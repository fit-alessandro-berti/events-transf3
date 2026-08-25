import torch
import torch .nn as nn
import numpy as np
from .learned_event_embedder import LearnedEventEmbedder
from .pretrained_event_embedder import PretrainedEventEmbedder
from .event_encoder import EventEncoder
from .prototypical_head import PrototypicalHead
from .task_confidence import TaskConfidenceHead, build_task_descriptor
class MetaLearner (nn .Module ):
    def __init__ (self ,strategy :str ,num_feat_dim :int ,d_model :int ,n_heads :int ,n_layers :int ,dropout :float =0.1 ,proto_head_config =None ,**kwargs ):
        super ().__init__ ()
        self .strategy =strategy
        self .encoder =EventEncoder (
        d_model ,n_heads ,n_layers ,dropout ,prefix_config =proto_head_config )
        self .proto_head =PrototypicalHead (feature_dim =d_model ,**(proto_head_config or {}))
        routing_config =proto_head_config or {}
        routing_enabled =routing_config .get ('expert_routing_confidence_enabled',False )
        if isinstance (routing_enabled ,str ):
            routing_enabled =routing_enabled .strip ().lower ()in {'1','true','yes','y','on'}
        self .expert_routing_confidence_enabled =bool (routing_enabled )
        self .task_confidence_head =None
        if self .expert_routing_confidence_enabled :
            self .task_confidence_head =TaskConfidenceHead (
            architecture =routing_config .get ('expert_routing_architecture','mlp'),
            hidden_dim =routing_config .get ('expert_routing_hidden_dim',32 ),
            dropout =routing_config .get ('expert_routing_dropout',0.0 ),
            )
        self .proj_head =nn .Sequential (
        nn .Linear (d_model ,d_model ),
        nn .GELU (),
        nn .Linear (d_model ,128 ),
        nn .LayerNorm (128 ),
        )
        self .d_model =d_model
        self .expert_adapter =None
        if kwargs .get ('expert_adapter_enabled',False ):
            expert_hidden =max (int (kwargs .get ('expert_adapter_hidden_dim',64 )),1 )
            self .expert_adapter =nn .Sequential (
            nn .LayerNorm (d_model ),
            nn .Linear (d_model ,expert_hidden ),
            nn .GELU (),
            nn .Linear (expert_hidden ,d_model ,bias =False ),
            )
            # Exact identity for checkpoint migration; balanced expert quotas
            # make the adapters diverge as soon as training starts.
            nn .init .zeros_ (self .expert_adapter [-1 ].weight )
        self .classification_embedding_adapter =None
        if routing_config .get ('classification_embedding_adapter_enabled',False ):
            adapter_hidden =max (int (routing_config .get (
            'classification_embedding_adapter_hidden_dim',64 )),1 )
            adapter_dropout =max (float (routing_config .get (
            'classification_embedding_adapter_dropout',0.0 )),0.0 )
            self .classification_embedding_adapter =nn .Sequential (
            nn .LayerNorm (d_model ),
            nn .Linear (d_model ,adapter_hidden ),
            nn .GELU (),
            nn .Dropout (adapter_dropout ),
            nn .Linear (adapter_hidden ,d_model ),
            )
            # Exact identity at initialization and when migrating a checkpoint.
            nn .init .zeros_ (self .classification_embedding_adapter [-1 ].weight )
            nn .init .zeros_ (self .classification_embedding_adapter [-1 ].bias )
        self .classification_retrieval_projection =None
        self .classification_decision_projection =None
        projection_enabled =routing_config .get (
        'classification_separate_projections_enabled',False )
        if isinstance (projection_enabled ,str ):
            projection_enabled =projection_enabled .strip ().lower ()in {
            '1','true','yes','y','on'}
        if projection_enabled :
            projection_hidden =max (int (routing_config .get (
            'classification_projection_hidden_dim',d_model )),8 )
            self .classification_retrieval_projection =self ._residual_projection (
            d_model ,projection_hidden )
            self .classification_decision_projection =self ._residual_projection (
            d_model ,projection_hidden )
        self .regression_embedding_adapter =None
        if routing_config .get ('regression_embedding_adapter_enabled',False ):
            adapter_hidden =max (int (routing_config .get (
            'regression_embedding_adapter_hidden_dim',64 )),1 )
            adapter_dropout =max (float (routing_config .get (
            'regression_embedding_adapter_dropout',0.0 )),0.0 )
            self .regression_embedding_adapter =nn .Sequential (
            nn .LayerNorm (d_model ),
            nn .Linear (d_model ,adapter_hidden ),
            nn .GELU (),
            nn .Dropout (adapter_dropout ),
            nn .Linear (adapter_hidden ,d_model ),
            )
            nn .init .zeros_ (self .regression_embedding_adapter [-1 ].weight )
            nn .init .zeros_ (self .regression_embedding_adapter [-1 ].bias )
        # Side-channel for episodic regression: branch diagnostics used by gate-aux.
        self .last_classification_diagnostics =None
        self .last_regression_diagnostics =None
        self .last_expert_confidence_logit =None
        self .last_regression_base_confidence =None
        if self .strategy =='pretrained':
            self .embedding_dim =kwargs ['embedding_dim']
            self .embedder =PretrainedEventEmbedder (
            self .embedding_dim ,num_feat_dim ,d_model ,dropout ,
            time_input_config =proto_head_config ,
            attribute_hash_buckets =kwargs .get ('attribute_hash_buckets',4096 ))
            self .pad_event ={'activity_embedding':np .zeros (self .embedding_dim ,dtype =np .float32 ),'resource_embedding':np .zeros (self .embedding_dim ,dtype =np .float32 ),'activity_id':0 ,'cost':0.0 ,'time_from_start':0.0 ,'time_from_previous':0.0 ,'timestamp':0.0 ,'case_id':'pad','calendar_features':(0.0 ,)*5 ,'resource_missing':1.0 ,'cost_missing':1.0 ,'lifecycle_missing':1.0 ,'generic_attributes':()}
            candidate_input_dim =self .embedding_dim
        elif self .strategy =='learned':
            max_string_length =kwargs .get ('max_string_length',64 )
            self .embedder =LearnedEventEmbedder (
            char_vocab_size =kwargs ['char_vocab_size'],
            char_emb_dim =kwargs ['char_embedding_dim'],
            char_cnn_out_dim =kwargs ['char_cnn_output_dim'],
            num_feat_dim =num_feat_dim ,
            d_model =d_model ,
            dropout =dropout
            ,time_input_config =proto_head_config
            ,max_string_length =max_string_length
            ,attribute_hash_buckets =kwargs .get ('attribute_hash_buckets',4096 )
            )
            self .pad_event ={
            'activity_name':'','resource_name':'','activity_id':-100 ,
            'cost':0.0 ,'time_from_start':0.0 ,'time_from_previous':0.0 ,
            'timestamp':0.0 ,'case_id':'pad'
            ,'calendar_features':(0.0 ,)*5 ,'resource_missing':1.0 ,
            'cost_missing':1.0 ,'lifecycle_name':'','lifecycle_missing':1.0 ,
            'generic_attributes':(),
            'activity_char_ids':(0 ,)*max_string_length ,
            'resource_char_ids':(0 ,)*max_string_length ,
            'lifecycle_char_ids':(0 ,)*max_string_length
            }
            candidate_input_dim =int (kwargs ['char_cnn_output_dim'])
        else :
            raise ValueError (f"Unknown embedding strategy: '{self .strategy }'")
        semantic_enabled =routing_config .get ('semantic_candidate_decoder_enabled',False )
        if isinstance (semantic_enabled ,str ):
            semantic_enabled =semantic_enabled .strip ().lower ()in {
            '1','true','yes','y','on'}
        self .semantic_candidate_decoder_enabled =bool (semantic_enabled )
        self .candidate_label_projection =None
        if self .semantic_candidate_decoder_enabled :
            self .candidate_label_projection =nn .Sequential (
            nn .LayerNorm (candidate_input_dim ),
            nn .Linear (candidate_input_dim ,d_model ),
            nn .GELU (),
            nn .LayerNorm (d_model ),
            )
    @staticmethod
    def _residual_projection (feature_dim ,hidden_dim ):
        projection =nn .Sequential (
        nn .LayerNorm (feature_dim ),
        nn .Linear (feature_dim ,hidden_dim ),
        nn .GELU (),
        nn .Linear (hidden_dim ,feature_dim ),
        )
        nn .init .zeros_ (projection [-1 ].weight )
        nn .init .zeros_ (projection [-1 ].bias )
        return projection
    def set_char_vocab (self ,char_to_id :dict ):
        if self .strategy =='learned':
            self .embedder .char_to_id =char_to_id
            print ("Character vocabulary set in LearnedEventEmbedder.")
    def task_confidence_descriptor (self ,support_set ,query_set ,task_type ):
        parameter =next (self .parameters ())
        return build_task_descriptor (
        support_set ,query_set ,task_type ,device =parameter .device ,dtype =parameter .dtype )
    def task_confidence_logit (self ,support_set ,query_set ,task_type ):
        """Return a pre-execution task reliability logit for this expert."""
        if self .task_confidence_head is None :
            return next (self .parameters ()).new_zeros (())
        descriptor =self .task_confidence_descriptor (support_set ,query_set ,task_type )
        return self .task_confidence_head (descriptor )
    def task_confidence_loss (self ,support_set ,query_set ,task_type ,target ):
        if self .task_confidence_head is None :
            return next (self .parameters ()).new_zeros (())
        descriptor =self .task_confidence_descriptor (support_set ,query_set ,task_type )
        return self .task_confidence_head .reliability_loss (descriptor ,target )
    def adapt_task_embeddings (self ,encoded ,task_type ):
        """Apply the opt-in task residual without changing the other task."""
        if self .expert_adapter is not None :
            encoded =encoded +self .expert_adapter (encoded )
        if task_type =='classification'and self .classification_embedding_adapter is not None :
            return encoded +self .classification_embedding_adapter (encoded )
        if task_type =='regression'and self .regression_embedding_adapter is not None :
            return encoded +self .regression_embedding_adapter (encoded )
        return encoded
    def classification_retrieval_features (self ,encoded ):
        if self .classification_retrieval_projection is None :return encoded
        return encoded +self .classification_retrieval_projection (encoded )
    def classification_decision_features (self ,encoded ):
        if self .classification_decision_projection is None :return encoded
        return encoded +self .classification_decision_projection (encoded )
    def encode_candidate_labels (self ,candidate_labels ):
        """Encode a complete log-local label schema without event context."""
        if not self .semantic_candidate_decoder_enabled :return None
        candidates =list (candidate_labels or ())
        device =next (self .parameters ()).device
        if not candidates :return torch .empty (0 ,self .d_model ,device =device )
        if self .strategy =='learned':
            names =[str (candidate .get ('activity_name',''))for candidate in candidates ]
            cached =[
            candidate .get ('activity_char_ids',())for candidate in candidates ]
            raw =self .embedder .char_embedder (
            names ,self .embedder .char_to_id ,
            cached_ids =cached if all (cached )else None )
        else :
            raw =torch .as_tensor (np .stack ([
            candidate ['activity_embedding']for candidate in candidates
            ]),dtype =torch .float32 ,device =device )
        return self .candidate_label_projection (raw )
    def _encode_batch (self ,batch_of_sequences ,task_type =None ,time_scale_factor =None ):
        device =next (self .parameters ()).device
        max_len =max (len (seq )for seq in batch_of_sequences )if batch_of_sequences else 0
        if max_len ==0 :return torch .empty (0 ,self .d_model ,device =device )
        padded_events ,masks =[],[]
        for seq in batch_of_sequences :
            pad_len =max_len -len (seq )
            mask =[False ]*len (seq )+[True ]*pad_len
            padded_events .extend (list (seq ))
            if pad_len >0 :
                padded_events .extend ([self .pad_event ]*pad_len )
            masks .append (mask )
        all_embeddings =self .embedder (
        padded_events ,use_time_adapter =(task_type =='regression'),
        time_scale_factor =time_scale_factor )
        embeddings_reshaped =all_embeddings .view (len (batch_of_sequences ),max_len ,-1 )
        mask_tensor =torch .tensor (masks ,dtype =torch .bool ,device =device )
        return self .encoder (
        embeddings_reshaped ,src_key_padding_mask =mask_tensor ,task_type =task_type )
    def _process_batch (self ,batch_of_sequences ,task_type =None ,time_scale_factor =None ):
        encoded =self ._encode_batch (
        batch_of_sequences ,task_type =task_type ,time_scale_factor =time_scale_factor )
        return self .adapt_task_embeddings (encoded ,task_type )
    def forward (self ,support_set ,query_set ,task_type ):
        support_seqs ,query_seqs =[s [0 ]for s in support_set ],[q [0 ]for q in query_set ]
        all_seqs =support_seqs +query_seqs
        if not all_seqs :return None ,None ,None
        time_scale_factor =None
        if task_type =='regression'and self .proto_head .regression_uses_time_transform_bank :
            time_scale_factor =self .proto_head .time_transform_bank .sample_augmentation_factor (
            next (self .parameters ()))
        all_encoded =self ._process_batch (
        all_seqs ,task_type =task_type ,time_scale_factor =time_scale_factor )
        if task_type =='classification':
            all_encoded =self .classification_decision_features (all_encoded )
        support_features =all_encoded [:len (support_seqs )]
        query_features =all_encoded [len (support_seqs ):]
        device =all_encoded .device
        if task_type =='classification':
            self .last_classification_diagnostics =None
            self .last_regression_diagnostics =None
            self .last_expert_confidence_logit =None
            self .last_regression_base_confidence =None
            support_labels =torch .LongTensor ([s [1 ]for s in support_set ]).to (device )
            query_labels =torch .LongTensor ([q [1 ]for q in query_set ]).to (device )
            predictions ,proto_classes ,confidence ,diagnostics =self .proto_head .forward_classification (
            support_features ,support_labels ,query_features ,return_diagnostics =True )
            if predictions is None :return None ,None ,None
            self .last_classification_diagnostics =diagnostics
            self .last_expert_confidence_logit =self .proto_head .classification_expert_confidence_logit (confidence )
            label_map ={orig_label .item ():new_label for new_label ,orig_label in enumerate (proto_classes )}
            mapped_labels =torch .tensor ([label_map .get (l .item (),-100 )for l in query_labels ],device =device ,dtype =torch .long )
            return predictions ,mapped_labels ,confidence
        elif task_type =='regression':
            self .last_classification_diagnostics =None
            self .last_expert_confidence_logit =None
            self .last_regression_base_confidence =None
            support_labels =torch .as_tensor ([s [1 ]for s in support_set ],dtype =torch .float32 ,device =device )
            query_labels =torch .as_tensor ([q [1 ]for q in query_set ],dtype =torch .float32 ,device =device )
            # Diagnostics are always materialised inside the head; request them so
            # episodic training can attach gate-aux without a second forward.
            predictions ,confidence ,diagnostics =self .proto_head .forward_regression (
            support_features ,support_labels ,query_features ,
            return_diagnostics =True ,augmentation_factor =time_scale_factor )
            self .last_regression_diagnostics =diagnostics
            self .last_regression_base_confidence =confidence
            self .last_expert_confidence_logit =self .proto_head .regression_expert_confidence_logit (
            predictions ,confidence ,diagnostics )
            return predictions ,self .proto_head .regression_labels_to_output (query_labels ),confidence
        else :raise ValueError (f"Unknown task type: {task_type }")
