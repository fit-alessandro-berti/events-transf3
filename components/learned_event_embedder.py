import torch
import torch .nn as nn
import pandas as pd
from .char_cnn_embedder import CharCNNEmbedder
from .attribute_encoder import GenericAttributeEncoder
from .temporal_adapter import (
    IndependentTemporalInputEncoder,
    LearnedTemporalInputAdapter,
)
class LearnedEventEmbedder (nn .Module ):
    def __init__ (self ,char_vocab_size :int ,char_emb_dim :int ,char_cnn_out_dim :int ,
    num_feat_dim :int ,d_model :int ,dropout :float =0.1 ,time_input_config =None ,
    max_string_length :int =64 ,attribute_hash_buckets :int =4096 ):
        super ().__init__ ()
        self .char_embedder =CharCNNEmbedder (
        char_vocab_size ,char_emb_dim ,char_cnn_out_dim ,max_word_len =max_string_length
        )
        self .char_to_id ={}
        total_input_dim =(2 *char_cnn_out_dim )+num_feat_dim
        self .projection =nn .Sequential (
        nn .LayerNorm (total_input_dim ),
        nn .Linear (total_input_dim ,d_model ),
        nn .GELU (),
        nn .LayerNorm (d_model )
        )
        self .dropout =nn .Dropout (dropout )
        self .context_projection =nn .Linear (char_cnn_out_dim +8 ,d_model ,bias =False )
        nn .init .zeros_ (self .context_projection .weight )
        self .attribute_encoder =GenericAttributeEncoder (
        d_model ,hash_buckets =attribute_hash_buckets )
        self .history_projection =nn .Linear (4 ,d_model ,bias =False )
        nn .init .zeros_ (self .history_projection .weight )
        self .history_transition_projection =nn .Linear (16 ,d_model ,bias =False )
        nn .init .zeros_ (self .history_transition_projection .weight )
        self .history_token =nn .Parameter (torch .zeros (d_model ))
        input_config =dict (time_input_config or {})
        self .temporal_input_encoder =None
        if input_config .get ('temporal_input_transforms',False ):
            self .temporal_input_encoder =IndependentTemporalInputEncoder (
            d_model ,**input_config )
        self .replace_legacy_temporal_inputs =bool (
        input_config .get ('temporal_replace_legacy_inputs',False ))
        # Legacy regression-only path retained for old checkpoint evaluation.
        self .time_input_adapter =None
        if input_config .get ('regression_time_input_transforms',False ):
            self .time_input_adapter =LearnedTemporalInputAdapter (
            d_model ,num_transforms =int (input_config .get ('regression_num_transforms',8 )),
            **input_config )
        self .replace_legacy_regression_times =bool (
        input_config .get ('regression_time_replace_legacy',False ))
    def forward (self ,events_df :pd .DataFrame ,use_time_adapter =False ,time_scale_factor =None ):
        events =(
        events_df .to_dict ('records')
        if isinstance (events_df ,pd .DataFrame )
        else list (events_df )
        )
        activity_names =[event ['activity_name']for event in events ]
        resource_names =[event ['resource_name']for event in events ]
        act_emb =self .char_embedder (
        activity_names ,self .char_to_id ,
        cached_ids =[event .get ('activity_char_ids',())for event in events ]
        if all ('activity_char_ids'in event for event in events )else None )
        res_emb =self .char_embedder (
        resource_names ,self .char_to_id ,
        cached_ids =[event .get ('resource_char_ids',())for event in events ]
        if all ('resource_char_ids'in event for event in events )else None )
        device =act_emb .device
        num_tensor =torch .as_tensor (
        [[event ['cost'],event ['time_from_start'],event ['time_from_previous']]
        for event in events ],dtype =torch .float32 ,device =device )
        num_feats =torch .empty_like (num_tensor )
        num_feats [:,0 ]=torch .sign (num_tensor [:,0 ])*torch .log1p (
        num_tensor [:,0 ].abs ())
        num_feats [:,1 :]=torch .log1p (num_tensor [:,1 :].clamp_min (0 ))
        temporal_tensor =num_tensor [:,1 :].clamp_min (0 )
        if self .temporal_input_encoder is not None and self .replace_legacy_temporal_inputs :
            num_feats =num_feats .clone ()
            num_feats [:,1 :]=0.0
        elif use_time_adapter and self .replace_legacy_regression_times :
            num_feats =num_feats .clone ()
            num_feats [:,1 :]=0.0
        combined_input =torch .cat ([act_emb ,res_emb ,num_feats ],dim =-1 )
        embedded =self .projection (combined_input )
        lifecycle_names =[event .get ('lifecycle_name','Unknown')for event in events ]
        lifecycle =self .char_embedder (
        lifecycle_names ,self .char_to_id ,
        cached_ids =[event .get ('lifecycle_char_ids',())for event in events ]
        if all ('lifecycle_char_ids'in event for event in events )else None )
        context =torch .as_tensor ([
        [*event .get ('calendar_features',(0.0 ,0.0 ,0.0 ,0.0 ,0.0 )),
        event .get ('resource_missing',0.0 ),event .get ('cost_missing',0.0 ),
        event .get ('lifecycle_missing',0.0 )]
        for event in events ],dtype =torch .float32 ,device =device )
        embedded =embedded +self .context_projection (torch .cat ([lifecycle ,context ],dim =-1 ))
        embedded =embedded +self .attribute_encoder (events ,device )
        history =torch .as_tensor ([
        event .get ('history_features',(0.0 ,0.0 ,0.0 ,0.0 ))
        for event in events ],dtype =torch .float32 ,device =device )
        history [:,:2 ]=torch .log1p (history [:,:2 ].clamp_min (0 ))
        history [:,3 ]=torch .log1p (history [:,3 ].clamp_min (0 ))
        history_mask =torch .as_tensor ([
        event .get ('is_history_summary',0.0 )for event in events
        ],dtype =torch .float32 ,device =device ).unsqueeze (1 )
        history_transitions =torch .as_tensor ([
        event .get ('history_transition_features',(0.0 ,)*16 )for event in events
        ],dtype =torch .float32 ,device =device )
        embedded =embedded +history_mask *(
        self .history_projection (history )+
        self .history_transition_projection (history_transitions )+
        self .history_token )
        if use_time_adapter and self .time_input_adapter is not None :
            embedded =embedded +self .time_input_adapter (
            temporal_tensor ,augmentation_factor =time_scale_factor )
        if self .temporal_input_encoder is not None :
            embedded =embedded +self .temporal_input_encoder (
            temporal_tensor ,augmentation_factor =time_scale_factor )
        return self .dropout (embedded )
