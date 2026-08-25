import torch
import os
import re
from config import CONFIG
from data_generator import XESLogLoader
from components .meta_learner import MetaLearner
from components .moe_model import MoEModel
def extract_model_state_dict (payload ):
    if isinstance (payload ,dict ):
        for key in ('model','model_state_dict','state_dict'):
            candidate =payload .get (key )
            if isinstance (candidate ,dict ):
                return candidate
    return payload
def init_loader (config ):
    strategy =config ['embedding_strategy']
    sbert_model_name =config ['pretrained_settings']['sbert_model']
    data_config =config .get ('data',{})or {}
    loader =XESLogLoader (
    strategy =strategy ,sbert_model_name =sbert_model_name ,
    sbert_model_revision =config ['pretrained_settings'].get ('revision'),
    max_string_length =config .get ('learned_settings',{}).get ('max_string_length',64 ),
    max_generic_attributes =data_config .get ('max_generic_attributes',16 ),
    attribute_hash_buckets =data_config .get ('attribute_hash_buckets',4096 ),
    )
    return loader
def create_model (config ,loader ,device ):
    strategy =config ['embedding_strategy']
    moe_config =config .get ('moe_settings',{})
    num_experts =moe_config .get ('num_experts',1 )
    if strategy =='pretrained':
        model_params ={'embedding_dim':config ['pretrained_settings']['embedding_dim']}
    else :
        if not loader .char_to_id :
            raise RuntimeError ("Loader must be fitted or artifacts loaded before creating 'learned' model.")
        model_params ={
        'char_vocab_size':len (loader .char_to_id ),
        'char_embedding_dim':config ['learned_settings']['char_embedding_dim'],
        'char_cnn_output_dim':config ['learned_settings']['char_cnn_output_dim'],
        'max_string_length':config ['learned_settings'].get ('max_string_length',64 ),
        }
    data_config =config .get ('data',{})or {}
    model_params ['attribute_hash_buckets']=data_config .get ('attribute_hash_buckets',4096 )
    model =MoEModel (
    num_experts =num_experts ,
    strategy =strategy ,
    num_feat_dim =config ['num_numerical_features'],
    d_model =config ['d_model'],
    n_heads =config ['n_heads'],
    n_layers =config ['n_layers'],
    dropout =config ['dropout'],
    proto_head_config =config .get ('fmv3_head',{}),
    shared_backbone =moe_config .get ('shared_backbone',False ),
    expert_adapter_enabled =moe_config .get ('expert_adapter_enabled',False ),
    expert_adapter_hidden_dim =moe_config .get ('expert_adapter_hidden_dim',64 ),
    **model_params
    ).to (device )
    if strategy =='learned':
        model .set_char_vocab (loader .char_to_id )
    return model
def load_state_dict_compatible (model ,state_dict ):
    """Load checkpoints across explicit temporal/prefix adapter migrations."""
    if getattr (model ,'shared_backbone',False ):
        state_dict =dict (state_dict )
        shared_suffixes ={}
        for key ,value in state_dict .items ():
            match =re .match (r'experts\.(\d+)\.(embedder|encoder)\.(.+)',key )
            if match :
                suffix =f"{match .group (2 )}.{match .group (3 )}"
                shared_suffixes .setdefault (suffix ,[]).append (value )
        for suffix ,values in shared_suffixes .items ():
            if len (values )>1 and values [0 ].is_floating_point ():
                shared_value =torch .stack ([
                value .detach ().to (dtype =torch .float32 ,device ='cpu')
                for value in values
                ]).mean (dim =0 ).to (dtype =values [0 ].dtype )
            else :
                shared_value =values [0 ]
            for expert_index in range (model .num_experts ):
                state_dict [f"experts.{expert_index}.{suffix}"]=shared_value
    incompatible =model .load_state_dict (state_dict ,strict =False )
    allowed_missing =[
    key for key in incompatible .missing_keys
    if '.proto_head.time_transform_bank.'in key
    or '.proto_head.classification_expert_confidence.'in key
    or '.proto_head.regression_expert_confidence.'in key
    or '.task_confidence_head.'in key
    or '.classification_embedding_adapter.'in key
    or '.classification_retrieval_projection.'in key
    or '.classification_decision_projection.'in key
    or '.candidate_label_projection.'in key
    or '.regression_embedding_adapter.'in key
    or '.proto_head.classification_example_selector.'in key
    or '.proto_head.regression_example_selector.'in key
    or '.embedder.time_input_adapter.'in key
    or '.embedder.temporal_input_encoder.'in key
    or 'encoder.state_aware_pool.'in key
    or 'encoder.classification_private_encoder.'in key
    or '.expert_adapter.'in key
    or '.embedder.context_projection.'in key
    or '.embedder.attribute_encoder.'in key
    or '.embedder.history_projection.'in key
    or '.embedder.history_transition_projection.'in key
    or '.embedder.history_token'in key
    or '.proto_head.regression_residual_head.'in key
    or '.proto_head.semantic_query_projection.'in key
    or '.proto_head.semantic_candidate_projection.'in key
    or '.proto_head.process_candidate_decoder.'in key
    or key .endswith ('.proto_head.semantic_logit_scale')
    or key .endswith ('.proto_head._support_gate_kappa_raw')
    or key .endswith ('.proto_head._support_fitted_regularization_raw')
    ]
    migrating_to_independent_inputs =any (
    '.embedder.temporal_input_encoder.'in key for key in allowed_missing )
    allowed_unexpected =[
    key for key in incompatible .unexpected_keys
    if (
    (migrating_to_independent_inputs and '.embedder.time_input_adapter.'in key)
    or '.proto_head.time_transform_bank.'in key
    or '.proto_head.classification_expert_confidence.'in key
    or '.proto_head.regression_expert_confidence.'in key
    or '.task_confidence_head.'in key
    or '.classification_embedding_adapter.'in key
    or '.classification_retrieval_projection.'in key
    or '.classification_decision_projection.'in key
    or '.candidate_label_projection.'in key
    or '.regression_embedding_adapter.'in key
    or '.proto_head.classification_example_selector.'in key
    or '.proto_head.regression_example_selector.'in key
    or 'encoder.classification_private_encoder.'in key
    or '.proto_head.semantic_query_projection.'in key
    or '.proto_head.semantic_candidate_projection.'in key
    or '.proto_head.process_candidate_decoder.'in key
    or key .endswith ('.proto_head.semantic_logit_scale')
    or key .endswith ('.proto_head._support_gate_kappa_raw')
    or key .endswith ('.proto_head._support_fitted_regularization_raw')
    )
    ]
    disallowed_missing =sorted (set (incompatible .missing_keys )-set (allowed_missing ))
    disallowed_unexpected =sorted (
    set (incompatible .unexpected_keys )-set (allowed_unexpected ))
    if disallowed_missing or disallowed_unexpected :
        raise RuntimeError (
        f"Checkpoint mismatch. Missing={disallowed_missing}; "
        f"unexpected={disallowed_unexpected}"
        )
    if allowed_missing :
        print (f"🆕 Initialized {len (allowed_missing )} learned adapter parameters from config.")
    if allowed_unexpected :
        print (f"♻️ Ignored {len (allowed_unexpected )} checkpoint parameters not used by current config.")
    return incompatible
def load_model_weights (model ,checkpoint_dir ,device ,epoch_num =None ):
    if not os .path .isdir (checkpoint_dir ):
        exit (f"❌ Error: Checkpoint directory not found at {checkpoint_dir }")
    checkpoint_path =None
    if epoch_num is not None :
        checkpoint_name =f"model_epoch_{epoch_num }.pth"
        checkpoint_path =os .path .join (checkpoint_dir ,checkpoint_name )
        if not os .path .exists (checkpoint_path ):
            exit (f"❌ Error: Specific checkpoint not found: {checkpoint_path }")
        print (f"🔍 Found specific checkpoint: {checkpoint_name }")
    else :
        checkpoints =[f for f in os .listdir (checkpoint_dir )if f .startswith ('model_epoch_')and f .endswith ('.pth')]
        if not checkpoints :
            exit (f"❌ Error: No model checkpoints found in {checkpoint_dir }.")
        latest_checkpoint_name =sorted (checkpoints ,key =lambda f :int (re .search (r'(\d+)',f ).group (1 )))[-1 ]
        checkpoint_path =os .path .join (
        checkpoint_dir ,
        latest_checkpoint_name
        )
        print (f"🔍 Found latest checkpoint: {latest_checkpoint_name }")
    print (f"💾 Loading weights from {checkpoint_path }...")
    load_state_dict_compatible (model ,torch .load (checkpoint_path ,map_location =device ))
    return checkpoint_path
