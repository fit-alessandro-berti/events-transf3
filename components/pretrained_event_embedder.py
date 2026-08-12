import torch
import torch .nn as nn
import numpy as np
import pandas as pd
from .temporal_adapter import (
    IndependentTemporalInputEncoder,
    LearnedTemporalInputAdapter,
)
class PretrainedEventEmbedder (nn .Module ):
    def __init__ (self ,embedding_dim :int ,num_feat_dim :int ,d_model :int ,dropout :float =0.1 ,time_input_config =None ):
        super ().__init__ ()
        total_input_dim =(2 *embedding_dim )+num_feat_dim
        self .projection =nn .Sequential (
        nn .LayerNorm (total_input_dim ),
        nn .Linear (total_input_dim ,d_model ),
        nn .GELU (),
        nn .LayerNorm (d_model )
        )
        self .dropout =nn .Dropout (dropout )
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
        device =next (self .parameters ()).device
        act_emb =torch .from_numpy (np .stack (events_df ['activity_embedding'].values )).float ().to (device )
        res_emb =torch .from_numpy (np .stack (events_df ['resource_embedding'].values )).float ().to (device )
        num_arr =events_df [['cost','time_from_start','time_from_previous']].values
        num_tensor =torch .as_tensor (num_arr .copy (),dtype =torch .float32 ,device =device ).clamp_min (0 )
        num_feats =torch .log1p (num_tensor )
        if self .temporal_input_encoder is not None and self .replace_legacy_temporal_inputs :
            num_feats =num_feats .clone ()
            num_feats [:,1 :]=0.0
        elif use_time_adapter and self .replace_legacy_regression_times :
            num_feats =num_feats .clone ()
            num_feats [:,1 :]=0.0
        combined_input =torch .cat ([act_emb ,res_emb ,num_feats ],dim =-1 )
        embedded =self .projection (combined_input )
        if use_time_adapter and self .time_input_adapter is not None :
            embedded =embedded +self .time_input_adapter (
            num_tensor [:,1 :],augmentation_factor =time_scale_factor )
        if self .temporal_input_encoder is not None :
            embedded =embedded +self .temporal_input_encoder (
            num_tensor [:,1 :],augmentation_factor =time_scale_factor )
        return self .dropout (embedded )
