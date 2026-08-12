import torch
import torch .nn as nn
import pandas as pd
from .char_cnn_embedder import CharCNNEmbedder
from .temporal_adapter import LearnedTemporalInputAdapter
class LearnedEventEmbedder (nn .Module ):
    def __init__ (self ,char_vocab_size :int ,char_emb_dim :int ,char_cnn_out_dim :int ,
    num_feat_dim :int ,d_model :int ,dropout :float =0.1 ,time_input_config =None ):
        super ().__init__ ()
        self .char_embedder =CharCNNEmbedder (
        char_vocab_size ,char_emb_dim ,char_cnn_out_dim
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
        input_config =dict (time_input_config or {})
        self .time_input_adapter =None
        if input_config .get ('regression_time_input_transforms',False ):
            self .time_input_adapter =LearnedTemporalInputAdapter (
            d_model ,num_transforms =int (input_config .get ('regression_num_transforms',8 )),
            **input_config )
        self .replace_legacy_regression_times =bool (
        input_config .get ('regression_time_replace_legacy',False ))
    def forward (self ,events_df :pd .DataFrame ,use_time_adapter =False ,time_scale_factor =None ):
        activity_names =events_df ['activity_name'].tolist ()
        resource_names =events_df ['resource_name'].tolist ()
        act_emb =self .char_embedder (activity_names ,self .char_to_id )
        res_emb =self .char_embedder (resource_names ,self .char_to_id )
        device =act_emb .device
        num_arr =events_df [['cost','time_from_start','time_from_previous']].values
        num_tensor =torch .as_tensor (num_arr .copy (),dtype =torch .float32 ,device =device ).clamp_min (0 )
        num_feats =torch .log1p (num_tensor )
        if use_time_adapter and self .replace_legacy_regression_times :
            num_feats =num_feats .clone ()
            num_feats [:,1 :]=0.0
        combined_input =torch .cat ([act_emb ,res_emb ,num_feats ],dim =-1 )
        embedded =self .projection (combined_input )
        if use_time_adapter and self .time_input_adapter is not None :
            embedded =embedded +self .time_input_adapter (
            num_tensor [:,1 :],augmentation_factor =time_scale_factor )
        return self .dropout (embedded )
