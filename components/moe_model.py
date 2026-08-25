import math
import torch
import torch .nn as nn
import torch .nn .functional as F
from .meta_learner import MetaLearner
class MoEModel (nn .Module ):
    def __init__ (self ,num_experts ,strategy ,shared_backbone =False ,**kwargs ):
        super ().__init__ ()
        self .num_experts =num_experts
        self .strategy =strategy
        self .shared_backbone =bool (shared_backbone )
        routing_config =kwargs .get ('proto_head_config')or {}
        enabled_value =routing_config .get ('expert_routing_confidence_enabled',False )
        if isinstance (enabled_value ,str ):
            enabled_value =enabled_value .strip ().lower ()in {'1','true','yes','y','on'}
        self .expert_routing_confidence_enabled =bool (enabled_value )
        self .expert_active_fraction =float (
        routing_config .get ('expert_active_fraction',0.5 ))
        if not 0.0 <self .expert_active_fraction <=1.0 :
            raise ValueError ("expert_active_fraction must be in (0, 1]")
        self .expert_routing_temperature =float (
        routing_config .get ('expert_routing_temperature',1.0 ))
        if not math .isfinite (self .expert_routing_temperature )or self .expert_routing_temperature <=0.0 :
            raise ValueError ("expert_routing_temperature must be finite and positive")
        self .last_routing_diagnostics =None
        self .experts =nn .ModuleList ([
        MetaLearner (strategy =strategy ,**kwargs )
        for _ in range (num_experts )
        ])
        if self .shared_backbone and self .experts :
            shared_embedder =self .experts [0 ].embedder
            shared_encoder =self .experts [0 ].encoder
            for expert in self .experts [1 :]:
                expert .embedder =shared_embedder
                expert .encoder =shared_encoder
        architecture ="shared backbone + lightweight experts"if self .shared_backbone else "independent experts"
        print (f"✅ Initialized MoEModel with {num_experts } expert(s): {architecture }.")
    @property
    def active_expert_count (self ):
        if not self .expert_routing_confidence_enabled :
            return self .num_experts
        # ceil keeps at least half active for odd expert counts and is exactly
        # half for the repository's default even-sized MoE.
        return max (1 ,min (self .num_experts ,int (math .ceil (
        self .num_experts *self .expert_active_fraction ))))
    def route_experts (self ,support_set ,query_set ,task_type ):
        """Choose experts from cheap learned task confidence before execution."""
        if not self .experts :
            return [],torch .empty (0 )
        reference =next (self .parameters ())
        if not self .expert_routing_confidence_enabled :
            indices =list (range (self .num_experts ))
            logits =reference .new_zeros (self .num_experts )
        else :
            logits =torch .stack ([
            expert .task_confidence_logit (support_set ,query_set ,task_type )
            for expert in self .experts
            ])
            # Stable low-index tie breaking matters for neutral initialization.
            tie_break =torch .arange (
            self .num_experts ,device =logits .device ,dtype =logits .dtype )*1e-7
            selected =torch .topk (
            logits -tie_break ,self .active_expert_count ,dim =0 ).indices
            indices =sorted (int (index )for index in selected .tolist ())
        probabilities =torch .sigmoid (logits /self .expert_routing_temperature )
        self .last_routing_diagnostics ={
        'task_type':str (task_type ),
        'selected_expert_indices':indices ,
        'active_expert_count':len (indices ),
        'total_expert_count':self .num_experts ,
        'confidence_logits':logits .detach ().cpu ().tolist (),
        'confidences':probabilities .detach ().cpu ().tolist (),
        }
        return indices ,logits
    def set_char_vocab (self ,char_to_id :dict ):
        if self .strategy =='learned':
            for expert in self .experts :
                expert .set_char_vocab (char_to_id )
    def _process_batch (self ,batch_of_sequences ,task_type =None ):
        if not self .experts :
            return None
        if self .shared_backbone :
            encoded =self .experts [0 ]._encode_batch (
            batch_of_sequences ,task_type =task_type )
            all_expert_embeddings =[
            expert .adapt_task_embeddings (encoded ,task_type )
            for expert in self .experts
            ]
        else :
            all_expert_embeddings =[
            expert ._process_batch (batch_of_sequences ,task_type =task_type )
            for expert in self .experts
            ]
        stacked_embeddings =torch .stack (all_expert_embeddings )
        avg_embeddings =torch .mean (stacked_embeddings ,dim =0 )
        return avg_embeddings
    def _aggregate_outputs (self ,expert_outputs ,task_type ,true_labels ):
        if task_type =='regression':
            all_preds =torch .stack ([out [0 ]for out in expert_outputs ])
            all_confs =torch .stack ([out [2 ]for out in expert_outputs ])
            learned_logits =[out [3 ]for out in expert_outputs if len (out )>3 and out [3 ]is not None]
            if len (learned_logits )==len (expert_outputs ):
                expert_weights =F .softmax (torch .stack (learned_logits ),dim =0 )
            else :
                expert_weights =all_confs /all_confs .sum (dim =0 ).clamp_min (1e-8 )
            final_preds =(all_preds *expert_weights ).sum (dim =0 )
            final_confidence =(all_confs *expert_weights ).sum (dim =0 )
            return final_preds ,true_labels ,final_confidence
        elif task_type =='classification':
            all_confs_stacked =torch .stack ([out [2 ]for out in expert_outputs ])
            learned_logits =[out [3 ]for out in expert_outputs if len (out )>3 and out [3 ]is not None]
            if len (learned_logits )==len (expert_outputs ):
                expert_weights =F .softmax (torch .stack (learned_logits ),dim =0 )
                final_predictions =(all_confs_stacked *expert_weights .unsqueeze (-1 )).sum (dim =0 )
            else :
                final_predictions =all_confs_stacked .mean (dim =0 )
            norm_confs =F .normalize (final_predictions ,p =1 ,dim =-1 )
            final_confidence ,_ =torch .max (norm_confs ,dim =-1 )
            return final_predictions ,true_labels ,final_confidence
    def forward (self ,support_set ,query_set ,task_type ,expert_id =None ):
        if self .training :
            if expert_id is None :
                raise ValueError ("MoEModel.forward() requires an 'expert_id' during training.")
            if expert_id >=self .num_experts :
                raise IndexError (f"Invalid expert_id {expert_id }. Max is {self .num_experts -1 }.")
            return self .experts [expert_id ](support_set ,query_set ,task_type )
        else :
            expert_outputs =[]
            all_true_labels =None
            selected_indices ,_=self .route_experts (
            support_set ,query_set ,task_type )
            for expert_index in selected_indices :
                expert =self .experts [expert_index ]
                preds ,labels ,confs =expert (support_set ,query_set ,task_type )
                if preds is None :continue
                expert_outputs .append ((preds ,labels ,confs ,getattr (expert ,'last_expert_confidence_logit',None )))
                if all_true_labels is None :
                    all_true_labels =labels
            if not expert_outputs :
                return None ,None ,None
            return self ._aggregate_outputs (expert_outputs ,task_type ,all_true_labels )
