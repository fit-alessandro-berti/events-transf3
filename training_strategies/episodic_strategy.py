import random
import torch
import torch .nn .functional as F
from utils .data_utils import create_episode
def run_episodic_step (model ,task_data_pool ,task_type ,config ,should_shuffle_labels ):
    progress_bar_task =task_type
    episode =None
    if task_type =='classification':
        episode =create_episode (
        task_data_pool ,config ['num_shots_range'],config ['num_queries'],
        num_ways_range =(3 ,10 ),shuffle_labels =should_shuffle_labels
        )
    else :
        if len (task_data_pool )<config ['num_shots_range'][1 ]+config ['num_queries']:
            episode =None
        else :
            random .shuffle (task_data_pool )
            num_shots =random .randint (config ['num_shots_range'][0 ],config ['num_shots_range'][1 ])
            support_set_raw =task_data_pool [:num_shots ]
            query_set_raw =task_data_pool [num_shots :num_shots +config ['num_queries']]
            support_set =[(s [0 ],s [1 ])for s in support_set_raw ]
            query_set =[(q [0 ],q [1 ])for q in query_set_raw ]
            episode =(support_set ,query_set )
    if episode is None or not episode [0 ]or not episode [1 ]:
        return None ,progress_bar_task
    support_set ,query_set =episode
    predictions ,true_labels ,_ =model (support_set ,query_set ,task_type )
    if predictions is None :
        return None ,progress_bar_task
    head_cfg =config .get ('fmv3_head',{})
    if task_type =='classification':
        smoothing =min (max (float (config .get ('classification_label_smoothing',0.05 )),0.0 ),1.0 )
        loss =F .cross_entropy (predictions ,true_labels ,ignore_index =-100 ,label_smoothing =smoothing )
        confidence_w =float (head_cfg .get ('classification_expert_confidence_loss_weight',0.0 ))
        if confidence_w >0:
            log_probs =F .log_softmax (predictions ,dim =-1 )
            probabilities =torch .exp (log_probs )
            loss =loss +confidence_w *model .proto_head .classification_expert_confidence_loss (
            probabilities ,true_labels )
    else :
        branch_predictions =None
        aggregation_weights =None
        diagnostics =getattr (model ,'last_regression_diagnostics',None )
        if (
        diagnostics is not None
        and model .proto_head .regression_outputs_hours
        and model .proto_head .regression_gate_aux_weight >0
        ):
            branch_predictions =diagnostics .get ('branch_predictions_hours')
            aggregation_weights =diagnostics .get ('aggregation_weights')
        loss =model .proto_head .regression_loss (
        predictions .squeeze (),
        true_labels ,
        labels_in_output_space =True ,
        branch_predictions =branch_predictions ,
        aggregation_weights =aggregation_weights ,
        )
        confidence_w =float (head_cfg .get ('regression_expert_confidence_loss_weight',0.0 ))
        if confidence_w >0 and diagnostics is not None:
            base_confidence =getattr (model ,'last_regression_base_confidence',None )
            if base_confidence is None:
                base_confidence =torch .ones_like (predictions .reshape (-1 ))
            loss =loss +confidence_w *model .proto_head .regression_expert_confidence_loss (
            predictions .squeeze (),true_labels ,base_confidence ,diagnostics ,
            labels_in_output_space =True )
    return loss ,progress_bar_task
