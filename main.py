import numpy as np
import torch
import os
import argparse
import shutil
import re
import random
from config import CONFIG
from config_utils import apply_experiment_config, save_yaml_config
from utils .data_utils import get_task_data
from utils .model_utils import init_loader ,create_model ,load_state_dict_compatible
from utils .parameter_utils import configure_trainable_scope
from training import train
from training_debug import save_validation_manifest, split_training_tasks_by_case
from metric_objectives import resolve_classification_objective, resolve_regression_metric_weights
from training_log_sets import combined_training_log_paths, resolve_training_log_sets
def main ():
    pre_parser =argparse .ArgumentParser (add_help =False )
    pre_parser .add_argument ('--config',type =str ,default =None )
    pre_parser .add_argument ('--set',dest ='config_overrides',action ='append',default =[] )
    pre_args ,_ =pre_parser .parse_known_args ()
    apply_experiment_config (CONFIG ,pre_args .config ,pre_args .config_overrides )
    parser =argparse .ArgumentParser (description ="Run the meta-learning model training script.")
    default_config =CONFIG
    parser .add_argument ('--checkpoint_dir',type =str ,default ='./checkpoints',help ="Directory to save checkpoints and training artifacts.")
    parser .add_argument ('--config',type =str ,default =pre_args .config ,help ="YAML experiment config (supports an extends key).")
    parser .add_argument ('--set',dest ='config_overrides',action ='append',default =pre_args .config_overrides ,help ="Override a config value with dotted.path=value.")
    parser .add_argument ('--resume',action ='store_true',help ="Resume training from the latest checkpoint in --checkpoint_dir.")
    parser .add_argument ('--stop_after_epoch',type =int ,default =None ,help ="Stop training after this specific epoch number completes (e.g., 1).")
    parser .add_argument ('--cleanup_checkpoints',action ='store_true',help ="Remove all intermediate checkpoints after training, keeping only the last one.")
    parser .add_argument ('--embedding_strategy',type =str ,default =default_config ['embedding_strategy'],choices =['learned','pretrained'],help =f"Embedding strategy to use. (default: {default_config ['embedding_strategy']})")
    parser .add_argument ('--num_experts',type =int ,default =default_config ['moe_settings']['num_experts'],help =f"Number of experts (MoE > 1). (default: {default_config ['moe_settings']['num_experts']})")
    parser .add_argument ('--d_model',type =int ,default =default_config ['d_model'],help =f"Model dimension. (default: {default_config ['d_model']})")
    parser .add_argument ('--n_heads',type =int ,default =default_config ['n_heads'],help =f"Number of attention heads. (default: {default_config ['n_heads']})")
    parser .add_argument ('--n_layers',type =int ,default =default_config ['n_layers'],help =f"Number of transformer layers. (default: {default_config ['n_layers']})")
    parser .add_argument ('--dropout',type =float ,default =default_config ['dropout'],help =f"Dropout rate. (default: {default_config ['dropout']})")
    parser .add_argument ('--lr',type =float ,default =default_config ['lr'],help =f"Learning rate. (default: {default_config ['lr']})")
    parser .add_argument ('--epochs',type =int ,default =default_config ['epochs'],help =f"Number of epochs. (default: {default_config ['epochs']})")
    parser .add_argument ('--episodes_per_epoch',type =int ,default =default_config ['episodes_per_epoch'],help =f"Episodes per epoch. (default: {default_config ['episodes_per_epoch']})")
    parser .add_argument ('--training_strategy',type =str ,default =default_config ['training_strategy'],choices =['episodic','retrieval','mixed'],help =f"Training strategy. (default: {default_config ['training_strategy']})")
    parser .add_argument ('--episodic_label_shuffle',type =str ,default =default_config ['episodic_label_shuffle'],choices =['no','yes','mixed'],help =f"Episodic label shuffle strategy. (default: {default_config ['episodic_label_shuffle']})")
    parser .add_argument ('--retrieval_train_k',type =int ,default =default_config ['retrieval_train_k'],help =f"k-value for retrieval training. (default: {default_config ['retrieval_train_k']})")
    parser .add_argument ('--num_shots_range',type =int ,nargs =2 ,default =default_config ['num_shots_range'],help =f"Min and max k-shots for training. (default: {default_config ['num_shots_range'][0 ]} {default_config ['num_shots_range'][1 ]})")
    parser .add_argument ('--num_queries',type =int ,default =default_config ['num_queries'],help =f"Number of queries per class in episodes. (default: {default_config ['num_queries']})")
    args =parser .parse_args ()
    CONFIG ['embedding_strategy']=args .embedding_strategy
    CONFIG ['moe_settings']['num_experts']=args .num_experts
    CONFIG ['d_model']=args .d_model
    CONFIG ['n_heads']=args .n_heads
    CONFIG ['n_layers']=args .n_layers
    CONFIG ['dropout']=args .dropout
    CONFIG ['lr']=args .lr
    CONFIG ['epochs']=args .epochs
    CONFIG ['episodes_per_epoch']=args .episodes_per_epoch
    CONFIG ['training_strategy']=args .training_strategy
    CONFIG ['episodic_label_shuffle']=args .episodic_label_shuffle
    CONFIG ['retrieval_train_k']=args .retrieval_train_k
    CONFIG ['num_shots_range']=tuple (args .num_shots_range )
    CONFIG ['num_queries']=args .num_queries
    training_log_sets =resolve_training_log_sets (CONFIG )
    print ("--- 🚀 Initializing Training Run with Configuration ---")
    print (f"  - Embedding Strategy: {CONFIG ['embedding_strategy']}")
    print (f"  - Num Experts (MoE): {CONFIG ['moe_settings']['num_experts']}")
    print (f"  - Training Strategy: {CONFIG ['training_strategy']}")
    print (f"  - Epochs: {CONFIG ['epochs']}")
    print (f"  - Learning Rate: {CONFIG ['lr']}")
    print (f"  - d_model: {CONFIG ['d_model']}, n_heads: {CONFIG ['n_heads']}, n_layers: {CONFIG ['n_layers']}")
    print (f"  - Num Shots Range: {CONFIG ['num_shots_range']}")
    print ("  - Training Log Sets:")
    for log_set in training_log_sets :
        print (
        f"    - {log_set ['name']}: {len (log_set ['log_paths'])} logs, "
        f"epochs [{log_set ['start_epoch']}, {log_set ['end_epoch']}]"
        )
    print (f"  - Checkpoint Directory: {args .checkpoint_dir }")
    print (f"  - Resume Training: {args .resume }")
    if args .stop_after_epoch :
        print (f"  - Stop After Epoch: {args .stop_after_epoch }")
    print (f"  - Cleanup Checkpoints: {args .cleanup_checkpoints }")
    classification_profile ,classification_weights =resolve_classification_objective (CONFIG )
    regression_profile ,regression_weights =resolve_regression_metric_weights (CONFIG .get ('fmv3_head',{})or {})
    print (f"  - Classification Objective: {classification_profile} {classification_weights}")
    print (f"  - Regression Objective: {regression_profile} {regression_weights}")
    seed =int (CONFIG .get ('seed',42 ))
    torch .manual_seed (seed )
    np .random .seed (seed )
    random .seed (seed )
    device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")
    print (f"Using device: {device }")
    strategy =CONFIG ['embedding_strategy']
    print (f"--- Running with embedding strategy: '{strategy }' ---")
    checkpoint_dir =args .checkpoint_dir
    os .makedirs (checkpoint_dir ,exist_ok =True )
    artifacts_path =os .path .join (checkpoint_dir ,'training_artifacts.pth')
    config_path =os .path .join (checkpoint_dir ,'training_config.pth')
    start_epoch =0
    latest_checkpoint_path =None
    if args .resume :
        print (f"--- 🔄 Resuming training from {checkpoint_dir } ---")
        checkpoints =[f for f in os .listdir (checkpoint_dir )if f .startswith ('model_epoch_')and f .endswith ('.pth')]
        if checkpoints :
            latest_checkpoint =sorted (checkpoints ,key =lambda f :int (re .search (r'(\d+)',f ).group (1 )))[-1 ]
            latest_epoch_num =int (re .search (r'(\d+)',latest_checkpoint ).group (1 ))
            start_epoch =latest_epoch_num
            latest_checkpoint_path =os .path .join (checkpoint_dir ,latest_checkpoint )
            print (f"Found latest checkpoint: {latest_checkpoint }. Resuming from epoch {start_epoch +1 }.")
        else :
            print ("No checkpoints found. Starting from epoch 1.")
        if os .path .exists (config_path ):
            print (f"Loading saved config from {config_path } for resume...")
            saved_config =torch .load (config_path )
            CONFIG ['moe_settings']=saved_config ['moe_settings']
            CONFIG ['embedding_strategy']=saved_config ['embedding_strategy']
            CONFIG ['d_model']=saved_config ['d_model']
            CONFIG ['n_heads']=saved_config ['n_heads']
            CONFIG ['n_layers']=saved_config ['n_layers']
            CONFIG ['dropout']=saved_config ['dropout']
            CONFIG ['pretrained_settings']=saved_config .get ('pretrained_settings',CONFIG ['pretrained_settings'])
            CONFIG ['learned_settings']=saved_config .get ('learned_settings',CONFIG ['learned_settings'])
        else :
            print ("No saved config found for resume, using current.")
    else :
        print (f"--- 🗑️ Starting new training run. Clearing {checkpoint_dir } ---")
        for filename in os .listdir (checkpoint_dir ):
            file_path =os .path .join (checkpoint_dir ,filename )
            try :
                if os .path .isfile (file_path )or os .path .islink (file_path ):
                    os .unlink (file_path )
                elif os .path .isdir (file_path ):
                    shutil .rmtree (file_path )
            except Exception as e :
                print (f'Failed to delete {file_path }. Reason: {e }')
    print ("Saving config...")
    torch .save (CONFIG ,config_path )
    save_yaml_config (CONFIG ,os .path .join (checkpoint_dir ,'training_config.yaml'))
    print ("--- Phase 1: Preparing Training Data ---")
    loader =init_loader (CONFIG )
    all_training_log_paths =combined_training_log_paths (training_log_sets )
    if args .resume and os .path .exists (artifacts_path ):
        print (f"Loading existing artifacts from {artifacts_path }...")
        loader .load_training_artifacts (artifacts_path )
    else :
        print ("Fitting new loader and saving artifacts...")
        loader .fit (all_training_log_paths )
        loader .save_training_artifacts (artifacts_path )
    print ("\n--- Phase 2: Creating Training Tasks ---")
    diagnostics_config =CONFIG .get ('training_diagnostics',{})or {}
    diagnostics_enabled =diagnostics_config .get ('enabled',False )
    training_tasks =[]
    validation_tasks =[]if diagnostics_enabled else None
    validation_manifest =[]
    for set_index ,log_set in enumerate (training_log_sets ):
        print (f"Preparing training log set '{log_set ['name']}'...")
        transformed_logs =loader .transform (log_set ['log_paths'])
        set_tasks ={
        'classification':[get_task_data (log ,'classification')for log in transformed_logs .values ()],
        'regression':[get_task_data (log ,'regression')for log in transformed_logs .values ()]
        }
        set_validation_tasks =None
        if diagnostics_enabled :
            training_log_names =[
            os .path .basename (str (path ))
            for path in log_set ['log_paths'].values ()
            ]
            set_tasks ,set_validation_tasks ,set_manifest =split_training_tasks_by_case (
            set_tasks ,
            diagnostics_config .get ('validation_fraction',0.10 ),
            seed +100000 *set_index ,
            log_names =training_log_names ,
            )
            for row in set_manifest :
                row ['log_set']=log_set ['name']
            validation_manifest .extend (set_manifest )
        task_set ={
        'name':log_set ['name'],
        'start_epoch':log_set ['start_epoch'],
        'end_epoch':log_set ['end_epoch'],
        'tasks':set_tasks ,
        }
        training_tasks .append (task_set )
        if diagnostics_enabled :
            validation_tasks .append ({**task_set ,'tasks':set_validation_tasks })
    if diagnostics_enabled :
        save_validation_manifest (checkpoint_dir ,validation_manifest ,CONFIG )
        print ("🔬 Created deterministic case-disjoint training/validation split.")
        for row in validation_manifest :
            print (
            f"  - {row ['log_set']}/{row ['log']}: "
            f"{row ['validation_cases']}/{row ['total_cases']} "
            "cases held out"
            )
    print ("\n--- Phase 3: Initializing Model ---")
    model =create_model (CONFIG ,loader ,device )
    if latest_checkpoint_path :
        print (f"Loading model weights from {latest_checkpoint_path }...")
        load_state_dict_compatible (model ,torch .load (latest_checkpoint_path ,map_location =device ))
    trainable_scope =str (CONFIG .get ('trainable_scope','all')).lower ()
    trainable_parameters =configure_trainable_scope (model ,trainable_scope )
    if trainable_scope in {
    'time_transform','regression_gate','expert_confidence','expert_routing_confidence','temporal_joint','prefix_attention','temporal_prefix_joint',
    'classification_adapter','regression_refinement','classification_example_selector',
    'regression_example_selector','example_selectors'}:
        print ("🔒 Frozen backbone; training only the selected adapter modules.")
        print (f"  - Trainable tensors: {len (trainable_parameters )}")
    print (f"Model has {sum (p .numel ()for p in model .parameters ()if p .requires_grad ):,} trainable parameters.")
    print ("\n--- Phase 4: Starting Model Training ---")
    if not CONFIG .get ('training_enabled',True ):
        torch .save (model .state_dict (),os .path .join (checkpoint_dir ,'model_epoch_0.pth'))
        print ("Training disabled by config; saved initialized model_epoch_0.pth.")
        return
    train (
    model ,
    training_tasks ,
    loader ,
    CONFIG ,
    checkpoint_dir =checkpoint_dir ,
    resume_epoch =start_epoch ,
    stop_after_epoch =args .stop_after_epoch ,
    cleanup_checkpoints =args .cleanup_checkpoints ,
    validation_tasks =validation_tasks ,
    )
    print ("\n✅ Training complete. Run 'testing.py' to evaluate.")
if __name__ =='__main__':
    main ()
