import numpy as np
import torch
import os
import argparse
import shutil
import re
import random
import json
import hashlib
from pathlib import Path
from config import CONFIG
from config_utils import (
apply_experiment_config ,save_yaml_config ,validate_exact_resume_config ,
validate_run_configuration ,stable_config_hash ,preprocessing_config_hash ,
model_architecture_config_hash ,
)
from utils .data_utils import get_classification_and_regression_tasks
from utils .model_utils import init_loader ,create_model ,load_state_dict_compatible, extract_model_state_dict
from utils .parameter_utils import configure_trainable_scope
from training import train
from training_debug import save_validation_manifest, split_training_tasks_by_case
from metric_objectives import resolve_classification_objective, resolve_regression_metric_weights
from training_log_sets import (
combined_training_log_paths ,resolve_training_log_sets ,
save_training_log_manifest ,training_log_manifest ,
validate_training_evaluation_disjointness ,
)


def _clear_training_outputs(checkpoint_dir):
    """Start a fresh optimizer trajectory without retaining stale states."""
    for filename in os .listdir (checkpoint_dir ):
        file_path =os .path .join (checkpoint_dir ,filename )
        try :
            if os .path .isfile (file_path )or os .path .islink (file_path ):
                os .unlink (file_path )
            elif os .path .isdir (file_path ):
                shutil .rmtree (file_path )
        except Exception as error :
            raise RuntimeError (
            f"Failed to clear prior training output {file_path }: {error }"
            )from error


def _file_sha256(path):
    digest =hashlib .sha256 ()
    with open (path ,'rb')as handle :
        for block in iter (lambda :handle .read (1024 *1024 ),b''):
            digest .update (block )
    return digest .hexdigest ()


def _reproducibility_metadata(config, corpus_manifest):
    return {
    'training_log_manifest_sha256':corpus_manifest ['manifest_sha256'],
    'preprocessing_sha256':preprocessing_config_hash (config ),
    'model_architecture_sha256':model_architecture_config_hash (config ),
    }


def _save_run_manifest(checkpoint_dir, config, corpus_manifest, metadata):
    manifest_path =Path (checkpoint_dir ,'training_run_manifest.json')
    if str (config .get ('run_mode','train')).lower ()=='resume':
        if not manifest_path .is_file ():
            raise FileNotFoundError (
            "Exact resume requires training_run_manifest.json with corpus and "
            "architecture fingerprints.")
        existing =json .loads (manifest_path .read_text (encoding ='utf-8'))
        mismatches =sorted (
        key for key ,value in metadata .items ()if existing .get (key )!=value )
        if mismatches :
            raise ValueError (
            "Exact resume reproducibility metadata differs at: "+", ".join (mismatches ))
        return
    payload ={
    'schema_version':1,
    'resolved_config_sha256':stable_config_hash (config ),
    **metadata ,
    'source_checkpoint':None,
    'source_artifacts':None,
    }
    if str (config .get ('run_mode','train')).lower ()=='initialize':
        checkpoint =os .fspath (config ['initialize_from_checkpoint'])
        artifacts =os .fspath (config ['initialize_from_artifacts'])
        payload ['source_checkpoint']={
        'path':checkpoint ,'sha256':_file_sha256 (checkpoint )}
        payload ['source_artifacts']={
        'path':artifacts ,'sha256':_file_sha256 (artifacts )}
    manifest_path .write_text (
    json .dumps (payload ,indent =2 ,sort_keys =True )+'\n',encoding ='utf-8')


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
    parser .add_argument ('--initialize_from',type =str ,default =None ,help ="Initialize model weights from a checkpoint while resetting optimizer/scheduler state.")
    parser .add_argument ('--initialize_artifacts',type =str ,default =None ,help ="Loader artifacts associated with --initialize_from.")
    parser .add_argument ('--source_epoch',type =int ,default =None ,help ="Epoch represented by --initialize_from.")
    parser .add_argument ('--additional_epochs',type =int ,default =None ,help ="New epochs to execute after --source_epoch.")
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
    if args .resume and args .initialize_from :
        parser .error ('--resume and --initialize_from are mutually exclusive')
    if args .resume :
        CONFIG ['run_mode']='resume'
    elif args .initialize_from :
        CONFIG ['run_mode']='initialize'
        CONFIG ['initialize_from_checkpoint']=args .initialize_from
        if args .initialize_artifacts is not None :
            CONFIG ['initialize_from_artifacts']=args .initialize_artifacts
        if args .source_epoch is not None :CONFIG ['source_epoch']=args .source_epoch
        if args .additional_epochs is not None :
            CONFIG ['additional_epochs']=args .additional_epochs
            if args .source_epoch is not None :
                CONFIG ['epochs']=args .source_epoch +args .additional_epochs
    run_mode =validate_run_configuration (CONFIG )
    if run_mode =='assemble':
        validate_run_configuration (CONFIG ,check_checkpoint_paths =True )
        from merge_task_isolated_checkpoints import assemble_from_config
        output ,classification_keys ,regression_keys =assemble_from_config (CONFIG )
        print (
        f"Assembled {output} from {len (classification_keys)} classification and "
        f"{len (regression_keys)} regression tensors."
        )
        return
    training_log_sets =resolve_training_log_sets (CONFIG )
    validate_training_evaluation_disjointness (CONFIG ,training_log_sets )
    corpus_manifest =training_log_manifest (training_log_sets )
    artifact_metadata =_reproducibility_metadata (CONFIG ,corpus_manifest )
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
    print (f"  - Run Mode: {run_mode }")
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
    resume_state =None
    initialized_model_state =None
    if run_mode =='resume':
        print (f"--- 🔄 Resuming training from {checkpoint_dir } ---")
        checkpoints =[f for f in os .listdir (checkpoint_dir )if f .startswith ('training_state_epoch_')and f .endswith ('.pth')]
        if not checkpoints :
            raise FileNotFoundError (
            "Exact resume requires training_state_epoch_N.pth. "
            "Use --initialize_from for a weights-only checkpoint."
            )
        latest_checkpoint =sorted (checkpoints ,key =lambda f :int (re .search (r'(\d+)',f ).group (1 )))[-1 ]
        resume_state =torch .load (
        os .path .join (checkpoint_dir ,latest_checkpoint ),
        map_location ='cpu',weights_only =False )
        validate_exact_resume_config (CONFIG ,resume_state ['config'])
        start_epoch =int (resume_state ['epoch'])
        print (f"Found exact training state: {latest_checkpoint }. Resuming from epoch {start_epoch +1 }.")
        if not os .path .exists (artifacts_path ):
            raise FileNotFoundError (
            f"Exact resume requires the original loader artifacts: {artifacts_path }"
            )
    elif run_mode =='initialize':
        latest_checkpoint_path =os .fspath (CONFIG ['initialize_from_checkpoint'])
        source_artifact_path =os .fspath (CONFIG ['initialize_from_artifacts'])
        if not os .path .isfile (latest_checkpoint_path ):
            raise FileNotFoundError (
            f"Initialization checkpoint not found: {latest_checkpoint_path }"
            )
        if not os .path .isfile (source_artifact_path ):
            raise FileNotFoundError (
            f"Initialization artifacts not found: {source_artifact_path }")
        output_root =Path (checkpoint_dir ).resolve ()
        unsafe_sources =[
        path for path in (latest_checkpoint_path ,source_artifact_path )
        if Path (path ).resolve ().is_relative_to (output_root )]
        if unsafe_sources :
            raise ValueError (
            "Initialization sources must be outside the output checkpoint "
            "directory because a new trajectory clears that directory: "
            +", ".join (unsafe_sources ))
        print (f"--- Initializing weights from {latest_checkpoint_path } ---")
        checkpoint_match =re .search (r'model_epoch_(\d+)\.pth$',latest_checkpoint_path )
        if checkpoint_match and int (checkpoint_match .group (1 ))!=int (CONFIG ['source_epoch']):
            raise ValueError (
            "initialize source_epoch does not match checkpoint filename: "
            f"{CONFIG ['source_epoch']} != {checkpoint_match .group (1 )}")
        payload =torch .load (
        latest_checkpoint_path ,map_location ='cpu',weights_only =False )
        initialized_model_state =extract_model_state_dict (payload )
        start_epoch =int (CONFIG ['source_epoch'])
        _clear_training_outputs (checkpoint_dir )
    else :
        print (f"--- 🗑️ Starting new training run. Clearing {checkpoint_dir } ---")
        _clear_training_outputs (checkpoint_dir )
    print ("Saving config...")
    _save_run_manifest (
    checkpoint_dir ,CONFIG ,corpus_manifest ,artifact_metadata )
    torch .save (CONFIG ,config_path )
    save_yaml_config (CONFIG ,os .path .join (checkpoint_dir ,'training_config.yaml'))
    save_training_log_manifest (checkpoint_dir ,training_log_sets )
    print ("--- Phase 1: Preparing Training Data ---")
    loader =init_loader (CONFIG )
    all_training_log_paths =combined_training_log_paths (training_log_sets )
    if run_mode =='resume'and os .path .exists (artifacts_path ):
        print (f"Loading existing artifacts from {artifacts_path }...")
        loader .load_training_artifacts (
        artifacts_path ,expected_metadata =artifact_metadata )
    elif run_mode =='initialize':
        source_artifacts =os .fspath (CONFIG ['initialize_from_artifacts'])
        print (f"Loading checkpoint-associated artifacts from {source_artifacts }...")
        expected_source_metadata =dict (artifact_metadata )
        if CONFIG .get ('allow_architecture_change',False ):
            expected_source_metadata .pop ('model_architecture_sha256',None )
        loader .load_training_artifacts (
        source_artifacts ,expected_metadata =expected_source_metadata )
        loader .save_training_artifacts (
        artifacts_path ,metadata =artifact_metadata )
    else :
        print ("Fitting new loader and saving artifacts...")
        loader .fit (all_training_log_paths )
        loader .save_training_artifacts (
        artifacts_path ,metadata =artifact_metadata )
    print ("\n--- Phase 2: Creating Training Tasks ---")
    diagnostics_config =CONFIG .get ('training_diagnostics',{})or {}
    diagnostics_enabled =diagnostics_config .get ('enabled',False )
    training_tasks =[]
    validation_tasks =[]if diagnostics_enabled else None
    validation_manifest =[]
    for set_index ,log_set in enumerate (training_log_sets ):
        print (f"Preparing training log set '{log_set ['name']}'...")
        transformed_logs =loader .transform (log_set ['log_paths'])
        joint_tasks =[
        get_classification_and_regression_tasks (log ,config =CONFIG )
        for log in transformed_logs .values ()
        ]
        set_tasks ={
        'classification':[tasks [0 ]for tasks in joint_tasks ],
        'regression':[tasks [1 ]for tasks in joint_tasks ],
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
        'weight_schedule':log_set .get ('weight_schedule'),
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
    if resume_state is not None :
        print (f"Loading exact model state from epoch {start_epoch }...")
        load_state_dict_compatible (model ,resume_state ['model'])
    elif initialized_model_state is not None :
        print (f"Loading model weights from {latest_checkpoint_path }...")
        load_state_dict_compatible (model ,initialized_model_state )
    trainable_scope =str (CONFIG .get ('trainable_scope','all')).lower ()
    trainable_parameters =configure_trainable_scope (model ,trainable_scope )
    if trainable_scope in {
    'time_transform','regression_gate','expert_confidence','expert_routing_confidence','temporal_joint','prefix_attention','temporal_prefix_joint',
    'classification_adapter','classification_redesign','regression_refinement','classification_example_selector',
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
    resume_state =resume_state ,
    )
    print ("\n✅ Training complete. Run 'testing.py' to evaluate.")
if __name__ =='__main__':
    main ()
