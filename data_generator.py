import pandas as pd
import random
import os
import numpy as np
import torch
import hashlib
import math
from itertools import chain
from sklearn .metrics .pairwise import cosine_similarity
from scipy .optimize import linear_sum_assignment
from config import CONFIG
from config_utils import stable_config_hash
from training_log_sets import combined_training_log_paths, resolve_training_log_sets

try:
    import pm4py
except ImportError:  # Model-only imports and tests do not require XES ingestion.
    pm4py = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # Learned embeddings do not require SentenceTransformer.
    SentenceTransformer = None

class XESLogLoader :
    def __init__ (self ,strategy :str ,sbert_model_name :str ='all-MiniLM-L6-v2',
    sbert_model_revision :str |None =None ,
    max_string_length :int =64 ,max_generic_attributes :int =16 ,
    attribute_hash_buckets :int =4096 ):
        self .strategy =strategy
        print (f"Data loader initialized with strategy: '{self .strategy }'")
        self .activity_to_id ={}
        self .char_to_id ={}
        self .training_activity_names =[]
        self .training_activity_embeddings =None
        self .activity_embedding_map ={}
        self .resource_embedding_map ={}
        self .PAD_TOKEN ='<PAD>'
        self .UNK_TOKEN ='<UNK>'
        self .pad_id =0
        self .unk_id =1
        self .max_string_length =max (1 ,int (max_string_length ))
        self .max_generic_attributes =max (0 ,int (max_generic_attributes ))
        self .attribute_hash_buckets =max (32 ,int (attribute_hash_buckets ))
        self .sbert_model =None
        self .sbert_model_name =sbert_model_name
        self .sbert_model_revision =sbert_model_revision
        if self .strategy =='pretrained':
            if SentenceTransformer is None:
                raise RuntimeError(
                "Pretrained strategy requires the sentence-transformers package."
                )
            try :
                model_kwargs ={}
                if sbert_model_revision :model_kwargs ['revision']=sbert_model_revision
                self .sbert_model =SentenceTransformer (sbert_model_name ,**model_kwargs )
                self .sbert_embedding_dim =self .sbert_model .get_sentence_embedding_dimension ()
            except Exception as e :
                raise RuntimeError (f"Pretrained strategy requires SentenceTransformer: {e }")
            self .pad_embedding =np .zeros (self .sbert_embedding_dim ,dtype =np .float32 )
    def fit (self ,training_log_paths :dict ,activity_key ='concept:name',resource_key ='org:resource'):
        print (f"Fitting on training data (strategy: '{self .strategy }')...")
        if pm4py is None:
            raise RuntimeError("XES ingestion requires the pm4py package.")
        all_activities ,all_resources ,all_lifecycles =set (),set (),set ()
        for _ ,path in training_log_paths .items ():
            if not os .path .exists (path ):continue
            try :
                df =pm4py .read_xes (path )
                if not resource_key in df :
                    df [resource_key ]="Unknown"
                all_activities .update (df [activity_key ].unique ())
                all_resources .update (df [resource_key ].fillna ('Unknown').unique ())
                lifecycle_key ='lifecycle:transition'
                if lifecycle_key in df :
                    all_lifecycles .update (
                    df [lifecycle_key ].fillna ('Unknown').astype (str ).unique ())
            except Exception as e :
                print (f"❌ Error reading file {path }: {e }")
        if not all_activities :raise ValueError ("No activities found in training logs.")
        self .training_activity_names =sorted (list (all_activities ))
        all_resource_names =sorted (list (all_resources ))
        self .activity_to_id ={name :i for i ,name in enumerate (self .training_activity_names )}
        if self .strategy =='learned':
            all_names =all_activities .union (all_resources ,all_lifecycles )
            all_chars =set ("".join (all_names ))
            self .char_to_id ={char :i +2 for i ,char in enumerate (sorted (list (all_chars )))}
            self .char_to_id [self .PAD_TOKEN ]=self .pad_id
            self .char_to_id [self .UNK_TOKEN ]=self .unk_id
            print (f"  - Created character vocabulary of size {len (self .char_to_id )}.")
        elif self .strategy =='pretrained':
            print ("  - Generating and storing embeddings for all training activities...")
            self .training_activity_embeddings =self .sbert_model .encode (
            self .training_activity_names ,show_progress_bar =True ,normalize_embeddings =True
            )
            self .activity_embedding_map ={name :emb for name ,emb in zip (
            self .training_activity_names ,self .training_activity_embeddings
            )}
            print ("  - Generating and storing embeddings for all training resources...")
            resource_embeddings =self .sbert_model .encode (
            all_resource_names ,show_progress_bar =True ,normalize_embeddings =True
            )
            self .resource_embedding_map ={name :emb for name ,emb in zip (
            all_resource_names ,resource_embeddings
            )}
        print ("✅ Fit complete.")
        return self
    def transform (self ,log_paths :dict ,case_id_key ='case:concept:name',activity_key ='concept:name',
    timestamp_key ='time:timestamp',resource_key ='org:resource',cost_key ='amount'):
        if not self .training_activity_names :raise RuntimeError ("Loader has not been fitted.")
        if pm4py is None:
            raise RuntimeError("XES ingestion requires the pm4py package.")
        print (f"\nTransforming logs: {list (log_paths .keys ())}")
        processed_logs ={}
        for name ,path in log_paths .items ():
            if not os .path .exists (path ):continue
            frame =pm4py .read_xes (path )
            processed_logs .update (self .transform_dataframes (
            {name :frame },case_id_key ,activity_key ,timestamp_key ,resource_key ,cost_key
            ))
        return processed_logs
    def transform_dataframes (self ,frames :dict ,case_id_key ='case:concept:name',activity_key ='concept:name',
    timestamp_key ='time:timestamp',resource_key ='org:resource',cost_key ='amount',activity_names_by_log =None ):
        if not self .training_activity_names :raise RuntimeError ("Loader has not been fitted.")
        if not frames :return {}
        processed_logs ={}
        for name ,source_df in frames .items ():
            group_df =source_df .copy ()
            if resource_key not in group_df :group_df [resource_key ]=None
            group_df ['orig_index']=np .arange (len (group_df ))
            raw_traces =self ._convert_df_to_raw_traces (group_df ,case_id_key ,activity_key ,timestamp_key ,
            resource_key ,cost_key )
            if self .strategy =='learned':
                names =activity_names_by_log .get (name )if activity_names_by_log else None
                processed_logs [name ]=self ._transform_learned (raw_traces ,names )
            else :
                processed_logs [name ]=self ._transform_pretrained (group_df ,raw_traces ,activity_key ,resource_key )
        print ("✅ Transformation complete.")
        return processed_logs
    def _transform_learned (self ,raw_traces ,activity_names =None ):
        all_activities_in_log =activity_names or set (event ['activity']for trace in raw_traces for event in trace )
        local_activity_to_id ={name :i for i ,name in enumerate (sorted (list (all_activities_in_log )))}
        log_with_strings =[]
        for raw_trace in raw_traces :
            processed_trace =[]
            for event in raw_trace :
                processed_event ={
                'activity_name':event ['activity'],
                'resource_name':event ['resource'],
                'activity_id':local_activity_to_id .get (event ['activity']),
                'cost':event ['cost'],
                'time_from_start':event ['time_from_start'],
                'time_from_previous':event ['time_from_previous'],
                'timestamp':event ['timestamp'],'case_id':event ['case_id']
                }
                processed_event .update (self ._context_features (event ))
                processed_event ['activity_char_ids']=self ._char_ids (event ['activity'])
                processed_event ['resource_char_ids']=self ._char_ids (event ['resource'])
                processed_event ['lifecycle_char_ids']=self ._char_ids (event ['lifecycle'])
                processed_trace .append (processed_event )
            log_with_strings .append (processed_trace )
        return log_with_strings
    def _convert_df_to_raw_traces (self ,df ,case_id_key ,activity_key ,timestamp_key ,resource_key ,cost_key ):
        raw_log =[]
        df [timestamp_key ]=pd .to_datetime (
        df [timestamp_key ],errors ='coerce',utc =True
        ).dt .tz_convert (None )
        df =df .dropna (subset =[timestamp_key ])
        if cost_key not in df :
            df [cost_key ]=np .nan
        # Sorting once and iterating over plain tuples is materially faster than
        # sorting every case and constructing a pandas Series for every event.
        lifecycle_key ='lifecycle:transition'
        if lifecycle_key not in df :
            df [lifecycle_key ]=None
        ignored ={
        case_id_key ,activity_key ,timestamp_key ,resource_key ,cost_key ,
        lifecycle_key ,'log_name','orig_index','index'
        }
        attribute_columns =sorted (
        column for column in df .columns
        if column not in ignored and not str (column ).startswith ('_fmv3_')
        and not df [column ].isna ().all ()
        )[:self .max_generic_attributes ]
        stable_key ='orig_index'if 'orig_index'in df else '_fmv3_original_order'
        if stable_key not in df :
            df [stable_key]=np .arange (len (df ))
        columns =[case_id_key ,activity_key ,timestamp_key ,resource_key ,cost_key ,
        lifecycle_key ,stable_key ,*attribute_columns ]
        ordered =df .sort_values (
        [case_id_key ,timestamp_key ,stable_key ],kind ='mergesort'
        )[columns ]
        for case_id ,trace_df in ordered .groupby (case_id_key ,sort =False ):
            if trace_df .empty :continue
            rows =trace_df .itertuples (index =False ,name =None )
            first =next (rows )
            start_time =first [2 ]
            prev_time =start_time
            trace =[]
            for event in chain ((first ,),rows ):
                _ ,activity ,current_time ,resource ,cost_val ,lifecycle ,_ =event [:7 ]
                resource_missing =pd .isna (resource )
                resource ='Unknown'if resource_missing else str (resource )
                cost_missing =not isinstance (cost_val ,(int ,float ,np .number ))or pd .isna (cost_val )
                if not isinstance (cost_val ,(int ,float ,np .number ))or pd .isna (cost_val ):
                    cost_val =0.0
                lifecycle_missing =pd .isna (lifecycle )
                lifecycle ='Unknown'if lifecycle_missing else str (lifecycle )
                attributes =[self ._encode_attribute (lifecycle_key ,lifecycle )]+[
                self ._encode_attribute (column ,value )
                for column ,value in zip (attribute_columns ,event [7 :])
                ]
                event_dict ={
                'case_id':case_id ,'activity':activity ,'timestamp':current_time .timestamp (),
                'resource':resource ,'cost':cost_val ,
                'resource_missing':float (resource_missing ),
                'cost_missing':float (cost_missing ),
                'lifecycle':lifecycle ,
                'lifecycle_missing':float (lifecycle_missing ),
                'calendar_features':self ._calendar_features (current_time ),
                'generic_attributes':attributes ,
                'time_from_start':(current_time -start_time ).total_seconds (),
                'time_from_previous':(current_time -prev_time ).total_seconds (),
                }
                trace .append (event_dict )
                prev_time =current_time
            if trace :raw_log .append (trace )
        return raw_log
    def _transform_pretrained (self ,df ,raw_traces ,activity_key ,resource_key ):
        activity_embedding_map =self .activity_embedding_map .copy ()
        resource_embedding_map =self .resource_embedding_map .copy ()
        current_activities =sorted (list (df [activity_key ].dropna ().unique ()))
        resources_in_log =sorted (list (df [resource_key ].fillna ('Unknown').unique ()))
        # Labels are always log-local. Semantic similarity is an input feature,
        # never the identity of the target class.
        final_activity_id_map ={name :index for index ,name in enumerate (current_activities )}
        unseen_activities_for_embed =[name for name in current_activities if name not in activity_embedding_map ]
        if unseen_activities_for_embed :
            print (f"  - Encoding {len (unseen_activities_for_embed )} new activity embeddings...")
            new_act_embs =self .sbert_model .encode (unseen_activities_for_embed ,normalize_embeddings =True )
            for name ,emb in zip (unseen_activities_for_embed ,new_act_embs ):
                activity_embedding_map [name ]=emb
        unseen_resources_for_embed =[name for name in resources_in_log if name not in resource_embedding_map ]
        if unseen_resources_for_embed :
            print (f"  - Encoding {len (unseen_resources_for_embed )} new resource embeddings...")
            new_res_embs =self .sbert_model .encode (unseen_resources_for_embed ,normalize_embeddings =True )
            for name ,emb in zip (unseen_resources_for_embed ,new_res_embs ):
                resource_embedding_map [name ]=emb
        log_with_embeddings =[]
        unknown_resource_emb =resource_embedding_map .get ('Unknown',self .pad_embedding )
        for raw_trace in raw_traces :
            processed_trace =[]
            for event in raw_trace :
                processed_trace .append ({
                'activity_embedding':activity_embedding_map .get (event ['activity'],self .pad_embedding ),
                'resource_embedding':resource_embedding_map .get (event ['resource'],unknown_resource_emb ),
                'activity_id':final_activity_id_map .get (event ['activity'],-100 ),
                'cost':event ['cost'],'time_from_start':event ['time_from_start'],
                'time_from_previous':event ['time_from_previous'],
                'timestamp':event ['timestamp'],'case_id':event ['case_id']
                })
                processed_trace [-1 ].update (self ._context_features (event ))
            log_with_embeddings .append (processed_trace )
        return log_with_embeddings
    def _char_ids (self ,value ):
        unknown =self .char_to_id .get (self .UNK_TOKEN ,self .unk_id )
        values =tuple (
        self .char_to_id .get (char ,unknown )
        for char in str (value )[:self .max_string_length ]
        )
        return values +(0 ,)*(self .max_string_length -len (values ))
    def _hash_bucket (self ,value ):
        digest =hashlib .blake2b (
        str (value ).encode ('utf-8',errors ='replace'),digest_size =8
        ).digest ()
        return 1 +int .from_bytes (digest ,'little')%(self .attribute_hash_buckets -1 )
    def _encode_attribute (self ,name ,value ):
        try :
            missing =bool (pd .isna (value ))
        except (TypeError ,ValueError ):
            value =str (value )
            missing =False
        if missing :
            type_id ,value_id ,numeric =0 ,0 ,0.0
        elif isinstance (value ,(bool ,np .bool_ )):
            type_id ,value_id ,numeric =3 ,self ._hash_bucket (bool (value )),float (value )
        elif isinstance (value ,(int ,float ,np .number )):
            type_id ,value_id ,numeric =1 ,0 ,float (value )
        else :
            type_id ,value_id ,numeric =2 ,self ._hash_bucket (value ),0.0
        return (self ._hash_bucket (name ),type_id ,value_id ,numeric ,float (missing ))
    @staticmethod
    def _calendar_features (timestamp ):
        hour =timestamp .hour +timestamp .minute /60.0 +timestamp .second /3600.0
        weekday =timestamp .weekday ()
        return (
        math .sin (2 *math .pi *hour /24.0 ),
        math .cos (2 *math .pi *hour /24.0 ),
        math .sin (2 *math .pi *weekday /7.0 ),
        math .cos (2 *math .pi *weekday /7.0 ),
        float (weekday >=5 ),
        )
    @staticmethod
    def _context_features (event ):
        return {
        'resource_missing':event ['resource_missing'],
        'cost_missing':event ['cost_missing'],
        'lifecycle_name':event ['lifecycle'],
        'lifecycle_missing':event ['lifecycle_missing'],
        'calendar_features':event ['calendar_features'],
        'generic_attributes':event ['generic_attributes'],
        }
    def save_training_artifacts (self ,path ,metadata =None ):
        artifacts ={'format_version':2 ,'strategy':self .strategy ,
        'activity_to_id':self .activity_to_id ,
        'training_activity_names':self .training_activity_names ,
        'metadata':dict (metadata or {})}
        if self .strategy =='pretrained':
            artifacts ['training_activity_embeddings']=self .training_activity_embeddings
            artifacts ['activity_embedding_map']=self .activity_embedding_map
            artifacts ['resource_embedding_map']=self .resource_embedding_map
            artifacts ['pad_embedding']=self .pad_embedding
            artifacts ['sbert_model_name']=self .sbert_model_name
            artifacts ['sbert_model_revision']=self .sbert_model_revision
        elif self .strategy =='learned':
            artifacts ['char_to_id']=self .char_to_id
        artifacts ['activity_vocabulary_sha256']=stable_config_hash (
        sorted ((str (key ),int (value ))for key ,value in self .activity_to_id .items ()))
        artifacts ['character_vocabulary_sha256']=(
        stable_config_hash (sorted ((str (key ),int (value ))for key ,value in self .char_to_id .items ()))
        if self .strategy =='learned'else None )
        torch .save (artifacts ,path )
        print (f"💾 Training artifacts for '{self .strategy }' strategy saved to {path }")
    def load_training_artifacts (self ,path ,expected_metadata =None ):
        if not os .path .exists (path ):raise FileNotFoundError (f"Artifacts file not found at {path }.")
        artifacts =torch .load (path ,weights_only =False )
        if artifacts ['strategy']!=self .strategy :
            raise ValueError (
            f"Artifact strategy '{artifacts ['strategy']}' does not match loader strategy '{self .strategy }'.")
        self .activity_to_id =artifacts ['activity_to_id']
        self .training_activity_names =artifacts ['training_activity_names']
        actual_activity_hash =stable_config_hash (
        sorted ((str (key ),int (value ))for key ,value in self .activity_to_id .items ()))
        saved_activity_hash =artifacts .get ('activity_vocabulary_sha256')
        if saved_activity_hash and saved_activity_hash !=actual_activity_hash :
            raise ValueError ("Training artifact activity vocabulary hash mismatch.")
        if self .strategy =='pretrained':
            required ={
            'training_activity_embeddings','activity_embedding_map',
            'resource_embedding_map','pad_embedding','sbert_model_name',
            'sbert_model_revision',
            }
            missing =sorted (required -set (artifacts ))
            if missing :
                raise ValueError (
                "Pretrained artifacts are incomplete and would require silent "
                "re-encoding: "+", ".join (missing ))
            self .training_activity_embeddings =artifacts ['training_activity_embeddings']
            self .activity_embedding_map =artifacts ['activity_embedding_map']
            self .resource_embedding_map =artifacts ['resource_embedding_map']
            self .pad_embedding =artifacts ['pad_embedding']
            if artifacts ['sbert_model_name']!=self .sbert_model_name or (
            artifacts ['sbert_model_revision']!=self .sbert_model_revision ):
                raise ValueError (
                "Pretrained artifact model name/revision does not match the "
                "configured SentenceTransformer."
                )
        elif self .strategy =='learned':
            self .char_to_id =artifacts ['char_to_id']
            actual_character_hash =stable_config_hash (
            sorted ((str (key ),int (value ))for key ,value in self .char_to_id .items ()))
            saved_character_hash =artifacts .get ('character_vocabulary_sha256')
            if saved_character_hash and saved_character_hash !=actual_character_hash :
                raise ValueError ("Training artifact character vocabulary hash mismatch.")
        if expected_metadata is not None :
            saved_metadata =artifacts .get ('metadata')
            if not saved_metadata :
                raise ValueError (
                "Training artifacts lack reproducibility metadata; refusing "
                "incompatible initialization."
                )
            mismatches =sorted (
            key for key ,value in expected_metadata .items ()
            if saved_metadata .get (key )!=value )
            if mismatches :
                raise ValueError (
                "Training artifact metadata mismatch at: "+", ".join (mismatches ))
        print (f"✅ Training artifacts loaded successfully from {path }")
def _summarize_processed_log (log_name ,log ,max_traces =2 ,max_events =5 ):
    total_events =sum (len (trace )for trace in log )
    case_ids ={trace [0 ].get ('case_id')for trace in log if trace }
    print (f"\nLog '{log_name }': traces={len (log )}, events={total_events }, cases={len (case_ids )}")
    if not log :
        return
    for t_idx ,trace in enumerate (log [:max_traces ],start =1 ):
        if not trace :
            continue
        case_id =trace [0 ].get ('case_id')
        print (f"  Trace {t_idx }: case_id={case_id }, events={len (trace )}")
        for e_idx ,event in enumerate (trace [:max_events ],start =1 ):
            if 'activity_name'in event :
                print (
                f"    {e_idx }. act={event .get ('activity_name')} | res={event .get ('resource_name')} "
                f"| t={event .get ('timestamp')}"
                )
            else :
                act_emb =event .get ('activity_embedding')
                res_emb =event .get ('resource_embedding')
                act_dim =int (getattr (act_emb ,"shape",[0 ])[0 ])if act_emb is not None else 0
                res_dim =int (getattr (res_emb ,"shape",[0 ])[0 ])if res_emb is not None else 0
                print (
                f"    {e_idx }. activity_id={event .get ('activity_id')} | emb_dim={act_dim } "
                f"| res_dim={res_dim } | t={event .get ('timestamp')}"
                )
if __name__ =="__main__":
    test_mode =CONFIG .get ('test_mode')
    if test_mode !='retrieval_augmented':
        print (f"Note: CONFIG['test_mode'] is '{test_mode }', not 'retrieval_augmented'.")
    print ("\n--- Data Generator Debug: Retrieval-Augmented Test Logs ---")
    print (f"Embedding strategy: {CONFIG .get ('embedding_strategy')}")
    train_logs =combined_training_log_paths (resolve_training_log_sets (CONFIG ))
    test_logs =CONFIG .get ('log_paths',{}).get ('testing',{})
    print (f"Training logs: {list (train_logs .keys ())}")
    print (f"Testing logs: {list (test_logs .keys ())}")
    loader =XESLogLoader (
    strategy =CONFIG ['embedding_strategy'],
    sbert_model_name =CONFIG ['pretrained_settings']['sbert_model']
    )
    loader .fit (train_logs )
    processed_test_logs =loader .transform (test_logs )
    for log_name ,log in processed_test_logs .items ():
        _summarize_processed_log (log_name ,log )
