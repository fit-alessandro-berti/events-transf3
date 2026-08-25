import math
import random
from contextlib import nullcontext
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from metric_objectives import classification_metric_objective
from training_debug import (
    average_metric_dicts,
    classification_head_metrics,
    regression_head_metrics,
    tensor_distribution,
)
from utils.retrieval_utils import find_knn_indices, class_diverse_topk_indices
from utils.data_utils import sample_task_batch


def _step_result(loss, task, diagnostics, return_diagnostics):
    if return_diagnostics:
        return loss, task, diagnostics
    return loss, task


def _encode_case_ids_to_int(case_ids_np: np.ndarray) -> torch.Tensor:
    """case_ids may be strings/objects -> map to contiguous ints for tensor ops."""
    uniq = list(dict.fromkeys(case_ids_np.tolist()))
    mapping = {cid: i for i, cid in enumerate(uniq)}
    return torch.tensor([mapping[cid] for cid in case_ids_np.tolist()], dtype=torch.long)


def select_mixed_negative_indices(
    query_embedding,
    candidate_embeddings,
    eligible_mask,
    k,
    *,
    pool_factor=4,
    random_fraction=0.25,
    generator=None,
):
    """Mix nearest and randomly sampled hard-pool negatives without duplicates."""
    k =max (int (k ),0 )
    if k ==0 :return torch .empty (0 ,dtype =torch .long ,device =candidate_embeddings .device )
    eligible_mask =torch .as_tensor (
    eligible_mask ,dtype =torch .bool ,device =candidate_embeddings .device ).reshape (-1 )
    candidate_indices =torch .where (eligible_mask )[0 ]
    if candidate_indices .numel ()==0 :return candidate_indices
    similarities =(query_embedding @candidate_embeddings .t ()).reshape (-1 )
    ranked =candidate_indices [
    torch .argsort (similarities [candidate_indices ],descending =True )]
    k =min (k ,int (ranked .numel ()))
    pool_size =min (
    int (ranked .numel ()),max (k ,k *max (1 ,int (pool_factor ))))
    hard_pool =ranked [:pool_size ]
    random_count =min (k ,max (0 ,int (round (k *float (random_fraction )))))
    hard_count =k -random_count
    selected =hard_pool [:hard_count ]
    remaining =hard_pool [hard_count :]
    if random_count >0 and remaining .numel ()>0 :
        order =torch .randperm (
        remaining .numel (),device =remaining .device ,generator =generator )
        selected =torch .cat ([selected ,remaining [order [:random_count ]]])
    if selected .numel ()<k :
        selected_set =set (selected .detach ().cpu ().tolist ())
        fill =[index for index in ranked .detach ().cpu ().tolist ()if index not in selected_set ]
        if fill :
            selected =torch .cat ([
            selected ,torch .as_tensor (
            fill [:k -selected .numel ()],dtype =torch .long ,device =selected .device )])
    return selected [:k ]


def _autocast_disabled_for(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=False)
    return nullcontext()


def _sample_balanced_classification_batch(
    task_pool, batch_size, min_per_class=2, max_classes=None,
    case_uniform_fraction=0.5,
):
    """
    Ensure selected classes appear at least min_per_class times, ideally across cases.
    Fall back to random sampling when constraints cannot be met.
    """
    by_label = defaultdict(list)
    for seq, label, case_id in task_pool:
        if label is None:
            continue
        if int(label) == -100:
            continue
        by_label[int(label)].append((seq, int(label), case_id))

    eligible = []
    for label, items in by_label.items():
        if len(items) < min_per_class:
            continue
        if len({cid for _, _, cid in items}) < 2:
            continue
        eligible.append(label)

    if not eligible:
        return sample_task_batch(
            task_pool, batch_size, case_uniform_fraction=case_uniform_fraction
        )

    num_classes = min(len(eligible), max(1, batch_size // min_per_class))
    if max_classes is not None:
        num_classes = min(num_classes, max(1, int(max_classes)))
    chosen_labels = random.sample(eligible, num_classes)

    batch = []
    for label in chosen_labels:
        items = by_label[label]
        by_case = defaultdict(list)
        for item in items:
            by_case[item[2]].append(item)
        cases = list(by_case.keys())
        random.shuffle(cases)

        for case_id in cases[:min_per_class]:
            batch.append(random.choice(by_case[case_id]))

        while sum(1 for item in batch if item[1] == label) < min_per_class:
            case_id = random.choice(cases)
            batch.append(random.choice(by_case[case_id]))

    if chosen_labels:
        label_cycle = chosen_labels[:]
        random.shuffle(label_cycle)
        cycle_idx = 0
        while len(batch) < batch_size:
            lbl = label_cycle[cycle_idx % len(label_cycle)]
            items = by_label[lbl]
            label_cases = list({item[2] for item in items})
            selected_case = random.choice(label_cases)
            batch.append(random.choice([
                item for item in items if item[2] == selected_case
            ]))
            cycle_idx += 1

    return batch[:batch_size]


def _valid_classification_items(task_pool):
    return [item for item in task_pool if item[1] is not None and int(item[1]) != -100]


def _weighted_episode_type(config):
    mix = config.get("fmv3_training", {}).get("episode_mix", {"balanced": 1.0})
    names = [name for name, weight in mix.items() if float(weight) > 0]
    weights = [float(mix[name]) for name in names]
    return random.choices(names, weights=weights, k=1)[0] if names else "balanced"


def _sample_deployment_classification_episode(task_pool, episode_type, config):
    """Build the same case-disjoint support/query shape used at evaluation."""
    valid =_valid_classification_items (task_pool )
    by_case =defaultdict (list )
    for item in valid :by_case [str (item [2 ])].append (item )
    cases =list (by_case )
    if len (cases )<2 :return None
    random .shuffle (cases )
    train_cfg =config .get ("fmv3_training",{})or {}
    query_fraction =min (max (float (train_cfg .get (
    "deployment_query_case_fraction",0.30 )),0.05 ),0.90 )
    query_case_count =min (
    max (1 ,int (round (len (cases )*query_fraction ))),len (cases )-1 )
    query_cases =set (cases [:query_case_count ])
    support_candidates =cases [query_case_count :]

    query_pool =[item for case in query_cases for item in by_case [case ]]
    if not query_pool :return None
    query_limit =max (1 ,int (train_cfg .get (
    "deployment_queries_per_episode",config .get ("num_queries",10 ))))
    if episode_type in {"long_tail","rare_path"}:
        label_counts =Counter (int (item [1 ])for item in valid )
        power =max (float (train_cfg .get ("long_tail_power",1.5 )),0.0 )
        available =query_pool [:]
        query_tasks =[]
        for _ in range (min (query_limit ,len (available ))):
            weights =[
            max (label_counts [int (item [1 ])],1 )**(-power )for item in available ]
            selected =random .choices (range (len (available )),weights =weights ,k =1 )[0 ]
            query_tasks .append (available .pop (selected ))
    else :
        query_tasks =random .sample (query_pool ,min (query_limit ,len (query_pool )))

    if episode_type =="missing_pool_label":
        target =int (random .choice (query_tasks )[1 ])
        matching_queries =[item for item in query_pool if int (item [1 ])==target ]
        if matching_queries :
            query_tasks =[random .choice (matching_queries )]
            support_candidates =[
            case for case in support_candidates
            if all (int (item [1 ])!=target for item in by_case [case ])]
    if not support_candidates :return None

    budgets =train_cfg .get (
    "support_case_budgets",[1 ,2 ,4 ,8 ,16 ,32 ,64 ,128 ,"full"])
    chosen_budget =random .choice (list (budgets )or [1 ])
    if str (chosen_budget ).lower ()=="full":
        budget =len (support_candidates )
    else :budget =max (1 ,int (chosen_budget ))
    support_cases =random .sample (
    support_candidates ,min (budget ,len (support_candidates )))
    support_tasks =[item for case in support_cases for item in by_case [case ]]
    support_cap =int (train_cfg .get ("deployment_support_max_prefixes",0 )or 0 )
    if support_cap >0 and len (support_tasks )>support_cap :
        support_tasks =sample_task_batch (
        support_tasks ,support_cap ,case_uniform_fraction =1.0 )
    if not support_tasks :return None
    if episode_type =="balanced":
        supported_labels ={int (item [1 ])for item in support_tasks }
        balanced_queries =[
        item for item in query_pool if int (item [1 ])in supported_labels ]
        if not balanced_queries :return None
        by_label =defaultdict (list )
        for item in balanced_queries :by_label [int (item [1 ])].append (item )
        query_tasks =[]
        labels =list (by_label )
        random .shuffle (labels )
        while len (query_tasks )<min (query_limit ,len (balanced_queries )):
            progressed =False
            for label in labels :
                available =[
                item for item in by_label [label ]if item not in query_tasks ]
                if available :
                    query_tasks .append (random .choice (available ))
                    progressed =True
                    if len (query_tasks )>=min (query_limit ,len (balanced_queries )):break
            if not progressed :break
    return support_tasks ,query_tasks


def _frequency_stratified_mean(losses, support_counts):
    if losses .numel ()==0 :return None
    strata =[
    support_counts ==0 ,support_counts ==1 ,
    (support_counts >=2 )&(support_counts <=5 ),support_counts >5 ,
    ]
    means =[losses [mask ].mean ()for mask in strata if bool (mask .any ())]
    return torch .stack (means ).mean ()if means else losses .mean ()


def _deployment_retrieval_ranking_loss(
    query_embeddings, support_embeddings, query_labels, support_labels,
    support_counts, margin,
):
    similarities =query_embeddings @support_embeddings .t ()
    losses ,counts =[],[]
    for row ,target in enumerate (query_labels ):
        positives =support_labels ==target
        negatives =support_labels !=target
        if not bool (positives .any ())or not bool (negatives .any ()):continue
        best_positive =similarities [row ,positives ].max ()
        hard_negative =similarities [row ,negatives ].max ()
        losses .append (F .relu (hard_negative -best_positive +float (margin )))
        counts .append (support_counts [row ])
    if not losses :return None
    return _frequency_stratified_mean (torch .stack (losses ),torch .stack (counts ))


def _activity_context(task, max_order):
    values =[
    int (event ["activity_id"])
    for event in task [0 ]
    if event .get ("activity_id")is not None
    and int (event ["activity_id"])!=-100 ]
    return tuple (values [-max (int (max_order ),1 ):])


def _deployment_structured_probabilities(
    support_tasks, query_tasks, candidate_classes, max_order=3, smoothing=0.5,
):
    """Exact support-only suffix backoff used by the evaluation protocol."""
    classes =[int (label )for label in candidate_classes .tolist ()]
    class_to_column ={label :column for column ,label in enumerate (classes )}
    global_counts =Counter (int (task [1 ])for task in support_tasks )
    tables ={order :defaultdict (Counter )for order in range (1 ,max_order +1 )}
    for task in support_tasks :
        context =_activity_context (task ,max_order )
        target =int (task [1 ])
        for order in range (1 ,min (len (context ),max_order )+1 ):
            tables [order ][context [-order :]][target ]+=1
    probabilities ,context_support =[],[]
    vocabulary_size =max (len (classes ),1 )
    smoothing =max (float (smoothing ),1e-8 )
    for task in query_tasks :
        context =_activity_context (task ,max_order )
        counts =None
        for order in range (min (len (context ),max_order ),0 ,-1 ):
            candidate =tables [order ].get (context [-order :])
            if candidate :counts =candidate ;break
        if counts is None :
            probabilities .append ([1.0 /vocabulary_size ]*vocabulary_size )
            context_support .append (0.0 )
            continue
        scores =[]
        for label in classes :
            scores .append (
            math .log (counts .get (label ,0 )+smoothing )-
            math .log (global_counts .get (label ,0 )+smoothing *vocabulary_size ))
        row =torch .softmax (torch .tensor (scores ,dtype =torch .float32 ),dim =0 )
        probabilities .append (row .tolist ())
        context_support .append (float (sum (counts .values ())))
    return probabilities ,context_support


def _run_deployment_classification_step(
    model, task_pool, episode_type, config, return_diagnostics,
):
    episode =_sample_deployment_classification_episode (
    task_pool ,episode_type ,config )
    diagnostics ={
    "data/episode_valid":0.0 ,"data/pool_prefixes":float (len (task_pool )),
    "data/deployment_episode":1.0 ,
    }
    task_name =f"retrieval_classification_{episode_type}"
    if episode is None :return _step_result (
    None ,f"{task_name}_empty",diagnostics ,return_diagnostics )
    support_tasks ,query_tasks =episode
    all_tasks =support_tasks +query_tasks
    base =model ._process_batch (
    [item [0 ]for item in all_tasks ],task_type ="classification")
    decision =model .classification_decision_features (base )
    retrieval =F .normalize (
    model .classification_retrieval_features (base ),p =2 ,dim =1 )
    n_support =len (support_tasks )
    support_decision ,query_decision =decision [:n_support ],decision [n_support :]
    support_retrieval ,query_retrieval =retrieval [:n_support ],retrieval [n_support :]
    device =base .device
    support_labels =torch .as_tensor (
    [int (item [1 ])for item in support_tasks ],dtype =torch .long ,device =device )
    query_labels =torch .as_tensor (
    [int (item [1 ])for item in query_tasks ],dtype =torch .long ,device =device )
    candidate_records =sorted (
    list (getattr (task_pool ,"candidate_labels",())or ()),
    key =lambda candidate :int (candidate ["label_id"]),
    )
    if candidate_records :
        candidate_classes =torch .as_tensor ([
        int (candidate ["label_id"])for candidate in candidate_records
        ],dtype =torch .long ,device =device )
    else :
        candidate_classes =torch .unique (torch .cat ([
        support_labels ,query_labels ]),sorted =True )
    candidate_features =model .encode_candidate_labels (candidate_records )
    if (model .semantic_candidate_decoder_enabled and (
    candidate_features is None or candidate_features .size (0 )!=candidate_classes .numel ())):
        return _step_result (
        None ,f"{task_name}_missing_schema",diagnostics ,return_diagnostics )

    support_count_vector =torch .stack ([
    (support_labels ==label ).sum ()for label in candidate_classes ]).float ()
    class_to_column ={int (label ):column for column ,label in enumerate (
    candidate_classes .tolist ())}
    target_columns =torch .as_tensor ([
    class_to_column [int (label )]for label in query_labels .tolist ()
    ],dtype =torch .long ,device =device )
    target_support_counts =support_count_vector [target_columns ]
    k =min (int (config .get ("retrieval_train_k",20 )),n_support )
    policy =str (config .get ("fmv3_training",{}).get (
    "classification_retrieval_policy","class_diverse" )).lower ()
    logits_rows =[None ]*len (query_tasks )
    semantic_rows =[None ]*len (query_tasks )
    process_rows =[None ]*len (query_tasks )
    probability_rows =[None ]*len (query_tasks )
    local_groups =defaultdict (list )
    semantic_candidates =None
    semantic_queries =None
    if candidate_features is not None :
        semantic_queries =F .normalize (
        model .proto_head .semantic_query_projection (query_decision ),p =2 ,dim =1 )
        semantic_candidates =F .normalize (
        model .proto_head .semantic_candidate_projection (
        candidate_features ),p =2 ,dim =1 )
    for row in range (len (query_tasks )):
        similarities =query_retrieval [row ]@support_retrieval .t ()
        eligible =torch .ones (n_support ,dtype =torch .bool ,device =device )
        if episode_type =="missing_local_label":
            eligible &=support_labels !=query_labels [row ]
        eligible_indices =torch .where (eligible )[0 ]
        if eligible_indices .numel ()==0 :continue
        local_k =min (k ,int (eligible_indices .numel ()))
        eligible_scores =similarities [eligible_indices ]
        if policy =="class_diverse":
            semantic_scores =(
            semantic_queries [row ]@semantic_candidates .t ()
            if semantic_candidates is not None else None )
            local_relative =class_diverse_topk_indices (
            eligible_scores ,support_labels [eligible_indices ],local_k ,
            classes_per_shortlist =int (config .get ("fmv3_training",{}).get (
            "class_diverse_shortlist_classes",10 )),
            examples_per_class =int (config .get ("fmv3_training",{}).get (
            "class_diverse_examples_per_class",2 )),
            candidate_classes =candidate_classes ,
            candidate_scores =semantic_scores ,
            semantic_weight =float (config .get ("fmv3_training",{}).get (
            "class_diverse_semantic_weight",1.0 )),
            )
        else :local_relative =torch .topk (eligible_scores ,local_k ).indices
        local_indices =eligible_indices [local_relative ]
        if episode_type =="balanced"and not bool ((
        support_labels [local_indices ]==query_labels [row ]).any ()):
            positive =torch .where (support_labels ==query_labels [row ])[0 ]
            if positive .numel ():
                best_positive =positive [torch .argmax (similarities [positive ])]
                local_indices =local_indices .clone ()
                local_indices [-1 ]=best_positive
        local_groups [int (local_indices .numel ())].append ((row ,local_indices ))
    for _local_k ,group in local_groups .items ():
        rows =torch .as_tensor ([item [0 ]for item in group ],device =device )
        local_indices =torch .stack ([item [1 ]for item in group ])
        result =model .proto_head .forward_classification_batched (
        support_decision [local_indices ],support_labels [local_indices ],
        query_decision [rows ],global_support_features =support_decision ,
        global_support_labels =support_labels ,
        candidate_classes =candidate_classes ,candidate_features =candidate_features ,
        return_diagnostics =True )
        group_logits ,_classes ,group_probabilities ,head_diagnostics =result
        if group_logits is None :continue
        for position ,row in enumerate (rows .tolist ()):
            logits_rows [row ]=group_logits [position ,:candidate_classes .numel ()]
            probability_rows [row ]=group_probabilities [position ,:candidate_classes .numel ()]
            semantic_rows [row ]=head_diagnostics ["semantic_evidence"][position ]
            process_rows [row ]=head_diagnostics ["process_evidence"][position ]
    if any (row is None for row in logits_rows ):
        return _step_result (
        None ,f"{task_name}_invalid",diagnostics ,return_diagnostics )
    logits =torch .stack (logits_rows )
    probabilities =torch .stack (probability_rows )
    train_cfg =config .get ("fmv3_training",{})or {}
    if bool (train_cfg .get ("structured_memory_enabled",True )):
        structured_probabilities ,context_support =_deployment_structured_probabilities (
        support_tasks ,query_tasks ,candidate_classes ,
        max_order =int (train_cfg .get ("structured_max_order",3 )),
        smoothing =float (train_cfg .get ("structured_smoothing",0.5 )))
        structured =torch .as_tensor (
        structured_probabilities ,dtype =logits .dtype ,device =device )
        context_support =torch .as_tensor (
        context_support ,dtype =logits .dtype ,device =device )
        tau =max (float (train_cfg .get ("structured_tau",0.5 )),0.0 )
        reliability =(
        torch .ones_like (context_support )if tau ==0.0
        else context_support /(context_support +tau ))
        structured_weight =float (train_cfg .get ("structured_weight",0.75 ))
        effective_weight =(structured_weight *reliability ).clamp (0.0 ,1.0 )
        base_log_probs =F .log_softmax (logits ,dim =1 )
        logits =(
        (1.0 -effective_weight .unsqueeze (1))*base_log_probs +
        effective_weight .unsqueeze (1)*torch .log (structured .clamp_min (1e-12 )))
        probabilities =F .softmax (logits ,dim =1 )
    smoothing =min (max (float (config .get (
    "classification_label_smoothing",0.0 )),0.0 ),1.0 )
    final_losses =F .cross_entropy (
    logits ,target_columns ,reduction ="none",label_smoothing =smoothing )
    semantic_losses =F .cross_entropy (
    torch .stack (semantic_rows ),target_columns ,reduction ="none",
    label_smoothing =smoothing )
    candidate_loss =_frequency_stratified_mean (
    semantic_losses ,target_support_counts )
    represented =target_support_counts >0
    decision_loss =(
    _frequency_stratified_mean (final_losses [represented ],
    target_support_counts [represented ])if bool (represented .any ())else None )
    process_loss =None
    if model .proto_head .process_candidate_decoder is not None :
        process_losses =F .cross_entropy (
        torch .stack (process_rows ),target_columns ,reduction ="none" )
        process_loss =_frequency_stratified_mean (
        process_losses ,target_support_counts )
    retrieval_loss =_deployment_retrieval_ranking_loss (
    query_retrieval ,support_retrieval ,query_labels ,support_labels ,
    target_support_counts ,float (config .get ("fmv3_training",{}).get (
    "retrieval_margin",0.15 )))
    total =float (train_cfg .get ("candidate_loss_weight",1.0 ))*candidate_loss
    if decision_loss is not None :
        total =total +float (train_cfg .get ("decision_loss_weight",1.0 ))*decision_loss
    if retrieval_loss is not None :
        total =total +float (train_cfg .get ("retrieval_loss_weight",0.25 ))*retrieval_loss
    if process_loss is not None :
        total =total +float (train_cfg .get ("process_loss_weight",0.25 ))*process_loss
    diagnostics .update ({
    "data/episode_valid":1.0 ,"data/support_cases":float (len ({
    str (item [2 ])for item in support_tasks })),
    "data/support_prefixes":float (len (support_tasks )),
    "data/query_cases":float (len ({str (item [2 ])for item in query_tasks })),
    "data/query_prefixes":float (len (query_tasks )),
    "data/zero_support_queries":float ((target_support_counts ==0 ).sum ().item ()),
    "loss/candidate":float (candidate_loss .detach ()),
    "loss/decision":float (decision_loss .detach ())if decision_loss is not None else 0.0 ,
    "loss/retrieval":float (retrieval_loss .detach ())if retrieval_loss is not None else 0.0 ,
    "loss/process":float (process_loss .detach ())if process_loss is not None else 0.0 ,
    "classification/accuracy":float ((
    probabilities .argmax (dim =1 )==target_columns ).float ().mean ().detach ()),
    })
    return _step_result (total ,task_name ,diagnostics ,return_diagnostics )


def _sample_classification_batch(task_pool, batch_size, episode_type, config):
    """Sample balanced, natural, long-tail, or random-shot deployment episodes."""
    valid = _valid_classification_items(task_pool)
    train_cfg = config.get("fmv3_training", {})
    case_range = train_cfg.get("cases_per_episode_range")
    if case_range and len(case_range) == 2:
        cases = list({str(item[2]) for item in valid})
        target_cases = random.randint(max(2, int(case_range[0])), max(2, int(case_range[1])))
        chosen_cases = set(random.sample(cases, min(target_cases, len(cases))))
        restricted = [item for item in valid if str(item[2]) in chosen_cases]
        if len(restricted) >= batch_size:
            valid = restricted
    if len(valid) < batch_size:
        return None
    if episode_type in {"balanced", "missing_local_label", "missing_pool_label"}:
        return _sample_balanced_classification_batch(
            valid,
            batch_size,
            min_per_class=int(config.get("retrieval_min_per_class", 2)),
            max_classes=config.get("retrieval_train_max_classes"),
            case_uniform_fraction=float(
                config.get("case_uniform_sampling_fraction", 0.5)
            ),
        )
    if episode_type == "natural":
        return sample_task_batch(
            valid,
            batch_size,
            case_uniform_fraction=float(
                config.get("case_uniform_sampling_fraction", 0.5)
            ),
        )

    by_label = defaultdict(list)
    for item in valid:
        by_label[int(item[1])].append(item)
    labels = list(by_label)
    if not labels:
        return None
    if episode_type in {"long_tail", "rare_path"}:
        labels.sort(key=lambda label: len(by_label[label]), reverse=True)
        power_range = train_cfg.get("long_tail_power_range")
        if power_range and len(power_range) == 2:
            power = random.uniform(float(power_range[0]), float(power_range[1]))
        else:
            power = float(train_cfg.get("long_tail_power", 1.5))
        power = max(power, 0.0)
        if episode_type == "long_tail":
            weights = np.asarray([(rank + 1) ** (-power) for rank in range(len(labels))], dtype=float)
        else:
            weights = np.asarray([len(by_label[label]) ** (-power) for label in labels], dtype=float)
        weights /= weights.sum()
        sampled = []
        for _ in range(batch_size):
            label = int(np.random.choice(labels, p=weights))
            items = by_label[label]
            selected_case = random.choice(list({item[2] for item in items}))
            sampled.append(random.choice([
                item for item in items if item[2] == selected_case
            ]))
        return sampled
    if episode_type == "random_shot":
        random.shuffle(labels)
        low = max(1, int(train_cfg.get("random_shot_min", 1)))
        high = max(low, int(train_cfg.get("random_shot_max", 20)))
        sampled = []
        for label in labels:
            shot = random.randint(low, high)
            items = by_label[label]
            sampled.extend(random.sample(items, min(shot, len(items))))
            if len(sampled) >= batch_size:
                break
        if len(sampled) < batch_size:
            sampled.extend(random.sample(valid, batch_size - len(sampled)))
        random.shuffle(sampled)
        return sampled[:batch_size]
    raise ValueError(f"Unknown FM-v3 episode type: {episode_type}")


def _supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    case_ids_int: torch.Tensor,
    temperature: float = 0.07,
):
    """
    Supervised contrastive loss.
    Positives are same-label, different-case pairs.
    Same-case pairs are removed from denominator by heavy negative masking.
    """
    device = z.device
    temp = max(float(temperature), 1e-6)

    with _autocast_disabled_for(device):
        z = F.normalize(z.float(), p=2, dim=1)
        batch_size = z.size(0)

        logits = (z @ z.t()) / temp

        self_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        same_case = case_ids_int.view(-1, 1).eq(case_ids_int.view(1, -1))
        ignore = self_mask | same_case

        # Use a large finite negative value to avoid fp16 overflow / NaNs in AMP paths.
        logits = logits.masked_fill(ignore, -1e4)

        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.t()) & (~ignore)
        pos_counts = pos_mask.sum(dim=1)
        valid = pos_counts > 0
        if not valid.any():
            return None

        log_prob = F.log_softmax(logits, dim=1)
        loss_per = -(log_prob * pos_mask.float()).sum(dim=1) / pos_counts.clamp_min(1).float()
        return loss_per[valid].mean()


def _regression_neighbor_contrastive(
    z: torch.Tensor,
    y: torch.Tensor,
    case_ids_int: torch.Tensor,
    temperature: float = 0.07,
    pos_k: int = 2,
):
    """
    Target-neighborhood contrastive objective for regression.
    Positives per anchor are nearest labels in target space (excluding same case).
    """
    device = z.device
    temp = max(float(temperature), 1e-6)
    pos_k = max(int(pos_k), 1)

    with _autocast_disabled_for(device):
        z = F.normalize(z.float(), p=2, dim=1)
        y = y.float().view(-1)
        batch_size = z.size(0)
        if batch_size == 0:
            return None

        logits = (z @ z.t()) / temp

        self_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        same_case = case_ids_int.view(-1, 1).eq(case_ids_int.view(1, -1))
        ignore = self_mask | same_case
        logits = logits.masked_fill(ignore, -1e4)

        log_prob = F.log_softmax(logits, dim=1)

        # Vectorized nearest-target positives: |y_i - y_j| with ignored pairs
        # pushed to +inf so they never win the top-k over real candidates.
        diffs = (y.view(-1, 1) - y.view(1, -1)).abs()
        diffs = diffs.masked_fill(ignore, float("inf"))
        valid_counts = (~ignore).sum(dim=1)
        valid = valid_counts > 0
        if not valid.any():
            return None
        k_cap = min(pos_k, batch_size - 1)
        if k_cap <= 0:
            return None
        positive_idx = torch.topk(diffs, k_cap, dim=1, largest=False).indices
        pos_diffs = diffs.gather(1, positive_idx)
        pos_log_prob = log_prob.gather(1, positive_idx)
        pos_valid = torch.isfinite(pos_diffs)
        pos_counts = pos_valid.sum(dim=1).clamp_min(1).float()
        loss_per = -(pos_log_prob * pos_valid.float()).sum(dim=1) / pos_counts
        return loss_per[valid].mean()


def _nca_knn_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    case_ids_int: torch.Tensor,
    temperature: float = 0.07,
):
    """Supervised NCA-style objective: maximize same-label probability mass."""
    device = z.device
    temp = max(float(temperature), 1e-6)

    with _autocast_disabled_for(device):
        z = F.normalize(z.float(), p=2, dim=1)
        batch_size = z.size(0)
        logits = (z @ z.t()) / temp

        self_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        same_case = case_ids_int.view(-1, 1).eq(case_ids_int.view(1, -1))
        ignore = self_mask | same_case
        logits = logits.masked_fill(ignore, -1e4)

        log_prob = F.log_softmax(logits, dim=1)
        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.t()) & (~ignore)
        pos_counts = pos_mask.sum(dim=1)
        valid = pos_counts > 0
        if not valid.any():
            return None

        log_p_pos = torch.logsumexp(log_prob.masked_fill(~pos_mask, -1e4), dim=1)
        return (-log_p_pos[valid]).mean()


def _classification_angular_margin_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    case_ids_int: torch.Tensor,
    temperature: float = 0.10,
    margin: float = 0.15,
):
    """Separate classes in the deployed embedding space with an angular margin.

    Every anchor is classified against leave-case-out class prototypes. The
    target prototype logit is reduced by ``margin`` before cross entropy, so a
    zero loss requires same-class cosine similarity to exceed competing class
    similarities by that margin. Excluding the anchor's whole case prevents
    prefixes from one trace providing a trivial same-case shortcut.

    Unlike the existing projection-head SupCon/NCA objectives, this loss acts
    directly on ``all_embeddings`` -- the representation used by retrieval at
    test time. It has no dependency on either posterior expert confidence or
    pre-execution routing confidence.
    """
    device = z.device
    temperature = max(float(temperature), 1e-6)
    margin = max(float(margin), 0.0)

    with _autocast_disabled_for(device):
        z = F.normalize(z.float(), p=2, dim=1)
        labels = labels.long().reshape(-1)
        case_ids_int = case_ids_int.long().reshape(-1)
        classes = torch.unique(labels, sorted=True)
        if z.size(0) < 3 or classes.numel() < 2:
            return None

        different_case = case_ids_int.view(-1, 1).ne(case_ids_int.view(1, -1))
        prototypes = []
        availability = []
        for cls in classes:
            members = different_case & labels.view(1, -1).eq(cls)
            counts = members.sum(dim=1)
            prototype = members.float() @ z
            prototype = prototype / counts.clamp_min(1).float().unsqueeze(1)
            prototypes.append(F.normalize(prototype, p=2, dim=1))
            availability.append(counts > 0)

        # [anchor, class, embedding] and [anchor, class]
        prototypes_t = torch.stack(prototypes, dim=1)
        availability_t = torch.stack(availability, dim=1)
        target_indices = torch.searchsorted(classes, labels)
        row_indices = torch.arange(z.size(0), device=device)
        valid = (
            availability_t[row_indices, target_indices]
            & (availability_t.sum(dim=1) >= 2)
        )
        if not valid.any():
            return None

        logits = torch.einsum("bd,bcd->bc", z, prototypes_t) / temperature
        logits = logits.masked_fill(~availability_t, -1e4)
        logits[row_indices, target_indices] -= margin / temperature
        return F.cross_entropy(logits[valid], target_indices[valid])


def _variance_loss(z: torch.Tensor, eps: float = 1e-4, target_std: float = 1.0):
    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(target_std - std))


def _covariance_loss(z: torch.Tensor):
    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)
    n, d = z.shape
    if n <= 1:
        return torch.tensor(0.0, device=z.device)
    cov = (z.t() @ z) / (n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / d


def run_retrieval_step(
    model, task_data_pool, task_type, config, return_diagnostics=False
):
    progress_bar_task = f"retrieval_{task_type}"
    diagnostics_out = {
        "data/episode_valid": 0.0,
        "data/pool_prefixes": float(len(task_data_pool)),
    }
    configured_k = int(config.get("retrieval_train_k", 5))
    configured_batch_size = int(config.get("retrieval_train_batch_size", 64))
    minimum_batch_size = max(2, int(config.get("retrieval_min_batch_size", 16)))
    retrieval_batch_size = min(configured_batch_size, len(task_data_pool))
    retrieval_k_train = min(configured_k, max(retrieval_batch_size - 1, 1))
    head_cfg = config.get("fmv3_head", {})
    cls_confidence_w = float(
        head_cfg.get(
            "classification_expert_confidence_loss_weight",
            config.get("classification_expert_confidence_loss_weight", 0.0),
        )
    )
    reg_confidence_w = float(
        head_cfg.get(
            "regression_expert_confidence_loss_weight",
            config.get("regression_expert_confidence_loss_weight", 0.0),
        )
    )
    routing_confidence_w = float(
        head_cfg.get("expert_routing_confidence_loss_weight", 0.0)
    )

    episode_type = "regression"
    if task_type == "classification":
        episode_type = _weighted_episode_type(config)
        deployment_enabled =config .get ("fmv3_training",{}).get (
        "deployment_case_episodes",False )
        if isinstance (deployment_enabled ,str ):
            deployment_enabled =deployment_enabled .strip ().lower ()in {
            "1","true","yes","y","on"}
        deployment_enabled =bool (deployment_enabled )or bool (getattr (
        model ,"semantic_candidate_decoder_enabled",False ))
        if deployment_enabled:
            return _run_deployment_classification_step (
            model ,task_data_pool ,episode_type ,config ,return_diagnostics )
    if retrieval_batch_size < minimum_batch_size:
        return _step_result(
            None, progress_bar_task, diagnostics_out, return_diagnostics
        )

    if task_type == "classification":
        batch_tasks_raw = _sample_classification_batch(
            task_data_pool, retrieval_batch_size, episode_type, config
        )
        if not batch_tasks_raw:
            return _step_result(
                None,
                f"{progress_bar_task}_empty",
                diagnostics_out,
                return_diagnostics,
            )
        progress_bar_task = f"{progress_bar_task}_{episode_type}"
    else:
        batch_tasks_raw = sample_task_batch(
            task_data_pool,
            retrieval_batch_size,
            case_uniform_fraction=float(
                config.get("case_uniform_sampling_fraction", 0.5)
            ),
        )

    batch_prefixes = [t[0] for t in batch_tasks_raw]
    batch_labels = np.array([t[1] for t in batch_tasks_raw])
    batch_case_ids = np.array([t[2] for t in batch_tasks_raw], dtype=object)
    diagnostics_out.update(
        {
            "data/episode_valid": 1.0,
            "data/batch_size": float(len(batch_tasks_raw)),
            "data/unique_cases": float(len(set(batch_case_ids.tolist()))),
            "data/unique_labels": float(len(set(batch_labels.tolist()))),
            "schedule/retrieval_k": float(retrieval_k_train),
        }
    )

    time_scale_factor =None
    if task_type =="regression"and model .proto_head .regression_uses_time_transform_bank :
        time_scale_factor =model .proto_head .time_transform_bank .sample_augmentation_factor (
            next (model .parameters ()))
    all_embeddings = model._process_batch(
        batch_prefixes, task_type=task_type, time_scale_factor=time_scale_factor
    )
    z_ssl = model.proj_head(all_embeddings) if hasattr(model, "proj_head") else all_embeddings
    device = all_embeddings.device
    if return_diagnostics:
        tensor_distribution(
            diagnostics_out, "embedding/deployed", all_embeddings, quantiles=False
        )
        tensor_distribution(
            diagnostics_out, "embedding/projection", z_ssl, quantiles=False
        )

    all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=1)
    all_embeddings_norm_detached = all_embeddings_norm.detach()
    cls_pos_k_cfg = int(config.get("retrieval_cls_pos_k", 2))
    neg_pool_factor = max(1, int(config.get("retrieval_neg_pool_factor", 4)))
    neg_random_frac = float(config.get("retrieval_neg_random_frac", 0.25))
    neg_random_frac = min(max(neg_random_frac, 0.0), 1.0)
    pos_use_nearest_cfg = config.get("retrieval_pos_use_nearest", True)
    if isinstance(pos_use_nearest_cfg, str):
        pos_use_nearest = pos_use_nearest_cfg.strip().lower() in {"1", "true", "yes", "y", "on"}
    else:
        pos_use_nearest = bool(pos_use_nearest_cfg)

    contrastive_w = float(config.get("retrieval_contrastive_weight", 0.2))
    contrastive_temp = float(config.get("retrieval_contrastive_temp", 0.07))
    knn_aux_w = float(config.get("retrieval_knn_aux_weight", 0.0))
    contrastive_loss = None
    nca_loss = None
    classification_separation_loss = None
    labels_t = None
    case_ids_int = None

    classification_separation_w = max(
        float(config.get("classification_separation_weight", 0.0)), 0.0
    )

    if (
        contrastive_w > 0
        or (
            task_type == "classification"
            and (knn_aux_w > 0 or classification_separation_w > 0)
        )
    ):
        case_ids_int = _encode_case_ids_to_int(batch_case_ids).to(device)

    if task_type == "classification":
        labels_t = torch.as_tensor(batch_labels, dtype=torch.long, device=device)

    if contrastive_w > 0:
        if task_type == "classification":
            contrastive_loss = _supcon_loss(
                z_ssl, labels_t, case_ids_int, temperature=contrastive_temp
            )
        else:
            y_t = torch.as_tensor(batch_labels, dtype=torch.float32, device=device)
            pos_k = int(config.get("retrieval_regression_pos_k", 2))
            contrastive_loss = _regression_neighbor_contrastive(
                z_ssl, y_t, case_ids_int, temperature=contrastive_temp, pos_k=pos_k
            )

    if task_type == "classification" and knn_aux_w > 0 and labels_t is not None and case_ids_int is not None:
        nca_loss = _nca_knn_loss(
            z_ssl,
            labels_t,
            case_ids_int,
            temperature=contrastive_temp,
        )

    if (
        task_type == "classification"
        and classification_separation_w > 0
        and labels_t is not None
        and case_ids_int is not None
    ):
        classification_separation_loss = _classification_angular_margin_loss(
            all_embeddings,
            labels_t,
            case_ids_int,
            temperature=float(
                config.get("classification_separation_temperature", 0.10)
            ),
            margin=float(config.get("classification_separation_margin", 0.15)),
        )

    total_loss_for_batch = 0.0
    queries_processed = 0
    routing_reliability = None
    classification_head_rows = []
    regression_head_rows = []
    primary_loss_value = None
    confidence_loss_value = None
    gate_auxiliary_value = None
    routing_loss_value = None
    variance_loss_value = None
    covariance_loss_value = None
    regression_components = None
    use_regression_gate_aux = (
        model.proto_head.regression_outputs_hours
        and model.proto_head.regression_gate_aux_weight > 0
    )

    if task_type == "regression":
        # Self-excluded kNN once for the batch, then head forward in groups of
        # equal neighborhood size (same math as the old per-query loop).
        labels_float = torch.as_tensor(batch_labels, dtype=torch.float32, device=device)
        if case_ids_int is None:
            case_ids_int = _encode_case_ids_to_int(batch_case_ids).to(device)
        with torch.no_grad():
            sims = all_embeddings_norm_detached @ all_embeddings_norm_detached.t()
            same_case = case_ids_int.view(-1, 1).eq(case_ids_int.view(1, -1))
            sims = sims.masked_fill(same_case, float("-inf"))
            valid_counts = torch.isfinite(sims).sum(dim=1)

        regression_predictions = []
        regression_targets = []
        regression_branch_predictions = []
        regression_aggregation_weights = []
        regression_confidence_losses = []
        # Group queries by k_eff so we preserve per-query neighborhood size
        # while still running the expensive head in batched mode.
        groups = defaultdict(list)
        for query_i in range(retrieval_batch_size):
            n_valid = int(valid_counts[query_i].item())
            if n_valid <= 0:
                continue
            k_eff = min(retrieval_k_train, n_valid)
            groups[k_eff].append(query_i)

        for k_eff, query_ids in groups.items():
            query_idx = torch.as_tensor(query_ids, dtype=torch.long, device=device)
            with torch.no_grad():
                neighbors = torch.topk(sims[query_idx], k_eff, dim=1).indices
            support_embeddings = all_embeddings[neighbors]
            support_labels_tensor = labels_float[neighbors]
            query_embeddings = all_embeddings[query_idx]
            if use_regression_gate_aux or reg_confidence_w > 0 or return_diagnostics:
                prediction, confidence, diagnostics = model.proto_head.forward_regression_batched(
                    support_embeddings,
                    support_labels_tensor,
                    query_embeddings,
                    return_diagnostics=True,
                    augmentation_factor=time_scale_factor,
                )
                if use_regression_gate_aux:
                    regression_branch_predictions.append(
                        diagnostics["branch_predictions_hours"]
                    )
                    regression_aggregation_weights.append(
                        diagnostics["aggregation_weights"]
                    )
                if reg_confidence_w > 0:
                    regression_confidence_losses.append(
                        model.proto_head.regression_expert_confidence_loss(
                            prediction,
                            labels_float[query_idx],
                            confidence,
                            diagnostics,
                        )
                    )
                if return_diagnostics:
                    regression_head_rows.append(
                        regression_head_metrics(
                            model.proto_head.regression_output_to_hours(prediction),
                            model.proto_head.regression_output_to_hours(
                                model.proto_head.regression_labels_to_output(
                                    labels_float[query_idx]
                                )
                            ),
                            confidence,
                            diagnostics,
                        )
                    )
            else:
                prediction, _ = model.proto_head.forward_regression_batched(
                    support_embeddings,
                    support_labels_tensor,
                    query_embeddings,
                    augmentation_factor=time_scale_factor,
                )
            regression_predictions.append(prediction.reshape(-1))
            regression_targets.append(labels_float[query_idx])

        if regression_predictions:
            joined_predictions = torch.cat(regression_predictions, dim=0)
            joined_targets = torch.cat(regression_targets, dim=0)
            regression_components = (
                model.proto_head.regression_loss_components(
                    joined_predictions, joined_targets
                )
                if model.proto_head.regression_objective_profile != "legacy"
                else None
            )
            primary_loss_value = model.proto_head.regression_loss(
                joined_predictions,
                joined_targets,
                branch_predictions=None,
                aggregation_weights=None,
            )
            total_loss_for_batch = model.proto_head.regression_loss(
                joined_predictions,
                joined_targets,
                branch_predictions=(
                    torch.cat(regression_branch_predictions, dim=1)
                    if regression_branch_predictions else None
                ),
                aggregation_weights=(
                    torch.cat(regression_aggregation_weights, dim=1)
                    if regression_aggregation_weights else None
                ),
            )
            queries_processed = 1
            output_targets = model.proto_head.regression_labels_to_output(
                joined_targets
            )
            routing_reliability = torch.exp(
                -(
                    (joined_predictions.detach() - output_targets.detach()).abs()
                    / output_targets.detach().abs().clamp_min(1.0)
                ).clamp(0.0, 20.0)
            ).mean()
            if regression_confidence_losses:
                confidence_loss_value = torch.stack(
                    regression_confidence_losses
                ).mean()
                total_loss_for_batch = total_loss_for_batch + (
                    reg_confidence_w * confidence_loss_value
                )
            if regression_branch_predictions and regression_aggregation_weights:
                gate_auxiliary_value = model.proto_head.regression_gate_auxiliary_loss(
                    torch.cat(regression_branch_predictions, dim=1),
                    torch.cat(regression_aggregation_weights, dim=1),
                    regression_components["targets"],
                )
    else:
        routing_correctness = []
        classification_logits_rows = []
        classification_target_indices = []
        classification_class_id_rows = []
        classification_confidence_losses = []
        classification_groups = defaultdict(list)
        for i in range(retrieval_batch_size):
            query_label = batch_labels[i]
            query_case_id = batch_case_ids[i]

            with torch.no_grad():
                query_embedding_norm = all_embeddings_norm_detached[i : i + 1]

            if int(query_label) == -100:
                continue
            with torch.no_grad():
                eligible = batch_case_ids != query_case_id
                if episode_type == "missing_pool_label":
                    eligible &= batch_labels != query_label
                if not eligible.any():
                    continue

                local_eligible = eligible.copy()
                if episode_type == "missing_local_label":
                    local_eligible &= batch_labels != query_label
                local_mask = torch.from_numpy(np.where(~local_eligible)[0]).to(device)

                # Balanced FM-v2 episodes preserve the historical guaranteed-positive
                # behavior. Other episode types use ordinary retrieval and may omit it.
                if episode_type == "balanced":
                    positive_np = np.where(
                        (batch_labels == query_label) & (batch_case_ids != query_case_id)
                    )[0]
                    if positive_np.size == 0:
                        continue
                    sims = (query_embedding_norm @ all_embeddings_norm_detached.t()).squeeze(0)
                    pos_k = min(cls_pos_k_cfg, int(positive_np.size), max(1, retrieval_k_train - 1))
                    positives = torch.from_numpy(positive_np).to(device)
                    if pos_use_nearest:
                        positives = positives[torch.topk(sims[positives], pos_k).indices]
                    else:
                        positives = positives[torch.randperm(positives.numel(), device=device)[:pos_k]]
                    negative_mask = (~local_eligible) | (batch_labels == query_label)
                    negatives = select_mixed_negative_indices(
                        query_embedding_norm,
                        all_embeddings_norm_detached,
                        ~torch .as_tensor (negative_mask ,dtype =torch .bool ,device =device ),
                        k=max(1, retrieval_k_train - pos_k),
                        pool_factor=neg_pool_factor,
                        random_fraction=neg_random_frac,
                    )
                    support_indices = torch.cat([positives, negatives])[:retrieval_k_train]
                else:
                    support_indices = find_knn_indices(
                        query_embedding_norm,
                        all_embeddings_norm_detached,
                        k=min(retrieval_k_train, int(local_eligible.sum())),
                        indices_to_mask=local_mask,
                    )

            if support_indices.numel() == 0:
                continue
            target_label = (
                int(config.get("fmv3_head", {}).get("abstain_label", -101))
                if episode_type == "missing_pool_label"
                else int(query_label)
            )
            classification_groups[int(support_indices.numel())].append(
                (i, support_indices, eligible, target_label)
            )

        smoothing = min(
            max(float(config.get("classification_label_smoothing", 0.05)), 0.0),
            1.0,
        )
        for _, group in classification_groups.items():
            query_ids = torch.as_tensor(
                [item[0] for item in group], dtype=torch.long, device=device
            )
            local_indices = torch.stack([item[1] for item in group])
            global_mask = torch.as_tensor(
                np.stack([item[2] for item in group]),
                dtype=torch.bool,
                device=device,
            )
            targets = torch.as_tensor(
                [item[3] for item in group], dtype=torch.long, device=device
            )
            classification_result = model.proto_head.forward_classification_batched(
                all_embeddings[local_indices],
                labels_t[local_indices],
                all_embeddings[query_ids],
                global_support_features=all_embeddings,
                global_support_labels=labels_t,
                global_support_mask=global_mask,
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                logits, proto_classes, probabilities, head_diagnostics = (
                    classification_result
                )
            else:
                logits, proto_classes, probabilities = classification_result
                head_diagnostics = None
            if logits is None:
                continue
            matches = targets.unsqueeze(1).eq(proto_classes.unsqueeze(0))
            valid = matches.any(dim=1)
            if not valid.any():
                continue
            mapped_labels = matches.long().argmax(dim=1)[valid]
            selected_logits = logits[valid]
            selected_probabilities = probabilities[valid]

            routing_correctness.append(
                (
                    selected_probabilities.argmax(dim=-1) == mapped_labels
                ).float().mean().detach()
            )
            classification_logits_rows.extend(selected_logits.unbind(dim=0))
            classification_target_indices.extend(mapped_labels.unbind(dim=0))
            classification_class_id_rows.extend(
                [proto_classes] * int(valid.sum().item())
            )
            if cls_confidence_w > 0:
                classification_confidence_losses.append(
                    model.proto_head.classification_expert_confidence_loss(
                        selected_probabilities, mapped_labels
                    )
                )

            if return_diagnostics:
                classification_head_rows.append(
                    classification_head_metrics(
                        selected_logits,
                        mapped_labels,
                        selected_probabilities,
                        head_diagnostics,
                    )
                )

        if classification_logits_rows:
            metric_objective = classification_metric_objective(
                classification_logits_rows,
                classification_target_indices,
                config,
                class_id_rows=classification_class_id_rows,
                label_smoothing=smoothing,
            )
            primary_loss_value = metric_objective.loss
            total_loss_for_batch = primary_loss_value
            diagnostics_out.update(metric_objective.diagnostics)
            if classification_confidence_losses:
                confidence_loss_value = torch.stack(
                    classification_confidence_losses
                ).mean()
                total_loss_for_batch = total_loss_for_batch + (
                    cls_confidence_w * confidence_loss_value
                )
            # The objective and confidence auxiliary are already episode means.
            queries_processed = 1
        if routing_correctness:
            routing_reliability = torch.stack(routing_correctness).mean()

    loss_out = None
    if queries_processed > 0:
        loss_out = total_loss_for_batch / queries_processed
        if contrastive_loss is not None:
            loss_out = loss_out + (contrastive_w * contrastive_loss)
    elif contrastive_loss is not None:
        loss_out = contrastive_w * contrastive_loss

    if nca_loss is not None and knn_aux_w > 0:
        if loss_out is None:
            loss_out = knn_aux_w * nca_loss
        else:
            loss_out = loss_out + (knn_aux_w * nca_loss)

    if classification_separation_loss is not None and classification_separation_w > 0:
        if loss_out is None:
            loss_out = classification_separation_w * classification_separation_loss
        else:
            loss_out = loss_out + (
                classification_separation_w * classification_separation_loss
            )

    if (
        loss_out is not None
        and routing_confidence_w > 0
        and routing_reliability is not None
        and getattr(model, "task_confidence_head", None) is not None
    ):
        split = max(1, len(batch_tasks_raw) // 2)
        routing_loss = model.task_confidence_loss(
            batch_tasks_raw[:split],
            batch_tasks_raw[split:],
            task_type,
            routing_reliability,
        )
        routing_loss_value = routing_loss
        loss_out = loss_out + routing_confidence_w * routing_loss

    var_w = float(config.get("retrieval_var_weight", 0.0))
    cov_w = float(config.get("retrieval_cov_weight", 0.0))
    if loss_out is not None and (var_w > 0 or cov_w > 0):
        with _autocast_disabled_for(device):
            reg = torch.tensor(0.0, device=device)
            if var_w > 0:
                variance_loss_value = _variance_loss(z_ssl)
                reg = reg + (var_w * variance_loss_value)
            if cov_w > 0:
                covariance_loss_value = _covariance_loss(z_ssl)
                reg = reg + (cov_w * covariance_loss_value)
        loss_out = loss_out + reg

    if return_diagnostics:
        diagnostics_out.update(average_metric_dicts(classification_head_rows))
        diagnostics_out.update(average_metric_dicts(regression_head_rows))
        effective_queries = (
            len(classification_head_rows)
            if task_type == "classification"
            else sum(
                int(row.get("head/regression/query_count", 0))
                for row in regression_head_rows
            )
        )
        diagnostics_out["data/effective_queries"] = float(effective_queries)
        diagnostics_out["data/skipped_queries"] = float(
            max(0, retrieval_batch_size - effective_queries)
        )
        if regression_components is not None:
            regression_weights = {
                "mae": model.proto_head.regression_mae_weight,
                "rmse": model.proto_head.regression_rmse_weight,
                "huber": model.proto_head.regression_huber_weight,
                "log_rmse": model.proto_head.regression_log_rmse_weight,
                "relative_mae": model.proto_head.regression_relative_mae_weight,
                "bias": model.proto_head.regression_bias_weight,
                "median_ae": model.proto_head.regression_median_ae_weight,
                "quantile": model.proto_head.regression_quantile_weight,
                "r2": model.proto_head.regression_r2_weight,
            }
            denominator = model.proto_head._regression_primary_weight_sum()
            for name, weight in regression_weights.items():
                diagnostics_out[f"loss/regression/{name}_raw"] = (
                    regression_components[name]
                )
                diagnostics_out[f"loss/regression/{name}_weighted"] = (
                    weight * regression_components[name] / denominator
                )
                diagnostics_out[f"objective/regression/{name}_weight"] = (
                    regression_components[name].new_tensor(weight)
                )
            diagnostics_out["loss/regression/normalizer_hours"] = (
                regression_components["normalizer"]
            )
        diagnostic_losses = {
            "loss/primary": primary_loss_value,
            "loss/contrastive_raw": contrastive_loss,
            "loss/contrastive_weighted": (
                contrastive_w * contrastive_loss
                if contrastive_loss is not None
                else None
            ),
            "loss/nca_raw": nca_loss,
            "loss/nca_weighted": (
                knn_aux_w * nca_loss if nca_loss is not None else None
            ),
            "loss/classification_separation_raw": classification_separation_loss,
            "loss/classification_separation_weighted": (
                classification_separation_w * classification_separation_loss
                if classification_separation_loss is not None
                else None
            ),
            "loss/confidence_raw": confidence_loss_value,
            "loss/confidence_weighted": (
                (cls_confidence_w if task_type == "classification" else reg_confidence_w)
                * confidence_loss_value
                if confidence_loss_value is not None
                else None
            ),
            "loss/regression_gate_aux_raw": gate_auxiliary_value,
            "loss/regression_gate_aux_weighted": (
                model.proto_head.regression_gate_aux_weight * gate_auxiliary_value
                if gate_auxiliary_value is not None
                else None
            ),
            "loss/routing_raw": routing_loss_value,
            "loss/routing_weighted": (
                routing_confidence_w * routing_loss_value
                if routing_loss_value is not None
                else None
            ),
            "loss/variance_raw": variance_loss_value,
            "loss/variance_weighted": (
                var_w * variance_loss_value
                if variance_loss_value is not None
                else None
            ),
            "loss/covariance_raw": covariance_loss_value,
            "loss/covariance_weighted": (
                cov_w * covariance_loss_value
                if covariance_loss_value is not None
                else None
            ),
            "loss/total": loss_out,
        }
        diagnostics_out.update(
            {key: value for key, value in diagnostic_losses.items() if value is not None}
        )
        if routing_reliability is not None:
            diagnostics_out["routing/target_reliability"] = routing_reliability

    return _step_result(
        loss_out, progress_bar_task, diagnostics_out, return_diagnostics
    )
