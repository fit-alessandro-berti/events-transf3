import torch
import numpy as np
def find_knn_indices (query_embedding_norm :torch .Tensor ,
search_embeddings_norm :torch .Tensor ,
k :int ,
indices_to_mask :torch .Tensor =None ):
    if k <=0 :
        return torch .tensor ([],dtype =torch .long ,device =query_embedding_norm .device )
    sims =query_embedding_norm @search_embeddings_norm .T
    if indices_to_mask is not None and indices_to_mask .numel ()>0 :
        sims [0 ,indices_to_mask ]=-float ('inf')
    num_valid =(sims [0 ]>-float ('inf')).sum ().item ()
    k_to_find =min (k ,num_valid )
    if k_to_find <=0 :
        return torch .tensor ([],dtype =torch .long ,device =sims .device )
    top_k_indices =torch .topk (sims .squeeze (0 ),k_to_find ).indices
    return top_k_indices


def class_diverse_topk_indices(
    similarities: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    *,
    classes_per_shortlist: int = 10,
    examples_per_class: int = 2,
    candidate_classes: torch.Tensor | None = None,
    candidate_scores: torch.Tensor | None = None,
    semantic_weight: float = 1.0,
):
    """Retrieve a class-diverse neighborhood and then fill by similarity.

    ``similarities`` may be ``[support]`` or ``[query, support]``.  Candidate
    classes are shortlisted by their best matching example.  When all observed
    classes fit in ``k``, every class receives one slot before the remainder is
    filled, which directly prevents high-frequency labels from crowding out the
    neighborhood.
    """
    squeeze =similarities .ndim ==1
    rows =similarities .unsqueeze (0 )if squeeze else similarities
    labels =torch .as_tensor (labels ,dtype =torch .long ,device =rows .device )
    k =min (max (int (k ),0 ),int (labels .numel ()))
    if k ==0 :
        empty =torch .empty ((rows .size (0 ),0 ),dtype =torch .long ,device =rows .device )
        return empty [0 ]if squeeze else empty
    unique =torch .unique (labels ,sorted =True )
    if candidate_scores is not None :
        candidate_scores =candidate_scores .unsqueeze (0 )if candidate_scores .ndim ==1 else candidate_scores
        candidate_classes =torch .as_tensor (
        candidate_classes ,dtype =torch .long ,device =rows .device )
        if candidate_scores .size (0 )not in {1 ,rows .size (0 )}:
            raise ValueError ("candidate_scores must have one row or align with queries")
    # Only the first ``depth`` members of each class and at most ``k`` global
    # members can be selected.  Materializing those bounded top-k sets in
    # batches is equivalent to fully sorting each support class for every
    # query, but avoids an O(query * support log support) sort in large logs.
    depth =max (int (examples_per_class ),1 )
    query_count =int (rows .size (0 ))
    class_count =int (unique .numel ())
    class_best =rows .new_empty ((query_count ,class_count ))
    ranked_by_class =torch .full (
    (query_count ,class_count ,depth ),-1 ,dtype =torch .long ,device =rows .device )
    for class_index ,label in enumerate (unique ):
        members =torch .where (labels ==label )[0 ]
        take =min (depth ,int (members .numel ()))
        values ,positions =torch .topk (
        rows [:,members ],take ,dim =1 ,largest =True ,sorted =True )
        class_best [:,class_index ]=values [:,0 ]
        ranked_by_class [:,class_index ,:take ]=members [positions ]
    if candidate_scores is not None :
        candidate_positions =torch .stack ([
        (candidate_classes ==label ).nonzero (as_tuple =False )[0 ,0 ]
        for label in unique ])
        semantic_for_observed =candidate_scores [:,candidate_positions ]
        if semantic_for_observed .size (0 )==1 and query_count >1 :
            semantic_for_observed =semantic_for_observed .expand (query_count ,-1 )
        class_best =class_best +float (semantic_weight )*semantic_for_observed
    shortlist_size =(
    class_count if class_count <=k
    else min (max (int (classes_per_shortlist ),1 ),class_count ,k ))
    shortlist_positions =torch .topk (
    class_best ,shortlist_size ,dim =1 ).indices
    shortlisted_rankings =torch .gather (
    ranked_by_class ,1 ,shortlist_positions .unsqueeze (-1 ).expand (
    -1 ,-1 ,depth ))
    # Match the historical depth-major order: one member from every class,
    # followed by the second member from every class, and so on.
    initial =shortlisted_rankings .permute (0 ,2 ,1 ).reshape (query_count ,-1 )
    max_initial =min (k ,int (initial .size (1 )))
    # ``k + max_initial`` guarantees at least k candidates remain after
    # removing every possible duplicate already chosen in the breadth pass.
    fill_width =min (int (labels .numel ()),k +max_initial )
    global_ranked =torch .topk (
    rows ,fill_width ,dim =1 ,largest =True ,sorted =True ).indices
    # Compact candidate lists are cheaper to de-duplicate on the host.  One
    # batched transfer avoids a device synchronization for every scalar index.
    initial_rows =initial .cpu ().tolist ()
    global_rows =global_ranked .cpu ().tolist ()
    output =[]
    for row_index in range (query_count ):
        selected =[]
        selected_ids =set ()
        for value in initial_rows [row_index ]:
            if value <0 or value in selected_ids :continue
            selected .append (value )
            selected_ids .add (value )
            if len (selected )==k :break
        if len (selected )<k :
            for value in global_rows [row_index ]:
                if value in selected_ids :continue
                selected .append (value )
                selected_ids .add (value )
                if len (selected )==k :break
        output .append (selected )
    result =torch .as_tensor (output ,dtype =torch .long ,device =rows .device )
    return result [0 ]if squeeze else result
