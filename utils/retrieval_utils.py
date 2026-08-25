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
    output =[]
    for row_index ,scores in enumerate (rows ):
        class_best =torch .stack ([
        scores [labels ==label ].max ()for label in unique ])
        if candidate_scores is not None :
            semantic_row =candidate_scores [
            0 if candidate_scores .size (0 )==1 else row_index ]
            semantic_for_observed =torch .stack ([
            semantic_row [(candidate_classes ==label ).nonzero (
            as_tuple =False )[0 ,0 ]]for label in unique ])
            class_best =class_best +float (semantic_weight )*semantic_for_observed
        shortlist_size =(
        int (unique .numel ())if unique .numel ()<=k
        else min (max (int (classes_per_shortlist ),1 ),int (unique .numel ()),k ))
        shortlisted =unique [torch .topk (class_best ,shortlist_size ).indices ]
        selected =[]
        # One pass guarantees breadth; later passes add limited within-class depth.
        depth =max (int (examples_per_class ),1 )
        ranked_by_class ={}
        for label in shortlisted :
            members =torch .where (labels ==label )[0 ]
            ranked_by_class [int (label )]=members [
            torch .argsort (scores [members ],descending =True )]
        for rank in range (depth ):
            for label in shortlisted :
                ranked =ranked_by_class [int (label )]
                if rank <ranked .numel ()and len (selected )<k :
                    selected .append (ranked [rank ])
        selected_ids ={int (index )for index in selected }
        if len (selected )<k :
            for index in torch .argsort (scores ,descending =True ):
                if int (index )not in selected_ids :
                    selected .append (index )
                    selected_ids .add (int (index ))
                    if len (selected )==k :break
        output .append (torch .stack (selected ))
    result =torch .stack (output )
    return result [0 ]if squeeze else result
