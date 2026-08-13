"""Utilities for making constrained fine-tuning scopes explicit and testable."""


def configure_trainable_scope(model, scope):
    selected = str(scope or "all").lower()
    if selected == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif selected in {
        "time_transform",
        "regression_gate",
        "expert_confidence",
        "expert_routing_confidence",
        "temporal_joint",
        "prefix_attention",
        "temporal_prefix_joint",
        "classification_adapter",
        "regression_refinement",
    }:
        for name, parameter in model.named_parameters():
            allowed = False
            if selected == "time_transform":
                allowed = (
                    "proto_head.time_transform_bank." in name
                    or "embedder.time_input_adapter." in name
                )
            elif selected == "regression_gate":
                allowed = (
                    "proto_head.time_transform_bank.dynamic_gate." in name
                    or name.endswith("proto_head.time_transform_bank.aggregation_logits")
                )
            elif selected == "expert_confidence":
                allowed = (
                    "proto_head.classification_expert_confidence." in name
                    or "proto_head.regression_expert_confidence." in name
                )
            elif selected == "expert_routing_confidence":
                allowed = "task_confidence_head." in name
            elif selected == "temporal_joint":
                allowed = (
                    "proto_head.time_transform_bank." in name
                    or "embedder.temporal_input_encoder." in name
                )
            elif selected == "prefix_attention":
                allowed = "encoder.state_aware_pool." in name
            elif selected == "temporal_prefix_joint":
                allowed = (
                    "encoder.state_aware_pool." in name
                    or "proto_head.time_transform_bank." in name
                    or "embedder.temporal_input_encoder." in name
                )
            elif selected == "classification_adapter":
                allowed = "classification_embedding_adapter." in name
            elif selected == "regression_refinement":
                allowed = (
                    "proto_head.time_transform_bank." in name
                    or "regression_embedding_adapter." in name
                    or "proto_head.regression_expert_confidence." in name
                )
            parameter.requires_grad_(allowed)
    else:
        raise ValueError(f"Unknown trainable_scope: {selected}")

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    constrained = {
        "time_transform",
        "regression_gate",
        "expert_confidence",
        "expert_routing_confidence",
        "temporal_joint",
        "prefix_attention",
        "temporal_prefix_joint",
        "classification_adapter",
        "regression_refinement",
    }
    if selected in constrained and not trainable:
        raise RuntimeError(
            f"{selected} scope selected, but the model has no matching parameters"
        )
    return trainable
