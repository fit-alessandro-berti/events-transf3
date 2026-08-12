"""Utilities for making constrained fine-tuning scopes explicit and testable."""


def configure_trainable_scope(model, scope):
    selected = str(scope or "all").lower()
    if selected == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif selected in {
        "time_transform",
        "temporal_joint",
        "prefix_attention",
        "temporal_prefix_joint",
    }:
        for name, parameter in model.named_parameters():
            allowed = False
            if selected == "time_transform":
                allowed = (
                    "proto_head.time_transform_bank." in name
                    or "embedder.time_input_adapter." in name
                )
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
            parameter.requires_grad_(allowed)
    else:
        raise ValueError(f"Unknown trainable_scope: {selected}")

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    constrained = {
        "time_transform",
        "temporal_joint",
        "prefix_attention",
        "temporal_prefix_joint",
    }
    if selected in constrained and not trainable:
        raise RuntimeError(
            f"{selected} scope selected, but the model has no matching parameters"
        )
    return trainable
