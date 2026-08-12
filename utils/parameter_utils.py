"""Utilities for making constrained fine-tuning scopes explicit and testable."""


def configure_trainable_scope(model, scope):
    selected = str(scope or "all").lower()
    if selected == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif selected in {"time_transform", "temporal_joint"}:
        for name, parameter in model.named_parameters():
            allowed = "proto_head.time_transform_bank." in name
            if selected == "time_transform":
                allowed = allowed or "embedder.time_input_adapter." in name
            else:
                allowed = allowed or "embedder.temporal_input_encoder." in name
            parameter.requires_grad_(allowed)
    else:
        raise ValueError(f"Unknown trainable_scope: {selected}")

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if selected in {"time_transform", "temporal_joint"} and not trainable:
        raise RuntimeError(
            f"{selected} scope selected, but the model has no temporal-transform parameters"
        )
    return trainable
