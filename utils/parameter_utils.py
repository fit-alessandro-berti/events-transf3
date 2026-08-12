"""Utilities for making constrained fine-tuning scopes explicit and testable."""


def configure_trainable_scope(model, scope):
    selected = str(scope or "all").lower()
    if selected == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif selected == "time_transform":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                "proto_head.time_transform_bank." in name
                or "embedder.time_input_adapter." in name
            )
    else:
        raise ValueError(f"Unknown trainable_scope: {selected}")

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if selected == "time_transform" and not trainable:
        raise RuntimeError("time_transform scope selected, but the model has no time-transform parameters")
    return trainable
