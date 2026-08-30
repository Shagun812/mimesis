
from dataclasses import dataclass
from typing import Callable

import torch

from . import config


# DATA STRUCTURES


@dataclass
class ActivationCache:
    """
    Honest residual-stream activations for one question.

    activations[layer] has shape:
        [batch, sequence_length, hidden_size]
    """

    activations: dict[int, torch.Tensor]


@dataclass
class ForwardResult:
    """Result from one model forward pass."""

    logits: torch.Tensor
    activations: ActivationCache | None = None


# UTILITY


def get_final_token_activation(
    activation: torch.Tensor,
) -> torch.Tensor:
    """
    Extract the final input-token activation.

    Expected input shape:
        [batch, sequence_length, hidden_size]

    Returned shape:
        [batch, hidden_size]
    """

    if activation.ndim != 3:
        raise ValueError(
            "Expected activation with shape "
            "[batch, sequence_length, hidden_size]. "
            f"Got {tuple(activation.shape)}."
        )

    return activation[:, -1, :]



# ACTIVATION CAPTURE

def capture_activations(
    model,
    inputs: dict[str, torch.Tensor],
    layers: list[int] | None = None,
) -> ActivationCache:
    """
    Run one forward pass and capture decoder-layer outputs.

    For MIMESIS, these outputs represent the residual-stream
    state passed downstream from each decoder block.

    Only the final input-token position is ultimately used
    for patching, but the full tensor is temporarily captured
    to make the hook behavior explicit and testable.
    """

    if layers is None:
        layers = list(
            range(model.config.num_hidden_layers)
        )

    activations: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int) -> Callable:
        def hook(module, module_inputs, output):
            if not torch.is_tensor(output):
                raise TypeError(
                    f"Layer {layer_idx} returned "
                    f"{type(output)}, expected Tensor."
                )

            if output.ndim != 3:
                raise ValueError(
                    f"Layer {layer_idx} output has shape "
                    f"{tuple(output.shape)}, expected "
                    "[batch, sequence, hidden]."
                )

            # Detach and clone so later computation cannot
            # mutate the cached Honest activation.
            activations[layer_idx] = (
                output.detach()
                .clone()
            )

        return hook

    try:
        for layer_idx in layers:
            layer = model.model.layers[layer_idx]

            handle = layer.register_forward_hook(
                make_hook(layer_idx)
            )

            handles.append(handle)

        with torch.inference_mode():
            model(**inputs)

    finally:
        for handle in handles:
            handle.remove()

    missing = [
        layer
        for layer in layers
        if layer not in activations
    ]

    if missing:
        raise RuntimeError(
            f"Failed to capture activations for layers: {missing}"
        )

    return ActivationCache(
        activations=activations
    )


# UNPATCHED FORWARD

@torch.inference_mode()
def run_unpatched(
    model,
    inputs: dict[str, torch.Tensor],
) -> ForwardResult:
    """
    Run a normal forward pass without intervention.
    """

    outputs = model(**inputs)

    return ForwardResult(
        logits=outputs.logits.detach(),
        activations=None,
    )



# SINGLE-LAYER PATCH

@torch.inference_mode()
def run_patched(
    model,
    inputs: dict[str, torch.Tensor],
    honest_cache: ActivationCache,
    layer_idx: int,
) -> ForwardResult:
    """
    Run the Sandbagging trajectory while replacing the
    selected layer's final-token activation with the
    corresponding Honest activation.

    Intervention:

        Sandbagging output[L][:, -1, :]
                         ↓
                    replaced by
                         ↓
        Honest output[L][:, -1, :]

    All other tokens and all other layers remain unchanged.
    """

    if layer_idx not in honest_cache.activations:
        raise KeyError(
            f"No Honest activation cached for layer {layer_idx}."
        )

    honest_activation = (
        honest_cache.activations[layer_idx]
    )

    def patch_hook(module, module_inputs, output):
        if not torch.is_tensor(output):
            raise TypeError(
                f"Layer {layer_idx} returned "
                f"{type(output)}, expected Tensor."
            )

        if output.ndim != 3:
            raise ValueError(
                f"Layer {layer_idx} output has shape "
                f"{tuple(output.shape)}, expected "
                "[batch, sequence, hidden]."
            )

        if honest_activation.ndim != 3:
            raise ValueError(
                "Cached Honest activation has shape "
                f"{tuple(honest_activation.shape)}, expected "
                "[batch, sequence, hidden]."
            )

        # Sequence lengths may differ across conditions.
        # Only batch size and hidden size must match because
        # the intervention replaces the final token only.

        if output.shape[0] != honest_activation.shape[0]:
            raise ValueError(
                "Honest and Sandbagging batch sizes differ: "
                f"{output.shape[0]} vs {honest_activation.shape[0]}."
            )

        if output.shape[2] != honest_activation.shape[2]:
            raise ValueError(
                "Honest and Sandbagging hidden sizes differ: "
                f"{output.shape[2]} vs {honest_activation.shape[2]}."
            )

        patched_output = output.clone()

        patched_output[:, -1, :] = (
            honest_activation[:, -1, :]
            .to(
                device=output.device,
                dtype=output.dtype,
            )
        )

        return patched_output

    layer = model.model.layers[layer_idx]

    handle = layer.register_forward_hook(
        patch_hook
    )

    try:
        outputs = model(**inputs)

    finally:
        handle.remove()

    return ForwardResult(
        logits=outputs.logits.detach(),
        activations=None,
    )



# LAYER-WISE PATCHING

def run_layer_sweep(
    model,
    honest_inputs: dict[str, torch.Tensor],
    sandbagging_inputs: dict[str, torch.Tensor],
) -> tuple[
    ActivationCache,
    ForwardResult,
    dict[int, ForwardResult],
]:
    """
    Run the complete layer-wise MIMESIS intervention
    for one question.

    Returns:

        honest_cache
        unpatched Sandbagging result
        patched result for every layer
    """

    
    # 1. Honest trajectory
    
    honest_cache = capture_activations(
        model=model,
        inputs=honest_inputs,
    )

    
    # 2. Unpatched Sandbagging trajectory
    
    unpatched = run_unpatched(
        model=model,
        inputs=sandbagging_inputs,
    )

    
    # 3. Patch every transformer layer
    

    num_layers = model.config.num_hidden_layers

    patched_results: dict[int, ForwardResult] = {}

    for layer_idx in range(num_layers):
        patched_results[layer_idx] = run_patched(
            model=model,
            inputs=sandbagging_inputs,
            honest_cache=honest_cache,
            layer_idx=layer_idx,
        )

    return (
        honest_cache,
        unpatched,
        patched_results,
    )



# POSITIVE CONTROL

def run_activation_replacement_test(
    model,
    inputs: dict[str, torch.Tensor],
    layer_idx: int,
) -> dict:
    """
    Minimal intervention sanity test.

    Captures a layer's own activation and then replaces
    the same layer output with that cached activation.

    This should preserve the forward computation up to
    numerical precision and is useful for validating:

        - hook registration
        - activation capture
        - activation replacement
        - tensor shape handling
        - hook cleanup

    This is an infrastructure test, not the scientific
    positive control used to establish behavioral recovery.
    """

    cache = capture_activations(
        model=model,
        inputs=inputs,
        layers=[layer_idx],
    )

    baseline = run_unpatched(
        model=model,
        inputs=inputs,
    )

    layer = model.model.layers[layer_idx]

    cached = cache.activations[layer_idx]

    def identity_patch(module, module_inputs, output):
        if not torch.is_tensor(output):
            raise TypeError(
                "Expected tensor layer output."
            )

        if output.shape != cached.shape:
            raise ValueError(
                "Cached and current activation shapes differ."
            )

        patched = output.clone()

        patched[:, -1, :] = (
            cached[:, -1, :]
            .to(
                device=output.device,
                dtype=output.dtype,
            )
        )

        return patched

    handle = layer.register_forward_hook(
        identity_patch
    )

    try:
        with torch.inference_mode():
            outputs = model(**inputs)

    finally:
        handle.remove()

    max_logit_difference = (
        baseline.logits - outputs.logits
    ).abs().max().item()

    return {
        "layer": layer_idx,
        "max_logit_difference": max_logit_difference,
        "passed": max_logit_difference < 1e-4,
    }



# VALIDATION

def validate_patch_configuration(model) -> None:
    """
    Validate assumptions required by MIMESIS patching.
    """

    if config.PATCH_COMPONENT != "residual_stream":
        raise ValueError(
            "MIMESIS currently requires residual-stream patching."
        )

    if config.PATCH_POSITION != "final_input_token":
        raise ValueError(
            "MIMESIS currently requires final-input-token patching."
        )

    if config.LAYER_SELECTION != "all":
        raise ValueError(
            "MIMESIS V1 requires an all-layer sweep."
        )

    num_layers = model.config.num_hidden_layers

    if num_layers <= 0:
        raise ValueError(
            "Model has no transformer layers."
        )



# MAIN

if __name__ == "__main__":
    print(
        "patching.py contains the MIMESIS activation-"
        "patching routines."
    )
    print(
        "Run model loading and intervention tests from "
        "the dedicated notebooks."
    )