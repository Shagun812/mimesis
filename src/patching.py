from dataclasses import dataclass
from typing import Callable

import torch

from . import config



@dataclass
class ActivationCache:
    """
    Cached Honest residual-stream activations for one question.

    activations[layer] has shape:
        [batch, hidden_size]

    Only the final input-token position is stored.

    For MIMESIS V1 this corresponds to the residual-stream state
    produced by the selected Qwen3 decoder layer at the final
    input-token position.
    """

    activations: dict[int, torch.Tensor]


@dataclass
class ForwardResult:
    """Result from one model forward pass."""

    logits: torch.Tensor
    activations: ActivationCache | None = None


@dataclass
class RestorationMetrics:
    """
    Continuous causal-restoration metrics for one patched result.

    All logit values refer to the final input position, whose
    logits determine the next-token prediction.
    """

    correct_logit: float
    best_wrong_logit: float
    correct_margin: float

    restoration_logit: float | None
    restoration_margin: float | None

    prediction: str | None
    correct: bool | None



def validate_qwen_layer_structure(model) -> None:
    """
    Verify the model exposes the Qwen3 layer structure expected by MIMESIS.
    """

    if not hasattr(model, "model"):
        raise AttributeError(
            "Expected causal LM to expose model.model."
        )

    backbone = model.model

    if not hasattr(backbone, "layers"):
        raise AttributeError(
            "Expected Qwen3 backbone to expose model.layers."
        )

    layers = backbone.layers

    if not isinstance(layers, torch.nn.ModuleList):
        raise TypeError(
            "Expected model.model.layers to be a "
            f"torch.nn.ModuleList; got {type(layers)}."
        )

    expected_layers = model.config.num_hidden_layers

    if len(layers) != expected_layers:
        raise ValueError(
            "Layer-count mismatch: "
            f"config reports {expected_layers}, "
            f"but model.model.layers contains {len(layers)}."
        )

    if expected_layers <= 0:
        raise ValueError(
            "Model reports no transformer layers."
        )

    # Qwen3-4B exposes Qwen3DecoderLayer objects.
    # Avoid importing the implementation class directly because
    # trust_remote_code/model-version details can vary.
    for idx, layer in enumerate(layers):
        class_name = layer.__class__.__name__

        if class_name != "Qwen3DecoderLayer":
            raise TypeError(
                f"Layer {idx} is {class_name}, expected "
                "Qwen3DecoderLayer."
            )



def get_final_token_id(
    inputs: dict[str, torch.Tensor],
) -> int:
    """
    Return the final input token ID for a batch of size one.
    """

    if "input_ids" not in inputs:
        raise KeyError(
            "inputs must contain input_ids."
        )

    input_ids = inputs["input_ids"]

    if input_ids.ndim != 2:
        raise ValueError(
            "Expected input_ids with shape "
            f"[batch, sequence]. Got {tuple(input_ids.shape)}."
        )

    if input_ids.shape[0] != 1:
        raise ValueError(
            "MIMESIS V1 position validation currently expects "
            "batch size 1."
        )

    return int(input_ids[0, -1].item())


def verify_final_token_is_answer_suffix(
    tokenizer,
    inputs: dict[str, torch.Tensor],
) -> dict:
    """
    Verify that the final input token belongs to the expected
    'Answer:' suffix.

    This does not require the final token itself to be a specific
    punctuation token because tokenization can vary.

    Instead, decode the final few input tokens and verify that the
    final input sequence ends with 'Answer:' after whitespace
    normalization.
    """

    if "input_ids" not in inputs:
        raise KeyError(
            "inputs must contain input_ids."
        )

    input_ids = inputs["input_ids"]

    if input_ids.ndim != 2:
        raise ValueError(
            "Expected input_ids with shape [batch, sequence]."
        )

    if input_ids.shape[0] != 1:
        raise ValueError(
            "Final-token verification currently expects batch size 1."
        )

    decoded = tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
    )

    normalized = decoded.rstrip()

    if not normalized.endswith("Answer:"):
        raise ValueError(
            "Final input sequence does not end with 'Answer:'. "
            f"Decoded suffix: {repr(normalized[-80:])}"
        )

    final_token_id = int(input_ids[0, -1].item())

    final_token_text = tokenizer.decode(
        [final_token_id],
        skip_special_tokens=False,
    )

    return {
        "final_token_id": final_token_id,
        "final_token_text": final_token_text,
        "decoded_suffix": normalized[-80:],
        "verified": True,
    }



def get_final_token_activation(
    activation: torch.Tensor,
) -> torch.Tensor:
    """
    Extract the final input-token activation.

    Expected input:
        [batch, sequence_length, hidden_size]

    Returned:
        [batch, hidden_size]
    """

    if activation.ndim != 3:
        raise ValueError(
            "Expected activation with shape "
            "[batch, sequence_length, hidden_size]. "
            f"Got {tuple(activation.shape)}."
        )

    if activation.shape[1] <= 0:
        raise ValueError(
            "Activation has empty sequence dimension."
        )

    return activation[:, -1, :]



def capture_activations(
    model,
    inputs: dict[str, torch.Tensor],
    layers: list[int] | None = None,
) -> ActivationCache:
    """
    Run one forward pass and capture final-token decoder-layer outputs.

    For each selected layer, only:

        output[:, -1, :]

    is retained.

    Cached shape:

        [batch, hidden_size]

    This avoids storing the full sequence for each layer and removes
    any dependence on Honest/Sandbagging sequence-length equality.
    """

    validate_qwen_layer_structure(model)

    if layers is None:
        layers = list(
            range(model.config.num_hidden_layers)
        )

    if not layers:
        raise ValueError(
            "At least one layer must be requested."
        )

    num_layers = model.config.num_hidden_layers

    for layer_idx in layers:
        if not isinstance(layer_idx, int):
            raise TypeError(
                f"Layer index must be int; got {type(layer_idx)}."
            )

        if not 0 <= layer_idx < num_layers:
            raise IndexError(
                f"Layer {layer_idx} is outside valid range "
                f"[0, {num_layers - 1}]."
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

            final_activation = (
                get_final_token_activation(output)
            )

            if final_activation.ndim != 2:
                raise RuntimeError(
                    f"Layer {layer_idx} final activation has "
                    f"unexpected shape "
                    f"{tuple(final_activation.shape)}."
                )

            activations[layer_idx] = (
                final_activation.detach()
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
            model(**inputs, use_cache=False)

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
            "Failed to capture activations for layers: "
            f"{missing}"
        )

    return ActivationCache(
        activations=activations
    )




@torch.inference_mode()
def run_unpatched(
    model,
    inputs: dict[str, torch.Tensor],
) -> ForwardResult:
    """
    Run a normal forward pass without intervention.
    """

    outputs = model(**inputs, use_cache=False)

    if not hasattr(outputs, "logits"):
        raise AttributeError(
            "Model output does not contain logits."
        )

    # Move logits off GPU immediately. Do not retain GPU-resident
    # vocabulary logits across repeated interventions.
    logits_cpu = outputs.logits.detach().cpu()
    del outputs

    return ForwardResult(
        logits=logits_cpu,
        activations=None,
    )




def get_final_logits(
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Extract next-token logits from the final input position.

    Input:
        [batch, sequence, vocab]

    Output:
        [batch, vocab]
    """

    if logits.ndim != 3:
        raise ValueError(
            "Expected logits with shape "
            "[batch, sequence, vocab]. "
            f"Got {tuple(logits.shape)}."
        )

    return logits[:, -1, :]


def get_answer_choice_logits(
    logits: torch.Tensor,
    choice_token_ids: dict[str, int],
) -> dict[str, float]:
    """
    Extract logits for A/B/C/D answer tokens.

    The supplied token IDs must correspond to the exact single-token
    answer labels used by the experiment.
    """

    final_logits = get_final_logits(logits)

    if final_logits.shape[0] != 1:
        raise ValueError(
            "Answer-choice extraction currently expects batch size 1."
        )

    result = {}

    for label in ("A", "B", "C", "D"):

        if label not in choice_token_ids:
            raise KeyError(
                f"Missing token ID for choice {label}."
            )

        token_id = choice_token_ids[label]

        if not isinstance(token_id, int):
            raise TypeError(
                f"Token ID for {label} must be int."
            )

        result[label] = float(
            final_logits[0, token_id].item()
        )

    return result


def calculate_correct_margin(
    choice_logits: dict[str, float],
    correct_label: str,
) -> tuple[float, float, float]:
    """
    Return:

        correct_logit
        best_wrong_logit
        correct_margin

    where:

        correct_margin =
            correct_logit - best_wrong_logit
    """

    if correct_label not in ("A", "B", "C", "D"):
        raise ValueError(
            f"Invalid correct label: {correct_label}"
        )

    correct_logit = choice_logits[correct_label]

    wrong_logits = [
        value
        for label, value in choice_logits.items()
        if label != correct_label
    ]

    best_wrong_logit = max(wrong_logits)

    margin = (
        correct_logit -
        best_wrong_logit
    )

    return (
        correct_logit,
        best_wrong_logit,
        margin,
    )



@torch.inference_mode()
def run_patched(
    model,
    inputs: dict[str, torch.Tensor],
    honest_cache: ActivationCache,
    layer_idx: int,
) -> ForwardResult:
    """
    Run the Sandbagging trajectory while replacing the selected
    layer's final-token activation with the corresponding Honest
    activation.

    Intervention:

        Sandbagging layer output:
            [batch, sequence_SB, hidden]

        Replace:
            output[:, -1, :]

        With:
            Honest cached final-token activation:
            [batch, hidden]

    Sequence lengths may differ between conditions.
    Only batch size and hidden size must match.
    """

    validate_qwen_layer_structure(model)

    if layer_idx not in honest_cache.activations:
        raise KeyError(
            f"No Honest activation cached for layer {layer_idx}."
        )

    honest_activation = (
        honest_cache.activations[layer_idx]
    )

    if honest_activation.ndim != 2:
        raise ValueError(
            "Cached Honest final-token activation must have shape "
            f"[batch, hidden]. Got {tuple(honest_activation.shape)}."
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

        if output.shape[0] != honest_activation.shape[0]:
            raise ValueError(
                "Honest and Sandbagging batch sizes differ: "
                f"{output.shape[0]} vs "
                f"{honest_activation.shape[0]}."
            )

        if output.shape[2] != honest_activation.shape[1]:
            raise ValueError(
                "Honest and Sandbagging hidden sizes differ: "
                f"{output.shape[2]} vs "
                f"{honest_activation.shape[1]}."
            )

        if output.shape[1] <= 0:
            raise ValueError(
                "Sandbagging activation has empty sequence dimension."
            )

        replacement = (
            honest_activation
            .to(
                device=output.device,
                dtype=output.dtype,
            )
        )

        patched_output = output.clone()

        original_final = (
            patched_output[:, -1, :]
            .detach()
            .clone()
        )

        patched_output[:, -1, :] = replacement

        # Verify the intervention changed only the final position.
        if output.shape[1] > 1:

            prefix_difference = (
                patched_output[:, :-1, :]
                - output[:, :-1, :]
            ).abs().max().item()

            if prefix_difference != 0.0:
                raise RuntimeError(
                    "Patch-integrity failure: activation values before "
                    "the final token were modified."
                )

        final_difference = (
            patched_output[:, -1, :]
            - original_final
        ).abs().max().item()

        # If the Honest and Sandbagging activations happen to be
        # numerically identical, a zero difference is legitimate.
        # We do not require a nonzero intervention.

        return patched_output

    layer = model.model.layers[layer_idx]

    handle = layer.register_forward_hook(
        patch_hook
    )

    try:
        outputs = model(**inputs, use_cache=False)

    finally:
        handle.remove()

    if not hasattr(outputs, "logits"):
        raise AttributeError(
            "Patched model output does not contain logits."
        )

    # Move logits off GPU immediately so patched results cannot retain
    # the model's vocabulary-sized output on the T4.
    logits_cpu = outputs.logits.detach().cpu()
    del outputs

    return ForwardResult(
        logits=logits_cpu,
        activations=None,
    )




def calculate_restoration_metrics(
    honest_logits: dict[str, float],
    sandbagging_logits: dict[str, float],
    patched_logits: dict[str, float],
    correct_label: str,
) -> RestorationMetrics:
    """
    Calculate continuous restoration from Sandbagging toward Honest.

    For the correct-answer logit:

        restoration =
            (patched - sandbagging)
            /
            (honest - sandbagging)

    The same calculation is performed for the
    correct-vs-best-wrong margin.

    Interpretation:

        0   = no restoration
        1   = full restoration to Honest
        >1  = overshoot beyond Honest
        <0  = movement away from Honest
    """

    (
        honest_correct,
        honest_best_wrong,
        honest_margin,
    ) = calculate_correct_margin(
        honest_logits,
        correct_label,
    )

    (
        sand_correct,
        sand_best_wrong,
        sand_margin,
    ) = calculate_correct_margin(
        sandbagging_logits,
        correct_label,
    )

    (
        patched_correct,
        patched_best_wrong,
        patched_margin,
    ) = calculate_correct_margin(
        patched_logits,
        correct_label,
    )

    honest_delta = (
        honest_correct -
        sand_correct
    )

    if abs(honest_delta) > 1e-12:
        restoration_logit = (
            patched_correct -
            sand_correct
        ) / honest_delta
    else:
        restoration_logit = None

    honest_margin_delta = (
        honest_margin -
        sand_margin
    )

    if abs(honest_margin_delta) > 1e-12:
        restoration_margin = (
            patched_margin -
            sand_margin
        ) / honest_margin_delta
    else:
        restoration_margin = None

    prediction = max(
        patched_logits,
        key=patched_logits.get,
    )

    correct = (
        prediction == correct_label
    )

    return RestorationMetrics(
        correct_logit=patched_correct,
        best_wrong_logit=patched_best_wrong,
        correct_margin=patched_margin,
        restoration_logit=restoration_logit,
        restoration_margin=restoration_margin,
        prediction=prediction,
        correct=correct,
    )



def run_layer_sweep(
    model,
    honest_inputs: dict[str, torch.Tensor],
    sandbagging_inputs: dict[str, torch.Tensor],
    choice_token_ids: dict[str, int] | None = None,
    correct_label: str | None = None,
    layers: list[int] | None = None,
) -> tuple[
    ActivationCache,
    ForwardResult,
    ForwardResult,
    dict[int, ForwardResult],
]:
    """
    Run the complete layer-wise MIMESIS intervention for one question.

    Returns:

        honest_cache
        honest_result
        unpatched_sandbagging_result
        patched_results

    Honest activations are cached only at the final input-token
    position.

    If choice_token_ids and correct_label are supplied, the caller
    can compute continuous restoration metrics from the returned
    logits.
    """

    validate_qwen_layer_structure(model)



    if honest_inputs["input_ids"].shape[0] != 1:
        raise ValueError(
            "MIMESIS V1 currently expects batch size 1."
        )

    if sandbagging_inputs["input_ids"].shape[0] != 1:
        raise ValueError(
            "MIMESIS V1 currently expects batch size 1."
        )



    honest_cache = capture_activations(
        model=model,
        inputs=honest_inputs,
        layers=layers,
    )

    honest_result = run_unpatched(
        model=model,
        inputs=honest_inputs,
    )



    unpatched = run_unpatched(
        model=model,
        inputs=sandbagging_inputs,
    )



    if layers is None:
        layers_to_patch = list(
            range(model.config.num_hidden_layers)
        )
    else:
        layers_to_patch = list(layers)

    patched_results: dict[int, ForwardResult] = {}

    for layer_idx in layers_to_patch:

        patched_results[layer_idx] = run_patched(
            model=model,
            inputs=sandbagging_inputs,
            honest_cache=honest_cache,
            layer_idx=layer_idx,
        )

    return (
        honest_cache,
        honest_result,
        unpatched,
        patched_results,
    )




def run_activation_replacement_test(
    model,
    inputs: dict[str, torch.Tensor],
    layer_idx: int,
) -> dict:
    """
    Infrastructure positive control.

    Capture a layer's final-token activation and replace that exact
    activation with itself.

    Expected:

        baseline logits == patched logits

    up to numerical precision.

    This validates:

        - correct layer access
        - hook registration
        - final-token extraction
        - activation replacement
        - patch integrity
        - hook cleanup
    """

    validate_qwen_layer_structure(model)

    cache = capture_activations(
        model=model,
        inputs=inputs,
        layers=[layer_idx],
    )

    baseline = run_unpatched(
        model=model,
        inputs=inputs,
    )

    cached = cache.activations[layer_idx]

    if cached.ndim != 2:
        raise RuntimeError(
            "Positive-control cache has unexpected shape: "
            f"{tuple(cached.shape)}."
        )

    layer = model.model.layers[layer_idx]

    patch_statistics = {
        "max_final_token_difference": None,
        "max_prefix_difference": None,
    }

    def identity_patch(module, module_inputs, output):

        if not torch.is_tensor(output):
            raise TypeError(
                "Expected tensor layer output."
            )

        if output.ndim != 3:
            raise ValueError(
                "Expected layer output with shape "
                "[batch, sequence, hidden]."
            )

        if output.shape[0] != cached.shape[0]:
            raise ValueError(
                "Batch-size mismatch in positive control."
            )

        if output.shape[2] != cached.shape[1]:
            raise ValueError(
                "Hidden-size mismatch in positive control."
            )

        patched = output.clone()

        original = output.detach().clone()

        replacement = cached.to(
            device=output.device,
            dtype=output.dtype,
        )

        patched[:, -1, :] = replacement

        final_difference = (
            patched[:, -1, :]
            - original[:, -1, :]
        ).abs().max().item()

        if output.shape[1] > 1:
            prefix_difference = (
                patched[:, :-1, :]
                - original[:, :-1, :]
            ).abs().max().item()
        else:
            prefix_difference = 0.0

        patch_statistics[
            "max_final_token_difference"
        ] = final_difference

        patch_statistics[
            "max_prefix_difference"
        ] = prefix_difference

        if prefix_difference != 0.0:
            raise RuntimeError(
                "Positive-control patch modified "
                "non-final token positions."
            )

        return patched

    handle = layer.register_forward_hook(
        identity_patch
    )

    try:
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=False)

    finally:
        handle.remove()

    if not hasattr(outputs, "logits"):
        raise AttributeError(
            "Positive-control model output lacks logits."
        )

    logits_difference = (
        baseline.logits -
        outputs.logits
    ).abs()

    max_logit_difference = (
        logits_difference.max().item()
    )

    baseline_prediction = (
        baseline.logits[:, -1, :]
        .argmax(dim=-1)
    )

    patched_prediction = (
        outputs.logits[:, -1, :]
        .argmax(dim=-1)
    )

    prediction_unchanged = bool(
        torch.equal(
            baseline_prediction,
            patched_prediction,
        )
    )

    # FP16 inference can introduce tiny numerical differences.
    # Keep the tolerance explicit rather than hiding it.
    tolerance = 1e-4

    logits_within_tolerance = (
        max_logit_difference < tolerance
    )

    passed = (
        logits_within_tolerance
        and prediction_unchanged
        and patch_statistics[
            "max_prefix_difference"
        ] == 0.0
    )

    return {
        "layer": layer_idx,
        "max_logit_difference": max_logit_difference,
        "logit_tolerance": tolerance,
        "prediction_unchanged": prediction_unchanged,
        "max_final_token_activation_difference": (
            patch_statistics[
                "max_final_token_difference"
            ]
        ),
        "max_prefix_activation_difference": (
            patch_statistics[
                "max_prefix_difference"
            ]
        ),
        "passed": passed,
    }




def validate_patch_configuration(model) -> None:
    """
    Validate all structural assumptions required by MIMESIS patching.
    """

    validate_qwen_layer_structure(model)

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

    if num_layers != 36:
        raise ValueError(
            "MIMESIS V1 is currently configured for Qwen3-4B "
            f"with 36 layers, but model reports {num_layers}."
        )

    if model.config.hidden_size != 2560:
        raise ValueError(
            "MIMESIS V1 is currently configured for Qwen3-4B "
            f"with hidden size 2560, but model reports "
            f"{model.config.hidden_size}."
        )

    if not hasattr(model, "lm_head"):
        raise AttributeError(
            "Model does not expose lm_head."
        )




if __name__ == "__main__":

    print(
        "patching.py contains the MIMESIS activation-"
        "patching routines."
    )

    print(
        "Expected architecture: Qwen3ForCausalLM"
    )

    print(
        "Expected transformer layers: 36"
    )

    print(
        "Expected hidden size: 2560"
    )

    print(
        "Patch component: residual_stream"
    )

    print(
        "Patch position: final_input_token"
    )

    print(
        "Run model loading, token-position inspection, "
        "positive-control tests, and scientific "
        "interventions from the dedicated notebooks."
    )