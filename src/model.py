

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config


@dataclass
class ModelBundle:
    """Loaded MIMESIS model and tokenizer."""

    model: Any
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype

    @property
    def num_layers(self) -> int:
        """Return the number of transformer layers."""
        return self.model.config.num_hidden_layers

    @property
    def hidden_size(self) -> int:
        """Return the model hidden size."""
        return self.model.config.hidden_size


def resolve_device() -> torch.device:
    """Resolve the configured device."""

    if config.DEVICE == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device(config.DEVICE)


def resolve_dtype(device: torch.device) -> torch.dtype:
    """Resolve the configured inference dtype."""

    if config.DTYPE == "float16":
        return torch.float16

    if config.DTYPE == "float32":
        return torch.float32

    if config.DTYPE == "bfloat16":
        return torch.bfloat16

    raise ValueError(
        f"Unsupported dtype: {config.DTYPE}"
    )


def load_tokenizer():
    """Load the tokenizer for the frozen model."""

    tokenizer = AutoTokenizer.from_pretrained(
        config.TOKENIZER_NAME,
        revision=config.MODEL_REVISION,
        trust_remote_code=True,
    )

    return tokenizer


def load_model(
    device: torch.device,
    dtype: torch.dtype,
):
    """Load Qwen3-3B for inference."""

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        revision=config.MODEL_REVISION,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    return model


def load_model_bundle() -> ModelBundle:
    """Load and validate the complete MIMESIS model bundle."""

    config.validate_config()

    device = resolve_device()
    dtype = resolve_dtype(device)

    tokenizer = load_tokenizer()
    model = load_model(device, dtype)

    bundle = ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )

    validate_model(bundle)

    return bundle


def validate_model(bundle: ModelBundle) -> None:
    """Validate assumptions needed by MIMESIS."""

    model = bundle.model

    if bundle.num_layers <= 0:
        raise ValueError(
            "Model reports no transformer layers."
        )

    if bundle.hidden_size <= 0:
        raise ValueError(
            "Model reports an invalid hidden size."
        )

    if model.training:
        raise ValueError(
            "Model must be in evaluation mode."
        )

    if bundle.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but CUDA is unavailable."
            )


def get_model_info(bundle: ModelBundle) -> dict:
    """Return model metadata for logging and reproducibility."""

    return {
        "model_name": config.MODEL_NAME,
        "model_revision": config.MODEL_REVISION,
        "tokenizer_name": config.TOKENIZER_NAME,
        "device": str(bundle.device),
        "dtype": str(bundle.dtype),
        "num_layers": bundle.num_layers,
        "hidden_size": bundle.hidden_size,
    }


@torch.inference_mode()
def generate(
    bundle: ModelBundle,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    **generation_kwargs,
):
    """Generate from the loaded model."""

    input_ids = input_ids.to(bundle.device)

    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.device)

    return bundle.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_kwargs,
    )


def tokenize(
    bundle: ModelBundle,
    text: str,
):
    """Tokenize a rendered prompt."""

    encoded = bundle.tokenizer(
        text,
        return_tensors="pt",
    )

    return {
        key: value.to(bundle.device)
        for key, value in encoded.items()
    }


def decode(
    bundle: ModelBundle,
    token_ids,
) -> str:
    """Decode token IDs into text."""

    return bundle.tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
    )