"""Shared PaliGemma/StreamForest compatibility for training and inference."""

import inspect
import logging
from pathlib import Path
import types

import torch
from peft import PeftModel
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


def last_token_logits(model, inputs):
    """Score the last prompt token with the supported Transformers keyword.

    Avoid materializing vocabulary logits for all 729 image tokens when the
    installed model supports keeping only the final position. Fall back to
    full logits for right-padded prompts and older model implementations.
    """
    attention = inputs["attention_mask"]
    kwargs = {}
    if attention[:, -1].bool().all():
        parameters = inspect.signature(_unwrap_model(model).forward).parameters
        for name in ("logits_to_keep", "num_logits_to_keep"):
            if name in parameters:
                kwargs[name] = 1
                break
    outputs = model(
        input_ids=inputs["input_ids"], attention_mask=attention,
        pixel_values=inputs["pixel_values"], use_cache=False, **kwargs,
    )
    if kwargs:
        return outputs.logits[:, -1]
    positions = torch.arange(attention.shape[1], device=attention.device)
    last_positions = positions.masked_fill(~attention.bool(), -1).max(dim=1).values
    return outputs.logits[torch.arange(attention.shape[0], device=attention.device), last_positions]


def _unwrap_model(model):
    return model.get_base_model() if isinstance(model, PeftModel) else model


def _get_vision_tower(model):
    model = _unwrap_model(model)
    if hasattr(model, "vision_tower"):
        return model.vision_tower
    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        return model.model.vision_tower
    raise AttributeError(f"Cannot locate PaliGemma vision tower on {type(model).__name__}")


def validate_lora_path(lora_path):
    """Reject an invalid requested local adapter instead of scoring the base model."""
    if not lora_path:
        return
    adapter_dir = Path(lora_path).expanduser()
    if not (adapter_dir / "adapter_config.json").is_file():
        raise ValueError(
            f"LoRA directory has no adapter_config.json: {adapter_dir}. "
            "Set PALIGEMMA_LORA_PATH to the completed training_<timestamp> "
            "directory or checkpoint directory containing the adapter."
        )
    if not any((adapter_dir / name).is_file() for name in (
        "adapter_model.safetensors", "adapter_model.bin"
    )):
        raise ValueError(f"LoRA directory has no adapter_model weights: {adapter_dir}")


def load_streamforest_vision_weights(model, weights_path, image_size=384):
    """Load legacy SigLIP keys into flat or nested Transformers vision towers."""
    logger.info("Loading StreamForest vision weights from: %s", weights_path)
    stream_weights = load_file(weights_path)
    remapped = {}
    for key, value in stream_weights.items():
        key = key.removeprefix("vision_tower.vision_tower.")
        if key.startswith("vision_model."):
            remapped[key] = value

    tower = _get_vision_tower(model)
    vision_model = getattr(tower, "vision_model", tower)
    embeddings = vision_model.embeddings
    prefix = "vision_model." if vision_model is not tower else ""
    if not prefix:
        remapped = {key.removeprefix("vision_model."): value
                    for key, value in remapped.items()}
    pos_key = prefix + "embeddings.position_embedding.weight"
    if pos_key not in remapped:
        raise ValueError(f"StreamForest weights are missing {pos_key}: {weights_path}")

    positions = remapped[pos_key]
    old_weight = embeddings.position_embedding.weight
    expected_positions = (image_size // embeddings.patch_size) ** 2
    if positions.ndim != 2 or positions.shape != (expected_positions, old_weight.shape[1]):
        raise ValueError(
            f"Invalid position embedding {tuple(positions.shape)} for "
            f"image_size={image_size}; expected {(expected_positions, old_weight.shape[1])}"
        )
    logger.info("  Position embedding: %s -> %s", tuple(old_weight.shape), tuple(positions.shape))
    embeddings.position_embedding = torch.nn.Embedding(
        *positions.shape, device=old_weight.device, dtype=old_weight.dtype
    )
    missing, unexpected = tower.load_state_dict(remapped, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected StreamForest vision keys: {unexpected[:5]}")
    logger.info("  Missing keys (kept from PaliGemma): %d; unexpected: %d", len(missing), len(unexpected))

    embeddings.num_positions = expected_positions
    embeddings.num_patches = expected_positions
    embeddings.image_size = image_size
    embeddings.register_buffer(
        "position_ids", torch.arange(expected_positions, device=old_weight.device).expand((1, -1)),
        persistent=False,
    )
    base = _unwrap_model(model)
    for config in (base.config.vision_config, tower.config):
        config.image_size = image_size
        config.num_image_tokens = expected_positions
        config.num_positions = expected_positions
    base.config.text_config.num_image_tokens = expected_positions
    return model


def patch_vision_feature_layer(model, select_layer=-2):
    """Select hidden states before native projection, preserving version-specific APIs.

    Transformers 4 returns a tensor from get_image_features; Transformers 5
    consumes a model output on the inner model. Patching the tower works for
    both and keeps each version's native projection/scaling intact.
    """
    tower = _get_vision_tower(model)
    num_states = tower.config.num_hidden_layers + 1
    if not -num_states <= select_layer < num_states:
        raise ValueError(f"Vision layer {select_layer} is outside {num_states} hidden states")
    if not hasattr(tower, "_reactvau_original_forward"):
        tower._reactvau_original_forward = tower.forward
    tower._reactvau_select_layer = select_layer

    def forward_with_layer_select(self, *args, **kwargs):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        outputs = self._reactvau_original_forward(*args, **kwargs)
        outputs.last_hidden_state = outputs.hidden_states[self._reactvau_select_layer]
        return outputs

    tower.forward = types.MethodType(forward_with_layer_select, tower)
    logger.info("  Selected vision feature layer %d", select_layer)
    return model
