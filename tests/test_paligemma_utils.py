"""CPU regressions for the Transformers 4/5 PaliGemma compatibility boundary."""
import re
import tempfile
import unittest
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from transformers import PaliGemmaConfig, PaliGemmaForConditionalGeneration

from vad.paligemma_utils import (
    _get_vision_tower, load_streamforest_vision_weights,
    patch_vision_feature_layer, validate_lora_path, last_token_logits,
)


def tiny_model(image_size=28):
    config = PaliGemmaConfig(
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                           num_attention_heads=2, image_size=image_size, patch_size=14),
        text_config=dict(model_type='gemma2', hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, vocab_size=128, pad_token_id=0),
        projection_dim=32, image_token_index=127,
    )
    return PaliGemmaForConditionalGeneration(config).eval()


class LegacyModel(torch.nn.Module):
    """Old flat PaliGemma and nested SigLIP layout with native tensor scaling."""
    def __init__(self, tower, config):
        super().__init__()
        self.vision_tower = torch.nn.Module()
        self.vision_tower.vision_model = tower
        self.vision_tower.config = tower.config
        self.vision_tower.forward = tower.forward
        self.multi_modal_projector = torch.nn.Linear(16, 32)
        self.config = config

    def get_image_features(self, pixels):
        return self.multi_modal_projector(self.vision_tower(pixels).last_hidden_state) / 32**0.5


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.model = tiny_model()
        self.pixels = torch.randn(1, 3, 28, 28)

    def test_native_feature_api_and_repeat_patch(self):
        tower = _get_vision_tower(self.model)
        hidden = tower(self.pixels, output_hidden_states=True).hidden_states
        projector = self.model.model.multi_modal_projector
        expected = projector(hidden[-2])
        last = self.model.get_image_features(self.pixels).pooler_output
        self.assertFalse(torch.allclose(last, expected))
        patch_vision_feature_layer(self.model, -3)
        patch_vision_feature_layer(self.model, -2)
        result = self.model.get_image_features(self.pixels)
        torch.testing.assert_close(result.pooler_output, expected)
        with self.assertRaises(ValueError):
            patch_vision_feature_layer(self.model, -100)

    def test_legacy_native_tensor_api_and_scaling(self):
        tower = _get_vision_tower(self.model)
        legacy = LegacyModel(tower, self.model.config)
        selected = tower(self.pixels, output_hidden_states=True).hidden_states[-2]
        expected = legacy.multi_modal_projector(selected) / 32**0.5
        patch_vision_feature_layer(legacy)
        torch.testing.assert_close(legacy.get_image_features(self.pixels), expected)

    def test_weights_resize_flat_and_nested_layouts(self):
        source = _get_vision_tower(self.model)
        for nested in (False, True):
            target = tiny_model(image_size=42)
            if nested:
                target = LegacyModel(_get_vision_tower(target), target.config)
            with tempfile.TemporaryDirectory() as tmp:
                weights = Path(tmp) / 'vision.safetensors'
                save_file({'vision_tower.vision_tower.vision_model.' + k: v.contiguous()
                           for k, v in source.state_dict().items()}, weights)
                load_streamforest_vision_weights(target, weights, image_size=28)
            tower = _get_vision_tower(target)
            inner = getattr(tower, 'vision_model', tower)
            self.assertEqual(inner.embeddings.num_positions, 4)
            self.assertEqual(inner.embeddings.num_patches, 4)
            self.assertEqual(inner.embeddings.image_size, 28)
            self.assertEqual(tuple(inner.embeddings.position_ids.shape), (1, 4))
            torch.testing.assert_close(tower(self.pixels).last_hidden_state,
                                       source(self.pixels).last_hidden_state)

    def test_invalid_adapter_and_weights_fail_clearly(self):
        validate_lora_path(None)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'adapter_config.json'):
                validate_lora_path(tmp)
            weights = Path(tmp) / 'bad.safetensors'
            save_file({'unrelated': torch.ones(1)}, weights)
            with self.assertRaisesRegex(ValueError, 'missing'):
                load_streamforest_vision_weights(self.model, weights)

    def test_peft_backward_with_frozen_vision(self):
        # Same language-only target expression as Stage 1; both historical layouts.
        pattern = (r'(?:model\.language_model|language_model\.model)\.layers\.\d+\.'
                   r'(self_attn\.(q_proj|k_proj|v_proj|o_proj)|'
                   r'mlp\.(gate_proj|up_proj|down_proj))')
        self.assertIsNotNone(re.fullmatch(pattern, 'language_model.model.layers.0.self_attn.q_proj'))
        model = get_peft_model(self.model, LoraConfig(
            r=2, lora_alpha=4, target_modules=pattern, task_type=TaskType.CAUSAL_LM,
            modules_to_save=['multi_modal_projector'],
        ))
        patch_vision_feature_layer(model)
        tower = _get_vision_tower(model)
        self.assertFalse(any(p.requires_grad for p in tower.parameters()))
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        ids = torch.tensor([[127, 127, 127, 127, 2, 3, 4]])
        labels = torch.tensor([[-100, -100, -100, -100, -100, 3, 4]])
        result = model(input_ids=ids, pixel_values=self.pixels,
                       attention_mask=torch.ones_like(ids), labels=labels,
                       token_type_ids=torch.tensor([[0, 0, 0, 0, 0, 1, 1]]), use_cache=False)
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(any('lora_B' in n and g is not None and g.abs().sum() > 0
                            for n, g in grads.items()))
        self.assertTrue(any('multi_modal_projector' in n and g is not None and g.abs().sum() > 0
                            for n, g in grads.items()))
        self.assertTrue(all(p.grad is None for p in tower.parameters()))

    def test_last_logits_matches_full_forward(self):
        ids = torch.tensor([[127, 127, 127, 127, 2, 3]])
        inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=self.pixels)
        with torch.no_grad():
            expected = self.model(**inputs, use_cache=False).logits[:, -1]
            actual = last_token_logits(self.model, inputs)
        torch.testing.assert_close(actual, expected)

    def test_last_logits_handles_both_padding_sides_and_old_keyword(self):
        class LegacyScorer(torch.nn.Module):
            def forward(self, input_ids, attention_mask, pixel_values, use_cache, num_logits_to_keep=0):
                self.kept = num_logits_to_keep
                logits = input_ids.float().unsqueeze(-1).expand(-1, -1, 3)
                return type('Output', (), {'logits': logits[:, -num_logits_to_keep:] if num_logits_to_keep else logits})()

        model = LegacyScorer()
        inputs = dict(input_ids=torch.tensor([[0, 3, 4], [5, 6, 0]]),
                      attention_mask=torch.tensor([[0, 1, 1], [1, 1, 0]]), pixel_values=None)
        result = last_token_logits(model, inputs)
        torch.testing.assert_close(result[:, 0], torch.tensor([4., 6.]))
        self.assertEqual(model.kept, 0)
        inputs['attention_mask'] = torch.tensor([[0, 1, 1], [1, 1, 1]])
        last_token_logits(model, inputs)
        self.assertEqual(model.kept, 1)


if __name__ == '__main__':
    unittest.main()
