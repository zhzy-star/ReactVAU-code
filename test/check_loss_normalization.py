"""Check that accumulation gives the same gradient as one complete batch."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
runtime = Path(__file__).resolve().parent / "artifacts" / "runtime"
for name, suffix in (("XDG_CACHE_HOME", "cache"), ("HF_HOME", "huggingface"),
                     ("TRITON_CACHE_DIR", "triton")):
    target = runtime / suffix
    target.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(target)

import torch
import torch.nn.functional as F

from train_paligemma_vad_384_ddp import DDPVADTrainer


class TinyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 5, dtype=torch.float64)
        self.head = torch.nn.Linear(5, 8, dtype=torch.float64)

    def forward(self, input_ids, labels, **kwargs):
        logits = self.head(self.embedding(input_ids))
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 8), labels[:, 1:].reshape(-1))
        return SimpleNamespace(logits=logits, loss=loss)


def run_checks():
    torch.manual_seed(42)
    reference = TinyLanguageModel()
    labels = torch.tensor([[-100, 1, 2, 3, 4, 5], [-100, -100, 2, 3, 4, 5],
                           [-100, -100, -100, 3, 4, 5], [-100, -100, -100, -100, 4, 5],
                           [-100, -100, -100, -100, -100, 5]])
    weights = (labels != -100).double() * torch.tensor([1., 2., 3., 4., 5., 6.])
    batch = {"input_ids": torch.randint(0, 8, (5, 6)), "labels": labels,
             "attention_mask": torch.ones(5, 6), "pixel_values": torch.zeros(5, 1),
             "token_weights": weights}
    checks = []
    for label_format in ("binary", "text"):
        for sizes in ((2, 2, 1), (1, 4)):
            trainer = object.__new__(DDPVADTrainer)
            trainer.args = SimpleNamespace(device=torch.device("cpu"), average_tokens_across_devices=False)
            trainer.label_format = label_format
            trainer.yes_token_id, trainer.no_token_id = 1, 2
            trainer._accumulated_yes_probs = []
            trainer._accumulated_no_probs = []
            trainer._accumulation_count = 0
            full_model, accumulated_model = TinyLanguageModel(), TinyLanguageModel()
            full_model.load_state_dict(reference.state_dict())
            accumulated_model.load_state_dict(reference.state_dict())
            full_loss = trainer.compute_loss(full_model, {k: v.clone() for k, v in batch.items()})
            full_loss.backward()
            start, micros = 0, []
            for size in sizes:
                micros.append({k: v[start:start + size].clone() for k, v in batch.items()})
                start += size
            micros, denominator = trainer.get_batch_samples(iter(micros), len(micros))
            accumulated_loss = 0.
            for micro in micros:
                loss = trainer.compute_loss(accumulated_model, micro, num_items_in_batch=denominator)
                accumulated_loss += loss.item()
                loss.backward()
            torch.testing.assert_close(torch.tensor(accumulated_loss, dtype=torch.float64),
                                       full_loss.detach(), rtol=1e-12, atol=1e-12)
            max_error = 0.
            for full, accumulated in zip(full_model.parameters(), accumulated_model.parameters()):
                torch.testing.assert_close(full.grad, accumulated.grad, rtol=1e-12, atol=1e-12)
                max_error = max(max_error, (full.grad - accumulated.grad).abs().max().item())
            checks.append({"label_format": label_format, "microbatch_sizes": sizes,
                           "max_gradient_error": max_error})
    return checks


if __name__ == "__main__":
    for check in run_checks():
        print("PASS", check)
