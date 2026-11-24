"""Model loading and trainer construction (the TranslateGemma SFT recipe, plan §05).

Planned implementation:

* **Model**: ``AutoModelForImageTextToText`` / ``AutoModelForCausalLM`` from ``cfg.model.init_from``
  at ``cfg.model.revision``, ``torch_dtype=bfloat16`` for compute with fp32 master weights.
  Freeze every parameter matching ``cfg.model.freeze`` (token embeddings, which the output layer
  shares, and the vision encoder): ≈10.8B of 12.2B parameters train.
* **Trainer**: TRL ``SFTTrainer`` with ``SFTConfig(packing=cfg.data.packing,
  max_length=cfg.data.max_length, completion_only_loss=True, bf16=True,
  gradient_checkpointing=True, gradient_accumulation_steps=cfg.training.grad_accum)``.
  The per-device batch is set from ``micro_batch_tokens`` (padding-free packing).
* **Optimizer**: ``transformers.optimization.Adafactor(lr=cfg.training.learning_rate, beta1=None,
  scale_parameter=False, relative_step=False, warmup_init=False)``, without momentum and factored.
* **Schedule**: ``cosine`` = warm-up 3% then cosine to ``min_lr_ratio`` (10%);
  ``wsd`` = warm-up, constant, then a (1−sqrt) cooldown over the final ``cooldown_ratio``.
  This needs a custom ``LambdaLR``. Cooldown branches (TG-10-CD06/CD10) resume a constant-rate
  checkpoint with a ``wsd`` schedule whose constant phase is zero.
* **Sharding**: accelerate FSDP2 (``fsdp_version: 2``), which keeps 2-D weight shapes so Adafactor
  stays factored; fused linear cross-entropy.
* **Saving**: ``cfg.training.save``. Either ``strategy: end``, or ``at_examples: [...]`` with
  ``with_optimizer: true`` (TG-10's size-check checkpoints). bf16 weights go to ``cfg.output.local_dir``
  and are pushed to ``cfg.output.hub_repo`` per ``cfg.output.push``.
* **Selection**: dev only (``data.eval``); test sets are never touched during training.
"""

from __future__ import annotations

from typing import Any

from src.pipeline.config import RunConfig


def load_model_and_tokenizer(cfg: RunConfig) -> tuple[Any, Any]:
    raise NotImplementedError("trainer.load_model_and_tokenizer")


def build_trainer(cfg: RunConfig, model: Any, tokenizer: Any, train_ds: Any, eval_sets: dict[str, Any]):
    raise NotImplementedError("trainer.build_trainer")


def push_to_hub(cfg: RunConfig, trainer: Any, manifest: dict[str, Any]) -> None:
    raise NotImplementedError("trainer.push_to_hub")
