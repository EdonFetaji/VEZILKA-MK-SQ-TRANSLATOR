"""Model loading and trainer construction for one NVIDIA DGX Spark (GB10), plan §3.

The recipe follows ``src/configs/tg_hyperparameters.csv``:

* **Model**: ``AutoModelForImageTextToText`` in fp32 (master weights) + ``AutoProcessor``. Parameters
  matching ``cfg.model.freeze`` (fnmatch) are frozen and, under bf16 autocast, cast to bf16 (they get no
  optimizer state, so fp32 buys nothing). About 10.8B of 12.2B parameters train.
* **Precision**: ``bf16=True`` means accelerate's autocast over the fp32 weights, with
  ``cache_enabled=False`` (CSV runtime_notes). Weights are never loaded in bf16.
* **Optimizer**: ``transformers.optimization.Adafactor`` from ``cfg.training.optimizer`` (no momentum,
  factored, fixed lr), with a ``LambdaLR`` over :func:`lr_at`, the CSV ``lr_formula`` column.
* **Steps are global**: a branch (TG-10-CD06/CD10/EXT) starts at ``resume.start_global_step``; the
  trainer counts local steps from 0 and the schedule adds the offset.
* **Loss**: TRL's SFTTrainer computes the completion-only loss through its fused LM head (chunked
  log-probs; the 262k-vocab logits are never built).
* **Layout** under ``runs/<id>/``: ``checkpoints/checkpoint-N`` (rotated, 2 kept),
  ``checkpoints/keep/step-NNNNNN`` (``save.keep_steps``, never rotated), ``metrics/train_log.jsonl``,
  ``metrics/dev_loss.jsonl`` and ``model/`` (final bf16 weights + processor). Pushing to the Hub is the
  orchestrator's job (``training_pipeline.py``).
"""

from __future__ import annotations

import fnmatch
import json
import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from src.pipeline.config import RunConfig

log = logging.getLogger(__name__)

#: Trainable-parameter window for the 12B model (CSV: ~10.8B); checked only for models > 5e9 params.
TRAINABLE_RANGE = (10.5e9, 11.0e9)
#: Disk per checkpoint (fp32 weights + Adafactor state), for the preflight check.
CHECKPOINT_BYTES = 50e9


# ── paths ─────────────────────────────────────────────────────────────────────


def run_dir_for(cfg: RunConfig, run_dir: str | Path | None = None) -> Path:
    return Path(run_dir or cfg.output.local_dir)


def keep_dir(run_dir: Path, global_step: int) -> Path:
    return run_dir / "checkpoints" / "keep" / f"step-{int(global_step):06d}"


def branch_source(cfg: RunConfig, run_dir: Path) -> Path | None:
    """The parent's kept checkpoint a branch run starts from (``runs/<from_run>/checkpoints/keep/...``)."""
    r = cfg.training.resume
    if not r.get("from_run"):
        return None
    return keep_dir(run_dir.parent / r["from_run"], r["from_step"])


def start_step(cfg: RunConfig) -> int:
    return int(cfg.training.resume.get("start_global_step") or 0)


# ── learning-rate schedule ──────────────────────────────────────────────────


def lr_at(step: int, sched: dict[str, Any], max_steps: int | None = None) -> float:
    """The CSV ``lr_formula`` at global step ``step`` (the lr used by update ``step + 1``).

    ``max_steps`` is the global step the run ends at (``start_global_step + training.max_steps``); only
    the cosine schedule needs it.
    """
    kind, peak, lo = sched["type"], float(sched["peak"]), float(sched["min"])
    warmup = int(sched.get("warmup_steps") or 0)

    def sqrt_cooldown(s: float) -> float:
        n = int(sched["cooldown_steps"])
        return lo + (peak - lo) * (1 - math.sqrt(min(max(s, 0), n) / n))

    if kind == "cooldown":  # no warmup; starts at the branch point
        return sqrt_cooldown(step - int(sched.get("constant_end_step") or 0))

    if step < warmup:
        start = float(sched["warmup_start"] if sched.get("warmup_start") is not None else peak)
        return start + (peak - start) * step / warmup

    if kind == "wsd":
        end = int(sched["constant_end_step"])
        return peak if step < end else sqrt_cooldown(step - end)

    if kind == "cosine":
        if max_steps is None:
            raise ValueError("lr_at: the cosine schedule needs max_steps")
        p = min(max((step - warmup) / max(max_steps - warmup, 1), 0.0), 1.0)
        return lo + 0.5 * (peak - lo) * (1 + math.cos(math.pi * p))

    raise ValueError(f"lr_at: unknown schedule type {kind!r}")


def build_optimizer(cfg: RunConfig, model: Any):
    from transformers.optimization import Adafactor

    o, peak = cfg.training.optimizer, float(cfg.training.lr_schedule["peak"])
    if o.get("name", "adafactor") != "adafactor":
        raise ValueError(f"optimizer {o.get('name')!r} is not supported; the recipe uses adafactor")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = Adafactor(
        params,
        lr=peak,
        eps=tuple(float(e) for e in o["eps"]),
        clip_threshold=float(o["clip_threshold"]),
        decay_rate=float(o["decay_rate"]),
        beta1=o["beta1"],
        weight_decay=float(o["weight_decay"]),
        scale_parameter=bool(o["scale_parameter"]),
        relative_step=bool(o["relative_step"]),
        warmup_init=bool(o["warmup_init"]),
    )
    log.info("optimizer: Adafactor %s", {k: v for k, v in opt.defaults.items()})
    return opt


def build_scheduler(cfg: RunConfig, optimizer: Any):
    """``LambdaLR`` with the optimizer at lr = peak; the lambda maps local step → ``lr_at(global)/peak``."""
    sched, offset = cfg.training.lr_schedule, start_step(cfg)
    peak, end = float(sched["peak"]), offset + int(cfg.training.max_steps)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_at(offset + s, sched, end) / peak
    )


# ── model ─────────────────────────────────────────────────────────────────────


def _resolve_init(cfg: RunConfig) -> tuple[str, str | None, str | None]:
    """``run:<ID>`` → the parent's pushed or local ``model/`` (``hub.resolve_init``); else the id/path itself."""
    init = cfg.model.init_from
    if not init.startswith("run:"):
        return init, None, cfg.model.revision
    try:
        from src.pipeline import hub
    except ImportError:
        return init.removeprefix("run:"), None, cfg.model.revision
    return hub.resolve_init(cfg, hub.load_state())


def allow_text_only_flash_attention(attn: str | None) -> list[str]:
    """Let Gemma3's multimodal classes run Flash Attention for text-only training.

    transformers marks ``Gemma3Model`` / ``Gemma3ForConditionalGeneration`` ``_supports_flash_attn = False``
    because Flash Attention cannot apply the bidirectional mask *between image tokens*. Our training and dev
    batches are text only (no ``pixel_values``, no image tokens), so that mask never exists and the causal +
    sliding-window attention Flash Attention computes is exactly the text model's. The text classes
    (``Gemma3TextModel``, ``Gemma3ForCausalLM``) support it already. Only touched when a flash implementation
    is requested; returns the classes it changed.
    """
    if not attn or "flash" not in attn:
        return []
    from transformers.models.gemma3 import modeling_gemma3 as g

    changed = []
    for cls in (g.Gemma3Model, g.Gemma3ForConditionalGeneration):
        if not cls._supports_flash_attn:
            cls._supports_flash_attn = True
            changed.append(cls.__name__)
    if changed:
        log.warning("attention: %s allowed for %s (text-only training: no image tokens, so the image "
                    "bidirectional mask that blocks it does not apply)", attn, ", ".join(changed))
    return changed


def load_model_and_tokenizer(cfg: RunConfig, run_dir: str | Path | None = None) -> tuple[Any, Any]:
    """fp32 model + processor, frozen per ``cfg.model.freeze``; branches load the parent's kept step."""
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    rd = run_dir_for(cfg, run_dir)
    src = branch_source(cfg, rd)
    if src is not None:
        path, kw = str(src), {}
        log.info("branch: weights from %s", src)
    else:
        repo, subfolder, revision = _resolve_init(cfg)
        path, kw = repo, {"revision": revision, **({"subfolder": subfolder} if subfolder else {})}

    cuda = torch.cuda.is_available()
    attn = cfg.model.attn_implementation if cuda else "sdpa"
    allow_text_only_flash_attention(attn)
    model = AutoModelForImageTextToText.from_pretrained(
        path,
        dtype=getattr(torch, cfg.model.dtype),
        attn_implementation=attn,
        device_map="cuda" if cuda else None,  # straight to the device: no second copy in unified memory
        **kw,
    )
    try:
        processor = AutoProcessor.from_pretrained(path, **kw)
    except (OSError, ValueError, TypeError):
        processor = AutoTokenizer.from_pretrained(path, **kw)

    autocast = cfg.training.bf16 and cuda
    freeze_parameters(model, cfg.model.freeze, cast_frozen_to=torch.bfloat16 if autocast else None)
    return model, processor


def freeze_parameters(model: Any, patterns: list[str], cast_frozen_to: torch.dtype | None = None) -> int:
    """Freeze parameters whose name matches any fnmatch pattern; return the trainable count."""
    total = trainable = 0
    for name, p in model.named_parameters():
        frozen = any(fnmatch.fnmatch(name, pat) for pat in patterns)
        p.requires_grad_(not frozen)
        if frozen and cast_frozen_to is not None:
            p.data = p.data.to(cast_frozen_to)
        total += p.numel()
        trainable += 0 if frozen else p.numel()
    log.info("parameters: %.3fB total, %.3fB trainable, %.3fB frozen", total / 1e9, trainable / 1e9,
             (total - trainable) / 1e9)
    lo, hi = TRAINABLE_RANGE
    if total > 5e9 and not lo < trainable < hi:
        raise RuntimeError(f"trainable parameters {trainable / 1e9:.3f}B outside ({lo / 1e9}, {hi / 1e9})B")
    return trainable


# ── preflight ───────────────────────────────────────────────────────────────


def preflight(cfg: RunConfig, run_dir: str | Path | None = None) -> None:
    """GPU free of other jobs, attention kernel loads, enough disk. Skipped (with a note) without CUDA."""
    if not torch.cuda.is_available():
        log.warning("preflight: no CUDA device; skipping GPU and disk checks")
        return
    if shutil.which("nvidia-smi"):
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        ).stdout
        others = [ln for ln in out.splitlines() if ln.strip() and ln.split(",")[0].strip() != str(os.getpid())]
        others = [ln for ln in others if ln.split(",")[0].strip().isdigit()]
        if others:
            raise RuntimeError("preflight: other CUDA processes are running (no vLLM during training): "
                               + "; ".join(others))
    cap = torch.cuda.get_device_capability()
    log.info("preflight: %s, sm_%d%d", torch.cuda.get_device_name(), *cap)
    attn = cfg.model.attn_implementation or ""
    if "/" in attn:  # a hub kernel, e.g. kernels-community/flash-attn2: load it now, before the 24 GB model
        from transformers.integrations.hub_kernels import load_and_register_attn_kernel

        try:
            load_and_register_attn_kernel(attn)
        except Exception as e:
            raise RuntimeError(f"preflight: attention kernel {attn!r} does not load here ({e}); "
                               "set model.attn_implementation: sdpa to fall back") from e
        log.info("preflight: attention kernel %s loaded", attn)
    rd = run_dir_for(cfg, run_dir)
    probe = next((p for p in (rd, *rd.parents) if p.exists()), Path("."))
    need = (2 + len(cfg.training.save.get("keep_steps") or [])) * CHECKPOINT_BYTES
    free = shutil.disk_usage(probe).free
    if free < need:
        raise RuntimeError(f"preflight: {free / 1e9:.0f} GB free under {probe}, need {need / 1e9:.0f} GB")


# ── trainer ───────────────────────────────────────────────────────────────────


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _copy_checkpoint(src: Path, dst: Path, with_optimizer: bool) -> None:
    """Hard-link (or copy) a checkpoint; the trainer never rewrites a finished checkpoint's files."""
    if dst.exists():
        shutil.rmtree(dst)

    def link(s: str, d: str) -> None:
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)

    ignore = None if with_optimizer else shutil.ignore_patterns("optimizer.pt")
    shutil.copytree(src, dst, copy_function=link, ignore=ignore)


SAVE_SHARD_SIZE = "2GB"  # safetensors copies a whole shard to CPU before writing; on the GB10 that is the same
                         # 119 GB pool as the 49 GB fp32 weights, so the 50GB default shard gets the process OOM-killed


def release_memory() -> None:
    """Return the CUDA caching allocator's free blocks to the (unified) system pool."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def small_shard_saves(model: Any, max_shard_size: str = SAVE_SHARD_SIZE) -> None:
    """Make every ``model.save_pretrained`` (Trainer checkpoints and ``save_final``) write small shards."""
    orig = model.save_pretrained

    def save_pretrained(*args, **kwargs):
        kwargs.setdefault("max_shard_size", max_shard_size)
        return orig(*args, **kwargs)

    model.save_pretrained = save_pretrained


def _callbacks(cfg: RunConfig, run_dir: Path):
    from transformers import TrainerCallback

    offset = start_step(cfg)
    keep = {int(s) - offset for s in cfg.training.save.get("keep_steps") or [] if int(s) > offset}
    with_opt = bool(cfg.training.save.get("with_optimizer"))
    metrics = run_dir / "metrics"

    class KeepAndLog(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if state.global_step in keep:
                control.should_save = True
            if control.should_save or control.should_evaluate:
                release_memory()  # unified memory: cached activation/grad blocks would starve the save copy
            return control

        def on_save(self, args, state, control, **kw):
            if state.global_step in keep:
                src = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                dst = keep_dir(run_dir, offset + state.global_step)
                _copy_checkpoint(src, dst, with_opt)
                log.info("kept %s", dst)
            return control

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs:
                return control
            row = {"step": offset + state.global_step, "local_step": state.global_step,
                   "tokens": state.num_input_tokens_seen, **logs}
            if any(k.startswith("eval_") for k in logs):
                _append_jsonl(metrics / "dev_loss.jsonl", row)
            elif "loss" in logs:
                _append_jsonl(metrics / "train_log.jsonl", row)
            return control

    from src.pipeline import tracking

    return [KeepAndLog(), tracking.callback(cfg, run_dir, offset)]


def build_trainer(
    cfg: RunConfig,
    model: Any,
    tokenizer: Any,
    train_ds: Any,
    eval_sets: dict[str, Any],
    run_dir: str | Path | None = None,
    continue_parent_stream: bool = True,
):
    """TRL ``SFTTrainer`` for the single-device recipe. ``tokenizer`` is the processor (or a tokenizer)."""
    from accelerate.utils import AutocastKwargs
    from trl import SFTConfig, SFTTrainer

    t, rd = cfg.training, run_dir_for(cfg, run_dir)
    cuda = torch.cuda.is_available()
    every = t.save.get("every_steps")
    try:
        import liger_kernel  # noqa: F401

        liger = cuda and t.fused_cross_entropy
    except ImportError:
        liger = False

    # transformers 5.2x renamed gradient_checkpointing* to activation_checkpointing* (old names still alias)
    # The frozen TG-D0 stream is already shuffled once: read it in order (no per-run reshuffle)
    sequential = {"train_sampling_strategy": "sequential"} if hasattr(SFTConfig, "train_sampling_strategy") else {}
    ck = "activation_checkpointing" if hasattr(SFTConfig, "activation_checkpointing") else "gradient_checkpointing"
    checkpointing = {ck: t.gradient_checkpointing,
                     f"{ck}_kwargs": {"use_reentrant": False} if t.gradient_checkpointing else None}
    args = SFTConfig(
        output_dir=str(rd / "checkpoints"),
        max_length=cfg.data.max_length,
        packing=cfg.data.packing,
        padding_free=cfg.data.packing,
        completion_only_loss=True,
        per_device_train_batch_size=t.micro_batch_rows,
        per_device_eval_batch_size=t.micro_batch_rows,
        gradient_accumulation_steps=t.grad_accum,
        max_steps=int(t.max_steps),
        bf16=t.bf16 and cuda,
        max_grad_norm=t.max_grad_norm,
        use_liger_kernel=liger,
        logging_steps=int(t.logging.get("every_steps", 10)),
        report_to=t.logging.get("report_to", "none"),
        eval_strategy="steps" if eval_sets and t.dev_loss_every_steps else "no",
        eval_steps=t.dev_loss_every_steps,
        save_strategy="steps" if every else "no",
        save_steps=every or 500,
        save_total_limit=2,
        seed=cfg.seed,
        data_seed=cfg.seed,
        include_num_input_tokens_seen=True,
        use_cpu=not cuda,
        dataloader_num_workers=0,
        **checkpointing,
        **sequential,
    )
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_sets or None,
        processing_class=getattr(tokenizer, "tokenizer", tokenizer),  # text only: the tokenizer, not the VLM processor
        optimizers=(optimizer, scheduler),
        callbacks=_callbacks(cfg, rd),
    )
    for cb in trainer.callback_handler.callbacks:  # TensorBoard into metrics/: local + pushed to HF with the run
        if type(cb).__name__ == "TensorBoardCallback":
            cb.logging_dir = str(rd / "metrics" / "tensorboard")  # its writer is created at train begin
    trainer.full_processor = tokenizer  # saved with the final weights (chat template, image processor)
    small_shard_saves(trainer.accelerator.unwrap_model(trainer.model))
    skip = continue_stream(cfg, trainer) if continue_parent_stream else 0
    if skip:
        log.info("branch: continuing %s's stream after %d packed rows", t.resume["from_run"], skip)
    for cb in trainer.callback_handler.callbacks:  # parameter counts + init for the W&B run config
        if hasattr(cb, "extra"):
            cb.extra.update(
                trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                total_params=sum(p.numel() for p in model.parameters()),
            )
            try:
                cb.extra["init"] = dict(zip(("repo_or_path", "subfolder", "revision"), _resolve_init(cfg),
                                            strict=True))
            except Exception:  # noqa: BLE001, S110 (metadata only)
                pass
    trainer.accelerator.autocast_handler = AutocastKwargs(cache_enabled=False)  # read when train() prepares

    src = branch_source(cfg, rd)
    if src is not None and t.resume.get("optimizer_state") and latest_checkpoint(cfg, rd) is None:
        state = torch.load(src / "optimizer.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state)  # tensors follow the params' device
        log.info("branch: optimizer state from %s", src / "optimizer.pt")
    log.info("schedule: %s from global step %d for %d steps", t.lr_schedule, start_step(cfg), t.max_steps)
    return trainer


def continue_stream(cfg: RunConfig, trainer: Any) -> int:
    """Branch runs (TG-10-CD06/CD10) train on the parent's stream *after* its kept step: drop the packed
    rows the parent consumed (start_global_step × micro_batch_rows × grad_accum, same packing)."""
    t = cfg.training
    if not t.resume.get("from_run"):
        return 0
    skip = int(t.resume.get("start_global_step") or 0) * t.micro_batch_rows * t.grad_accum
    ds = trainer.train_dataset
    if skip >= len(ds):
        raise ValueError(f"branch skip {skip} >= {len(ds)} packed rows")
    trainer.train_dataset = ds.select(range(skip, len(ds)))
    return skip


def latest_checkpoint(cfg: RunConfig, run_dir: str | Path | None = None) -> str | None:
    """The newest ``checkpoint-N`` in ``runs/<id>/checkpoints`` (crash restart), or None."""
    from transformers.trainer_utils import get_last_checkpoint

    ckpts = run_dir_for(cfg, run_dir) / "checkpoints"
    return get_last_checkpoint(str(ckpts)) if ckpts.is_dir() else None


def train(trainer: Any, cfg: RunConfig, run_dir: str | Path | None = None) -> Any:
    last = latest_checkpoint(cfg, run_dir)
    if last:
        log.info("resuming from %s", last)
    return trainer.train(resume_from_checkpoint=last)


def save_final(cfg: RunConfig, trainer: Any, run_dir: str | Path | None = None) -> Path:
    """bf16 weights + processor into ``runs/<id>/model/`` (the model is cast in place: training is over)."""
    out = run_dir_for(cfg, run_dir) / "model"
    model = trainer.accelerator.unwrap_model(trainer.model)
    release_memory()
    model.to(torch.bfloat16).save_pretrained(out, max_shard_size=SAVE_SHARD_SIZE)
    proc = getattr(trainer, "full_processor", None) or trainer.processing_class
    if proc is not None:
        proc.save_pretrained(out)
    log.info("saved bf16 model to %s", out)
    return out
