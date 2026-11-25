"""Weights & Biases tracking, shared by every training stage.

One ``TrainerCallback`` (:class:`WandbTracker`, added in ``trainer._callbacks``) streams the live run;
two helpers attach to the same W&B run afterwards, from the orchestrator:

* :func:`log_checkpoint` after ``hub.push``: a ``model`` artifact that *references* the run's Hugging
  Face folder at its commit (HF stays the only checkpoint store; nothing is uploaded to W&B).
* :func:`log_eval` when ``runs/<id>/eval/<point>/scores.json`` exists (BLEU, chrF++, COMET, ...).

Organisation: project ``logging.wandb.project``, ``group`` = training stage (``general_mt``,
``domain_mt``, ``lexical``, ``rl``, ``final``), ``job_type`` = run family (``TG-33``), ``name`` = run id.
Every metric is plotted against ``global_step``, so branch runs (TG-10-CD06/CD10/EXT) line up with
TG-10. The W&B run id is kept in ``runs/<id>/wandb.json``, so a crash restart continues the same run.

Mode: ``WANDB_MODE`` / ``WANDB_DISABLED`` env (``training_pipeline.py run --wandb``) beats
``logging.wandb.mode`` in the YAML (``online`` | ``offline`` | ``disabled``). Offline runs land in
``runs/<id>/wandb/``; upload them later with ``wandb sync``.

Only global rank 0 logs. Tracking never breaks training: a missing package, failed login or any wandb
error logs one warning and turns tracking off for the rest of the run. Never log data rows here.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from src.pipeline.config import RunConfig

log = logging.getLogger(__name__)

MODES = ("online", "offline", "disabled")
WANDB_KEYS = {"enabled", "mode", "project", "entity", "system_interval_s"}
DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "mode": "online",
    "project": "translategemma-mk-sq",
    "entity": None,
    "system_interval_s": 10,
}
#: Suffixes of the Trainer's ``eval_<SET>_<metric>`` keys (longest first when matching).
EVAL_SUFFIXES = sorted(
    (
        "loss",
        "runtime",
        "samples_per_second",
        "steps_per_second",
        "mean_token_accuracy",
        "entropy",
        "num_tokens",
    ),
    key=len,
    reverse=True,
)


# ── settings ──────────────────────────────────────────────────────────────────


def settings(cfg: RunConfig) -> dict[str, Any]:
    """``logging.wandb`` merged over the defaults, with the env override applied to ``mode``."""
    s = {**DEFAULTS, **(cfg.training.logging.get("wandb") or {})}
    env = os.environ.get("WANDB_MODE")
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        env = "disabled"
    if env in MODES:
        s["mode"] = env
    if not s["enabled"]:
        s["mode"] = "disabled"
    return s


def stage_group(cfg: RunConfig) -> str:
    """CSV ``stage`` → W&B group: general (incl. size-check branches and the extension) → general_mt."""
    stage = str(cfg.plan.get("stage", "")).lower()
    for key, group in (
        ("general", "general_mt"),
        ("domain", "domain_mt"),
        ("lexical", "lexical"),
        ("rl", "rl"),
        ("final", "final"),
    ):
        if stage.startswith(key):
            return group
    return stage or "other"


def is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def info_path(run_dir: Path) -> Path:
    return Path(run_dir) / "wandb.json"


def read_info(run_dir: Path) -> dict[str, Any]:
    p = info_path(run_dir)
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def _import_wandb():
    try:
        import wandb

        return wandb
    except ImportError:
        log.warning("wandb is not installed: tracking disabled")
        return None


def _versions() -> dict[str, str]:
    out = {}
    for name in ("torch", "transformers", "trl", "accelerate", "wandb"):
        with contextlib.suppress(Exception):
            out[name] = __import__(name).__version__
    return out


def run_config(cfg: RunConfig, run_dir: Path, **extra: Any) -> dict[str, Any]:
    """Everything W&B shows as the run's config: resolved YAML, CSV plan row, provenance. No data rows."""
    from src.pipeline import hub

    out: dict[str, Any] = {**cfg.to_dict(), "stage_group": stage_group(cfg), "versions": _versions()}
    with contextlib.suppress(Exception):
        out["git_sha"] = hub._git_sha()
    manifest = Path(run_dir) / "manifest.json"
    if manifest.exists():
        try:
            out["data_manifest"] = json.loads(manifest.read_text())
        except ValueError:
            pass
    return {**out, **extra}


def tags(cfg: RunConfig) -> list[str]:
    t = [stage_group(cfg), cfg.group or cfg.run_id, f"seed{cfg.seed}"]
    if cfg.plan.get("priority"):
        t.append(f"priority{str(cfg.plan['priority']).split()[0]}")
    if cfg.output.repo:
        t.append(f"repo:{cfg.output.repo}")
    if cfg.training.resume.get("from_run"):
        t.append("branch")
    return t


# ── the live callback ─────────────────────────────────────────────────────────


def _trainer_callback_base():
    try:
        from transformers import TrainerCallback

        return TrainerCallback
    except ImportError:  # tests without transformers
        return object


class WandbTracker(_trainer_callback_base()):
    """Streams Trainer logs, checkpoint events and memory to W&B. Safe: errors disable it, never raise."""

    def __init__(self, cfg: RunConfig, run_dir: str | Path, start_global_step: int = 0) -> None:
        self.cfg, self.run_dir, self.offset = cfg, Path(run_dir), start_global_step
        self.s = settings(cfg)
        self.wandb = None
        self.run = None
        self.extra: dict[str, Any] = {}  # added to the run config at init (e.g. parameter counts)
        self._last: tuple[float, int, int] | None = None  # (time, tokens, local step) of the previous log

    # -- safety -------------------------------------------------------------
    def _safe(self, what: str, fn, *a, **kw):
        if self.run is None and what != "init":
            return None
        try:
            return fn(*a, **kw)
        except Exception as e:  # noqa: BLE001
            log.warning("wandb %s failed (%s: %s); tracking disabled for this run", what, type(e).__name__, e)
            self._shutdown(quiet=True)
            return None

    def _shutdown(self, quiet: bool = False, exit_code: int = 0) -> None:
        run, self.run = self.run, None
        if run is not None:
            with contextlib.suppress(Exception):
                run.finish(exit_code=exit_code, quiet=quiet)

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> Any:
        """Init (or resume) the W&B run. Called from ``on_train_begin``; idempotent; never raises."""
        try:
            return self._start()
        except Exception as e:  # noqa: BLE001
            log.warning("wandb start failed (%s: %s); tracking disabled for this run", type(e).__name__, e)
            self._shutdown(quiet=True)
            return None

    def _start(self) -> Any:
        if self.run is not None or self.s["mode"] == "disabled" or not is_rank_zero():
            return self.run
        self.wandb = self.wandb or _import_wandb()
        if self.wandb is None:
            return None
        info = read_info(self.run_dir)
        run_id = info.get("id") or uuid.uuid4().hex[:12]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        kwargs = {
            "project": self.s["project"],
            "entity": self.s["entity"],
            "name": self.cfg.run_id,
            "group": stage_group(self.cfg),
            "job_type": self.cfg.group or self.cfg.run_id,
            "tags": tags(self.cfg),
            "notes": self.cfg.description,
            "config": run_config(self.cfg, self.run_dir, **self.extra),
            "dir": str(self.run_dir),
            "id": run_id,
            "resume": "allow",
        }
        used = None
        for mode in dict.fromkeys((self.s["mode"], "offline")):  # online failure → one offline retry
            try:
                settings_ = self.wandb.Settings(x_stats_sampling_interval=float(self.s["system_interval_s"]))
            except Exception:  # noqa: BLE001 (older/newer wandb without this setting)
                settings_ = None
            try:
                self.run = self.wandb.init(mode=mode, settings=settings_, **kwargs)
                used = mode
                break
            except Exception as e:  # noqa: BLE001
                log.warning("wandb init (%s) failed: %s: %s", mode, type(e).__name__, e)
        if self.run is None:
            return None
        self._safe("define_metric", self._define_metrics)
        url = getattr(self.run, "url", None)
        entity = getattr(self.run, "entity", None) or self.s["entity"]
        info = {"id": run_id, "project": self.s["project"], "entity": entity, "url": url, "mode": used}
        info_path(self.run_dir).write_text(json.dumps(info, indent=2) + "\n")
        log.info("wandb run %s (%s) %s", self.cfg.run_id, used, url or "")
        return self.run

    def _define_metrics(self) -> None:
        self.run.define_metric("global_step")
        self.run.define_metric("*", step_metric="global_step")

    def close(self, failed: bool = False) -> None:
        self._shutdown(exit_code=1 if failed else 0)

    # -- metrics ------------------------------------------------------------
    def row(self, logs: dict[str, Any], local_step: int, tokens: int) -> dict[str, Any]:
        """Trainer ``logs`` → one W&B row (pure; unit-tested)."""
        out: dict[str, Any] = {
            "global_step": self.offset + local_step,
            "local_step": local_step,
            "tokens_seen": tokens,
        }
        for k, v in logs.items():
            if not isinstance(v, int | float) or isinstance(v, bool):
                continue
            if k.startswith("eval_"):
                rest = k[5:]
                for suf in EVAL_SUFFIXES:
                    if rest.endswith("_" + suf):
                        out[f"eval/{rest[: -len(suf) - 1]}/{suf}"] = v
                        break
                else:
                    out[f"eval/{rest}"] = v
            else:
                out[f"train/{k.removeprefix('train_')}"] = v
        now = time.monotonic()
        if self._last is not None and "train/loss" in out:
            t0, tok0, s0 = self._last
            dt = now - t0
            if dt > 0:
                out["perf/tokens_per_s"] = (tokens - tok0) / dt
                out["perf/steps_per_s"] = (local_step - s0) / dt
        if "train/loss" in out:
            self._last = (now, tokens, local_step)
        out.update(memory())
        return out

    # -- TrainerCallback hooks ---------------------------------------------
    def on_train_begin(self, args, state, control, **kw):
        if getattr(state, "is_world_process_zero", True):
            self.start()
            self._last = (
                time.monotonic(),
                getattr(state, "num_input_tokens_seen", 0) or 0,
                state.global_step,
            )
        return control

    def on_log(self, args, state, control, logs=None, **kw):
        if self.run is not None and logs:
            tokens = getattr(state, "num_input_tokens_seen", 0) or 0
            self._safe("log", lambda: self.run.log(self.row(logs, state.global_step, tokens)))
        return control

    def on_save(self, args, state, control, **kw):
        if self.run is not None:
            self._safe("summary", self._saved, args, state)
        return control

    def _saved(self, args, state) -> None:
        step = self.offset + state.global_step
        keep = step in {int(s) for s in self.cfg.training.save.get("keep_steps") or []}
        path = str(Path(args.output_dir) / f"checkpoint-{state.global_step}")
        self.run.summary.update({"last_local_checkpoint": {"global_step": step, "path": path, "kept": keep}})

    def on_train_end(self, args, state, control, **kw):
        self.close(failed=False)
        return control


def memory() -> dict[str, float]:
    """torch CUDA memory and system RAM in GB (GB10 memory is unified: NVML may report N/A)."""
    out: dict[str, float] = {}
    try:
        import torch

        if torch.cuda.is_available():
            g = 1e9
            out["mem/cuda_allocated_gb"] = torch.cuda.memory_allocated() / g
            out["mem/cuda_reserved_gb"] = torch.cuda.memory_reserved() / g
            out["mem/cuda_peak_gb"] = torch.cuda.max_memory_allocated() / g
    except Exception:  # noqa: BLE001, S110 (metrics are best effort)
        pass
    try:
        import psutil

        vm = psutil.virtual_memory()
        out["mem/system_used_gb"] = (vm.total - vm.available) / 1e9
        out["mem/system_percent"] = vm.percent
    except Exception:  # noqa: BLE001, S110
        pass
    return out


def callback(cfg: RunConfig, run_dir: str | Path, start_global_step: int = 0) -> WandbTracker:
    return WandbTracker(cfg, run_dir, start_global_step)


# ── after the run (orchestrator) ──────────────────────────────────────────────


def _attach(cfg: RunConfig, run_dir: Path):
    """Re-open the run's W&B run (same id) for post-training logging, or None."""
    s = settings(cfg)
    info = read_info(run_dir)
    if s["mode"] == "disabled" or not info.get("id"):
        return None
    wandb = _import_wandb()
    if wandb is None:
        return None
    mode = s["mode"] if info.get("mode") != "offline" else "offline"
    return wandb.init(
        project=info.get("project") or s["project"],
        entity=info.get("entity") or s["entity"],
        id=info["id"],
        resume="allow",
        mode=mode,
        dir=str(run_dir),
        name=cfg.run_id,
        group=stage_group(cfg),
        job_type=cfg.group or cfg.run_id,
    )


def hf_url(repo: str, commit: str | None, folder: str) -> str:
    return f"https://huggingface.co/{repo}/tree/{commit or 'main'}/{folder}"


def log_checkpoint(cfg: RunConfig, run_dir: str | Path, record: dict[str, Any]) -> bool:
    """Reference the pushed HF folder from the W&B run. ``record`` is the run's pipeline-state entry."""
    run_dir = Path(run_dir)
    try:
        run = _attach(cfg, run_dir)
        if run is None:
            return False
        import wandb

        repo, folder, commit = record.get("repo"), record.get("folder") or cfg.run_id, record.get("commit")
        url = hf_url(repo, commit, folder)
        meta = {
            "repo": repo,
            "folder": folder,
            "commit": commit,
            "url": url,
            "global_step": record.get("step"),
            "tokens": record.get("tokens"),
            "dev_loss": record.get("dev_loss"),
            "weights_pushed": record.get("weights_pushed"),
            "bytes": record.get("bytes"),
            "pushed_at": record.get("pushed_at"),
        }
        art = wandb.Artifact(
            f"{cfg.run_id}-hf",
            type="model",
            metadata=meta,
            description=f"Hugging Face checkpoint of {cfg.run_id} (reference only)",
        )
        art.add_reference(url, checksum=False)
        run.log_artifact(art, aliases=["latest", (commit or "main")[:12]])
        run.summary.update(
            {
                "hf/repo": repo,
                "hf/folder": folder,
                "hf/commit": commit,
                "hf/url": url,
                "hf/global_step": record.get("step"),
            }
        )
        run.finish()
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("wandb log_checkpoint failed (%s: %s); the HF push itself succeeded", type(e).__name__, e)
        return False


def flatten_scores(point: str, scores: dict[str, Any]) -> dict[str, float]:
    """``{SET.dir: {bleu, chrf++, comet}}`` (or already flat) → ``eval/<point>/<SET.dir>/<metric>``."""
    out: dict[str, float] = {}
    for key, val in scores.items():
        if isinstance(val, dict):
            for metric, x in val.items():
                if isinstance(x, int | float) and not isinstance(x, bool):
                    out[f"eval/{point}/{key}/{metric}"] = x
        elif isinstance(val, int | float) and not isinstance(val, bool):
            out[f"eval/{point}/{key}"] = val
    return out


def log_eval(
    cfg: RunConfig, run_dir: str | Path, point: str, scores: dict[str, Any], global_step: int | None = None
) -> bool:
    try:
        run = _attach(cfg, Path(run_dir))
        if run is None:
            return False
        row = flatten_scores(point, scores)
        if global_step is not None:
            row["global_step"] = global_step
        run.log(row)
        run.summary.update(row)
        run.finish()
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("wandb log_eval failed (%s: %s)", type(e).__name__, e)
        return False
