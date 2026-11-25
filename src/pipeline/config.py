"""Run configs: load a YAML file, resolve ``extends``, validate, return frozen dataclasses.

One YAML per run in ``src/configs/``, named after its ``run_id`` (``TG-33-s1.yaml``). Shared settings
live in ``_base.yaml``; a run file says ``extends: _base.yaml`` and overrides only what differs.
Merging is deep for mappings (child wins) and replaces lists whole.

Validation is strict: unknown keys, unknown loaders/tasks/splits/selection rules, a task applied to
the wrong record kind, or a batch size that doesn't add up all raise ``ConfigError``.

The ``TG-*.yaml`` run files are generated from ``tg_hyperparameters.csv`` by
``python training_pipeline.py configs``; only their ``data:`` section is edited by hand.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.pipeline import registry
from src.train_tasks.base import DIRECTIONS

LR_SCHEDULES = ("wsd", "cosine", "cooldown")
PUSH_MODES = ("never", "end", "every_save")
REPO_CHOICES = ("public", "private", "auto")

#: Defaults of the dict-valued training fields. A YAML value is merged over these (so every key is
#: always present for the trainer); keys not listed here are rejected (except in ``optimizer``).
OPTIMIZER_DEFAULT: dict[str, Any] = {
    "name": "adafactor",
    "beta1": None,
    "decay_rate": -0.8,
    "clip_threshold": 1.0,
    "eps": [1e-30, 1e-3],
    "scale_parameter": False,
    "relative_step": False,
    "warmup_init": False,
    "weight_decay": 0.0,
}
LR_SCHEDULE_DEFAULT: dict[str, Any] = {
    "type": "cosine",
    "peak": 5e-6,
    "min": 5e-7,
    "warmup_start": None,
    "warmup_steps": 0,
    "constant_end_step": None,
    "cooldown_steps": None,
}
SAVE_DEFAULT: dict[str, Any] = {"every_steps": None, "keep_steps": [], "with_optimizer": False}
RESUME_DEFAULT: dict[str, Any] = {
    "from_run": None,
    "from_step": None,
    "optimizer_state": False,
    "start_global_step": 0,
}
_STRICT_DICTS = {"lr_schedule": LR_SCHEDULE_DEFAULT, "save": SAVE_DEFAULT, "resume": RESUME_DEFAULT}


class ConfigError(ValueError):
    pass


# ── dataclasses ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelConfig:
    init_from: str
    revision: str | None = None
    #: Master-weight dtype. fp32: the forward runs under bf16 autocast (``training.bf16``).
    dtype: str = "float32"
    attn_implementation: str | None = "flash_attention_2"
    #: Parameter-name patterns (fnmatch) kept frozen.
    freeze: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SourceEntry:
    """One loader + task pair. Used for training mixtures and evaluation sets."""

    loader: str
    task: str
    split: str = "train"
    directions: str = "both"
    name: str | None = None
    select: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    task_options: dict[str, Any] = field(default_factory=dict)
    #: Presentations per record and direction (lexical tasks: 3 in the plan).
    repeats: int | None = None
    #: Token budget in millions; the mixture samples or repeats to hit it (e.g. G_REPLAY 12.0).
    tokens_m: float | None = None


@dataclass(frozen=True)
class DataConfig:
    #: The run's own training data (for an arm: its treatment).
    mixture: list[SourceEntry] = field(default_factory=list)
    #: A stream shared by a family of runs, set once in a base file (``_arms.yaml``: L_ARMS +
    #: G_REPLAY). Built as its own block, identical in order for a given seed across the family.
    shared: list[SourceEntry] = field(default_factory=list)
    eval: list[SourceEntry] = field(default_factory=list)
    max_length: int = 2048
    packing: bool = True
    #: Seed for the one shuffle of the mixture. Defaults to the run seed.
    shuffle_seed: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    #: Optimizer updates (CSV ``planned_updates``; an estimate until tokens are counted).
    max_steps: int | None = None
    micro_batch_rows: int = 4
    micro_batch_tokens: int = 8192
    grad_accum: int = 8
    tokens_per_step: int = 65536
    optimizer: dict[str, Any] = field(default_factory=lambda: dict(OPTIMIZER_DEFAULT))
    max_grad_norm: float = 1.0
    #: ``{type: wsd|cosine|cooldown, peak, min, warmup_start, warmup_steps, constant_end_step,
    #: cooldown_steps}`` (see ``LR_SCHEDULE_DEFAULT``).
    lr_schedule: dict[str, Any] = field(default_factory=lambda: dict(LR_SCHEDULE_DEFAULT))
    dev_loss_every_steps: int | None = None
    #: ``{every_steps, keep_steps, with_optimizer}``: resumable checkpoints and kept branch points.
    save: dict[str, Any] = field(default_factory=lambda: dict(SAVE_DEFAULT))
    #: ``{from_run, from_step, optimizer_state, start_global_step}``: branch from a kept checkpoint.
    resume: dict[str, Any] = field(default_factory=lambda: dict(RESUME_DEFAULT))
    gradient_checkpointing: bool = True
    fused_cross_entropy: bool = True
    bf16: bool = True
    eval: dict[str, Any] = field(default_factory=dict)
    logging: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DistributedConfig:
    strategy: str = "single"
    num_gpus: int = 1
    fsdp: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputConfig:
    local_dir: str
    #: ``public`` | ``private`` | ``auto`` (private if the data or the parent chain is restricted).
    repo: str = "auto"
    #: Top-level folder in the HF repo; ``None`` means the ``run_id``.
    folder: str | None = None
    push_weights: bool = True
    push: str = "end"


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    seed: int
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    output: OutputConfig
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    group: str | None = None
    description: str = ""
    #: The CSV row's descriptive columns, verbatim (stage, priority, run_condition, ...). Not interpreted.
    plan: dict[str, Any] = field(default_factory=dict)
    #: Path of the YAML this config was loaded from (set by ``load_config``).
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)


# ── loading ───────────────────────────────────────────────────────────────────


def deep_merge(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def read_with_extends(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in _seen:
        raise ConfigError(f"circular extends: {' -> '.join(p.name for p in (*_seen, path))}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name}: top level must be a mapping")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return deep_merge(read_with_extends(path.parent / parent, (*_seen, path)), raw)


def _build(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(names)}")
    try:
        return cls(**data)
    except TypeError as e:  # missing required field
        raise ConfigError(f"{where}: {e}") from None


def _entries(items: Any, where: str) -> list[SourceEntry]:
    if not isinstance(items, list):
        raise ConfigError(f"{where}: expected a list")
    return [_build(SourceEntry, item, f"{where}[{i}]") for i, item in enumerate(items)]


def parse(raw: dict[str, Any], source: str = "") -> RunConfig:
    raw = dict(raw)
    sections = {}
    if isinstance(raw.get("training"), dict):
        training = dict(raw["training"])
        if isinstance(training.get("optimizer"), dict):
            training["optimizer"] = {**OPTIMIZER_DEFAULT, **training["optimizer"]}
        for key, default in _STRICT_DICTS.items():
            value = training.get(key)
            if value is None:
                continue
            if not isinstance(value, dict):
                raise ConfigError(f"training.{key}: expected a mapping")
            unknown = set(value) - set(default)
            if unknown:
                raise ConfigError(
                    f"training.{key}: unknown keys {sorted(unknown)}; allowed: {sorted(default)}"
                )
            training[key] = {**default, **value}
        raw["training"] = training
    for key, cls in (
        ("model", ModelConfig),
        ("training", TrainingConfig),
        ("distributed", DistributedConfig),
        ("output", OutputConfig),
    ):
        if key in raw:
            sections[key] = _build(cls, raw.pop(key), key)
    if "data" in raw:
        data = dict(raw.pop("data"))
        for key in ("mixture", "shared"):
            if key in data:
                data[key] = _entries(data[key], f"data.{key}")
        if "eval" in data:
            data["eval"] = _entries(data["eval"], "data.eval")
        sections["data"] = _build(DataConfig, data, "data")
    return _build(RunConfig, {**raw, **sections, "source": source}, "run")


def validate(cfg: RunConfig, stem: str | None = None) -> None:
    if stem is not None and cfg.run_id != stem:
        raise ConfigError(f"run_id {cfg.run_id!r} must equal the file name {stem!r}")
    if not cfg.data.mixture and not cfg.data.shared:
        raise ConfigError("data.mixture and data.shared are both empty")

    for where, entries in (
        ("data.shared", cfg.data.shared),
        ("data.mixture", cfg.data.mixture),
        ("data.eval", cfg.data.eval),
    ):
        for i, e in enumerate(entries):
            at = f"{where}[{i}] ({e.name or e.loader})"
            try:
                loader = registry.get_loader(e.loader)
                task = registry.get_task(e.task)
            except KeyError as err:
                raise ConfigError(f"{at}: {err.args[0]}") from None
            if task.accepts != loader.kind:
                raise ConfigError(
                    f"{at}: task {e.task!r} needs {task.accepts.value} records, "
                    f"loader {e.loader!r} gives {loader.kind.value}"
                )
            if e.split not in loader.splits:
                raise ConfigError(f"{at}: split {e.split!r} not in {loader.splits}")
            bad = set(e.select) - set(loader.select_rules)
            if bad:
                raise ConfigError(
                    f"{at}: loader {e.loader!r} has no selection rules {sorted(bad)}; "
                    f"known: {list(loader.select_rules)}"
                )
            if e.directions not in DIRECTIONS:
                raise ConfigError(f"{at}: directions {e.directions!r} not in {DIRECTIONS}")
            if e.repeats is not None and e.tokens_m is not None:
                raise ConfigError(f"{at}: set repeats or tokens_m, not both")
        if where == "data.eval":
            names = [e.name for e in entries]
            if None in names or len(set(names)) != len(names):
                raise ConfigError("data.eval: every entry needs a unique name")

    t, d = cfg.training, cfg.distributed
    if t.micro_batch_tokens * t.grad_accum * d.num_gpus != t.tokens_per_step:
        raise ConfigError(
            f"training: micro_batch_tokens × grad_accum × num_gpus = "
            f"{t.micro_batch_tokens} × {t.grad_accum} × {d.num_gpus} ≠ tokens_per_step {t.tokens_per_step}"
        )
    wb = t.logging.get("wandb")
    if wb is not None:
        from src.pipeline.tracking import MODES, WANDB_KEYS

        if not isinstance(wb, dict) or set(wb) - WANDB_KEYS:
            raise ConfigError(f"training.logging.wandb: allowed keys {sorted(WANDB_KEYS)}")
        if wb.get("mode", "online") not in MODES:
            raise ConfigError(f"training.logging.wandb.mode must be one of {MODES}")
    sched = t.lr_schedule
    if sched.get("type") not in LR_SCHEDULES:
        raise ConfigError(f"training.lr_schedule.type must be one of {LR_SCHEDULES}")
    if sched["type"] == "wsd" and (sched["constant_end_step"] is None or sched["cooldown_steps"] is None):
        raise ConfigError("training.lr_schedule: wsd needs constant_end_step and cooldown_steps")
    if sched["type"] == "cooldown" and sched["cooldown_steps"] is None:
        raise ConfigError("training.lr_schedule: cooldown needs cooldown_steps")
    if t.resume["from_run"] is not None and t.resume["from_step"] is None:
        raise ConfigError("training.resume: from_run needs from_step")
    if cfg.output.push not in PUSH_MODES:
        raise ConfigError(f"output.push must be one of {PUSH_MODES}")
    if cfg.output.repo not in REPO_CHOICES:
        raise ConfigError(f"output.repo must be one of {REPO_CHOICES}")


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    if path.name.startswith("_"):
        raise ConfigError(f"{path.name} is a shared base, not a run; load a run config that extends it")
    cfg = parse(read_with_extends(path), source=str(path))
    validate(cfg, stem=path.stem)
    return cfg
