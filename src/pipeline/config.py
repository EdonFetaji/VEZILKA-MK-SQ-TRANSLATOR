"""Run configs: load a YAML file, resolve ``extends``, validate, return frozen dataclasses.

One YAML per run in ``src/configs/``, named after its ``run_id`` (``TG-33-s1.yaml``). Shared settings
live in ``_base.yaml``; a run file says ``extends: _base.yaml`` and overrides only what differs.
Merging is deep for mappings (child wins) and replaces lists whole.

Validation is strict: unknown keys, unknown loaders/tasks/splits/selection rules, a task applied to
the wrong record kind, or a batch size that doesn't add up all raise ``ConfigError``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.pipeline import registry
from src.train_tasks.base import DIRECTIONS

LR_SCHEDULES = ("cosine", "wsd", "constant")
PUSH_MODES = ("never", "end", "every_save")


class ConfigError(ValueError):
    pass


# ── dataclasses ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelConfig:
    init_from: str
    revision: str | None = None
    dtype: str = "bfloat16"
    attn_implementation: str | None = None
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
    epochs: float = 1.0
    learning_rate: float = 1e-5
    optimizer: dict[str, Any] = field(default_factory=lambda: {"name": "adafactor"})
    lr_schedule: dict[str, Any] = field(default_factory=lambda: {"type": "cosine"})
    tokens_per_step: int = 65536
    micro_batch_tokens: int = 8192
    grad_accum: int = 4
    gradient_checkpointing: bool = True
    bf16: bool = True
    eval: dict[str, Any] = field(default_factory=dict)
    save: dict[str, Any] = field(default_factory=lambda: {"strategy": "end"})
    logging: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DistributedConfig:
    strategy: str = "fsdp2"
    num_gpus: int = 2
    fsdp: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputConfig:
    local_dir: str
    hub_repo: str | None = None
    private: bool = True
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
    if t.lr_schedule.get("type") not in LR_SCHEDULES:
        raise ConfigError(f"training.lr_schedule.type must be one of {LR_SCHEDULES}")
    if cfg.output.push not in PUSH_MODES:
        raise ConfigError(f"output.push must be one of {PUSH_MODES}")
    if cfg.output.push != "never" and not cfg.output.hub_repo:
        raise ConfigError("output.hub_repo is required unless output.push is 'never'")


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    if path.name.startswith("_"):
        raise ConfigError(f"{path.name} is a shared base, not a run; load a run config that extends it")
    cfg = parse(read_with_extends(path), source=str(path))
    validate(cfg, stem=path.stem)
    return cfg
