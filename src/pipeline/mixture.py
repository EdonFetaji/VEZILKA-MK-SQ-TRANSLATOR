"""Build the training stream and evaluation sets from a run config.

For each ``data.shared`` and ``data.mixture`` entry:

1. ``loader = get_loader(e.loader)(**e.options)``; ``ds = loader.select(loader.load(e.split), **e.select)``
2. ``task = get_task(e.task)(processor, **e.task_options)``; every record becomes prompt–completion
   examples via ``task.examples(record, e.directions, index)``
3. Examples longer than ``data.max_length`` tokens (prompt + completion) are **dropped, never
   truncated**, and counted per source in the manifest (CSV ``over_length_policy``)

A frozen TG-D0 tag (``directions: frozen``, e.g. G for TG-10) is already in its one shuffled order and
is used as is: no reshuffle. Mixing several non-frozen entries (``repeats``, ``tokens_m``, a seeded
interleave) is for later stages and raises ``NotImplementedError`` for now.

Everything stays a memory-mapped ``datasets.Dataset`` (Arrow files in the HF datasets cache next to
the source), so the 1.75M examples are never all in RAM.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from typing import Any

from src.pipeline.config import RunConfig, SourceEntry
from src.pipeline.registry import get_loader, get_task

MAP_BATCH = 2_048


def _tokenizer(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def entry_examples(e: SourceEntry, processor: Any, max_length: int) -> tuple[Any, dict[str, Any]]:
    """One config entry → ``Dataset`` with ``prompt`` / ``completion`` (+ ``source_id``), and its stats."""
    if e.repeats is not None or e.tokens_m is not None:
        raise NotImplementedError(f"{e.name or e.loader}: repeats / tokens_m mixing is not implemented yet")
    loader = get_loader(e.loader)(**e.options)
    ds = loader.select(loader.load(e.split), **e.select)
    task = get_task(e.task)(processor, **e.task_options)
    tok = _tokenizer(processor)
    directions = e.directions

    def render(batch: dict[str, list], idx: list[int]) -> dict[str, list]:
        keys = list(batch)
        out: dict[str, list] = {"prompt": [], "completion": [], "source_id": [], "n_tokens": []}
        for j, i in enumerate(idx):
            record = {k: batch[k][j] for k in keys}
            for ex in task.examples(record, directions, i):
                out["prompt"].append(ex["prompt"])
                out["completion"].append(ex["completion"])
                out["source_id"].append(record.get("source_id", e.loader))
        ids = tok([p + c for p, c in zip(out["prompt"], out["completion"], strict=True)])["input_ids"]
        out["n_tokens"] = [len(x) for x in ids]
        return out

    rendered = ds.map(render, batched=True, batch_size=MAP_BATCH, with_indices=True,
                      remove_columns=ds.column_names, desc=f"render {e.name or e.loader}")
    lengths = rendered["n_tokens"]
    too_long = Counter(s for s, n in zip(rendered["source_id"], lengths, strict=True) if n > max_length)
    if too_long:
        rendered = rendered.filter(lambda b: [n <= max_length for n in b["n_tokens"]], batched=True,
                                   batch_size=65_536, desc="drop over-length")
    stats = {
        "name": e.name, "loader": e.loader, "split": e.split, "select": e.select, "task": e.task,
        "directions": e.directions, "records": len(ds), "examples": len(rendered),
        "tokens": int(sum(lengths)) - sum(n for n in lengths if n > max_length),
        "dropped_over_length": dict(too_long), "max_tokens": max(lengths) if lengths else 0,
        "per_source": dict(Counter(rendered["source_id"])),
        "loader_describe": loader.describe(),
    }
    frozen = getattr(loader, "frozen_path", None)
    if frozen is not None and e.select.get("tag"):
        side = frozen(e.select["tag"], int(e.select.get("seed", 1))) / "train.json"
        if side.exists():
            stats["frozen"] = json.loads(side.read_text())
    return rendered.remove_columns(["source_id", "n_tokens"]), stats


def build_train_dataset(cfg: RunConfig, tokenizer: Any):
    """Return a ``datasets.Dataset`` of prompt–completion examples plus a manifest dict."""
    entries = [*cfg.data.shared, *cfg.data.mixture]
    if not entries:
        raise ValueError("no training entries")
    parts, stats = [], []
    for e in entries:
        ds, st = entry_examples(e, tokenizer, cfg.data.max_length)
        parts.append(ds)
        stats.append(st)
    if len(parts) == 1 and entries[0].directions == "frozen":
        train = parts[0]  # the frozen TG-D0 order: no reshuffle
        order = "frozen TG-D0 order (shuffled once at freeze time)"
    else:
        raise NotImplementedError(
            "mixing several entries (seeded interleave / shuffle) is not implemented yet; "
            "TG-10 uses one frozen entry: {loader: opus, select: {tag: G}, directions: frozen}"
        )
    manifest = {
        "run_id": cfg.run_id, "seed": cfg.seed, "order": order, "examples": len(train),
        "tokens": sum(s["tokens"] for s in stats), "max_length": cfg.data.max_length, "entries": stats,
    }
    return train, manifest


def build_eval_sets(cfg: RunConfig, tokenizer: Any) -> dict[str, Any]:
    """``{"<NAME>.<direction>": Dataset}`` for every ``data.eval`` entry (unshuffled, one pass)."""
    out: dict[str, Any] = {}
    for e in cfg.data.eval:
        dirs = ["mk-sq", "sq-mk"] if e.directions in ("both", "alternating") else [e.directions]
        for d in dirs:
            single = dataclasses.replace(e, directions=d)
            ds, _ = entry_examples(single, tokenizer, cfg.data.max_length)
            out[f"{e.name}.{d}"] = ds
    return out
