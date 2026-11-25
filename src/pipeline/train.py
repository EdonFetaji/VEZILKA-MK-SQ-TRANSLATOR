"""Train one run from its YAML config, on one device (the orchestrator, ``training_pipeline.py``, calls this).

    python -m src.pipeline.train src/configs/TG-33-s1.yaml --dry-run      # validate + print resolved config
    python -m src.pipeline.train src/configs/TG-10.yaml --run-dir runs/TG-10
    python -m src.pipeline.train src/configs/TG-10.yaml --max-steps 3      # pilot
    python -m src.pipeline.train src/configs/TG-10.yaml --smoke            # tiny in-memory data, no mixture

Writes ``<run-dir>/status.json`` = ``{"status": "done"|"failed", "step": ..., "tokens": ...}`` on exit.
A rerun resumes from the newest checkpoint in ``<run-dir>/checkpoints``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import sys
import traceback
from pathlib import Path

from src.pipeline.config import ConfigError, RunConfig, load_config

SMOKE_WORDS = [
    "the", "a", "house", "water", "city", "book", "river", "school",
    "friend", "road", "light", "sea", "tree", "day", "night", "word",
]


def smoke_dataset(n: int = 64, seed: int = 0):
    """``n`` conversational prompt–completion rows of random words (no tokenizer or network needed)."""
    from datasets import Dataset

    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        src = " ".join(rng.choices(SMOKE_WORDS, k=rng.randint(3, 12)))
        tgt = " ".join(reversed(src.split()))
        rows.append({"prompt": [{"role": "user", "content": src}],
                     "completion": [{"role": "assistant", "content": tgt}]})
    return Dataset.from_list(rows)


def write_status(run_dir: Path, status: str, trainer=None, start_global_step: int = 0, **extra) -> None:
    state = getattr(trainer, "state", None)
    local = getattr(state, "global_step", 0) or 0
    row = {"status": status, "step": start_global_step + local, "local_step": local,
           "tokens": getattr(state, "num_input_tokens_seen", 0) or 0, **extra}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(json.dumps(row, indent=2) + "\n")


def close_tracking(trainer) -> None:
    """Finish the W&B run as failed (``on_train_end`` never fires when training raises)."""
    for cb in getattr(getattr(trainer, "callback_handler", None), "callbacks", []):
        if hasattr(cb, "close"):
            cb.close(failed=True)


def with_max_steps(cfg: RunConfig, max_steps: int) -> RunConfig:
    return dataclasses.replace(cfg, training=dataclasses.replace(cfg.training, max_steps=max_steps))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to a run config, e.g. src/configs/TG-10.yaml")
    ap.add_argument("--dry-run", action="store_true", help="validate the config and print it resolved")
    ap.add_argument("--run-dir", help="run output directory (default: output.local_dir, runs/<id>)")
    ap.add_argument("--max-steps", type=int, help="override training.max_steps (pilots)")
    ap.add_argument("--smoke", action="store_true", help="64-row in-memory dataset instead of the mixture")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"config error in {args.config}: {e}", file=sys.stderr)
        return 2
    if args.max_steps is not None:
        cfg = with_max_steps(cfg, args.max_steps)

    if args.dry_run:
        print(cfg.to_yaml(), end="")
        print(f"# OK: {cfg.run_id} is valid", file=sys.stderr)
        return 0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(cfg, Path(args.run_dir or cfg.output.local_dir), smoke=args.smoke)


def run(cfg: RunConfig, run_dir: Path, smoke: bool = False) -> int:
    from src.pipeline import trainer as tr

    hf_trainer, offset = None, tr.start_step(cfg)
    try:
        tr.preflight(cfg, run_dir)
        model, processor = tr.load_model_and_tokenizer(cfg, run_dir)
        if smoke:
            train_ds = smoke_dataset(64, cfg.seed)
            eval_sets = {"SMOKE_DEV": smoke_dataset(8, cfg.seed + 1)}
        else:
            from src.pipeline import mixture

            train_ds, manifest = mixture.build_train_dataset(cfg, processor)
            eval_sets = mixture.build_eval_sets(cfg, processor)
            (run_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        hf_trainer = tr.build_trainer(cfg, model, processor, train_ds, eval_sets, run_dir)
        tr.train(hf_trainer, cfg, run_dir)
        tr.save_final(cfg, hf_trainer, run_dir)
    except BaseException as e:  # noqa: BLE001 (Ctrl-C and OOM also leave a "failed" status)
        write_status(run_dir, "failed", hf_trainer, offset, error=f"{type(e).__name__}: {e}")
        close_tracking(hf_trainer)
        traceback.print_exc()
        return 1
    write_status(run_dir, "done", hf_trainer, offset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
