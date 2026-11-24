"""Train one run from its YAML config.

    python -m src.pipeline.train src/configs/TG-33-s1.yaml --dry-run   # validate + print resolved config
    accelerate launch -m src.pipeline.train src/configs/TG-33-s1.yaml  # train (2 GPUs, FSDP2)
"""

from __future__ import annotations

import argparse
import sys

from src.pipeline.config import ConfigError, load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to a run config, e.g. src/configs/TG-10.yaml")
    ap.add_argument("--dry-run", action="store_true", help="validate the config and print it resolved")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"config error in {args.config}: {e}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(cfg.to_yaml(), end="")
        print(f"# OK: {cfg.run_id} is valid", file=sys.stderr)
        return 0

    from src.pipeline import mixture, trainer

    model, tokenizer = trainer.load_model_and_tokenizer(cfg)
    train_ds, manifest = mixture.build_train_dataset(cfg, tokenizer)
    eval_sets = mixture.build_eval_sets(cfg, tokenizer)
    hf_trainer = trainer.build_trainer(cfg, model, tokenizer, train_ds, eval_sets)
    hf_trainer.train()
    trainer.push_to_hub(cfg, hf_trainer, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
