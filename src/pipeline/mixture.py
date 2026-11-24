"""Build the training stream and evaluation sets from a run config.

For each ``data.shared`` and ``data.mixture`` entry:

1. ``loader = get_loader(e.loader)(**e.options)``; ``ds = loader.select(loader.load(e.split), **e.select)``
2. ``task = get_task(e.task)(tokenizer, **e.task_options)``; map every record through
   ``task.examples(record, e.directions, index)``
3. Size: ``repeats`` copies per record and direction, or sample/repeat until ``tokens_m`` million
   tokens (counted with the run's tokenizer and chat template, the same count the packer sees)
4. Record ``loader.describe()`` and the example/token counts for the run manifest

Then concatenate every entry and shuffle **once** with ``data.shuffle_seed`` (default: the run
seed). The shared arm stream (L_ARMS + G_REPLAY) must come out in the same order for a given
seed in every arm: ``data.shared`` is built and shuffled as its own block, then the treatment is
mixed in, so the arms differ only in the added data.
"""

from __future__ import annotations

from typing import Any

from src.pipeline.config import RunConfig


def build_train_dataset(cfg: RunConfig, tokenizer: Any):
    """Return a ``datasets.Dataset`` of prompt–completion examples plus a manifest dict."""
    raise NotImplementedError("mixture.build_train_dataset")


def build_eval_sets(cfg: RunConfig, tokenizer: Any) -> dict[str, Any]:
    """Return ``{name: Dataset}`` for every ``data.eval`` entry (unshuffled, one pass)."""
    raise NotImplementedError("mixture.build_eval_sets")
