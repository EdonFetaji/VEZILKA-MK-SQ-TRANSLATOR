# Does learning a term's definition help a model translate it?

Macedonian ↔ Albanian legal MT with **TranslateGemma-12B**. The model is trained in three steps:
general MK–SQ (TG-10), legal language from the Official Gazette (TG-20), and then the TG-30 arms.
The arms test whether a model taught legal terms *with their definitions* (VERBIS) translates
better than one taught the bare terms, or one given the same GPU time of plain parallel text.

The full design is in [`research_plan/`](research_plan/TranslateGemma_MK-SQ_Experiment_Plan.md).
Its run and dataset tags (`TG-10`, `L_ARMS`, `V_DEF_CTX`, …) are the names used everywhere in the code.

## Layout

| Path | What it holds |
| --- | --- |
| `research_plan/` | The experiment plan (.md/.html) and its twins `tg_experiment_plan.csv` (runs) and `tg_datasets.csv` (dataset tags) |
| `src/datasets/` | One **loader** per source (OPUS, Gazette, VERBIS, Media Literacy, MUSE, phrasebook, FLORES+, NTREX). Each returns a HF `Dataset` with fixed columns |
| `src/train_tasks/` | One **task** per instruction format (translation, term pair, term + definition, shuffled definition, definition translation) |
| `src/configs/` | One **YAML per run**, named after its run id. `_base.yaml` is the shared recipe and `_arms.yaml` the shared TG-30 setup |
| `src/pipeline/` | Config loading and validation, the name registry, mixture building, the trainer, and the `train` CLI |
| `notebooks/gazette_corpus/` | How the Gazette parallel corpus was analysed, cleaned and SONAR-rescored |
| `notebooks/nllb_pilot/` | Zero-shot NLLB notebooks from the pilot |
| `pilot_nllb/` | The finished NLLB-600M pilot (E00–E02): results, report, tables, and archived code |
| `demo/`, `data/`, `runs/` | Local only, never committed |

## Running a config

```bash
python -m src.pipeline.train src/configs/TG-33-s1.yaml --dry-run    # validate, print the resolved config
accelerate launch -m src.pipeline.train src/configs/TG-33-s1.yaml   # train (2× A100, FSDP2)
```

Always run as a module from the repo root. Never put `src/` itself on `PYTHONPATH`, or
`src/datasets` will shadow Hugging Face `datasets`.

Status: the interfaces, registry, config schema and validation work. Loaders, tasks, the mixture
builder and the trainer are stubs that raise `NotImplementedError`; their docstrings say what each
one must do.

## Extending

- **New data source**: subclass `DatasetLoader` (`src/datasets/base.py`), set `kind`, `splits` and
  `select_rules`, decorate it with `@register_loader("name")`, and import it in `src/datasets/__init__.py`.
- **New task**: subclass `TrainTask` (`src/train_tasks/base.py`), set `accepts`, implement
  `format(record, direction)` to return `{"prompt": [...], "completion": "..."}`, decorate it with
  `@register_task("name")`, and import it in `src/train_tasks/__init__.py`.
- **New run**: add `src/configs/<RUN_ID>.yaml` with `extends: _base.yaml` or `_arms.yaml`, and
  override only what differs from the plan's shared recipe. Validation rejects unknown keys,
  loaders, tasks, splits and selection rules, any task/loader kind mismatch, and batch sizes that
  don't add up.
