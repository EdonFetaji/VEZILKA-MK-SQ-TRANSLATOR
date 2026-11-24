# E02_gazette_only (arm M1) — done

Gazette-only control. It is identical to [E01_gazette_verbis_mix](../E01_gazette_verbis_mix/REPORT.md)
except for `data.verbis.use_for_training: false`. Verbis is still loaded, but only to measure
terminology, so E01 vs E02 isolates the contribution of the Verbis lexicon.

**Full report: [REPORT.md](REPORT.md)**. Every number in it is in `results.json`.

## Key results

- In-domain (Gazette test), E02 is marginally ahead of E01 (+0.16 to +0.26 BLEU), so training on Verbis
  gave no in-domain benefit.
- In the general domain (FLORES+, NTREX-128), E02 is 2.1–3.5 BLEU below E01 in all 4 comparisons. Every
  difference is significant (p ≤ 0.001, paired bootstrap). Verbis data reduced the loss of general-domain quality.
- Overall Term Success Rate is the same as E01's (within 0.24 points). E01 is ahead only on rare MK→SQ terms
  (116 vs 106 of 220).

## Files

| Path | What | Origin |
|---|---|---|
| `config.yaml` | copy of `configs/E02_gazette_only.yaml` | git |
| `run_info.json`, `train_log.csv`, `dev_curve.csv`, `progress.log` | run record (`progress.log`: 25 example translations removed) | results repo, see `SOURCES.json` |
| `results/*.csv` | BLEU / chrF++ per test set, direction and system | results repo, see `SOURCES.json` |
| `analysis/*.csv` | overlap classes, Term Success Rate, number fidelity | results repo, see `SOURCES.json` |
| `REPORT.md`, `results.json`, `tables/*.tex`, `figures/*.png`, `MANIFEST.json` | report, with E01 comparison and significance tests | `build_report.py` |

Results repo (private): `EdonFetaji/mk-sq-nllb600m-e02-gazette-only` @ `a5d93046ed56f5442cb3b434f95d5f0f39774108`.

## Run notes

- Training ran at `ccd58f9` (clean tree). Steps 1–1000 and data preparation ran at `bd3c440`; the only
  code difference between these two commits is the `dev_curve.csv` writer.
- Step 12b analysis failed in the training run (a variable shadowed `table()`). It was re-run alone at
  `fcb4355` (= `ccd58f9` + the fix) on 2026-10-07. Step 13 (`DONE`, export) never ran; the files were
  imported with `scripts/registry/import_hf_artifacts.py E02_gazette_only`.
- `dev_curve.csv` lacks the step-3650 dev row. It is in `progress.log`, and the report uses it from there.

Rebuild the report (no GPU, about 2 minutes):

    uv run python experiments/E02_gazette_only/build_report.py
