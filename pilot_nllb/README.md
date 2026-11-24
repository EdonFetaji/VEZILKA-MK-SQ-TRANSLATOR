# NLLB-200-distilled-600M pilot (E00–E02): finished

The first round of experiments: the base model (E00), Gazette + VERBIS terms and definitions
(E01), and Gazette only (E02), one seed each. It motivates the TranslateGemma design (research plan §04).

| BLEU, MK→SQ | E00 base | E01 Gazette + VERBIS | E02 Gazette only |
| --- | --- | --- | --- |
| Gazette test | 27.74 | 54.02 | 54.28 |
| FLORES+ devtest | 22.08 | 19.45 | 15.99 |
| NTREX-128 | 23.14 | 18.13 | 15.25 |

VERBIS protected general quality (2.1–3.5 BLEU more retained), but the pilot can't say why.

| Path | Contents |
| --- | --- |
| `experiments/` | Per-experiment results, analysis, figures, tables and REPORT.md; `registry.csv` |
| `paper/` | Pilot paper tables/figures and `results_registry.csv` |
| `reports/experiments_report.html` | The combined E00–E02 report |
| `data/manifests/` | Pinned dataset revisions used by the pilot |
| `data/release_meta/` | Gazette release metadata (column pruning, redaction log, HF dataset card) |
| `code/` | **Archived, not maintained.** The pilot's scripts, the old `src/mksq` package and configs, kept for reference. Their paths assume the old layout and no longer run as-is |

Model repos: `EdonFetaji/mk-sq-nllb600m-gazette-verbis` (E01), `EdonFetaji/mk-sq-nllb600m-e02-gazette-only` (E02).
