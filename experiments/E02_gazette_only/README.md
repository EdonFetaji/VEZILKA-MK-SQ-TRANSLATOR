# E02_gazette_only (arm M1) — planned

Gazette-only control. It is identical to [E01_gazette_verbis_mix](../E01_gazette_verbis_mix/REPORT.md)
except for `data.verbis.use_for_training: false`. Verbis is still loaded, but only to measure
terminology, so E01 vs E02 isolates the contribution of the Verbis lexicon.

- Config: [`configs/E02_gazette_only.yaml`](../../configs/E02_gazette_only.yaml)
- Base model: `facebook/nllb-200-distilled-600M@f8d333a098d19b4fd9a8b18f94170487ad3f821d`
- Data: gazette `07af5bb`, flores_plus `5fec6c1`, ntrex `468c6b6` (pinned, see `data/manifests/`)
- Results repo (private): `EdonFetaji/mk-sq-nllb600m-e02-gazette-only`

Run (on the GPU machine, from a clean checkout):

    python3 scripts/train/smoke_test_nllb_run.py E02_gazette_only --quick
    mkdir -p runs/E02_gazette_only
    nohup python3 scripts/train/train_nllb_gazette_verbis_ct2.py E02_gazette_only > runs/E02_gazette_only/nohup.out 2>&1 &

When the run finishes it copies the small artifacts (results, curves, logs, `SOURCES.json`) into
this folder. Commit them together with the updated row in `experiments/registry.csv`.
