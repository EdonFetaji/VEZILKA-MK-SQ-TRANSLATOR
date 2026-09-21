#!/usr/bin/env python
"""Fine-tune NLLB-200-distilled-600M for Macedonian <-> Albanian on the Gazette
corpus and the Verbis dictionary, export it to CTranslate2 int8, and evaluate it
on the Gazette test split, FLORES+ devtest and NTREX-128.

    cd ~/VEZILKA-MK-SQ-TRANSLATOR
    mkdir -p work
    nohup .venv/bin/python scripts/train_nllb_gazette_verbis_ct2.py > work/nohup.out 2>&1 &

One command, no arguments, no interactive input. Every step writes its output
under WORK_DIR and is skipped on the next run when that output exists, so
re-running after a crash, reboot or Ctrl-C continues where it stopped (training
resumes from the last checkpoint). To rebuild a stage, delete its output file.

Results, logs, checkpoints and models are uploaded to the PRIVATE Hugging Face
repo HF_RESULTS_REPO, whose privacy is re-checked before every upload batch:

    (root)                     final model + tokenizer (loads with from_pretrained)
    checkpoints/checkpoint-*/  last 3 + best checkpoints (for resuming anywhere)
    ct2/finetuned_int8/, ct2/base_int8/
    data/verbis/               raw + processed Verbis data (nothing else from data/)
    eval/                      logs, curves, results, translations
    progress.log, DONE

Training uses only Gazette train and Verbis; Gazette dev is used only for
checkpoint selection; the test sets are touched only in step 12.
"""

from __future__ import annotations

import csv
import dataclasses
import fnmatch
import gc
import importlib
import inspect
import json
import logging
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
HF_GAZETTE_REPO = "EdonFetaji/slvesnik-mk-sq"
HF_RESULTS_REPO = "EdonFetaji/mk-sq-nllb600m-gazette-verbis"  # PRIVATE
VERBIS_PATH = "data/verbis_mk_sq.parquet"  # local cache of VERBIS_HF_PATH
VERBIS_HF_PATH = "data/verbis/verbis_mk_sq.parquet"  # in the PRIVATE HF_RESULTS_REPO
WORK_DIR = "work"  # relative to repo root
BASE_MODEL = "facebook/nllb-200-distilled-600M"
SEED = 42

MK, SQ = "mkd_Cyrl", "als_Latn"  # NLLB language codes
MAX_TOKENS = 192  # longer examples are dropped, never truncated
MIN_FREE_VRAM_GB = 14
LABSE_MODEL = "sentence-transformers/LaBSE"
LABSE_MIN_SCORE = 0.70
DEV_EVAL_SIZE = 500  # per direction
EVAL_STEPS = 1000
LOG_STEPS = 50
SUMMARY_STEPS = 500
UPLOAD_MIN_INTERVAL_S = 30 * 60
GEN_BATCH = 32
MAX_NEW_TOKENS = 256
FINAL_BEAMS = 4
EVAL_CHUNK = 2000  # final evaluation: save + log progress every N sentences
FLORES_REPO = "openlanguagedata/flores_plus"
NTREX_URL = "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128/newstest2019-ref.{code}.txt"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_REQUIRED = ("transformers", "datasets", "accelerate", "sentencepiece", "sacrebleu", "ctranslate2",
             "sentence_transformers", "polars", "pyarrow", "huggingface_hub", "matplotlib")


def _check_imports() -> None:
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: torch is not importable ({type(e).__name__}: {e}).\n"
              "torch is installed separately with a CUDA build that supports the RTX 5080 (Blackwell);\n"
              "it is intentionally not in scripts/requirements-train.txt.", flush=True)
        sys.exit(1)
    missing = []
    for name in _REQUIRED:
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{name} ({type(e).__name__}: {e})")
    if missing:
        print("ERROR: missing Python packages:\n  " + "\n  ".join(missing)
              + "\nInstall them with:\n  uv pip install -r scripts/requirements-train.txt", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    _check_imports()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import sacrebleu  # noqa: E402
import torch  # noqa: E402
import datasets  # noqa: E402
import huggingface_hub  # noqa: E402
import transformers  # noqa: E402
from datasets import Dataset, load_dataset, load_from_disk  # noqa: E402
from huggingface_hub import (  # noqa: E402
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
    snapshot_download,
)
from sacrebleu.metrics import BLEU, CHRF  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right  # noqa: E402
from transformers.optimization import Adafactor  # noqa: E402
from transformers.trainer_callback import PrinterCallback  # noqa: E402
from transformers.trainer_pt_utils import LabelSmoother, LengthGroupedSampler  # noqa: E402
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402

try:
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
except ImportError:  # older huggingface_hub
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

datasets.disable_progress_bars()
transformers.logging.set_verbosity_warning()
transformers.utils.logging.disable_progress_bar()

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
W = ROOT / WORK_DIR
RAW = W / "data" / "raw"
PROC = W / "data" / "processed"
TOKD = W / "data" / "tokenized"
MODELS = W / "models"
CKPT = MODELS / "checkpoints"
FINAL = MODELS / "final"
CT2 = W / "ct2"
CT2_FT = CT2 / "finetuned_int8"
CT2_BASE = CT2 / "base_int8"
EVAL = W / "eval"
PROGRESS = W / "progress.log"
DONE = W / "DONE"
RUN_INFO = EVAL / "run_info.json"
TRAIN_LOG = EVAL / "train_log.csv"
DEV_CURVE = EVAL / "dev_curve.csv"
CURVES_PNG = EVAL / "training_curves.png"
DEV_IDS = EVAL / "dev_eval_ids.parquet"
PROBE_JSON = MODELS / "memory_probe.json"

TRAIN_LOG_FIELDS = ("timestamp", "step", "epoch", "loss", "learning_rate", "grad_norm", "samples_per_sec",
                    "tokens_per_sec", "gpu_mem_allocated_gb", "gpu_mem_peak_gb", "gpu_util_pct", "gpu_temp_c",
                    "gpu_power_w", "elapsed_h", "eta_h")
DEV_CURVE_FIELDS = ("step", "bleu_mk_sq", "chrf_mk_sq", "bleu_sq_mk", "chrf_sq_mk", "chrf_mean")

# Final evaluation layout.
TESTSETS = (  # (name, processed parquet, results csv)
    ("gazette_test", "gazette_test.parquet", "gazette_test_results.csv"),
    ("flores", "flores_devtest.parquet", "flores_results.csv"),
    ("ntrex", "ntrex.parquet", "ntrex_results.csv"),
)
MODEL_COLS = ("base", "finetuned_hf", "finetuned_ct2")
DIRECTIONS = (  # (name, source column, reference column, source lang, target lang)
    ("mk_sq", "mk", "sq", MK, SQ),
    ("sq_mk", "sq", "mk", SQ, MK),
)

TOKEN: str | None = None
UPLOADER: Uploader | None = None
LOG = logging.getLogger("mksq")
_CTX = {"step": "start"}


class Fatal(Exception):
    """A condition the run cannot continue from; the message says what to do."""


# ----------------------------------------------------------------------------
# Logging, step wrapper, small file helpers
# ----------------------------------------------------------------------------
class _StepFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.step = _CTX["step"]
        return True


def setup_logging() -> None:
    W.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(step)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(PROGRESS, encoding="utf-8")):
        handler.setFormatter(fmt)
        handler.addFilter(_StepFilter())
        LOG.addHandler(handler)


def run_step(num: int, name: str, fn, *args):
    _CTX["step"] = f"step {num:>2} {name}"
    LOG.info("===== start")
    t0 = time.time()
    try:
        out = fn(*args)
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001 -- includes KeyboardInterrupt
        LOG.error("FAILED after %s: %s: %s", fmt_dur(time.time() - t0), type(e).__name__, e)
        LOG.error("traceback:\n%s", traceback.format_exc())
        if UPLOADER is not None and not UPLOADER.violated and not isinstance(e, KeyboardInterrupt):
            UPLOADER.upload_files([(PROGRESS, "progress.log")], "progress.log after failure")
        sys.exit(130 if isinstance(e, KeyboardInterrupt) else 1)
    LOG.info("===== done in %s", fmt_dur(time.time() - t0))
    return out


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def fmt_dur(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "?"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(W))
    except ValueError:
        return str(path)


def _tmp(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def have(path: Path) -> bool:
    if path.exists():
        LOG.info("loaded from disk: %s", rel(path))
        return True
    return False


def write_parquet(df: pl.DataFrame, path: Path, quiet: bool = False) -> None:
    tmp = _tmp(path)
    df.write_parquet(tmp, compression="zstd")
    os.replace(tmp, path)
    if not quiet:
        LOG.info("wrote %s (%d rows)", rel(path), df.height)


def write_text(path: Path, text: str) -> None:
    tmp = _tmp(path)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


def write_csv(df: pl.DataFrame, path: Path) -> None:
    tmp = _tmp(path)
    df.write_csv(tmp)
    os.replace(tmp, path)


def save_dataset(ds: Dataset, path: Path) -> None:
    tmp = _tmp(path)
    if tmp.exists():
        shutil.rmtree(tmp)
    ds.save_to_disk(str(tmp))
    os.replace(tmp, path)
    LOG.info("wrote %s (%d rows)", rel(path), len(ds))


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_run_info() -> dict:
    return json.loads(RUN_INFO.read_text(encoding="utf-8")) if RUN_INFO.exists() else {}


def update_run_info(upload: bool = True, **fields) -> None:
    info = read_run_info()
    info.update(fields)
    info["updated_at"] = now_iso()
    write_json(RUN_INFO, info)
    if upload and UPLOADER is not None:
        UPLOADER.upload_files([(RUN_INFO, "eval/run_info.json")], "update run_info.json")


def update_dataset_sizes(upload: bool = False, **sizes) -> None:
    merged = read_run_info().get("dataset_sizes", {})
    merged.update(sizes)
    update_run_info(upload=upload, dataset_sizes=merged)


def nvidia_smi(fields: list[str]) -> dict | None:
    cmd = ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if visible.isdigit():
        cmd += ["-i", visible]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        values = [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]
        return dict(zip(fields, values))
    except Exception:  # noqa: BLE001
        return None


def gpu_live_stats() -> dict:
    raw = nvidia_smi(["utilization.gpu", "temperature.gpu", "power.draw"]) or {}

    def num(key):
        try:
            return float(raw[key])
        except (KeyError, ValueError):
            return None

    return {"gpu_util_pct": num("utilization.gpu"), "gpu_temp_c": num("temperature.gpu"),
            "gpu_power_w": num("power.draw")}


def _norm(col: str) -> pl.Expr:
    """Lowercased, whitespace-normalised text, used for leakage matching."""
    return pl.col(col).cast(pl.Utf8).str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars()


def _drop_empty(df: pl.DataFrame, what: str) -> pl.DataFrame:
    before = df.height
    df = df.with_columns(pl.col("mk").cast(pl.Utf8).str.strip_chars(), pl.col("sq").cast(pl.Utf8).str.strip_chars())
    df = df.filter(pl.col("mk").is_not_null() & pl.col("sq").is_not_null()
                   & (pl.col("mk").str.len_chars() > 0) & (pl.col("sq").str.len_chars() > 0))
    if df.height < before:
        LOG.warning("%s: dropped %d rows with an empty mk or sq side", what, before - df.height)
    return df


# ----------------------------------------------------------------------------
# Private results repo
# ----------------------------------------------------------------------------
# Every path the run may create in HF_RESULTS_REPO. Anything else is refused.
_ALLOWED_REMOTE = tuple(re.compile(p) for p in (
    r"[^/]+",  # repo root: final model + tokenizer, progress.log, DONE
    r"checkpoints/checkpoint-\d+/.+",
    r"ct2/(finetuned_int8|base_int8)/.+",
    r"eval/[^/]+",
    r"data/verbis/(verbis_terms|verbis_defs_scored|verbis_defs)\.parquet",
))
VERBIS_OUTPUTS = ("verbis_terms", "verbis_defs_scored", "verbis_defs")


class PrivacyViolation(Fatal):
    """HF_RESULTS_REPO is public: nothing is uploaded and the run stops."""


def _ckpt_step(name: str) -> int:
    return int(name.rsplit("-", 1)[1])


class Uploader:
    """Uploads only to HF_RESULTS_REPO, only paths in _ALLOWED_REMOTE, and only
    after model_info() confirms the repo is private -- checked before every
    upload batch. If the repo is ever public, nothing is uploaded, an ERROR is
    logged and PrivacyViolation stops the run.

    Any other upload failure never raises unless asked to: it logs a warning and
    the next upload point tries again. Local files in WORK_DIR are the source of truth.
    """

    def __init__(self, token: str | None):
        self.api = HfApi(token=token)
        self.url = f"https://huggingface.co/{HF_RESULTS_REPO}"
        self.last_periodic = time.time()
        self.violated = False
        self.pending: dict = {}  # checkpoint name -> Future of its background upload
        self.retry: set[str] = set()  # checkpoints whose upload failed; retried at the next save

    # -- primitives: the only methods that talk to the Hub (the smoke test replaces them)
    def _is_private(self) -> bool | None:
        return self.api.model_info(HF_RESULTS_REPO).private

    def _commit(self, pairs: list[tuple[Path, str]], message: str) -> None:
        # Small files are read into memory first, so a file that is still being
        # appended to (progress.log) is uploaded as a consistent snapshot.
        ops = [CommitOperationAdd(path_in_repo=remote,
                                  path_or_fileobj=local.read_bytes() if local.stat().st_size < 50_000_000 else str(local))
               for local, remote in pairs]
        self.api.create_commit(HF_RESULTS_REPO, operations=ops, commit_message=message, repo_type="model")

    def _folder(self, local: Path, path_in_repo: str, message: str, ignore: list[str], run_as_future: bool = False):
        return self.api.upload_folder(repo_id=HF_RESULTS_REPO, repo_type="model", folder_path=str(local),
                                      path_in_repo=path_in_repo or None, commit_message=message,
                                      ignore_patterns=ignore, run_as_future=run_as_future)

    def _background(self, fn, *args):
        # HfApi runs futures on ONE worker thread, in submission order.
        return self.api.run_as_future(fn, *args)

    def _repo_files(self) -> list[str]:
        return self.api.list_repo_files(HF_RESULTS_REPO, repo_type="model")

    def _delete_folders(self, paths: list[str], message: str) -> None:
        ops = [CommitOperationDelete(path_in_repo=p.rstrip("/") + "/", is_folder=True) for p in paths]
        self.api.create_commit(HF_RESULTS_REPO, operations=ops, commit_message=message, repo_type="model")

    def _download(self, patterns: list[str], local_dir: Path) -> None:
        snapshot_download(HF_RESULTS_REPO, repo_type="model", token=self.api.token, allow_patterns=patterns,
                          local_dir=str(local_dir))

    # -- guards
    def ensure_private_repo(self) -> None:
        try:
            private = self._is_private()
        except RepositoryNotFoundError:
            LOG.info("results repo %s does not exist -- creating it as PRIVATE", HF_RESULTS_REPO)
            self.api.create_repo(HF_RESULTS_REPO, repo_type="model", private=True)
            private = self._is_private()
        if private is not True:
            self.violated = True
            raise PrivacyViolation(f"{self.url} is NOT private. Refusing to upload anything to a public repo. "
                                   "Make it private in the repo settings and re-run.")
        LOG.info("results repo verified PRIVATE: %s", self.url)

    def _guard(self) -> None:
        """Before every upload batch. Network errors propagate as ordinary (retryable) upload failures."""
        if self.violated:
            raise PrivacyViolation(f"{self.url} was found public earlier -- uploads are disabled")
        private = self._is_private()
        if private is False:
            self.violated = True
            LOG.error("PRIVACY GUARD: %s is PUBLIC -- nothing uploaded, stopping the run", self.url)
            raise PrivacyViolation(f"{self.url} is public. Nothing was uploaded; make it private and re-run.")
        if private is not True:
            raise RuntimeError(f"could not verify that {self.url} is private (model_info().private={private!r})")

    @staticmethod
    def _check_allowed(remotes: list[str]) -> None:
        bad = [r for r in remotes if not any(p.fullmatch(r) for p in _ALLOWED_REMOTE)]
        if bad:
            raise ValueError(f"refusing to upload paths outside the allowed repo layout: {bad[:5]}")

    @staticmethod
    def _folder_remotes(local: Path, path_in_repo: str, ignore: list[str]) -> list[str]:
        return [f"{path_in_repo}/{p.relative_to(local).as_posix()}".lstrip("/") for p in sorted(local.rglob("*"))
                if p.is_file() and not any(fnmatch.fnmatch(p.relative_to(local).as_posix(), pat) for pat in ignore)]

    # -- uploads
    def upload_files(self, pairs, message: str, raise_errors: bool = False) -> bool:
        pairs = [(Path(local), remote) for local, remote in pairs if Path(local).exists()]
        if not pairs:
            return True
        try:
            self._check_allowed([remote for _, remote in pairs])
            self._guard()
            self._commit(pairs, message)
            LOG.info("uploaded: %s", ", ".join(remote for _, remote in pairs))
            return True
        except PrivacyViolation:
            raise
        except Exception as e:  # noqa: BLE001
            if raise_errors:
                raise
            LOG.warning("upload failed (%s: %s) -- will retry at the next upload point", type(e).__name__, e)
            return False

    def upload_folder(self, local: Path, path_in_repo: str, message: str, ignore: list[str] | None = None,
                      raise_errors: bool = False) -> bool:
        if not local.exists():
            return True
        ignore = ["*.tmp", *(ignore or [])]
        try:
            self._check_allowed(self._folder_remotes(local, path_in_repo, ignore))
            self._guard()
            self._folder(local, path_in_repo, message, ignore)
            LOG.info("uploaded folder %s -> %s/%s", rel(local), self.url, path_in_repo)
            return True
        except PrivacyViolation:
            raise
        except Exception as e:  # noqa: BLE001
            if raise_errors:
                raise
            LOG.warning("upload of %s failed (%s: %s) -- will retry at the next upload point",
                        rel(local), type(e).__name__, e)
            return False

    def periodic_training_upload(self) -> None:
        if time.time() - self.last_periodic < UPLOAD_MIN_INTERVAL_S:
            return
        self.last_periodic = time.time()
        self.upload_files([(PROGRESS, "progress.log"), (TRAIN_LOG, "eval/train_log.csv"),
                           (DEV_CURVE, "eval/dev_curve.csv"), (CURVES_PNG, "eval/training_curves.png"),
                           (RUN_INFO, "eval/run_info.json")], "training progress")

    def upload_verbis_outputs(self, raise_errors: bool = False) -> bool:
        return self.upload_files([(PROC / f"{n}.parquet", f"data/verbis/{n}.parquet") for n in VERBIS_OUTPUTS],
                                 "processed Verbis data", raise_errors=raise_errors)

    # -- checkpoints
    def upload_checkpoint(self, ckpt_dir: Path, best_name: str | None) -> None:
        """Called from on_save. Queues the upload in the background; training never waits for it."""
        self._collect_finished()
        names = sorted(self.retry | {ckpt_dir.name}, key=_ckpt_step)
        self.retry.clear()
        try:
            self._guard()
        except PrivacyViolation:
            raise
        except Exception as e:  # noqa: BLE001
            LOG.warning("checkpoint upload of %s postponed (%s) -- retrying at the next save", ", ".join(names), e)
            self.retry.update(names)
            return
        queued = {}
        for name in names:
            local = ckpt_dir.parent / name
            if not (local / "trainer_state.json").exists():  # rotated away locally in the meantime
                continue
            try:
                self._check_allowed(self._folder_remotes(local, f"checkpoints/{name}", ["*.tmp"]))
                queued[name] = self.pending[name] = self._folder(local, f"checkpoints/{name}", f"checkpoint {name}",
                                                                 ["*.tmp"], run_as_future=True)
            except Exception as e:  # noqa: BLE001
                LOG.warning("could not queue the upload of %s (%s) -- retrying at the next save", name, e)
                self.retry.add(name)
        if queued:
            LOG.info("checkpoint upload queued in the background: %s", ", ".join(queued))
            self._background(self._prune_checkpoints, best_name, queued).add_done_callback(self._log_prune_result)

    def _collect_finished(self) -> None:
        for name, fut in list(self.pending.items()):
            if not fut.done():
                continue
            del self.pending[name]
            exc = fut.exception()
            if exc is None:
                LOG.info("checkpoint %s uploaded to %s/checkpoints/%s", name, self.url, name)
            else:
                LOG.warning("background upload of %s failed (%s: %s) -- retrying at the next save",
                            name, type(exc).__name__, exc)
                self.retry.add(name)

    @staticmethod
    def _log_prune_result(fut) -> None:
        if fut.exception() is not None:
            LOG.warning("pruning old checkpoints in the repo failed (%s) -- retried after the next save",
                        fut.exception())

    def hub_checkpoints(self) -> list[str]:
        """Complete checkpoints in the repo (a folder upload is one commit, so trainer_state.json marks it complete)."""
        found = {m.group(1) for f in self._repo_files()
                 if (m := re.fullmatch(r"checkpoints/(checkpoint-\d+)/trainer_state\.json", f))}
        return sorted(found, key=_ckpt_step)

    def _prune_checkpoints(self, best_name: str | None, just_uploaded: dict) -> None:
        """Keep the last 3 + the best checkpoint in the repo. Runs after the uploads it follows
        (single FIFO worker), and only if all of them succeeded."""
        if any(f.exception() is not None for f in just_uploaded.values()):
            return
        on_hub = self.hub_checkpoints()
        keep = set(on_hub[-3:]) | ({best_name} if best_name else set())
        drop = [n for n in on_hub if n not in keep]
        if drop:
            self._delete_folders([f"checkpoints/{n}" for n in drop], f"prune checkpoints: {', '.join(drop)}")
            LOG.info("removed from the repo: %s (kept %s)", ", ".join(drop), ", ".join(sorted(keep, key=_ckpt_step)))

    def sync_checkpoints(self, ckpt_root: Path) -> None:
        """Final upload: wait for background uploads, upload local checkpoints missing in the repo, prune."""
        for fut in list(self.pending.values()):
            try:
                fut.result()
            except Exception:  # noqa: BLE001 -- re-uploaded below
                pass
        self._collect_finished()
        self.retry.clear()
        local = sorted((p.name for p in ckpt_root.glob("checkpoint-*") if (p / "trainer_state.json").exists()),
                       key=_ckpt_step) if ckpt_root.exists() else []
        if not local:
            return
        best = json.loads((ckpt_root / local[-1] / "trainer_state.json").read_text()).get("best_model_checkpoint")
        on_hub = set(self.hub_checkpoints())
        for name in local:
            if name not in on_hub:
                self.upload_folder(ckpt_root / name, f"checkpoints/{name}", f"checkpoint {name}", raise_errors=True)
        self._prune_checkpoints(Path(best).name if best else None, {})

    def download_checkpoint(self, name: str, ckpt_root: Path) -> Path:
        tmp = ckpt_root.parent / "_hub_download"
        if tmp.exists():
            shutil.rmtree(tmp)
        self._download([f"checkpoints/{name}/*"], tmp)
        dest = ckpt_root / name
        ckpt_root.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp / "checkpoints" / name, dest)
        shutil.rmtree(tmp, ignore_errors=True)
        return dest

    def download_file(self, remote: str, dest: Path) -> bool:
        tmp = dest.parent / f"_hub_download_{dest.name}"
        try:
            self._download([remote], tmp)
            if not (tmp / remote).exists():
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp / remote, dest)
            return True
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def upload_everything(self) -> None:
        """Final upload; raises on failure so the caller can retry."""
        self.sync_checkpoints(CKPT)
        self.upload_folder(FINAL, "", "fine-tuned model (best dev chrF++)", ignore=["training_args.bin"],
                           raise_errors=True)
        self.upload_folder(CT2_FT, "ct2/finetuned_int8", "CTranslate2 int8: fine-tuned", raise_errors=True)
        self.upload_folder(CT2_BASE, "ct2/base_int8", "CTranslate2 int8: base", raise_errors=True)
        self.upload_verbis_outputs(raise_errors=True)
        self.upload_folder(EVAL, "eval", "evaluation results", raise_errors=True)
        self.upload_files([(PROGRESS, "progress.log"), (DONE, "DONE")], "progress.log and DONE", raise_errors=True)


# ----------------------------------------------------------------------------
# Translation helpers (shared by dev evaluation, smoke test and final evaluation)
# ----------------------------------------------------------------------------
@torch.inference_mode()
def hf_translate(model, tok, texts: list[str], src_lang: str, tgt_lang: str, num_beams: int,
                 batch_size: int = GEN_BATCH) -> list[str]:
    """Translate with a Hugging Face model; batches sorted by length, halved on OOM."""
    tok.src_lang = src_lang
    forced_bos = tok.convert_tokens_to_ids(tgt_lang)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    out = [""] * len(texts)
    pos, bs = 0, batch_size
    while pos < len(order):
        idx = order[pos:pos + bs]
        enc = gen = None
        oom = False
        try:
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                      max_length=1024).to(model.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen = model.generate(**enc, forced_bos_token_id=forced_bos, num_beams=num_beams, do_sample=False,
                                     max_new_tokens=MAX_NEW_TOKENS, use_cache=True)
        except torch.cuda.OutOfMemoryError:
            oom = True
        if oom:
            enc = gen = None
            free_gpu()
            if bs == 1:
                raise Fatal("out of GPU memory translating a single sentence")
            bs //= 2
            LOG.warning("out of GPU memory during generation -- batch size halved to %d", bs)
            continue
        for i, text in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = text
        pos += len(idx)
    return out


def ct2_translate(translator, tok, texts: list[str], src_lang: str, tgt_lang: str, num_beams: int,
                  batch_size: int = GEN_BATCH) -> list[str]:
    """Translate with CTranslate2; target_prefix forces the language, whose token is then dropped."""
    tok.src_lang = src_lang
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    out = [""] * len(texts)
    pos, bs = 0, batch_size
    while pos < len(order):
        idx = order[pos:pos + bs]
        ids = tok([texts[i] for i in idx], truncation=True, max_length=1024)["input_ids"]
        tokens = [tok.convert_ids_to_tokens(x) for x in ids]
        try:
            results = translator.translate_batch(tokens, target_prefix=[[tgt_lang]] * len(tokens),
                                                 beam_size=num_beams, max_batch_size=bs,
                                                 max_decoding_length=MAX_NEW_TOKENS + 1)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or bs == 1:
                raise
            bs //= 2
            LOG.warning("CTranslate2 out of memory -- batch size halved to %d", bs)
            continue
        for i, res in zip(idx, results):
            hyp = res.hypotheses[0][1:]  # drop the target language code
            out[i] = tok.decode(tok.convert_tokens_to_ids(hyp), skip_special_tokens=True)
        pos += len(idx)
    return out


def load_hf_model(path_or_name: str | Path):
    model = AutoModelForSeq2SeqLM.from_pretrained(str(path_or_name), token=TOKEN)
    return model.to("cuda", dtype=torch.bfloat16).eval()


def load_ct2(path: Path, device: str, compute_type: str):
    import ctranslate2
    return ctranslate2.Translator(str(path), device=device, compute_type=compute_type, inter_threads=1,
                                  intra_threads=16 if device == "cpu" else 0)


def corpus_scores(hyps: list[str], refs: list[str]) -> tuple[float, float, str, str]:
    bleu, chrf = BLEU(), CHRF(word_order=2)
    b = bleu.corpus_score(hyps, [refs]).score
    c = chrf.corpus_score(hyps, [refs]).score
    return b, c, str(bleu.get_signature()), str(chrf.get_signature())


# ============================================================================
# STEP 1 -- preflight
# ============================================================================
def _gpu_preflight() -> dict:
    if not torch.cuda.is_available():
        raise Fatal("CUDA is not available to torch (torch.cuda.is_available() is False). "
                    "Check the NVIDIA driver and that the CUDA build of torch is installed.")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    smi = nvidia_smi(["driver_version", "memory.total"]) or {}
    gpu = {"gpu_name": name, "compute_capability": f"{cap[0]}.{cap[1]}", "driver": smi.get("driver_version"),
           "cuda_torch": torch.version.cuda, "torch": torch.__version__,
           "torch_arch_list": torch.cuda.get_arch_list()}
    LOG.info("GPU: %s | compute capability %s | driver %s | torch %s (CUDA %s)",
             name, gpu["compute_capability"], gpu["driver"], torch.__version__, torch.version.cuda)
    try:  # a torch build without kernels for this GPU fails here, not 3 hours in
        x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        float((x @ x).float().sum())
    except Exception as e:  # noqa: BLE001
        raise Fatal(f"torch cannot run kernels on {name} (sm_{cap[0]}{cap[1]}): {e}. "
                    f"torch arch list: {torch.cuda.get_arch_list()}. Install a torch build for this GPU.") from e
    if not torch.cuda.is_bf16_supported():
        raise Fatal(f"{name} does not support bf16")
    free, total = torch.cuda.mem_get_info()
    gpu["vram_free_gb"], gpu["vram_total_gb"] = round(free / 1e9, 2), round(total / 1e9, 2)
    LOG.info("VRAM: %.2f GB free of %.2f GB", free / 1e9, total / 1e9)
    if free / 1e9 < MIN_FREE_VRAM_GB:
        try:
            smi_out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout
        except Exception as e:  # noqa: BLE001
            smi_out = f"(nvidia-smi failed: {e})"
        LOG.error("other GPU processes:\n%s", smi_out)
        raise Fatal(f"only {free / 1e9:.2f} GB VRAM free, need >= {MIN_FREE_VRAM_GB} GB. "
                    "Stop the other GPU processes listed above and re-run.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    LOG.info("TF32 enabled; bf16 supported")
    return gpu


def _git_commit() -> dict:
    def git(*args):
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except Exception:  # noqa: BLE001
        return {"commit": None}


def _versions() -> dict:
    import accelerate
    import ctranslate2
    import sentence_transformers
    return {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
            "datasets": datasets.__version__, "ctranslate2": ctranslate2.__version__,
            "sacrebleu": sacrebleu.__version__, "huggingface_hub": huggingface_hub.__version__,
            "accelerate": accelerate.__version__, "sentence_transformers": sentence_transformers.__version__,
            "polars": pl.__version__}


def step1_preflight() -> None:
    global TOKEN, UPLOADER
    LOG.info("repo root: %s | work dir: %s", ROOT, W)
    TOKEN = os.environ.get("HF_TOKEN") or huggingface_hub.get_token()
    if not TOKEN:
        raise Fatal("no Hugging Face token found. Set HF_TOKEN (WRITE access) or run `hf auth login` once.")
    source = "environment variable HF_TOKEN" if os.environ.get("HF_TOKEN") else "cached login"
    try:
        who = HfApi(token=TOKEN).whoami()
    except Exception as e:  # noqa: BLE001
        raise Fatal(f"the Hugging Face token (from {source}) was rejected: {e}") from e
    LOG.info("Hugging Face token: found (%s), user %s", source, who.get("name"))

    gpu = _gpu_preflight()

    for d in (RAW, PROC, TOKD, MODELS, CT2, EVAL):
        d.mkdir(parents=True, exist_ok=True)

    info = read_run_info()
    invocations = info.get("invocations", []) + [now_iso()]
    constants = {"HF_GAZETTE_REPO": HF_GAZETTE_REPO, "HF_RESULTS_REPO": HF_RESULTS_REPO, "VERBIS_PATH": VERBIS_PATH,
                 "VERBIS_HF_PATH": VERBIS_HF_PATH, "WORK_DIR": WORK_DIR, "BASE_MODEL": BASE_MODEL, "SEED": SEED, "MK": MK, "SQ": SQ,
                 "MAX_TOKENS": MAX_TOKENS, "LABSE_MIN_SCORE": LABSE_MIN_SCORE, "DEV_EVAL_SIZE": DEV_EVAL_SIZE,
                 "EVAL_STEPS": EVAL_STEPS, "FINAL_BEAMS": FINAL_BEAMS, "MAX_NEW_TOKENS": MAX_NEW_TOKENS}
    update_run_info(upload=False, script=Path(__file__).name, git=_git_commit(), versions=_versions(), gpu=gpu,
                    constants=constants, results_repo=f"https://huggingface.co/{HF_RESULTS_REPO}",
                    invocations=invocations)
    LOG.info("run info:\n%s", RUN_INFO.read_text(encoding="utf-8"))

    UPLOADER = Uploader(TOKEN)
    LOG.info("results repo: %s", UPLOADER.url)
    try:
        UPLOADER.ensure_private_repo()
        UPLOADER.upload_files([(RUN_INFO, "eval/run_info.json")], "preflight: run_info.json", raise_errors=True)
    except Fatal:
        UPLOADER = None
        raise
    except Exception as e:  # noqa: BLE001
        UPLOADER = None
        raise Fatal(f"preflight upload to {HF_RESULTS_REPO} failed ({type(e).__name__}: {e}). "
                    "Check that the token has WRITE access to this repo and that huggingface.co is reachable.") from e
    LOG.info("preflight test upload OK")


# ============================================================================
# STEP 2 -- Gazette corpus
# ============================================================================
MK_COL_CANDIDATES = ("mk", "mk_text", "text_mk", "macedonian", "mkd")
SQ_COL_CANDIDATES = ("sq", "sq_text", "text_sq", "albanian", "sqi", "als")
GROUP_COL_CANDIDATES = ("issue_key", "source_pdf", "document_id", "doc_id", "document", "issue")
ID_COL_CANDIDATES = ("pair_id", "id")


def _holdout_dev(train: pl.DataFrame, group_col: str | None) -> tuple[pl.DataFrame, pl.DataFrame]:
    target = int(round(0.015 * train.height))
    if group_col is None:
        LOG.warning("no dev split and no document/issue column -- holding out %d random rows as dev", target)
        shuffled = train.sample(fraction=1.0, shuffle=True, seed=SEED)
        return shuffled.slice(target), shuffled.slice(0, target)
    sizes = dict(train.group_by(group_col).len().iter_rows())
    groups = sorted(g for g in sizes if g is not None)
    random.Random(SEED).shuffle(groups)
    picked, n = [], 0
    for g in groups:
        if n >= target:
            break
        picked.append(g)
        n += sizes[g]
    is_dev = pl.col(group_col).is_in(picked)
    LOG.info("no dev split -- held out %d %s groups (%d rows, %.2f%%) as dev",
             len(picked), group_col, n, 100 * n / max(train.height, 1))
    return train.filter(~is_dev), train.filter(is_dev)


def step2_gazette() -> None:
    outs = {s: PROC / f"gazette_{s}.parquet" for s in ("train", "dev", "test")}
    if all(p.exists() for p in outs.values()):
        for p in outs.values():
            have(p)
    else:
        dd = load_dataset(HF_GAZETTE_REPO, token=TOKEN)
        for split, ds in dd.items():
            LOG.info("Gazette split %r: %d rows | schema: %s", split, ds.num_rows,
                     {k: getattr(v, "dtype", type(v).__name__) for k, v in ds.features.items()})
        if "train" not in dd or "test" not in dd:
            raise Fatal(f"{HF_GAZETTE_REPO} must have 'train' and 'test' splits, found {list(dd)}")
        splits = {"train": "train", "test": "test",
                  "dev": next((s for s in ("dev", "validation", "valid") if s in dd), None)}
        cols = dd["train"].column_names

        def pick(candidates, what):
            col = next((c for c in candidates if c in cols), None)
            if col is None:
                raise Fatal(f"cannot find the {what} column in {HF_GAZETTE_REPO}; columns are {cols}")
            return col

        mk_col, sq_col = pick(MK_COL_CANDIDATES, "Macedonian"), pick(SQ_COL_CANDIDATES, "Albanian")
        group_col = next((c for c in GROUP_COL_CANDIDATES if c in cols), None)
        id_col = next((c for c in ID_COL_CANDIDATES if c in cols), None)
        keep = [mk_col, sq_col] + [c for c in (id_col, group_col) if c]
        LOG.info("column mapping: %s -> mk, %s -> sq (also kept: %s)", mk_col, sq_col, keep[2:] or "none")
        frames = {}
        for name, split in splits.items():
            if split is None:
                continue
            table = dd[split].select_columns(keep).with_format("arrow")[:]
            frames[name] = _drop_empty(pl.from_arrow(table).rename({mk_col: "mk", sq_col: "sq"}), f"gazette {name}")
        if splits["dev"] is None:
            frames["train"], frames["dev"] = _holdout_dev(frames["train"], group_col)
        for name, df in frames.items():
            write_parquet(df, outs[name])
    sizes = {}
    for name, p in outs.items():
        sizes[f"gazette_{name}"] = pl.scan_parquet(p).select(pl.len()).collect().item()
        LOG.info("Gazette %-5s: %d rows", name, sizes[f"gazette_{name}"])
    update_dataset_sizes(**sizes)


# ============================================================================
# STEP 3 -- external test sets (FLORES+ devtest, NTREX-128)
# ============================================================================
def _flores_side(lang: str) -> pl.DataFrame:
    api = HfApi(token=TOKEN)
    try:
        files = api.list_repo_files(FLORES_REPO, repo_type="dataset")
        cands = [f for f in files if f.split("/")[0] == "devtest" and Path(f).name.split(".")[0] == lang]
        if not cands:
            raise Fatal(f"no devtest file for {lang} in {FLORES_REPO}")
        fname = sorted(cands, key=lambda f: (not f.endswith(".parquet"), f))[0]
        local = hf_hub_download(FLORES_REPO, fname, repo_type="dataset", token=TOKEN)
    except GatedRepoError as e:
        raise Fatal(f"{FLORES_REPO} is gated: accept its terms at https://huggingface.co/datasets/{FLORES_REPO} "
                    "with the account that owns the token, then re-run.") from e
    LOG.info("FLORES+ %s: %s", lang, fname)
    return pl.read_parquet(local) if local.endswith(".parquet") else pl.read_ndjson(local)


def _download(url: str, dest: Path) -> None:
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            tmp = _tmp(dest)
            tmp.write_bytes(data)
            os.replace(tmp, dest)
            LOG.info("downloaded %s -> %s (%d bytes)", url, rel(dest), len(data))
            return
        except Exception as e:  # noqa: BLE001
            LOG.warning("download attempt %d of %s failed: %s", attempt, url, e)
            time.sleep(10 * attempt)
    raise Fatal(f"could not download {url}")


def _read_lines(path: Path) -> list[str]:
    # split on "\n" only: str.splitlines() also splits on U+2028 etc. and would misalign the sides
    lines = path.read_text(encoding="utf-8-sig").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line.rstrip("\r") for line in lines]


def step3_external() -> None:
    flores_p, ntrex_p = PROC / "flores_devtest.parquet", PROC / "ntrex.parquet"
    if not have(flores_p):
        mk, sq = _flores_side(MK), _flores_side(SQ)
        if mk.height != sq.height:
            raise Fatal(f"FLORES+ devtest sides differ: {MK}={mk.height} lines, {SQ}={sq.height} lines")
        if "id" in mk.columns and "id" in sq.columns:
            df = (mk.select("id", mk="text").join(sq.select("id", sq="text"), on="id", how="inner", validate="1:1")
                  .sort("id").drop("id"))
        else:
            df = pl.DataFrame({"mk": mk["text"], "sq": sq["text"]})
        assert df.height == mk.height, "FLORES+ id alignment lost rows"
        if df.height != 1012:
            LOG.warning("FLORES+ devtest has %d sentences, expected 1,012", df.height)
        write_parquet(df, flores_p)
    if not have(ntrex_p):
        ntrex_dir = RAW / "ntrex"
        ntrex_dir.mkdir(parents=True, exist_ok=True)
        sides = {}
        for key, code in (("mk", "mkd"), ("sq", "sqi")):
            path = ntrex_dir / f"newstest2019-ref.{code}.txt"
            if not have(path):
                _download(NTREX_URL.format(code=code), path)
            sides[key] = _read_lines(path)
        if len(sides["mk"]) != len(sides["sq"]):
            raise Fatal(f"NTREX-128 sides differ: mkd={len(sides['mk'])} lines, sqi={len(sides['sq'])} lines")
        write_parquet(pl.DataFrame(sides), ntrex_p)
    sizes = {"flores_devtest": pl.read_parquet(flores_p).height, "ntrex": pl.read_parquet(ntrex_p).height}
    LOG.info("FLORES+ devtest: %d sentences | NTREX-128: %d sentences", sizes["flores_devtest"], sizes["ntrex"])
    update_dataset_sizes(**sizes)


# ============================================================================
# STEP 4 -- Verbis dictionary
# ============================================================================
_PAREN_TAIL = re.compile(r"\s*\([^()]*\)[\s;:,.]*$")
_INFLECTION_TAIL = re.compile(r",\s*-\s*[^\s,]+$")  # Albanian headword endings: "AR,-I", "ALTERNATIVË,-A"


def _clean_headword(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()
    prev = None
    while prev != s:  # "X (A) (B)" -> "X"
        prev = s
        s = _PAREN_TAIL.sub("", s).strip()
        s = _INFLECTION_TAIL.sub("", s).strip()
        s = s.rstrip(";:,").strip()
    s = re.sub(r"\s+", " ", s)
    if s.isupper() and len(s) > 4:
        s = s.lower()
    return s


def _labse_scores(mk: list[str], sq: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    # fp32 on purpose: the job is small, and the 0.70 cut-off should not move with bf16 rounding.
    model = SentenceTransformer(LABSE_MODEL, device="cuda")
    try:
        kw = {"batch_size": 256, "convert_to_tensor": True, "normalize_embeddings": True, "show_progress_bar": False}
        e_mk, e_sq = model.encode(mk, **kw), model.encode(sq, **kw)
        scores = (e_mk * e_sq).sum(dim=1).float().cpu().numpy()
    finally:
        del model
        free_gpu()
    LOG.info("LaBSE freed; GPU memory allocated now %.2f GB", torch.cuda.memory_allocated() / 1e9)
    return scores


def step4_verbis() -> None:
    src = ROOT / VERBIS_PATH
    if not have(src):
        try:
            cached = hf_hub_download(HF_RESULTS_REPO, VERBIS_HF_PATH, repo_type="model", token=TOKEN)
        except Exception as e:  # noqa: BLE001
            raise Fatal(f"Verbis is not at {src} and could not be downloaded from "
                        f"{HF_RESULTS_REPO}/{VERBIS_HF_PATH}: {type(e).__name__}: {e}") from e
        src.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, _tmp(src))
        os.replace(_tmp(src), src)
        LOG.info("downloaded %s/%s -> %s", HF_RESULTS_REPO, VERBIS_HF_PATH, src)
    terms_p, scored_p, defs_p = PROC / "verbis_terms.parquet", PROC / "verbis_defs_scored.parquet", PROC / "verbis_defs.parquet"
    need = {"mk", "mk_description", "sq", "sq_description"}
    if not have(terms_p):
        v = pl.read_parquet(src)
        if not need <= set(v.columns):
            raise Fatal(f"{src} must have columns {sorted(need)}, found {v.columns}")
        LOG.info("Verbis: %d entries", v.height)
        terms = v.select(mk=pl.col("mk").map_elements(_clean_headword, return_dtype=pl.Utf8),
                         sq=pl.col("sq").map_elements(_clean_headword, return_dtype=pl.Utf8))
        terms = _drop_empty(terms, "verbis terms").unique(maintain_order=True)
        write_parquet(terms, terms_p)
    if not have(scored_p):
        v = pl.read_parquet(src)
        defs = _drop_empty(v.select(mk="mk_description", sq="sq_description"), "verbis definitions")
        defs = defs.unique(maintain_order=True)
        LOG.info("scoring %d definition pairs with %s", defs.height, LABSE_MODEL)
        scores = _labse_scores(defs["mk"].to_list(), defs["sq"].to_list())
        write_parquet(defs.with_columns(labse_score=pl.Series(scores, dtype=pl.Float32)), scored_p)
    if not have(defs_p):
        scored = pl.read_parquet(scored_p)
        write_parquet(scored.filter(pl.col("labse_score") >= LABSE_MIN_SCORE).select("mk", "sq"), defs_p)
    terms, scored, defs = pl.read_parquet(terms_p), pl.read_parquet(scored_p), pl.read_parquet(defs_p)
    LOG.info("Verbis term pairs kept: %d", terms.height)
    LOG.info("Verbis definition pairs: %d scored, %d kept with LaBSE >= %.2f (%.1f%%), median score %.3f",
             scored.height, defs.height, LABSE_MIN_SCORE, 100 * defs.height / max(scored.height, 1),
             float(scored["labse_score"].median() or 0))
    update_dataset_sizes(verbis_terms=terms.height, verbis_defs_scored=scored.height, verbis_defs=defs.height)
    UPLOADER.upload_verbis_outputs()


# ============================================================================
# STEP 5 -- leakage removal (train only; dev and test sets are never modified)
# ============================================================================
def step5_leakage() -> None:
    out_p, rep_p = PROC / "train_clean.parquet", PROC / "leakage_report.json"
    if out_p.exists() and rep_p.exists():
        have(out_p)
        have(rep_p)
    else:
        held_out = {"gazette_dev": PROC / "gazette_dev.parquet", "gazette_test": PROC / "gazette_test.parquet",
                    "flores_devtest": PROC / "flores_devtest.parquet", "ntrex": PROC / "ntrex.parquet"}
        frames = [pl.read_parquet(p, columns=["mk", "sq"]) for p in held_out.values()]
        mk_keys = pl.concat([f.select(k_mk=_norm("mk")) for f in frames]).unique().with_columns(hit_mk=pl.lit(True))
        sq_keys = pl.concat([f.select(k_sq=_norm("sq")) for f in frames]).unique().with_columns(hit_sq=pl.lit(True))
        LOG.info("held-out texts: %d unique mk, %d unique sq", mk_keys.height, sq_keys.height)
        parts, report = [], {}
        for source, path in (("gazette", PROC / "gazette_train.parquet"),
                             ("verbis_terms", PROC / "verbis_terms.parquet"),
                             ("verbis_defs", PROC / "verbis_defs.parquet")):
            df = pl.read_parquet(path, columns=["mk", "sq"])
            flagged = (df.with_columns(k_mk=_norm("mk"), k_sq=_norm("sq"))
                       .join(mk_keys, on="k_mk", how="left").join(sq_keys, on="k_sq", how="left")
                       .with_columns(pl.col("hit_mk").fill_null(False), pl.col("hit_sq").fill_null(False)))
            clean = flagged.filter(~(pl.col("hit_mk") | pl.col("hit_sq"))).select("mk", "sq")
            report[source] = {"rows_before": df.height, "rows_removed": df.height - clean.height,
                              "matched_on_mk": int(flagged["hit_mk"].sum()),
                              "matched_on_sq": int(flagged["hit_sq"].sum()), "rows_after": clean.height}
            parts.append(clean.with_columns(source=pl.lit(source)))
        write_parquet(pl.concat(parts), out_p)
        write_json(rep_p, {"removed_per_source": report, "held_out_sets": list(held_out),
                           "normalisation": "lowercase, whitespace collapsed, stripped",
                           "rule": "a train pair is removed if its mk OR its sq text appears in any held-out set"})
    report = json.loads(rep_p.read_text(encoding="utf-8"))["removed_per_source"]
    for source, r in report.items():
        LOG.info("leakage %-12s: removed %d of %d rows (mk matches %d, sq matches %d) -> %d",
                 source, r["rows_removed"], r["rows_before"], r["matched_on_mk"], r["matched_on_sq"], r["rows_after"])
    update_dataset_sizes(leakage_removed={s: r["rows_removed"] for s, r in report.items()},
                         train_clean_pairs=pl.scan_parquet(out_p).select(pl.len()).collect().item())


# ============================================================================
# STEP 6 -- bidirectional training set
# ============================================================================
def step6_bidirectional() -> None:
    out_p = PROC / "train_bidirectional.parquet"
    if not have(out_p):
        df = pl.read_parquet(PROC / "train_clean.parquet")
        fwd = df.select(src_text="mk", tgt_text="sq", src_lang=pl.lit(MK), tgt_lang=pl.lit(SQ), source="source")
        bwd = df.select(src_text="sq", tgt_text="mk", src_lang=pl.lit(SQ), tgt_lang=pl.lit(MK), source="source")
        write_parquet(pl.concat([fwd, bwd]).sample(fraction=1.0, shuffle=True, seed=SEED), out_p)
    df = pl.read_parquet(out_p, columns=["source", "src_lang"])
    comp = df.group_by("source").len().sort("len", descending=True)
    total = df.height
    LOG.info("final training composition (%d examples, both directions):", total)
    for source, n in comp.iter_rows():
        LOG.info("  %-13s %9d examples (%5.1f%%)", source, n, 100 * n / total)
    by_dir = dict(df.group_by("src_lang").len().iter_rows())
    LOG.info("  directions: mk->sq %d, sq->mk %d", by_dir.get(MK, 0), by_dir.get(SQ, 0))
    update_dataset_sizes(train_bidirectional=total, train_composition=dict(comp.iter_rows()))


# ============================================================================
# STEP 7 -- tokenisation
# ============================================================================
def make_tokenize_fn(tok, drop_long: bool):
    """Batched map function; each direction is tokenised separately with its own src/tgt language."""
    def fn(batch):
        n = len(batch["src_text"])
        ids_out, lab_out = [None] * n, [None] * n
        for sl, tl in ((MK, SQ), (SQ, MK)):
            idx = [i for i in range(n) if batch["src_lang"][i] == sl and batch["tgt_lang"][i] == tl]
            if not idx:
                continue
            tok.src_lang, tok.tgt_lang = sl, tl
            enc = tok([batch["src_text"][i] for i in idx], text_target=[batch["tgt_text"][i] for i in idx])
            for j, i in enumerate(idx):
                ids_out[i], lab_out[i] = enc["input_ids"][j], enc["labels"][j]
        out = {"input_ids": [], "attention_mask": [], "labels": [], "length": []}
        for ids, lab in zip(ids_out, lab_out):
            if ids is None:
                raise ValueError("example with an unknown language direction")
            if drop_long and (len(ids) > MAX_TOKENS or len(lab) > MAX_TOKENS):
                continue
            out["input_ids"].append(ids)
            out["attention_mask"].append([1] * len(ids))
            out["labels"].append(lab)
            out["length"].append(max(len(ids), len(lab)))
        return out
    return fn


def _check_lang_tokens(ds: Dataset, tok, what: str) -> None:
    mk_id, sq_id = tok.convert_tokens_to_ids(MK), tok.convert_tokens_to_ids(SQ)
    sample = ds.select(range(min(2000, len(ds))))
    seen = set()
    for ids, lab in zip(sample["input_ids"], sample["labels"]):
        pair = (ids[0], lab[0])
        assert pair in {(mk_id, sq_id), (sq_id, mk_id)}, f"{what}: example starts with {pair}, not a language pair"
        seen.add(pair)
    assert len(seen) == 2, f"{what}: only one direction in the first {len(sample)} examples"
    LOG.info("%s: language tokens verified on %d examples (both directions present)", what, len(sample))


def step7_tokenize() -> None:
    train_dir, dev_dir = TOKD / "train", TOKD / "dev"
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=TOKEN)
    mk_id, sq_id = tok.convert_tokens_to_ids(MK), tok.convert_tokens_to_ids(SQ)
    dev_df = pl.read_parquet(PROC / "gazette_dev.parquet", columns=["mk", "sq"])

    # One example per direction, through exactly the function used by map().
    fn = make_tokenize_fn(tok, drop_long=False)
    for sl, tl, s_col, t_col, want in ((MK, SQ, "mk", "sq", (mk_id, sq_id)), (SQ, MK, "sq", "mk", (sq_id, mk_id))):
        enc = fn({"src_text": [dev_df[s_col][0]], "tgt_text": [dev_df[t_col][0]], "src_lang": [sl], "tgt_lang": [tl]})
        got = (enc["input_ids"][0][0], enc["labels"][0][0])
        assert got == want, f"{sl}->{tl}: input_ids/labels start with {got}, expected {want}"
        LOG.info("%s->%s: input_ids start %s, labels start %s -- OK", sl, tl,
                 tok.convert_ids_to_tokens(enc["input_ids"][0][:3]), tok.convert_ids_to_tokens(enc["labels"][0][:3]))

    if not have(train_dir):
        ds = Dataset.from_parquet(str(PROC / "train_bidirectional.parquet"))
        out = ds.map(make_tokenize_fn(tok, drop_long=True), batched=True, batch_size=1000, num_proc=8,
                     remove_columns=ds.column_names)
        LOG.info("train: %d -> %d examples (%d dropped for > %d tokens, %.2f%%)", len(ds), len(out),
                 len(ds) - len(out), MAX_TOKENS, 100 * (len(ds) - len(out)) / max(len(ds), 1))
        save_dataset(out, train_dir)
    if not have(dev_dir):
        rows = {"src_text": dev_df["mk"].to_list() + dev_df["sq"].to_list(),
                "tgt_text": dev_df["sq"].to_list() + dev_df["mk"].to_list(),
                "src_lang": [MK] * dev_df.height + [SQ] * dev_df.height,
                "tgt_lang": [SQ] * dev_df.height + [MK] * dev_df.height}
        ds = Dataset.from_dict(rows)
        # Dev is never filtered; this tokenised copy only satisfies the Trainer --
        # dev scores come from generation on the raw text (step 8).
        out = ds.map(make_tokenize_fn(tok, drop_long=False), batched=True, batch_size=1000, num_proc=8,
                     remove_columns=ds.column_names)
        save_dataset(out, dev_dir)
    train, dev = load_from_disk(str(train_dir)), load_from_disk(str(dev_dir))
    _check_lang_tokens(train, tok, "train")
    _check_lang_tokens(dev.shuffle(seed=SEED), tok, "dev")
    lengths = train.data.column("length").to_numpy()
    LOG.info("tokenised train: %d examples, length mean %.1f / p95 %d / max %d tokens | dev: %d examples",
             len(train), lengths.mean(), int(np.percentile(lengths, 95)), int(lengths.max()), len(dev))
    update_dataset_sizes(upload=True, train_tokenized=len(train), dev_tokenized=len(dev),
                         train_dropped_over_max_tokens=read_run_info().get("dataset_sizes", {})
                         .get("train_bidirectional", len(train)) - len(train))


# ============================================================================
# STEP 8 -- training
# ============================================================================
def build_dev_eval_sets() -> dict:
    """Two fixed 500-sentence Gazette dev sets, one per direction; ids saved to eval/."""
    dev = pl.read_parquet(PROC / "gazette_dev.parquet", columns=["mk", "sq"])
    if not have(DEV_IDS):
        rng = np.random.default_rng(SEED)
        n = min(DEV_EVAL_SIZE, dev.height)
        ids = pl.concat([pl.DataFrame({"direction": [d] * n,
                                       "row_idx": np.sort(rng.choice(dev.height, n, replace=False)).tolist()})
                         for d, *_ in DIRECTIONS])
        write_parquet(ids, DEV_IDS)
    ids = pl.read_parquet(DEV_IDS)
    sets = {}
    for d, s_col, t_col, sl, tl in DIRECTIONS:
        rows = dev.select(pl.all().gather(ids.filter(pl.col("direction") == d)["row_idx"].to_list()))
        sets[d] = (rows[s_col].to_list(), rows[t_col].to_list(), sl, tl)
        LOG.info("dev eval set %s: %d sentences", d, len(sets[d][0]))
    return sets


def dev_evaluate(model, tok, dev_sets: dict) -> dict:
    was_training = model.training
    model.eval()
    metrics = {}
    try:
        for d, (src, ref, sl, tl) in dev_sets.items():
            hyp = hf_translate(model, tok, src, sl, tl, num_beams=1)
            b, c, _, _ = corpus_scores(hyp, ref)
            metrics[f"bleu_{d}"], metrics[f"chrf_{d}"] = b, c
    finally:
        if was_training:
            model.train()
    metrics["chrf"] = (metrics["chrf_mk_sq"] + metrics["chrf_sq_mk"]) / 2
    return metrics


class NllbCollator:
    """DataCollatorForSeq2Seq plus decoder_input_ids.

    NLLB (M2M100) has no prepare_decoder_input_ids_from_labels, and with label
    smoothing the Trainer pops `labels` before the forward pass, so the model
    would otherwise get no decoder input at all.
    """

    def __init__(self, tok, decoder_start_token_id: int):
        self.inner = DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, pad_to_multiple_of=8)
        self.pad_id, self.start_id = tok.pad_token_id, decoder_start_token_id

    def __call__(self, features):
        batch = self.inner(features)
        batch["decoder_input_ids"] = shift_tokens_right(batch["labels"], self.pad_id, self.start_id)
        return batch


def _worst_case_batch(rows: Dataset, tok, model) -> dict:
    """The given (longest) examples, padded to MAX_TOKENS on both sides: the largest batch training can see."""
    n = len(rows)
    ids = torch.full((n, MAX_TOKENS), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((n, MAX_TOKENS), dtype=torch.long)
    labels = torch.full((n, MAX_TOKENS), -100, dtype=torch.long)
    for i, (src, tgt) in enumerate(zip(rows["input_ids"], rows["labels"])):
        ids[i, :len(src)] = torch.tensor(src)
        mask[i, :len(src)] = 1
        labels[i, :len(tgt)] = torch.tensor(tgt)
    dec = shift_tokens_right(labels, tok.pad_token_id, model.config.decoder_start_token_id)
    return {"input_ids": ids.cuda(), "attention_mask": mask.cuda(), "decoder_input_ids": dec.cuda(),
            "labels": labels.cuda()}


def memory_probe(train_ds: Dataset, tok) -> dict:
    """3 forward/backward/optimizer steps on the longest examples; halve the batch on OOM."""
    if have(PROBE_JSON):
        probe = json.loads(PROBE_JSON.read_text(encoding="utf-8"))
        LOG.info("memory probe (from disk): batch %d x accumulation %d, gradient checkpointing %s, peak %s GB%s",
                 probe["per_device_train_batch_size"], probe["gradient_accumulation_steps"],
                 probe["gradient_checkpointing"], probe.get("peak_gb"),
                 f" -- {probe['source']}" if probe.get("source") else "")
        return probe
    lengths = train_ds.data.column("length").to_numpy()
    order = np.argsort(-lengths, kind="stable")
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, token=TOKEN).cuda()
    model.train()
    smoother = LabelSmoother(epsilon=0.1)
    attempts = [(16, 8, False), (8, 16, False), (4, 32, False), (16, 8, True), (8, 16, True), (4, 32, True)]
    chosen = None
    tried = []
    for bs, ga, gc_on in attempts:
        if gc_on and not model.is_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
        batch = _worst_case_batch(train_ds.select(order[:bs].tolist()), tok, model)
        labels = batch.pop("labels")
        opt = Adafactor(model.parameters(), lr=1e-4, scale_parameter=False, relative_step=False, warmup_init=False)
        free_gpu()
        torch.cuda.reset_peak_memory_stats()
        oom = False
        try:
            for _ in range(3):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**batch)
                    # fp32 logits, as accelerate's mixed-precision wrapper returns them
                    loss = smoother({"logits": out.logits.float()}, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                out = loss = None
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            oom = True
        peak = torch.cuda.max_memory_allocated() / 1e9
        out = loss = None
        opt.zero_grad(set_to_none=True)
        del opt, batch, labels
        free_gpu()
        tried.append({"batch": bs, "accumulation": ga, "gradient_checkpointing": gc_on, "ok": not oom,
                      "peak_gb": round(peak, 2)})
        if oom:
            LOG.warning("memory probe: batch %d (gradient checkpointing %s) ran out of memory", bs, gc_on)
            continue
        LOG.info("memory probe: batch %d x accumulation %d (gradient checkpointing %s) OK, peak %.2f GB",
                 bs, ga, gc_on, peak)
        chosen = {"per_device_train_batch_size": bs, "gradient_accumulation_steps": ga,
                  "gradient_checkpointing": gc_on, "peak_gb": round(peak, 2),
                  "probe_example_lengths": int(lengths[order[0]]), "attempts": tried}
        break
    del model
    free_gpu()
    if chosen is None:
        raise Fatal(f"memory probe failed at every setting, including batch 4 with gradient checkpointing: {tried}")
    write_json(PROBE_JSON, chosen)
    return chosen


def make_training_args(probe: dict) -> Seq2SeqTrainingArguments:
    wanted = dict(
        output_dir=str(CKPT), optim="adafactor", learning_rate=1e-4, warmup_steps=1000, lr_scheduler_type="linear",
        num_train_epochs=1, label_smoothing_factor=0.1, weight_decay=0.01, max_grad_norm=1.0,
        bf16=True, tf32=True, group_by_length=True,
        per_device_train_batch_size=probe["per_device_train_batch_size"],
        gradient_accumulation_steps=probe["gradient_accumulation_steps"],
        gradient_checkpointing=probe["gradient_checkpointing"], per_device_eval_batch_size=GEN_BATCH,
        dataloader_num_workers=8, save_total_limit=3,
        eval_strategy="steps", eval_steps=EVAL_STEPS, save_strategy="steps", save_steps=EVAL_STEPS,
        load_best_model_at_end=True, metric_for_best_model="eval_chrf", greater_is_better=True,
        logging_strategy="steps", logging_steps=LOG_STEPS, report_to="none", seed=SEED, data_seed=SEED,
        disable_tqdm=True, save_safetensors=True, predict_with_generate=False, remove_unused_columns=True,
        length_column_name="length", gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    optional = {"tf32", "disable_tqdm", "save_safetensors", "data_seed", "report_to", "length_column_name",
                "gradient_checkpointing_kwargs", "predict_with_generate"}
    fields = {f.name for f in dataclasses.fields(Seq2SeqTrainingArguments)}
    if "eval_strategy" not in fields:
        wanted["evaluation_strategy"] = wanted.pop("eval_strategy")
    dropped = sorted(k for k in wanted if k not in fields)
    if set(dropped) - optional:
        raise Fatal(f"this transformers version ({transformers.__version__}) lacks training arguments {dropped}")
    if dropped:
        LOG.warning("training arguments not supported by transformers %s, ignored: %s",
                    transformers.__version__, dropped)
    return Seq2SeqTrainingArguments(**{k: v for k, v in wanted.items() if k in fields})


def _last_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    df = pl.read_csv(path)
    return df.row(-1, named=True) if df.height else None


def _truncate_csv(path: Path, max_step: int) -> None:
    """On resume, drop log rows written after the checkpoint we resume from."""
    if not path.exists():
        return
    df = pl.read_csv(path)
    kept = df.filter(pl.col("step") <= max_step)
    if kept.height < df.height:
        write_csv(kept, path)
        LOG.info("resume: dropped %d rows after step %d from %s", df.height - kept.height, max_step, rel(path))


def _append_train_log(row: dict) -> None:
    new = not TRAIN_LOG.exists()
    with open(TRAIN_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRAIN_LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: ("" if row.get(k) is None else row[k]) for k in TRAIN_LOG_FIELDS})


def _append_dev_curve(row: dict) -> pl.DataFrame:
    new = pl.DataFrame([row])
    if DEV_CURVE.exists():
        old = pl.read_csv(DEV_CURVE).filter(pl.col("step") != row["step"])
        new = pl.concat([old, new], how="vertical_relaxed").sort("step")
    write_csv(new, DEV_CURVE)
    return new


def plot_curves() -> None:
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        if TRAIN_LOG.exists():
            tl = pl.read_csv(TRAIN_LOG)
            if tl.height:
                ax1.plot(tl["step"].to_list(), tl["loss"].to_list(), lw=1)
        ax1.set(xlabel="optimizer step", ylabel="training loss (label-smoothed)", title="Training loss")
        ax1.grid(alpha=0.3)
        if DEV_CURVE.exists():
            dc = pl.read_csv(DEV_CURVE)
            if dc.height:
                ax2.plot(dc["step"].to_list(), dc["chrf_mk_sq"].to_list(), marker="o", label="mk→sq")
                ax2.plot(dc["step"].to_list(), dc["chrf_sq_mk"].to_list(), marker="o", label="sq→mk")
                ax2.legend()
        ax2.set(xlabel="optimizer step", ylabel="chrF++", title="Gazette dev chrF++ (greedy, 500 sent./direction)")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        tmp = _tmp(CURVES_PNG)
        fig.savefig(tmp, format="png", dpi=120)
        os.replace(tmp, CURVES_PNG)
    except Exception as e:  # noqa: BLE001
        LOG.warning("could not draw %s: %s", rel(CURVES_PNG), e)
    finally:
        plt.close("all")


class TrainingMonitor(TrainerCallback):
    """train_log.csv rows, progress summaries, warnings, dev-curve bookkeeping."""

    def __init__(self):
        self.trainer = None
        self.nan_stop = None
        self.grad_norms: list[float] = []
        self.rates: list[float] = []
        self.slow_streak = 0
        self.last_temp_warn = 0.0
        self.projected = False
        self.pause_s = 0.0
        self._eval_end = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t_begin = self.t_last = time.time()
        self.step_begin = state.global_step
        self.tok_last, self.samples_last = 0, 0
        last = _last_row(TRAIN_LOG)
        self.elapsed_offset_h = float(last["elapsed_h"]) if last and last.get("elapsed_h") is not None else 0.0
        LOG.info("training %s at step %d of %d (%.1f optimizer steps per 1%%)",
                 "resumes" if self.step_begin else "starts", self.step_begin, state.max_steps, state.max_steps / 100)

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step >= state.max_steps and step % EVAL_STEPS:
            control.should_evaluate = True  # the tail after the last multiple of 1000 is also a candidate
            control.should_save = True
        done = step - self.step_begin
        if not self.projected and done >= 200:
            self.projected = True
            per_step = (time.time() - self.t_begin) / done
            LOG.info("projection after %d steps: %.2f s/step -> full epoch (%d steps) ~ %s, remaining ~ %s "
                     "(plus ~1-2 min per dev evaluation)", done, per_step, state.max_steps,
                     fmt_dur(per_step * state.max_steps), fmt_dur(per_step * (state.max_steps - step)))

    def on_save(self, args, state, control, **kwargs):
        if UPLOADER is not None:
            best = Path(state.best_model_checkpoint).name if state.best_model_checkpoint else None
            UPLOADER.upload_checkpoint(Path(args.output_dir) / f"checkpoint-{state.global_step}", best)
        if self._eval_end is not None:
            self.pause_s += time.time() - self._eval_end
            self._eval_end = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        now, step, tr = time.time(), state.global_step, self.trainer
        tok_total, samples_total = int(tr.tok_count.item()), tr.sample_count
        paused = self.pause_s > 0
        dt = max(now - self.t_last - self.pause_s, 1e-6)
        sps, tps = (samples_total - self.samples_last) / dt, (tok_total - self.tok_last) / dt
        self.t_last, self.tok_last, self.samples_last, self.pause_s = now, tok_total, samples_total, 0.0
        loss = float(logs["loss"])
        gn = logs.get("grad_norm")
        gn = float(gn) if gn is not None else None
        lr = logs.get("learning_rate")
        done = step - self.step_begin
        eta_s = (now - self.t_begin) / done * (state.max_steps - step) if done > 0 else None
        live = gpu_live_stats()
        row = {"timestamp": now_iso(), "step": step, "epoch": round(state.epoch or 0.0, 4), "loss": round(loss, 5),
               "learning_rate": lr, "grad_norm": round(gn, 4) if gn is not None else None,
               "samples_per_sec": round(sps, 2), "tokens_per_sec": round(tps, 1),
               "gpu_mem_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
               "gpu_mem_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), **live,
               "elapsed_h": round(self.elapsed_offset_h + (now - self.t_begin) / 3600, 4),
               "eta_h": round(eta_s / 3600, 3) if eta_s is not None else None}
        _append_train_log(row)

        if not math.isfinite(loss):
            path = MODELS / f"nan_debug_step{step}"
            LOG.error("loss is %s at step %d -- saving the model to %s and stopping", loss, step, rel(path))
            try:
                tr.save_model(str(path))
            except Exception as e:  # noqa: BLE001
                LOG.error("could not save the NaN-debug model: %s", e)
            self.nan_stop = (step, path)
            control.should_training_stop = True
            return
        if gn is not None and math.isfinite(gn):
            if len(self.grad_norms) >= 10 and gn > 10 * statistics.median(self.grad_norms):
                LOG.warning("grad_norm %.3f at step %d is > 10x its running median %.3f",
                            gn, step, statistics.median(self.grad_norms))
            self.grad_norms.append(gn)
        temp = live.get("gpu_temp_c")
        if temp is not None and temp > 85 and now - self.last_temp_warn > 600:
            self.last_temp_warn = now
            LOG.warning("GPU temperature %.0f °C at step %d (> 85 °C)", temp, step)
        if not paused:  # intervals that contain a dev evaluation or a checkpoint save are not comparable
            if len(self.rates) >= 5 and sps < 0.5 * statistics.median(self.rates):
                self.slow_streak += 1
            else:
                self.slow_streak = 0
            if self.slow_streak == 3:
                LOG.warning("throughput %.1f samples/s is > 50%% below its running median %.1f for 3 log intervals "
                            "in a row (thermal throttling or another process on the GPU?)",
                            sps, statistics.median(self.rates))
                self.slow_streak = 0
            self.rates.append(sps)
        if step % SUMMARY_STEPS == 0:
            LOG.info("step %d/%d (%.1f%%) | loss %.4f | lr %.2e | %.1f samples/s | peak GPU mem %.2f GB | ETA %s",
                     step, state.max_steps, 100 * step / state.max_steps, loss, lr or 0.0, sps,
                     row["gpu_mem_peak_gb"], fmt_dur(eta_s))

    def after_dev_eval(self, step: int, metrics: dict, seconds: float) -> None:
        self.pause_s += seconds
        row = {"step": step, "bleu_mk_sq": round(metrics["eval_bleu_mk_sq"], 3),
               "chrf_mk_sq": round(metrics["eval_chrf_mk_sq"], 3), "bleu_sq_mk": round(metrics["eval_bleu_sq_mk"], 3),
               "chrf_sq_mk": round(metrics["eval_chrf_sq_mk"], 3), "chrf_mean": round(metrics["eval_chrf"], 3)}
        curve = _append_dev_curve(row)
        best = curve.sort("chrf_mean", descending=True).row(0, named=True)
        LOG.info("dev @ step %d: mk→sq BLEU %.2f chrF++ %.2f | sq→mk BLEU %.2f chrF++ %.2f | mean chrF++ %.2f "
                 "(best %.2f @ step %d) | %.0fs", step, row["bleu_mk_sq"], row["chrf_mk_sq"], row["bleu_sq_mk"],
                 row["chrf_sq_mk"], row["chrf_mean"], best["chrf_mean"], best["step"], seconds)
        plot_curves()
        if UPLOADER is not None:
            UPLOADER.periodic_training_upload()
        self._eval_end = time.time()


class BidirectionalTrainer(Seq2SeqTrainer):
    """Dev evaluation by generation, each direction with its own forced_bos_token_id."""

    def setup(self, tok, dev_sets: dict, monitor: TrainingMonitor) -> None:
        self.nllb_tok, self.dev_sets, self.monitor = tok, dev_sets, monitor
        self.tok_count = torch.zeros((), dtype=torch.long, device="cuda")
        self.sample_count = 0

    def training_step(self, model, inputs, *args, **kwargs):
        # non-padding source + target tokens, accumulated on the GPU (no sync per step)
        self.tok_count += inputs["attention_mask"].sum().to(self.tok_count.device)
        self.tok_count += (inputs["labels"] != -100).sum().to(self.tok_count.device)
        self.sample_count += int(inputs["input_ids"].shape[0])
        return super().training_step(model, inputs, *args, **kwargs)

    def _get_train_sampler(self, *args, **kwargs):
        if not self.args.group_by_length:
            return super()._get_train_sampler(*args, **kwargs)
        lengths = self.train_dataset.data.column("length").to_pylist()
        return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps,
                                    lengths=lengths, model_input_name="input_ids")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval", **kwargs):
        t0 = time.time()
        raw = dev_evaluate(self.model, self.nllb_tok, self.dev_sets)
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in raw.items()}
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self.monitor.after_dev_eval(self.state.global_step, metrics, time.time() - t0)
        return metrics


def _resume_from_hub() -> str | None:
    """No local checkpoint: download the latest (and the best) checkpoint from the private repo."""
    if UPLOADER is None:
        return None
    try:
        names = UPLOADER.hub_checkpoints()
    except Exception as e:  # noqa: BLE001
        LOG.warning("could not list checkpoints in %s (%s) -- starting from the base model", UPLOADER.url, e)
        return None
    if not names:
        return None
    latest = names[-1]
    LOG.info("no local checkpoint -- downloading %s from %s to resume from it", latest, UPLOADER.url)
    path = UPLOADER.download_checkpoint(latest, CKPT)
    state_p = path / "trainer_state.json"
    state = json.loads(state_p.read_text())
    if state.get("best_model_checkpoint"):
        best = Path(state["best_model_checkpoint"]).name
        if best != latest and best in names and not (CKPT / best).exists():
            LOG.info("also downloading the best checkpoint %s", best)
            UPLOADER.download_checkpoint(best, CKPT)
        state["best_model_checkpoint"] = str(CKPT / best)  # the path recorded on the machine that saved it
        write_json(state_p, state)
    for local, remote in ((TRAIN_LOG, "eval/train_log.csv"), (DEV_CURVE, "eval/dev_curve.csv")):
        if not local.exists():
            try:
                if UPLOADER.download_file(remote, local):
                    LOG.info("downloaded %s from the repo", remote)
            except Exception as e:  # noqa: BLE001
                LOG.warning("could not download %s (%s) -- it restarts empty", remote, e)
    if not PROBE_JSON.exists():
        # Keep the batch layout of the checkpoint: a different one would change the data order and step count.
        ta = torch.load(path / "training_args.bin", weights_only=False)
        write_json(PROBE_JSON, {"per_device_train_batch_size": ta.per_device_train_batch_size,
                                "gradient_accumulation_steps": ta.gradient_accumulation_steps,
                                "gradient_checkpointing": bool(ta.gradient_checkpointing), "peak_gb": None,
                                "source": f"training_args.bin of {latest} (resumed from the repo)"})
    return str(path)


def step8_train():
    if (FINAL / "model.safetensors").exists():
        LOG.info("loaded from disk: %s -- training already finished, skipped", rel(FINAL))
        return None
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=TOKEN)
    train_ds = load_from_disk(str(TOKD / "train"))
    dev_ds = load_from_disk(str(TOKD / "dev"))
    dev_sets = build_dev_eval_sets()

    last = get_last_checkpoint(str(CKPT)) if CKPT.exists() else None
    if last is None:
        last = _resume_from_hub()
    probe = memory_probe(train_ds, tok)
    args = make_training_args(probe)
    eff = args.per_device_train_batch_size * args.gradient_accumulation_steps
    LOG.info("training settings: batch %d x accumulation %d = effective %d | gradient checkpointing %s | "
             "adafactor lr %.0e, warmup %d, linear decay, %d epoch | label smoothing %.1f | bf16 + tf32",
             args.per_device_train_batch_size, args.gradient_accumulation_steps, eff, args.gradient_checkpointing,
             args.learning_rate, args.warmup_steps, args.num_train_epochs, args.label_smoothing_factor)
    update_run_info(memory_probe=probe, training_args=args.to_dict())

    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, token=TOKEN)
    if probe["gradient_checkpointing"]:
        model.config.use_cache = False
    collator = NllbCollator(tok, model.config.decoder_start_token_id)

    resume_step = 0
    if last:
        resume_step = json.loads((Path(last) / "trainer_state.json").read_text())["global_step"]
        LOG.info("resuming from %s (step %d)", rel(Path(last)), resume_step)
    _truncate_csv(TRAIN_LOG, resume_step)
    _truncate_csv(DEV_CURVE, resume_step)

    monitor = TrainingMonitor()
    params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    tok_kw = {"processing_class": tok} if "processing_class" in params else {"tokenizer": tok}
    trainer = BidirectionalTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds,
                                   data_collator=collator, callbacks=[monitor], **tok_kw)
    trainer.remove_callback(PrinterCallback)  # our monitor logs instead of printing every log dict
    trainer.setup(tok, dev_sets, monitor)
    monitor.trainer = trainer

    t0 = time.time()
    trainer.train(resume_from_checkpoint=last)
    if monitor.nan_stop:
        step, path = monitor.nan_stop
        raise Fatal(f"training stopped: loss became NaN/inf at step {step}. Model saved to {path}. "
                    f"Re-running would resume from the last checkpoint in {CKPT}; inspect "
                    f"{rel(TRAIN_LOG)} first (learning rate, grad_norm) before resuming.")

    last_row = _last_row(TRAIN_LOG) or {}
    best_ckpt = trainer.state.best_model_checkpoint
    m = re.search(r"checkpoint-(\d+)", best_ckpt or "")
    summary = {"total_training_time_h": last_row.get("elapsed_h"), "this_invocation_s": round(time.time() - t0),
               "global_step": trainer.state.global_step, "best_dev_step": int(m.group(1)) if m else None,
               "best_dev_chrf_mean": trainer.state.best_metric, "best_checkpoint": best_ckpt,
               "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    LOG.info("training finished: total training time %s | best dev step %s | best dev chrF++ (mean) %s",
             fmt_dur((last_row.get("elapsed_h") or 0) * 3600), summary["best_dev_step"],
             f"{trainer.state.best_metric:.2f}" if trainer.state.best_metric is not None else "n/a")
    update_run_info(training_summary=summary)
    return trainer


# ============================================================================
# STEP 9 -- save the best model
# ============================================================================
def step9_save_final(trainer) -> None:
    if have(FINAL / "model.safetensors"):
        return
    if trainer is None:
        raise Fatal(f"no trained model in memory and no {rel(FINAL)} on disk")
    LOG.info("best checkpoint (loaded by load_best_model_at_end): %s", trainer.state.best_model_checkpoint)
    tmp = _tmp(FINAL)
    if tmp.exists():
        shutil.rmtree(tmp)
    trainer.model.config.use_cache = True
    trainer.save_model(str(tmp))
    AutoTokenizer.from_pretrained(BASE_MODEL, token=TOKEN).save_pretrained(str(tmp))
    os.replace(tmp, FINAL)
    AutoTokenizer.from_pretrained(str(FINAL))  # reload check
    LOG.info("saved %s: %s", rel(FINAL), sorted(p.name for p in FINAL.iterdir()))


# ============================================================================
# STEP 10 -- CTranslate2 conversion
# ============================================================================
def _ct2_converter() -> list[str]:
    cand = Path(sys.executable).with_name("ct2-transformers-converter")
    if cand.exists():
        return [str(cand)]
    found = shutil.which("ct2-transformers-converter")
    return [found] if found else [sys.executable, "-m", "ctranslate2.converters.transformers"]


def _ct2_convert(src: Path, out: Path) -> None:
    if (out / "model.bin").exists():
        LOG.info("loaded from disk: %s", rel(out))
        return
    tmp = _tmp(out)
    if tmp.exists():
        shutil.rmtree(tmp)
    copy = [f for f in TOKENIZER_FILES if (src / f).exists()]
    cmd = [*_ct2_converter(), "--model", str(src), "--output_dir", str(tmp), "--quantization", "int8"]
    if copy:
        cmd += ["--copy_files", *copy]
    LOG.info("converting %s -> %s (int8)", src, rel(out))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise Fatal(f"ct2-transformers-converter failed (exit {r.returncode}):\n{r.stderr[-3000:]}")
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    LOG.info("wrote %s: %s", rel(out), sorted(p.name for p in out.iterdir()))


def step10_ct2() -> None:
    _ct2_convert(FINAL, CT2_FT)
    if not (CT2_BASE / "model.bin").exists():
        base_local = Path(snapshot_download(BASE_MODEL, token=TOKEN,
                                            allow_patterns=["*.json", "*.bin", "*.safetensors", "*.model"]))
        _ct2_convert(base_local, CT2_BASE)
    else:
        LOG.info("loaded from disk: %s", rel(CT2_BASE))
    UPLOADER.upload_folder(FINAL, "", "fine-tuned model (best dev chrF++)", ignore=["training_args.bin"])
    UPLOADER.upload_folder(CT2_FT, "ct2/finetuned_int8", "CTranslate2 int8: fine-tuned")
    UPLOADER.upload_folder(CT2_BASE, "ct2/base_int8", "CTranslate2 int8: base NLLB-600M")


# ============================================================================
# STEP 11 -- CTranslate2 smoke test and device choice
# ============================================================================
def _detect_ct2_device() -> tuple[str, str]:
    """Try CT2 on CUDA in a subprocess (a Blackwell/CUDA-library mismatch can abort the process)."""
    code = textwrap.dedent(f"""
        import sys, ctranslate2
        if ctranslate2.get_cuda_device_count() < 1:
            sys.exit("CTranslate2 sees no CUDA device")
        t = ctranslate2.Translator({str(CT2_FT)!r}, device="cuda", compute_type="int8_float16")
        r = t.translate_batch([["{MK}", "▁Здраво", "</s>"]], target_prefix=[["{SQ}"]], max_decoding_length=8)
        print("ok", r[0].hypotheses[0])
    """)
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and r.stdout.startswith("ok"):
            LOG.info("CTranslate2 runs on CUDA (compute_type int8_float16)")
            return "cuda", "int8_float16"
        reason = (r.stderr or r.stdout).strip()[-1500:]
    except subprocess.TimeoutExpired:
        reason = "timed out"
    LOG.warning("CTranslate2 cannot use this GPU (%s) -- falling back to CPU, compute_type int8, 16 threads", reason)
    return "cpu", "int8"


def ct2_device() -> tuple[str, str]:
    info = read_run_info()
    if info.get("ct2_device"):
        return info["ct2_device"], info["ct2_compute_type"]
    device, ctype = _detect_ct2_device()
    update_run_info(ct2_device=device, ct2_compute_type=ctype)
    return device, ctype


def step11_smoke() -> None:
    device, ctype = ct2_device()
    LOG.info("CTranslate2 device: %s (%s) -- also used in step 12", device, ctype)
    tok = AutoTokenizer.from_pretrained(str(FINAL))
    rows = pl.read_parquet(PROC / "gazette_dev.parquet", columns=["mk", "sq"]).sample(n=5, seed=SEED)
    model = translator = None
    try:
        model = load_hf_model(FINAL)
        translator = load_ct2(CT2_FT, device, ctype)
        for d, s_col, _, sl, tl in DIRECTIONS:
            src = rows[s_col].to_list()
            hf_out = hf_translate(model, tok, src, sl, tl, num_beams=FINAL_BEAMS)
            ct_out = ct2_translate(translator, tok, src, sl, tl, num_beams=FINAL_BEAMS)
            same = sum(a.strip() == b.strip() for a, b in zip(hf_out, ct_out))
            LOG.info("smoke test %s (Gazette dev): HF and CT2 identical on %d/5", d, same)
            for s, h, c in zip(src, hf_out, ct_out):
                LOG.info("  SRC: %s\n      HF : %s\n      CT2: %s", s, h, c)
    finally:
        del model, translator
        free_gpu()


# ============================================================================
# ============================================================================
# STEP 12 -- FINAL EVALUATION
#
# Standalone: reloads models from models/final/ and ct2/, test sets from
# data/processed/. Gazette dev is NOT used here. Translations are cached in
# eval/translations_<testset>.parquet (saved every EVAL_CHUNK sentences, so an
# interruption resumes mid-way); each test set's results CSV is written and
# uploaded as soon as that set is finished.
# ============================================================================
# ============================================================================
class Engine:
    """One of the three evaluated systems, loaded on demand."""

    def __init__(self, col: str, device: str, ctype: str):
        self.col = col
        if col == "base":
            self.tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=TOKEN)
            self.model = load_hf_model(BASE_MODEL)
        elif col == "finetuned_hf":
            self.tok = AutoTokenizer.from_pretrained(str(FINAL))
            self.model = load_hf_model(FINAL)
        else:
            self.tok = AutoTokenizer.from_pretrained(str(FINAL))
            self.model = load_ct2(CT2_FT, device, ctype)
        LOG.info("loaded %s (%s)", col, "CTranslate2 " + device if col == "finetuned_ct2" else "HF, GPU bf16")

    def __call__(self, texts, sl, tl):
        fn = ct2_translate if self.col == "finetuned_ct2" else hf_translate
        return fn(self.model, self.tok, texts, sl, tl, num_beams=FINAL_BEAMS)

    def close(self):
        self.model = None
        free_gpu()


def _translate_testset(name: str, data_path: Path, device: str, ctype: str) -> pl.DataFrame:
    path = EVAL / f"translations_{name}.parquet"
    if path.exists():
        df = pl.read_parquet(path)
        LOG.info("loaded from disk: %s", rel(path))
    else:
        src = pl.read_parquet(data_path, columns=["mk", "sq"])
        df = pl.concat([src.select(source=pl.col(s), reference=pl.col(t), direction=pl.lit(d))
                        for d, s, t, _, _ in DIRECTIONS])
        df = df.with_columns([pl.lit(None, dtype=pl.Utf8).alias(c) for c in MODEL_COLS])
        write_parquet(df, path)
    sources, dirs = df["source"].to_list(), df["direction"].to_list()
    cols = {c: df[c].to_list() for c in MODEL_COLS}

    def save():
        write_parquet(df.with_columns([pl.Series(c, cols[c], dtype=pl.Utf8) for c in MODEL_COLS]), path, quiet=True)

    for col in MODEL_COLS:
        todo_all = [i for i, v in enumerate(cols[col]) if v is None]
        if not todo_all:
            LOG.info("%s / %s: all %d translations cached", name, col, len(cols[col]))
            continue
        engine = Engine(col, device, ctype)
        try:
            for d, _, _, sl, tl in DIRECTIONS:
                todo = sorted((i for i in todo_all if dirs[i] == d), key=lambda i: len(sources[i]), reverse=True)
                total = sum(1 for x in dirs if x == d)
                done0 = done = total - len(todo)
                t0 = time.time()
                for k in range(0, len(todo), EVAL_CHUNK):
                    chunk = todo[k:k + EVAL_CHUNK]
                    for i, out in zip(chunk, engine([sources[i] for i in chunk], sl, tl)):
                        cols[col][i] = out
                    save()
                    done += len(chunk)
                    rate = (done - done0) / max(time.time() - t0, 1e-6)
                    LOG.info("%s / %s / %s: %d / %d sentences (%.1f sent/s, ETA %s)", name, col, d, done, total,
                             rate, fmt_dur((total - done) / rate if rate else None))
        finally:
            engine.close()
    return pl.read_parquet(path)


def _score_testset(df: pl.DataFrame) -> pl.DataFrame:
    rows, base = [], {}
    for col in MODEL_COLS:  # "base" first, so deltas are defined
        for d, *_ in DIRECTIONS:
            sub = df.filter(pl.col("direction") == d)
            b, c, b_sig, c_sig = corpus_scores(sub[col].to_list(), sub["reference"].to_list())
            if col == "base":
                base[d] = (b, c)
            rows.append({"model": col, "direction": d, "n_sentences": sub.height, "bleu": round(b, 2),
                         "chrf": round(c, 2), "bleu_delta_vs_base": round(b - base[d][0], 2),
                         "chrf_delta_vs_base": round(c - base[d][1], 2), "bleu_signature": b_sig,
                         "chrf_signature": c_sig, "timestamp": now_iso()})
    return pl.DataFrame(rows)


def format_results(name: str, table: pl.DataFrame) -> str:
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=160, tbl_hide_dataframe_shape=True,
                   tbl_hide_column_data_types=True):
        body = str(table.drop("bleu_signature", "chrf_signature", "timestamp"))
    return (f"{name}\n{body}\nBLEU signature: {table['bleu_signature'][0]}\n"
            f"chrF signature: {table['chrf_signature'][0]}")


def _log_examples(name: str, df: pl.DataFrame) -> None:
    ex = df.sample(n=min(5, df.height), seed=SEED)
    LOG.info("%s: 5 random examples", name)
    for r in ex.iter_rows(named=True):
        LOG.info("  [%s] SRC: %s\n      REF : %s\n      BASE: %s\n      FT  : %s\n      CT2 : %s", r["direction"],
                 r["source"], r["reference"], r["base"], r["finetuned_hf"], r["finetuned_ct2"])


def step12_final_eval() -> None:
    for p in [FINAL / "model.safetensors", CT2_FT / "model.bin", *(PROC / f for _, f, _ in TESTSETS)]:
        if not p.exists():
            raise Fatal(f"final evaluation needs {p}, which does not exist")
    device, ctype = ct2_device()
    LOG.info("models: base NLLB-600M (HF, GPU) | fine-tuned (HF, GPU) | fine-tuned (CTranslate2 int8, %s)", device)
    for name, data_file, res_file in TESTSETS:
        res_p = EVAL / res_file
        if have(res_p):
            continue
        LOG.info("---- test set %s ----", name)
        trans = _translate_testset(name, PROC / data_file, device, ctype)
        table = _score_testset(trans)
        write_csv(table, res_p)
        LOG.info("results\n%s", format_results(name, table))
        _log_examples(name, trans)
        UPLOADER.upload_files([(res_p, f"eval/{res_file}"),
                               (EVAL / f"translations_{name}.parquet", f"eval/translations_{name}.parquet"),
                               (PROGRESS, "progress.log")], f"results: {name}")


# ============================================================================
# STEP 13 -- finish
# ============================================================================
def step13_finish(t_start: float) -> bool:
    info = read_run_info()
    tables = [format_results(name, pl.read_csv(EVAL / res)) for name, _, res in TESTSETS]
    summary = info.get("training_summary", {})
    runtime = time.time() - t_start
    text = "\n".join([
        f"DONE {now_iso()}",
        f"private results repo: {UPLOADER.url}",
        f"runtime of this invocation: {fmt_dur(runtime)}",
        f"total training time: {fmt_dur((summary.get('total_training_time_h') or 0) * 3600)}",
        f"best dev step: {summary.get('best_dev_step')} | best dev chrF++ (mean): {summary.get('best_dev_chrf_mean')}",
        f"CTranslate2 device: {info.get('ct2_device')} ({info.get('ct2_compute_type')})",
        "", *[t + "\n" for t in tables],
    ])
    write_text(DONE, text + "\n")
    update_run_info(upload=False, finished_at=now_iso(), last_invocation_runtime_s=round(runtime))
    LOG.info("summary:\n%s", text)
    for attempt in range(1, 4):
        try:
            UPLOADER.upload_everything()
            LOG.info("final upload complete: %s", UPLOADER.url)
            return True
        except PrivacyViolation:
            raise
        except Exception as e:  # noqa: BLE001
            LOG.warning("final upload attempt %d/3 failed: %s: %s", attempt, type(e).__name__, e)
            if attempt < 3:
                time.sleep(60 * 2 ** (attempt - 1))
    LOG.error("final upload failed 3 times. Local files in %s are complete; re-run the script to retry "
              "(every step loads from disk). Repo: %s", W, UPLOADER.url)
    return False


def main() -> None:
    t_start = time.time()
    setup_logging()
    LOG.info("=" * 78)
    LOG.info("run started (pid %d): %s", os.getpid(), " ".join(sys.argv))
    run_step(1, "preflight", step1_preflight)
    run_step(2, "gazette corpus", step2_gazette)
    run_step(3, "external test sets", step3_external)
    run_step(4, "verbis", step4_verbis)
    run_step(5, "leakage removal", step5_leakage)
    run_step(6, "bidirectional training set", step6_bidirectional)
    run_step(7, "tokenisation", step7_tokenize)
    trainer = run_step(8, "training", step8_train)
    run_step(9, "save best model", step9_save_final, trainer)
    del trainer
    free_gpu()
    run_step(10, "ctranslate2 conversion", step10_ct2)
    run_step(11, "ctranslate2 smoke test", step11_smoke)
    run_step(12, "FINAL EVALUATION", step12_final_eval)
    ok = run_step(13, "finish", step13_finish, t_start)
    LOG.info("results repo: %s", UPLOADER.url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
