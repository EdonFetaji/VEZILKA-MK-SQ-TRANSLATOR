#!/usr/bin/env python
"""Smoke test for scripts/train_nllb_gazette_verbis_ct2.py -- run it on the
training machine BEFORE starting the 8-hour run:

    cd ~/VEZILKA-MK-SQ-TRANSLATOR
    .venv/bin/python scripts/smoke_test_nllb_run.py --quick   # ~5 min
    .venv/bin/python scripts/smoke_test_nllb_run.py           # full, 30-60 min

Exit code 0 means every check passed.

--quick checks only what would otherwise fail late or expensively:
  preflight (token, GPU, >= 14 GB free VRAM, private repo, test upload),
  every data source reachable (Gazette, FLORES+ gated access, NTREX, Verbis),
  a background (run_as_future) upload, checkpoints already in the repo,
  worst-case GPU memory: the memory probe, then a dev-evaluation generation
  (batch 32 x 256 new tokens) on top of the training state, then the final
  evaluation's beam-4 batch, and CTranslate2 on this GPU (it converts the base
  model into work/ct2/base_int8/, which the real run then reuses).
  It does not prepare data or run the kill/resume tests: data preparation
  fails, if at all, in the first ~15 minutes of the real run, which resumes
  from disk; kill/resume is what the full test is for.

Full test:

Phase A -- real data, real scale (work/):
  Runs the real steps 1-7 (preflight, test upload, all data preparation,
  tokenisation) on the full data. Their outputs are exactly what the real run
  would produce, so the real run reuses them and skips these steps. Then checks
  data integrity (no leakage, no nulls, lengths, language tokens), runs the
  same steps a second time to prove nothing is recomputed, and runs the real
  memory probe on the real longest examples.

Phase B -- sandbox (work/_smoke/, deleted on success):
  The real training / CTranslate2 / evaluation code with the real batch size,
  on a 60-step subset made of the LONGEST training examples (every step is
  worst-case memory):
    - training is hard-killed (SIGKILL) and resumed from the local checkpoint;
      hard-killed again, local checkpoints/logs/probe wiped (a fresh machine)
      and resumed from the checkpoint in the (fake) Hub; logs, dev curve and
      best-checkpoint tracking must survive; the Hub keeps last 3 + best
    - best model saved, CTranslate2 conversion, CT2 GPU/CPU device detection
    - final evaluation on small test subsets, hard-killed after a partial
      save and resumed; results CSVs, DONE and the upload set are checked
  Nothing from phase B is uploaded: the real upload logic runs against a local
  fake Hub (work/_smoke/fake_hub/), and every file is checked against the upload
  rules. Phase A does upload for real (preflight test upload, processed Verbis).

It measures peak RAM, minimum free RAM, peak GPU memory, and throughput, and
projects the duration of the real run. Report: work/smoke_report.json, child
logs: work/smoke_logs/.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import Future
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "scripts" / "train_nllb_gazette_verbis_ct2.py"
ROLE = os.environ.get("SMOKE_ROLE")
MARKER = "##### SMOKE CHILD START"

SMOKE_STEPS = 80  # optimizer steps of the sandbox training run (4 checkpoints, so repo pruning is exercised)
SMOKE_EVAL_STEPS = 20
KILL_AT_STEP = 30  # 1st SIGKILL once this step is logged (after checkpoint-20): resume from the local checkpoint
KILL2_AT_STEP = 50  # 2nd SIGKILL (after checkpoint-40 is in the fake Hub); then local state is wiped: resume from the Hub
SUBSET_ROWS = {"gazette_test": 150, "flores": 60, "ntrex": 60}
MIN_DISK_GB = 40
MIN_RAM_AVAILABLE_GB = 4
MIN_VRAM_HEADROOM_GB = 0.75


def load_main():
    spec = importlib.util.spec_from_file_location("nllb_main", MAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def point_to(m, root: Path) -> None:
    """Redirect every WORK_DIR path of the main script to `root` (same layout)."""
    m.W = root
    m.RAW, m.PROC, m.TOKD = root / "data" / "raw", root / "data" / "processed", root / "data" / "tokenized"
    m.MODELS = root / "models"
    m.CKPT, m.FINAL = m.MODELS / "checkpoints", m.MODELS / "final"
    m.CT2 = root / "ct2"
    m.CT2_FT, m.CT2_BASE = m.CT2 / "finetuned_int8", m.CT2 / "base_int8"
    m.EVAL = root / "eval"
    m.PROGRESS, m.DONE = root / "progress.log", root / "DONE"
    m.RUN_INFO, m.TRAIN_LOG, m.DEV_CURVE = m.EVAL / "run_info.json", m.EVAL / "train_log.csv", m.EVAL / "dev_curve.csv"
    m.CURVES_PNG, m.DEV_IDS = m.EVAL / "training_curves.png", m.EVAL / "dev_eval_ids.parquet"
    m.PROBE_JSON = m.MODELS / "memory_probe.json"


# ============================================================================
# Child roles (run in their own process so they can be killed and measured)
# ============================================================================
def make_dry_uploader(m, sandbox: Path):
    """The real Uploader logic (privacy guard, allow-list, background checkpoint uploads,
    pruning, resume downloads) running against a local fake Hub in work/_smoke/fake_hub/."""
    hub = sandbox / "fake_hub"
    record = sandbox / "upload_calls.jsonl"

    def rec(local, remote: str) -> None:
        local = Path(local).resolve()
        try:
            parts = local.relative_to(sandbox.resolve()).parts
        except ValueError:
            parts = ("OUTSIDE_WORK_DIR",)
        in_layout = any(p.fullmatch(remote) for p in m._ALLOWED_REMOTE)
        bad_source = (parts[0] == "OUTSIDE_WORK_DIR" or "tokenized" in parts
                      or (parts[0] == "data" and not remote.startswith("data/verbis/")))
        with open(record, "a", encoding="utf-8") as f:
            f.write(json.dumps({"local": "/".join(parts), "remote": remote,
                                "forbidden": (not in_layout) or bad_source}) + "\n")
        (hub / remote).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, hub / remote)

    def done(fn, *args):
        fut = Future()
        try:
            fut.set_result(fn(*args))
        except Exception as e:  # noqa: BLE001
            fut.set_exception(e)
        return fut

    class DryRunUploader(m.Uploader):
        def __init__(self):
            super().__init__(token=None)
            self.url = "fake-hub (smoke test, nothing uploaded)"
            self.last_periodic = 0.0

        def _is_private(self):
            return True

        def _commit(self, pairs, message):
            for local, remote in pairs:
                rec(local, remote)

        def _folder(self, local, path_in_repo, message, ignore, run_as_future=False):
            def copy():
                for remote in self._folder_remotes(local, path_in_repo, ignore):
                    rec(Path(local) / remote[len(path_in_repo):].lstrip("/"), remote)
            return done(copy) if run_as_future else copy()

        def _background(self, fn, *args):
            return done(fn, *args)

        def _repo_files(self):
            return [p.relative_to(hub).as_posix() for p in hub.rglob("*") if p.is_file()] if hub.exists() else []

        def _delete_folders(self, paths, message):
            for p in paths:
                shutil.rmtree(hub / p, ignore_errors=True)

        def _download(self, patterns, local_dir):
            for f in self._repo_files():
                if any(fnmatch.fnmatch(f, pat) for pat in patterns):
                    (Path(local_dir) / f).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(hub / f, Path(local_dir) / f)

    return DryRunUploader()


def sandbox_child(m) -> None:
    import huggingface_hub
    sandbox = m.W / "_smoke"
    point_to(m, sandbox)
    m.EVAL_STEPS, m.LOG_STEPS, m.SUMMARY_STEPS = SMOKE_EVAL_STEPS, 5, SMOKE_EVAL_STEPS
    m.DEV_EVAL_SIZE, m.EVAL_CHUNK, m.UPLOAD_MIN_INTERVAL_S = 32, 50, 0
    m.setup_logging()
    m.TOKEN = os.environ.get("HF_TOKEN") or huggingface_hub.get_token()
    m.UPLOADER = make_dry_uploader(m, sandbox)
    m._CTX["step"] = "smoke setup"
    m._gpu_preflight()  # also enables TF32, as step 1 does in the real run


def role_data(m) -> None:
    m.setup_logging()
    m.LOG.info("SMOKE TEST: real steps 1-7")
    m.run_step(1, "preflight", m.step1_preflight)
    m.run_step(2, "gazette corpus", m.step2_gazette)
    m.run_step(3, "external test sets", m.step3_external)
    m.run_step(4, "verbis", m.step4_verbis)
    m.run_step(5, "leakage removal", m.step5_leakage)
    m.run_step(6, "bidirectional training set", m.step6_bidirectional)
    m.run_step(7, "tokenisation", m.step7_tokenize)


def role_probe(m) -> None:
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    m.setup_logging()

    def probe():
        m._gpu_preflight()
        tok = AutoTokenizer.from_pretrained(m.BASE_MODEL)
        m.memory_probe(load_from_disk(str(m.TOKD / "train")), tok)

    m.run_step(8, "memory probe (smoke test)", probe)


def role_train(m) -> None:
    sandbox_child(m)
    trainer = m.run_step(8, "smoke training", m.step8_train)
    m.run_step(9, "smoke save best model", m.step9_save_final, trainer)


def role_ct2(m) -> None:
    sandbox_child(m)
    m.run_step(10, "smoke ctranslate2 conversion", m.step10_ct2)
    m.run_step(11, "smoke ctranslate2 smoke test", m.step11_smoke)


def role_eval(m, interrupt: bool) -> None:
    sandbox_child(m)
    timing = m.W / "engine_timing.jsonl"
    base_engine = m.Engine

    class TimedEngine(base_engine):
        def __call__(self, texts, sl, tl):
            t0 = time.time()
            out = super().__call__(texts, sl, tl)
            with open(timing, "a", encoding="utf-8") as f:
                f.write(json.dumps({"col": self.col, "n": len(texts), "secs": time.time() - t0}) + "\n")
            return out

    m.Engine = TimedEngine
    if interrupt:
        real_write = m.write_parquet
        saves = {"n": 0}

        def write_then_die(df, path, quiet=False):
            real_write(df, path, quiet)
            saves["n"] += quiet  # quiet writes are the partial-progress saves
            if saves["n"] >= 3:
                m.LOG.warning("SMOKE TEST: simulated hard kill after a partial save")
                for h in m.LOG.handlers:
                    h.flush()
                os._exit(9)

        m.write_parquet = write_then_die
    m.run_step(12, "smoke FINAL EVALUATION", m.step12_final_eval)
    m.run_step(13, "smoke finish (dry-run upload)", m.step13_finish, time.time())


def role_quick(m) -> None:
    """All quick checks in one GPU process; results go to work/quick_result.json after every stage."""
    import traceback
    import urllib.request

    import polars as pl
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers.optimization import Adafactor
    from transformers.trainer_pt_utils import LabelSmoother

    m.setup_logging()
    m.LOG.info("QUICK PREFLIGHT")
    real_w = m.W
    scratch = real_w / "_quick"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    result_path = real_w / "quick_result.json"
    results: dict = {}

    def stage(name, fn) -> bool:
        m._CTX["step"] = f"quick {name}"
        t0 = time.time()
        try:
            ok, detail = True, fn()
        except BaseException as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
            m.LOG.error("%s failed:\n%s", name, traceback.format_exc())
        results[name] = {"ok": ok, "secs": round(time.time() - t0, 1), "detail": detail}
        result_path.write_text(json.dumps(results, indent=2, default=str))
        return ok

    if not stage("preflight: HF token, GPU, >= 14 GB free VRAM, private repo, test upload", m.step1_preflight):
        return
    api = HfApi(token=m.TOKEN)

    def gazette():
        files = [s.rfilename for s in api.dataset_info(m.HF_GAZETTE_REPO).siblings]
        splits = [sp for sp in ("train", "validation", "dev", "test") if any(f"/{sp}" in f or f.startswith(sp) for f in files)]
        assert "train" in splits and "test" in splits, f"train/test split files not found in {files}"
        return f"{m.HF_GAZETTE_REPO}: split files {splits}"

    def flores():
        return f"gated access OK, {m._flores_side(m.MK).height} {m.MK} devtest sentences"

    def ntrex():
        sizes = []
        for code in ("mkd", "sqi"):
            req = urllib.request.Request(m.NTREX_URL.format(code=code), method="HEAD")
            with urllib.request.urlopen(req, timeout=30) as r:
                sizes.append(f"{code} HTTP {r.status}")
        return ", ".join(sizes)

    def verbis():
        local = m.ROOT / m.VERBIS_PATH
        path = str(local) if local.exists() else hf_hub_download(m.HF_RESULTS_REPO, m.VERBIS_HF_PATH, token=m.TOKEN)
        cols = pl.read_parquet_schema(path)
        need = {"mk", "mk_description", "sq", "sq_description"}
        assert need <= set(cols), f"columns {list(cols)} lack {sorted(need - set(cols))}"
        return f"{'local cache' if local.exists() else m.VERBIS_HF_PATH + ' in the private repo'}: columns OK"

    def background_upload():
        folder = scratch / "bg"
        folder.mkdir()
        shutil.copy(m.RUN_INFO, folder / "run_info.json")
        up = m.UPLOADER
        up._check_allowed(up._folder_remotes(folder, "eval", ["*.tmp"]))
        up._guard()
        up._folder(folder, "eval", "preflight: background upload test", ["*.tmp"], run_as_future=True).result(timeout=300)
        on_hub = up.hub_checkpoints()
        local = m.get_last_checkpoint(str(m.CKPT)) if m.CKPT.exists() else None
        note = (f"; repo already has {on_hub} -- with no local checkpoint the run RESUMES from {on_hub[-1]}"
                if on_hub and not local else "; no checkpoints in the repo" if not on_hub else "")
        return "run_as_future upload OK" + note

    for name, fn in (("Gazette dataset reachable", gazette), ("FLORES+ devtest (gated) downloadable", flores),
                     ("NTREX-128 reachable on GitHub", ntrex), ("Verbis available", verbis),
                     ("background upload + checkpoints in the repo", background_upload)):
        stage(name, fn)

    tok = AutoTokenizer.from_pretrained(m.BASE_MODEL, token=m.TOKEN)
    mk_id, sq_id, eos = tok.convert_tokens_to_ids(m.MK), tok.convert_tokens_to_ids(m.SQ), tok.eos_token_id
    g = torch.Generator().manual_seed(0)

    def rand_ids(n, lang):  # a MAX_TOKENS-long example: lang code + random ordinary tokens + </s>
        return [lang] + torch.randint(1000, 250_000, (n - 2,), generator=g).tolist() + [eos]

    L = m.MAX_TOKENS
    worst = Dataset.from_dict({"input_ids": [rand_ids(L, mk_id) for _ in range(16)],
                               "labels": [rand_ids(L, sq_id) for _ in range(16)], "length": [L] * 16})
    probe: dict = {}

    def total_gb() -> float:
        return torch.cuda.mem_get_info()[1] / 1e9

    def memory_probe():
        # Memory depends only on shapes, and the probe pads every example to MAX_TOKENS on
        # both sides, so random MAX_TOKENS-long examples are exactly the real worst case.
        m.PROBE_JSON = scratch / "memory_probe.json"
        probe.update(m.memory_probe(worst, tok))
        return (f"batch {probe['per_device_train_batch_size']} x accumulation {probe['gradient_accumulation_steps']}, "
                f"gradient checkpointing {probe['gradient_checkpointing']}, peak {probe['peak_gb']} GB of {total_gb():.1f}")

    def dev_generation_on_top_of_training():
        model = AutoModelForSeq2SeqLM.from_pretrained(m.BASE_MODEL, token=m.TOKEN).cuda()
        model.train()
        if probe["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        opt = Adafactor(model.parameters(), lr=1e-4, scale_parameter=False, relative_step=False, warmup_init=False)
        batch = m._worst_case_batch(worst.select(range(probe["per_device_train_batch_size"])), tok, model)
        labels = batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = LabelSmoother(epsilon=0.1)({"logits": model(**batch).logits.float()}, labels)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        del batch, labels, loss
        m.free_gpu()
        torch.cuda.reset_peak_memory_stats()
        model.eval()  # as in the Trainer's evaluate(): model + optimizer state stay in memory
        enc = torch.tensor([rand_ids(L, mk_id) for _ in range(m.GEN_BATCH)], device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.generate(input_ids=enc, attention_mask=torch.ones_like(enc), forced_bos_token_id=sq_id,
                           num_beams=1, do_sample=False, min_new_tokens=m.MAX_NEW_TOKENS,
                           max_new_tokens=m.MAX_NEW_TOKENS, use_cache=True)
        peak = torch.cuda.max_memory_allocated() / 1e9
        del model, opt, enc
        m.free_gpu()
        assert total_gb() - peak >= MIN_VRAM_HEADROOM_GB, f"peak {peak:.2f} GB of {total_gb():.1f} GB"
        return f"batch {m.GEN_BATCH} x {m.MAX_NEW_TOKENS} new tokens with the training state loaded: peak {peak:.2f} GB"

    def final_eval_generation():
        model = m.load_hf_model(m.BASE_MODEL)
        torch.cuda.reset_peak_memory_stats()
        enc = torch.tensor([rand_ids(L, mk_id) for _ in range(m.GEN_BATCH)], device="cuda")
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                model.generate(input_ids=enc, attention_mask=torch.ones_like(enc), forced_bos_token_id=sq_id,
                               num_beams=m.FINAL_BEAMS, do_sample=False, min_new_tokens=m.MAX_NEW_TOKENS,
                               max_new_tokens=m.MAX_NEW_TOKENS)
            return f"beam {m.FINAL_BEAMS} x batch {m.GEN_BATCH} fits: peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        except torch.cuda.OutOfMemoryError:
            return "beam 4 x batch 32 does not fit -- the evaluation will halve the batch automatically (slower, not fatal)"
        finally:
            del model, enc
            m.free_gpu()

    def ctranslate2():
        # Converting the base model is step 10's work anyway: written to the real work/ct2/base_int8/.
        if not (m.CT2_BASE / "model.bin").exists():
            base = snapshot_download(m.BASE_MODEL, token=m.TOKEN,
                                     allow_patterns=["*.json", "*.bin", "*.safetensors", "*.model"])
            m._ct2_convert(Path(base), m.CT2_BASE)
        real_ft = m.CT2_FT
        m.CT2_FT = m.CT2_BASE  # device detection runs on whichever CT2 model exists
        try:
            device, ctype = m._detect_ct2_device()
        finally:
            m.CT2_FT = real_ft
        note = "" if device == "cuda" else " -- the final evaluation's CT2 column will run on CPU (slower, not fatal)"
        return f"conversion OK, CT2 runs on {device} ({ctype}){note}"

    for name, fn in (("GPU: memory probe (worst-case 192+192-token batches)", memory_probe),
                     ("GPU: dev-eval generation on top of the training state", dev_generation_on_top_of_training),
                     ("GPU: final-eval generation (beam 4, batch 32)", final_eval_generation),
                     ("CTranslate2: base conversion + GPU run", ctranslate2)):
        if name.startswith("GPU: dev") and not results.get("GPU: memory probe (worst-case 192+192-token batches)",
                                                            {}).get("ok"):
            continue
        stage(name, fn)
    shutil.rmtree(scratch, ignore_errors=True)


def run_role(role: str) -> None:
    m = load_main()
    {"quick": role_quick, "data": role_data, "probe": role_probe, "train": role_train, "ct2": role_ct2,
     "eval_interrupt": lambda mm: role_eval(mm, True), "eval": lambda mm: role_eval(mm, False)}[role](m)


# ============================================================================
# Parent: orchestration, measurements, checks
# ============================================================================
class Smoke:
    def __init__(self, m):
        import polars as pl
        self.m, self.pl = m, pl
        self.real = m.W
        self.sandbox = m.W / "_smoke"
        self.logs = m.W / "smoke_logs"
        self.checks: list[dict] = []
        self.quick = False
        self.measure: dict = {}
        self.t0 = time.time()

    # -- output ---------------------------------------------------------------
    def say(self, msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})
        self.say(f"{'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
        return bool(ok)

    def warn(self, name: str, detail: str) -> None:
        self.checks.append({"check": name, "ok": True, "warning": True, "detail": detail})
        self.say(f"WARN  {name}  -- {detail}")

    # -- system probes ------------------------------------------------------
    @staticmethod
    def mem_available_gb() -> float | None:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 ** 2
        except OSError:
            return None
        return None

    @staticmethod
    def gpu_used_total_gb() -> tuple[float, float] | None:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            used, total = (float(x) for x in out.stdout.strip().splitlines()[0].split(","))
            return used / 1024, total / 1024
        except Exception:  # noqa: BLE001
            return None

    def run_child(self, role: str, timeout_s: int, kill_when=None) -> dict:
        self.logs.mkdir(parents=True, exist_ok=True)
        log_path = self.logs / f"{role}.log"
        self.say(f"running {role} (log: {log_path.relative_to(ROOT)})")
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n{MARKER} {time.strftime('%Y-%m-%d %H:%M:%S')} {role}\n")
            log.flush()
            p = subprocess.Popen([sys.executable, str(Path(__file__).resolve())], cwd=ROOT, stdout=log,
                                 stderr=subprocess.STDOUT, env={**os.environ, "SMOKE_ROLE": role},
                                 start_new_session=True)
            t0, last_sample = time.time(), 0.0
            min_avail, max_gpu = math.inf, 0.0
            killed = timed_out = False
            while True:
                pid, status, usage = os.wait4(p.pid, os.WNOHANG)
                if pid:
                    break
                if time.time() - last_sample >= 5:
                    last_sample = time.time()
                    avail = self.mem_available_gb()
                    if avail is not None:
                        min_avail = min(min_avail, avail)
                    gpu = self.gpu_used_total_gb()
                    if gpu:
                        max_gpu = max(max_gpu, gpu[0])
                if not killed and kill_when is not None and kill_when():
                    self.say(f"  hard-killing {role} (SIGKILL) to test recovery")
                    os.killpg(p.pid, signal.SIGKILL)
                    killed = True
                if not killed and time.time() - t0 > timeout_s:
                    self.say(f"  {role} exceeded {timeout_s // 60} min -- killing")
                    os.killpg(p.pid, signal.SIGKILL)
                    killed = timed_out = True
                time.sleep(1)
            p.returncode = os.waitstatus_to_exitcode(status)
        try:  # dataloader workers etc. must not outlive a killed child
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        res = {"role": role, "exit_code": p.returncode, "seconds": round(time.time() - t0),
               "peak_rss_gb": round(usage.ru_maxrss / 1024 ** 2, 2),  # Linux: KiB
               "min_ram_available_gb": round(min_avail, 2) if math.isfinite(min_avail) else None,
               "max_gpu_used_gb": round(max_gpu, 2), "killed": killed, "timed_out": timed_out, "log": str(log_path)}
        self.measure.setdefault("children", []).append(res)
        self.say(f"  {role}: exit {res['exit_code']}, {res['seconds']}s, peak RSS {res['peak_rss_gb']} GB, "
                 f"min free RAM {res['min_ram_available_gb']} GB, peak GPU {res['max_gpu_used_gb']} GB")
        return res

    @staticmethod
    def last_run_text(res: dict) -> str:
        """Log output of this invocation only (a role's log file is appended to across invocations)."""
        return Path(res["log"]).read_text(encoding="utf-8", errors="replace").split(MARKER)[-1]

    def tail(self, res: dict, n: int = 40) -> str:
        lines = self.last_run_text(res).splitlines()
        return "\n".join(lines[-n:])

    def child_ok(self, name: str, res: dict, expect_exit: int = 0) -> bool:
        ok = res["exit_code"] == expect_exit and not res["timed_out"]
        if not self.check(name, ok, f"exit {res['exit_code']} in {res['seconds']}s"):
            print(self.tail(res), flush=True)
        return ok

    # -- phase 0 ------------------------------------------------------------------
    def environment(self) -> bool:
        m = self.m
        other = subprocess.run(["pgrep", "-f", MAIN.name], capture_output=True, text=True).stdout.split()
        if not self.check("no real training run is active", not other, f"pids {other}" if other else ""):
            return False
        self.real.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.real).free / 1e9
        ok = self.check(f"disk space >= {MIN_DISK_GB} GB free", free >= MIN_DISK_GB, f"{free:.0f} GB free")
        avail = self.mem_available_gb()
        if avail is not None and avail < 16:
            self.warn("free RAM", f"only {avail:.1f} GB available before starting")
        ok &= self.check("nvidia-smi works", self.gpu_used_total_gb() is not None)
        if (m.CKPT).exists():
            self.warn("existing checkpoints", f"{m.CKPT} exists -- the real run will resume from it")
        return ok

    # -- phase A --------------------------------------------------------------------
    def integrity(self) -> bool:
        m, pl = self.m, self.pl
        from datasets import load_from_disk
        from transformers import AutoTokenizer
        import pyarrow.compute as pc
        P = m.PROC
        ok = True
        text_files = ["gazette_train", "gazette_dev", "gazette_test", "flores_devtest", "ntrex",
                      "verbis_terms", "verbis_defs", "train_clean"]
        for name in text_files:
            df = pl.read_parquet(P / f"{name}.parquet", columns=["mk", "sq"])
            bad = df.filter(pl.col("mk").is_null() | pl.col("sq").is_null()
                            | (pl.col("mk").str.strip_chars() == "") | (pl.col("sq").str.strip_chars() == "")).height
            ok &= self.check(f"{name}: no empty/null sides", bad == 0 and df.height > 0, f"{df.height} rows, {bad} bad")
        ok &= self.check("FLORES+ devtest has 1,012 sentences",
                         pl.read_parquet(P / "flores_devtest.parquet").height == 1012)

        held = pl.concat([pl.read_parquet(P / f"{n}.parquet", columns=["mk", "sq"])
                          for n in ("gazette_dev", "gazette_test", "flores_devtest", "ntrex")])
        train = pl.read_parquet(P / "train_clean.parquet")
        mk_hits = train.select(k=m._norm("mk")).join(held.select(k=m._norm("mk")).unique(), on="k", how="semi").height
        sq_hits = train.select(k=m._norm("sq")).join(held.select(k=m._norm("sq")).unique(), on="k", how="semi").height
        ok &= self.check("no train text appears in dev/test/FLORES+/NTREX (re-checked independently)",
                         mk_hits == 0 and sq_hits == 0, f"mk overlaps {mk_hits}, sq overlaps {sq_hits}")
        ok &= self.check("train sources are only gazette/verbis_terms/verbis_defs",
                         set(train["source"].unique()) <= {"gazette", "verbis_terms", "verbis_defs"})
        bi = pl.read_parquet(P / "train_bidirectional.parquet", columns=["src_lang"])
        by_dir = dict(bi.group_by("src_lang").len().iter_rows())
        ok &= self.check("bidirectional set = 2 x clean pairs, balanced directions",
                         bi.height == 2 * train.height and by_dir.get(m.MK) == by_dir.get(m.SQ) == train.height,
                         f"{bi.height} examples {by_dir}")

        tok = AutoTokenizer.from_pretrained(m.BASE_MODEL)
        for split in ("train", "dev"):
            ds = load_from_disk(str(m.TOKD / split))
            max_in = pc.max(pc.list_value_length(ds.data.column("input_ids"))).as_py()
            max_lab = pc.max(pc.list_value_length(ds.data.column("labels"))).as_py()
            if split == "train":
                ok &= self.check(f"tokenised train: every example <= {m.MAX_TOKENS} tokens",
                                 max_in <= m.MAX_TOKENS and max_lab <= m.MAX_TOKENS,
                                 f"max input {max_in}, max labels {max_lab}, {len(ds)} examples")
                self.measure["train_examples"] = len(ds)
                self.measure["train_tokens"] = int(pc.sum(pc.list_value_length(ds.data.column("input_ids"))).as_py()
                                                   + pc.sum(pc.list_value_length(ds.data.column("labels"))).as_py())
            try:
                m._check_lang_tokens(ds.shuffle(seed=1), tok, split)
                ok &= self.check(f"tokenised {split}: language tokens correct (random 2,000)", True)
            except AssertionError as e:
                ok &= self.check(f"tokenised {split}: language tokens correct (random 2,000)", False, str(e))
        return ok

    def phase_a(self) -> bool:
        m = self.m
        r = self.run_child("data", timeout_s=3 * 3600)
        if not self.child_ok("real steps 1-7 (preflight, private repo, test upload, data, tokenisation)", r):
            return False
        if not self.integrity():
            return False
        r2 = self.run_child("data", timeout_s=1800)
        if not self.child_ok("re-run of steps 1-7", r2):
            return False
        text = self.last_run_text(r2)
        rewrote = [ln for ln in text.splitlines() if "| wrote " in ln]
        self.check("re-run recomputes nothing (every stage 'loaded from disk')", not rewrote,
                   f"{r2['seconds']}s" + (f"; rewrote: {rewrote[:3]}" if rewrote else ""))
        r3 = self.run_child("probe", timeout_s=1800)
        if not self.child_ok("memory probe on the real longest examples", r3):
            return False
        probe = json.loads(m.PROBE_JSON.read_text())
        self.measure["memory_probe"] = probe
        gpu = self.gpu_used_total_gb()
        total = gpu[1] if gpu else 16.0
        self.check("memory probe found a batch size", True,
                   f"batch {probe['per_device_train_batch_size']} x accumulation {probe['gradient_accumulation_steps']}, "
                   f"gradient checkpointing {probe['gradient_checkpointing']}, peak {probe['peak_gb']} GB of {total:.1f}")
        return True

    # -- phase B --------------------------------------------------------------------
    def build_sandbox(self) -> int:
        m, pl = self.m, self.pl
        import numpy as np
        from datasets import load_from_disk
        S = self.sandbox
        if S.exists():
            shutil.rmtree(S)
        for d in ("data/processed", "data/tokenized", "models", "ct2", "eval"):
            (S / d).mkdir(parents=True, exist_ok=True)
        probe = json.loads(m.PROBE_JSON.read_text())
        shutil.copy(m.PROBE_JSON, S / "models" / "memory_probe.json")
        eff = probe["per_device_train_batch_size"] * probe["gradient_accumulation_steps"]
        train = load_from_disk(str(m.TOKD / "train"))
        lengths = train.data.column("length").to_numpy()
        idx = np.sort(np.argsort(-lengths, kind="stable")[:SMOKE_STEPS * eff])
        train.select(idx.tolist()).save_to_disk(str(S / "data/tokenized/train"))
        dev = load_from_disk(str(m.TOKD / "dev"))
        dev.select(range(min(512, len(dev)))).save_to_disk(str(S / "data/tokenized/dev"))
        shutil.copy(m.PROC / "gazette_dev.parquet", S / "data/processed/gazette_dev.parquet")
        for name, data_file, _ in m.TESTSETS:
            df = pl.read_parquet(m.PROC / data_file)
            df.sample(n=min(SUBSET_ROWS[name], df.height), seed=m.SEED).write_parquet(S / "data/processed" / data_file)
        self.say(f"sandbox built: {len(idx)} longest training examples (length {int(lengths[idx].min())}-"
                 f"{int(lengths[idx].max())} tokens) = {SMOKE_STEPS} steps at effective batch {eff}")
        return eff

    def phase_b(self) -> bool:
        m, pl, S = self.m, self.pl, self.sandbox
        self.build_sandbox()
        train_log, dev_curve = S / "eval/train_log.csv", S / "eval/dev_curve.csv"
        ckpt = S / "models/checkpoints"

        def killable() -> bool:
            try:
                return ((ckpt / f"checkpoint-{SMOKE_EVAL_STEPS}" / "trainer_state.json").exists()
                        and pl.read_csv(train_log)["step"].max() >= KILL_AT_STEP)
            except Exception:  # noqa: BLE001
                return False

        r = self.run_child("train", timeout_s=3600, kill_when=killable)
        if not self.check("sandbox training ran past checkpoint-20 and was hard-killed", r["killed"] and not r["timed_out"],
                          f"exit {r['exit_code']}"):
            print(self.tail(r), flush=True)
            return False

        def killable2() -> bool:
            try:
                return ((S / "fake_hub/checkpoints" / f"checkpoint-{2 * SMOKE_EVAL_STEPS}" / "trainer_state.json").exists()
                        and pl.read_csv(train_log)["step"].max() >= KILL2_AT_STEP)
            except Exception:  # noqa: BLE001
                return False

        r = self.run_child("train", timeout_s=3600, kill_when=killable2)
        if not self.check("training resumed from the local checkpoint, then hard-killed again",
                          r["killed"] and not r["timed_out"] and "resuming from" in self.last_run_text(r),
                          f"exit {r['exit_code']}"):
            print(self.tail(r), flush=True)
            return False
        # a fresh machine: no local checkpoints, logs or probe result
        shutil.rmtree(ckpt)
        for f in (train_log, dev_curve, S / "models/memory_probe.json"):
            f.unlink(missing_ok=True)
        r = self.run_child("train", timeout_s=3600)
        if not self.child_ok("training resumed from the checkpoint in the (fake) Hub and finished", r):
            return False
        text = self.last_run_text(r)
        self.check("hub resume downloaded checkpoint-40 and its logs",
                   f"downloading checkpoint-{2 * SMOKE_EVAL_STEPS}" in text and "downloaded eval/train_log.csv" in text)
        probe = json.loads((S / "models/memory_probe.json").read_text())
        self.check("hub resume kept the checkpoint's batch layout", "source" in probe,
                   f"batch {probe['per_device_train_batch_size']} x {probe['gradient_accumulation_steps']}")
        steps = pl.read_csv(train_log)["step"].to_list()
        self.check("train_log.csv: continuous across two kills and a hub resume, no duplicate steps",
                   steps == list(range(5, SMOKE_STEPS + 1, 5)), f"{len(steps)} rows, steps {steps[0]}..{steps[-1]}")
        dev_steps = pl.read_csv(dev_curve)["step"].to_list()
        self.check("dev_curve.csv: one row per evaluation across the kills",
                   dev_steps == list(range(SMOKE_EVAL_STEPS, SMOKE_STEPS + 1, SMOKE_EVAL_STEPS)), str(dev_steps))
        state = json.loads((ckpt / f"checkpoint-{SMOKE_STEPS}" / "trainer_state.json").read_text())
        best_name = Path(state.get("best_model_checkpoint") or "").name
        self.check("best-checkpoint tracking survived (trainer_state.json)",
                   bool(best_name) and state.get("best_metric") is not None,
                   f"best {state.get('best_metric')} at {best_name}")
        hub_ckpts = sorted((p.name for p in (S / "fake_hub/checkpoints").iterdir()), key=lambda n: int(n.split("-")[1]))
        want = sorted({f"checkpoint-{s}" for s in range(SMOKE_STEPS - 2 * SMOKE_EVAL_STEPS, SMOKE_STEPS + 1,
                                                          SMOKE_EVAL_STEPS)} | {best_name},
                      key=lambda n: int(n.split("-")[1]))
        self.check("repo keeps exactly the last 3 + best checkpoints", hub_ckpts == want, f"repo has {hub_ckpts}")
        files = {p.name for p in (S / "fake_hub/checkpoints" / f"checkpoint-{SMOKE_STEPS}").iterdir()}
        need = {"model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin"}
        self.check("uploaded checkpoint is complete (weights, optimizer, scheduler, rng, state, args, tokenizer)",
                   need <= files and any(f.startswith("rng_state") for f in files) and "tokenizer.json" in files,
                   f"missing {sorted(need - files)}" if need - files else f"{len(files)} files")
        self.check("best model saved to models/final", (S / "models/final/model.safetensors").exists())
        self.check("training_curves.png written", (S / "eval/training_curves.png").exists())
        tl = pl.read_csv(train_log)
        peak = tl["gpu_mem_peak_gb"].max()
        total = (self.gpu_used_total_gb() or (0, 16.0))[1]
        self.check("GPU memory headroom during worst-case training + dev generation",
                   total - peak >= MIN_VRAM_HEADROOM_GB, f"peak allocated {peak} GB of {total:.1f} GB")
        self.measure["smoke_tokens_per_sec"] = float(tl["tokens_per_sec"].slice(2).median())

        r = self.run_child("ct2", timeout_s=3600)
        if not self.child_ok("CTranslate2 conversion (fine-tuned + base) and smoke test", r):
            return False
        info = json.loads((S / "eval/run_info.json").read_text())
        self.measure["ct2_device"] = f"{info.get('ct2_device')} ({info.get('ct2_compute_type')})"
        self.check("CTranslate2 device chosen", info.get("ct2_device") in ("cuda", "cpu"), self.measure["ct2_device"])
        self.measure["ct2_seconds"] = r["seconds"]

        r = self.run_child("eval_interrupt", timeout_s=3600)
        if not self.child_ok("final evaluation hard-killed after a partial save", r, expect_exit=9):
            return False
        part = pl.read_parquet(S / "eval/translations_gazette_test.parquet")
        n_done = part["base"].is_not_null().sum()
        self.check("partial translations were saved before the kill", 0 < n_done < part.height,
                   f"{n_done}/{part.height} base translations cached")
        r = self.run_child("eval", timeout_s=5400)
        if not self.child_ok("final evaluation resumed and finished", r):
            return False
        text = self.last_run_text(r)
        self.check("evaluation resumed from cached translations", "loaded from disk: eval/translations_gazette_test" in text)
        cols = ["model", "direction", "n_sentences", "bleu", "chrf", "bleu_delta_vs_base", "chrf_delta_vs_base",
                "bleu_signature", "chrf_signature", "timestamp"]
        for name, data_file, res_file in m.TESTSETS:
            res = pl.read_csv(S / "eval" / res_file)
            tr = pl.read_parquet(S / "eval" / f"translations_{name}.parquet")
            nulls = sum(tr[c].null_count() for c in m.MODEL_COLS)
            want_n = min(SUBSET_ROWS[name], pl.read_parquet(m.PROC / data_file).height)
            self.check(f"{res_file}: 6 rows, right columns, every sentence scored",
                       res.columns == cols and res.height == 6 and set(res["n_sentences"]) == {want_n} and nulls == 0,
                       f"{res.height} rows, n={sorted(set(res['n_sentences']))}, {nulls} missing translations")
        self.check("DONE written", (S / "DONE").exists())

        calls = [json.loads(x) for x in (S / "upload_calls.jsonl").read_text().splitlines()]
        remotes = {c["remote"] for c in calls}
        forbidden = [c for c in calls if c["forbidden"]]
        self.check("upload rules: only the allowed repo layout; from data/ only processed Verbis; no tokenized data",
                   not forbidden,
                   f"{len(forbidden)} violations" + (f", e.g. {forbidden[0]}" if forbidden else ""))
        expected = {"config.json", "model.safetensors", "ct2/finetuned_int8/model.bin", "ct2/base_int8/model.bin",
                    f"checkpoints/checkpoint-{SMOKE_STEPS}/trainer_state.json",
                    "eval/gazette_test_results.csv", "eval/flores_results.csv", "eval/ntrex_results.csv",
                    "eval/train_log.csv", "eval/dev_curve.csv", "eval/training_curves.png", "eval/run_info.json",
                    "progress.log", "DONE"}
        self.check("upload set complete (model at repo root, checkpoints/, ct2/, eval/, progress.log, DONE)",
                   expected <= remotes and "training_args.bin" not in remotes,
                   f"missing {sorted(expected - remotes)}" if expected - remotes else f"{len(remotes)} files")
        self.measure["engine_timing"] = self.engine_rates()
        return True

    def engine_rates(self) -> dict:
        rates: dict = {}
        for line in (self.sandbox / "engine_timing.jsonl").read_text().splitlines():
            r = json.loads(line)
            n, s = rates.get(r["col"], (0, 0.0))
            rates[r["col"]] = (n + r["n"], s + r["secs"])
        return {c: round(n / s, 2) for c, (n, s) in rates.items() if s > 0}

    # -- projection -------------------------------------------------------------------
    def projection(self) -> None:
        m, pl, meas = self.m, self.pl, self.measure
        probe = meas["memory_probe"]
        eff = probe["per_device_train_batch_size"] * probe["gradient_accumulation_steps"]
        steps = math.ceil(meas["train_examples"] / eff)
        train_h = meas["train_tokens"] / meas["smoke_tokens_per_sec"] / 3600
        dev_secs = []
        for line in (self.sandbox / "progress.log").read_text(encoding="utf-8").splitlines():
            if "dev @ step" in line and line.rstrip().endswith("s"):
                try:
                    dev_secs.append(float(line.rsplit("|", 1)[1].strip().rstrip("s")))
                except ValueError:
                    pass
        per_dev_eval = (sorted(dev_secs)[len(dev_secs) // 2] if dev_secs else 60) * (m.DEV_EVAL_SIZE / 32)
        dev_h = (steps // m.EVAL_STEPS + 1) * per_dev_eval / 3600
        n_eval = 2 * sum(pl.read_parquet(m.PROC / f).height for _, f, _ in m.TESTSETS)
        eval_h = sum(n_eval / rate / 3600 for rate in meas["engine_timing"].values())
        total = train_h + dev_h + eval_h + meas.get("ct2_seconds", 600) / 3600
        meas["projection"] = {"optimizer_steps": steps, "training_h": round(train_h, 2), "dev_evals_h": round(dev_h, 2),
                              "final_eval_sentences": n_eval, "final_eval_h": round(eval_h, 2),
                              "total_after_data_steps_h": round(total, 2)}
        self.say(f"projected real run (data steps already done): training ~{train_h:.1f} h ({steps} steps), "
                 f"dev evaluations ~{dev_h:.1f} h, final evaluation ~{eval_h:.1f} h "
                 f"({n_eval} translations x 3 systems) -> total ~{total:.1f} h")
        self.say("  (training time is extrapolated from tokens/s on the longest examples; real batches are "
                 "shorter, so this is closer to an upper bound)")
        if total > 8:
            self.warn("projected duration", f"~{total:.1f} h, longer than 8 h")

    # -- run ----------------------------------------------------------------------------
    def finish(self, completed: bool) -> None:
        mins = [c["min_ram_available_gb"] for c in self.measure.get("children", []) if c["min_ram_available_gb"]]
        if mins:
            self.check(f"free RAM never dropped below {MIN_RAM_AVAILABLE_GB} GB", min(mins) >= MIN_RAM_AVAILABLE_GB,
                       f"minimum {min(mins):.1f} GB available; peak child RSS "
                       f"{max(c['peak_rss_gb'] for c in self.measure['children'])} GB")
        failed = [c for c in self.checks if not c["ok"]]
        passed = completed and not failed
        report = {"passed": passed, "minutes": round((time.time() - self.t0) / 60, 1),
                  "checks": self.checks, "measurements": self.measure}
        out = self.real / ("quick_report.json" if self.quick else "smoke_report.json")
        out.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print("\n" + "=" * 78)
        if passed:
            shutil.rmtree(self.sandbox, ignore_errors=True)
            if self.quick:
                self.say("Not covered by --quick: data preparation (first ~15 min of the real run; resumes from disk)")
                self.say("and kill/resume (the full smoke test).")
            self.say(f"ALL {len(self.checks)} CHECKS PASSED in {report['minutes']} min. Sandbox removed.")
            self.say("Start the real run:" if self.quick else "Start the real run (steps 1-7 will load from disk):")
            self.say("  nohup .venv/bin/python scripts/train_nllb_gazette_verbis_ct2.py > work/nohup.out 2>&1 &")
        else:
            self.say(f"SMOKE TEST FAILED ({len(failed)} failed checks{'' if completed else ', stopped early'}):")
            for c in failed:
                self.say(f"  - {c['check']}: {c['detail']}")
            self.say(f"Sandbox kept for inspection: {self.sandbox}")
        self.say(f"report: {out}   child logs: {self.logs}")
        sys.exit(0 if passed else 1)

    def run_quick(self) -> None:
        self.say(f"QUICK preflight for {MAIN.name}; work dir {self.real}")
        if not self.environment():
            self.finish(False)
        result_path = self.real / "quick_result.json"
        result_path.unlink(missing_ok=True)
        r = self.run_child("quick", timeout_s=1800)
        results = json.loads(result_path.read_text()) if result_path.exists() else {}
        for name, v in results.items():
            self.check(name, v["ok"], f"{v['secs']}s | {v['detail']}")
        ok = self.child_ok("quick checks process", r) and bool(results)
        self.finish(ok)

    def run(self) -> None:
        self.say(f"smoke test for {MAIN.name}; work dir {self.real}")
        if not self.environment():
            self.finish(False)
        self.say("---- phase A: real steps 1-7 on the full data ----")
        if not self.phase_a():
            self.finish(False)
        self.say("---- phase B: sandbox training / kill / resume / CT2 / evaluation ----")
        if not self.phase_b():
            self.finish(False)
        self.projection()
        self.finish(True)


def main() -> None:
    try:
        m = load_main()
    except ImportError as e:
        print(f"ERROR: {e}\nInstall the dependencies with:\n  uv pip install -r scripts/requirements-train.txt",
              flush=True)
        sys.exit(1)
    if sys.platform != "linux":
        print("ERROR: run this on the Linux training machine (it needs the GPU and /proc).", flush=True)
        sys.exit(1)
    smoke = Smoke(m)
    if "--quick" in sys.argv[1:]:
        smoke.quick = True
        smoke.sandbox = m.W / "_quick"
        smoke.run_quick()
    smoke.run()


if __name__ == "__main__":
    if ROLE:
        run_role(ROLE)
    else:
        main()
