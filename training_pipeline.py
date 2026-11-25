"""Orchestrate the TranslateGemma runs: CSV -> configs, plan, run, decide, push.

    python training_pipeline.py configs [--check]        # tg_hyperparameters.csv -> src/configs/TG-*.yaml
    python training_pipeline.py plan                     # order, status, parents, repos, private budget
    python training_pipeline.py run TG-10 [--wandb online|offline|disabled] [-- <train.py args>]
    python training_pipeline.py run --next | --all [--priority 1]
    python training_pipeline.py decide TG-35-s1 go|skip  # outcome of an "only if ..." condition
    python training_pipeline.py select-endpoint TG-10 TG-10-CD10
    python training_pipeline.py push TG-20 [--dry-run] [--replace] [--no-weights]
    python training_pipeline.py status
    python training_pipeline.py prune TG-10              # delete local checkpoints once children are done

State lives in runs/pipeline_state.json (gitignored). Each run trains in a fresh subprocess
(``python -m src.pipeline.train <cfg>``), which writes runs/<id>/status.json, model/ and metrics/.
HF layout, repo choice and the private budget guard: src/pipeline/hub.py.
Live W&B tracking (one run per training run, grouped by stage; HF pushes logged as reference
artifacts): src/pipeline/tracking.py.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from src.pipeline import hub, tracking
from src.pipeline.config import ConfigError, RunConfig, load_config

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "src" / "configs"
CSV_PATH = CONFIG_DIR / "tg_hyperparameters.csv"

#: CSV columns copied verbatim into ``plan:`` (descriptive; validation does not interpret them).
PLAN_COLUMNS = (
    "plan_order",
    "stage",
    "priority",
    "run_condition",
    "init_checkpoint",
    "train_data",
    "est_train_hours_at_500_tok_s",
    "checkpoint_selection",
    "save_at_end",
    "decoded_eval_points",
    "retention_check",
    "notes",
)

HEADER = """\
# GENERATED from src/configs/tg_hyperparameters.csv by `python training_pipeline.py configs`.
# Edit the CSV and regenerate; only the `data:` section is hand-edited (it is kept on regeneration).
# The CSV wins over the SFT summary where they differ: max_grad_norm 1.0 (summary 0); lr peak {peak}
# (summary 1e-5 or 5e-5); one GB10 device without FSDP (summary FSDP2 on 2 GPUs).
"""


# ── data templates for runs whose file does not exist yet ─────────────────────


def _lex(name: str, loader: str, task: str, select: dict[str, Any], **kw: Any) -> dict[str, Any]:
    entry = {
        "name": name,
        "loader": loader,
        "split": "train",
        "select": dict(select),
        "task": task,
        "directions": "both",
        **kw,
    }
    if not select:
        del entry["select"]
    return entry


_HOLDOUT = {"exclude": "V_HOLDOUT"}
#: ``run_id -> (comment or None, data dict)``. Runs not listed copy another run's data (``COPY_DATA``).
DATA_TEMPLATES: dict[str, tuple[str | None, dict[str, Any]]] = {
    "TG-21": (
        None,
        {
            "mixture": [
                {
                    "name": "L_MAIN",
                    "loader": "gazette",
                    "split": "train",
                    "select": {"slice": "L_MAIN"},
                    "task": "translation",
                    "directions": "both",
                },
            ]
        },
    ),
    "TG-34-s1": (
        "V_DEF_PAR: definition pairs, LaBSE >= 0.70; TODO tokens_m (~1.7 passes, ~8.4M tokens)",
        {
            "mixture": [
                _lex(
                    "V_DEF_PAR/verbis",
                    "verbis",
                    "definition_translation",
                    {**_HOLDOUT, "def_labse_floor": 0.70},
                ),
                _lex(
                    "V_DEF_PAR/media_literacy",
                    "media_literacy",
                    "definition_translation",
                    {**_HOLDOUT, "def_labse_floor": 0.70},
                ),
            ]
        },
    ),
    "TG-35-s1": (
        "V_DEF_SHUF: TG-33's examples with definitions swapped within a domain",
        {
            "mixture": [
                _lex("V_DEF_SHUF/verbis", "verbis", "shuffled_definition", _HOLDOUT, repeats=3),
                _lex(
                    "V_DEF_SHUF/media_literacy", "media_literacy", "shuffled_definition", _HOLDOUT, repeats=3
                ),
            ]
        },
    ),
    "TG-36-s1": (
        "V_TERM + W_GEN (MUSE pairs + Goethe phrases), 3x per direction",
        {
            "mixture": [
                _lex("V_TERM/verbis", "verbis", "term_pair", _HOLDOUT, repeats=3),
                _lex("V_TERM/media_literacy", "media_literacy", "term_pair", _HOLDOUT, repeats=3),
                {
                    "name": "W_GEN/muse",
                    "loader": "muse",
                    "split": "train",
                    "task": "term_pair",
                    "directions": "both",
                    "repeats": 3,
                },
                {
                    "name": "W_GEN/phrasebook",
                    "loader": "phrasebook",
                    "split": "train",
                    "task": "translation",
                    "directions": "both",
                    "repeats": 3,
                },
            ]
        },
    ),
    "TG-90": (
        "TODO provisional: the recipe follows the core results (task format, W_GEN, replay size)",
        {
            "mixture": [
                {
                    "name": "L_ARMS",
                    "loader": "gazette",
                    "split": "train",
                    "select": {"slice": "L_ARMS"},
                    "task": "translation",
                    "directions": "both",
                },
                {
                    "name": "L_FRESH",
                    "loader": "gazette",
                    "split": "train",
                    "select": {"slice": "L_FRESH"},
                    "task": "translation",
                    "directions": "both",
                },
                _lex("V_ALL/verbis", "verbis", "term_definition", {}, repeats=3),
                _lex("V_ALL/media_literacy", "media_literacy", "term_definition", {}, repeats=3),
                {
                    "name": "G_REPLAY",
                    "loader": "opus",
                    "split": "train",
                    "select": {"tag": "G_REPLAY"},
                    "task": "translation",
                    "directions": "alternating",
                },
            ]
        },
    ),
}
COPY_NOTES = {
    "TG-10-EXT": "TODO: rest of G from the cooldown start + next-ranked NLLB pairs (LaBSE >= 0.80), ~2.2M "
    "examples in total",
}


def copy_source(run_id: str) -> str | None:
    if run_id.startswith("TG-10-"):
        return "TG-10"
    m = re.fullmatch(r"(.+)-s[2-9]", run_id)
    return f"{m.group(1)}-s1" if m else None


# ── CSV -> YAML ───────────────────────────────────────────────────────────────


def read_csv() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return sorted(rows, key=lambda r: int(r["plan_order"]))


def _num(s: str, cast: type = float) -> Any:
    s = s.strip()
    return None if s == "" else cast(float(s)) if cast is int else cast(s)


def _flag(s: str) -> bool:
    return s.strip().lower().startswith("true")


def _schedule_type(s: str) -> str:
    s = s.strip().lower()
    if s.startswith("wsd"):
        return "wsd"
    if s.startswith("cosine"):
        return "cosine"
    if "cooldown only" in s:
        return "cooldown"
    raise ValueError(f"unknown lr_schedule {s!r}")


def _eps(s: str) -> list[float]:
    return [float(x) for x in s.strip("() ").split(",")]


def training_from_row(row: dict[str, str], rows: list[dict[str, str]]) -> dict[str, Any]:
    rid = row["run_id"]
    # Kept (never-rotated) checkpoints: wherever a child branches with optimizer state.
    keep = sorted(
        int(r["start_global_step"])
        for r in rows
        if r["init_from"] == rid and _flag(r["resume_optimizer_state"])
    )
    resumes = _flag(row["resume_optimizer_state"])
    start = int(row["start_global_step"] or 0)
    out: dict[str, Any] = {
        "max_steps": int(row["planned_updates"]),
        "micro_batch_rows": int(row["micro_batch_rows"]),
        "micro_batch_tokens": int(row["micro_batch_tokens"]),
        "grad_accum": int(row["grad_accum_steps"]),
        "tokens_per_step": int(row["tokens_per_update"]),
        "optimizer": {
            "name": "adafactor",
            "beta1": None
            if row["adafactor_beta1"].strip().lower() in ("none", "")
            else float(row["adafactor_beta1"]),
            "decay_rate": float(row["adafactor_decay_rate"]),
            "clip_threshold": float(row["adafactor_clip_threshold"]),
            "eps": _eps(row["adafactor_eps"]),
            "scale_parameter": _flag(row["adafactor_scale_parameter"]),
            "relative_step": _flag(row["adafactor_relative_step"]),
            "warmup_init": _flag(row["adafactor_warmup_init"]),
            "weight_decay": float(row["weight_decay"]),
        },
        "max_grad_norm": float(row["max_grad_norm"]),
        "lr_schedule": {
            "type": _schedule_type(row["lr_schedule"]),
            "peak": _num(row["lr_peak"]),
            "min": _num(row["lr_min"]),
            "warmup_start": _num(row["lr_warmup_start"]),
            "warmup_steps": _num(row["warmup_steps"], int),
            "constant_end_step": _num(row["constant_end_step"], int),
            "cooldown_steps": _num(row["cooldown_steps"], int),
        },
        "dev_loss_every_steps": _num(row["dev_loss_every_steps"], int),
        "save": {
            "every_steps": _num(row["resumable_checkpoint_every_steps"], int),
            "keep_steps": keep,
            "with_optimizer": bool(keep),
        },
        "resume": {
            "from_run": row["init_from"] if resumes else None,
            "from_step": start if resumes else None,
            "optimizer_state": resumes,
            "start_global_step": start,
        },
        "gradient_checkpointing": _flag(row["gradient_checkpointing"]),
        "fused_cross_entropy": _flag(row["fused_cross_entropy"]),
        "bf16": row["compute_dtype"].strip() == "bf16",
    }
    if rid.startswith("TG-10"):  # TG-10's eval list is FLORES_DEV only
        out["eval"] = {"select_on": ["FLORES_DEV"], "metric": "chrf++"}
    return out


def extract_data_block(text: str) -> str | None:
    """The ``data:`` section of a YAML file, verbatim (up to the next top-level key)."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if re.match(r"data:\s*(#.*)?$", ln)), None)
    if start is None:
        return None
    end = next((j for j in range(start + 1, len(lines)) if re.match(r"[A-Za-z_]", lines[j])), len(lines))
    block = lines[start:end]
    while block and not block[-1].strip():
        block.pop()
    return "\n".join(block) + "\n"


def _dump(obj: dict[str, Any], flow: bool | None = False, width: int = 110) -> str:
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True, default_flow_style=flow, width=width)


def _dump_data(data: dict[str, Any]) -> str:
    """A template ``data:`` section, one flow-style line per entry (the hand-written files' style)."""
    lines = ["data:"]
    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"  {key}:")
            lines += [f"    - {_dump(e, flow=True, width=10**6).strip()}" for e in value]
        else:
            lines.append(f"  {key}: {_dump(value, flow=True).strip()}")
    return "\n".join(lines) + "\n"


def data_block(run_id: str, existing: dict[str, str]) -> str:
    """Existing file's ``data:`` > copy of the family source (``-s1`` / TG-10) > template."""
    if run_id in existing and (block := extract_data_block(existing[run_id])):
        return block
    src = copy_source(run_id)
    if src is not None:
        block = data_block(src, existing)
        if run_id in COPY_NOTES:
            first, rest = block.split("\n", 1)
            block = f"{first}\n  # {COPY_NOTES[run_id]}\n{rest}"
        return block
    if run_id not in DATA_TEMPLATES:
        raise KeyError(f"{run_id}: no existing file and no data template")
    comment, data = DATA_TEMPLATES[run_id]
    block = _dump_data(data)
    if comment:
        first, rest = block.split("\n", 1)
        block = f"{first}\n  # {comment}\n{rest}"
    return block


def _description(run_id: str, existing: dict[str, str], row: dict[str, str]) -> str:
    for rid in (run_id, copy_source(run_id) if re.search(r"-s[2-9]$", run_id) else None):
        if rid and rid in existing:
            desc = (yaml.safe_load(existing[rid]) or {}).get("description")
            if desc:
                return desc
    return row["train_data"]


def render(row: dict[str, str], rows: list[dict[str, str]], existing: dict[str, str]) -> str:
    rid = row["run_id"]
    training = training_from_row(row, rows)
    init = hub.BASE_MODEL if row["init_from"] == "TG-00" else f"run:{row['init_from']}"
    head = {
        "extends": "_arms.yaml" if rid.startswith("TG-3") else "_base.yaml",
        "run_id": rid,
        "group": re.sub(r"-(s\d+|CD\d+|EXT)$", "", rid),
        "description": _description(rid, existing, row),
        "seed": int(row["seed"]),
    }
    parts = [
        HEADER.format(peak=training["lr_schedule"]["peak"]),
        _dump(head),
        _dump({"model": {"init_from": init}}),
        data_block(rid, existing),
        _dump({"training": training}, flow=None, width=10**6),
        _dump({"output": {"local_dir": f"runs/{rid}", "repo": "auto", "push_weights": True}}),
        _dump({"plan": {c: row[c] for c in PLAN_COLUMNS}}),
    ]
    return "\n".join(p.rstrip("\n") + "\n" for p in parts)


def generate() -> dict[Path, str]:
    rows = read_csv()
    existing = {p.stem: p.read_text() for p in CONFIG_DIR.glob("TG-*.yaml")}
    return {CONFIG_DIR / f"{r['run_id']}.yaml": render(r, rows, existing) for r in rows}


def cmd_configs(args: argparse.Namespace) -> int:
    out = generate()
    problems = 0
    stray = sorted(set(CONFIG_DIR.glob("TG-*.yaml")) - set(out))
    for p in stray:
        print(f"not in the CSV: {p.name}")
        problems += 1
    for path, text in out.items():
        old = path.read_text() if path.exists() else ""
        if old == text:
            continue
        if args.check:
            problems += 1
            diff = list(
                difflib.unified_diff(old.splitlines(), text.splitlines(), path.name, "generated", lineterm="")
            )
            print(f"--- {path.name}: {'missing' if not old else 'differs'}")
            print("\n".join(diff[:40]) + ("\n  ..." if len(diff) > 40 else ""))
        else:
            path.write_text(text)
            print(f"wrote {path.relative_to(ROOT)}")
    for path in out:
        if path.exists():
            try:
                load_config(path)
            except ConfigError as e:
                problems += 1
                print(f"INVALID {path.name}: {e}")
    print(f"{len(out)} configs, {problems} problem(s)" + (" (check only)" if args.check else ""))
    return 1 if problems else 0


# ── orchestration ─────────────────────────────────────────────────────────────


def load_all() -> dict[str, RunConfig]:
    return {r["run_id"]: load_config(CONFIG_DIR / f"{r['run_id']}.yaml") for r in read_csv()}


def is_conditional(cfg: RunConfig) -> bool:
    return cfg.plan.get("run_condition", "").strip().lower().startswith("only if")


def needs_endpoint(configs: dict[str, RunConfig]) -> set[str]:
    """Runs with size-check branches: their non-branch children wait for ``select-endpoint``."""
    return {c.training.resume["from_run"] for c in configs.values() if c.training.resume["from_run"]}


def blockers(
    cfg: RunConfig, configs: dict[str, RunConfig], state: dict[str, Any], explicit: bool
) -> list[str]:
    reasons = []
    r = state["runs"].get(cfg.run_id, {})
    if r.get("decision") == "skip":
        reasons.append("decided skip")
    elif is_conditional(cfg) and r.get("decision") != "go" and not explicit:
        reasons.append(f"conditional, undecided ({cfg.plan['run_condition']})")
    parent = hub.parent_id(cfg)
    if parent is None:
        return reasons
    resume = cfg.training.resume
    if resume["from_run"]:
        if not hub.keep_dir(resume["from_run"], int(resume["from_step"])).exists():
            reasons.append(f"needs {resume['from_run']} kept checkpoint at step {resume['from_step']}")
        return reasons
    if parent in needs_endpoint(configs) and parent not in state["endpoints"]:
        reasons.append(f"run `select-endpoint {parent} <run>` first")
        return reasons
    target = state["endpoints"].get(parent, parent)
    if state["runs"].get(target, {}).get("status") != "done":
        reasons.append(f"parent {target} not done")
    else:
        try:
            hub.resolve_init(cfg, state)
        except hub.HubError as e:
            reasons.append(str(e))
    return reasons


def _priority_ok(cfg: RunConfig, max_priority: int | None) -> bool:
    if max_priority is None:
        return True
    m = re.match(r"\d+", cfg.plan.get("priority", ""))
    return bool(m) and int(m.group()) <= max_priority


def next_ready(configs: dict[str, RunConfig], state: dict[str, Any], max_priority: int | None) -> str | None:
    for rid, cfg in configs.items():
        st = state["runs"].get(rid, {}).get("status")
        if st == "done" or not _priority_ok(cfg, max_priority):
            continue
        if not blockers(cfg, configs, state, explicit=False):
            return rid
    return None


def gpu_busy() -> list[str]:
    if shutil.which("nvidia-smi") is None:
        return []
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def evaluate(cfg: RunConfig) -> None:
    """Decoded evaluation (vLLM) hook. TODO: no eval module yet; dev loss is logged during training.

    Whatever scores exist (``runs/<id>/eval/<point>/scores.json``: BLEU, chrF++, COMET per set and
    direction) are sent to the run's W&B run.
    """
    print(f"WARNING evaluate({cfg.run_id}) decoding skipped: no decoded-eval module yet (dev loss is in metrics/)")
    status = hub._read_json(hub.run_dir(cfg) / "status.json")
    for scores in sorted((hub.run_dir(cfg) / "eval").glob("*/scores.json")):
        point = scores.parent.name
        if tracking.log_eval(cfg, hub.run_dir(cfg), point, hub._read_json(scores), status.get("step")):
            print(f"  wandb: logged eval/{point}")


def track_push(cfg: RunConfig, state: dict[str, Any]) -> None:
    """Reference the just-pushed HF folder + commit from the run's W&B run (never fails the push)."""
    r = state["runs"].get(cfg.run_id, {})
    if r.get("commit") and tracking.log_checkpoint(cfg, hub.run_dir(cfg), r):
        print(f"  wandb: artifact {cfg.run_id}-hf -> {tracking.hf_url(r['repo'], r['commit'], r.get('folder') or cfg.run_id)}")


def preflight(
    cfg: RunConfig, configs: dict[str, RunConfig], state: dict[str, Any], args: argparse.Namespace
) -> None:
    problems = blockers(cfg, configs, state, explicit=True)
    if problems:
        raise hub.HubError(f"{cfg.run_id} not ready: " + "; ".join(problems))
    if is_conditional(cfg) and state["runs"].get(cfg.run_id, {}).get("decision") != "go":
        print(f"WARNING {cfg.run_id} is conditional and undecided; running because it was named explicitly")
    repo = hub.assign_repo(cfg, configs, state)
    print(f"{cfg.run_id}: init {hub.resolve_init(cfg, state) if hub.parent_id(cfg) else cfg.model.init_from}")
    print(f"{cfg.run_id}: will push to {repo}/{hub.folder_of(cfg)}/")
    if not args.no_push and cfg.output.push != "never":
        from huggingface_hub import HfApi

        api = HfApi()
        print(f"HF user: {hub.check_auth(api)}")
        if repo == hub.PRIVATE_REPO and cfg.output.push_weights:
            used = sum(hub.repo_files(api, repo).values())
            if not hub.fits_budget(used, hub.BF16_MODEL_BYTES_EST):
                print(f"WARNING private repo at {hub.gb(used)}: weights will not fit the budget")
    if not args.skip_download:
        from src.pipeline import registry

        for e in (*cfg.data.shared, *cfg.data.mixture, *cfg.data.eval):
            path = registry.get_loader(e.loader)(**e.options).download(e.split)
            print(f"  data ready: {e.name or e.loader} ({e.loader}/{e.split}) -> {path}")
    busy = gpu_busy()
    if busy:
        raise hub.HubError(f"GPU busy (no other GPU job during training): {busy}")


def run_one(rid: str, configs: dict[str, RunConfig], state: dict[str, Any], args: argparse.Namespace) -> bool:
    cfg = configs[rid]
    preflight(cfg, configs, state, args)
    cmd = [sys.executable, "-m", "src.pipeline.train", str(CONFIG_DIR / f"{rid}.yaml"), *args.train_args]
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    if args.wandb:
        env["WANDB_MODE"] = args.wandb
    print("$ " + " ".join(cmd))
    if args.dry_run:
        return True
    r = hub.run_state(state, rid)
    r.update(status="running")
    hub.save_state(state)
    code = subprocess.run(cmd, check=False, cwd=ROOT, env=env).returncode
    status_file = hub.run_dir(cfg) / "status.json"
    status = hub._read_json(status_file)
    r.update(
        status=status.get("status", "failed"),
        step=status.get("step"),
        tokens=status.get("tokens"),
        exit_code=code,
    )
    if code != 0 and r["status"] == "done":
        r["status"] = "failed"
    hub.save_state(state)
    if r["status"] != "done":
        print(f"{rid} FAILED (exit {code}, {status_file} = {status or 'missing'})")
        return False
    print(f"{rid} done: step {r['step']}, tokens {r['tokens']}")
    wb = tracking.read_info(hub.run_dir(cfg))
    if wb.get("url") or wb.get("id"):
        r["wandb"] = wb.get("url") or f"offline:{wb['id']}"
    evaluate(cfg)
    if not args.no_push and cfg.output.push != "never":
        from huggingface_hub import HfApi

        hub.push(cfg, configs, state, HfApi())
        hub.save_state(state)
        track_push(cfg, state)
    return True


def cmd_run(args: argparse.Namespace) -> int:
    configs, state = load_all(), hub.load_state()
    if args.run:
        if args.run not in configs:
            raise SystemExit(f"unknown run {args.run}")
        return 0 if run_one(args.run, configs, state, args) else 1
    while (rid := next_ready(configs, state, args.priority)) is not None:
        if not run_one(rid, configs, state, args):
            return 1
        if args.next or args.dry_run:
            return 0
    print("nothing ready to run" + ("" if args.next else " (all remaining runs are done or blocked)"))
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    configs, state = load_all(), hub.load_state()
    if args.run not in configs:
        raise SystemExit(f"unknown run {args.run}")
    if not is_conditional(configs[args.run]):
        print(f"note: {args.run} has no 'only if' condition ({configs[args.run].plan.get('run_condition')})")
    hub.run_state(state, args.run)["decision"] = args.decision
    hub.save_state(state)
    print(f"{args.run}: {args.decision}")
    return 0


def cmd_select_endpoint(args: argparse.Namespace) -> int:
    configs, state = load_all(), hub.load_state()
    branches = {c.run_id for c in configs.values() if c.training.resume["from_run"] == args.run}
    if args.endpoint not in {args.run, *branches}:
        raise SystemExit(f"{args.endpoint} is not {args.run} or one of its branches {sorted(branches)}")
    if state["runs"].get(args.endpoint, {}).get("status") != "done":
        raise SystemExit(f"{args.endpoint} is not done")
    state["endpoints"][args.run] = args.endpoint
    hub.save_state(state)
    print(f"run:{args.run} now resolves to {args.endpoint}")
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    from huggingface_hub import HfApi

    configs, state = load_all(), hub.load_state()
    if args.run not in configs:
        raise SystemExit(f"unknown run {args.run}")
    weights = False if args.no_weights else None
    hub.push(
        configs[args.run],
        configs,
        state,
        HfApi(),
        replace=args.replace,
        dry_run=args.dry_run,
        weights=weights,
    )
    if not args.dry_run:
        hub.save_state(state)
        track_push(configs[args.run], state)
    return 0


def _weight_bytes(cfg: RunConfig, r: dict[str, Any]) -> tuple[int, str]:
    if r.get("weights_pushed"):
        return r.get("bytes", 0), "pushed"
    local = hub.dir_bytes(hub.run_dir(cfg) / "model")
    if local:
        return local, "local"
    return (hub.BF16_MODEL_BYTES_EST, "est") if cfg.output.push_weights else (0, "-")


def cmd_plan(args: argparse.Namespace) -> int:
    configs, state = load_all(), hub.load_state()
    fmt = "{:>3}  {:<11} {:<8} {:<9} {:<11} {:>6}  {:<8} {:>9}  {}"
    print(fmt.format("#", "run", "status", "decision", "parent", "est h", "repo", "weights", "condition"))
    pushed_private = planned_private = 0
    n_private = 0
    for rid, cfg in configs.items():
        r = state["runs"].get(rid, {})
        parent = hub.parent_id(cfg) or "-"
        if parent in state["endpoints"] and not cfg.training.resume["from_run"]:
            parent = f"{parent}>{state['endpoints'][parent]}"
        repo = hub.assign_repo(cfg, configs, state)
        nbytes, how = _weight_bytes(cfg, r)
        if repo == hub.PRIVATE_REPO:
            n_private += 1
            planned_private += nbytes
            pushed_private += r.get("bytes", 0) if r.get("repo") == hub.PRIVATE_REPO else 0
        cond = cfg.plan.get("run_condition", "")
        print(
            fmt.format(
                cfg.plan.get("plan_order", ""),
                rid,
                r.get("status", "pending"),
                r.get("decision", "-"),
                parent,
                cfg.plan.get("est_train_hours_at_500_tok_s", ""),
                "private" if repo == hub.PRIVATE_REPO else "public",
                f"{nbytes / 1e9:.1f}G {how}" if nbytes else how,
                cond[:60] + ("…" if len(cond) > 60 else ""),
            )
        )
    budget = hub.PRIVATE_BUDGET_BYTES
    print(
        f"\nprivate repo {hub.PRIVATE_REPO}: {hub.gb(pushed_private)} pushed (state); "
        f"{n_private} private runs "
        f"would need {hub.gb(planned_private)} with all weights; budget {hub.gb(budget)} "
        f"= {int((budget - pushed_private) // hub.BF16_MODEL_BYTES_EST)} more bf16 models"
    )
    fit = int(budget // hub.BF16_MODEL_BYTES_EST)
    full = hub.ACCOUNT_USED_ELSEWHERE_BYTES + fit * hub.BF16_MODEL_BYTES_EST
    print(
        f"account: ~{hub.gb(hub.ACCOUNT_USED_ELSEWHERE_BYTES)} used elsewhere + {fit} private bf16 models "
        f"= ~{hub.gb(full)} of {hub.gb(hub.ACCOUNT_QUOTA_BYTES)} (public repo storage is free)"
    )
    print(f"endpoints: {state['endpoints'] or 'none selected'}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    state = hub.load_state()
    print(f"state file: {hub.state_path()}")
    for rid, r in sorted(state["runs"].items()):
        items = ", ".join(f"{k}={v}" for k, v in sorted(r.items()))
        print(f"  {rid}: {items}")
    print(f"endpoints: {state['endpoints'] or 'none selected'}")
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    configs, state = load_all(), hub.load_state()
    children = [
        c.run_id
        for c in configs.values()
        if hub.parent_id(c) == args.run or c.training.resume["from_run"] == args.run
    ]
    open_children = [
        c
        for c in children
        if state["runs"].get(c, {}).get("status") != "done"
        and state["runs"].get(c, {}).get("decision") != "skip"
    ]
    if open_children:
        raise SystemExit(f"{args.run}: children not done or skipped: {open_children}")
    ckpt = hub.RUNS_DIR / args.run / "checkpoints"
    if not ckpt.exists():
        print(f"nothing to prune: {ckpt} does not exist")
        return 0
    size = hub.gb(hub.dir_bytes(ckpt))
    if not args.yes and input(f"delete {ckpt} ({size})? [y/N] ").strip().lower() != "y":
        print("kept")
        return 0
    shutil.rmtree(ckpt)
    print(f"deleted {ckpt} ({size})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("configs", help="generate src/configs/TG-*.yaml from the CSV and validate them")
    p.add_argument("--check", action="store_true", help="report differences, write nothing")
    p.set_defaults(fn=cmd_configs)

    sub.add_parser("plan", help="runs in plan order with status, parent, repo and bytes").set_defaults(
        fn=cmd_plan
    )
    sub.add_parser("status", help="print the pipeline state").set_defaults(fn=cmd_status)

    p = sub.add_parser("run", help="preflight, train in a subprocess, publish")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("run", nargs="?", help="run id, e.g. TG-10")
    g.add_argument("--next", action="store_true", help="the first ready run in plan order")
    g.add_argument(
        "--all", action="store_true", help="every ready run in plan order (never undecided conditionals)"
    )
    p.add_argument(
        "--priority", type=int, help="only runs with priority <= N (1 core, 2 interpret, 3 extend)"
    )
    p.add_argument("--no-push", action="store_true", help="train only; publish later with `push`")
    p.add_argument("--skip-download", action="store_true", help="do not pre-download the data in preflight")
    p.add_argument("--dry-run", action="store_true", help="preflight and print the command, do not train")
    p.add_argument(
        "--wandb", choices=tracking.MODES, help="W&B mode for this run (default: logging.wandb.mode in the YAML)"
    )
    p.epilog = "Arguments after `--` go to src.pipeline.train, e.g. `run TG-10 -- --max-steps 3`."
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("decide", help="record the outcome of a run's 'only if' condition")
    p.add_argument("run")
    p.add_argument("decision", choices=("go", "skip"))
    p.set_defaults(fn=cmd_decide)

    p = sub.add_parser("select-endpoint", help="choose what run:<RUN> resolves to (the size check)")
    p.add_argument("run")
    p.add_argument("endpoint")
    p.set_defaults(fn=cmd_select_endpoint)

    p = sub.add_parser("push", help="(re)publish a finished run without retraining")
    p.add_argument("run")
    p.add_argument("--dry-run", action="store_true", help="print the target and staged files, upload nothing")
    p.add_argument(
        "--replace", action="store_true", help="delete the existing folder, squash history, upload"
    )
    p.add_argument("--no-weights", action="store_true", help="leave model/ out")
    p.set_defaults(fn=cmd_push)

    p = sub.add_parser("prune", help="delete a run's local checkpoints once its children are done")
    p.add_argument("run")
    p.add_argument("--yes", action="store_true", help="do not ask")
    p.set_defaults(fn=cmd_prune)

    argv = sys.argv[1:] if argv is None else list(argv)
    train_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    args = ap.parse_args(argv[: argv.index("--")] if "--" in argv else argv)
    args.train_args = train_args
    try:
        return args.fn(args)
    except (hub.HubError, ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
