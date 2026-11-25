"""Hugging Face storage for finished runs: repo choice, layout, budget guard, upload, ``run:`` resolution.

Two model repos with the same layout, one top-level folder per run (branches included)::

    README.md                  index of the runs in this repo (regenerated on every push)
    TG-10/README.md            run card: description, CSV row, data, status
    TG-10/config.yaml          resolved config as trained
    TG-10/manifest.json        provenance (written by the trainer; a minimal one otherwise)
    TG-10/metrics/*.jsonl      train_log.jsonl, dev_loss.jsonl
    TG-10/eval/...             decoded evaluation, once the eval hook exists
    TG-10/model/               bf16 save_pretrained + processor (loads in transformers / vLLM)

* ``PUBLIC_REPO`` holds runs on releasable data; ``PRIVATE_REPO`` holds every run that trained on a
  ``restricted`` loader (VERBIS, media literacy, phrasebook) or descends from such a run.
* Resumable checkpoints (fp32 + optimizer) are never pushed.
* Write once: a run folder is uploaded once. ``push(..., replace=True)`` deletes the folder and squashes
  the history (deleted files still count towards quota until then), then uploads again.
* The private repo is capped at ``PRIVATE_BUDGET_BYTES``; over it, the weights are left out and the
  rest of the run folder is still uploaded.

Load a stored run: ``snapshot_download(repo, allow_patterns="TG-20/model/*", revision=sha)``, then
``from_pretrained(<dir>/TG-20/model)``. ``resolve_init`` does this resolution for ``init_from: run:<ID>``.

No VERBIS (or other restricted) rows ever go into a card, manifest or log written here: only loader
names, options and the sidecars' counts and hashes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.pipeline import registry

if TYPE_CHECKING:
    from src.pipeline.config import RunConfig

PUBLIC_REPO = "EdonFetaji/translategemma-mk-sq"
PRIVATE_REPO = "EdonFetaji/translategemma-mk-sq-private"
BASE_MODEL = "google/translategemma-12b-it"
#: Hard cap on the private repo's stored files (the account had 25 of 100 GB used, 2026-11).
PRIVATE_BUDGET_BYTES = 75_000_000_000
ACCOUNT_USED_ELSEWHERE_BYTES = 25_000_000_000
ACCOUNT_QUOTA_BYTES = 100_000_000_000
#: A full bf16 save of TranslateGemma-12B, for planning before a run has been saved.
BF16_MODEL_BYTES_EST = 24_400_000_000

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"

GEMMA_NOTICE = (
    "Gemma is provided under and subject to the Gemma Terms of Use found at "
    "[ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms). These weights are a fine-tune of "
    f"[{BASE_MODEL}](https://huggingface.co/{BASE_MODEL})."
)


class HubError(RuntimeError):
    pass


# ── pipeline state (runs/pipeline_state.json) ────────────────────────────────


def state_path() -> Path:
    return RUNS_DIR / "pipeline_state.json"


def load_state() -> dict[str, Any]:
    """``{"runs": {run_id: {...}}, "endpoints": {run_id: chosen_run_id}}``; empty if no file yet."""
    path = state_path()
    state = json.loads(path.read_text()) if path.exists() else {}
    state.setdefault("runs", {})
    state.setdefault("endpoints", {})
    return state


def save_state(state: dict[str, Any]) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def run_state(state: dict[str, Any], run_id: str) -> dict[str, Any]:
    return state["runs"].setdefault(run_id, {})


# ── run identity ──────────────────────────────────────────────────────────────


def parent_id(cfg: RunConfig) -> str | None:
    """The run this one starts from (``init_from: run:<ID>``), or ``None`` for a hub model id."""
    init = cfg.model.init_from
    return init[len("run:") :] if init.startswith("run:") else None


def folder_of(cfg: RunConfig) -> str:
    return cfg.output.folder or cfg.run_id


def run_dir(cfg: RunConfig) -> Path:
    p = Path(cfg.output.local_dir)
    return p if p.is_absolute() else REPO_ROOT / p


def keep_dir(run_id: str, step: int) -> Path:
    """Where the trainer keeps a never-rotated checkpoint (with optimizer) of ``run_id`` at ``step``."""
    return RUNS_DIR / run_id / "checkpoints" / "keep" / f"step-{step:06d}"


def restricted_sources(cfg: RunConfig) -> list[str]:
    """Names of the training entries (mixture + shared) that read a restricted loader."""
    return [
        e.name or e.loader
        for e in (*cfg.data.shared, *cfg.data.mixture)
        if getattr(registry.get_loader(e.loader), "restricted", False)
    ]


def is_restricted(
    cfg: RunConfig, configs: Mapping[str, RunConfig], state: Mapping[str, Any] | None = None
) -> bool:
    """True if the run trains on restricted data, or any run up its ``init_from`` chain does
    (or was stored in the private repo)."""
    seen: set[str] = set()
    todo = [cfg]
    while todo:
        c = todo.pop()
        if restricted_sources(c):
            return True
        parent = parent_id(c)
        if parent is None:
            continue
        candidates = {parent}
        if state:
            candidates.add(state.get("endpoints", {}).get(parent, parent))
        for p in candidates - seen:
            seen.add(p)
            if state and state.get("runs", {}).get(p, {}).get("repo") == PRIVATE_REPO:
                return True
            if p not in configs:
                raise HubError(f"{c.run_id}: parent run {p!r} has no config")
            todo.append(configs[p])
    return False


def assign_repo(
    cfg: RunConfig, configs: Mapping[str, RunConfig], state: Mapping[str, Any] | None = None
) -> str:
    """The HF repo a run is stored in (``output.repo``: public | private | auto)."""
    restricted = is_restricted(cfg, configs, state)
    if cfg.output.repo == "public" and restricted:
        raise HubError(
            f"{cfg.run_id}: output.repo is 'public' but the run (or its parent chain) trains on restricted "
            f"data {restricted_sources(cfg) or '(via the parent)'}"
        )
    if cfg.output.repo == "private" or restricted:
        return PRIVATE_REPO
    return PUBLIC_REPO


def resolve_init(cfg: RunConfig, state: Mapping[str, Any]) -> tuple[str, str | None, str | None]:
    """Where to load the starting weights from: ``(repo_or_path, subfolder_or_None, revision)``.

    * a hub model id (``google/...``): itself, at ``model.revision``;
    * a branch (``training.resume.from_run``): the parent's local kept checkpoint at ``from_step``;
    * ``run:<ID>``: the chosen endpoint for ``<ID>`` if one was selected (``select-endpoint``), then
      that run's pushed folder at its pinned commit, else its local ``model/`` directory.
    """
    parent = parent_id(cfg)
    if parent is None:
        return cfg.model.init_from, None, cfg.model.revision
    resume = cfg.training.resume
    if resume.get("from_run"):
        path = keep_dir(resume["from_run"], int(resume["from_step"]))
        if not path.exists():
            raise HubError(
                f"{cfg.run_id}: kept checkpoint {path} does not exist (has {resume['from_run']} run?)"
            )
        return str(path), None, None
    target = state.get("endpoints", {}).get(parent, parent)
    info = state.get("runs", {}).get(target, {})
    if info.get("commit") and info.get("weights_pushed"):
        return info["repo"], f"{info.get('folder') or target}/model", info["commit"]
    local = RUNS_DIR / target / "model"
    if local.exists():
        return str(local), None, None
    raise HubError(f"{cfg.run_id}: parent {target} has neither pushed weights nor a local model/ directory")


# ── hub listing and the budget guard ──────────────────────────────────────────


def repo_files(api: Any, repo: str) -> dict[str, int]:
    """``{path: size}`` of every file in ``repo`` (empty if the repo does not exist yet)."""
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        items = api.list_repo_tree(repo, recursive=True, repo_type="model")
        return {it.path: it.size for it in items if getattr(it, "size", None) is not None}
    except RepositoryNotFoundError:
        return {}


def folder_bytes(files: Mapping[str, int]) -> dict[str, int]:
    """Bytes per top-level folder (``README.md`` at the root counts as ``""``)."""
    out: dict[str, int] = {}
    for path, size in files.items():
        top = path.split("/", 1)[0] if "/" in path else ""
        out[top] = out.get(top, 0) + size
    return out


def fits_budget(used_bytes: int, new_bytes: int, budget: int = PRIVATE_BUDGET_BYTES) -> bool:
    return used_bytes + new_bytes <= budget


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def check_repo_visibility(api: Any, repo: str, create: bool) -> None:
    """The private repo must exist as private (created if ``create``); the public one is created public.

    Refuses a push to ``PRIVATE_REPO`` if it has become public.
    """
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        info = api.repo_info(repo, repo_type="model")
    except RepositoryNotFoundError:
        if not create:
            return
        api.create_repo(repo, repo_type="model", private=(repo == PRIVATE_REPO), exist_ok=True)
        return
    if repo == PRIVATE_REPO and not info.private:
        raise HubError(f"{repo} is PUBLIC; refusing to push restricted runs there. Make it private first.")


def check_auth(api: Any) -> str:
    try:
        return api.whoami()["name"]
    except Exception as e:  # noqa: BLE001  (any failure means: not logged in)
        raise HubError(f"HF auth failed ({type(e).__name__}); run `hf auth login`") from None


# ── run folder contents ───────────────────────────────────────────────────────


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=False, cwd=REPO_ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _last_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return json.loads(lines[-1]) if lines else {}


def dev_loss_of(cfg: RunConfig) -> float | None:
    rec = _last_jsonl(run_dir(cfg) / "metrics" / "dev_loss.jsonl")
    for key in ("eval_loss", "dev_loss", "loss"):
        if isinstance(rec.get(key), int | float):
            return float(rec[key])
    return None


def minimal_manifest(cfg: RunConfig, state: Mapping[str, Any]) -> dict[str, Any]:
    """Used only when the trainer wrote no ``manifest.json``: provenance without any data rows."""
    loaders = []
    for e in (*cfg.data.shared, *cfg.data.mixture):
        try:
            desc = registry.get_loader(e.loader)(**e.options).describe()
        except Exception as err:  # noqa: BLE001
            desc = {"loader": e.loader, "error": type(err).__name__}
        loaders.append({"entry": e.name or e.loader, "split": e.split, "task": e.task, "describe": desc})
    parent = parent_id(cfg)
    target = state.get("endpoints", {}).get(parent, parent) if parent else None
    return {
        "run_id": cfg.run_id,
        "git_sha": _git_sha(),
        "seed": cfg.seed,
        "init_from": cfg.model.init_from,
        "parent": target,
        "parent_commit": state.get("runs", {}).get(target, {}).get("commit") if target else None,
        "status": _read_json(run_dir(cfg) / "status.json"),
        "lr_schedule": cfg.training.lr_schedule,
        "max_steps": cfg.training.max_steps,
        "loaders": loaders,
        "note": "minimal manifest written by src/pipeline/hub.py (the trainer wrote none)",
    }


def wandb_line(cfg: RunConfig) -> str:
    """``- W&B: <url>`` for the run card, from ``runs/<id>/wandb.json`` (empty if not tracked)."""
    info = _read_json(run_dir(cfg) / "wandb.json")
    if info.get("url"):
        return f"- W&B: {info['url']}\n"
    return f"- W&B: offline run `{info['id']}`\n" if info.get("id") else ""


def run_card(cfg: RunConfig, repo: str, status: Mapping[str, Any], weights: bool) -> str:
    entries = "\n".join(
        f"- `{e.name or e.loader}`: loader `{e.loader}` ({e.split}), task `{e.task}`, {e.directions}"
        for e in (*cfg.data.shared, *cfg.data.mixture)
    )
    load = (
        f"```python\nfrom huggingface_hub import snapshot_download\n"
        f'd = snapshot_download("{repo}", allow_patterns="{folder_of(cfg)}/model/*", revision="<commit>")\n'
        f'# then from_pretrained(f"{{d}}/{folder_of(cfg)}/model")\n```'
        if weights
        else "Weights not stored in this repo for this run."
    )
    plan = yaml.safe_dump(cfg.plan, sort_keys=False, allow_unicode=True, width=100)
    return (
        f"# {cfg.run_id}\n\n{cfg.description}\n\n"
        f"- init_from: `{cfg.model.init_from}`\n- seed: {cfg.seed}\n"
        f"- steps: {status.get('step')} / planned {cfg.training.max_steps}\n"
        f"- tokens: {status.get('tokens')}\n"
        f"- lr: {cfg.training.lr_schedule['type']} peak {cfg.training.lr_schedule['peak']}\n"
        f"- weights: {'yes (bf16, `model/`)' if weights else 'no'}\n"
        f"{wandb_line(cfg)}\n"
        f"## Training data\n\n{entries}\n\n## Plan (tg_hyperparameters.csv)\n\n```yaml\n{plan}```\n\n"
        f"## Load\n\n{load}\n\n{GEMMA_NOTICE}\n"
    )


def index_card(repo: str, state: Mapping[str, Any]) -> str:
    rows = []
    for rid, r in sorted(state.get("runs", {}).items()):
        if r.get("repo") != repo:
            continue
        rows.append(
            f"| [{rid}]({r.get('folder') or rid}/) | {r.get('stage', '')} | {r.get('init_from', '')} | "
            f"{r.get('step', '')} | {r.get('tokens', '')} | {r.get('dev_loss', '')} | "
            f"{'yes' if r.get('weights_pushed') else 'no'} | {(r.get('commit') or '')[:10]} |"
        )
    vis = "Private: runs trained on licence-restricted data." if repo == PRIVATE_REPO else "Public runs."
    return (
        f"---\nlicense: gemma\nbase_model: {BASE_MODEL}\nlanguage: [mk, sq]\n"
        "pipeline_tag: translation\n---\n\n"
        f"# TranslateGemma MK-SQ runs\n\n{vis} One folder per run; see each folder's README.\n\n"
        "| run | stage | init_from | steps | tokens | dev loss | weights | commit |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n" + "\n".join(rows) + f"\n\n{GEMMA_NOTICE}\n"
    )


def collect(cfg: RunConfig, repo: str, state: Mapping[str, Any], weights: bool) -> dict[str, Path | str]:
    """What goes into the run folder: ``{path inside the folder: local file | generated text}``."""
    rdir = run_dir(cfg)
    status = _read_json(rdir / "status.json")
    out: dict[str, Path | str] = {"README.md": run_card(cfg, repo, status, weights)}
    out["config.yaml"] = rdir / "config.yaml" if (rdir / "config.yaml").exists() else cfg.to_yaml()
    manifest = rdir / "manifest.json"
    out["manifest.json"] = (
        manifest if manifest.exists() else json.dumps(minimal_manifest(cfg, state), indent=2)
    )
    subdirs = ("metrics", "eval", "model") if weights else ("metrics", "eval")
    for sub in subdirs:
        for p in sorted((rdir / sub).rglob("*")) if (rdir / sub).exists() else []:
            if p.is_file():
                out[p.relative_to(rdir).as_posix()] = p
    return out


def stage(files: Mapping[str, Path | str], dest: Path) -> None:
    """Materialise ``files`` under ``dest`` (hard links for large files, copies across devices)."""
    if dest.exists():
        shutil.rmtree(dest)
    for rel, src in files.items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(src, str):
            target.write_text(src)
        else:
            try:
                os.link(src, target)
            except OSError:
                shutil.copy2(src, target)


def _size(src: Path | str) -> int:
    return len(src.encode()) if isinstance(src, str) else src.stat().st_size


# ── push ──────────────────────────────────────────────────────────────────────


def push(
    cfg: RunConfig,
    configs: Mapping[str, RunConfig],
    state: dict[str, Any],
    api: Any,
    *,
    replace: bool = False,
    dry_run: bool = False,
    weights: bool | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Publish one finished run folder. Returns a summary; updates ``state`` (not saved here)."""
    repo = assign_repo(cfg, configs, state)
    folder = folder_of(cfg)
    rdir = run_dir(cfg)
    status = _read_json(rdir / "status.json")
    if status.get("status") != "done":
        msg = f"{cfg.run_id}: status.json says {status.get('status')!r}, not 'done'"
        if not dry_run:
            raise HubError(msg)
        log(f"WARNING {msg}")

    # Public pushes: the data and the whole parent chain must be releasable (assign_repo enforces it).
    if repo == PUBLIC_REPO and is_restricted(cfg, configs, state):
        raise HubError(f"{cfg.run_id}: restricted data in the run or its parent chain; never public")
    try:
        check_repo_visibility(api, repo, create=not dry_run)
        files = repo_files(api, repo)
    except HubError:
        raise
    except Exception as e:  # offline / bad token
        if not dry_run:
            raise
        log(f"(dry run) could not list {repo}: {type(e).__name__}; budget and write-once not checked")
        files = {}

    existing = {p: s for p, s in files.items() if p.startswith(folder + "/")}
    if existing and not replace:
        raise HubError(
            f"{repo}/{folder} already exists ({len(existing)} files); run folders are write-once. "
            "Use `push --replace` to delete it, squash the history and upload again."
        )

    want_weights = cfg.output.push_weights if weights is None else weights
    model_dir = rdir / "model"
    if want_weights and not model_dir.exists():
        log(f"WARNING {model_dir} missing; pushing without weights")
        want_weights = False
    used = sum(files.values()) - sum(existing.values())
    if repo == PRIVATE_REPO and want_weights:
        other = sum(_size(v) for v in collect(cfg, repo, state, False).values())
        new = other + dir_bytes(model_dir)
        if not fits_budget(used, new):
            want_weights = False
            log(
                f"REFUSED weights for {cfg.run_id}: private repo holds {gb(used)}, + {gb(new)} > "
                f"{gb(PRIVATE_BUDGET_BYTES)} budget. Metrics, config and manifest still upload."
            )
            for top, size in sorted(folder_bytes(files).items(), key=lambda kv: -kv[1]):
                if top and size > 1e9:
                    log(f"  prune candidate: {top}/ {gb(size)}  (push {top} --replace --no-weights)")

    entries = collect(cfg, repo, state, want_weights)
    total = sum(_size(v) for v in entries.values())
    summary = {"repo": repo, "folder": folder, "files": len(entries), "bytes": total, "weights": want_weights}
    if dry_run:
        log(f"(dry run) {cfg.run_id} -> {repo}/{folder}/  ({len(entries)} files, {gb(total)})")
        for rel, src in entries.items():
            log(f"  {folder}/{rel}  {_size(src):,} B")
        if repo == PRIVATE_REPO:
            log(f"  private repo after push: {gb(used + total)} of {gb(PRIVATE_BUDGET_BYTES)}")
        return summary

    if existing:
        log(f"--replace: deleting {repo}/{folder}/ and squashing history (frees the old files' quota)")
        api.delete_folder(folder, repo, repo_type="model", commit_message=f"Delete {folder} (republish)")
        api.super_squash_history(
            repo, repo_type="model", commit_message=f"Squash history after deleting {folder}"
        )

    staging = RUNS_DIR / "_upload" / cfg.run_id
    stage(entries, staging / folder)
    api.upload_large_folder(
        repo_id=repo, folder_path=staging, repo_type="model", private=(repo == PRIVATE_REPO)
    )

    r = run_state(state, cfg.run_id)
    r.update(
        repo=repo,
        folder=folder,
        bytes=total,
        weights_pushed=want_weights,
        step=status.get("step"),
        tokens=status.get("tokens"),
        dev_loss=dev_loss_of(cfg),
        stage=cfg.plan.get("stage", ""),
        init_from=cfg.model.init_from,
        pushed_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    api.upload_file(
        path_or_fileobj=index_card(repo, state).encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="model",
        commit_message=f"Index: add {cfg.run_id}",
    )
    sha = api.repo_info(repo, repo_type="model").sha
    r["commit"] = sha
    if existing:  # squashing rewrote every commit: re-pin the other runs (their files are unchanged)
        for rid, other in state["runs"].items():
            if rid != cfg.run_id and other.get("repo") == repo and other.get("commit"):
                log(f"  re-pinned {rid} to {sha[:10]} (history squashed; old sha in child manifests)")
                other["commit"] = sha
    shutil.rmtree(staging, ignore_errors=True)
    log(f"pushed {cfg.run_id} -> {repo}/{folder}/ @ {sha[:10]} ({gb(total)}, weights={want_weights})")
    return {**summary, "commit": sha}
