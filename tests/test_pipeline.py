"""Configs from the CSV, repo assignment, ``run:`` resolution and the HF push rules (HfApi is mocked:
nothing here touches the network)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import training_pipeline as tp
from src.pipeline import hub
from src.pipeline.config import ConfigError, load_config, parse

PRIVATE_RUNS = {
    "TG-32-s1",
    "TG-32-s2",
    "TG-32-s3",
    "TG-33-s1",
    "TG-33-s2",
    "TG-33-s3",
    "TG-34-s1",
    "TG-34-s2",
    "TG-34-s3",
    "TG-35-s1",
    "TG-36-s1",
    "TG-36-s2",
    "TG-36-s3",
    "TG-90",
}


@pytest.fixture(scope="module")
def configs():
    return tp.load_all()


# ── configs ───────────────────────────────────────────────────────────────────


def test_generated_configs_match_csv():
    out = tp.generate()
    assert len(out) == 23
    stale = [p.name for p, text in out.items() if not p.exists() or p.read_text() != text]
    assert not stale, f"run `python training_pipeline.py configs`: {stale}"


def test_all_configs_load(configs):
    assert len(configs) == 23
    for cfg in configs.values():
        t, d = cfg.training, cfg.distributed
        assert t.micro_batch_tokens * t.grad_accum * d.num_gpus == t.tokens_per_step == 65536
        assert d.strategy == "single" and cfg.model.dtype == "float32"


def test_csv_mapping(configs):
    tg10 = configs["TG-10"]
    assert tg10.model.init_from == "google/translategemma-12b-it"
    assert tg10.training.lr_schedule["type"] == "wsd"
    assert tg10.training.lr_schedule["warmup_start"] == 4e-7
    assert tg10.training.save == {
        "every_steps": 250,
        "keep_steps": [1196, 1994, 2957],
        "with_optimizer": True,
    }
    assert tg10.training.max_steps == 3479 and tg10.seed == 1

    for rid, step, kind in (
        ("TG-10-CD06", 1196, "cooldown"),
        ("TG-10-CD10", 1994, "cooldown"),
        ("TG-10-EXT", 2957, "wsd"),
    ):
        c = configs[rid]
        assert c.training.lr_schedule["type"] == kind
        assert c.training.resume == {
            "from_run": "TG-10",
            "from_step": step,
            "optimizer_state": True,
            "start_global_step": step,
        }
    assert configs["TG-10-CD06"].training.save["every_steps"] is None
    assert configs["TG-10-CD06"].training.lr_schedule["warmup_start"] is None

    tg20 = configs["TG-20"]
    assert tg20.model.init_from == "run:TG-10" and tg20.training.lr_schedule["type"] == "cosine"
    assert tg20.training.resume["from_run"] is None and tg20.training.resume["start_global_step"] == 0
    assert configs["TG-33-s2"].data.mixture == configs["TG-33-s1"].data.mixture
    assert configs["TG-33-s2"].seed == 2
    assert configs["TG-34-s1"].plan["stage"] == "lexical"


def test_validation_rejects_bad_schedule_and_repo(configs):
    raw = json.loads(json.dumps(configs["TG-31-s1"].to_dict()))
    raw.pop("source")
    raw["training"]["lr_schedule"]["type"] = "constant"
    with pytest.raises(ConfigError):
        from src.pipeline.config import validate

        validate(parse(raw))
    raw["training"]["lr_schedule"] = {"type": "cosine", "bogus": 1}
    with pytest.raises(ConfigError, match="unknown keys"):
        parse(raw)


def test_check_mode_reports_drift(tmp_path, monkeypatch, capsys):
    for p in tp.CONFIG_DIR.glob("*.yaml"):
        (tmp_path / p.name).write_text(p.read_text())
    (tmp_path / "TG-21.yaml").write_text(
        (tmp_path / "TG-21.yaml").read_text().replace("max_steps: 778", "max_steps: 1")
    )
    monkeypatch.setattr(tp, "CONFIG_DIR", tmp_path)
    assert tp.main(["configs", "--check"]) == 1
    assert "TG-21.yaml: differs" in capsys.readouterr().out
    assert "max_steps: 1" in (tmp_path / "TG-21.yaml").read_text()  # --check writes nothing


# ── repo assignment ───────────────────────────────────────────────────────────


def test_repo_assignment(configs):
    for rid, cfg in configs.items():
        want = hub.PRIVATE_REPO if rid in PRIVATE_RUNS else hub.PUBLIC_REPO
        assert hub.assign_repo(cfg, configs, {}) == want, rid


def _child(configs, parent: str, **output):
    """A gazette-only run (releasable data) that starts from ``parent``."""
    base = configs["TG-31-s1"]
    model = dataclasses.replace(base.model, init_from=f"run:{parent}")
    return dataclasses.replace(
        base, run_id="TG-X", model=model, output=dataclasses.replace(base.output, **output)
    )


def test_restricted_parent_chain_goes_private(configs):
    assert hub.assign_repo(_child(configs, "TG-31-s1"), configs, {}) == hub.PUBLIC_REPO
    child = _child(configs, "TG-33-s1")
    assert not hub.restricted_sources(child)
    assert hub.assign_repo(child, configs, {}) == hub.PRIVATE_REPO
    grandchild = _child({**configs, "TG-X": child}, "TG-X")
    assert hub.assign_repo(grandchild, {**configs, "TG-X": child}, {}) == hub.PRIVATE_REPO
    # a parent recorded as stored privately also makes the child private
    state = {"runs": {"TG-31-s1": {"repo": hub.PRIVATE_REPO}}, "endpoints": {}}
    assert hub.assign_repo(_child(configs, "TG-31-s1"), configs, state) == hub.PRIVATE_REPO


def test_public_override_on_restricted_run_is_refused(configs):
    with pytest.raises(hub.HubError, match="restricted"):
        hub.assign_repo(_child(configs, "TG-33-s1", repo="public"), configs, {})


# ── run: resolution ───────────────────────────────────────────────────────────


def test_resolve_init(configs, tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "RUNS_DIR", tmp_path)
    assert hub.resolve_init(configs["TG-10"], {}) == ("google/translategemma-12b-it", None, None)
    state = {
        "runs": {
            "TG-10-CD10": {
                "repo": hub.PUBLIC_REPO,
                "folder": "TG-10-CD10",
                "commit": "abc",
                "weights_pushed": True,
            }
        },
        "endpoints": {"TG-10": "TG-10-CD10"},
    }
    assert hub.resolve_init(configs["TG-20"], state) == (hub.PUBLIC_REPO, "TG-10-CD10/model", "abc")
    with pytest.raises(hub.HubError, match="kept checkpoint"):
        hub.resolve_init(configs["TG-10-CD06"], state)
    keep = hub.keep_dir("TG-10", 1196)
    keep.mkdir(parents=True)
    assert hub.resolve_init(configs["TG-10-CD06"], state) == (str(keep), None, None)


def test_tg20_waits_for_endpoint(configs, tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "RUNS_DIR", tmp_path)
    state = {"runs": {"TG-10": {"status": "done"}}, "endpoints": {}}
    assert any("select-endpoint" in r for r in tp.blockers(configs["TG-20"], configs, state, explicit=True))
    assert tp.blockers(configs["TG-35-s1"], configs, state, explicit=False)[0].startswith("conditional")


# ── push: budget guard, write-once, visibility (mocked HfApi) ─────────────────


def _api(tree: dict[str, int], private: bool = True):
    api = MagicMock()
    api.list_repo_tree.return_value = [SimpleNamespace(path=p, size=s) for p, s in tree.items()]
    api.repo_info.return_value = SimpleNamespace(private=private, sha="f" * 40)
    api.uploaded = []

    def upload(repo_id, folder_path, **kw):
        api.uploaded = sorted(
            p.relative_to(folder_path).as_posix() for p in Path(folder_path).rglob("*") if p.is_file()
        )

    api.upload_large_folder.side_effect = upload
    return api


@pytest.fixture
def finished(configs, tmp_path, monkeypatch):
    """TG-33-s1 (private) finished locally: status done, a small model/ and metrics/."""
    monkeypatch.setattr(hub, "RUNS_DIR", tmp_path / "runs")
    rdir = tmp_path / "runs" / "TG-33-s1"
    (rdir / "model").mkdir(parents=True)
    (rdir / "model" / "model.safetensors").write_bytes(b"\0" * 1000)
    (rdir / "metrics").mkdir()
    (rdir / "metrics" / "dev_loss.jsonl").write_text('{"step": 25, "eval_loss": 1.5}\n')
    (rdir / "checkpoints").mkdir()
    (rdir / "checkpoints" / "optimizer.pt").write_bytes(b"x")
    (rdir / "status.json").write_text(json.dumps({"status": "done", "step": 494, "tokens": 32400000}))
    cfg = configs["TG-33-s1"]
    return dataclasses.replace(cfg, output=dataclasses.replace(cfg.output, local_dir=str(rdir)))


def test_budget_guard_threshold():
    assert hub.fits_budget(70_000_000_000, 5_000_000_000)
    assert not hub.fits_budget(70_000_000_000, 5_000_000_001)


def test_push_private_with_room(configs, finished):
    api = _api({"README.md": 100, "TG-32-s1/model/model.safetensors": 24_400_000_000})
    state = {"runs": {}, "endpoints": {}}
    out = hub.push(finished, configs, state, api, log=lambda s: None)
    assert out["repo"] == hub.PRIVATE_REPO and out["weights"]
    assert "TG-33-s1/model/model.safetensors" in api.uploaded
    assert "TG-33-s1/metrics/dev_loss.jsonl" in api.uploaded
    assert not any("checkpoints" in p for p in api.uploaded)  # resumable state is never pushed
    assert {"TG-33-s1/README.md", "TG-33-s1/config.yaml", "TG-33-s1/manifest.json"} <= set(api.uploaded)
    r = state["runs"]["TG-33-s1"]
    assert r["commit"] == "f" * 40 and r["weights_pushed"] and r["dev_loss"] == 1.5
    api.create_repo.assert_not_called()


def test_push_private_over_budget_drops_weights(configs, finished):
    logs = []
    tree = {f"TG-3{i}-s1/model/model.safetensors": 24_999_999_000 for i in (1, 2, 4)}  # 74.999997 GB
    api = _api(tree)
    state = {"runs": {}, "endpoints": {}}
    out = hub.push(finished, configs, state, api, log=logs.append)
    assert not out["weights"] and not state["runs"]["TG-33-s1"]["weights_pushed"]
    assert api.uploaded and not any("/model/" in p for p in api.uploaded)
    assert "TG-33-s1/manifest.json" in api.uploaded  # the rest still uploads
    assert any(line.startswith("REFUSED weights") for line in logs)
    assert any("prune candidate" in line for line in logs)


def test_push_is_write_once(configs, finished):
    api = _api({"TG-33-s1/README.md": 10})
    with pytest.raises(hub.HubError, match="write-once"):
        hub.push(finished, configs, {"runs": {}, "endpoints": {}}, api, log=lambda s: None)
    api.upload_large_folder.assert_not_called()


def test_push_replace_deletes_and_squashes(configs, finished):
    api = _api({"TG-33-s1/README.md": 10, "TG-32-s1/README.md": 10})
    state = {"runs": {"TG-32-s1": {"repo": hub.PRIVATE_REPO, "commit": "old"}}, "endpoints": {}}
    hub.push(finished, configs, state, api, replace=True, log=lambda s: None)
    api.delete_folder.assert_called_once()
    api.super_squash_history.assert_called_once()
    assert state["runs"]["TG-32-s1"]["commit"] == "f" * 40  # re-pinned after the squash


def test_refuses_restricted_push_if_private_repo_became_public(configs, finished):
    api = _api({}, private=False)
    with pytest.raises(hub.HubError, match="PUBLIC"):
        hub.push(finished, configs, {"runs": {}, "endpoints": {}}, api, log=lambda s: None)
    api.upload_large_folder.assert_not_called()


def test_missing_private_repo_is_created_private(configs, finished):
    from huggingface_hub.utils import RepositoryNotFoundError

    api = _api({})
    api.repo_info.side_effect = [  # missing at the visibility check, present after the upload
        RepositoryNotFoundError("nope", response=MagicMock()),
        SimpleNamespace(private=True, sha="a" * 40),
    ]
    hub.push(finished, configs, {"runs": {}, "endpoints": {}}, api, log=lambda s: None)
    api.create_repo.assert_called_once_with(hub.PRIVATE_REPO, repo_type="model", private=True, exist_ok=True)


def test_dry_run_uploads_nothing(configs, finished):
    logs = []
    api = _api({})
    out = hub.push(finished, configs, {"runs": {}, "endpoints": {}}, api, dry_run=True, log=logs.append)
    assert out["weights"] and out["files"] >= 5
    for name in (
        "create_repo",
        "upload_large_folder",
        "upload_file",
        "delete_folder",
        "super_squash_history",
    ):
        getattr(api, name).assert_not_called()
    assert any("TG-33-s1/model/model.safetensors" in line for line in logs)


def test_unfinished_run_is_not_pushed(configs, finished):
    (Path(finished.output.local_dir) / "status.json").write_text('{"status": "failed"}')
    with pytest.raises(hub.HubError, match="not 'done'"):
        hub.push(finished, configs, {"runs": {}, "endpoints": {}}, _api({}), log=lambda s: None)


def test_card_has_licence_and_no_rows(configs, finished):
    card = hub.index_card(hub.PRIVATE_REPO, {"runs": {"TG-33-s1": {"repo": hub.PRIVATE_REPO}}})
    assert card.startswith("---\nlicense: gemma\n") and "Gemma Terms of Use" in card
    run = hub.run_card(finished, hub.PRIVATE_REPO, {"step": 1}, True)
    assert "loader `verbis`" in run and "Gemma Terms of Use" in run


def test_load_config_on_generated_example():
    cfg = load_config(tp.CONFIG_DIR / "TG-36-s1.yaml")
    assert set(hub.restricted_sources(cfg)) == {"V_TERM/verbis", "V_TERM/media_literacy", "W_GEN/phrasebook"}
