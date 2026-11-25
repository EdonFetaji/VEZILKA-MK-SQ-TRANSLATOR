"""W&B tracking: grouping, rank/mode gating, failure isolation, global steps, HF references (wandb faked)."""

from __future__ import annotations

import dataclasses
import json
import sys
import types
from pathlib import Path

import pytest

from src.pipeline import tracking
from src.pipeline.config import ConfigError, load_config, parse, read_with_extends, validate

CONFIGS = Path(__file__).resolve().parents[1] / "src" / "configs"


def cfg_of(run_id: str):
    return load_config(CONFIGS / f"{run_id}.yaml")


class FakeRun:
    def __init__(self, fake, kw):
        self.fake, self.kw = fake, kw
        self.rows, self.summary, self.artifacts, self.finished = [], {}, [], None
        self.url = f"https://wandb.ai/me/{kw.get('project')}/runs/{kw.get('id')}"
        self.entity = "me"

    def log(self, row):
        if self.fake.fail_log:
            raise RuntimeError("network down")
        self.rows.append(row)

    def define_metric(self, *a, **kw):
        pass

    def log_artifact(self, art, aliases=None):
        self.artifacts.append((art, aliases))

    def finish(self, exit_code=0, quiet=False):
        self.finished = exit_code


class FakeArtifact:
    def __init__(self, name, type, metadata=None, description=None):
        self.name, self.type, self.metadata, self.refs = name, type, metadata, []

    def add_reference(self, uri, checksum=True):
        self.refs.append(uri)


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = types.ModuleType("wandb")
    fake.runs, fake.fail_log, fake.fail_modes = [], False, set()

    def init(**kw):
        if kw.get("mode") in fake.fail_modes:
            raise RuntimeError(f"cannot init {kw.get('mode')}")
        run = FakeRun(fake, kw)
        fake.runs.append(run)
        return run

    fake.init = init
    fake.Settings = lambda **kw: kw
    fake.Artifact = FakeArtifact
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    return fake


class State:
    def __init__(self, step=0, tokens=0):
        self.global_step, self.num_input_tokens_seen, self.is_world_process_zero = step, tokens, True


def test_stage_groups():
    assert tracking.stage_group(cfg_of("TG-10")) == "general_mt"
    assert tracking.stage_group(cfg_of("TG-10-CD06")) == "general_mt"
    assert tracking.stage_group(cfg_of("TG-20")) == "domain_mt"
    assert tracking.stage_group(cfg_of("TG-33-s1")) == "lexical"
    assert tracking.stage_group(cfg_of("TG-90")) == "final"
    rl = dataclasses.replace(cfg_of("TG-20"), plan={"stage": "rl"})
    assert tracking.stage_group(rl) == "rl"


def test_disabled_and_non_zero_rank_never_init(fake_wandb, tmp_path, monkeypatch):
    cfg = cfg_of("TG-20")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert tracking.WandbTracker(cfg, tmp_path).start() is None
    monkeypatch.delenv("WANDB_MODE")
    monkeypatch.setenv("RANK", "1")
    assert tracking.WandbTracker(cfg, tmp_path).start() is None
    assert fake_wandb.runs == []
    off = dataclasses.replace(
        cfg, training=dataclasses.replace(cfg.training, logging={"wandb": {"enabled": False}})
    )
    monkeypatch.delenv("RANK")
    assert tracking.WandbTracker(off, tmp_path).start() is None
    assert fake_wandb.runs == []


def test_init_groups_and_resumes_same_id(fake_wandb, tmp_path):
    cfg = cfg_of("TG-33-s1")
    run = tracking.WandbTracker(cfg, tmp_path).start()
    kw = run.kw
    assert (kw["name"], kw["group"], kw["job_type"], kw["resume"]) == (
        "TG-33-s1",
        "lexical",
        "TG-33",
        "allow",
    )
    assert "seed1" in kw["tags"] and kw["config"]["run_id"] == "TG-33-s1" and "plan" in kw["config"]
    info = json.loads((tmp_path / "wandb.json").read_text())
    again = tracking.WandbTracker(cfg, tmp_path).start()  # crash restart
    assert again.kw["id"] == info["id"] == kw["id"]


def test_online_failure_falls_back_to_offline_then_gives_up(fake_wandb, tmp_path):
    fake_wandb.fail_modes = {"online"}
    tr = tracking.WandbTracker(cfg_of("TG-20"), tmp_path)
    assert tr.start() is not None
    assert json.loads((tmp_path / "wandb.json").read_text())["mode"] == "offline"
    fake_wandb.fail_modes = {"online", "offline"}
    assert tracking.WandbTracker(cfg_of("TG-20"), tmp_path / "b").start() is None  # no raise


def test_branch_rows_use_global_step(fake_wandb, tmp_path):
    cfg = cfg_of("TG-10-CD06")
    tr = tracking.WandbTracker(cfg, tmp_path, start_global_step=1196)
    tr.on_train_begin(None, State(0, 0), None)
    tr.on_log(
        None,
        State(5, 327680),
        None,
        logs={"loss": 1.5, "learning_rate": 9e-6, "grad_norm": 0.4, "epoch": 0.01},
    )
    tr.on_log(
        None,
        State(5, 327680),
        None,
        logs={"eval_FLORES_DEV.mk-sq_loss": 2.1, "eval_GAZ_DEV.sq-mk_runtime": 3.0},
    )
    train_row, eval_row = fake_wandb.runs[0].rows
    assert train_row["global_step"] == 1201 and train_row["local_step"] == 5
    assert train_row["train/loss"] == 1.5 and train_row["train/learning_rate"] == 9e-6
    assert train_row["train/grad_norm"] == 0.4 and train_row["train/epoch"] == 0.01
    assert "perf/tokens_per_s" in train_row and "mem/system_used_gb" in train_row
    assert eval_row["eval/FLORES_DEV.mk-sq/loss"] == 2.1 and eval_row["eval/GAZ_DEV.sq-mk/runtime"] == 3.0


def test_logging_errors_never_raise(fake_wandb, tmp_path):
    tr = tracking.WandbTracker(cfg_of("TG-20"), tmp_path)
    tr.on_train_begin(None, State(), None)
    fake_wandb.fail_log = True
    tr.on_log(None, State(1, 10), None, logs={"loss": 1.0})  # must not raise
    assert tr.run is None  # disabled for the rest of the run
    tr.on_log(None, State(2, 20), None, logs={"loss": 1.0})
    tr.on_save(types.SimpleNamespace(output_dir=str(tmp_path)), State(2), None)
    tr.close(failed=True)


def test_broken_wandb_module_does_not_break_start(monkeypatch, tmp_path):
    broken = types.ModuleType("wandb")  # no init / Settings at all
    monkeypatch.setitem(sys.modules, "wandb", broken)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    assert tracking.WandbTracker(cfg_of("TG-20"), tmp_path).start() is None


def test_log_checkpoint_references_hf_commit(fake_wandb, tmp_path):
    cfg = cfg_of("TG-20")
    tracking.WandbTracker(cfg, tmp_path).start()
    rec = {
        "repo": "EdonFetaji/translategemma-mk-sq",
        "folder": "TG-20",
        "commit": "abc123def4567890",
        "step": 2594,
        "tokens": 170_000_000,
        "dev_loss": 1.23,
        "weights_pushed": True,
    }
    assert tracking.log_checkpoint(cfg, tmp_path, rec)
    run = fake_wandb.runs[-1]
    assert run.kw["id"] == json.loads((tmp_path / "wandb.json").read_text())["id"]
    art, aliases = run.artifacts[0]
    url = "https://huggingface.co/EdonFetaji/translategemma-mk-sq/tree/abc123def4567890/TG-20"
    assert art.name == "TG-20-hf" and art.type == "model" and art.refs == [url]
    assert art.metadata["commit"] == "abc123def4567890" and art.metadata["global_step"] == 2594
    assert run.summary["hf/url"] == url and "abc123def456" in aliases


def test_post_run_helpers_skip_untracked_runs(fake_wandb, tmp_path):
    assert not tracking.log_checkpoint(cfg_of("TG-20"), tmp_path, {"repo": "r", "commit": "c"})
    assert not tracking.log_eval(cfg_of("TG-20"), tmp_path, "final", {"x": 1.0})
    assert fake_wandb.runs == []


def test_log_eval_metric_keys(fake_wandb, tmp_path):
    cfg = cfg_of("TG-33-s1")
    tracking.WandbTracker(cfg, tmp_path).start()
    scores = {"FLORES_DEV.mk-sq": {"bleu": 20.1, "chrf++": 50.2, "comet": 0.85}, "TERM_DEV": 0.61}
    assert tracking.log_eval(cfg, tmp_path, "final", scores, global_step=494)
    row = fake_wandb.runs[-1].rows[-1]
    assert (
        row["eval/final/FLORES_DEV.mk-sq/chrf++"] == 50.2 and row["eval/final/FLORES_DEV.mk-sq/comet"] == 0.85
    )
    assert row["eval/final/TERM_DEV"] == 0.61 and row["global_step"] == 494


def test_config_rejects_bad_wandb_block():
    raw = read_with_extends(CONFIGS / "TG-20.yaml")
    raw["training"]["logging"]["wandb"]["mode"] = "sometimes"
    with pytest.raises(ConfigError, match="mode"):
        validate(parse(raw))
    raw["training"]["logging"]["wandb"] = {"projct": "x"}
    with pytest.raises(ConfigError, match="allowed keys"):
        validate(parse(raw))
