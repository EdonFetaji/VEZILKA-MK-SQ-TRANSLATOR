"""``lr_at`` against the CSV ``lr_formula`` (plan Verification §4). Schedules are read from the CSV itself."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch

from src.pipeline.trainer import lr_at

CSV = Path(__file__).resolve().parents[1] / "src" / "configs" / "tg_hyperparameters.csv"


def _num(v: str) -> float | None:
    return float(v) if v.strip() else None


def csv_schedule(run_id: str) -> tuple[dict, int, int]:
    """(sched, start_global_step, end_global_step) for one CSV row."""
    row = next(r for r in csv.DictReader(CSV.open()) if r["run_id"] == run_id)
    text = row["lr_schedule"]
    kind = "cosine" if text.startswith("cosine") else "wsd" if text.startswith("wsd") else "cooldown"
    start = int(row["start_global_step"] or 0)
    sched = {
        "type": kind,
        "peak": float(row["lr_peak"]),
        "min": float(row["lr_min"]),
        "warmup_start": _num(row["lr_warmup_start"]),
        "warmup_steps": int(row["warmup_steps"] or 0),
        "constant_end_step": int(row["constant_end_step"]) if row["constant_end_step"] else None,
        "cooldown_steps": int(row["cooldown_steps"]) if row["cooldown_steps"] else None,
    }
    return sched, start, start + int(row["planned_updates"])


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-15)


def test_tg10_wsd():
    s, start, end = csv_schedule("TG-10")
    assert (start, end) == (0, 3479)
    assert close(lr_at(0, s, end), 4e-7)
    assert close(lr_at(52, s, end), 4e-7 + (1e-5 - 4e-7) * 52 / 104)
    assert close(lr_at(104, s, end), 1e-5)
    assert close(lr_at(1196, s, end), 1e-5)
    assert close(lr_at(2957, s, end), 1e-5)
    assert close(lr_at(2957 + 261, s, end), 1e-6 + 9e-6 * (1 - math.sqrt(0.5)))
    assert close(lr_at(3479, s, end), 1e-6)


@pytest.mark.parametrize("run_id,branch", [("TG-10-CD06", 1196), ("TG-10-CD10", 1994)])
def test_cooldown_branches(run_id, branch):
    s, start, end = csv_schedule(run_id)
    assert start == branch
    assert close(lr_at(start, s, end), 1e-5)
    assert lr_at(start + 1, s, end) < 1e-5
    assert close(lr_at(end, s, end), 1e-6)


def test_cd06_end_step():
    s, _start, end = csv_schedule("TG-10-CD06")
    assert end == 1316
    assert close(lr_at(1316, s, end), 1e-6)


def test_tg10_ext():
    s, start, end = csv_schedule("TG-10-EXT")
    assert (start, end) == (2957, 4386)
    assert close(lr_at(2957, s, end), 1e-5)
    assert close(lr_at(3727, s, end), 1e-5)
    assert close(lr_at(4386, s, end), 1e-6)


def test_tg20_cosine():
    s, _start, end = csv_schedule("TG-20")
    assert close(lr_at(0, s, end), 2e-7)
    assert close(lr_at(78, s, end), 5e-6)
    assert close(lr_at(end, s, end), 5e-7)
    mid = 78 + (end - 78) // 2
    assert lr_at(mid, s, end) == pytest.approx((5e-6 + 5e-7) / 2, rel=1e-3)


def test_cosine_needs_max_steps():
    s, _, _ = csv_schedule("TG-20")
    with pytest.raises(ValueError):
        lr_at(100, s)


def test_lambda_lr_offsets_global_step():
    """The trainer's LambdaLR counts local steps from 0; the branch offset makes them global."""
    from types import SimpleNamespace

    from src.pipeline.trainer import build_scheduler

    s, start, end = csv_schedule("TG-10-CD06")
    cfg = SimpleNamespace(training=SimpleNamespace(
        lr_schedule=s, max_steps=end - start, resume={"start_global_step": start}))
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=s["peak"])
    sched = build_scheduler(cfg, opt)
    lrs = []
    for _ in range(end - start + 1):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert close(lrs[0], 1e-5) and close(lrs[-1], 1e-6)
