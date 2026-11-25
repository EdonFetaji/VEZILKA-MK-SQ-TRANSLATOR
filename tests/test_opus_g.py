"""Tag G (TG-D0) for TG-10: selection, freeze, select(), native prompt, mixture, branch stream."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from src.datasets import opus
from src.datasets.opus import G_HUMAN, OpusLoader, freeze_g, select_g
from src.pipeline import mixture
from src.pipeline.config import load_config
from src.train_tasks.translation import TranslationTask

CONFIGS = Path(__file__).resolve().parents[1] / "src" / "configs"


# ── select_g (pure) ───────────────────────────────────────────────────────────


def arrays(n_nllb=50, n_hplt=30, n_subs=40):
    src = (["SETIMES"] * 10 + ["TED2020"] * 4 + ["NLLB"] * n_nllb + ["MultiHPLT"] * n_hplt
           + ["MultiMaCoCu"] * 6 + ["OpenSubtitles"] * n_subs + ["QED"] * 5 + ["WikiMatrix"] * 3)
    sources = np.array(src, dtype=object)
    n = len(sources)
    rng = np.random.default_rng(0)
    scores = np.full(n, np.nan, dtype=np.float32)
    mined = np.isin(sources, ["NLLB", "MultiHPLT", "MultiMaCoCu"])
    scores[mined] = rng.uniform(0.6, 1.0, mined.sum())
    held = np.zeros(n, dtype=bool)
    subs_ok = (sources == "OpenSubtitles") & (np.arange(n) % 2 == 0)
    return sources, scores, held, subs_ok


def test_select_g_rules(monkeypatch):
    monkeypatch.setattr(opus, "G_MINED_TOP", {"NLLB": 20, "MultiHPLT": 10, "MultiMaCoCu": None})
    monkeypatch.setattr(opus, "G_SUBS_ROWS", 25)
    sources, scores, held, subs_ok = arrays()
    held[0] = True  # one SETIMES row repeats a dev sentence
    rows, dirs, stats = select_g(sources, scores, held, subs_ok, seed=1)

    assert stats["SETIMES"]["selected"] == 9 and 0 not in rows  # held-out removed
    for src, top in (("NLLB", 20), ("MultiHPLT", 10)):
        picked = np.unique(rows[sources[rows] == src])
        assert len(picked) <= top and (scores[picked] >= 0.80).all()
        cand = np.flatnonzero((sources == src) & (scores >= 0.80))
        assert set(picked) == set(cand[np.argsort(-scores[cand])][: len(picked)])  # best first
    macocu = np.unique(rows[sources[rows] == "MultiMaCoCu"])
    assert set(macocu) == set(np.flatnonzero((sources == "MultiMaCoCu") & (scores >= 0.80)))
    subs = np.unique(rows[sources[rows] == "OpenSubtitles"])
    assert len(subs) == 25 and subs_ok[subs].sum() == 20  # all 20 preferred rows, then 5 others
    assert not np.isin(sources[rows], ["QED", "WikiMatrix"]).any()

    human = np.isin(sources[rows], G_HUMAN)
    for r in np.unique(rows[human]):  # human rows: both directions exactly once
        assert sorted(dirs[rows == r]) == [0, 1]
    one = ~human
    assert len(np.unique(rows[one])) == one.sum()  # mined/subs: one example per row
    assert abs(int(dirs[one].sum()) - one.sum() // 2) <= 1  # 50/50
    n_human, n_one = len(np.unique(rows[human])), len(np.unique(rows[one]))
    assert stats["_total"]["examples"] == len(rows) == 2 * n_human + n_one
    assert stats["_total"]["rows"] == n_human + n_one
    again = select_g(sources, scores, held, subs_ok, seed=1)
    assert (again[0] == rows).all() and (again[1] == dirs).all()  # deterministic
    other = select_g(sources, scores, held, subs_ok, seed=2)
    assert not (other[0] == rows).all()  # the seed changes the order


# ── freeze_g end to end on a tiny fake repository cache ───────────────────────


def fake_cache(tmp_path: Path, monkeypatch) -> OpusLoader:
    monkeypatch.setattr(opus, "G_MINED_TOP", {"NLLB": 3, "MultiHPLT": 2, "MultiMaCoCu": None})
    monkeypatch.setattr(opus, "G_SUBS_ROWS", 2)
    monkeypatch.setattr(opus, "SCORE_BLOCK", 4)  # several cached parts
    loader = OpusLoader(cache_dir=str(tmp_path))
    src = ["SETIMES"] * 3 + ["NLLB"] * 6 + ["MultiHPLT"] * 4 + ["MultiMaCoCu"] * 2 + ["OpenSubtitles"] * 4 + ["QED"]
    n = len(src)
    mk = [f"мк реченица број {i}" for i in range(n)]
    mk[0] = "ова е dev реченица"
    tbl = pa.table({"mk": mk, "sq": [f"fjali shqip numër {i}" for i in range(n)], "source_id": src,
                    "doc_id": [f"p{i}" for i in range(n)], "meta": ["{}"] * n})
    path = loader.path("train")
    path.parent.mkdir(parents=True)
    with pa.OSFile(str(path), "wb") as f, pa.ipc.new_stream(f, tbl.schema) as w:
        w.write_table(tbl, max_chunksize=5)
    path.with_suffix(".json").write_text(json.dumps({"rows": n, "revision": "abc123def4567890", "sha256": "x"}))
    return loader


class FakeScorer:
    name = "fake-labse"

    def __init__(self):
        self.calls = 0

    def __call__(self, mk, sq):
        self.calls += 1
        return np.array([0.5 + (int(s.split()[-1]) % 5) / 10 for s in sq], dtype=np.float32)


def test_freeze_and_select(tmp_path, monkeypatch):
    loader = fake_cache(tmp_path, monkeypatch)
    scorer = FakeScorer()
    out = freeze_g(loader, seed=1, scorer=scorer, held_out={opus.norm("Ова е DEV   реченица")}, log=lambda *_: None)
    side = json.loads(out.with_suffix(".json").read_text())
    assert side["sources"]["SETIMES"]["selected"] == 2  # row 0 repeats a dev sentence
    assert side["examples"] == 2 * 2 + side["sources"]["NLLB"]["selected"] + side["sources"]["MultiHPLT"][
        "selected"] + side["sources"]["MultiMaCoCu"]["selected"] + 2
    assert scorer.calls == 3  # 12 mined rows in blocks of 4

    # cached scores: a rebuild needs no scorer calls
    again = FakeScorer()
    freeze_g(loader, seed=1, scorer=again, held_out={opus.norm("ова е dev реченица")}, force=True,
             log=lambda *_: None)
    assert again.calls == 0

    ds = loader.select(None, tag="G")
    assert len(ds) == side["examples"]
    assert {"mk", "sq", "source_id", "doc_id", "meta", "direction", "labse", "example_id"} <= set(ds.column_names)
    assert set(ds["direction"]) <= {"mk-sq", "sq-mk"} and "QED" not in set(ds["source_id"])
    assert ds["example_id"] == [f"G:{i:07d}" for i in range(len(ds))]
    only = loader.select(None, tag="G", source_ids=["SETIMES"])
    assert len(only) == 4 and set(only["source_id"]) == {"SETIMES"}
    assert len(loader.select(None, tag="G", stream_from=2, stream_to=5)) == 3


def test_select_needs_freeze(tmp_path):
    with pytest.raises(FileNotFoundError, match="freeze G"):
        OpusLoader(cache_dir=str(tmp_path)).select(None, tag="G")


# ── the native TranslateGemma prompt ──────────────────────────────────────────


class FakeTokenizer:
    bos_token, bos_token_id, eos_token = "<bos>", 2, "<eos>"

    def __call__(self, text):
        texts = [text] if isinstance(text, str) else text
        ids = [[2, *range(10, 10 + len(t.split()))] for t in texts]
        return {"input_ids": ids[0] if isinstance(text, str) else ids}


class FakeProcessor:
    """Stands in for TranslateGemma's processor: records what reaches apply_chat_template."""

    def __init__(self):
        self.tokenizer, self.seen = FakeTokenizer(), []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.seen.append((messages, tokenize, add_generation_prompt))
        c = messages[0]["content"][0]
        return (f"<bos><start_of_turn>user\n[{c['source_lang_code']}->{c['target_lang_code']}] {c['text']}"
                "<end_of_turn>\n<start_of_turn>model\n")


def test_translation_uses_native_template():
    proc = FakeProcessor()
    ex = TranslationTask(proc).format({"mk": "Добар ден", "sq": "Mirëdita"}, "mk-sq")
    messages, tokenize, gen = proc.seen[0]
    assert messages == [{"role": "user", "content": [{"type": "text", "source_lang_code": "mk",
                                                      "target_lang_code": "sq", "text": "Добар ден"}]}]
    assert tokenize is False and gen is True
    assert not ex["prompt"].startswith("<bos>")  # the tokenizer adds the one <bos>
    assert ex["prompt"].endswith("<start_of_turn>model\n") and ex["completion"] == "Mirëdita<end_of_turn>"
    back = TranslationTask(proc).format({"mk": "Добар ден", "sq": "Mirëdita"}, "sq-mk")
    assert proc.seen[-1][0][0]["content"][0]["source_lang_code"] == "sq"
    assert back["completion"] == "Добар ден<end_of_turn>"


# ── mixture: frozen order kept, over-length dropped and counted ───────────────


def test_mixture_frozen_g(tmp_path, monkeypatch):
    loader = fake_cache(tmp_path, monkeypatch)
    freeze_g(loader, seed=1, scorer=FakeScorer(), held_out=set(), log=lambda *_: None)
    cfg = load_config(CONFIGS / "TG-10.yaml")
    entry = cfg.data.mixture[0]
    monkeypatch.setattr(mixture, "get_loader", lambda name: lambda **kw: OpusLoader(cache_dir=str(tmp_path)))
    ds, manifest = mixture.build_train_dataset(cfg, FakeProcessor())
    frozen = loader.select(None, tag="G")
    assert len(ds) == len(frozen) and set(ds.column_names) == {"prompt", "completion"}
    for i in (0, len(ds) // 2, len(ds) - 1):  # same order, direction from the frozen file
        d = frozen[i]["direction"]
        assert ds[i]["completion"] == frozen[i][d.split("-")[1]] + "<end_of_turn>"
    assert manifest["entries"][0]["frozen"]["tag"] == "G" and entry.directions == "frozen"

    short = dataclass_replace_max_length(cfg, 8)  # fake tokens = words + 1: long examples go
    ds2, m2 = mixture.build_train_dataset(short, FakeProcessor())
    dropped = m2["entries"][0]["dropped_over_length"]
    assert sum(dropped.values()) == len(frozen) - len(ds2) > 0


def dataclass_replace_max_length(cfg, n):
    import dataclasses

    return dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, max_length=n))


# ── branch runs continue the parent's packed stream ───────────────────────────


def test_continue_stream_skips_parent_steps():
    from src.pipeline.trainer import continue_stream

    cfg = load_config(CONFIGS / "TG-10-CD06.yaml")
    class DS(list):
        def select(self, idx):
            return DS(self[i] for i in idx)

    trainer = types.SimpleNamespace(train_dataset=DS(range(200_000)))
    skip = continue_stream(cfg, trainer)
    assert skip == 1196 * 4 * 8 and trainer.train_dataset[0] == skip
    plain = types.SimpleNamespace(train_dataset=DS(range(10)))
    assert continue_stream(load_config(CONFIGS / "TG-10.yaml"), plain) == 0 and len(plain.train_dataset) == 10
