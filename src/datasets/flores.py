"""FLORES+ mkd_Cyrl–als_Latn (evaluation only)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from src.datasets.base import DatasetLoader, RecordKind
from src.pipeline.registry import register_loader

LANGS = {"mk": "mkd_Cyrl", "sq": "als_Latn"}


@register_loader("flores")
class FloresLoader(DatasetLoader):
    """FLORES+ at a pinned revision. ``dev`` (997, FLORES_DEV, selection) and ``devtest``
    (1,012, FLORES_DEVTEST, the primary general-quality test). Never trained on.

    Gated: accept the terms on Hugging Face with the account whose token is used. The two sides are
    joined on ``id``. ``doc_id`` is the source article URL, so sentences of one article bootstrap
    together.
    """

    kind = RecordKind.PARALLEL
    splits = ("dev", "devtest")
    repo = "openlanguagedata/flores_plus"
    revision = "5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06"

    def _side(self, split: str, lang: str) -> dict[int, dict]:
        from huggingface_hub import HfApi, hf_hub_download

        repo, rev = self.opt("repo"), self.opt("revision")
        files = HfApi().list_repo_files(repo, repo_type="dataset", revision=rev)
        cands = [f for f in files if f.split("/")[0] == split and Path(f).name.split(".")[0] == lang]
        if not cands:
            raise FileNotFoundError(f"{repo}@{rev}: no {split} file for {lang}")
        name = min(cands, key=lambda f: (not f.endswith(".parquet"), f))
        local = hf_hub_download(repo, name, repo_type="dataset", revision=rev)
        if local.endswith(".parquet"):
            rows = pq.read_table(local).to_pylist()
        else:
            rows = [json.loads(line) for line in Path(local).read_text(encoding="utf-8").splitlines() if line]
        self._source.setdefault("files", []).append(name)
        return {int(r["id"]): r for r in rows}

    def _read(self, split: str):
        self._source = {"repo": self.opt("repo"), "revision": self.opt("revision")}
        mk, sq = self._side(split, LANGS["mk"]), self._side(split, LANGS["sq"])
        if mk.keys() != sq.keys():
            raise ValueError(f"FLORES+ {split}: the two sides have different ids")
        ids = sorted(mk)
        yield {
            "mk": [mk[i]["text"] for i in ids],
            "sq": [sq[i]["text"] for i in ids],
            "source_id": ["flores"] * len(ids),
            "doc_id": [str(mk[i].get("url") or i) for i in ids],
            "meta": [
                json.dumps(
                    {"id": i, "domain": mk[i].get("domain"), "topic": mk[i].get("topic")}, ensure_ascii=False
                )
                for i in ids
            ],
        }
