"""Precomputed NLLB token lengths.

Length filtering happens on every run and in every arm of the matrix. Tokenising
600k texts each time is minutes of GPU-idle wall clock for a number that never
changes, so it is computed once here and stored as a column.

The count is the length of ``tokenizer(text).input_ids`` with the source language
set -- i.e. what the model actually sees, including the language token and EOS,
not a whitespace approximation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MK = "mkd_Cyrl"
SQ = "als_Latn"


def load_tokenizer(model_name: str, lang: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, src_lang=lang)


def token_lengths(
    texts: pd.Series,
    *,
    model_name: str,
    lang: str,
    batch_size: int = 512,
    progress: bool = True,
) -> pd.Series:
    """Token count per text, in the same order and index as `texts`."""
    from tqdm import tqdm

    tokenizer = load_tokenizer(model_name, lang)
    values = texts.fillna("").tolist()
    lengths = np.zeros(len(values), dtype="int32")

    batches = range(0, len(values), batch_size)
    for start in tqdm(batches, desc=f"tokenise {lang}", disable=not progress, unit="batch"):
        chunk = values[start : start + batch_size]
        encoded = tokenizer(chunk, add_special_tokens=True, truncation=False)["input_ids"]
        lengths[start : start + len(chunk)] = [len(ids) for ids in encoded]

    return pd.Series(lengths, index=texts.index, name=f"{lang}_tokens")


def assert_language_codes(model_name: str) -> None:
    """Both codes must already be in the tokenizer -- no vocabulary surgery."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = set(getattr(tokenizer, "additional_special_tokens", []) or [])
    lang_codes = set(getattr(tokenizer, "lang_code_to_id", {}) or {}) | vocab
    missing = [code for code in (MK, SQ) if code not in lang_codes]
    if missing:
        raise RuntimeError(f"{model_name} tokenizer has no language code for {missing}")
