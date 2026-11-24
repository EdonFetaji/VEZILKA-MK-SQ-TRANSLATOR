"""Character-level repair of OCR substitutions in the pair texts.

Two defects reach the corpus from the extraction pipeline and, left alone, are
*learned* by any model fine-tuned on it.

**`[` for `ë` (Albanian side).** The evidence that this is a substitution and not
a bracket, measured over all 295,628 pairs:

* `[` outnumbers `]` 27,035 to 233 -- 116:1. On the Macedonian side, where
  brackets are genuine, the ratio is 7:7.
* 91% of `[` are immediately preceded by a letter, i.e. word-internal.
* 98.7% of the words containing `[` resolve, on substituting `ë`, to a word
  attested elsewhere in the corpus.
* The most frequent are the language's most frequent function words:
  `t[`->`të`, `n[`->`në`, `p[r`->`për`, `s[`->`së`.

**Cyrillic homoglyphs inside Latin words (Albanian side).** `Е` (U+0415) for `E`,
`а` (U+0430) for `a`, and so on -- the same defect `scripts/verbis_harvest.py`
already handles for the APJ dictionary, and handled here with the same idiom.
Applied only inside words that are *mixed-script*, so an Albanian sentence quoting
a Macedonian title in Cyrillic is left alone.

Both repairs are conservative: a `[` that does not touch a letter is left as a
bracket, and a word written wholly in one script is never touched.
"""

from __future__ import annotations

import re

# --- `[` for `ë` -------------------------------------------------------------
LATIN_LETTER = r"A-Za-zÇçËë"

# A `[` that touches a Latin letter on either side. Word-initial `[sht[` -> `është`
# and word-final `t[` -> `të` both qualify; a standalone `[1]` does not.
BRACKET_AS_E = re.compile(rf"(?<=[{LATIN_LETTER}])\[|\[(?=[{LATIN_LETTER}])")

# --- Cyrillic homoglyphs for Latin letters -----------------------------------
# Same table as scripts/verbis_harvest.py, kept in step with it.
CYR_TO_LAT = str.maketrans("АВЕКМНОРСТУХаеорсухіјѕЁё", "ABEKMHOPCTYXaeopcyxijsËë")

WORD = re.compile(rf"[{LATIN_LETTER}Ѐ-ӿ]+")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(rf"[{LATIN_LETTER}]")


def repair_bracket_e(text: str) -> str:
    """`N[ paragrafin 2 pas fjal[ve` -> `Në paragrafin 2 pas fjalëve`."""
    if "[" not in text:
        return text
    return BRACKET_AS_E.sub("ë", text)


def repair_mixed_script(text: str) -> str:
    """Fold Cyrillic homoglyphs to Latin, but only inside mixed-script words."""

    def fix(match: re.Match) -> str:
        word = match.group(0)
        cyrillic, latin = len(CYRILLIC.findall(word)), len(LATIN.findall(word))
        if cyrillic and latin:            # mixed: the Cyrillic is homoglyph noise
            return word.translate(CYR_TO_LAT)
        return word                       # wholly one script: a real quotation

    return WORD.sub(fix, text)


def repair_sq(text: str, *, mixed_script: bool = True) -> str:
    """The Albanian side, repaired. `mixed_script=False` does the `[` fix only."""
    if text is None:
        return text
    out = repair_bracket_e(text)
    return repair_mixed_script(out) if mixed_script else out


def residual_brackets(text: str) -> int:
    """`[` left after the repair -- these are treated as genuine brackets."""
    return repair_bracket_e(text).count("[")
