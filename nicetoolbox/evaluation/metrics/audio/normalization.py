"""Build jiwer transform chains for the transcription normalization matrix."""

import re
from typing import Literal

import jiwer

# Filler disfluencies removed from BOTH sides when `remove_filler=True`..
_FILLER_REGEXES = {
    r"(?i)\[\s*u[mh]\s*\]": " ",
    r"(?i)\b(?:um|uh|uhm|uhh|erm|hmm|ähm|äh)\b": " ",
}

# When `remove_filler=False` (verbatim) we keep fillers but map CrisperWhisper's bracketed tokens
# onto the annotation convention (hypothesis side only) so formatting differences aren't scored as
# substitutions. Override via the metric's `filler_map` config.
_DEFAULT_FILLER_MAP = {"[UM]": "um", "[UH]": "uh"}

# Consistent contraction expansion applied to both sides when `expand_contractions=True`.
_CONTRACTION_REGEXES = {
    r"(?i)\bwon't\b": "will not",
    r"(?i)\bcan't\b": "cannot",
    r"(?i)n't\b": " not",  # no leading \b: must match inside "don't"/"isn't"
    r"(?i)\b(i)'m\b": r"\1 am",
    r"(?i)\b(it|he|she|that|there|what|who|here)'s\b": r"\1 is",
    r"(?i)\b(i|you|we|they)'re\b": r"\1 are",
    r"(?i)\b(i|you|we|they|he|she)'ll\b": r"\1 will",
    r"(?i)\b(i|you|we|they|he|she)'ve\b": r"\1 have",
    r"(?i)\b(i|you|we|they|he|she)'d\b": r"\1 would",
}

# Strip digit-grouping separators (1,000 -> 1000) when `normalize_numbers=True`.
_NUMBER_REGEXES = {r"(\d)[,](\d)": r"\1\2"}


def _filler_transforms(side: str, remove_filler: bool, filler_map: dict[str, str]) -> list:
    if remove_filler:
        return [jiwer.SubstituteRegexes(dict(_FILLER_REGEXES))]
    # verbatim: only the hypothesis carries CrisperWhisper bracket tokens that need remapping.
    # Use regex substitution (escaping the token) because jiwer's word-level SubstituteWords does
    # not match bracketed tokens like "[UH]".
    if side == "hypothesis":
        mapping = dict(filler_map) if filler_map else dict(_DEFAULT_FILLER_MAP)
        if mapping:
            return [jiwer.SubstituteRegexes({re.escape(k): v for k, v in mapping.items()})]
    return []


def build_transform(
    side: Literal["reference", "hypothesis"],
    level: Literal["word", "char"] = "word",
    *,
    remove_filler: bool,
    lower_case: bool,
    strip_punctuation: bool,
    expand_contractions: bool,
    normalize_numbers: bool,
    filler_map: dict[str, str] | None = None,
) -> jiwer.Compose:
    """Assemble the jiwer `Compose` chain for one side and one granularity.

    Args:
        side: `"reference"` or `"hypothesis"` - the two can differ (only the hypothesis needs the
            CrisperWhisper -> annotation filler remap in the verbatim regime).
        level: `"word"` (terminates with `ReduceToListOfListOfWords`, for `process_words`) or
            `"char"` (terminates with `ReduceToListOfListOfChars`, for `process_characters`).
        remove_filler: drop disfluencies from both sides instead of remapping them.
        lower_case: fold case.
        strip_punctuation: drop punctuation.
        expand_contractions: rewrite "won't" as "will not" etc. on both sides.
        normalize_numbers: strip digit-grouping separators (1,000 -> 1000).
        filler_map: optional override of the verbatim CrisperWhisper -> annotation token mapping.

    Returns:
        A `jiwer.Compose` ready to pass as `reference_transform` / `hypothesis_transform`.
    """
    steps: list = []

    # 1. fillers first - before punctuation removal would strip the brackets.
    steps += _filler_transforms(side, remove_filler, filler_map or {})

    # 2. contractions and numbers while apostrophes are still present.
    if expand_contractions:
        steps.append(jiwer.SubstituteRegexes(dict(_CONTRACTION_REGEXES)))
    if normalize_numbers:
        steps.append(jiwer.SubstituteRegexes(dict(_NUMBER_REGEXES)))

    # 3. punctuation, 4. case.
    if strip_punctuation:
        steps.append(jiwer.RemovePunctuation())
    if lower_case:
        steps.append(jiwer.ToLowerCase())

    # 5. whitespace cleanup, then terminal reduction.
    steps.append(jiwer.RemoveMultipleSpaces())
    steps.append(jiwer.Strip())
    steps.append(jiwer.ReduceToListOfListOfWords() if level == "word" else jiwer.ReduceToListOfListOfChars())

    return jiwer.Compose(steps)
