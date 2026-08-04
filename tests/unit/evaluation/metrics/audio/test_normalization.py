"""Tests for the jiwer transform chains behind the normalization matrix."""

import jiwer
import pytest

from nicetoolbox.evaluation.metrics.audio import normalization
from tests.unit.evaluation.metrics.audio.conftest import LENIENT_FLAGS


def wer(reference: str, hypothesis: str, **overrides) -> float:
    """Score one pair through both sides' word chains, with the lenient preset as baseline."""
    flags = {**LENIENT_FLAGS, **overrides}
    return jiwer.process_words(
        reference,
        hypothesis,
        reference_transform=normalization.build_transform("reference", "word", **flags),
        hypothesis_transform=normalization.build_transform("hypothesis", "word", **flags),
    ).wer


def cer(reference: str, hypothesis: str, **overrides) -> float:
    flags = {**LENIENT_FLAGS, **overrides}
    return jiwer.process_characters(
        reference,
        hypothesis,
        reference_transform=normalization.build_transform("reference", "char", **flags),
        hypothesis_transform=normalization.build_transform("hypothesis", "char", **flags),
    ).cer


# ---------------------------------------------------------------------------
# Filler handling
# ---------------------------------------------------------------------------


class TestFiller:
    def test_removed_from_both_sides(self):
        assert wer("um hello world", "hello [UH] world", remove_filler=True) == 0.0

    def test_bracket_tokens_remapped_when_kept(self):
        """Verbatim: the hypothesis' [UM] is remapped so it matches the reference's plain 'um'."""
        assert wer("um hello world", "[UM] hello world", remove_filler=False) == 0.0

    def test_custom_filler_map_overrides_default(self):
        assert wer("erm hello", "[UM] hello", remove_filler=False, filler_map={"[UM]": "erm"}) == 0.0


# ---------------------------------------------------------------------------
# Case and punctuation
# ---------------------------------------------------------------------------


class TestCase:
    def test_folded_when_enabled(self):
        assert wer("Hello World", "hello world", lower_case=True) == 0.0

    def test_preserved_when_disabled(self):
        assert wer("Hello World", "hello world", lower_case=False) == 1.0


class TestPunctuation:
    def test_stripped_when_enabled(self):
        assert wer("hello, world!", "hello world", strip_punctuation=True) == 0.0

    def test_kept_when_disabled(self):
        assert wer("hello, world", "hello world", strip_punctuation=False, lower_case=True) > 0.0


# ---------------------------------------------------------------------------
# The two axes split out of the old single `normalize_numbers` flag
# ---------------------------------------------------------------------------


class TestContractions:
    def test_expanded_consistently(self):
        assert wer("do not go", "don't go", expand_contractions=True) == 0.0

    def test_substitution_when_disabled(self):
        """Without expansion, "dont" (post punctuation-strip) mismatches "do not": 1 sub + 1 del."""
        assert wer("do not go", "don't go", expand_contractions=False) == pytest.approx(2 / 3)


class TestNumbers:
    def test_thousands_separator_stripped(self):
        assert wer("we had 1000 items", "we had 1,000 items", normalize_numbers=True) == 0.0

    def test_separator_kept_when_disabled(self):
        """Punctuation stripping turns "1,000" into "1000" too, so isolate it to see the flag."""
        assert wer(
            "we had 1000 items",
            "we had 1,000 items",
            normalize_numbers=False,
            strip_punctuation=False,
        ) == pytest.approx(1 / 4)

    def test_independent_of_contractions(self):
        """Numbers normalize even with contraction expansion off - the axes are separate."""
        assert wer("1000", "1,000", normalize_numbers=True, expand_contractions=False) == 0.0


# ---------------------------------------------------------------------------
# Character level
# ---------------------------------------------------------------------------


class TestCharLevel:
    def test_cer_counts_characters(self):
        assert cer("abc", "abd") == pytest.approx(1 / 3)

    def test_cer_zero_for_normalized_match(self):
        assert cer("Hello, World", "hello world") == 0.0
