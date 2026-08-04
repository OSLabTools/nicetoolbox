"""Tests for the transcription error rate metric."""

import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nicetoolbox.configs.schemas.evaluation_metrics_config import TranscriptionErrorRateConfig
from nicetoolbox.evaluation.data.results_saver import save_results
from nicetoolbox.evaluation.metrics.audio.transcription_error_rate import TranscriptionErrorRateMetric
from tests.unit.evaluation.metrics.audio.conftest import LENIENT_FLAGS, write_transcript


def make_metric(pred_path, gt_path, *, track="room", gt_track=None, name="wer_test", **overrides):
    """Build a configured metric. The ConfigHandler is unused by audio metrics."""
    config = TranscriptionErrorRateConfig(
        metric_type="transcription_error_rate",
        predictions={"path": pred_path, "track": track},
        ground_truth={"path": gt_path, "track": gt_track or track},
        **{**LENIENT_FLAGS, **overrides},
    )
    config._metric_name = name
    return TranscriptionErrorRateMetric(config, MagicMock())


def summary_row(result):
    return result.summary.summaries["summary"].iloc[0]


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------


class TestMeasures:
    def test_identical_transcripts_score_zero(self, tmp_path):
        pred = write_transcript(tmp_path, "pred", {"room": "hello world"})
        gt = write_transcript(tmp_path, "gt", {"room": "hello world"})

        row = summary_row(make_metric(pred, gt).compute())

        assert row["wer"] == 0.0
        assert row["cer"] == 0.0
        assert (row["substitutions"], row["deletions"], row["insertions"]) == (0, 0, 0)
        assert row["ref_words"] == 2

    def test_counts_and_rates_match_hand_calculation(self, tmp_path):
        # reference: 4 words; hypothesis substitutes "brown" and drops "dog" -> 1 sub + 1 del
        pred = write_transcript(tmp_path, "pred", {"room": "the quick red"})
        gt = write_transcript(tmp_path, "gt", {"room": "the quick brown dog"})

        row = summary_row(make_metric(pred, gt).compute())

        assert row["ref_words"] == 4
        assert row["substitutions"] == 1
        assert row["deletions"] == 1
        assert row["insertions"] == 0
        assert row["wer"] == pytest.approx(0.5)

    def test_normalization_flags_change_the_score(self, tmp_path):
        """Fillers kept: the hypothesis' remapped "um" is an insertion against a clean reference."""
        pred = write_transcript(tmp_path, "pred", {"room": "[UM] hello world"})
        gt = write_transcript(tmp_path, "gt", {"room": "hello world"})

        lenient = summary_row(make_metric(pred, gt, remove_filler=True).compute())
        verbatim = summary_row(make_metric(pred, gt, remove_filler=False).compute())

        assert lenient["wer"] == 0.0
        assert verbatim["wer"] > lenient["wer"]


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


class TestSummary:
    def test_single_row_identifying_the_prediction(self, tmp_path):
        pred = write_transcript(tmp_path, "pred", {"room": "hello"}, algorithm="crisper_whisper")
        gt = write_transcript(tmp_path, "gt", {"room": "hello"}, algorithm="elan")

        df = make_metric(pred, gt, name="my_metric").compute().summary.summaries["summary"]

        assert len(df) == 1
        row = df.iloc[0]
        assert row["metric"] == "my_metric"  # what makes several 1-row CSVs concatenable
        assert row["pred_algorithm"] == "crisper_whisper"
        assert row["pred_component"] == "audio_transcription"

    def test_reference_side_is_not_described(self, tmp_path):
        """Only the prediction is identified - reference provenance and paths would add width only."""
        pred = write_transcript(tmp_path, "pred", {"room": "hello"})
        gt = write_transcript(tmp_path, "gt", {"room": "hello"}, algorithm="elan")

        df = make_metric(pred, gt).compute().summary.summaries["summary"]

        assert not {"gt_component", "gt_algorithm", "gt_track", "pred_path", "gt_path"} & set(df.columns)

    def test_measures_filter_rates_but_keep_counts(self, tmp_path):
        pred = write_transcript(tmp_path, "pred", {"room": "hello"})
        gt = write_transcript(tmp_path, "gt", {"room": "hello world"})

        df = make_metric(pred, gt, measures=["wer"]).compute().summary.summaries["summary"]

        assert "wer" in df.columns
        assert not {"cer", "mer", "wil", "wip"} & set(df.columns)
        assert {"hits", "substitutions", "deletions", "insertions", "ref_words"} <= set(df.columns)

    def test_no_frame_arrays_produced(self, tmp_path):
        pred = write_transcript(tmp_path, "pred", {"room": "hello"})
        gt = write_transcript(tmp_path, "gt", {"room": "hello"})

        assert make_metric(pred, gt).compute().frames is None


# ---------------------------------------------------------------------------
# Inputs and errors
# ---------------------------------------------------------------------------


class TestErrors:
    def test_empty_reference_is_rejected(self, tmp_path):
        pred = write_transcript(tmp_path, "pred", {"room": "hello world"})
        gt = write_transcript(tmp_path, "gt", {"room": "   "})

        with pytest.raises(ValueError, match="undefined"):
            make_metric(pred, gt).compute()

    def test_reference_that_normalizes_to_empty_is_rejected(self, tmp_path):
        """jiwer returns a nonsense rate here rather than raising, so the guard must catch it."""
        pred = write_transcript(tmp_path, "pred", {"room": "hello world"})
        gt = write_transcript(tmp_path, "gt", {"room": "um uh"})

        with pytest.raises(ValueError, match="undefined"):
            make_metric(pred, gt, remove_filler=True).compute()

    def test_missing_track_is_reported(self, tmp_path):
        pred = write_transcript(tmp_path, "pred", {"left_mic": "hello"})
        gt = write_transcript(tmp_path, "gt", {"room": "hello"})

        with pytest.raises(KeyError, match="Available"):
            make_metric(pred, gt).compute()

    def test_wrong_component_file_is_rejected(self, tmp_path):
        """A diarization JSON has no `total`, so it never validates as a transcription."""
        path = tmp_path / "diarization.json"
        path.write_text(json.dumps({"meta": {}, "tracks": {"room": {"segments": []}}}))
        gt = write_transcript(tmp_path, "gt", {"room": "hello"})

        with pytest.raises(ValidationError):
            make_metric(path, gt).compute()


class TestConfig:
    @pytest.mark.parametrize("omitted", sorted(LENIENT_FLAGS))
    def test_every_normalization_flag_is_mandatory(self, omitted, tmp_path):
        flags = {k: v for k, v in LENIENT_FLAGS.items() if k != omitted}

        with pytest.raises(ValidationError, match=omitted):
            TranscriptionErrorRateConfig(
                metric_type="transcription_error_rate",
                predictions={"path": tmp_path / "p.json", "track": "room"},
                ground_truth={"path": tmp_path / "g.json", "track": "room"},
                **flags,
            )


# ---------------------------------------------------------------------------
# Output artifacts
# ---------------------------------------------------------------------------


class TestOutputs:
    def test_visualization_toggle(self, tmp_path):
        pred = write_transcript(tmp_path, "pred", {"room": "hello"})
        gt = write_transcript(tmp_path, "gt", {"room": "hello world"})

        assert "error_composition" in make_metric(pred, gt, visualize=True).compute().plots.figures
        assert make_metric(pred, gt, visualize=False).compute().plots.figures == {}

    def test_results_are_written_to_disk(self, tmp_path):
        pred = write_transcript(tmp_path, "pred", {"room": "hello"})
        gt = write_transcript(tmp_path, "gt", {"room": "hello world"})
        out_dir = tmp_path / "out"

        save_results(make_metric(pred, gt, name="wer_room").compute(), out_dir)

        csv_path = out_dir / "wer_room" / "summary.csv"
        assert csv_path.exists()
        assert len(csv_path.read_text().strip().splitlines()) == 2  # header + one data row
        assert (out_dir / "wer_room" / "visualization" / "error_composition.png").exists()
