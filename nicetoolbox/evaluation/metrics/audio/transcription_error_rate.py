"""WER/CER of one predicted transcript track against one reference track."""

import logging

import jiwer
import pandas as pd

from nicetoolbox_core.data.json_schema import JsonMeta

from ....configs.schemas.evaluation_metrics_config import TranscriptionErrorRateConfig
from ...data import plots
from ...data.transcript_loader import load_transcript
from ..base_metric import BaseMetric
from ..metric_result import MetricResult, PlotResult, SummaryResult
from . import normalization


class TranscriptionErrorRateMetric(BaseMetric):
    """WER/CER (and MER/WIL/WIP) for one transcript track against one reference track."""

    metric_config: TranscriptionErrorRateConfig

    def _init_metric(self) -> None:
        cfg = self.metric_config
        kwargs = dict(
            remove_filler=cfg.remove_filler,
            lower_case=cfg.lower_case,
            strip_punctuation=cfg.strip_punctuation,
            expand_contractions=cfg.expand_contractions,
            normalize_numbers=cfg.normalize_numbers,
            filler_map=cfg.filler_map,
        )
        # Build the four chains once. Reference and hypothesis differ: only the hypothesis carries
        # CrisperWhisper's bracketed filler tokens that need remapping in the verbatim regime.
        self._ref_word = normalization.build_transform("reference", "word", **kwargs)
        self._hyp_word = normalization.build_transform("hypothesis", "word", **kwargs)
        self._ref_char = normalization.build_transform("reference", "char", **kwargs)
        self._hyp_char = normalization.build_transform("hypothesis", "char", **kwargs)

    def compute(self) -> MetricResult:
        cfg = self.metric_config
        pred_meta, pred_track = load_transcript(cfg.predictions)
        gt_meta, gt_track = load_transcript(cfg.ground_truth)

        self._warn_on_mismatch(pred_meta, gt_meta)

        reference, hypothesis = gt_track.total.text, pred_track.total.text
        words = jiwer.process_words(
            reference, hypothesis, reference_transform=self._ref_word, hypothesis_transform=self._hyp_word
        )
        chars = jiwer.process_characters(
            reference, hypothesis, reference_transform=self._ref_char, hypothesis_transform=self._hyp_char
        )

        ref_words = words.hits + words.substitutions + words.deletions
        if ref_words == 0:
            # jiwer does not raise here, it returns a rate divided by an empty reference. Fail loudly
            # instead: this also catches a reference that normalizes to empty, e.g. "um" with
            # remove_filler = true.
            raise ValueError(
                f"[{self.metric_name}] ground-truth track '{cfg.ground_truth.track}' in "
                f"'{cfg.ground_truth.path}' is empty after normalization; error rates are undefined."
            )

        detail = pd.DataFrame([self._build_row(pred_meta, words, chars, ref_words)])
        figures = self._visualize(detail) if cfg.visualize else {}

        return MetricResult(
            self.metric_name,
            summary=SummaryResult({"summary": detail}),
            plots=PlotResult(figures),
        )

    def _warn_on_mismatch(self, pred_meta: JsonMeta, gt_meta: JsonMeta) -> None:
        """Warn about pairings that are valid config but almost certainly a mistake."""
        cfg = self.metric_config
        if cfg.predictions.track != cfg.ground_truth.track:
            logging.warning(
                f"[{self.metric_name}] scoring prediction track '{cfg.predictions.track}' against "
                f"reference track '{cfg.ground_truth.track}' - these are different audio sources."
            )
        pred_window = (pred_meta.subsequence_start, pred_meta.subsequence_length)
        gt_window = (gt_meta.subsequence_start, gt_meta.subsequence_length)
        if pred_window != gt_window:
            logging.warning(
                f"[{self.metric_name}] subsequence windows differ: prediction covers "
                f"start={pred_window[0]}s length={pred_window[1]}s but reference covers "
                f"start={gt_window[0]}s length={gt_window[1]}s. Error rates across different "
                f"time spans are not meaningful."
            )

    def _build_row(
        self,
        pred_meta: JsonMeta,
        words: jiwer.WordOutput,
        chars: jiwer.CharacterOutput,
        ref_words: int,
    ) -> dict:
        """
        Assemble the single result row: identity, selected measures, then word counts.

        Only the prediction is identified: the reference is the same for every row being compared,
        and a mismatched track or window is already reported by _warn_on_mismatch.
        """
        cfg = self.metric_config
        available = {
            "wer": words.wer,
            "mer": words.mer,
            "wil": words.wil,
            "wip": words.wip,
            "cer": chars.cer,
        }
        return {
            # `metric` is what makes several single-row CSVs concatenable + comparible
            "metric": self.metric_name,
            "pred_component": pred_meta.component,
            "pred_algorithm": pred_meta.algorithm,
            "pred_track": cfg.predictions.track,
            **{name: value for name, value in available.items() if name in cfg.measures},
            "hits": words.hits,
            "substitutions": words.substitutions,
            "deletions": words.deletions,
            "insertions": words.insertions,
            "ref_words": ref_words,
            "hyp_words": words.hits + words.substitutions + words.insertions,
        }

    def _visualize(self, detail: pd.DataFrame) -> dict:
        """
        Error composition is the only chart that says anything about a single pair.

        Bar charts, per-track grouping and heatmaps all need at least two rows to compare.
        Deferred for now.
        """
        return {
            "error_composition": plots.plot_error_composition(
                detail,
                group_col="metric",
                title=f"{self.metric_name}: error composition (S/D/I per reference word)",
            )
        }
