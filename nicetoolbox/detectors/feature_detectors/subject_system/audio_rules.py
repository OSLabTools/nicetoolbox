import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, List

from pydantic import ValidationError

from nicetoolbox_core.data.json_schema import AudioDiarization, DiarizationSegment, TrackDiarization

from ....configs.schemas.subject_rules import BaseRuleConfig
from ...detector_inputs import BaseDetectorInput, JsonDetectorInput
from ...subjects import assign_speakers_from_ranges, merge_ranges, range_coverage, subtract_ranges

# TODO: remove cycle? need this just for annotation
if TYPE_CHECKING:
    from .audio_subject_detector import AudioSubjectDetector


# =======================================
#           Base Audio Rule
# ========================================


class BaseAudioRule(ABC):
    rule_type: str  # subclass sets to the rule's discriminator string

    def __init__(self, detector: "AudioSubjectDetector", track_name: str, rule: BaseRuleConfig):
        self.detector = detector
        self.rule = rule
        self.track_name = track_name
        self.subjects = list(detector.subjects_descr)

        # segment timings are relative to the subsequence start
        self.duration = detector.subsequence_length

    @classmethod
    def required_inputs(cls, rule: BaseRuleConfig) -> List[BaseDetectorInput]:  # noqa: ARG003
        """
        Upstream inputs this rule needs, declared on the audio subject detector before it runs.

        Input names must be unique per (component, algorithm), named "<component>_<algorithm>": tracks
        whose rules read the same algorithm share one input.
        """
        return []

    @abstractmethod
    def compute(self) -> List[DiarizationSegment]:
        """
        Return the subject segments for this specific audio track.

        Each segment's speaker is a subject name; start/end are seconds relative to the
        subsequence start. Time not covered by any segment carries no subject.
        """
        ...


# =======================================
#           Simple Audio Rule
# ========================================


class NoneAudioRule(BaseAudioRule):
    """No subject is placed on this track; it has no segments."""

    rule_type = "none"

    def compute(self):
        return []


class AssignAudioRule(BaseAudioRule):
    """The whole track is assigned to one named subject."""

    rule_type = "assign"

    def compute(self):
        if self.rule.subject not in self.subjects:
            raise ValueError(
                f"audio_subject_detector: track '{self.track_name}' assigns unknown subject "
                f"'{self.rule.subject}'. Available: {self.subjects}."
            )
        return [DiarizationSegment(speaker=self.rule.subject, start=0.0, end=self.duration)]


# =======================================
#           External Audio Rule
# ========================================


class PathAudioRule(BaseAudioRule):
    """Reads subject-labeled turns from a track of an audio_diarization JSON on disk."""

    rule_type = "path"

    def compute(self):
        diarization = self._load()
        if self.rule.track not in diarization.tracks:
            raise ValueError(f"{self._where}: no track '{self.rule.track}'. Available: {list(diarization.tracks)}.")

        # file time base -> processed subsequence time base
        shift = diarization.meta.subsequence_start - self.detector.subsequence_start
        subject_times, unknown_speakers = {}, set()
        for segment in diarization.tracks[self.rule.track].segments:
            if segment.speaker not in self.subjects:
                unknown_speakers.add(segment.speaker)
                continue
            start = max(segment.start + shift, 0.0)
            end = min(segment.end + shift, self.duration)
            if end > start:
                subject_times.setdefault(segment.speaker, []).append((start, end))
        if unknown_speakers:
            logging.warning(
                f"{self._where}: speakers {sorted(unknown_speakers)} are not subjects of the sequence "
                f"{self.subjects}; their turns are dropped."
            )

        segments = [
            DiarizationSegment(speaker=subject, start=start, end=end)
            for subject, times in subject_times.items()
            for start, end in merge_ranges(times)
        ]
        return sorted(segments, key=lambda segment: (segment.start, segment.end))

    @property
    def _where(self) -> str:
        return f"audio_subject_detector: track '{self.track_name}' source '{self.rule.source}'"

    def _load(self) -> AudioDiarization:
        path = Path(self.rule.source)
        if not path.is_file():
            raise FileNotFoundError(f"{self._where}: file does not exist.")
        with open(path) as f:
            raw = json.load(f)
        try:
            return AudioDiarization.model_validate(raw)
        except ValidationError as e:
            raise ValueError(f"{self._where}: not an audio_diarization JSON:\n{e}") from None


# =======================================
#           Diarization Audio Rule
# ========================================


class AudioRangeRule(BaseAudioRule):
    """
    Assigns the anonymous speakers of a diarization to subjects, using per subject a list of ranges
    in which only that subject speaks.

    Speakers are matched to subjects by how much of each range they cover. The ranges themselves are
    known, so each is emitted as a segment of its subject even if the diarization found no speech there,
    and the turns are cut out of them; the rest of each turn gets its speaker's subject.
    """

    rule_type = "audio_range"

    @classmethod
    def required_inputs(cls, rule):
        component = "audio_diarization"
        return [JsonDetectorInput(component, f"{component}_{rule.algorithm}", schema=AudioDiarization)]

    def compute(self):
        # get diarization tracks from the algorithm (i.e. pyannote)
        diarization = self.detector.loaded_inputs[f"audio_diarization_{self.rule.algorithm}"]
        if self.track_name not in diarization.json.tracks:
            raise ValueError(
                f"{self._where}: '{self.rule.algorithm}' did not diarize this track. Diarized tracks: "
                f"{list(diarization.json.tracks)}. Add the track to its track_names."
            )
        track = diarization.json.tracks[self.track_name]

        # read the rule, extract and validate subjects expected ranges
        ranges = self._subject_ranges()

        # based on rules expected ranges and actual diarazation results assign speakers segments
        config = self.detector.subsequence_context.audio_subject_assignment
        assignment = assign_speakers_from_ranges(track.segments, ranges, config)
        logging.info(f"{self._where}: speaker assignment {assignment}")
        self._warn_unmatched(track, ranges, assignment, config)

        # we inject rules expectation as a segment, even if diariazation didn't find any segment there
        subject_times = {subject: list(subject_ranges) for subject, subject_ranges in ranges.items()}
        all_ranges = [time_range for subject_ranges in ranges.values() for time_range in subject_ranges]
        for segment in track.segments:
            subject = assignment.get(segment.speaker)
            if subject is None:
                continue
            subject_times.setdefault(subject, []).extend(subtract_ranges(segment.start, segment.end, all_ranges))

        # sometimes we have overlapping same speaker subject - merge them
        segments = [
            DiarizationSegment(speaker=subject, start=start, end=end)
            for subject, times in subject_times.items()
            for start, end in merge_ranges(times)
        ]
        return sorted(segments, key=lambda segment: (segment.start, segment.end))

    @property
    def _where(self) -> str:
        return f"audio_subject_detector: track '{self.track_name}' ({self.rule.algorithm})"

    def _subject_ranges(self) -> dict[str, list[tuple[float, float]]]:
        # source recording seconds, so the overlap check does not depend on the processed subsequence
        fps = self.detector.data.fps
        source_ranges = {}
        for subject, time_ranges in self.rule.ranges.items():
            try:
                source_ranges[subject] = merge_ranges([time_range.to_seconds(fps) for time_range in time_ranges])
            except ValueError as e:
                raise ValueError(f"{self._where}: range for '{subject}': {e}") from None

        # ranges of different subjects must not overlap; any overlap shows up between neighbours in
        # start order, and overlapping neighbours are always different subjects (each subject's are merged)
        labeled = sorted(
            (s, e, subject) for subject, subject_ranges in source_ranges.items() for s, e in subject_ranges
        )
        for (_, prev_end, prev_subject), (range_start, _, subject) in zip(labeled, labeled[1:]):
            if range_start < prev_end:
                raise ValueError(
                    f"{self._where}: ranges of '{prev_subject}' and '{subject}' overlap around {range_start:.3f}s; "
                    "a range must contain a single subject's speech."
                )

        start = self.detector.subsequence_start
        ranges = {}
        for subject, subject_source_ranges in source_ranges.items():
            subject_ranges = []
            for range_start, range_end in subject_source_ranges:
                range_start = max(range_start - start, 0.0)
                range_end = min(range_end - start, self.duration)
                if range_end > range_start:
                    subject_ranges.append((range_start, range_end))
            if not subject_ranges:
                logging.warning(
                    f"{self._where}: no range for '{subject}' lies in the processed subsequence "
                    f"({start:.3f}s - {start + self.duration:.3f}s); it gets no speaker on this track."
                )
                continue
            ranges[subject] = subject_ranges
        return ranges

    def _warn_unmatched(self, track: TrackDiarization, ranges, assignment, config) -> None:
        """Log merged voices, subjects without a speaker and speakers without a subject."""
        # One speaker covering several subjects' ranges: the diarization merged their voices into one cluster
        for speaker, coverage in range_coverage(track.segments, ranges).items():
            covered = {subject: round(value, 2) for subject, value in coverage.items() if value >= config.min_overlap}
            if len(covered) > 1:
                logging.warning(
                    f"{self._where}: speaker {speaker} covers the ranges of several subjects {covered}; "
                    f"their voices are likely merged. Each subject keeps its ranges, "
                    f"the rest of {speaker} gets '{assignment.get(speaker)}'."
                )

        unassigned_subjects = sorted(set(ranges) - set(assignment.values()))
        if unassigned_subjects:
            logging.warning(
                f"{self._where}: no speaker matched the ranges of {unassigned_subjects}; they get only "
                "their ranges. Check the rule ranges or lower subject_assignment.audio.min_overlap."
            )
        unassigned_speakers = sorted({segment.speaker for segment in track.segments} - set(assignment))
        if unassigned_speakers:
            logging.warning(
                f"{self._where}: speakers {unassigned_speakers} match no subject; "
                "their turns outside the subjects' ranges are dropped."
            )
