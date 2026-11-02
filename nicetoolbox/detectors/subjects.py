import warnings

import numpy as np

from nicetoolbox_core.data.json_schema import (
    AlignedSegment,
    AlignedTranscription,
    AlignedWord,
    AudioDiarization,
    AudioTranscription,
    DiarizationSegment,
    SpeakerAlignedTranscription,
    TrackTranscription,
)
from nicetoolbox_core.data.loaded_array import NpzArray

from ..configs.schemas.detectors_config import AudioSubjectAssignmentConfig, VideoSubjectAssignmentConfig


def cam_sees_subjects_from_regions(subject_regions: NpzArray) -> dict[str, list[int]]:
    """Per-camera "seen" subject-index lists derived from a loaded subject_regions array.

    A subject counts as seen by a camera when any of its frames on that camera has
    a non-NaN region box.
    """
    data = subject_regions.data  # (n_subjects, n_cameras, n_frames, 4+)
    cameras = subject_regions.axes.cameras
    per_sc = np.any(~np.isnan(data[..., :4]), axis=(2, 3))  # (n_subjects, n_cameras)
    return {cam: np.flatnonzero(per_sc[:, c]).tolist() for c, cam in enumerate(cameras)}


def bboxes_from_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Axis-aligned [x0, y0, x1, y1] bounding boxes from per-detection 2D keypoints.

    Args:
        landmarks: (n_det, n_kp, 2) or (n_det, n_kp, 3+). Only the first two columns
            of the last axis are used; extra columns (e.g. confidence) are ignored.

    Returns:
        (n_det, 4) float array of [x0, y0, x1, y1].
    """
    landmarks = np.asarray(landmarks, dtype=float)
    xy = landmarks[..., :2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN detection is expected
        x0 = np.nanmin(xy[..., 0], axis=-1)
        y0 = np.nanmin(xy[..., 1], axis=-1)
        x1 = np.nanmax(xy[..., 0], axis=-1)
        y1 = np.nanmax(xy[..., 1], axis=-1)
    return np.stack([x0, y0, x1, y1], axis=-1)


def _iod_matrix(boxes: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Compute IoD = |D ∩ R| / |D| for each (detection, region) pair. Vectorized."""
    n_det = boxes.shape[0]
    n_reg = regions.shape[0]
    if n_det == 0 or n_reg == 0:
        return np.zeros((n_det, n_reg), dtype=float)

    d = boxes[:, None, :]  # (n_det, 1, 4)
    r = regions[None, :, :]  # (1, n_reg, 4)

    inter_x0 = np.maximum(d[..., 0], r[..., 0])
    inter_y0 = np.maximum(d[..., 1], r[..., 1])
    inter_x1 = np.minimum(d[..., 2], r[..., 2])
    inter_y1 = np.minimum(d[..., 3], r[..., 3])

    inter_w = np.clip(inter_x1 - inter_x0, 0.0, None)
    inter_h = np.clip(inter_y1 - inter_y0, 0.0, None)
    inter = inter_w * inter_h

    det_area = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    det_area = det_area[:, None]  # (n_det, 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        iod = np.where(det_area > 0, inter / det_area, 0.0)

    # NaN region rows mean "subject absent from this camera/frame": zero those columns out.
    region_valid = ~np.any(np.isnan(regions), axis=1)  # (n_reg,)
    iod = iod * region_valid[None, :]
    iod = np.nan_to_num(iod, nan=0.0, posinf=0.0, neginf=0.0)
    return iod


def assign_detections(
    boxes: np.ndarray,
    confidence: np.ndarray | None,
    subject_regions: np.ndarray,
    config: VideoSubjectAssignmentConfig,
) -> np.ndarray:
    """
    Assign each detection to at most one subject slot.

    Args:
        boxes: (n_det, 4) detection boxes as [x0, y0, x1, y1].
        confidence: (n_det,) detection confidences, or None (defaults to 1.0).
        subject_regions: (n_subjects, 4) subject regions; a row of NaN means
            the subject is absent from this camera/frame.
        config: Shared assignment knobs (min_overlap, confidence_weight, ...).

    Returns:
        (n_det,) int array holding the assigned subject index per input
        detection, or -1 where the detection was left unassigned.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    n_det = boxes.shape[0]
    subject_regions = np.asarray(subject_regions, dtype=float).reshape(-1, 4)
    n_subj = subject_regions.shape[0]

    assigned = np.full(n_det, -1, dtype=int)
    if n_det == 0 or n_subj == 0:
        return assigned

    if confidence is None:
        confidence = np.ones(n_det, dtype=float)
    else:
        confidence = np.asarray(confidence, dtype=float).reshape(-1)

    # Stable ordering: lexsort so tied cells resolve deterministically.
    order = np.lexsort((boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0]))
    sorted_boxes = boxes[order]
    sorted_scores = confidence[order]

    iod = _iod_matrix(sorted_boxes, subject_regions)
    iod = np.where(iod >= config.min_overlap, iod, 0.0)

    score_term = config.confidence_weight * sorted_scores[:, None]
    active = iod > 0.0
    ranking = np.where(active, iod + score_term, 0.0)

    # Greedy: repeatedly take the argmax cell, assign, zero its row and column.
    sorted_assign = np.full(n_det, -1, dtype=int)
    while True:
        flat_idx = int(np.argmax(ranking))
        best = ranking.flat[flat_idx]
        if best <= 0.0:
            break
        det_i, subj_j = np.unravel_index(flat_idx, ranking.shape)
        sorted_assign[det_i] = subj_j
        ranking[det_i, :] = 0.0
        ranking[:, subj_j] = 0.0

    # Un-sort back to input order.
    assigned[order] = sorted_assign
    return assigned


def assign_speakers_from_ranges(
    segments: list[DiarizationSegment],
    ranges: dict[str, list[tuple[float, float]]],
    config: AudioSubjectAssignmentConfig,
) -> dict[str, str]:
    """
    Assign anonymous diarization speakers to subjects using known ranges per subject.

    Each range is a stretch of time in which only that subject speaks. A speaker's score for a
    subject is the fraction of the subject's total range time it covers. Pairs scoring below
    min_overlap are discarded, then the best remaining pair is assigned greedily, one speaker per subject.

    Args:
        segments: Diarization segments labeled with anonymous speakers (e.g. "SPEAKER_00").
        ranges: Subject name -> non-empty (start, end) ranges in the same time base as the segments.
        config: Shared assignment knobs (min_overlap).

    Returns:
        Anonymous speaker -> subject name, for assigned speakers only.
    """
    coverage = range_coverage(segments, ranges)
    speakers = list(coverage)
    subjects = list(ranges)
    if not speakers or not subjects:
        return {}

    score = np.array([[coverage[speaker][subject] for subject in subjects] for speaker in speakers], dtype=float)
    score = np.where(score >= config.min_overlap, score, 0.0)

    # Greedy: repeatedly take the argmax cell, assign, zero its row and column.
    assignment = {}
    while True:
        flat_idx = int(np.argmax(score))
        if score.flat[flat_idx] <= 0.0:
            break
        speaker_i, subject_j = np.unravel_index(flat_idx, score.shape)
        assignment[speakers[speaker_i]] = subjects[subject_j]
        score[speaker_i, :] = 0.0
        score[:, subject_j] = 0.0
    return assignment


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Length of the intersection of two time intervals, 0 if they are disjoint."""
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _ranges_overlap(start: float, end: float, ranges: list[tuple[float, float]]) -> float:
    """Total length of the intersection of an interval with non-overlapping ranges."""
    return sum(_overlap(start, end, range_start, range_end) for range_start, range_end in ranges)


def range_coverage(
    segments: list[DiarizationSegment], ranges: dict[str, list[tuple[float, float]]]
) -> dict[str, dict[str, float]]:
    """Per anonymous speaker (sorted), the fraction of each subject's total range time during which it speaks."""
    totals = {subject: sum(end - start for start, end in subject_ranges) for subject, subject_ranges in ranges.items()}
    coverage = {speaker: dict.fromkeys(ranges, 0.0) for speaker in sorted({segment.speaker for segment in segments})}
    for segment in segments:
        for subject, subject_ranges in ranges.items():
            if totals[subject] > 0.0:
                overlap = _ranges_overlap(segment.start, segment.end, subject_ranges)
                coverage[segment.speaker][subject] += overlap / totals[subject]
    return coverage


def merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping or touching ranges into disjoint ranges, in time order."""
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def subtract_ranges(start: float, end: float, ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The parts of the interval [start, end] not covered by any of the ranges, in time order."""
    pieces = []
    for range_start, range_end in sorted(ranges):
        if range_end <= start or range_start >= end:
            continue
        if range_start > start:
            pieces.append((start, range_start))
        start = max(start, range_end)
    if end > start:
        pieces.append((start, end))
    return pieces


def to_speaker_aligned_transcription(
    transcription: AudioTranscription, diarization: AudioDiarization, algorithm: str
) -> SpeakerAlignedTranscription:
    """Label every transcribed track with the speakers of its subject diarization.

    Args:
        transcription: A transcription detector's output for each track.
        diarization: Subject-labeled speaker turns per track, in the same subsequence time base.
        algorithm: Detector instance name that produced this output.
    """
    tracks = {}
    for track_name, track in transcription.tracks.items():
        if track_name not in diarization.tracks:
            raise ValueError(
                f"Track '{track_name}' is not in the '{diarization.meta.algorithm}' diarization "
                f"(tracks: {sorted(diarization.tracks)})."
            )
        tracks[track_name] = assign_transcription_speakers(track, diarization.tracks[track_name].segments)

    meta = transcription.meta.model_copy(update={"component": "speaker_aligned_transcription", "algorithm": algorithm})
    return SpeakerAlignedTranscription(meta=meta, tracks=tracks)


def assign_transcription_speakers(
    transcription: TrackTranscription, diarization: list[DiarizationSegment]
) -> AlignedTranscription:
    """
    Label a track's transcribed words and segments with the speakers of its diarization.

    Words and segments are labeled independently, each with the speaker whose turns overlap it
    the longest (the rule of WhisperX's assign_word_speakers). A segment's speaker_confidence is
    the fraction of its words carrying that same speaker.

    Args:
        transcription: The track's transcription, in the same time base as the diarization.
        diarization: The track's speaker turns, labeled with subjects.

    Returns:
        The transcription with a speaker on every word and segment; None where no turn
        overlaps it or alignment left it without timings.
    """
    words = []
    for word in transcription.words:
        speaker = _dominant_speaker(word.start, word.end, diarization)
        words.append(AlignedWord(**word.model_dump(), speaker=speaker))

    segment_words = [[] for _ in transcription.segments]
    for word in words:
        if word.segment_index is not None:
            segment_words[word.segment_index].append(word)

    segments = []
    for segment, own_words in zip(transcription.segments, segment_words, strict=True):
        speaker = _dominant_speaker(segment.start, segment.end, diarization)
        agreeing = sum(1 for word in own_words if word.speaker is not None and word.speaker == speaker)
        confidence = agreeing / len(own_words) if own_words else 0.0
        segments.append(AlignedSegment(**segment.model_dump(), speaker=speaker, speaker_confidence=confidence))

    return AlignedTranscription(total=transcription.total, segments=segments, words=words)


def _dominant_speaker(start: float | None, end: float | None, diarization: list[DiarizationSegment]) -> str | None:
    """The speaker whose turns overlap [start, end] the longest, None if none overlaps it or it is untimed."""
    if start is None or end is None:
        return None

    overlap = {}
    for turn in diarization:
        overlap[turn.speaker] = overlap.get(turn.speaker, 0.0) + _overlap(start, end, turn.start, turn.end)

    best = max(overlap, key=overlap.get, default=None)
    if best is None or overlap[best] <= 0.0:
        return None
    return best
