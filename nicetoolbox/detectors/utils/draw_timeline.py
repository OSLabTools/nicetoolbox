"""
Shared timeline rendering for audio detectors.

Speaker turns of a diarization-like output (diarization, subject segments, ...) are drawn per
audio track with one row per speaker, in the style of pyannote's own plots:
- draw_diarization_timeline: a static plot of the whole track (PNG).
- render_diarization_video: the track's camera frames (or a black canvas) with a scrolling timeline
  strip below them and the track's audio, frame by frame (MP4).
"""

import logging
import os
from typing import Any, Optional

import cv2
import matplotlib
import numpy as np
from matplotlib.figure import Figure

from nicetoolbox_core.data.json_schema import AudioDiarization, TrackDiarization
from nicetoolbox_core.input_recipes import InputRecipes
from nicetoolbox_core.video_loaders import ImagePathsByFrameIndexLoader

from ...utils import video as vd

# Frame size when a track has no camera to take frames from
BLACK_FRAME_SIZE = (1280, 720)  # (width, height)


def draw_diarization_timeline(
    diarization: AudioDiarization, out_folder: str, speakers: Optional[list[str]] = None
) -> None:
    """
    Save one timeline per track as <out_folder>/<track>.png.

    Args:
        diarization: The output to draw; title and time axis come from its meta.
        out_folder: Folder for the PNGs.
        speakers: Rows to draw on every track, top first, e.g. the sequence's subjects: each keeps
            its color in every plot and a speaker without turns still gets an (empty) row. Turns of
            other speakers are not drawn. If None, each track shows the speakers found on it, sorted.
    """
    meta = diarization.meta
    for track_name, track in diarization.tracks.items():
        _draw_track(
            track,
            rows=_row_colors(_track_rows(track, speakers)),
            length=meta.subsequence_length,
            title=f"{meta.algorithm} {meta.component} - {track_name}",
            out_path=os.path.join(out_folder, f"{track_name}.png"),
        )


def render_diarization_video(
    diarization: AudioDiarization,
    out_folder: str,
    input_recipes: InputRecipes,
    speakers: Optional[list[str]] = None,
    fallback_camera: Optional[str] = None,
    window: float = 10.0,
) -> None:
    """
    Save one video per track as <out_folder>/<track>.mp4: frames on top, a scrolling timeline strip
    below them (one row per speaker, `window` seconds centered on the current time) and the track's audio.

    Frames come from the camera the track is embedded in, else from `fallback_camera`, else a black
    canvas. Rows and colors follow draw_diarization_timeline, so the video matches the PNG.

    Args:
        diarization: The output to draw; its time base is the subsequence.
        out_folder: Folder for the videos.
        input_recipes: The sequence's input recipes; give each track's audio file and the video frames.
        speakers: Rows to draw on every track, as in draw_diarization_timeline.
        fallback_camera: Camera for tracks that are not embedded in one; None draws on a black canvas.
        window: Seconds of the timeline visible at once.
    """
    video_recipe = input_recipes.video_input_recipe
    audio_recipe = input_recipes.audio_input_recipe
    fps = video_recipe.fps
    # audio files are full length: cut the subsequence's time range, the segments' time base
    audio_start = audio_recipe.start_time_ms / 1000.0
    audio_duration = (audio_recipe.end_time_ms - audio_recipe.start_time_ms) / 1000.0

    for track_name, track in diarization.tracks.items():
        stream = audio_recipe.streams[track_name]
        camera = stream.camera or fallback_camera
        if camera is not None and camera not in video_recipe.camera_names:
            logging.warning(f"Track '{track_name}': camera '{camera}' has no frames, drawing on a black canvas.")
            camera = None

        rows = _row_colors(_track_rows(track, speakers))
        strip = _TimelineStrip(track, rows, window)
        out_path = os.path.join(out_folder, f"{track_name}.mp4")
        os.makedirs(out_folder, exist_ok=True)

        writer = None
        try:
            for frame_idx, image in enumerate(_frames(input_recipes, camera)):
                frame = np.vstack([image, strip.draw(frame_idx / fps, width=image.shape[1])])
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = vd.FFmpegPipeWriter(
                        out_path,
                        size=(width, height),
                        save_frames_dir=None,
                        fps=fps,
                        audio_path=stream.path,
                        audio_start=audio_start,
                        audio_duration=audio_duration,
                    )
                writer.write_frame(frame)
        finally:
            if writer is not None:
                writer.close()
        logging.info(f"Diarization video for track '{track_name}' saved to '{out_path}'.")


def _frames(input_recipes: InputRecipes, camera: Optional[str]):
    """Yield the subsequence's frames of a camera as BGR images, or black frames if camera is None."""
    video_recipe = input_recipes.video_input_recipe
    if camera is None:
        width, height = BLACK_FRAME_SIZE
        for _ in range(video_recipe.range_start, video_recipe.range_end, video_recipe.step):
            yield np.zeros((height, width, 3), dtype=np.uint8)
        return

    for _, paths in ImagePathsByFrameIndexLoader(config=input_recipes, expected_cameras=[camera]):
        image = cv2.imread(paths[camera])
        if image is None:
            raise RuntimeError(f"Failed to read image from '{paths[camera]}'")
        yield image


class _TimelineStrip:
    """Draws the strip below a frame: the speaker turns within a time window around the current time."""

    ROW_HEIGHT = 28
    LABEL_WIDTH = 160
    PADDING = 8
    BACKGROUND = (30, 30, 30)
    TEXT = (230, 230, 230)

    def __init__(self, track: TrackDiarization, rows: dict[str, Any], window: float):
        self.window = window
        self.labels = list(rows)
        # matplotlib RGBA in [0, 1] -> OpenCV BGR in [0, 255], same colors as the PNG timeline
        self.colors = [tuple(int(255 * c) for c in reversed(rgba[:3])) for rgba in rows.values()]
        # per row, the turns' start and end times, to select the visible ones per frame
        self.turns = []
        for speaker in self.labels:
            segments = [segment for segment in track.segments if segment.speaker == speaker]
            self.turns.append(
                (np.array([s.start for s in segments], dtype=float), np.array([s.end for s in segments], dtype=float))
            )
        self.height = 2 * self.PADDING + max(len(self.labels), 1) * self.ROW_HEIGHT + self.ROW_HEIGHT

    def draw(self, now: float, width: int) -> np.ndarray:
        """The strip for time `now` (seconds in the subsequence), `width` pixels wide."""
        strip = np.full((self.height, width, 3), self.BACKGROUND, dtype=np.uint8)
        plot_x0 = self.LABEL_WIDTH
        plot_width = max(width - plot_x0 - self.PADDING, 1)
        window_start = now - self.window / 2
        window_end = now + self.window / 2

        def to_x(seconds: float) -> int:
            return int(round(plot_x0 + (seconds - window_start) / self.window * plot_width))

        for row, (label, color, (starts, ends)) in enumerate(zip(self.labels, self.colors, self.turns)):
            y0 = self.PADDING + row * self.ROW_HEIGHT
            y_mid = y0 + self.ROW_HEIGHT // 2
            cv2.putText(strip, label, (self.PADDING, y_mid + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cv2.line(strip, (plot_x0, y_mid), (plot_x0 + plot_width, y_mid), (70, 70, 70), 1)

            visible = (starts < window_end) & (ends > window_start)
            for start, end in zip(starts[visible], ends[visible]):
                x0, x1 = to_x(max(start, window_start)), to_x(min(end, window_end))
                cv2.rectangle(strip, (x0, y0 + 6), (x1, y0 + self.ROW_HEIGHT - 6), color, thickness=-1)

        # current time: cursor in the middle of the window, and its value below it
        cursor_x = to_x(now)
        rows_bottom = self.PADDING + max(len(self.labels), 1) * self.ROW_HEIGHT
        cv2.line(strip, (cursor_x, self.PADDING), (cursor_x, rows_bottom), self.TEXT, 1)
        cv2.putText(
            strip, f"{now:.1f} s", (cursor_x - 25, rows_bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.TEXT, 1
        )
        return strip


def _track_rows(track: TrackDiarization, speakers: Optional[list[str]]) -> list[str]:
    """The rows to draw for a track, top first."""
    if speakers is not None:  # fixed rows, the same on every track
        return list(speakers)
    # rows from the speakers that have turns on this track
    return sorted({segment.speaker for segment in track.segments})


def _row_colors(rows: list[str]) -> dict[str, Any]:
    """Map each row to a color by its position."""
    palette = matplotlib.colormaps["tab10"]
    return {name: palette(idx % palette.N) for idx, name in enumerate(rows)}


def _draw_track(track: TrackDiarization, rows: dict[str, Any], length: float, title: str, out_path: str) -> None:
    """Draw each row's turns as lines on its own row (speaker -> color), with ticks at the turn boundaries."""
    fig = Figure(figsize=(20, 1 + 0.5 * len(rows)))
    ax = fig.subplots()
    for row, (speaker, color) in enumerate(rows.items()):
        # first row on top
        y = len(rows) - row
        for segment in track.segments:
            if segment.speaker != speaker:
                continue
            ax.hlines(y, segment.start, segment.end, color=color, linewidth=3)
            ax.vlines([segment.start, segment.end], y - 0.15, y + 0.15, color=color, linewidth=1)

    ax.set(xlim=(0.0, length), ylim=(0.5, len(rows) + 0.5), xlabel="time [s]")
    ax.set_yticks(range(len(rows), 0, -1), labels=list(rows))
    ax.grid(axis="x", linestyle=":", linewidth=0.5)
    fig.suptitle(title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100)
