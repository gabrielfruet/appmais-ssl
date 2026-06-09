"""Curate a diverse subset of videos for the frame extraction pipeline.

For every ``*.mp4`` in the input directory, score it on sharpness,
motion, and brightness (sampled from the middle 90% of the video),
print a ranked table of every video, then pick a top-N subset that
maximises hive-id diversity and write the picked video paths to a
text file.

Usage:
    uv run python scripts/curate_videos.py data/videos_raw \\
        --output data/curated_videos.txt --top-n 8
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import click
import cv2
import numpy as np

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
DEFAULT_INPUT_DIR = Path("data/videos_raw")
DEFAULT_OUTPUT_FILE = Path("data/curated_videos.txt")
MOTION_DOWN_SAMPLE = (160, 120)
RANK_TABLE_WIDTH = 110


@dataclass(frozen=True)
class ScoredVideo:
    path: Path
    hive_id: str
    sharpness: float
    motion: float
    brightness: float
    score: float

    @property
    def video(self) -> str:
        return self.path.name


def _find_videos(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def _hive_id_from_path(path: Path) -> str:
    """Return the hive id embedded in ``<hive>@<date>@<time>.mp4``."""
    return path.stem.split("@", 1)[0]


def _sample_timestamps(duration_seconds: float, num_samples: int) -> list[float]:
    """Return ``num_samples`` evenly spaced timestamps in the middle 90%."""
    if duration_seconds <= 0.0 or num_samples <= 0:
        return []
    margin = duration_seconds * 0.05
    inner_start = margin
    inner_end = max(inner_start, duration_seconds - margin)
    if inner_end <= inner_start:
        return [duration_seconds * 0.5] * num_samples
    if num_samples == 1:
        return [(inner_start + inner_end) * 0.5]
    return [
        inner_start + (inner_end - inner_start) * i / float(num_samples - 1)
        for i in range(num_samples)
    ]


def _read_frame_at(
    capture: cv2.VideoCapture, timestamp_seconds: float
) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    return frame


def _frame_sharpness_brightness(frame: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())
    brightness = float(gray.mean())
    return sharpness, brightness


def _frame_motion(prev_gray_small: np.ndarray, gray: np.ndarray) -> float:
    resized = cv2.resize(gray, MOTION_DOWN_SAMPLE, interpolation=cv2.INTER_AREA)
    return float(
        np.mean(np.abs(resized.astype(np.int32) - prev_gray_small.astype(np.int32)))
    )


def score_video(video_path: Path, num_samples: int) -> ScoredVideo:
    """Sample ``num_samples`` frames and compute sharpness, motion, brightness."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise click.ClickException(f"Could not open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_seconds = total_frames / fps if fps > 0.0 and total_frames > 0 else 0.0
        timestamps = _sample_timestamps(duration_seconds, num_samples)
        sharpness_values: list[float] = []
        brightness_values: list[float] = []
        motion_values: list[float] = []
        prev_gray_small: np.ndarray | None = None
        for ts in timestamps:
            frame = _read_frame_at(capture, ts)
            if frame is None:
                continue
            sharpness, brightness = _frame_sharpness_brightness(frame)
            sharpness_values.append(sharpness)
            brightness_values.append(brightness)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray_small is not None:
                motion_values.append(_frame_motion(prev_gray_small, gray))
            prev_gray_small = cv2.resize(
                gray, MOTION_DOWN_SAMPLE, interpolation=cv2.INTER_AREA
            )
        if not sharpness_values:
            raise click.ClickException(f"Could not read any samples from: {video_path}")
        sharpness = float(np.mean(sharpness_values))
        brightness = float(np.mean(brightness_values))
        motion = float(np.mean(motion_values)) if motion_values else 0.0
        score = sharpness * (1.0 + motion)
        return ScoredVideo(
            path=video_path,
            hive_id=_hive_id_from_path(video_path),
            sharpness=sharpness,
            motion=motion,
            brightness=brightness,
            score=score,
        )
    finally:
        capture.release()


def filter_videos(
    scored: Sequence[ScoredVideo],
    sharpness_min: float,
    motion_min: float,
    brightness_min: float,
) -> list[ScoredVideo]:
    return [
        video
        for video in scored
        if video.sharpness >= sharpness_min
        and video.motion >= motion_min
        and video.brightness >= brightness_min
    ]


def pick_diverse(candidates: Sequence[ScoredVideo], top_n: int) -> list[ScoredVideo]:
    """Pick ``top_n`` videos, preferring unseen hive ids in descending score order."""
    if top_n <= 0:
        return []
    sorted_candidates = sorted(candidates, key=lambda v: v.score, reverse=True)
    seen_hives: set[str] = set()
    picked: list[ScoredVideo] = []
    for video in sorted_candidates:
        if len(picked) >= top_n:
            break
        if video.hive_id in seen_hives:
            continue
        picked.append(video)
        seen_hives.add(video.hive_id)
    return picked


def _print_ranked_table(scored: Sequence[ScoredVideo]) -> None:
    sorted_videos = sorted(scored, key=lambda v: v.score, reverse=True)
    header = (
        f"{'rank':>4}  {'video':<55}  {'sharp':>8}  "
        f"{'motion':>8}  {'bright':>8}  {'hive_id'}"
    )
    click.echo(header)
    click.echo("-" * RANK_TABLE_WIDTH)
    for rank, video in enumerate(sorted_videos, start=1):
        click.echo(
            f"{rank:>4}  {video.video:<55}  {video.sharpness:>8.2f}  "
            f"{video.motion:>8.3f}  {video.brightness:>8.2f}  {video.hive_id}"
        )


def _write_curated_file(path: Path, picked: Sequence[ScoredVideo]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Curated video list (one absolute path per line)"]
    lines.extend(str(video.path.resolve()) for video in picked)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@click.command()
@click.argument(
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_INPUT_DIR,
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_OUTPUT_FILE,
    show_default=True,
    help="Where to write the curated video list (one absolute path per line).",
)
@click.option(
    "--top-n",
    type=int,
    default=8,
    show_default=True,
    help="Number of diverse videos to keep after filtering.",
)
@click.option(
    "--sharpness-min",
    type=float,
    default=100.0,
    show_default=True,
    help="Drop videos whose mean Laplacian variance is below this.",
)
@click.option(
    "--motion-min",
    type=float,
    default=0.5,
    show_default=True,
    help="Drop videos whose mean inter-frame pixel diff is below this.",
)
@click.option(
    "--brightness-min",
    type=float,
    default=60.0,
    show_default=True,
    help="Drop videos whose mean grayscale value is below this.",
)
@click.option(
    "--num-samples",
    type=int,
    default=8,
    show_default=True,
    help="Frames to sample per video when computing scores.",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Exit non-zero if the curated list ends up with fewer than --top-n videos.",
)
def main(
    input_dir: Path,
    output: Path,
    top_n: int,
    sharpness_min: float,
    motion_min: float,
    brightness_min: float,
    num_samples: int,
    strict: bool,
) -> None:
    if top_n <= 0:
        raise click.ClickException("--top-n must be positive")
    if num_samples <= 0:
        raise click.ClickException("--num-samples must be positive")

    videos = _find_videos(input_dir)
    if not videos:
        raise click.ClickException(f"No videos found in {input_dir}")

    scored = [score_video(path, num_samples) for path in videos]
    _print_ranked_table(scored)

    survivors = filter_videos(
        scored,
        sharpness_min=sharpness_min,
        motion_min=motion_min,
        brightness_min=brightness_min,
    )
    click.echo(
        f"Survivors after filters: {len(survivors)}/{len(scored)} "
        f"(sharp>={sharpness_min}, motion>={motion_min}, bright>={brightness_min})"
    )

    picked = pick_diverse(survivors, top_n)
    if not picked:
        click.echo("WARNING: curated list is empty.")
    elif len(picked) < top_n:
        click.echo(
            f"WARNING: only {len(picked)} diverse hive ids available; "
            f"requested {top_n}."
        )

    _write_curated_file(output, picked)
    click.echo(f"Wrote {len(picked)} curated video path(s) to {output}")

    if strict and len(picked) < top_n:
        raise click.ClickException(
            f"--strict: curated list has {len(picked)} entries, need {top_n}"
        )


if __name__ == "__main__":
    main()
