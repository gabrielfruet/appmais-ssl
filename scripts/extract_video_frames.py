"""Extract representative frames from videos.

Usage:
    python scripts/extract_video_frames.py input.mp4 output/frames
    python scripts/extract_video_frames.py data/videos_raw output/frames
"""

from pathlib import Path

import click
import cv2
import numpy as np
from tqdm import tqdm

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
THUMBNAIL_SIZE = (64, 64)


def find_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    videos = [
        path
        for path in sorted(input_path.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not videos:
        raise click.ClickException(f"No videos found in {input_path}")
    return videos


def frame_signature(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def frame_difference(current: np.ndarray, previous: np.ndarray) -> float:
    return float(np.mean(np.abs(current - previous)))


def safe_stem(path: Path) -> str:
    return path.stem.replace("/", "_").replace(":", "-")


def output_dir_for_video(output_dir: Path, video_path: Path) -> Path:
    return output_dir / safe_stem(video_path)


def output_path_for_frame(
    output_dir: Path, video_path: Path, saved_count: int, timestamp_seconds: float
) -> Path:
    video_dir = output_dir_for_video(output_dir, video_path)
    return video_dir / f"frame_{saved_count:06d}_t{timestamp_seconds:08.1f}s.jpg"


def extract_frames(
    video_path: Path,
    output_dir: Path,
    sample_every_seconds: float,
    diff_threshold: float,
    min_gap_seconds: float,
    max_frames: int,
    jpeg_quality: int,
    overwrite: bool,
) -> int:
    video_output_dir = output_dir_for_video(output_dir, video_path)
    existing_frames = sorted(video_output_dir.glob("*.jpg"))
    if existing_frames and not overwrite:
        click.echo(
            f"{video_path.name}: skipped, found {len(existing_frames)} "
            f"existing frame(s) in {video_output_dir}. "
            "Use --overwrite to regenerate."
        )
        return 0
    if existing_frames and overwrite:
        for frame_path in existing_frames:
            frame_path.unlink()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise click.ClickException(f"Could not open video: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0.0 or total_frames <= 0:
            raise click.ClickException(f"Could not read video metadata: {video_path}")

        duration_seconds = total_frames / fps
        candidate_count = int(duration_seconds / sample_every_seconds) + 1
        last_saved_signature: np.ndarray | None = None
        last_saved_time = -min_gap_seconds
        saved_count = 0
        sampled_count = 0

        progress = tqdm(
            range(candidate_count),
            desc=video_path.name,
            unit="sample",
            leave=False,
        )
        for candidate_index in progress:
            if saved_count >= max_frames:
                break

            timestamp_seconds = candidate_index * sample_every_seconds
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue

            sampled_count += 1
            signature = frame_signature(frame)
            enough_gap = timestamp_seconds - last_saved_time >= min_gap_seconds
            different_enough = (
                last_saved_signature is None
                or frame_difference(signature, last_saved_signature) >= diff_threshold
            )
            if not enough_gap or not different_enough:
                continue

            saved_count += 1
            frame_path = output_path_for_frame(
                output_dir, video_path, saved_count, timestamp_seconds
            )
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(
                str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if not ok:
                raise click.ClickException(f"Could not write frame: {frame_path}")

            last_saved_signature = signature
            last_saved_time = timestamp_seconds

        click.echo(
            f"{video_path.name}: sampled {sampled_count}, saved {saved_count} "
            f"to {video_output_dir}"
        )
        return saved_count
    finally:
        capture.release()


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--sample-every-seconds",
    type=float,
    default=2.0,
    show_default=True,
    help="How often to inspect a candidate frame.",
)
@click.option(
    "--diff-threshold",
    type=float,
    default=8.0,
    show_default=True,
    help="Minimum average grayscale pixel difference from the last saved frame.",
)
@click.option(
    "--min-gap-seconds",
    type=float,
    default=5.0,
    show_default=True,
    help="Minimum time between saved frames.",
)
@click.option(
    "--max-frames-per-video",
    type=int,
    default=200,
    show_default=True,
    help="Maximum number of frames to save from each video.",
)
@click.option(
    "--jpeg-quality",
    type=click.IntRange(1, 100),
    default=95,
    show_default=True,
    help="JPEG quality for saved frames.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Delete existing JPG frames for a video and regenerate them.",
)
def main(
    input_path: Path,
    output_dir: Path,
    sample_every_seconds: float,
    diff_threshold: float,
    min_gap_seconds: float,
    max_frames_per_video: int,
    jpeg_quality: int,
    overwrite: bool,
) -> None:
    if sample_every_seconds <= 0.0:
        raise click.ClickException("--sample-every-seconds must be positive")
    if diff_threshold < 0.0:
        raise click.ClickException("--diff-threshold cannot be negative")
    if min_gap_seconds < 0.0:
        raise click.ClickException("--min-gap-seconds cannot be negative")
    if max_frames_per_video <= 0:
        raise click.ClickException("--max-frames-per-video must be positive")

    videos = find_videos(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for video_path in videos:
        total_saved += extract_frames(
            video_path=video_path,
            output_dir=output_dir,
            sample_every_seconds=sample_every_seconds,
            diff_threshold=diff_threshold,
            min_gap_seconds=min_gap_seconds,
            max_frames=max_frames_per_video,
            jpeg_quality=jpeg_quality,
            overwrite=overwrite,
        )

    click.echo(f"Saved {total_saved} frames from {len(videos)} video(s).")


if __name__ == "__main__":
    main()
