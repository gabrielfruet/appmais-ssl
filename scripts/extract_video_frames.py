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
MOG2_BLUR_KERNEL_SIZE = (5, 5)


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


def output_path_for_mask(frame_path: Path) -> Path:
    return frame_path.with_name(f"{frame_path.stem}_mask.png")


def downsample_for_mog2(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame

    scale = max_width / width
    resized_height = max(1, round(height * scale))
    return cv2.resize(frame, (max_width, resized_height), interpolation=cv2.INTER_AREA)


def preprocess_for_mog2(frame: np.ndarray, max_width: int) -> np.ndarray:
    downsampled = downsample_for_mog2(frame, max_width)
    return cv2.GaussianBlur(downsampled, MOG2_BLUR_KERNEL_SIZE, 0)


def normalize_mog2_mask(mask: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(mask, dtype=np.uint8)
    normalized[mask == 127] = 127
    normalized[mask > 127] = 255
    return normalized


def resize_mask_to_frame(mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    if mask.shape[:2] == (frame_height, frame_width):
        return mask
    return cv2.resize(
        mask,
        (frame_width, frame_height),
        interpolation=cv2.INTER_NEAREST,
    )


def write_frame(
    frame_path: Path,
    frame: np.ndarray,
    jpeg_quality: int,
    foreground_mask: np.ndarray | None,
) -> None:
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise click.ClickException(f"Could not write frame: {frame_path}")

    if foreground_mask is None:
        return

    mask_path = output_path_for_mask(frame_path)
    ok = cv2.imwrite(str(mask_path), foreground_mask)
    if not ok:
        raise click.ClickException(f"Could not write foreground mask: {mask_path}")


def should_save_frame(
    frame: np.ndarray,
    timestamp_seconds: float,
    last_saved_signature: np.ndarray | None,
    last_saved_time: float,
    min_gap_seconds: float,
    diff_threshold: float,
) -> tuple[bool, np.ndarray]:
    signature = frame_signature(frame)
    enough_gap = timestamp_seconds - last_saved_time >= min_gap_seconds
    different_enough = (
        last_saved_signature is None
        or frame_difference(signature, last_saved_signature) >= diff_threshold
    )
    return enough_gap and different_enough, signature


def extract_frames(
    video_path: Path,
    output_dir: Path,
    sample_every_seconds: float,
    diff_threshold: float,
    min_gap_seconds: float,
    max_frames: int,
    jpeg_quality: int,
    skip_start_seconds: float,
    overwrite: bool,
    foreground_masks: bool,
    mog2_history: int,
    mog2_var_threshold: float,
    mog2_downsample_width: int,
) -> int:
    video_output_dir = output_dir_for_video(output_dir, video_path)
    existing_frames = sorted(video_output_dir.glob("*.jpg"))
    existing_masks = sorted(video_output_dir.glob("*_mask.png"))
    if existing_frames and not overwrite:
        click.echo(
            f"{video_path.name}: skipped, found {len(existing_frames)} "
            f"existing frame(s) in {video_output_dir}. "
            "Use --overwrite to regenerate."
        )
        return 0
    if overwrite:
        for output_path in [*existing_frames, *existing_masks]:
            output_path.unlink()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise click.ClickException(f"Could not open video: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0.0 or total_frames <= 0:
            raise click.ClickException(f"Could not read video metadata: {video_path}")

        duration_seconds = total_frames / fps
        if skip_start_seconds >= duration_seconds:
            click.echo(
                f"{video_path.name}: skipped, duration {duration_seconds:.1f}s "
                f"is not longer than --skip-start-seconds {skip_start_seconds:.1f}s"
            )
            return 0

        export_duration_seconds = duration_seconds - skip_start_seconds
        candidate_count = int(export_duration_seconds / sample_every_seconds) + 1
        last_saved_signature: np.ndarray | None = None
        last_saved_time = -min_gap_seconds
        saved_count = 0
        sampled_count = 0

        if foreground_masks:
            background_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=mog2_history,
                varThreshold=mog2_var_threshold,
                detectShadows=True,
            )
            next_sample_time = skip_start_seconds
            frame_index = 0
            progress = tqdm(
                total=candidate_count,
                desc=video_path.name,
                unit="sample",
                leave=False,
            )
            while saved_count < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break

                timestamp_seconds = frame_index / fps
                mog2_frame = preprocess_for_mog2(frame, mog2_downsample_width)
                raw_mask = background_subtractor.apply(mog2_frame)
                frame_index += 1

                if timestamp_seconds + 1e-9 < next_sample_time:
                    continue

                progress.update(1)
                sampled_count += 1
                should_save, signature = should_save_frame(
                    frame=frame,
                    timestamp_seconds=timestamp_seconds,
                    last_saved_signature=last_saved_signature,
                    last_saved_time=last_saved_time,
                    min_gap_seconds=min_gap_seconds,
                    diff_threshold=diff_threshold,
                )
                while next_sample_time <= timestamp_seconds:
                    next_sample_time += sample_every_seconds
                if not should_save:
                    continue

                saved_count += 1
                frame_path = output_path_for_frame(
                    output_dir, video_path, saved_count, timestamp_seconds
                )
                write_frame(
                    frame_path=frame_path,
                    frame=frame,
                    jpeg_quality=jpeg_quality,
                    foreground_mask=resize_mask_to_frame(
                        normalize_mog2_mask(raw_mask), frame
                    ),
                )

                last_saved_signature = signature
                last_saved_time = timestamp_seconds
            progress.close()
        else:
            progress = tqdm(
                range(candidate_count),
                desc=video_path.name,
                unit="sample",
                leave=False,
            )
            for candidate_index in progress:
                if saved_count >= max_frames:
                    break

                timestamp_seconds = skip_start_seconds + (
                    candidate_index * sample_every_seconds
                )
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
                ok, frame = capture.read()
                if not ok:
                    continue

                sampled_count += 1
                should_save, signature = should_save_frame(
                    frame=frame,
                    timestamp_seconds=timestamp_seconds,
                    last_saved_signature=last_saved_signature,
                    last_saved_time=last_saved_time,
                    min_gap_seconds=min_gap_seconds,
                    diff_threshold=diff_threshold,
                )
                if not should_save:
                    continue

                saved_count += 1
                frame_path = output_path_for_frame(
                    output_dir, video_path, saved_count, timestamp_seconds
                )
                write_frame(
                    frame_path=frame_path,
                    frame=frame,
                    jpeg_quality=jpeg_quality,
                    foreground_mask=None,
                )

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
    "--skip-start-seconds",
    type=float,
    default=5.0,
    show_default=True,
    help="Do not export frames from the first N seconds of each video.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help=(
        "Delete existing JPG frames and foreground masks for a video "
        "and regenerate them."
    ),
)
@click.option(
    "--foreground-masks",
    is_flag=True,
    help=(
        "Export a MOG2 foreground mask beside each saved frame. "
        "Mask values are 0=background, 127=shadow, 255=foreground."
    ),
)
@click.option(
    "--mog2-history",
    type=int,
    default=500,
    show_default=True,
    help="Number of frames used by MOG2 to model the background.",
)
@click.option(
    "--mog2-var-threshold",
    type=float,
    default=4.0,
    show_default=True,
    help="MOG2 variance threshold; lower values make detection more sensitive.",
)
@click.option(
    "--mog2-downsample-width",
    type=int,
    default=320,
    show_default=True,
    help=(
        "Downsample frames to this width before MOG2; masks are resized "
        "back to the exported frame size."
    ),
)
def main(
    input_path: Path,
    output_dir: Path,
    sample_every_seconds: float,
    diff_threshold: float,
    min_gap_seconds: float,
    max_frames_per_video: int,
    jpeg_quality: int,
    skip_start_seconds: float,
    overwrite: bool,
    foreground_masks: bool,
    mog2_history: int,
    mog2_var_threshold: float,
    mog2_downsample_width: int,
) -> None:
    if sample_every_seconds <= 0.0:
        raise click.ClickException("--sample-every-seconds must be positive")
    if diff_threshold < 0.0:
        raise click.ClickException("--diff-threshold cannot be negative")
    if min_gap_seconds < 0.0:
        raise click.ClickException("--min-gap-seconds cannot be negative")
    if max_frames_per_video <= 0:
        raise click.ClickException("--max-frames-per-video must be positive")
    if skip_start_seconds < 0.0:
        raise click.ClickException("--skip-start-seconds cannot be negative")
    if mog2_history <= 0:
        raise click.ClickException("--mog2-history must be positive")
    if mog2_var_threshold <= 0.0:
        raise click.ClickException("--mog2-var-threshold must be positive")
    if mog2_downsample_width <= 0:
        raise click.ClickException("--mog2-downsample-width must be positive")

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
            skip_start_seconds=skip_start_seconds,
            overwrite=overwrite,
            foreground_masks=foreground_masks,
            mog2_history=mog2_history,
            mog2_var_threshold=mog2_var_threshold,
            mog2_downsample_width=mog2_downsample_width,
        )

    click.echo(f"Saved {total_saved} frames from {len(videos)} video(s).")


if __name__ == "__main__":
    main()
