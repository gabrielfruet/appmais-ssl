"""Extract representative frames from videos.

Usage:
    python scripts/extract_video_frames.py input.mp4 output/frames
    python scripts/extract_video_frames.py data/videos_raw output/frames
    python scripts/extract_video_frames.py data/videos_raw output/frames \\
        --videos-file data/curated_videos.txt
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import cv2
import numpy as np
from tqdm import tqdm

from engine.bee_crop import find_bee_components

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
THUMBNAIL_SIZE = (64, 64)
MOG2_BLUR_KERNEL_SIZE = (9, 9)


@dataclass(frozen=True)
class _VideoCaptureInfo:
    capture: cv2.VideoCapture
    fps: float
    total_frames: int
    duration_seconds: float


@dataclass(frozen=True)
class _ExtractionStats:
    saved_count: int
    sampled_count: int
    skipped_no_bee: int = 0
    original_frame_shape: tuple[int, int] | None = None


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


def parse_videos_file(path: Path) -> list[Path]:
    """Parse a text file with one video path per line into absolute Paths.

    Lines starting with ``#`` and empty lines are ignored. Each kept
    line must resolve to an existing video file.
    """
    parsed: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        candidate = Path(stripped).expanduser().resolve()
        if not candidate.exists():
            raise click.ClickException(f"Video listed in {path} not found: {candidate}")
        if candidate.suffix.lower() not in VIDEO_EXTENSIONS:
            raise click.ClickException(
                f"Video listed in {path} is not a video file: {candidate}"
            )
        parsed.append(candidate)
    return parsed


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


def output_path_for_background(video_output_dir: Path) -> Path:
    return video_output_dir / "background.png"


def save_background_image(
    video_path: Path,
    video_output_dir: Path,
    background_subtractor: cv2.BackgroundSubtractor,
    original_frame_shape: tuple[int, int] | None = None,
) -> None:
    background = background_subtractor.getBackgroundImage()
    if background is None:
        return
    if (
        original_frame_shape is not None
        and background.shape[:2] != original_frame_shape
    ):
        background = cv2.resize(
            background,
            (original_frame_shape[1], original_frame_shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    background_path = output_path_for_background(video_output_dir)
    background_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(background_path), background)
    if not ok:
        raise click.ClickException(
            f"Could not write background image: {background_path}"
        )
    click.echo(
        f"{video_path.name}: saved background "
        f"({background.shape[1]}x{background.shape[0]}) to {background_path}"
    )


def _feed_mog2_pair(
    mask_subtractor: cv2.BackgroundSubtractor,
    background_subtractor: cv2.BackgroundSubtractor,
    frame: np.ndarray,
    downsample_width: int,
) -> np.ndarray:
    """Update both MOG2 instances and return the foreground mask.

    The ``mask_subtractor`` receives a Gaussian-blurred, downsampled
    frame so its foreground mask is stable. The ``background_subtractor``
    receives the same downsampled frame *without* the Gaussian blur so
    its learned background stays sharp for the saved ``background.png``.
    We only return the mask from the blurred one — the unblurred
    subtractor's mask is discarded; only its background model is used.
    """
    mog2_frame = preprocess_for_mog2(frame, downsample_width)
    raw_mask = mask_subtractor.apply(mog2_frame)
    background_subtractor.apply(downsample_for_mog2(frame, downsample_width))
    return raw_mask


def _open_capture(video_path: Path) -> _VideoCaptureInfo:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise click.ClickException(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0.0 or total_frames <= 0:
        capture.release()
        raise click.ClickException(f"Could not read video metadata: {video_path}")

    return _VideoCaptureInfo(
        capture=capture,
        fps=fps,
        total_frames=total_frames,
        duration_seconds=total_frames / fps,
    )


def _extract_with_mog2(
    capture_info: _VideoCaptureInfo,
    video_path: Path,
    output_dir: Path,
    sample_every_seconds: float,
    diff_threshold: float,
    min_gap_seconds: float,
    max_frames: int,
    jpeg_quality: int,
    skip_start_seconds: float,
    mog2_downsample_width: int,
    min_bee_area: int,
    mask_subtractor: cv2.BackgroundSubtractor,
    background_subtractor: cv2.BackgroundSubtractor,
) -> _ExtractionStats:
    export_duration_seconds = capture_info.duration_seconds - skip_start_seconds
    candidate_count = int(export_duration_seconds / sample_every_seconds) + 1
    last_saved_signature: np.ndarray | None = None
    last_saved_time = -min_gap_seconds
    saved_count = 0
    sampled_count = 0
    skipped_no_bee = 0
    original_frame_shape: tuple[int, int] | None = None
    next_sample_time = skip_start_seconds
    frame_index = 0
    progress = tqdm(
        total=candidate_count,
        desc=video_path.name,
        unit="sample",
        leave=False,
    )
    try:
        while saved_count < max_frames:
            ok, frame = capture_info.capture.read()
            if not ok:
                break

            timestamp_seconds = frame_index / capture_info.fps
            if original_frame_shape is None:
                original_frame_shape = frame.shape[:2]
            raw_mask = _feed_mog2_pair(
                mask_subtractor=mask_subtractor,
                background_subtractor=background_subtractor,
                frame=frame,
                downsample_width=mog2_downsample_width,
            )
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

            full_mask = resize_mask_to_frame(normalize_mog2_mask(raw_mask), frame)
            if not find_bee_components(full_mask, min_area=min_bee_area):
                skipped_no_bee += 1
                continue

            saved_count += 1
            frame_path = output_path_for_frame(
                output_dir, video_path, saved_count, timestamp_seconds
            )
            write_frame(
                frame_path=frame_path,
                frame=frame,
                jpeg_quality=jpeg_quality,
                foreground_mask=full_mask,
            )

            last_saved_signature = signature
            last_saved_time = timestamp_seconds
    finally:
        progress.close()

    return _ExtractionStats(
        saved_count=saved_count,
        sampled_count=sampled_count,
        skipped_no_bee=skipped_no_bee,
        original_frame_shape=original_frame_shape,
    )


def _extract_without_mog2(
    capture_info: _VideoCaptureInfo,
    video_path: Path,
    output_dir: Path,
    sample_every_seconds: float,
    diff_threshold: float,
    min_gap_seconds: float,
    max_frames: int,
    jpeg_quality: int,
    skip_start_seconds: float,
) -> _ExtractionStats:
    export_duration_seconds = capture_info.duration_seconds - skip_start_seconds
    candidate_count = int(export_duration_seconds / sample_every_seconds) + 1
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

        timestamp_seconds = skip_start_seconds + (
            candidate_index * sample_every_seconds
        )
        capture_info.capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
        ok, frame = capture_info.capture.read()
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

    return _ExtractionStats(saved_count=saved_count, sampled_count=sampled_count)


def _safe_extract_video(
    job: tuple[Callable[..., int], Path],
) -> tuple[Path, int, str | None]:
    """Run a single ``extract_frames`` job in a worker process, capturing errors."""
    worker_fn, video_path = job
    try:
        return (video_path, int(worker_fn(video_path)), None)
    except Exception as exc:  # noqa: BLE001 - want to surface any per-video failure
        return (video_path, 0, f"{type(exc).__name__}: {exc}")


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
    save_background: bool,
    mog2_history: int,
    mog2_var_threshold: float,
    mog2_downsample_width: int,
    min_bee_area: int = 50,
) -> int:
    video_output_dir = output_dir_for_video(output_dir, video_path)
    existing_frames = sorted(video_output_dir.glob("*.jpg"))
    existing_masks = sorted(video_output_dir.glob("*_mask.png"))
    existing_background = output_path_for_background(video_output_dir)
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
        if existing_background.exists():
            existing_background.unlink()

    capture_info = _open_capture(video_path)
    try:
        if skip_start_seconds >= capture_info.duration_seconds:
            click.echo(
                f"{video_path.name}: skipped, duration "
                f"{capture_info.duration_seconds:.1f}s is not longer than "
                f"--skip-start-seconds {skip_start_seconds:.1f}s"
            )
            return 0

        if foreground_masks:
            mask_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=mog2_history,
                varThreshold=mog2_var_threshold,
                detectShadows=True,
            )
            background_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=mog2_history,
                varThreshold=mog2_var_threshold,
                detectShadows=True,
            )
            stats = _extract_with_mog2(
                capture_info=capture_info,
                video_path=video_path,
                output_dir=output_dir,
                sample_every_seconds=sample_every_seconds,
                diff_threshold=diff_threshold,
                min_gap_seconds=min_gap_seconds,
                max_frames=max_frames,
                jpeg_quality=jpeg_quality,
                skip_start_seconds=skip_start_seconds,
                mog2_downsample_width=mog2_downsample_width,
                min_bee_area=min_bee_area,
                mask_subtractor=mask_subtractor,
                background_subtractor=background_subtractor,
            )
            if save_background:
                save_background_image(
                    video_path,
                    video_output_dir,
                    background_subtractor,
                    original_frame_shape=stats.original_frame_shape,
                )
        else:
            stats = _extract_without_mog2(
                capture_info=capture_info,
                video_path=video_path,
                output_dir=output_dir,
                sample_every_seconds=sample_every_seconds,
                diff_threshold=diff_threshold,
                min_gap_seconds=min_gap_seconds,
                max_frames=max_frames,
                jpeg_quality=jpeg_quality,
                skip_start_seconds=skip_start_seconds,
            )

        click.echo(
            f"{video_path.name}: sampled {stats.sampled_count}, "
            f"saved {stats.saved_count}, skipped {stats.skipped_no_bee} "
            f"no-bee to {video_output_dir}"
        )
        return stats.saved_count
    finally:
        capture_info.capture.release()


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--videos-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional path to a text file with one video path per line. "
        "If given, only these videos are processed. Lines starting "
        "with '#' are ignored. When set, this overrides the videos "
        "discovered under INPUT_PATH."
    ),
)
@click.option(
    "--min-bee-area",
    type=int,
    default=50,
    show_default=True,
    help=(
        "Minimum foreground component area (in pixels) for a frame to "
        "be saved. Frames with no bee component of at least this size "
        "are skipped. Requires --foreground-masks; ignored otherwise."
    ),
)
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
    "--save-background",
    is_flag=True,
    help=(
        "After the MOG2 loop finishes, write the learned background image "
        "to <video_output_dir>/background.png as a BGR PNG. The image is "
        "produced by a second MOG2 trained on unblurred (sharp), "
        "downsampled frames, then upscaled with INTER_CUBIC to the full "
        "frame resolution (e.g. 640x480). The foreground-mask MOG2 is "
        "trained on Gaussian-blurred frames for stable detection, but its "
        "background is *not* used here. Requires --foreground-masks; "
        "ignored otherwise."
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
    default=2.0,
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
@click.option(
    "--workers",
    type=click.IntRange(1, 64),
    default=1,
    show_default=True,
    help=(
        "Number of worker processes for per-video extraction. "
        "1 keeps the original sequential behavior; 2-8 is a good range "
        "for a multi-core machine. Per-video failures are caught and "
        "logged without aborting the batch."
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
    save_background: bool,
    mog2_history: int,
    mog2_var_threshold: float,
    mog2_downsample_width: int,
    videos_file: Path | None = None,
    min_bee_area: int = 50,
    workers: int = 1,
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
    if min_bee_area <= 0:
        raise click.ClickException("--min-bee-area must be positive")
    if save_background and not foreground_masks:
        click.echo("--save-background requires --foreground-masks; ignoring.")
        save_background = False

    if videos_file is not None:
        videos = parse_videos_file(videos_file)
    else:
        videos = find_videos(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    worker_kwargs: dict[str, Any] = dict(
        output_dir=output_dir,
        sample_every_seconds=sample_every_seconds,
        diff_threshold=diff_threshold,
        min_gap_seconds=min_gap_seconds,
        max_frames=max_frames_per_video,
        jpeg_quality=jpeg_quality,
        skip_start_seconds=skip_start_seconds,
        overwrite=overwrite,
        foreground_masks=foreground_masks,
        save_background=save_background,
        mog2_history=mog2_history,
        mog2_var_threshold=mog2_var_threshold,
        mog2_downsample_width=mog2_downsample_width,
        min_bee_area=min_bee_area,
    )

    total_saved = 0
    if workers <= 1:
        for video_path in videos:
            total_saved += extract_frames(video_path=video_path, **worker_kwargs)
    else:
        import functools
        import multiprocessing

        worker_fn = functools.partial(extract_frames, **worker_kwargs)
        jobs: list[tuple[functools.partial[int], Path]] = [
            (worker_fn, video_path) for video_path in videos
        ]
        with multiprocessing.Pool(workers) as pool:
            for video_path, saved, error in tqdm(
                pool.imap_unordered(_safe_extract_video, jobs, chunksize=1),
                total=len(videos),
                desc="Extracting",
                unit="video",
            ):
                if error is not None:
                    click.echo(f"  ! {video_path.name}: {error}")
                else:
                    total_saved += saved

    click.echo(f"Saved {total_saved} frames from {len(videos)} video(s).")


if __name__ == "__main__":
    main()
