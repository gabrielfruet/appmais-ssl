"""Visualize extracted video frames as contact sheets.

Usage:
    python scripts/visualize_extracted_frames.py data/frames data/frame_visualizations
"""

from math import ceil
from pathlib import Path

import click
import cv2
import numpy as np

FRAME_EXTENSIONS = {".jpeg", ".jpg"}
LABEL_HEIGHT = 26
TEXT_ORIGIN = (6, 18)
TEXT_SCALE = 0.45
TEXT_THICKNESS = 1
SHADOW_COLOR = (0, 255, 255)
FOREGROUND_COLOR = (0, 0, 255)


def find_frame_groups(input_dir: Path) -> list[Path]:
    if list_frame_paths(input_dir):
        return [input_dir]

    groups = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_dir() and list_frame_paths(path)
    ]
    if not groups:
        raise click.ClickException(f"No extracted JPG frames found in {input_dir}")
    return groups


def list_frame_paths(frame_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(frame_dir.iterdir())
        if path.is_file()
        and path.suffix.lower() in FRAME_EXTENSIONS
        and not path.stem.endswith("_mask")
    ]


def select_evenly(paths: list[Path], max_count: int) -> list[Path]:
    if len(paths) <= max_count:
        return paths
    if max_count == 1:
        return [paths[0]]

    last_index = len(paths) - 1
    return [
        paths[round(index * last_index / (max_count - 1))] for index in range(max_count)
    ]


def mask_path_for_frame(frame_path: Path) -> Path:
    return frame_path.with_name(f"{frame_path.stem}_mask.png")


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    height, original_width = image.shape[:2]
    if original_width == width:
        return image

    resized_height = max(1, round(height * width / original_width))
    return cv2.resize(image, (width, resized_height), interpolation=cv2.INTER_AREA)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    colors = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    colors[mask == 127] = SHADOW_COLOR
    colors[mask > 127] = FOREGROUND_COLOR
    return colors


def overlay_mask(frame: np.ndarray, mask_path: Path, alpha: float) -> np.ndarray:
    if not mask_path.exists():
        return frame

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise click.ClickException(f"Could not read mask: {mask_path}")

    frame_height, frame_width = frame.shape[:2]
    if mask.shape[:2] != (frame_height, frame_width):
        mask = cv2.resize(
            mask,
            (frame_width, frame_height),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_pixels = mask > 0
    if not np.any(mask_pixels):
        return frame

    colors = colorize_mask(mask)
    blended = cv2.addWeighted(frame, 1.0 - alpha, colors, alpha, 0.0)
    output = frame.copy()
    output[mask_pixels] = blended[mask_pixels]
    return output


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = cv2.copyMakeBorder(
        image,
        0,
        LABEL_HEIGHT,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    cv2.putText(
        labeled,
        label,
        (TEXT_ORIGIN[0] + 1, image.shape[0] + TEXT_ORIGIN[1] + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        (0, 0, 0),
        TEXT_THICKNESS + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        labeled,
        label,
        (TEXT_ORIGIN[0], image.shape[0] + TEXT_ORIGIN[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_SCALE,
        (255, 255, 255),
        TEXT_THICKNESS,
        cv2.LINE_AA,
    )
    return labeled


def make_preview(
    frame_path: Path,
    thumbnail_width: int,
    show_masks: bool,
    mask_alpha: float,
) -> np.ndarray:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise click.ClickException(f"Could not read frame: {frame_path}")

    if show_masks:
        frame = overlay_mask(frame, mask_path_for_frame(frame_path), mask_alpha)

    preview = resize_to_width(frame, thumbnail_width)
    return add_label(preview, frame_path.stem)


def make_contact_sheet(previews: list[np.ndarray], columns: int) -> np.ndarray:
    cell_width = max(preview.shape[1] for preview in previews)
    cell_height = max(preview.shape[0] for preview in previews)
    rows = ceil(len(previews) / columns)
    sheet = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)

    for index, preview in enumerate(previews):
        row = index // columns
        column = index % columns
        y = row * cell_height
        x = column * cell_width
        sheet[y : y + preview.shape[0], x : x + preview.shape[1]] = preview

    return sheet


def output_path_for_group(output_dir: Path, group_dir: Path) -> Path:
    safe_name = group_dir.name.replace("/", "_").replace(":", "-")
    return output_dir / f"{safe_name}_contact_sheet.jpg"


def visualize_group(
    group_dir: Path,
    output_dir: Path,
    max_frames: int,
    columns: int,
    thumbnail_width: int,
    show_masks: bool,
    mask_alpha: float,
    jpeg_quality: int,
) -> None:
    frame_paths = select_evenly(list_frame_paths(group_dir), max_frames)
    previews = [
        make_preview(
            frame_path=frame_path,
            thumbnail_width=thumbnail_width,
            show_masks=show_masks,
            mask_alpha=mask_alpha,
        )
        for frame_path in frame_paths
    ]
    sheet = make_contact_sheet(previews, columns)
    output_path = output_path_for_group(output_dir, group_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise click.ClickException(f"Could not write visualization: {output_path}")

    click.echo(f"{group_dir.name}: wrote {len(frame_paths)} frame(s) to {output_path}")


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--max-frames-per-video",
    type=int,
    default=40,
    show_default=True,
    help="Maximum number of frames to include per contact sheet.",
)
@click.option(
    "--columns",
    type=int,
    default=5,
    show_default=True,
    help="Number of columns in each contact sheet.",
)
@click.option(
    "--thumbnail-width",
    type=int,
    default=240,
    show_default=True,
    help="Width of each frame thumbnail in pixels.",
)
@click.option(
    "--show-masks/--hide-masks",
    default=True,
    show_default=True,
    help="Overlay matching foreground masks when present.",
)
@click.option(
    "--mask-alpha",
    type=click.FloatRange(0.0, 1.0),
    default=0.45,
    show_default=True,
    help="Opacity of foreground mask overlays.",
)
@click.option(
    "--jpeg-quality",
    type=click.IntRange(1, 100),
    default=95,
    show_default=True,
    help="JPEG quality for saved contact sheets.",
)
def main(
    input_dir: Path,
    output_dir: Path,
    max_frames_per_video: int,
    columns: int,
    thumbnail_width: int,
    show_masks: bool,
    mask_alpha: float,
    jpeg_quality: int,
) -> None:
    if max_frames_per_video <= 0:
        raise click.ClickException("--max-frames-per-video must be positive")
    if columns <= 0:
        raise click.ClickException("--columns must be positive")
    if thumbnail_width <= 0:
        raise click.ClickException("--thumbnail-width must be positive")

    groups = find_frame_groups(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for group_dir in groups:
        visualize_group(
            group_dir=group_dir,
            output_dir=output_dir,
            max_frames=max_frames_per_video,
            columns=columns,
            thumbnail_width=thumbnail_width,
            show_masks=show_masks,
            mask_alpha=mask_alpha,
            jpeg_quality=jpeg_quality,
        )

    click.echo(f"Wrote {len(groups)} contact sheet(s).")


if __name__ == "__main__":
    main()
