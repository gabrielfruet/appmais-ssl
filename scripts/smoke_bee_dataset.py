"""Smoke-test the BeeCropDataset and save sample outputs.

Usage:
    uv run python scripts/smoke_bee_dataset.py data/frames \\
        --num-samples 16 --output samples/bees
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import click
import cv2
import numpy as np
import torch

from engine.bee_crop import (
    BeeBBox,
    crop_with_border,
    find_bee_components,
    sample_bee_bbox,
    square_window,
)
from engine.dataset import BeeCropDataset

CROP_SIZE = 224
SWAP_PROBABILITY = 0.5
SEED = 0
THUMB_SIZE = 224
GUTTER = 4
LUMINANCE_FLOOR = 5.0
ORIGINAL_LUMINANCE_FLOOR = 30.0
SWAP_DIFF_FLOOR = 5.0
BG_DETAIL_FLOOR = 8.0
CENTER_LOW = 0.2
CENTER_HIGH = 0.8
COVERAGE_LOW = 0.02
COVERAGE_HIGH = 0.60
SWAP_RATIO_LOW = 0.20
SWAP_RATIO_HIGH = 0.80


def _tensor_to_uint8_rgb(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    array = (array * 255.0).clip(0, 255).astype(np.uint8)
    return np.transpose(array, (1, 2, 0))


def _load_original_crop(
    frame_path: Path,
    mask_path: Path,
    crop_size: int,
    padding_factor: float,
    min_area: int,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the original (unswapped) crop for a sample.

    Uses the same RNG seed as the dataset (SEED + index) so the picked
    component matches. Raises if no foreground component is found;
    after the dataset's init filter this should be unreachable in
    practice.
    """
    image_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image_bgr is None:
        raise ValueError(f"Could not read frame: {frame_path}")
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    components = find_bee_components(mask, min_area)
    if not components:
        raise ValueError(f"No foreground components in {mask_path}")

    rng = np.random.default_rng(SEED + index)
    bbox: BeeBBox = sample_bee_bbox(components, rng)  # type: ignore[assignment]
    if bbox is None:
        raise ValueError(f"Could not sample a component in {mask_path}")

    window = square_window(bbox, image_bgr.shape[:2], padding_factor)
    image_crop = crop_with_border(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), window)
    mask_crop = crop_with_border(mask, window)
    image_resized = cv2.resize(
        image_crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA
    )
    mask_resized = cv2.resize(
        mask_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST
    )
    return image_resized, mask_resized


def _save_uint8(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise click.ClickException(f"Could not write image: {path}")


def _make_montage(
    images: Sequence[np.ndarray], cols: int, thumb_w: int, thumb_h: int
) -> np.ndarray:
    if not images:
        raise ValueError("No images to montage")
    rows = (len(images) + cols - 1) // cols
    canvas = np.full(
        (rows * thumb_h + (rows + 1) * GUTTER, cols * thumb_w + (cols + 1) * GUTTER, 3),
        32,
        dtype=np.uint8,
    )
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        x = GUTTER + col * (thumb_w + GUTTER)
        y = GUTTER + row * (thumb_h + GUTTER)
        resized = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        canvas[y : y + thumb_h, x : x + thumb_w] = resized
    return canvas


def _check_centering(
    mask_classes: np.ndarray, crop_size: int
) -> tuple[bool, float, float]:
    foreground = mask_classes == 2
    total = int(foreground.sum())
    if total == 0:
        return False, -1.0, -1.0
    ys, xs = np.where(foreground)
    cy = float(ys.mean())
    cx = float(xs.mean())
    low = CENTER_LOW * crop_size
    high = CENTER_HIGH * crop_size
    ok = low <= cy <= high and low <= cx <= high
    return ok, cx, cy


def _check_coverage(mask_classes: np.ndarray) -> tuple[bool, float]:
    foreground = mask_classes == 2
    total_pixels = mask_classes.size
    coverage = float(foreground.sum()) / float(total_pixels)
    return COVERAGE_LOW <= coverage <= COVERAGE_HIGH, coverage


def _check_swap_diff(
    original_rgb: np.ndarray, swapped_rgb: np.ndarray, mask_classes: np.ndarray
) -> tuple[bool, float]:
    non_fg = mask_classes != 2
    if not non_fg.any():
        return False, 0.0
    diff = np.abs(original_rgb.astype(np.int32) - swapped_rgb.astype(np.int32))
    mean_diff = float(diff[non_fg].mean())
    return mean_diff >= SWAP_DIFF_FLOOR, mean_diff


def _check_luminance(image_rgb: np.ndarray, floor: float) -> tuple[bool, float]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mean = float(gray.mean())
    return mean >= floor, mean


def _check_background_detail(
    swapped_rgb: np.ndarray, mask_classes: np.ndarray
) -> tuple[bool, float]:
    background = mask_classes == 0
    if not background.any():
        return False, 0.0
    region = swapped_rgb[background]
    std = float(region.std())
    return std >= BG_DETAIL_FLOOR, std


def _check_background_sizes(dataset: BeeCropDataset, failures: list[str]) -> None:
    """Fail if any background.png is significantly smaller than its frames."""
    for background_path in dataset.background_pool:
        bg = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        if bg is None:
            failures.append(f"could not read background: {background_path}")
            continue
        bg_h, bg_w = bg.shape[:2]
        video_dir = background_path.parent
        frame_paths = sorted(
            [
                p
                for p in video_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                and not p.stem.endswith("_mask")
            ]
        )
        if not frame_paths:
            failures.append(f"no frames found beside {background_path}")
            continue
        first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
        if first_frame is None:
            failures.append(f"could not read first frame in {video_dir}")
            continue
        frame_h, frame_w = first_frame.shape[:2]
        if bg_w < frame_w or bg_h < frame_h:
            failures.append(
                f"background.png in {video_dir.name} is {bg_w}x{bg_h}, "
                f"frames are {frame_w}x{frame_h} — blurry-background regression"
            )


@click.command()
@click.argument(
    "root", default="data/frames", type=click.Path(exists=True, path_type=Path)
)
@click.option("--num-samples", type=int, default=16, show_default=True)
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("samples/bees"),
)
def main(root: Path, num_samples: int, output: Path) -> None:
    if num_samples <= 0:
        raise click.ClickException("--num-samples must be positive")

    output.mkdir(parents=True, exist_ok=True)
    dataset = BeeCropDataset(
        root=root,
        crop_size=CROP_SIZE,
        swap_background_prob=SWAP_PROBABILITY,
        seed=SEED,
    )

    failures: list[str] = []

    # Pre-flight checks (these are the new strictness gates).
    if len(dataset) == 0:
        raise click.ClickException(f"No samples found in {root}")
    if len(dataset) < num_samples:
        raise click.ClickException(
            f"dataset has {len(dataset)} samples, "
            f"need at least num_samples={num_samples}"
        )
    pool_size = len(dataset.background_pool)
    if pool_size < 2:
        click.echo(
            f"WARNING: background pool has {pool_size} entry; "
            "swap-diff check is toothless with a single background."
        )
    _check_background_sizes(dataset, failures)

    num_to_show = min(num_samples, len(dataset))
    click.echo(
        f"Dataset size: {len(dataset)}, showing {num_to_show}, pool size: {pool_size}"
    )

    swapped_count = 0
    saved_swapped: list[np.ndarray] = []
    original_rgbs: list[np.ndarray] = []
    mask_classes_list: list[np.ndarray] = []
    swapped_rgbs: list[np.ndarray] = []

    for index in range(num_to_show):
        sample = dataset[index]
        image_tensor = cast(torch.Tensor, sample["image"])
        swapped_rgb = _tensor_to_uint8_rgb(image_tensor)
        mask_classes = cast(torch.Tensor, sample["mask"]).cpu().numpy()
        swapped_flag = bool(sample["swapped"])

        # Reproduce the original (unswapped) crop for side-by-side comparison.
        sample_meta = dataset._samples[index]
        frame_path, mask_path = sample_meta.frame_path, sample_meta.mask_path
        original_rgb, _ = _load_original_crop(
            frame_path=frame_path,
            mask_path=mask_path,
            crop_size=CROP_SIZE,
            padding_factor=dataset._padding_factor,
            min_area=dataset._min_area,
            index=index,
        )

        if swapped_flag:
            swapped_count += 1
            saved_swapped.append(swapped_rgb)

        original_rgbs.append(original_rgb)
        mask_classes_list.append(mask_classes)
        swapped_rgbs.append(swapped_rgb)

        # No-empty-mask: every sample must have foreground pixels.
        if int((mask_classes == 2).sum()) == 0:
            failures.append(f"sample {index}: mask has zero foreground pixels")

        # Luminance on the swapped crop (cheap, always checked).
        ok_brightness, mean_lum = _check_luminance(swapped_rgb, LUMINANCE_FLOOR)
        if not ok_brightness:
            failures.append(
                f"sample {index}: swapped mean luminance {mean_lum:.1f} "
                f"below {LUMINANCE_FLOOR}"
            )

        # Luminance on the original (catches full-hive or black fallbacks).
        ok_orig, mean_orig = _check_luminance(original_rgb, ORIGINAL_LUMINANCE_FLOOR)
        if not ok_orig:
            failures.append(
                f"sample {index}: original mean luminance {mean_orig:.1f} "
                f"below {ORIGINAL_LUMINANCE_FLOOR}"
            )

        # Centering + coverage on the foreground.
        ok_center, cx, cy = _check_centering(mask_classes, CROP_SIZE)
        if not ok_center:
            failures.append(
                f"sample {index}: foreground centroid ({cx:.1f}, {cy:.1f}) "
                f"outside [{CENTER_LOW * CROP_SIZE}, {CENTER_HIGH * CROP_SIZE}]"
            )
        ok_cov, coverage = _check_coverage(mask_classes)
        if not ok_cov:
            failures.append(
                f"sample {index}: coverage {coverage:.3f} outside "
                f"[{COVERAGE_LOW}, {COVERAGE_HIGH}]"
            )

        # Swap diff: non-foreground region of swapped must differ from original.
        if swapped_flag:
            ok_diff, mean_diff = _check_swap_diff(
                original_rgb, swapped_rgb, mask_classes
            )
            if not ok_diff:
                failures.append(
                    f"sample {index}: swap diff {mean_diff:.1f} below {SWAP_DIFF_FLOOR}"
                )
            ok_detail, std = _check_background_detail(swapped_rgb, mask_classes)
            if not ok_detail:
                failures.append(
                    f"sample {index}: swapped background region std {std:.1f} "
                    f"below {BG_DETAIL_FLOOR}"
                )

        # Save per-sample files.
        suffix = "_swapped" if swapped_flag else ""
        _save_uint8(
            output / f"sample_{index:03d}{suffix}.jpg",
            cv2.cvtColor(swapped_rgb, cv2.COLOR_RGB2BGR),
        )
        _save_uint8(
            output / f"sample_{index:03d}_original.jpg",
            cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR),
        )
        mask_vis = mask_classes.astype(np.uint8) * 127
        cv2.imwrite(str(output / f"sample_{index:03d}_mask.png"), mask_vis)

    swap_ratio = swapped_count / num_to_show
    if pool_size >= 2 and not (SWAP_RATIO_LOW <= swap_ratio <= SWAP_RATIO_HIGH):
        click.echo(
            f"WARNING: swap ratio {swap_ratio:.2f} outside "
            f"[{SWAP_RATIO_LOW}, {SWAP_RATIO_HIGH}] (pool size {pool_size})"
        )

    # Contact sheet: if no samples were swapped, fall back to unswapped crops.
    if saved_swapped:
        contact_sheet = _make_montage(
            saved_swapped, cols=4, thumb_w=THUMB_SIZE, thumb_h=THUMB_SIZE
        )
    else:
        click.echo("No swapped samples; contact sheet uses unswapped crops.")
        contact_sheet = _make_montage(
            swapped_rgbs, cols=4, thumb_w=THUMB_SIZE, thumb_h=THUMB_SIZE
        )
    _save_uint8(output / "contact_sheet.jpg", contact_sheet)

    compare_tiles: list[np.ndarray] = []
    for original_rgb, mask_classes, swapped_rgb in zip(
        original_rgbs, mask_classes_list, swapped_rgbs, strict=False
    ):
        mask_vis = np.stack([mask_classes.astype(np.uint8) * 127] * 3, axis=-1)
        compare_tiles.extend([original_rgb, mask_vis, swapped_rgb])
    compare_sheet = _make_montage(
        compare_tiles, cols=3, thumb_w=THUMB_SIZE, thumb_h=THUMB_SIZE
    )
    _save_uint8(output / "compare.jpg", compare_sheet)

    click.echo(f"Items: {num_to_show}, swapped: {swapped_count}")

    if failures:
        click.echo("FAILED checks:")
        for failure in failures:
            click.echo(f"  - {failure}")
        raise click.ClickException(f"{len(failures)} sanity check(s) failed")

    click.echo("All sanity checks passed.")


if __name__ == "__main__":
    main()
