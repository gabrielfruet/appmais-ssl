"""Smoke-test the BeeCropDataset and save sample outputs.

Quantitative sanity checks (pre-flight + per-sample checks + the
``compare.jpg`` montage) run on the primary seed only. The other
contact-sheet seeds are visual aids: they iterate the dataset with a
different RNG seed so the population-level swap behaviour is visible
across multiple draws.

Usage:
    uv run python scripts/smoke_bee_dataset.py data/frames \\
        --num-samples 16 --output samples/bees
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import click
import cv2
import numpy as np
import torch

from engine.bee_crop import (
    crop_with_border,
    sample_center_from_distance_transform,
)
from engine.dataset import BeeCropDataset

CROP_SIZE = 128
SWAP_PROBABILITY = 0.5
CONTACT_SHEET_SEEDS: tuple[int, ...] = (0, 1, 2)
THUMB_SIZE = 224
GUTTER = 4
LUMINANCE_FLOOR = 5.0
ORIGINAL_LUMINANCE_FLOOR = 30.0
SWAP_DIFF_FLOOR = 5.0
BG_DETAIL_FLOOR = 8.0
CENTER_LOW = 0.4
CENTER_HIGH = 0.6
COVERAGE_LOW = 0.02
COVERAGE_HIGH = 0.95
SWAP_RATIO_LOW = 0.20
SWAP_RATIO_HIGH = 0.80


@dataclass(frozen=True)
class _SampleResult:
    index: int
    image_rgb: np.ndarray
    mask_classes: np.ndarray
    original_rgb: np.ndarray
    swapped: bool
    edt_peak: tuple[int, int]


def _tensor_to_uint8_rgb(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    array = (array * 255.0).clip(0, 255).astype(np.uint8)
    return np.transpose(array, (1, 2, 0))


def _load_original_crop(
    frame_path: Path,
    mask_path: Path,
    crop_size: int,
    min_area: int,
    index: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the original (unswapped) crop for a sample.

    Uses the same EDT-driven center sampling as the dataset
    (``seed + index + 2 + epoch=0``) so the crop matches what
    ``BeeCropDataset.__getitem__`` would return without the
    background swap. Raises if the mask has no foreground
    (should be unreachable after the dataset's init filter).
    """
    image_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image_bgr is None:
        raise ValueError(f"Could not read frame: {frame_path}")
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if not _has_foreground(mask, min_area):
        raise ValueError(f"No foreground components in {mask_path}")

    center_rng = np.random.default_rng(seed + index + 2)
    cy, cx = sample_center_from_distance_transform(mask, center_rng, crop_size)
    half = crop_size // 2
    x0 = cx - half
    y0 = cy - half
    window = (x0, y0, x0 + crop_size, y0 + crop_size)
    image_rgb = crop_with_border(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), window)
    mask_crop = crop_with_border(mask, window)
    return image_rgb, mask_crop


def _has_foreground(mask: np.ndarray, min_area: int) -> bool:
    binary = (mask == 255).astype(np.uint8)
    _num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    for label in range(1, stats.shape[0]):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            return True
    return False


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
    edt_peak: tuple[int, int], crop_size: int
) -> tuple[bool, float, float]:
    """Check the EDT peak (from the full mask) is centred in the crop.

    The pipeline centres the crop on the EDT peak of the strict
    foreground (with a small ±3 px jitter). The peak is stored in
    the dataset output in crop coordinates; this check verifies it
    lands inside the centre window, which is the only correct
    "is the bee centred?" proxy given the pipeline design (the
    foreground centroid is biased for asymmetric bees, and the EDT
    peak computed from the crop mask can land at the edge of a
    strict foreground that has internal holes).
    """
    cy_f = float(edt_peak[0])
    cx_f = float(edt_peak[1])
    low = CENTER_LOW * crop_size
    high = CENTER_HIGH * crop_size
    ok = low <= cy_f <= high and low <= cx_f <= high
    return ok, cx_f, cy_f


def _check_coverage(mask_classes: np.ndarray) -> tuple[bool, float]:
    """Check the bee (shadow + foreground) covers a reasonable fraction of the crop.

    The shadow ring is part of the bee per the shadow-halo feathering
    contract, so the coverage is measured on the relaxed foreground
    (``mask >= 1``). The lower bound catches tiny / off-frame bees;
    the upper bound catches samples where the EDT landed on a sliver
    of foreground at the edge of the frame.
    """
    foreground = mask_classes >= 1
    total_pixels = mask_classes.size
    coverage = float(foreground.sum()) / float(total_pixels)
    return COVERAGE_LOW <= coverage <= COVERAGE_HIGH, coverage


def _check_swap_diff(
    original_rgb: np.ndarray, swapped_rgb: np.ndarray, mask_classes: np.ndarray
) -> tuple[bool, float]:
    """Check the strict-background region changed between original and swapped.

    With shadow-halo feathering (``mask >= 127`` is in-bee), the shadow
    region is copied from the source frame, so it is *not* swapped.
    Only the strict background (``mask == 0``) is actually replaced.
    Checking the strict background gives a clean signal of swap quality.
    """
    background = mask_classes == 0
    if not background.any():
        return False, 0.0
    diff = np.abs(original_rgb.astype(np.int32) - swapped_rgb.astype(np.int32))
    mean_diff = float(diff[background].mean())
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


def _run_preflight(
    dataset: BeeCropDataset, num_samples: int, failures: list[str]
) -> None:
    if len(dataset) == 0:
        failures.append("dataset has zero samples")
    if len(dataset) < num_samples:
        failures.append(
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


def _check_sample(result: _SampleResult, failures: list[str], crop_size: int) -> None:
    if int((result.mask_classes == 2).sum()) == 0:
        failures.append(f"sample {result.index}: mask has zero foreground pixels")

    ok_brightness, mean_lum = _check_luminance(result.image_rgb, LUMINANCE_FLOOR)
    if not ok_brightness:
        failures.append(
            f"sample {result.index}: swapped mean luminance {mean_lum:.1f} "
            f"below {LUMINANCE_FLOOR}"
        )

    ok_orig, mean_orig = _check_luminance(result.original_rgb, ORIGINAL_LUMINANCE_FLOOR)
    if not ok_orig:
        failures.append(
            f"sample {result.index}: original mean luminance {mean_orig:.1f} "
            f"below {ORIGINAL_LUMINANCE_FLOOR}"
        )

    ok_center, cx, cy = _check_centering(result.edt_peak, crop_size)
    if not ok_center:
        failures.append(
            f"sample {result.index}: EDT peak ({cx:.1f}, {cy:.1f}) "
            f"outside [{CENTER_LOW * crop_size}, {CENTER_HIGH * crop_size}]"
        )
    ok_cov, coverage = _check_coverage(result.mask_classes)
    if not ok_cov:
        failures.append(
            f"sample {result.index}: coverage {coverage:.3f} outside "
            f"[{COVERAGE_LOW}, {COVERAGE_HIGH}]"
        )

    if not result.swapped:
        return
    ok_diff, mean_diff = _check_swap_diff(
        result.original_rgb, result.image_rgb, result.mask_classes
    )
    if not ok_diff:
        failures.append(
            f"sample {result.index}: swap diff {mean_diff:.1f} below {SWAP_DIFF_FLOOR}"
        )
    ok_detail, std = _check_background_detail(result.image_rgb, result.mask_classes)
    if not ok_detail:
        failures.append(
            f"sample {result.index}: swapped background region std {std:.1f} "
            f"below {BG_DETAIL_FLOOR}"
        )


def _save_sample_files(output_dir: Path, result: _SampleResult) -> None:
    suffix = "_swapped" if result.swapped else ""
    _save_uint8(
        output_dir / f"sample_{result.index:03d}{suffix}.jpg",
        cv2.cvtColor(result.image_rgb, cv2.COLOR_RGB2BGR),
    )
    _save_uint8(
        output_dir / f"sample_{result.index:03d}_original.jpg",
        cv2.cvtColor(result.original_rgb, cv2.COLOR_RGB2BGR),
    )
    mask_vis = result.mask_classes.astype(np.uint8) * 127
    cv2.imwrite(str(output_dir / f"sample_{result.index:03d}_mask.png"), mask_vis)


def _collect_samples(
    dataset: BeeCropDataset,
    num_samples: int,
    failures: list[str],
    output_dir: Path,
    crop_size: int,
    seed: int,
    save_samples: bool = True,
) -> list[_SampleResult]:
    results: list[_SampleResult] = []
    num_to_show = min(num_samples, len(dataset))
    for index in range(num_to_show):
        sample = dataset[index]
        image_tensor = cast(torch.Tensor, sample["image"])
        image_rgb = _tensor_to_uint8_rgb(image_tensor)
        mask_classes = cast(torch.Tensor, sample["mask"]).cpu().numpy()
        swapped = bool(sample["swapped"])
        edt_peak_tensor = cast(torch.Tensor, sample["edt_peak"])
        edt_peak = (int(edt_peak_tensor[0]), int(edt_peak_tensor[1]))

        sample_meta = dataset._samples[index]
        original_rgb, _ = _load_original_crop(
            frame_path=sample_meta.frame_path,
            mask_path=sample_meta.mask_path,
            crop_size=crop_size,
            min_area=dataset._min_area,
            index=index,
            seed=seed,
        )
        result = _SampleResult(
            index=index,
            image_rgb=image_rgb,
            mask_classes=mask_classes,
            original_rgb=original_rgb,
            swapped=swapped,
            edt_peak=edt_peak,
        )
        _check_sample(result, failures, crop_size)
        if save_samples:
            _save_sample_files(output_dir, result)
        results.append(result)
    return results


def _build_contact_sheet(
    results: Sequence[_SampleResult], output_dir: Path, seed: int
) -> None:
    """Write ``contact_sheet_<seed>.jpg`` from ``results`` (pure I/O)."""
    swapped_rgbs = [result.image_rgb for result in results]
    saved_swapped = [result.image_rgb for result in results if result.swapped]
    if saved_swapped:
        contact_sheet = _make_montage(
            saved_swapped, cols=4, thumb_w=THUMB_SIZE, thumb_h=THUMB_SIZE
        )
    else:
        click.echo(
            f"seed={seed}: no swapped samples; contact sheet uses unswapped crops."
        )
        contact_sheet = _make_montage(
            swapped_rgbs, cols=4, thumb_w=THUMB_SIZE, thumb_h=THUMB_SIZE
        )
    _save_uint8(output_dir / f"contact_sheet_{seed}.jpg", contact_sheet)


def _build_compare_sheet(results: Sequence[_SampleResult], output_dir: Path) -> None:
    """Write ``compare.jpg`` from the primary seed's results (pure I/O)."""
    compare_tiles: list[np.ndarray] = []
    for result in results:
        mask_vis = np.stack([result.mask_classes.astype(np.uint8) * 127] * 3, axis=-1)
        compare_tiles.extend([result.original_rgb, mask_vis, result.image_rgb])
    compare_sheet = _make_montage(
        compare_tiles, cols=3, thumb_w=THUMB_SIZE, thumb_h=THUMB_SIZE
    )
    _save_uint8(output_dir / "compare.jpg", compare_sheet)


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
@click.option(
    "--crop-size",
    type=int,
    default=CROP_SIZE,
    show_default=True,
    help="Bee crop size in pixels. The contact sheet's tile size is "
    "THUMB_SIZE, independent of this.",
)
@click.option(
    "--seeds",
    type=str,
    default=",".join(str(s) for s in CONTACT_SHEET_SEEDS),
    show_default=True,
    help="Comma-separated list of seeds for the contact sheets. The "
    "first seed is the primary run (quantitative checks + "
    "compare.jpg); the rest are visual aids.",
)
def main(
    root: Path, num_samples: int, output: Path, crop_size: int, seeds: str
) -> None:
    if num_samples <= 0:
        raise click.ClickException("--num-samples must be positive")
    if crop_size <= 0:
        raise click.ClickException("--crop-size must be positive")

    seen: set[int] = set()
    seed_list: list[int] = []
    for piece in seeds.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value in seen:
            continue
        seen.add(value)
        seed_list.append(value)
    if not seed_list:
        raise click.ClickException("--seeds must list at least one integer")

    primary_seed = seed_list[0]
    output.mkdir(parents=True, exist_ok=True)
    primary_dataset = BeeCropDataset(
        root=root,
        crop_size=crop_size,
        swap_background_prob=SWAP_PROBABILITY,
        seed=primary_seed,
    )

    failures: list[str] = []
    _run_preflight(primary_dataset, num_samples, failures)

    num_to_show = min(num_samples, len(primary_dataset))
    pool_size = len(primary_dataset.background_pool)
    click.echo(
        f"Dataset size: {len(primary_dataset)}, showing {num_to_show}, "
        f"pool size: {pool_size}, seeds: {seed_list}"
    )

    primary_results = _collect_samples(
        dataset=primary_dataset,
        num_samples=num_samples,
        failures=failures,
        output_dir=output,
        crop_size=crop_size,
        seed=primary_seed,
        save_samples=True,
    )
    swapped_count = sum(result.swapped for result in primary_results)
    swap_ratio = swapped_count / max(1, len(primary_results))
    if pool_size >= 2 and not (SWAP_RATIO_LOW <= swap_ratio <= SWAP_RATIO_HIGH):
        click.echo(
            f"WARNING: swap ratio {swap_ratio:.2f} outside "
            f"[{SWAP_RATIO_LOW}, {SWAP_RATIO_HIGH}] (pool size {pool_size})"
        )

    for seed in seed_list:
        if seed == primary_seed:
            sheet_results = primary_results
        else:
            other = BeeCropDataset(
                root=root,
                crop_size=crop_size,
                swap_background_prob=SWAP_PROBABILITY,
                seed=seed,
            )
            sheet_results = _collect_samples(
                dataset=other,
                num_samples=num_samples,
                failures=[],
                output_dir=output,
                crop_size=crop_size,
                seed=seed,
                save_samples=False,
            )
        _build_contact_sheet(sheet_results, output, seed)

    _build_compare_sheet(primary_results, output)
    click.echo(f"Items: {num_to_show}, swapped: {swapped_count}")

    if failures:
        click.echo("FAILED checks:")
        for failure in failures:
            click.echo(f"  - {failure}")
        raise click.ClickException(f"{len(failures)} sanity check(s) failed")

    click.echo("All sanity checks passed.")


if __name__ == "__main__":
    main()
