"""Tests for engine.dataset.BeeCropDataset."""

import pathlib

import cv2
import numpy as np
import pytest
import torch

from engine.dataset import (
    BeeCropDataset,
    _CropResult,
    _load_background,
    _Sample,
    _WindowInfo,
)


def _write_sample(
    directory: object,
    video: str,
    frame_stem: str,
    fg: bool = True,
) -> tuple[str, str]:
    vdir = pathlib.Path(str(directory)) / video
    vdir.mkdir(parents=True, exist_ok=True)
    img = np.full((100, 100, 3), 100, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    if fg:
        mask[30:70, 30:70] = 255
        img[30:70, 30:70] = (0, 0, 255)
    cv2.imwrite(str(vdir / f"{frame_stem}.jpg"), img)
    cv2.imwrite(str(vdir / f"{frame_stem}_mask.png"), mask)
    cv2.imwrite(
        str(vdir / "background.png"), np.full((100, 100, 3), 50, dtype=np.uint8)
    )
    return str(vdir / f"{frame_stem}.jpg"), str(vdir / f"{frame_stem}_mask.png")


def test_len_and_getitem_shapes(tmp_path: object) -> None:
    for v in ("vid_a", "vid_b"):
        _write_sample(tmp_path, v, "frame_000001")
    ds = BeeCropDataset(str(tmp_path), crop_size=64)
    assert len(ds) == 2
    item = ds[0]
    assert item["image"].dtype == torch.float32 and item["image"].shape == (3, 64, 64)
    assert item["mask"].dtype == torch.int64 and item["mask"].shape == (64, 64)
    assert item["bbox"].dtype == torch.float32 and item["bbox"].shape == (4,)
    assert isinstance(item["swapped"], bool)
    assert isinstance(item["video_id"], str) and isinstance(item["frame_id"], str)


def test_swap_probability_one(tmp_path: object) -> None:
    for v in ("vid_a", "vid_b"):
        _write_sample(tmp_path, v, "frame_000001")
    ds = BeeCropDataset(str(tmp_path), crop_size=64, swap_background_prob=1.0)
    assert all(ds[i]["swapped"] is True for i in range(len(ds)))


def test_transform_invoked(tmp_path: object) -> None:
    _write_sample(tmp_path, "vid_a", "frame_000001")

    def mark(sample: dict[str, object]) -> dict[str, object]:
        return {**sample, "marker": True}

    ds = BeeCropDataset(str(tmp_path), crop_size=64, transform=mark)
    assert ds[0]["marker"] is True


def test_no_bee_filtered_at_init(tmp_path: object) -> None:
    """Frames whose masks have no foreground are dropped at init."""
    _write_sample(tmp_path, "vid_a", "frame_with_bee", fg=True)
    _write_sample(tmp_path, "vid_a", "frame_no_bee", fg=False)
    ds = BeeCropDataset(str(tmp_path), crop_size=64)
    assert len(ds) == 1
    assert all((item["mask"] == 2).sum() > 0 for item in (ds[0],))


def test_set_epoch_changes_crop(tmp_path: object) -> None:
    """Different epochs sample different centers from the EDT."""
    _write_sample(tmp_path, "vid_a", "frame_with_bee", fg=True)
    ds = BeeCropDataset(str(tmp_path), crop_size=64)
    ds.set_epoch(0)
    a = ds[0]
    ds.set_epoch(1)
    b = ds[0]
    # Same frame, different epochs -> different center samples
    assert not (a["image"] == b["image"]).all()


def test_load_sample_raises_on_missing_frame() -> None:
    """`_load_sample` raises if the frame is unreadable."""
    ds = BeeCropDataset.__new__(BeeCropDataset)
    bad = _Sample(
        frame_path=pathlib.Path("/nonexistent.jpg"),
        mask_path=pathlib.Path("/nonexistent_mask.png"),
        video_id="vid_a",
        frame_id="bad",
    )
    with pytest.raises(ValueError, match="Could not read frame"):
        ds._load_sample(bad)


def test_sample_window_clamps_to_bounds(tmp_path: object) -> None:
    """`_sample_window` returns a window that fits in the source frame."""
    _write_sample(tmp_path, "vid_a", "frame_with_bee", fg=True)
    ds = BeeCropDataset(str(tmp_path), crop_size=64)
    sample = ds._samples[0]
    sample_data = ds._load_sample(sample)
    window_info = ds._sample_window(sample_data, idx=0)
    x0, y0, x2, y2 = window_info.window
    assert x0 >= 0 and x2 <= sample_data.width and (x2 - x0) == 64
    assert y0 >= 0 and y2 <= sample_data.height and (y2 - y0) == 64


def test_build_crop_forced_swap_returns_rgb_crop(tmp_path: object) -> None:
    """`_build_crop` hides the swap decision and returns crop metadata."""
    for video in ("vid_a", "vid_b"):
        _write_sample(tmp_path, video, "frame_with_bee", fg=True)
    ds = BeeCropDataset(str(tmp_path), crop_size=64, swap_background_prob=1.0)
    sample_data = ds._load_sample(ds._samples[0])
    window_info = ds._sample_window(sample_data, idx=0)
    crop = ds._build_crop(sample_data, window_info, idx=0)
    assert crop.swapped is True
    assert crop.image_rgb.shape == (64, 64, 3)
    assert crop.mask_crop.shape == (64, 64)


def test_assemble_output_builds_tensors(tmp_path: object) -> None:
    """`_assemble_output` converts numpy crops into the dataset output dict."""
    _write_sample(tmp_path, "vid_a", "frame_with_bee", fg=True)
    ds = BeeCropDataset(str(tmp_path), crop_size=64)
    sample_data = ds._load_sample(ds._samples[0])
    window_info = _WindowInfo(
        window=(0, 0, 64, 64),
        bbox=ds._sample_window(sample_data, idx=0).bbox,
    )
    image_rgb = np.full((64, 64, 3), 255, dtype=np.uint8)
    mask_crop = np.zeros((64, 64), dtype=np.uint8)
    mask_crop[20:40, 20:40] = 255
    crop = _CropResult(image_rgb=image_rgb, mask_crop=mask_crop, swapped=False)
    output = ds._assemble_output(ds._samples[0], crop, window_info)
    assert output["image"].dtype == torch.float32
    assert output["mask"].dtype == torch.int64
    assert output["image"].shape == (3, 64, 64)
    assert output["mask"].shape == (64, 64)


def test_load_background_returns_bgr_for_now(tmp_path: object) -> None:
    """Step 1 keeps the existing background color contract unchanged."""
    path = pathlib.Path(str(tmp_path)) / "bg.png"
    red_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    red_bgr[:, :, 2] = 255
    cv2.imwrite(str(path), red_bgr)
    loaded = _load_background(path, frame_shape=(10, 10))
    assert int(loaded[:, :, 0].mean()) == 0
    assert int(loaded[:, :, 2].mean()) == 255
