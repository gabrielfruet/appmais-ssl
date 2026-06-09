"""Tests for engine.dataset.BeeCropDataset."""

import cv2
import numpy as np
import torch

from engine.dataset import BeeCropDataset


def _write_sample(
    directory: object,
    video: str,
    frame_stem: str,
    fg: bool = True,
) -> tuple[str, str]:
    import pathlib

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
