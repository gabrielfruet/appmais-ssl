"""Tests for the pure helpers in engine.bee_crop."""

import numpy as np

from engine.bee_crop import (
    BeeBBox,
    build_swapped_crop,
    find_bee_components,
    mask_to_classes,
    square_window,
)


def test_find_bee_components() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:40, 10:40] = 255
    mask[60:90, 60:90] = 255
    mask[5:7, 5:7] = 255
    comps = find_bee_components(mask, min_area=50)
    assert len(comps) == 2 and all(c.area == 900 for c in comps)
    assert find_bee_components(mask, min_area=1000) == []


def test_square_window() -> None:
    bbox = BeeBBox(x=50, y=50, w=20, h=20, area=400)
    x1, y1, x2, y2 = square_window(bbox, (200, 200), padding_factor=1.5)
    assert x2 - x1 == 30 and (x1 + x2) // 2 == 60 and (y1 + y2) // 2 == 60


def test_mask_to_classes() -> None:
    mask = np.array([[0, 127, 255], [255, 0, 127]], dtype=np.uint8)
    classes = mask_to_classes(mask)
    assert classes.dtype == np.int64
    assert (classes == np.array([[0, 1, 2], [2, 0, 1]])).all()


def test_build_swapped_crop_alignment() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60] = (255, 0, 0)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    bg = np.full((100, 100, 3), 50, dtype=np.uint8)
    swapped, mask_out = build_swapped_crop(img, mask, bg, (30, 30, 70, 70), 40)
    assert (swapped[10:30, 10:30] == (255, 0, 0)).all()
    assert (swapped[0:10, :] == 50).all()
    assert mask_out.shape == (40, 40) and (mask_out == 255).sum() == 400


def test_build_swapped_crop_resizes_background() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    bg_small = np.full((50, 50, 3), 200, dtype=np.uint8)
    swapped, _ = build_swapped_crop(img, mask, bg_small, (30, 30, 70, 70), 40)
    assert swapped.shape == (40, 40, 3) and (swapped[0:10, :] == 200).all()
