"""Tests for the pure helpers in engine.bee_crop."""

import numpy as np
import pytest

from engine.bee_crop import (
    build_swapped_crop,
    find_bee_components,
    mask_to_classes,
    sample_center_from_distance_transform,
)


def test_find_bee_components() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:40, 10:40] = 255
    mask[60:90, 60:90] = 255
    mask[5:7, 5:7] = 255
    comps = find_bee_components(mask, min_area=50)
    assert len(comps) == 2 and all(c.area == 900 for c in comps)
    assert find_bee_components(mask, min_area=1000) == []


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


def test_swap_includes_shadow_halo() -> None:
    """`build_swapped_crop` uses `mask >= 127` (shadow + foreground)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60] = (10, 20, 30)  # dark "bee" body
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255  # foreground
    mask[38:40, 38:62] = 127  # shadow ring (top edge only)
    bg = np.full((100, 100, 3), 200, dtype=np.uint8)
    swapped, _ = build_swapped_crop(img, mask, bg, (30, 30, 70, 70), 40)
    # Top shadow ring lands in the crop at [8:10, 8:32]; the ring must come
    # from the source image (0, 0, 0), not the background (200, 200, 200).
    assert (swapped[8:10, 8:32] == 0).all(), (
        f"shadow ring pixel {swapped[8:10, 8:32][0]} should be the "
        "source image (0, 0, 0), not the background (200, 200, 200)"
    )


def test_sample_center_from_distance_transform() -> None:
    """Center is sampled at the EDT peak and clamped to valid bounds."""
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[11:21, 11:21] = 1
    rng = np.random.default_rng(0)
    for _ in range(20):
        cy, cx = sample_center_from_distance_transform(mask, rng, crop_size=8)
        # post-clamp: half=4, so cy in [4, 27] and cx in [4, 27]
        assert 4 <= cy < 28
        assert 4 <= cx < 28
    # No-foreground mask raises
    empty = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError):
        sample_center_from_distance_transform(empty, rng, crop_size=8)
    # Over-large crop_size raises
    small = np.zeros((16, 16), dtype=np.uint8)
    small[7:9, 7:9] = 1
    with pytest.raises(ValueError):
        sample_center_from_distance_transform(small, rng, crop_size=32)
