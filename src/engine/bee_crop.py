"""Pure helpers for the bee-centered crop dataset.

No I/O, no torch, no class state. Each helper takes numpy arrays and
returns numpy arrays (or small structured values). This keeps the
behaviour easy to unit-test and the dataset body thin.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import cv2
import numpy as np


class BeeBBox(NamedTuple):
    x: int
    y: int
    w: int
    h: int
    area: int


def find_bee_components(mask: np.ndarray, min_area: int) -> list[BeeBBox]:
    """Return connected foreground components in ``mask`` with area >= min_area.

    The mask uses the MOG2 convention: 0 = background, 127 = shadow,
    255 = foreground. We threshold to binary foreground (mask == 255)
    and run cv2.connectedComponentsWithStats.
    """
    binary = (mask == 255).astype(np.uint8)
    _num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    components: list[BeeBBox] = []
    for label in range(1, stats.shape[0]):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        components.append(BeeBBox(x=x, y=y, w=w, h=h, area=area))
    return components


def sample_bee_bbox(
    components: Sequence[BeeBBox], rng: np.random.Generator
) -> BeeBBox | None:
    """Pick one component uniformly at random. Returns None if empty."""
    if not components:
        return None
    index = int(rng.integers(0, len(components)))
    return components[index]


def crop_with_border(
    image: np.ndarray, window: tuple[int, int, int, int]
) -> np.ndarray:
    """Crop ``image`` to ``window``, padding with replicated border if needed."""
    x1, y1, x2, y2 = window
    height, width = image.shape[:2]
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(width, x2)
    crop_y2 = min(height, y2)
    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
    top = max(0, -y1)
    bottom = max(0, y2 - height)
    left = max(0, -x1)
    right = max(0, x2 - width)
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return cropped
    return cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_REPLICATE)


def build_swapped_crop(
    image: np.ndarray,
    mask: np.ndarray,
    background: np.ndarray,
    window: tuple[int, int, int, int],
    output_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut-paste a bee crop onto a different background.

    All inputs (``image`` and ``background``) are RGB uint8. ``mask``
    is the MOG2 grayscale mask using values {0, 127, 255} (background,
    shadow halo, foreground). The in-bee region is ``mask >= 127``
    (shadow + foreground), so the bee comes with its natural shadow
    halo from the source frame and the cut-out edge is naturally
    feathered. ``image`` and ``background`` are interpreted in the
    same (H, W) coordinate system; ``background`` is resized to the
    frame size if needed. The window is allowed to extend past image
    bounds; both crops are aligned by cropping at the same window in
    both scenes.
    """
    height, width = image.shape[:2]
    if background.shape[:2] != (height, width):
        background = cv2.resize(
            background, (width, height), interpolation=cv2.INTER_AREA
        )

    image_crop = crop_with_border(image, window)
    mask_crop = crop_with_border(mask, window)
    background_crop = crop_with_border(background, window)

    foreground = mask_crop == 255
    swapped = np.where(foreground[..., None], image_crop, background_crop)

    swapped_resized = cv2.resize(
        swapped, (output_size, output_size), interpolation=cv2.INTER_AREA
    )
    mask_resized = cv2.resize(
        mask_crop, (output_size, output_size), interpolation=cv2.INTER_NEAREST
    )
    return swapped_resized, mask_resized


def mask_to_classes(mask: np.ndarray) -> np.ndarray:
    """Map MOG2 mask values {0, 127, 255} to class indices {0, 1, 2} as int64."""
    classes = np.zeros(mask.shape, dtype=np.int64)
    classes[mask == 127] = 1
    classes[mask == 255] = 2
    return classes


def sample_center_from_distance_transform(
    mask: np.ndarray, rng: np.random.Generator, crop_size: int
) -> tuple[int, int]:
    """Return ``(cy, cx)`` sampled with probability proportional to the EDT of ``mask``.

    The binary foreground is ``(mask >= 1).astype(uint8)`` (bee body + shadow
    halo treated as one region). The EDT peak lands inside the bee body;
    deeper foreground pixels are more likely to be sampled as the crop
    center. The returned center is clamped so the ``crop_size x crop_size``
    window stays in bounds.

    Raises ``ValueError`` if the mask has no foreground or if ``crop_size``
    does not fit in the mask shape.
    """
    foreground = (mask >= 1).astype(np.uint8)
    edt = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
    flat = edt.ravel().astype(np.float64)
    total = float(flat.sum())
    if total <= 0:
        raise ValueError("mask has no foreground — caller should have filtered")
    choice = rng.choice(flat.size, p=flat / total)
    cy, cx = np.unravel_index(choice, edt.shape)
    height, width = foreground.shape
    half = int(crop_size) // 2
    if half <= 0 or 2 * half > width or 2 * half > height:
        raise ValueError(f"crop_size={crop_size} does not fit in {(height, width)}")
    cx = int(np.clip(cx, half, width - half - 1))
    cy = int(np.clip(cy, half, height - half - 1))
    return cy, cx
