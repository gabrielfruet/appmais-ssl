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


def square_window(
    bbox: BeeBBox,
    image_shape: tuple[int, int],
    padding_factor: float,
) -> tuple[int, int, int, int]:
    """Return a square window (x1, y1, x2, y2) centered on ``bbox``.

    The window is allowed to extend outside the image bounds; consumers
    pad with replicated border. ``image_shape`` is (H, W).
    """
    height, width = image_shape
    side = int(round(max(bbox.w, bbox.h) * padding_factor))
    side = min(side, min(height, width))
    center_x = bbox.x + bbox.w // 2
    center_y = bbox.y + bbox.h // 2
    half = side // 2
    x1 = center_x - half
    y1 = center_y - half
    x2 = x1 + side
    y2 = y1 + side
    return x1, y1, x2, y2


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

    ``image`` and ``background`` are interpreted in the same (H, W)
    coordinate system; ``background`` is resized to the frame size if
    needed. The window is allowed to extend past image bounds; both
    crops are aligned by cropping at the same window in both scenes.
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
