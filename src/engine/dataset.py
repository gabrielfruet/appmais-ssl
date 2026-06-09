"""Bee crop dataset with optional background swapping.

The dataset walks a directory of frame/mask pairs, picks a connected
foreground component per sample, crops a square window around it, and
optionally pastes the crop onto a background drawn from a pool of
``background.png`` images saved by the frame-extraction pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from engine.bee_crop import (
    BeeBBox,
    build_swapped_crop,
    crop_with_border,
    find_bee_components,
    mask_to_classes,
    sample_bee_bbox,
    square_window,
)

FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class _Sample:
    frame_path: Path
    mask_path: Path
    video_id: str
    frame_id: str


def _looks_like_mask(frame_stem: str, mask_path: Path) -> bool:
    return mask_path.stem == f"{frame_stem}_mask"


def discover_samples(root: Path) -> list[_Sample]:
    """Walk ``root`` and return one ``_Sample`` per (frame, mask) pair."""
    samples: list[_Sample] = []
    for frame_path in sorted(root.rglob("*")):
        if not frame_path.is_file():
            continue
        if frame_path.suffix.lower() not in FRAME_EXTENSIONS:
            continue
        if frame_path.stem.endswith("_mask"):
            continue
        mask_path = frame_path.with_name(f"{frame_path.stem}_mask.png")
        if not mask_path.exists():
            continue
        if not _looks_like_mask(frame_path.stem, mask_path):
            continue
        samples.append(
            _Sample(
                frame_path=frame_path,
                mask_path=mask_path,
                video_id=frame_path.parent.name,
                frame_id=frame_path.stem,
            )
        )
    return samples


def discover_backgrounds(root: Path) -> list[Path]:
    return sorted(root.rglob("background.png"))


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _resize_crop(
    image_crop: np.ndarray, mask_crop: np.ndarray, crop_size: int
) -> tuple[np.ndarray, np.ndarray]:
    image_resized = cv2.resize(
        image_crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA
    )
    mask_resized = cv2.resize(
        mask_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST
    )
    return image_resized, mask_resized


def _load_background(path: Path, frame_shape: tuple[int, int]) -> np.ndarray:
    background = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if background is None:
        raise ValueError(f"Could not read background image: {path}")
    height, width = frame_shape
    if background.shape[:2] != (height, width):
        background = cv2.resize(
            background, (width, height), interpolation=cv2.INTER_AREA
        )
    return background


class BeeCropDataset(Dataset[dict[str, object]]):
    """Bee-centered crop dataset with optional background swapping."""

    def __init__(
        self,
        root: str | Path,
        crop_size: int = 224,
        padding_factor: float = 1.5,
        min_area: int = 50,
        swap_background_prob: float = 0.5,
        background_pool: Sequence[str | Path] | None = None,
        transform: Callable[[dict[str, object]], dict[str, object]] | None = None,
        seed: int = 0,
    ) -> None:
        self._root = Path(root)
        self._crop_size = int(crop_size)
        self._padding_factor = float(padding_factor)
        self._min_area = int(min_area)
        self._seed = int(seed)
        self._transform = transform

        self._samples = discover_samples(self._root)
        if not self._samples:
            raise ValueError(f"No frame/mask pairs found under {self._root}")

        if background_pool is not None:
            self._background_pool = [Path(p) for p in background_pool]
        else:
            self._background_pool = discover_backgrounds(self._root)

        if self._background_pool:
            self._swap_probability = float(swap_background_prob)
        else:
            self._swap_probability = 0.0

    @property
    def background_pool(self) -> list[Path]:
        return list(self._background_pool)

    def __len__(self) -> int:
        return len(self._samples)

    def _rng(self, idx: int) -> np.random.Generator:
        # Mirror torch DataLoader worker seeding by adding idx to base seed.
        return np.random.default_rng(self._seed + int(idx))

    def _no_bee_sample(
        self, sample: _Sample, frame_bgr: np.ndarray
    ) -> dict[str, object]:
        image_rgb = _bgr_to_rgb(frame_bgr)
        image_resized, _ = _resize_crop(
            image_rgb,
            np.zeros(image_rgb.shape[:2], dtype=np.uint8),
            self._crop_size,
        )
        image_tensor = (
            torch.from_numpy(image_resized).float().div(255.0).permute(2, 0, 1)
        )
        zeros_mask = torch.zeros((self._crop_size, self._crop_size), dtype=torch.int64)
        bbox_tensor = torch.tensor(
            [0.0, 0.0, float(self._crop_size), float(self._crop_size)],
            dtype=torch.float32,
        )
        result: dict[str, object] = {
            "image": image_tensor,
            "mask": zeros_mask,
            "bbox": bbox_tensor,
            "video_id": sample.video_id,
            "frame_id": sample.frame_id,
            "swapped": False,
        }
        if self._transform is not None:
            result = self._transform(result)
        return result

    def _choose_background(
        self, current_video_id: str, rng: np.random.Generator
    ) -> Path:
        if len(self._background_pool) == 1:
            return self._background_pool[0]
        other = [
            path
            for path in self._background_pool
            if path.parent.name != current_video_id
        ]
        if other:
            return other[int(rng.integers(0, len(other)))]
        return self._background_pool[int(rng.integers(0, len(self._background_pool)))]

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self._samples[idx]
        rng = self._rng(idx)

        frame_bgr = cv2.imread(str(sample.frame_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError(f"Could not read frame: {sample.frame_path}")
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Could not read mask: {sample.mask_path}")
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        height, width = frame_bgr.shape[:2]
        components = find_bee_components(mask, self._min_area)
        if not components:
            return self._no_bee_sample(sample, frame_bgr)

        bbox: BeeBBox = sample_bee_bbox(components, rng)  # type: ignore[assignment]
        window = square_window(
            bbox=bbox,
            image_shape=(height, width),
            padding_factor=self._padding_factor,
        )
        window_x1, window_y1, window_x2, window_y2 = window
        window_w = window_x2 - window_x1
        window_h = window_y2 - window_y1

        do_swap = self._swap_probability > 0.0 and rng.random() < self._swap_probability

        if do_swap:
            background_path = self._choose_background(sample.video_id, rng)
            background = _load_background(background_path, frame_shape=(height, width))
            image_rgb, mask_resized = build_swapped_crop(
                image=_bgr_to_rgb(frame_bgr),
                mask=mask,
                background=background,
                window=window,
                output_size=self._crop_size,
            )
            swapped = True
        else:
            image_crop = crop_with_border(_bgr_to_rgb(frame_bgr), window)
            mask_crop = crop_with_border(mask, window)
            image_rgb, mask_resized = _resize_crop(
                image_crop, mask_crop, self._crop_size
            )
            swapped = False

        image_tensor = torch.from_numpy(image_rgb).float().div(255.0).permute(2, 0, 1)
        mask_classes = mask_to_classes(mask_resized)
        mask_tensor = torch.from_numpy(mask_classes)

        scale_x = self._crop_size / float(window_w)
        scale_y = self._crop_size / float(window_h)
        bbox_tensor = torch.tensor(
            [
                float((bbox.x - window_x1) * scale_x),
                float((bbox.y - window_y1) * scale_y),
                float((bbox.x + bbox.w - window_x1) * scale_x),
                float((bbox.y + bbox.h - window_y1) * scale_y),
            ],
            dtype=torch.float32,
        )

        result: dict[str, object] = {
            "image": image_tensor,
            "mask": mask_tensor,
            "bbox": bbox_tensor,
            "video_id": sample.video_id,
            "frame_id": sample.frame_id,
            "swapped": swapped,
        }

        if self._transform is not None:
            result = self._transform(result)
        return result
