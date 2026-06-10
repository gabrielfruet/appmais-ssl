"""Bee crop dataset with optional background swapping.

The dataset walks a directory of frame/mask pairs, drops any pair whose
mask has no foreground component of at least ``min_area``, then for
each kept sample samples a bee-centered crop from the foreground mask
and optionally pastes the crop onto a background drawn from a pool of
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
    sample_center_from_distance_transform,
)

FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class _Sample:
    frame_path: Path
    mask_path: Path
    video_id: str
    frame_id: str


@dataclass(frozen=True)
class _SampleData:
    frame_bgr: np.ndarray
    mask: np.ndarray
    height: int
    width: int


@dataclass(frozen=True)
class _WindowInfo:
    window: tuple[int, int, int, int]
    # Bbox is the sampled foreground component in crop coordinates' source frame.
    # The EDT center can differ from the bbox center; downstream tasks still use
    # the bbox as a useful component label.
    bbox: BeeBBox


@dataclass(frozen=True)
class _CropResult:
    image_rgb: np.ndarray
    mask_crop: np.ndarray
    swapped: bool


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


def _filter_samples_with_bees(samples: list[_Sample], min_area: int) -> list[_Sample]:
    """Drop samples whose mask has no foreground component of at least ``min_area``."""
    kept: list[_Sample] = []
    for sample in samples:
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if find_bee_components(mask, min_area):
            kept.append(sample)
    return kept


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_background(path: Path, frame_shape: tuple[int, int]) -> np.ndarray:
    """Load a background image and return it as RGB uint8.

    The dataset's color contract is RGB throughout: the source frame
    is converted from BGR to RGB in ``_build_crop``, so the background
    must be RGB here too. Otherwise the cut-paste mixes channels and
    the swapped crop ends up with a blue/peach cast. Resizes to the
    frame shape if needed.
    """
    background = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if background is None:
        raise ValueError(f"Could not read background image: {path}")
    height, width = frame_shape
    if background.shape[:2] != (height, width):
        background = cv2.resize(
            background, (width, height), interpolation=cv2.INTER_AREA
        )
    return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)


class BeeCropDataset(Dataset[dict[str, object]]):
    """Bee-centered crop dataset with optional background swapping."""

    def __init__(
        self,
        root: str | Path,
        crop_size: int = 224,
        min_area: int = 50,
        swap_background_prob: float = 0.5,
        background_pool: Sequence[str | Path] | None = None,
        transform: Callable[[dict[str, object]], dict[str, object]] | None = None,
        seed: int = 0,
    ) -> None:
        self._root = Path(root)
        self._crop_size = int(crop_size)
        self._min_area = int(min_area)
        self._seed = int(seed)
        self._epoch: int = 0
        self._transform = transform

        self._samples = discover_samples(self._root)
        if not self._samples:
            raise ValueError(f"No frame/mask pairs found under {self._root}")
        self._samples = _filter_samples_with_bees(self._samples, self._min_area)
        if not self._samples:
            raise ValueError(
                f"No frame/mask pairs with foreground components "
                f"(min_area={self._min_area}) under {self._root}"
            )

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

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch counter used by the center-sampling RNG.

        Same frame yields different crops across epochs (data
        augmentation); the same ``__getitem__`` call is reproducible
        within an epoch.
        """
        self._epoch = int(epoch)

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

    def _load_sample(self, sample: _Sample) -> _SampleData:
        frame_bgr = cv2.imread(str(sample.frame_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError(f"Could not read frame: {sample.frame_path}")
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Could not read mask: {sample.mask_path}")
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        height, width = frame_bgr.shape[:2]
        return _SampleData(frame_bgr=frame_bgr, mask=mask, height=height, width=width)

    def _sample_window(self, sample_data: _SampleData, idx: int) -> _WindowInfo:
        components = find_bee_components(sample_data.mask, self._min_area)
        if not components:
            sample = self._samples[idx]
            raise ValueError(
                f"No foreground components in {sample.frame_path} "
                f"(min_area={self._min_area})"
            )

        bbox = sample_bee_bbox(components, self._rng(idx))
        if bbox is None:
            raise ValueError("Could not sample a foreground component")

        center_rng = np.random.default_rng(self._seed + int(idx) + 2 + self._epoch)
        cy, cx = sample_center_from_distance_transform(
            sample_data.mask, center_rng, self._crop_size
        )
        half = self._crop_size // 2
        x0 = cx - half
        y0 = cy - half
        window = (x0, y0, x0 + self._crop_size, y0 + self._crop_size)
        return _WindowInfo(window=window, bbox=bbox)

    def _build_crop(
        self, sample_data: _SampleData, window_info: _WindowInfo, idx: int
    ) -> _CropResult:
        swap_rng = np.random.default_rng(self._seed + int(idx) + 1)
        do_swap = (
            self._swap_probability > 0.0 and swap_rng.random() < self._swap_probability
        )

        image_rgb = _bgr_to_rgb(sample_data.frame_bgr)
        if do_swap:
            sample = self._samples[idx]
            background_path = self._choose_background(sample.video_id, swap_rng)
            background = _load_background(
                background_path, frame_shape=(sample_data.height, sample_data.width)
            )
            crop_rgb, mask_crop = build_swapped_crop(
                image=image_rgb,
                mask=sample_data.mask,
                background=background,
                window=window_info.window,
                output_size=self._crop_size,
            )
            return _CropResult(image_rgb=crop_rgb, mask_crop=mask_crop, swapped=True)

        crop_rgb = crop_with_border(image_rgb, window_info.window)
        mask_crop = crop_with_border(sample_data.mask, window_info.window)
        return _CropResult(image_rgb=crop_rgb, mask_crop=mask_crop, swapped=False)

    def _assemble_output(
        self, sample: _Sample, crop: _CropResult, window_info: _WindowInfo
    ) -> dict[str, object]:
        image_tensor = (
            torch.from_numpy(crop.image_rgb).float().div(255.0).permute(2, 0, 1)
        )
        mask_classes = mask_to_classes(crop.mask_crop)
        mask_tensor = torch.from_numpy(mask_classes)

        x0, y0, _x2, _y2 = window_info.window
        bbox = window_info.bbox
        bbox_tensor = torch.tensor(
            [
                float(bbox.x - x0),
                float(bbox.y - y0),
                float(bbox.x + bbox.w - x0),
                float(bbox.y + bbox.h - y0),
            ],
            dtype=torch.float32,
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "bbox": bbox_tensor,
            "video_id": sample.video_id,
            "frame_id": sample.frame_id,
            "swapped": crop.swapped,
        }

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self._samples[idx]
        sample_data = self._load_sample(sample)
        window_info = self._sample_window(sample_data, idx)
        crop = self._build_crop(sample_data, window_info, idx)
        result = self._assemble_output(sample, crop, window_info)
        if self._transform is not None:
            result = self._transform(result)
        return result
