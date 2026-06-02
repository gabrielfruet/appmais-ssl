"""Create pseudo-foreground masks for extracted frames with DINOv3.

Usage:
    python scripts/dino_foreground_masks.py data/frames
"""

from pathlib import Path
from typing import Any, cast

import click
import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from tqdm import tqdm

FRAME_EXTENSIONS = {".jpeg", ".jpg"}
MODEL_NAME = "vit_small_patch16_dinov3"
THRESHOLD = 0.6
BATCH_SIZE = 8
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_frame_paths(input_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file()
        and path.suffix.lower() in FRAME_EXTENSIONS
        and not path.stem.endswith("_mask")
    ]


def find_frame_groups(input_dir: Path) -> list[Path]:
    if list_frame_paths(input_dir):
        return [input_dir]

    groups = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_dir() and list_frame_paths(path)
    ]
    if not groups:
        raise click.ClickException(f"No extracted JPG frames found in {input_dir}")
    return groups


def mask_path_for_frame(frame_path: Path) -> Path:
    return frame_path.with_name(f"{frame_path.stem}_mask.png")


def image_size_for_model(model: Any) -> int:
    input_size = model.default_cfg.get("input_size", (3, 256, 256))
    height = int(input_size[1])
    width = int(input_size[2])
    if height != width:
        raise click.ClickException(
            f"Expected a square model input size, got {input_size}"
        )
    return height


def frame_to_tensor(
    frame: np.ndarray, image_size: int, device: torch.device
) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    image = rgb.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0).to(device)


def cls_and_patch_tokens_from_features(
    features: torch.Tensor | dict[str, Any], model: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(features, dict):
        if "x_norm_clstoken" in features and "x_norm_patchtokens" in features:
            return features["x_norm_clstoken"].unsqueeze(1), features[
                "x_norm_patchtokens"
            ]
        features = cast(torch.Tensor, features.get("x", next(iter(features.values()))))

    prefix_tokens = getattr(model, "num_prefix_tokens", 1)
    return features[:, :1, :], features[:, prefix_tokens:, :]


def patch_grid_size(patch_tokens: torch.Tensor) -> int:
    patch_count = patch_tokens.shape[1]
    grid_size = int(patch_count**0.5)
    if grid_size * grid_size != patch_count:
        raise click.ClickException(
            f"Expected a square patch grid, got {patch_count} patch tokens"
        )
    return grid_size


def foreground_heatmaps_for_frames(
    frames: list[np.ndarray], model: Any, image_size: int, device: torch.device
) -> list[np.ndarray]:
    tensor = torch.cat(
        [frame_to_tensor(frame, image_size, device) for frame in frames], dim=0
    )

    with torch.inference_mode():
        features = model.forward_features(tensor)
        cls_token, patch_tokens = cls_and_patch_tokens_from_features(features, model)
        similarities = F.cosine_similarity(
            patch_tokens, cls_token.expand_as(patch_tokens), dim=-1
        )

    grid_size = patch_grid_size(patch_tokens)
    heatmaps = []
    for frame, similarity in zip(frames, similarities, strict=True):
        heatmap = similarity.reshape(grid_size, grid_size).float().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        heatmap = cv2.resize(
            heatmap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC
        )
        heatmaps.append(np.clip(heatmap, 0.0, 1.0))

    return heatmaps


def write_mask(frame_path: Path, heatmap: np.ndarray, threshold: float) -> bool:
    mask_path = mask_path_for_frame(frame_path)
    mask = np.zeros(heatmap.shape[:2], dtype=np.uint8)
    mask[heatmap >= threshold] = 255
    ok = cv2.imwrite(str(mask_path), mask)
    if not ok:
        raise click.ClickException(f"Could not write foreground mask: {mask_path}")
    return bool(np.any(mask))


def process_group(
    group_dir: Path,
    model: Any,
    image_size: int,
    device: torch.device,
    threshold: float,
    batch_size: int,
    overwrite: bool,
) -> int:
    frame_paths = list_frame_paths(group_dir)
    if not overwrite:
        frame_paths = [
            frame_path
            for frame_path in frame_paths
            if not mask_path_for_frame(frame_path).exists()
        ]

    written_count = 0
    foreground_count = 0
    batch_paths: list[Path] = []
    batch_frames: list[np.ndarray] = []

    def flush_batch() -> None:
        nonlocal written_count, foreground_count
        if not batch_frames:
            return

        heatmaps = foreground_heatmaps_for_frames(
            batch_frames, model=model, image_size=image_size, device=device
        )
        for frame_path, heatmap in zip(batch_paths, heatmaps, strict=True):
            if write_mask(frame_path, heatmap, threshold):
                foreground_count += 1
            written_count += 1

        batch_paths.clear()
        batch_frames.clear()

    for frame_path in tqdm(frame_paths, desc=group_dir.name, unit="frame", leave=False):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise click.ClickException(f"Could not read frame: {frame_path}")

        batch_paths.append(frame_path)
        batch_frames.append(frame)
        if len(batch_frames) >= batch_size:
            flush_batch()

    flush_batch()
    click.echo(
        f"{group_dir.name}: wrote {written_count} mask(s), "
        f"{foreground_count} with foreground pixels"
    )
    return written_count


@click.command()
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--model-name",
    default=MODEL_NAME,
    show_default=True,
    help="timm DINO model to use for pseudo-foreground masks.",
)
@click.option(
    "--threshold",
    type=click.FloatRange(0.0, 1.0),
    default=THRESHOLD,
    show_default=True,
    help="Normalized CLS-vs-patch similarity threshold for foreground pixels.",
)
@click.option(
    "--batch-size",
    type=int,
    default=BATCH_SIZE,
    show_default=True,
    help="Number of frames to process at once.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing _mask.png files instead of skipping them.",
)
def main(
    input_dir: Path,
    model_name: str,
    threshold: float,
    batch_size: int,
    overwrite: bool,
) -> None:
    if batch_size <= 0:
        raise click.ClickException("--batch-size must be positive")

    device = pick_device()
    click.echo(f"Loading {model_name} on {device}...")
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device)
    model.eval()
    image_size = image_size_for_model(model)

    groups = find_frame_groups(input_dir)
    total_written = 0
    for group_dir in groups:
        total_written += process_group(
            group_dir=group_dir,
            model=model,
            image_size=image_size,
            device=device,
            threshold=threshold,
            batch_size=batch_size,
            overwrite=overwrite,
        )

    click.echo(f"Wrote {total_written} DINO foreground mask(s).")


if __name__ == "__main__":
    main()
