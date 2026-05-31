"""Create a DINO-style heatmap overlay video.

Usage:
    python scripts/dino_video_heatmap.py input.mp4 output.mp4
"""

from pathlib import Path
from typing import Any, Literal, cast

import click
import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from tqdm import tqdm

MODEL_NAME = "vit_base_patch14_dinov2"
IMAGE_SIZE = 518
HEATMAP_ALPHA = 0.45
FRAME_ALPHA = 1.0 - HEATMAP_ALPHA
COLORMAP = cv2.COLORMAP_INFERNO
BATCH_SIZE = 1
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
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


def heatmaps_for_frames(
    frames: list[np.ndarray],
    model: Any,
    device: torch.device,
    similarity_mode: Literal["cls", "avg"] = "cls",
) -> list[np.ndarray]:
    tensor = torch.cat([frame_to_tensor(frame, device) for frame in frames], dim=0)

    with torch.inference_mode():
        features = model.forward_features(tensor)
        cls_token, patch_tokens = cls_and_patch_tokens_from_features(features, model)
        ref_token = (
            cls_token
            if similarity_mode == "cls"
            else patch_tokens.mean(dim=1, keepdim=True)
        )
        similarities = F.cosine_similarity(
            patch_tokens, ref_token.expand_as(patch_tokens), dim=-1
        )

    heatmaps = []
    for frame, similarity in zip(frames, similarities, strict=True):
        grid_size = int(similarity.numel() ** 0.5)
        heatmap = similarity[: grid_size * grid_size].reshape(grid_size, grid_size)
        heatmap = heatmap.float().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        heatmap = cv2.resize(
            heatmap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        heatmaps.append(heatmap)

    return heatmaps


def overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heatmap_u8 = (255 * heatmap).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_u8, COLORMAP)
    return cv2.addWeighted(frame, FRAME_ALPHA, colored, HEATMAP_ALPHA, 0.0)


@click.command()
@click.argument(
    "input_video", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("output_video", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--similarity-mode",
    type=click.Choice(["cls", "avg"], case_sensitive=False),
    default="cls",
    help=(
        "How to compute similarity: 'cls' uses the CLS token as reference, while "
        "'avg' uses the average of all patch tokens. 'cls' typically highlights "
        "the main subject, while 'avg' may produce more diffuse heatmaps."
    ),
)
@click.option(
    "--max-frames",
    type=int,
    default=None,
    help="Stop early; useful for quick smoke tests.",
)
def main(
    input_video: Path,
    output_video: Path,
    similarity_mode: Literal["avg", "cls"],
    max_frames: int | None,
) -> None:
    device = pick_device()
    click.echo(f"Loading {MODEL_NAME} on {device}...")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).to(device)
    model.eval()

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise click.ClickException(f"Could not open input video: {input_video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise click.ClickException(f"Could not open output video: {output_video}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    if max_frames is not None:
        total_frames = min(total_frames, max_frames) if total_frames else max_frames

    frame_count = 0
    batch_frames: list[np.ndarray] = []

    def flush_batch(progress: tqdm) -> None:
        nonlocal frame_count
        if not batch_frames:
            return
        heatmaps = heatmaps_for_frames(batch_frames, model, device, similarity_mode)
        for frame, heatmap in zip(batch_frames, heatmaps, strict=True):
            writer.write(overlay_heatmap(frame, heatmap))
            frame_count += 1
            progress.update(1)
        batch_frames.clear()

    try:
        with tqdm(total=total_frames, desc="Frames", unit="frame") as progress:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                batch_frames.append(frame)
                if len(batch_frames) >= BATCH_SIZE:
                    flush_batch(progress)

                if (
                    max_frames is not None
                    and (frame_count + len(batch_frames)) >= max_frames
                ):
                    break

            flush_batch(progress)
    finally:
        capture.release()
        writer.release()

    click.echo(f"Wrote {frame_count} frames to {output_video}")


if __name__ == "__main__":
    main()
