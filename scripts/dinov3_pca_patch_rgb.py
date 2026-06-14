"""Visualize DINOv3 ViT-Large patch embeddings with PCA RGB colors.

Usage:
    python scripts/dinov3_pca_patch_rgb.py input.jpg output.png
"""

from pathlib import Path
from typing import Any, cast

import click
import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F

MODEL_NAME = "vit_large_patch16_dinov3"
THRESHOLD = 0.6
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def image_size_for_model(model: Any) -> int:
    input_size = model.default_cfg.get("input_size", (3, 256, 256))
    height = int(input_size[1])
    width = int(input_size[2])
    if height != width:
        raise click.ClickException(
            f"Expected a square model input size, got {input_size}"
        )
    return height


def image_to_tensor(
    image_bgr: np.ndarray, image_size: int, device: torch.device
) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(
        image_rgb, (image_size, image_size), interpolation=cv2.INTER_CUBIC
    )
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0).to(device)


def cls_and_patch_tokens_from_features(
    features: torch.Tensor | dict[str, Any], model: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(features, dict):
        if "x_norm_clstoken" in features and "x_norm_patchtokens" in features:
            return features["x_norm_clstoken"], features["x_norm_patchtokens"]
        features = cast(torch.Tensor, features.get("x", next(iter(features.values()))))

    prefix_tokens = getattr(model, "num_prefix_tokens", 1)
    return features[:, 0, :], features[:, prefix_tokens:, :]


def patch_grid_size(patch_tokens: torch.Tensor) -> int:
    patch_count = patch_tokens.shape[1]
    grid_size = int(patch_count**0.5)
    if grid_size * grid_size != patch_count:
        raise click.ClickException(
            f"Expected a square patch grid, got {patch_count} patch tokens"
        )
    return grid_size


def minmax(values: torch.Tensor) -> torch.Tensor:
    return (values - values.min()) / (values.max() - values.min() + 1e-6)


def pca_rgb_for_selected_patches(selected_embeddings: torch.Tensor) -> torch.Tensor:
    if selected_embeddings.shape[0] < 3:
        raise click.ClickException(
            "Threshold kept fewer than 3 patches; lower --threshold and try again."
        )

    centered = selected_embeddings - selected_embeddings.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered.float(), full_matrices=False)
    scores = centered.float() @ vh[:3].T

    rgb = torch.empty_like(scores)
    for channel in range(3):
        rgb[:, channel] = minmax(scores[:, channel])
    return rgb


def make_pca_visualization(
    image_bgr: np.ndarray,
    model: Any,
    image_size: int,
    device: torch.device,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = image_to_tensor(image_bgr, image_size, device)

    with torch.inference_mode():
        features = model.forward_features(tensor)
        cls_token, patch_tokens = cls_and_patch_tokens_from_features(features, model)
        similarities = F.cosine_similarity(
            patch_tokens, cls_token.unsqueeze(1).expand_as(patch_tokens), dim=-1
        )[0]

    normalized_similarities = minmax(similarities.float())
    keep_mask = normalized_similarities >= threshold
    selected_embeddings = patch_tokens[0, keep_mask].detach().cpu()
    selected_rgb = pca_rgb_for_selected_patches(selected_embeddings)

    grid_size = patch_grid_size(patch_tokens)
    patch_rgb = torch.zeros((patch_tokens.shape[1], 3), dtype=torch.float32)
    patch_rgb[keep_mask.cpu()] = selected_rgb
    patch_rgb_image = patch_rgb.reshape(grid_size, grid_size, 3).numpy()

    patch_mask = (
        keep_mask.reshape(grid_size, grid_size).cpu().numpy().astype(np.uint8) * 255
    )
    output_rgb = cv2.resize(
        (patch_rgb_image * 255).astype(np.uint8),
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    mask = cv2.resize(
        patch_mask,
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return output_bgr, mask


@click.command()
@click.argument(
    "input_image", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("output_image", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--model-name",
    default=MODEL_NAME,
    show_default=True,
    help="timm DINOv3 ViT model to use.",
)
@click.option(
    "--threshold",
    type=click.FloatRange(0.0, 1.0),
    default=THRESHOLD,
    show_default=True,
    help="Normalized CLS-vs-patch cosine similarity threshold.",
)
@click.option(
    "--mask-output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to save the binary relevant-patch mask.",
)
def main(
    input_image: Path,
    output_image: Path,
    model_name: str,
    threshold: float,
    mask_output: Path | None,
) -> None:
    image_bgr = cv2.imread(str(input_image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise click.ClickException(f"Could not read input image: {input_image}")

    device = pick_device()
    click.echo(f"Loading {model_name} on {device}...")
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device)
    model.eval()
    image_size = image_size_for_model(model)

    output_bgr, mask = make_pca_visualization(
        image_bgr=image_bgr,
        model=model,
        image_size=image_size,
        device=device,
        threshold=threshold,
    )

    output_image.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_image), output_bgr):
        raise click.ClickException(f"Could not write output image: {output_image}")

    if mask_output is not None:
        mask_output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_output), mask):
            raise click.ClickException(f"Could not write mask image: {mask_output}")

    kept = int(np.count_nonzero(mask))
    total = int(mask.size)
    click.echo(f"Wrote {output_image} with {kept / total:.1%} relevant pixels.")


if __name__ == "__main__":
    main()
