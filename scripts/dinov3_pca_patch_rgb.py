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
INFERENCE_MAX_SIZE = 1024
UPSAMPLE_METHOD = "nearest"
INFERENCE_DTYPE = "bfloat16"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def patch_size_for_model(model: Any) -> tuple[int, int]:
    patch_embed = getattr(model, "patch_embed", None)
    patch_size = getattr(patch_embed, "patch_size", (16, 16))
    return int(patch_size[0]), int(patch_size[1])


def pad_to_patch_multiple(
    image_bgr: np.ndarray, patch_size: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image_bgr.shape[:2]
    patch_h, patch_w = patch_size
    padded_h = ((height + patch_h - 1) // patch_h) * patch_h
    padded_w = ((width + patch_w - 1) // patch_w) * patch_w
    pad_bottom = padded_h - height
    pad_right = padded_w - width
    if pad_bottom == 0 and pad_right == 0:
        return image_bgr, (padded_h, padded_w)

    padded = cv2.copyMakeBorder(
        image_bgr,
        0,
        pad_bottom,
        0,
        pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return padded, (padded_h, padded_w)


def resize_for_inference(image_bgr: np.ndarray, inference_max_size: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    largest_side = max(height, width)
    scale = min(1.0, inference_max_size / largest_side)
    if scale == 1.0:
        return image_bgr

    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    return cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def image_to_tensor(
    image_bgr: np.ndarray, patch_size: tuple[int, int], device: torch.device
) -> tuple[torch.Tensor, tuple[int, int]]:
    image_bgr, padded_size = pad_to_patch_multiple(image_bgr, patch_size)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0).to(device), padded_size


def cls_and_patch_tokens_from_features(
    features: torch.Tensor | dict[str, Any], model: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(features, dict):
        if "x_norm_clstoken" in features and "x_norm_patchtokens" in features:
            return features["x_norm_clstoken"], features["x_norm_patchtokens"]
        features = cast(torch.Tensor, features.get("x", next(iter(features.values()))))

    prefix_tokens = getattr(model, "num_prefix_tokens", 1)
    return features[:, 0, :], features[:, prefix_tokens:, :]


def patch_grid_shape(
    patch_tokens: torch.Tensor,
    padded_size: tuple[int, int],
    patch_size: tuple[int, int],
) -> tuple[int, int]:
    padded_h, padded_w = padded_size
    patch_h, patch_w = patch_size
    grid_h = padded_h // patch_h
    grid_w = padded_w // patch_w
    patch_count = patch_tokens.shape[1]
    if grid_h * grid_w != patch_count:
        raise click.ClickException(
            f"Expected {grid_h}x{grid_w}={grid_h * grid_w} patch tokens, "
            f"got {patch_count}"
        )
    return grid_h, grid_w


def minmax(values: torch.Tensor) -> torch.Tensor:
    return (values - values.min()) / (values.max() - values.min() + 1e-6)


def pca_rgb_for_selected_patches(selected_embeddings: torch.Tensor) -> torch.Tensor:
    if selected_embeddings.shape[0] < 3:
        raise click.ClickException(
            "Threshold kept fewer than 3 patches; lower --threshold and try again."
        )

    centered = selected_embeddings.float() - selected_embeddings.float().mean(
        dim=0, keepdim=True
    )
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    scores = centered @ vh[:3].T

    rgb = torch.empty_like(scores)
    for channel in range(3):
        rgb[:, channel] = minmax(scores[:, channel])
    return rgb


def make_pca_visualization(
    image_bgr: np.ndarray,
    model: Any,
    device: torch.device,
    threshold: float,
    inference_max_size: int,
    upsample_method: int,
    inference_dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    original_h, original_w = image_bgr.shape[:2]
    inference_bgr = resize_for_inference(image_bgr, inference_max_size)
    inference_h, inference_w = inference_bgr.shape[:2]
    patch_size = patch_size_for_model(model)
    tensor, padded_size = image_to_tensor(inference_bgr, patch_size, device)
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=inference_dtype)

    with torch.inference_mode():
        features = model.forward_features(tensor)
        cls_token, patch_tokens = cls_and_patch_tokens_from_features(features, model)
        similarities = F.cosine_similarity(
            patch_tokens.float(),
            cls_token.float().unsqueeze(1).expand_as(patch_tokens),
            dim=-1,
        )[0]
        similarities = torch.nan_to_num(similarities, nan=0.0, posinf=1.0, neginf=-1.0)

    normalized_similarities = minmax(similarities.float())
    keep_mask = normalized_similarities >= threshold
    selected_embeddings = patch_tokens[0, keep_mask].detach().cpu()
    selected_rgb = pca_rgb_for_selected_patches(selected_embeddings)

    grid_h, grid_w = patch_grid_shape(patch_tokens, padded_size, patch_size)
    patch_rgb = torch.zeros((patch_tokens.shape[1], 3), dtype=torch.float32)
    patch_rgb[keep_mask.cpu()] = selected_rgb
    patch_rgb_image = patch_rgb.reshape(grid_h, grid_w, 3).numpy()

    patch_mask = keep_mask.reshape(grid_h, grid_w).cpu().numpy().astype(np.float32)
    padded_h, padded_w = padded_size
    output_rgb = cv2.resize(
        patch_rgb_image,
        (padded_w, padded_h),
        interpolation=upsample_method,
    )[:inference_h, :inference_w]
    mask_float = cv2.resize(
        patch_mask,
        (padded_w, padded_h),
        interpolation=upsample_method,
    )[:inference_h, :inference_w]

    if (inference_h, inference_w) != (original_h, original_w):
        output_rgb = cv2.resize(
            output_rgb,
            (original_w, original_h),
            interpolation=upsample_method,
        )
        mask_float = cv2.resize(
            mask_float,
            (original_w, original_h),
            interpolation=upsample_method,
        )

    output_rgb = np.clip(output_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    mask = (mask_float >= 0.5).astype(np.uint8) * 255
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
@click.option(
    "--inference-max-size",
    type=click.IntRange(16),
    default=INFERENCE_MAX_SIZE,
    show_default=True,
    help=(
        "Downsample the largest input side to at most this many pixels before "
        "DINO inference."
    ),
)
@click.option(
    "--upsample-method",
    type=click.Choice(["nearest", "bilinear", "bicubic"], case_sensitive=False),
    default=UPSAMPLE_METHOD,
    show_default=True,
    help=(
        "OpenCV interpolation used to upsample the PCA RGB and mask to the "
        "original input size."
    ),
)
@click.option(
    "--inference-dtype",
    type=click.Choice(["float32", "float16", "bfloat16"], case_sensitive=False),
    default=INFERENCE_DTYPE,
    show_default=True,
    help=(
        "Dtype used to load the DINO model and run forward inference. "
        "`bfloat16` is the recommended reduced-precision dtype (DINOv3's rotary "
        "embeddings can produce NaNs in plain `float16`); `float32` is the most "
        "accurate but slowest."
    ),
)
def main(
    input_image: Path,
    output_image: Path,
    model_name: str,
    threshold: float,
    mask_output: Path | None,
    inference_max_size: int,
    upsample_method: str,
    inference_dtype: str,
) -> None:
    image_bgr = cv2.imread(str(input_image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise click.ClickException(f"Could not read input image: {input_image}")

    device = pick_device()
    click.echo(f"Loading {model_name} on {device} ({inference_dtype})...")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model_dtype = dtype_map[inference_dtype.lower()]
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(
        device=device, dtype=model_dtype
    )
    model.eval()

    upsample_flag = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
    }[upsample_method.lower()]

    output_bgr, mask = make_pca_visualization(
        image_bgr=image_bgr,
        model=model,
        device=device,
        threshold=threshold,
        inference_max_size=inference_max_size,
        upsample_method=upsample_flag,
        inference_dtype=model_dtype,
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
