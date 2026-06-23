"""Render a temporally-coherent two-stage PCA-RGB video from DINOv3 patch tokens.

Pipeline (designed for flicker-free bee clips):

  1. Fit, once, on a subsample of frames:
       - Stage A PCA on ALL patch tokens -> 1st component as a foreground mask.
       - Stage B PCA on foreground patches only -> 3 components -> RGB basis.
       - Fix each component's sign (positive skew) and record per-component
         percentile clip anchors (1-99%) on the fit set.
  2. Render every frame with that FROZEN basis -> temporally stable colors.

Usage:
    python scripts/dino_pca_video.py input.mp4 output.mp4
    python scripts/dino_pca_video.py input.mp4 output.mp4 --side-by-side --mask-video
    # Render a second clip with the SAME projection for a fair visual comparison:
    python scripts/dino_pca_video.py other.mp4 other_pca.mp4 --load-basis basis.npz
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click
import cv2
import numpy as np
import timm
import torch
from tqdm import tqdm

# Make the sibling script importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dinov3_pca_patch_rgb as dpr  # noqa: E402

MODEL_NAME = "vit_small_patch16_dinov3"
INFERENCE_SIZE = 1 * 1280  # longest input side (px); may upscale above native
INFERENCE_DTYPE = "bfloat16"
PCA_FIT_FRAMES = 48
CLIP_SECONDS = 10.0  # render + fit only the middle N seconds (0 = whole video)
FG_QUANTILE = 0.90  # keep top 20% by 1st-component projection
CLIP_PERCENTILE = 1.0  # clip [1, 99]
UPSAMPLE_METHOD = "bilinear"
UPSAMPLE_METHOD_CHOICES = ["nearest", "bilinear", "bicubic", "lanczos4"]
BATCH_SIZE = 8

# --- EUPE (Meta, Efficient Universal Perception Encoder) backend ---
# Clone: git clone https://github.com/facebookresearch/EUPE.git external/EUPE
EUPE_REPO_PATH = Path(__file__).resolve().parent.parent / "external" / "EUPE"
EUPE_ARCH_CHOICES = ["vitt16", "vits16", "vitb16"]
EUPE_WEIGHTS_URL = {
    "vitt16": "https://huggingface.co/facebook/EUPE-ViT-T/resolve/main/EUPE-ViT-T.pt",
    "vits16": "https://huggingface.co/facebook/EUPE-ViT-S/resolve/main/EUPE-ViT-S.pt",
    "vitb16": "https://huggingface.co/facebook/EUPE-ViT-B/resolve/main/EUPE-ViT-B.pt",
}


class PcaBasis:
    """Frozen PCA basis for two-stage PCA-RGB rendering.

    Stage A: 1st principal component over all patches -> foreground mask.
    Stage B: 3 principal components over foreground patches -> RGB.
    """

    def __init__(
        self,
        stage_a_mean: np.ndarray,
        stage_a_comp1: np.ndarray,
        fg_threshold: float,
        stage_b_mean: np.ndarray,
        stage_b_components: np.ndarray,
        clip_lo: np.ndarray,
        clip_hi: np.ndarray,
    ) -> None:
        self.stage_a_mean = stage_a_mean
        self.stage_a_comp1 = stage_a_comp1
        self.fg_threshold = fg_threshold
        self.stage_b_mean = stage_b_mean
        self.stage_b_components = stage_b_components  # (3, D)
        self.clip_lo = clip_lo  # (3,)
        self.clip_hi = clip_hi  # (3,)

    def save(self, path: Path) -> None:
        np.savez(
            path,
            stage_a_mean=self.stage_a_mean,
            stage_a_comp1=self.stage_a_comp1,
            fg_threshold=np.array(self.fg_threshold),
            stage_b_mean=self.stage_b_mean,
            stage_b_components=self.stage_b_components,
            clip_lo=self.clip_lo,
            clip_hi=self.clip_hi,
        )

    @classmethod
    def load(cls, path: Path) -> PcaBasis:
        data = np.load(path)
        return cls(
            stage_a_mean=data["stage_a_mean"],
            stage_a_comp1=data["stage_a_comp1"],
            fg_threshold=float(data["fg_threshold"]),
            stage_b_mean=data["stage_b_mean"],
            stage_b_components=data["stage_b_components"],
            clip_lo=data["clip_lo"],
            clip_hi=data["clip_hi"],
        )


def fit_stage_a(all_tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First component of PCA over all patches. Returns (mean, comp1[D])."""
    mean = all_tokens.mean(axis=0)
    centered = all_tokens - mean
    # Economy SVD on (N, D); principal axes are rows of Vt[:1].
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return mean, vt[0]


def fit_stage_b(fg_tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-3 components of PCA over foreground patches. Returns (mean, comps[3,D])."""
    mean = fg_tokens.mean(axis=0)
    centered = fg_tokens - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return mean, vt[:3]


def fix_sign(comp: np.ndarray, projections: np.ndarray) -> np.ndarray:
    """Flip a component so its projections over the fit set have positive skew.

    Makes the sign deterministic regardless of SVD sign convention.
    """
    skew_sign = np.sign(np.cbrt(np.mean(projections**3)))
    if skew_sign < 0:
        comp = -comp
    return comp


def fit_basis(
    fit_tokens: np.ndarray, fg_quantile: float, clip_percentile: float
) -> PcaBasis:
    if fit_tokens.shape[0] < 4:
        raise click.ClickException(
            f"Not enough fit patches ({fit_tokens.shape[0]}); "
            "increase --pca-fit-frames or --inference-size."
        )

    # Stage A: 1st component over all patches -> foreground.
    a_mean, a_comp1 = fit_stage_a(fit_tokens)
    a_proj = fit_tokens @ a_comp1  # subtracting mean cancels in sign/score
    a_comp1 = fix_sign(a_comp1, a_proj)
    a_proj = fit_tokens @ a_comp1
    fg_threshold = float(np.quantile(a_proj, fg_quantile))
    fg_mask = a_proj >= fg_threshold

    explained_var_ratio = (a_proj.var()) / (fit_tokens.var() + 1e-12)
    click.echo(
        f"Stage A: comp1 explained-var ratio {explained_var_ratio:.3f}, "
        f"fg threshold {fg_threshold:.4f} (q={fg_quantile:.2f}), "
        f"kept {int(fg_mask.sum())}/{len(fg_mask)} fit patches."
    )

    fg_tokens = fit_tokens[fg_mask]
    if fg_tokens.shape[0] < 4:
        raise click.ClickException(
            "Foreground kept fewer than 4 patches; lower --fg-quantile."
        )

    # Stage B: 3 components over foreground.
    b_mean, b_comps = fit_stage_b(fg_tokens)
    b_proj = fg_tokens @ b_comps.T
    for i in range(3):
        b_comps[i] = fix_sign(b_comps[i], b_proj[:, i])
    b_proj = fg_tokens @ b_comps.T
    clip_lo = np.percentile(b_proj, clip_percentile, axis=0)
    clip_hi = np.percentile(b_proj, 100.0 - clip_percentile, axis=0)
    click.echo(
        "Stage B: 3 components fit; clip anchors "
        f"lo={np.round(clip_lo, 3)} hi={np.round(clip_hi, 3)}"
    )

    return PcaBasis(
        stage_a_mean=a_mean,
        stage_a_comp1=a_comp1,
        fg_threshold=fg_threshold,
        stage_b_mean=b_mean,
        stage_b_components=b_comps,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
    )


def render_frame_rgb(
    patch_tokens: np.ndarray,  # (P, D) float32
    grid_h: int,
    grid_w: int,
    basis: PcaBasis,
    frame_h: int,
    frame_w: int,
    upsample_flag: int,
) -> np.ndarray:
    """Project patches onto the frozen basis and return a (H, W, 3) BGR image."""
    fg = (patch_tokens @ basis.stage_a_comp1) >= basis.fg_threshold

    rgb = np.zeros((patch_tokens.shape[0], 3), dtype=np.float32)
    if np.any(fg):
        fg_proj = (patch_tokens[fg] - basis.stage_b_mean) @ basis.stage_b_components.T
        fg_proj = np.clip(fg_proj, basis.clip_lo, basis.clip_hi)
        denom = np.where(
            basis.clip_hi - basis.clip_lo > 1e-6,
            basis.clip_hi - basis.clip_lo,
            1e-6,
        )
        rgb[fg] = (fg_proj - basis.clip_lo) / denom

    grid_img = rgb.reshape(grid_h, grid_w, 3)
    upsampled = cv2.resize(grid_img, (frame_w, frame_h), interpolation=upsample_flag)
    upsampled = np.clip(upsampled * 255.0, 0.0, 255.0).astype(np.uint8)
    # Built as RGB; cv2 expects BGR.
    return cv2.cvtColor(upsampled, cv2.COLOR_RGB2BGR)


def render_frame_mask(
    patch_tokens: np.ndarray, grid_h: int, grid_w: int, basis: PcaBasis
) -> np.ndarray:
    fg = (patch_tokens @ basis.stage_a_comp1) >= basis.fg_threshold
    return (fg.reshape(grid_h, grid_w).astype(np.uint8)) * 255


def resize_inference(image_bgr: np.ndarray, target_long_side: int) -> np.ndarray:
    """Resize so the longest side is ``target_long_side`` (up- or down-scale).

    Unlike a downscale-only helper, this can also *upscale* the input so
    ``--inference-size`` above the native resolution yields a denser patch grid.
    Uses ``INTER_AREA`` when shrinking and ``INTER_LINEAR`` when enlarging.
    """
    height, width = image_bgr.shape[:2]
    longest = max(height, width)
    if longest == target_long_side:
        return image_bgr
    scale = target_long_side / longest
    resized_w = max(1, round(width * scale))
    resized_h = max(1, round(height * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image_bgr, (resized_w, resized_h), interpolation=interp)


def frames_to_patch_tokens(
    frames: list[np.ndarray],
    model: Any,
    device: torch.device,
    inference_size: int,
    dtype: torch.dtype,
) -> tuple[list[np.ndarray], list[tuple[int, int]], list[tuple[int, int]]]:
    """Batched forward; returns per-frame (patch_tokens[P,D] cpu float32,
    padded_size, inference_size_hw)."""
    tensors: list[torch.Tensor] = []
    padded_sizes: list[tuple[int, int]] = []
    for frame in frames:
        inference_bgr = resize_inference(frame, inference_size)
        patch_size = dpr.patch_size_for_model(model)
        tensor, padded = dpr.image_to_tensor(inference_bgr, patch_size, device)
        tensors.append(tensor.to(dtype=dtype))
        padded_sizes.append(padded)

    batch = torch.cat(tensors, dim=0)
    with torch.inference_mode():
        features = model.forward_features(batch)
        _, patch_tokens = dpr.cls_and_patch_tokens_from_features(features, model)

    patch_tokens_cpu = patch_tokens.detach().cpu().float().numpy()
    return (
        list(patch_tokens_cpu),
        padded_sizes,
        [dpr.patch_size_for_model(model) for _ in frames],
    )


def grid_shape_from_tokens(
    patch_tokens: np.ndarray, padded_size: tuple[int, int], patch_size: tuple[int, int]
) -> tuple[int, int]:
    padded_h, padded_w = padded_size
    ph, pw = patch_size
    grid_h, grid_w = padded_h // ph, padded_w // pw
    if grid_h * grid_w != patch_tokens.shape[0]:
        raise click.ClickException(
            f"Patch grid mismatch: {grid_h}x{grid_w} != {patch_tokens.shape[0]}"
        )
    return grid_h, grid_w


def clip_window(total_frames: int, fps: float, clip_seconds: float) -> tuple[int, int]:
    """Centered [start, end) frame range covering ``clip_seconds`` seconds.

    ``clip_seconds <= 0`` (or unknown length) returns the whole video as
    ``(0, total_frames)``.
    """
    if total_frames <= 0 or clip_seconds <= 0:
        return 0, total_frames
    win = min(int(round(clip_seconds * fps)), total_frames)
    start = max((total_frames - win) // 2, 0)
    end = min(start + win, total_frames)
    return start, end


def sample_fit_frame_indices(start: int, end: int, n_fit: int) -> np.ndarray:
    """Evenly-spaced frame indices in [start, end) for fitting."""
    span = max(end - start, n_fit)
    n_fit = min(max(n_fit, 1), span)
    return (start + np.linspace(0, max(span - 1, 0), n_fit)).round().astype(np.int64)


def video_props(capture: cv2.VideoCapture) -> tuple[int, int, float, int]:
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return width, height, fps, total


def upsample_flag(name: str) -> int:
    return {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
    }[name.lower()]


def load_eupe_model(arch: str, dtype: torch.dtype, device: torch.device) -> Any:
    """Load an EUPE ViT (Meta) backbone from a local repo clone + HF weights.

    EUPE ViT is a DINOv3-family model: ``forward_features`` returns the same
    dict (``x_norm_clstoken`` / ``x_norm_patchtokens``) the rest of this
    pipeline already consumes, so no other pipeline changes are needed.
    """
    if not EUPE_REPO_PATH.is_dir():
        raise click.ClickException(
            "EUPE repo not found at "
            f"{EUPE_REPO_PATH}. Clone it first: "
            "git clone https://github.com/facebookresearch/EUPE.git "
            f"{EUPE_REPO_PATH}"
        )
    if str(EUPE_REPO_PATH) not in sys.path:
        sys.path.insert(0, str(EUPE_REPO_PATH))
    import importlib

    # Imported dynamically: ``external/EUPE`` is an optional local clone
    # resolved at runtime via sys.path, not a static package dependency.
    backbones = importlib.import_module("eupe.hub.backbones")

    builders = {
        "vitt16": backbones.eupe_vitt16,
        "vits16": backbones.eupe_vits16,
        "vitb16": backbones.eupe_vitb16,
    }
    model = builders[arch](pretrained=True, weights=EUPE_WEIGHTS_URL[arch])
    return model.to(device=device, dtype=dtype).eval()


def read_fit_tokens(
    video_path: Path,
    fit_indices: np.ndarray,
    model: Any,
    device: torch.device,
    inference_size: int,
    dtype: torch.dtype,
    batch_size: int,
) -> np.ndarray:
    """Seek to and forward only the fit frames; concatenate all their patch tokens."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise click.ClickException(f"Could not open input video: {video_path}")
    wanted = set(int(i) for i in fit_indices)
    idx = min(wanted) if wanted else 0
    if idx:
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
    tokens_list: list[np.ndarray] = []
    buffer: list[np.ndarray] = []
    pending: list[int] = []
    try:
        with tqdm(total=len(wanted), desc="Fit", unit="frame", leave=False) as pbar:
            while wanted:
                ok, frame = capture.read()
                if not ok:
                    break
                if idx in wanted:
                    buffer.append(frame)
                    pending.append(idx)
                    if len(buffer) >= batch_size:
                        tokens, _, _ = frames_to_patch_tokens(
                            buffer, model, device, inference_size, dtype
                        )
                        tokens_list.extend(tokens)
                        buffer.clear()
                        pbar.update(len(pending))
                        for i in pending:
                            wanted.discard(i)
                        pending.clear()
                idx += 1
            if buffer:
                tokens, _, _ = frames_to_patch_tokens(
                    buffer, model, device, inference_size, dtype
                )
                tokens_list.extend(tokens)
                pbar.update(len(pending))
    finally:
        capture.release()
    if not tokens_list:
        raise click.ClickException("No fit frames could be read.")
    return np.concatenate(tokens_list, axis=0)


@click.command()
@click.argument(
    "input_video", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("output_video", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--backend",
    type=click.Choice(["dinov3", "eupe"], case_sensitive=False),
    default="dinov3",
    show_default=True,
    help=(
        "Feature backbone. 'dinov3' loads a timm DINOv3 ViT; 'eupe' loads "
        "Meta's EUPE ViT (clone at external/EUPE) — same patch-token format."
    ),
)
@click.option(
    "--model-name",
    default=MODEL_NAME,
    show_default=True,
    help="timm DINO model used with --backend dinov3 (default: DINOv3 small).",
)
@click.option(
    "--eupe-arch",
    type=click.Choice(EUPE_ARCH_CHOICES, case_sensitive=False),
    default="vits16",
    show_default=True,
    help="EUPE ViT arch used with --backend eupe (vits16 ~ DINOv3 small).",
)
@click.option(
    "--inference-size",
    type=click.IntRange(64),
    default=INFERENCE_SIZE,
    show_default=True,
    help=(
        "Longest input side (px) before DINO inference. Above the native "
        "resolution upscales the frame (denser patch grid, more detail); "
        "below downscales it."
    ),
)
@click.option(
    "--inference-dtype",
    type=click.Choice(["float32", "float16", "bfloat16"], case_sensitive=False),
    default=INFERENCE_DTYPE,
    show_default=True,
    help="Model/forward dtype. bfloat16 recommended (float16 NaNs on RoPE).",
)
@click.option(
    "--pca-fit-frames",
    type=click.IntRange(4),
    default=PCA_FIT_FRAMES,
    show_default=True,
    help="Number of evenly-spaced frames used to fit the (frozen) PCA basis.",
)
@click.option(
    "--fg-quantile",
    type=click.FloatRange(0.0, 1.0),
    default=FG_QUANTILE,
    show_default=True,
    help="Foreground = top (1 - q) of patches by stage-A 1st component.",
)
@click.option(
    "--clip-percentile",
    type=click.FloatRange(0.0, 49.0),
    default=CLIP_PERCENTILE,
    show_default=True,
    help="Per-component percentile clip [p, 100-p] before mapping to RGB.",
)
@click.option(
    "--upsample",
    type=click.Choice(UPSAMPLE_METHOD_CHOICES, case_sensitive=False),
    default=UPSAMPLE_METHOD,
    show_default=True,
    help="Interpolation for patch-grid -> frame-size upsampling.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(1),
    default=BATCH_SIZE,
    show_default=True,
    help="Frames per forward batch.",
)
@click.option(
    "--side-by-side",
    is_flag=True,
    help="Also write <output>_sidebyside.mp4 (original | PCA).",
)
@click.option(
    "--mask-video",
    is_flag=True,
    help="Also write <output>_mask.mp4 (binary foreground mask).",
)
@click.option(
    "--save-basis",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Save the frozen PCA basis to this .npz file.",
)
@click.option(
    "--load-basis",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Load a frozen PCA basis instead of fitting (cross-clip comparability).",
)
@click.option(
    "--clip-seconds",
    type=click.FloatRange(0.0),
    default=CLIP_SECONDS,
    show_default=True,
    help=(
        "Render AND fit only the middle N seconds of the video (centered "
        "window). 0 = whole video."
    ),
)
@click.option(
    "--max-frames",
    type=click.IntRange(1),
    default=None,
    help="Cap the number of rendered frames (within the clip window).",
)
def main(
    input_video: Path,
    output_video: Path,
    backend: str,
    model_name: str,
    eupe_arch: str,
    inference_size: int,
    inference_dtype: str,
    pca_fit_frames: int,
    fg_quantile: float,
    clip_percentile: float,
    upsample: str,
    batch_size: int,
    side_by_side: bool,
    mask_video: bool,
    save_basis: Path | None,
    load_basis: Path | None,
    clip_seconds: float,
    max_frames: int | None,
) -> None:
    if save_basis is not None and load_basis is not None:
        raise click.ClickException(
            "--save-basis and --load-basis are mutually exclusive."
        )

    device = dpr.pick_device()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[inference_dtype.lower()]
    if backend.lower() == "eupe":
        click.echo(f"Loading EUPE {eupe_arch} on {device} ({inference_dtype})...")
        model = load_eupe_model(eupe_arch.lower(), dtype, device)
    else:
        click.echo(f"Loading {model_name} on {device} ({inference_dtype})...")
        model = timm.create_model(model_name, pretrained=True, num_classes=0).to(
            device=device, dtype=dtype
        )
        model.eval()

    cap_probe = cv2.VideoCapture(str(input_video))
    if not cap_probe.isOpened():
        raise click.ClickException(f"Could not open input video: {input_video}")
    width, height, fps, total_frames = video_props(cap_probe)
    cap_probe.release()
    win_start, win_end = clip_window(total_frames, fps, clip_seconds)
    if max_frames is not None and (win_end - win_start) > max_frames:
        win_end = win_start + max_frames
    render_count = max(win_end - win_start, 0)
    click.echo(
        f"Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames; "
        f"window [{win_start}, {win_end}) "
        f"({render_count if render_count else 'all'} frames)."
    )

    upsample_int = upsample_flag(upsample)

    # ----- Pass 1: fit or load basis -----
    if load_basis is not None:
        click.echo(f"Loading PCA basis from {load_basis}...")
        basis = PcaBasis.load(load_basis)
    else:
        fit_idx = sample_fit_frame_indices(win_start, win_end, pca_fit_frames)
        click.echo(f"Fitting PCA basis on {len(fit_idx)} frame(s)...")
        fit_tokens = read_fit_tokens(
            input_video, fit_idx, model, device, inference_size, dtype, batch_size
        )
        basis = fit_basis(fit_tokens, fg_quantile, clip_percentile)
        if save_basis is not None:
            save_basis.parent.mkdir(parents=True, exist_ok=True)
            basis.save(save_basis)
            click.echo(f"Saved PCA basis to {save_basis}")

    # ----- Pass 2: render with frozen basis -----
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writers = _make_writers(output_video, width, height, fps, side_by_side, mask_video)

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise click.ClickException(f"Could not re-open input video: {input_video}")
    if win_start:
        capture.set(cv2.CAP_PROP_POS_FRAMES, win_start)
    frame_idx = 0
    buffer: list[np.ndarray] = []
    try:
        with tqdm(total=render_count or None, desc="Render", unit="frame") as pbar:
            while render_count <= 0 or frame_idx + len(buffer) < render_count:
                ok, frame = capture.read()
                if not ok:
                    break
                buffer.append(frame)
                if len(buffer) >= batch_size:
                    _render_and_write_batch(
                        buffer,
                        model,
                        device,
                        inference_size,
                        dtype,
                        basis,
                        upsample_int,
                        writers,
                    )
                    pbar.update(len(buffer))
                    frame_idx += len(buffer)
                    buffer.clear()
            if buffer:
                _render_and_write_batch(
                    buffer,
                    model,
                    device,
                    inference_size,
                    dtype,
                    basis,
                    upsample_int,
                    writers,
                )
                pbar.update(len(buffer))
                frame_idx += len(buffer)
                buffer.clear()
    finally:
        capture.release()
        for w in writers.values():
            w["writer"].release()

    click.echo(f"Wrote {frame_idx} frame(s) to {output_video}.")


def _make_writers(
    output_video: Path,
    width: int,
    height: int,
    fps: float,
    side_by_side: bool,
    mask_video: bool,
) -> dict[str, dict[str, Any]]:
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writers: dict[str, dict[str, Any]] = {}

    def add(path: Path, w: int, h: int, key: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if not writer.isOpened():
            raise click.ClickException(f"Could not open output video: {path}")
        writers[key] = {"writer": writer, "path": path}

    add(output_video, width, height, "pca")
    if side_by_side:
        add(
            output_video.with_name(f"{output_video.stem}_sidebyside.mp4"),
            width * 2,
            height,
            "side",
        )
    if mask_video:
        add(
            output_video.with_name(f"{output_video.stem}_mask.mp4"),
            width,
            height,
            "mask",
        )
    return writers


def _render_and_write_batch(
    frames: list[np.ndarray],
    model: Any,
    device: torch.device,
    inference_size: int,
    dtype: torch.dtype,
    basis: PcaBasis,
    upsample_int: int,
    writers: dict[str, dict[str, Any]],
) -> None:
    tokens, padded_sizes, patch_sizes = frames_to_patch_tokens(
        frames, model, device, inference_size, dtype
    )
    for frame, toks, padded, psize in zip(
        frames, tokens, padded_sizes, patch_sizes, strict=True
    ):
        h, w = frame.shape[:2]
        grid_h, grid_w = grid_shape_from_tokens(toks, padded, psize)
        pca_bgr = render_frame_rgb(toks, grid_h, grid_w, basis, h, w, upsample_int)
        writers["pca"]["writer"].write(pca_bgr)
        if "side" in writers:
            side = np.concatenate([frame, pca_bgr], axis=1)
            writers["side"]["writer"].write(side)
        if "mask" in writers:
            mask = render_frame_mask(toks, grid_h, grid_w, basis)
            mask_bgr = cv2.cvtColor(
                cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST),
                cv2.COLOR_GRAY2BGR,
            )
            writers["mask"]["writer"].write(mask_bgr)


if __name__ == "__main__":
    main()
