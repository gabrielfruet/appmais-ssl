# Scripts

## `scripts/curate_videos.py`

Scores every `*.mp4` in a directory on sharpness (Laplacian variance), motion (mean absolute frame difference at 160x120 grayscale) and brightness (mean grayscale), samples from the middle 90% of each video, prints a ranked table of every video, and writes a curated list of the top-N diverse videos (one absolute path per line) to a text file. The "diverse" picking rule is greedy by score with a hive-id guard: the next picked video must come from a hive that isn't already represented in the current top-N.

```bash
uv run python scripts/curate_videos.py data/videos_raw \
    --output data/curated_videos.txt --top-n 8
```

Useful options:

- `--top-n 8`: number of diverse videos to keep after filtering.
- `--sharpness-min 100.0`: drop videos whose mean Laplacian variance is below this.
- `--motion-min 0.5`: drop videos whose mean inter-frame pixel diff is below this.
- `--brightness-min 60.0`: drop videos whose mean grayscale value is below this.
- `--num-samples 8`: frames to sample per video when computing scores.
- `--strict`: exit non-zero if the curated list ends up with fewer than `--top-n` entries (e.g. when a tighter filter starves the picker).

Output file format (`data/curated_videos.txt` by default): a `#`-prefixed comment header line followed by one absolute video path per line. `scripts/extract_video_frames.py --videos-file` reads this file directly.

## `scripts/extract_video_frames.py`

Extracts representative frames from videos without saving every repetitive frame. The script samples candidate frames at a fixed time interval, then only saves a frame when it is visually different enough from the last saved frame.

```bash
uv run python scripts/extract_video_frames.py data/videos_raw data/frames
```

For one video:

```bash
uv run python scripts/extract_video_frames.py input.mp4 data/frames
```

For a curated list of videos (produced by `scripts/curate_videos.py`):

```bash
uv run python scripts/extract_video_frames.py data/videos_raw data/frames \
    --videos-file data/curated_videos.txt \
    --foreground-masks --save-background --overwrite
```

Useful options:

- `--sample-every-seconds 2.0`: how often to inspect a candidate frame.
- `--diff-threshold 8.0`: minimum visual difference from the last saved frame.
- `--min-gap-seconds 5.0`: minimum time between saved frames.
- `--max-frames-per-video 200`: cap saved frames per video.
- `--skip-start-seconds 5.0`: ignore the first N seconds of each video before exporting frames.
- `--videos-file PATH`: optional text file with one video path per line (`#` comments and blank lines ignored). When set, only these videos are processed and the directory scan is skipped. Mutually exclusive in spirit with passing a directory that has many videos you don't want to process.
- `--min-bee-area 50`: minimum foreground component area (in pixels) for a frame to be saved. Frames whose MOG2 mask has no component of at least this size are skipped and counted in the per-video summary. Requires `--foreground-masks`; ignored otherwise.
- `--foreground-masks`: save a MOG2 foreground mask beside each exported frame.
- `--save-background`: after the MOG2 loop finishes, write the learned background image to `<video_output_dir>/background.png` as a BGR PNG. The image is produced by a second MOG2 trained on unblurred (sharp), downsampled frames, then upscaled with `cv2.INTER_CUBIC` to the full frame resolution (e.g. 640x480). The foreground-mask MOG2 is trained on Gaussian-blurred frames for stable detection, but its background is *not* used here. Requires `--foreground-masks`; ignored otherwise. Re-runs require `--overwrite` to refresh the file.
- `--mog2-history 500`: number of frames used by MOG2 to model the background (shared by both the mask and background MOG2s).
- `--mog2-var-threshold 4.0`: MOG2 variance threshold; lower values make foreground detection more sensitive.
- `--mog2-downsample-width 320`: run MOG2 on frames downsampled to this width, then resize masks back to the exported frame size. The mask MOG2 sees a Gaussian-blurred downsampled frame; the background MOG2 sees the same downsampled frame *without* the blur so its learned background stays sharp.
- `--overwrite`: delete existing JPG frames, foreground masks and background.png for a video and regenerate them.

Foreground masks are saved as PNG files with the same stem as each JPG plus `_mask`, for example `frame_000001_t000000.0s_mask.png`. Mask pixels use `0` for background, `127` for shadow, and `255` for foreground. When `--save-background` is also passed, a `background.png` is written to the same video directory at the full frame resolution.

```bash
uv run python scripts/extract_video_frames.py data/videos_raw data/frames --foreground-masks
```

By default, videos with existing extracted JPG frames are skipped, and the first 5 seconds of each video are ignored. This makes reruns safe and deterministic and avoids exporting startup frames. If too many similar frames are saved, increase `--diff-threshold` or `--min-gap-seconds`. If too few frames are saved, decrease them. When `--foreground-masks` is enabled, the script reads each video sequentially so MOG2 can learn temporal background history. The script runs **two MOG2 instances in lockstep** with identical hyperparameters: a *mask MOG2* fed Gaussian-blurred, downsampled frames (so the foreground mask is stable) and a *background MOG2* fed the same downsampled frame *without* the blur (so the learned background stays sharp). The mask MOG2 produces the per-frame foreground masks; the background MOG2's `getBackgroundImage()` is what `--save-background` writes to `background.png`. Saved masks are resized back to the exported frame size; the saved background is upsampled with INTER_CUBIC to the full frame size.

## `scripts/visualize_extracted_frames.py`

Creates static contact-sheet images for extracted video frames. Point it at the frame extraction output directory; it writes one JPG contact sheet per video subdirectory.

```bash
uv run python scripts/visualize_extracted_frames.py data/frames data/frame_visualizations
```

Useful options:

- `--max-frames-per-video 40`: maximum frames to show per contact sheet, selected evenly across the extracted frames.
- `--columns 5`: number of thumbnails per row.
- `--thumbnail-width 240`: thumbnail width in pixels.
- `--hide-masks`: do not overlay foreground masks.
- `--mask-alpha 0.45`: foreground mask overlay opacity.

If matching `_mask.png` files exist, the script overlays them by default: shadow pixels are yellow and foreground pixels are red.

## `scripts/smoke_bee_dataset.py`

Smoke-tests `engine.dataset.BeeCropDataset` by iterating N samples, saving per-sample original/mask/swapped crops, and building contact sheets plus a per-sample compare montage for visual inspection.

```bash
uv run python scripts/smoke_bee_dataset.py data/frames \
    --num-samples 16 --output samples/bees
```

Defaults: `ROOT=data/frames`, `num_samples=16`, `output=samples/bees`, `crop_size=128`, `seeds="0,1,2"`. The script instantiates the dataset with `crop_size=128` and `swap_background_prob=0.5`, then for each item saves:

- `sample_<idx:03d>_original.jpg` — the unswapped crop at the same window.
- `sample_<idx:03d>_mask.png` — the mask with class 0/1/2 scaled by 127.
- `sample_<idx:03d>{_swapped}.jpg` — the final (possibly swapped) crop; the `_swapped` suffix is added when the background was actually swapped.

It also writes the following montages:

- `contact_sheet_<seed>.jpg` — 4×4 grid of swapped crops, one per seed in `--seeds`. Contact sheets show population-level swap consistency across seeds.
- `compare.jpg` — 3-column montage (`original | mask | swapped`), one row per sample, built from the primary seed (first entry in `--seeds`). It shows per-sample detail.

Quantitative sanity checks (centering, coverage, swap diff, swap ratio, non-black) run on the primary seed only. A swap-ratio warning is logged (not failed) when the ratio falls outside `[0.2, 0.8]` with a pool of >=2 entries. The remaining seeds are visual aids: they iterate the dataset with a different RNG seed so the population-level swap behaviour is visible across multiple draws. Exit code is non-zero with a clear message if any hard check fails.

Useful options:

- `--crop-size INT`: bee crop size in pixels. The contact sheet's tile size is `THUMB_SIZE` (display), independent of this.
- `--seeds "0,1,2"`: comma-separated list of seeds for the contact sheets. The first seed is the primary run (quantitative checks + `compare.jpg`); the rest are visual aids.

The swap uses the MOG2 mask values `127` (shadow) and `255` (foreground) as the in-bee region (`mask >= 127`), so the bee is copied from the source frame together with its natural shadow halo. This naturally feathers the cut-out edge between the bee and the new background.

## `scripts/dinov3_pca_patch_rgb.py`

Visualizes one image with DINOv3 ViT-Large patch embeddings. It computes CLS-vs-patch cosine similarity, masks low-similarity patches, runs PCA on the remaining patch embeddings, and maps the first three PCA components to RGB.

```bash
uv run python scripts/dinov3_pca_patch_rgb.py beehive_entrance.jpg outputs/beehive_pca.png \
    --mask-output outputs/beehive_mask.png
```

Useful options:

- `--model-name vit_large_patch16_dinov3`: timm DINOv3 model to use.
- `--threshold 0.0`: normalized CLS-vs-patch similarity threshold; lower values keep more patches.
- `--mask-output PATH`: optionally save the binary relevant-patch mask.
- `--inference-max-size 1024`: downsample the largest input side before DINO inference.
- `--upsample-method nearest`: OpenCV interpolation used to upsample the PCA RGB and mask to the original input size. Choices: `nearest`, `bilinear`, `bicubic`, `lanczos4`.
- `--inference-dtype bfloat16`: dtype used to load the DINO model and run forward inference. Choices: `float32`, `float16`, `bfloat16`. `bfloat16` is recommended (DINOv3's rotary embeddings can produce NaNs in plain `float16`).
- `--threshold-list "0.1,0.2,0.3,0.4,0.5"`: comma-separated thresholds to sweep. The script runs DINO inference once and writes one PCA/mask pair per threshold; filenames get a `_t<threshold*100>` suffix.

The script automatically uses CUDA, then MPS, then CPU. It downsamples large images before DINO inference, pads only to the next patch multiple when needed, renders masked patches as black, and upsamples the PCA RGB image back to the original input size with the selected interpolation.

## `scripts/dino_video_heatmap.py`

Creates a DINO-style heatmap overlay video from an input video using CLS-vs-patch cosine similarity.

```bash
uv run python scripts/dino_video_heatmap.py input.mp4 output.mp4
```

The script automatically uses CUDA, then MPS, then CPU.
Use `--max-frames N` for quick smoke tests.

## `scripts/dino_foreground_masks.py`

Creates pseudo-foreground masks for already extracted JPG frames with DINOv3. The script uses CLS-vs-patch cosine similarity, normalizes each frame's patch heatmap to 0-1, thresholds it, and writes matching `_mask.png` files for `scripts/visualize_extracted_frames.py`.

```bash
uv run python scripts/dino_foreground_masks.py data/frames
```

Useful options:

- `--model-name vit_small_patch16_dinov3`: timm DINO model to use.
- `--threshold 0.6`: normalized similarity threshold; lower values create larger foreground masks.
- `--batch-size 8`: number of frames to process at once.
- `--overwrite`: replace existing `_mask.png` files instead of skipping them.

DINOv3 is not a segmentation model, so these masks are saliency-style pseudo-foreground masks rather than true object masks. The script automatically uses CUDA, then MPS, then CPU.

## `scripts/appmais_download_diverse.py`

Downloads a diverse AppMAIS video sample without listing every video upfront. It lists hives/days, shuffles hive/day pairs by seed, then lazily lists times only for the current hive/day.

```bash
uv run python scripts/appmais_download_diverse.py \
  --output data/appmais/videos \
  --count 500 \
  --seed 0 \
  --per-day 1
```

Resume files are stored in the output directory:

- `download_state.json` tracks the current hive/day position.
- `download_manifest.jsonl` records each attempted video as `downloaded`, `unavailable`, or `failed`.
- Partially downloaded files use `.part` and are only renamed to `.mp4` after success.

Use `--hives`, `--start-date`, and `--end-date` to restrict the sample. The script waits between AppMAIS requests by default; use `--delay SECONDS` and `--max-retries N` if the server returns rate-limit errors.

## `scripts/dino_pca_video.py`

Renders a **temporally-coherent** two-stage PCA-RGB video from DINOv3 patch tokens, designed for flicker-free bee clips. It uses a register-equipped model by default (`vit_small_patch16_dinov3`) for clean maps. By default it processes only the **middle 10 seconds** of the input (override with `--clip-seconds`, or `--clip-seconds 0` for the whole video).

The pipeline has two passes, both restricted to the clip window:

1. **Fit (once):** sample `--pca-fit-frames` evenly-spaced frames *within the window* and fit a *frozen* PCA basis:
   - **Stage A** — PCA over all patch tokens; the 1st component becomes the foreground mask (threshold at `--fg-quantile`).
   - **Stage B** — PCA over foreground patches only; the top 3 components become the RGB basis.
   - Each component's sign is fixed (positive skew) and per-component percentile clip anchors (`--clip-percentile`, default 1–99%) are recorded on the fit set.
2. **Render:** every frame is projected onto that frozen basis → temporally stable colors. Non-foreground patches render black.

```bash
uv run python scripts/dino_pca_video.py input.mp4 output.mp4
uv run python scripts/dino_pca_video.py input.mp4 output.mp4 \
    --side-by-side --mask-video --save-basis basis.npz
```

Useful options:

- `--model-name vit_small_patch16_dinov3`: timm DINO model. Default has registers (DINOv3 small). Register-equipped variants (DINOv2-reg, DINOv3) give cleaner maps than no-register variants.
- `--inference-size 1280`: longest input side (px), default ~2× the clips' native resolution. Above native it **upscales** the frame (with `INTER_LINEAR`) for a denser patch grid and more detail; below it downscales (with `INTER_AREA`). Patch count scales quadratically: 1280 (4800 patches/frame) is ~16× the forward cost of 640px (1200 patches) — ~1 fps on MPS. The mid-10s default keeps a full run to ~4 min.
- `--inference-dtype bfloat16`: model/forward dtype. `bfloat16` recommended (DINOv3's rotary embeddings can NaN in plain `float16`).
- `--pca-fit-frames 48`: number of evenly-spaced frames used to fit the (frozen) PCA basis.
- `--fg-quantile 0.60`: foreground = top `(1 - q)` of patches by stage-A 1st component (so `0.60` keeps the top 40%).
- `--clip-percentile 1.0`: per-component percentile clip `[p, 100-p]` before mapping to RGB.
- `--upsample bilinear`: patch-grid → frame-size interpolation. Choices: `nearest`, `bilinear`, `bicubic`, `lanczos4`.
- `--batch-size 8`: frames per forward batch.
- `--side-by-side`: also write `<output>_sidebyside.mp4` (original | PCA).
- `--mask-video`: also write `<output>_mask.mp4` (binary foreground mask).
- `--save-basis PATH` / `--load-basis PATH`: save or load the frozen PCA basis as `.npz`.
- `--clip-seconds 10.0`: render AND fit only the middle N seconds of the video (centered window). `0` = whole video.
- `--max-frames N`: cap the number of rendered frames within the clip window (smoke tests).

### Cross-clip visual comparison

To compare two clips fairly, use the **same** projection for both: fit on one clip and reuse for the other.

```bash
uv run python scripts/dino_pca_video.py clip_a.mp4 a_pca.mp4 --save-basis a.npz
uv run python scripts/dino_pca_video.py clip_b.mp4 b_pca.mp4 --load-basis a.npz
```

The script automatically uses CUDA, then MPS, then CPU. It pads each frame only to the next patch multiple (no square squash), and registers are stripped via `num_prefix_tokens` before the patch grid is formed, so they never appear as spurious grid cells.
