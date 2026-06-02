# Scripts

## `scripts/extract_video_frames.py`

Extracts representative frames from videos without saving every repetitive frame. The script samples candidate frames at a fixed time interval, then only saves a frame when it is visually different enough from the last saved frame.

```bash
uv run python scripts/extract_video_frames.py data/videos_raw data/frames
```

For one video:

```bash
uv run python scripts/extract_video_frames.py input.mp4 data/frames
```

Useful options:

- `--sample-every-seconds 2.0`: how often to inspect a candidate frame.
- `--diff-threshold 8.0`: minimum visual difference from the last saved frame.
- `--min-gap-seconds 5.0`: minimum time between saved frames.
- `--max-frames-per-video 200`: cap saved frames per video.
- `--foreground-masks`: save a MOG2 foreground mask beside each exported frame.
- `--mog2-history 500`: number of frames used by MOG2 to model the background.
- `--mog2-var-threshold 16.0`: MOG2 variance threshold; lower values make foreground detection more sensitive.
- `--overwrite`: delete existing JPG frames and foreground masks for a video and regenerate them.

Foreground masks are saved as PNG files with the same stem as each JPG plus `_mask`, for example `frame_000001_t000000.0s_mask.png`. Mask pixels use `0` for background, `127` for shadow, and `255` for foreground.

```bash
uv run python scripts/extract_video_frames.py data/videos_raw data/frames --foreground-masks
```

By default, videos with existing extracted JPG frames are skipped. This makes reruns safe and deterministic. If too many similar frames are saved, increase `--diff-threshold` or `--min-gap-seconds`. If too few frames are saved, decrease them. When `--foreground-masks` is enabled, the script reads each video sequentially so MOG2 can learn temporal background history; this is slower than the default seek-based extraction but produces better masks.

## `scripts/dino_video_heatmap.py`

Creates a DINO-style heatmap overlay video from an input video using CLS-vs-patch cosine similarity.

```bash
uv run python scripts/dino_video_heatmap.py input.mp4 output.mp4
```

The script automatically uses CUDA, then MPS, then CPU.
Use `--max-frames N` for quick smoke tests.

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
