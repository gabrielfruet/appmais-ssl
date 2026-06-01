# Scripts

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
