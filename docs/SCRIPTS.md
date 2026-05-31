# Scripts

## `scripts/dino_video_heatmap.py`

Creates a DINO-style heatmap overlay video from an input video using CLS-vs-patch cosine similarity.

```bash
uv run python scripts/dino_video_heatmap.py input.mp4 output.mp4
```

The script automatically uses CUDA, then MPS, then CPU.
Use `--max-frames N` for quick smoke tests.
