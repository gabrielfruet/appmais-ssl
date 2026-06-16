# Scale Download — Plan

## Goal
Download 5,000–10,000 videos from the AppMAIS bee archive and end up
with 50,000–100,000 extracted frames, on the local machine, to feed
SSL pre-training and a small hand-labeled object-detection subset.

## Probe results (5 videos, 2 hives, last day)
- Resolution: 640×480 @ 25 fps
- Duration: ~71.8 s per video
- Frames per video: ~1,794 raw
- Size: median 5 MB, range 0.8–11.1 MB
- Download time: 1–1.5 s per video
- Archive: 52 hives total; HTTP 429s if hammered

## Pipeline (3 stages + 1 deferred)

### Stage 1 — Download
```bash
uv run python scripts/appmais_download_diverse.py \
    --output data/videos \
    --count 8000 --per-day 1 --seed 0 --delay 2
```
- Reuses the existing diverse downloader (2 s delay, exp backoff,
  `Retry-After`, manifest, resume).
- `--count 8000` sits in the middle of 5k–10k.
- `--per-day 1` + shuffling maximizes diversity across the archive.
- Output: `data/videos/*.mp4` + `data/videos/download_manifest.jsonl`.

### Stage 2 — Parallel extraction
```bash
uv run python scripts/extract_video_frames.py \
    data/videos data/dataset_v0/frames \
    --workers 8 --max-frames-per-video 15
```
- New `--workers N` option (multiprocessing fan-out). Default 1 keeps
  the original sequential behavior; 2–8 is a good range for a
  multi-core machine.
- `--max-frames-per-video 15` is a safety cap; the diff + min-gap +
  bee filters keep most videos at 5–12 frames.
- Time: ~1–3 h with 8 workers (vs 7–28 h sequential).
- Output: `data/dataset_v0/frames/<video_stem>/frame_NNNN.jpg`.

### Stage 3 — Frame index
```bash
uv run python scripts/build_frame_index.py \
    --frames-dir data/dataset_v0/frames \
    --source-videos data/videos \
    --version v0
```
- Walks frames, writes:
  - `data/dataset_v0/index.jsonl` — one row per saved frame
    (`{video, frame_path, frame_idx, size_bytes}`)
  - `data/dataset_v0/video_summary.jsonl` — one row per video
    (`{video, frame_count, total_bytes, first_mtime, last_mtime}`)
  - `data/dataset_v0/manifest.json` — dataset-level
    (`version, created_at, frame_count, video_count, total_bytes,
    frames_dir, source_videos, git_commit`)
- Time: seconds.
- Per-video summary surfaces videos that produced 0 frames (e.g. no
  bees, corrupt file) without scanning 8k directories by hand.

### Stage 4 (deferred) — Label candidate selection
- Will use the per-frame + per-video index to diversity-pick frames
  to label. Reuses the greedy-diverse-pick pattern from
  `scripts/curate_videos.py`. Output path TBD
  (`data/dataset_v0/labels/` vs sibling `data/labels_v0/`).

## Disk layout

```
data/
  probe/                          # from scripts/probe_archive.py
  videos/                         # raw mp4s, ~25–50 GB
    AppMAIS14L@2024-04-15@12-50-00.mp4
    ...
    download_manifest.jsonl
    download_state.json
  dataset_v0/                     # the dataset, version 0
    frames/
      AppMAIS14L@2024-04-15@12-50-00/
        frame_0001.jpg
        ...
    index.jsonl
    video_summary.jsonl
    manifest.json
    README.md
```

## Success criteria
- 5,000–10,000 mp4s in `data/videos/`, every manifest entry `downloaded`.
- 50,000–100,000 jpgs in `data/dataset_v0/frames/`.
- Per-video frame count: median 5–15, p95 < 20.
- Total disk < 70 GB.
- No `failed` entries in the manifest that we didn't knowingly skip.

## Risks
- **429s during download** — bump `--delay` to 3–5 s if seen.
- **Extraction variance** — some days have no bees (0 frames saved).
  Acceptable; the per-video summary surfaces it.
- **Mac thermals** — start with `--workers 6`, bump only if temps are fine.
- **One bad video** — extractor's existing per-video error handling
  is fine; the parallel path wraps each video in try/except so a single
  failure doesn't kill the batch.
