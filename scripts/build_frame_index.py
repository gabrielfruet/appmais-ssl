"""Build per-frame + per-video + dataset-level indexes from extracted frames.

Walks ``--frames-dir`` (one subdir per video with ``frame_NNNN.jpg``
files) and writes, inside ``--dataset-dir``:

- ``index.jsonl``         : one row per saved frame
- ``video_summary.jsonl`` : one row per video
- ``manifest.json``       : dataset-level metadata (version, counts,
                            source path, git commit, timestamp)

Usage:
    python scripts/build_frame_index.py --frames-dir data/dataset_v0/frames
    python scripts/build_frame_index.py \\
        --frames-dir data/dataset_v0/frames \\
        --source-videos data/videos --version v0
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

FRAME_PATTERN = re.compile(r"^frame_(\d+)_t[\d.]+s\.jpg$")
DESCRIPTION = (
    "Build per-frame + per-video + dataset-level indexes from extracted frames."
)


@dataclass(frozen=True)
class _FrameRow:
    video: str
    frame_path: str
    frame_idx: int
    size_bytes: int


@dataclass(frozen=True)
class _VideoSummary:
    video: str
    frame_count: int
    total_bytes: int
    first_mtime: float
    last_mtime: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="Path to the per-video frames directory (one subdir per video).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Where to write index.jsonl, video_summary.jsonl, manifest.json. "
            "Defaults to the parent of --frames-dir."
        ),
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v0",
        help="Version label to record in manifest.json.",
    )
    parser.add_argument(
        "--source-videos",
        type=Path,
        default=None,
        help=(
            "Optional path to the raw videos directory, recorded in the "
            "manifest for provenance."
        ),
    )
    return parser.parse_args()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _build_indexes(
    frames_dir: Path,
    dataset_dir: Path,
    version: str,
    source_videos: Path | None,
) -> dict[str, object]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    index_path = dataset_dir / "index.jsonl"
    summary_path = dataset_dir / "video_summary.jsonl"
    manifest_path = dataset_dir / "manifest.json"

    video_dirs = sorted(p for p in frames_dir.iterdir() if p.is_dir())
    if not video_dirs:
        print(f"No video subdirs found in {frames_dir}", file=sys.stderr)
        return {}

    project_root = dataset_dir.parent
    frame_rows: list[_FrameRow] = []
    summaries: list[_VideoSummary] = []
    total_bytes = 0

    for video_dir in video_dirs:
        frames: list[tuple[int, Path]] = []
        for path in video_dir.iterdir():
            match = FRAME_PATTERN.match(path.name)
            if match is None or not path.is_file():
                continue
            frames.append((int(match.group(1)), path))
        frames.sort(key=lambda item: item[0])

        if not frames:
            summaries.append(
                _VideoSummary(
                    video=video_dir.name,
                    frame_count=0,
                    total_bytes=0,
                    first_mtime=0.0,
                    last_mtime=0.0,
                )
            )
            continue

        video_total = 0
        first_mtime = frames[0][1].stat().st_mtime
        last_mtime = first_mtime
        for frame_idx, frame_path in frames:
            stat = frame_path.stat()
            size = stat.st_size
            video_total += size
            total_bytes += size
            last_mtime = max(last_mtime, stat.st_mtime)
            try:
                rel = frame_path.relative_to(project_root)
            except ValueError:
                rel = frame_path
            frame_rows.append(
                _FrameRow(
                    video=video_dir.name,
                    frame_path=str(rel),
                    frame_idx=frame_idx,
                    size_bytes=size,
                )
            )
        summaries.append(
            _VideoSummary(
                video=video_dir.name,
                frame_count=len(frames),
                total_bytes=video_total,
                first_mtime=first_mtime,
                last_mtime=last_mtime,
            )
        )

    with index_path.open("w", encoding="utf-8") as file:
        for row in frame_rows:
            file.write(json.dumps(asdict(row), sort_keys=True))
            file.write("\n")

    with summary_path.open("w", encoding="utf-8") as file:
        for row in summaries:
            file.write(json.dumps(asdict(row), sort_keys=True))
            file.write("\n")

    manifest = {
        "version": version,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "frame_count": len(frame_rows),
        "video_count": len(summaries),
        "total_bytes": total_bytes,
        "frames_dir": str(frames_dir),
        "source_videos": str(source_videos) if source_videos is not None else None,
        "git_commit": _git_commit(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "frame_count": len(frame_rows),
        "video_count": len(summaries),
        "total_bytes": total_bytes,
        "index_path": str(index_path),
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
    }


def main() -> None:
    args = parse_args()
    frames_dir = args.frames_dir.resolve()
    dataset_dir = (args.dataset_dir or frames_dir.parent).resolve()
    if not frames_dir.exists():
        raise SystemExit(f"--frames-dir does not exist: {frames_dir}")

    print(f"Frames dir:   {frames_dir}")
    print(f"Dataset dir:  {dataset_dir}")
    print(f"Version:      {args.version}")

    result = _build_indexes(
        frames_dir=frames_dir,
        dataset_dir=dataset_dir,
        version=args.version,
        source_videos=args.source_videos,
    )
    if not result:
        return
    print(f"Videos:       {result['video_count']}")
    print(f"Frames:       {result['frame_count']}")
    print(f"Total bytes:  {result['total_bytes']:,}")
    print(f"Index:        {result['index_path']}")
    print(f"Summary:      {result['summary_path']}")
    print(f"Manifest:     {result['manifest_path']}")


if __name__ == "__main__":
    main()
