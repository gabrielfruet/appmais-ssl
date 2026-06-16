"""Probe the AppMAIS archive by downloading a small, diverse video sample.

Reports per-video size, duration, fps, and frame count, plus aggregate
size stats. Useful for estimating disk and time budgets before launching
a large download. Safe to re-run; partial files are cleaned up on error.

To stay polite to the archive, only the most recent ``--probe-days`` days
per hive are scanned, and a configurable delay is applied between every
HTTP request (with exponential backoff on HTTP 429).

Usage:
    python scripts/probe_archive.py --count 5
    python scripts/probe_archive.py --count 10 --probe-days 14
    python scripts/probe_archive.py --count 5 --hives AppMAIS14L
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import httpx

from appmais import AppMaisClient

PROBE_DIRNAME = "data/probe"
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
DESCRIPTION = "Probe the AppMAIS archive by downloading a small, diverse video sample."


@dataclass(frozen=True)
class _VideoMeta:
    hive: str
    day: str
    time: str
    path: str
    size_bytes: int
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--count", type=int, default=5, help="Number of videos to download."
    )
    parser.add_argument(
        "--hives",
        nargs="+",
        default=None,
        help="Restrict probe to these hives. Defaults to all available hives.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Deterministic sample seed."
    )
    parser.add_argument(
        "--probe-days",
        type=int,
        default=7,
        help="Only scan the most recent N days per hive (politeness + speed).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between AppMAIS API requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retries for HTTP 429 rate-limit responses.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(PROBE_DIRNAME),
        help="Where to write probe videos and report.",
    )
    return parser.parse_args()


def retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def request_with_retries[T](
    label: str,
    args: argparse.Namespace,
    request: Callable[[], T],
) -> T:
    attempts = 0
    while True:
        if args.delay > 0:
            time.sleep(args.delay)
        try:
            return request()
        except httpx.HTTPStatusError as exc:
            if (
                exc.response.status_code not in TRANSIENT_STATUS_CODES
                or attempts >= args.max_retries
            ):
                raise
            attempts += 1
            wait = retry_after_seconds(exc.response) or max(
                args.delay, min(60.0, 2.0**attempts)
            )
            print(
                f"  rate limited during {label}; waiting {wait:.1f}s "
                f"(retry {attempts}/{args.max_retries})"
            )
            time.sleep(wait)


def pick_targets(
    client: AppMaisClient,
    args: argparse.Namespace,
    hives: list[str],
    count: int,
) -> list[tuple[str, str, str]]:
    """Sample (hive, day, time) tuples from a recent window per hive."""
    rng = random.Random(args.seed)
    pool: list[tuple[str, str, str]] = []
    for hive in hives:
        try:
            days = request_with_retries(
                f"list days for {hive}", args, lambda hive=hive: client.list_days(hive)
            )
        except Exception as exc:
            print(f"  skip {hive}: list_days failed ({exc})")
            continue

        recent_days = days[-args.probe_days :]
        for day in recent_days:
            try:
                times = request_with_retries(
                    f"list times for {hive} {day}",
                    args,
                    lambda hive=hive, day=day: client.list_times(hive, day),
                )
            except Exception as exc:
                print(f"  skip {hive}/{day}: list_times failed ({exc})")
                continue
            for time_str in times:
                pool.append((hive, day, time_str))

    if not pool:
        return []

    rng.shuffle(pool)
    chosen: list[tuple[str, str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    for candidate in pool:
        key = (candidate[0], candidate[1])
        if key in seen_keys:
            continue
        chosen.append(candidate)
        seen_keys.add(key)
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        for candidate in pool:
            if candidate in chosen:
                continue
            chosen.append(candidate)
            if len(chosen) >= count:
                break
    return chosen[:count]


def probe_video(
    client: AppMaisClient,
    args: argparse.Namespace,
    hive: str,
    day: str,
    time_str: str,
) -> _VideoMeta:
    video = request_with_retries(
        f"resolve {hive} {day} {time_str}",
        args,
        lambda: client.get_video(hive, day, time_str),
    )
    started = time.perf_counter()
    path = client.download(video, args.output)
    elapsed = time.perf_counter() - started
    size_bytes = path.stat().st_size

    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()

    duration = (frame_count / fps) if fps > 0 else 0.0
    print(
        f"  + {hive} {day} {time_str}  "
        f"{size_bytes / 1e6:.1f} MB  "
        f"{width}x{height}@{fps:.1f}fps  "
        f"{duration:.1f}s  "
        f"{elapsed:.1f}s download"
    )
    return _VideoMeta(
        hive=hive,
        day=day,
        time=time_str,
        path=str(path),
        size_bytes=size_bytes,
        duration_seconds=duration,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def summarize(metas: list[_VideoMeta]) -> dict[str, float]:
    if not metas:
        return {}
    sizes = [m.size_bytes for m in metas]
    durations = [m.duration_seconds for m in metas]
    frames = [m.frame_count for m in metas]
    return {
        "videos": len(metas),
        "size_min_mb": round(min(sizes) / 1e6, 2),
        "size_max_mb": round(max(sizes) / 1e6, 2),
        "size_median_mb": round(statistics.median(sizes) / 1e6, 2),
        "size_total_mb": round(sum(sizes) / 1e6, 2),
        "duration_median_s": round(statistics.median(durations), 2),
        "frames_total": sum(frames),
        "frames_per_video_median": int(statistics.median(frames)),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with AppMaisClient(timeout=60.0) as client:
        hives = request_with_retries("list hives", args, client.list_hives)
        if not hives:
            print("No hives available.", file=sys.stderr)
            return
        if args.hives:
            hives = [h for h in args.hives if h in hives]
        print(f"Probing across {len(hives)} hive(s): {hives}")
        print(
            f"Window: most recent {args.probe_days} day(s) per hive; "
            f"delay {args.delay}s."
        )

        targets = pick_targets(client, args, hives, args.count)
        if not targets:
            print("No (hive, day, time) tuples available to probe.")
            return

        metas: list[_VideoMeta] = []
        for hive, day, time_str in targets:
            try:
                metas.append(probe_video(client, args, hive, day, time_str))
            except Exception as exc:
                print(f"  ! {hive} {day} {time_str}: {exc}")
                # also write a half-finished report so the user sees what was tried
                errors = [
                    {
                        "hive": hive,
                        "day": day,
                        "time": time_str,
                        "reason": str(exc),
                    }
                ]
                report = {
                    "summary": summarize(metas),
                    "videos": [asdict(m) for m in metas],
                    "errors": errors,
                }
                report_path = args.output / "probe_report.json"
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = summarize(metas)
    report = {"summary": summary, "videos": [asdict(m) for m in metas]}
    report_path = args.output / "probe_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
