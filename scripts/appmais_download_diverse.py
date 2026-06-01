from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import time
from collections.abc import Callable, Iterable
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, cast

import httpx

from appmais import AppMaisClient, AppMaisVideo, parse_date

Status = Literal["downloaded", "unavailable", "failed"]
VideoKey = tuple[str, str, str]

UNAVAILABLE_STATUS_CODES = {403, 404, 410}
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class Args(argparse.Namespace):
    output: Path
    count: int
    seed: int
    per_day: int
    hives: list[str] | None
    start_date: dt.date | None
    end_date: dt.date | None
    delay: float
    max_retries: int


class ManifestRecord(TypedDict):
    status: Status
    hive: str
    day: str
    time: str
    path: NotRequired[str]
    reason: NotRequired[str]


class State(TypedDict):
    signature: dict[str, Any]
    next_day_index: int


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output / "download_manifest.jsonl"
    state_path = args.output / "download_state.json"

    statuses = load_manifest(manifest_path)
    downloaded_total = count_downloaded(statuses)
    attempted_this_run: set[VideoKey] = set()

    with AppMaisClient(timeout=60.0) as client:
        schedule = build_day_schedule(client, args)

        signature = schedule_signature(args)
        start_index = load_next_day_index(state_path, signature)
        if start_index >= len(schedule) and downloaded_total < args.count:
            start_index = 0

        downloaded_by_day = count_downloaded_by_day(statuses)

        print(f"Schedule has {len(schedule)} hive/day pairs.")
        print(f"Already downloaded according to manifest: {downloaded_total}")

        for day_index in range(start_index, len(schedule)):
            if downloaded_total >= args.count:
                break

            hive, day = schedule[day_index]
            save_state(state_path, signature, day_index)
            print(f"\n[{day_index + 1}/{len(schedule)}] {hive} {day}")

            day_successes = downloaded_by_day.get((hive, day), 0)
            if day_successes >= args.per_day:
                save_state(state_path, signature, day_index + 1)
                continue

            try:
                times = request_with_retries(
                    f"list times for {hive} {day}",
                    args,
                    lambda hive=hive, day=day: client.list_times(hive, day),
                )
            except Exception as exc:
                print(f"  failed to list times: {exc}")
                save_state(state_path, signature, day_index + 1)
                continue

            for time in diverse_times(times, args.per_day, args.seed, hive, day):
                if downloaded_total >= args.count or day_successes >= args.per_day:
                    break

                key = (hive, day, time)
                status = statuses.get(key)
                if status in {"downloaded", "unavailable"} or key in attempted_this_run:
                    continue

                result = try_download_video(client, args, hive, day, time)
                attempted_this_run.add(key)
                append_manifest(manifest_path, result)
                statuses[key] = result["status"]

                if result["status"] == "downloaded":
                    downloaded_total += 1
                    day_successes += 1
                    downloaded_by_day[(hive, day)] = day_successes
                    print(f"  downloaded {time} ({downloaded_total}/{args.count})")
                else:
                    reason = result.get("reason", "unknown")
                    print(f"  {result['status']} {time}: {reason}")

            save_state(state_path, signature, day_index + 1)

    print(f"\nDone. Downloaded total: {downloaded_total}/{args.count}")


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Download a diverse, resumable AppMAIS video sample."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for videos, state, and manifest.",
    )
    parser.add_argument(
        "--count",
        type=positive_int,
        required=True,
        help="Target number of successfully downloaded videos.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic hive/day/time ordering.",
    )
    parser.add_argument(
        "--per-day",
        type=positive_int,
        default=1,
        help="Maximum successful downloads per hive/day pair.",
    )
    parser.add_argument(
        "--hives",
        nargs="+",
        default=None,
        help="Optional hive names. Defaults to all hives.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_optional_date,
        default=None,
        help="Optional first day, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_optional_date,
        default=None,
        help="Optional last day, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--delay",
        type=non_negative_float,
        default=2.0,
        help="Seconds to wait before each AppMAIS request.",
    )
    parser.add_argument(
        "--max-retries",
        type=non_negative_int,
        default=5,
        help="Retries for HTTP 429 rate-limit responses.",
    )
    return cast(Args, parser.parse_args())


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or positive")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or positive")
    return parsed


def parse_optional_date(value: str) -> dt.date:
    return parse_date(value)


def build_day_schedule(client: AppMaisClient, args: Args) -> list[tuple[str, str]]:
    hives = (
        args.hives
        if args.hives is not None
        else request_with_retries("list hives", args, client.list_hives)
    )
    schedule: list[tuple[str, str]] = []

    for hive in hives:
        try:
            days = request_with_retries(
                f"list days for {hive}", args, lambda hive=hive: client.list_days(hive)
            )
        except Exception as exc:
            print(f"Skipping hive {hive}: failed to list days: {exc}")
            continue

        for day in days:
            parsed_day = parse_date(day)
            if args.start_date is not None and parsed_day < args.start_date:
                continue
            if args.end_date is not None and parsed_day > args.end_date:
                continue
            schedule.append((hive, day))

    rng = random.Random(args.seed)
    rng.shuffle(schedule)
    return schedule


def schedule_signature(args: Args) -> dict[str, Any]:
    return {
        "seed": args.seed,
        "per_day": args.per_day,
        "hives": args.hives,
        "start_date": args.start_date.isoformat() if args.start_date else None,
        "end_date": args.end_date.isoformat() if args.end_date else None,
    }


def request_with_retries[T](label: str, args: Args, request: Callable[[], T]) -> T:
    attempts = 0
    while True:
        if args.delay > 0:
            time.sleep(args.delay)

        try:
            return request()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 429 or attempts >= args.max_retries:
                raise

            attempts += 1
            wait_seconds = retry_after_seconds(exc.response) or max(
                args.delay, min(60.0, 2.0**attempts)
            )
            print(
                f"  rate limited during {label}; waiting {wait_seconds:.1f}s "
                f"(retry {attempts}/{args.max_retries})"
            )
            time.sleep(wait_seconds)


def retry_after_seconds(response: httpx.Response) -> float | None:
    retry_after = response.headers.get("Retry-After")
    if retry_after is None:
        return None

    try:
        return max(0.0, float(retry_after))
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(retry_after)
    except (TypeError, ValueError):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=dt.UTC)
    now = dt.datetime.now(dt.UTC)
    return max(0.0, (retry_at - now).total_seconds())


def diverse_times(
    times: Iterable[str], per_day: int, seed: int, hive: str, day: str
) -> list[str]:
    sorted_times = sorted(set(times))
    if len(sorted_times) <= 1 or per_day <= 1:
        rng = random.Random(f"{seed}:{hive}:{day}:times")
        rng.shuffle(sorted_times)
        return sorted_times

    bucket_count = min(per_day, len(sorted_times))
    buckets: list[list[str]] = []
    rng = random.Random(f"{seed}:{hive}:{day}:times")

    for bucket_index in range(bucket_count):
        start = bucket_index * len(sorted_times) // bucket_count
        end = (bucket_index + 1) * len(sorted_times) // bucket_count
        bucket = sorted_times[start:end]
        rng.shuffle(bucket)
        buckets.append(bucket)

    ordered: list[str] = []
    max_bucket_size = max(len(bucket) for bucket in buckets)
    for item_index in range(max_bucket_size):
        for bucket in buckets:
            if item_index < len(bucket):
                ordered.append(bucket[item_index])
    return ordered


def try_download_video(
    client: AppMaisClient, args: Args, hive: str, day: str, time: str
) -> ManifestRecord:
    try:
        video = request_with_retries(
            f"resolve video {hive} {day} {time}",
            args,
            lambda hive=hive, day=day, time=time: client.get_video(hive, day, time),
        )
    except Exception as exc:
        status = classify_exception(exc)
        return make_record(hive, day, time, status, reason=str(exc))

    destination = args.output / video.filename
    if destination.exists():
        return make_record(hive, day, time, "downloaded", path=destination)

    part_destination = destination.with_suffix(destination.suffix + ".part")
    if part_destination.exists():
        part_destination.unlink()

    try:
        request_with_retries(
            f"download video {hive} {day} {time}",
            args,
            lambda video=video, part_destination=part_destination: download_to_part(
                client, video, part_destination
            ),
        )
        part_destination.replace(destination)
    except Exception as exc:
        if part_destination.exists():
            part_destination.unlink()
        status = classify_exception(exc)
        return make_record(hive, day, time, status, reason=str(exc))

    return make_record(hive, day, time, "downloaded", path=destination)


def make_record(
    hive: str,
    day: str,
    time: str,
    status: Status,
    *,
    path: Path | None = None,
    reason: str | None = None,
) -> ManifestRecord:
    record: ManifestRecord = {
        "status": status,
        "hive": hive,
        "day": day,
        "time": time,
    }
    if path is not None:
        record["path"] = str(path)
    if reason is not None:
        record["reason"] = reason
    return record


def download_to_part(
    client: AppMaisClient, video: AppMaisVideo, part_destination: Path
) -> None:
    part_destination.parent.mkdir(parents=True, exist_ok=True)
    client.download(video, part_destination)


def classify_exception(exc: Exception) -> Status:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        if status_code in UNAVAILABLE_STATUS_CODES:
            return "unavailable"
        if status_code in TRANSIENT_STATUS_CODES:
            return "failed"
    if isinstance(exc, httpx.TimeoutException | httpx.NetworkError):
        return "failed"
    if isinstance(exc, ValueError):
        return "unavailable"
    return "failed"


def load_manifest(path: Path) -> dict[VideoKey, Status]:
    statuses: dict[VideoKey, Status] = {}
    if not path.exists():
        return statuses

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Ignoring bad manifest line {line_number}: {exc}")
                continue

            if not isinstance(record, dict):
                continue
            key = manifest_key(record)
            status = record.get("status")
            if key is None or status not in {"downloaded", "unavailable", "failed"}:
                continue
            statuses[key] = cast(Status, status)

    return statuses


def manifest_key(record: dict[str, Any]) -> VideoKey | None:
    hive = record.get("hive")
    day = record.get("day")
    time = record.get("time")
    if not isinstance(hive, str) or not isinstance(day, str):
        return None
    if not isinstance(time, str):
        return None
    return hive, day, time


def count_downloaded(statuses: dict[VideoKey, Status]) -> int:
    return sum(status == "downloaded" for status in statuses.values())


def count_downloaded_by_day(
    statuses: dict[VideoKey, Status],
) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for (hive, day, _time), status in statuses.items():
        if status == "downloaded":
            counts[(hive, day)] = counts.get((hive, day), 0) + 1
    return counts


def append_manifest(path: Path, record: ManifestRecord) -> None:
    with path.open("a", encoding="utf-8") as file:
        json.dump(record, file, sort_keys=True)
        file.write("\n")


def load_next_day_index(path: Path, signature: dict[str, Any]) -> int:
    if not path.exists():
        return 0

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0

    if not isinstance(data, dict):
        return 0
    if data.get("signature") != signature:
        return 0

    next_day_index = data.get("next_day_index")
    if not isinstance(next_day_index, int) or next_day_index < 0:
        return 0
    return next_day_index


def save_state(path: Path, signature: dict[str, Any], next_day_index: int) -> None:
    state: State = {"signature": signature, "next_day_index": next_day_index}
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary_path.replace(path)


if __name__ == "__main__":
    main()
