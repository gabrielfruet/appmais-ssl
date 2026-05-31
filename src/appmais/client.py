from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import httpx

APPMAIS_URL = "https://appmais.cs.appstate.edu"
VIDEO_BASE_URL = f"{APPMAIS_URL}/videos/"

_VIDEO_NAME_RE = re.compile(
    r"^(?P<hive>[^@]+)@"
    r"(?P<date>\d{4}-\d{2}-\d{2})@"
    r"(?P<time>\d{2}-\d{2}-\d{2})"
    r"(?:\.[^.]+)?$"
)


@dataclass(frozen=True)
class AppMaisVideoName:
    hive: str
    date: dt.date
    time: dt.time

    @property
    def filename(self) -> str:
        return build_video_filename(self.hive, self.date, self.time)


@dataclass(frozen=True)
class AppMaisVideo:
    hive: str
    date: dt.date
    time: dt.time
    source_path: str
    url: str

    @property
    def filename(self) -> str:
        return build_video_filename(self.hive, self.date, self.time)


def parse_video_name(name: str | Path) -> AppMaisVideoName:
    """Parse names like AppMAIS14L@2024-04-08@14-05-00.mp4."""
    match = _VIDEO_NAME_RE.match(Path(name).name)
    if match is None:
        raise ValueError(f"Invalid AppMAIS video name: {name}")

    return AppMaisVideoName(
        hive=match.group("hive"),
        date=parse_date(match.group("date")),
        time=parse_time(match.group("time").replace("-", ":")),
    )


def build_video_filename(hive: str, date: str | dt.date, time: str | dt.time) -> str:
    parsed_date = parse_date(date)
    parsed_time = parse_time(time)
    time_part = parsed_time.strftime("%H-%M-%S")
    return f"{hive}@{parsed_date.isoformat()}@{time_part}.mp4"


def parse_date(value: str | dt.date) -> dt.date:
    """Accept YYYY-MM-DD or the AppMAIS UI format MM/DD/YYYY."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value

    text = value.strip()
    for date_format in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(text, date_format).date()
        except ValueError:
            pass
    raise ValueError(f"Invalid date: {value!r}. Use YYYY-MM-DD or MM/DD/YYYY.")


def parse_time(value: str | dt.time) -> dt.time:
    """Accept HH:MM:SS or the AppMAIS UI format H:MM:SS am/pm."""
    if isinstance(value, dt.time):
        return value.replace(microsecond=0)

    text = value.strip()
    for time_format in ("%H:%M:%S", "%I:%M:%S %p"):
        try:
            return dt.datetime.strptime(text.upper(), time_format).time()
        except ValueError:
            pass
    raise ValueError(f"Invalid time: {value!r}. Use HH:MM:SS or H:MM:SS am/pm.")


class AppMaisClient:
    def __init__(
        self,
        base_url: str = APPMAIS_URL,
        video_base_url: str = VIDEO_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.video_base_url = video_base_url.rstrip("/") + "/"
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> AppMaisClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def list_hives(self) -> list[str]:
        return self._get_string_list("/api/hives")

    def list_days(self, hive: str) -> list[str]:
        """Return available days as YYYY-MM-DD strings."""
        days = self._get_string_list(f"/api/videofiles/{_url_part(hive)}/days/")
        return [parse_date(day).isoformat() for day in days]

    def list_times(self, hive: str, day: str | dt.date) -> list[str]:
        """Return available times as HH:MM:SS strings."""
        day_part = parse_date(day).isoformat()
        times = self._get_string_list(
            f"/api/videofiles/{_url_part(hive)}/times/{_url_part(day_part)}"
        )
        return [parse_time(time).strftime("%H:%M:%S") for time in times]

    def get_video(
        self,
        hive: str,
        day: str | dt.date,
        time: str | dt.time,
    ) -> AppMaisVideo:
        parsed_date = parse_date(day)
        parsed_time = parse_time(time)
        data = self._get_json(
            f"/api/videofiles/{_url_part(hive)}/"
            f"{_url_part(parsed_date.isoformat())}/"
            f"{_url_part(parsed_time.strftime('%H:%M:%S'))}/filepath"
        )

        if not isinstance(data, dict) or "FilePath" not in data:
            raise ValueError(f"Unexpected filepath response: {data!r}")

        source_path = str(data["FilePath"])
        return AppMaisVideo(
            hive=hive,
            date=parsed_date,
            time=parsed_time,
            source_path=source_path,
            url=self.video_url_from_source_path(source_path),
        )

    def iter_videos(
        self,
        hive: str,
        day: str | dt.date,
    ) -> Iterator[AppMaisVideo]:
        for time in self.list_times(hive, day):
            yield self.get_video(hive, day, time)

    def download(self, video_or_url: AppMaisVideo | str, output: str | Path) -> Path:
        url = (
            video_or_url.url if isinstance(video_or_url, AppMaisVideo) else video_or_url
        )
        filename = (
            video_or_url.filename
            if isinstance(video_or_url, AppMaisVideo)
            else Path(httpx.URL(url).path).name
        )
        destination = _download_destination(output, filename)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with self._client.stream("GET", url) as response:
            response.raise_for_status()
            with destination.open("wb") as file:
                for chunk in response.iter_bytes():
                    file.write(chunk)

        return destination

    def video_url_from_source_path(self, source_path: str) -> str:
        marker = "appmais/"
        if marker not in source_path:
            raise ValueError(f"Unexpected AppMAIS source path: {source_path}")

        relative_path = source_path.split(marker, 1)[1]
        if relative_path.endswith(".h264"):
            relative_path = relative_path.removesuffix(".h264") + ".mp4"

        return self.video_base_url + relative_path.lstrip("/")

    def _get_string_list(self, path: str) -> list[str]:
        data = self._get_json(path)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list from {path}, got {data!r}")
        return [str(item) for item in data]

    def _get_json(self, path: str) -> Any:
        response = self._client.get(path)
        response.raise_for_status()
        return cast(Any, response.json())


def _url_part(value: str) -> str:
    return quote(value, safe="")


def _download_destination(output: str | Path, filename: str) -> Path:
    path = Path(output)
    if path.exists() and path.is_dir():
        return path / filename
    if path.suffix:
        return path
    return path / filename
