# tcc

## AppMAIS archive downloads

Small Python helpers for listing and downloading videos from the AppMAIS archive.

```python
from appmais import AppMaisClient, parse_video_name

with AppMaisClient() as client:
    hives = client.list_hives()
    days = client.list_days("AppMAIS14L")
    times = client.list_times("AppMAIS14L", "2024-04-08")

    video = client.get_video("AppMAIS14L", "2024-04-08", "14:05:00")
    path = client.download(video, "data/videos/")

print(path)
print(parse_video_name("AppMAIS14L@2024-04-08@14-05-00.mp4"))
```

Useful API:

- `list_hives() -> list[str]`
- `list_days(hive) -> list[str]` returns `YYYY-MM-DD` dates
- `list_times(hive, day) -> list[str]` returns `HH:MM:SS` times
- `get_video(hive, day, time) -> AppMaisVideo`
- `iter_videos(hive, day)` lazily resolves videos for one day
- `download(video_or_url, output) -> Path`

Dates can be `YYYY-MM-DD` or `MM/DD/YYYY`. Times can be `HH:MM:SS` or `H:MM:SS am/pm`.
