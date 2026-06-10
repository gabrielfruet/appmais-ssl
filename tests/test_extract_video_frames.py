"""Tests for scripts.extract_video_frames helpers."""

import pathlib

from scripts.extract_video_frames import parse_videos_file


def test_parse_videos_file_ignores_comments_and_blanks(tmp_path: object) -> None:
    root = pathlib.Path(str(tmp_path))
    first = root / "a.mp4"
    second = root / "b.mov"
    first.write_bytes(b"not a real video")
    second.write_bytes(b"not a real video")
    videos_file = root / "videos.txt"
    videos_file.write_text(
        f"# curated videos\n\n{first}\n  {second}  \n",
        encoding="utf-8",
    )

    assert parse_videos_file(videos_file) == [first.resolve(), second.resolve()]
