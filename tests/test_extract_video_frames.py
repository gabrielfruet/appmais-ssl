"""Tests for scripts.extract_video_frames helpers."""

import pathlib

import cv2
import numpy as np

from scripts.extract_video_frames import _feed_mog2_pair, parse_videos_file


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


def test_feed_mog2_pair_updates_both_subtractors() -> None:
    """`_feed_mog2_pair` updates both MOG2s and returns the mask subtractor's mask."""
    mask_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=10,
        varThreshold=2.0,
        detectShadows=True,
    )
    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=10,
        varThreshold=2.0,
        detectShadows=True,
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    raw_mask = _feed_mog2_pair(
        mask_subtractor=mask_subtractor,
        background_subtractor=background_subtractor,
        frame=frame,
        downsample_width=64,
    )

    # The returned mask comes from the (blurred) mask subtractor.
    assert raw_mask is not None
    assert raw_mask.shape == (64, 64)
    # Both subtractor models have been updated.
    assert mask_subtractor.getBackgroundImage() is not None
    assert background_subtractor.getBackgroundImage() is not None


def test_feed_mog2_pair_background_uses_unblurred_input() -> None:
    """The background subtractor sees a sharper frame than the mask subtractor.

    Feed a frame with a sharp horizontal edge to both MOG2s. The mask
    subtractor sees the Gaussian-blurred version, so its background is
    also blurry (the edge diffuses). The background subtractor sees the
    unblurred frame, so its background keeps the sharp edge (lower blur
    score when re-blurred at evaluation time).
    """
    mask_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=20,
        varThreshold=2.0,
        detectShadows=True,
    )
    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=20,
        varThreshold=2.0,
        detectShadows=True,
    )
    height, width = 64, 64
    frame = np.full((height, width, 3), 0, dtype=np.uint8)
    frame[:, 32:] = 200  # sharp vertical edge at x=32

    # Feed many frames so both models converge.
    for _ in range(20):
        _feed_mog2_pair(
            mask_subtractor=mask_subtractor,
            background_subtractor=background_subtractor,
            frame=frame,
            downsample_width=width,
        )

    mask_bg = mask_subtractor.getBackgroundImage()
    sharp_bg = background_subtractor.getBackgroundImage()
    assert mask_bg is not None and sharp_bg is not None

    # Re-blur both at evaluation time; the unblurred-trained background
    # should blur less (its edges were sharper to begin with).
    kernel = np.ones((5, 5), np.float32) / 25.0
    mask_residual = float(np.mean(np.abs(mask_bg - cv2.filter2D(mask_bg, -1, kernel))))
    sharp_residual = float(
        np.mean(np.abs(sharp_bg - cv2.filter2D(sharp_bg, -1, kernel)))
    )
    assert sharp_residual < mask_residual, (
        f"Expected unblurred-trained background to be sharper than the "
        f"blurred-trained one; got sharp_residual={sharp_residual:.3f} >= "
        f"mask_residual={mask_residual:.3f}"
    )
