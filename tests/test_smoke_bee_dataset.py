"""Tests for scripts.smoke_bee_dataset helpers."""

from engine.dataset import BeeCropDataset
from scripts.smoke_bee_dataset import _run_preflight
from tests.test_dataset import _write_sample


def test_run_preflight_accepts_tiny_valid_dataset(tmp_path: object) -> None:
    """`_run_preflight` appends no failures for a valid tiny dataset."""
    _write_sample(tmp_path, "vid_a", "frame_with_bee", fg=True)
    dataset = BeeCropDataset(str(tmp_path), crop_size=64)
    failures: list[str] = []

    _run_preflight(dataset, num_samples=1, failures=failures)

    assert failures == []
