"""Download every dataset listed in ``seed_detector_datasets.yaml`` as COCO.

For each entry we ask the Roboflow SDK for the project's named version and
unpack the zip into ``<root>/raw/<workspace>__<project>/<version>/`` with the
Roboflow-native layout (``train/``, ``valid/``, ``test/`` + per-split
``_annotations.coco.json``).

The download is resumable: if the per-dataset directory already exists and
contains a ``_annotations.coco.json`` in each split, the entry is skipped
unless ``--overwrite`` is passed.

Usage:
    uv run python scripts/seed_detector_download.py
    uv run python scripts/seed_detector_download.py --root /media/data/seed_detector
    uv run python scripts/seed_detector_download.py --overwrite
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from roboflow import Roboflow

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "scripts" / "seed_detector_datasets.yaml"
DEFAULT_ROOT = Path("/media/data/seed_detector")


def _slug(ws: str, proj: str) -> str:
    return f"{ws}__{proj}"


def _has_complete_split(root: Path) -> bool:
    """Return True if every Roboflow split already has its COCO JSON unpacked."""
    for split in ("train", "valid", "test"):
        sub = root / split
        if sub.is_dir() and not (sub / "_annotations.coco.json").exists():
            return False
    return (root / "train" / "_annotations.coco.json").exists()


def _load_api_key() -> str:
    load_dotenv(REPO_ROOT / ".env")
    import os

    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        sys.exit("ROBOFLOW_API_KEY missing from .env")
    return key


def _load_registry(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, list):
        sys.exit(f"{path}: registry must be a YAML list")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_YAML)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="data root (raw/ is created here)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="re-download even when files exist"
    )
    args = parser.parse_args()

    raw_root = args.root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    api_key = _load_api_key()
    rf = Roboflow(api_key=api_key)
    entries = _load_registry(args.registry)

    for entry in entries:
        ws = entry["workspace"]
        proj = entry["project"]
        version = entry.get("version")
        slug = _slug(ws, proj)
        target = raw_root / slug

        if target.exists() and _has_complete_split(target) and not args.overwrite:
            print(f"[skip] {slug}: already present at {target}")
            continue

        if target.exists() and args.overwrite:
            print(f"[clean] {slug}: removing {target}")
            shutil.rmtree(target)

        try:
            project = rf.workspace(ws).project(proj)
            if version is None or version == "latest":
                v = project.versions()[0]
            else:
                if str(version).isdigit():
                    v = project.version(int(version))
                else:
                    v = project.version(version)  # type: ignore[arg-type]
            print(f"[get ] {slug}: downloading v{v.version} ({v.name}) as coco")
            ds = v.download("coco", location=str(target))
            print(f"[done] {slug}: {ds.location}")
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {slug}: {exc}", file=sys.stderr)
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            continue


if __name__ == "__main__":
    main()
