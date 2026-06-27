"""Merge every downloaded Roboflow dataset into a single COCO dataset.

For each `(workspace, project)` we have in ``seed_detector_datasets.yaml`` we
read the train / valid / test ``_annotations.coco.json``, remap every category
through ``seed_detector_class_mapping.map_class`` and emit one merged
``_annotations.coco.json`` per split under ``<root>/merged/{train,val,test}/``
together with a flat copy of every image (prefixed with the source slug so
filenames stay unique).

Categories are fixed at the canonical 4 classes with contiguous ids:

    0 drone
    1 worker
    2 pollen
    3 enemy

Anything that doesn't map to a target bucket is dropped (with a warning per
class so the audit script can flag it). Image and annotation ids are offset
per source so they stay unique across the merged file. The Roboflow-native
``valid`` split is renamed to ``val`` to match the RF-DETR / COCO convention.

Usage:
    uv run python scripts/seed_detector_merge.py
    uv run python scripts/seed_detector_merge.py --root /media/data/seed_detector --overwrite
"""  # noqa: E501

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from seed_detector_class_mapping import TARGET_TO_ID, TARGETS, map_class

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "scripts" / "seed_detector_datasets.yaml"
DEFAULT_ROOT = Path("/media/data/seed_detector")

SPLIT_SRC_TO_DST = {"train": "train", "valid": "val", "test": "test"}


@dataclass
class DatasetStats:
    slug: str
    source_cats: list[dict[str, Any]]
    bucket_for_source_id: dict[int, str]
    dropped_cats: Counter = field(default_factory=Counter)
    images_per_split: Counter = field(default_factory=Counter)
    anns_per_split: Counter = field(default_factory=Counter)


def _slug(ws: str, proj: str) -> str:
    return f"{ws}__{proj}"


def _load_registry(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, list):
        sys.exit(f"{path}: registry must be a YAML list")
    return data


def _categories_for(coco: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"id": int(c["id"]), "name": str(c["name"])} for c in coco.get("categories", [])
    ]


def _build_cat_map(cats: list[dict[str, Any]]) -> tuple[dict[int, str], list[str]]:
    """Return source_cat_id -> target_bucket and the dropped source class names."""
    keep: dict[int, str] = {}
    dropped: list[str] = []
    for cat in cats:
        bucket = map_class(cat["name"])
        if bucket is None:
            dropped.append(cat["name"])
        else:
            keep[int(cat["id"])] = bucket
    return keep, sorted(set(dropped))


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def _process_split(
    src_split_dir: Path,
    dst_split_dir: Path,
    slug: str,
    cat_map: dict[int, str],
    dropped_class_names: set[str],
    image_id_offset: int,
    ann_id_offset: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str], int, int]:
    """Remap one split and copy the kept images.

    Returns (images, annotations, dropped_counts, next_image_id, next_ann_id).
    """
    json_path = src_split_dir / "_annotations.coco.json"
    coco = json.loads(json_path.read_text())

    src_images = coco.get("images", [])
    src_anns = coco.get("annotations", [])
    if not src_images:
        return [], [], Counter(), image_id_offset, ann_id_offset

    name_by_id = {int(c["id"]): c["name"] for c in coco.get("categories", [])}

    # Bucket annotations by image_id for kept categories; tally dropped ones.
    kept_by_image: dict[int, list[dict[str, Any]]] = {}
    dropped: Counter[str] = Counter()
    for ann in src_anns:
        src_cat = int(ann["category_id"])
        if src_cat in cat_map:
            kept_by_image.setdefault(int(ann["image_id"]), []).append(ann)
        else:
            dropped[name_by_id.get(src_cat, str(src_cat))] += 1

    out_images: list[dict[str, Any]] = []
    out_anns: list[dict[str, Any]] = []
    next_image_id = image_id_offset
    next_ann_id = ann_id_offset

    for src_img in src_images:
        sid = int(src_img["id"])
        bucket = kept_by_image.get(sid)
        if not bucket:
            continue
        new_image_id = next_image_id
        next_image_id += 1
        src_name = src_img["file_name"]
        new_name = f"{slug}___{src_name}"
        out_images.append(
            {
                "id": new_image_id,
                "file_name": new_name,
                "width": int(src_img["width"]),
                "height": int(src_img["height"]),
                "license": int(src_img.get("license", 0)),
                "date_captured": src_img.get("date_captured", ""),
            }
        )
        _copy_image(src_split_dir / src_name, dst_split_dir / new_name)

        for src_ann in bucket:
            new_ann_id = next_ann_id
            next_ann_id += 1
            out_anns.append(
                {
                    "id": new_ann_id,
                    "image_id": new_image_id,
                    "category_id": TARGET_TO_ID[cat_map[int(src_ann["category_id"])]],
                    "bbox": [float(v) for v in src_ann["bbox"]],
                    "area": float(src_ann.get("area", 0.0)),
                    "segmentation": src_ann.get("segmentation", []),
                    "iscrowd": int(src_ann.get("iscrowd", 0)),
                }
            )

    return out_images, out_anns, dropped, next_image_id, next_ann_id


def _write_split(
    dst_dir: Path, images: list[dict[str, Any]], anns: list[dict[str, Any]]
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    coco = {
        "info": {"description": "Merged seed detector dataset (RF-DETR ready)"},
        "licenses": [],
        "categories": [
            {"id": TARGET_TO_ID[name], "name": name, "supercategory": "seed-detector"}
            for name in TARGETS
        ],
        "images": images,
        "annotations": anns,
    }
    (dst_dir / "_annotations.coco.json").write_text(json.dumps(coco, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--overwrite", action="store_true", help="wipe merged/ before merging"
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    merged_root = args.root / "merged"
    if merged_root.exists() and args.overwrite:
        print(f"[clean] {merged_root}")
        shutil.rmtree(merged_root)
    merged_root.mkdir(parents=True, exist_ok=True)
    for split_dst in SPLIT_SRC_TO_DST.values():
        (merged_root / split_dst).mkdir(exist_ok=True)

    entries = _load_registry(args.registry)
    stats: list[DatasetStats] = []

    image_id_counter = 1
    ann_id_counter = 1
    split_images: dict[str, list[dict[str, Any]]] = {
        d: [] for d in SPLIT_SRC_TO_DST.values()
    }
    split_anns: dict[str, list[dict[str, Any]]] = {
        d: [] for d in SPLIT_SRC_TO_DST.values()
    }

    for entry in entries:
        ws, proj, version = entry["workspace"], entry["project"], entry.get("version")
        slug = _slug(ws, proj)
        raw_root = args.root / "raw" / slug
        if not raw_root.exists():
            print(f"[skip] {slug}: no raw data at {raw_root}", file=sys.stderr)
            continue

        # Accept either raw/<slug>/<version>/{train,...} or raw/<slug>/{train,...}.
        candidate = raw_root / str(version) if version else raw_root
        if not (candidate / "train").exists():
            candidate = raw_root

        # Collect categories from any split that has one.
        first_coco: dict[str, Any] | None = None
        for split_src in SPLIT_SRC_TO_DST:
            jp = candidate / split_src / "_annotations.coco.json"
            if jp.exists():
                first_coco = json.loads(jp.read_text())
                break
        if first_coco is None:
            print(
                f"[skip] {slug}: no _annotations.coco.json under {candidate}",
                file=sys.stderr,
            )
            continue

        cats = _categories_for(first_coco)
        cat_map, dropped_proto = _build_cat_map(cats)
        dropped_names = set(dropped_proto)
        ds_stats = DatasetStats(
            slug=slug, source_cats=cats, bucket_for_source_id=cat_map
        )
        # Track every source class so the summary can show counts even when a
        # class is unmapped *and* has zero annotations (otherwise it would
        # vanish from the output entirely).
        for cat in cats:
            cid = int(cat["id"])
            if cid not in cat_map:
                ds_stats.dropped_cats.setdefault(cat["name"], 0)

        for split_src, split_dst in SPLIT_SRC_TO_DST.items():
            src_split = candidate / split_src
            if not (src_split / "_annotations.coco.json").exists():
                continue
            images, anns, dropped, image_id_counter, ann_id_counter = _process_split(
                src_split_dir=src_split,
                dst_split_dir=merged_root / split_dst,
                slug=slug,
                cat_map=cat_map,
                dropped_class_names=dropped_names,
                image_id_offset=image_id_counter,
                ann_id_offset=ann_id_counter,
            )
            split_images[split_dst].extend(images)
            split_anns[split_dst].extend(anns)
            ds_stats.images_per_split[split_dst] = len(images)
            ds_stats.anns_per_split[split_dst] = len(anns)
            ds_stats.dropped_cats.update(dropped)

        stats.append(ds_stats)
        # Dedupe by (name, bucket) so datasets that have two category ids both
        # named "bee" don't show the mapping twice.
        seen_pairs: set[tuple[str, str]] = set()
        kept: list[str] = []
        for c in cats:
            cid = int(c["id"])
            if cid not in cat_map:
                continue
            pair = (c["name"], cat_map[cid])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            kept.append(f"{pair[0]}->{pair[1]}")
        dropped_str = (
            ", ".join(f"{n}={k}" for n, k in ds_stats.dropped_cats.items()) or "none"
        )
        print(
            f"[merge] {slug}: kept=[{', '.join(kept)}] dropped=[{dropped_str}] "
            f"images={dict(ds_stats.images_per_split)} "
            f"anns={dict(ds_stats.anns_per_split)}"
        )

    for split_dst in SPLIT_SRC_TO_DST.values():
        _write_split(
            merged_root / split_dst, split_images[split_dst], split_anns[split_dst]
        )

    print(
        f"[done] merged dataset at {merged_root}  "
        f"(next image_id={image_id_counter - 1}, next ann_id={ann_id_counter - 1})"
    )


if __name__ == "__main__":
    main()
