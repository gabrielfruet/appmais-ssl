"""Audit the merged seed-detector dataset.

Reads ``<root>/merged/{train,val,test}/_annotations.coco.json`` and prints:

- per-split image and annotation counts
- per-class annotation counts (overall and per split)
- per-source-dataset image and annotation counts (slug = filename prefix)
- per-source-dataset per-class annotation counts

With ``--contact-sheet`` it also writes a single ``outputs/seed_detector_audit.jpg``
contact sheet with up to ``--samples`` random training images and their boxes
overlaid (uses OpenCV). Run after ``seed_detector_merge.py``.

Usage:
    uv run python scripts/seed_detector_audit.py
    uv run python scripts/seed_detector_audit.py --root /media/data/seed_detector
    uv run python scripts/seed_detector_audit.py --contact-sheet --samples 32
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from seed_detector_class_mapping import TARGETS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/media/data/seed_detector")
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "seed_detector_audit.jpg"

# BGR colours per bucket for the contact-sheet overlay.
_BUCKET_COLOURS: dict[str, tuple[int, int, int]] = {
    "drone": (255, 128, 0),
    "worker": (0, 200, 0),
    "enemy": (0, 0, 255),
}


def _split_roots(root: Path) -> dict[str, Path]:
    return {split: root / "merged" / split for split in ("train", "val", "test")}


def _slug_of(file_name: str) -> str:
    return file_name.split("___", 1)[0] if "___" in file_name else "<unknown>"


def _summarise(root: Path) -> dict:
    by_split_images: dict[str, int] = {}
    by_split_anns: dict[str, int] = {}
    class_counts: Counter[str] = Counter()
    class_counts_per_split: dict[str, Counter[str]] = defaultdict(Counter)
    slug_images: Counter[str] = Counter()
    slug_anns: dict[str, int] = {}
    slug_class_anns: dict[str, Counter[str]] = defaultdict(Counter)
    images_by_id: dict[tuple[str, int], dict] = {}
    anns_by_id: dict[tuple[str, int], dict] = {}
    file_to_split: dict[str, str] = {}

    for split, split_root in _split_roots(root).items():
        jp = split_root / "_annotations.coco.json"
        if not jp.exists():
            continue
        coco = json.loads(jp.read_text())
        images = coco["images"]
        anns = coco["annotations"]
        by_split_images[split] = len(images)
        by_split_anns[split] = len(anns)
        for img in images:
            images_by_id[(split, int(img["id"]))] = img
            file_to_split[img["file_name"]] = split
        for ann in anns:
            anns_by_id[(split, int(ann["id"]))] = ann
            cat_id = int(ann["category_id"])
            name = TARGETS[cat_id]
            class_counts[name] += 1
            class_counts_per_split[split][name] += 1
            img = images_by_id[(split, int(ann["image_id"]))]
            slug = _slug_of(img["file_name"])
            slug_class_anns[slug][name] += 1
        for img in images:
            slug = _slug_of(img["file_name"])
            slug_images[slug] += 1

    slug_anns = {slug: sum(c.values()) for slug, c in slug_class_anns.items()}

    return {
        "by_split_images": by_split_images,
        "by_split_anns": by_split_anns,
        "class_counts": class_counts,
        "class_counts_per_split": class_counts_per_split,
        "slug_images": slug_images,
        "slug_anns": slug_anns,
        "slug_class_anns": slug_class_anns,
        "images_by_id": images_by_id,
        "anns_by_id": anns_by_id,
        "file_to_split": file_to_split,
    }


def _print_report(summary: dict) -> None:
    print("\n=== merged seed-detector dataset ===")
    print("images per split:", dict(summary["by_split_images"]))
    print("anns   per split:", dict(summary["by_split_anns"]))
    total = sum(summary["class_counts"].values())
    print(f"\nclass totals (anns={total}):")
    for name in TARGETS:
        n = summary["class_counts"].get(name, 0)
        pct = (100.0 * n / total) if total else 0.0
        print(f"  {name:<6} {n:>6}  {pct:5.1f}%")
    print("\nclass counts per split:")
    for split in ("train", "val", "test"):
        if split in summary["class_counts_per_split"]:
            row = summary["class_counts_per_split"][split]
            print(
                f"  {split:<5}: " + ", ".join(f"{n}={row.get(n, 0)}" for n in TARGETS)
            )

    print("\nper source dataset (slug):")
    slugs = sorted(summary["slug_images"])
    header = f"  {'slug':<48} {'images':>7} {'anns':>7}  " + "  ".join(
        f"{n:>7}" for n in TARGETS
    )
    print(header)
    for slug in slugs:
        cls_counts = summary["slug_class_anns"][slug]
        row = (
            f"  {slug:<48} {summary['slug_images'][slug]:>7} "
            f"{summary['slug_anns'][slug]:>7}  "
        )
        row += "  ".join(f"{cls_counts.get(n, 0):>7}" for n in TARGETS)
        print(row)


def _draw_boxes(
    img_bgr: np.ndarray, boxes: list[tuple[list[float], str]]
) -> np.ndarray:
    for (x, y, w, h), name in boxes:
        colour = _BUCKET_COLOURS.get(name, (255, 255, 255))
        x0, y0 = int(round(x)), int(round(y))
        x1, y1 = int(round(x + w)), int(round(y + h))
        cv2.rectangle(img_bgr, (x0, y0), (x1, y1), colour, 2)
        label = f"{name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x0, max(0, y0 - th - 4)), (x0 + tw, y0), colour, -1)
        cv2.putText(
            img_bgr,
            label,
            (x0, y0 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return img_bgr


def _contact_sheet(root: Path, samples: int, output: Path, seed: int) -> None:
    train_root = root / "merged" / "train"
    jp = train_root / "_annotations.coco.json"
    if not jp.exists():
        print(f"[contact-sheet] missing {jp}; skipping", file=sys.stderr)
        return
    coco = json.loads(jp.read_text())
    images = coco["images"]
    if not images:
        print("[contact-sheet] train split is empty; skipping", file=sys.stderr)
        return
    rng = random.Random(seed)
    rng.shuffle(images)
    chosen = images[:samples]

    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)

    cell_w, cell_h = 320, 320
    cols = 4
    rows = (len(chosen) + cols - 1) // cols
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 32, dtype=np.uint8)
    for idx, img in enumerate(chosen):
        path = train_root / img["file_name"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = np.full((cell_h, cell_w, 3), 64, dtype=np.uint8)
        bgr = cv2.resize(bgr, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        boxes = []
        for ann in anns_by_image.get(int(img["id"]), []):
            name = TARGETS[int(ann["category_id"])]
            boxes.append((ann["bbox"], name))
        bgr = _draw_boxes(bgr, boxes)
        # Footer text = source slug (handy to spot dataset-specific quirks).
        slug = _slug_of(img["file_name"])
        cv2.putText(
            bgr,
            slug,
            (4, cell_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        r, c = divmod(idx, cols)
        sheet[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = bgr

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)
    print(f"[contact-sheet] wrote {output} ({len(chosen)} samples)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--contact-sheet", action="store_true")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = _summarise(args.root)
    if not summary["by_split_images"]:
        sys.exit(f"no merged dataset found at {args.root / 'merged'}")
    _print_report(summary)

    if args.contact_sheet:
        _contact_sheet(args.root, args.samples, args.output, args.seed)


if __name__ == "__main__":
    main()
