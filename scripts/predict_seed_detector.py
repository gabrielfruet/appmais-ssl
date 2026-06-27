"""Run a trained RF-DETR seed-detector checkpoint over a folder of images.

Loads a checkpoint (default: latest ``checkpoint_best_total.pth`` under
``/media/data/seed_detector/checkpoints/``) and writes:

- an annotated contact sheet at ``<output-dir>/contact_sheet.jpg`` so you can
  eyeball whether the detector is doing the right thing
- a JSON dump of detections at ``<output-dir>/detections.json`` (one entry
  per image, with class_id, score, bbox, and bucket name) so downstream
  tools (e.g. the eventual autolabeler) can consume them

Usage:
    uv run python scripts/predict_seed_detector.py
    uv run python scripts/predict_seed_detector.py --checkpoint path/to/best.pth
    uv run python scripts/predict_seed_detector.py --image-dir /media/data/seed_detector/merged/val
"""  # noqa: E501

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rfdetr import RFDETRSmall
from seed_detector_class_mapping import TARGETS

DEFAULT_CHECKPOINT_ROOT = Path("/media/data/seed_detector/checkpoints")
DEFAULT_OUTPUT_DIR = Path("/media/data/seed_detector/predictions")

# BGR colours per bucket — must match seed_detector_audit.py.
_BUCKET_COLOURS: dict[str, tuple[int, int, int]] = {
    "drone": (255, 128, 0),
    "worker": (0, 200, 0),
    "enemy": (0, 0, 255),
}


def _resolve_checkpoint(arg: str | None) -> Path:
    """Find the checkpoint to load.

    Priority: explicit --checkpoint, else the most recent
    ``checkpoint_best_total.pth`` under ``--checkpoint-root``.
    """
    if arg is not None:
        p = Path(arg).expanduser().resolve()
        if not p.exists():
            sys.exit(f"--checkpoint {p} does not exist")
        return p

    candidates = sorted(DEFAULT_CHECKPOINT_ROOT.glob("*/checkpoint_best_total.pth"))
    if not candidates:
        # Fall back to the most recent checkpoint.pth (in-progress run).
        candidates = sorted(DEFAULT_CHECKPOINT_ROOT.glob("*/checkpoint.pth"))
    if not candidates:
        sys.exit(
            f"no checkpoint found under {DEFAULT_CHECKPOINT_ROOT}; "
            "pass --checkpoint explicitly or train one first"
        )
    return candidates[-1]


def _draw(img_bgr: np.ndarray, det: Any) -> np.ndarray:
    for bbox, score, class_id in zip(
        det.xyxy,  # type: ignore[attr-defined]
        det.confidence,  # type: ignore[attr-defined]
        det.class_id,  # type: ignore[attr-defined]
        strict=True,
    ):
        try:
            name = TARGETS[int(class_id)]
        except (IndexError, ValueError):
            name = f"class_{class_id}"
        colour = _BUCKET_COLOURS.get(name, (255, 255, 255))
        x0, y0 = int(round(float(bbox[0]))), int(round(float(bbox[1])))
        x1, y1 = int(round(float(bbox[2]))), int(round(float(bbox[3])))
        cv2.rectangle(img_bgr, (x0, y0), (x1, y1), colour, 2)
        label = f"{name} {float(score):.2f}"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="path to a .pth checkpoint")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="dir of images (default: merged/val)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument(
        "--max-images",
        type=int,
        default=32,
        help="cap on images to predict (contact-sheet limit)",
    )
    parser.add_argument(
        "--model",
        choices=("small",),
        default="small",
        help="must match the checkpoint variant",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ckpt = _resolve_checkpoint(args.checkpoint)
    image_dir = args.image_dir or Path("/media/data/seed_detector/merged/val")
    image_dir = image_dir.expanduser().resolve()
    if not image_dir.is_dir():
        sys.exit(f"--image-dir {image_dir} is not a directory")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[predict] checkpoint={ckpt}")
    print(f"[predict] image_dir={image_dir}")
    print(f"[predict] output_dir={args.output_dir}")
    print(f"[predict] target classes (in order): {list(TARGETS)}")

    model_cls = {"small": RFDETRSmall}[args.model]
    model = model_cls(pretrain_weights=str(ckpt))

    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        sys.exit(f"no images under {image_dir}")
    image_paths = image_paths[: args.max_images]

    detections_out: list[dict[str, Any]] = []
    cell_w = cell_h = 320
    cols = 4
    rows = (len(image_paths) + cols - 1) // cols
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 32, dtype=np.uint8)

    for idx, path in enumerate(image_paths):
        det = model.predict(str(path), threshold=args.threshold)
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[warn] could not read {path}", file=sys.stderr)
            continue
        # Draw on a copy sized for the contact sheet
        drawn = cv2.resize(img_bgr, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        drawn = _draw(drawn, det)

        counts: dict[str, int] = defaultdict(int)
        per_image: list[dict[str, Any]] = []
        for bbox, score, class_id in zip(
            det.xyxy,  # type: ignore[attr-defined]
            det.confidence,  # type: ignore[attr-defined]
            det.class_id,  # type: ignore[attr-defined]
            strict=True,
        ):
            try:
                name = TARGETS[int(class_id)]
            except (IndexError, ValueError):
                name = f"class_{class_id}"
            counts[name] += 1
            per_image.append(
                {
                    "class_id": int(class_id),
                    "class_name": name,
                    "score": float(score),
                    "bbox_xyxy": [float(v) for v in bbox],
                }
            )
        detections_out.append(
            {"file": path.name, "counts": dict(counts), "detections": per_image}
        )

        # Footer: source slug (filename prefix before ___).
        slug = path.name.split("___", 1)[0] if "___" in path.name else "?"
        cv2.putText(
            drawn,
            f"{slug}  |  " + " ".join(f"{n}:{c}" for n, c in counts.items()),
            (4, cell_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        r, c = divmod(idx, cols)
        sheet[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = drawn

    out_json = args.output_dir / "detections.json"
    out_json.write_text(json.dumps(detections_out, indent=2))
    out_sheet = args.output_dir / "contact_sheet.jpg"
    cv2.imwrite(str(out_sheet), sheet)
    print(f"[done] wrote {out_sheet} ({len(image_paths)} samples) and {out_json}")
    total = sum(len(d["detections"]) for d in detections_out)
    print(f"[done] {total} detections total across {len(detections_out)} images")


if __name__ == "__main__":
    main()
