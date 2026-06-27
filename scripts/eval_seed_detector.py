"""Compute COCO-style mAP for a trained seed-detector checkpoint.

Runs the model over a split's images, converts detections into the COCO
results format, and evaluates against the split's ground-truth
``_annotations.coco.json`` via ``pycocotools``. Reports overall and
per-class AP@.50:.95, AP@.50, AP@.75.

Default split is ``test/`` (held out from training and validation).

Usage:
    uv run python scripts/eval_seed_detector.py
    uv run python scripts/eval_seed_detector.py --split valid
    uv run python scripts/eval_seed_detector.py \
        --checkpoint /path/to/checkpoint_best_total.pth
"""  # noqa: E501

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from rfdetr import RFDETRSmall
from seed_detector_class_mapping import TARGETS

DEFAULT_DATASET_DIR = Path("/media/data/seed_detector/merged")
DEFAULT_CHECKPOINT_ROOT = Path("/media/data/seed_detector/checkpoints")
DEFAULT_OUTPUT_DIR = Path("/media/data/seed_detector/eval")


def _resolve_checkpoint(arg: str | None) -> Path:
    if arg is not None:
        p = Path(arg).expanduser().resolve()
        if not p.exists():
            sys.exit(f"--checkpoint {p} does not exist")
        return p
    candidates = sorted(DEFAULT_CHECKPOINT_ROOT.glob("*/checkpoint_best_total.pth"))
    if not candidates:
        sys.exit(
            f"no checkpoint found under {DEFAULT_CHECKPOINT_ROOT}; "
            "pass --checkpoint explicitly"
        )
    return candidates[-1]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument(
        "--split",
        choices=("test", "valid", "train"),
        default="test",
        help="which split to evaluate against (default: test)",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--threshold", type=float, default=0.0, help="score threshold (0 keeps all)"
    )
    p.add_argument("--max-images", type=int, default=None, help="cap on images (debug)")
    p.add_argument("--model", choices=("small",), default="small")
    p.add_argument("--seed", type=int, default=0)
    return p


def _per_class_table(coco_eval: Any) -> dict[str, dict[str, float]]:
    """Pull per-class AP@.50:.95, AP@.50, AP@.75 from a COCOeval object."""
    precision = coco_eval.eval["precision"]  # [T, R, K, A, M]
    # iouThr=0.5 is row 0, iouThr=0.75 is row 2; area=large idx 3, maxDets=100 idx 1.
    per_class: dict[str, dict[str, float]] = {}
    for k, name in enumerate(coco_eval.cocoGt.dataset["categories"]):
        # -1 means no positive samples for that class.
        p_all = precision[:, :, k, 0, 1]
        p_50 = precision[0, :, k, 0, 1]
        p_75 = precision[2, :, k, 0, 1]
        ap_all = float(p_all[p_all > -1].mean()) if (p_all > -1).any() else float("nan")
        ap_50 = float(p_50[p_50 > -1].mean()) if (p_50 > -1).any() else float("nan")
        ap_75 = float(p_75[p_75 > -1].mean()) if (p_75 > -1).any() else float("nan")
        per_class[name["name"]] = {"ap": ap_all, "ap50": ap_50, "ap75": ap_75}
    return per_class


def main() -> None:
    args = _build_argparser().parse_args()
    ckpt = _resolve_checkpoint(args.checkpoint)
    split_dir = (args.dataset_dir / args.split).resolve()
    if not split_dir.is_dir():
        sys.exit(f"split dir not found: {split_dir}")
    gt_path = split_dir / "_annotations.coco.json"
    if not gt_path.exists():
        sys.exit(f"missing {gt_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] checkpoint={ckpt}")
    print(f"[eval] split={args.split}  gt={gt_path}")
    print(f"[eval] output_dir={args.output_dir}")

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(gt_path))
    img_ids = sorted(coco_gt.imgs.keys())
    if args.max_images is not None:
        img_ids = img_ids[: args.max_images]
    print(f"[eval] {len(img_ids)} images, {len(coco_gt.anns)} GT annotations")

    model_cls = {"small": RFDETRSmall}[args.model]
    model = model_cls(pretrain_weights=str(ckpt))

    results: list[dict[str, Any]] = []
    per_class_counts: dict[str, int] = defaultdict(int)

    for img_id in img_ids:
        img_info = coco_gt.imgs[img_id]
        path = split_dir / img_info["file_name"]
        if not path.exists():
            print(f"[warn] missing image {path}", file=sys.stderr)
            continue
        det = model.predict(str(path), threshold=args.threshold)
        for bbox, score, class_id in zip(
            det.xyxy,  # type: ignore[attr-defined]
            det.confidence,  # type: ignore[attr-defined]
            det.class_id,  # type: ignore[attr-defined]
            strict=True,
        ):
            cid = int(class_id)
            if cid < 0 or cid >= len(TARGETS):
                # RF-DETR can emit garbage class ids when threshold is very low
                # and NMS keeps redundant queries. Drop them — they don't map
                # to anything real in our schema.
                continue
            x0, y0, x1, y1 = (float(v) for v in bbox)
            name = TARGETS[cid]
            per_class_counts[name] += 1
            results.append(
                {
                    "image_id": int(img_id),
                    "category_id": cid,  # merged dataset uses 0-indexed COCO cats
                    "bbox": [x0, y0, x1 - x0, y1 - y0],  # xywh
                    "score": float(score),
                }
            )

    results_path = args.output_dir / f"results_{args.split}.json"
    results_path.write_text(json.dumps(results))
    print(f"[eval] wrote {len(results)} detections to {results_path}")

    if not results:
        sys.exit("no detections produced; nothing to evaluate")

    coco_dt = coco_gt.loadRes(str(results_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    # Silence COCOeval's default print; we format our own summary below.
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()

    stats = coco_eval.stats  # [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ...]
    summary = {
        "checkpoint": str(ckpt),
        "split": args.split,
        "n_images": len(img_ids),
        "n_detections": len(results),
        "per_class_detections": dict(per_class_counts),
        "overall": {
            "AP_50_95": float(stats[0]),
            "AP_50": float(stats[1]),
            "AP_75": float(stats[2]),
            "AP_small": float(stats[3]),
            "AP_medium": float(stats[4]),
            "AP_large": float(stats[5]),
        },
        "per_class": _per_class_table(coco_eval),
    }
    out_summary = args.output_dir / f"summary_{args.split}.json"
    out_summary.write_text(json.dumps(summary, indent=2))

    print("\n=== Overall ===")
    print(f"  AP@.50:.95 : {stats[0]:.4f}")
    print(f"  AP@.50     : {stats[1]:.4f}")
    print(f"  AP@.75     : {stats[2]:.4f}")
    print("=== Per-class AP@.50:.95 ===")
    for name, m in summary["per_class"].items():
        ap = m["ap"]
        # NaN-safe: NaN != NaN, so we use math.isnan explicitly.
        import math

        print(f"  {name:<8}: {ap:.4f}" if not math.isnan(ap) else f"  {name:<8}: n/a")
    print(f"\n[done] summary -> {out_summary}")


if __name__ == "__main__":
    main()
