"""Fine-tune RF-DETR on the merged seed-detector dataset.

This is a thin wrapper around RF-DETR's high-level ``model.train()`` API
(https://rfdetr.roboflow.com/latest/learn/train/) that locks in the settings
that fit a single RTX 3060 (12 GB VRAM, the "Low" memory tier in RF-DETR
docs) and our 3-class (drone / worker / enemy) schema.

Defaults chosen here (overridable via flags) match RF-DETR's published
recommendations for our setup:

- ``RFDETRSmall`` at 512x512 — best speed/accuracy tradeoff that still fits
  on 12 GB VRAM.
- ``batch_size=2`` + ``grad_accum_steps=8`` → effective batch 16, the value
  RF-DETR docs target across GPU tiers.
- ``gradient_checkpointing=True`` — required to fit Small at 512 on 12 GB.
- ``epochs=30`` — RF-DETR docs' guidance for the 2k-10k image range.
- ``early_stopping=True`` + ``patience=10`` + ``skip_best_epochs=3`` — the
  docs explicitly recommend ``skip_best_epochs`` for fine-tuning from
  pretrained weights so the initial-finetuning burn-in doesn't pin the best
  checkpoint to the random-pretrained weights.

The model picks up the class count from the dataset's ``_annotations.coco.json``
automatically (we currently have 3 categories: drone / worker / enemy).

Usage:
    uv run python scripts/train_seed_detector.py
    uv run python scripts/train_seed_detector.py --model small --epochs 5  # smoke test
    uv run python scripts/train_seed_detector.py --model nano --resolution 384
    uv run python scripts/train_seed_detector.py --resume output/checkpoint.pth
"""  # noqa: E501

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from seed_detector_class_mapping import TARGETS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = Path("/media/data/seed_detector/merged")
DEFAULT_OUTPUT_ROOT = Path("/media/data/seed_detector/checkpoints")

# (variant symbol, rfdetr import path, native resolution)
MODEL_VARIANTS: dict[str, tuple[str, int]] = {
    "nano": ("RFDETRNano", 384),
    "small": ("RFDETRSmall", 512),
    "medium": ("RFDETRMedium", 576),
    "large": ("RFDETRLarge", 704),
}


def _read_num_classes(dataset_dir: Path) -> int:
    """Return the number of categories in the train COCO JSON."""
    jp = dataset_dir / "train" / "_annotations.coco.json"
    if not jp.exists():
        sys.exit(f"missing {jp} — run scripts/seed_detector_merge.py first")
    coco = json.loads(jp.read_text())
    return len(coco.get("categories", []))


def _import_model(symbol: str):  # type: ignore[no-untyped-def]
    """Import and return the RFDETR variant class by name."""
    import rfdetr  # local import: rfdetr is heavy

    try:
        return getattr(rfdetr, symbol)
    except AttributeError as exc:
        sys.exit(f"rfdetr has no class {symbol!r}: {exc}")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        choices=sorted(MODEL_VARIANTS),
        default="small",
        help="RF-DETR variant (default: small — best fit for a 12 GB GPU).",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Path to a COCO dataset with train/ and val/ subdirs.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where checkpoints + logs go (default: <root>/<model>-<timestamp>/).",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-encoder", type=float, default=1.5e-4)
    p.add_argument(
        "--resolution", type=int, default=None, help="override native resolution"
    )
    p.add_argument(
        "--no-grad-checkpointing",
        action="store_true",
        help="disable gradient checkpointing",
    )
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--no-early-stopping", action="store_true")
    p.add_argument("--early-stopping-patience", type=int, default=10)
    p.add_argument("--early-stopping-min-delta", type=float, default=0.005)
    p.add_argument("--skip-best-epochs", type=int, default=3)
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run", type=str, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="override detected class count (default: len(categories) in the COCO JSON)",  # noqa: E501
    )
    p.add_argument("--device", default="cuda")
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    symbol, native_res = MODEL_VARIANTS[args.model]
    resolution = args.resolution if args.resolution is not None else native_res

    num_classes = (
        args.num_classes
        if args.num_classes is not None
        else _read_num_classes(args.dataset_dir)
    )
    train_coco = json.loads(
        (args.dataset_dir / "train" / "_annotations.coco.json").read_text()
    )
    coco_cats = [c["name"] for c in train_coco["categories"]]
    if sorted(TARGETS) != sorted(coco_cats):
        print(
            f"[warn] COCO categories {coco_cats} do not match TARGETS {list(TARGETS)} "
            "— proceeding anyway.",
            file=sys.stderr,
        )

    output_dir = (
        args.output_dir or DEFAULT_OUTPUT_ROOT / f"{args.model}-{int(time.time())}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cls = _import_model(symbol)
    model = model_cls()

    print(f"[train] model={symbol} resolution={resolution} num_classes={num_classes}")
    print(f"[train] dataset_dir={args.dataset_dir}")
    print(f"[train] output_dir={output_dir}")
    print(
        f"[train] epochs={args.epochs} batch_size={args.batch_size} "
        f"grad_accum_steps={args.grad_accum_steps} "
        f"effective_batch={args.batch_size * args.grad_accum_steps}"
    )

    train_kwargs: dict = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "resolution": resolution,
        "gradient_checkpointing": not args.no_grad_checkpointing,
        "use_ema": not args.no_ema,
        "early_stopping": not args.no_early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "skip_best_epochs": args.skip_best_epochs,
        "tensorboard": not args.no_tensorboard,
        "device": args.device,
    }
    if args.wandb:
        train_kwargs["wandb"] = True
        if args.wandb_project:
            train_kwargs["project"] = args.wandb_project
        if args.wandb_run:
            train_kwargs["run"] = args.wandb_run
    if args.resume is not None:
        train_kwargs["resume"] = str(args.resume)

    # The rfdetr API auto-derives num_classes from the dataset; we don't pass it
    # explicitly here, but if the explicit CLI flag ever needs to be forwarded,
    # add ``num_classes=num_classes`` to train_kwargs.
    model.train(**train_kwargs)
    print(f"[done] checkpoints in {output_dir}")


if __name__ == "__main__":
    main()
