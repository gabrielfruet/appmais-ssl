# Seed Detector — Training

We fine-tune [RF-DETR](https://rfdetr.roboflow.com/latest/) (Roboflow's
real-time DETR-family detection model, ICLR 2026) on the merged dataset
described in [`docs/DATASETS.md`](./DATASETS.md). The result is the
"seed detector" used for autolabelling your own data later.

## Hardware

- **GPU**: NVIDIA RTX 3060, 12 GB VRAM (`strongandcourageous`)
- **Storage**: merged dataset under `/media/data/seed_detector/merged/`,
  checkpoints under `/media/data/seed_detector/checkpoints/`

## Why these defaults

RF-DETR docs (https://rfdetr.roboflow.com/latest/learn/train/, ../advanced/)
class 12 GB VRAM as the "Low" memory tier and recommend:

| Setting | Value | Source |
| --- | --- | --- |
| `model` | `RFDETRSmall` (512×512) | Speed/accuracy sweet spot for 12 GB |
| `batch_size` | `2` | "Low" memory tier table |
| `grad_accum_steps` | `8` | → effective batch 16 |
| `gradient_checkpointing` | `True` | Required to fit Small at 512 on 12 GB |
| `epochs` | `30` | Docs' table for 2k–10k image datasets |
| `early_stopping` | `True`, `patience=10`, `min_delta=0.005` | Docs' Early Stopping Example |
| `skip_best_epochs` | `3` | Docs explicitly recommend for fine-tuning from pretrained weights |
| `lr` / `lr_encoder` | `1e-4` / `1.5e-4` | RF-DETR defaults for fine-tuning |
| `use_ema` | `True` | Default; usually improves final mAP |
| `tensorboard` | `True` | Default; no account needed |

These are the defaults baked into `scripts/train_seed_detector.py`; every
one is overridable via CLI flags.

## Current state

The merged dataset is **3-class** (drone / worker / enemy) — pollen was
removed from the target schema because no source dataset exports pollen
annotations in COCO today. RF-DETR auto-detects the class count from the
dataset's `categories` array, so no manual `num_classes` flag is required.

Per-class totals:

| bucket | annotations | share |
| --- | ---: | ---: |
| drone | 3 952 | 4.9 % |
| worker | 76 930 | 94.8 % |
| enemy | 306 | 0.4 % |

Severe class imbalance; we are training **as-is** for the seed detector
(rebalanced sampling is a future improvement once you have your own
labelled data).

## How to train

```bash
ssh strongandcourageous
cd /home/fruet/dev/pytorch/appmais-ssl
export PATH="$HOME/.local/bin:$PATH"

# 1) Smoke test: 1 epoch, Nano (smallest, fastest). Confirms the pipeline works
#    end-to-end before committing hours.
uv run python scripts/train_seed_detector.py --model nano --epochs 1 \
    --output-dir /media/data/seed_detector/checkpoints/smoke

# 2) Real run: defaults (Small, 30 epochs, early stopping on).
uv run python scripts/train_seed_detector.py

# 3) Resume if training was interrupted
uv run python scripts/train_seed_detector.py \
    --resume /media/data/seed_detector/checkpoints/small-<ts>/checkpoint.pth

# 4) Predict on the val split (sanity-check before training on real data)
uv run python scripts/predict_seed_detector.py \
    --checkpoint /media/data/seed_detector/checkpoints/small-<ts>/checkpoint_best_total.pth \
    --image-dir /media/data/seed_detector/merged/valid \
    --max-images 32
```

## What to watch

- **Training loss should drop steadily** across the first few hundred steps.
- **Validation mAP** (`checkpoint_best_total.pth` is selected on COCO
  box mAP@.50:.95) should rise and plateau.
- **Early stopping** triggers if `mAP` does not improve by `min_delta`
  (default 0.005) for `patience` epochs (default 10). The first 3 epochs are
  ignored (`skip_best_epochs`) so the best-checkpoint tracker isn't pinned
  to the pretrained weights.
- **Per-class AP** at the end: drone and worker should be the highest,
  enemy will be lowest (only 306 annotations across 723 images).
- **GPU memory**: if you hit OOM, drop `batch_size` to `1` and bump
  `grad_accum_steps` to `16`. If still OOM, lower `--resolution` to `384`
  (Small's checkpoint supports it; it's the Nano native resolution).

## Output artefacts

For a run in `/media/data/seed_detector/checkpoints/small-<ts>/`:

| File | What it is |
| --- | --- |
| `checkpoint.pth` | Most recent epoch (used to resume) |
| `checkpoint_best_ema.pth` | Best validation mAP, EMA weights |
| `checkpoint_best_regular.pth` | Best validation mAP, raw weights |
| `checkpoint_best_total.pth` | Final best model — **use this for inference** |
| `checkpoint_<N>.pth` | Periodic snapshots at epoch N |
| TensorBoard logs | Open with `tensorboard --logdir <output_dir>` |

## Adding pollen later

When a Roboflow dataset with actual pollen-basket annotations becomes
available:

1. Add it to `scripts/seed_detector_datasets.yaml`.
2. Re-run `scripts/seed_detector_download.py` and
   `scripts/seed_detector_merge.py` to regenerate the merged dataset.
3. In `scripts/seed_detector_class_mapping.py`, add `"pollen"` back to
   `TARGETS` (and re-add the `pollen` rule in `_RULES` for completeness).
4. Re-train from scratch with the same command — RF-DETR will auto-detect
   the new 4-class schema from the COCO categories.