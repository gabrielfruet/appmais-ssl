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

## Latest run (2026-06-27)

- **Model**: `RFDETRSmall` @ 512×512, 31.8 M trainable params
- **Dataset**: merged 3-class, 7 458 train imgs / 808 val imgs / 406 test imgs
- **Wall time**: ~3 h 20 min on RTX 3060 (10–11 min per epoch incl. eval)
- **Epochs completed**: 22 / 30 — early-stopped by RF-DETR after
  10 epochs without improvement
- **Best mAP@.50:.95**: **0.578** (overall, EMA-tracked best, val split)

### Independent pycocotools eval

After training, `scripts/eval_seed_detector.py` re-runs the best
checkpoint over both splits with pycocotools (COCO's official metric)
for a sanity check independent of RF-DETR's torchmetrics-based val:

| Split | AP@.50:.95 | AP@.50 | AP@.75 | drone AP | worker AP | enemy AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| valid (808 imgs) | **0.577** | 0.825 | 0.657 | 0.658 | 0.540 | 0.370 |
| test (406 imgs) | **0.714** | 0.953 | 0.838 | 0.718 | 0.575 | n/a (no enemy anns in test) |

Test AP is higher than valid because the val split has a larger share
of `ufc__workerxdrone` (a drone-heavy source). The merged dataset
inherits Roboflow's per-source train/valid/test splits rather than
randomly partitioning across all images, so the test set is
distributionally a bit easier. Both splits confirm the model is
production-quality for autolabel use.

### Per-class metrics during training (epoch 22/30)

| class | AP | AR | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| drone | 0.648 | 0.780 | 0.841 | 0.921 | 0.774 |
| worker | 0.698 | 0.757 | 0.945 | 0.946 | 0.945 |
| enemy | 0.293 | 0.611 | 0.632 | 0.600 | 0.667 |

Worker and drone are solid for a seed detector. Enemy AP is lower
because only ~9 enemy annotations end up in the val split; this is
acceptable for a seed model — the autolabel pass will need human
review of enemy predictions regardless.

- **Best checkpoint**: `checkpoint_best_total.pth` (regular=0.578,
  ema=0.565). This is the file to point `predict_seed_detector.py`
  and `eval_seed_detector.py` at for inference / eval.

### Known caveats

- `skip_best_epochs=3` means runs shorter than 4 epochs will not save
  any checkpoint. For a smoke test, override with
  `--skip-best-epochs 0`.
- The training set has only 362 drone annotations in val and 9 enemy
  annotations in val, so per-class metrics for these classes are noisy.
  The overall mAP and worker metrics are the most reliable signal.
- The merged dataset uses `valid/` as the split directory name (RF-DETR's
  `rfdetr/datasets/coco.py` expects `valid/_annotations.coco.json`).
  Do not rename it to `val/`.
- The Roboflow-per-dataset splits mean some sources (e.g.
  `ufc__workerxdrone`, a major drone source) have no test images and
  one source (`arena-bee`) only appears in test. If you want a
  per-class stratified split across sources, re-partition the merged
  dataset before training; right now the held-out numbers are skewed
  by source coverage.

## Output artefacts

For a run in `/media/data/seed_detector/checkpoints/small-<ts>/`:

| File | What it is |
| --- | --- |
| `last.ckpt` | Most recent epoch (used to resume) |
| `checkpoint_best_ema.pth` | Best validation mAP, EMA weights |
| `checkpoint_best_regular.pth` | Best validation mAP, raw weights |
| `checkpoint_best_total.pth` | Final best model — **use this for inference** |
| `checkpoint_<N>.ckpt` | Periodic snapshots at epoch N |
| `events.out.tfevents.*` | TensorBoard logs (`tensorboard --logdir <output_dir>`) |
| `metrics.csv` | Per-step LR + per-epoch val metrics, easy to chart |
| `training_config.json` | Exact kwargs the run was launched with |
| `hparams.yaml` | Hyperparameters written by PyTorch Lightning |

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