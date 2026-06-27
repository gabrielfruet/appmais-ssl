# Seed Detector — Datasets Used

This is the registry of Roboflow Universe datasets that the seed detector is
built from. Every entry below was downloaded via the Roboflow SDK (key in
`.env`), unpacked under `/media/data/seed_detector/raw/<workspace>__<project>/`,
and then merged into a single COCO dataset at
`/media/data/seed_detector/merged/{train,val,test}/` by
`scripts/seed_detector_merge.py`.

The actual list lives in `scripts/seed_detector_datasets.yaml` and is read by
both `scripts/seed_detector_download.py` and `scripts/seed_detector_merge.py`;
edit that file (not this one) when adding or removing datasets, then re-run the
download and merge scripts.

## Source datasets

10/11 datasets were usable. The 11th (`hornet-detection-1wa7m`) is
unreachable with our API key (workspace permissions error) and is skipped.

| Slug (workspace\_\_project) | Version | Source classes | Mapping notes |
| --- | --- | --- | --- |
| `ufc__workerxdrone` | 5 | Bees, drone, enemy, worker | All four used directly. `Bees` (Roboflow's supercategory slot) maps to worker. |
| `datalabeling-yvo8b__bee-detection-iys4x` | 1 | bee (two ids) | Two different category ids both named `bee` (Roboflow quirk); both map to worker. |
| `bee-wz4v8__bee-detection-er0lm` | 1 | bees, Drone, Worker | Direct hits; useful drone/worker seed. |
| `beecounting__bee-counting-utltw` | 2 | bee (two ids) | Worker only; the largest single contributor (21 701 anns). |
| `dima-unddima__bee-8sqnq` | 9 | B, Bee | Probe advertised `Bee, Pollen` but v9's actual COCO only has `B, Bee`. Pollen seed **not obtained** from this dataset. |
| `arena-bee__arena-bee` | 1 | bee (two ids) | Worker only. **Test split only** — no train/val. Images end up in the merged `test/` split only. |
| `james-3iwyb__bee-counting-gsdmj` | 2 | bee (two ids) | Worker only; second largest contributor (17 932 anns). |
| `carls-workspace-m5woj__bee-project-lj9sk` | latest (timestamped) | Bee-Project, Drone, Worker | Direct hits; useful drone/worker seed. |
| `ramans-workspace-njyvf__bee-fpc1w` | 1 | bee (two ids) | Small (246 anns) worker-only contribution. |
| `project-enurp__bee-vx7uy` | 2 | bee, fanning, forager, foreign, guard, queen, initial-labels | `foreign` → enemy; `guard`, `fanning`, `forager`, `bee` → worker; `queen` dropped (no target bucket); `initial-labels` is an unused class id. **Train split only.** |

### Excluded / unreachable

- **`hornet-detection-1wa7m/bee-detection-kx8ggo`** — rejected by the Roboflow
  API with `Unsupported request / missing permissions`, so the dataset cannot
  be downloaded with the current API key. The workspace slug matches the URL
  the user supplied; if access is restored later, add it back to
  `scripts/seed_detector_datasets.yaml` with `version: 1` and re-run.

## Class-mapping rules

Implemented in `scripts/seed_detector_class_mapping.py`. Substring match
(case-insensitive), first rule wins:

| Substring | Target bucket |
| --- | --- |
| `drone` | `drone` |
| `hornet`, `yellowjacket`, `wasp`, `predator`, `intruder`, `foreign`, `enemy` | `enemy` |
| `worker`, `forager`, `guard`, `fanning`, `bee` | `worker` |

Source classes that match no rule are dropped. Final counts from a real merge
run (recorded by `scripts/seed_detector_audit.py`):

| Bucket | Annotations | % of total |
| --- | ---: | ---: |
| `drone` | 3 952 | 4.9 % |
| `worker` | 76 930 | 94.8 % |
| `enemy` | 306 | 0.4 % |
| **total** | **81 188** | 100 % |

The target schema is intentionally **3 classes** (drone / worker / enemy);
the pollen bucket was dropped because no source dataset exports pollen
annotations in COCO today. RF-DETR auto-detects the class count from the
dataset's `categories` array, so no manual override is needed. See
`docs/TRAINING.md` for how to re-introduce pollen once a real pollen dataset
is added.

## Findings and follow-ups

- **`pollen` is deferred.** None of the 10 reachable datasets has pollen
  annotations in its exported COCO (the dima-unddima project advertises
  `Pollen` at the project level, but v9's exported categories are `B, Bee`
  only). We will train the seed detector 3-class and add a pollen head when
  real pollen data is available.
- **Severe class imbalance.** Workers make up 94.8 % of annotations, drones
  4.9 %, enemies 0.4 %. Training proceeds as-is for the seed detector; a
  rebalanced sampler / weighted loss is a follow-up once you have your own
  labelled data.
- **`arena-bee` is test-only.** All 86 of its images land in the merged
  `test/` split. RF-DETR only uses `train` + `val`, so this source is unused
  for training; useful for held-out evaluation only.
- **`project-enurp` is train-only.** All 723 images land in `train/`. Its
  9 `queen` annotations were dropped (no target bucket — see mapping rules
  above). Its 261 `foreign` annotations are the bulk of the `enemy` bucket.
- **`ufc__workerxdrone` has no test split**; `arena-bee__arena-bee` has no
  train/val. After merging, the smallest per-split counts are val=808 and
  test=406 images, which is enough for sanity checks but small for proper
  evaluation. Consider re-splitting if you want larger val/test.

## Reproducing the merge

```bash
ssh strongandcourageous
cd /home/fruet/dev/pytorch/appmais-ssl
uv run python scripts/seed_detector_download.py
uv run python scripts/seed_detector_merge.py --overwrite
uv run python scripts/seed_detector_audit.py --contact-sheet --samples 32
```

The merged layout is what RF-DETR expects:

```
/media/data/seed_detector/merged/
  train/
    _annotations.coco.json
    <slug>___<original_filename>.jpg
  valid/
    _annotations.coco.json
    ...
  test/
    _annotations.coco.json
    ...
```

Point `rfdetr` (or `RoboflowTrainer`) at `/media/data/seed_detector/merged/`
to train the seed detector.