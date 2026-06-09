# Ralph Loop Prompt: Bee Crop Dataset with Background Swapping

You are iterating on a small, well-scoped task inside the `tcc` repository at
`/Users/gabrielfruet/dev/python/tcc`. The repo already has a video
frame-extraction pipeline built on MOG2 and a `BeeCropDataset` that yields
bee-centered crops with optional background swapping.

Your job in this iteration is to **fix the pipeline so the smoke test
output is actually usable**. The previous iteration claimed DONE but the
output was bad: 11/16 samples were "no-bee" fallbacks (full hive frame,
not a bee crop), the swapped samples were nearly black, and only 4/100
videos were processed — most of which were the blurriest, most-static
videos in the dataset.

## What went wrong last time (do not repeat)

1. **No video curation.** The extract step ran on every `.mp4` in
   `data/videos_raw/`, including the worst ones
   (e.g. `AppMAIS10R@2022-12-05@18-20-00` is mostly black,
   `AppMAIS4RB@2024-11-17@18-00-00` is dark) and 4 videos took all the
   per-video MOG2 budget while 96 sat unprocessed.
2. **MOG2 background saved at downsampled resolution.** MOG2 was given
   320-px-wide frames; the saved `background.png` is therefore 320×240
   and is blurry when the dataset upsamples it to the frame's
   640×480. The swapped crops ended up looking like a bee pasted onto
   a soft, dark wash.
3. **No-bee frames were kept as "fallback" samples.** When MOG2 found
   zero foreground components, the dataset returned the full hive frame
   with a "no-bee" mask. The smoke test happily accepted these. Most
   of the contact sheet was therefore full-hive tiles, not bee crops.
4. **Smoke test did not catch any of this.** It checked that each saved
   image had *some* luminance (the full hive frame is bright) and that
   swap pixels differed (a 320×240 blur upscaled to 640×480 does differ
   from the sharp hive frame, just not usefully).

## What changes in this iteration

- **Step 0 (NEW)**: A new `scripts/curate_videos.py` scores every video
  on sharpness + motion + brightness, prints a ranked table, and writes
  a curated list of the top-N most-diverse videos to
  `data/curated_videos.txt`.
- **Step 1 (MODIFIED)**: `extract_video_frames.py` accepts a
  `--videos-file` flag, **skips frames with no detected bee** (uses the
  existing `find_bee_components` helper), and **saves the MOG2
  background at full frame resolution** by upscaling with
  `cv2.INTER_CUBIC`.
- **Step 2 (UNCHANGED)**: Pure helpers in `src/engine/bee_crop.py` are
  done; you should not need to touch them.
- **Step 3 (MODIFIED)**: `BeeCropDataset` **removes the no-bee
  fallback** entirely and **filters samples at `__init__`** so it only
  contains frames that have at least one foreground component.
- **Step 4 (MODIFIED)**: The smoke test is stricter and would have
  failed last time:
  - fails if `len(dataset) < num_samples`
  - fails if any sample has zero foreground pixels
  - fails if any `background.png` is significantly smaller than the
    frames in the same video
  - fails if a swapped sample's background region is a low-detail wash
- **Step 5 (MODIFIED)**: One new test confirms the dataset filters
  no-bee samples at init.
- **Steps 6–8 (MODIFIED)**: Docs, final gate, visual inspection all
  updated for the new flow.

## Repository conventions (read `AGENTS.md` first)

- Python 3.13, managed with `uv`. Run tools via `uv run ...`.
- Keep code KISS: simple, readable, static > clever. Small functions, no big
  god-functions. Use type hints everywhere.
- Match the existing style of `scripts/extract_video_frames.py`: small
  top-level functions, one Click command per script, clear `-> None` returns.
- New library code goes under `src/engine/`. The `engine` package is already
  initialized; just add modules.
- New script goes under `scripts/` and gets documented in `docs/SCRIPTS.md`.
- Before finishing, all three of these must pass cleanly:
  - `uv run ruff format . --check`
  - `uv run ruff check .`
  - `uv run basedpyright`
- Follow the style in `~/.dotfiles/prompts/PYTORCH-AGENTS.md` for any
  tensor code (functional ops, explicit dtypes, full type hints).

## Final goal (do not lose sight of this)

A user can run:

```bash
# 1) Curate the input videos (ranks all, picks top-N diverse)
uv run python scripts/curate_videos.py data/videos_raw \
    --output data/curated_videos.txt --top-n 8

# 2) Extract frames + MOG2 masks + sharp background for those videos
uv run python scripts/extract_video_frames.py data/videos_raw data/frames \
    --videos-file data/curated_videos.txt \
    --foreground-masks --save-background --overwrite

# 3) Smoke-test the dataset
uv run python scripts/smoke_bee_dataset.py data/frames \
    --output samples/bees
```

…and get a `BeeCropDataset` whose contact sheet is a 4×4 grid of
**real bee-centered crops** (no full-hive fallbacks, no black tiles)
with a sharp, recognizably-different background on the swapped ones.

---

## Step 0 — Video curation (NEW)

**File:** `scripts/curate_videos.py` (new)
**Output:** `data/curated_videos.txt` (one absolute path per line)

The 100 videos in `data/videos_raw/` vary wildly in quality — some are
sharp, well-lit, full of moving bees; others are blurry, dark, or
static. Running the full extract pipeline on all 100 is slow and yields
mostly junk. Curate first.

### Behaviour

Click command, single file, with these options (defaults shown):

```
uv run python scripts/curate_videos.py [INPUT_DIR] \
    [--output data/curated_videos.txt] \
    [--top-n 8] \
    [--sharpness-min 100] \
    [--motion-min 0.5] \
    [--brightness-min 60] \
    [--num-samples 8]
```

- `INPUT_DIR` defaults to `data/videos_raw`. Walk it for `*.mp4`.
- For each video, sample `num-samples` frames (default 8) evenly across
  the *middle 90%* of the video (skip first and last 5%). Use
  `cv2.VideoCapture.set(CAP_PROP_POS_MSEC, ts*1000)` + `read()` — do
  not decode the whole video.
- For each sampled frame, compute:
  - `sharpness` = variance of the Laplacian (grayscale) — float
  - `brightness` = mean grayscale value — float
- For each pair of consecutive sampled frames, compute:
  - `motion` = mean absolute difference between the two frames after
    downsampling both to 160×120 grayscale. The video's overall
    `motion` is the mean of these per-pair values.
- Per video, take `mean` of the per-frame `sharpness` and `brightness`.
- Print a ranked table to stdout with columns
  `rank | video | sharp | motion | bright | hive_id` sorted by
  `sharpness * (1 + motion)` descending. The table must include
  **every** video (not just the kept ones) so the next iteration can
  eyeball the rankings.
- Apply hard filters: drop videos where
  `sharpness < sharpness-min` OR `motion < motion-min` OR
  `brightness < brightness-min`. (These defaults are conservative —
  raise them later if the curated list still has bad videos.)
- From the survivors, pick `top-n` videos **maximizing hive-id
  diversity**: greedy by score, skipping a video if its hive id is
  already represented in the current top-n. (A simple
  `seen_hives: set[str]` guard. The point is: with 27 hives and 100
  videos, do not let one hive's many videos crowd out the others.)
- Write the picked videos to `--output`, one **absolute** path per
  line. Lines starting with `#` are comments. Create the parent
  directory if missing.
- Exit 0 even if the curated list is empty, but `click.echo` a clear
  warning when it is.
- If `top-n` survivors is less than `top-n` and `--strict` is passed,
  exit non-zero. (No default `--strict`; the smoke test will check the
  count anyway.)

### Style

- Module-level pure functions, one Click command at the bottom — same
  shape as `scripts/extract_video_frames.py`.
- The score computation must be a function that takes
  `Sequence[np.ndarray]` and returns a small NamedTuple or dataclass
  with `sharpness`, `motion`, `brightness` — easy to unit-test.
- The picking logic must be a function that takes
  `Sequence[ScoredVideo]` and `int top_n` and returns
  `list[ScoredVideo]`. This is the part that picks diverse hives.

### Verify

```bash
uv run python scripts/curate_videos.py data/videos_raw \
    --output data/curated_videos.txt --top-n 8
```

Pass conditions:
- Exit code 0.
- `data/curated_videos.txt` exists with 8 lines (or as many as survived
  filtering, with a clear warning if fewer).
- All 8 lines point to files that exist and end in `.mp4`.
- The ranked table was printed to stdout. **Open it in your
  scratch and visually verify**: the top 8 should be from at least 6
  different hives, the bottom of the all-videos table should contain
  the known-bad videos (`AppMAIS10R@2022-12-05@18-20-00`,
  `AppMAIS4RB@2024-11-17@18-00-00`, `AppMAIS15R@2023-12-11@17-05-00`),
  and there should be no `.mp4` listed in the curated file that has
  `sharpness < 200` unless all surviving videos are below that.

If those eyeball checks fail, raise the thresholds or change the
picking rule. Do not move on until the curated list looks right.

### Reset the previous run's outputs

Before Step 1, remove the artefacts of the previous (bad) run:

```bash
rm -rf data/frames samples/bees DONE
git status
```

The first three commands must be silent. `git status` will show
deletions of files that were committed by the previous iteration —
that's expected and you should commit them in the Step 1 commit.

---

## Step 1 — Extract on the curated list, sharp background, no-bee filter

**File:** `scripts/extract_video_frames.py`

### 1a. Accept a curated videos file

Add a Click option:

```python
@click.option(
    "--videos-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional path to a text file with one video path per line. "
        "If given, only these videos are processed. Lines starting "
        "with '#' are ignored. Mutually exclusive with passing a "
        "directory that already has the curated list as its only "
        "contents — use the option explicitly."
    ),
)
```

In `main`, after `videos = find_videos(input_path)`:
- If `--videos-file` is set, replace `videos` with the parsed lines.
  Parse defensively: strip whitespace, skip empty/`#` lines, resolve
  to absolute paths, error out on missing files.
- If both the curated list and the input directory are given, the
  curated list wins.

Document in `docs/SCRIPTS.md`.

### 1b. Skip frames where MOG2 found no bee

In the MOG2 branch of `extract_frames`, **after** the candidate passes
the existing `should_save_frame` gate and **before** the
`write_frame(...)` call:

- Read `mask` as grayscale (use the same resize-to-frame logic the
  existing code already does — pass the full-res mask to
  `find_bee_components`).
- Call `find_bee_components(mask_resized, min_area=...)` where
  `min_area` is a new Click option `--min-bee-area` (default 50,
  shared with the dataset's default).
- If the component list is **empty**: do **not** write the frame,
  bump a new `skipped_no_bee` counter, and `continue`. Do not advance
  `last_saved_signature` or `last_saved_time` (we are skipping, not
  saving).
- Otherwise proceed as today.

Import the helper at the top of the script:
`from engine.bee_crop import find_bee_components`.

Add a `skipped_no_bee` counter and `click.echo` it in the per-video
summary line:
`"<video>: sampled N, saved M, skipped K no-bee to <dir>"`.

If `foreground_masks` is False, the filtering is irrelevant (no
foreground is being computed) and `skipped_no_bee` is always 0. Do
nothing special in that branch.

### 1c. Save the MOG2 background at full frame resolution

In `extract_frames` (MOG2 branch), track the original frame shape
**once**, the first time you read a frame successfully:

```python
original_frame_shape: tuple[int, int] | None = None
...
ok, frame = capture.read()
if not ok:
    break
if original_frame_shape is None:
    original_frame_shape = frame.shape[:2]  # (H, W)
```

Pass `original_frame_shape` to `save_background_image(...)`. The
helper, after `getBackgroundImage()` and **before** `cv2.imwrite`:

```python
if original_frame_shape is not None and background.shape[:2] != original_frame_shape:
    background = cv2.resize(
        background,
        (original_frame_shape[1], original_frame_shape[0]),  # (W, H)
        interpolation=cv2.INTER_CUBIC,
    )
```

Update `save_background_image`'s signature and the `click.echo` line
to mention the new behaviour: `saved background (WxH) to <path>`.

### Verify

```bash
uv run python scripts/curate_videos.py data/videos_raw \
    --output data/curated_videos.txt --top-n 8
uv run python scripts/extract_video_frames.py data/videos_raw data/frames \
    --videos-file data/curated_videos.txt \
    --foreground-masks --save-background --overwrite
```

Pass conditions:
- Exit code 0.
- `data/frames/` has exactly 8 subdirectories, one per curated video.
- Each subdirectory has at least e.g. 5 `frame_*.jpg` files (MOG2
  should not have skipped every frame from any of the curated videos —
  if it did, that video is a bad pick and Step 0 needs tighter
  filters).
- Each subdirectory has a `background.png` whose shape matches the
  frame shape (640×480 for this dataset). Check this with
  `python -c "import cv2; print(cv2.imread('data/frames/<vid>/background.png').shape)"`
  on a couple of them.
- Each subdirectory has `_mask.png` files matching `frame_*.jpg` files
  1:1.

Then run without `--overwrite` on a fresh output dir and confirm
nothing is created (regression check for the existing skip-existing
behaviour).

---

## Step 2 — Pure helpers for the dataset (already done)

**File:** `src/engine/bee_crop.py`

This file is already in place from the previous iteration. Read it,
but do not change it unless Step 1c reveals a real bug.

`find_bee_components(mask, min_area)` is now imported by both the
extract script and the dataset — that is the intended reuse.

---

## Step 3 — Dataset: remove the no-bee fallback, filter at init

**File:** `src/engine/dataset.py`

### 3a. Remove the no-bee fallback

Delete the `_no_bee_sample` method entirely. Update the class docstring
if it mentions the fallback (it does — "or returns a no-bee
fallback…"; remove that).

In `__getitem__`, the `if not components: return self._no_bee_sample(...)`
branch becomes `if not components: raise ValueError(f"No foreground components in {sample.frame_path}")`.
This branch should be unreachable after the init filter below, but the
explicit `raise` is the safety net.

### 3b. Filter samples at `__init__`

After `self._samples = discover_samples(self._root)` and before the
`if not self._samples: raise ValueError(...)` check, filter the sample
list in place. The cleanest form is a small helper that loads each
mask and returns the kept samples. Because reading every mask at
`__init__` time costs I/O, do it once and cache:

```python
def _filter_samples_with_bees(
    samples: list[_Sample], min_area: int
) -> list[_Sample]:
    kept: list[_Sample] = []
    for sample in samples:
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if find_bee_components(mask, min_area):
            kept.append(sample)
    return kept
```

Call it: `self._samples = _filter_samples_with_bees(self._samples, self._min_area)`.
Then `if not self._samples: raise ValueError(f"No frame/mask pairs with foreground components (min_area={self._min_area}) under {self._root}")`.

The `__init__` should now accept the case where the on-disk fixture
contains no-bee frames silently — they are dropped without affecting
the public surface.

### Verify

From a Python REPL:

```python
from engine.dataset import BeeCropDataset
ds = BeeCropDataset("data/frames", crop_size=128)
print(len(ds))                              # > 0
for i in range(3):
    item = ds[i]
    assert (item["mask"] == 2).sum() > 0    # every sample has foreground
```

If `data/frames` is empty in your environment, run Step 1 first on the
curated list.

---

## Step 4 — Smoke-test script (stricter, no fallback)

**File:** `scripts/smoke_bee_dataset.py`

The script already builds montages and runs sanity checks. Make these
changes:

### 4a. Drop the fallback concept

- Remove the `is_fallback` boolean and the fallback branch in
  `_load_original_crop` (just return `(None, None)` if no component is
  found — and the caller can also raise in that case; the dataset
  filters, so this is unreachable in practice).
- The smoke test no longer counts "fallback" samples. Drop the
  `fallback_count` variable and the `fallback_count >= 1` warning from
  the output line.
- In the contact sheet, drop the "if no swapped, fall back to all"
  branch — if the pool is empty, the dataset sets
  `swap_background_prob = 0` and every sample is unswapped, which is
  fine for the contact sheet (it just shows the un-swapped crops).

### 4b. New required pre-flight checks (before the per-sample loop)

1. `len(dataset) > 0`, else exit non-zero.
2. `len(dataset) >= num_samples`, else exit non-zero with
   `"dataset has N samples, need at least num_samples={num_samples}"`.
3. **Background-size check**: for each `background.png` in
   `dataset.background_pool`, read its shape. For each frame in the
   same video, read the first frame's shape. If the background is
   smaller than 1.5× (in either dim) the smaller of the two frame
   dims, fail with `"background.png in {video} is {bg_w}x{bg_h}, frames
   are {frame_w}x{frame_h} — blurry-background regression"`. This
   check is what catches the original bug.
4. `len(dataset.background_pool) >= 2`, else warn (the swap diff check
   below becomes toothless with a 1-entry pool, but the smoke test
   still runs).

### 4c. Per-sample checks (replace the existing ones where noted)

Keep:
- **Centering** (unchanged): foreground centroid in
  `[0.2, 0.8] * crop_size` on both axes.
- **Coverage** (unchanged): foreground area in `[0.02, 0.60] * total`.
- **Luminance** (unchanged): mean luma `>= 5/255`.
- **Swap diff** (unchanged): for `swapped=True` samples, per-pixel mean
  abs diff over the *non-foreground* region must be `>= 5/255`.

Add:
- **No-empty-mask**: for every sample, `(mask == 2).sum() > 0`. Should
  be impossible after the init filter; this is a regression tripwire.
- **Background detail** (new): for every `swapped=True` sample, the
  *background* region (`mask == 0`) of the swapped image must have
  `np.std() >= 8.0` (over 0–255 uint8). Catches a low-detail blurry
  wash that happens to differ from the original by 5/255.
- **Luminance on the original too** (new, cheap): the unswapped
  `original_rgb` must have mean luma `>= 30/255`. The 5/255 floor was
  too lax — the full-hive fallback would pass it.

Adjust the per-sample failure aggregator to include the new check
names.

### 4d. Drop or warn on out-of-band swap ratio

The existing 20–80% swap ratio warning stays. With `seed=0` and
`swap_background_prob=0.5`, the actual ratio is deterministic but can
still be 0/16 with a tiny pool. The new "Background detail" check is
the more useful signal.

### Verify

```bash
uv run python scripts/smoke_bee_dataset.py data/frames \
    --num-samples 16 --output samples/bees
```

Pass conditions:
- Exit code 0.
- `samples/bees/contact_sheet.jpg` and `samples/bees/compare.jpg` exist
  and are non-empty.
- All quantitative checks above pass.
- No "fallback" samples were encountered (the `Items: N, swapped: M`
  line no longer has a `fallbacks:` field).
- No exceptions during iteration.

If any check fails, do not lower the thresholds. Diagnose the
underlying problem (the dataset/extract step is wrong, the curated
list is bad, etc.) and fix it.

---

## Step 5 — Tests (lean, functionality only)

**Files:**
- `pyproject.toml` — `pytest` is already in `[dependency-groups].dev`
  and `[tool.pytest.ini_options]` has `pythonpath = ["src"]`. No
  changes.
- `tests/test_bee_crop.py` — already passing, no changes.
- `tests/test_dataset.py` — add one new test (see below). Do not touch
  the existing ones unless they reference the removed
  `_no_bee_sample` (they do not — the current tests only touch the
  happy path with foreground).

### New test in `tests/test_dataset.py`

```python
def test_no_bee_filtered_at_init(tmp_path: object) -> None:
    """Frames whose masks have no foreground are dropped at init."""
    _write_sample(tmp_path, "vid_a", "frame_with_bee", fg=True)
    _write_sample(tmp_path, "vid_a", "frame_no_bee", fg=False)
    ds = BeeCropDataset(str(tmp_path), crop_size=64)
    assert len(ds) == 1
    assert all((item["mask"] == 2).sum() > 0 for item in (ds[0],))
```

Reuse the existing `_write_sample` fixture (it already takes an `fg`
flag). If the helper needs a small tweak to honour `fg=False`, do it.

### Verify

```bash
uv run pytest
```

All tests pass (the previous 8 + this new one = 9), exit code `0`.

---

## Step 6 — Docs

**File:** `docs/SCRIPTS.md`

- Add a section for `scripts/curate_videos.py` (its flags, its output
  file, an example).
- Update the `scripts/extract_video_frames.py` section to mention the
  new `--videos-file` and `--min-bee-area` flags.
- Update the `background.png` description to say it is saved at the
  frame's full resolution (upscaled with INTER_CUBIC from the MOG2
  internal size), not at the MOG2 downsampled size.

---

## Step 7 — Final gate (this is what tells the ralph loop to stop)

Run all of the following from the repo root and confirm clean output:

```bash
rm -rf data/frames samples/bees DONE

uv run ruff format . --check
uv run ruff check .
uv run basedpyright
uv run pytest

uv run python scripts/curate_videos.py data/videos_raw \
    --output data/curated_videos.txt --top-n 8
uv run python scripts/extract_video_frames.py data/videos_raw data/frames \
    --videos-file data/curated_videos.txt \
    --foreground-masks --save-background --overwrite
uv run python scripts/smoke_bee_dataset.py data/frames \
    --num-samples 16 --output samples/bees
```

If any step fails, fix it. Do not declare success until every command
exits zero.

---

## Step 8 — Visual inspection (mandatory, do not skip)

The numerical checks catch obvious breakage but cannot tell you
whether the bee is actually centered, the mask outlines a bee, or the
swapped background looks like a different hive scene. After the smoke
test exits 0, you **must look at the output images yourself** before
declaring success. You have native vision — use it: open the images
with the Read tool and look at them directly. No external vision API
calls.

Inspect:

1. `samples/bees/contact_sheet.jpg` — 4×4 of the final crops. This
   time, the centre of every tile must be a recognizable bee (or
   bee-like cluster) on a sharp hive background. No all-black tile.
   No full-hive-with-bees-in-the-corner tile.
2. `samples/bees/compare.jpg` — per-sample `original | mask | swapped`.
   The mask column must be a small white shape in the centre of the
   crop on every row. The right column (swapped) must be a different
   hive scene than the left column (original) — different texture,
   different colours, plausibly a different camera angle. The bee in
   the right column must sit at the same location as in the left
   column (this is the spatial-alignment check).
3. `data/frames/<video>/background.png` (spot-check 2 videos) — the
   background image must be a sharp, recognisable hive scene at
   640×480, not a soft blur at 320×240. If it is blurry, Step 1c did
   not apply and the smoke test missed it.

For each, answer these questions in your scratch:

- Is there a recognizable bee in the centre of every crop?
- Does the mask in `compare.jpg` look like it outlines that bee, with
  the white region matching the dark bee shape?
- For swapped samples in `compare.jpg`, does the background region
  clearly differ from the original — different texture, different
  colours, plausibly a different hive?
- Are the per-video `background.png` files sharp at 640×480?

If the answer to any of these is "no" or "I can't tell", do not
declare success. Diagnose and fix.

Only declare success once you have read the images, the answers above
are "yes", and every Step 7 command exits 0.

---

## Exit signal

When **everything** above is done — every step committed, the final gate
passes, the visual inspection is clean, and `git status` is clean —
create an empty file at the repo root:

```bash
touch DONE
```

The ralph loop (`prompts/ralph.sh`) watches for this file and stops
iterating as soon as it appears. The loop also stops on a red test
tree or a `pi` crash, in which case you should NOT have created
`DONE` — a human will pick it up.

Rules:

- Do not `touch DONE` if any of the eight steps is incomplete.
- Do not `touch DONE` if `git status` shows uncommitted changes; commit
  first, then create the file.
- Do not `touch DONE` based on partial verification. Visual inspection
  is mandatory — the numerical checks alone are not enough.
- If you decide to stop early because of a hard blocker, do NOT create
  `DONE`. Just stop and let the loop's pytest/pi gate pause for the
  human.

---

## Commit cadence

- One commit per step with a clear conventional message
  (`feat: ...`, `fix: ...`, `chore: ...`).
- Reference the step number in the body when useful.
- The first commit of this iteration should include the `rm -rf` of
  the previous run's `data/frames`, `samples/bees`, and `DONE` (if
  they were committed in the prior iteration) plus the rewrite of
  this prompt.
- Do not squash at the end; leave the history clean for review.

---

## Anti-goals (do NOT do these)

- No clever abstractions. No base classes for "BeePreprocessor", no
  plugin systems, no factory functions.
- No second pass over the video just to learn a better background.
- No new dependencies for their own sake. New runtime or dev deps are
  fine when they meaningfully simplify the code; prefer stdlib +
  existing `pyproject.toml` deps otherwise. If you add one, mention
  it in the commit body.
- No edge-blending / feathering / soft masks in the swap. Naive
  cut-paste is the spec.
- No torchvision transforms inside the dataset body. The dataset
  returns raw tensors; the user composes `v2` transforms and passes
  them via the `transforms` argument.
- Do NOT re-add the no-bee fallback. If a frame has no foreground, it
  is dropped, not converted to a "no-bee" sample. We will deal with
  that case later in a different dataset.
- Do not lower the smoke-test thresholds to make it pass. If a check
  fires, fix the underlying pipeline.
- Do not extract from the full `data/videos_raw/` directory. Always
  go through the curated list.
