# Ralph Loop Prompt: Bee Crop Dataset with Background Swapping

You are iterating on a small, well-scoped task inside the `tcc` repository at
`/Users/gabrielfruet/dev/python/tcc`. The repo already has a video
frame-extraction pipeline built on MOG2 and a `BeeCropDataset` that yields
bee-centered crops with optional background swapping.

This is a **polish pass**. The previous iteration delivered EDT-driven
center sampling and removed the no-bee fallback; the smoke test exits
0 and the contact sheet is generated. **The pipeline is working but the
output has four visible problems and the code is harder to test than
it needs to be.** Your job is to fix the visible problems and refactor
the few big functions into smaller, testable methods.

## What this iteration fixes (do not lose sight of this)

1. **Wrong colors in the swap.** The contact sheet has a peach/blue
   color cast — `build_swapped_crop` is fed an RGB `image` (the
   dataset converts the source frame to RGB before the call) but a
   BGR `background` (loaded with `cv2.imread`, never converted). The
   swap mixes channels: foreground in RGB, background in BGR, all
   interpreted as RGB in the output tensor. The user described it as
   "bees are blue" — that's the channel swap.
2. **Hard cut-out edges around the bee.** `build_swapped_crop` uses
   `mask == 255` only, so the bee comes with no halo. The shadow ring
   that MOG2 detected (mask value 127) stays on the new background,
   leaving a thin dark ring around the pasted bee. Switch the swap's
   foreground rule to `mask >= 127` so the bee + its shadow halo come
   from the source frame — natural feathering at zero cost.
3. **Crops are too big.** Default `crop_size=224` is 1/3 of the source
   640×480 frame in each dim, so the hive background gets clipped
   before the swap can do anything interesting. Drop the default to
   **128** (≈ 1/5 in each dim), still plenty for a bee + a context
   patch.
4. **One contact sheet is not enough.** A single `contact_sheet.jpg`
   doesn't show whether the swap is *consistently* good. Render
   **three** contact sheets with different seeds so each iteration
   can eyeball the swap quality at a glance.
5. **A few functions are still too big.** `BeeCropDataset.__getitem__`
   is 60+ lines, `extract_video_frames.extract_frames` is 100+
   lines with two big branches, and `smoke_bee_dataset.main` is 100+
   lines. Break them into smaller methods that each have a clear
   test target. **Avoid method explosion** — only extract when
   there is a clear unit-test target. A 3-line method with no test
   is not refactoring, it's noise.

## What this iteration does NOT do

- No new abstractions. No base classes, no factory functions, no
  plugin systems.
- No clever MOG2 / shadow-detection code — we just use the data MOG2
  already produced.
- No new dependencies.
- No torchvision transforms inside the dataset. The dataset returns
  raw tensors; the user composes `v2` transforms via the
  `transforms` argument.
- No re-introduction of the no-bee fallback. The dataset filters at
  init, the extract step filters during the loop, and that's that.
- Do not lower the smoke-test thresholds to make it pass. The
  numerical checks are tightened in Step 3 to catch the BGR/RGB bug.

## Repository conventions (read `AGENTS.md` first)

- Python 3.13, managed with `uv`. Run tools via `uv run ...`.
- Keep code KISS: simple, readable, static > clever. Small functions,
  no big god-functions. Use type hints everywhere.
- Match the existing style of `scripts/extract_video_frames.py`:
  small top-level functions, one Click command per script, clear
  `-> None` returns.
- New library code goes under `src/engine/`. The `engine` package
  is already initialized; just add or edit modules.
- New script goes under `scripts/` and gets documented in
  `docs/SCRIPTS.md`.
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

…and get a `samples/bees/` directory containing:

- `contact_sheet_0.jpg`, `contact_sheet_1.jpg`, `contact_sheet_2.jpg` —
  three 4×4 grids of bee crops, one per seed (`seed=0,1,2`). Every
  tile is a 128×128 crop with a recognizable bee at the centre, on a
  sharp, naturally-colored hive background, with the bee + its
  shadow halo coming from the source frame (no visible dark ring at
  the swap boundary).
- `compare.jpg` — 3-column `original | mask | swapped` montage for
  the `seed=0` run. The right column's bee is the **same bee** as the
  left column at the **same position**, with **plausibly natural
  colors** (no blue/peach channel-swap cast).
- `sample_NNN{_swapped}.jpg`, `sample_NNN_original.jpg`,
  `sample_NNN_mask.png` per sample.

The crops must be 128×128, the swap must be RGB-correct end-to-end,
and the bee must come with its natural shadow halo.

---

## Step 1 — Refactor for testability (NEW, do this first)

**Files:** `src/engine/dataset.py`, `scripts/extract_video_frames.py`,
`scripts/smoke_bee_dataset.py`

The code is KISS-shaped but a few functions do too many things.
The point of this refactor is **testability**: every new method
should be a named responsibility that maps to one focused unit
test. **Do not invent new abstractions for their own sake.** A
method that just renames three lines is method explosion, not
refactoring. The litmus test: can you describe what the new method
does in one sentence and write a focused test for it? If no, leave
the code where it is.

### 1a. `BeeCropDataset.__getitem__` (dataset.py)

Currently 60+ lines doing six things in one method. Refactor into
**4 helpers**. Suggested shape (you may adjust names/responsibilities
slightly if a unit test demands it):

1. `_load_sample(self, sample: _Sample) -> _SampleData` — reads the
   frame (BGR) and mask (grayscale) from disk, raises on I/O
   failure, returns a small dataclass with `frame_bgr`, `mask`,
   `height`, `width`. The height/width come from `frame_bgr.shape[:2]`.
   `frame_bgr` stays BGR — color conversion is the caller's
   responsibility (this keeps `_load_sample` "do one thing").
2. `_sample_window(self, sample_data: _SampleData, idx: int) -> _WindowInfo` —
   finds foreground components, raises if empty, samples the center
   via `sample_center_from_distance_transform` with the
   epoch-seeded RNG, computes the `(x0, y0, x2, y2)` window
   clamped to bounds, and (if you want to keep the bbox output)
   picks one bbox from the components. Returns a small dataclass
   with `window: tuple[int, int, int, int]` and `bbox: BeeBBox`.
   **The center sampling and the window clamping live together**
   — they always travel as a unit, and the clamp's bounds depend
   on the sampled center.
3. `_build_crop(
       self, sample_data: _SampleData, window_info: _WindowInfo, idx: int,
   ) -> _CropResult` — decides whether to swap (with the
   `seed + idx + 1` RNG), loads the background if so, calls
   `build_swapped_crop` or `crop_with_border` to get the crop, and
   returns a small dataclass with `image_rgb: np.ndarray`,
   `mask_crop: np.ndarray`, `swapped: bool`. **The swap decision,
   the background load, and the cut-paste all live in this one
   method** — three responsibilities, but they are tightly coupled
   and breaking them further would mean the caller has to know
   about the swap-decision RNG stream, which is exactly the
   hidden-state this helper exists to hide.
4. `_assemble_output(
       self, sample: _Sample, crop: _CropResult, window_info: _WindowInfo,
   ) -> dict[str, object]` — converts `image_rgb` to a
   `float32` CHW tensor, `mask_crop` to a class-indexed `int64`
   tensor (via `mask_to_classes`), builds `bbox_tensor` (if you
   keep the bbox output), builds the output dict. Does **not**
   apply the transform — that stays in `__getitem__` (so the
   transform can be a no-op for tests).

After the refactor, `__getitem__` should read as 5–7 lines that
call the helpers in order. The four small dataclasses
(`_SampleData`, `_WindowInfo`, `_CropResult`, plus the existing
`_Sample`) live at module scope so they can be imported by tests.

**Tests for the helpers** — add at least one focused test per
helper in `tests/test_dataset.py`. The existing five tests must
keep passing unchanged:

- `test_len_and_getitem_shapes`
- `test_swap_probability_one`
- `test_transform_invoked`
- `test_no_bee_filtered_at_init`
- `test_set_epoch_changes_crop`

For example:

```python
def test_load_sample_raises_on_missing_frame(tmp_path: object) -> None:
    """`_load_sample` raises if the frame is unreadable."""
    ds = BeeCropDataset.__new__(BeeCropDataset)  # bypass __init__
    from engine.dataset import _Sample
    bad = _Sample(
        frame_path=pathlib.Path("/nonexistent.jpg"),
        mask_path=pathlib.Path("/nonexistent_mask.png"),
        video_id="vid_a", frame_id="bad",
    )
    with pytest.raises(ValueError, match="Could not read frame"):
        ds._load_sample(bad)


def test_sample_window_clamps_to_bounds(tmp_path: object) -> None:
    """`_sample_window` returns a window that fits in the source frame."""
    _write_sample(tmp_path, "vid_a", "frame_with_bee", fg=True)
    ds = BeeCropDataset(str(tmp_path), crop_size=64)
    sample = ds._samples[0]
    sample_data = ds._load_sample(sample)
    window_info = ds._sample_window(sample_data, idx=0)
    x0, y0, x2, y2 = window_info.window
    h, w = sample_data.height, sample_data.width
    assert 0 <= x0 and x2 <= w and (x2 - x0) == 64
    assert 0 <= y0 and y2 <= h and (y2 - y0) == 64
```

(Do not be religious about the test names — the point is that
each helper has at least one focused test.)

### 1b. `extract_video_frames.extract_frames` (extract_video_frames.py)

The function is 100+ lines with two big branches (MOG2 vs.
non-MOG2). Refactor into **3 helpers**:

1. `_open_capture(video_path: Path) -> _VideoCaptureInfo` — opens
   the `VideoCapture`, reads `fps` and `total_frames`, raises on
   bad metadata. Returns a small dataclass with the capture and
   the parsed metadata. Pure setup helper.
2. `_extract_with_mog2(capture_info, ... MOG2 params ...) -> int` —
   the entire MOG2 branch as its own function. Returns the saved
   count. **Does not** open the capture — the caller passes in the
   already-opened capture info from `_open_capture`.
3. `_extract_without_mog2(capture_info, ... no-MOG2 params ...) -> int` —
   the non-MOG2 branch as its own function. Returns the saved
   count. Same input shape as `_extract_with_mog2` minus the MOG2
   params.

After the refactor, `extract_frames` is the dispatcher that opens
the capture, decides the branch, calls one of the two
`_extract_*` functions, and (if `save_background` is set) writes
the background image after the MOG2 branch finishes.

The two `_extract_*` functions are testable in isolation if you
stub a `VideoCapture` (record a tiny synthetic mp4 once and reuse
it as a pytest fixture under `tests/fixtures/`; or just rely on
the existing manual verify step for these — they touch disk and
video decoding, which is awkward to mock).

**At minimum**, add a test that calls
`parse_videos_file` (already a top-level function) with a tiny
synthetic `.txt` file in `tmp_path` and asserts the returned list
is what you wrote. That helper is the only one that's truly
pure in this file.

### 1c. `smoke_bee_dataset.main` (smoke_bee_dataset.py)

The `main` function is 100+ lines. Refactor into **3 helpers**:

1. `_run_preflight(
       dataset: BeeCropDataset, num_samples: int, failures: list[str],
   ) -> None` — runs the pre-flight checks: `len(dataset) == 0`,
   `len(dataset) < num_samples`, `len(background_pool) < 2` warning,
   and the background-size check. Appends to `failures` (does not
   raise; the caller decides what to do with `failures`).
2. `_collect_samples(
       dataset: BeeCropDataset, num_samples: int, failures: list[str],
   ) -> list[_SampleResult]` — runs the per-sample loop. For each
   sample: load original crop, gather the dataset output, run the
   per-sample checks, save the per-sample files. Returns a list
   of small dataclasses with everything needed to build the
   sheets later (image tensors, masks, swapped flags, original
   rgbs). Appends to `failures` on a check failure.
3. `_build_sheets(results: list[_SampleResult], output_dir: Path, seeds: list[int]) -> None` —
   builds the three contact sheets (one per seed — see Step 5
   for the seed-tuple design) and the `compare.jpg`, writes them
   to disk. **No checks, no failures** — pure I/O. Takes the
   `seeds` list so the caller controls how many contact sheets
   are rendered (Step 5 makes this 3 by default).

After the refactor, `main` is a small dispatcher: build the
dataset → preflight → collect → sheets → check failures.

**Test for the preflight helper** — at least one test that builds
a tiny dataset with `_write_sample`, calls `_run_preflight` with
it, and asserts that the failure list is empty (or non-empty,
depending on the test conditions).

### 1d. Dead-code cleanup (small, opportunistic)

During the refactor, **remove any helper that has no callers**.
For example: `square_window` in `src/engine/bee_crop.py` is no
longer used by the dataset after the EDT switch — it is tested
but uncalled. If you find a test that still references it, delete
the test too. **Do not** delete `BeeBBox`, `sample_bee_bbox`,
`mask_to_classes`, `find_bee_components`, `crop_with_border`,
`build_swapped_crop`, or `sample_center_from_distance_transform`
— all of these still have callers after Step 3.

### Verify

```bash
uv run pytest
uv run ruff format . --check
uv run ruff check .
uv run basedpyright
```

All checks green. The test count goes from 11 to ~14+ (one new
test per helper extracted, give or take).

---

## Step 2 — Fix the BGR/RGB color mismatch (NEW)

**File:** `src/engine/dataset.py`

The bug: `_load_background` reads a background PNG with
`cv2.imread(...)` and returns BGR. The dataset converts the
source frame to RGB before calling `build_swapped_crop`. Result:
the swap mixes RGB foreground pixels with BGR background pixels.
The output tensor interprets both as RGB, so the hive background
appears as a channel-swapped version of itself (warm → cool/blue
cast) and the bee region is RGB (correct). The user saw "blue
bees" — actually the background goes blue while the bee stays
correct, but the *whole image* has a wrong color cast.

The cleanest fix is to make `_load_background` return **RGB**,
matching the rest of the dataset's color contract. The single
change is one `cv2.cvtColor` call at the source — no other
conversion points need to change.

```python
def _load_background(path: Path, frame_shape: tuple[int, int]) -> np.ndarray:
    background = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if background is None:
        raise ValueError(f"Could not read background image: {path}")
    height, width = frame_shape
    if background.shape[:2] != (height, width):
        background = cv2.resize(
            background, (width, height), interpolation=cv2.INTER_AREA
        )
    return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)  # NEW
```

Update the `build_swapped_crop` docstring to say explicitly:
"all inputs (image and background) are RGB uint8".

The dataset-side call site is unchanged — it already passes
`_bgr_to_rgb(frame_bgr)` for the image and the (now-RGB)
`_load_background` result for the background. They are
consistent. **Do not** also convert at the call site — pick
one place (the source of the BGR data, which is
`_load_background`).

### Test

Add a focused test in `tests/test_dataset.py` that asserts
`_load_background` returns RGB. The cleanest way is to write a
PNG with a recognizable color and assert the channel that
contains that color:

```python
def test_load_background_returns_rgb(tmp_path: object) -> None:
    """`_load_background` returns RGB, not BGR (BGR/RGB mismatch fix)."""
    import pathlib
    path = pathlib.Path(str(tmp_path)) / "bg.png"
    # BGR red: B=255, G=0, R=0
    red_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    red_bgr[:, :, 0] = 0
    red_bgr[:, :, 1] = 0
    red_bgr[:, :, 2] = 255
    cv2.imwrite(str(path), red_bgr)
    loaded = _load_background(path, frame_shape=(10, 10))
    # After the fix, R channel = 255 (RGB red). Before the fix,
    # R channel = 0 and B channel = 255 (BGR red).
    assert int(loaded[:, :, 0].mean()) == 255, (
        f"R channel mean is {loaded[:, :, 0].mean()}, expected 255 "
        "(BGR/RGB mismatch?)"
    )
    assert int(loaded[:, :, 2].mean()) == 0
```

Add a similar test for the end-to-end swap: build a dataset
with one sample, force `swap_background_prob=1.0`, get an item,
and assert the foreground (bee region) is RGB-red (R > B) — the
existing `_write_sample` writes a `(0, 0, 255)` BGR-red bee, so
in RGB the bee's R channel is 255 and the B channel is 0.

### Verify

```bash
uv run pytest
```

The two new tests pass. The previous 11 tests still pass.

---

## Step 3 — Use shadow + foreground in the swap (NEW)

**File:** `src/engine/bee_crop.py`

Currently `build_swapped_crop` does:

```python
foreground = mask_crop == 255
```

This is too tight: the bee comes with no halo, so the hard
cut-out edge is visible. The MOG2 mask already detects a
shadow ring (value 127) around the bee — including it in the
foreground means the bee + its shadow halo come from the
original frame, which naturally feathers the cut-out edge.

Change to:

```python
foreground = mask_crop >= 127
```

This treats `mask == 127` (shadow) and `mask == 255`
(foreground) both as "in-bee" — they are copied from the
source frame into the swap.

The mask *returned* from `build_swapped_crop` (the
`mask_resized`) is still the class-form `{0, 1, 2}` tensor, so
downstream consumers see the same labels. Only the *alpha* of
the cut-paste changes.

### Alternative: morphological dilation

If the shadow ring turns out to be too sparse on this dataset
(you can spot-check by opening a `_mask.png` — if the shadow
ring is broken or absent), fall back to dilation instead:

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
foreground = cv2.dilate(
    (mask_crop == 255).astype(np.uint8), kernel, iterations=1
).astype(bool)
```

Pick **one** approach — don't combine. The shadow-mask
approach is preferred because it uses the data MOG2 already
computed (no new parameters, no new hyperparameters to tune).
Only switch to dilation if the shadow-mask visual inspection
shows it's not enough.

Update the `build_swapped_crop` docstring to document the rule
("in-bee = `mask >= 127`, i.e. shadow halo + foreground").

### Test

Add a test in `tests/test_bee_crop.py` that asserts the shadow
ring is included in the swap:

```python
def test_swap_includes_shadow_halo() -> None:
    """`build_swapped_crop` uses `mask >= 127` (shadow + foreground)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60] = (10, 20, 30)  # dark "bee"
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255  # foreground
    mask[38:40, 38:62] = 127  # shadow ring (top edge only)
    bg = np.full((100, 100, 3), 200, dtype=np.uint8)
    swapped, _ = build_swapped_crop(img, mask, bg, (30, 30, 70, 70), 40)
    # The top shadow ring at crop [8:10, 8:32] should come from `img`
    # (where it was (0, 0, 0)), not the background (200, 200, 200).
    assert (swapped[8:10, 8:32] == 0).all(), (
        f"shadow ring pixel {swapped[8:10, 8:32][0]} should be the "
        "source image (0, 0, 0), not the background (200, 200, 200)"
    )
```

The existing `test_build_swapped_crop_alignment` and
`test_build_swapped_crop_resizes_background` must keep passing
— both use masks with only 0 and 255 values, so the change from
`== 255` to `>= 127` doesn't affect them.

### Verify

```bash
uv run pytest
```

The new test passes; the 14+ existing tests still pass.

---

## Step 4 — Smaller crops (default 128) (MODIFIED)

**Files:** `src/engine/dataset.py`, `scripts/smoke_bee_dataset.py`

Change `BeeCropDataset`'s default `crop_size` from 224 to 128.
Rationale: 128 is large enough to fit a bee + a small context
patch (≈ 1/5 of the source frame area), small enough to leave
room for the hive background to be visible, and divisible by 2
in both dims (no off-by-one in the EDT clamp).

Update:

- `src/engine/dataset.py` — `def __init__(self, ..., crop_size: int = 128, ...)`
- `scripts/smoke_bee_dataset.py` — `CROP_SIZE = 128`

Add a `--crop-size` Click option to the smoke test's `main` so
the user can override the default at the CLI (e.g. to revert
to 224 for a side-by-side comparison). The `THUMB_SIZE`
constant (display size of the contact sheet) stays at 224 — it
controls the *display* size, not the *dataset* crop size. The
two are now independent: `crop_size` is what the dataset
yields, `THUMB_SIZE` is what the sheet shows.

```python
@click.option(
    "--crop-size", type=int, default=CROP_SIZE, show_default=True,
    help="Bee crop size in pixels. The contact sheet's tile size "
         "is THUMB_SIZE, independent of this.",
)
def main(root: Path, num_samples: int, output: Path, crop_size: int) -> None:
    ...
    dataset = BeeCropDataset(
        root=root, crop_size=crop_size,
        swap_background_prob=SWAP_PROBABILITY, seed=SEED,
    )
    ...
```

(If you want to be more flexible, also pass `crop_size` to
`_collect_samples` and `_load_original_crop` so the helper
doesn't read `CROP_SIZE` from module scope. Module-scope
constants are fine for now — KISS — but threading the value
through is also OK.)

### Verify

The smoke test's `samples/bees/sample_*.jpg` files should be
128×128. Check with:

```bash
uv run python -c "import cv2; print(cv2.imread('samples/bees/sample_000_original.jpg').shape)"
# Should print (128, 128, 3)
```

---

## Step 5 — Multiple contact sheets (NEW)

**File:** `scripts/smoke_bee_dataset.py`

One `contact_sheet.jpg` is not enough to judge the swap
quality — a single bad sample can hide the inconsistency, and
with a 50% swap probability, one sheet might just happen to
have all unswapped samples (or all swapped). Add **three**
contact sheets with different seeds (`seed=0`, `seed=1`,
`seed=2`) so each iteration can eyeball the swap quality at a
glance across the population.

The existing `compare.jpg` (3-column `original | mask |
swapped` for one seed) stays — it shows per-sample detail.
The three new `contact_sheet_<seed>.jpg` files show the
**population-level** behavior at a glance.

### Behaviour

```python
CONTACT_SHEET_SEEDS: tuple[int, ...] = (0, 1, 2)
```

For each seed, the smoke test:

1. Builds a `BeeCropDataset` with `seed=<seed>` and the
   configured `crop_size` / `swap_background_prob`.
2. Iterates `num_samples` items.
3. Builds a 4×4 montage of the swapped crops, saves as
   `contact_sheet_<seed>.jpg`.

**Quantitative checks run on `seed=0` only** — the canonical
"primary" run. The other two seeds are visual aids (no
checks, no failures). Document this in the smoke test's
module docstring.

Add a CLI option `--seeds "0,1,2"` (default `"0,1,2"`) that
parses a comma-separated list of integers and overrides the
default tuple.

### Cleanest implementation

Build the per-seed datasets inside a small loop, not by
re-running `_collect_samples` for each seed. The simplest
shape:

```python
# After preflight, run the primary seed (seed=0) to do all checks.
primary_dataset = BeeCropDataset(root=root, crop_size=crop_size,
                                 swap_background_prob=SWAP_PROBABILITY,
                                 seed=CONTACT_SHEET_SEEDS[0])
_run_preflight(primary_dataset, num_samples, failures)
results = _collect_samples(primary_dataset, num_samples, failures)

# Render contact sheets for every requested seed.
for seed in CONTACT_SHEET_SEEDS:
    if seed == CONTACT_SHEET_SEEDS[0]:
        sheet_results = results
    else:
        # Just iterate and build the crops; no checks.
        other = BeeCropDataset(root=root, crop_size=crop_size,
                               swap_background_prob=SWAP_PROBABILITY, seed=seed)
        sheet_results = _collect_samples(other, num_samples, failures)
    _build_sheet_for_seed(sheet_results, output, seed)
_build_compare_sheet(results, output)
```

(If your refactor in Step 1 makes `_collect_samples` accept a
dataset, this is straightforward. If not, the simplest
option is to keep `results` as a list of dataclasses and
have `_build_sheet_for_seed` just render from it — but the
primary `results` only exist for `seed=0`, so the other
seeds need their own dataset iteration. The point is: keep
the per-seed rendering logic in a small loop, not
hand-duplicated.)

### Verify

```bash
uv run python scripts/smoke_bee_dataset.py data/frames \
    --num-samples 16 --output samples/bees
ls samples/bees/contact_sheet_*.jpg
# Should print:
#   samples/bees/contact_sheet_0.jpg
#   samples/bees/contact_sheet_1.jpg
#   samples/bees/contact_sheet_2.jpg
```

Visual inspection: each sheet is a 4×4 grid of bee crops.
The swap decisions differ across sheets — a sample that is
swapped in `seed=0` may be unswapped in `seed=1`. The three
sheets together show the full distribution. Across all
three, the majority of samples should be recognizably
"bee-like" (a dark blob on a hive background, with the
swap boundary invisible).

---

## Step 6 — Documentation (MODIFIED)

**File:** `docs/SCRIPTS.md`

Update the `scripts/smoke_bee_dataset.py` section to mention:

- The default `crop_size` is now **128** (was 224). Add the
  new `--crop-size` option to the options list.
- The script now produces **three** contact sheets
  (`contact_sheet_0.jpg`, `contact_sheet_1.jpg`,
  `contact_sheet_2.jpg`) — one per seed — plus the existing
  `compare.jpg`. Add a one-line description of what each is
  for: "Contact sheets show population-level swap
  consistency across seeds; compare.jpg shows per-sample
  detail for the primary seed."
- Add a note that the swapped images use the **shadow mask
  (value 127) + foreground mask (value 255)** as the
  in-bee region, so the bee comes with its natural shadow
  halo from the original frame.

Update the `scripts/extract_video_frames.py` section if
anything in its behavior changed (it shouldn't have, but
double-check the MOG2 background section — it already
mentions INTER_CUBIC and full-frame resolution, no change
needed there).

If `docs/ENGINE.md` exists, update the `BeeCropDataset`
section to mention:

- The default `crop_size` is 128.
- The swap uses `mask >= 127` (shadow + foreground) as the
  in-bee region.
- The colors are RGB throughout the dataset; the
  background is converted from BGR to RGB at load time.

If `docs/ENGINE.md` does not exist, **do not create it in
this iteration** — that's a separate task. Just update
`docs/SCRIPTS.md`.

---

## Step 7 — Final gate (this is what tells the ralph loop to stop)

Run all of the following from the repo root and confirm clean
output:

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

If any step fails, fix it. Do not declare success until every
command exits zero.

---

## Step 8 — Visual inspection (mandatory, do not skip)

The numerical checks catch obvious breakage but cannot tell
you whether the bee is actually centered, the colors are
natural, or the swap boundary is invisible. After the smoke
test exits 0, you **must look at the output images yourself**
before declaring success. You have native vision — use it:
open the images with the Read tool and look at them directly.
No external vision API calls.

Inspect:

1. `samples/bees/contact_sheet_0.jpg`,
   `contact_sheet_1.jpg`, `contact_sheet_2.jpg` — three
   4×4 grids. Every tile must have a recognizable bee (or
   bee-like cluster) at the **center of the tile** (EDT
   peak), with **naturally-feathered edges** (no visible
   dark ring at the swap boundary — the shadow halo from
   Step 3 must be visible as a soft transition from bee to
   background), and a **plausibly natural color** (warm
   tones for the hive, dark tones for the bee — **no
   blue-tinted bees, no peach-tinted backgrounds**). Across
   the three sheets, the majority of samples must look
   bee-like — that's the population-level check.
2. `samples/bees/compare.jpg` — 3-column
   `original | mask | swapped`. The right column (swapped)
   must show **the same bee** as the left column (original)
   at the **same position**. The colors must be plausibly
   natural — the hive background in the right column must
   be a recognizably hive-like scene (warm browns, yellows,
   ambers — depending on the source video), **not** a
   blue-shifted version of itself. **No blue bees.**
3. `data/frames/<video>/background.png` (spot-check 2
   videos) — the background image must be a sharp,
   recognisable hive scene at 640×480, not a soft blur at
   320×240. If it is blurry, Step 1c of the previous
   iteration (the upscaling fix) regressed.

For each, answer these questions in your scratch:

- Is there a recognizable bee in the centre of every crop?
- Does the mask in `compare.jpg` look like it outlines that
  bee, with the white region matching the dark bee shape?
- For swapped samples in `compare.jpg`, does the background
  region clearly differ from the original — different
  texture, different colors, plausibly a different hive?
- Is the swap edge natural — **no visible dark ring at the
  cut-out boundary**?
- Are the colors plausibly natural — **no blue-tinted
  bees, no peach/blue channel-swap cast**?
- Are the per-video `background.png` files sharp at
  640×480?
- Across the three contact sheets, do most samples look
  bee-like? (Population-level consistency.)

If the answer to any of these is "no" or "I can't tell", do
not declare success. Diagnose and fix.

Only declare success once you have read the images, the
answers above are "yes", and every Step 7 command exits 0.

---

## Exit signal

When **everything** above is done — every step committed, the
final gate passes, the visual inspection is clean, and
`git status` is clean — create an empty file at the repo
root:

```bash
touch DONE
```

The ralph loop (`prompts/ralph.sh`) watches for this file and
stops iterating as soon as it appears. The loop also stops on
a red test tree or a `pi` crash, in which case you should NOT
have created `DONE` — a human will pick it up.

Rules:

- Do not `touch DONE` if any of the eight steps is
  incomplete.
- Do not `touch DONE` if `git status` shows uncommitted
  changes; commit first, then create the file.
- Do not `touch DONE` based on partial verification. Visual
  inspection is mandatory — the numerical checks alone are
  not enough.
- If you decide to stop early because of a hard blocker, do
  NOT create `DONE`. Just stop and let the loop's pytest/pi
  gate pause for the human.

---

## Commit cadence

- One commit per step with a clear conventional message
  (`feat: ...`, `fix: ...`, `chore: ...`, `refactor: ...`,
  `docs: ...`, `test: ...`).
- Reference the step number in the body when useful.
- The first commit of this iteration should include the
  `rm -rf` of the previous run's `data/frames`,
  `samples/bees`, and `DONE` (if they were committed in the
  prior iteration) plus the rewrite of this prompt. The
  commit body should list the 8 steps of this iteration in
  one line each so a reviewer can scan the plan.
- Do not squash at the end; leave the history clean for
  review.
- Step 1 (refactor) is one commit. Step 2 (BGR/RGB) is one
  commit. Step 3 (shadow + foreground) is one commit. Step 4
  (smaller crops) is one commit. Step 5 (multiple sheets) is
  one commit. Step 6 (docs) is one commit. Steps 7+8 are
  not separate commits — they are verification of the
  previous six. You can amend a step's commit if you find
  a small fix during Step 7's gate, but don't merge the
  steps into one big "polish" commit.

---

## Anti-goals (do NOT do these)

- No clever abstractions. No base classes for
  "BeePreprocessor", no plugin systems, no factory
  functions. The refactor in Step 1 is for *testability*,
  not for new abstractions.
- No method extraction for its own sake. Every new method
  in Step 1 must have a clear unit-test target. A 3-line
  method with no test is method explosion, not
  refactoring.
- No new dependencies for their own sake. New runtime or
  dev deps are fine when they meaningfully simplify the
  code; prefer stdlib + existing `pyproject.toml` deps
  otherwise. If you add one, mention it in the commit body.
- No edge-blending / feathering / soft masks beyond Step 3.
  The shadow-halo rule (`mask >= 127`) is the spec.
- No torchvision transforms inside the dataset body. The
  dataset returns raw tensors; the user composes `v2`
  transforms and passes them via the `transforms` argument.
- Do NOT re-add the no-bee fallback. If a frame has no
  foreground, it is dropped, not converted to a "no-bee"
  sample.
- Do not lower the smoke-test thresholds to make it pass.
  If a check fires, fix the underlying pipeline.
- Do not extract from the full `data/videos_raw/`
  directory. Always go through the curated list.
- Do not combine shadow-mask + dilation in the swap. Pick
  one (default: shadow-mask).
- Do not lower the contact-sheet count to 1 "to save
  time". The user explicitly asked for more than one.
- Do not introduce a new color abstraction
  ("`ImageArray` NewType for RGB vs BGR") — KISS. Just
  add the one `cv2.cvtColor` call in `_load_background`
  and document the contract in the docstring.
- Do not delete the bbox output from the dataset dict
  just because the EDT center doesn't match it. The bbox
  is a useful "largest component" label for downstream
  tasks; keep it (and document it as such in the
  `_WindowInfo` dataclass comment).
