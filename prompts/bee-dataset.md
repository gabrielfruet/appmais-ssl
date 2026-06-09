# Ralph Loop Prompt: Bee Crop Dataset with Background Swapping

You are iterating on a small, well-scoped task inside the `tcc` repository at
`/Users/gabrielfruet/dev/python/tcc`. The repo already has a video
frame-extraction pipeline built on MOG2. Your job is to add (a) a saved
background image per video and (b) a torch `Dataset` that yields bee-centered
crops with optional background swapping.

Work in small, verifiable steps. After every meaningful change, run the
verification listed for that step. Do not move on until verification passes.

---

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

---

## Final goal (do not lose sight of this)

A user can run:

```bash
uv run python scripts/extract_video_frames.py data/videos_raw data/frames \
    --foreground-masks --save-background

uv run python scripts/smoke_bee_dataset.py data/frames --output samples/bees
```

…and get a working `BeeCropDataset` that yields bee-centered crops with
optional background swapping, plus a contact sheet of N sample crops for
visual confirmation.

---

## Step 1 — Save the MOG2 background per video

**File:** `scripts/extract_video_frames.py`

- Add a `--save-background` Click flag (default: off, to keep current
  behavior identical).
- When set AND `--foreground-masks` is also set, after the MOG2 loop finishes
  successfully, call `background_subtractor.getBackgroundImage()` and write
  the result to `<video_output_dir>/background.png` as a BGR PNG.
- The image is the size MOG2 was operating on (downsampled width). That is
  fine; consumers resize as needed. Do not add a second pass.
- Click.echo a short line on success, matching the existing
  `sampled/saved` log line.
- Update `docs/SCRIPTS.md` to document the flag and the new output file.

**Verify:** run

```bash
uv run python scripts/extract_video_frames.py data/videos_raw data/frames \
    --foreground-masks --save-background
```

on at least one real video. After it finishes, confirm
`data/frames/<video_tag>/background.png` exists and is a non-empty PNG. Then
re-run the same command without `--save-background` and confirm no
`background.png` is created (no behavior change for existing users).

---

## Step 2 — Pure helpers for the dataset

**File:** `src/engine/bee_crop.py` (new)

Write small, pure, fully-typed functions. No I/O, no torch, no class state.
Return values are numpy arrays or small NamedTuples.

Required helpers (names are a suggestion, you can rename if clearer):

- `find_bee_components(mask: np.ndarray, min_area: int) -> list[BeeBBox]`
  - Input mask: `uint8` HxW with values `0` (bg), `127` (shadow), `255` (fg).
  - Threshold to binary foreground (`mask == 255`).
  - Use `cv2.connectedComponentsWithStats` to get stats.
  - Skip background label 0 and components smaller than `min_area`.
  - Return each component as a NamedTuple `BeeBBox(x: int, y: int, w: int, h: int, area: int)`.

- `sample_bee_bbox(components: Sequence[BeeBBox], rng: np.random.Generator) -> BeeBBox | None`
  - Returns `None` if the list is empty. Otherwise a uniform random pick.

- `square_window(bbox: BeeBBox, image_shape: tuple[int, int], padding_factor: float) -> tuple[int, int, int, int]`
  - `image_shape` is `(H, W)`.
  - Center the square on the bbox center. Side length is
    `max(bbox.w, bbox.h) * padding_factor`, clamped to `min(H, W)`.
  - Do NOT clamp to image bounds. The window is allowed to go slightly
    negative or past `H`/`W`; `crop_with_border` handles the padding.
    Return `(x1, y1, x2, y2)` as ints.

- `crop_with_border(image: np.ndarray, window: tuple[int, int, int, int]) -> np.ndarray`
  - Crops a single image to `window`. The window is in image coordinates
    and is allowed to go out of bounds (e.g. for a bee near the edge). Pad
    with replicated border using `cv2.copyMakeBorder(..., BORDER_REPLICATE)`
    as needed. The output is exactly the window size, regardless of bounds.

- `build_swapped_crop(image: np.ndarray, mask: np.ndarray, background: np.ndarray, window: tuple[int, int, int, int], output_size: int) -> tuple[np.ndarray, np.ndarray]`
  - **Spatial alignment is the whole point of this function.** Both the
    original frame and `background` are interpreted in the same `(H, W)`
    coordinate system, even though `background` is saved at MOG2's
    downsampled size. The function:
    1. Resizes `background` to the frame's `(H, W)` if needed
       (`cv2.INTER_AREA`).
    2. Crops the same `window` from `image`, `mask`, and the resized
       `background` using `crop_with_border` (so all three crops are
       pixel-aligned at full resolution, including at image edges).
    3. Builds a naive cut-paste: pixels where `mask == 255` keep `image`;
       everything else (including shadow) becomes `background`.
    4. Resizes the cut-pasted crop and the mask to `(output_size,
       output_size)` (`INTER_AREA` for the image, `INTER_NEAREST` for
       the mask).
    5. Returns `(swapped_image, mask)` at `output_size`.

- `mask_to_classes(mask: np.ndarray) -> np.ndarray`
  - Maps `{0, 127, 255}` -> `{0, 1, 2}` (`int64`). This is the class encoding
    the dataset will return.

**Verify:** from a Python REPL, create a synthetic mask with a couple of
forefront blobs and a synthetic background, run each helper, and assert
the outputs make sense (no crash, correct shapes, correct class mapping,
swap is pixel-aligned with the original — e.g. by checking that a known
bee pixel in the crop is identical to the original frame at that
location). This is exploratory; you do not need to commit the scratch.

---

## Step 3 — The Dataset

**File:** `src/engine/dataset.py` (new)

Define `BeeCropDataset(torch.utils.data.Dataset)` with this surface
(adjust names/types if you have a strong reason):

```python
class BeeCropDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        root: str | Path,
        crop_size: int = 224,
        padding_factor: float = 1.5,
        min_area: int = 50,
        swap_background_prob: float = 0.5,
        background_pool: Sequence[str | Path] | None = None,
        transform: Callable[[dict[str, object]], dict[str, object]] | None = None,
        seed: int = 0,
    ) -> None: ...

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> dict[str, object]: ...
```

Behavior:

- In `__init__`, walk `root` recursively. A "sample" is a frame file
  (`*.jpg` or `*.png`) that has a sibling `*_mask.png` with the same stem.
  Skip frames without masks. Store tuples of `(frame_path, mask_path,
  video_id, frame_id)` where `video_id` is the parent directory name and
  `frame_id` is the file stem.
- Build a `background_pool`:
  - If `background_pool` is provided, use it as-is.
  - Otherwise, glob `**/background.png` under `root` and use those.
  - If the pool is empty, set `swap_background_prob` effectively to 0
    (still safe).
- `__getitem__(idx)`:
  1. Load the frame (BGR via cv2) and the mask (grayscale via cv2 with
     `IMREAD_UNCHANGED`). The frame is `(H, W, 3)`; the mask is `(H, W)`.
  2. Run `find_bee_components`. If empty, return a deterministic "no-bee"
     sample: the full frame resized to `crop_size`, mask of all zeros, bbox
     equal to the frame, `swapped=False`. Log nothing in normal use; do not
     crash.
  3. Pick one component via `sample_bee_bbox` with a per-worker RNG seeded
     from `seed + idx` (use `np.random.default_rng`).
  4. Compute the `square_window` in **frame coordinates** (`(H, W)` of the
     loaded frame). The window is allowed to extend slightly past image
     bounds; the helpers pad with replicated border as needed.
  5. **No-swap path** (`swap_background_prob == 0`, or empty pool, or the
     random draw missed): crop the window from `image` and `mask` using
     `crop_with_border`, then resize both to `crop_size` (`INTER_AREA` for
     image, `INTER_NEAREST` for mask). `swapped=False`.
  6. **Swap path**: with probability `swap_background_prob` and a non-empty
     pool, pick a random background from the pool (uniformly, prefer one
     from a *different* video when the pool has more than one). Call
     `build_swapped_crop(image, mask, background, window, crop_size)` and
     use its outputs. `swapped=True`. The function handles resizing the
     candidate background to `(H, W)`, the aligned cut-paste, and the
     final crop resize — the dataset just calls it.
  7. Convert to tensors:
     - `image`: `torch.from_numpy` of a `float32` `[C, H, W]` array in
       `[0, 1]` (divide by 255). Channel order is RGB (convert BGR→RGB).
     - `mask`: `torch.from_numpy` of the class-mapped `int64` `[H, W]`
       array (use `mask_to_classes`).
     - `bbox`: `torch.tensor([x1, y1, x2, y2], dtype=torch.float32)` in the
       crop's coordinate system (i.e. relative to the crop's top-left,
       which is the window origin). For the no-bee fallback, use
       `[0, 0, crop_size, crop_size]`.
  8. Return `{"image": ..., "mask": ..., "bbox": ..., "video_id": ...,
     "frame_id": ..., "swapped": swapped}`.
  9. If `self.transforms is not None`, call it on the dict and return the
     result. **Contract:** "dict in, dict out". A typical `transforms` is
     a `torchvision.transforms.v2.Compose` (or a thin wrapper around it)
     that knows how to read the `image` / `mask` / `bbox` keys. The
     dataset does not import `torchvision`; it just calls whatever
     callable the user provided.

Use type hints everywhere. Keep each method short; delegate work to helpers
in `bee_crop.py`.

**Background swap semantics — do not forget:** the new background is taken
from a *different* video's `background.png`, resized to the current
frame's `(H, W)`, then cropped at the **same window** in the original
frame. Pixel `(0, 0)` of the resulting crop corresponds to pixel
`(window.x1, window.y1)` in both the original frame and the new
background's image space. The bee ends up at the same image coordinates
in both scenes, which is the whole point — a model trained on this
should not be able to latch onto scene-specific background features.

**Verify:** open a Python REPL and instantiate
`BeeCropDataset("data/frames", crop_size=128)`. Iterate over `range(5)`.
Assert that:
- `image` is `float32` shape `[3, 128, 128]`, values in `[0, 1]`.
- `mask` is `int64` shape `[128, 128]`, values in `{0, 1, 2}`.
- `bbox` is `float32` shape `[4]`.
- `swapped` is a `bool`.
- `video_id` and `frame_id` are `str`.

If `data/frames` is empty in your environment, run Step 1 first on at least
one real video.

---

## Step 4 — Smoke-test script (this is the loop's exit signal)

**File:** `scripts/smoke_bee_dataset.py` (new)

A small Click command:

```
uv run python scripts/smoke_bee_dataset.py [ROOT] [--num-samples 16] [--output samples/bees]
```

- Defaults: `ROOT=data/frames`, `num_samples=16`, `output=samples/bees`.
- Instantiate `BeeCropDataset(root, crop_size=224)` with
  `swap_background_prob=0.5`.
- Iterate `num_samples` items. For each, keep both the final swapped
  image and the underlying **original** (unswapped) crop at the same
  window so we can compare them side-by-side below.
- Track and `click.echo`:
  - number of items returned
  - number of items with `swapped == True`
  - number of "no-bee" fallbacks (`bbox` equal to full image and
    `mask.sum() == 0`)
- Save per-sample files into `--output`:
  - `sample_<idx:03d>_original.jpg` — the original (unswapped) crop.
  - `sample_<idx:03d>_mask.png` — the mask (class 0/1/2 scaled by
    127 for visibility: bg=black, shadow=mid-gray, fg=white).
  - `sample_<idx:03d>{_swapped}.jpg` — the final (possibly swapped)
    crop, after converting the tensor back to `uint8` HxWx3 RGB.
- Compose two montages (use cv2 or PIL, your call — keep it minimal):
  - `contact_sheet.jpg` — a 4×4 grid of the **swapped** crops so the
    final output can be scanned quickly.
  - `compare.jpg` — a 3-column montage (one row per sample, in order):
    `original | mask | swapped`. With 16 samples that is 16×3 tiles.
    This is the file the visual inspection step looks at.
- Run the quantitative sanity checks below; if any fail, exit non-zero
  with a clear `click.echo` line saying which check failed.
- Exit zero on success.

**Quantitative sanity checks** (cheap, no vision model needed):

- **Centering:** for each non-fallback sample, compute the centroid of
  foreground pixels (`mask == 2`) in crop coordinates. The centroid
  must be within `[crop_size * 0.2, crop_size * 0.8]` on both axes.
  If it isn't, the window is not actually centered on the bee.
- **Coverage:** the foreground area must be between 2% and 60% of the
  crop. Outside this range the crop is either empty (no bee) or
  basically all-bee (window too tight).
- **Swap actually changes pixels:** for samples where `swapped=True`,
  the per-pixel mean absolute difference between `original` and
  `swapped` (over the *non-foreground* region of the mask) must be at
  least 5/255. If a swap looks identical to the original, the pool is
  not being used (or the same background is being picked every time).
- **Swap ratio is plausible:** if the background pool has >=2 entries
  and `swap_background_prob=0.5`, allow between 20% and 80% swapped
  samples. Outside that band, log a warning (do not fail) — the RNG
  seed is fixed, so the actual fraction is deterministic and small
  sample sizes can be misleading.
- **Non-black images:** every saved JPG must have a mean luminance
  above 5/255. A black crop means the cv2 read or the BGR→RGB
  conversion is broken.

**Verify:**

```bash
uv run python scripts/smoke_bee_dataset.py data/frames --num-samples 16 \
    --output samples/bees
```

Pass conditions:
- Exit code `0`.
- `samples/bees/contact_sheet.jpg` and `samples/bees/compare.jpg` exist
  and are non-empty.
- All four quantitative checks above pass.
- Roughly half the per-sample JPGs are flagged `_swapped` (within
  reason; if the background pool has 0–1 entries, expect 0 swaps and
  the swap-difference check should be skipped, not failed).
- No exceptions during iteration.

---

## Step 5 — Tests (lean, functionality only)

**Files:**
- `pyproject.toml` — add `pytest` to `[dependency-groups].dev`.
- `pyproject.toml` — add a `[tool.pytest.ini_options]` block with
  `pythonpath = ["src"]` so `import engine...` works from `tests/`
  without an editable install.
- `tests/__init__.py` (empty file).
- `tests/test_bee_crop.py` (new).
- `tests/test_dataset.py` (new).

**Philosophy:** undertest, not overtest. Test the **functional
contract** of each module, not every edge case. One assertion per
behavior. No fixtures file, no parametrized matrices, no coverage
hunting. If a test feels like it duplicates the smoke test, drop it.

**`tests/test_bee_crop.py`** — keep it under ~30 lines total. Tests:

- `find_bee_components`: build a synthetic `uint8` mask with two
  30×30 white squares, assert it returns two components with the
  expected areas and that the `min_area` filter drops small ones.
- `square_window`: bbox `(50, 50, 20, 20)` on a 200×200 image with
  `padding_factor=1.5`, assert the returned window's center is near
  `(60, 60)` and the side length is `max(20, 20) * 1.5 = 30` (give or
  take 1 px for rounding).
- `mask_to_classes`: input `[0, 127, 255]` -> output `[0, 1, 2]`
  `int64`.
- `build_swapped_crop`: feed a known 100×100 image, a mask with one
  foreground square, and a known 100×100 background. Assert:
  (a) foreground pixels in the output equal the original image at the
  same coordinates (this is the spatial-alignment check);
  (b) non-foreground pixels equal the new background at the same
  coordinates;
  (c) output is the requested `output_size`.

**`tests/test_dataset.py`** — keep it under ~40 lines total. Use
`tmp_path` (pytest's built-in fixture) to build a tiny on-disk fixture
with one video directory containing one frame JPG, one matching mask
PNG, and one `background.png`. Then:

- `len()` matches the number of (frame, mask) pairs on disk.
- One `__getitem__` call returns tensors with the expected shapes and
  dtypes (`image` is `float32 [3, crop_size, crop_size]` in `[0, 1]`;
  `mask` is `int64 [crop_size, crop_size]` in `{0, 1, 2}`; `bbox` is
  `float32 [4]`; `swapped` is `bool`; `video_id` / `frame_id` are
  `str`).
- With `swap_background_prob=1.0` and a non-empty pool, every item
  has `swapped=True`. (This is the only end-to-end dataset test; the
  smoke test already covers ratio and visual quality.)
- A `transforms` callable is invoked. Build a tiny `transforms` (e.g.
  a function that copies the dict and adds a `"marker": True` key) and
  pass it to the dataset. Assert the returned dict contains the
  marker. This is the only test that exercises the `transforms`
  argument; do not also test that a `torchvision.transforms.v2.Compose`
  works — that is the user's responsibility.

Do not test: extract_video_frames.py (the smoke test exercises the
script end-to-end), error paths, performance, the smoke-test
montages, or anything that duplicates the visual inspection step.

**Verify:**

```bash
uv run pytest
```

All tests pass, exit code `0`.

---

## Step 6 — Docs

**File:** `docs/SCRIPTS.md`

Add a section for `scripts/smoke_bee_dataset.py` and a short bullet
documenting the new `--save-background` flag on
`scripts/extract_video_frames.py` (mention the new `background.png`
output).

---

## Step 7 — Final gate (this is what tells the ralph loop to stop)

Run all of the following from the repo root and confirm clean output:

```bash
uv run ruff format . --check
uv run ruff check .
uv run basedpyright
uv run pytest
uv run python scripts/extract_video_frames.py data/videos_raw data/frames \
    --foreground-masks --save-background
uv run python scripts/smoke_bee_dataset.py data/frames \
    --num-samples 16 --output samples/bees
```

If any step fails, fix it. Do not declare success until every command
exits zero. If a video dataset is not available in the environment,
use a synthetic one (a short looped video created with
`np.zeros + cv2.VideoWriter`) to exercise the extract script, or skip
that one command and document why.

## Step 8 — Visual inspection (mandatory, do not skip)

The numerical checks above catch obvious breakage but cannot tell you
whether the bee is actually centered, the mask outlines a bee, or the
swapped background looks like a different hive scene. After the smoke
test exits 0, you **must look at the output images yourself** before
declaring success. You have native vision — use it: open the images
with the Read tool and look at them directly. No external vision API
calls.

Inspect:

1. `samples/bees/contact_sheet.jpg` — 4×4 of the final crops.
2. `samples/bees/compare.jpg` — per-sample `original | mask | swapped`.

For each, answer these questions (in your own scratch notes, not
committed — they drive your accept/reject decision):

- Is there a recognizable bee (or bee-like dark blob) in the center of
  most crops?
- Does the mask in `compare.jpg` look like it outlines that bee, with
  the white (foreground) region matching the dark blob?
- For swapped samples in `compare.jpg`, does the **background** region
  (everything outside the mask) clearly differ from the original —
  different texture, different colors, plausibly a different hive?
  In particular, is pixel `(0, 0)` of the swapped crop visibly
  different from pixel `(0, 0)` of the original crop (this is the
  spatial-alignment check from Step 3)?
- Are there obvious artifacts: a fully black tile, a tile that's
  identical to its neighbors, a bee that's clipped to one corner,
  vertical/horizontal seams from cv2 border padding?

If the answer to any of these is "no" or "I can't tell", do not
declare success. Diagnose: is the mask wrong, is the crop mis-centered,
is the background pool empty in disguise, is the swap leaking the
foreground, etc. Fix the underlying code (not the smoke test) and
re-run from Step 4.

Only declare success once you have read both images, the answers above
are "yes", and the smoke test still exits 0.

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
tree or a `pi` crash, in which case you should NOT have created `DONE`
— a human will pick it up.

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
- Do not squash at the end; leave the history clean for review.

---

## Anti-goals (do NOT do these)

- No clever abstractions. No base classes for "BeePreprocessor", no plugin
  systems, no factory functions.
- No second pass over the video just to learn a better background.
- No new dependencies for their own sake. New runtime or dev deps are
  fine when they meaningfully simplify the code; prefer stdlib + existing
  `pyproject.toml` deps otherwise. If you add one, mention it in the
  commit body.
- No edge-blending / feathering / soft masks in the swap. Naive cut-paste is
  the spec.
- No torchvision transforms inside the dataset body. The dataset returns
  raw tensors; the user composes `v2` transforms and passes them via the
  `transforms` argument.
- No edits to existing scripts' behavior beyond adding the
  `--save-background` flag.
