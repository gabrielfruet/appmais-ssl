# Agent Notes

- Keep scripts KISS: simple, readable, and static is better than overly configurable.
- Put small one-off utilities in `scripts/`.
- Document scripts briefly in `docs/SCRIPTS.md` when adding or changing them.
- Support CUDA, MPS, and CPU for local model scripts when practical; use automatic device selection unless there is a good reason not to.
- Before finishing Python changes, run:
  - `uv run ruff format . --check`
  - `uv run ruff check .`
  - `uv run basedpyright`

## Git

You should do regular commits with meaningful, but concise, commit messages.
