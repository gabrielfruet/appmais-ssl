#!/usr/bin/env bash
# Ralph loop around pi. Each iteration is a fresh `pi -p` process = clean context.
#
# Usage: ./ralph.sh <prompt_file> [max_iters]
#   prompt_file  path to the markdown prompt to feed pi on every iteration
#   max_iters    max number of iterations before giving up (default 30)
#
# The loop stops early when a `DONE` file appears at the repo root, or when
# `pytest -q` exits non-zero, or when `pi` itself crashes.
set -uo pipefail

PROMPT_FILE="${1:-}"
MAX_ITERS="${2:-30}"

if [[ -z "$PROMPT_FILE" || ! -f "$PROMPT_FILE" ]]; then
    echo "Error: prompt file '$PROMPT_FILE' does not exist."
    echo "Usage: $0 <prompt_file> [max_iters]"
    exit 1
fi

# Anchor at the repo root (parent of the prompts/ dir) so DONE / pytest /
# relative paths in the prompt resolve consistently regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

last_i=0
for ((i = 1; i <= MAX_ITERS; i++)); do
  last_i="$i"

  if [[ -f "$REPO_ROOT/DONE" ]]; then
    echo "DONE file present at $REPO_ROOT/DONE. Stopping."
    break
  fi

  echo "===== Ralph iteration ${i}/${MAX_ITERS} ====="

  # Fresh context every iteration: --no-session => nothing persisted/continued.
  # pi default tools (read/write/edit/bash) are what the build needs.
  if ! cat "$PROMPT_FILE" | pi -p --no-session; then
    echo "pi exited non-zero on iteration ${i}. Pausing for human review."
    break
  fi

  # External safety gate: never proceed on a red tree, even if the agent slipped.
  if ! uv run pytest -q; then
    echo "Tests RED after iteration ${i}. Pausing for human review."
    break
  fi
done

echo "Loop ended at iteration ${last_i}."
