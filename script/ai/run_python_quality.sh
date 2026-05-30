#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing python executable: $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"
"$PYTHON_BIN" -m pre_commit run --all-files
