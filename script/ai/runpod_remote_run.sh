#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${1:-$HOME/runpod-experiment}
MANIFEST_PATH=${2:-$ROOT_DIR/manifest.json}
ENV_PATH=${3:-$ROOT_DIR/.runpod_env.sh}

if [ ! -f "$MANIFEST_PATH" ]; then
  printf 'missing manifest: %s\n' "$MANIFEST_PATH" >&2
  exit 1
fi

if [ -f "$ENV_PATH" ]; then
  source "$ENV_PATH"
fi

COMMAND=$(python3 - <<'PY' "$MANIFEST_PATH"
import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as fh:
    manifest = json.load(fh)

print(manifest['command'])
PY
)

RESULTS_PATH=$(python3 - <<'PY' "$MANIFEST_PATH"
import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as fh:
    manifest = json.load(fh)

print(manifest.get('results_path', ''))
PY
)

cd "$ROOT_DIR"

if [ -n "$RESULTS_PATH" ]; then
  mkdir -p "$ROOT_DIR/$RESULTS_PATH"
  LOG_PATH="$ROOT_DIR/$RESULTS_PATH/remote_run.log"
  {
    printf 'command=%s\n' "$COMMAND"
    printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$LOG_PATH"
  set +e
  bash -euo pipefail -c "$COMMAND" >> "$LOG_PATH" 2>&1
  STATUS=$?
  {
    printf 'exit_status=%s\n' "$STATUS"
    printf 'finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >> "$LOG_PATH"
  set -e
  exit "$STATUS"
else
  bash -c "$COMMAND"
fi
