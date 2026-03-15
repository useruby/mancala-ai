#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${1:-$HOME/runpod-experiment}
MANIFEST_PATH=${2:-$ROOT_DIR/manifest.json}
ENV_PATH="$ROOT_DIR/.runpod_env.sh"

mkdir -p "$ROOT_DIR"
cd "$ROOT_DIR"

INSTALL_REQUIREMENTS=false
INSTALL_RUBY=false
RUBY_VERSION=4.0.0
if [ -f "$MANIFEST_PATH" ]; then
  eval "$(python3 - <<'PY' "$MANIFEST_PATH"
import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as fh:
    manifest = json.load(fh)

print(f"INSTALL_REQUIREMENTS={'true' if manifest.get('install_python_requirements') else 'false'}")
print(f"INSTALL_RUBY={'true' if manifest.get('install_ruby_dependencies') else 'false'}")
print(f"RUBY_VERSION={manifest.get('ruby_version', '4.0.0')}")
PY
)"
fi

python3 -m venv "$ROOT_DIR/.venv"

if [ "$INSTALL_RUBY" = "true" ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y build-essential git curl ca-certificates pkg-config rustc libssl-dev zlib1g-dev libyaml-dev libreadline-dev libffi-dev libgmp-dev libsqlite3-dev sqlite3

  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v mise >/dev/null 2>&1; then
    curl -fsSL https://mise.run | sh
  fi

  export PATH="$HOME/.local/bin:$PATH"
  mise install "ruby@${RUBY_VERSION}"
  mise use -g "ruby@${RUBY_VERSION}"

  cat > "$ENV_PATH" <<EOF
export PATH="$HOME/.local/bin:$HOME/.local/share/mise/shims:$ROOT_DIR/.venv/bin:$PATH"
export BUNDLE_GEMFILE="$ROOT_DIR/Gemfile"
EOF
  source "$ENV_PATH"

  gem install bundler
  bundle config set --local path vendor/bundle
  bundle install
else
  cat > "$ENV_PATH" <<EOF
export PATH="$ROOT_DIR/.venv/bin:$PATH"
EOF
fi

source "$ENV_PATH"
source "$ROOT_DIR/.venv/bin/activate"

if [ "$INSTALL_REQUIREMENTS" = "true" ] && [ -f "$ROOT_DIR/ml/alphazero_lite/requirements.txt" ]; then
  pip install --upgrade pip
  pip install -r "$ROOT_DIR/ml/alphazero_lite/requirements.txt"
fi

printf 'bootstrap_ready root=%s\n' "$ROOT_DIR"
