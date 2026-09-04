#!/usr/bin/env bash
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
mkdir -p "$root/.tmp/kalah_v1_tablebase_build"
g++ -std=c++17 -O3 -Wall -Wextra -Werror "$root/native/kalah_v1_tablebase/kalah_v1_tablebase.cc" -o "$root/.tmp/kalah_v1_tablebase_build/kalah_v1_tablebase"
printf '%s\n' "$root/.tmp/kalah_v1_tablebase_build/kalah_v1_tablebase"
