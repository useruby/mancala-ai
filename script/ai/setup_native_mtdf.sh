#!/usr/bin/env bash
set -euo pipefail
root=$(git rev-parse --show-toplevel); dir=${1:-"$root/.native-mtdf/girving-kalah"}; sha=0a0c00702908f4ba865f2a1baa548ac0e3724950
if [[ ! -d "$dir/.git" ]]; then mkdir -p "$(dirname "$dir")"; git clone https://github.com/girving/kalah.git "$dir"; fi
git -C "$dir" fetch --depth=1 origin "$sha"; git -C "$dir" checkout --detach "$sha"
if git -C "$dir" diff --quiet; then
  git -C "$dir" apply --ignore-space-change "$root/third_party/girving-kalah/compatibility.patch"
fi
cp "$root/third_party/girving-kalah/native_probe.c" "$root/third_party/girving-kalah/native_hash_tests.c" "$root/third_party/girving-kalah/canonical_kvtb.c" "$root/third_party/girving-kalah/canonical_kvtb.h" "$dir"
flags='-O3 -funroll-loops -Winline -Wall -std=gnu11 -fgnu89-inline -DDISABLE_FUTILITY -DCANONICAL_TABLEBASE'
if [[ ${SANITIZE:-0} == 1 ]]; then flags='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -std=gnu11 -fgnu89-inline'; fi
if [[ ${NO_TT:-0} == 1 ]]; then flags="$flags -DDISABLE_TT"; fi
gcc $flags -DNOCILK -x c -c -o "$dir/crunch-s.o" "$dir/crunch.cilk"; gcc $flags -DNOCILK -x c -c -o "$dir/hash-s.o" "$dir/hash.cilk"; gcc $flags -c -o "$dir/rules.o" "$dir/rules.c"; gcc $flags -c -o "$dir/endgame.o" "$dir/endgame.c"
gcc $flags -c -o "$dir/canonical_kvtb.o" "$dir/canonical_kvtb.c"; gcc $flags -DNOCILK -o "$dir/native_probe" "$dir/native_probe.c" "$dir/canonical_kvtb.o" "$dir/crunch-s.o" "$dir/hash-s.o" "$dir/rules.o" "$dir/endgame.o"; gcc $flags -DNOCILK -o "$dir/native_hash_tests" "$dir/native_hash_tests.c"; "$dir/native_hash_tests"; printf '%s\n' "$dir/native_probe"
