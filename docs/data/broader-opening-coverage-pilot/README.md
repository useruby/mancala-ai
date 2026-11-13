# Broader-opening P-vs-P coverage pilot freeze

This is a freeze-only registration for a **62-matchable-opening** pilot. It
registers 124 seat-swapped P-vs-P pairs (248 games), with one seed per matched
opening shared across control and treatment. Both player seats use P in both
arms; only designated-P rows are retained. No games are launched by preparation,
preflight, or synthetic tests.

The pilot's estimand is explicitly limited to these 62 matchable openings. The
two unmatched #580 controls are `opening-0002` (10-ply prefix, 33 active stones,
side 0) and `opening-0022` (11-ply prefix, 24 active stones, side 1). The latter
could never contribute a position above 32 active stones; the former could.
Their exclusion limits interpretation to the frozen 62-opening subset.

## Selection and identity checks

Control is the ordered #580 `openings.json` manifest. The wider pool is the
public 1,024-row seed556 screen-0005 `openings.jsonl`. Visit controls in manifest
order. Choose the eligible pool opening with the lowest `opening_index` (then
canonical identity as a tie-break) that exactly matches prefix length, active
stones, and `current_player`. Reject candidates whose canonical **or** production
`kalah_v3` float32 identity is already present among controls, and consume each
treatment opening only once under both schemes. No criteria are relaxed.

The frozen matching audit records 62 matches and 2 unmatched controls. The
selected control and treatment manifests publish both identities per opening.
The selection audit verifies 64 unique control identities in each scheme, 62
unique treatment identities in each scheme, no cross-arm start identity
collisions, and exact signature equality for each matched pair.

## Frozen run and metric

The pilot-specific harness uses the unchanged shared #580/#582 modules and the
validated v2 warmed-probe lifecycle. Its P-vs-P adapter and single-worker
scheduler live in this new harness, leaving both shared frozen runners
byte-for-byte intact. Each pair owns one warmed exact-root probe; execution is
bound to **pair concurrency 1** and one search worker per game. Search, target,
runtime assets, CPU quota, CPU weight, memory cap, CPU set, and nice level are
identical between arms and fixed in `freeze.json`. Assignment seeds are matched
between the two arms for each opening. Attempts and CPU, including reaped native
child CPU, must reconcile before interpretation; capped or incomplete games
contribute cost and no accepted rows.

For each identity scheme independently, remove the union of all supplied
control and treatment opening positions from retained designated-P rows before
counting distinct states with **more than 32 active stones**. Divide by arm CPU
hours. Also report overlap and novelty against the published #556–#560
`identity-inventory.json`; that inventory is explicitly scoped to #556–#560 and
is not a complete historical exclusion census.

Propose a larger run only if treatment yield is at least **1.15× control under
both canonical and production float32 identities**, with all attempts and CPU
accounted for. The gate applies only to the 62 matchable openings. A zero
control yield or unreconciled accounting is inconclusive. #580's decision is
unchanged; there is no training, serving change, sealed-suite access, or reopening
of #483.

## Asset and commands

The tablebase copy used for preflight was copied only after the source bytes
matched the sidecar-registered SHA-256
`f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7`; the copy
was re-hashed to the same value. Keep the copy at
`.tmp/broader-opening-coverage/kalah_v1_21.kvtb` for eventual execution. It is
not committed. If recreating it after merge, first run `sha256sum`
`/home/alex/Mancala/ai/model-artifact/runtime/kalah_v1_21.kvtb` and copy only if
the result is the registered digest above; then verify the copy's digest again.
Native probe, tablebase, model components, runner sources,
assignments, and both opening manifests are hash-bound by the freeze.

Before creating run identity, checkpoint, or attempt evidence, launch and resume
verify the live cgroup CPU quota, CPU weight, memory maximum, effective CPU set,
process affinity, and nice level against the frozen limits. Direct invocation
outside the registered systemd/taskset scope fails closed. Analysis verifies the
exact published novelty-inventory bytes against the freeze digest before reading
identities; mutation aborts analysis.

The host omits both cgroup cpuset files. The freeze explicitly permits the
`taskset_affinity_fallback` only in that case; the harness records that source
and requires the current process affinity to equal CPUs 14–31. The no-games
preflight ran inside the exact registered systemd/taskset scope and recorded
CPU quota 1800%, CPU weight 100, 36 GiB memory cap, CPUs 14–31, nice 19, and the
taskset affinity fallback in `preflight.json`.

The recorded no-games preflight is `preflight.json` and reports 124 pairs, 248
games, concurrency 1, and `games_launched: 0`. `synthetic-checks.json` records
the passing synthetic test command and outcomes. Synthetic tests cover deterministic
matching, both identity schemes, supplied-opening exclusion in both arms, P
against P in treatment, rejection of mutated treatment-row opponent metadata,
preservation of prior runner hashes, resource-scope rejection before evidence
creation, one-buffer inventory hash/parse and mutation rejection, taskset
fallback verification, and v2 checkpoint/resume skipping a verified pair.

After the freeze merges, launch with the digest independently reviewed from
`freeze.json`:

```bash
export PILOT_FREEZE_SHA256=3f2ccca4fae605f826b4fb00e0adb6f88c8c87ad8ca1373152aea051133f75b5
export AZLITE_KALAH_V1_TABLEBASE=.tmp/broader-opening-coverage/kalah_v1_21.kvtb
systemd-run --user --scope --quiet \
  --property=CPUQuota=1800% --property=CPUWeight=100 \
  --property=MemoryMax=38654705664 --property=AllowedCPUs=14-31 --nice=19 -- \
  taskset -c 14-31 /home/alex/azlite-venv/bin/python \
  -m ml.alphazero_lite.broader_opening_coverage_pilot run \
  --freeze docs/data/broader-opening-coverage-pilot/freeze.json \
  --expected-freeze-sha256 "$PILOT_FREEZE_SHA256" \
  --checkpoint-dir docs/data/broader-opening-coverage-pilot/pairs
```

Resume uses the same command and adds `--resume`. Analyze only after all pairs
and attempts reconcile:

```bash
/home/alex/azlite-venv/bin/python \
  -m ml.alphazero_lite.broader_opening_coverage_pilot analyze \
  --freeze docs/data/broader-opening-coverage-pilot/freeze.json \
  --expected-freeze-sha256 "$PILOT_FREEZE_SHA256" \
  --checkpoint-dir docs/data/broader-opening-coverage-pilot/pairs \
  --attempt-dir docs/data/broader-opening-coverage-pilot/attempts
```
