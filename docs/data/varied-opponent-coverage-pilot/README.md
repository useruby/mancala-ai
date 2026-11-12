# Varied-opponent P coverage pilot freeze

This is the freeze-only preparation for the 128-pair, 256-game coverage pilot.
No production games are launched by this publication. `freeze.json` binds the
ordered public openings, complete assignments, frozen opponent split, targets,
artifacts, runtime assets, execution code, resource scope, metric, and decision
rule. `game-assignments.json` is the exact worker input; `openings.json` is the
selected ordered opening manifest.

## Frozen design

The first 64 rows in public seed556 screen-0005 `openings.jsonl` are retained in
source order. Seed 578064 shuffles the opening indexes and assigns the first 32
to initializer and the remaining 32 to T512. Each opening's single seed and
prefix are shared across its four games: two seat-swapped P-vs-P controls and
two seat-swapped treatment games. Treatment opponent remains fixed across that
seat swap. Only designated-P turns are retained.

Primary yield is the number of distinct accepted designated-P positions with
more than 32 active stones divided by arm CPU-hours. Identities are counted
separately using canonical state and production `kalah_v3` float32 identity.
CPU is the sum of pair CPU, including model setup and reaped native-probe CPU;
capped/incomplete games contribute cost and zero accepted rows. A larger
generation run is proposed only when treatment yield is at least 1.15x control
under both identity schemes. Zero control yield or irreconcilable accounting is
inconclusive.

## Resume and attempt evidence

Pair commits are one atomically replaced JSON file containing the exact pair
assignment, both replay trajectories, retained P rows, and reconciled pair
process/native-child CPU. Resume validates all existing commits before work is
dispatched and skips committed pair IDs. Each dispatched pair also has a
separate attempt record. A surviving `started` attempt without a verified pair
commit is treated as unreconciled CPU and stops resume; it is not silently
replayed. Such a pilot is inconclusive unless its cost can be reconciled.

## Commands

The no-games preflight command, result, and freeze digest are recorded in
`preflight.json`. The synthetic checkpoint/resume suite exercises pair commit,
skip, idempotence, and conflicting evidence. After this freeze merges, the
registered launch command is:

```bash
export AZLITE_KALAH_V1_TABLEBASE=/home/alex/Mancala/ai/model-artifact/runtime/kalah_v1_21.kvtb
systemd-run --user --scope --quiet \
  --property=CPUQuota=1800% --property=CPUWeight=100 \
  --property=MemoryMax=38654705664 --property=AllowedCPUs=14-31 --nice=19 -- \
  /home/alex/azlite-venv/bin/python -m ml.alphazero_lite.run_varied_opponent_pilot \
  --freeze docs/data/varied-opponent-coverage-pilot/freeze.json \
  --checkpoint-dir docs/data/varied-opponent-coverage-pilot/pairs
```

Resume uses the same command plus `--resume`; committed pairs are validated and
skipped, and interrupted attempts remain preserved in `attempts/`.

No training, strength screening, sealed-suite access, or serving change is part
of this pilot. #483 remains closed.
