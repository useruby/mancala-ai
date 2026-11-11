# Varied-opponent P coverage pilot: preparation

This preparation adds an experiment-local game worker, archive verifier, tests,
and a no-games launch preflight. It does not freeze or generate the real pilot,
train a model, run a strength screen, access the sealed suite, or change serving.
The hash-pinned #548/#549 implementation remains unchanged.

## Published artifact and pool rationale

The three runtime bundles are read from
`seed556-exploratory-arena/screen-0005/runtime-artifacts/{P-E4,T512,initializer}`.
The worker pins and checks every metadata, model, search-policy, and weights file
against the twelve SHA-256 values recorded in #556's `registration.json` before
loading evaluators or starting search.

Serving P is P-E4 policy and trunk composed with initializer value and trunk.
Treatment opponent `initializer` is the complete initializer; treatment opponent
`T512` uses T512 policy and trunk composed with the same initializer value and
trunk. Control uses P on both seats. The pool rationale is the public archived
decision-divergence check in #556 screen-0005: for indexed opening 0000, the
archived T512 and P first actions are 5 and 3, respectively. This is evidence of
different decisions on a public position, not a strength claim.

## Worker and verifier contract

Each game spec binds `opening_id`, `pair_id`, seed, arm, designated P seat,
opponent, and `search_workers=1`. Each pair contains both designated-seat
assignments. A treatment pair has one frozen opponent label across the seat
swap. Assignment and all artifact hashes are checked for the entire launch set
before evaluator construction and before any search.

The artifact runtime-search sidecar semantics are frozen separately from the
host runtime assets. The sidecars agree on the PUCT/exact-root hybrid, threshold
16, final-score-margin objective, network-prior-then-lowest-index tie rule, and
disabled exact-leaf solving. The preflight checks the probe/tablebase bytes at
the current worktree's materialized runtime paths against the hashes in all
three sidecars; it does not rewrite the published artifacts.

The worker asks Kalah for its actual `current_player` before every move, so an
extra turn is routed to the same seat's evaluator. It records every move and
pre-move state in the full replay trajectory. Only designated-P turns become
rows, including in P-vs-P control. Target generation is fixed for both arms:
128 PUCT simulations, `c_puct=1.25`, temperature 1.0 for plies 0–8 then 0.1,
Dirichlet alpha/epsilon 0.3/0.25 for action sampling in those early plies,
denoised default visit-count policy targets, selected-one-hot exact-root targets
at the registered threshold of 16, and default terminal-outcome value targets.
Exact-leaf solving is disabled by the artifact runtime policy.

The verifier replays every move from the archived opening, checks player/extra-
turn transitions, rebuilds the expected P-seat row IDs, and rejects missing,
duplicate, opponent-seat, wrong-state, wrong-ply, illegal-move-mass, and
wrong-terminal-target rows. It also checks the target-setting declaration on
every row and game against the common frozen settings. A nonterminal capped or
incomplete game retains its trajectory and is classified separately; none of
its rows are accepted.

## Parallelism and CPU accounting

Pair jobs are the 18-way concurrency unit, dispatched to 18 isolated worker
processes. Each process handles one pair and exits, so model initialization is
attributable to that pair. The scope binds CPU quota 1800%, CPU weight 100,
memory maximum 36 GiB, CPUs 14–31, and nice level 19. Each game search remains
single-worker; PyTorch intra-op and inter-op thread counts are both set to one
in each pair process. The worker records process CPU time per game and per pair;
the per-pair amount includes model initialization, native tablebase setup, and
both seat-swapped games, without mixing concurrent pairs. Sum pair CPU time
separately by arm; CPU-hours are `sum(pair_cpu_time_ns) / 3_600_000_000_000`.
Wall time is not used as the primary-efficiency denominator. Capped and
incomplete games still incur CPU cost and remain in cost accounting.

## Preparation checks

Focused tests (scripted game fixtures only; no production-model games):

```bash
/home/alex/azlite-venv/bin/python -m pytest -q \
  ml/alphazero_lite/test_varied_opponent_pilot.py
/home/alex/azlite-venv/bin/ruff check \
  ml/alphazero_lite/varied_opponent_pilot.py \
  ml/alphazero_lite/test_varied_opponent_pilot.py
```

The read-only preflight also authenticates all twelve files and starts a trivial
`systemd-run` command under the configured resource scope; it launches no game.
For a planned frozen spec JSON, invoke:

```bash
/home/alex/azlite-venv/bin/python -m ml.alphazero_lite.varied_opponent_pilot \
  --preflight --specs /path/to/frozen-game-specs.json \
  --artifacts docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts \
  --tablebase /path/to/materialized/kalah_v1_21.kvtb
```

The tablebase path must point to the materialized 709,635,072-byte object whose
SHA-256 is bound by #556; the tracked Git-LFS pointer itself is not used as a
solver table. The native probe is resolved from the worktree and independently
hash-checked.

The actual pilot freeze must separately record its public opening manifest,
complete game assignments, effective search policy, runtime limits, coverage
gain criterion, stopping rule, and opening/pair-level accounting before any
real game is launched.
