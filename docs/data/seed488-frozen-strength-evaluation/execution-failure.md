# Seed488 execution attempt — stopped before first game

**Status: blocked; no evaluation games were completed.** The frozen runner was
launched once with the registered command. Opening 0 reached arena preflight, but
arena rejected the runner-provided search-options override before starting either
game. The registered interruption policy forbids silently rerunning an
uncommitted opening; execution stopped with opening 0 uncommitted. No frozen
source, registration, suite, receipt, report, or outcome ledger was edited.

## Commands and environment

Readiness and focused tests were run in the frozen `.venv-azlite` environment:

```text
./.venv-azlite/bin/python -m ml.alphazero_lite.verify_seed488_strength_evaluation
./.venv-azlite/bin/python -m pytest ml/alphazero_lite/test_seed488_strength_evaluation.py -q
./.venv-azlite/bin/python -m ml.alphazero_lite.seed488_strength_evaluation --execute
```

The readiness result was `prepared_not_executed`, with 512 openings. The focused
test suite passed: 21 tests. Runtime was Python 3.14.7 and NumPy 2.5.2.

The runner's final arena command for opening 0 used:

```text
./.venv-azlite/bin/python -m ml.alphazero_lite.arena --challenger docs/data/seed488-frozen-strength-evaluation/runtime-artifacts/F --current docs/data/seed488-frozen-strength-evaluation/runtime-artifacts/initializer --games 2 --games-per-opening 2 --workers 1 --challenger-simulations 1200 --current-simulations 1200 --c-puct 1.25 --seed-contract azlite_eval_seed_v2 --challenger-search-options-json '{"fpu_mode":"zero","normalize_values":false,"random_opening_plies":0,"reuse_subtree":false,"root_policy_mode":"deterministic","root_prior_transform":null,"root_temperature":0.0,"tactical_root_bias":0.0}' --current-search-options-json '{"fpu_mode":"zero","normalize_values":false,"random_opening_plies":0,"reuse_subtree":false,"root_policy_mode":"deterministic","root_prior_transform":null,"root_temperature":0.0,"tactical_root_bias":0.0}' --max-moves 200 --seed 488 --opening-prefixes-jsonl docs/data/seed488-frozen-strength-evaluation/resume/opening-0000.jsonl --suite-sha256 4dee384ce68692d75ce8b5e0d1ca67cafefc7f87330aeb9005938f6e55a13c7b --game-jsonl docs/data/seed488-frozen-strength-evaluation/resume/games-0000.jsonl --out docs/data/seed488-frozen-strength-evaluation/resume/report-0000.json
```

## Preserved first-batch evidence

- Parent suite SHA-256: `53841cad6c5d34355cd7d6533a0fa435f4164260fd40a2da760a3d78c784cff8`.
- Parent opening index and local opening index: `0` and `0`.
- Parent state hash: `51ad78828a2b5ed64945ff68d37cc054f9e31f75b64a13fa5bf447d14326da26`.
- Exact one-entry batch SHA-256: `4dee384ce68692d75ce8b5e0d1ca67cafefc7f87330aeb9005938f6e55a13c7b`.
- Base seed: `488` (`488 + parent_opening_index`).
- Exported runtime artifacts and their manifest remain in `runtime-artifacts/` and
  `runtime-artifacts.json`; a read-only check against both frozen endpoint states
  passed. Manifest SHA-256:
  `a3bbca1f701247957d0d7d1aa3fab0cfd49de5d70fd17bfaa56edd6cd24e1ab2`.
- The complete runner stdout/stderr is retained in `execution.log`.
- There is no `games-0000.jsonl`, `report-0000.json`, committed outcome ledger,
  or pending atomic commit. Thus completed games: `0`; capped games: `0`; paired
  score and bootstrap intervals: not estimable; advance: no decision.

## Exact arena failure

```text
ValueError: unsupported search-options override fields: ['random_opening_plies']
```

Arena terminated during command-line option parsing, before a game began. The
current evidence includes only the deterministic exported artifacts and the
one-entry batch written before arena launch. This is a concrete runtime-contract
failure. Do not resume or rerun this uncommitted opening under the current
registration.

The unresolved seed461 continuation-overlap limitation remains: non-overlap
cannot be certified because authenticated state identities for the 1,129,643
receipt-matched continuation actions are unavailable. This is an exploratory
evaluation attempt, not fresh confirmation; no promotion, tuning, retraining, or
production replacement is authorized.
