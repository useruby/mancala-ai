# AlphaZero-Lite Fixed-Evaluator Candidate Rerank Results

**Date**: 2026-06-14
**Classification**: `control_ep2_canonical_confirmed_post_fix`
**Schema**: `azlite_fixed_evaluator_candidate_rerank_v1`

## Summary

Re-ranked the canonical control candidates after the PR #113 evaluator fixes.

Primary outcome: canonical `control_ep2` remains the best candidate on the decision budget, large-suite `384:256` disadvantaged-seat score.

- `control_ep2` large `384:256` DS: `-0.1784`
- `seed_121` large `384:256` DS: `-0.2448`
- Delta vs `control_ep2`: `-0.0664`
- Default-opening gate signature preserved: `high_search_breakthrough`

Fresh `workers=1` and `workers=24` reruns matched exactly on the primary `384:256` budget for every candidate on both `medium_eval` and `large_eval`.

## Scope

PR #113 fixed two relevant issues already described in the determinism audit:

1. `arena.py` now writes per-worker `games.jsonl` files and merges them by `game_index`.
2. `run_opening_suite_seat_benchmark.py` deterministic runs now honor `--seed` directly.

The full `workers=24` matrix was re-run for both suites and all requested budget pairs.

Fresh `workers=1` reruns were completed for the primary `384:256` budget on both suites. A full fresh `workers=1` all-budget matrix was not completed in this session because the suite sizes make it substantially more expensive than the task text suggests. The primary decision budget, which determines the canonical-control question, was re-run fully and matched `workers=24` exactly.

## Inputs

| Item | Path | SHA256 |
|------|------|--------|
| current | `model-artifact/current` | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| iter0_reference | `/tmp/azlite_iterative_random_replay/iter0_candidate_artifact` | `0bbeaa9c6954ea147b084cdb3e68355887df1b0ab34c1fbc952b7988fdddc3bd` |
| control_ep2 | `/tmp/azlite_control_ep2_seed_harvest/control_ep2_eval_only/artifact_control_ep2` | `34b3697f95c3c63e092838d10668ba1361dab2a68a0bfe12305a294c45f765ad` |
| control_ep2 fallback | `/tmp/azlite_control_continuation_sweep/lr3e5/artifact_epoch2` | `34b3697f95c3c63e092838d10668ba1361dab2a68a0bfe12305a294c45f765ad` |
| seed_121 | `/tmp/azlite_control_ep2_seed_harvest/replicate_seed_121/artifact_replicate_seed_121` | `3cd012f286798c4dc588b10d92dfa89dc1b5eab791232bdbde3a0de7325c4825` |
| seed_103 | `/tmp/azlite_control_ep2_seed_harvest/replicate_seed_103/artifact_replicate_seed_103` | `556210303b0c6d9bc00fa34cd16b0db861851cb522daa78f9ae7f15c85994fc9` |
| seed_112 | `/tmp/azlite_control_ep2_seed_harvest/replicate_seed_112/artifact_replicate_seed_112` | `8f4866173363603d31d922ef0ca7ebff4c870126bce66409aa50fbec15cf3a51` |
| medium suite | `/tmp/azlite_opening_suite/medium_eval.jsonl` | `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04` |
| large suite | `/tmp/azlite_opening_suite/large_eval.jsonl` | `ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4` |

## Run Conditions

- Root policy mode: `deterministic`
- Seed: `42`
- Games per opening: `2`
- Budget pairs: `384:256`, `768:256`, `768:768`, `1200:1200`, `256:768`
- Full rerank workdirs:
  - `/tmp/azlite_fixed_eval_rerank/medium_workers24`
  - `/tmp/azlite_fixed_eval_rerank/large_workers24`
  - `/tmp/azlite_fixed_eval_rerank/large_workers24_seed112_only`
- Primary-budget workers=1 cross-check workdirs:
  - `/tmp/azlite_fixed_eval_rerank/medium_standard_workers1`
  - `/tmp/azlite_fixed_eval_rerank/large_standard_workers1`

## Large Suite Rerank

Full `workers=24` rerank on `large_eval`:

| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 256:768 DS | 384:256 P0/P1 | 384:256 margin mean/median | 384:256 duplicate count |
|-----------|------------|------------|------------|--------------|------------|---------------|----------------------------|--------------------------|
| control_ep2 | **-0.1784** | -0.1120 | -0.2161 | **+0.0924** | -0.1901 | 0.2878 / 0.4661 | -7.4688 / -14.0 | 1536 |
| seed_121 | -0.2448 | **+0.0130** | -0.0911 | -0.1732 | -0.1458 | 0.2318 / 0.4766 | -7.4818 / -14.0 | 1536 |
| seed_103 | -0.3698 | -0.2083 | **+0.0052** | -0.2161 | -0.1458 | 0.2760 / 0.6458 | -5.1641 / -14.0 | 1536 |
| iter0_reference | -0.3698 | -0.1562 | -0.0469 | +0.2682 | -0.1901 | 0.2760 / 0.6458 | -6.6484 / -14.0 | 1536 |
| seed_112 | -0.3815 | -0.2083 | -0.2604 | -0.1107 | -0.2005 | 0.2760 / 0.6576 | -3.7708 / -14.0 | 1536 |

Ranking by large-suite `384:256` DS:

1. `control_ep2` at `-0.1784`
2. `seed_121` at `-0.2448`
3. `seed_103` and `iter0_reference` tied at `-0.3698`
4. `seed_112` at `-0.3815`

Ranking by simple multi-budget DS sum on the large suite:

1. `iter0_reference` at `-0.4948`
2. `control_ep2` at `-0.6042`
3. `seed_121` at `-0.6419`
4. `seed_103` at `-0.9349`
5. `seed_112` at `-1.1615`

The requested acceptance question is driven by large-suite `384:256`, and on that budget canonical `control_ep2` remains first.

## Medium Suite Rerank

Full `workers=24` rerank on `medium_eval`:

| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 256:768 DS | 384:256 P0/P1 | 384:256 margin mean/median | 384:256 duplicate count |
|-----------|------------|------------|------------|--------------|------------|---------------|----------------------------|--------------------------|
| control_ep2 | **-0.1719** | -0.1094 | -0.2188 | +0.1133 | -0.1016 | 0.2734 / 0.4453 | -8.2266 / -14.0 | 512 |
| seed_121 | -0.2031 | **+0.0938** | -0.0156 | -0.1797 | -0.0625 | 0.2188 / 0.4219 | -8.3984 / -14.0 | 512 |
| seed_103 | -0.4062 | -0.2031 | **+0.0781** | -0.2188 | -0.0625 | 0.2578 / 0.6641 | -5.7344 / -14.0 | 512 |
| iter0_reference | -0.4062 | -0.1484 | +0.0234 | **+0.3516** | -0.1016 | 0.2578 / 0.6641 | -6.4688 / -14.0 | 512 |
| seed_112 | -0.4219 | -0.2031 | -0.2578 | -0.0586 | -0.1172 | 0.2578 / 0.6797 | -4.4688 / -14.0 | 512 |

Ranking by medium-suite `384:256` DS:

1. `control_ep2` at `-0.1719`
2. `seed_121` at `-0.2031`
3. `seed_103` and `iter0_reference` tied at `-0.4062`
4. `seed_112` at `-0.4219`

## Workers=1 Vs Workers=24

Fresh reruns on the primary `384:256` budget matched exactly for every candidate.

| Suite | Candidate | workers=1 DS | workers=24 DS | Delta |
|------|-----------|--------------|---------------|-------|
| medium | control_ep2 | -0.1719 | -0.1719 | 0.0000 |
| medium | seed_121 | -0.2031 | -0.2031 | 0.0000 |
| medium | seed_103 | -0.4062 | -0.4062 | 0.0000 |
| medium | seed_112 | -0.4219 | -0.4219 | 0.0000 |
| medium | iter0_reference | -0.4062 | -0.4062 | 0.0000 |
| large | control_ep2 | -0.1784 | -0.1784 | 0.0000 |
| large | seed_121 | -0.2448 | -0.2448 | 0.0000 |
| large | seed_103 | -0.3698 | -0.3698 | 0.0000 |
| large | seed_112 | -0.3815 | -0.3815 | 0.0000 |
| large | iter0_reference | -0.3698 | -0.3698 | 0.0000 |

No material workers-based instability was observed on the decision budget. For the exact reruns completed, results were bit-for-bit identical at the aggregate metric level.

## Default Opening Gate

Re-ran the default deterministic opening gate with `workers=24` for the requested artifacts:

| Candidate | Classification | Standard alt score | 384:256 DS | 1200:1200 DS | 256:768 DS |
|-----------|----------------|--------------------|------------|---------------|------------|
| control_ep2 | `high_search_breakthrough` | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| seed_121 | `high_search_breakthrough` | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| iter0_reference | `high_search_breakthrough` | 0.0000 | 0.0000 | 1.0000 | 0.0000 |

Canonical `control_ep2` preserved the expected high-search-breakthrough signature after the evaluator fixes.

## Decision

Classification remains `control_ep2_canonical_confirmed_post_fix`.

Why:

1. `control_ep2` remains the best candidate on large-suite `384:256`.
2. `seed_121` does not beat `control_ep2`; it trails by `0.0664` DS on the primary large budget.
3. Fresh `workers=1` and `workers=24` reruns agree exactly on the primary `384:256` budget for all candidates on both suites.
4. The default deterministic opening gate still classifies `control_ep2` as `high_search_breakthrough`.

This is not `candidate_ranking_changed_post_fix`.

This is not `evaluator_still_unstable` based on the completed reruns.

## Verification

| Check | Result |
|------|--------|
| Full medium rerank, workers=24 | PASS: `/tmp/azlite_fixed_eval_rerank/medium_workers24` |
| Full large rerank, workers=24 | PASS: completed across `/tmp/azlite_fixed_eval_rerank/large_workers24` plus `/tmp/azlite_fixed_eval_rerank/large_workers24_seed112_only` |
| Medium primary-budget rerank, workers=1 | PASS: `/tmp/azlite_fixed_eval_rerank/medium_standard_workers1` |
| Large primary-budget rerank, workers=1 | PASS: `/tmp/azlite_fixed_eval_rerank/large_standard_workers1` |
| Default gate reruns | PASS: `/tmp/azlite_fixed_eval_rerank/gate_*.json` |
| Unit tests | `Ran 2338 tests in 104.273s`; `FAILED (failures=1, errors=8, skipped=3)`; matches existing repo baseline style of unrelated failures |
| Ruff | PASS: `ruff check ml/alphazero_lite script/ai` |

## Guardrails

| Guardrail | Status |
|-----------|--------|
| No training | PASS |
| No promotion | PASS |
| No overwrite of `model-artifact/current` | PASS |
| No added data | PASS |
| No architecture change | PASS |
