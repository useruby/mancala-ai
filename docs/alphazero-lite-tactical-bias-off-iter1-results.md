# AlphaZero-Lite Tactical-Bias-Off iter1 Training Results

**Date:** 2026-06-08

## Summary

Training `iter1` candidates from the `iter0_reference` checkpoint on tactical-bias-off
replay data (generated with `tactical_root_bias=0.0` PUCT player) does **not** improve
disadvantaged-seat performance. Both the tactical-bias-off lane and the control lane
(continued training on the same data with no new replay) **regress** compared to
`iter0_reference`.

**Result: REJECTED.** Tactical-bias-off training loses the 1200:1200 breakthrough,
shows no DS gain at any budget, and collapses at 256:768.

**Primary classification: `eval_only_effect`** -- tactical_root_bias=0.0 helps at
evaluation time but training on replay from tactical-bias-off search does not help.

## Artifact Lineage

| Artifact | Path | SHA256 |
|----------|------|--------|
| current production | model-artifact/current | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| iter0_reference weights.json | /tmp/azlite_iterative_random_replay/iter0_candidate_artifact | `0bbeaa9c6954ea147b084cdb3e68355887df1b0ab34c1fbc952b7988fdddc3bd` |
| iter0_reference checkpoint | /tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz | `c2430b56e7a68386343579aeac3610f4c88e1c35904e39179902e437fef36c18` |
| iter1_control_top1 weights.json | /tmp/azlite_tactical_bias_off_iter1/eval_artifacts/iter1_control_top1 | `a4a86dceb064a2be63d186785e9e3e30e5e5bed52bd4be2fb878d42edc45a5a9` |
| iter1_control_top2 weights.json | /tmp/azlite_tactical_bias_off_iter1/iter1_control/checkpoint.top2.npz | `66d1be6e3dcc382b968ab692015799c318babe8c31fb2d923122338f622d5a89` |
| iter1_control_top3 weights.json | /tmp/azlite_tactical_bias_off_iter1/iter1_control/checkpoint.top3.npz | `9c4e4df67e39d42c1cd9ddec8872505fd1f002eeda5786ff544842428e6e92da` |
| iter1_tbo_top1 weights.json | /tmp/azlite_tactical_bias_off_iter1/eval_artifacts/iter1_tbo_top1 | `ad927d23e9e5bd9c377be2909c4f1010b80146f958f392f06e234e2b3c831f8d` |
| iter1_tbo_top2 weights.json | /tmp/azlite_tactical_bias_off_iter1/iter1_tbo/checkpoint.top2.npz | `4f74c868a208cddd4932fda389af50addfec764ba149c1f4137240a046d77291` |
| iter1_tbo_top3 weights.json | /tmp/azlite_tactical_bias_off_iter1/iter1_tbo/checkpoint.top3.npz | `4951b0df1a8c53c3d428e35e4befa7860cff1624c0d38574055efb636810f5e1` |

iter0_reference was NOT regenerated; existing artifact from PR #93/PR #94 reused.

## Dataset

| Dataset | Path | SHA256 | Rows |
|---------|------|--------|------|
| generic bootstrap | /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl | `5d01a60e...` | — |
| old current-mined random replay (train) | /tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl | `7ca93389...` | — |
| new tactical-bias-off random replay (train) | /tmp/azlite_tactical_bias_off_iter1/tbo_random_train.jsonl | `774ccf13...` | 2374 |
| new tactical-bias-off random replay (holdout) | /tmp/azlite_tactical_bias_off_iter1/tbo_random_holdout.jsonl | `fbc0d804...` | 594 |
| new tactical-bias-off source states | /tmp/azlite_tactical_bias_off_iter1/tbo_source_states.jsonl | `f22e7720...` | 2968 |

### Dataset Comparison

| Metric | Value |
|--------|-------|
| New replay unique states | 1879 |
| Overlap with old replay | 0.69% (13 / 1879) |
| Overlap with generic bootstrap | 4.47% (84 / 1879) |
| Mined games | 363 (from 800 played) |
| Positions visited | 26,014 |
| Teacher top-1 visit share (mean) | 0.595 |
| Teacher/player top-move agreement rate | 48.19% |
| Capture rate | 90.2% |
| Extra-turn rate | 44.7% |
| Duplicate state count | 4201 |
| Capped state count | 0 |

### Phase Distribution (train)

| Phase | Rows | % |
|-------|------|---|
| Early | 279 | 9.4% |
| Mid | 1428 | 48.1% |
| Late | 1261 | 42.5% |

### First-Move Distribution (train teacher policy top-1)

| Pit | Count | % |
|-----|-------|---|
| 0 | 1284 | 54.1% |
| 1 | 471 | 19.8% |
| 2 | 242 | 10.2% |
| 3 | 165 | 7.0% |
| 4 | 126 | 5.3% |
| 5 | 86 | 3.6% |

### Mining Configuration

```
player_simulations: 96
tactical_root_bias: 0.0
root_policy_mode: deterministic
c_puct: 1.25
teacher_simulations: 1200
teacher_mode: classic_mcts
sampling_mode: random
input_encoding: kalah_v3
policy_target_mode: sharpened
value_target_mode: sharpened
games: 800, seed: 45
max_positions_per_game: 12
```

### Effective Replay Sample Share

| Lane | Generic Weight | Old Replay Weight | New TBO Replay Weight | TBO Share |
|------|---------------|-------------------|-----------------------|-----------|
| iter1_control | 4 | 1 | — | 0% |
| iter1_tbo | 4 | 1 | 1 | ~16.7% |

## Training

| Metric | iter1_control | iter1_tbo |
|--------|--------------|-----------|
| Init checkpoint | iter0_reference | iter0_reference |
| Data files | generic (4x) + old replay (1x) | generic (4x) + old replay (1x) + new tbo replay (1x) |
| Architecture | residual_v3 (96,3) | residual_v3 (96,3) |
| Input encoding | kalah_v3 | kalah_v3 |
| Epochs | 10 | 10 |
| Batch size | 512 | 512 |
| Learning rate | 1e-3 | 1e-3 |
| Value loss | huber (weight=0.3) | huber (weight=0.3) |
| Grad clip | 1.0 | 1.0 |
| Policy loss | 0.8566 | 0.8780 |
| Value loss | 0.2333 | 0.2451 |
| Best val loss | 1.1685 | 1.1839 |
| Top-1 val loss | 1.1685 | 1.1839 |
| Top-2 val loss | 1.1723 | 1.1839 |
| Top-3 val loss | 1.1740 | 1.1856 |
| Seed | 42 | 42 |

Note: control lane has lower val loss (1.1685 vs 1.1839), consistent with training
on a subset of the tbo lane's data (same distribution, fewer files).

## Seat-Aware Strength

All evaluations at 120 games, seed=42, c_puct=1.25, root_policy_mode=deterministic,
challenger vs current (model-artifact/current).

### iter0_reference (baseline, eval only)

Matches PR #99 results:

| Eval | Budget | Alt | DS | P0 W/L/D | P1 W/L/D | Margin | Len | Dup | CI95 |
|------|--------|-----|----|----------|----------|--------|-----|-----|------|
| default | 384:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -4 | 39 | 118 | (0, 0.06) |
| default | 768:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | 4 | 32 | 118 | (0, 0.06) |
| default | **1200:1200** | **1.00** | **1.00** | **60/0/0** | **60/0/0** | **13** | **41** | **118** | **(0.94, 1.00)** |
| default | 256:768 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -6 | 36 | 118 | (0, 0.06) |
| tbo | 384:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -4 | 40 | 118 | (0, 0.06) |
| tbo | **768:256** | **0.75** | **0.50** | **60/0/0** | **0/0/60** | **3** | **42** | **118** | **(0.38, 0.62)** |
| tbo | 1200:1200 | 0.75 | 0.50 | 60/0/0 | 0/0/60 | 10 | 37 | 118 | (0.38, 0.62) |
| tbo | 256:768 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -5 | 36 | 118 | (0, 0.06) |

Classification: `high_search_only` (default), `search_compression_promising` (tbo at 768:256)

### iter1_control_top1 (continued training, no new data)

| Eval | Budget | Alt | DS | P0 W/L/D | P1 W/L/D | Margin | Len | Dup | CI95 |
|------|--------|-----|----|----------|----------|--------|-----|-----|------|
| default | 384:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -8 | 38 | 118 | (0, 0.06) |
| default | 768:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -5 | 35 | 118 | (0, 0.06) |
| default | **1200:1200** | **0.75** | **0.50** | **60/0/0** | **0/0/60** | **2** | **40** | **118** | **(0.38, 0.62)** |
| default | **256:768** | **0.00** | **0.00** | **0/60/0** | **0/60/0** | **-17** | **34** | **118** | **(0, 0.06)** |
| tbo | 384:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -5 | 42 | 118 | (0, 0.06) |
| tbo | 768:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -10 | 39 | 118 | (0, 0.06) |
| tbo | **1200:1200** | **0.50** | **0.00** | **60/0/0** | **0/60/0** | **-9** | **38** | **118** | **(0, 0.06)** |
| tbo | **256:768** | **0.00** | **0.00** | **0/60/0** | **0/60/0** | **-23** | **36** | **118** | **(0, 0.06)** |

Classification: **REJECTED (extra_training_not_tactical_data)** -- 1200:1200
breakthrough degraded from DS=1.00 to DS=0.50, and 256:768 collapsed from winning
P0 to losing all games from both seats. Additional 10 epochs on the same data
damages rather than improves the model.

### iter1_tbo_top1 (tactical-bias-off replay)

| Eval | Budget | Alt | DS | P0 W/L/D | P1 W/L/D | Margin | Len | Dup | CI95 |
|------|--------|-----|----|----------|----------|--------|-----|-----|------|
| default | 384:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -4 | 38 | 118 | (0, 0.06) |
| default | **768:256** | **0.00** | **0.00** | **0/60/0** | **0/60/0** | **-13** | **46** | **118** | **(0, 0.06)** |
| default | **1200:1200** | **0.50** | **0.00** | **60/0/0** | **0/60/0** | **-10** | **38** | **118** | **(0, 0.06)** |
| default | **256:768** | **0.00** | **0.00** | **0/60/0** | **0/60/0** | **-12** | **34** | **118** | **(0, 0.06)** |
| tbo | 384:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -9 | 36 | 118 | (0, 0.06) |
| tbo | **768:256** | **0.00** | **0.00** | **0/60/0** | **0/60/0** | **-13** | **44** | **118** | **(0, 0.06)** |
| tbo | **1200:1200** | **0.50** | **0.00** | **60/0/0** | **0/60/0** | **-9** | **38** | **118** | **(0, 0.06)** |
| tbo | 256:768 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -8 | 47 | 118 | (0, 0.06) |

Classification: **REJECTED** -- Lost 1200:1200 breakthrough entirely (DS=1.00 ->
0.00). Collapsed at 768:256 (alt=0.00, loses all games from both seats). Collapsed
at 256:768 under default eval. Adding tactical-bias-off replay data during training
severely degrades model strength.

### Latency Summary

| Eval | Budget | iter0_ref | iter1_control | iter1_tbo |
|------|--------|-----------|---------------|-----------|
| default | 1200:1200 | 74.9 ms (p95: 109.1) | 68.5 ms (p95: 109.7) | 69.8 ms (p95: 108.8) |
| default | 384:256 | 19.5 ms (p95: 30.5) | 18.8 ms (p95: 31.0) | 18.5 ms (p95: 30.1) |

Latency is comparable across all candidates. No meaningful difference.

## Classification

| Lane | Classification | Rationale |
|------|---------------|-----------|
| iter0_reference (eval only) | high_search_only (default), search_compression_promising (tbo) | Matches PR #99: breakthrough at 1200:1200 only; tbo draws at 768:256 |
| iter1_control | extra_training_not_tactical_data (REJECTED) | Degraded from iter0: lost 1200:1200 wins, collapsed at 256:768 |
| iter1_tbo | REJECTED | Lost breakthrough entirely, collapsed at multiple budgets, no DS gain |

## Primary Finding

**eval_only_effect confirmed:** Tactical_root_bias=0.0 at evaluation time helps
iter0_reference reach DS=0.50 at 768:256 (all draws from disadvantaged seat).
However, training from iter0_reference on replay data generated with
tactical_root_bias=0.0 does not improve default-eval disadvantaged-seat scores.
Instead, any additional training from the iter0 checkpoint degrades the model.

The iter0_reference model appears to be at a delicate optimum for its training
regime. Continuing optimization from this checkpoint -- whether on the same data
(control) or on new tactical-bias-off data -- shifts weights away from this optimum
and reduces strength, most visibly at the 1200:1200 breakthrough and 256:768 asymmetry
budgets.

## What Was Tested vs What Was Not

Tested:
- Training from iter0_reference with 10 additional epochs
- Adding tactical-bias-off mined random replay (PUCT player at 96 sims, bias=0.0)
- Seat-aware evaluation at 4 budget pairs with both default and tbo eval
- 120-game evaluations (up from 60 in PR #99)

Not tested (intentionally excluded per guardrails):
- c_puct changes (kept at 1.25)
- residual_v4 architecture
- Tablebase overlay
- Agreement filtering
- Failure/disagreement sampling
- Root-prior transforms
- Promotion (no model promoted)
- Replay-weight sweeps

## Guardrail Compliance

| Guardrail | Status |
|-----------|--------|
| No promotion | PASS: no model promoted |
| No overwrite model-artifact/current | PASS: current unchanged |
| No change to local_promotion_gate defaults | PASS: not run |
| No change to global default tactical_root_bias | PASS: unchanged |
| No c_puct changes | PASS: all at 1.25 |
| No residual_v4 | PASS: residual_v3 only |
| No tablebase overlay | PASS: not used |
| No failure/disagreement or agreement filtering | PASS: random sampling only |
| No replay-weight sweep | PASS: fixed weights 4,1 and 4,1,1 |

## Miner Code Changes

Added PUCT-based player support to `ml/alphazero_lite/mine_failure_replay_dataset.py`:

- New CLI args: `--player-simulations`, `--tactical-root-bias`, `--root-policy-mode`, `--c-puct`
- When `--player-simulations > 0`, the current model uses PUCT search for move selection
  during game generation instead of pure NN top-policy evaluation
- Default behavior (player_simulations=0) is unchanged
- Imported `PUCT`, `SUPPORTED_ROOT_POLICY_MODES`, `DEFAULT_SEARCH_OPTIONS` from self_play
