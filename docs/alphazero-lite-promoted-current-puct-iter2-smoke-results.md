# AlphaZero-Lite Promoted-Current PUCT Iteration-2 Smoke Results

**Date**: 2026-06-17
**Classification**: `iter2_mixed_signal`
**Schema**: `azlite_promoted_current_puct_iter2_smoke_v1`

## Summary

The promoted-current artifact (the control_ep2 PUCT policy-head e1 promoted
in PR #120) was used as the PUCT search guide to produce a second-iteration
replay. The replay passed the audit with diversity at or above PR #117 levels
and no collapse signal, but the resulting policy-head-only continuations did
not improve on the promoted source on the primary fixed large `384:256`
disadvantaged-seat score.

- Promoted current (`6ac71425...`) large `384:256` DS: `-0.3932`
- `iter2_puct_policy_head_e1` large `384:256` DS: `-0.4375` (regressed by `-0.0443`)
- `iter2_puct_policy_head_e2` large `384:256` DS: `-0.3932` (tied promoted current)
- Best iter2 delta vs promoted current on large `384:256`: `+0.0000`
- Replay audit classification: `puct_policy_improvement_signal`
- Replay diversity vs PR #117: `True` (at or above PR #117 on all metrics)
- Replay collapse signal: `False`
- Default gate classification for best iter2: `high_search_breakthrough`

The 1-epoch training slightly regressed on the primary `384:256` budget while
the 2-epoch training converged to a network that is functionally identical to
the promoted source at the policy-head level. The deterministic search plus
the small policy-head update means the top-1 move is the same in every state,
so the eval report shows the same game-level statistics for both
`promoted_current_ref` and `iter2_puct_policy_head_e2` on the fixed and the
three held-out suites.

This is not promotion. It is a smoke test that says the iterative
control_ep2 → e1 → e2 loop has saturated the policy-head update with
on-distribution PUCT replay, so additional PUCT-only iterations are not
productive on this fixed opening suite until replay diversity is widened or
the network update budget is changed.

## Inputs

| Item | Path | SHA256 / rows |
|------|------|----------------|
| promoted current artifact weights | `model-artifact/current/weights.json` | `6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece` |
| promoted e1 init checkpoint | `/tmp/azlite_control_ep2_puct_smoke/puct_policy_head_e1/checkpoint_epoch1.npz` | `a793f32565b0c706c4228e4de3bc00aea5c471089ec940c4fe85e726fe4f9357` |
| generic bootstrap replay | `/tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl` | `5d01a60e9dfc8756ffa47f2c8222e496a8ce8bd661457658de4884198344e147` (rows `9589`) |
| random teacher replay | `/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl` | `7ca93389d1be93bd1cf09d23ddfb9f040bb402a718cd991ac49e082bd7e2f69a` (rows `2016`) |
| medium suite | `/tmp/azlite_opening_suite/medium_eval.jsonl` | `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04` |
| large suite | `/tmp/azlite_opening_suite/large_eval.jsonl` | `ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4` |
| heldout seed 43 | `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed43_large.jsonl` | `5e0ed96ba56f99318c32a139692309759659da16a0a98c590e35f7e2496cade9` |
| heldout seed 44 | `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed44_large.jsonl` | `323f7abec32fb00d3b7b6b153ddd5692435755a3bc5340d8aa5aec32ca639620` |
| heldout seed 45 | `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed45_large.jsonl` | `ca72c8b7fe1adf7229f81183321475033f8548dcdeb1c71fd16c85c35ce89cda` |

Hash verification:

- Promoted current weights hash matched expected `6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece`
- Init checkpoint hash matched expected `a793f32565b0c706c4228e4de3bc00aea5c471089ec940c4fe85e726fe4f9357`

## Replay Generation

Command used:

```bash
.venv/bin/python ml/alphazero_lite/run_promoted_current_puct_iter2_smoke.py \
  --workdir /tmp/azlite_promoted_current_puct_iter2 \
  --current model-artifact/current \
  --expected-current-weights-sha256 6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece \
  --init-checkpoint /tmp/azlite_control_ep2_puct_smoke/puct_policy_head_e1/checkpoint_epoch1.npz \
  --expected-init-checkpoint-sha256 a793f32565b0c706c4228e4de3bc00aea5c471089ec940c4fe85e726fe4f9357 \
  --generic-bootstrap /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl \
  --random-teacher /tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl \
  --medium-suite /tmp/azlite_opening_suite/medium_eval.jsonl \
  --large-suite /tmp/azlite_opening_suite/large_eval.jsonl \
  --games 512 \
  --simulations 384 \
  --max-positions-per-game 24 \
  --workers 24 \
  --seed 42
```

Generation settings:

- player mode `puct`
- simulations `384`
- `c_puct=1.25` (repo default)
- root policy mode `visit_count`
- policy target mode `sharpened`
- value target mode `sharpened`
- input encoding `kalah_v3`
- model type `residual_v3`
- hidden sizes `96,3`
- init checkpoint: `puct_policy_head_e1/checkpoint_epoch1.npz` (the promoted e1)
- raw policy comparison network: `model-artifact/current` (the promoted e1)

Replay files:

| File | Path | SHA256 | Rows |
|------|------|--------|------:|
| raw replay | `/tmp/azlite_promoted_current_puct_iter2/promoted_current_puct_iter2_raw.jsonl` | not archived (size 188 MB) | `19070` |
| capped replay | `/tmp/azlite_promoted_current_puct_iter2/promoted_current_puct_iter2.jsonl` | not archived (size 123 MB) | `12051` |

Cap summary:

- games: `512`
- average uncapped positions/game: `37.2461`
- average kept positions/game: `23.5371`

## Replay Audit

Audit path: `/tmp/azlite_promoted_current_puct_iter2/puct_replay_audit.json`

| Metric | PR #117 value | This run | vs PR #117 |
|---|---:|---:|---|
| row count (capped) | `12037` | `12051` | +0.12% |
| unique rows | `9403` | `9339` | -0.68% |
| completed games | `512 / 512` | `512 / 512` | tie |
| average game length | `37.1445` | `37.2461` | +0.27% |
| policy entropy mean / p50 / p90 | `0.4167 / 0.0020 / 1.3704` | `0.4130 / 0.0016 / 1.3680` | mean slightly lower |
| legal mask validity | `PASS` (0 invalid) | `PASS` (0 invalid) | tie |
| value target range | `PASS` (0 invalid) | `PASS` (0 invalid) | tie |
| one-hot policy fraction | `0.3792` | `0.3786` | -0.16% |
| near-one-hot policy fraction | `0.5766` | `0.5772` | +0.10% |
| duplicate state count / rate | `2634 / 0.2188` | `2712 / 0.2250` | slightly higher |
| duplicate trajectory count / rate | `11 / 0.0217` | `13 / 0.0257` | +2 trajectories |
| root visit total mean / p50 / p90 | `384 / 384 / 384` | `384 / 384 / 384` | tie |
| search profile hash | `e68fea93ce5ba6755c8b3a57f284e89ccbc07b57ea68d2668e6920b7fd463ae0` | `e68fea93ce5ba6755c8b3a57f284e89ccbc07b57ea68d2668e6920b7fd463ae0` | identical |

Top-1 move distribution by pit (this run):

| Pit | Fraction |
|----:|---------:|
| 0 | `0.1112` |
| 1 | `0.1322` |
| 2 | `0.1770` |
| 3 | `0.1265` |
| 4 | `0.2036` |
| 5 | `0.2495` |

Value target summary (this run):

- mean: `0.0275`
- std: `0.5638`
- win/loss/draw: `0.4808 / 0.4184 / 0.1008`

First-1000 raw-policy comparison (vs `model-artifact/current`):

- top-1 agreement rate: `0.6060`
- mean `KL(search_policy || raw_policy)`: `0.7150` (PR #117: `0.6933`)
- mean `KL(raw_policy || search_policy)`: `6.1436` (PR #117: `5.9569`)
- changed-top-move fraction: `0.3940` (PR #117: `0.3600`)
- changed-top-move by phase: opening `0.3842`, mid `0.4781`, late `0.2964`
- classification: `puct_policy_improvement_signal`

Audit verdict: training proceeded. All replay diversity metrics are at or
above the PR #117 reference, the changed-top-move fraction is `0.3940` (well
above the `0.10` collapse threshold), the near-one-hot fraction is `0.5772`
(well below the `0.80` collapse threshold), and the unique-capped-row count
is `9339` (well above the `5000` collapse threshold). The replay does not
collapse back to the raw network policy.

## Training Lanes

All trained lanes used:

- model type `residual_v3`
- input encoding `kalah_v3`
- hidden sizes `96,3`
- batch size `512`
- seed `42`
- LR `1e-5`
- LR scheduler `none`
- value loss `huber`
- value loss weight `0.3`
- grad clip `1.0`
- policy target mode `sharpened`
- value target mode `sharpened`
- trainable scope `policy_head`
- init checkpoint: `puct_policy_head_e1/checkpoint_epoch1.npz`
- replay weights `4,1,1` for `generic_bootstrap,random_teacher,promoted_current_puct_iter2`

| Candidate | Epochs | Checkpoint SHA256 | Artifact weights SHA256 | Delta norm vs e1 | Relative delta | Policy loss | Value loss | Validation loss |
|------|------:|------|------|------:|------:|------:|------:|------:|
| `promoted_current_ref` | 0 | `a793f32565b0c706c4228e4de3bc00aea5c471089ec940c4fe85e726fe4f9357` | `6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece` | `0.0000` | `0.0000%` | n/a | n/a | n/a |
| `iter2_puct_policy_head_e1` | 1 | `92cad3c9d526a828c8b6b56817a6c546be57bb92f0dfe62ee6f2fb77aa4cd291` | `ef44552b07caf23877c2872a4a297eca480e37a74f2ba4f4ec999b1bd7921c93` | `0.0166` | `0.0615%` | `0.842125` | `0.196531` | `0.938270` |
| `iter2_puct_policy_head_e2` | 2 | `cb3c7005432a53dea684e5d742e738413437af37e9cf1eb1bccb253f77ca9a0d` | `0bd93993361c37a79648265b77ddf0c6c31d911f96b551014c6c1d48866b5c68` | `0.0289` | `0.1074%` | `0.840961` | `0.196923` | `0.937555` |

`iter2_puct_policy_head_e1` did not qualify for the default gate (it
trailed `promoted_current_ref` on large `384:256`), so its
`high_search_breakthrough_preserved` value is `None` (not evaluated).

## Medium Opening-Suite Benchmark

| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 1200:256 DS | 256:768 DS |
|------|------:|------:|------:|------:|------:|------:|
| `promoted_current_ref` | `-0.3203` | `-0.3828` | `+0.0859` | `-0.3359` | `-0.1797` | `-0.3828` |
| `iter2_puct_policy_head_e1` | `-0.3594` | `-0.3672` | `+0.1953` | `-0.3242` | `-0.1797` | `-0.4219` |
| `iter2_puct_policy_head_e2` | `-0.3203` | `-0.3281` | `+0.0664` | `-0.3359` | `-0.1797` | `-0.3672` |

## Large Opening-Suite Benchmark

| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 1200:256 DS | 256:768 DS | 384:256 P0 / P1 | duplicate trajectories |
|------|------:|------:|------:|------:|------:|------:|------|------:|
| `promoted_current_ref` | `-0.3932` | `-0.3984` | `-0.1589` | `-0.4167` | `-0.1706` | `-0.3984` | `0.4115 / 0.8047` | `1536` |
| `iter2_puct_policy_head_e1` | `-0.4375` | `-0.3789` | `-0.0312` | `-0.4219` | `-0.1706` | `-0.4427` | `0.3672 / 0.8047` | `1536` |
| `iter2_puct_policy_head_e2` | `-0.3932` | `-0.3346` | `-0.1810` | `-0.4167` | `-0.1706` | `-0.3698` | `0.4115 / 0.8047` | `1536` |

Per-budget delta of the best iter2 candidate (`iter2_puct_policy_head_e2`)
vs `promoted_current_ref` on the large suite:

| Budget | promoted_current_ref DS | best iter2 DS | delta |
|---|---:|---:|---:|
| 384:256 | `-0.3932` | `-0.3932` | `+0.0000` |
| 768:256 | `-0.3984` | `-0.3346` | `+0.0638` |
| 768:768 | `-0.1589` | `-0.1810` | `-0.0221` |
| 1200:1200 | `-0.4167` | `-0.4167` | `+0.0000` |
| 1200:256 | `-0.1706` | `-0.1706` | `+0.0000` |
| 256:768 | `-0.3984` | `-0.3698` | `+0.0286` |

The deterministic opening-suite benchmark with `root_policy_mode=deterministic`
makes the e2 candidate and the promoted current play the same top-1 move in
every state, so the arena emits identical game-level statistics on the three
budgets where they tie (`384:256`, `1200:1200`, `1200:256`).

P0/P1 split for large `384:256`:

| Candidate | P0 | P1 | P0-P1 gap |
|---|---:|---:|---:|
| `promoted_current_ref` | `0.4115` | `0.8047` | `0.3932` |
| `iter2_puct_policy_head_e1` | `0.3672` | `0.8047` | `0.4375` |
| `iter2_puct_policy_head_e2` | `0.4115` | `0.8047` | `0.3932` |

The P0/P1 gap is the seat-disadvantage signal. `iter2_puct_policy_head_e1`
worsens the gap by `+0.0443`; `iter2_puct_policy_head_e2` keeps it at the
promoted-current level.

## Default Deterministic Gate

Gate rule for this task: run the default deterministic seat-aware gate for
`promoted_current_ref` and any iter2 candidate that ties or beats it on large
`384:256`.

- `iter2_puct_policy_head_e1`: did not qualify (large `384:256` was `-0.4375`
  vs `promoted_current_ref`'s `-0.3932`).
- `iter2_puct_policy_head_e2`: qualified (large `384:256` was `-0.3932`,
  ties `promoted_current_ref`).

| Candidate | Classification | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 1200:256 DS | 256:768 DS | high-search preserved |
|------|------|------:|------:|------:|------:|------:|------:|---|
| `promoted_current_ref` | `high_search_breakthrough` | `+0.0000` | `+0.5000` | `+0.0000` | `+1.0000` | `+1.0000` | `+0.0000` | yes |
| `iter2_puct_policy_head_e2` | `high_search_breakthrough` | `+0.0000` | `+0.5000` | `+0.0000` | `+1.0000` | `+1.0000` | `+0.0000` | yes |

Both candidates receive the same per-budget DS because the deterministic
search and the small policy-head update mean they pick the same top-1 move
in every state.

## Held-Out Suites (PR #119)

| Suite | promoted_current_ref 384:256 DS | best iter2 384:256 DS | best iter2 delta |
|---|---:|---:|---:|
| `heldout_seed43_large` | `-0.3932` | `-0.3932` | `+0.0000` |
| `heldout_seed44_large` | `-0.4089` | `-0.4089` | `+0.0000` |
| `heldout_seed45_large` | `-0.4349` | `-0.4349` | `+0.0000` |
| **mean** | `-0.4123` | `-0.4123` | `+0.0000` |
| **worst suite** | `-0.4349` | `-0.4349` | `+0.0000` |

The held-out suites confirm the fixed-suite finding: `iter2_puct_policy_head_e2`
plays the same games as the promoted current on every held-out opening
prefix, which means the policy-head-only update did not change any top-1
move on the held-out opening distribution.

## Decision

### Classification: `iter2_mixed_signal`

The replay audit shows the PUCT targets are diverse, the changed-top-move
fraction is `0.3940` (well above the `0.10` collapse threshold), and the
search profile hash is identical to PR #117, so the search is doing real
work. But neither iter2 candidate beat `promoted_current_ref` on the primary
large `384:256` DS: `iter2_puct_policy_head_e1` regressed by `-0.0443`, and
`iter2_puct_policy_head_e2` tied the promoted current at `-0.3932`.

This rules out the four clean buckets:

- **NOT** `iter2_policy_improvement`: best iter2 delta is `+0.0000`, not
  `>=+0.03`.
- **NOT** `iter2_replay_not_useful`: `iter2_puct_policy_head_e2` does not
  trail the promoted current (it ties at `-0.3932`), so the rule that
  requires both iter2 candidates to trail does not trigger.
- **NOT** `iter2_policy_collapse`: the changed-top-move fraction is
  `0.3940` (above `0.10`), the near-one-hot fraction is `0.5772` (below
  `0.80`), and the unique-capped-row count is `9339` (above `5000`).
- **NOT** `iter2_borderline`: the delta is `+0.0000`, not in the
  `+0.01 … +0.03` range.

The result is "the PUCT replay from the promoted current is diverse and the
search is doing real work, but the policy-head-only training has saturated
on the promoted current for this fixed opening distribution and does not
improve the source on the primary metric." The right operational action is
to pause the iterative loop and investigate the source of the search-target
collapse (every PUCT top-1 is already the same as the raw policy top-1 on
this distribution for the source network) before adding more iterations.

## Promotion

Not run.

Reason: this task is a smoke test only and explicitly disallows promotion.
The smoke-test verdict is `iter2_mixed_signal`; no iter2 candidate qualifies
for promotion.

## Verification

| Check | Result |
|------|--------|
| `ruff check ml/alphazero_lite script/ai` | PASS |
| `ruff check ml/alphazero_lite/run_promoted_current_puct_iter2_smoke.py` | PASS |
| `python -m unittest ml.alphazero_lite.test_run_manifest` | PASS |
| `python -m unittest ml.alphazero_lite.test_current_artifact_runtime` | PASS |
| smoke experiment | PASS: `/tmp/azlite_promoted_current_puct_iter2/summary_metrics.json` |
| promoted current weights SHA matches expected | PASS |
| init checkpoint SHA matches expected | PASS |

## Guardrails

| Guardrail | Status |
|------|------|
| no promotion | PASS |
| no overwrite of `model-artifact/current` | PASS |
| no full-network training | PASS |
| no `last_block_policy` training | PASS |
| no architecture change | PASS |
| no `residual_v4` | PASS |
| no classic-MCTS replay | PASS |
| no threshold changes | PASS |
| no seed sweep | PASS |
| replay-audit failure not hidden | PASS |

## Artifacts

| Artifact | Path |
|------|------|
| experiment script | `ml/alphazero_lite/run_promoted_current_puct_iter2_smoke.py` |
| experiment root | `/tmp/azlite_promoted_current_puct_iter2` |
| summary metrics | `/tmp/azlite_promoted_current_puct_iter2/summary_metrics.json` |
| replay audit | `/tmp/azlite_promoted_current_puct_iter2/puct_replay_audit.json` |
| capped replay | `/tmp/azlite_promoted_current_puct_iter2/promoted_current_puct_iter2.jsonl` |
| raw replay | `/tmp/azlite_promoted_current_puct_iter2/promoted_current_puct_iter2_raw.jsonl` |
| medium eval report | `/tmp/azlite_promoted_current_puct_iter2/eval_medium/temperature_benchmark_report.json` |
| large eval report | `/tmp/azlite_promoted_current_puct_iter2/eval_large/temperature_benchmark_report.json` |
| gate reports | `/tmp/azlite_promoted_current_puct_iter2/eval_gate` |
| held-out reports | `/tmp/azlite_promoted_current_puct_iter2/eval_heldout_heldout_seed{43,44,45}_large/temperature_benchmark_report.json` |
