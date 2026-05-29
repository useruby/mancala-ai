# AlphaZero-lite Rule-Conditioned Opening Full Guarded Results

## Outcome

Ran the guarded opening-family experiment with exactly two replay weights, `1` and `2`, using:

- artifact: `/tmp/azlite_failure_family_diag/rule_conditioned_opening_family_full_guarded_artifact.jsonl`
- summary: `/tmp/azlite_failure_family_diag/rule_conditioned_opening_family_full_guarded_artifact_summary.json`
- incumbent path: `storage/ai/alphazero_lite/current`

The artifact was verified in-place before training and already matched the expected guarded composition:

- `row_count=15`
- tracked opening rows present: `capture_available-005`, `capture_available-006`, `capture_available-007`, `capture_available-008`
- rule-collision guard rows present: `capture_available-002`, `capture_available-003`
- replay roles present:
  - `capture_protection`
  - `capture_preservation`
  - `nearby_preservation`
  - `opening_capture_extra_turn_reference`
  - `opening_capture_no_extra_turn_reference`
  - `rule_collision_extra_turn_reference_guard`
  - `rule_collision_no_extra_turn_reference_guard`

Both candidates failed the required pre-arena `002/003` local kill gate.

- `capture_available-002` still search-selected the wrong extra-turn move `2` instead of the reference move `4`
- `capture_available-003` kept the reference move `1`, but its reference visit share materially degraded versus incumbent
- arena, hard-state validation, `MCTS1200`, benchmark, top-k, and local promotion gate were skipped for both weights as required
- no model was promoted

## Runtime Artifacts

- experiment summary: `/tmp/azlite_rule_conditioned_opening_full_guarded/rule-conditioned-opening-full-guarded/experiment_summary.json`
- runtime config `w1`: `/tmp/azlite_rule_conditioned_opening_full_guarded/rule-conditioned-opening-full-guarded/w1/runtime_config.json`
- runtime config `w2`: `/tmp/azlite_rule_conditioned_opening_full_guarded/rule-conditioned-opening-full-guarded/w2/runtime_config.json`
- candidate dir `w1`: `/tmp/azlite_rule_conditioned_opening_full_guarded/rule-conditioned-opening-full-guarded/w1/versions/aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w1-iter1`
- candidate dir `w2`: `/tmp/azlite_rule_conditioned_opening_full_guarded/rule-conditioned-opening-full-guarded/w2/versions/aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1`
- local gate `w1`: `/tmp/azlite_rule_conditioned_opening_full_guarded/rule-conditioned-opening-full-guarded/w1/versions/aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w1-iter1/capture_002_003_rule_conditioned_gate.json`
- local gate `w2`: `/tmp/azlite_rule_conditioned_opening_full_guarded/rule-conditioned-opening-full-guarded/w2/versions/aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1/capture_002_003_rule_conditioned_gate.json`

## Results Table

| variant | replay_weight | artifact_row_count | has_002_guard | has_003_guard | row_002_reference_move | row_002_searched_move | row_002_reference_visit_share | row_002_wrong_extra_turn_visit_share | row_002_gate_pass | row_003_reference_move | row_003_searched_move | row_003_reference_visit_share | row_003_gate_pass | hard_state_average_regret | hard_state_value_calibration_mae | arena_score | arena_ci_low | arena_ci_high | unstable_decision | mcts1200_score | benchmark_pass | decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `w1` | `1` | `15` | `true` | `true` | `4` | `2` | `0.0547` | `0.6797` | `false` | `1` | `1` | `0.7292` | `false` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `reject_local_gate` | `002` collision persisted and `003` reference visit share fell from `0.8932` to `0.7292`; skipped arena and all downstream evaluation |
| `w2` | `2` | `15` | `true` | `true` | `4` | `2` | `0.0547` | `0.6693` | `false` | `1` | `1` | `0.6875` | `false` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `reject_local_gate` | `002` collision persisted and `003` reference visit share fell from `0.8932` to `0.6875`; skipped arena and all downstream evaluation |

## Row-Pair Gate Readout

`w1`:

- `capture_available-002`
  - reference move: `4`
  - searched selected move: `2`
  - reference move visit share: `0.0547`
  - wrong extra-turn move visit share: `0.6797`
  - policy reference probability: `0.0617`
  - selected-minus-reference Q margin: `0.0417`
  - gate result: fail
  - reason: `row_002_reference_move_not_selected,row_002_wrong_extra_turn_still_dominates`
- `capture_available-003`
  - reference move: `1`
  - searched selected move: `1`
  - reference move visit share: `0.7292`
  - incumbent reference move visit share: `0.8932`
  - gate result: fail
  - reason: `row_003_reference_visit_share_materially_degraded`

`w2`:

- `capture_available-002`
  - reference move: `4`
  - searched selected move: `2`
  - reference move visit share: `0.0547`
  - wrong extra-turn move visit share: `0.6693`
  - policy reference probability: `0.0668`
  - selected-minus-reference Q margin: `0.0496`
  - gate result: fail
  - reason: `row_002_reference_move_not_selected,row_002_wrong_extra_turn_still_dominates`
- `capture_available-003`
  - reference move: `1`
  - searched selected move: `1`
  - reference move visit share: `0.6875`
  - incumbent reference move visit share: `0.8932`
  - gate result: fail
  - reason: `row_003_reference_visit_share_materially_degraded`

## Decision

- `w1`: `reject_local_gate`
- `w2`: `reject_local_gate`

Why:

- neither weight fixed `capture_available-002`
- both weights preserved the selected move on `capture_available-003`, but both materially degraded reference visit share there
- the local pre-arena guard did exactly what it was supposed to do and stopped the lane before expensive arena evaluation

## Recommendation

Next action: **run a search-prior control experiment instead of another replay lane**.
