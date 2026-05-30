# AlphaZero-lite Policy Target Encoding Audit Results

## Context

- Goal: audit whether the persistent `capture_available-002` family failure is best explained by policy-target inconsistency, teacher instability, an input-encoding gap, or a search/policy interaction.
- Guardrails held for this run: no production training, no arena, no promotion.
- Audited rows: `capture_available-002`, `003`, `005`, `006`, `007`, `008`.
- Generated artifacts:
  - `/tmp/azlite_policy_target_encoding_audit/policy_target_consistency_summary.json`
  - `/tmp/azlite_policy_target_encoding_audit/policy_target_encoding_audit_results.json`

## Current Encoding Summary

- Both the current artifact and guarded-`w2` artifact still use `input_encoding: "kalah_v3"` with `feature_count: 27`.
- `kalah_v3` contains board-state features plus aggregate tactical flags such as `player_extra_turn_available` and `player_capture_available`, but no per-action consequence features.
- Diagnostic-only helpers were added to expose per-move consequences without changing production encoding.

## Known Row Audit

| row_id | reference_move | current learned top move | guarded-w2 learned top move | current search selected move | reference gives extra turn | reference produces capture | wrong extra-turn move | wrong gives extra turn | wrong produces capture | reference policy current | wrong policy current | reference policy guarded-w2 | wrong policy guarded-w2 | diagnosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `capture_available-002` | `4` | `1` | `1` | `2` | `false` | `false` | `2` | `true` | `false` | `0.0103` | `0.2371` | `0.0868` | `0.3105` | `known_policy_mismatch` |
| `capture_available-003` | `1` | `1` | `0` | `1` | `true` | `false` | `-` | `-` | `-` | `0.7158` | `-` | `0.3664` | `-` | `none` |
| `capture_available-005` | `4` | `1` | `0` | `1` | `false` | `false` | `1` | `true` | `false` | `0.0156` | `0.5949` | `0.0345` | `0.3238` | `none` |
| `capture_available-006` | `2` | `1` | `1` | `2` | `true` | `false` | `-` | `-` | `-` | `0.1098` | `-` | `0.1335` | `-` | `none` |
| `capture_available-007` | `1` | `1` | `1` | `1` | `true` | `false` | `-` | `-` | `-` | `0.7587` | `-` | `0.3661` | `-` | `none` |
| `capture_available-008` | `1` | `1` | `1` | `1` | `true` | `false` | `-` | `-` | `-` | `0.8061` | `-` | `0.4927` | `-` | `none` |

Observations:

- `002` is the only audited row with an explicit learned-policy mismatch note in this run.
- On `002`, both raw policies strongly underweight the reference move `4` relative to the wrong extra-turn move `2`.
- Other rows include some raw-policy oddities, but search preserved the reference move for `003`, `006`, `007`, and `008`.

## Dataset Policy-Target Consistency

Summary:

- `conflicting_policy_targets_total: 0`
- `cross_source_conflicting_canonical_states: []`
- `duplicate_canonical_states_total: 1`

| source | rows | duplicate canonical states | conflicting policy targets | extra-turn top-move rate | no-extra-turn capture top-move rate | rows with extra-turn-over-capture conflict |
| --- | --- | --- | --- | --- | --- | --- |
| `tactical_opening_capture_family_replay.jsonl` | `12` | `1` | `0` | `0.0` | `0.25` | `0` |
| `tactical_balanced_replay.jsonl` | `10` | `0` | `0` | `0.1` | `0.3` | `0` |
| `tactical_balanced_replay_source.jsonl` | `198` | `0` | `0` | `0.2525` | `0.1313` | `0` |
| `tactical_capture_protection.jsonl` | `5` | `0` | `0` | `0.8` | `0.2` | `0` |
| `human_games_combined.jsonl` | `0` | `0` | `0` | `0.0` | `0.0` | `0` |
| forensic references | `224` | `0` | `0` | `0.0` | `0.0` | `0` |

Interpretation:

- This audit found no evidence that duplicated or cross-source canonical states are carrying conflicting top-move targets.
- The failure family is not explained by a target-generation or canonicalization bug in the scanned datasets.

## Per-Move Consequence Separability Probe

| feature_set | train_rows | dev_rows | top1_accuracy | capture_002_reference_rank | capture_003_reference_rank | average_cross_entropy |
| --- | --- | --- | --- | --- | --- | --- |
| `state_only_kalah_v3` | `4` | `2` | `0.5` | `2` | `1` | `0.8550` |
| `state_plus_flattened_consequences` | `4` | `2` | `0.5` | `3` | `1` | `1.5188` |
| `action_scoring_state_plus_consequences` | `19` | `9` | `0.6667` | `2` | `1` | `0.4296` |

Interpretation:

- Adding action-aware consequence features improved the small probe's overall ranking accuracy and cross-entropy.
- It did not improve the reference rank for `capture_available-002`; that row stayed rank `2`.
- On this evidence alone, the audit does not justify `encoding_gap_confirmed`.

## 002 Teacher/Target Audit

Teacher policy budgets:

| teacher_budget | top_move | policy_move_2 | policy_move_4 | move_4_minus_move_2 |
| --- | --- | --- | --- | --- |
| `384` | `4` | `0.2500` | `0.3047` | `0.0547` |
| `1200` | `4` | `0.3225` | `0.4167` | `0.0942` |
| `2400` | `4` | `0.3246` | `0.4942` | `0.1696` |

Classic search summaries:

| teacher_budget | selected_move | move_2_visits | move_4_visits | move_2_q | move_4_q | move_4_minus_move_2_q |
| --- | --- | --- | --- | --- | --- | --- |
| `384` | `4` | `96` | `117` | `0.6458` | `0.7179` | `0.0721` |
| `1200` | `4` | `387` | `500` | `0.6408` | `0.6860` | `0.0452` |
| `2400` | `4` | `779` | `1186` | `0.5995` | `0.6509` | `0.0514` |

Interpretation:

- `move_4_consistently_preferred` is `true` across all audited budgets.
- The teacher signal is stable and not close to flipping back to move `2`.
- That rules out `teacher_label_uncertain` for this row.

## Interpretation

- Final classification: `unresolved_search_policy_interaction`
- Why not `target_inconsistency`: scanned datasets and forensic references showed `0` conflicting policy targets and no cross-source canonical-state disagreements.
- Why not `teacher_label_uncertain`: `capture_available-002` kept move `4` as the teacher top move at `384`, `1200`, and `2400` simulations, with positive `Q` separation over move `2`.
- Why not `encoding_gap_confirmed`: the consequence-aware probe improved aggregate ranking metrics, but it did not improve `capture_available-002` reference rank.
- Why not `optimization_or_capacity_gap`: the probe result is too small and mixed to cleanly isolate model capacity/optimization as the next action.
- Combined with the prior root-prior intervention results, the remaining evidence points to a search/policy/value interaction around `002`, not a confirmed target bug and not yet a confirmed encoding-only failure.

## Recommended Next Action

Recommendation: **return to search/value trace audit, not training**.
