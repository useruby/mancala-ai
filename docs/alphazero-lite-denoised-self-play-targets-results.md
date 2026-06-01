# AlphaZero-lite Denoised Self-Play Targets Results

## 1. Context

- Commit `2dc740b` identified noisy self-play policy targets as the leading cause of corrected guard regression.
- This run added optional denoised target generation while leaving default production behavior unchanged.
- This run stayed diagnostic-only: no arena, no promotion, no overwrite of `storage/ai/alphazero_lite/current`, and no broad replay-weight sweep.

## 2. Why Noisy Targets Are Now The Suspected Cause

- The prior opening-017 audit showed root Dirichlet noise restored bad target mass on corrected descendants.
- The corrective patch trace still regressed under training, which pointed back to the self-play labels themselves rather than replay anchoring.

## 3. Self-Play Implementation Change

- Added `--policy-target-noise-mode noisy|denoised` with default `noisy`.
- In `denoised` mode, action sampling can still use root Dirichlet noise, but the stored policy target is generated from a separate no-noise root search.
- Added optional `--write-root-target-telemetry` row metadata for root visits, priors, and stored targets.

## 4. Target-Quality Audit

| row_id | mode | simulations | corrected_reference_move | top_target_move | corrected_reference_mass | target_entropy | corrected_reference_is_top | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| opening_plies_1_8-017 | old_noisy_target_generation | 192 | 2 | 1 | 0.2623 | 1.6901 | false | matches_prior_audit_baseline |
| capture_available-002 | old_noisy_target_generation | 192 | 2 | 2 | 0.4329 | 1.7098 | true | matches_prior_audit_baseline |
| capture_available-003 | old_noisy_target_generation | 192 | 2 | 1 | 0.4697 | 1.3281 | false | matches_prior_audit_baseline |
| capture_available-006 | old_noisy_target_generation | 192 | 2 | 2 | 0.6221 | 1.4361 | true | matches_prior_audit_baseline |
| capture_available-007 | old_noisy_target_generation | 192 | 2 | 1 | 0.5275 | 1.2572 | false | matches_prior_audit_baseline |
| capture_available-008 | old_noisy_target_generation | 192 | 1 | 1 | 0.9087 | 0.5373 | true | matches_prior_audit_baseline |
| opening_plies_1_8-017 | denoised_target_generation | 192 | 2 | 1 | 0.2284 | 1.4302 | false | matches_prior_audit_017_remains_unstable |
| capture_available-002 | denoised_target_generation | 192 | 2 | 2 | 0.3802 | 1.9404 | true | matches_prior_audit_improvement |
| capture_available-003 | denoised_target_generation | 192 | 2 | 1 | 0.2789 | 1.2451 | false | matches_prior_audit_003_still_requires_higher_sims |
| capture_available-006 | denoised_target_generation | 192 | 2 | 2 | 0.6250 | 1.6456 | true | matches_prior_audit_improvement |
| capture_available-007 | denoised_target_generation | 192 | 2 | 2 | 0.4995 | 1.2196 | true | matches_prior_audit_improvement |
| capture_available-008 | denoised_target_generation | 192 | 1 | 1 | 0.9573 | 0.3166 | true | matches_prior_audit_improvement |
| opening_plies_1_8-017 | denoised_target_generation_higher_sims_384 | 384 | 2 | 1 | 0.1778 | 2.0447 | false | matches_prior_audit_017_remains_unstable |
| capture_available-002 | denoised_target_generation_higher_sims_384 | 384 | 2 | 2 | 0.3514 | 2.1570 | true | matches_prior_audit_improvement |
| capture_available-003 | denoised_target_generation_higher_sims_384 | 384 | 2 | 2 | 0.4121 | 1.7414 | true | better_than_prior_audit |
| capture_available-006 | denoised_target_generation_higher_sims_384 | 384 | 2 | 2 | 0.4833 | 2.0148 | true | matches_prior_audit_improvement |
| capture_available-007 | denoised_target_generation_higher_sims_384 | 384 | 2 | 2 | 0.5277 | 1.6172 | true | matches_prior_audit_improvement |
| capture_available-008 | denoised_target_generation_higher_sims_384 | 384 | 1 | 1 | 0.7415 | 1.3086 | true | matches_prior_audit_improvement |

## 5. Diagnostic Self-Play Scan

| row_id | dataset | hit_count | corrected_reference_move | average_reference_mass | top_target_distribution | notes |
| --- | --- | --- | --- | --- | --- | --- |
| opening_plies_1_8-017 | pr36_noisy_self_play | 32 | 2 | 0.1673 | `{0:1, 1:2, 2:3, 3:4, 4:10, 5:12}` | ok |
| capture_available-002 | pr36_noisy_self_play | 2 | 2 | 0.1849 | `{1:1, 2:1}` | ok |
| capture_available-003 | pr36_noisy_self_play | 7 | 2 | 0.3898 | `{1:5, 2:2}` | ok |
| capture_available-006 | pr36_noisy_self_play | 1 | 2 | 0.1175 | `{3:1}` | ok |
| capture_available-007 | pr36_noisy_self_play | 5 | 2 | 0.3316 | `{1:3, 2:2}` | ok |
| capture_available-008 | pr36_noisy_self_play | 8 | 1 | 0.4863 | `{0:1, 1:7}` | ok |
| opening_plies_1_8-017 | denoised_diagnostic_self_play | 19 | 2 | 0.2265 | `{4:19}` | ok |
| capture_available-002 | denoised_diagnostic_self_play | 3 | 2 | 0.1857 | `{1:3}` | ok |
| capture_available-003 | denoised_diagnostic_self_play | 1 | 2 | 0.4854 | `{2:1}` | ok |
| capture_available-006 | denoised_diagnostic_self_play | 0 | 2 | - | `-` | no_exact_hits |
| capture_available-007 | denoised_diagnostic_self_play | 4 | 2 | 0.3046 | `{1:4}` | ok |
| capture_available-008 | denoised_diagnostic_self_play | 3 | 1 | 0.4574 | `{1:3}` | ok |

## 6. Low-Cost Training Trace

- `noisy_self_play_reproduction`: data `['/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_versions/exp-v3-guard-safe-opening-selected-w1-iter1/self_play.jsonl']` epochs `[1, 2, 4]`.
- `denoised_self_play_only`: data `['/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_denoised_targets_versions/exp-v3-guard-safe-opening-selected-w1-denoised-targets-iter1/self_play.jsonl']` epochs `[1, 2, 4]`.
- `denoised_self_play_plus_selected_artifact_w1`: data `['/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_denoised_targets_versions/exp-v3-guard-safe-opening-selected-w1-denoised-targets-iter1/self_play.jsonl', '/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl']` epochs `[1, 2, 4]`.

## 7. Corrected Guard Kill-Gate Results

| trace_name | epoch | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | reference_visit_share_384 | reference_visit_share_1200 | row_pass | gate_pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noisy_self_play_reproduction | 1 | capture_available-002 | 2 | 1 | 2 | 0.2396 | 0.3442 | false | false | selected_move_not_reference |
| noisy_self_play_reproduction | 1 | capture_available-003 | 2 | 1 | 2 | 0.3438 | 0.4092 | false | false | selected_move_not_reference |
| noisy_self_play_reproduction | 1 | capture_available-006 | 2 | 2 | 2 | 0.3125 | 0.5458 | true | false | pass_reference_selected |
| noisy_self_play_reproduction | 1 | capture_available-007 | 2 | 2 | 2 | 0.3542 | 0.4208 | true | false | pass_reference_selected |
| noisy_self_play_reproduction | 1 | capture_available-008 | 1 | 1 | 1 | 0.5938 | 0.6967 | true | false | pass_reference_selected |
| noisy_self_play_reproduction | 2 | capture_available-002 | 2 | 1 | 1 | 0.2005 | 0.3425 | false | false | selected_move_not_reference |
| noisy_self_play_reproduction | 2 | capture_available-003 | 2 | 1 | 1 | 0.2865 | 0.3633 | false | false | selected_move_not_reference |
| noisy_self_play_reproduction | 2 | capture_available-006 | 2 | 1 | 2 | 0.2500 | 0.3700 | false | false | selected_move_not_reference |
| noisy_self_play_reproduction | 2 | capture_available-007 | 2 | 1 | 2 | 0.2708 | 0.4333 | false | false | selected_move_not_reference |
| noisy_self_play_reproduction | 2 | capture_available-008 | 1 | 1 | 1 | 0.5859 | 0.6842 | true | false | pass_reference_selected |
| noisy_self_play_reproduction | 4 | capture_available-002 | 2 | 1 | 2 | 0.3203 | 0.4508 | false | false | selected_move_not_reference |
| noisy_self_play_reproduction | 4 | capture_available-003 | 2 | 1 | 2 | 0.2292 | 0.4392 | false | false | selected_move_not_reference |
| noisy_self_play_reproduction | 4 | capture_available-006 | 2 | 2 | 2 | 0.4401 | 0.4992 | true | false | pass_reference_selected |
| noisy_self_play_reproduction | 4 | capture_available-007 | 2 | 1 | 2 | 0.2500 | 0.4867 | false | false | selected_move_not_reference |
| noisy_self_play_reproduction | 4 | capture_available-008 | 1 | 1 | 1 | 0.6901 | 0.7433 | true | false | pass_reference_selected |
| denoised_self_play_only | 1 | capture_available-002 | 2 | 0 | 0 | 0.1693 | 0.3242 | false | false | selected_move_not_reference |
| denoised_self_play_only | 1 | capture_available-003 | 2 | 1 | 1 | 0.3255 | 0.3117 | false | false | selected_move_not_reference |
| denoised_self_play_only | 1 | capture_available-006 | 2 | 2 | 2 | 0.2995 | 0.4742 | true | false | pass_reference_selected |
| denoised_self_play_only | 1 | capture_available-007 | 2 | 1 | 2 | 0.3255 | 0.4608 | false | false | selected_move_not_reference |
| denoised_self_play_only | 1 | capture_available-008 | 1 | 1 | 1 | 0.5781 | 0.6983 | true | false | pass_reference_selected |
| denoised_self_play_only | 2 | capture_available-002 | 2 | 0 | 2 | 0.2240 | 0.4583 | false | false | selected_move_not_reference |
| denoised_self_play_only | 2 | capture_available-003 | 2 | 1 | 2 | 0.3516 | 0.4300 | false | false | selected_move_not_reference |
| denoised_self_play_only | 2 | capture_available-006 | 2 | 2 | 2 | 0.3958 | 0.5592 | true | false | pass_reference_selected |
| denoised_self_play_only | 2 | capture_available-007 | 2 | 2 | 2 | 0.3880 | 0.5875 | true | false | pass_reference_selected |
| denoised_self_play_only | 2 | capture_available-008 | 1 | 1 | 1 | 0.6172 | 0.7367 | true | false | pass_reference_selected |
| denoised_self_play_only | 4 | capture_available-002 | 2 | 1 | 2 | 0.2422 | 0.4192 | false | false | selected_move_not_reference |
| denoised_self_play_only | 4 | capture_available-003 | 2 | 1 | 2 | 0.3177 | 0.4492 | false | false | selected_move_not_reference |
| denoised_self_play_only | 4 | capture_available-006 | 2 | 2 | 2 | 0.3750 | 0.4925 | true | false | pass_reference_selected |
| denoised_self_play_only | 4 | capture_available-007 | 2 | 1 | 2 | 0.3047 | 0.4133 | false | false | selected_move_not_reference |
| denoised_self_play_only | 4 | capture_available-008 | 1 | 1 | 1 | 0.5833 | 0.6975 | true | false | pass_reference_selected |
| denoised_self_play_plus_selected_artifact_w1 | 1 | capture_available-002 | 2 | 1 | 2 | 0.2734 | 0.4242 | false | false | selected_move_not_reference |
| denoised_self_play_plus_selected_artifact_w1 | 1 | capture_available-003 | 2 | 1 | 1 | 0.3750 | 0.3750 | false | false | selected_move_not_reference |
| denoised_self_play_plus_selected_artifact_w1 | 1 | capture_available-006 | 2 | 2 | 2 | 0.3724 | 0.5267 | true | false | pass_reference_selected |
| denoised_self_play_plus_selected_artifact_w1 | 1 | capture_available-007 | 2 | 1 | 2 | 0.3646 | 0.4725 | false | false | selected_move_not_reference |
| denoised_self_play_plus_selected_artifact_w1 | 1 | capture_available-008 | 1 | 1 | 1 | 0.7161 | 0.7867 | true | false | pass_reference_selected |
| denoised_self_play_plus_selected_artifact_w1 | 2 | capture_available-002 | 2 | 0 | 2 | 0.2240 | 0.3817 | false | false | selected_move_not_reference |
| denoised_self_play_plus_selected_artifact_w1 | 2 | capture_available-003 | 2 | 1 | 2 | 0.3490 | 0.4942 | false | false | selected_move_not_reference |
| denoised_self_play_plus_selected_artifact_w1 | 2 | capture_available-006 | 2 | 2 | 2 | 0.3099 | 0.5217 | true | false | pass_reference_selected |
| denoised_self_play_plus_selected_artifact_w1 | 2 | capture_available-007 | 2 | 1 | 2 | 0.3568 | 0.5683 | false | false | selected_move_not_reference |
| denoised_self_play_plus_selected_artifact_w1 | 2 | capture_available-008 | 1 | 1 | 1 | 0.6771 | 0.7692 | true | false | pass_reference_selected |
| denoised_self_play_plus_selected_artifact_w1 | 4 | capture_available-002 | 2 | 1 | 2 | 0.2396 | 0.4225 | false | false | selected_move_not_reference |
| denoised_self_play_plus_selected_artifact_w1 | 4 | capture_available-003 | 2 | 1 | 2 | 0.2969 | 0.4525 | false | false | selected_move_not_reference |
| denoised_self_play_plus_selected_artifact_w1 | 4 | capture_available-006 | 2 | 2 | 2 | 0.2995 | 0.4658 | true | false | pass_reference_selected |
| denoised_self_play_plus_selected_artifact_w1 | 4 | capture_available-007 | 2 | 1 | 2 | 0.3359 | 0.4700 | false | false | selected_move_not_reference |
| denoised_self_play_plus_selected_artifact_w1 | 4 | capture_available-008 | 1 | 1 | 1 | 0.6016 | 0.7217 | true | false | pass_reference_selected |

## 8. Interpretation

- Classification: `denoised_targets_partial_003_needs_more_sims`.
- `capture_available-003` still needs higher sims: `true`.
- `opening_plies_1_8-017` remains unstable: `true`.
- Low-cost trace achieved a full corrected guard pass: `false`.

## 9. Exactly One Recommended Next Action

Recommendation: **test denoised targets plus opening minimum 384 sims, diagnostic only.**
