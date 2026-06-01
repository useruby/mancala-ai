# Final Clean-Root Confirmation Results

## Context

- This final confirmation used the already-proven clean-root single-iteration mixture.
- The train step also matched the passing controlled isolate settings:
  - `val_split=0`
  - `seed=442`
  - `lr_scheduler=none`
- Goal: verify whether the lane can both pass the corrected guard gate and retain usable arena strength under the production entrypoint.

## Config

- config: `ml/alphazero_lite/configs/exp_v3_denoised_opening_min384_selected_w1_guard_w2_clean_root_confirm.json`
- versions dir: `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_clean_root_confirm_versions`
- run id: `exp-v3-denoised-opening-min384-selected-w1-guard-w2-clean-root-confirm`

## Verified Train Settings

Train command included:

- `--data-files iter1_self_play,selected_artifact,guard_controls_artifact`
- `--replay-weights 1,1,2`
- `--seed 442`
- `--val-split 0`
- `--lr-scheduler none`

## Training Result

- iteration dir: `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_clean_root_confirm_versions/exp-v3-denoised-opening-min384-selected-w1-guard-w2-clean-root-confirm-iter1`
- `policy_loss=1.107766`
- `value_loss=0.063119`
- `best_val_loss=1.126701`

## Top-k Corrected Guard Gate Results

| checkpoint | `002` | `003` | `006` | `007` | `008` | guard_gate_pass | arena_skipped_due_to_guard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `checkpoint.npz` | `2 / 2` | `2 / 2` | `2 / 2` | `2 / 2` | `1 / 1` | true | false |
| `checkpoint.top1.npz` | `2 / 2` | `2 / 2` | `2 / 2` | `2 / 2` | `1 / 1` | true | false |
| `checkpoint.top2.npz` | `1 / 2` | `1 / 2` | `2 / 2` | `1 / 2` | `1 / 1` | false | true |
| `checkpoint.top3.npz` | guard failed | guard failed | guard failed | guard failed | guard failed | false | true |

## Top-k Arena Results

| checkpoint | guard_gate_pass | arena_score | wins | losses | draws | selected_for_local_gate | notes |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `checkpoint.npz` | true | 0.0000 | 0 | 60 | 0 | false | guard passed, arena collapsed |
| `checkpoint.top1.npz` | true | 0.0000 | 0 | 60 | 0 | false | identical outcome to `checkpoint.npz` |
| `checkpoint.top2.npz` | false | - | - | - | - | false | arena skipped due to guard |
| `checkpoint.top3.npz` | false | - | - | - | - | false | arena skipped due to guard |

## Promotion Gate

- Not run.
- Reason: although at least one candidate passed the corrected guard gate, no guard-passing candidate was above the arena threshold.

## Classification

- `guard_safe_no_strength_gain`

## Interpretation

- The lane can be made guard-safe under the production entrypoint when the train settings match the proven isolate.
- However, the resulting guard-safe candidate has no usable playing strength against `storage/ai/alphazero_lite/current` in the 60-game arena screen.
- This means the opening replay intervention solved the guard rows at the cost of catastrophic strength loss, so it is not a promotion candidate.

## Exactly One Recommended Next Action

Recommendation: **stop opening replay work on this lane and mine corrected non-opening failure families instead, because the guard-safe configuration now appears strength-negative rather than promotion-worthy.**
