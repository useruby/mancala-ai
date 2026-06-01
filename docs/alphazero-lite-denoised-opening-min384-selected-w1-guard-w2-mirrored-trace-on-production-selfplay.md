# Mirrored Diagnostic Trace On Frozen Production Self-Play

## Context

- The `train.py` frozen-data `val_split=0` isolate still failed the corrected guard gate on `capture_available-002` and `capture_available-003`.
- That left two possibilities:
  - the regenerated production self-play corpus itself was enough to cause the regression
  - the remaining regression came from differences between the PR #40 diagnostic trainer path and `train.py`
- To isolate that, this run replayed only the single PR #40 target trace on the frozen production `self_play.jsonl`, using the same diagnostic `run_trace` path, replay mix, and seed slot as the passing PR #40 target trace.

## Inputs

- self-play: `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/exp-v3-denoised-opening-min384-selected-w1-guard-w2-iter1/self_play.jsonl`
- selected replay: `/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl`
- guard controls: `/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl`
- replay weights: `[1, 1, 2]`
- initializer: `storage/ai/alphazero_lite/current`
- mirrored trace seed: `442`

## Output Paths

- summary: `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/frozen_data_mirrored_trace/single_trace_summary.json`
- checkpoints:
  - `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/frozen_data_mirrored_trace/traces/denoised_opening_min384_plus_selected_w1_guard_controls_w2_mirrored_production_self_play/checkpoints/epoch_1.npz`
  - `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/frozen_data_mirrored_trace/traces/denoised_opening_min384_plus_selected_w1_guard_controls_w2_mirrored_production_self_play/checkpoints/epoch_2.npz`
  - `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/frozen_data_mirrored_trace/traces/denoised_opening_min384_plus_selected_w1_guard_controls_w2_mirrored_production_self_play/checkpoints/epoch_4.npz`

## Corrected Guard Result

Epoch-4 guard rows for `denoised_opening_min384_plus_selected_w1_guard_controls_w2_mirrored_production_self_play`:

| row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | reference_visit_share_384 | reference_visit_share_1200 | row_pass | gate_pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `capture_available-002` | 2 | 2 | 2 | 0.3828 | 0.5075 | true | true |
| `capture_available-003` | 2 | 2 | 2 | 0.4089 | 0.5700 | true | true |
| `capture_available-006` | 2 | 2 | 2 | 0.4531 | 0.5875 | true | true |
| `capture_available-007` | 2 | 2 | 2 | 0.4531 | 0.5983 | true | true |
| `capture_available-008` | 1 | 1 | 1 | 0.7370 | 0.7825 | true | true |

- Outcome: full corrected guard pass at epoch 4.

## Stability Result

| row_id | selected_256 | selected_384 | selected_512 | selected_768 | selected_1200 | first_budget_reference_selected | diagnosis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `capture_available-002` | 1 | 2 | 2 | 2 | 2 | 384 | `recovers_by_384` |
| `capture_available-003` | 1 | 2 | 2 | 2 | 2 | 384 | `recovers_by_384` |
| `capture_available-006` | 2 | 2 | 2 | 2 | 2 | 256 | `stable_from_256` |
| `capture_available-007` | 2 | 2 | 2 | 2 | 2 | 256 | `stable_from_256` |
| `capture_available-008` | 1 | 1 | 1 | 1 | 1 | 256 | `stable_from_256` |

## Hash Comparison

Checkpoint hashes:

| artifact | sha256 |
| --- | --- |
| PR #40 passing diagnostic epoch-4 checkpoint | `472a3536945148636cca8b5b1a131b17eb8239f2838731fe9180ef28f8bbdbc1` |
| mirrored production-self-play epoch-4 checkpoint | `472a3536945148636cca8b5b1a131b17eb8239f2838731fe9180ef28f8bbdbc1` |
| `train.py` frozen-data `val_split=0` epoch-4 checkpoint | `32cbfb8214fb44c784fe2b348c74d3c2e98a6dcb6e4c5e38552c8d63361690f1` |

Exported weights hashes:

| artifact | sha256 |
| --- | --- |
| PR #40 passing diagnostic epoch-4 weights | `1b424902ee7622ad137db16dd8f50224e7b346ac505eff97a8a3a180d03820ee` |
| mirrored production-self-play epoch-4 weights | `1b424902ee7622ad137db16dd8f50224e7b346ac505eff97a8a3a180d03820ee` |
| `train.py` frozen-data `val_split=0` epoch-4 weights | `10009d6fe39a5872167276d578a85e1a0c5619f82818648d2d8bb0265d2043d0` |

## Interpretation

- The frozen production self-play corpus is not what broke the PR #40 trace.
- When the frozen production self-play corpus is trained with the PR #40 diagnostic `run_trace` path, the result is bit-for-bit identical to the original passing PR #40 checkpoint.
- The remaining regression is therefore explained by trainer dynamics in `train.py`, not by the regenerated self-play corpus.
- The relevant production-path differences still in play are the `train.py` training implementation itself, especially its optimizer/scheduler/checkpoint-selection behavior relative to the diagnostic loop.

## Conclusion

- Corpus effect: ruled out as the primary cause.
- Diagnostic trainer effect: confirmed.
- `train.py` path and PR #40 diagnostic path are not interchangeable for this lane.

## Exactly One Recommended Next Action

Recommendation: **diff `train.py` against the PR #40 mirrored `run_trace` training loop and align the minimal behavior difference needed to reproduce the passing epoch-4 checkpoint under the production training entrypoint before running another production lane.**
