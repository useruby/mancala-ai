# Self-Play Reproducibility Investigation

## Question

Was fresh self-play generation the remaining source of instability in the clean-root production-style lane?

## Compared Corpora

| label | path |
| --- | --- |
| frozen passing corpus | `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/exp-v3-denoised-opening-min384-selected-w1-guard-w2-iter1/self_play.jsonl` |
| clean-root iter1 corpus | `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_clean_root_versions/exp-v3-denoised-opening-min384-selected-w1-guard-w2-clean-root-iter1/self_play.jsonl` |

## File-Level Result

The two self-play corpora are identical.

| check | result |
| --- | --- |
| full-file sha256 | both `9341e344e813d8bc5103b83b152f684b638b4650c716f4acf43681f93e3836e4` |
| order-insensitive multiset digest | both `3f01a73e7aac5e717c55ad50aafc9c4a146b59f3bd82b24acf668ec598b9372b` |
| total rows | both `63959` |
| unique row hashes | both `63894` |

## Targeted Guard/Open Support

The targeted corrected-guard and opening-anchor support is also identical.

| row_id | frozen hit_count | clean-root hit_count | frozen top distribution | clean-root top distribution |
| --- | ---: | ---: | --- | --- |
| `capture_available-002` | 4 | 4 | `{2:4}` | `{2:4}` |
| `capture_available-003` | 0 | 0 | `{}` | `{}` |
| `capture_available-006` | 0 | 0 | `{}` | `{}` |
| `capture_available-007` | 3 | 3 | `{2:3}` | `{2:3}` |
| `capture_available-008` | 3 | 3 | `{1:3}` | `{1:3}` |
| `opening_plies_1_8-017` | 16 | 16 | `{2:16}` | `{2:16}` |

## Conclusion

- Fresh self-play generation is not the source of the clean-root lane failure.
- The clean-root iter1 self-play corpus is exactly the same corpus that was used in the earlier frozen passing isolate.
- The remaining instability therefore comes from train-path settings layered on top of that same corpus.

## Practical Implication

At this point the unresolved production-style differences are no longer about corpus generation. The meaningful remaining levers are the lane's train settings, especially:

- `val_split=0.1` versus the passing isolate's `val_split=0`
- lane seed `42` versus the passing isolate's `442`

Those settings are enough to explain why the same corpus can both pass and fail depending on how `train.py` is invoked.

## Exactly One Recommended Next Action

Recommendation: **if you want one more production-style confirmation, run a single clean-root lane that keeps the aligned shared loop and no-scheduler setting but also sets `val_split=0` and training seed `442`, because corpus reproducibility is already proven and the remaining instability is in the train configuration.**
