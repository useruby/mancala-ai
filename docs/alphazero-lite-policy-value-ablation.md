# AlphaZero-lite Policy-Value Search Ablation

## Artifact

`model-artifact/current`

## Budgets

`128`, `384`, `1200`

## Overall Attribution

- `128`: `classic_only` is strongest at `0.6205` top-1 agreement. The learned-search variants cluster well below that: `full` reaches `0.4375`, `value_only` reaches `0.4330`, and `policy_only` reaches `0.4152`. Within the learned variants, attribution still marks `value` as the larger contributor because `value_only` stays closer to `full` than `policy_only` does.
- `384`: `classic_only` again leads at `0.7232`. `value_only` reaches `0.4062` and slightly exceeds `full` at `0.3929`, while `policy_only` trails at `0.3304`. Attribution again marks `value` as the larger contributor.
- `1200`: `classic_only` remains best at `0.7679`. `value_only` reaches `0.4821` and again exceeds `full` at `0.4464`, while `policy_only` reaches `0.4152`. Attribution again marks `value` as the larger contributor.

## Bucket Findings

- `capture_available`: `classic_only` is best at every budget. Among the learned variants, `value_only` leads at `128` and `1200`, while `full` leads at `384`.
- `opening_plies_1_8`: `classic_only` is best at every budget, and `value_only` is the best learned variant at `128`, `384`, and `1200`.
- `incumbent_proxy_disagreement`: `classic_only` is best at every budget. Among learned variants, `value_only` leads at `128` and `384`, while `full` leads at `1200`.
- `high_value_swing`: hardest weakness bucket for the combined search. `classic_only` is best at all three budgets, and `full` is never best.
- `high_imbalance`: another strong classic-search bucket. `classic_only` is best at all three budgets; among learned variants, `value_only` is best at `384`, while `policy_only` and `value_only` tie at `128` and `1200`.
- `early_extra_turn`: mixed. `full` is best at `128`, but `classic_only` is best at `384` and `1200`.
- `sparse_endgame`: `classic_only` is best at every budget. `full` beats the other learned variants at `128` and `384`, then all learned variants tie at `1200`.
- `starvation_pressure`: `classic_only` is best at every budget. `full` is the best learned variant at `128` and `384`, while `policy_only` is best among learned variants at `1200`.

## Answer

Overall larger contributor among the learned search signals: `value`.
Overall strongest mode after restoring the intended baseline: `classic_only` at all three budgets.
Buckets where `full` underperforms a simpler mode: every tracked bucket versus `classic_only`; among the learned variants specifically, the clearest recurring weak spots are `opening_plies_1_8`, `incumbent_proxy_disagreement`, `high_imbalance`, and `high_value_swing`.
