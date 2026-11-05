# Seed554 results: fixed-state teacher-budget training

## Scope

The comparison uses the same ordered **13,818** control-training occurrences in
both arms. The frozen #549 schedule is unchanged at **108 updates / 54,896
exposures** per arm. C128 is a new temperature-1.0 control; it is not a request
to reproduce historical C108's generation-temperature targets.

All 5,635 unique canonical training states above 16 active stones received
paired 128- and 512-simulation ordinary PUCT searches, with exact-leaf solving
disabled, seed `554000 + state_index`, fresh trees and RNGs, and alternating
budget-first order. The 4,798 ≤16-stone occurrences retained their original
policy targets. Each arm retained all occurrences and original row weights,
values, legal actions, and inputs.

## Policy-target differences

| Cohort | Occurrences | Bitwise-different targets | Highest-target action differs | Mean TV | Median TV |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pooled | 13,818 | 8,928 | 1,215 | 0.0912552 | 0.0371094 |
| >32 stones | 4,743 | 4,730 | 637 | 0.1494242 | 0.1035156 |
| 17–32 stones | 4,277 | 4,198 | 578 | 0.1291200 | 0.0878906 |
| ≤16 stones | 4,798 | 0 | 0 | 0 | 0 |

TV is half the legal-policy L1 distance. Values are occurrence-weighted, so
repeated states retain their original frequencies. These differences show that
the two budgets produced different training labels; they do not indicate which
teacher targets are better.

## Production-loaded tensor hashes

SHA256 hashes below are over the contiguous production-loaded float32 tensors.
Inputs, legal masks, values, and policy weights are identical between arms;
the policy tensor differs.

| Tensor | C128 | T512 |
| --- | --- | --- |
| Input | `b15449bf80cebeb34782ea0087866390bcb8916bd434cf4a271fa232c5268da2` | `b15449bf80cebeb34782ea0087866390bcb8916bd434cf4a271fa232c5268da2` |
| Legal mask | `99d9175bf45d31b79930570bd1c33ed461d5daebd0e8ea7faf27080867fbaf04` | `99d9175bf45d31b79930570bd1c33ed461d5daebd0e8ea7faf27080867fbaf04` |
| Value target | `3bed13e285e77450dfa7ddc11f1246671c8e2ffc2acf9f3f75578ebd1daeed1e` | `3bed13e285e77450dfa7ddc11f1246671c8e2ffc2acf9f3f75578ebd1daeed1e` |
| Policy weight | `1d128376fedcc388789a1e5769fbf1ed1f397e9e4d662862cde25d9699ec7c67` | `1d128376fedcc388789a1e5769fbf1ed1f397e9e4d662862cde25d9699ec7c67` |
| Policy target | `1b264a3023fb04f6033208a3745c7d84383968e931d6028294f7ee47b0a87c5f` | `7270ba81d755b87cd7bd543fcee2e063d4a226129af101ce978d603e83295167` |

## Training gate and development diagnostics

Both arms completed exactly 108 updates / 54,896 exposures from the same
initializer with fresh Adam states. Before T512 started, C128 matched the
production `train_one_epoch` primitive on **all 108 updates**, including
pre/post parameter and Adam optimizer states. Checkpoints were saved at updates
27, 54, 81, and 108. T512 update 108 is the sole treatment candidate.

On the unchanged 4,738-occurrence development membership, combined pooled
policy CE was 0.745624 / 0.741250 (exposure / equal-input) for the initializer,
0.628356 / 0.625407 for C128, and 0.631068 / 0.628258 for T512. Corresponding
actual-value MSE was 0.337509 / 0.342456, 0.303804 / 0.312747, and 0.303879 /
0.312780. The full endpoint metrics and differences by source, stone cohort,
and weighting are in `development-results.json`; occurrence-level predictions
and unchanged targets are in `development-predictions.jsonl`.

These development diagnostics measure fit to existing targets only. They do not
measure target quality, playing strength, or advancement. No arena, games,
serving-asset changes, promotion, or deployment were part of this task.

## Verification and bounded input authentication

The read-only verifier independently rebuilt all occurrence mappings and target
bindings, replayed the paired teacher searches (excluding elapsed timings),
replayed both training trajectories and checkpoints, and reconstructed endpoint
predictions, metrics, and effects. Its derivative replay receipt is
`derivative-verification.json`; the separate training-input binding is
`training-input-binding.json`.

The #553 bounded-authentication limitation remains explicit: seed554
authenticates source shards, occurrence identities, published receipts, the
unchanged schedule, development membership, and initializer bytes. It does not
claim that #549's full historical verifier passed in the tablebase-mismatched
runtime. Seed554's teacher search path uses ordinary PUCT with exact-leaf
solving disabled and performs no native solver queries.
