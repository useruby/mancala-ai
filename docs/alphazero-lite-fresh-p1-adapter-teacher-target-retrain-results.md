# PR #214 Adapter Teacher Target Retrain

Three matched A16 lanes use only different policy target sources. No trajectories are generated and no promotion occurs.

## Invariants

```json
{
  "a16_state_hash": true,
  "all_passed": true,
  "full_replay_state_round_trip": true,
  "non_adapter_parameters_bit_identical": true,
  "p1_checkpoint_hash": true,
  "pr214_batch_plan": true,
  "replay_hash": true,
  "same_initial_state": true,
  "three_matched_lanes": true
}
```

## Network Fit

| Lane | CE improvement vs P1 | Mean legal L1 | P95 legal L1 |
| --- | ---: | ---: | ---: |
| stored384 | 0.000713 | 0.003667 | 0.007078 |
| clean384 | 0.001009 | 0.004954 | 0.009622 |
| clean1200 | 0.000632 | 0.003262 | 0.005947 |

## Canonical Arenas

| Lane | 384:256 effect | 1200:1200 effect |
| --- | ---: | ---: |
| stored384 | -0.0195 | -0.0234 |
| clean384 | -0.0742 | -0.0312 |
| clean1200 | -0.0742 | -0.0195 |
