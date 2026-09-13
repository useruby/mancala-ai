# Internal State-Cluster Provenance Audit

## Selected Cluster

Definition: `{"capture_actions":[],"extra_turn_actions":[0],"legal_count":4,"own_zero_mask":"001001","player":0}`.
Deterministic selection: ranked by selected-family degradation events, then SHA-256 of sorted canonical states; top cluster has 68 events and 73 opportunities.

## State Evidence

| State | Events | Parent top | Control top | Uniform top |
| --- | ---: | ---: | ---: | ---: |
| `{"current_player":0,"opponent_pits":[2,0,7,7,1,6],"opponent_store":2,"player_pits":[6,2,0,6,6,0],"player_store":3}` | 68 | 0 | 3 | 3 |

## Replay Accounting

| Lane | Source | Exact effective rows | Parent effective rows | Neighbor effective rows |
| --- | --- | ---: | ---: | ---: |
| control | `dynamic_current_self_play` | 0 | 0 | 3 |
| control | `family_leave_one_out_without_opening_missed_extra_turn_continuation.jsonl` | 0 | 0 | 0 |
| control | `guard_safe_controls_only.jsonl` | 0 | 0 | 0 |
| uniform1200 | `dynamic_current_self_play` | 0 | 0 | 5 |
| uniform1200 | `family_leave_one_out_without_opening_missed_extra_turn_continuation.jsonl` | 0 | 0 | 0 |
| uniform1200 | `guard_safe_controls_only.jsonl` | 0 | 0 | 0 |

## Conclusion

Provenance classification: `structural_neighbor_provenance_identified`.
Training-data intervention justified: `False`.

## Next Experiment

Exactly one next experiment: construct a frozen cluster evaluation set and measure policy change under normal iterative self-play before changing replay content.
