# Outcome-Value Scale Replication

**Classification:** `outcome_value_scale_effect_not_replicated`

The frozen PR #189 transform was evaluated without refitting:

```
clip(1.425456519226987 * v_current + 0.061720579402024627, -1, +1)
```

The causal corpus contained 1,024 unique fresh states: 512 current-model standard-start self-play states, 256 independent opening-family states, and 256 mid/late enriched states. It excluded all 4,336 states persisted by PR #188 and PR #189. State and source hashes, player, phase, legal-move count, current value, and policy entropy are frozen in `docs/data/alphazero-lite-outcome-value-scale-replication-summary.json`.

| Originating budget | Disagreement | True Q-rank change | Visit JS | Margin delta, continue 768 | Margin delta, continue 1200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D384 | 10.84% | 43.75% | 0.01184 | +0.01877 [-0.01764, +0.05593] | +0.02252 [-0.01426, +0.05968] |
| D768 | 9.38% | 45.90% | 0.01373 | +0.03429 [-0.00955, +0.07812] | +0.03385 [-0.00521, +0.07248] |
| D1200 | 9.08% | 42.97% | 0.01322 | +0.02957 [-0.02195, +0.08020] | +0.01568 [-0.02912, +0.06093] |

| Originating budget | Root-value delta | Q/U ratio delta | Visit-margin delta |
| --- | ---: | ---: | ---: |
| D384 | +0.00597 | +0.46912 | +15.41699 |
| D768 | +0.00834 | +0.60265 | +23.26855 |
| D1200 | +0.00874 | +0.75348 | +32.38379 |

D1200 produced 93 disagreement states and non-negative binary-outcome point estimates (+0.09677 at 768 and +0.10753 at 1200). However, both normalized-margin bootstrap lower bounds were below zero. The forced-move replication therefore fails its preregistered criterion.

The canonical medium-suite arena diagnostic was not run because it is conditional on a passing D1200 causal replication. No training, scale sweep, c_puct tuning, policy/trunk change, replay change, or promotion was performed. Isolated value-head/value-scale tuning is closed; the next branch should investigate joint trunk-plus-value representation learning.
