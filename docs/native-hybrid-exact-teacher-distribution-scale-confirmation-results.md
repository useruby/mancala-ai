# Exact-Teacher Mixed Distribution Scale Confirmation

Classification: `exact_distribution_scale_gate_not_met`.

No checkpoint was promoted. The acceptance gate is unchanged from PR #283.

## Dataset

Lane SHA: `b99ef0559d47d7e7d1b8baf3e458e6075f48e710dfa2c3249bc35e3724bb71f3`. Rows: 5,880 (2,940 opening, 2,940 midgame).

## Arena gate

Passed seeds: 0/3.

| seed | 384:256 effect (delta vs PR #283) | 768:768 effect (delta) | 1200:1200 effect (delta) | 1200:256 effect (delta) | gate |
| --- | --- | --- | --- | --- | --- |
| seed42 | -0.467 (+0.322) | -0.133 (+0.367) | -0.236 (+0.236) | -0.326 (+0.523) | fail |
| seed43 | -0.523 (+0.293) | -0.094 (+0.391) | -0.277 (+0.223) | -0.408 (+0.488) | fail |
| seed44 | -0.484 (+0.016) | -0.246 (+0.164) | -0.191 (+0.207) | -0.639 (+0.238) | fail |

## Decision

The scaled mixed lane did not pass the frozen gate in at least two seeds. Stop exact-teacher training work; do not run a rescue experiment.
