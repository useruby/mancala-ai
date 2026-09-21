# Value-Head Oracle Audit

Classification: `value_head_degradation_explains_mcts_regression`.

This evaluation-only audit uses the unchanged 200-state PR #340 corpus and all six byte-verified PR #351 `fresh_w1`/`fresh_w4` pairs (sources 401, 407, 413, 419, 443, and 449). It performed no training, self-play, replay change, canonical gate, selection, or promotion.

## Context

PR #351 found no clear raw-policy gain from fresh replay upweighting: raw optimal mass was +0.0039 (95% paired CI [-0.0048, 0.0100]) and raw expected regret was -0.0435 ([-0.0909, 0.0274]). The diagnostic arena moved +0.0736, but its CI crossed zero ([-0.0036, 0.1439]). In contrast, MCTS-384 exact-optimal mass regressed by -0.0113 ([-0.0223, -0.0012]).

## Matched Search

All conditions used 384 simulations, `c_puct=1.25`, zero FPU, no subtree reuse, no value normalisation, deterministic root policy, no tactical bias, and no Dirichlet noise. The normal-value results reproduce PR #351 exactly.

| Condition | w1 optimal mass | w4 optimal mass | w1 expected regret | w4 expected regret |
| --- | ---: | ---: | ---: | ---: |
| Network value | 0.6965 | 0.6853 | 1.6385 | 1.6982 |
| Zero nonterminal value | 0.6424 | 0.6445 | 2.1542 | 2.1119 |
| Exact WDL nonterminal value | 0.7096 | 0.7114 | 1.5544 | 1.5192 |

Zeroing nonterminal values hurts both arms, so the learned value head is useful on average. Exact WDL values improve both arms over their normal value heads and reverse the normal w4 deficit: w4-w1 optimal-mass delta is -0.0113 with network values, +0.0021 with zero values, and +0.0019 with exact WDL values.

Root value MAE is worse for w4 (0.4648) than w1 (0.4181), paired w4-w1 delta +0.0467 with 95% CI [0.0340, 0.0629]. One-ply ranking regret is slightly lower for w4 (1.8167 versus 1.9367); this does not support a policy-ranking regression. Thus the attributable signal is degraded root outcome calibration and search use of those values, not a uniform deterioration of one-ply action ranking.

The exact oracle covered 71,951 unique nonterminal leaves. No evaluated leaf was outside tier-21 coverage and no network fallback was used. Perspective conversion uses current-player identity, including Kalah extra turns; terminal outcomes remain PUCT's normal exact game values.

## Paired Bootstrap

Ten thousand paired source-level bootstrap samples used seed 352.

| w4-w1 metric | Estimate | 95% CI |
| --- | ---: | --- |
| Root value MAE | +0.0467 | [0.0340, 0.0629] |
| One-ply value-ranking regret | -0.1200 | [-0.4317, 0.2083] |
| Network-value MCTS optimal mass | -0.0113 | [-0.0223, -0.0012] |
| Zero-value MCTS optimal mass | +0.0021 | [-0.0072, 0.0098] |
| Exact-WDL MCTS optimal mass | +0.0019 | [-0.0047, 0.0068] |
| Network-value expected regret | +0.0597 | [-0.0065, 0.1258] |
| Zero-value expected regret | -0.0423 | [-0.0860, 0.0147] |
| Exact-WDL expected regret | -0.0352 | [-0.0720, 0.0098] |

## Conclusion

The normal w4 regression disappears when nonterminal value use is removed or made exact, while perfect WDL values improve search over normal values. The learned value channel remains beneficial relative to zero, but w4's poorer value calibration is the primary explanation for its MCTS regression under this matched audit.

Next experiment: compare `value_target_mode=default` with the current `value_target_mode=sharpened` using existing self-play only, keeping policy, replay, and search fixed.

Complete provenance, calibration bins/buckets, one-ply rows, PUCT telemetry, paired bootstrap, and the 25 diagnostic disagreements are in `docs/data/alphazero-lite-value-head-oracle-audit/`.
