# PUCT Exact WDL+Margin Leaf Ablation

Primary classification: `exact_margin_leaf_loses_tactical_fix`.

This read-only diagnostic tested the fixed threshold-10 `wdl_margin` leaf
mapping. It did not train, generate self-play, tune search parameters, run a
canonical gate, or promote seed455.

## PR #340 Reproduction

The disabled and flat-WDL baselines reproduced PR #359 exactly enough for the
pre-registered comparison: optimal mass was `0.7047` / `0.6878` and expected
regret was `1.5296` / `1.6358`.

The fixed WDL+margin mapping improved on flat WDL globally, with optimal mass
`0.6978` and expected regret `1.5128`. It did not restore optimal mass to the
disabled value.

## Decisive Rejection

For `sparse_endgame-023`, the exact action margins are move 1 loss `-12`, move
4 draw `0`, and move 5 loss `-6`. Flat WDL selected move 4, but WDL+margin
selected move 5 at 384 simulations. This violates the primary tactical
requirement, so the mapping is rejected regardless of its aggregate PR #340
improvement.

No diagnostic arena was run. The generation remains rejected.
