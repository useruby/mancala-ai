# AlphaZero-Lite 768:768 Value-Search Integration Diagnostic

- source summary: `/tmp/azlite_terminal_outcome_selfplay_iteration`
- candidate: `value_head_only_best_probe`
- current artifact: `model-artifact/current`
- candidate artifact: `/tmp/azlite_terminal_outcome_selfplay_iteration/value_head_only_e2/artifact_value_head_only_e2`

## Aggregate

- changed rows analyzed: `32`
- mean root value delta: `-0.0769`
- mean |root value delta|: `+0.1571`
- mean selected-q delta: `-0.0968`
- mean selected-score delta: `-0.0882`
- mean selected-prior delta: `+0.2426`
- candidate-selected move was teacher move: `+0.4062`
- current-selected move was teacher move: `+0.2500`
- phase counts: `{"late": 5, "mid": 4, "opening": 23}`
- current->candidate move flips: `{"0->4": 7, "1->3": 1, "1->4": 2, "1->5": 1, "2->0": 1, "3->2": 1, "4->1": 1, "4->5": 2, "5->1": 15, "5->3": 1}`

## Changed Cases

| Game | Ply | Phase | Seat | Cur move | Cand move | Delta V | Delta Q(sel) | Delta Score(sel) | Delta Prior(sel) |
|---|---|---|---|---|---|---|---|---|---|
| 948 | 31 | late | player_0 | 4 | 5 | +0.2566 | +0.2962 | +0.2888 | -0.2196 |
| 1004 | 31 | late | player_0 | 4 | 5 | +0.2566 | +0.2962 | +0.2888 | -0.2196 |
| 894 | 27 | mid | player_0 | 4 | 1 | +0.2263 | +0.2255 | +0.2240 | -0.0076 |
| 877 | 15 | mid | player_0 | 2 | 0 | -0.2132 | -0.6349 | -0.6118 | +0.6172 |
| 947 | 39 | late | player_1 | 1 | 3 | +0.2040 | +0.2993 | +0.3045 | +0.1245 |
| 975 | 38 | late | player_1 | 0 | 4 | -0.1600 | -0.2321 | -0.2151 | +0.1787 |
| 888 | 2 | opening | player_1 | 5 | 1 | -0.1536 | -0.1660 | -0.1512 | +0.4780 |
| 892 | 2 | opening | player_1 | 5 | 1 | -0.1536 | -0.1660 | -0.1512 | +0.4780 |
| 906 | 2 | opening | player_1 | 5 | 1 | -0.1536 | -0.1660 | -0.1512 | +0.4780 |
| 913 | 2 | opening | player_1 | 5 | 1 | -0.1536 | -0.1660 | -0.1512 | +0.4780 |

