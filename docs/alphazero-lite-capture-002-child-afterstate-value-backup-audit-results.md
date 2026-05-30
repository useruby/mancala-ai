# AlphaZero-lite Capture 002 Child-Afterstate Value/Backup Audit Results

## Context

- No production training, no arena, and no promotion were run.
- Focus row: `capture_available-002`.
- Preservation controls loaded: `capture_available-003`, `006`, `007`, `008`.
- Summary artifact: `/tmp/azlite_capture_002_child_afterstate_value_backup_audit/capture_002_child_afterstate_value_backup_audit_summary.json`

## Root and child state extraction

| move | gives_extra_turn | side_to_move_after | immediate_store_delta | capture_count | game_over_after_move | root_perspective_conversion | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | true | 1 | 1 | 0 | false | identity: child raw value already uses root-player perspective | extra turn keeps perspective |
| 4 | false | 0 | 1 | 0 | false | sign flip: child raw value is opponent-to-move perspective, negate for root | turn handoff requires sign flip |

## Perspective and backup convention check

- Root player is `1`.
- Move `2` child side to move: `1`.
- Move `4` child side to move: `0`.
- Neural value is evaluated from `state.current_player` perspective and converted back to root perspective with sign `+1 when child current_player == root_player else -1`.
- PUCT backup convention check: `_search negates returned child value exactly when child.game.current_player != parent.game.current_player`.
- Extra-turn audit finding: `move 2 keeps the same player to move in this position; move 4 hands off and therefore must flip sign back to root perspective`.

## Raw neural value audit

| artifact | state | raw_value | root_perspective_value | policy_top_move | policy_move_2 | policy_move_4 | value_child4_minus_child2 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current | root | 0.1135 | 0.1135 | 0 | 0.1299 | 0.1101 | -0.1765 | root value already aligned with root player |
| current | child_after_move_2 | 0.063 | 0.063 | 4 | 0.0 | 0.3249 | -0.1765 | move 2 extra-turn child |
| current | child_after_move_4 | 0.1136 | -0.1136 | 5 | 0.3763 | 0.0908 | -0.1765 | move 4 handoff child |
| guarded-w2 | root | 0.0413 | 0.0413 | 0 | 0.115 | 0.0906 | -0.1336 | root value already aligned with root player |
| guarded-w2 | child_after_move_2 | 0.0613 | 0.0613 | 1 | 0.0 | 0.2386 | -0.1336 | move 2 extra-turn child |
| guarded-w2 | child_after_move_4 | 0.0723 | -0.0723 | 2 | 0.3163 | 0.1938 | -0.1336 | move 4 handoff child |

## Child-afterstate teacher audit

| child_move | budget | child_selected_move | child_value_raw | child_value_root_perspective | visits | q_summary | child4_minus_child2_root_value | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 384 | 5 | 0.7769 | 0.7769 | `{"0": 18, "1": 33, "3": 106, "4": 97, "5": 130}` | `{"0": -0.2778, "1": 0.1515, "3": 0.6981, "4": 0.6804, "5": 0.7769}` | -1.0166 | teacher search from child afterstate |
| 2 | 1200 | 5 | 0.7456 | 0.7456 | `{"0": 27, "1": 69, "3": 392, "4": 260, "5": 452}` | `{"0": -0.3704, "1": 0.1739, "3": 0.7245, "4": 0.6346, "5": 0.7456}` | -1.2074 | teacher search from child afterstate |
| 2 | 2400 | 5 | 0.7161 | 0.7161 | `{"0": 67, "1": 79, "3": 819, "4": 438, "5": 997}` | `{"0": -0.0149, "1": 0.0633, "3": 0.6886, "4": 0.589, "5": 0.7161}` | -1.2537 | teacher search from child afterstate |
| 2 | 5000 | 5 | 0.6968 | 0.6968 | `{"0": 87, "1": 88, "3": 1890, "4": 788, "5": 2147}` | `{"0": -0.0115, "1": -0.0114, "3": 0.6852, "4": 0.5799, "5": 0.6968}` | -1.2626 | teacher search from child afterstate |
| 4 | 384 | 3 | 0.2397 | -0.2397 | `{"0": 16, "1": 24, "2": 108, "3": 121, "4": 74, "5": 41}` | `{"0": -0.875, "1": -0.5833, "2": 0.2037, "3": 0.2397, "4": 0.0541, "5": -0.2195}` | -1.0166 | teacher search from child afterstate |
| 4 | 1200 | 2 | 0.4618 | -0.4618 | `{"0": 36, "1": 44, "2": 641, "3": 198, "4": 186, "5": 95}` | `{"0": -0.5278, "1": -0.3864, "2": 0.4618, "3": 0.2121, "4": 0.1935, "5": -0.0316}` | -1.2074 | teacher search from child afterstate |
| 4 | 2400 | 2 | 0.5376 | -0.5376 | `{"0": 58, "1": 50, "2": 1622, "3": 279, "4": 292, "5": 99}` | `{"0": -0.3103, "1": -0.4, "2": 0.5376, "3": 0.2616, "4": 0.2877, "5": -0.0606}` | -1.2537 | teacher search from child afterstate |
| 4 | 5000 | 2 | 0.5658 | -0.5658 | `{"0": 63, "1": 55, "2": 2863, "3": 352, "4": 1555, "5": 112}` | `{"0": -0.3333, "1": -0.4182, "2": 0.5658, "3": 0.2642, "4": 0.4926, "5": -0.0804}` | -1.2626 | teacher search from child afterstate |

## Child-afterstate PUCT audit

| artifact | child_move | simulations | child_selected_move | child_value_raw | child_value_root_perspective | visit_entropy | q_summary | child4_minus_child2_root_value | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current | 2 | 64 | 3 | 0.063 | 0.063 | 0.7305 | `{"0": -0.1554, "1": -0.1741, "3": 0.0436, "4": 0.0355, "5": 0.0672}` | -0.1766 | direct artifact PUCT from child afterstate |
| current | 2 | 128 | 5 | 0.063 | 0.063 | 0.6854 | `{"0": -0.1554, "1": -0.01, "3": 0.0436, "4": 0.0355, "5": 0.1043}` | -0.1766 | direct artifact PUCT from child afterstate |
| current | 2 | 384 | 5 | 0.063 | 0.063 | 0.3985 | `{"0": -0.1554, "1": -0.0141, "3": 0.0436, "4": 0.0272, "5": 0.0827}` | -0.1766 | direct artifact PUCT from child afterstate |
| current | 2 | 1200 | 5 | 0.063 | 0.063 | 0.1844 | `{"0": -0.0501, "1": -0.0163, "3": 0.0436, "4": 0.0136, "5": 0.1157}` | -0.1766 | direct artifact PUCT from child afterstate |
| current | 4 | 64 | 5 | 0.1136 | -0.1136 | 0.3438 | `{"0": 0.0, "1": -0.0472, "2": -0.1455, "3": -0.2038, "4": 0.0367, "5": -0.0056}` | -0.1766 | direct artifact PUCT from child afterstate |
| current | 4 | 128 | 4 | 0.1136 | -0.1136 | 0.5488 | `{"0": -0.0915, "1": -0.0472, "2": -0.1234, "3": -0.2038, "4": 0.0298, "5": -0.0056}` | -0.1766 | direct artifact PUCT from child afterstate |
| current | 4 | 384 | 5 | 0.1136 | -0.1136 | 0.503 | `{"0": -0.0915, "1": -0.0843, "2": -0.0471, "3": -0.2038, "4": -0.0269, "5": -0.0296}` | -0.1766 | direct artifact PUCT from child afterstate |
| current | 4 | 1200 | 4 | 0.1136 | -0.1136 | 0.6405 | `{"0": -0.0915, "1": -0.0728, "2": -0.0722, "3": -0.1479, "4": -0.0663, "5": -0.0699}` | -0.1766 | direct artifact PUCT from child afterstate |
| guarded-w2 | 2 | 64 | 4 | 0.0613 | 0.0613 | 0.6512 | `{"0": -0.058, "1": -0.0603, "3": -0.0034, "4": -0.0068, "5": -0.0386}` | -0.1336 | direct artifact PUCT from child afterstate |
| guarded-w2 | 2 | 128 | 4 | 0.0613 | 0.0613 | 0.7477 | `{"0": -0.058, "1": -0.0603, "3": -0.0176, "4": -0.0188, "5": 0.0303}` | -0.1336 | direct artifact PUCT from child afterstate |
| guarded-w2 | 2 | 384 | 5 | 0.0613 | 0.0613 | 0.4949 | `{"0": -0.0573, "1": -0.0836, "3": -0.0176, "4": -0.0188, "5": 0.0775}` | -0.1336 | direct artifact PUCT from child afterstate |
| guarded-w2 | 2 | 1200 | 5 | 0.0613 | 0.0613 | 0.2461 | `{"0": -0.0555, "1": -0.0172, "3": -0.0176, "4": -0.0188, "5": 0.0973}` | -0.1336 | direct artifact PUCT from child afterstate |
| guarded-w2 | 4 | 64 | 5 | 0.0723 | -0.0723 | 0.6032 | `{"0": 0.0027, "1": -0.0579, "2": -0.035, "3": -0.0634, "4": 0.0097, "5": 0.0554}` | -0.1336 | direct artifact PUCT from child afterstate |
| guarded-w2 | 4 | 128 | 5 | 0.0723 | -0.0723 | 0.4895 | `{"0": 0.0027, "1": -0.0579, "2": 0.0069, "3": -0.0634, "4": 0.0045, "5": 0.0379}` | -0.1336 | direct artifact PUCT from child afterstate |
| guarded-w2 | 4 | 384 | 5 | 0.0723 | -0.0723 | 0.4389 | `{"0": 0.0101, "1": -0.0579, "2": -0.0101, "3": 0.0326, "4": 0.0124, "5": 0.0279}` | -0.1336 | direct artifact PUCT from child afterstate |
| guarded-w2 | 4 | 1200 | 5 | 0.0723 | -0.0723 | 0.4097 | `{"0": 0.0101, "1": -0.0139, "2": -0.0212, "3": 0.0231, "4": 0.0094, "5": 0.0279}` | -0.1336 | direct artifact PUCT from child afterstate |

## Root selection-score trace with child-value annotations

| artifact | simulations | selected_move | n_move_2 | n_move_4 | p_move_2 | p_move_4 | q_move_2 | q_move_4 | u_move_2 | u_move_4 | score_move_2 | score_move_4 | score_margin_2_minus_4 | q_margin_4_minus_2 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current | 1 | 0 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.221 | 0.1059 | 0.1331 | 0.018 | 0.1151 | 0.0 | move_2_score_gt_move_4 |
| current | 2 | 1 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.3126 | 0.1498 | 0.8126 | 0.6498 | 0.1628 | 0.0 | move_2_score_gt_move_4 |
| current | 3 | 1 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.3828 | 0.1834 | 1.0495 | 0.8501 | 0.1994 | 0.0 | move_2_score_gt_move_4 |
| current | 4 | 1 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.4421 | 0.2118 | 1.1921 | 0.9618 | 0.2303 | 0.0 | move_2_score_gt_move_4 |
| current | 5 | 1 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.4942 | 0.2368 | 1.2942 | 1.0368 | 0.2575 | 0.0 | move_2_score_gt_move_4 |
| current | 6 | 1 | 1 | 0 | 0.1768 | 0.0847 | 0.063 | 0.0 | 0.2707 | 0.2594 | 1.2707 | 1.0229 | 0.2478 | -0.063 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 7 | 1 | 2 | 0 | 0.1768 | 0.0847 | 0.0467 | 0.0 | 0.1949 | 0.2802 | 1.1918 | 1.1364 | 0.0554 | -0.0467 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 8 | 1 | 2 | 0 | 0.1768 | 0.0847 | 0.0467 | 0.0 | 0.2084 | 0.2995 | 1.0972 | 1.1467 | -0.0495 | -0.0467 | move_2_q_gt_move_4 |
| current | 9 | 1 | 2 | 0 | 0.1768 | 0.0847 | 0.0467 | 0.0 | 0.221 | 0.3177 | 1.1212 | 1.1844 | -0.0632 | -0.0467 | move_2_q_gt_move_4 |
| current | 10 | 1 | 2 | 1 | 0.1768 | 0.0847 | 0.0467 | -0.1136 | 0.233 | 0.1674 | 1.1478 | 0.1674 | 0.9803 | -0.1603 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 11 | 1 | 2 | 1 | 0.1768 | 0.0847 | 0.0467 | -0.1136 | 0.2444 | 0.1756 | 1.1887 | 0.1756 | 1.0131 | -0.1603 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 12 | 1 | 3 | 1 | 0.1768 | 0.0847 | 0.0582 | -0.1136 | 0.1914 | 0.1834 | 1.1914 | 0.1834 | 1.008 | -0.1718 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 13 | 1 | 4 | 1 | 0.1768 | 0.0847 | 0.0303 | -0.1136 | 0.1594 | 0.1909 | 1.0069 | 0.1909 | 0.816 | -0.1439 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 14 | 1 | 4 | 1 | 0.1768 | 0.0847 | 0.0303 | -0.1136 | 0.1654 | 0.1981 | 1.1287 | 0.1981 | 0.9306 | -0.1439 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 15 | 1 | 4 | 1 | 0.1768 | 0.0847 | 0.0303 | -0.1136 | 0.1712 | 0.2051 | 1.1712 | 0.2051 | 0.9662 | -0.1439 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 16 | 1 | 5 | 1 | 0.1768 | 0.0847 | 0.0286 | -0.1136 | 0.1474 | 0.2118 | 1.1474 | 0.2118 | 0.9356 | -0.1422 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 17 | 1 | 6 | 1 | 0.1768 | 0.0847 | 0.0398 | -0.1136 | 0.1302 | 0.2183 | 1.1302 | 0.2183 | 0.9119 | -0.1533 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 18 | 1 | 6 | 1 | 0.1768 | 0.0847 | 0.0398 | -0.1136 | 0.134 | 0.2246 | 1.134 | 0.2246 | 0.9093 | -0.1533 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 19 | 1 | 6 | 1 | 0.1768 | 0.0847 | 0.0398 | -0.1136 | 0.1376 | 0.2308 | 1.1376 | 0.2308 | 0.9069 | -0.1533 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 20 | 1 | 7 | 1 | 0.1768 | 0.0847 | 0.0565 | -0.1136 | 0.1236 | 0.2368 | 1.1236 | 0.2368 | 0.8868 | -0.17 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 21 | 1 | 8 | 1 | 0.1768 | 0.0847 | 0.0535 | -0.1136 | 0.1125 | 0.2426 | 1.1125 | 0.2426 | 0.8699 | -0.1671 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 22 | 2 | 9 | 1 | 0.1768 | 0.0847 | 0.0674 | -0.1136 | 0.1037 | 0.2483 | 1.1037 | 0.2483 | 0.8553 | -0.1809 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 23 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.0964 | 0.2539 | 1.0964 | 0.2539 | 0.8425 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 24 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.0984 | 0.2594 | 1.0984 | 0.5113 | 0.5872 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 25 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1005 | 0.2647 | 1.1005 | 0.5166 | 0.5839 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 26 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1025 | 0.27 | 1.1025 | 0.5219 | 0.5806 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 27 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1044 | 0.2751 | 1.1044 | 0.527 | 0.5774 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 28 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1063 | 0.2802 | 1.1063 | 0.532 | 0.5743 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 29 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1082 | 0.2851 | 1.0859 | 0.5314 | 0.5545 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 30 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1101 | 0.29 | 1.0611 | 0.5296 | 0.5316 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 31 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1119 | 0.2948 | 1.1119 | 0.5467 | 0.5652 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 32 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1137 | 0.2995 | 1.1137 | 0.5514 | 0.5623 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 33 | 1 | 11 | 1 | 0.1768 | 0.0847 | 0.046 | -0.1136 | 0.1058 | 0.3041 | 1.1058 | 0.5528 | 0.553 | -0.1595 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 34 | 1 | 11 | 1 | 0.1768 | 0.0847 | 0.046 | -0.1136 | 0.1074 | 0.3087 | 1.1074 | 0.5574 | 0.5501 | -0.1595 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 35 | 1 | 12 | 1 | 0.1768 | 0.0847 | 0.0464 | -0.1136 | 0.1006 | 0.3132 | 1.1006 | 0.5614 | 0.5392 | -0.1599 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 36 | 1 | 13 | 1 | 0.1768 | 0.0847 | 0.0308 | -0.1136 | 0.0947 | 0.3177 | 1.0947 | 0.5854 | 0.5093 | -0.1444 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 37 | 1 | 13 | 1 | 0.1768 | 0.0847 | 0.0308 | -0.1136 | 0.096 | 0.3221 | 1.096 | 0.5898 | 0.5063 | -0.1444 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 38 | 1 | 14 | 1 | 0.1768 | 0.0847 | 0.0309 | -0.1136 | 0.0908 | 0.3264 | 1.0908 | 0.594 | 0.4968 | -0.1445 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 39 | 1 | 15 | 1 | 0.1768 | 0.0847 | 0.0431 | -0.1136 | 0.0863 | 0.3306 | 1.0863 | 0.5827 | 0.5036 | -0.1567 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 40 | 2 | 16 | 1 | 0.1768 | 0.0847 | 0.0579 | -0.1136 | 0.0822 | 0.3349 | 1.0822 | 0.5703 | 0.512 | -0.1715 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 41 | 2 | 17 | 1 | 0.1768 | 0.0847 | 0.0606 | -0.1136 | 0.0786 | 0.339 | 1.0786 | 0.5716 | 0.507 | -0.1742 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 42 | 2 | 18 | 1 | 0.1768 | 0.0847 | 0.0766 | -0.1136 | 0.0754 | 0.3431 | 1.0754 | 0.5604 | 0.5149 | -0.1901 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 43 | 2 | 19 | 1 | 0.1768 | 0.0847 | 0.0824 | -0.1136 | 0.0725 | 0.3472 | 1.0725 | 0.5594 | 0.513 | -0.1959 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 44 | 2 | 20 | 1 | 0.1768 | 0.0847 | 0.0841 | -0.1136 | 0.0698 | 0.3512 | 1.0698 | 0.562 | 0.5079 | -0.1977 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 45 | 2 | 21 | 1 | 0.1768 | 0.0847 | 0.0763 | -0.1136 | 0.0674 | 0.3552 | 1.0674 | 0.5727 | 0.4947 | -0.1899 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 46 | 2 | 22 | 1 | 0.1768 | 0.0847 | 0.0742 | -0.1136 | 0.0652 | 0.3591 | 1.0652 | 0.5786 | 0.4866 | -0.1877 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 47 | 2 | 23 | 1 | 0.1768 | 0.0847 | 0.0656 | -0.1136 | 0.0631 | 0.363 | 1.0631 | 0.5906 | 0.4726 | -0.1792 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 48 | 2 | 24 | 1 | 0.1768 | 0.0847 | 0.0591 | -0.1136 | 0.0613 | 0.3668 | 1.0613 | 0.601 | 0.4602 | -0.1726 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 49 | 2 | 25 | 1 | 0.1768 | 0.0847 | 0.0563 | -0.1136 | 0.0595 | 0.3706 | 1.0595 | 0.6078 | 0.4517 | -0.1698 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 50 | 2 | 25 | 1 | 0.1768 | 0.0847 | 0.0563 | -0.1136 | 0.0601 | 0.3744 | 1.0601 | 0.6115 | 0.4486 | -0.1698 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 51 | 2 | 26 | 1 | 0.1768 | 0.0847 | 0.0514 | -0.1136 | 0.0585 | 0.3781 | 1.0585 | 0.6205 | 0.4379 | -0.165 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 52 | 2 | 27 | 1 | 0.1768 | 0.0847 | 0.0549 | -0.1136 | 0.0569 | 0.3818 | 1.0569 | 0.6204 | 0.4365 | -0.1684 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 53 | 2 | 28 | 1 | 0.1768 | 0.0847 | 0.057 | -0.1136 | 0.0555 | 0.3854 | 1.0555 | 0.6218 | 0.4337 | -0.1706 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 54 | 2 | 29 | 1 | 0.1768 | 0.0847 | 0.0642 | -0.1136 | 0.0541 | 0.3891 | 1.0541 | 0.6181 | 0.4361 | -0.1777 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 55 | 2 | 30 | 1 | 0.1768 | 0.0847 | 0.0656 | -0.1136 | 0.0529 | 0.3927 | 1.0529 | 0.6202 | 0.4326 | -0.1792 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 56 | 2 | 31 | 1 | 0.1768 | 0.0847 | 0.0705 | -0.1136 | 0.0517 | 0.3962 | 1.0517 | 0.6191 | 0.4326 | -0.1841 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 57 | 2 | 32 | 1 | 0.1768 | 0.0847 | 0.0761 | -0.1136 | 0.0506 | 0.3997 | 1.0506 | 0.6175 | 0.4331 | -0.1896 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 58 | 2 | 33 | 1 | 0.1768 | 0.0847 | 0.0806 | -0.1136 | 0.0495 | 0.4032 | 1.0495 | 0.617 | 0.4325 | -0.1941 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 59 | 2 | 34 | 1 | 0.1768 | 0.0847 | 0.0729 | -0.1136 | 0.0485 | 0.4067 | 1.0485 | 0.6273 | 0.4212 | -0.1865 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 60 | 2 | 35 | 1 | 0.1768 | 0.0847 | 0.0679 | -0.1136 | 0.0476 | 0.4101 | 1.0476 | 0.6354 | 0.4121 | -0.1815 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 61 | 2 | 36 | 1 | 0.1768 | 0.0847 | 0.0654 | -0.1136 | 0.0467 | 0.4135 | 1.0467 | 0.6413 | 0.4054 | -0.179 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 62 | 2 | 37 | 1 | 0.1768 | 0.0847 | 0.0625 | -0.1136 | 0.0458 | 0.4169 | 1.0458 | 0.6476 | 0.3982 | -0.176 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 63 | 2 | 38 | 1 | 0.1768 | 0.0847 | 0.0536 | -0.1136 | 0.045 | 0.4202 | 1.045 | 0.6603 | 0.3847 | -0.1671 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 64 | 2 | 38 | 1 | 0.1768 | 0.0847 | 0.0536 | -0.1136 | 0.0453 | 0.4236 | 1.0453 | 0.6636 | 0.3817 | -0.1671 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1 | 0 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.221 | 0.1059 | 0.1331 | 0.018 | 0.1151 | 0.0 | move_2_score_gt_move_4 |
| current | 2 | 1 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.3126 | 0.1498 | 0.8126 | 0.6498 | 0.1628 | 0.0 | move_2_score_gt_move_4 |
| current | 3 | 1 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.3828 | 0.1834 | 1.0495 | 0.8501 | 0.1994 | 0.0 | move_2_score_gt_move_4 |
| current | 4 | 1 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.4421 | 0.2118 | 1.1921 | 0.9618 | 0.2303 | 0.0 | move_2_score_gt_move_4 |
| current | 5 | 1 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.4942 | 0.2368 | 1.2942 | 1.0368 | 0.2575 | 0.0 | move_2_score_gt_move_4 |
| current | 6 | 1 | 1 | 0 | 0.1768 | 0.0847 | 0.063 | 0.0 | 0.2707 | 0.2594 | 1.2707 | 1.0229 | 0.2478 | -0.063 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 7 | 1 | 2 | 0 | 0.1768 | 0.0847 | 0.0467 | 0.0 | 0.1949 | 0.2802 | 1.1918 | 1.1364 | 0.0554 | -0.0467 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 8 | 1 | 2 | 0 | 0.1768 | 0.0847 | 0.0467 | 0.0 | 0.2084 | 0.2995 | 1.0972 | 1.1467 | -0.0495 | -0.0467 | move_2_q_gt_move_4 |
| current | 9 | 1 | 2 | 0 | 0.1768 | 0.0847 | 0.0467 | 0.0 | 0.221 | 0.3177 | 1.1212 | 1.1844 | -0.0632 | -0.0467 | move_2_q_gt_move_4 |
| current | 10 | 1 | 2 | 1 | 0.1768 | 0.0847 | 0.0467 | -0.1136 | 0.233 | 0.1674 | 1.1478 | 0.1674 | 0.9803 | -0.1603 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 11 | 1 | 2 | 1 | 0.1768 | 0.0847 | 0.0467 | -0.1136 | 0.2444 | 0.1756 | 1.1887 | 0.1756 | 1.0131 | -0.1603 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 12 | 1 | 3 | 1 | 0.1768 | 0.0847 | 0.0582 | -0.1136 | 0.1914 | 0.1834 | 1.1914 | 0.1834 | 1.008 | -0.1718 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 13 | 1 | 4 | 1 | 0.1768 | 0.0847 | 0.0303 | -0.1136 | 0.1594 | 0.1909 | 1.0069 | 0.1909 | 0.816 | -0.1439 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 14 | 1 | 4 | 1 | 0.1768 | 0.0847 | 0.0303 | -0.1136 | 0.1654 | 0.1981 | 1.1287 | 0.1981 | 0.9306 | -0.1439 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 15 | 1 | 4 | 1 | 0.1768 | 0.0847 | 0.0303 | -0.1136 | 0.1712 | 0.2051 | 1.1712 | 0.2051 | 0.9662 | -0.1439 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 16 | 1 | 5 | 1 | 0.1768 | 0.0847 | 0.0286 | -0.1136 | 0.1474 | 0.2118 | 1.1474 | 0.2118 | 0.9356 | -0.1422 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 17 | 1 | 6 | 1 | 0.1768 | 0.0847 | 0.0398 | -0.1136 | 0.1302 | 0.2183 | 1.1302 | 0.2183 | 0.9119 | -0.1533 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 18 | 1 | 6 | 1 | 0.1768 | 0.0847 | 0.0398 | -0.1136 | 0.134 | 0.2246 | 1.134 | 0.2246 | 0.9093 | -0.1533 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 19 | 1 | 6 | 1 | 0.1768 | 0.0847 | 0.0398 | -0.1136 | 0.1376 | 0.2308 | 1.1376 | 0.2308 | 0.9069 | -0.1533 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 20 | 1 | 7 | 1 | 0.1768 | 0.0847 | 0.0565 | -0.1136 | 0.1236 | 0.2368 | 1.1236 | 0.2368 | 0.8868 | -0.17 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 21 | 1 | 8 | 1 | 0.1768 | 0.0847 | 0.0535 | -0.1136 | 0.1125 | 0.2426 | 1.1125 | 0.2426 | 0.8699 | -0.1671 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 22 | 2 | 9 | 1 | 0.1768 | 0.0847 | 0.0674 | -0.1136 | 0.1037 | 0.2483 | 1.1037 | 0.2483 | 0.8553 | -0.1809 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 23 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.0964 | 0.2539 | 1.0964 | 0.2539 | 0.8425 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 24 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.0984 | 0.2594 | 1.0984 | 0.5113 | 0.5872 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 25 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1005 | 0.2647 | 1.1005 | 0.5166 | 0.5839 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 26 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1025 | 0.27 | 1.1025 | 0.5219 | 0.5806 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 27 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1044 | 0.2751 | 1.1044 | 0.527 | 0.5774 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 28 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1063 | 0.2802 | 1.1063 | 0.532 | 0.5743 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 29 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1082 | 0.2851 | 1.0859 | 0.5314 | 0.5545 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 30 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1101 | 0.29 | 1.0611 | 0.5296 | 0.5316 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 31 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1119 | 0.2948 | 1.1119 | 0.5467 | 0.5652 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 32 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1137 | 0.2995 | 1.1137 | 0.5514 | 0.5623 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 33 | 1 | 11 | 1 | 0.1768 | 0.0847 | 0.046 | -0.1136 | 0.1058 | 0.3041 | 1.1058 | 0.5528 | 0.553 | -0.1595 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 34 | 1 | 11 | 1 | 0.1768 | 0.0847 | 0.046 | -0.1136 | 0.1074 | 0.3087 | 1.1074 | 0.5574 | 0.5501 | -0.1595 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 35 | 1 | 12 | 1 | 0.1768 | 0.0847 | 0.0464 | -0.1136 | 0.1006 | 0.3132 | 1.1006 | 0.5614 | 0.5392 | -0.1599 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 36 | 1 | 13 | 1 | 0.1768 | 0.0847 | 0.0308 | -0.1136 | 0.0947 | 0.3177 | 1.0947 | 0.5854 | 0.5093 | -0.1444 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 37 | 1 | 13 | 1 | 0.1768 | 0.0847 | 0.0308 | -0.1136 | 0.096 | 0.3221 | 1.096 | 0.5898 | 0.5063 | -0.1444 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 38 | 1 | 14 | 1 | 0.1768 | 0.0847 | 0.0309 | -0.1136 | 0.0908 | 0.3264 | 1.0908 | 0.594 | 0.4968 | -0.1445 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 39 | 1 | 15 | 1 | 0.1768 | 0.0847 | 0.0431 | -0.1136 | 0.0863 | 0.3306 | 1.0863 | 0.5827 | 0.5036 | -0.1567 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 40 | 2 | 16 | 1 | 0.1768 | 0.0847 | 0.0579 | -0.1136 | 0.0822 | 0.3349 | 1.0822 | 0.5703 | 0.512 | -0.1715 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 41 | 2 | 17 | 1 | 0.1768 | 0.0847 | 0.0606 | -0.1136 | 0.0786 | 0.339 | 1.0786 | 0.5716 | 0.507 | -0.1742 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 42 | 2 | 18 | 1 | 0.1768 | 0.0847 | 0.0766 | -0.1136 | 0.0754 | 0.3431 | 1.0754 | 0.5604 | 0.5149 | -0.1901 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 43 | 2 | 19 | 1 | 0.1768 | 0.0847 | 0.0824 | -0.1136 | 0.0725 | 0.3472 | 1.0725 | 0.5594 | 0.513 | -0.1959 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 44 | 2 | 20 | 1 | 0.1768 | 0.0847 | 0.0841 | -0.1136 | 0.0698 | 0.3512 | 1.0698 | 0.562 | 0.5079 | -0.1977 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 45 | 2 | 21 | 1 | 0.1768 | 0.0847 | 0.0763 | -0.1136 | 0.0674 | 0.3552 | 1.0674 | 0.5727 | 0.4947 | -0.1899 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 46 | 2 | 22 | 1 | 0.1768 | 0.0847 | 0.0742 | -0.1136 | 0.0652 | 0.3591 | 1.0652 | 0.5786 | 0.4866 | -0.1877 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 47 | 2 | 23 | 1 | 0.1768 | 0.0847 | 0.0656 | -0.1136 | 0.0631 | 0.363 | 1.0631 | 0.5906 | 0.4726 | -0.1792 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 48 | 2 | 24 | 1 | 0.1768 | 0.0847 | 0.0591 | -0.1136 | 0.0613 | 0.3668 | 1.0613 | 0.601 | 0.4602 | -0.1726 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 49 | 2 | 25 | 1 | 0.1768 | 0.0847 | 0.0563 | -0.1136 | 0.0595 | 0.3706 | 1.0595 | 0.6078 | 0.4517 | -0.1698 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 50 | 2 | 25 | 1 | 0.1768 | 0.0847 | 0.0563 | -0.1136 | 0.0601 | 0.3744 | 1.0601 | 0.6115 | 0.4486 | -0.1698 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 51 | 2 | 26 | 1 | 0.1768 | 0.0847 | 0.0514 | -0.1136 | 0.0585 | 0.3781 | 1.0585 | 0.6205 | 0.4379 | -0.165 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 52 | 2 | 27 | 1 | 0.1768 | 0.0847 | 0.0549 | -0.1136 | 0.0569 | 0.3818 | 1.0569 | 0.6204 | 0.4365 | -0.1684 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 53 | 2 | 28 | 1 | 0.1768 | 0.0847 | 0.057 | -0.1136 | 0.0555 | 0.3854 | 1.0555 | 0.6218 | 0.4337 | -0.1706 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 54 | 2 | 29 | 1 | 0.1768 | 0.0847 | 0.0642 | -0.1136 | 0.0541 | 0.3891 | 1.0541 | 0.6181 | 0.4361 | -0.1777 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 55 | 2 | 30 | 1 | 0.1768 | 0.0847 | 0.0656 | -0.1136 | 0.0529 | 0.3927 | 1.0529 | 0.6202 | 0.4326 | -0.1792 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 56 | 2 | 31 | 1 | 0.1768 | 0.0847 | 0.0705 | -0.1136 | 0.0517 | 0.3962 | 1.0517 | 0.6191 | 0.4326 | -0.1841 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 57 | 2 | 32 | 1 | 0.1768 | 0.0847 | 0.0761 | -0.1136 | 0.0506 | 0.3997 | 1.0506 | 0.6175 | 0.4331 | -0.1896 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 58 | 2 | 33 | 1 | 0.1768 | 0.0847 | 0.0806 | -0.1136 | 0.0495 | 0.4032 | 1.0495 | 0.617 | 0.4325 | -0.1941 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 59 | 2 | 34 | 1 | 0.1768 | 0.0847 | 0.0729 | -0.1136 | 0.0485 | 0.4067 | 1.0485 | 0.6273 | 0.4212 | -0.1865 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 60 | 2 | 35 | 1 | 0.1768 | 0.0847 | 0.0679 | -0.1136 | 0.0476 | 0.4101 | 1.0476 | 0.6354 | 0.4121 | -0.1815 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 61 | 2 | 36 | 1 | 0.1768 | 0.0847 | 0.0654 | -0.1136 | 0.0467 | 0.4135 | 1.0467 | 0.6413 | 0.4054 | -0.179 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 62 | 2 | 37 | 1 | 0.1768 | 0.0847 | 0.0625 | -0.1136 | 0.0458 | 0.4169 | 1.0458 | 0.6476 | 0.3982 | -0.176 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 63 | 2 | 38 | 1 | 0.1768 | 0.0847 | 0.0536 | -0.1136 | 0.045 | 0.4202 | 1.045 | 0.6603 | 0.3847 | -0.1671 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 64 | 2 | 38 | 1 | 0.1768 | 0.0847 | 0.0536 | -0.1136 | 0.0453 | 0.4236 | 1.0453 | 0.6636 | 0.3817 | -0.1671 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 65 | 2 | 39 | 1 | 0.1768 | 0.0847 | 0.0487 | -0.1136 | 0.0446 | 0.4269 | 1.0446 | 0.6723 | 0.3722 | -0.1623 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 66 | 2 | 40 | 1 | 0.1768 | 0.0847 | 0.0439 | -0.1136 | 0.0438 | 0.4301 | 1.0438 | 0.6813 | 0.3625 | -0.1574 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 67 | 2 | 41 | 1 | 0.1768 | 0.0847 | 0.0404 | -0.1136 | 0.0431 | 0.4334 | 1.0431 | 0.6888 | 0.3543 | -0.1539 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 68 | 2 | 42 | 1 | 0.1768 | 0.0847 | 0.0424 | -0.1136 | 0.0424 | 0.4366 | 1.0424 | 0.6895 | 0.3529 | -0.156 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 69 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0417 | 0.4398 | 1.0417 | 0.6941 | 0.3476 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 70 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.042 | 0.443 | 1.042 | 0.6973 | 0.3448 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 71 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0423 | 0.4461 | 1.0423 | 0.7004 | 0.3419 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 72 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0426 | 0.4493 | 1.0426 | 0.7035 | 0.3391 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 73 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0429 | 0.4524 | 1.0429 | 0.7067 | 0.3363 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 74 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0432 | 0.4555 | 1.0432 | 0.7097 | 0.3335 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 75 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0435 | 0.4585 | 1.0435 | 0.7128 | 0.3307 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 76 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0438 | 0.4616 | 1.0438 | 0.7159 | 0.3279 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 77 | 2 | 44 | 1 | 0.1768 | 0.0847 | 0.0372 | -0.1136 | 0.0431 | 0.4646 | 1.0431 | 0.7239 | 0.3192 | -0.1508 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 78 | 2 | 44 | 1 | 0.1768 | 0.0847 | 0.0372 | -0.1136 | 0.0434 | 0.4676 | 1.0434 | 0.7269 | 0.3165 | -0.1508 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 79 | 2 | 45 | 1 | 0.1768 | 0.0847 | 0.0344 | -0.1136 | 0.0427 | 0.4706 | 1.0427 | 0.7336 | 0.3091 | -0.148 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 80 | 2 | 46 | 1 | 0.1768 | 0.0847 | 0.0299 | -0.1136 | 0.0421 | 0.4736 | 1.0421 | 0.7425 | 0.2995 | -0.1435 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 81 | 2 | 47 | 1 | 0.1768 | 0.0847 | 0.0279 | -0.1136 | 0.0414 | 0.4765 | 1.0414 | 0.7483 | 0.2931 | -0.1414 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 82 | 2 | 48 | 1 | 0.1768 | 0.0847 | 0.0247 | -0.1136 | 0.0408 | 0.4794 | 1.0408 | 0.7558 | 0.285 | -0.1382 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 83 | 2 | 48 | 1 | 0.1768 | 0.0847 | 0.0247 | -0.1136 | 0.0411 | 0.4824 | 1.0411 | 0.7587 | 0.2824 | -0.1382 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 84 | 2 | 49 | 1 | 0.1768 | 0.0847 | 0.026 | -0.1136 | 0.0405 | 0.4852 | 1.0405 | 0.7598 | 0.2808 | -0.1395 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 85 | 2 | 50 | 1 | 0.1768 | 0.0847 | 0.028 | -0.1136 | 0.04 | 0.4881 | 1.04 | 0.7598 | 0.2802 | -0.1416 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 86 | 2 | 51 | 1 | 0.1768 | 0.0847 | 0.0297 | -0.1136 | 0.0394 | 0.491 | 1.0394 | 0.7603 | 0.2791 | -0.1433 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 87 | 2 | 52 | 1 | 0.1768 | 0.0847 | 0.0326 | -0.1136 | 0.0389 | 0.4938 | 1.0389 | 0.7592 | 0.2797 | -0.1462 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 88 | 2 | 53 | 1 | 0.1768 | 0.0847 | 0.0372 | -0.1136 | 0.0384 | 0.4967 | 1.0384 | 0.7561 | 0.2823 | -0.1507 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 89 | 2 | 54 | 1 | 0.1768 | 0.0847 | 0.0383 | -0.1136 | 0.0379 | 0.4995 | 1.0379 | 0.7575 | 0.2805 | -0.1518 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 90 | 2 | 55 | 1 | 0.1768 | 0.0847 | 0.0333 | -0.1136 | 0.0374 | 0.5023 | 1.0374 | 0.7668 | 0.2707 | -0.1468 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 91 | 2 | 56 | 1 | 0.1768 | 0.0847 | 0.0292 | -0.1136 | 0.037 | 0.5051 | 1.037 | 0.775 | 0.262 | -0.1428 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 92 | 2 | 57 | 1 | 0.1768 | 0.0847 | 0.0264 | -0.1136 | 0.0366 | 0.5078 | 1.0366 | 0.7817 | 0.2548 | -0.14 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 93 | 2 | 57 | 1 | 0.1768 | 0.0847 | 0.0264 | -0.1136 | 0.0368 | 0.5106 | 1.0368 | 0.7845 | 0.2523 | -0.14 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 94 | 2 | 58 | 1 | 0.1768 | 0.0847 | 0.025 | -0.1136 | 0.0363 | 0.5133 | 1.0363 | 0.7892 | 0.2472 | -0.1386 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 95 | 2 | 59 | 1 | 0.1768 | 0.0847 | 0.0289 | -0.1136 | 0.0359 | 0.516 | 1.0359 | 0.7865 | 0.2494 | -0.1424 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 96 | 2 | 60 | 1 | 0.1768 | 0.0847 | 0.028 | -0.1136 | 0.0355 | 0.5188 | 1.0355 | 0.7904 | 0.2451 | -0.1415 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 97 | 2 | 61 | 1 | 0.1768 | 0.0847 | 0.0276 | -0.1136 | 0.0351 | 0.5214 | 1.0351 | 0.7936 | 0.2415 | -0.1412 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 98 | 2 | 62 | 1 | 0.1768 | 0.0847 | 0.0249 | -0.1136 | 0.0347 | 0.5241 | 1.0347 | 0.8002 | 0.2346 | -0.1384 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 99 | 2 | 63 | 1 | 0.1768 | 0.0847 | 0.0249 | -0.1136 | 0.0344 | 0.5268 | 1.0344 | 0.8029 | 0.2315 | -0.1384 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 100 | 2 | 64 | 1 | 0.1768 | 0.0847 | 0.024 | -0.1136 | 0.034 | 0.5295 | 1.034 | 0.8068 | 0.2272 | -0.1375 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 101 | 2 | 65 | 1 | 0.1768 | 0.0847 | 0.0237 | -0.1136 | 0.0337 | 0.5321 | 1.0337 | 0.8098 | 0.2238 | -0.1373 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 102 | 2 | 66 | 1 | 0.1768 | 0.0847 | 0.0198 | -0.1136 | 0.0333 | 0.5347 | 1.0333 | 0.8183 | 0.215 | -0.1333 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 103 | 2 | 67 | 1 | 0.1768 | 0.0847 | 0.02 | -0.1136 | 0.033 | 0.5373 | 1.033 | 0.8206 | 0.2124 | -0.1336 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 104 | 2 | 68 | 1 | 0.1768 | 0.0847 | 0.0236 | -0.1136 | 0.0327 | 0.5399 | 1.0327 | 0.8179 | 0.2148 | -0.1371 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 105 | 2 | 69 | 1 | 0.1768 | 0.0847 | 0.02 | -0.1136 | 0.0324 | 0.5425 | 1.0324 | 0.8258 | 0.2065 | -0.1336 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 106 | 2 | 70 | 1 | 0.1768 | 0.0847 | 0.0143 | -0.1136 | 0.0321 | 0.5451 | 1.0321 | 0.8374 | 0.1947 | -0.1278 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 107 | 2 | 70 | 1 | 0.1768 | 0.0847 | 0.0143 | -0.1136 | 0.0322 | 0.5477 | 1.0322 | 0.8399 | 0.1923 | -0.1278 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 108 | 2 | 71 | 1 | 0.1768 | 0.0847 | 0.0141 | -0.1136 | 0.0319 | 0.5502 | 1.0319 | 0.8428 | 0.1891 | -0.1276 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 109 | 2 | 72 | 1 | 0.1768 | 0.0847 | 0.0091 | -0.1136 | 0.0316 | 0.5528 | 1.0316 | 0.8537 | 0.1779 | -0.1226 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 110 | 2 | 73 | 1 | 0.1768 | 0.0847 | 0.0112 | -0.1136 | 0.0313 | 0.5553 | 1.0313 | 0.8526 | 0.1787 | -0.1248 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 111 | 2 | 74 | 1 | 0.1768 | 0.0847 | 0.0134 | -0.1136 | 0.031 | 0.5578 | 1.031 | 0.8514 | 0.1796 | -0.127 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 112 | 2 | 75 | 1 | 0.1768 | 0.0847 | 0.0157 | -0.1136 | 0.0308 | 0.5603 | 1.0308 | 0.8502 | 0.1805 | -0.1293 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 113 | 2 | 76 | 1 | 0.1768 | 0.0847 | 0.0188 | -0.1136 | 0.0305 | 0.5628 | 1.0305 | 0.848 | 0.1826 | -0.1323 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 114 | 2 | 77 | 1 | 0.1768 | 0.0847 | 0.0206 | -0.1136 | 0.0303 | 0.5653 | 1.0303 | 0.8477 | 0.1826 | -0.1341 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 115 | 2 | 78 | 1 | 0.1768 | 0.0847 | 0.0208 | -0.1136 | 0.03 | 0.5678 | 1.03 | 0.8499 | 0.1801 | -0.1343 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 116 | 2 | 79 | 1 | 0.1768 | 0.0847 | 0.0206 | -0.1136 | 0.0298 | 0.5702 | 1.0298 | 0.8526 | 0.1772 | -0.1342 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 117 | 2 | 80 | 1 | 0.1768 | 0.0847 | 0.0197 | -0.1136 | 0.0295 | 0.5727 | 1.0295 | 0.8564 | 0.1731 | -0.1333 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 118 | 2 | 81 | 1 | 0.1768 | 0.0847 | 0.0199 | -0.1136 | 0.0293 | 0.5751 | 1.0293 | 0.8586 | 0.1707 | -0.1335 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 119 | 2 | 82 | 1 | 0.1768 | 0.0847 | 0.019 | -0.1136 | 0.0291 | 0.5776 | 1.0291 | 0.8623 | 0.1667 | -0.1326 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 120 | 2 | 83 | 1 | 0.1768 | 0.0847 | 0.0176 | -0.1136 | 0.0288 | 0.58 | 1.0288 | 0.867 | 0.1619 | -0.1312 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 121 | 2 | 84 | 1 | 0.1768 | 0.0847 | 0.0198 | -0.1136 | 0.0286 | 0.5824 | 1.0286 | 0.866 | 0.1626 | -0.1333 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 122 | 2 | 85 | 1 | 0.1768 | 0.0847 | 0.0227 | -0.1136 | 0.0284 | 0.5848 | 1.0284 | 0.8641 | 0.1643 | -0.1363 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 123 | 2 | 86 | 1 | 0.1768 | 0.0847 | 0.0259 | -0.1136 | 0.0282 | 0.5872 | 1.0282 | 0.8618 | 0.1663 | -0.1394 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 124 | 2 | 87 | 1 | 0.1768 | 0.0847 | 0.0265 | -0.1136 | 0.028 | 0.5896 | 1.028 | 0.8633 | 0.1647 | -0.1401 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 125 | 2 | 88 | 1 | 0.1768 | 0.0847 | 0.0278 | -0.1136 | 0.0278 | 0.5919 | 1.0278 | 0.8639 | 0.1638 | -0.1413 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 126 | 2 | 89 | 1 | 0.1768 | 0.0847 | 0.0288 | -0.1136 | 0.0276 | 0.5943 | 1.0276 | 0.8648 | 0.1628 | -0.1424 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 127 | 2 | 90 | 1 | 0.1768 | 0.0847 | 0.0295 | -0.1136 | 0.0274 | 0.5967 | 1.0274 | 0.8662 | 0.1612 | -0.1431 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 128 | 2 | 91 | 1 | 0.1768 | 0.0847 | 0.0282 | -0.1136 | 0.0272 | 0.599 | 1.0272 | 0.8704 | 0.1568 | -0.1417 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1 | 0 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.221 | 0.1059 | 0.1331 | 0.018 | 0.1151 | 0.0 | move_2_score_gt_move_4 |
| current | 2 | 1 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.3126 | 0.1498 | 0.8126 | 0.6498 | 0.1628 | 0.0 | move_2_score_gt_move_4 |
| current | 3 | 1 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.3828 | 0.1834 | 1.0495 | 0.8501 | 0.1994 | 0.0 | move_2_score_gt_move_4 |
| current | 4 | 1 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.4421 | 0.2118 | 1.1921 | 0.9618 | 0.2303 | 0.0 | move_2_score_gt_move_4 |
| current | 5 | 1 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.4942 | 0.2368 | 1.2942 | 1.0368 | 0.2575 | 0.0 | move_2_score_gt_move_4 |
| current | 6 | 1 | 1 | 0 | 0.1768 | 0.0847 | 0.063 | 0.0 | 0.2707 | 0.2594 | 1.2707 | 1.0229 | 0.2478 | -0.063 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 7 | 1 | 2 | 0 | 0.1768 | 0.0847 | 0.0467 | 0.0 | 0.1949 | 0.2802 | 1.1918 | 1.1364 | 0.0554 | -0.0467 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 8 | 1 | 2 | 0 | 0.1768 | 0.0847 | 0.0467 | 0.0 | 0.2084 | 0.2995 | 1.0972 | 1.1467 | -0.0495 | -0.0467 | move_2_q_gt_move_4 |
| current | 9 | 1 | 2 | 0 | 0.1768 | 0.0847 | 0.0467 | 0.0 | 0.221 | 0.3177 | 1.1212 | 1.1844 | -0.0632 | -0.0467 | move_2_q_gt_move_4 |
| current | 10 | 1 | 2 | 1 | 0.1768 | 0.0847 | 0.0467 | -0.1136 | 0.233 | 0.1674 | 1.1478 | 0.1674 | 0.9803 | -0.1603 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 11 | 1 | 2 | 1 | 0.1768 | 0.0847 | 0.0467 | -0.1136 | 0.2444 | 0.1756 | 1.1887 | 0.1756 | 1.0131 | -0.1603 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 12 | 1 | 3 | 1 | 0.1768 | 0.0847 | 0.0582 | -0.1136 | 0.1914 | 0.1834 | 1.1914 | 0.1834 | 1.008 | -0.1718 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 13 | 1 | 4 | 1 | 0.1768 | 0.0847 | 0.0303 | -0.1136 | 0.1594 | 0.1909 | 1.0069 | 0.1909 | 0.816 | -0.1439 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 14 | 1 | 4 | 1 | 0.1768 | 0.0847 | 0.0303 | -0.1136 | 0.1654 | 0.1981 | 1.1287 | 0.1981 | 0.9306 | -0.1439 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 15 | 1 | 4 | 1 | 0.1768 | 0.0847 | 0.0303 | -0.1136 | 0.1712 | 0.2051 | 1.1712 | 0.2051 | 0.9662 | -0.1439 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 16 | 1 | 5 | 1 | 0.1768 | 0.0847 | 0.0286 | -0.1136 | 0.1474 | 0.2118 | 1.1474 | 0.2118 | 0.9356 | -0.1422 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 17 | 1 | 6 | 1 | 0.1768 | 0.0847 | 0.0398 | -0.1136 | 0.1302 | 0.2183 | 1.1302 | 0.2183 | 0.9119 | -0.1533 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 18 | 1 | 6 | 1 | 0.1768 | 0.0847 | 0.0398 | -0.1136 | 0.134 | 0.2246 | 1.134 | 0.2246 | 0.9093 | -0.1533 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 19 | 1 | 6 | 1 | 0.1768 | 0.0847 | 0.0398 | -0.1136 | 0.1376 | 0.2308 | 1.1376 | 0.2308 | 0.9069 | -0.1533 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 20 | 1 | 7 | 1 | 0.1768 | 0.0847 | 0.0565 | -0.1136 | 0.1236 | 0.2368 | 1.1236 | 0.2368 | 0.8868 | -0.17 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 21 | 1 | 8 | 1 | 0.1768 | 0.0847 | 0.0535 | -0.1136 | 0.1125 | 0.2426 | 1.1125 | 0.2426 | 0.8699 | -0.1671 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 22 | 2 | 9 | 1 | 0.1768 | 0.0847 | 0.0674 | -0.1136 | 0.1037 | 0.2483 | 1.1037 | 0.2483 | 0.8553 | -0.1809 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 23 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.0964 | 0.2539 | 1.0964 | 0.2539 | 0.8425 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 24 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.0984 | 0.2594 | 1.0984 | 0.5113 | 0.5872 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 25 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1005 | 0.2647 | 1.1005 | 0.5166 | 0.5839 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 26 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1025 | 0.27 | 1.1025 | 0.5219 | 0.5806 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 27 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1044 | 0.2751 | 1.1044 | 0.527 | 0.5774 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 28 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1063 | 0.2802 | 1.1063 | 0.532 | 0.5743 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 29 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1082 | 0.2851 | 1.0859 | 0.5314 | 0.5545 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 30 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1101 | 0.29 | 1.0611 | 0.5296 | 0.5316 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 31 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1119 | 0.2948 | 1.1119 | 0.5467 | 0.5652 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 32 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1137 | 0.2995 | 1.1137 | 0.5514 | 0.5623 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 33 | 1 | 11 | 1 | 0.1768 | 0.0847 | 0.046 | -0.1136 | 0.1058 | 0.3041 | 1.1058 | 0.5528 | 0.553 | -0.1595 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 34 | 1 | 11 | 1 | 0.1768 | 0.0847 | 0.046 | -0.1136 | 0.1074 | 0.3087 | 1.1074 | 0.5574 | 0.5501 | -0.1595 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 35 | 1 | 12 | 1 | 0.1768 | 0.0847 | 0.0464 | -0.1136 | 0.1006 | 0.3132 | 1.1006 | 0.5614 | 0.5392 | -0.1599 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 36 | 1 | 13 | 1 | 0.1768 | 0.0847 | 0.0308 | -0.1136 | 0.0947 | 0.3177 | 1.0947 | 0.5854 | 0.5093 | -0.1444 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 37 | 1 | 13 | 1 | 0.1768 | 0.0847 | 0.0308 | -0.1136 | 0.096 | 0.3221 | 1.096 | 0.5898 | 0.5063 | -0.1444 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 38 | 1 | 14 | 1 | 0.1768 | 0.0847 | 0.0309 | -0.1136 | 0.0908 | 0.3264 | 1.0908 | 0.594 | 0.4968 | -0.1445 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 39 | 1 | 15 | 1 | 0.1768 | 0.0847 | 0.0431 | -0.1136 | 0.0863 | 0.3306 | 1.0863 | 0.5827 | 0.5036 | -0.1567 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 40 | 2 | 16 | 1 | 0.1768 | 0.0847 | 0.0579 | -0.1136 | 0.0822 | 0.3349 | 1.0822 | 0.5703 | 0.512 | -0.1715 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 41 | 2 | 17 | 1 | 0.1768 | 0.0847 | 0.0606 | -0.1136 | 0.0786 | 0.339 | 1.0786 | 0.5716 | 0.507 | -0.1742 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 42 | 2 | 18 | 1 | 0.1768 | 0.0847 | 0.0766 | -0.1136 | 0.0754 | 0.3431 | 1.0754 | 0.5604 | 0.5149 | -0.1901 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 43 | 2 | 19 | 1 | 0.1768 | 0.0847 | 0.0824 | -0.1136 | 0.0725 | 0.3472 | 1.0725 | 0.5594 | 0.513 | -0.1959 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 44 | 2 | 20 | 1 | 0.1768 | 0.0847 | 0.0841 | -0.1136 | 0.0698 | 0.3512 | 1.0698 | 0.562 | 0.5079 | -0.1977 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 45 | 2 | 21 | 1 | 0.1768 | 0.0847 | 0.0763 | -0.1136 | 0.0674 | 0.3552 | 1.0674 | 0.5727 | 0.4947 | -0.1899 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 46 | 2 | 22 | 1 | 0.1768 | 0.0847 | 0.0742 | -0.1136 | 0.0652 | 0.3591 | 1.0652 | 0.5786 | 0.4866 | -0.1877 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 47 | 2 | 23 | 1 | 0.1768 | 0.0847 | 0.0656 | -0.1136 | 0.0631 | 0.363 | 1.0631 | 0.5906 | 0.4726 | -0.1792 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 48 | 2 | 24 | 1 | 0.1768 | 0.0847 | 0.0591 | -0.1136 | 0.0613 | 0.3668 | 1.0613 | 0.601 | 0.4602 | -0.1726 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 49 | 2 | 25 | 1 | 0.1768 | 0.0847 | 0.0563 | -0.1136 | 0.0595 | 0.3706 | 1.0595 | 0.6078 | 0.4517 | -0.1698 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 50 | 2 | 25 | 1 | 0.1768 | 0.0847 | 0.0563 | -0.1136 | 0.0601 | 0.3744 | 1.0601 | 0.6115 | 0.4486 | -0.1698 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 51 | 2 | 26 | 1 | 0.1768 | 0.0847 | 0.0514 | -0.1136 | 0.0585 | 0.3781 | 1.0585 | 0.6205 | 0.4379 | -0.165 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 52 | 2 | 27 | 1 | 0.1768 | 0.0847 | 0.0549 | -0.1136 | 0.0569 | 0.3818 | 1.0569 | 0.6204 | 0.4365 | -0.1684 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 53 | 2 | 28 | 1 | 0.1768 | 0.0847 | 0.057 | -0.1136 | 0.0555 | 0.3854 | 1.0555 | 0.6218 | 0.4337 | -0.1706 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 54 | 2 | 29 | 1 | 0.1768 | 0.0847 | 0.0642 | -0.1136 | 0.0541 | 0.3891 | 1.0541 | 0.6181 | 0.4361 | -0.1777 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 55 | 2 | 30 | 1 | 0.1768 | 0.0847 | 0.0656 | -0.1136 | 0.0529 | 0.3927 | 1.0529 | 0.6202 | 0.4326 | -0.1792 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 56 | 2 | 31 | 1 | 0.1768 | 0.0847 | 0.0705 | -0.1136 | 0.0517 | 0.3962 | 1.0517 | 0.6191 | 0.4326 | -0.1841 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 57 | 2 | 32 | 1 | 0.1768 | 0.0847 | 0.0761 | -0.1136 | 0.0506 | 0.3997 | 1.0506 | 0.6175 | 0.4331 | -0.1896 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 58 | 2 | 33 | 1 | 0.1768 | 0.0847 | 0.0806 | -0.1136 | 0.0495 | 0.4032 | 1.0495 | 0.617 | 0.4325 | -0.1941 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 59 | 2 | 34 | 1 | 0.1768 | 0.0847 | 0.0729 | -0.1136 | 0.0485 | 0.4067 | 1.0485 | 0.6273 | 0.4212 | -0.1865 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 60 | 2 | 35 | 1 | 0.1768 | 0.0847 | 0.0679 | -0.1136 | 0.0476 | 0.4101 | 1.0476 | 0.6354 | 0.4121 | -0.1815 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 61 | 2 | 36 | 1 | 0.1768 | 0.0847 | 0.0654 | -0.1136 | 0.0467 | 0.4135 | 1.0467 | 0.6413 | 0.4054 | -0.179 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 62 | 2 | 37 | 1 | 0.1768 | 0.0847 | 0.0625 | -0.1136 | 0.0458 | 0.4169 | 1.0458 | 0.6476 | 0.3982 | -0.176 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 63 | 2 | 38 | 1 | 0.1768 | 0.0847 | 0.0536 | -0.1136 | 0.045 | 0.4202 | 1.045 | 0.6603 | 0.3847 | -0.1671 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 64 | 2 | 38 | 1 | 0.1768 | 0.0847 | 0.0536 | -0.1136 | 0.0453 | 0.4236 | 1.0453 | 0.6636 | 0.3817 | -0.1671 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 65 | 2 | 39 | 1 | 0.1768 | 0.0847 | 0.0487 | -0.1136 | 0.0446 | 0.4269 | 1.0446 | 0.6723 | 0.3722 | -0.1623 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 66 | 2 | 40 | 1 | 0.1768 | 0.0847 | 0.0439 | -0.1136 | 0.0438 | 0.4301 | 1.0438 | 0.6813 | 0.3625 | -0.1574 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 67 | 2 | 41 | 1 | 0.1768 | 0.0847 | 0.0404 | -0.1136 | 0.0431 | 0.4334 | 1.0431 | 0.6888 | 0.3543 | -0.1539 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 68 | 2 | 42 | 1 | 0.1768 | 0.0847 | 0.0424 | -0.1136 | 0.0424 | 0.4366 | 1.0424 | 0.6895 | 0.3529 | -0.156 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 69 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0417 | 0.4398 | 1.0417 | 0.6941 | 0.3476 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 70 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.042 | 0.443 | 1.042 | 0.6973 | 0.3448 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 71 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0423 | 0.4461 | 1.0423 | 0.7004 | 0.3419 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 72 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0426 | 0.4493 | 1.0426 | 0.7035 | 0.3391 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 73 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0429 | 0.4524 | 1.0429 | 0.7067 | 0.3363 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 74 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0432 | 0.4555 | 1.0432 | 0.7097 | 0.3335 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 75 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0435 | 0.4585 | 1.0435 | 0.7128 | 0.3307 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 76 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0438 | 0.4616 | 1.0438 | 0.7159 | 0.3279 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 77 | 2 | 44 | 1 | 0.1768 | 0.0847 | 0.0372 | -0.1136 | 0.0431 | 0.4646 | 1.0431 | 0.7239 | 0.3192 | -0.1508 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 78 | 2 | 44 | 1 | 0.1768 | 0.0847 | 0.0372 | -0.1136 | 0.0434 | 0.4676 | 1.0434 | 0.7269 | 0.3165 | -0.1508 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 79 | 2 | 45 | 1 | 0.1768 | 0.0847 | 0.0344 | -0.1136 | 0.0427 | 0.4706 | 1.0427 | 0.7336 | 0.3091 | -0.148 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 80 | 2 | 46 | 1 | 0.1768 | 0.0847 | 0.0299 | -0.1136 | 0.0421 | 0.4736 | 1.0421 | 0.7425 | 0.2995 | -0.1435 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 81 | 2 | 47 | 1 | 0.1768 | 0.0847 | 0.0279 | -0.1136 | 0.0414 | 0.4765 | 1.0414 | 0.7483 | 0.2931 | -0.1414 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 82 | 2 | 48 | 1 | 0.1768 | 0.0847 | 0.0247 | -0.1136 | 0.0408 | 0.4794 | 1.0408 | 0.7558 | 0.285 | -0.1382 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 83 | 2 | 48 | 1 | 0.1768 | 0.0847 | 0.0247 | -0.1136 | 0.0411 | 0.4824 | 1.0411 | 0.7587 | 0.2824 | -0.1382 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 84 | 2 | 49 | 1 | 0.1768 | 0.0847 | 0.026 | -0.1136 | 0.0405 | 0.4852 | 1.0405 | 0.7598 | 0.2808 | -0.1395 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 85 | 2 | 50 | 1 | 0.1768 | 0.0847 | 0.028 | -0.1136 | 0.04 | 0.4881 | 1.04 | 0.7598 | 0.2802 | -0.1416 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 86 | 2 | 51 | 1 | 0.1768 | 0.0847 | 0.0297 | -0.1136 | 0.0394 | 0.491 | 1.0394 | 0.7603 | 0.2791 | -0.1433 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 87 | 2 | 52 | 1 | 0.1768 | 0.0847 | 0.0326 | -0.1136 | 0.0389 | 0.4938 | 1.0389 | 0.7592 | 0.2797 | -0.1462 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 88 | 2 | 53 | 1 | 0.1768 | 0.0847 | 0.0372 | -0.1136 | 0.0384 | 0.4967 | 1.0384 | 0.7561 | 0.2823 | -0.1507 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 89 | 2 | 54 | 1 | 0.1768 | 0.0847 | 0.0383 | -0.1136 | 0.0379 | 0.4995 | 1.0379 | 0.7575 | 0.2805 | -0.1518 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 90 | 2 | 55 | 1 | 0.1768 | 0.0847 | 0.0333 | -0.1136 | 0.0374 | 0.5023 | 1.0374 | 0.7668 | 0.2707 | -0.1468 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 91 | 2 | 56 | 1 | 0.1768 | 0.0847 | 0.0292 | -0.1136 | 0.037 | 0.5051 | 1.037 | 0.775 | 0.262 | -0.1428 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 92 | 2 | 57 | 1 | 0.1768 | 0.0847 | 0.0264 | -0.1136 | 0.0366 | 0.5078 | 1.0366 | 0.7817 | 0.2548 | -0.14 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 93 | 2 | 57 | 1 | 0.1768 | 0.0847 | 0.0264 | -0.1136 | 0.0368 | 0.5106 | 1.0368 | 0.7845 | 0.2523 | -0.14 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 94 | 2 | 58 | 1 | 0.1768 | 0.0847 | 0.025 | -0.1136 | 0.0363 | 0.5133 | 1.0363 | 0.7892 | 0.2472 | -0.1386 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 95 | 2 | 59 | 1 | 0.1768 | 0.0847 | 0.0289 | -0.1136 | 0.0359 | 0.516 | 1.0359 | 0.7865 | 0.2494 | -0.1424 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 96 | 2 | 60 | 1 | 0.1768 | 0.0847 | 0.028 | -0.1136 | 0.0355 | 0.5188 | 1.0355 | 0.7904 | 0.2451 | -0.1415 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 97 | 2 | 61 | 1 | 0.1768 | 0.0847 | 0.0276 | -0.1136 | 0.0351 | 0.5214 | 1.0351 | 0.7936 | 0.2415 | -0.1412 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 98 | 2 | 62 | 1 | 0.1768 | 0.0847 | 0.0249 | -0.1136 | 0.0347 | 0.5241 | 1.0347 | 0.8002 | 0.2346 | -0.1384 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 99 | 2 | 63 | 1 | 0.1768 | 0.0847 | 0.0249 | -0.1136 | 0.0344 | 0.5268 | 1.0344 | 0.8029 | 0.2315 | -0.1384 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 100 | 2 | 64 | 1 | 0.1768 | 0.0847 | 0.024 | -0.1136 | 0.034 | 0.5295 | 1.034 | 0.8068 | 0.2272 | -0.1375 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 101 | 2 | 65 | 1 | 0.1768 | 0.0847 | 0.0237 | -0.1136 | 0.0337 | 0.5321 | 1.0337 | 0.8098 | 0.2238 | -0.1373 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 102 | 2 | 66 | 1 | 0.1768 | 0.0847 | 0.0198 | -0.1136 | 0.0333 | 0.5347 | 1.0333 | 0.8183 | 0.215 | -0.1333 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 103 | 2 | 67 | 1 | 0.1768 | 0.0847 | 0.02 | -0.1136 | 0.033 | 0.5373 | 1.033 | 0.8206 | 0.2124 | -0.1336 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 104 | 2 | 68 | 1 | 0.1768 | 0.0847 | 0.0236 | -0.1136 | 0.0327 | 0.5399 | 1.0327 | 0.8179 | 0.2148 | -0.1371 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 105 | 2 | 69 | 1 | 0.1768 | 0.0847 | 0.02 | -0.1136 | 0.0324 | 0.5425 | 1.0324 | 0.8258 | 0.2065 | -0.1336 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 106 | 2 | 70 | 1 | 0.1768 | 0.0847 | 0.0143 | -0.1136 | 0.0321 | 0.5451 | 1.0321 | 0.8374 | 0.1947 | -0.1278 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 107 | 2 | 70 | 1 | 0.1768 | 0.0847 | 0.0143 | -0.1136 | 0.0322 | 0.5477 | 1.0322 | 0.8399 | 0.1923 | -0.1278 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 108 | 2 | 71 | 1 | 0.1768 | 0.0847 | 0.0141 | -0.1136 | 0.0319 | 0.5502 | 1.0319 | 0.8428 | 0.1891 | -0.1276 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 109 | 2 | 72 | 1 | 0.1768 | 0.0847 | 0.0091 | -0.1136 | 0.0316 | 0.5528 | 1.0316 | 0.8537 | 0.1779 | -0.1226 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 110 | 2 | 73 | 1 | 0.1768 | 0.0847 | 0.0112 | -0.1136 | 0.0313 | 0.5553 | 1.0313 | 0.8526 | 0.1787 | -0.1248 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 111 | 2 | 74 | 1 | 0.1768 | 0.0847 | 0.0134 | -0.1136 | 0.031 | 0.5578 | 1.031 | 0.8514 | 0.1796 | -0.127 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 112 | 2 | 75 | 1 | 0.1768 | 0.0847 | 0.0157 | -0.1136 | 0.0308 | 0.5603 | 1.0308 | 0.8502 | 0.1805 | -0.1293 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 113 | 2 | 76 | 1 | 0.1768 | 0.0847 | 0.0188 | -0.1136 | 0.0305 | 0.5628 | 1.0305 | 0.848 | 0.1826 | -0.1323 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 114 | 2 | 77 | 1 | 0.1768 | 0.0847 | 0.0206 | -0.1136 | 0.0303 | 0.5653 | 1.0303 | 0.8477 | 0.1826 | -0.1341 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 115 | 2 | 78 | 1 | 0.1768 | 0.0847 | 0.0208 | -0.1136 | 0.03 | 0.5678 | 1.03 | 0.8499 | 0.1801 | -0.1343 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 116 | 2 | 79 | 1 | 0.1768 | 0.0847 | 0.0206 | -0.1136 | 0.0298 | 0.5702 | 1.0298 | 0.8526 | 0.1772 | -0.1342 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 117 | 2 | 80 | 1 | 0.1768 | 0.0847 | 0.0197 | -0.1136 | 0.0295 | 0.5727 | 1.0295 | 0.8564 | 0.1731 | -0.1333 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 118 | 2 | 81 | 1 | 0.1768 | 0.0847 | 0.0199 | -0.1136 | 0.0293 | 0.5751 | 1.0293 | 0.8586 | 0.1707 | -0.1335 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 119 | 2 | 82 | 1 | 0.1768 | 0.0847 | 0.019 | -0.1136 | 0.0291 | 0.5776 | 1.0291 | 0.8623 | 0.1667 | -0.1326 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 120 | 2 | 83 | 1 | 0.1768 | 0.0847 | 0.0176 | -0.1136 | 0.0288 | 0.58 | 1.0288 | 0.867 | 0.1619 | -0.1312 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 121 | 2 | 84 | 1 | 0.1768 | 0.0847 | 0.0198 | -0.1136 | 0.0286 | 0.5824 | 1.0286 | 0.866 | 0.1626 | -0.1333 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 122 | 2 | 85 | 1 | 0.1768 | 0.0847 | 0.0227 | -0.1136 | 0.0284 | 0.5848 | 1.0284 | 0.8641 | 0.1643 | -0.1363 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 123 | 2 | 86 | 1 | 0.1768 | 0.0847 | 0.0259 | -0.1136 | 0.0282 | 0.5872 | 1.0282 | 0.8618 | 0.1663 | -0.1394 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 124 | 2 | 87 | 1 | 0.1768 | 0.0847 | 0.0265 | -0.1136 | 0.028 | 0.5896 | 1.028 | 0.8633 | 0.1647 | -0.1401 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 125 | 2 | 88 | 1 | 0.1768 | 0.0847 | 0.0278 | -0.1136 | 0.0278 | 0.5919 | 1.0278 | 0.8639 | 0.1638 | -0.1413 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 126 | 2 | 89 | 1 | 0.1768 | 0.0847 | 0.0288 | -0.1136 | 0.0276 | 0.5943 | 1.0276 | 0.8648 | 0.1628 | -0.1424 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 127 | 2 | 90 | 1 | 0.1768 | 0.0847 | 0.0295 | -0.1136 | 0.0274 | 0.5967 | 1.0274 | 0.8662 | 0.1612 | -0.1431 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 128 | 2 | 91 | 1 | 0.1768 | 0.0847 | 0.0282 | -0.1136 | 0.0272 | 0.599 | 1.0272 | 0.8704 | 0.1568 | -0.1417 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 129 | 2 | 92 | 1 | 0.1768 | 0.0847 | 0.027 | -0.1136 | 0.027 | 0.6013 | 1.027 | 0.8744 | 0.1526 | -0.1406 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 130 | 2 | 93 | 1 | 0.1768 | 0.0847 | 0.0271 | -0.1136 | 0.0268 | 0.6037 | 1.0268 | 0.8766 | 0.1502 | -0.1406 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 131 | 2 | 94 | 1 | 0.1768 | 0.0847 | 0.0308 | -0.1136 | 0.0266 | 0.606 | 1.0266 | 0.8738 | 0.1528 | -0.1443 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 132 | 2 | 95 | 1 | 0.1768 | 0.0847 | 0.0347 | -0.1136 | 0.0265 | 0.6083 | 1.0265 | 0.8709 | 0.1555 | -0.1482 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 133 | 2 | 96 | 1 | 0.1768 | 0.0847 | 0.0391 | -0.1136 | 0.0263 | 0.6106 | 1.0263 | 0.8675 | 0.1588 | -0.1527 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 134 | 2 | 97 | 1 | 0.1768 | 0.0847 | 0.0398 | -0.1136 | 0.0261 | 0.6129 | 1.0261 | 0.8689 | 0.1572 | -0.1534 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 135 | 2 | 98 | 1 | 0.1768 | 0.0847 | 0.0395 | -0.1136 | 0.0259 | 0.6152 | 1.0259 | 0.8716 | 0.1543 | -0.1531 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 136 | 2 | 99 | 1 | 0.1768 | 0.0847 | 0.0404 | -0.1136 | 0.0258 | 0.6174 | 1.0258 | 0.8728 | 0.153 | -0.154 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 137 | 2 | 100 | 1 | 0.1768 | 0.0847 | 0.0428 | -0.1136 | 0.0256 | 0.6197 | 1.0256 | 0.8721 | 0.1535 | -0.1564 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 138 | 2 | 101 | 1 | 0.1768 | 0.0847 | 0.0442 | -0.1136 | 0.0255 | 0.622 | 1.0255 | 0.8727 | 0.1527 | -0.1577 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 139 | 2 | 102 | 1 | 0.1768 | 0.0847 | 0.0448 | -0.1136 | 0.0253 | 0.6242 | 1.0253 | 0.8742 | 0.1511 | -0.1584 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 140 | 2 | 103 | 1 | 0.1768 | 0.0847 | 0.0434 | -0.1136 | 0.0251 | 0.6265 | 1.0251 | 0.8781 | 0.147 | -0.157 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 141 | 2 | 104 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.025 | 0.6287 | 1.025 | 0.8806 | 0.1444 | -0.1567 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 142 | 2 | 105 | 1 | 0.1768 | 0.0847 | 0.0419 | -0.1136 | 0.0248 | 0.6309 | 1.0248 | 0.8845 | 0.1404 | -0.1554 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 143 | 2 | 106 | 1 | 0.1768 | 0.0847 | 0.0417 | -0.1136 | 0.0247 | 0.6331 | 1.0247 | 0.8869 | 0.1378 | -0.1553 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 144 | 2 | 107 | 1 | 0.1768 | 0.0847 | 0.0439 | -0.1136 | 0.0246 | 0.6353 | 1.0246 | 0.8864 | 0.1382 | -0.1575 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 145 | 2 | 108 | 1 | 0.1768 | 0.0847 | 0.0456 | -0.1136 | 0.0244 | 0.6375 | 1.0244 | 0.8866 | 0.1378 | -0.1592 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 146 | 2 | 109 | 1 | 0.1768 | 0.0847 | 0.0456 | -0.1136 | 0.0243 | 0.6397 | 1.0243 | 0.8888 | 0.1355 | -0.1591 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 147 | 2 | 110 | 1 | 0.1768 | 0.0847 | 0.049 | -0.1136 | 0.0241 | 0.6419 | 1.0241 | 0.887 | 0.1371 | -0.1626 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 148 | 2 | 111 | 1 | 0.1768 | 0.0847 | 0.0505 | -0.1136 | 0.024 | 0.6441 | 1.024 | 0.8875 | 0.1365 | -0.1641 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 149 | 2 | 112 | 1 | 0.1768 | 0.0847 | 0.0537 | -0.1136 | 0.0239 | 0.6463 | 1.0239 | 0.8862 | 0.1376 | -0.1672 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 150 | 2 | 113 | 1 | 0.1768 | 0.0847 | 0.0529 | -0.1136 | 0.0237 | 0.6484 | 1.0237 | 0.8892 | 0.1345 | -0.1665 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 151 | 2 | 114 | 1 | 0.1768 | 0.0847 | 0.0516 | -0.1136 | 0.0236 | 0.6506 | 1.0236 | 0.8928 | 0.1308 | -0.1652 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 152 | 2 | 115 | 1 | 0.1768 | 0.0847 | 0.052 | -0.1136 | 0.0235 | 0.6528 | 1.0235 | 0.8945 | 0.1289 | -0.1655 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 153 | 2 | 116 | 1 | 0.1768 | 0.0847 | 0.0512 | -0.1136 | 0.0234 | 0.6549 | 1.0234 | 0.8975 | 0.1258 | -0.1648 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 154 | 2 | 117 | 1 | 0.1768 | 0.0847 | 0.0503 | -0.1136 | 0.0232 | 0.657 | 1.0232 | 0.9007 | 0.1225 | -0.1639 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 155 | 2 | 118 | 1 | 0.1768 | 0.0847 | 0.0508 | -0.1136 | 0.0231 | 0.6592 | 1.0231 | 0.9023 | 0.1209 | -0.1644 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 156 | 2 | 119 | 1 | 0.1768 | 0.0847 | 0.0499 | -0.1136 | 0.023 | 0.6613 | 1.023 | 0.9054 | 0.1176 | -0.1635 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 157 | 2 | 120 | 1 | 0.1768 | 0.0847 | 0.0507 | -0.1136 | 0.0229 | 0.6634 | 1.0229 | 0.9066 | 0.1162 | -0.1642 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 158 | 2 | 121 | 1 | 0.1768 | 0.0847 | 0.0528 | -0.1136 | 0.0228 | 0.6655 | 1.0228 | 0.9064 | 0.1164 | -0.1664 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 159 | 2 | 122 | 1 | 0.1768 | 0.0847 | 0.0541 | -0.1136 | 0.0227 | 0.6676 | 1.0227 | 0.907 | 0.1156 | -0.1677 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 160 | 2 | 123 | 1 | 0.1768 | 0.0847 | 0.054 | -0.1136 | 0.0225 | 0.6697 | 1.0225 | 0.9093 | 0.1133 | -0.1676 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 161 | 2 | 124 | 1 | 0.1768 | 0.0847 | 0.0567 | -0.1136 | 0.0224 | 0.6718 | 1.0224 | 0.9085 | 0.1139 | -0.1702 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 162 | 2 | 125 | 1 | 0.1768 | 0.0847 | 0.0541 | -0.1136 | 0.0223 | 0.6739 | 1.0223 | 0.9134 | 0.109 | -0.1677 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 163 | 2 | 126 | 1 | 0.1768 | 0.0847 | 0.055 | -0.1136 | 0.0222 | 0.676 | 1.0222 | 0.9145 | 0.1077 | -0.1685 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 164 | 2 | 127 | 1 | 0.1768 | 0.0847 | 0.0556 | -0.1136 | 0.0221 | 0.678 | 1.0221 | 0.9159 | 0.1062 | -0.1691 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 165 | 2 | 128 | 1 | 0.1768 | 0.0847 | 0.0537 | -0.1136 | 0.022 | 0.6801 | 1.022 | 0.92 | 0.102 | -0.1673 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 166 | 2 | 129 | 1 | 0.1768 | 0.0847 | 0.0526 | -0.1136 | 0.0219 | 0.6822 | 1.0219 | 0.9233 | 0.0986 | -0.1661 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 167 | 2 | 130 | 1 | 0.1768 | 0.0847 | 0.0542 | -0.1136 | 0.0218 | 0.6842 | 1.0218 | 0.9235 | 0.0983 | -0.1678 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 168 | 2 | 131 | 1 | 0.1768 | 0.0847 | 0.0552 | -0.1136 | 0.0217 | 0.6862 | 1.0217 | 0.9245 | 0.0972 | -0.1688 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 169 | 2 | 132 | 1 | 0.1768 | 0.0847 | 0.0541 | -0.1136 | 0.0216 | 0.6883 | 1.0216 | 0.9277 | 0.0939 | -0.1677 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 170 | 2 | 133 | 1 | 0.1768 | 0.0847 | 0.0558 | -0.1136 | 0.0215 | 0.6903 | 1.0215 | 0.9279 | 0.0936 | -0.1694 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 171 | 2 | 134 | 1 | 0.1768 | 0.0847 | 0.0576 | -0.1136 | 0.0214 | 0.6923 | 1.0214 | 0.9281 | 0.0933 | -0.1711 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 172 | 2 | 135 | 1 | 0.1768 | 0.0847 | 0.0597 | -0.1136 | 0.0213 | 0.6944 | 1.0213 | 0.928 | 0.0933 | -0.1732 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 173 | 2 | 136 | 1 | 0.1768 | 0.0847 | 0.0609 | -0.1136 | 0.0212 | 0.6964 | 1.0212 | 0.9287 | 0.0925 | -0.1745 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 174 | 2 | 137 | 1 | 0.1768 | 0.0847 | 0.0605 | -0.1136 | 0.0211 | 0.6984 | 1.0211 | 0.9311 | 0.09 | -0.1741 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 175 | 2 | 138 | 1 | 0.1768 | 0.0847 | 0.0596 | -0.1136 | 0.021 | 0.7004 | 1.021 | 0.9341 | 0.0869 | -0.1731 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 176 | 2 | 139 | 1 | 0.1768 | 0.0847 | 0.0589 | -0.1136 | 0.0209 | 0.7024 | 1.0209 | 0.9368 | 0.0841 | -0.1724 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 177 | 2 | 140 | 1 | 0.1768 | 0.0847 | 0.0603 | -0.1136 | 0.0209 | 0.7044 | 1.0209 | 0.9373 | 0.0835 | -0.1738 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 178 | 2 | 141 | 1 | 0.1768 | 0.0847 | 0.0611 | -0.1136 | 0.0208 | 0.7064 | 1.0208 | 0.9385 | 0.0823 | -0.1747 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 179 | 2 | 142 | 1 | 0.1768 | 0.0847 | 0.061 | -0.1136 | 0.0207 | 0.7084 | 1.0207 | 0.9406 | 0.0801 | -0.1745 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 180 | 2 | 143 | 1 | 0.1768 | 0.0847 | 0.0606 | -0.1136 | 0.0206 | 0.7103 | 1.0206 | 0.9429 | 0.0776 | -0.1742 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 181 | 2 | 144 | 1 | 0.1768 | 0.0847 | 0.0602 | -0.1136 | 0.0205 | 0.7123 | 1.0205 | 0.9453 | 0.0752 | -0.1738 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 182 | 2 | 145 | 1 | 0.1768 | 0.0847 | 0.0599 | -0.1136 | 0.0204 | 0.7143 | 1.0204 | 0.9476 | 0.0728 | -0.1734 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 183 | 2 | 146 | 1 | 0.1768 | 0.0847 | 0.0595 | -0.1136 | 0.0203 | 0.7162 | 1.0203 | 0.95 | 0.0703 | -0.173 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 184 | 2 | 147 | 1 | 0.1768 | 0.0847 | 0.0598 | -0.1136 | 0.0203 | 0.7182 | 1.0203 | 0.9517 | 0.0686 | -0.1733 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 185 | 2 | 148 | 1 | 0.1768 | 0.0847 | 0.0602 | -0.1136 | 0.0202 | 0.7201 | 1.0202 | 0.9532 | 0.067 | -0.1737 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 186 | 2 | 149 | 1 | 0.1768 | 0.0847 | 0.0584 | -0.1136 | 0.0201 | 0.7221 | 1.0201 | 0.957 | 0.0631 | -0.172 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 187 | 2 | 150 | 1 | 0.1768 | 0.0847 | 0.0599 | -0.1136 | 0.02 | 0.724 | 1.02 | 0.9573 | 0.0627 | -0.1735 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 188 | 2 | 151 | 1 | 0.1768 | 0.0847 | 0.0611 | -0.1136 | 0.0199 | 0.7259 | 1.0199 | 0.958 | 0.0619 | -0.1747 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 189 | 2 | 152 | 1 | 0.1768 | 0.0847 | 0.063 | -0.1136 | 0.0199 | 0.7279 | 1.0199 | 0.958 | 0.0618 | -0.1766 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 190 | 2 | 153 | 1 | 0.1768 | 0.0847 | 0.0633 | -0.1136 | 0.0198 | 0.7298 | 1.0198 | 0.9596 | 0.0601 | -0.1769 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 191 | 2 | 154 | 1 | 0.1768 | 0.0847 | 0.0657 | -0.1136 | 0.0197 | 0.7317 | 1.0197 | 0.9592 | 0.0605 | -0.1793 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 192 | 2 | 155 | 1 | 0.1768 | 0.0847 | 0.0668 | -0.1136 | 0.0196 | 0.7336 | 1.0196 | 0.96 | 0.0596 | -0.1804 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 193 | 2 | 156 | 1 | 0.1768 | 0.0847 | 0.0669 | -0.1136 | 0.0196 | 0.7355 | 1.0196 | 0.9619 | 0.0577 | -0.1804 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 194 | 2 | 157 | 1 | 0.1768 | 0.0847 | 0.0675 | -0.1136 | 0.0195 | 0.7374 | 1.0195 | 0.9632 | 0.0563 | -0.1811 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 195 | 2 | 158 | 1 | 0.1768 | 0.0847 | 0.0692 | -0.1136 | 0.0194 | 0.7393 | 1.0194 | 0.9634 | 0.056 | -0.1828 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 196 | 2 | 159 | 1 | 0.1768 | 0.0847 | 0.0698 | -0.1136 | 0.0193 | 0.7412 | 1.0193 | 0.9648 | 0.0545 | -0.1833 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 197 | 2 | 160 | 1 | 0.1768 | 0.0847 | 0.0688 | -0.1136 | 0.0193 | 0.7431 | 1.0193 | 0.9676 | 0.0517 | -0.1824 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 198 | 2 | 161 | 1 | 0.1768 | 0.0847 | 0.0698 | -0.1136 | 0.0192 | 0.745 | 1.0192 | 0.9685 | 0.0507 | -0.1834 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 199 | 2 | 162 | 1 | 0.1768 | 0.0847 | 0.071 | -0.1136 | 0.0191 | 0.7469 | 1.0191 | 0.9693 | 0.0498 | -0.1846 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 200 | 2 | 163 | 1 | 0.1768 | 0.0847 | 0.0704 | -0.1136 | 0.0191 | 0.7488 | 1.0191 | 0.9718 | 0.0473 | -0.1839 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 201 | 2 | 164 | 1 | 0.1768 | 0.0847 | 0.0705 | -0.1136 | 0.019 | 0.7506 | 1.019 | 0.9735 | 0.0455 | -0.1841 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 202 | 2 | 165 | 1 | 0.1768 | 0.0847 | 0.0701 | -0.1136 | 0.0189 | 0.7525 | 1.0189 | 0.9758 | 0.0431 | -0.1836 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 203 | 2 | 166 | 1 | 0.1768 | 0.0847 | 0.0696 | -0.1136 | 0.0189 | 0.7544 | 1.0189 | 0.9781 | 0.0408 | -0.1832 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 204 | 2 | 167 | 1 | 0.1768 | 0.0847 | 0.0703 | -0.1136 | 0.0188 | 0.7562 | 1.0188 | 0.9793 | 0.0395 | -0.1839 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 205 | 2 | 168 | 1 | 0.1768 | 0.0847 | 0.0705 | -0.1136 | 0.0187 | 0.7581 | 1.0187 | 0.9809 | 0.0378 | -0.1841 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 206 | 2 | 169 | 1 | 0.1768 | 0.0847 | 0.071 | -0.1136 | 0.0187 | 0.7599 | 1.0187 | 0.9824 | 0.0363 | -0.1845 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 207 | 2 | 170 | 1 | 0.1768 | 0.0847 | 0.0696 | -0.1136 | 0.0186 | 0.7617 | 1.0186 | 0.9855 | 0.0331 | -0.1831 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 208 | 2 | 171 | 1 | 0.1768 | 0.0847 | 0.0696 | -0.1136 | 0.0185 | 0.7636 | 1.0185 | 0.9873 | 0.0312 | -0.1831 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 209 | 2 | 172 | 1 | 0.1768 | 0.0847 | 0.0704 | -0.1136 | 0.0185 | 0.7654 | 1.0185 | 0.9884 | 0.0301 | -0.184 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 210 | 2 | 173 | 1 | 0.1768 | 0.0847 | 0.07 | -0.1136 | 0.0184 | 0.7672 | 1.0184 | 0.9906 | 0.0278 | -0.1836 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 211 | 2 | 174 | 1 | 0.1768 | 0.0847 | 0.0695 | -0.1136 | 0.0183 | 0.7691 | 1.0183 | 0.9929 | 0.0254 | -0.1831 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 212 | 2 | 175 | 1 | 0.1768 | 0.0847 | 0.0695 | -0.1136 | 0.0183 | 0.7709 | 1.0183 | 0.9947 | 0.0236 | -0.1831 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 213 | 2 | 176 | 1 | 0.1768 | 0.0847 | 0.0689 | -0.1136 | 0.0182 | 0.7727 | 1.0182 | 0.9971 | 0.0211 | -0.1825 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 214 | 2 | 177 | 1 | 0.1768 | 0.0847 | 0.0696 | -0.1136 | 0.0182 | 0.7745 | 1.0182 | 0.9983 | 0.0199 | -0.1831 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 215 | 2 | 178 | 1 | 0.1768 | 0.0847 | 0.0708 | -0.1136 | 0.0181 | 0.7763 | 1.0181 | 0.9989 | 0.0192 | -0.1844 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 216 | 2 | 179 | 1 | 0.1768 | 0.0847 | 0.0717 | -0.1136 | 0.018 | 0.7781 | 1.018 | 0.9999 | 0.0181 | -0.1852 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 217 | 2 | 180 | 1 | 0.1768 | 0.0847 | 0.0738 | -0.1136 | 0.018 | 0.7799 | 1.018 | 0.9998 | 0.0182 | -0.1873 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 218 | 2 | 181 | 1 | 0.1768 | 0.0847 | 0.0729 | -0.1136 | 0.0179 | 0.7817 | 1.0179 | 1.0024 | 0.0156 | -0.1865 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 219 | 2 | 182 | 1 | 0.1768 | 0.0847 | 0.0739 | -0.1136 | 0.0179 | 0.7835 | 1.0179 | 1.0032 | 0.0146 | -0.1875 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 220 | 2 | 183 | 1 | 0.1768 | 0.0847 | 0.0746 | -0.1136 | 0.0178 | 0.7853 | 1.0178 | 1.0044 | 0.0134 | -0.1881 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 221 | 2 | 184 | 1 | 0.1768 | 0.0847 | 0.0765 | -0.1136 | 0.0178 | 0.7871 | 1.0178 | 1.0045 | 0.0133 | -0.1901 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 222 | 2 | 185 | 1 | 0.1768 | 0.0847 | 0.0754 | -0.1136 | 0.0177 | 0.7889 | 1.0177 | 1.0072 | 0.0105 | -0.189 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 223 | 2 | 186 | 1 | 0.1768 | 0.0847 | 0.0754 | -0.1136 | 0.0177 | 0.7906 | 1.0177 | 1.009 | 0.0086 | -0.1889 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 224 | 2 | 187 | 1 | 0.1768 | 0.0847 | 0.075 | -0.1136 | 0.0176 | 0.7924 | 1.0176 | 1.0112 | 0.0064 | -0.1885 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 225 | 2 | 188 | 1 | 0.1768 | 0.0847 | 0.0743 | -0.1136 | 0.0175 | 0.7942 | 1.0175 | 1.0136 | 0.004 | -0.1879 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 226 | 2 | 189 | 1 | 0.1768 | 0.0847 | 0.0741 | -0.1136 | 0.0175 | 0.7959 | 1.0175 | 1.0155 | 0.002 | -0.1877 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 227 | 2 | 190 | 1 | 0.1768 | 0.0847 | 0.0733 | -0.1136 | 0.0174 | 0.7977 | 1.0174 | 1.018 | -0.0006 | -0.1868 | move_2_q_gt_move_4, selected_move_2 |
| current | 228 | 2 | 190 | 2 | 0.1768 | 0.0847 | 0.0733 | -0.0194 | 0.0175 | 0.533 | 1.0175 | 1.1462 | -0.1287 | -0.0927 | move_2_q_gt_move_4, selected_move_2 |
| current | 229 | 2 | 190 | 3 | 0.1768 | 0.0847 | 0.0733 | 0.0716 | 0.0175 | 0.4006 | 1.0175 | 1.3937 | -0.3762 | -0.0016 | move_2_q_gt_move_4, selected_move_2 |
| current | 230 | 2 | 190 | 4 | 0.1768 | 0.0847 | 0.0733 | 0.073 | 0.0176 | 0.3212 | 1.0176 | 1.3202 | -0.3026 | -0.0002 | move_2_q_gt_move_4, selected_move_2 |
| current | 231 | 2 | 190 | 5 | 0.1768 | 0.0847 | 0.0733 | 0.0992 | 0.0176 | 0.2682 | 0.9199 | 1.2682 | -0.3483 | 0.0259 | selected_move_2 |
| current | 232 | 2 | 190 | 6 | 0.1768 | 0.0847 | 0.0733 | 0.0905 | 0.0176 | 0.2304 | 0.9504 | 1.2304 | -0.28 | 0.0173 | selected_move_2 |
| current | 233 | 2 | 190 | 7 | 0.1768 | 0.0847 | 0.0733 | 0.0767 | 0.0177 | 0.202 | 1.0035 | 1.202 | -0.1985 | 0.0034 | selected_move_2 |
| current | 234 | 2 | 190 | 8 | 0.1768 | 0.0847 | 0.0733 | 0.0796 | 0.0177 | 0.18 | 0.9918 | 1.18 | -0.1882 | 0.0064 | selected_move_2 |
| current | 235 | 2 | 190 | 9 | 0.1768 | 0.0847 | 0.0733 | 0.081 | 0.0177 | 0.1623 | 0.9866 | 1.1623 | -0.1757 | 0.0077 | selected_move_2 |
| current | 236 | 2 | 190 | 10 | 0.1768 | 0.0847 | 0.0733 | 0.0883 | 0.0178 | 0.1479 | 0.9586 | 1.1479 | -0.1893 | 0.0151 | selected_move_2 |
| current | 237 | 2 | 190 | 11 | 0.1768 | 0.0847 | 0.0733 | 0.0918 | 0.0178 | 0.1358 | 0.9461 | 1.1358 | -0.1897 | 0.0185 | selected_move_2 |
| current | 238 | 2 | 190 | 12 | 0.1768 | 0.0847 | 0.0733 | 0.0707 | 0.0179 | 0.1257 | 1.0179 | 1.1149 | -0.0971 | -0.0026 | move_2_q_gt_move_4, selected_move_2 |
| current | 239 | 2 | 190 | 13 | 0.1768 | 0.0847 | 0.0733 | 0.0569 | 0.0179 | 0.1169 | 1.0179 | 1.0486 | -0.0307 | -0.0164 | move_2_q_gt_move_4, selected_move_2 |
| current | 240 | 2 | 190 | 14 | 0.1768 | 0.0847 | 0.0733 | 0.0595 | 0.0179 | 0.1094 | 1.0179 | 1.0519 | -0.034 | -0.0138 | move_2_q_gt_move_4, selected_move_2 |
| current | 241 | 2 | 190 | 15 | 0.1768 | 0.0847 | 0.0733 | 0.0551 | 0.018 | 0.1027 | 1.018 | 1.0272 | -0.0092 | -0.0181 | move_2_q_gt_move_4, selected_move_2 |
| current | 242 | 2 | 190 | 16 | 0.1768 | 0.0847 | 0.0733 | 0.0548 | 0.018 | 0.0969 | 1.018 | 1.02 | -0.002 | -0.0184 | move_2_q_gt_move_4, selected_move_2 |
| current | 243 | 2 | 190 | 17 | 0.1768 | 0.0847 | 0.0733 | 0.0447 | 0.018 | 0.0917 | 1.018 | 0.9725 | 0.0456 | -0.0286 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 244 | 2 | 191 | 17 | 0.1768 | 0.0847 | 0.0746 | 0.0447 | 0.018 | 0.0919 | 1.018 | 0.9677 | 0.0503 | -0.0299 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 245 | 2 | 192 | 17 | 0.1768 | 0.0847 | 0.0749 | 0.0447 | 0.0179 | 0.0921 | 1.0179 | 0.9667 | 0.0513 | -0.0303 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 246 | 2 | 193 | 17 | 0.1768 | 0.0847 | 0.0753 | 0.0447 | 0.0179 | 0.0923 | 1.0179 | 0.9657 | 0.0522 | -0.0306 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 247 | 2 | 194 | 17 | 0.1768 | 0.0847 | 0.0751 | 0.0447 | 0.0178 | 0.0925 | 1.0178 | 0.9664 | 0.0514 | -0.0304 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 248 | 2 | 195 | 17 | 0.1768 | 0.0847 | 0.0751 | 0.0447 | 0.0178 | 0.0926 | 1.0178 | 0.9666 | 0.0512 | -0.0304 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 249 | 2 | 196 | 17 | 0.1768 | 0.0847 | 0.0747 | 0.0447 | 0.0177 | 0.0928 | 1.0177 | 0.9683 | 0.0494 | -0.03 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 250 | 2 | 197 | 17 | 0.1768 | 0.0847 | 0.0757 | 0.0447 | 0.0177 | 0.093 | 1.0177 | 0.9649 | 0.0528 | -0.031 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 251 | 2 | 198 | 17 | 0.1768 | 0.0847 | 0.0744 | 0.0447 | 0.0176 | 0.0932 | 1.0176 | 0.9696 | 0.048 | -0.0298 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 252 | 2 | 199 | 17 | 0.1768 | 0.0847 | 0.074 | 0.0447 | 0.0175 | 0.0934 | 1.0175 | 0.9714 | 0.0462 | -0.0293 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 253 | 2 | 200 | 17 | 0.1768 | 0.0847 | 0.0732 | 0.0447 | 0.0175 | 0.0936 | 1.0175 | 0.9744 | 0.0431 | -0.0285 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 254 | 2 | 201 | 17 | 0.1768 | 0.0847 | 0.0731 | 0.0447 | 0.0174 | 0.0938 | 1.0174 | 0.9751 | 0.0423 | -0.0284 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 255 | 2 | 202 | 17 | 0.1768 | 0.0847 | 0.0737 | 0.0447 | 0.0174 | 0.0939 | 1.0174 | 0.9731 | 0.0443 | -0.029 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 256 | 2 | 203 | 17 | 0.1768 | 0.0847 | 0.0733 | 0.0447 | 0.0173 | 0.0941 | 1.0173 | 0.9748 | 0.0425 | -0.0286 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 257 | 2 | 204 | 17 | 0.1768 | 0.0847 | 0.0725 | 0.0447 | 0.0173 | 0.0943 | 1.0173 | 0.978 | 0.0393 | -0.0278 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 258 | 2 | 205 | 17 | 0.1768 | 0.0847 | 0.0724 | 0.0447 | 0.0172 | 0.0945 | 1.0172 | 0.9784 | 0.0388 | -0.0277 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 259 | 2 | 206 | 17 | 0.1768 | 0.0847 | 0.0715 | 0.0447 | 0.0172 | 0.0947 | 1.0172 | 0.9818 | 0.0354 | -0.0268 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 260 | 2 | 207 | 17 | 0.1768 | 0.0847 | 0.0733 | 0.0447 | 0.0171 | 0.0949 | 1.0171 | 0.9753 | 0.0418 | -0.0287 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 261 | 2 | 208 | 17 | 0.1768 | 0.0847 | 0.074 | 0.0447 | 0.0171 | 0.095 | 1.0171 | 0.9729 | 0.0442 | -0.0294 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 262 | 2 | 209 | 17 | 0.1768 | 0.0847 | 0.0748 | 0.0447 | 0.017 | 0.0952 | 1.017 | 0.9705 | 0.0466 | -0.0301 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 263 | 2 | 210 | 17 | 0.1768 | 0.0847 | 0.0746 | 0.0447 | 0.017 | 0.0954 | 1.017 | 0.9713 | 0.0457 | -0.0299 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 264 | 2 | 211 | 17 | 0.1768 | 0.0847 | 0.0762 | 0.0447 | 0.0169 | 0.0956 | 1.0169 | 0.9655 | 0.0514 | -0.0316 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 265 | 2 | 212 | 17 | 0.1768 | 0.0847 | 0.076 | 0.0447 | 0.0169 | 0.0958 | 1.0169 | 0.9667 | 0.0502 | -0.0313 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 266 | 2 | 213 | 17 | 0.1768 | 0.0847 | 0.075 | 0.0447 | 0.0168 | 0.0959 | 1.0168 | 0.9703 | 0.0465 | -0.0303 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 267 | 2 | 214 | 17 | 0.1768 | 0.0847 | 0.0755 | 0.0447 | 0.0168 | 0.0961 | 1.0168 | 0.9687 | 0.0481 | -0.0308 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 268 | 2 | 215 | 17 | 0.1768 | 0.0847 | 0.0756 | 0.0447 | 0.0168 | 0.0963 | 1.0168 | 0.9684 | 0.0483 | -0.0309 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 269 | 2 | 216 | 17 | 0.1768 | 0.0847 | 0.0747 | 0.0447 | 0.0167 | 0.0965 | 1.0167 | 0.9719 | 0.0448 | -0.03 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 270 | 2 | 217 | 17 | 0.1768 | 0.0847 | 0.0734 | 0.0447 | 0.0167 | 0.0967 | 1.0167 | 0.9768 | 0.0399 | -0.0287 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 271 | 2 | 218 | 17 | 0.1768 | 0.0847 | 0.0731 | 0.0447 | 0.0166 | 0.0968 | 1.0166 | 0.9782 | 0.0384 | -0.0284 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 272 | 2 | 219 | 17 | 0.1768 | 0.0847 | 0.0721 | 0.0447 | 0.0166 | 0.097 | 1.0166 | 0.982 | 0.0346 | -0.0274 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 273 | 2 | 220 | 17 | 0.1768 | 0.0847 | 0.0712 | 0.0447 | 0.0165 | 0.0972 | 1.0165 | 0.9855 | 0.031 | -0.0265 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 274 | 2 | 221 | 17 | 0.1768 | 0.0847 | 0.0701 | 0.0447 | 0.0165 | 0.0974 | 1.0165 | 0.9898 | 0.0267 | -0.0254 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 275 | 2 | 222 | 17 | 0.1768 | 0.0847 | 0.0685 | 0.0447 | 0.0164 | 0.0976 | 1.0164 | 0.996 | 0.0204 | -0.0238 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 276 | 2 | 223 | 17 | 0.1768 | 0.0847 | 0.0671 | 0.0447 | 0.0164 | 0.0977 | 1.0164 | 1.0015 | 0.0149 | -0.0225 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 277 | 2 | 224 | 17 | 0.1768 | 0.0847 | 0.0675 | 0.0447 | 0.0163 | 0.0979 | 1.0163 | 1.0003 | 0.016 | -0.0228 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 278 | 2 | 225 | 17 | 0.1768 | 0.0847 | 0.0677 | 0.0447 | 0.0163 | 0.0981 | 1.0163 | 0.9997 | 0.0166 | -0.023 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 279 | 2 | 226 | 17 | 0.1768 | 0.0847 | 0.0679 | 0.0447 | 0.0163 | 0.0983 | 1.0163 | 0.9991 | 0.0171 | -0.0232 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 280 | 2 | 227 | 17 | 0.1768 | 0.0847 | 0.0686 | 0.0447 | 0.0162 | 0.0984 | 1.0162 | 0.9965 | 0.0197 | -0.024 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 281 | 2 | 228 | 17 | 0.1768 | 0.0847 | 0.0695 | 0.0447 | 0.0162 | 0.0986 | 1.0162 | 0.9935 | 0.0227 | -0.0248 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 282 | 2 | 229 | 17 | 0.1768 | 0.0847 | 0.0683 | 0.0447 | 0.0161 | 0.0988 | 1.0161 | 0.9983 | 0.0179 | -0.0236 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 283 | 2 | 230 | 17 | 0.1768 | 0.0847 | 0.0676 | 0.0447 | 0.0161 | 0.099 | 1.0161 | 1.0012 | 0.0149 | -0.0229 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 284 | 2 | 231 | 17 | 0.1768 | 0.0847 | 0.0661 | 0.0447 | 0.0161 | 0.0991 | 1.0161 | 1.007 | 0.009 | -0.0214 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 285 | 2 | 232 | 17 | 0.1768 | 0.0847 | 0.0664 | 0.0447 | 0.016 | 0.0993 | 1.016 | 1.0059 | 0.0102 | -0.0218 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 286 | 2 | 233 | 17 | 0.1768 | 0.0847 | 0.0665 | 0.0447 | 0.016 | 0.0995 | 1.016 | 1.0059 | 0.01 | -0.0218 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 287 | 2 | 233 | 17 | 0.1768 | 0.0847 | 0.0665 | 0.0447 | 0.016 | 0.0997 | 1.016 | 1.0061 | 0.0099 | -0.0218 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 288 | 2 | 234 | 17 | 0.1768 | 0.0847 | 0.0667 | 0.0447 | 0.016 | 0.0998 | 1.016 | 1.0053 | 0.0107 | -0.022 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 289 | 2 | 235 | 17 | 0.1768 | 0.0847 | 0.0679 | 0.0447 | 0.0159 | 0.1 | 1.0159 | 1.0008 | 0.0151 | -0.0232 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 290 | 2 | 236 | 17 | 0.1768 | 0.0847 | 0.0692 | 0.0447 | 0.0159 | 0.1002 | 1.0159 | 0.9961 | 0.0198 | -0.0245 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 291 | 2 | 237 | 17 | 0.1768 | 0.0847 | 0.0692 | 0.0447 | 0.0158 | 0.1004 | 1.0158 | 0.9964 | 0.0195 | -0.0245 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 292 | 2 | 238 | 17 | 0.1768 | 0.0847 | 0.0694 | 0.0447 | 0.0158 | 0.1005 | 1.0158 | 0.9956 | 0.0202 | -0.0247 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 293 | 2 | 239 | 17 | 0.1768 | 0.0847 | 0.0687 | 0.0447 | 0.0158 | 0.1007 | 1.0158 | 0.9983 | 0.0174 | -0.0241 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 294 | 2 | 240 | 17 | 0.1768 | 0.0847 | 0.0693 | 0.0447 | 0.0157 | 0.1009 | 1.0157 | 0.9963 | 0.0194 | -0.0246 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 295 | 2 | 241 | 17 | 0.1768 | 0.0847 | 0.068 | 0.0447 | 0.0157 | 0.101 | 1.0157 | 1.0013 | 0.0143 | -0.0234 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 296 | 2 | 242 | 17 | 0.1768 | 0.0847 | 0.0666 | 0.0447 | 0.0156 | 0.1012 | 1.0156 | 1.0072 | 0.0085 | -0.0219 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 297 | 2 | 243 | 17 | 0.1768 | 0.0847 | 0.067 | 0.0447 | 0.0156 | 0.1014 | 1.0156 | 1.0059 | 0.0097 | -0.0223 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 298 | 2 | 244 | 17 | 0.1768 | 0.0847 | 0.0659 | 0.0447 | 0.0156 | 0.1016 | 1.0156 | 1.0101 | 0.0055 | -0.0213 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 299 | 2 | 245 | 17 | 0.1768 | 0.0847 | 0.0664 | 0.0447 | 0.0155 | 0.1017 | 1.0155 | 1.0083 | 0.0073 | -0.0218 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 300 | 2 | 246 | 17 | 0.1768 | 0.0847 | 0.0667 | 0.0447 | 0.0155 | 0.1019 | 1.0155 | 1.0075 | 0.008 | -0.022 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 301 | 2 | 247 | 17 | 0.1768 | 0.0847 | 0.0677 | 0.0447 | 0.0155 | 0.1021 | 1.0155 | 1.0037 | 0.0118 | -0.023 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 302 | 2 | 248 | 17 | 0.1768 | 0.0847 | 0.0675 | 0.0447 | 0.0154 | 0.1022 | 1.0154 | 1.0045 | 0.011 | -0.0229 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 303 | 2 | 249 | 17 | 0.1768 | 0.0847 | 0.0687 | 0.0447 | 0.0154 | 0.1024 | 1.0154 | 1.0003 | 0.0151 | -0.024 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 304 | 2 | 250 | 17 | 0.1768 | 0.0847 | 0.0682 | 0.0447 | 0.0154 | 0.1026 | 1.0154 | 1.0022 | 0.0132 | -0.0235 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 305 | 2 | 251 | 17 | 0.1768 | 0.0847 | 0.0677 | 0.0447 | 0.0153 | 0.1027 | 1.0153 | 1.0044 | 0.0109 | -0.023 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 306 | 2 | 252 | 17 | 0.1768 | 0.0847 | 0.067 | 0.0447 | 0.0153 | 0.1029 | 1.0153 | 1.0071 | 0.0082 | -0.0224 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 307 | 2 | 253 | 17 | 0.1768 | 0.0847 | 0.0679 | 0.0447 | 0.0152 | 0.1031 | 1.0152 | 1.0039 | 0.0114 | -0.0232 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 308 | 2 | 254 | 17 | 0.1768 | 0.0847 | 0.068 | 0.0447 | 0.0152 | 0.1032 | 1.0152 | 1.0038 | 0.0114 | -0.0233 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 309 | 2 | 255 | 17 | 0.1768 | 0.0847 | 0.0676 | 0.0447 | 0.0152 | 0.1034 | 1.0152 | 1.0054 | 0.0098 | -0.0229 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 310 | 2 | 256 | 17 | 0.1768 | 0.0847 | 0.0682 | 0.0447 | 0.0151 | 0.1036 | 1.0151 | 1.0032 | 0.012 | -0.0236 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 311 | 2 | 257 | 17 | 0.1768 | 0.0847 | 0.0688 | 0.0447 | 0.0151 | 0.1037 | 1.0151 | 1.0012 | 0.0139 | -0.0241 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 312 | 2 | 258 | 17 | 0.1768 | 0.0847 | 0.0693 | 0.0447 | 0.0151 | 0.1039 | 1.0151 | 0.9995 | 0.0155 | -0.0246 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 313 | 2 | 259 | 17 | 0.1768 | 0.0847 | 0.0695 | 0.0447 | 0.015 | 0.1041 | 1.015 | 0.9987 | 0.0164 | -0.0249 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 314 | 2 | 260 | 17 | 0.1768 | 0.0847 | 0.0706 | 0.0447 | 0.015 | 0.1042 | 1.015 | 0.995 | 0.0201 | -0.0259 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 315 | 2 | 261 | 17 | 0.1768 | 0.0847 | 0.0717 | 0.0447 | 0.015 | 0.1044 | 1.015 | 0.9911 | 0.0239 | -0.027 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 316 | 2 | 262 | 17 | 0.1768 | 0.0847 | 0.073 | 0.0447 | 0.0149 | 0.1046 | 1.0149 | 0.9863 | 0.0286 | -0.0283 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 317 | 2 | 263 | 17 | 0.1768 | 0.0847 | 0.0722 | 0.0447 | 0.0149 | 0.1047 | 1.0149 | 0.9893 | 0.0256 | -0.0275 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 318 | 2 | 264 | 17 | 0.1768 | 0.0847 | 0.0722 | 0.0447 | 0.0149 | 0.1049 | 1.0149 | 0.9894 | 0.0254 | -0.0275 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 319 | 2 | 265 | 17 | 0.1768 | 0.0847 | 0.0717 | 0.0447 | 0.0148 | 0.1051 | 1.0148 | 0.9916 | 0.0233 | -0.027 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 320 | 2 | 266 | 17 | 0.1768 | 0.0847 | 0.071 | 0.0447 | 0.0148 | 0.1052 | 1.0148 | 0.9943 | 0.0205 | -0.0263 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 321 | 2 | 267 | 17 | 0.1768 | 0.0847 | 0.0706 | 0.0447 | 0.0148 | 0.1054 | 1.0148 | 0.996 | 0.0187 | -0.0259 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 322 | 2 | 268 | 17 | 0.1768 | 0.0847 | 0.0699 | 0.0447 | 0.0147 | 0.1056 | 1.0147 | 0.9987 | 0.016 | -0.0252 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 323 | 2 | 269 | 17 | 0.1768 | 0.0847 | 0.0692 | 0.0447 | 0.0147 | 0.1057 | 1.0147 | 1.0016 | 0.0131 | -0.0245 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 324 | 2 | 270 | 17 | 0.1768 | 0.0847 | 0.0681 | 0.0447 | 0.0147 | 0.1059 | 1.0147 | 1.0061 | 0.0086 | -0.0234 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 325 | 2 | 271 | 17 | 0.1768 | 0.0847 | 0.0678 | 0.0447 | 0.0146 | 0.1061 | 1.0146 | 1.0075 | 0.0072 | -0.0231 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 326 | 2 | 272 | 17 | 0.1768 | 0.0847 | 0.0683 | 0.0447 | 0.0146 | 0.1062 | 1.0146 | 1.0055 | 0.0091 | -0.0236 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 327 | 2 | 273 | 17 | 0.1768 | 0.0847 | 0.0695 | 0.0447 | 0.0146 | 0.1064 | 1.0146 | 1.0011 | 0.0135 | -0.0248 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 328 | 2 | 274 | 17 | 0.1768 | 0.0847 | 0.0692 | 0.0447 | 0.0146 | 0.1065 | 1.0146 | 1.0024 | 0.0121 | -0.0245 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 329 | 2 | 275 | 17 | 0.1768 | 0.0847 | 0.0689 | 0.0447 | 0.0145 | 0.1067 | 1.0145 | 1.0037 | 0.0109 | -0.0242 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 330 | 2 | 276 | 17 | 0.1768 | 0.0847 | 0.0686 | 0.0447 | 0.0145 | 0.1069 | 1.0145 | 1.0049 | 0.0096 | -0.024 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 331 | 2 | 277 | 17 | 0.1768 | 0.0847 | 0.0691 | 0.0447 | 0.0145 | 0.107 | 1.0145 | 1.0034 | 0.0111 | -0.0244 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 332 | 2 | 278 | 17 | 0.1768 | 0.0847 | 0.0698 | 0.0447 | 0.0144 | 0.1072 | 1.0144 | 1.0008 | 0.0137 | -0.0251 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 333 | 2 | 279 | 17 | 0.1768 | 0.0847 | 0.0693 | 0.0447 | 0.0144 | 0.1074 | 1.0144 | 1.0027 | 0.0117 | -0.0247 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 334 | 2 | 280 | 17 | 0.1768 | 0.0847 | 0.0693 | 0.0447 | 0.0144 | 0.1075 | 1.0144 | 1.003 | 0.0113 | -0.0246 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 335 | 2 | 281 | 17 | 0.1768 | 0.0847 | 0.0684 | 0.0447 | 0.0143 | 0.1077 | 1.0143 | 1.0066 | 0.0077 | -0.0237 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 336 | 2 | 282 | 17 | 0.1768 | 0.0847 | 0.0677 | 0.0447 | 0.0143 | 0.1078 | 1.0143 | 1.0095 | 0.0049 | -0.023 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 337 | 2 | 282 | 17 | 0.1768 | 0.0847 | 0.0677 | 0.0447 | 0.0143 | 0.108 | 1.0143 | 1.0096 | 0.0047 | -0.023 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 338 | 2 | 283 | 17 | 0.1768 | 0.0847 | 0.0668 | 0.0447 | 0.0143 | 0.1082 | 1.0143 | 1.0132 | 0.0011 | -0.0221 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 339 | 2 | 284 | 17 | 0.1768 | 0.0847 | 0.0666 | 0.0447 | 0.0143 | 0.1083 | 1.0143 | 1.0143 | -0.0 | -0.0219 | move_2_q_gt_move_4, selected_move_2 |
| current | 340 | 2 | 284 | 18 | 0.1768 | 0.0847 | 0.0666 | 0.0391 | 0.0143 | 0.1028 | 1.0143 | 0.9848 | 0.0295 | -0.0275 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 341 | 2 | 285 | 18 | 0.1768 | 0.0847 | 0.0668 | 0.0391 | 0.0143 | 0.1029 | 1.0143 | 0.9841 | 0.0302 | -0.0277 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 342 | 2 | 286 | 18 | 0.1768 | 0.0847 | 0.0662 | 0.0391 | 0.0142 | 0.1031 | 1.0142 | 0.9866 | 0.0277 | -0.0271 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 343 | 2 | 287 | 18 | 0.1768 | 0.0847 | 0.0659 | 0.0391 | 0.0142 | 0.1032 | 1.0142 | 0.9878 | 0.0264 | -0.0268 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 344 | 2 | 288 | 18 | 0.1768 | 0.0847 | 0.0653 | 0.0391 | 0.0142 | 0.1034 | 1.0142 | 0.9902 | 0.024 | -0.0262 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 345 | 2 | 289 | 18 | 0.1768 | 0.0847 | 0.0642 | 0.0391 | 0.0142 | 0.1035 | 1.0142 | 0.9945 | 0.0196 | -0.0251 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 346 | 2 | 290 | 18 | 0.1768 | 0.0847 | 0.0644 | 0.0391 | 0.0141 | 0.1037 | 1.0141 | 0.9941 | 0.02 | -0.0253 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 347 | 2 | 291 | 18 | 0.1768 | 0.0847 | 0.0651 | 0.0391 | 0.0141 | 0.1038 | 1.0141 | 0.9915 | 0.0226 | -0.026 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 348 | 2 | 292 | 18 | 0.1768 | 0.0847 | 0.0661 | 0.0391 | 0.0141 | 0.104 | 1.0141 | 0.9877 | 0.0264 | -0.027 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 349 | 2 | 293 | 18 | 0.1768 | 0.0847 | 0.0674 | 0.0391 | 0.014 | 0.1041 | 1.014 | 0.9829 | 0.0311 | -0.0283 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 350 | 2 | 294 | 18 | 0.1768 | 0.0847 | 0.0672 | 0.0391 | 0.014 | 0.1043 | 1.014 | 0.9838 | 0.0302 | -0.0281 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 351 | 2 | 295 | 18 | 0.1768 | 0.0847 | 0.0671 | 0.0391 | 0.014 | 0.1044 | 1.014 | 0.9844 | 0.0296 | -0.028 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 352 | 2 | 296 | 18 | 0.1768 | 0.0847 | 0.0685 | 0.0391 | 0.014 | 0.1046 | 1.014 | 0.9794 | 0.0345 | -0.0294 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 353 | 2 | 297 | 18 | 0.1768 | 0.0847 | 0.0687 | 0.0391 | 0.0139 | 0.1047 | 1.0139 | 0.9789 | 0.035 | -0.0296 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 354 | 2 | 298 | 18 | 0.1768 | 0.0847 | 0.0689 | 0.0391 | 0.0139 | 0.1049 | 1.0139 | 0.9781 | 0.0358 | -0.0298 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 355 | 2 | 299 | 18 | 0.1768 | 0.0847 | 0.0696 | 0.0391 | 0.0139 | 0.105 | 1.0139 | 0.9759 | 0.038 | -0.0305 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 356 | 2 | 300 | 18 | 0.1768 | 0.0847 | 0.0701 | 0.0391 | 0.0139 | 0.1052 | 1.0139 | 0.9741 | 0.0398 | -0.031 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 357 | 2 | 301 | 18 | 0.1768 | 0.0847 | 0.0708 | 0.0391 | 0.0138 | 0.1053 | 1.0138 | 0.9716 | 0.0423 | -0.0317 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 358 | 2 | 302 | 18 | 0.1768 | 0.0847 | 0.0706 | 0.0391 | 0.0138 | 0.1054 | 1.0138 | 0.9724 | 0.0414 | -0.0315 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 359 | 2 | 303 | 18 | 0.1768 | 0.0847 | 0.0701 | 0.0391 | 0.0138 | 0.1056 | 1.0138 | 0.9746 | 0.0392 | -0.031 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 360 | 2 | 304 | 18 | 0.1768 | 0.0847 | 0.0704 | 0.0391 | 0.0138 | 0.1057 | 1.0138 | 0.9735 | 0.0402 | -0.0313 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 361 | 2 | 305 | 18 | 0.1768 | 0.0847 | 0.0707 | 0.0391 | 0.0137 | 0.1059 | 1.0137 | 0.9727 | 0.041 | -0.0316 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 362 | 2 | 306 | 18 | 0.1768 | 0.0847 | 0.0693 | 0.0391 | 0.0137 | 0.106 | 1.0137 | 0.9779 | 0.0358 | -0.0302 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 363 | 2 | 307 | 18 | 0.1768 | 0.0847 | 0.0682 | 0.0391 | 0.0137 | 0.1062 | 1.0137 | 0.9819 | 0.0317 | -0.0291 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 364 | 2 | 308 | 18 | 0.1768 | 0.0847 | 0.068 | 0.0391 | 0.0136 | 0.1063 | 1.0136 | 0.983 | 0.0306 | -0.0289 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 365 | 2 | 309 | 18 | 0.1768 | 0.0847 | 0.0672 | 0.0391 | 0.0136 | 0.1065 | 1.0136 | 0.9861 | 0.0275 | -0.0281 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 366 | 2 | 310 | 18 | 0.1768 | 0.0847 | 0.0674 | 0.0391 | 0.0136 | 0.1066 | 1.0136 | 0.9856 | 0.028 | -0.0283 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 367 | 2 | 311 | 18 | 0.1768 | 0.0847 | 0.0675 | 0.0391 | 0.0136 | 0.1068 | 1.0136 | 0.9853 | 0.0283 | -0.0284 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 368 | 2 | 312 | 18 | 0.1768 | 0.0847 | 0.0677 | 0.0391 | 0.0135 | 0.1069 | 1.0135 | 0.9847 | 0.0288 | -0.0286 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 369 | 2 | 313 | 18 | 0.1768 | 0.0847 | 0.0684 | 0.0391 | 0.0135 | 0.1071 | 1.0135 | 0.9823 | 0.0312 | -0.0293 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 370 | 2 | 314 | 18 | 0.1768 | 0.0847 | 0.0686 | 0.0391 | 0.0135 | 0.1072 | 1.0135 | 0.9817 | 0.0318 | -0.0295 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 371 | 2 | 315 | 18 | 0.1768 | 0.0847 | 0.0692 | 0.0391 | 0.0135 | 0.1073 | 1.0135 | 0.9795 | 0.034 | -0.0301 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 372 | 2 | 316 | 18 | 0.1768 | 0.0847 | 0.0684 | 0.0391 | 0.0134 | 0.1075 | 1.0134 | 0.9827 | 0.0308 | -0.0293 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 373 | 2 | 317 | 18 | 0.1768 | 0.0847 | 0.0683 | 0.0391 | 0.0134 | 0.1076 | 1.0134 | 0.9832 | 0.0302 | -0.0292 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 374 | 2 | 318 | 18 | 0.1768 | 0.0847 | 0.0678 | 0.0391 | 0.0134 | 0.1078 | 1.0134 | 0.9853 | 0.0281 | -0.0287 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 375 | 2 | 319 | 18 | 0.1768 | 0.0847 | 0.0679 | 0.0391 | 0.0134 | 0.1079 | 1.0134 | 0.9849 | 0.0285 | -0.0288 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 376 | 2 | 320 | 18 | 0.1768 | 0.0847 | 0.0673 | 0.0391 | 0.0134 | 0.1081 | 1.0134 | 0.9875 | 0.0258 | -0.0282 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 377 | 2 | 321 | 18 | 0.1768 | 0.0847 | 0.0666 | 0.0391 | 0.0133 | 0.1082 | 1.0133 | 0.9903 | 0.023 | -0.0275 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 378 | 2 | 322 | 18 | 0.1768 | 0.0847 | 0.0661 | 0.0391 | 0.0133 | 0.1084 | 1.0133 | 0.9921 | 0.0212 | -0.027 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 379 | 2 | 323 | 18 | 0.1768 | 0.0847 | 0.0653 | 0.0391 | 0.0133 | 0.1085 | 1.0133 | 0.9953 | 0.018 | -0.0262 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 380 | 2 | 324 | 18 | 0.1768 | 0.0847 | 0.0643 | 0.0391 | 0.0133 | 0.1086 | 1.0133 | 0.9992 | 0.014 | -0.0252 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 381 | 2 | 325 | 18 | 0.1768 | 0.0847 | 0.0639 | 0.0391 | 0.0132 | 0.1088 | 1.0132 | 1.0012 | 0.012 | -0.0248 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 382 | 2 | 326 | 18 | 0.1768 | 0.0847 | 0.0633 | 0.0391 | 0.0132 | 0.1089 | 1.0132 | 1.0036 | 0.0096 | -0.0242 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 383 | 2 | 327 | 18 | 0.1768 | 0.0847 | 0.0626 | 0.0391 | 0.0132 | 0.1091 | 1.0132 | 1.0064 | 0.0067 | -0.0235 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 384 | 2 | 328 | 18 | 0.1768 | 0.0847 | 0.0623 | 0.0391 | 0.0132 | 0.1092 | 1.0132 | 1.0077 | 0.0055 | -0.0232 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1 | 0 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.221 | 0.1059 | 0.1331 | 0.018 | 0.1151 | 0.0 | move_2_score_gt_move_4 |
| current | 2 | 1 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.3126 | 0.1498 | 0.8126 | 0.6498 | 0.1628 | 0.0 | move_2_score_gt_move_4 |
| current | 3 | 1 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.3828 | 0.1834 | 1.0495 | 0.8501 | 0.1994 | 0.0 | move_2_score_gt_move_4 |
| current | 4 | 1 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.4421 | 0.2118 | 1.1921 | 0.9618 | 0.2303 | 0.0 | move_2_score_gt_move_4 |
| current | 5 | 1 | 0 | 0 | 0.1768 | 0.0847 | 0.0 | 0.0 | 0.4942 | 0.2368 | 1.2942 | 1.0368 | 0.2575 | 0.0 | move_2_score_gt_move_4 |
| current | 6 | 1 | 1 | 0 | 0.1768 | 0.0847 | 0.063 | 0.0 | 0.2707 | 0.2594 | 1.2707 | 1.0229 | 0.2478 | -0.063 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 7 | 1 | 2 | 0 | 0.1768 | 0.0847 | 0.0467 | 0.0 | 0.1949 | 0.2802 | 1.1918 | 1.1364 | 0.0554 | -0.0467 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 8 | 1 | 2 | 0 | 0.1768 | 0.0847 | 0.0467 | 0.0 | 0.2084 | 0.2995 | 1.0972 | 1.1467 | -0.0495 | -0.0467 | move_2_q_gt_move_4 |
| current | 9 | 1 | 2 | 0 | 0.1768 | 0.0847 | 0.0467 | 0.0 | 0.221 | 0.3177 | 1.1212 | 1.1844 | -0.0632 | -0.0467 | move_2_q_gt_move_4 |
| current | 10 | 1 | 2 | 1 | 0.1768 | 0.0847 | 0.0467 | -0.1136 | 0.233 | 0.1674 | 1.1478 | 0.1674 | 0.9803 | -0.1603 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 11 | 1 | 2 | 1 | 0.1768 | 0.0847 | 0.0467 | -0.1136 | 0.2444 | 0.1756 | 1.1887 | 0.1756 | 1.0131 | -0.1603 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 12 | 1 | 3 | 1 | 0.1768 | 0.0847 | 0.0582 | -0.1136 | 0.1914 | 0.1834 | 1.1914 | 0.1834 | 1.008 | -0.1718 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 13 | 1 | 4 | 1 | 0.1768 | 0.0847 | 0.0303 | -0.1136 | 0.1594 | 0.1909 | 1.0069 | 0.1909 | 0.816 | -0.1439 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 14 | 1 | 4 | 1 | 0.1768 | 0.0847 | 0.0303 | -0.1136 | 0.1654 | 0.1981 | 1.1287 | 0.1981 | 0.9306 | -0.1439 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 15 | 1 | 4 | 1 | 0.1768 | 0.0847 | 0.0303 | -0.1136 | 0.1712 | 0.2051 | 1.1712 | 0.2051 | 0.9662 | -0.1439 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 16 | 1 | 5 | 1 | 0.1768 | 0.0847 | 0.0286 | -0.1136 | 0.1474 | 0.2118 | 1.1474 | 0.2118 | 0.9356 | -0.1422 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 17 | 1 | 6 | 1 | 0.1768 | 0.0847 | 0.0398 | -0.1136 | 0.1302 | 0.2183 | 1.1302 | 0.2183 | 0.9119 | -0.1533 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 18 | 1 | 6 | 1 | 0.1768 | 0.0847 | 0.0398 | -0.1136 | 0.134 | 0.2246 | 1.134 | 0.2246 | 0.9093 | -0.1533 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 19 | 1 | 6 | 1 | 0.1768 | 0.0847 | 0.0398 | -0.1136 | 0.1376 | 0.2308 | 1.1376 | 0.2308 | 0.9069 | -0.1533 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 20 | 1 | 7 | 1 | 0.1768 | 0.0847 | 0.0565 | -0.1136 | 0.1236 | 0.2368 | 1.1236 | 0.2368 | 0.8868 | -0.17 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 21 | 1 | 8 | 1 | 0.1768 | 0.0847 | 0.0535 | -0.1136 | 0.1125 | 0.2426 | 1.1125 | 0.2426 | 0.8699 | -0.1671 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 22 | 2 | 9 | 1 | 0.1768 | 0.0847 | 0.0674 | -0.1136 | 0.1037 | 0.2483 | 1.1037 | 0.2483 | 0.8553 | -0.1809 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 23 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.0964 | 0.2539 | 1.0964 | 0.2539 | 0.8425 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 24 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.0984 | 0.2594 | 1.0984 | 0.5113 | 0.5872 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 25 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1005 | 0.2647 | 1.1005 | 0.5166 | 0.5839 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 26 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1025 | 0.27 | 1.1025 | 0.5219 | 0.5806 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 27 | 2 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1044 | 0.2751 | 1.1044 | 0.527 | 0.5774 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 28 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1063 | 0.2802 | 1.1063 | 0.532 | 0.5743 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 29 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1082 | 0.2851 | 1.0859 | 0.5314 | 0.5545 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 30 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1101 | 0.29 | 1.0611 | 0.5296 | 0.5316 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 31 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1119 | 0.2948 | 1.1119 | 0.5467 | 0.5652 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 32 | 1 | 10 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.1137 | 0.2995 | 1.1137 | 0.5514 | 0.5623 | -0.1568 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 33 | 1 | 11 | 1 | 0.1768 | 0.0847 | 0.046 | -0.1136 | 0.1058 | 0.3041 | 1.1058 | 0.5528 | 0.553 | -0.1595 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 34 | 1 | 11 | 1 | 0.1768 | 0.0847 | 0.046 | -0.1136 | 0.1074 | 0.3087 | 1.1074 | 0.5574 | 0.5501 | -0.1595 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 35 | 1 | 12 | 1 | 0.1768 | 0.0847 | 0.0464 | -0.1136 | 0.1006 | 0.3132 | 1.1006 | 0.5614 | 0.5392 | -0.1599 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 36 | 1 | 13 | 1 | 0.1768 | 0.0847 | 0.0308 | -0.1136 | 0.0947 | 0.3177 | 1.0947 | 0.5854 | 0.5093 | -0.1444 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 37 | 1 | 13 | 1 | 0.1768 | 0.0847 | 0.0308 | -0.1136 | 0.096 | 0.3221 | 1.096 | 0.5898 | 0.5063 | -0.1444 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 38 | 1 | 14 | 1 | 0.1768 | 0.0847 | 0.0309 | -0.1136 | 0.0908 | 0.3264 | 1.0908 | 0.594 | 0.4968 | -0.1445 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 39 | 1 | 15 | 1 | 0.1768 | 0.0847 | 0.0431 | -0.1136 | 0.0863 | 0.3306 | 1.0863 | 0.5827 | 0.5036 | -0.1567 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| current | 40 | 2 | 16 | 1 | 0.1768 | 0.0847 | 0.0579 | -0.1136 | 0.0822 | 0.3349 | 1.0822 | 0.5703 | 0.512 | -0.1715 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 41 | 2 | 17 | 1 | 0.1768 | 0.0847 | 0.0606 | -0.1136 | 0.0786 | 0.339 | 1.0786 | 0.5716 | 0.507 | -0.1742 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 42 | 2 | 18 | 1 | 0.1768 | 0.0847 | 0.0766 | -0.1136 | 0.0754 | 0.3431 | 1.0754 | 0.5604 | 0.5149 | -0.1901 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 43 | 2 | 19 | 1 | 0.1768 | 0.0847 | 0.0824 | -0.1136 | 0.0725 | 0.3472 | 1.0725 | 0.5594 | 0.513 | -0.1959 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 44 | 2 | 20 | 1 | 0.1768 | 0.0847 | 0.0841 | -0.1136 | 0.0698 | 0.3512 | 1.0698 | 0.562 | 0.5079 | -0.1977 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 45 | 2 | 21 | 1 | 0.1768 | 0.0847 | 0.0763 | -0.1136 | 0.0674 | 0.3552 | 1.0674 | 0.5727 | 0.4947 | -0.1899 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 46 | 2 | 22 | 1 | 0.1768 | 0.0847 | 0.0742 | -0.1136 | 0.0652 | 0.3591 | 1.0652 | 0.5786 | 0.4866 | -0.1877 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 47 | 2 | 23 | 1 | 0.1768 | 0.0847 | 0.0656 | -0.1136 | 0.0631 | 0.363 | 1.0631 | 0.5906 | 0.4726 | -0.1792 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 48 | 2 | 24 | 1 | 0.1768 | 0.0847 | 0.0591 | -0.1136 | 0.0613 | 0.3668 | 1.0613 | 0.601 | 0.4602 | -0.1726 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 49 | 2 | 25 | 1 | 0.1768 | 0.0847 | 0.0563 | -0.1136 | 0.0595 | 0.3706 | 1.0595 | 0.6078 | 0.4517 | -0.1698 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 50 | 2 | 25 | 1 | 0.1768 | 0.0847 | 0.0563 | -0.1136 | 0.0601 | 0.3744 | 1.0601 | 0.6115 | 0.4486 | -0.1698 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 51 | 2 | 26 | 1 | 0.1768 | 0.0847 | 0.0514 | -0.1136 | 0.0585 | 0.3781 | 1.0585 | 0.6205 | 0.4379 | -0.165 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 52 | 2 | 27 | 1 | 0.1768 | 0.0847 | 0.0549 | -0.1136 | 0.0569 | 0.3818 | 1.0569 | 0.6204 | 0.4365 | -0.1684 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 53 | 2 | 28 | 1 | 0.1768 | 0.0847 | 0.057 | -0.1136 | 0.0555 | 0.3854 | 1.0555 | 0.6218 | 0.4337 | -0.1706 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 54 | 2 | 29 | 1 | 0.1768 | 0.0847 | 0.0642 | -0.1136 | 0.0541 | 0.3891 | 1.0541 | 0.6181 | 0.4361 | -0.1777 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 55 | 2 | 30 | 1 | 0.1768 | 0.0847 | 0.0656 | -0.1136 | 0.0529 | 0.3927 | 1.0529 | 0.6202 | 0.4326 | -0.1792 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 56 | 2 | 31 | 1 | 0.1768 | 0.0847 | 0.0705 | -0.1136 | 0.0517 | 0.3962 | 1.0517 | 0.6191 | 0.4326 | -0.1841 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 57 | 2 | 32 | 1 | 0.1768 | 0.0847 | 0.0761 | -0.1136 | 0.0506 | 0.3997 | 1.0506 | 0.6175 | 0.4331 | -0.1896 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 58 | 2 | 33 | 1 | 0.1768 | 0.0847 | 0.0806 | -0.1136 | 0.0495 | 0.4032 | 1.0495 | 0.617 | 0.4325 | -0.1941 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 59 | 2 | 34 | 1 | 0.1768 | 0.0847 | 0.0729 | -0.1136 | 0.0485 | 0.4067 | 1.0485 | 0.6273 | 0.4212 | -0.1865 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 60 | 2 | 35 | 1 | 0.1768 | 0.0847 | 0.0679 | -0.1136 | 0.0476 | 0.4101 | 1.0476 | 0.6354 | 0.4121 | -0.1815 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 61 | 2 | 36 | 1 | 0.1768 | 0.0847 | 0.0654 | -0.1136 | 0.0467 | 0.4135 | 1.0467 | 0.6413 | 0.4054 | -0.179 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 62 | 2 | 37 | 1 | 0.1768 | 0.0847 | 0.0625 | -0.1136 | 0.0458 | 0.4169 | 1.0458 | 0.6476 | 0.3982 | -0.176 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 63 | 2 | 38 | 1 | 0.1768 | 0.0847 | 0.0536 | -0.1136 | 0.045 | 0.4202 | 1.045 | 0.6603 | 0.3847 | -0.1671 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 64 | 2 | 38 | 1 | 0.1768 | 0.0847 | 0.0536 | -0.1136 | 0.0453 | 0.4236 | 1.0453 | 0.6636 | 0.3817 | -0.1671 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 65 | 2 | 39 | 1 | 0.1768 | 0.0847 | 0.0487 | -0.1136 | 0.0446 | 0.4269 | 1.0446 | 0.6723 | 0.3722 | -0.1623 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 66 | 2 | 40 | 1 | 0.1768 | 0.0847 | 0.0439 | -0.1136 | 0.0438 | 0.4301 | 1.0438 | 0.6813 | 0.3625 | -0.1574 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 67 | 2 | 41 | 1 | 0.1768 | 0.0847 | 0.0404 | -0.1136 | 0.0431 | 0.4334 | 1.0431 | 0.6888 | 0.3543 | -0.1539 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 68 | 2 | 42 | 1 | 0.1768 | 0.0847 | 0.0424 | -0.1136 | 0.0424 | 0.4366 | 1.0424 | 0.6895 | 0.3529 | -0.156 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 69 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0417 | 0.4398 | 1.0417 | 0.6941 | 0.3476 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 70 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.042 | 0.443 | 1.042 | 0.6973 | 0.3448 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 71 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0423 | 0.4461 | 1.0423 | 0.7004 | 0.3419 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 72 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0426 | 0.4493 | 1.0426 | 0.7035 | 0.3391 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 73 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0429 | 0.4524 | 1.0429 | 0.7067 | 0.3363 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 74 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0432 | 0.4555 | 1.0432 | 0.7097 | 0.3335 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 75 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0435 | 0.4585 | 1.0435 | 0.7128 | 0.3307 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 76 | 2 | 43 | 1 | 0.1768 | 0.0847 | 0.0413 | -0.1136 | 0.0438 | 0.4616 | 1.0438 | 0.7159 | 0.3279 | -0.1548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 77 | 2 | 44 | 1 | 0.1768 | 0.0847 | 0.0372 | -0.1136 | 0.0431 | 0.4646 | 1.0431 | 0.7239 | 0.3192 | -0.1508 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 78 | 2 | 44 | 1 | 0.1768 | 0.0847 | 0.0372 | -0.1136 | 0.0434 | 0.4676 | 1.0434 | 0.7269 | 0.3165 | -0.1508 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 79 | 2 | 45 | 1 | 0.1768 | 0.0847 | 0.0344 | -0.1136 | 0.0427 | 0.4706 | 1.0427 | 0.7336 | 0.3091 | -0.148 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 80 | 2 | 46 | 1 | 0.1768 | 0.0847 | 0.0299 | -0.1136 | 0.0421 | 0.4736 | 1.0421 | 0.7425 | 0.2995 | -0.1435 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 81 | 2 | 47 | 1 | 0.1768 | 0.0847 | 0.0279 | -0.1136 | 0.0414 | 0.4765 | 1.0414 | 0.7483 | 0.2931 | -0.1414 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 82 | 2 | 48 | 1 | 0.1768 | 0.0847 | 0.0247 | -0.1136 | 0.0408 | 0.4794 | 1.0408 | 0.7558 | 0.285 | -0.1382 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 83 | 2 | 48 | 1 | 0.1768 | 0.0847 | 0.0247 | -0.1136 | 0.0411 | 0.4824 | 1.0411 | 0.7587 | 0.2824 | -0.1382 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 84 | 2 | 49 | 1 | 0.1768 | 0.0847 | 0.026 | -0.1136 | 0.0405 | 0.4852 | 1.0405 | 0.7598 | 0.2808 | -0.1395 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 85 | 2 | 50 | 1 | 0.1768 | 0.0847 | 0.028 | -0.1136 | 0.04 | 0.4881 | 1.04 | 0.7598 | 0.2802 | -0.1416 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 86 | 2 | 51 | 1 | 0.1768 | 0.0847 | 0.0297 | -0.1136 | 0.0394 | 0.491 | 1.0394 | 0.7603 | 0.2791 | -0.1433 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 87 | 2 | 52 | 1 | 0.1768 | 0.0847 | 0.0326 | -0.1136 | 0.0389 | 0.4938 | 1.0389 | 0.7592 | 0.2797 | -0.1462 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 88 | 2 | 53 | 1 | 0.1768 | 0.0847 | 0.0372 | -0.1136 | 0.0384 | 0.4967 | 1.0384 | 0.7561 | 0.2823 | -0.1507 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 89 | 2 | 54 | 1 | 0.1768 | 0.0847 | 0.0383 | -0.1136 | 0.0379 | 0.4995 | 1.0379 | 0.7575 | 0.2805 | -0.1518 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 90 | 2 | 55 | 1 | 0.1768 | 0.0847 | 0.0333 | -0.1136 | 0.0374 | 0.5023 | 1.0374 | 0.7668 | 0.2707 | -0.1468 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 91 | 2 | 56 | 1 | 0.1768 | 0.0847 | 0.0292 | -0.1136 | 0.037 | 0.5051 | 1.037 | 0.775 | 0.262 | -0.1428 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 92 | 2 | 57 | 1 | 0.1768 | 0.0847 | 0.0264 | -0.1136 | 0.0366 | 0.5078 | 1.0366 | 0.7817 | 0.2548 | -0.14 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 93 | 2 | 57 | 1 | 0.1768 | 0.0847 | 0.0264 | -0.1136 | 0.0368 | 0.5106 | 1.0368 | 0.7845 | 0.2523 | -0.14 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 94 | 2 | 58 | 1 | 0.1768 | 0.0847 | 0.025 | -0.1136 | 0.0363 | 0.5133 | 1.0363 | 0.7892 | 0.2472 | -0.1386 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 95 | 2 | 59 | 1 | 0.1768 | 0.0847 | 0.0289 | -0.1136 | 0.0359 | 0.516 | 1.0359 | 0.7865 | 0.2494 | -0.1424 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 96 | 2 | 60 | 1 | 0.1768 | 0.0847 | 0.028 | -0.1136 | 0.0355 | 0.5188 | 1.0355 | 0.7904 | 0.2451 | -0.1415 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 97 | 2 | 61 | 1 | 0.1768 | 0.0847 | 0.0276 | -0.1136 | 0.0351 | 0.5214 | 1.0351 | 0.7936 | 0.2415 | -0.1412 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 98 | 2 | 62 | 1 | 0.1768 | 0.0847 | 0.0249 | -0.1136 | 0.0347 | 0.5241 | 1.0347 | 0.8002 | 0.2346 | -0.1384 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 99 | 2 | 63 | 1 | 0.1768 | 0.0847 | 0.0249 | -0.1136 | 0.0344 | 0.5268 | 1.0344 | 0.8029 | 0.2315 | -0.1384 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 100 | 2 | 64 | 1 | 0.1768 | 0.0847 | 0.024 | -0.1136 | 0.034 | 0.5295 | 1.034 | 0.8068 | 0.2272 | -0.1375 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 101 | 2 | 65 | 1 | 0.1768 | 0.0847 | 0.0237 | -0.1136 | 0.0337 | 0.5321 | 1.0337 | 0.8098 | 0.2238 | -0.1373 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 102 | 2 | 66 | 1 | 0.1768 | 0.0847 | 0.0198 | -0.1136 | 0.0333 | 0.5347 | 1.0333 | 0.8183 | 0.215 | -0.1333 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 103 | 2 | 67 | 1 | 0.1768 | 0.0847 | 0.02 | -0.1136 | 0.033 | 0.5373 | 1.033 | 0.8206 | 0.2124 | -0.1336 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 104 | 2 | 68 | 1 | 0.1768 | 0.0847 | 0.0236 | -0.1136 | 0.0327 | 0.5399 | 1.0327 | 0.8179 | 0.2148 | -0.1371 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 105 | 2 | 69 | 1 | 0.1768 | 0.0847 | 0.02 | -0.1136 | 0.0324 | 0.5425 | 1.0324 | 0.8258 | 0.2065 | -0.1336 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 106 | 2 | 70 | 1 | 0.1768 | 0.0847 | 0.0143 | -0.1136 | 0.0321 | 0.5451 | 1.0321 | 0.8374 | 0.1947 | -0.1278 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 107 | 2 | 70 | 1 | 0.1768 | 0.0847 | 0.0143 | -0.1136 | 0.0322 | 0.5477 | 1.0322 | 0.8399 | 0.1923 | -0.1278 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 108 | 2 | 71 | 1 | 0.1768 | 0.0847 | 0.0141 | -0.1136 | 0.0319 | 0.5502 | 1.0319 | 0.8428 | 0.1891 | -0.1276 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 109 | 2 | 72 | 1 | 0.1768 | 0.0847 | 0.0091 | -0.1136 | 0.0316 | 0.5528 | 1.0316 | 0.8537 | 0.1779 | -0.1226 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 110 | 2 | 73 | 1 | 0.1768 | 0.0847 | 0.0112 | -0.1136 | 0.0313 | 0.5553 | 1.0313 | 0.8526 | 0.1787 | -0.1248 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 111 | 2 | 74 | 1 | 0.1768 | 0.0847 | 0.0134 | -0.1136 | 0.031 | 0.5578 | 1.031 | 0.8514 | 0.1796 | -0.127 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 112 | 2 | 75 | 1 | 0.1768 | 0.0847 | 0.0157 | -0.1136 | 0.0308 | 0.5603 | 1.0308 | 0.8502 | 0.1805 | -0.1293 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 113 | 2 | 76 | 1 | 0.1768 | 0.0847 | 0.0188 | -0.1136 | 0.0305 | 0.5628 | 1.0305 | 0.848 | 0.1826 | -0.1323 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 114 | 2 | 77 | 1 | 0.1768 | 0.0847 | 0.0206 | -0.1136 | 0.0303 | 0.5653 | 1.0303 | 0.8477 | 0.1826 | -0.1341 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 115 | 2 | 78 | 1 | 0.1768 | 0.0847 | 0.0208 | -0.1136 | 0.03 | 0.5678 | 1.03 | 0.8499 | 0.1801 | -0.1343 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 116 | 2 | 79 | 1 | 0.1768 | 0.0847 | 0.0206 | -0.1136 | 0.0298 | 0.5702 | 1.0298 | 0.8526 | 0.1772 | -0.1342 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 117 | 2 | 80 | 1 | 0.1768 | 0.0847 | 0.0197 | -0.1136 | 0.0295 | 0.5727 | 1.0295 | 0.8564 | 0.1731 | -0.1333 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 118 | 2 | 81 | 1 | 0.1768 | 0.0847 | 0.0199 | -0.1136 | 0.0293 | 0.5751 | 1.0293 | 0.8586 | 0.1707 | -0.1335 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 119 | 2 | 82 | 1 | 0.1768 | 0.0847 | 0.019 | -0.1136 | 0.0291 | 0.5776 | 1.0291 | 0.8623 | 0.1667 | -0.1326 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 120 | 2 | 83 | 1 | 0.1768 | 0.0847 | 0.0176 | -0.1136 | 0.0288 | 0.58 | 1.0288 | 0.867 | 0.1619 | -0.1312 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 121 | 2 | 84 | 1 | 0.1768 | 0.0847 | 0.0198 | -0.1136 | 0.0286 | 0.5824 | 1.0286 | 0.866 | 0.1626 | -0.1333 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 122 | 2 | 85 | 1 | 0.1768 | 0.0847 | 0.0227 | -0.1136 | 0.0284 | 0.5848 | 1.0284 | 0.8641 | 0.1643 | -0.1363 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 123 | 2 | 86 | 1 | 0.1768 | 0.0847 | 0.0259 | -0.1136 | 0.0282 | 0.5872 | 1.0282 | 0.8618 | 0.1663 | -0.1394 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 124 | 2 | 87 | 1 | 0.1768 | 0.0847 | 0.0265 | -0.1136 | 0.028 | 0.5896 | 1.028 | 0.8633 | 0.1647 | -0.1401 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 125 | 2 | 88 | 1 | 0.1768 | 0.0847 | 0.0278 | -0.1136 | 0.0278 | 0.5919 | 1.0278 | 0.8639 | 0.1638 | -0.1413 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 126 | 2 | 89 | 1 | 0.1768 | 0.0847 | 0.0288 | -0.1136 | 0.0276 | 0.5943 | 1.0276 | 0.8648 | 0.1628 | -0.1424 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 127 | 2 | 90 | 1 | 0.1768 | 0.0847 | 0.0295 | -0.1136 | 0.0274 | 0.5967 | 1.0274 | 0.8662 | 0.1612 | -0.1431 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 128 | 2 | 91 | 1 | 0.1768 | 0.0847 | 0.0282 | -0.1136 | 0.0272 | 0.599 | 1.0272 | 0.8704 | 0.1568 | -0.1417 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 129 | 2 | 92 | 1 | 0.1768 | 0.0847 | 0.027 | -0.1136 | 0.027 | 0.6013 | 1.027 | 0.8744 | 0.1526 | -0.1406 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 130 | 2 | 93 | 1 | 0.1768 | 0.0847 | 0.0271 | -0.1136 | 0.0268 | 0.6037 | 1.0268 | 0.8766 | 0.1502 | -0.1406 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 131 | 2 | 94 | 1 | 0.1768 | 0.0847 | 0.0308 | -0.1136 | 0.0266 | 0.606 | 1.0266 | 0.8738 | 0.1528 | -0.1443 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 132 | 2 | 95 | 1 | 0.1768 | 0.0847 | 0.0347 | -0.1136 | 0.0265 | 0.6083 | 1.0265 | 0.8709 | 0.1555 | -0.1482 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 133 | 2 | 96 | 1 | 0.1768 | 0.0847 | 0.0391 | -0.1136 | 0.0263 | 0.6106 | 1.0263 | 0.8675 | 0.1588 | -0.1527 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 134 | 2 | 97 | 1 | 0.1768 | 0.0847 | 0.0398 | -0.1136 | 0.0261 | 0.6129 | 1.0261 | 0.8689 | 0.1572 | -0.1534 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 135 | 2 | 98 | 1 | 0.1768 | 0.0847 | 0.0395 | -0.1136 | 0.0259 | 0.6152 | 1.0259 | 0.8716 | 0.1543 | -0.1531 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 136 | 2 | 99 | 1 | 0.1768 | 0.0847 | 0.0404 | -0.1136 | 0.0258 | 0.6174 | 1.0258 | 0.8728 | 0.153 | -0.154 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 137 | 2 | 100 | 1 | 0.1768 | 0.0847 | 0.0428 | -0.1136 | 0.0256 | 0.6197 | 1.0256 | 0.8721 | 0.1535 | -0.1564 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 138 | 2 | 101 | 1 | 0.1768 | 0.0847 | 0.0442 | -0.1136 | 0.0255 | 0.622 | 1.0255 | 0.8727 | 0.1527 | -0.1577 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 139 | 2 | 102 | 1 | 0.1768 | 0.0847 | 0.0448 | -0.1136 | 0.0253 | 0.6242 | 1.0253 | 0.8742 | 0.1511 | -0.1584 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 140 | 2 | 103 | 1 | 0.1768 | 0.0847 | 0.0434 | -0.1136 | 0.0251 | 0.6265 | 1.0251 | 0.8781 | 0.147 | -0.157 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 141 | 2 | 104 | 1 | 0.1768 | 0.0847 | 0.0432 | -0.1136 | 0.025 | 0.6287 | 1.025 | 0.8806 | 0.1444 | -0.1567 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 142 | 2 | 105 | 1 | 0.1768 | 0.0847 | 0.0419 | -0.1136 | 0.0248 | 0.6309 | 1.0248 | 0.8845 | 0.1404 | -0.1554 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 143 | 2 | 106 | 1 | 0.1768 | 0.0847 | 0.0417 | -0.1136 | 0.0247 | 0.6331 | 1.0247 | 0.8869 | 0.1378 | -0.1553 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 144 | 2 | 107 | 1 | 0.1768 | 0.0847 | 0.0439 | -0.1136 | 0.0246 | 0.6353 | 1.0246 | 0.8864 | 0.1382 | -0.1575 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 145 | 2 | 108 | 1 | 0.1768 | 0.0847 | 0.0456 | -0.1136 | 0.0244 | 0.6375 | 1.0244 | 0.8866 | 0.1378 | -0.1592 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 146 | 2 | 109 | 1 | 0.1768 | 0.0847 | 0.0456 | -0.1136 | 0.0243 | 0.6397 | 1.0243 | 0.8888 | 0.1355 | -0.1591 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 147 | 2 | 110 | 1 | 0.1768 | 0.0847 | 0.049 | -0.1136 | 0.0241 | 0.6419 | 1.0241 | 0.887 | 0.1371 | -0.1626 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 148 | 2 | 111 | 1 | 0.1768 | 0.0847 | 0.0505 | -0.1136 | 0.024 | 0.6441 | 1.024 | 0.8875 | 0.1365 | -0.1641 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 149 | 2 | 112 | 1 | 0.1768 | 0.0847 | 0.0537 | -0.1136 | 0.0239 | 0.6463 | 1.0239 | 0.8862 | 0.1376 | -0.1672 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 150 | 2 | 113 | 1 | 0.1768 | 0.0847 | 0.0529 | -0.1136 | 0.0237 | 0.6484 | 1.0237 | 0.8892 | 0.1345 | -0.1665 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 151 | 2 | 114 | 1 | 0.1768 | 0.0847 | 0.0516 | -0.1136 | 0.0236 | 0.6506 | 1.0236 | 0.8928 | 0.1308 | -0.1652 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 152 | 2 | 115 | 1 | 0.1768 | 0.0847 | 0.052 | -0.1136 | 0.0235 | 0.6528 | 1.0235 | 0.8945 | 0.1289 | -0.1655 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 153 | 2 | 116 | 1 | 0.1768 | 0.0847 | 0.0512 | -0.1136 | 0.0234 | 0.6549 | 1.0234 | 0.8975 | 0.1258 | -0.1648 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 154 | 2 | 117 | 1 | 0.1768 | 0.0847 | 0.0503 | -0.1136 | 0.0232 | 0.657 | 1.0232 | 0.9007 | 0.1225 | -0.1639 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 155 | 2 | 118 | 1 | 0.1768 | 0.0847 | 0.0508 | -0.1136 | 0.0231 | 0.6592 | 1.0231 | 0.9023 | 0.1209 | -0.1644 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 156 | 2 | 119 | 1 | 0.1768 | 0.0847 | 0.0499 | -0.1136 | 0.023 | 0.6613 | 1.023 | 0.9054 | 0.1176 | -0.1635 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 157 | 2 | 120 | 1 | 0.1768 | 0.0847 | 0.0507 | -0.1136 | 0.0229 | 0.6634 | 1.0229 | 0.9066 | 0.1162 | -0.1642 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 158 | 2 | 121 | 1 | 0.1768 | 0.0847 | 0.0528 | -0.1136 | 0.0228 | 0.6655 | 1.0228 | 0.9064 | 0.1164 | -0.1664 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 159 | 2 | 122 | 1 | 0.1768 | 0.0847 | 0.0541 | -0.1136 | 0.0227 | 0.6676 | 1.0227 | 0.907 | 0.1156 | -0.1677 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 160 | 2 | 123 | 1 | 0.1768 | 0.0847 | 0.054 | -0.1136 | 0.0225 | 0.6697 | 1.0225 | 0.9093 | 0.1133 | -0.1676 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 161 | 2 | 124 | 1 | 0.1768 | 0.0847 | 0.0567 | -0.1136 | 0.0224 | 0.6718 | 1.0224 | 0.9085 | 0.1139 | -0.1702 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 162 | 2 | 125 | 1 | 0.1768 | 0.0847 | 0.0541 | -0.1136 | 0.0223 | 0.6739 | 1.0223 | 0.9134 | 0.109 | -0.1677 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 163 | 2 | 126 | 1 | 0.1768 | 0.0847 | 0.055 | -0.1136 | 0.0222 | 0.676 | 1.0222 | 0.9145 | 0.1077 | -0.1685 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 164 | 2 | 127 | 1 | 0.1768 | 0.0847 | 0.0556 | -0.1136 | 0.0221 | 0.678 | 1.0221 | 0.9159 | 0.1062 | -0.1691 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 165 | 2 | 128 | 1 | 0.1768 | 0.0847 | 0.0537 | -0.1136 | 0.022 | 0.6801 | 1.022 | 0.92 | 0.102 | -0.1673 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 166 | 2 | 129 | 1 | 0.1768 | 0.0847 | 0.0526 | -0.1136 | 0.0219 | 0.6822 | 1.0219 | 0.9233 | 0.0986 | -0.1661 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 167 | 2 | 130 | 1 | 0.1768 | 0.0847 | 0.0542 | -0.1136 | 0.0218 | 0.6842 | 1.0218 | 0.9235 | 0.0983 | -0.1678 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 168 | 2 | 131 | 1 | 0.1768 | 0.0847 | 0.0552 | -0.1136 | 0.0217 | 0.6862 | 1.0217 | 0.9245 | 0.0972 | -0.1688 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 169 | 2 | 132 | 1 | 0.1768 | 0.0847 | 0.0541 | -0.1136 | 0.0216 | 0.6883 | 1.0216 | 0.9277 | 0.0939 | -0.1677 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 170 | 2 | 133 | 1 | 0.1768 | 0.0847 | 0.0558 | -0.1136 | 0.0215 | 0.6903 | 1.0215 | 0.9279 | 0.0936 | -0.1694 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 171 | 2 | 134 | 1 | 0.1768 | 0.0847 | 0.0576 | -0.1136 | 0.0214 | 0.6923 | 1.0214 | 0.9281 | 0.0933 | -0.1711 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 172 | 2 | 135 | 1 | 0.1768 | 0.0847 | 0.0597 | -0.1136 | 0.0213 | 0.6944 | 1.0213 | 0.928 | 0.0933 | -0.1732 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 173 | 2 | 136 | 1 | 0.1768 | 0.0847 | 0.0609 | -0.1136 | 0.0212 | 0.6964 | 1.0212 | 0.9287 | 0.0925 | -0.1745 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 174 | 2 | 137 | 1 | 0.1768 | 0.0847 | 0.0605 | -0.1136 | 0.0211 | 0.6984 | 1.0211 | 0.9311 | 0.09 | -0.1741 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 175 | 2 | 138 | 1 | 0.1768 | 0.0847 | 0.0596 | -0.1136 | 0.021 | 0.7004 | 1.021 | 0.9341 | 0.0869 | -0.1731 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 176 | 2 | 139 | 1 | 0.1768 | 0.0847 | 0.0589 | -0.1136 | 0.0209 | 0.7024 | 1.0209 | 0.9368 | 0.0841 | -0.1724 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 177 | 2 | 140 | 1 | 0.1768 | 0.0847 | 0.0603 | -0.1136 | 0.0209 | 0.7044 | 1.0209 | 0.9373 | 0.0835 | -0.1738 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 178 | 2 | 141 | 1 | 0.1768 | 0.0847 | 0.0611 | -0.1136 | 0.0208 | 0.7064 | 1.0208 | 0.9385 | 0.0823 | -0.1747 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 179 | 2 | 142 | 1 | 0.1768 | 0.0847 | 0.061 | -0.1136 | 0.0207 | 0.7084 | 1.0207 | 0.9406 | 0.0801 | -0.1745 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 180 | 2 | 143 | 1 | 0.1768 | 0.0847 | 0.0606 | -0.1136 | 0.0206 | 0.7103 | 1.0206 | 0.9429 | 0.0776 | -0.1742 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 181 | 2 | 144 | 1 | 0.1768 | 0.0847 | 0.0602 | -0.1136 | 0.0205 | 0.7123 | 1.0205 | 0.9453 | 0.0752 | -0.1738 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 182 | 2 | 145 | 1 | 0.1768 | 0.0847 | 0.0599 | -0.1136 | 0.0204 | 0.7143 | 1.0204 | 0.9476 | 0.0728 | -0.1734 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 183 | 2 | 146 | 1 | 0.1768 | 0.0847 | 0.0595 | -0.1136 | 0.0203 | 0.7162 | 1.0203 | 0.95 | 0.0703 | -0.173 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 184 | 2 | 147 | 1 | 0.1768 | 0.0847 | 0.0598 | -0.1136 | 0.0203 | 0.7182 | 1.0203 | 0.9517 | 0.0686 | -0.1733 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 185 | 2 | 148 | 1 | 0.1768 | 0.0847 | 0.0602 | -0.1136 | 0.0202 | 0.7201 | 1.0202 | 0.9532 | 0.067 | -0.1737 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 186 | 2 | 149 | 1 | 0.1768 | 0.0847 | 0.0584 | -0.1136 | 0.0201 | 0.7221 | 1.0201 | 0.957 | 0.0631 | -0.172 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 187 | 2 | 150 | 1 | 0.1768 | 0.0847 | 0.0599 | -0.1136 | 0.02 | 0.724 | 1.02 | 0.9573 | 0.0627 | -0.1735 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 188 | 2 | 151 | 1 | 0.1768 | 0.0847 | 0.0611 | -0.1136 | 0.0199 | 0.7259 | 1.0199 | 0.958 | 0.0619 | -0.1747 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 189 | 2 | 152 | 1 | 0.1768 | 0.0847 | 0.063 | -0.1136 | 0.0199 | 0.7279 | 1.0199 | 0.958 | 0.0618 | -0.1766 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 190 | 2 | 153 | 1 | 0.1768 | 0.0847 | 0.0633 | -0.1136 | 0.0198 | 0.7298 | 1.0198 | 0.9596 | 0.0601 | -0.1769 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 191 | 2 | 154 | 1 | 0.1768 | 0.0847 | 0.0657 | -0.1136 | 0.0197 | 0.7317 | 1.0197 | 0.9592 | 0.0605 | -0.1793 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 192 | 2 | 155 | 1 | 0.1768 | 0.0847 | 0.0668 | -0.1136 | 0.0196 | 0.7336 | 1.0196 | 0.96 | 0.0596 | -0.1804 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 193 | 2 | 156 | 1 | 0.1768 | 0.0847 | 0.0669 | -0.1136 | 0.0196 | 0.7355 | 1.0196 | 0.9619 | 0.0577 | -0.1804 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 194 | 2 | 157 | 1 | 0.1768 | 0.0847 | 0.0675 | -0.1136 | 0.0195 | 0.7374 | 1.0195 | 0.9632 | 0.0563 | -0.1811 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 195 | 2 | 158 | 1 | 0.1768 | 0.0847 | 0.0692 | -0.1136 | 0.0194 | 0.7393 | 1.0194 | 0.9634 | 0.056 | -0.1828 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 196 | 2 | 159 | 1 | 0.1768 | 0.0847 | 0.0698 | -0.1136 | 0.0193 | 0.7412 | 1.0193 | 0.9648 | 0.0545 | -0.1833 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 197 | 2 | 160 | 1 | 0.1768 | 0.0847 | 0.0688 | -0.1136 | 0.0193 | 0.7431 | 1.0193 | 0.9676 | 0.0517 | -0.1824 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 198 | 2 | 161 | 1 | 0.1768 | 0.0847 | 0.0698 | -0.1136 | 0.0192 | 0.745 | 1.0192 | 0.9685 | 0.0507 | -0.1834 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 199 | 2 | 162 | 1 | 0.1768 | 0.0847 | 0.071 | -0.1136 | 0.0191 | 0.7469 | 1.0191 | 0.9693 | 0.0498 | -0.1846 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 200 | 2 | 163 | 1 | 0.1768 | 0.0847 | 0.0704 | -0.1136 | 0.0191 | 0.7488 | 1.0191 | 0.9718 | 0.0473 | -0.1839 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 201 | 2 | 164 | 1 | 0.1768 | 0.0847 | 0.0705 | -0.1136 | 0.019 | 0.7506 | 1.019 | 0.9735 | 0.0455 | -0.1841 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 202 | 2 | 165 | 1 | 0.1768 | 0.0847 | 0.0701 | -0.1136 | 0.0189 | 0.7525 | 1.0189 | 0.9758 | 0.0431 | -0.1836 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 203 | 2 | 166 | 1 | 0.1768 | 0.0847 | 0.0696 | -0.1136 | 0.0189 | 0.7544 | 1.0189 | 0.9781 | 0.0408 | -0.1832 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 204 | 2 | 167 | 1 | 0.1768 | 0.0847 | 0.0703 | -0.1136 | 0.0188 | 0.7562 | 1.0188 | 0.9793 | 0.0395 | -0.1839 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 205 | 2 | 168 | 1 | 0.1768 | 0.0847 | 0.0705 | -0.1136 | 0.0187 | 0.7581 | 1.0187 | 0.9809 | 0.0378 | -0.1841 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 206 | 2 | 169 | 1 | 0.1768 | 0.0847 | 0.071 | -0.1136 | 0.0187 | 0.7599 | 1.0187 | 0.9824 | 0.0363 | -0.1845 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 207 | 2 | 170 | 1 | 0.1768 | 0.0847 | 0.0696 | -0.1136 | 0.0186 | 0.7617 | 1.0186 | 0.9855 | 0.0331 | -0.1831 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 208 | 2 | 171 | 1 | 0.1768 | 0.0847 | 0.0696 | -0.1136 | 0.0185 | 0.7636 | 1.0185 | 0.9873 | 0.0312 | -0.1831 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 209 | 2 | 172 | 1 | 0.1768 | 0.0847 | 0.0704 | -0.1136 | 0.0185 | 0.7654 | 1.0185 | 0.9884 | 0.0301 | -0.184 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 210 | 2 | 173 | 1 | 0.1768 | 0.0847 | 0.07 | -0.1136 | 0.0184 | 0.7672 | 1.0184 | 0.9906 | 0.0278 | -0.1836 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 211 | 2 | 174 | 1 | 0.1768 | 0.0847 | 0.0695 | -0.1136 | 0.0183 | 0.7691 | 1.0183 | 0.9929 | 0.0254 | -0.1831 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 212 | 2 | 175 | 1 | 0.1768 | 0.0847 | 0.0695 | -0.1136 | 0.0183 | 0.7709 | 1.0183 | 0.9947 | 0.0236 | -0.1831 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 213 | 2 | 176 | 1 | 0.1768 | 0.0847 | 0.0689 | -0.1136 | 0.0182 | 0.7727 | 1.0182 | 0.9971 | 0.0211 | -0.1825 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 214 | 2 | 177 | 1 | 0.1768 | 0.0847 | 0.0696 | -0.1136 | 0.0182 | 0.7745 | 1.0182 | 0.9983 | 0.0199 | -0.1831 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 215 | 2 | 178 | 1 | 0.1768 | 0.0847 | 0.0708 | -0.1136 | 0.0181 | 0.7763 | 1.0181 | 0.9989 | 0.0192 | -0.1844 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 216 | 2 | 179 | 1 | 0.1768 | 0.0847 | 0.0717 | -0.1136 | 0.018 | 0.7781 | 1.018 | 0.9999 | 0.0181 | -0.1852 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 217 | 2 | 180 | 1 | 0.1768 | 0.0847 | 0.0738 | -0.1136 | 0.018 | 0.7799 | 1.018 | 0.9998 | 0.0182 | -0.1873 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 218 | 2 | 181 | 1 | 0.1768 | 0.0847 | 0.0729 | -0.1136 | 0.0179 | 0.7817 | 1.0179 | 1.0024 | 0.0156 | -0.1865 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 219 | 2 | 182 | 1 | 0.1768 | 0.0847 | 0.0739 | -0.1136 | 0.0179 | 0.7835 | 1.0179 | 1.0032 | 0.0146 | -0.1875 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 220 | 2 | 183 | 1 | 0.1768 | 0.0847 | 0.0746 | -0.1136 | 0.0178 | 0.7853 | 1.0178 | 1.0044 | 0.0134 | -0.1881 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 221 | 2 | 184 | 1 | 0.1768 | 0.0847 | 0.0765 | -0.1136 | 0.0178 | 0.7871 | 1.0178 | 1.0045 | 0.0133 | -0.1901 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 222 | 2 | 185 | 1 | 0.1768 | 0.0847 | 0.0754 | -0.1136 | 0.0177 | 0.7889 | 1.0177 | 1.0072 | 0.0105 | -0.189 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 223 | 2 | 186 | 1 | 0.1768 | 0.0847 | 0.0754 | -0.1136 | 0.0177 | 0.7906 | 1.0177 | 1.009 | 0.0086 | -0.1889 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 224 | 2 | 187 | 1 | 0.1768 | 0.0847 | 0.075 | -0.1136 | 0.0176 | 0.7924 | 1.0176 | 1.0112 | 0.0064 | -0.1885 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 225 | 2 | 188 | 1 | 0.1768 | 0.0847 | 0.0743 | -0.1136 | 0.0175 | 0.7942 | 1.0175 | 1.0136 | 0.004 | -0.1879 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 226 | 2 | 189 | 1 | 0.1768 | 0.0847 | 0.0741 | -0.1136 | 0.0175 | 0.7959 | 1.0175 | 1.0155 | 0.002 | -0.1877 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 227 | 2 | 190 | 1 | 0.1768 | 0.0847 | 0.0733 | -0.1136 | 0.0174 | 0.7977 | 1.0174 | 1.018 | -0.0006 | -0.1868 | move_2_q_gt_move_4, selected_move_2 |
| current | 228 | 2 | 190 | 2 | 0.1768 | 0.0847 | 0.0733 | -0.0194 | 0.0175 | 0.533 | 1.0175 | 1.1462 | -0.1287 | -0.0927 | move_2_q_gt_move_4, selected_move_2 |
| current | 229 | 2 | 190 | 3 | 0.1768 | 0.0847 | 0.0733 | 0.0716 | 0.0175 | 0.4006 | 1.0175 | 1.3937 | -0.3762 | -0.0016 | move_2_q_gt_move_4, selected_move_2 |
| current | 230 | 2 | 190 | 4 | 0.1768 | 0.0847 | 0.0733 | 0.073 | 0.0176 | 0.3212 | 1.0176 | 1.3202 | -0.3026 | -0.0002 | move_2_q_gt_move_4, selected_move_2 |
| current | 231 | 2 | 190 | 5 | 0.1768 | 0.0847 | 0.0733 | 0.0992 | 0.0176 | 0.2682 | 0.9199 | 1.2682 | -0.3483 | 0.0259 | selected_move_2 |
| current | 232 | 2 | 190 | 6 | 0.1768 | 0.0847 | 0.0733 | 0.0905 | 0.0176 | 0.2304 | 0.9504 | 1.2304 | -0.28 | 0.0173 | selected_move_2 |
| current | 233 | 2 | 190 | 7 | 0.1768 | 0.0847 | 0.0733 | 0.0767 | 0.0177 | 0.202 | 1.0035 | 1.202 | -0.1985 | 0.0034 | selected_move_2 |
| current | 234 | 2 | 190 | 8 | 0.1768 | 0.0847 | 0.0733 | 0.0796 | 0.0177 | 0.18 | 0.9918 | 1.18 | -0.1882 | 0.0064 | selected_move_2 |
| current | 235 | 2 | 190 | 9 | 0.1768 | 0.0847 | 0.0733 | 0.081 | 0.0177 | 0.1623 | 0.9866 | 1.1623 | -0.1757 | 0.0077 | selected_move_2 |
| current | 236 | 2 | 190 | 10 | 0.1768 | 0.0847 | 0.0733 | 0.0883 | 0.0178 | 0.1479 | 0.9586 | 1.1479 | -0.1893 | 0.0151 | selected_move_2 |
| current | 237 | 2 | 190 | 11 | 0.1768 | 0.0847 | 0.0733 | 0.0918 | 0.0178 | 0.1358 | 0.9461 | 1.1358 | -0.1897 | 0.0185 | selected_move_2 |
| current | 238 | 2 | 190 | 12 | 0.1768 | 0.0847 | 0.0733 | 0.0707 | 0.0179 | 0.1257 | 1.0179 | 1.1149 | -0.0971 | -0.0026 | move_2_q_gt_move_4, selected_move_2 |
| current | 239 | 2 | 190 | 13 | 0.1768 | 0.0847 | 0.0733 | 0.0569 | 0.0179 | 0.1169 | 1.0179 | 1.0486 | -0.0307 | -0.0164 | move_2_q_gt_move_4, selected_move_2 |
| current | 240 | 2 | 190 | 14 | 0.1768 | 0.0847 | 0.0733 | 0.0595 | 0.0179 | 0.1094 | 1.0179 | 1.0519 | -0.034 | -0.0138 | move_2_q_gt_move_4, selected_move_2 |
| current | 241 | 2 | 190 | 15 | 0.1768 | 0.0847 | 0.0733 | 0.0551 | 0.018 | 0.1027 | 1.018 | 1.0272 | -0.0092 | -0.0181 | move_2_q_gt_move_4, selected_move_2 |
| current | 242 | 2 | 190 | 16 | 0.1768 | 0.0847 | 0.0733 | 0.0548 | 0.018 | 0.0969 | 1.018 | 1.02 | -0.002 | -0.0184 | move_2_q_gt_move_4, selected_move_2 |
| current | 243 | 2 | 190 | 17 | 0.1768 | 0.0847 | 0.0733 | 0.0447 | 0.018 | 0.0917 | 1.018 | 0.9725 | 0.0456 | -0.0286 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 244 | 2 | 191 | 17 | 0.1768 | 0.0847 | 0.0746 | 0.0447 | 0.018 | 0.0919 | 1.018 | 0.9677 | 0.0503 | -0.0299 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 245 | 2 | 192 | 17 | 0.1768 | 0.0847 | 0.0749 | 0.0447 | 0.0179 | 0.0921 | 1.0179 | 0.9667 | 0.0513 | -0.0303 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 246 | 2 | 193 | 17 | 0.1768 | 0.0847 | 0.0753 | 0.0447 | 0.0179 | 0.0923 | 1.0179 | 0.9657 | 0.0522 | -0.0306 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 247 | 2 | 194 | 17 | 0.1768 | 0.0847 | 0.0751 | 0.0447 | 0.0178 | 0.0925 | 1.0178 | 0.9664 | 0.0514 | -0.0304 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 248 | 2 | 195 | 17 | 0.1768 | 0.0847 | 0.0751 | 0.0447 | 0.0178 | 0.0926 | 1.0178 | 0.9666 | 0.0512 | -0.0304 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 249 | 2 | 196 | 17 | 0.1768 | 0.0847 | 0.0747 | 0.0447 | 0.0177 | 0.0928 | 1.0177 | 0.9683 | 0.0494 | -0.03 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 250 | 2 | 197 | 17 | 0.1768 | 0.0847 | 0.0757 | 0.0447 | 0.0177 | 0.093 | 1.0177 | 0.9649 | 0.0528 | -0.031 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 251 | 2 | 198 | 17 | 0.1768 | 0.0847 | 0.0744 | 0.0447 | 0.0176 | 0.0932 | 1.0176 | 0.9696 | 0.048 | -0.0298 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 252 | 2 | 199 | 17 | 0.1768 | 0.0847 | 0.074 | 0.0447 | 0.0175 | 0.0934 | 1.0175 | 0.9714 | 0.0462 | -0.0293 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 253 | 2 | 200 | 17 | 0.1768 | 0.0847 | 0.0732 | 0.0447 | 0.0175 | 0.0936 | 1.0175 | 0.9744 | 0.0431 | -0.0285 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 254 | 2 | 201 | 17 | 0.1768 | 0.0847 | 0.0731 | 0.0447 | 0.0174 | 0.0938 | 1.0174 | 0.9751 | 0.0423 | -0.0284 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 255 | 2 | 202 | 17 | 0.1768 | 0.0847 | 0.0737 | 0.0447 | 0.0174 | 0.0939 | 1.0174 | 0.9731 | 0.0443 | -0.029 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 256 | 2 | 203 | 17 | 0.1768 | 0.0847 | 0.0733 | 0.0447 | 0.0173 | 0.0941 | 1.0173 | 0.9748 | 0.0425 | -0.0286 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 257 | 2 | 204 | 17 | 0.1768 | 0.0847 | 0.0725 | 0.0447 | 0.0173 | 0.0943 | 1.0173 | 0.978 | 0.0393 | -0.0278 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 258 | 2 | 205 | 17 | 0.1768 | 0.0847 | 0.0724 | 0.0447 | 0.0172 | 0.0945 | 1.0172 | 0.9784 | 0.0388 | -0.0277 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 259 | 2 | 206 | 17 | 0.1768 | 0.0847 | 0.0715 | 0.0447 | 0.0172 | 0.0947 | 1.0172 | 0.9818 | 0.0354 | -0.0268 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 260 | 2 | 207 | 17 | 0.1768 | 0.0847 | 0.0733 | 0.0447 | 0.0171 | 0.0949 | 1.0171 | 0.9753 | 0.0418 | -0.0287 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 261 | 2 | 208 | 17 | 0.1768 | 0.0847 | 0.074 | 0.0447 | 0.0171 | 0.095 | 1.0171 | 0.9729 | 0.0442 | -0.0294 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 262 | 2 | 209 | 17 | 0.1768 | 0.0847 | 0.0748 | 0.0447 | 0.017 | 0.0952 | 1.017 | 0.9705 | 0.0466 | -0.0301 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 263 | 2 | 210 | 17 | 0.1768 | 0.0847 | 0.0746 | 0.0447 | 0.017 | 0.0954 | 1.017 | 0.9713 | 0.0457 | -0.0299 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 264 | 2 | 211 | 17 | 0.1768 | 0.0847 | 0.0762 | 0.0447 | 0.0169 | 0.0956 | 1.0169 | 0.9655 | 0.0514 | -0.0316 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 265 | 2 | 212 | 17 | 0.1768 | 0.0847 | 0.076 | 0.0447 | 0.0169 | 0.0958 | 1.0169 | 0.9667 | 0.0502 | -0.0313 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 266 | 2 | 213 | 17 | 0.1768 | 0.0847 | 0.075 | 0.0447 | 0.0168 | 0.0959 | 1.0168 | 0.9703 | 0.0465 | -0.0303 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 267 | 2 | 214 | 17 | 0.1768 | 0.0847 | 0.0755 | 0.0447 | 0.0168 | 0.0961 | 1.0168 | 0.9687 | 0.0481 | -0.0308 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 268 | 2 | 215 | 17 | 0.1768 | 0.0847 | 0.0756 | 0.0447 | 0.0168 | 0.0963 | 1.0168 | 0.9684 | 0.0483 | -0.0309 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 269 | 2 | 216 | 17 | 0.1768 | 0.0847 | 0.0747 | 0.0447 | 0.0167 | 0.0965 | 1.0167 | 0.9719 | 0.0448 | -0.03 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 270 | 2 | 217 | 17 | 0.1768 | 0.0847 | 0.0734 | 0.0447 | 0.0167 | 0.0967 | 1.0167 | 0.9768 | 0.0399 | -0.0287 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 271 | 2 | 218 | 17 | 0.1768 | 0.0847 | 0.0731 | 0.0447 | 0.0166 | 0.0968 | 1.0166 | 0.9782 | 0.0384 | -0.0284 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 272 | 2 | 219 | 17 | 0.1768 | 0.0847 | 0.0721 | 0.0447 | 0.0166 | 0.097 | 1.0166 | 0.982 | 0.0346 | -0.0274 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 273 | 2 | 220 | 17 | 0.1768 | 0.0847 | 0.0712 | 0.0447 | 0.0165 | 0.0972 | 1.0165 | 0.9855 | 0.031 | -0.0265 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 274 | 2 | 221 | 17 | 0.1768 | 0.0847 | 0.0701 | 0.0447 | 0.0165 | 0.0974 | 1.0165 | 0.9898 | 0.0267 | -0.0254 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 275 | 2 | 222 | 17 | 0.1768 | 0.0847 | 0.0685 | 0.0447 | 0.0164 | 0.0976 | 1.0164 | 0.996 | 0.0204 | -0.0238 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 276 | 2 | 223 | 17 | 0.1768 | 0.0847 | 0.0671 | 0.0447 | 0.0164 | 0.0977 | 1.0164 | 1.0015 | 0.0149 | -0.0225 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 277 | 2 | 224 | 17 | 0.1768 | 0.0847 | 0.0675 | 0.0447 | 0.0163 | 0.0979 | 1.0163 | 1.0003 | 0.016 | -0.0228 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 278 | 2 | 225 | 17 | 0.1768 | 0.0847 | 0.0677 | 0.0447 | 0.0163 | 0.0981 | 1.0163 | 0.9997 | 0.0166 | -0.023 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 279 | 2 | 226 | 17 | 0.1768 | 0.0847 | 0.0679 | 0.0447 | 0.0163 | 0.0983 | 1.0163 | 0.9991 | 0.0171 | -0.0232 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 280 | 2 | 227 | 17 | 0.1768 | 0.0847 | 0.0686 | 0.0447 | 0.0162 | 0.0984 | 1.0162 | 0.9965 | 0.0197 | -0.024 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 281 | 2 | 228 | 17 | 0.1768 | 0.0847 | 0.0695 | 0.0447 | 0.0162 | 0.0986 | 1.0162 | 0.9935 | 0.0227 | -0.0248 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 282 | 2 | 229 | 17 | 0.1768 | 0.0847 | 0.0683 | 0.0447 | 0.0161 | 0.0988 | 1.0161 | 0.9983 | 0.0179 | -0.0236 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 283 | 2 | 230 | 17 | 0.1768 | 0.0847 | 0.0676 | 0.0447 | 0.0161 | 0.099 | 1.0161 | 1.0012 | 0.0149 | -0.0229 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 284 | 2 | 231 | 17 | 0.1768 | 0.0847 | 0.0661 | 0.0447 | 0.0161 | 0.0991 | 1.0161 | 1.007 | 0.009 | -0.0214 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 285 | 2 | 232 | 17 | 0.1768 | 0.0847 | 0.0664 | 0.0447 | 0.016 | 0.0993 | 1.016 | 1.0059 | 0.0102 | -0.0218 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 286 | 2 | 233 | 17 | 0.1768 | 0.0847 | 0.0665 | 0.0447 | 0.016 | 0.0995 | 1.016 | 1.0059 | 0.01 | -0.0218 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 287 | 2 | 233 | 17 | 0.1768 | 0.0847 | 0.0665 | 0.0447 | 0.016 | 0.0997 | 1.016 | 1.0061 | 0.0099 | -0.0218 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 288 | 2 | 234 | 17 | 0.1768 | 0.0847 | 0.0667 | 0.0447 | 0.016 | 0.0998 | 1.016 | 1.0053 | 0.0107 | -0.022 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 289 | 2 | 235 | 17 | 0.1768 | 0.0847 | 0.0679 | 0.0447 | 0.0159 | 0.1 | 1.0159 | 1.0008 | 0.0151 | -0.0232 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 290 | 2 | 236 | 17 | 0.1768 | 0.0847 | 0.0692 | 0.0447 | 0.0159 | 0.1002 | 1.0159 | 0.9961 | 0.0198 | -0.0245 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 291 | 2 | 237 | 17 | 0.1768 | 0.0847 | 0.0692 | 0.0447 | 0.0158 | 0.1004 | 1.0158 | 0.9964 | 0.0195 | -0.0245 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 292 | 2 | 238 | 17 | 0.1768 | 0.0847 | 0.0694 | 0.0447 | 0.0158 | 0.1005 | 1.0158 | 0.9956 | 0.0202 | -0.0247 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 293 | 2 | 239 | 17 | 0.1768 | 0.0847 | 0.0687 | 0.0447 | 0.0158 | 0.1007 | 1.0158 | 0.9983 | 0.0174 | -0.0241 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 294 | 2 | 240 | 17 | 0.1768 | 0.0847 | 0.0693 | 0.0447 | 0.0157 | 0.1009 | 1.0157 | 0.9963 | 0.0194 | -0.0246 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 295 | 2 | 241 | 17 | 0.1768 | 0.0847 | 0.068 | 0.0447 | 0.0157 | 0.101 | 1.0157 | 1.0013 | 0.0143 | -0.0234 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 296 | 2 | 242 | 17 | 0.1768 | 0.0847 | 0.0666 | 0.0447 | 0.0156 | 0.1012 | 1.0156 | 1.0072 | 0.0085 | -0.0219 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 297 | 2 | 243 | 17 | 0.1768 | 0.0847 | 0.067 | 0.0447 | 0.0156 | 0.1014 | 1.0156 | 1.0059 | 0.0097 | -0.0223 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 298 | 2 | 244 | 17 | 0.1768 | 0.0847 | 0.0659 | 0.0447 | 0.0156 | 0.1016 | 1.0156 | 1.0101 | 0.0055 | -0.0213 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 299 | 2 | 245 | 17 | 0.1768 | 0.0847 | 0.0664 | 0.0447 | 0.0155 | 0.1017 | 1.0155 | 1.0083 | 0.0073 | -0.0218 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 300 | 2 | 246 | 17 | 0.1768 | 0.0847 | 0.0667 | 0.0447 | 0.0155 | 0.1019 | 1.0155 | 1.0075 | 0.008 | -0.022 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 301 | 2 | 247 | 17 | 0.1768 | 0.0847 | 0.0677 | 0.0447 | 0.0155 | 0.1021 | 1.0155 | 1.0037 | 0.0118 | -0.023 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 302 | 2 | 248 | 17 | 0.1768 | 0.0847 | 0.0675 | 0.0447 | 0.0154 | 0.1022 | 1.0154 | 1.0045 | 0.011 | -0.0229 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 303 | 2 | 249 | 17 | 0.1768 | 0.0847 | 0.0687 | 0.0447 | 0.0154 | 0.1024 | 1.0154 | 1.0003 | 0.0151 | -0.024 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 304 | 2 | 250 | 17 | 0.1768 | 0.0847 | 0.0682 | 0.0447 | 0.0154 | 0.1026 | 1.0154 | 1.0022 | 0.0132 | -0.0235 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 305 | 2 | 251 | 17 | 0.1768 | 0.0847 | 0.0677 | 0.0447 | 0.0153 | 0.1027 | 1.0153 | 1.0044 | 0.0109 | -0.023 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 306 | 2 | 252 | 17 | 0.1768 | 0.0847 | 0.067 | 0.0447 | 0.0153 | 0.1029 | 1.0153 | 1.0071 | 0.0082 | -0.0224 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 307 | 2 | 253 | 17 | 0.1768 | 0.0847 | 0.0679 | 0.0447 | 0.0152 | 0.1031 | 1.0152 | 1.0039 | 0.0114 | -0.0232 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 308 | 2 | 254 | 17 | 0.1768 | 0.0847 | 0.068 | 0.0447 | 0.0152 | 0.1032 | 1.0152 | 1.0038 | 0.0114 | -0.0233 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 309 | 2 | 255 | 17 | 0.1768 | 0.0847 | 0.0676 | 0.0447 | 0.0152 | 0.1034 | 1.0152 | 1.0054 | 0.0098 | -0.0229 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 310 | 2 | 256 | 17 | 0.1768 | 0.0847 | 0.0682 | 0.0447 | 0.0151 | 0.1036 | 1.0151 | 1.0032 | 0.012 | -0.0236 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 311 | 2 | 257 | 17 | 0.1768 | 0.0847 | 0.0688 | 0.0447 | 0.0151 | 0.1037 | 1.0151 | 1.0012 | 0.0139 | -0.0241 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 312 | 2 | 258 | 17 | 0.1768 | 0.0847 | 0.0693 | 0.0447 | 0.0151 | 0.1039 | 1.0151 | 0.9995 | 0.0155 | -0.0246 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 313 | 2 | 259 | 17 | 0.1768 | 0.0847 | 0.0695 | 0.0447 | 0.015 | 0.1041 | 1.015 | 0.9987 | 0.0164 | -0.0249 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 314 | 2 | 260 | 17 | 0.1768 | 0.0847 | 0.0706 | 0.0447 | 0.015 | 0.1042 | 1.015 | 0.995 | 0.0201 | -0.0259 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 315 | 2 | 261 | 17 | 0.1768 | 0.0847 | 0.0717 | 0.0447 | 0.015 | 0.1044 | 1.015 | 0.9911 | 0.0239 | -0.027 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 316 | 2 | 262 | 17 | 0.1768 | 0.0847 | 0.073 | 0.0447 | 0.0149 | 0.1046 | 1.0149 | 0.9863 | 0.0286 | -0.0283 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 317 | 2 | 263 | 17 | 0.1768 | 0.0847 | 0.0722 | 0.0447 | 0.0149 | 0.1047 | 1.0149 | 0.9893 | 0.0256 | -0.0275 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 318 | 2 | 264 | 17 | 0.1768 | 0.0847 | 0.0722 | 0.0447 | 0.0149 | 0.1049 | 1.0149 | 0.9894 | 0.0254 | -0.0275 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 319 | 2 | 265 | 17 | 0.1768 | 0.0847 | 0.0717 | 0.0447 | 0.0148 | 0.1051 | 1.0148 | 0.9916 | 0.0233 | -0.027 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 320 | 2 | 266 | 17 | 0.1768 | 0.0847 | 0.071 | 0.0447 | 0.0148 | 0.1052 | 1.0148 | 0.9943 | 0.0205 | -0.0263 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 321 | 2 | 267 | 17 | 0.1768 | 0.0847 | 0.0706 | 0.0447 | 0.0148 | 0.1054 | 1.0148 | 0.996 | 0.0187 | -0.0259 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 322 | 2 | 268 | 17 | 0.1768 | 0.0847 | 0.0699 | 0.0447 | 0.0147 | 0.1056 | 1.0147 | 0.9987 | 0.016 | -0.0252 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 323 | 2 | 269 | 17 | 0.1768 | 0.0847 | 0.0692 | 0.0447 | 0.0147 | 0.1057 | 1.0147 | 1.0016 | 0.0131 | -0.0245 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 324 | 2 | 270 | 17 | 0.1768 | 0.0847 | 0.0681 | 0.0447 | 0.0147 | 0.1059 | 1.0147 | 1.0061 | 0.0086 | -0.0234 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 325 | 2 | 271 | 17 | 0.1768 | 0.0847 | 0.0678 | 0.0447 | 0.0146 | 0.1061 | 1.0146 | 1.0075 | 0.0072 | -0.0231 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 326 | 2 | 272 | 17 | 0.1768 | 0.0847 | 0.0683 | 0.0447 | 0.0146 | 0.1062 | 1.0146 | 1.0055 | 0.0091 | -0.0236 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 327 | 2 | 273 | 17 | 0.1768 | 0.0847 | 0.0695 | 0.0447 | 0.0146 | 0.1064 | 1.0146 | 1.0011 | 0.0135 | -0.0248 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 328 | 2 | 274 | 17 | 0.1768 | 0.0847 | 0.0692 | 0.0447 | 0.0146 | 0.1065 | 1.0146 | 1.0024 | 0.0121 | -0.0245 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 329 | 2 | 275 | 17 | 0.1768 | 0.0847 | 0.0689 | 0.0447 | 0.0145 | 0.1067 | 1.0145 | 1.0037 | 0.0109 | -0.0242 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 330 | 2 | 276 | 17 | 0.1768 | 0.0847 | 0.0686 | 0.0447 | 0.0145 | 0.1069 | 1.0145 | 1.0049 | 0.0096 | -0.024 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 331 | 2 | 277 | 17 | 0.1768 | 0.0847 | 0.0691 | 0.0447 | 0.0145 | 0.107 | 1.0145 | 1.0034 | 0.0111 | -0.0244 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 332 | 2 | 278 | 17 | 0.1768 | 0.0847 | 0.0698 | 0.0447 | 0.0144 | 0.1072 | 1.0144 | 1.0008 | 0.0137 | -0.0251 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 333 | 2 | 279 | 17 | 0.1768 | 0.0847 | 0.0693 | 0.0447 | 0.0144 | 0.1074 | 1.0144 | 1.0027 | 0.0117 | -0.0247 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 334 | 2 | 280 | 17 | 0.1768 | 0.0847 | 0.0693 | 0.0447 | 0.0144 | 0.1075 | 1.0144 | 1.003 | 0.0113 | -0.0246 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 335 | 2 | 281 | 17 | 0.1768 | 0.0847 | 0.0684 | 0.0447 | 0.0143 | 0.1077 | 1.0143 | 1.0066 | 0.0077 | -0.0237 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 336 | 2 | 282 | 17 | 0.1768 | 0.0847 | 0.0677 | 0.0447 | 0.0143 | 0.1078 | 1.0143 | 1.0095 | 0.0049 | -0.023 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 337 | 2 | 282 | 17 | 0.1768 | 0.0847 | 0.0677 | 0.0447 | 0.0143 | 0.108 | 1.0143 | 1.0096 | 0.0047 | -0.023 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 338 | 2 | 283 | 17 | 0.1768 | 0.0847 | 0.0668 | 0.0447 | 0.0143 | 0.1082 | 1.0143 | 1.0132 | 0.0011 | -0.0221 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 339 | 2 | 284 | 17 | 0.1768 | 0.0847 | 0.0666 | 0.0447 | 0.0143 | 0.1083 | 1.0143 | 1.0143 | -0.0 | -0.0219 | move_2_q_gt_move_4, selected_move_2 |
| current | 340 | 2 | 284 | 18 | 0.1768 | 0.0847 | 0.0666 | 0.0391 | 0.0143 | 0.1028 | 1.0143 | 0.9848 | 0.0295 | -0.0275 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 341 | 2 | 285 | 18 | 0.1768 | 0.0847 | 0.0668 | 0.0391 | 0.0143 | 0.1029 | 1.0143 | 0.9841 | 0.0302 | -0.0277 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 342 | 2 | 286 | 18 | 0.1768 | 0.0847 | 0.0662 | 0.0391 | 0.0142 | 0.1031 | 1.0142 | 0.9866 | 0.0277 | -0.0271 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 343 | 2 | 287 | 18 | 0.1768 | 0.0847 | 0.0659 | 0.0391 | 0.0142 | 0.1032 | 1.0142 | 0.9878 | 0.0264 | -0.0268 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 344 | 2 | 288 | 18 | 0.1768 | 0.0847 | 0.0653 | 0.0391 | 0.0142 | 0.1034 | 1.0142 | 0.9902 | 0.024 | -0.0262 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 345 | 2 | 289 | 18 | 0.1768 | 0.0847 | 0.0642 | 0.0391 | 0.0142 | 0.1035 | 1.0142 | 0.9945 | 0.0196 | -0.0251 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 346 | 2 | 290 | 18 | 0.1768 | 0.0847 | 0.0644 | 0.0391 | 0.0141 | 0.1037 | 1.0141 | 0.9941 | 0.02 | -0.0253 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 347 | 2 | 291 | 18 | 0.1768 | 0.0847 | 0.0651 | 0.0391 | 0.0141 | 0.1038 | 1.0141 | 0.9915 | 0.0226 | -0.026 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 348 | 2 | 292 | 18 | 0.1768 | 0.0847 | 0.0661 | 0.0391 | 0.0141 | 0.104 | 1.0141 | 0.9877 | 0.0264 | -0.027 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 349 | 2 | 293 | 18 | 0.1768 | 0.0847 | 0.0674 | 0.0391 | 0.014 | 0.1041 | 1.014 | 0.9829 | 0.0311 | -0.0283 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 350 | 2 | 294 | 18 | 0.1768 | 0.0847 | 0.0672 | 0.0391 | 0.014 | 0.1043 | 1.014 | 0.9838 | 0.0302 | -0.0281 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 351 | 2 | 295 | 18 | 0.1768 | 0.0847 | 0.0671 | 0.0391 | 0.014 | 0.1044 | 1.014 | 0.9844 | 0.0296 | -0.028 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 352 | 2 | 296 | 18 | 0.1768 | 0.0847 | 0.0685 | 0.0391 | 0.014 | 0.1046 | 1.014 | 0.9794 | 0.0345 | -0.0294 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 353 | 2 | 297 | 18 | 0.1768 | 0.0847 | 0.0687 | 0.0391 | 0.0139 | 0.1047 | 1.0139 | 0.9789 | 0.035 | -0.0296 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 354 | 2 | 298 | 18 | 0.1768 | 0.0847 | 0.0689 | 0.0391 | 0.0139 | 0.1049 | 1.0139 | 0.9781 | 0.0358 | -0.0298 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 355 | 2 | 299 | 18 | 0.1768 | 0.0847 | 0.0696 | 0.0391 | 0.0139 | 0.105 | 1.0139 | 0.9759 | 0.038 | -0.0305 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 356 | 2 | 300 | 18 | 0.1768 | 0.0847 | 0.0701 | 0.0391 | 0.0139 | 0.1052 | 1.0139 | 0.9741 | 0.0398 | -0.031 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 357 | 2 | 301 | 18 | 0.1768 | 0.0847 | 0.0708 | 0.0391 | 0.0138 | 0.1053 | 1.0138 | 0.9716 | 0.0423 | -0.0317 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 358 | 2 | 302 | 18 | 0.1768 | 0.0847 | 0.0706 | 0.0391 | 0.0138 | 0.1054 | 1.0138 | 0.9724 | 0.0414 | -0.0315 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 359 | 2 | 303 | 18 | 0.1768 | 0.0847 | 0.0701 | 0.0391 | 0.0138 | 0.1056 | 1.0138 | 0.9746 | 0.0392 | -0.031 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 360 | 2 | 304 | 18 | 0.1768 | 0.0847 | 0.0704 | 0.0391 | 0.0138 | 0.1057 | 1.0138 | 0.9735 | 0.0402 | -0.0313 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 361 | 2 | 305 | 18 | 0.1768 | 0.0847 | 0.0707 | 0.0391 | 0.0137 | 0.1059 | 1.0137 | 0.9727 | 0.041 | -0.0316 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 362 | 2 | 306 | 18 | 0.1768 | 0.0847 | 0.0693 | 0.0391 | 0.0137 | 0.106 | 1.0137 | 0.9779 | 0.0358 | -0.0302 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 363 | 2 | 307 | 18 | 0.1768 | 0.0847 | 0.0682 | 0.0391 | 0.0137 | 0.1062 | 1.0137 | 0.9819 | 0.0317 | -0.0291 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 364 | 2 | 308 | 18 | 0.1768 | 0.0847 | 0.068 | 0.0391 | 0.0136 | 0.1063 | 1.0136 | 0.983 | 0.0306 | -0.0289 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 365 | 2 | 309 | 18 | 0.1768 | 0.0847 | 0.0672 | 0.0391 | 0.0136 | 0.1065 | 1.0136 | 0.9861 | 0.0275 | -0.0281 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 366 | 2 | 310 | 18 | 0.1768 | 0.0847 | 0.0674 | 0.0391 | 0.0136 | 0.1066 | 1.0136 | 0.9856 | 0.028 | -0.0283 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 367 | 2 | 311 | 18 | 0.1768 | 0.0847 | 0.0675 | 0.0391 | 0.0136 | 0.1068 | 1.0136 | 0.9853 | 0.0283 | -0.0284 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 368 | 2 | 312 | 18 | 0.1768 | 0.0847 | 0.0677 | 0.0391 | 0.0135 | 0.1069 | 1.0135 | 0.9847 | 0.0288 | -0.0286 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 369 | 2 | 313 | 18 | 0.1768 | 0.0847 | 0.0684 | 0.0391 | 0.0135 | 0.1071 | 1.0135 | 0.9823 | 0.0312 | -0.0293 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 370 | 2 | 314 | 18 | 0.1768 | 0.0847 | 0.0686 | 0.0391 | 0.0135 | 0.1072 | 1.0135 | 0.9817 | 0.0318 | -0.0295 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 371 | 2 | 315 | 18 | 0.1768 | 0.0847 | 0.0692 | 0.0391 | 0.0135 | 0.1073 | 1.0135 | 0.9795 | 0.034 | -0.0301 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 372 | 2 | 316 | 18 | 0.1768 | 0.0847 | 0.0684 | 0.0391 | 0.0134 | 0.1075 | 1.0134 | 0.9827 | 0.0308 | -0.0293 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 373 | 2 | 317 | 18 | 0.1768 | 0.0847 | 0.0683 | 0.0391 | 0.0134 | 0.1076 | 1.0134 | 0.9832 | 0.0302 | -0.0292 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 374 | 2 | 318 | 18 | 0.1768 | 0.0847 | 0.0678 | 0.0391 | 0.0134 | 0.1078 | 1.0134 | 0.9853 | 0.0281 | -0.0287 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 375 | 2 | 319 | 18 | 0.1768 | 0.0847 | 0.0679 | 0.0391 | 0.0134 | 0.1079 | 1.0134 | 0.9849 | 0.0285 | -0.0288 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 376 | 2 | 320 | 18 | 0.1768 | 0.0847 | 0.0673 | 0.0391 | 0.0134 | 0.1081 | 1.0134 | 0.9875 | 0.0258 | -0.0282 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 377 | 2 | 321 | 18 | 0.1768 | 0.0847 | 0.0666 | 0.0391 | 0.0133 | 0.1082 | 1.0133 | 0.9903 | 0.023 | -0.0275 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 378 | 2 | 322 | 18 | 0.1768 | 0.0847 | 0.0661 | 0.0391 | 0.0133 | 0.1084 | 1.0133 | 0.9921 | 0.0212 | -0.027 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 379 | 2 | 323 | 18 | 0.1768 | 0.0847 | 0.0653 | 0.0391 | 0.0133 | 0.1085 | 1.0133 | 0.9953 | 0.018 | -0.0262 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 380 | 2 | 324 | 18 | 0.1768 | 0.0847 | 0.0643 | 0.0391 | 0.0133 | 0.1086 | 1.0133 | 0.9992 | 0.014 | -0.0252 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 381 | 2 | 325 | 18 | 0.1768 | 0.0847 | 0.0639 | 0.0391 | 0.0132 | 0.1088 | 1.0132 | 1.0012 | 0.012 | -0.0248 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 382 | 2 | 326 | 18 | 0.1768 | 0.0847 | 0.0633 | 0.0391 | 0.0132 | 0.1089 | 1.0132 | 1.0036 | 0.0096 | -0.0242 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 383 | 2 | 327 | 18 | 0.1768 | 0.0847 | 0.0626 | 0.0391 | 0.0132 | 0.1091 | 1.0132 | 1.0064 | 0.0067 | -0.0235 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 384 | 2 | 328 | 18 | 0.1768 | 0.0847 | 0.0623 | 0.0391 | 0.0132 | 0.1092 | 1.0132 | 1.0077 | 0.0055 | -0.0232 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 385 | 2 | 329 | 18 | 0.1768 | 0.0847 | 0.0625 | 0.0391 | 0.0131 | 0.1094 | 1.0131 | 1.0072 | 0.006 | -0.0234 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 386 | 2 | 330 | 18 | 0.1768 | 0.0847 | 0.0619 | 0.0391 | 0.0131 | 0.1095 | 1.0131 | 1.0096 | 0.0036 | -0.0228 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 387 | 2 | 330 | 18 | 0.1768 | 0.0847 | 0.0619 | 0.0391 | 0.0131 | 0.1096 | 1.0131 | 1.0097 | 0.0034 | -0.0228 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 388 | 2 | 331 | 18 | 0.1768 | 0.0847 | 0.0624 | 0.0391 | 0.0131 | 0.1098 | 1.0131 | 1.0079 | 0.0052 | -0.0233 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 389 | 2 | 332 | 18 | 0.1768 | 0.0847 | 0.0629 | 0.0391 | 0.0131 | 0.1099 | 1.0131 | 1.0061 | 0.007 | -0.0238 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 390 | 2 | 333 | 18 | 0.1768 | 0.0847 | 0.0637 | 0.0391 | 0.0131 | 0.1101 | 1.0131 | 1.003 | 0.0101 | -0.0246 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 391 | 2 | 334 | 18 | 0.1768 | 0.0847 | 0.0641 | 0.0391 | 0.013 | 0.1102 | 1.013 | 1.0019 | 0.0112 | -0.025 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 392 | 2 | 335 | 18 | 0.1768 | 0.0847 | 0.0648 | 0.0391 | 0.013 | 0.1103 | 1.013 | 0.999 | 0.014 | -0.0257 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 393 | 2 | 336 | 18 | 0.1768 | 0.0847 | 0.065 | 0.0391 | 0.013 | 0.1105 | 1.013 | 0.9986 | 0.0144 | -0.0259 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 394 | 2 | 337 | 18 | 0.1768 | 0.0847 | 0.0653 | 0.0391 | 0.013 | 0.1106 | 1.013 | 0.9977 | 0.0153 | -0.0262 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 395 | 2 | 338 | 18 | 0.1768 | 0.0847 | 0.065 | 0.0391 | 0.013 | 0.1108 | 1.013 | 0.9987 | 0.0142 | -0.0259 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 396 | 2 | 339 | 18 | 0.1768 | 0.0847 | 0.0648 | 0.0391 | 0.0129 | 0.1109 | 1.0129 | 0.9998 | 0.0132 | -0.0257 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 397 | 2 | 340 | 18 | 0.1768 | 0.0847 | 0.0643 | 0.0391 | 0.0129 | 0.111 | 1.0129 | 1.0019 | 0.011 | -0.0252 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 398 | 2 | 341 | 18 | 0.1768 | 0.0847 | 0.0641 | 0.0391 | 0.0129 | 0.1112 | 1.0129 | 1.0026 | 0.0103 | -0.025 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 399 | 2 | 342 | 18 | 0.1768 | 0.0847 | 0.0641 | 0.0391 | 0.0129 | 0.1113 | 1.0129 | 1.0027 | 0.0101 | -0.025 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 400 | 2 | 343 | 18 | 0.1768 | 0.0847 | 0.0647 | 0.0391 | 0.0129 | 0.1115 | 1.0129 | 1.0008 | 0.012 | -0.0256 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 401 | 2 | 344 | 18 | 0.1768 | 0.0847 | 0.0648 | 0.0391 | 0.0128 | 0.1116 | 1.0128 | 1.0003 | 0.0125 | -0.0257 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 402 | 2 | 345 | 18 | 0.1768 | 0.0847 | 0.0655 | 0.0391 | 0.0128 | 0.1117 | 1.0128 | 0.9978 | 0.0151 | -0.0264 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 403 | 2 | 346 | 18 | 0.1768 | 0.0847 | 0.0656 | 0.0391 | 0.0128 | 0.1119 | 1.0128 | 0.9978 | 0.015 | -0.0265 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 404 | 2 | 347 | 18 | 0.1768 | 0.0847 | 0.0653 | 0.0391 | 0.0128 | 0.112 | 1.0128 | 0.9987 | 0.014 | -0.0263 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 405 | 2 | 348 | 18 | 0.1768 | 0.0847 | 0.0662 | 0.0391 | 0.0127 | 0.1122 | 1.0127 | 0.9954 | 0.0173 | -0.0272 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 406 | 2 | 349 | 18 | 0.1768 | 0.0847 | 0.0672 | 0.0391 | 0.0127 | 0.1123 | 1.0127 | 0.992 | 0.0208 | -0.0281 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 407 | 2 | 350 | 18 | 0.1768 | 0.0847 | 0.0669 | 0.0391 | 0.0127 | 0.1124 | 1.0127 | 0.9931 | 0.0196 | -0.0278 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 408 | 2 | 351 | 18 | 0.1768 | 0.0847 | 0.0666 | 0.0391 | 0.0127 | 0.1126 | 1.0127 | 0.9946 | 0.0181 | -0.0275 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 409 | 2 | 352 | 18 | 0.1768 | 0.0847 | 0.0666 | 0.0391 | 0.0127 | 0.1127 | 1.0127 | 0.9946 | 0.0181 | -0.0275 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 410 | 2 | 353 | 18 | 0.1768 | 0.0847 | 0.0671 | 0.0391 | 0.0126 | 0.1128 | 1.0126 | 0.9927 | 0.0199 | -0.028 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 411 | 2 | 354 | 18 | 0.1768 | 0.0847 | 0.0672 | 0.0391 | 0.0126 | 0.113 | 1.0126 | 0.9926 | 0.02 | -0.0281 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 412 | 2 | 355 | 18 | 0.1768 | 0.0847 | 0.0675 | 0.0391 | 0.0126 | 0.1131 | 1.0126 | 0.9916 | 0.021 | -0.0284 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 413 | 2 | 356 | 18 | 0.1768 | 0.0847 | 0.0679 | 0.0391 | 0.0126 | 0.1133 | 1.0126 | 0.9902 | 0.0224 | -0.0288 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 414 | 2 | 357 | 18 | 0.1768 | 0.0847 | 0.0687 | 0.0391 | 0.0126 | 0.1134 | 1.0126 | 0.9876 | 0.025 | -0.0296 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 415 | 2 | 358 | 18 | 0.1768 | 0.0847 | 0.0688 | 0.0391 | 0.0125 | 0.1135 | 1.0125 | 0.9871 | 0.0255 | -0.0297 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 416 | 2 | 359 | 18 | 0.1768 | 0.0847 | 0.0696 | 0.0391 | 0.0125 | 0.1137 | 1.0125 | 0.9843 | 0.0282 | -0.0305 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 417 | 2 | 360 | 18 | 0.1768 | 0.0847 | 0.0691 | 0.0391 | 0.0125 | 0.1138 | 1.0125 | 0.9863 | 0.0262 | -0.03 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 418 | 2 | 361 | 18 | 0.1768 | 0.0847 | 0.0685 | 0.0391 | 0.0125 | 0.1139 | 1.0125 | 0.9888 | 0.0237 | -0.0294 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 419 | 2 | 362 | 18 | 0.1768 | 0.0847 | 0.0679 | 0.0391 | 0.0125 | 0.1141 | 1.0125 | 0.9913 | 0.0212 | -0.0288 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 420 | 2 | 363 | 18 | 0.1768 | 0.0847 | 0.0677 | 0.0391 | 0.0124 | 0.1142 | 1.0124 | 0.9922 | 0.0203 | -0.0286 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 421 | 2 | 364 | 18 | 0.1768 | 0.0847 | 0.0677 | 0.0391 | 0.0124 | 0.1144 | 1.0124 | 0.9921 | 0.0203 | -0.0286 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 422 | 2 | 365 | 18 | 0.1768 | 0.0847 | 0.0681 | 0.0391 | 0.0124 | 0.1145 | 1.0124 | 0.991 | 0.0215 | -0.029 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 423 | 2 | 366 | 18 | 0.1768 | 0.0847 | 0.068 | 0.0391 | 0.0124 | 0.1146 | 1.0124 | 0.9915 | 0.0209 | -0.0289 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 424 | 2 | 367 | 18 | 0.1768 | 0.0847 | 0.0675 | 0.0391 | 0.0124 | 0.1148 | 1.0124 | 0.9933 | 0.0191 | -0.0284 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 425 | 2 | 368 | 18 | 0.1768 | 0.0847 | 0.0674 | 0.0391 | 0.0123 | 0.1149 | 1.0123 | 0.9938 | 0.0185 | -0.0283 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 426 | 2 | 369 | 18 | 0.1768 | 0.0847 | 0.068 | 0.0391 | 0.0123 | 0.115 | 1.0123 | 0.9919 | 0.0205 | -0.0289 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 427 | 2 | 370 | 18 | 0.1768 | 0.0847 | 0.0679 | 0.0391 | 0.0123 | 0.1152 | 1.0123 | 0.9921 | 0.0202 | -0.0288 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 428 | 2 | 371 | 18 | 0.1768 | 0.0847 | 0.0682 | 0.0391 | 0.0123 | 0.1153 | 1.0123 | 0.9913 | 0.021 | -0.0291 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 429 | 2 | 372 | 18 | 0.1768 | 0.0847 | 0.0678 | 0.0391 | 0.0123 | 0.1154 | 1.0123 | 0.9928 | 0.0195 | -0.0287 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 430 | 2 | 373 | 18 | 0.1768 | 0.0847 | 0.0684 | 0.0391 | 0.0123 | 0.1156 | 1.0123 | 0.9908 | 0.0215 | -0.0293 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 431 | 2 | 374 | 18 | 0.1768 | 0.0847 | 0.0679 | 0.0391 | 0.0122 | 0.1157 | 1.0122 | 0.9927 | 0.0195 | -0.0288 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 432 | 2 | 375 | 18 | 0.1768 | 0.0847 | 0.0676 | 0.0391 | 0.0122 | 0.1158 | 1.0122 | 0.9941 | 0.0181 | -0.0285 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 433 | 2 | 376 | 18 | 0.1768 | 0.0847 | 0.0673 | 0.0391 | 0.0122 | 0.116 | 1.0122 | 0.9952 | 0.017 | -0.0282 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 434 | 2 | 377 | 18 | 0.1768 | 0.0847 | 0.0669 | 0.0391 | 0.0122 | 0.1161 | 1.0122 | 0.9969 | 0.0153 | -0.0278 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 435 | 2 | 378 | 18 | 0.1768 | 0.0847 | 0.0666 | 0.0391 | 0.0122 | 0.1162 | 1.0122 | 0.9982 | 0.0139 | -0.0275 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 436 | 2 | 379 | 18 | 0.1768 | 0.0847 | 0.0665 | 0.0391 | 0.0121 | 0.1164 | 1.0121 | 0.9986 | 0.0135 | -0.0274 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 437 | 2 | 380 | 18 | 0.1768 | 0.0847 | 0.0665 | 0.0391 | 0.0121 | 0.1165 | 1.0121 | 0.9988 | 0.0134 | -0.0274 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 438 | 2 | 381 | 18 | 0.1768 | 0.0847 | 0.0661 | 0.0391 | 0.0121 | 0.1166 | 1.0121 | 1.0005 | 0.0116 | -0.027 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 439 | 2 | 382 | 18 | 0.1768 | 0.0847 | 0.0659 | 0.0391 | 0.0121 | 0.1168 | 1.0121 | 1.0015 | 0.0106 | -0.0268 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 440 | 2 | 383 | 18 | 0.1768 | 0.0847 | 0.0656 | 0.0391 | 0.0121 | 0.1169 | 1.0121 | 1.0026 | 0.0095 | -0.0265 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 441 | 2 | 384 | 18 | 0.1768 | 0.0847 | 0.0656 | 0.0391 | 0.0121 | 0.117 | 1.0121 | 1.0026 | 0.0095 | -0.0265 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 442 | 2 | 385 | 18 | 0.1768 | 0.0847 | 0.0656 | 0.0391 | 0.012 | 0.1172 | 1.012 | 1.0028 | 0.0093 | -0.0265 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 443 | 2 | 386 | 18 | 0.1768 | 0.0847 | 0.0657 | 0.0391 | 0.012 | 0.1173 | 1.012 | 1.0025 | 0.0095 | -0.0266 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 444 | 2 | 387 | 18 | 0.1768 | 0.0847 | 0.0662 | 0.0391 | 0.012 | 0.1174 | 1.012 | 1.0009 | 0.0111 | -0.0271 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 445 | 2 | 388 | 18 | 0.1768 | 0.0847 | 0.0667 | 0.0391 | 0.012 | 0.1176 | 1.012 | 0.9991 | 0.0128 | -0.0276 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 446 | 2 | 389 | 18 | 0.1768 | 0.0847 | 0.067 | 0.0391 | 0.012 | 0.1177 | 1.012 | 0.9982 | 0.0137 | -0.0279 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 447 | 2 | 390 | 18 | 0.1768 | 0.0847 | 0.0673 | 0.0391 | 0.012 | 0.1178 | 1.012 | 0.997 | 0.015 | -0.0282 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 448 | 2 | 391 | 18 | 0.1768 | 0.0847 | 0.0678 | 0.0391 | 0.0119 | 0.118 | 1.0119 | 0.9955 | 0.0165 | -0.0287 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 449 | 2 | 392 | 18 | 0.1768 | 0.0847 | 0.0688 | 0.0391 | 0.0119 | 0.1181 | 1.0119 | 0.9916 | 0.0203 | -0.0297 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 450 | 2 | 393 | 18 | 0.1768 | 0.0847 | 0.0689 | 0.0391 | 0.0119 | 0.1182 | 1.0119 | 0.9914 | 0.0205 | -0.0298 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 451 | 2 | 394 | 18 | 0.1768 | 0.0847 | 0.0687 | 0.0391 | 0.0119 | 0.1184 | 1.0119 | 0.9926 | 0.0193 | -0.0296 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 452 | 2 | 395 | 18 | 0.1768 | 0.0847 | 0.0685 | 0.0391 | 0.0119 | 0.1185 | 1.0119 | 0.9932 | 0.0187 | -0.0294 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 453 | 2 | 396 | 18 | 0.1768 | 0.0847 | 0.0683 | 0.0391 | 0.0118 | 0.1186 | 1.0118 | 0.9941 | 0.0178 | -0.0292 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 454 | 2 | 397 | 18 | 0.1768 | 0.0847 | 0.0687 | 0.0391 | 0.0118 | 0.1187 | 1.0118 | 0.9928 | 0.019 | -0.0296 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 455 | 2 | 398 | 18 | 0.1768 | 0.0847 | 0.0683 | 0.0391 | 0.0118 | 0.1189 | 1.0118 | 0.9945 | 0.0173 | -0.0292 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 456 | 2 | 399 | 18 | 0.1768 | 0.0847 | 0.0682 | 0.0391 | 0.0118 | 0.119 | 1.0118 | 0.9948 | 0.017 | -0.0291 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 457 | 2 | 400 | 18 | 0.1768 | 0.0847 | 0.0677 | 0.0391 | 0.0118 | 0.1191 | 1.0118 | 0.9969 | 0.0149 | -0.0286 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 458 | 2 | 401 | 18 | 0.1768 | 0.0847 | 0.0675 | 0.0391 | 0.0118 | 0.1193 | 1.0118 | 0.9977 | 0.0141 | -0.0284 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 459 | 2 | 402 | 18 | 0.1768 | 0.0847 | 0.0675 | 0.0391 | 0.0118 | 0.1194 | 1.0118 | 0.9978 | 0.0139 | -0.0284 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 460 | 2 | 403 | 18 | 0.1768 | 0.0847 | 0.0674 | 0.0391 | 0.0117 | 0.1195 | 1.0117 | 0.9984 | 0.0134 | -0.0283 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 461 | 2 | 404 | 18 | 0.1768 | 0.0847 | 0.0673 | 0.0391 | 0.0117 | 0.1197 | 1.0117 | 0.9989 | 0.0128 | -0.0282 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 462 | 2 | 405 | 18 | 0.1768 | 0.0847 | 0.0679 | 0.0391 | 0.0117 | 0.1198 | 1.0117 | 0.9967 | 0.015 | -0.0288 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 463 | 2 | 406 | 18 | 0.1768 | 0.0847 | 0.0681 | 0.0391 | 0.0117 | 0.1199 | 1.0117 | 0.996 | 0.0156 | -0.029 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 464 | 2 | 407 | 18 | 0.1768 | 0.0847 | 0.0681 | 0.0391 | 0.0117 | 0.12 | 1.0117 | 0.9963 | 0.0153 | -0.029 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 465 | 2 | 408 | 18 | 0.1768 | 0.0847 | 0.0684 | 0.0391 | 0.0117 | 0.1202 | 1.0117 | 0.9953 | 0.0164 | -0.0293 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 466 | 2 | 409 | 18 | 0.1768 | 0.0847 | 0.0689 | 0.0391 | 0.0116 | 0.1203 | 1.0116 | 0.9937 | 0.018 | -0.0298 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 467 | 2 | 410 | 18 | 0.1768 | 0.0847 | 0.0694 | 0.0391 | 0.0116 | 0.1204 | 1.0116 | 0.992 | 0.0196 | -0.0303 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 468 | 2 | 411 | 18 | 0.1768 | 0.0847 | 0.0695 | 0.0391 | 0.0116 | 0.1206 | 1.0116 | 0.9915 | 0.0201 | -0.0304 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 469 | 2 | 412 | 18 | 0.1768 | 0.0847 | 0.0701 | 0.0391 | 0.0116 | 0.1207 | 1.0116 | 0.9894 | 0.0222 | -0.031 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 470 | 2 | 413 | 18 | 0.1768 | 0.0847 | 0.0703 | 0.0391 | 0.0116 | 0.1208 | 1.0116 | 0.9891 | 0.0225 | -0.0312 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 471 | 2 | 414 | 18 | 0.1768 | 0.0847 | 0.0701 | 0.0391 | 0.0116 | 0.121 | 1.0116 | 0.9898 | 0.0218 | -0.031 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 472 | 2 | 415 | 18 | 0.1768 | 0.0847 | 0.0709 | 0.0391 | 0.0115 | 0.1211 | 1.0115 | 0.9872 | 0.0243 | -0.0318 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 473 | 2 | 416 | 18 | 0.1768 | 0.0847 | 0.0712 | 0.0391 | 0.0115 | 0.1212 | 1.0115 | 0.9859 | 0.0256 | -0.0322 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 474 | 2 | 417 | 18 | 0.1768 | 0.0847 | 0.0718 | 0.0391 | 0.0115 | 0.1213 | 1.0115 | 0.9841 | 0.0274 | -0.0327 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 475 | 2 | 418 | 18 | 0.1768 | 0.0847 | 0.0718 | 0.0391 | 0.0115 | 0.1215 | 1.0115 | 0.9843 | 0.0272 | -0.0327 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 476 | 2 | 419 | 18 | 0.1768 | 0.0847 | 0.072 | 0.0391 | 0.0115 | 0.1216 | 1.0115 | 0.9836 | 0.0279 | -0.0329 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 477 | 2 | 420 | 18 | 0.1768 | 0.0847 | 0.0715 | 0.0391 | 0.0115 | 0.1217 | 1.0115 | 0.9853 | 0.0261 | -0.0324 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 478 | 2 | 421 | 18 | 0.1768 | 0.0847 | 0.0722 | 0.0391 | 0.0115 | 0.1218 | 1.0115 | 0.9831 | 0.0284 | -0.0331 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 479 | 2 | 422 | 18 | 0.1768 | 0.0847 | 0.0727 | 0.0391 | 0.0114 | 0.122 | 1.0114 | 0.9814 | 0.03 | -0.0336 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 480 | 2 | 423 | 18 | 0.1768 | 0.0847 | 0.0734 | 0.0391 | 0.0114 | 0.1221 | 1.0114 | 0.9791 | 0.0323 | -0.0343 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 481 | 2 | 424 | 18 | 0.1768 | 0.0847 | 0.0738 | 0.0391 | 0.0114 | 0.1222 | 1.0114 | 0.9778 | 0.0336 | -0.0347 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 482 | 2 | 425 | 18 | 0.1768 | 0.0847 | 0.0744 | 0.0391 | 0.0114 | 0.1224 | 1.0114 | 0.9758 | 0.0356 | -0.0353 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 483 | 2 | 426 | 18 | 0.1768 | 0.0847 | 0.0747 | 0.0391 | 0.0114 | 0.1225 | 1.0114 | 0.9747 | 0.0367 | -0.0356 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 484 | 2 | 427 | 18 | 0.1768 | 0.0847 | 0.0755 | 0.0391 | 0.0114 | 0.1226 | 1.0114 | 0.972 | 0.0394 | -0.0364 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 485 | 2 | 428 | 18 | 0.1768 | 0.0847 | 0.0761 | 0.0391 | 0.0113 | 0.1227 | 1.0113 | 0.9702 | 0.0412 | -0.037 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 486 | 2 | 429 | 18 | 0.1768 | 0.0847 | 0.0766 | 0.0391 | 0.0113 | 0.1229 | 1.0113 | 0.9686 | 0.0427 | -0.0375 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 487 | 2 | 430 | 18 | 0.1768 | 0.0847 | 0.077 | 0.0391 | 0.0113 | 0.123 | 1.0113 | 0.9672 | 0.0441 | -0.0379 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 488 | 2 | 431 | 18 | 0.1768 | 0.0847 | 0.0773 | 0.0391 | 0.0113 | 0.1231 | 1.0113 | 0.9662 | 0.0451 | -0.0382 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 489 | 2 | 432 | 18 | 0.1768 | 0.0847 | 0.0771 | 0.0391 | 0.0113 | 0.1232 | 1.0113 | 0.967 | 0.0443 | -0.038 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 490 | 2 | 433 | 18 | 0.1768 | 0.0847 | 0.0769 | 0.0391 | 0.0113 | 0.1234 | 1.0113 | 0.968 | 0.0433 | -0.0378 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 491 | 2 | 434 | 18 | 0.1768 | 0.0847 | 0.0765 | 0.0391 | 0.0113 | 0.1235 | 1.0113 | 0.9693 | 0.042 | -0.0375 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 492 | 2 | 435 | 18 | 0.1768 | 0.0847 | 0.0765 | 0.0391 | 0.0112 | 0.1236 | 1.0112 | 0.9696 | 0.0416 | -0.0374 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 493 | 2 | 436 | 18 | 0.1768 | 0.0847 | 0.0771 | 0.0391 | 0.0112 | 0.1237 | 1.0112 | 0.9678 | 0.0435 | -0.038 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 494 | 2 | 437 | 18 | 0.1768 | 0.0847 | 0.0771 | 0.0391 | 0.0112 | 0.1239 | 1.0112 | 0.9678 | 0.0435 | -0.038 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 495 | 2 | 438 | 18 | 0.1768 | 0.0847 | 0.0779 | 0.0391 | 0.0112 | 0.124 | 1.0112 | 0.9653 | 0.0459 | -0.0388 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 496 | 2 | 439 | 18 | 0.1768 | 0.0847 | 0.0776 | 0.0391 | 0.0112 | 0.1241 | 1.0112 | 0.9663 | 0.0449 | -0.0385 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 497 | 2 | 440 | 18 | 0.1768 | 0.0847 | 0.0771 | 0.0391 | 0.0112 | 0.1242 | 1.0112 | 0.9681 | 0.0431 | -0.038 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 498 | 2 | 441 | 18 | 0.1768 | 0.0847 | 0.0776 | 0.0391 | 0.0112 | 0.1244 | 1.0112 | 0.9665 | 0.0446 | -0.0385 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 499 | 2 | 442 | 18 | 0.1768 | 0.0847 | 0.0783 | 0.0391 | 0.0111 | 0.1245 | 1.0111 | 0.9642 | 0.047 | -0.0392 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 500 | 2 | 443 | 18 | 0.1768 | 0.0847 | 0.0786 | 0.0391 | 0.0111 | 0.1246 | 1.0111 | 0.9634 | 0.0477 | -0.0395 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 501 | 2 | 444 | 18 | 0.1768 | 0.0847 | 0.0792 | 0.0391 | 0.0111 | 0.1247 | 1.0111 | 0.9614 | 0.0497 | -0.0401 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 502 | 2 | 445 | 18 | 0.1768 | 0.0847 | 0.08 | 0.0391 | 0.0111 | 0.1249 | 1.0111 | 0.9587 | 0.0524 | -0.041 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 503 | 2 | 446 | 18 | 0.1768 | 0.0847 | 0.0806 | 0.0391 | 0.0111 | 0.125 | 1.0111 | 0.9571 | 0.054 | -0.0415 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 504 | 2 | 447 | 18 | 0.1768 | 0.0847 | 0.0807 | 0.0391 | 0.0111 | 0.1251 | 1.0111 | 0.9566 | 0.0545 | -0.0416 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 505 | 2 | 448 | 18 | 0.1768 | 0.0847 | 0.0807 | 0.0391 | 0.0111 | 0.1252 | 1.0111 | 0.9568 | 0.0543 | -0.0416 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 506 | 2 | 449 | 18 | 0.1768 | 0.0847 | 0.0806 | 0.0391 | 0.011 | 0.1254 | 1.011 | 0.9573 | 0.0538 | -0.0415 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 507 | 2 | 450 | 18 | 0.1768 | 0.0847 | 0.0804 | 0.0391 | 0.011 | 0.1255 | 1.011 | 0.9582 | 0.0529 | -0.0413 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 508 | 2 | 451 | 18 | 0.1768 | 0.0847 | 0.0803 | 0.0391 | 0.011 | 0.1256 | 1.011 | 0.9585 | 0.0525 | -0.0412 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 509 | 2 | 452 | 18 | 0.1768 | 0.0847 | 0.0804 | 0.0391 | 0.011 | 0.1257 | 1.011 | 0.9583 | 0.0527 | -0.0413 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 510 | 2 | 453 | 18 | 0.1768 | 0.0847 | 0.0804 | 0.0391 | 0.011 | 0.1259 | 1.011 | 0.9584 | 0.0526 | -0.0413 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 511 | 2 | 454 | 18 | 0.1768 | 0.0847 | 0.0804 | 0.0391 | 0.011 | 0.126 | 1.011 | 0.9587 | 0.0523 | -0.0413 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 512 | 2 | 455 | 18 | 0.1768 | 0.0847 | 0.0803 | 0.0391 | 0.011 | 0.1261 | 1.011 | 0.959 | 0.052 | -0.0412 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 513 | 2 | 456 | 18 | 0.1768 | 0.0847 | 0.0805 | 0.0391 | 0.011 | 0.1262 | 1.011 | 0.9586 | 0.0523 | -0.0414 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 514 | 2 | 457 | 18 | 0.1768 | 0.0847 | 0.0803 | 0.0391 | 0.0109 | 0.1264 | 1.0109 | 0.9594 | 0.0516 | -0.0412 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 515 | 2 | 458 | 18 | 0.1768 | 0.0847 | 0.08 | 0.0391 | 0.0109 | 0.1265 | 1.0109 | 0.9604 | 0.0505 | -0.0409 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 516 | 2 | 459 | 18 | 0.1768 | 0.0847 | 0.0798 | 0.0391 | 0.0109 | 0.1266 | 1.0109 | 0.9614 | 0.0495 | -0.0407 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 517 | 2 | 460 | 18 | 0.1768 | 0.0847 | 0.0796 | 0.0391 | 0.0109 | 0.1267 | 1.0109 | 0.9621 | 0.0488 | -0.0405 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 518 | 2 | 461 | 18 | 0.1768 | 0.0847 | 0.0795 | 0.0391 | 0.0109 | 0.1268 | 1.0109 | 0.9627 | 0.0482 | -0.0404 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 519 | 2 | 462 | 18 | 0.1768 | 0.0847 | 0.0801 | 0.0391 | 0.0109 | 0.127 | 1.0109 | 0.9605 | 0.0504 | -0.041 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 520 | 2 | 463 | 18 | 0.1768 | 0.0847 | 0.0796 | 0.0391 | 0.0109 | 0.1271 | 1.0109 | 0.9625 | 0.0484 | -0.0405 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 521 | 2 | 464 | 18 | 0.1768 | 0.0847 | 0.0794 | 0.0391 | 0.0108 | 0.1272 | 1.0108 | 0.9633 | 0.0475 | -0.0403 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 522 | 2 | 465 | 18 | 0.1768 | 0.0847 | 0.079 | 0.0391 | 0.0108 | 0.1273 | 1.0108 | 0.9647 | 0.0461 | -0.0399 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 523 | 2 | 466 | 18 | 0.1768 | 0.0847 | 0.0789 | 0.0391 | 0.0108 | 0.1275 | 1.0108 | 0.9653 | 0.0455 | -0.0398 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 524 | 2 | 467 | 18 | 0.1768 | 0.0847 | 0.0788 | 0.0391 | 0.0108 | 0.1276 | 1.0108 | 0.9656 | 0.0452 | -0.0397 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 525 | 2 | 468 | 18 | 0.1768 | 0.0847 | 0.0787 | 0.0391 | 0.0108 | 0.1277 | 1.0108 | 0.9661 | 0.0447 | -0.0396 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 526 | 2 | 469 | 18 | 0.1768 | 0.0847 | 0.0787 | 0.0391 | 0.0108 | 0.1278 | 1.0108 | 0.9663 | 0.0445 | -0.0396 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 527 | 2 | 470 | 18 | 0.1768 | 0.0847 | 0.0785 | 0.0391 | 0.0108 | 0.1279 | 1.0108 | 0.9671 | 0.0437 | -0.0394 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 528 | 2 | 471 | 18 | 0.1768 | 0.0847 | 0.0784 | 0.0391 | 0.0108 | 0.1281 | 1.0108 | 0.9675 | 0.0432 | -0.0393 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 529 | 2 | 472 | 18 | 0.1768 | 0.0847 | 0.0781 | 0.0391 | 0.0107 | 0.1282 | 1.0107 | 0.9687 | 0.042 | -0.039 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 530 | 2 | 473 | 18 | 0.1768 | 0.0847 | 0.078 | 0.0391 | 0.0107 | 0.1283 | 1.0107 | 0.9691 | 0.0416 | -0.0389 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 531 | 2 | 474 | 18 | 0.1768 | 0.0847 | 0.0783 | 0.0391 | 0.0107 | 0.1284 | 1.0107 | 0.9683 | 0.0424 | -0.0392 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 532 | 2 | 475 | 18 | 0.1768 | 0.0847 | 0.0783 | 0.0391 | 0.0107 | 0.1285 | 1.0107 | 0.9684 | 0.0423 | -0.0392 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 533 | 2 | 476 | 18 | 0.1768 | 0.0847 | 0.0783 | 0.0391 | 0.0107 | 0.1287 | 1.0107 | 0.9684 | 0.0423 | -0.0392 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 534 | 2 | 477 | 18 | 0.1768 | 0.0847 | 0.0787 | 0.0391 | 0.0107 | 0.1288 | 1.0107 | 0.9672 | 0.0435 | -0.0396 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 535 | 2 | 478 | 18 | 0.1768 | 0.0847 | 0.0783 | 0.0391 | 0.0107 | 0.1289 | 1.0107 | 0.9687 | 0.042 | -0.0392 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 536 | 2 | 479 | 18 | 0.1768 | 0.0847 | 0.0787 | 0.0391 | 0.0107 | 0.129 | 1.0107 | 0.9675 | 0.0432 | -0.0396 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 537 | 2 | 480 | 18 | 0.1768 | 0.0847 | 0.0792 | 0.0391 | 0.0106 | 0.1291 | 1.0106 | 0.9659 | 0.0448 | -0.0401 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 538 | 2 | 481 | 18 | 0.1768 | 0.0847 | 0.079 | 0.0391 | 0.0106 | 0.1293 | 1.0106 | 0.9667 | 0.0439 | -0.0399 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 539 | 2 | 482 | 18 | 0.1768 | 0.0847 | 0.0787 | 0.0391 | 0.0106 | 0.1294 | 1.0106 | 0.9677 | 0.043 | -0.0396 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 540 | 2 | 483 | 18 | 0.1768 | 0.0847 | 0.0789 | 0.0391 | 0.0106 | 0.1295 | 1.0106 | 0.9671 | 0.0435 | -0.0398 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 541 | 2 | 484 | 18 | 0.1768 | 0.0847 | 0.0794 | 0.0391 | 0.0106 | 0.1296 | 1.0106 | 0.9655 | 0.0451 | -0.0403 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 542 | 2 | 485 | 18 | 0.1768 | 0.0847 | 0.0798 | 0.0391 | 0.0106 | 0.1297 | 1.0106 | 0.9644 | 0.0462 | -0.0407 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 543 | 2 | 486 | 18 | 0.1768 | 0.0847 | 0.0802 | 0.0391 | 0.0106 | 0.1299 | 1.0106 | 0.9631 | 0.0474 | -0.0411 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 544 | 2 | 487 | 18 | 0.1768 | 0.0847 | 0.0806 | 0.0391 | 0.0106 | 0.13 | 1.0106 | 0.9619 | 0.0487 | -0.0415 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 545 | 2 | 488 | 18 | 0.1768 | 0.0847 | 0.0805 | 0.0391 | 0.0106 | 0.1301 | 1.0106 | 0.9625 | 0.048 | -0.0414 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 546 | 2 | 489 | 18 | 0.1768 | 0.0847 | 0.0809 | 0.0391 | 0.0105 | 0.1302 | 1.0105 | 0.9612 | 0.0493 | -0.0418 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 547 | 2 | 490 | 18 | 0.1768 | 0.0847 | 0.0807 | 0.0391 | 0.0105 | 0.1303 | 1.0105 | 0.9619 | 0.0486 | -0.0416 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 548 | 2 | 491 | 18 | 0.1768 | 0.0847 | 0.0806 | 0.0391 | 0.0105 | 0.1305 | 1.0105 | 0.9625 | 0.0481 | -0.0415 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 549 | 2 | 492 | 18 | 0.1768 | 0.0847 | 0.0806 | 0.0391 | 0.0105 | 0.1306 | 1.0105 | 0.9625 | 0.0481 | -0.0415 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 550 | 2 | 492 | 18 | 0.1768 | 0.0847 | 0.0806 | 0.0391 | 0.0105 | 0.1307 | 1.0105 | 0.9626 | 0.0479 | -0.0415 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 551 | 2 | 493 | 18 | 0.1768 | 0.0847 | 0.0807 | 0.0391 | 0.0105 | 0.1308 | 1.0105 | 0.9625 | 0.048 | -0.0416 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 552 | 2 | 494 | 18 | 0.1768 | 0.0847 | 0.0806 | 0.0391 | 0.0105 | 0.1309 | 1.0105 | 0.9629 | 0.0476 | -0.0415 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 553 | 2 | 495 | 18 | 0.1768 | 0.0847 | 0.0811 | 0.0391 | 0.0105 | 0.1311 | 1.0105 | 0.9612 | 0.0493 | -0.042 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 554 | 2 | 496 | 18 | 0.1768 | 0.0847 | 0.0816 | 0.0391 | 0.0105 | 0.1312 | 1.0105 | 0.9596 | 0.0508 | -0.0425 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 555 | 2 | 497 | 18 | 0.1768 | 0.0847 | 0.0822 | 0.0391 | 0.0105 | 0.1313 | 1.0105 | 0.9579 | 0.0526 | -0.0431 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 556 | 2 | 498 | 18 | 0.1768 | 0.0847 | 0.0822 | 0.0391 | 0.0104 | 0.1314 | 1.0104 | 0.9581 | 0.0523 | -0.0431 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 557 | 2 | 499 | 18 | 0.1768 | 0.0847 | 0.0822 | 0.0391 | 0.0104 | 0.1315 | 1.0104 | 0.9582 | 0.0522 | -0.0431 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 558 | 2 | 500 | 18 | 0.1768 | 0.0847 | 0.0822 | 0.0391 | 0.0104 | 0.1316 | 1.0104 | 0.9581 | 0.0523 | -0.0431 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 559 | 2 | 501 | 18 | 0.1768 | 0.0847 | 0.0822 | 0.0391 | 0.0104 | 0.1318 | 1.0104 | 0.9583 | 0.0521 | -0.0431 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 560 | 2 | 502 | 18 | 0.1768 | 0.0847 | 0.0821 | 0.0391 | 0.0104 | 0.1319 | 1.0104 | 0.9589 | 0.0515 | -0.043 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 561 | 2 | 503 | 18 | 0.1768 | 0.0847 | 0.0818 | 0.0391 | 0.0104 | 0.132 | 1.0104 | 0.96 | 0.0504 | -0.0427 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 562 | 2 | 504 | 18 | 0.1768 | 0.0847 | 0.0818 | 0.0391 | 0.0104 | 0.1321 | 1.0104 | 0.96 | 0.0504 | -0.0427 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 563 | 2 | 505 | 18 | 0.1768 | 0.0847 | 0.0816 | 0.0391 | 0.0104 | 0.1322 | 1.0104 | 0.9608 | 0.0496 | -0.0425 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 564 | 2 | 505 | 18 | 0.1768 | 0.0847 | 0.0816 | 0.0391 | 0.0104 | 0.1324 | 1.0104 | 0.9609 | 0.0495 | -0.0425 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 565 | 2 | 506 | 18 | 0.1768 | 0.0847 | 0.0817 | 0.0391 | 0.0104 | 0.1325 | 1.0104 | 0.9609 | 0.0495 | -0.0426 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 566 | 2 | 507 | 18 | 0.1768 | 0.0847 | 0.0814 | 0.0391 | 0.0104 | 0.1326 | 1.0104 | 0.9618 | 0.0485 | -0.0423 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 567 | 2 | 508 | 18 | 0.1768 | 0.0847 | 0.0817 | 0.0391 | 0.0103 | 0.1327 | 1.0103 | 0.9608 | 0.0495 | -0.0426 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 568 | 2 | 509 | 18 | 0.1768 | 0.0847 | 0.0814 | 0.0391 | 0.0103 | 0.1328 | 1.0103 | 0.9621 | 0.0483 | -0.0423 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 569 | 2 | 510 | 18 | 0.1768 | 0.0847 | 0.0812 | 0.0391 | 0.0103 | 0.1329 | 1.0103 | 0.9627 | 0.0476 | -0.0421 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 570 | 2 | 511 | 18 | 0.1768 | 0.0847 | 0.0809 | 0.0391 | 0.0103 | 0.1331 | 1.0103 | 0.964 | 0.0463 | -0.0418 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 571 | 2 | 512 | 18 | 0.1768 | 0.0847 | 0.0809 | 0.0391 | 0.0103 | 0.1332 | 1.0103 | 0.9642 | 0.0461 | -0.0418 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 572 | 2 | 513 | 18 | 0.1768 | 0.0847 | 0.0805 | 0.0391 | 0.0103 | 0.1333 | 1.0103 | 0.9655 | 0.0448 | -0.0414 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 573 | 2 | 514 | 18 | 0.1768 | 0.0847 | 0.0802 | 0.0391 | 0.0103 | 0.1334 | 1.0103 | 0.9668 | 0.0435 | -0.0411 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 574 | 2 | 515 | 18 | 0.1768 | 0.0847 | 0.0799 | 0.0391 | 0.0103 | 0.1335 | 1.0103 | 0.968 | 0.0423 | -0.0408 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 575 | 2 | 516 | 18 | 0.1768 | 0.0847 | 0.0802 | 0.0391 | 0.0103 | 0.1336 | 1.0103 | 0.9669 | 0.0433 | -0.0411 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 576 | 2 | 517 | 18 | 0.1768 | 0.0847 | 0.0801 | 0.0391 | 0.0102 | 0.1338 | 1.0102 | 0.9672 | 0.043 | -0.041 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 577 | 2 | 518 | 18 | 0.1768 | 0.0847 | 0.0806 | 0.0391 | 0.0102 | 0.1339 | 1.0102 | 0.9659 | 0.0444 | -0.0415 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 578 | 2 | 519 | 18 | 0.1768 | 0.0847 | 0.0805 | 0.0391 | 0.0102 | 0.134 | 1.0102 | 0.9662 | 0.044 | -0.0414 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 579 | 2 | 520 | 18 | 0.1768 | 0.0847 | 0.0808 | 0.0391 | 0.0102 | 0.1341 | 1.0102 | 0.9652 | 0.045 | -0.0417 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 580 | 2 | 521 | 18 | 0.1768 | 0.0847 | 0.081 | 0.0391 | 0.0102 | 0.1342 | 1.0102 | 0.9648 | 0.0454 | -0.0419 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 581 | 2 | 522 | 18 | 0.1768 | 0.0847 | 0.0806 | 0.0391 | 0.0102 | 0.1343 | 1.0102 | 0.9662 | 0.044 | -0.0415 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 582 | 2 | 522 | 18 | 0.1768 | 0.0847 | 0.0806 | 0.0391 | 0.0102 | 0.1345 | 1.0102 | 0.9663 | 0.0439 | -0.0415 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 583 | 2 | 523 | 18 | 0.1768 | 0.0847 | 0.0806 | 0.0391 | 0.0102 | 0.1346 | 1.0102 | 0.9665 | 0.0436 | -0.0415 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 584 | 2 | 524 | 18 | 0.1768 | 0.0847 | 0.0805 | 0.0391 | 0.0102 | 0.1347 | 1.0102 | 0.9669 | 0.0433 | -0.0414 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 585 | 2 | 525 | 18 | 0.1768 | 0.0847 | 0.0805 | 0.0391 | 0.0102 | 0.1348 | 1.0102 | 0.967 | 0.0432 | -0.0414 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 586 | 2 | 526 | 18 | 0.1768 | 0.0847 | 0.0803 | 0.0391 | 0.0102 | 0.1349 | 1.0102 | 0.9678 | 0.0424 | -0.0412 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 587 | 2 | 527 | 18 | 0.1768 | 0.0847 | 0.0806 | 0.0391 | 0.0101 | 0.135 | 1.0101 | 0.9671 | 0.043 | -0.0415 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 588 | 2 | 528 | 18 | 0.1768 | 0.0847 | 0.0807 | 0.0391 | 0.0101 | 0.1351 | 1.0101 | 0.9668 | 0.0433 | -0.0416 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 589 | 2 | 529 | 18 | 0.1768 | 0.0847 | 0.0805 | 0.0391 | 0.0101 | 0.1353 | 1.0101 | 0.9674 | 0.0427 | -0.0414 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 590 | 2 | 530 | 18 | 0.1768 | 0.0847 | 0.0801 | 0.0391 | 0.0101 | 0.1354 | 1.0101 | 0.969 | 0.0411 | -0.041 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 591 | 2 | 531 | 18 | 0.1768 | 0.0847 | 0.08 | 0.0391 | 0.0101 | 0.1355 | 1.0101 | 0.9694 | 0.0407 | -0.0409 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 592 | 2 | 532 | 18 | 0.1768 | 0.0847 | 0.0799 | 0.0391 | 0.0101 | 0.1356 | 1.0101 | 0.9699 | 0.0401 | -0.0408 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 593 | 2 | 533 | 18 | 0.1768 | 0.0847 | 0.0795 | 0.0391 | 0.0101 | 0.1357 | 1.0101 | 0.9712 | 0.0388 | -0.0404 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 594 | 2 | 534 | 18 | 0.1768 | 0.0847 | 0.0795 | 0.0391 | 0.0101 | 0.1358 | 1.0101 | 0.9715 | 0.0386 | -0.0404 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 595 | 2 | 535 | 18 | 0.1768 | 0.0847 | 0.0794 | 0.0391 | 0.0101 | 0.1359 | 1.0101 | 0.9721 | 0.038 | -0.0403 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 596 | 2 | 536 | 18 | 0.1768 | 0.0847 | 0.0792 | 0.0391 | 0.01 | 0.1361 | 1.01 | 0.9726 | 0.0375 | -0.0402 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 597 | 2 | 537 | 18 | 0.1768 | 0.0847 | 0.0789 | 0.0391 | 0.01 | 0.1362 | 1.01 | 0.9738 | 0.0362 | -0.0398 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 598 | 2 | 538 | 18 | 0.1768 | 0.0847 | 0.0793 | 0.0391 | 0.01 | 0.1363 | 1.01 | 0.9728 | 0.0373 | -0.0402 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 599 | 2 | 539 | 18 | 0.1768 | 0.0847 | 0.0791 | 0.0391 | 0.01 | 0.1364 | 1.01 | 0.9735 | 0.0365 | -0.04 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 600 | 2 | 539 | 18 | 0.1768 | 0.0847 | 0.0791 | 0.0391 | 0.01 | 0.1365 | 1.01 | 0.9736 | 0.0364 | -0.04 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 601 | 2 | 540 | 18 | 0.1768 | 0.0847 | 0.0791 | 0.0391 | 0.01 | 0.1366 | 1.01 | 0.9737 | 0.0364 | -0.04 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 602 | 2 | 541 | 18 | 0.1768 | 0.0847 | 0.0792 | 0.0391 | 0.01 | 0.1367 | 1.01 | 0.9733 | 0.0367 | -0.0401 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 603 | 2 | 542 | 18 | 0.1768 | 0.0847 | 0.0789 | 0.0391 | 0.01 | 0.1369 | 1.01 | 0.9747 | 0.0353 | -0.0398 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 604 | 2 | 543 | 18 | 0.1768 | 0.0847 | 0.0785 | 0.0391 | 0.01 | 0.137 | 1.01 | 0.976 | 0.034 | -0.0394 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 605 | 2 | 543 | 18 | 0.1768 | 0.0847 | 0.0785 | 0.0391 | 0.01 | 0.1371 | 1.01 | 0.9761 | 0.0339 | -0.0394 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 606 | 2 | 543 | 18 | 0.1768 | 0.0847 | 0.0785 | 0.0391 | 0.01 | 0.1372 | 1.01 | 0.9762 | 0.0338 | -0.0394 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 607 | 2 | 543 | 18 | 0.1768 | 0.0847 | 0.0785 | 0.0391 | 0.01 | 0.1373 | 1.01 | 0.9764 | 0.0337 | -0.0394 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 608 | 2 | 543 | 18 | 0.1768 | 0.0847 | 0.0785 | 0.0391 | 0.01 | 0.1374 | 1.01 | 0.9765 | 0.0335 | -0.0394 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 609 | 2 | 544 | 18 | 0.1768 | 0.0847 | 0.079 | 0.0391 | 0.01 | 0.1375 | 1.01 | 0.9748 | 0.0352 | -0.0399 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 610 | 2 | 545 | 18 | 0.1768 | 0.0847 | 0.0793 | 0.0391 | 0.01 | 0.1376 | 1.01 | 0.974 | 0.036 | -0.0402 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 611 | 2 | 546 | 18 | 0.1768 | 0.0847 | 0.0798 | 0.0391 | 0.01 | 0.1378 | 1.01 | 0.9725 | 0.0375 | -0.0407 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 612 | 2 | 547 | 18 | 0.1768 | 0.0847 | 0.0798 | 0.0391 | 0.01 | 0.1379 | 1.01 | 0.9725 | 0.0375 | -0.0407 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 613 | 2 | 548 | 18 | 0.1768 | 0.0847 | 0.0803 | 0.0391 | 0.01 | 0.138 | 1.01 | 0.9709 | 0.0391 | -0.0412 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 614 | 2 | 549 | 18 | 0.1768 | 0.0847 | 0.0808 | 0.0391 | 0.01 | 0.1381 | 1.01 | 0.9694 | 0.0405 | -0.0417 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 615 | 2 | 550 | 18 | 0.1768 | 0.0847 | 0.0809 | 0.0391 | 0.0099 | 0.1382 | 1.0099 | 0.9693 | 0.0407 | -0.0418 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 616 | 2 | 551 | 18 | 0.1768 | 0.0847 | 0.0814 | 0.0391 | 0.0099 | 0.1383 | 1.0099 | 0.9675 | 0.0425 | -0.0423 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 617 | 2 | 552 | 18 | 0.1768 | 0.0847 | 0.082 | 0.0391 | 0.0099 | 0.1384 | 1.0099 | 0.9656 | 0.0444 | -0.0429 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 618 | 2 | 553 | 18 | 0.1768 | 0.0847 | 0.0821 | 0.0391 | 0.0099 | 0.1385 | 1.0099 | 0.9656 | 0.0443 | -0.043 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 619 | 2 | 554 | 18 | 0.1768 | 0.0847 | 0.0816 | 0.0391 | 0.0099 | 0.1387 | 1.0099 | 0.9671 | 0.0428 | -0.0425 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 620 | 2 | 555 | 18 | 0.1768 | 0.0847 | 0.0818 | 0.0391 | 0.0099 | 0.1388 | 1.0099 | 0.9666 | 0.0433 | -0.0427 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 621 | 2 | 556 | 18 | 0.1768 | 0.0847 | 0.0821 | 0.0391 | 0.0099 | 0.1389 | 1.0099 | 0.9657 | 0.0442 | -0.043 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 622 | 2 | 557 | 18 | 0.1768 | 0.0847 | 0.0824 | 0.0391 | 0.0099 | 0.139 | 1.0099 | 0.965 | 0.0448 | -0.0433 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 623 | 2 | 558 | 18 | 0.1768 | 0.0847 | 0.083 | 0.0391 | 0.0099 | 0.1391 | 1.0099 | 0.9631 | 0.0468 | -0.0439 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 624 | 2 | 559 | 18 | 0.1768 | 0.0847 | 0.0826 | 0.0391 | 0.0099 | 0.1392 | 1.0099 | 0.9644 | 0.0455 | -0.0435 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 625 | 2 | 560 | 18 | 0.1768 | 0.0847 | 0.0823 | 0.0391 | 0.0098 | 0.1393 | 1.0098 | 0.9656 | 0.0442 | -0.0432 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 626 | 2 | 561 | 18 | 0.1768 | 0.0847 | 0.0823 | 0.0391 | 0.0098 | 0.1394 | 1.0098 | 0.9658 | 0.044 | -0.0432 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 627 | 2 | 562 | 18 | 0.1768 | 0.0847 | 0.0825 | 0.0391 | 0.0098 | 0.1396 | 1.0098 | 0.9652 | 0.0447 | -0.0434 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 628 | 2 | 563 | 18 | 0.1768 | 0.0847 | 0.0821 | 0.0391 | 0.0098 | 0.1397 | 1.0098 | 0.9666 | 0.0432 | -0.043 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 629 | 2 | 564 | 18 | 0.1768 | 0.0847 | 0.082 | 0.0391 | 0.0098 | 0.1398 | 1.0098 | 0.967 | 0.0428 | -0.0429 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 630 | 2 | 565 | 18 | 0.1768 | 0.0847 | 0.0819 | 0.0391 | 0.0098 | 0.1399 | 1.0098 | 0.9673 | 0.0425 | -0.0428 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 631 | 2 | 566 | 18 | 0.1768 | 0.0847 | 0.0823 | 0.0391 | 0.0098 | 0.14 | 1.0098 | 0.9661 | 0.0437 | -0.0432 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 632 | 2 | 567 | 18 | 0.1768 | 0.0847 | 0.0825 | 0.0391 | 0.0098 | 0.1401 | 1.0098 | 0.9658 | 0.044 | -0.0434 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 633 | 2 | 568 | 18 | 0.1768 | 0.0847 | 0.0827 | 0.0391 | 0.0098 | 0.1402 | 1.0098 | 0.965 | 0.0447 | -0.0436 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 634 | 2 | 569 | 18 | 0.1768 | 0.0847 | 0.0831 | 0.0391 | 0.0098 | 0.1403 | 1.0098 | 0.9641 | 0.0457 | -0.044 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 635 | 2 | 570 | 18 | 0.1768 | 0.0847 | 0.0834 | 0.0391 | 0.0098 | 0.1404 | 1.0098 | 0.9629 | 0.0468 | -0.0443 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 636 | 2 | 571 | 18 | 0.1768 | 0.0847 | 0.084 | 0.0391 | 0.0097 | 0.1406 | 1.0097 | 0.9613 | 0.0485 | -0.0449 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 637 | 2 | 572 | 18 | 0.1768 | 0.0847 | 0.0839 | 0.0391 | 0.0097 | 0.1407 | 1.0097 | 0.9615 | 0.0482 | -0.0448 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 638 | 2 | 573 | 18 | 0.1768 | 0.0847 | 0.0839 | 0.0391 | 0.0097 | 0.1408 | 1.0097 | 0.9618 | 0.0479 | -0.0448 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 639 | 2 | 574 | 18 | 0.1768 | 0.0847 | 0.0838 | 0.0391 | 0.0097 | 0.1409 | 1.0097 | 0.9622 | 0.0475 | -0.0447 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 640 | 2 | 575 | 18 | 0.1768 | 0.0847 | 0.0841 | 0.0391 | 0.0097 | 0.141 | 1.0097 | 0.9613 | 0.0484 | -0.045 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 641 | 2 | 576 | 18 | 0.1768 | 0.0847 | 0.0844 | 0.0391 | 0.0097 | 0.1411 | 1.0097 | 0.9604 | 0.0493 | -0.0453 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 642 | 2 | 577 | 18 | 0.1768 | 0.0847 | 0.0848 | 0.0391 | 0.0097 | 0.1412 | 1.0097 | 0.9592 | 0.0505 | -0.0457 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 643 | 2 | 578 | 18 | 0.1768 | 0.0847 | 0.0851 | 0.0391 | 0.0097 | 0.1413 | 1.0097 | 0.9584 | 0.0512 | -0.046 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 644 | 2 | 579 | 18 | 0.1768 | 0.0847 | 0.0853 | 0.0391 | 0.0097 | 0.1414 | 1.0097 | 0.9578 | 0.0519 | -0.0462 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 645 | 2 | 580 | 18 | 0.1768 | 0.0847 | 0.0856 | 0.0391 | 0.0097 | 0.1415 | 1.0097 | 0.9569 | 0.0527 | -0.0465 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 646 | 2 | 581 | 18 | 0.1768 | 0.0847 | 0.0859 | 0.0391 | 0.0097 | 0.1417 | 1.0097 | 0.9562 | 0.0534 | -0.0468 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 647 | 2 | 582 | 18 | 0.1768 | 0.0847 | 0.0855 | 0.0391 | 0.0096 | 0.1418 | 1.0096 | 0.9576 | 0.052 | -0.0464 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 648 | 2 | 583 | 18 | 0.1768 | 0.0847 | 0.0856 | 0.0391 | 0.0096 | 0.1419 | 1.0096 | 0.9574 | 0.0523 | -0.0465 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 649 | 2 | 584 | 18 | 0.1768 | 0.0847 | 0.0862 | 0.0391 | 0.0096 | 0.142 | 1.0096 | 0.9553 | 0.0543 | -0.0471 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 650 | 2 | 585 | 18 | 0.1768 | 0.0847 | 0.0865 | 0.0391 | 0.0096 | 0.1421 | 1.0096 | 0.9545 | 0.0551 | -0.0474 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 651 | 2 | 586 | 18 | 0.1768 | 0.0847 | 0.0864 | 0.0391 | 0.0096 | 0.1422 | 1.0096 | 0.9551 | 0.0546 | -0.0473 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 652 | 2 | 587 | 18 | 0.1768 | 0.0847 | 0.0864 | 0.0391 | 0.0096 | 0.1423 | 1.0096 | 0.9552 | 0.0544 | -0.0473 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 653 | 2 | 588 | 18 | 0.1768 | 0.0847 | 0.0864 | 0.0391 | 0.0096 | 0.1424 | 1.0096 | 0.9553 | 0.0542 | -0.0473 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 654 | 2 | 589 | 18 | 0.1768 | 0.0847 | 0.0865 | 0.0391 | 0.0096 | 0.1425 | 1.0096 | 0.9552 | 0.0544 | -0.0474 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 655 | 2 | 590 | 18 | 0.1768 | 0.0847 | 0.0866 | 0.0391 | 0.0096 | 0.1426 | 1.0096 | 0.955 | 0.0546 | -0.0475 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 656 | 2 | 591 | 18 | 0.1768 | 0.0847 | 0.0869 | 0.0391 | 0.0096 | 0.1427 | 1.0096 | 0.954 | 0.0556 | -0.0478 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 657 | 2 | 592 | 18 | 0.1768 | 0.0847 | 0.0871 | 0.0391 | 0.0096 | 0.1429 | 1.0096 | 0.9534 | 0.0561 | -0.048 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 658 | 2 | 593 | 18 | 0.1768 | 0.0847 | 0.0875 | 0.0391 | 0.0095 | 0.143 | 1.0095 | 0.9524 | 0.0572 | -0.0484 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 659 | 2 | 594 | 18 | 0.1768 | 0.0847 | 0.088 | 0.0391 | 0.0095 | 0.1431 | 1.0095 | 0.9507 | 0.0588 | -0.0489 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 660 | 2 | 595 | 18 | 0.1768 | 0.0847 | 0.0882 | 0.0391 | 0.0095 | 0.1432 | 1.0095 | 0.9501 | 0.0594 | -0.0492 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 661 | 2 | 596 | 18 | 0.1768 | 0.0847 | 0.0879 | 0.0391 | 0.0095 | 0.1433 | 1.0095 | 0.9512 | 0.0583 | -0.0488 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 662 | 2 | 597 | 18 | 0.1768 | 0.0847 | 0.0875 | 0.0391 | 0.0095 | 0.1434 | 1.0095 | 0.9528 | 0.0567 | -0.0484 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 663 | 2 | 598 | 18 | 0.1768 | 0.0847 | 0.0878 | 0.0391 | 0.0095 | 0.1435 | 1.0095 | 0.952 | 0.0576 | -0.0487 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 664 | 2 | 599 | 18 | 0.1768 | 0.0847 | 0.0876 | 0.0391 | 0.0095 | 0.1436 | 1.0095 | 0.9527 | 0.0568 | -0.0485 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 665 | 2 | 600 | 18 | 0.1768 | 0.0847 | 0.0883 | 0.0391 | 0.0095 | 0.1437 | 1.0095 | 0.9506 | 0.0589 | -0.0492 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 666 | 2 | 601 | 18 | 0.1768 | 0.0847 | 0.0886 | 0.0391 | 0.0095 | 0.1438 | 1.0095 | 0.9496 | 0.0598 | -0.0495 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 667 | 2 | 602 | 18 | 0.1768 | 0.0847 | 0.0887 | 0.0391 | 0.0095 | 0.1439 | 1.0095 | 0.9494 | 0.0601 | -0.0496 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 668 | 2 | 603 | 18 | 0.1768 | 0.0847 | 0.0891 | 0.0391 | 0.0095 | 0.144 | 1.0095 | 0.9484 | 0.061 | -0.05 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 669 | 2 | 604 | 18 | 0.1768 | 0.0847 | 0.0897 | 0.0391 | 0.0094 | 0.1442 | 1.0094 | 0.9464 | 0.0631 | -0.0507 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 670 | 2 | 605 | 18 | 0.1768 | 0.0847 | 0.0896 | 0.0391 | 0.0094 | 0.1443 | 1.0094 | 0.947 | 0.0624 | -0.0505 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 671 | 2 | 606 | 18 | 0.1768 | 0.0847 | 0.0898 | 0.0391 | 0.0094 | 0.1444 | 1.0094 | 0.9463 | 0.0631 | -0.0507 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 672 | 2 | 607 | 18 | 0.1768 | 0.0847 | 0.09 | 0.0391 | 0.0094 | 0.1445 | 1.0094 | 0.9458 | 0.0636 | -0.0509 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 673 | 2 | 608 | 18 | 0.1768 | 0.0847 | 0.0905 | 0.0391 | 0.0094 | 0.1446 | 1.0094 | 0.9444 | 0.065 | -0.0514 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 674 | 2 | 609 | 18 | 0.1768 | 0.0847 | 0.0908 | 0.0391 | 0.0094 | 0.1447 | 1.0094 | 0.9435 | 0.0659 | -0.0517 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 675 | 2 | 610 | 18 | 0.1768 | 0.0847 | 0.0907 | 0.0391 | 0.0094 | 0.1448 | 1.0094 | 0.9442 | 0.0652 | -0.0516 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 676 | 2 | 610 | 18 | 0.1768 | 0.0847 | 0.0907 | 0.0391 | 0.0094 | 0.1449 | 1.0094 | 0.9443 | 0.0651 | -0.0516 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 677 | 2 | 611 | 18 | 0.1768 | 0.0847 | 0.0903 | 0.0391 | 0.0094 | 0.145 | 1.0094 | 0.9454 | 0.064 | -0.0512 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 678 | 2 | 612 | 18 | 0.1768 | 0.0847 | 0.0902 | 0.0391 | 0.0094 | 0.1451 | 1.0094 | 0.9461 | 0.0633 | -0.0511 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 679 | 2 | 613 | 18 | 0.1768 | 0.0847 | 0.0902 | 0.0391 | 0.0094 | 0.1452 | 1.0094 | 0.9461 | 0.0633 | -0.0511 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 680 | 2 | 614 | 18 | 0.1768 | 0.0847 | 0.09 | 0.0391 | 0.0094 | 0.1453 | 1.0094 | 0.9467 | 0.0627 | -0.0509 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 681 | 2 | 615 | 18 | 0.1768 | 0.0847 | 0.0895 | 0.0391 | 0.0094 | 0.1454 | 1.0094 | 0.9484 | 0.0609 | -0.0504 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 682 | 2 | 616 | 18 | 0.1768 | 0.0847 | 0.0895 | 0.0391 | 0.0094 | 0.1455 | 1.0094 | 0.9486 | 0.0607 | -0.0504 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 683 | 2 | 617 | 18 | 0.1768 | 0.0847 | 0.0893 | 0.0391 | 0.0093 | 0.1457 | 1.0093 | 0.9493 | 0.0601 | -0.0502 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 684 | 2 | 618 | 18 | 0.1768 | 0.0847 | 0.0897 | 0.0391 | 0.0093 | 0.1458 | 1.0093 | 0.9482 | 0.0612 | -0.0506 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 685 | 2 | 619 | 18 | 0.1768 | 0.0847 | 0.0901 | 0.0391 | 0.0093 | 0.1459 | 1.0093 | 0.9471 | 0.0623 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 686 | 2 | 620 | 18 | 0.1768 | 0.0847 | 0.0903 | 0.0391 | 0.0093 | 0.146 | 1.0093 | 0.9463 | 0.063 | -0.0513 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 687 | 2 | 621 | 18 | 0.1768 | 0.0847 | 0.0908 | 0.0391 | 0.0093 | 0.1461 | 1.0093 | 0.9449 | 0.0644 | -0.0517 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 688 | 2 | 622 | 18 | 0.1768 | 0.0847 | 0.0912 | 0.0391 | 0.0093 | 0.1462 | 1.0093 | 0.9438 | 0.0655 | -0.0521 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 689 | 2 | 623 | 18 | 0.1768 | 0.0847 | 0.0916 | 0.0391 | 0.0093 | 0.1463 | 1.0093 | 0.9427 | 0.0666 | -0.0525 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 690 | 2 | 624 | 18 | 0.1768 | 0.0847 | 0.0918 | 0.0391 | 0.0093 | 0.1464 | 1.0093 | 0.9422 | 0.0671 | -0.0527 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 691 | 2 | 625 | 18 | 0.1768 | 0.0847 | 0.0919 | 0.0391 | 0.0093 | 0.1465 | 1.0093 | 0.9422 | 0.0671 | -0.0528 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 692 | 2 | 626 | 18 | 0.1768 | 0.0847 | 0.092 | 0.0391 | 0.0093 | 0.1466 | 1.0093 | 0.9417 | 0.0676 | -0.0529 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 693 | 2 | 627 | 18 | 0.1768 | 0.0847 | 0.0918 | 0.0391 | 0.0093 | 0.1467 | 1.0093 | 0.9426 | 0.0667 | -0.0527 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 694 | 2 | 628 | 18 | 0.1768 | 0.0847 | 0.0921 | 0.0391 | 0.0093 | 0.1468 | 1.0093 | 0.9417 | 0.0676 | -0.053 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 695 | 2 | 629 | 18 | 0.1768 | 0.0847 | 0.0926 | 0.0391 | 0.0092 | 0.1469 | 1.0092 | 0.9404 | 0.0688 | -0.0535 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 696 | 2 | 630 | 18 | 0.1768 | 0.0847 | 0.0928 | 0.0391 | 0.0092 | 0.147 | 1.0092 | 0.9397 | 0.0695 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 697 | 2 | 631 | 18 | 0.1768 | 0.0847 | 0.093 | 0.0391 | 0.0092 | 0.1471 | 1.0092 | 0.9393 | 0.07 | -0.0539 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 698 | 2 | 632 | 18 | 0.1768 | 0.0847 | 0.0931 | 0.0391 | 0.0092 | 0.1472 | 1.0092 | 0.939 | 0.0702 | -0.054 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 699 | 2 | 633 | 18 | 0.1768 | 0.0847 | 0.0936 | 0.0391 | 0.0092 | 0.1473 | 1.0092 | 0.9376 | 0.0716 | -0.0545 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 700 | 2 | 634 | 18 | 0.1768 | 0.0847 | 0.094 | 0.0391 | 0.0092 | 0.1475 | 1.0092 | 0.9365 | 0.0727 | -0.0549 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 701 | 2 | 635 | 18 | 0.1768 | 0.0847 | 0.0939 | 0.0391 | 0.0092 | 0.1476 | 1.0092 | 0.937 | 0.0722 | -0.0548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 702 | 2 | 636 | 18 | 0.1768 | 0.0847 | 0.0946 | 0.0391 | 0.0092 | 0.1477 | 1.0092 | 0.935 | 0.0742 | -0.0555 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 703 | 2 | 637 | 18 | 0.1768 | 0.0847 | 0.0944 | 0.0391 | 0.0092 | 0.1478 | 1.0092 | 0.9357 | 0.0735 | -0.0553 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 704 | 2 | 638 | 18 | 0.1768 | 0.0847 | 0.0944 | 0.0391 | 0.0092 | 0.1479 | 1.0092 | 0.9358 | 0.0734 | -0.0553 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 705 | 2 | 639 | 18 | 0.1768 | 0.0847 | 0.094 | 0.0391 | 0.0092 | 0.148 | 1.0092 | 0.9371 | 0.0721 | -0.0549 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 706 | 2 | 640 | 18 | 0.1768 | 0.0847 | 0.0938 | 0.0391 | 0.0092 | 0.1481 | 1.0092 | 0.9379 | 0.0712 | -0.0547 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 707 | 2 | 641 | 18 | 0.1768 | 0.0847 | 0.094 | 0.0391 | 0.0092 | 0.1482 | 1.0092 | 0.9374 | 0.0718 | -0.0549 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 708 | 2 | 642 | 18 | 0.1768 | 0.0847 | 0.0937 | 0.0391 | 0.0091 | 0.1483 | 1.0091 | 0.9383 | 0.0709 | -0.0546 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 709 | 2 | 643 | 18 | 0.1768 | 0.0847 | 0.0937 | 0.0391 | 0.0091 | 0.1484 | 1.0091 | 0.9384 | 0.0707 | -0.0546 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 710 | 2 | 644 | 18 | 0.1768 | 0.0847 | 0.0935 | 0.0391 | 0.0091 | 0.1485 | 1.0091 | 0.9392 | 0.0699 | -0.0544 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 711 | 2 | 644 | 18 | 0.1768 | 0.0847 | 0.0935 | 0.0391 | 0.0091 | 0.1486 | 1.0091 | 0.9393 | 0.0698 | -0.0544 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 712 | 2 | 645 | 18 | 0.1768 | 0.0847 | 0.0934 | 0.0391 | 0.0091 | 0.1487 | 1.0091 | 0.9398 | 0.0694 | -0.0543 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 713 | 2 | 646 | 18 | 0.1768 | 0.0847 | 0.0935 | 0.0391 | 0.0091 | 0.1488 | 1.0091 | 0.9394 | 0.0697 | -0.0544 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 714 | 2 | 647 | 18 | 0.1768 | 0.0847 | 0.0935 | 0.0391 | 0.0091 | 0.1489 | 1.0091 | 0.9396 | 0.0695 | -0.0544 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 715 | 2 | 648 | 18 | 0.1768 | 0.0847 | 0.0936 | 0.0391 | 0.0091 | 0.149 | 1.0091 | 0.9393 | 0.0698 | -0.0545 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 716 | 2 | 649 | 18 | 0.1768 | 0.0847 | 0.0938 | 0.0391 | 0.0091 | 0.1491 | 1.0091 | 0.9389 | 0.0702 | -0.0547 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 717 | 2 | 650 | 18 | 0.1768 | 0.0847 | 0.0939 | 0.0391 | 0.0091 | 0.1492 | 1.0091 | 0.9388 | 0.0703 | -0.0548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 718 | 2 | 651 | 18 | 0.1768 | 0.0847 | 0.0939 | 0.0391 | 0.0091 | 0.1493 | 1.0091 | 0.9388 | 0.0703 | -0.0548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 719 | 2 | 652 | 18 | 0.1768 | 0.0847 | 0.0944 | 0.0391 | 0.0091 | 0.1494 | 1.0091 | 0.9374 | 0.0717 | -0.0553 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 720 | 2 | 653 | 18 | 0.1768 | 0.0847 | 0.0946 | 0.0391 | 0.0091 | 0.1495 | 1.0091 | 0.937 | 0.0721 | -0.0555 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 721 | 2 | 654 | 18 | 0.1768 | 0.0847 | 0.0946 | 0.0391 | 0.0091 | 0.1496 | 1.0091 | 0.937 | 0.0721 | -0.0555 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 722 | 2 | 655 | 18 | 0.1768 | 0.0847 | 0.0951 | 0.0391 | 0.0091 | 0.1498 | 1.0091 | 0.9356 | 0.0734 | -0.056 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 723 | 2 | 656 | 18 | 0.1768 | 0.0847 | 0.0951 | 0.0391 | 0.009 | 0.1499 | 1.009 | 0.9357 | 0.0734 | -0.056 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 724 | 2 | 657 | 18 | 0.1768 | 0.0847 | 0.0952 | 0.0391 | 0.009 | 0.15 | 1.009 | 0.9354 | 0.0736 | -0.0561 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 725 | 2 | 658 | 18 | 0.1768 | 0.0847 | 0.095 | 0.0391 | 0.009 | 0.1501 | 1.009 | 0.9362 | 0.0728 | -0.0559 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 726 | 2 | 659 | 18 | 0.1768 | 0.0847 | 0.0949 | 0.0391 | 0.009 | 0.1502 | 1.009 | 0.9366 | 0.0724 | -0.0558 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 727 | 2 | 660 | 18 | 0.1768 | 0.0847 | 0.0947 | 0.0391 | 0.009 | 0.1503 | 1.009 | 0.9373 | 0.0717 | -0.0556 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 728 | 2 | 661 | 18 | 0.1768 | 0.0847 | 0.0946 | 0.0391 | 0.009 | 0.1504 | 1.009 | 0.9377 | 0.0713 | -0.0555 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 729 | 2 | 662 | 18 | 0.1768 | 0.0847 | 0.0945 | 0.0391 | 0.009 | 0.1505 | 1.009 | 0.938 | 0.071 | -0.0554 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 730 | 2 | 663 | 18 | 0.1768 | 0.0847 | 0.0942 | 0.0391 | 0.009 | 0.1506 | 1.009 | 0.939 | 0.07 | -0.0551 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 731 | 2 | 664 | 18 | 0.1768 | 0.0847 | 0.0945 | 0.0391 | 0.009 | 0.1507 | 1.009 | 0.9381 | 0.0708 | -0.0554 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 732 | 2 | 665 | 18 | 0.1768 | 0.0847 | 0.0946 | 0.0391 | 0.009 | 0.1508 | 1.009 | 0.938 | 0.071 | -0.0555 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 733 | 2 | 666 | 18 | 0.1768 | 0.0847 | 0.0949 | 0.0391 | 0.009 | 0.1509 | 1.009 | 0.9372 | 0.0718 | -0.0558 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 734 | 2 | 667 | 18 | 0.1768 | 0.0847 | 0.0945 | 0.0391 | 0.009 | 0.151 | 1.009 | 0.9385 | 0.0705 | -0.0554 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 735 | 2 | 668 | 18 | 0.1768 | 0.0847 | 0.0949 | 0.0391 | 0.009 | 0.1511 | 1.009 | 0.9376 | 0.0714 | -0.0558 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 736 | 2 | 669 | 18 | 0.1768 | 0.0847 | 0.0948 | 0.0391 | 0.0089 | 0.1512 | 1.0089 | 0.9378 | 0.0712 | -0.0557 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 737 | 2 | 670 | 18 | 0.1768 | 0.0847 | 0.0948 | 0.0391 | 0.0089 | 0.1513 | 1.0089 | 0.9379 | 0.0711 | -0.0557 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 738 | 2 | 671 | 18 | 0.1768 | 0.0847 | 0.0947 | 0.0391 | 0.0089 | 0.1514 | 1.0089 | 0.9383 | 0.0706 | -0.0556 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 739 | 2 | 672 | 18 | 0.1768 | 0.0847 | 0.095 | 0.0391 | 0.0089 | 0.1515 | 1.0089 | 0.9377 | 0.0712 | -0.0559 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 740 | 2 | 673 | 18 | 0.1768 | 0.0847 | 0.0952 | 0.0391 | 0.0089 | 0.1516 | 1.0089 | 0.9372 | 0.0717 | -0.0561 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 741 | 2 | 674 | 18 | 0.1768 | 0.0847 | 0.0954 | 0.0391 | 0.0089 | 0.1517 | 1.0089 | 0.9365 | 0.0724 | -0.0563 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 742 | 2 | 675 | 18 | 0.1768 | 0.0847 | 0.0953 | 0.0391 | 0.0089 | 0.1518 | 1.0089 | 0.937 | 0.072 | -0.0562 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 743 | 2 | 676 | 18 | 0.1768 | 0.0847 | 0.0951 | 0.0391 | 0.0089 | 0.1519 | 1.0089 | 0.9376 | 0.0713 | -0.056 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 744 | 2 | 677 | 18 | 0.1768 | 0.0847 | 0.0957 | 0.0391 | 0.0089 | 0.152 | 1.0089 | 0.9362 | 0.0727 | -0.0566 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 745 | 2 | 678 | 18 | 0.1768 | 0.0847 | 0.096 | 0.0391 | 0.0089 | 0.1521 | 1.0089 | 0.9352 | 0.0737 | -0.0569 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 746 | 2 | 679 | 18 | 0.1768 | 0.0847 | 0.0964 | 0.0391 | 0.0089 | 0.1522 | 1.0089 | 0.9342 | 0.0747 | -0.0573 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 747 | 2 | 680 | 18 | 0.1768 | 0.0847 | 0.0964 | 0.0391 | 0.0089 | 0.1523 | 1.0089 | 0.9343 | 0.0745 | -0.0573 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 748 | 2 | 681 | 18 | 0.1768 | 0.0847 | 0.0961 | 0.0391 | 0.0089 | 0.1524 | 1.0089 | 0.9352 | 0.0737 | -0.057 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 749 | 2 | 682 | 18 | 0.1768 | 0.0847 | 0.096 | 0.0391 | 0.0089 | 0.1525 | 1.0089 | 0.9357 | 0.0731 | -0.0569 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 750 | 2 | 683 | 18 | 0.1768 | 0.0847 | 0.0957 | 0.0391 | 0.0088 | 0.1526 | 1.0088 | 0.9365 | 0.0723 | -0.0566 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 751 | 2 | 684 | 18 | 0.1768 | 0.0847 | 0.0955 | 0.0391 | 0.0088 | 0.1527 | 1.0088 | 0.9373 | 0.0716 | -0.0564 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 752 | 2 | 685 | 18 | 0.1768 | 0.0847 | 0.0959 | 0.0391 | 0.0088 | 0.1528 | 1.0088 | 0.9363 | 0.0725 | -0.0568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 753 | 2 | 686 | 18 | 0.1768 | 0.0847 | 0.0961 | 0.0391 | 0.0088 | 0.1529 | 1.0088 | 0.9358 | 0.073 | -0.057 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 754 | 2 | 687 | 18 | 0.1768 | 0.0847 | 0.0966 | 0.0391 | 0.0088 | 0.153 | 1.0088 | 0.9343 | 0.0745 | -0.0575 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 755 | 2 | 688 | 18 | 0.1768 | 0.0847 | 0.0969 | 0.0391 | 0.0088 | 0.1531 | 1.0088 | 0.9334 | 0.0754 | -0.0578 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 756 | 2 | 689 | 18 | 0.1768 | 0.0847 | 0.097 | 0.0391 | 0.0088 | 0.1532 | 1.0088 | 0.9333 | 0.0755 | -0.0579 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 757 | 2 | 690 | 18 | 0.1768 | 0.0847 | 0.0974 | 0.0391 | 0.0088 | 0.1533 | 1.0088 | 0.9322 | 0.0766 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 758 | 2 | 691 | 18 | 0.1768 | 0.0847 | 0.0977 | 0.0391 | 0.0088 | 0.1534 | 1.0088 | 0.9316 | 0.0772 | -0.0586 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 759 | 2 | 692 | 18 | 0.1768 | 0.0847 | 0.0975 | 0.0391 | 0.0088 | 0.1535 | 1.0088 | 0.9323 | 0.0765 | -0.0584 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 760 | 2 | 693 | 18 | 0.1768 | 0.0847 | 0.0976 | 0.0391 | 0.0088 | 0.1536 | 1.0088 | 0.932 | 0.0768 | -0.0585 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 761 | 2 | 694 | 18 | 0.1768 | 0.0847 | 0.0974 | 0.0391 | 0.0088 | 0.1537 | 1.0088 | 0.9326 | 0.0762 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 762 | 2 | 695 | 18 | 0.1768 | 0.0847 | 0.0974 | 0.0391 | 0.0088 | 0.1538 | 1.0088 | 0.9327 | 0.0761 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 763 | 2 | 696 | 18 | 0.1768 | 0.0847 | 0.0974 | 0.0391 | 0.0088 | 0.1539 | 1.0088 | 0.9328 | 0.076 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 764 | 2 | 697 | 18 | 0.1768 | 0.0847 | 0.0977 | 0.0391 | 0.0088 | 0.154 | 1.0088 | 0.9321 | 0.0767 | -0.0586 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 765 | 2 | 698 | 18 | 0.1768 | 0.0847 | 0.0976 | 0.0391 | 0.0087 | 0.1541 | 1.0087 | 0.9325 | 0.0762 | -0.0585 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 766 | 2 | 699 | 18 | 0.1768 | 0.0847 | 0.0981 | 0.0391 | 0.0087 | 0.1542 | 1.0087 | 0.9312 | 0.0775 | -0.059 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 767 | 2 | 700 | 18 | 0.1768 | 0.0847 | 0.0982 | 0.0391 | 0.0087 | 0.1543 | 1.0087 | 0.9309 | 0.0778 | -0.0591 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 768 | 2 | 701 | 18 | 0.1768 | 0.0847 | 0.0984 | 0.0391 | 0.0087 | 0.1544 | 1.0087 | 0.9305 | 0.0782 | -0.0593 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 769 | 2 | 702 | 18 | 0.1768 | 0.0847 | 0.0988 | 0.0391 | 0.0087 | 0.1545 | 1.0087 | 0.9295 | 0.0792 | -0.0597 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 770 | 2 | 703 | 18 | 0.1768 | 0.0847 | 0.0989 | 0.0391 | 0.0087 | 0.1546 | 1.0087 | 0.9292 | 0.0795 | -0.0598 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 771 | 2 | 704 | 18 | 0.1768 | 0.0847 | 0.0992 | 0.0391 | 0.0087 | 0.1547 | 1.0087 | 0.9284 | 0.0803 | -0.0601 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 772 | 2 | 705 | 18 | 0.1768 | 0.0847 | 0.0995 | 0.0391 | 0.0087 | 0.1548 | 1.0087 | 0.9276 | 0.0811 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 773 | 2 | 706 | 18 | 0.1768 | 0.0847 | 0.0997 | 0.0391 | 0.0087 | 0.155 | 1.0087 | 0.9273 | 0.0814 | -0.0606 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 774 | 2 | 707 | 18 | 0.1768 | 0.0847 | 0.1002 | 0.0391 | 0.0087 | 0.1551 | 1.0087 | 0.9257 | 0.0829 | -0.0611 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 775 | 2 | 708 | 18 | 0.1768 | 0.0847 | 0.1006 | 0.0391 | 0.0087 | 0.1552 | 1.0087 | 0.9247 | 0.084 | -0.0615 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 776 | 2 | 709 | 18 | 0.1768 | 0.0847 | 0.1008 | 0.0391 | 0.0087 | 0.1553 | 1.0087 | 0.9243 | 0.0844 | -0.0617 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 777 | 2 | 710 | 18 | 0.1768 | 0.0847 | 0.1008 | 0.0391 | 0.0087 | 0.1554 | 1.0087 | 0.9245 | 0.0842 | -0.0617 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 778 | 2 | 711 | 18 | 0.1768 | 0.0847 | 0.1007 | 0.0391 | 0.0087 | 0.1555 | 1.0087 | 0.9247 | 0.0839 | -0.0616 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 779 | 2 | 712 | 18 | 0.1768 | 0.0847 | 0.101 | 0.0391 | 0.0087 | 0.1556 | 1.0087 | 0.9241 | 0.0845 | -0.0619 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 780 | 2 | 713 | 18 | 0.1768 | 0.0847 | 0.1007 | 0.0391 | 0.0086 | 0.1557 | 1.0086 | 0.925 | 0.0837 | -0.0616 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 781 | 2 | 714 | 18 | 0.1768 | 0.0847 | 0.101 | 0.0391 | 0.0086 | 0.1557 | 1.0086 | 0.9242 | 0.0845 | -0.0619 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 782 | 2 | 715 | 18 | 0.1768 | 0.0847 | 0.1011 | 0.0391 | 0.0086 | 0.1558 | 1.0086 | 0.9241 | 0.0846 | -0.062 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 783 | 2 | 716 | 18 | 0.1768 | 0.0847 | 0.101 | 0.0391 | 0.0086 | 0.1559 | 1.0086 | 0.9245 | 0.0841 | -0.0619 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 784 | 2 | 717 | 18 | 0.1768 | 0.0847 | 0.1009 | 0.0391 | 0.0086 | 0.156 | 1.0086 | 0.9248 | 0.0838 | -0.0618 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 785 | 2 | 718 | 18 | 0.1768 | 0.0847 | 0.1011 | 0.0391 | 0.0086 | 0.1561 | 1.0086 | 0.9244 | 0.0843 | -0.062 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 786 | 2 | 719 | 18 | 0.1768 | 0.0847 | 0.1014 | 0.0391 | 0.0086 | 0.1562 | 1.0086 | 0.9237 | 0.0849 | -0.0623 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 787 | 2 | 720 | 18 | 0.1768 | 0.0847 | 0.1016 | 0.0391 | 0.0086 | 0.1563 | 1.0086 | 0.923 | 0.0856 | -0.0625 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 788 | 2 | 721 | 18 | 0.1768 | 0.0847 | 0.1022 | 0.0391 | 0.0086 | 0.1564 | 1.0086 | 0.9215 | 0.0871 | -0.0631 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 789 | 2 | 722 | 18 | 0.1768 | 0.0847 | 0.1019 | 0.0391 | 0.0086 | 0.1565 | 1.0086 | 0.9224 | 0.0861 | -0.0628 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 790 | 2 | 723 | 18 | 0.1768 | 0.0847 | 0.1022 | 0.0391 | 0.0086 | 0.1566 | 1.0086 | 0.9216 | 0.087 | -0.0631 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 791 | 2 | 724 | 18 | 0.1768 | 0.0847 | 0.1022 | 0.0391 | 0.0086 | 0.1567 | 1.0086 | 0.9218 | 0.0868 | -0.0631 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 792 | 2 | 725 | 18 | 0.1768 | 0.0847 | 0.1024 | 0.0391 | 0.0086 | 0.1568 | 1.0086 | 0.9212 | 0.0874 | -0.0633 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 793 | 2 | 726 | 18 | 0.1768 | 0.0847 | 0.1028 | 0.0391 | 0.0086 | 0.1569 | 1.0086 | 0.9204 | 0.0882 | -0.0637 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 794 | 2 | 727 | 18 | 0.1768 | 0.0847 | 0.1031 | 0.0391 | 0.0086 | 0.157 | 1.0086 | 0.9196 | 0.089 | -0.064 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 795 | 2 | 728 | 18 | 0.1768 | 0.0847 | 0.1031 | 0.0391 | 0.0085 | 0.1571 | 1.0085 | 0.9196 | 0.089 | -0.064 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 796 | 2 | 729 | 18 | 0.1768 | 0.0847 | 0.1034 | 0.0391 | 0.0085 | 0.1572 | 1.0085 | 0.9188 | 0.0897 | -0.0643 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 797 | 2 | 730 | 18 | 0.1768 | 0.0847 | 0.1036 | 0.0391 | 0.0085 | 0.1573 | 1.0085 | 0.9185 | 0.09 | -0.0645 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 798 | 2 | 731 | 18 | 0.1768 | 0.0847 | 0.1036 | 0.0391 | 0.0085 | 0.1574 | 1.0085 | 0.9185 | 0.09 | -0.0645 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 799 | 2 | 732 | 18 | 0.1768 | 0.0847 | 0.1038 | 0.0391 | 0.0085 | 0.1575 | 1.0085 | 0.9181 | 0.0904 | -0.0647 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 800 | 2 | 733 | 18 | 0.1768 | 0.0847 | 0.1041 | 0.0391 | 0.0085 | 0.1576 | 1.0085 | 0.9172 | 0.0913 | -0.065 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 801 | 2 | 734 | 18 | 0.1768 | 0.0847 | 0.1042 | 0.0391 | 0.0085 | 0.1577 | 1.0085 | 0.9171 | 0.0914 | -0.0651 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 802 | 2 | 735 | 18 | 0.1768 | 0.0847 | 0.1047 | 0.0391 | 0.0085 | 0.1578 | 1.0085 | 0.9159 | 0.0927 | -0.0656 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 803 | 2 | 736 | 18 | 0.1768 | 0.0847 | 0.1046 | 0.0391 | 0.0085 | 0.1579 | 1.0085 | 0.9161 | 0.0924 | -0.0655 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 804 | 2 | 737 | 18 | 0.1768 | 0.0847 | 0.1046 | 0.0391 | 0.0085 | 0.158 | 1.0085 | 0.9163 | 0.0921 | -0.0655 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 805 | 2 | 738 | 18 | 0.1768 | 0.0847 | 0.1048 | 0.0391 | 0.0085 | 0.1581 | 1.0085 | 0.9157 | 0.0928 | -0.0657 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 806 | 2 | 739 | 18 | 0.1768 | 0.0847 | 0.1048 | 0.0391 | 0.0085 | 0.1582 | 1.0085 | 0.9159 | 0.0926 | -0.0657 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 807 | 2 | 740 | 18 | 0.1768 | 0.0847 | 0.1047 | 0.0391 | 0.0085 | 0.1583 | 1.0085 | 0.9163 | 0.0922 | -0.0656 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 808 | 2 | 741 | 18 | 0.1768 | 0.0847 | 0.1051 | 0.0391 | 0.0085 | 0.1584 | 1.0085 | 0.9153 | 0.0932 | -0.066 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 809 | 2 | 742 | 18 | 0.1768 | 0.0847 | 0.105 | 0.0391 | 0.0085 | 0.1585 | 1.0085 | 0.9155 | 0.0929 | -0.0659 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 810 | 2 | 743 | 18 | 0.1768 | 0.0847 | 0.105 | 0.0391 | 0.0085 | 0.1586 | 1.0085 | 0.9157 | 0.0928 | -0.0659 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 811 | 2 | 744 | 18 | 0.1768 | 0.0847 | 0.1054 | 0.0391 | 0.0084 | 0.1587 | 1.0084 | 0.9146 | 0.0938 | -0.0663 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 812 | 2 | 745 | 18 | 0.1768 | 0.0847 | 0.1056 | 0.0391 | 0.0084 | 0.1588 | 1.0084 | 0.9143 | 0.0941 | -0.0665 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 813 | 2 | 746 | 18 | 0.1768 | 0.0847 | 0.106 | 0.0391 | 0.0084 | 0.1589 | 1.0084 | 0.9132 | 0.0953 | -0.0669 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 814 | 2 | 747 | 18 | 0.1768 | 0.0847 | 0.106 | 0.0391 | 0.0084 | 0.159 | 1.0084 | 0.9134 | 0.095 | -0.0669 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 815 | 2 | 748 | 18 | 0.1768 | 0.0847 | 0.1059 | 0.0391 | 0.0084 | 0.1591 | 1.0084 | 0.9136 | 0.0948 | -0.0668 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 816 | 2 | 749 | 18 | 0.1768 | 0.0847 | 0.1063 | 0.0391 | 0.0084 | 0.1592 | 1.0084 | 0.9127 | 0.0957 | -0.0672 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 817 | 2 | 750 | 18 | 0.1768 | 0.0847 | 0.1066 | 0.0391 | 0.0084 | 0.1593 | 1.0084 | 0.912 | 0.0964 | -0.0675 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 818 | 2 | 751 | 18 | 0.1768 | 0.0847 | 0.1072 | 0.0391 | 0.0084 | 0.1594 | 1.0084 | 0.9105 | 0.0979 | -0.0681 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 819 | 2 | 752 | 18 | 0.1768 | 0.0847 | 0.1075 | 0.0391 | 0.0084 | 0.1595 | 1.0084 | 0.9098 | 0.0986 | -0.0684 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 820 | 2 | 753 | 18 | 0.1768 | 0.0847 | 0.1076 | 0.0391 | 0.0084 | 0.1596 | 1.0084 | 0.9094 | 0.099 | -0.0685 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 821 | 2 | 754 | 18 | 0.1768 | 0.0847 | 0.1079 | 0.0391 | 0.0084 | 0.1597 | 1.0084 | 0.9088 | 0.0996 | -0.0688 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 822 | 2 | 755 | 18 | 0.1768 | 0.0847 | 0.108 | 0.0391 | 0.0084 | 0.1598 | 1.0084 | 0.9086 | 0.0998 | -0.0689 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 823 | 2 | 756 | 18 | 0.1768 | 0.0847 | 0.1085 | 0.0391 | 0.0084 | 0.1599 | 1.0084 | 0.9073 | 0.1011 | -0.0694 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 824 | 2 | 757 | 18 | 0.1768 | 0.0847 | 0.1088 | 0.0391 | 0.0084 | 0.16 | 1.0084 | 0.9066 | 0.1018 | -0.0697 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 825 | 2 | 758 | 18 | 0.1768 | 0.0847 | 0.1091 | 0.0391 | 0.0084 | 0.1601 | 1.0084 | 0.9059 | 0.1024 | -0.07 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 826 | 2 | 759 | 18 | 0.1768 | 0.0847 | 0.1088 | 0.0391 | 0.0084 | 0.1602 | 1.0084 | 0.9068 | 0.1016 | -0.0697 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 827 | 2 | 760 | 18 | 0.1768 | 0.0847 | 0.1086 | 0.0391 | 0.0084 | 0.1603 | 1.0084 | 0.9074 | 0.1009 | -0.0695 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 828 | 2 | 761 | 18 | 0.1768 | 0.0847 | 0.1084 | 0.0391 | 0.0083 | 0.1604 | 1.0083 | 0.908 | 0.1003 | -0.0693 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 829 | 2 | 762 | 18 | 0.1768 | 0.0847 | 0.1083 | 0.0391 | 0.0083 | 0.1605 | 1.0083 | 0.9085 | 0.0999 | -0.0692 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 830 | 2 | 763 | 18 | 0.1768 | 0.0847 | 0.108 | 0.0391 | 0.0083 | 0.1606 | 1.0083 | 0.9093 | 0.099 | -0.0689 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 831 | 2 | 764 | 18 | 0.1768 | 0.0847 | 0.1077 | 0.0391 | 0.0083 | 0.1607 | 1.0083 | 0.9104 | 0.098 | -0.0686 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 832 | 2 | 765 | 18 | 0.1768 | 0.0847 | 0.1081 | 0.0391 | 0.0083 | 0.1608 | 1.0083 | 0.9094 | 0.0989 | -0.069 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 833 | 2 | 766 | 18 | 0.1768 | 0.0847 | 0.1079 | 0.0391 | 0.0083 | 0.1609 | 1.0083 | 0.9099 | 0.0984 | -0.0688 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 834 | 2 | 767 | 18 | 0.1768 | 0.0847 | 0.1078 | 0.0391 | 0.0083 | 0.1609 | 1.0083 | 0.9103 | 0.098 | -0.0687 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 835 | 2 | 768 | 18 | 0.1768 | 0.0847 | 0.1078 | 0.0391 | 0.0083 | 0.161 | 1.0083 | 0.9103 | 0.098 | -0.0687 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 836 | 2 | 769 | 18 | 0.1768 | 0.0847 | 0.1082 | 0.0391 | 0.0083 | 0.1611 | 1.0083 | 0.9094 | 0.0989 | -0.0691 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 837 | 2 | 770 | 18 | 0.1768 | 0.0847 | 0.1083 | 0.0391 | 0.0083 | 0.1612 | 1.0083 | 0.9092 | 0.0991 | -0.0692 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 838 | 2 | 771 | 18 | 0.1768 | 0.0847 | 0.1083 | 0.0391 | 0.0083 | 0.1613 | 1.0083 | 0.9094 | 0.0989 | -0.0692 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 839 | 2 | 772 | 18 | 0.1768 | 0.0847 | 0.1085 | 0.0391 | 0.0083 | 0.1614 | 1.0083 | 0.9089 | 0.0993 | -0.0694 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 840 | 2 | 773 | 18 | 0.1768 | 0.0847 | 0.1088 | 0.0391 | 0.0083 | 0.1615 | 1.0083 | 0.9081 | 0.1001 | -0.0697 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 841 | 2 | 774 | 18 | 0.1768 | 0.0847 | 0.1089 | 0.0391 | 0.0083 | 0.1616 | 1.0083 | 0.908 | 0.1003 | -0.0698 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 842 | 2 | 775 | 18 | 0.1768 | 0.0847 | 0.1089 | 0.0391 | 0.0083 | 0.1617 | 1.0083 | 0.9081 | 0.1002 | -0.0698 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 843 | 2 | 776 | 18 | 0.1768 | 0.0847 | 0.109 | 0.0391 | 0.0083 | 0.1618 | 1.0083 | 0.9079 | 0.1004 | -0.0699 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 844 | 2 | 777 | 18 | 0.1768 | 0.0847 | 0.1094 | 0.0391 | 0.0083 | 0.1619 | 1.0083 | 0.9069 | 0.1013 | -0.0703 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 845 | 2 | 778 | 18 | 0.1768 | 0.0847 | 0.1093 | 0.0391 | 0.0082 | 0.162 | 1.0082 | 0.9073 | 0.1009 | -0.0702 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 846 | 2 | 779 | 18 | 0.1768 | 0.0847 | 0.1093 | 0.0391 | 0.0082 | 0.1621 | 1.0082 | 0.9075 | 0.1008 | -0.0702 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 847 | 2 | 780 | 18 | 0.1768 | 0.0847 | 0.1094 | 0.0391 | 0.0082 | 0.1622 | 1.0082 | 0.9072 | 0.101 | -0.0703 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 848 | 2 | 781 | 18 | 0.1768 | 0.0847 | 0.1098 | 0.0391 | 0.0082 | 0.1623 | 1.0082 | 0.9063 | 0.102 | -0.0707 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 849 | 2 | 782 | 18 | 0.1768 | 0.0847 | 0.1102 | 0.0391 | 0.0082 | 0.1624 | 1.0082 | 0.9052 | 0.1031 | -0.0711 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 850 | 2 | 783 | 18 | 0.1768 | 0.0847 | 0.1106 | 0.0391 | 0.0082 | 0.1625 | 1.0082 | 0.9044 | 0.1038 | -0.0715 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 851 | 2 | 784 | 18 | 0.1768 | 0.0847 | 0.1102 | 0.0391 | 0.0082 | 0.1626 | 1.0082 | 0.9054 | 0.1029 | -0.0711 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 852 | 2 | 785 | 18 | 0.1768 | 0.0847 | 0.11 | 0.0391 | 0.0082 | 0.1627 | 1.0082 | 0.906 | 0.1022 | -0.0709 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 853 | 2 | 786 | 18 | 0.1768 | 0.0847 | 0.11 | 0.0391 | 0.0082 | 0.1628 | 1.0082 | 0.9063 | 0.1019 | -0.0709 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 854 | 2 | 787 | 18 | 0.1768 | 0.0847 | 0.11 | 0.0391 | 0.0082 | 0.1629 | 1.0082 | 0.9062 | 0.102 | -0.0709 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 855 | 2 | 788 | 18 | 0.1768 | 0.0847 | 0.1102 | 0.0391 | 0.0082 | 0.163 | 1.0082 | 0.9059 | 0.1023 | -0.0711 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 856 | 2 | 789 | 18 | 0.1768 | 0.0847 | 0.11 | 0.0391 | 0.0082 | 0.1631 | 1.0082 | 0.9065 | 0.1017 | -0.0709 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 857 | 2 | 790 | 18 | 0.1768 | 0.0847 | 0.1097 | 0.0391 | 0.0082 | 0.1632 | 1.0082 | 0.9073 | 0.1009 | -0.0706 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 858 | 2 | 791 | 18 | 0.1768 | 0.0847 | 0.1095 | 0.0391 | 0.0082 | 0.1632 | 1.0082 | 0.9081 | 0.1001 | -0.0704 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 859 | 2 | 792 | 18 | 0.1768 | 0.0847 | 0.1091 | 0.0391 | 0.0082 | 0.1633 | 1.0082 | 0.9092 | 0.099 | -0.07 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 860 | 2 | 793 | 18 | 0.1768 | 0.0847 | 0.1087 | 0.0391 | 0.0082 | 0.1634 | 1.0082 | 0.9103 | 0.0978 | -0.0696 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 861 | 2 | 794 | 18 | 0.1768 | 0.0847 | 0.1085 | 0.0391 | 0.0082 | 0.1635 | 1.0082 | 0.9111 | 0.0971 | -0.0694 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 862 | 2 | 795 | 18 | 0.1768 | 0.0847 | 0.1081 | 0.0391 | 0.0082 | 0.1636 | 1.0082 | 0.9121 | 0.0961 | -0.069 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 863 | 2 | 796 | 18 | 0.1768 | 0.0847 | 0.1082 | 0.0391 | 0.0081 | 0.1637 | 1.0081 | 0.9121 | 0.096 | -0.0691 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 864 | 2 | 797 | 18 | 0.1768 | 0.0847 | 0.108 | 0.0391 | 0.0081 | 0.1638 | 1.0081 | 0.9128 | 0.0954 | -0.0689 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 865 | 2 | 798 | 18 | 0.1768 | 0.0847 | 0.1077 | 0.0391 | 0.0081 | 0.1639 | 1.0081 | 0.9136 | 0.0945 | -0.0686 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 866 | 2 | 799 | 18 | 0.1768 | 0.0847 | 0.1076 | 0.0391 | 0.0081 | 0.164 | 1.0081 | 0.9139 | 0.0943 | -0.0685 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 867 | 2 | 800 | 18 | 0.1768 | 0.0847 | 0.1076 | 0.0391 | 0.0081 | 0.1641 | 1.0081 | 0.9141 | 0.094 | -0.0685 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 868 | 2 | 800 | 18 | 0.1768 | 0.0847 | 0.1076 | 0.0391 | 0.0081 | 0.1642 | 1.0081 | 0.9142 | 0.0939 | -0.0685 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 869 | 2 | 801 | 18 | 0.1768 | 0.0847 | 0.1076 | 0.0391 | 0.0081 | 0.1643 | 1.0081 | 0.9141 | 0.094 | -0.0685 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 870 | 2 | 802 | 18 | 0.1768 | 0.0847 | 0.1075 | 0.0391 | 0.0081 | 0.1644 | 1.0081 | 0.9145 | 0.0936 | -0.0684 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 871 | 2 | 803 | 18 | 0.1768 | 0.0847 | 0.1072 | 0.0391 | 0.0081 | 0.1645 | 1.0081 | 0.9156 | 0.0925 | -0.0681 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 872 | 2 | 804 | 18 | 0.1768 | 0.0847 | 0.1072 | 0.0391 | 0.0081 | 0.1646 | 1.0081 | 0.9157 | 0.0924 | -0.0681 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 873 | 2 | 805 | 18 | 0.1768 | 0.0847 | 0.1071 | 0.0391 | 0.0081 | 0.1647 | 1.0081 | 0.916 | 0.0921 | -0.068 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 874 | 2 | 806 | 18 | 0.1768 | 0.0847 | 0.1071 | 0.0391 | 0.0081 | 0.1648 | 1.0081 | 0.916 | 0.0921 | -0.068 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 875 | 2 | 807 | 18 | 0.1768 | 0.0847 | 0.1073 | 0.0391 | 0.0081 | 0.1649 | 1.0081 | 0.9157 | 0.0924 | -0.0682 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 876 | 2 | 808 | 18 | 0.1768 | 0.0847 | 0.1072 | 0.0391 | 0.0081 | 0.165 | 1.0081 | 0.916 | 0.0921 | -0.0681 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 877 | 2 | 809 | 18 | 0.1768 | 0.0847 | 0.1076 | 0.0391 | 0.0081 | 0.165 | 1.0081 | 0.915 | 0.0931 | -0.0685 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 878 | 2 | 810 | 18 | 0.1768 | 0.0847 | 0.1074 | 0.0391 | 0.0081 | 0.1651 | 1.0081 | 0.9155 | 0.0926 | -0.0683 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 879 | 2 | 811 | 18 | 0.1768 | 0.0847 | 0.1074 | 0.0391 | 0.0081 | 0.1652 | 1.0081 | 0.9156 | 0.0924 | -0.0683 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 880 | 2 | 812 | 18 | 0.1768 | 0.0847 | 0.1076 | 0.0391 | 0.0081 | 0.1653 | 1.0081 | 0.9152 | 0.0929 | -0.0685 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 881 | 2 | 813 | 18 | 0.1768 | 0.0847 | 0.108 | 0.0391 | 0.0081 | 0.1654 | 1.0081 | 0.9142 | 0.0938 | -0.0689 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 882 | 2 | 814 | 18 | 0.1768 | 0.0847 | 0.1081 | 0.0391 | 0.0081 | 0.1655 | 1.0081 | 0.9141 | 0.0939 | -0.069 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 883 | 2 | 815 | 18 | 0.1768 | 0.0847 | 0.1081 | 0.0391 | 0.008 | 0.1656 | 1.008 | 0.9142 | 0.0939 | -0.069 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 884 | 2 | 816 | 18 | 0.1768 | 0.0847 | 0.1078 | 0.0391 | 0.008 | 0.1657 | 1.008 | 0.915 | 0.093 | -0.0687 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 885 | 2 | 817 | 18 | 0.1768 | 0.0847 | 0.1077 | 0.0391 | 0.008 | 0.1658 | 1.008 | 0.9154 | 0.0927 | -0.0686 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 886 | 2 | 818 | 18 | 0.1768 | 0.0847 | 0.1076 | 0.0391 | 0.008 | 0.1659 | 1.008 | 0.9158 | 0.0923 | -0.0685 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 887 | 2 | 819 | 18 | 0.1768 | 0.0847 | 0.1075 | 0.0391 | 0.008 | 0.166 | 1.008 | 0.9162 | 0.0918 | -0.0684 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 888 | 2 | 820 | 18 | 0.1768 | 0.0847 | 0.1075 | 0.0391 | 0.008 | 0.1661 | 1.008 | 0.9162 | 0.0918 | -0.0684 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 889 | 2 | 821 | 18 | 0.1768 | 0.0847 | 0.1074 | 0.0391 | 0.008 | 0.1662 | 1.008 | 0.9167 | 0.0914 | -0.0683 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 890 | 2 | 822 | 18 | 0.1768 | 0.0847 | 0.1073 | 0.0391 | 0.008 | 0.1663 | 1.008 | 0.917 | 0.0911 | -0.0682 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 891 | 2 | 823 | 18 | 0.1768 | 0.0847 | 0.1067 | 0.0391 | 0.008 | 0.1664 | 1.008 | 0.9186 | 0.0894 | -0.0676 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 892 | 2 | 824 | 18 | 0.1768 | 0.0847 | 0.1067 | 0.0391 | 0.008 | 0.1665 | 1.008 | 0.9187 | 0.0893 | -0.0676 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 893 | 2 | 825 | 18 | 0.1768 | 0.0847 | 0.1067 | 0.0391 | 0.008 | 0.1665 | 1.008 | 0.9189 | 0.089 | -0.0676 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 894 | 2 | 825 | 18 | 0.1768 | 0.0847 | 0.1067 | 0.0391 | 0.008 | 0.1666 | 1.008 | 0.919 | 0.089 | -0.0676 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 895 | 2 | 826 | 18 | 0.1768 | 0.0847 | 0.1066 | 0.0391 | 0.008 | 0.1667 | 1.008 | 0.9195 | 0.0885 | -0.0675 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 896 | 2 | 827 | 18 | 0.1768 | 0.0847 | 0.1065 | 0.0391 | 0.008 | 0.1668 | 1.008 | 0.9198 | 0.0882 | -0.0674 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 897 | 2 | 828 | 18 | 0.1768 | 0.0847 | 0.1067 | 0.0391 | 0.008 | 0.1669 | 1.008 | 0.9193 | 0.0887 | -0.0676 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 898 | 2 | 829 | 18 | 0.1768 | 0.0847 | 0.1066 | 0.0391 | 0.008 | 0.167 | 1.008 | 0.9198 | 0.0882 | -0.0675 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 899 | 2 | 830 | 18 | 0.1768 | 0.0847 | 0.1064 | 0.0391 | 0.008 | 0.1671 | 1.008 | 0.9202 | 0.0878 | -0.0673 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 900 | 2 | 831 | 18 | 0.1768 | 0.0847 | 0.1063 | 0.0391 | 0.008 | 0.1672 | 1.008 | 0.9207 | 0.0872 | -0.0672 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 901 | 2 | 832 | 18 | 0.1768 | 0.0847 | 0.1063 | 0.0391 | 0.008 | 0.1673 | 1.008 | 0.9209 | 0.0871 | -0.0672 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 902 | 2 | 833 | 18 | 0.1768 | 0.0847 | 0.106 | 0.0391 | 0.008 | 0.1674 | 1.008 | 0.9219 | 0.0861 | -0.0669 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 903 | 2 | 834 | 18 | 0.1768 | 0.0847 | 0.1056 | 0.0391 | 0.008 | 0.1675 | 1.008 | 0.9229 | 0.0851 | -0.0665 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 904 | 2 | 835 | 18 | 0.1768 | 0.0847 | 0.1053 | 0.0391 | 0.0079 | 0.1676 | 1.0079 | 0.924 | 0.084 | -0.0662 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 905 | 2 | 836 | 18 | 0.1768 | 0.0847 | 0.105 | 0.0391 | 0.0079 | 0.1677 | 1.0079 | 0.9248 | 0.0832 | -0.0659 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 906 | 2 | 837 | 18 | 0.1768 | 0.0847 | 0.1047 | 0.0391 | 0.0079 | 0.1678 | 1.0079 | 0.9257 | 0.0822 | -0.0656 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 907 | 2 | 838 | 18 | 0.1768 | 0.0847 | 0.1047 | 0.0391 | 0.0079 | 0.1678 | 1.0079 | 0.9259 | 0.082 | -0.0656 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 908 | 2 | 839 | 18 | 0.1768 | 0.0847 | 0.1046 | 0.0391 | 0.0079 | 0.1679 | 1.0079 | 0.9262 | 0.0817 | -0.0655 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 909 | 2 | 840 | 18 | 0.1768 | 0.0847 | 0.1046 | 0.0391 | 0.0079 | 0.168 | 1.0079 | 0.9263 | 0.0816 | -0.0655 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 910 | 2 | 841 | 18 | 0.1768 | 0.0847 | 0.1048 | 0.0391 | 0.0079 | 0.1681 | 1.0079 | 0.9257 | 0.0823 | -0.0658 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 911 | 2 | 842 | 18 | 0.1768 | 0.0847 | 0.1047 | 0.0391 | 0.0079 | 0.1682 | 1.0079 | 0.9263 | 0.0817 | -0.0656 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 912 | 2 | 843 | 18 | 0.1768 | 0.0847 | 0.1047 | 0.0391 | 0.0079 | 0.1683 | 1.0079 | 0.9263 | 0.0816 | -0.0656 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 913 | 2 | 844 | 18 | 0.1768 | 0.0847 | 0.1048 | 0.0391 | 0.0079 | 0.1684 | 1.0079 | 0.926 | 0.0819 | -0.0657 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 914 | 2 | 845 | 18 | 0.1768 | 0.0847 | 0.1049 | 0.0391 | 0.0079 | 0.1685 | 1.0079 | 0.9258 | 0.0821 | -0.0658 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 915 | 2 | 846 | 18 | 0.1768 | 0.0847 | 0.1052 | 0.0391 | 0.0079 | 0.1686 | 1.0079 | 0.9253 | 0.0826 | -0.0661 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 916 | 2 | 847 | 18 | 0.1768 | 0.0847 | 0.1054 | 0.0391 | 0.0079 | 0.1687 | 1.0079 | 0.9248 | 0.0831 | -0.0663 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 917 | 2 | 848 | 18 | 0.1768 | 0.0847 | 0.1058 | 0.0391 | 0.0079 | 0.1688 | 1.0079 | 0.9237 | 0.0842 | -0.0667 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 918 | 2 | 849 | 18 | 0.1768 | 0.0847 | 0.1058 | 0.0391 | 0.0079 | 0.1689 | 1.0079 | 0.9237 | 0.0842 | -0.0667 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 919 | 2 | 850 | 18 | 0.1768 | 0.0847 | 0.1063 | 0.0391 | 0.0079 | 0.169 | 1.0079 | 0.9226 | 0.0853 | -0.0672 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 920 | 2 | 851 | 18 | 0.1768 | 0.0847 | 0.1062 | 0.0391 | 0.0079 | 0.169 | 1.0079 | 0.9228 | 0.0851 | -0.0671 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 921 | 2 | 852 | 18 | 0.1768 | 0.0847 | 0.106 | 0.0391 | 0.0079 | 0.1691 | 1.0079 | 0.9234 | 0.0844 | -0.0669 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 922 | 2 | 853 | 18 | 0.1768 | 0.0847 | 0.1061 | 0.0391 | 0.0079 | 0.1692 | 1.0079 | 0.9233 | 0.0845 | -0.067 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 923 | 2 | 854 | 18 | 0.1768 | 0.0847 | 0.1061 | 0.0391 | 0.0079 | 0.1693 | 1.0079 | 0.9234 | 0.0845 | -0.067 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 924 | 2 | 855 | 18 | 0.1768 | 0.0847 | 0.1062 | 0.0391 | 0.0078 | 0.1694 | 1.0078 | 0.9231 | 0.0847 | -0.0671 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 925 | 2 | 856 | 18 | 0.1768 | 0.0847 | 0.1064 | 0.0391 | 0.0078 | 0.1695 | 1.0078 | 0.9227 | 0.0851 | -0.0673 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 926 | 2 | 857 | 18 | 0.1768 | 0.0847 | 0.1065 | 0.0391 | 0.0078 | 0.1696 | 1.0078 | 0.9225 | 0.0854 | -0.0674 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 927 | 2 | 858 | 18 | 0.1768 | 0.0847 | 0.1063 | 0.0391 | 0.0078 | 0.1697 | 1.0078 | 0.9231 | 0.0848 | -0.0672 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 928 | 2 | 859 | 18 | 0.1768 | 0.0847 | 0.1064 | 0.0391 | 0.0078 | 0.1698 | 1.0078 | 0.9231 | 0.0847 | -0.0673 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 929 | 2 | 860 | 18 | 0.1768 | 0.0847 | 0.1062 | 0.0391 | 0.0078 | 0.1699 | 1.0078 | 0.9237 | 0.0841 | -0.0671 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 930 | 2 | 861 | 18 | 0.1768 | 0.0847 | 0.1061 | 0.0391 | 0.0078 | 0.17 | 1.0078 | 0.924 | 0.0838 | -0.067 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 931 | 2 | 862 | 18 | 0.1768 | 0.0847 | 0.1059 | 0.0391 | 0.0078 | 0.1701 | 1.0078 | 0.9246 | 0.0832 | -0.0668 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 932 | 2 | 863 | 18 | 0.1768 | 0.0847 | 0.1058 | 0.0391 | 0.0078 | 0.1701 | 1.0078 | 0.9251 | 0.0827 | -0.0667 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 933 | 2 | 864 | 18 | 0.1768 | 0.0847 | 0.1056 | 0.0391 | 0.0078 | 0.1702 | 1.0078 | 0.9256 | 0.0822 | -0.0665 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 934 | 2 | 865 | 18 | 0.1768 | 0.0847 | 0.1054 | 0.0391 | 0.0078 | 0.1703 | 1.0078 | 0.9263 | 0.0815 | -0.0663 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 935 | 2 | 866 | 18 | 0.1768 | 0.0847 | 0.1056 | 0.0391 | 0.0078 | 0.1704 | 1.0078 | 0.9257 | 0.082 | -0.0665 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 936 | 2 | 867 | 18 | 0.1768 | 0.0847 | 0.1055 | 0.0391 | 0.0078 | 0.1705 | 1.0078 | 0.9263 | 0.0815 | -0.0664 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 937 | 2 | 868 | 18 | 0.1768 | 0.0847 | 0.1054 | 0.0391 | 0.0078 | 0.1706 | 1.0078 | 0.9267 | 0.0811 | -0.0663 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 938 | 2 | 869 | 18 | 0.1768 | 0.0847 | 0.1052 | 0.0391 | 0.0078 | 0.1707 | 1.0078 | 0.9272 | 0.0806 | -0.0661 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 939 | 2 | 870 | 18 | 0.1768 | 0.0847 | 0.1052 | 0.0391 | 0.0078 | 0.1708 | 1.0078 | 0.9273 | 0.0804 | -0.0661 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 940 | 2 | 871 | 18 | 0.1768 | 0.0847 | 0.1051 | 0.0391 | 0.0078 | 0.1709 | 1.0078 | 0.9277 | 0.08 | -0.066 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 941 | 2 | 872 | 18 | 0.1768 | 0.0847 | 0.1051 | 0.0391 | 0.0078 | 0.171 | 1.0078 | 0.9277 | 0.0801 | -0.066 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 942 | 2 | 873 | 18 | 0.1768 | 0.0847 | 0.1052 | 0.0391 | 0.0078 | 0.1711 | 1.0078 | 0.9277 | 0.08 | -0.0661 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 943 | 2 | 873 | 18 | 0.1768 | 0.0847 | 0.1052 | 0.0391 | 0.0078 | 0.1711 | 1.0078 | 0.9278 | 0.08 | -0.0661 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 944 | 2 | 874 | 18 | 0.1768 | 0.0847 | 0.1051 | 0.0391 | 0.0078 | 0.1712 | 1.0078 | 0.928 | 0.0797 | -0.066 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 945 | 2 | 875 | 18 | 0.1768 | 0.0847 | 0.1049 | 0.0391 | 0.0078 | 0.1713 | 1.0078 | 0.9287 | 0.0791 | -0.0658 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 946 | 2 | 876 | 18 | 0.1768 | 0.0847 | 0.1049 | 0.0391 | 0.0078 | 0.1714 | 1.0078 | 0.9288 | 0.0789 | -0.0658 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 947 | 2 | 877 | 18 | 0.1768 | 0.0847 | 0.1046 | 0.0391 | 0.0077 | 0.1715 | 1.0077 | 0.9297 | 0.0781 | -0.0655 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 948 | 2 | 878 | 18 | 0.1768 | 0.0847 | 0.1043 | 0.0391 | 0.0077 | 0.1716 | 1.0077 | 0.9306 | 0.0771 | -0.0652 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 949 | 2 | 879 | 18 | 0.1768 | 0.0847 | 0.1044 | 0.0391 | 0.0077 | 0.1717 | 1.0077 | 0.9306 | 0.0771 | -0.0653 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 950 | 2 | 880 | 18 | 0.1768 | 0.0847 | 0.1043 | 0.0391 | 0.0077 | 0.1718 | 1.0077 | 0.9309 | 0.0768 | -0.0652 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 951 | 2 | 881 | 18 | 0.1768 | 0.0847 | 0.1041 | 0.0391 | 0.0077 | 0.1719 | 1.0077 | 0.9314 | 0.0763 | -0.065 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 952 | 2 | 882 | 18 | 0.1768 | 0.0847 | 0.104 | 0.0391 | 0.0077 | 0.172 | 1.0077 | 0.9317 | 0.076 | -0.065 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 953 | 2 | 883 | 18 | 0.1768 | 0.0847 | 0.104 | 0.0391 | 0.0077 | 0.172 | 1.0077 | 0.932 | 0.0757 | -0.0649 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 954 | 2 | 884 | 18 | 0.1768 | 0.0847 | 0.1041 | 0.0391 | 0.0077 | 0.1721 | 1.0077 | 0.9316 | 0.0761 | -0.0651 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 955 | 2 | 885 | 18 | 0.1768 | 0.0847 | 0.1042 | 0.0391 | 0.0077 | 0.1722 | 1.0077 | 0.9315 | 0.0762 | -0.0651 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 956 | 2 | 886 | 18 | 0.1768 | 0.0847 | 0.1041 | 0.0391 | 0.0077 | 0.1723 | 1.0077 | 0.9319 | 0.0758 | -0.065 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 957 | 2 | 887 | 18 | 0.1768 | 0.0847 | 0.1039 | 0.0391 | 0.0077 | 0.1724 | 1.0077 | 0.9327 | 0.075 | -0.0648 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 958 | 2 | 888 | 18 | 0.1768 | 0.0847 | 0.1041 | 0.0391 | 0.0077 | 0.1725 | 1.0077 | 0.9322 | 0.0755 | -0.065 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 959 | 2 | 889 | 18 | 0.1768 | 0.0847 | 0.1044 | 0.0391 | 0.0077 | 0.1726 | 1.0077 | 0.9313 | 0.0763 | -0.0653 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 960 | 2 | 890 | 18 | 0.1768 | 0.0847 | 0.1047 | 0.0391 | 0.0077 | 0.1727 | 1.0077 | 0.9306 | 0.0771 | -0.0656 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 961 | 2 | 891 | 18 | 0.1768 | 0.0847 | 0.105 | 0.0391 | 0.0077 | 0.1728 | 1.0077 | 0.9299 | 0.0777 | -0.0659 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 962 | 2 | 892 | 18 | 0.1768 | 0.0847 | 0.1049 | 0.0391 | 0.0077 | 0.1729 | 1.0077 | 0.9303 | 0.0774 | -0.0658 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 963 | 2 | 893 | 18 | 0.1768 | 0.0847 | 0.1048 | 0.0391 | 0.0077 | 0.1729 | 1.0077 | 0.9306 | 0.0771 | -0.0657 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 964 | 2 | 894 | 18 | 0.1768 | 0.0847 | 0.1049 | 0.0391 | 0.0077 | 0.173 | 1.0077 | 0.9305 | 0.0771 | -0.0658 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 965 | 2 | 895 | 18 | 0.1768 | 0.0847 | 0.105 | 0.0391 | 0.0077 | 0.1731 | 1.0077 | 0.9304 | 0.0773 | -0.0659 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 966 | 2 | 896 | 18 | 0.1768 | 0.0847 | 0.1052 | 0.0391 | 0.0077 | 0.1732 | 1.0077 | 0.9298 | 0.0779 | -0.0661 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 967 | 2 | 897 | 18 | 0.1768 | 0.0847 | 0.1054 | 0.0391 | 0.0077 | 0.1733 | 1.0077 | 0.9292 | 0.0784 | -0.0663 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 968 | 2 | 898 | 18 | 0.1768 | 0.0847 | 0.1053 | 0.0391 | 0.0076 | 0.1734 | 1.0076 | 0.9296 | 0.078 | -0.0662 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 969 | 2 | 899 | 18 | 0.1768 | 0.0847 | 0.1052 | 0.0391 | 0.0076 | 0.1735 | 1.0076 | 0.9302 | 0.0775 | -0.0661 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 970 | 2 | 900 | 18 | 0.1768 | 0.0847 | 0.105 | 0.0391 | 0.0076 | 0.1736 | 1.0076 | 0.9306 | 0.0771 | -0.066 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 971 | 2 | 901 | 18 | 0.1768 | 0.0847 | 0.105 | 0.0391 | 0.0076 | 0.1737 | 1.0076 | 0.9308 | 0.0769 | -0.0659 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 972 | 2 | 902 | 18 | 0.1768 | 0.0847 | 0.1051 | 0.0391 | 0.0076 | 0.1738 | 1.0076 | 0.9305 | 0.0771 | -0.066 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 973 | 2 | 903 | 18 | 0.1768 | 0.0847 | 0.105 | 0.0391 | 0.0076 | 0.1738 | 1.0076 | 0.9309 | 0.0767 | -0.0659 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 974 | 2 | 904 | 18 | 0.1768 | 0.0847 | 0.1048 | 0.0391 | 0.0076 | 0.1739 | 1.0076 | 0.9316 | 0.076 | -0.0657 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 975 | 2 | 905 | 18 | 0.1768 | 0.0847 | 0.1046 | 0.0391 | 0.0076 | 0.174 | 1.0076 | 0.9323 | 0.0753 | -0.0655 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 976 | 2 | 906 | 18 | 0.1768 | 0.0847 | 0.1045 | 0.0391 | 0.0076 | 0.1741 | 1.0076 | 0.9325 | 0.0751 | -0.0654 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 977 | 2 | 907 | 18 | 0.1768 | 0.0847 | 0.1043 | 0.0391 | 0.0076 | 0.1742 | 1.0076 | 0.9332 | 0.0744 | -0.0652 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 978 | 2 | 908 | 18 | 0.1768 | 0.0847 | 0.1041 | 0.0391 | 0.0076 | 0.1743 | 1.0076 | 0.9339 | 0.0737 | -0.065 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 979 | 2 | 909 | 18 | 0.1768 | 0.0847 | 0.1039 | 0.0391 | 0.0076 | 0.1744 | 1.0076 | 0.9345 | 0.0731 | -0.0649 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 980 | 2 | 910 | 18 | 0.1768 | 0.0847 | 0.1039 | 0.0391 | 0.0076 | 0.1745 | 1.0076 | 0.9348 | 0.0728 | -0.0648 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 981 | 2 | 911 | 18 | 0.1768 | 0.0847 | 0.1038 | 0.0391 | 0.0076 | 0.1746 | 1.0076 | 0.9349 | 0.0727 | -0.0647 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 982 | 2 | 912 | 18 | 0.1768 | 0.0847 | 0.1038 | 0.0391 | 0.0076 | 0.1746 | 1.0076 | 0.9351 | 0.0724 | -0.0647 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 983 | 2 | 913 | 18 | 0.1768 | 0.0847 | 0.1039 | 0.0391 | 0.0076 | 0.1747 | 1.0076 | 0.9348 | 0.0728 | -0.0648 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 984 | 2 | 914 | 18 | 0.1768 | 0.0847 | 0.1038 | 0.0391 | 0.0076 | 0.1748 | 1.0076 | 0.9354 | 0.0722 | -0.0647 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 985 | 2 | 915 | 18 | 0.1768 | 0.0847 | 0.1038 | 0.0391 | 0.0076 | 0.1749 | 1.0076 | 0.9354 | 0.0722 | -0.0647 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 986 | 2 | 916 | 18 | 0.1768 | 0.0847 | 0.1037 | 0.0391 | 0.0076 | 0.175 | 1.0076 | 0.9357 | 0.0719 | -0.0646 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 987 | 2 | 917 | 18 | 0.1768 | 0.0847 | 0.104 | 0.0391 | 0.0076 | 0.1751 | 1.0076 | 0.935 | 0.0725 | -0.0649 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 988 | 2 | 918 | 18 | 0.1768 | 0.0847 | 0.1041 | 0.0391 | 0.0076 | 0.1752 | 1.0076 | 0.9348 | 0.0728 | -0.065 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 989 | 2 | 919 | 18 | 0.1768 | 0.0847 | 0.104 | 0.0391 | 0.0076 | 0.1753 | 1.0076 | 0.9352 | 0.0724 | -0.0649 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 990 | 2 | 920 | 18 | 0.1768 | 0.0847 | 0.1039 | 0.0391 | 0.0076 | 0.1754 | 1.0076 | 0.9357 | 0.0719 | -0.0648 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 991 | 2 | 921 | 18 | 0.1768 | 0.0847 | 0.1039 | 0.0391 | 0.0075 | 0.1754 | 1.0075 | 0.9356 | 0.0719 | -0.0648 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 992 | 2 | 922 | 18 | 0.1768 | 0.0847 | 0.1039 | 0.0391 | 0.0075 | 0.1755 | 1.0075 | 0.9356 | 0.0719 | -0.0648 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 993 | 2 | 923 | 18 | 0.1768 | 0.0847 | 0.1039 | 0.0391 | 0.0075 | 0.1756 | 1.0075 | 0.9358 | 0.0717 | -0.0648 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 994 | 2 | 924 | 18 | 0.1768 | 0.0847 | 0.1038 | 0.0391 | 0.0075 | 0.1757 | 1.0075 | 0.9363 | 0.0713 | -0.0647 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 995 | 2 | 925 | 18 | 0.1768 | 0.0847 | 0.1038 | 0.0391 | 0.0075 | 0.1758 | 1.0075 | 0.9361 | 0.0714 | -0.0648 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 996 | 2 | 926 | 18 | 0.1768 | 0.0847 | 0.1039 | 0.0391 | 0.0075 | 0.1759 | 1.0075 | 0.9362 | 0.0713 | -0.0648 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 997 | 2 | 927 | 18 | 0.1768 | 0.0847 | 0.1039 | 0.0391 | 0.0075 | 0.176 | 1.0075 | 0.9363 | 0.0712 | -0.0648 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 998 | 2 | 928 | 18 | 0.1768 | 0.0847 | 0.1039 | 0.0391 | 0.0075 | 0.1761 | 1.0075 | 0.9364 | 0.0712 | -0.0648 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 999 | 2 | 929 | 18 | 0.1768 | 0.0847 | 0.1037 | 0.0391 | 0.0075 | 0.1762 | 1.0075 | 0.9368 | 0.0707 | -0.0646 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1000 | 2 | 930 | 18 | 0.1768 | 0.0847 | 0.1037 | 0.0391 | 0.0075 | 0.1762 | 1.0075 | 0.9371 | 0.0704 | -0.0646 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1001 | 2 | 931 | 18 | 0.1768 | 0.0847 | 0.1035 | 0.0391 | 0.0075 | 0.1763 | 1.0075 | 0.9375 | 0.07 | -0.0644 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1002 | 2 | 932 | 18 | 0.1768 | 0.0847 | 0.1034 | 0.0391 | 0.0075 | 0.1764 | 1.0075 | 0.9379 | 0.0696 | -0.0644 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1003 | 2 | 933 | 18 | 0.1768 | 0.0847 | 0.1034 | 0.0391 | 0.0075 | 0.1765 | 1.0075 | 0.9381 | 0.0694 | -0.0643 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1004 | 2 | 934 | 18 | 0.1768 | 0.0847 | 0.1033 | 0.0391 | 0.0075 | 0.1766 | 1.0075 | 0.9385 | 0.069 | -0.0642 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1005 | 2 | 935 | 18 | 0.1768 | 0.0847 | 0.1035 | 0.0391 | 0.0075 | 0.1767 | 1.0075 | 0.9381 | 0.0694 | -0.0644 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1006 | 2 | 936 | 18 | 0.1768 | 0.0847 | 0.1034 | 0.0391 | 0.0075 | 0.1768 | 1.0075 | 0.9385 | 0.069 | -0.0643 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1007 | 2 | 937 | 18 | 0.1768 | 0.0847 | 0.1033 | 0.0391 | 0.0075 | 0.1769 | 1.0075 | 0.9389 | 0.0686 | -0.0642 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1008 | 2 | 938 | 18 | 0.1768 | 0.0847 | 0.1034 | 0.0391 | 0.0075 | 0.1769 | 1.0075 | 0.9385 | 0.069 | -0.0643 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1009 | 2 | 939 | 18 | 0.1768 | 0.0847 | 0.1035 | 0.0391 | 0.0075 | 0.177 | 1.0075 | 0.9384 | 0.0691 | -0.0644 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1010 | 2 | 940 | 18 | 0.1768 | 0.0847 | 0.1037 | 0.0391 | 0.0075 | 0.1771 | 1.0075 | 0.938 | 0.0695 | -0.0646 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1011 | 2 | 941 | 18 | 0.1768 | 0.0847 | 0.1034 | 0.0391 | 0.0075 | 0.1772 | 1.0075 | 0.9388 | 0.0686 | -0.0643 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1012 | 2 | 942 | 18 | 0.1768 | 0.0847 | 0.1033 | 0.0391 | 0.0075 | 0.1773 | 1.0075 | 0.9392 | 0.0683 | -0.0642 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1013 | 2 | 943 | 18 | 0.1768 | 0.0847 | 0.1031 | 0.0391 | 0.0075 | 0.1774 | 1.0075 | 0.9399 | 0.0675 | -0.064 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1014 | 2 | 944 | 18 | 0.1768 | 0.0847 | 0.103 | 0.0391 | 0.0074 | 0.1775 | 1.0074 | 0.9402 | 0.0672 | -0.0639 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1015 | 2 | 945 | 18 | 0.1768 | 0.0847 | 0.1029 | 0.0391 | 0.0074 | 0.1776 | 1.0074 | 0.9406 | 0.0669 | -0.0638 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1016 | 2 | 946 | 18 | 0.1768 | 0.0847 | 0.1028 | 0.0391 | 0.0074 | 0.1776 | 1.0074 | 0.9411 | 0.0664 | -0.0637 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1017 | 2 | 947 | 18 | 0.1768 | 0.0847 | 0.1026 | 0.0391 | 0.0074 | 0.1777 | 1.0074 | 0.9416 | 0.0658 | -0.0635 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1018 | 2 | 947 | 18 | 0.1768 | 0.0847 | 0.1026 | 0.0391 | 0.0074 | 0.1778 | 1.0074 | 0.9417 | 0.0657 | -0.0635 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1019 | 2 | 948 | 18 | 0.1768 | 0.0847 | 0.1028 | 0.0391 | 0.0074 | 0.1779 | 1.0074 | 0.9413 | 0.0661 | -0.0637 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1020 | 2 | 949 | 18 | 0.1768 | 0.0847 | 0.1029 | 0.0391 | 0.0074 | 0.178 | 1.0074 | 0.9411 | 0.0663 | -0.0638 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1021 | 2 | 950 | 18 | 0.1768 | 0.0847 | 0.1027 | 0.0391 | 0.0074 | 0.1781 | 1.0074 | 0.9418 | 0.0656 | -0.0636 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1022 | 2 | 951 | 18 | 0.1768 | 0.0847 | 0.1025 | 0.0391 | 0.0074 | 0.1782 | 1.0074 | 0.9424 | 0.065 | -0.0634 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1023 | 2 | 952 | 18 | 0.1768 | 0.0847 | 0.1022 | 0.0391 | 0.0074 | 0.1783 | 1.0074 | 0.9433 | 0.0641 | -0.0631 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1024 | 2 | 953 | 18 | 0.1768 | 0.0847 | 0.1021 | 0.0391 | 0.0074 | 0.1783 | 1.0074 | 0.9437 | 0.0637 | -0.063 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1025 | 2 | 953 | 18 | 0.1768 | 0.0847 | 0.1021 | 0.0391 | 0.0074 | 0.1784 | 1.0074 | 0.9438 | 0.0636 | -0.063 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1026 | 2 | 954 | 18 | 0.1768 | 0.0847 | 0.1019 | 0.0391 | 0.0074 | 0.1785 | 1.0074 | 0.9443 | 0.0631 | -0.0628 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1027 | 2 | 955 | 18 | 0.1768 | 0.0847 | 0.1018 | 0.0391 | 0.0074 | 0.1786 | 1.0074 | 0.9449 | 0.0625 | -0.0627 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1028 | 2 | 956 | 18 | 0.1768 | 0.0847 | 0.1016 | 0.0391 | 0.0074 | 0.1787 | 1.0074 | 0.9454 | 0.062 | -0.0625 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1029 | 2 | 957 | 18 | 0.1768 | 0.0847 | 0.1016 | 0.0391 | 0.0074 | 0.1788 | 1.0074 | 0.9455 | 0.0619 | -0.0625 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1030 | 2 | 958 | 18 | 0.1768 | 0.0847 | 0.1016 | 0.0391 | 0.0074 | 0.1789 | 1.0074 | 0.9456 | 0.0618 | -0.0625 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1031 | 2 | 959 | 18 | 0.1768 | 0.0847 | 0.1019 | 0.0391 | 0.0074 | 0.1789 | 1.0074 | 0.9449 | 0.0625 | -0.0628 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1032 | 2 | 960 | 18 | 0.1768 | 0.0847 | 0.102 | 0.0391 | 0.0074 | 0.179 | 1.0074 | 0.9447 | 0.0626 | -0.0629 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1033 | 2 | 961 | 18 | 0.1768 | 0.0847 | 0.1022 | 0.0391 | 0.0074 | 0.1791 | 1.0074 | 0.9442 | 0.0631 | -0.0631 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1034 | 2 | 962 | 18 | 0.1768 | 0.0847 | 0.1019 | 0.0391 | 0.0074 | 0.1792 | 1.0074 | 0.945 | 0.0624 | -0.0628 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1035 | 2 | 963 | 18 | 0.1768 | 0.0847 | 0.1017 | 0.0391 | 0.0074 | 0.1793 | 1.0074 | 0.9457 | 0.0617 | -0.0626 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1036 | 2 | 964 | 18 | 0.1768 | 0.0847 | 0.1015 | 0.0391 | 0.0074 | 0.1794 | 1.0074 | 0.9463 | 0.0611 | -0.0624 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1037 | 2 | 965 | 18 | 0.1768 | 0.0847 | 0.1014 | 0.0391 | 0.0074 | 0.1795 | 1.0074 | 0.9469 | 0.0605 | -0.0623 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1038 | 2 | 966 | 18 | 0.1768 | 0.0847 | 0.1012 | 0.0391 | 0.0074 | 0.1796 | 1.0074 | 0.9474 | 0.06 | -0.0621 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1039 | 2 | 967 | 18 | 0.1768 | 0.0847 | 0.101 | 0.0391 | 0.0074 | 0.1796 | 1.0074 | 0.9481 | 0.0593 | -0.0619 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1040 | 2 | 968 | 18 | 0.1768 | 0.0847 | 0.101 | 0.0391 | 0.0074 | 0.1797 | 1.0074 | 0.9482 | 0.0591 | -0.0619 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1041 | 2 | 968 | 18 | 0.1768 | 0.0847 | 0.101 | 0.0391 | 0.0074 | 0.1798 | 1.0074 | 0.9483 | 0.0591 | -0.0619 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1042 | 2 | 969 | 18 | 0.1768 | 0.0847 | 0.1008 | 0.0391 | 0.0074 | 0.1799 | 1.0074 | 0.949 | 0.0583 | -0.0617 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1043 | 2 | 970 | 18 | 0.1768 | 0.0847 | 0.1007 | 0.0391 | 0.0074 | 0.18 | 1.0074 | 0.9493 | 0.0581 | -0.0616 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1044 | 2 | 971 | 18 | 0.1768 | 0.0847 | 0.1008 | 0.0391 | 0.0073 | 0.1801 | 1.0073 | 0.949 | 0.0584 | -0.0617 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1045 | 2 | 972 | 18 | 0.1768 | 0.0847 | 0.101 | 0.0391 | 0.0073 | 0.1802 | 1.0073 | 0.9485 | 0.0588 | -0.0619 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1046 | 2 | 973 | 18 | 0.1768 | 0.0847 | 0.1013 | 0.0391 | 0.0073 | 0.1802 | 1.0073 | 0.948 | 0.0594 | -0.0622 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1047 | 2 | 974 | 18 | 0.1768 | 0.0847 | 0.1011 | 0.0391 | 0.0073 | 0.1803 | 1.0073 | 0.9486 | 0.0587 | -0.062 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1048 | 2 | 975 | 18 | 0.1768 | 0.0847 | 0.1007 | 0.0391 | 0.0073 | 0.1804 | 1.0073 | 0.9496 | 0.0577 | -0.0616 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1049 | 2 | 976 | 18 | 0.1768 | 0.0847 | 0.1006 | 0.0391 | 0.0073 | 0.1805 | 1.0073 | 0.95 | 0.0573 | -0.0615 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1050 | 2 | 977 | 18 | 0.1768 | 0.0847 | 0.1004 | 0.0391 | 0.0073 | 0.1806 | 1.0073 | 0.9508 | 0.0566 | -0.0613 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1051 | 2 | 978 | 18 | 0.1768 | 0.0847 | 0.1003 | 0.0391 | 0.0073 | 0.1807 | 1.0073 | 0.9513 | 0.056 | -0.0612 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1052 | 2 | 979 | 18 | 0.1768 | 0.0847 | 0.1002 | 0.0391 | 0.0073 | 0.1808 | 1.0073 | 0.9514 | 0.0559 | -0.0611 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1053 | 2 | 980 | 18 | 0.1768 | 0.0847 | 0.1002 | 0.0391 | 0.0073 | 0.1808 | 1.0073 | 0.9517 | 0.0556 | -0.0611 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1054 | 2 | 981 | 18 | 0.1768 | 0.0847 | 0.1 | 0.0391 | 0.0073 | 0.1809 | 1.0073 | 0.9524 | 0.0549 | -0.0609 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1055 | 2 | 982 | 18 | 0.1768 | 0.0847 | 0.0999 | 0.0391 | 0.0073 | 0.181 | 1.0073 | 0.9527 | 0.0546 | -0.0608 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1056 | 2 | 983 | 18 | 0.1768 | 0.0847 | 0.0999 | 0.0391 | 0.0073 | 0.1811 | 1.0073 | 0.9528 | 0.0545 | -0.0608 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1057 | 2 | 984 | 18 | 0.1768 | 0.0847 | 0.0999 | 0.0391 | 0.0073 | 0.1812 | 1.0073 | 0.9528 | 0.0545 | -0.0608 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1058 | 2 | 985 | 18 | 0.1768 | 0.0847 | 0.0999 | 0.0391 | 0.0073 | 0.1813 | 1.0073 | 0.9528 | 0.0545 | -0.0608 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1059 | 2 | 986 | 18 | 0.1768 | 0.0847 | 0.0998 | 0.0391 | 0.0073 | 0.1814 | 1.0073 | 0.9532 | 0.0541 | -0.0607 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1060 | 2 | 987 | 18 | 0.1768 | 0.0847 | 0.0997 | 0.0391 | 0.0073 | 0.1814 | 1.0073 | 0.9536 | 0.0537 | -0.0606 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1061 | 2 | 988 | 18 | 0.1768 | 0.0847 | 0.0996 | 0.0391 | 0.0073 | 0.1815 | 1.0073 | 0.9539 | 0.0534 | -0.0605 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1062 | 2 | 988 | 18 | 0.1768 | 0.0847 | 0.0996 | 0.0391 | 0.0073 | 0.1816 | 1.0073 | 0.954 | 0.0533 | -0.0605 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1063 | 2 | 989 | 18 | 0.1768 | 0.0847 | 0.0995 | 0.0391 | 0.0073 | 0.1817 | 1.0073 | 0.9544 | 0.0529 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1064 | 2 | 990 | 18 | 0.1768 | 0.0847 | 0.0995 | 0.0391 | 0.0073 | 0.1818 | 1.0073 | 0.9547 | 0.0526 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1065 | 2 | 991 | 18 | 0.1768 | 0.0847 | 0.0994 | 0.0391 | 0.0073 | 0.1819 | 1.0073 | 0.955 | 0.0523 | -0.0603 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1066 | 2 | 992 | 18 | 0.1768 | 0.0847 | 0.0995 | 0.0391 | 0.0073 | 0.182 | 1.0073 | 0.9549 | 0.0524 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1067 | 2 | 993 | 18 | 0.1768 | 0.0847 | 0.0996 | 0.0391 | 0.0073 | 0.182 | 1.0073 | 0.9546 | 0.0527 | -0.0605 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1068 | 2 | 994 | 18 | 0.1768 | 0.0847 | 0.0998 | 0.0391 | 0.0073 | 0.1821 | 1.0073 | 0.9541 | 0.0531 | -0.0607 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1069 | 2 | 995 | 18 | 0.1768 | 0.0847 | 0.0998 | 0.0391 | 0.0073 | 0.1822 | 1.0073 | 0.9542 | 0.053 | -0.0607 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1070 | 2 | 996 | 18 | 0.1768 | 0.0847 | 0.0998 | 0.0391 | 0.0073 | 0.1823 | 1.0073 | 0.9542 | 0.053 | -0.0607 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1071 | 2 | 997 | 18 | 0.1768 | 0.0847 | 0.0999 | 0.0391 | 0.0072 | 0.1824 | 1.0072 | 0.954 | 0.0532 | -0.0608 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1072 | 2 | 998 | 18 | 0.1768 | 0.0847 | 0.0998 | 0.0391 | 0.0072 | 0.1825 | 1.0072 | 0.9544 | 0.0529 | -0.0607 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1073 | 2 | 999 | 18 | 0.1768 | 0.0847 | 0.0998 | 0.0391 | 0.0072 | 0.1826 | 1.0072 | 0.9546 | 0.0526 | -0.0607 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1074 | 2 | 1000 | 18 | 0.1768 | 0.0847 | 0.0997 | 0.0391 | 0.0072 | 0.1826 | 1.0072 | 0.9549 | 0.0524 | -0.0606 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1075 | 2 | 1001 | 18 | 0.1768 | 0.0847 | 0.0998 | 0.0391 | 0.0072 | 0.1827 | 1.0072 | 0.9545 | 0.0527 | -0.0607 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1076 | 2 | 1002 | 18 | 0.1768 | 0.0847 | 0.1 | 0.0391 | 0.0072 | 0.1828 | 1.0072 | 0.9542 | 0.053 | -0.0609 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1077 | 2 | 1003 | 18 | 0.1768 | 0.0847 | 0.1001 | 0.0391 | 0.0072 | 0.1829 | 1.0072 | 0.954 | 0.0532 | -0.061 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1078 | 2 | 1004 | 18 | 0.1768 | 0.0847 | 0.1001 | 0.0391 | 0.0072 | 0.183 | 1.0072 | 0.9541 | 0.0531 | -0.061 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1079 | 2 | 1004 | 18 | 0.1768 | 0.0847 | 0.1001 | 0.0391 | 0.0072 | 0.1831 | 1.0072 | 0.7864 | 0.2208 | -0.061 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1080 | 2 | 1004 | 18 | 0.1768 | 0.0847 | 0.1001 | 0.0391 | 0.0072 | 0.1832 | 1.0072 | 0.7865 | 0.2208 | -0.061 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1081 | 2 | 1004 | 18 | 0.1768 | 0.0847 | 0.1001 | 0.0391 | 0.0072 | 0.1832 | 0.9642 | 0.7606 | 0.2036 | -0.061 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1082 | 2 | 1004 | 18 | 0.1768 | 0.0847 | 0.1001 | 0.0391 | 0.0072 | 0.1833 | 0.8045 | 0.6643 | 0.1402 | -0.061 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1083 | 2 | 1004 | 18 | 0.1768 | 0.0847 | 0.1001 | 0.0391 | 0.0072 | 0.1834 | 0.8109 | 0.6682 | 0.1426 | -0.061 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1084 | 2 | 1004 | 18 | 0.1768 | 0.0847 | 0.1001 | 0.0391 | 0.0072 | 0.1835 | 0.8792 | 0.7096 | 0.1696 | -0.061 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1085 | 2 | 1004 | 18 | 0.1768 | 0.0847 | 0.1001 | 0.0391 | 0.0072 | 0.1836 | 1.0072 | 0.7869 | 0.2204 | -0.061 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1086 | 2 | 1004 | 18 | 0.1768 | 0.0847 | 0.1001 | 0.0391 | 0.0072 | 0.1837 | 1.0072 | 0.787 | 0.2203 | -0.061 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1087 | 2 | 1004 | 18 | 0.1768 | 0.0847 | 0.1001 | 0.0391 | 0.0073 | 0.1837 | 1.0073 | 0.7871 | 0.2202 | -0.061 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1088 | 2 | 1005 | 18 | 0.1768 | 0.0847 | 0.1003 | 0.0391 | 0.0072 | 0.1838 | 1.0072 | 0.7864 | 0.2209 | -0.0612 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1089 | 2 | 1006 | 18 | 0.1768 | 0.0847 | 0.1004 | 0.0391 | 0.0072 | 0.1839 | 1.0072 | 0.7861 | 0.2212 | -0.0613 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1090 | 2 | 1007 | 18 | 0.1768 | 0.0847 | 0.1002 | 0.0391 | 0.0072 | 0.184 | 1.0072 | 0.787 | 0.2203 | -0.0611 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1091 | 2 | 1008 | 18 | 0.1768 | 0.0847 | 0.1003 | 0.0391 | 0.0072 | 0.1841 | 1.0072 | 0.7865 | 0.2207 | -0.0612 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1092 | 2 | 1009 | 18 | 0.1768 | 0.0847 | 0.1001 | 0.0391 | 0.0072 | 0.1842 | 1.0072 | 0.7873 | 0.2199 | -0.061 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1093 | 2 | 1010 | 18 | 0.1768 | 0.0847 | 0.1004 | 0.0391 | 0.0072 | 0.1843 | 1.0072 | 0.7864 | 0.2209 | -0.0613 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1094 | 2 | 1011 | 18 | 0.1768 | 0.0847 | 0.1002 | 0.0391 | 0.0072 | 0.1843 | 1.0072 | 0.7871 | 0.2201 | -0.0611 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1095 | 2 | 1012 | 18 | 0.1768 | 0.0847 | 0.1002 | 0.0391 | 0.0072 | 0.1844 | 1.0072 | 0.7872 | 0.22 | -0.0611 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1096 | 2 | 1013 | 18 | 0.1768 | 0.0847 | 0.1002 | 0.0391 | 0.0072 | 0.1845 | 1.0072 | 0.7873 | 0.2199 | -0.0611 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1097 | 2 | 1014 | 18 | 0.1768 | 0.0847 | 0.1002 | 0.0391 | 0.0072 | 0.1846 | 1.0072 | 0.7876 | 0.2196 | -0.0611 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1098 | 2 | 1015 | 18 | 0.1768 | 0.0847 | 0.1001 | 0.0391 | 0.0072 | 0.1847 | 1.0072 | 0.7879 | 0.2193 | -0.061 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1099 | 2 | 1016 | 18 | 0.1768 | 0.0847 | 0.0998 | 0.0391 | 0.0072 | 0.1848 | 1.0072 | 0.7891 | 0.2181 | -0.0607 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1100 | 2 | 1017 | 18 | 0.1768 | 0.0847 | 0.0997 | 0.0391 | 0.0072 | 0.1848 | 1.0072 | 0.7895 | 0.2177 | -0.0606 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1101 | 2 | 1018 | 18 | 0.1768 | 0.0847 | 0.1 | 0.0391 | 0.0072 | 0.1849 | 1.0072 | 0.7887 | 0.2185 | -0.0609 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1102 | 2 | 1019 | 18 | 0.1768 | 0.0847 | 0.0999 | 0.0391 | 0.0072 | 0.185 | 1.0072 | 0.7889 | 0.2183 | -0.0608 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1103 | 2 | 1020 | 18 | 0.1768 | 0.0847 | 0.1002 | 0.0391 | 0.0072 | 0.1851 | 1.0072 | 0.7879 | 0.2193 | -0.0611 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1104 | 2 | 1021 | 18 | 0.1768 | 0.0847 | 0.1003 | 0.0391 | 0.0072 | 0.1852 | 1.0072 | 0.7875 | 0.2197 | -0.0612 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1105 | 2 | 1022 | 18 | 0.1768 | 0.0847 | 0.1006 | 0.0391 | 0.0072 | 0.1853 | 1.0072 | 0.7865 | 0.2207 | -0.0615 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1106 | 2 | 1023 | 18 | 0.1768 | 0.0847 | 0.1005 | 0.0391 | 0.0072 | 0.1853 | 1.0072 | 0.787 | 0.2202 | -0.0614 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1107 | 2 | 1024 | 18 | 0.1768 | 0.0847 | 0.1006 | 0.0391 | 0.0072 | 0.1854 | 1.0072 | 0.7868 | 0.2203 | -0.0615 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1108 | 2 | 1025 | 18 | 0.1768 | 0.0847 | 0.1005 | 0.0391 | 0.0072 | 0.1855 | 1.0072 | 0.787 | 0.2201 | -0.0614 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1109 | 2 | 1026 | 18 | 0.1768 | 0.0847 | 0.1007 | 0.0391 | 0.0072 | 0.1856 | 1.0072 | 0.7864 | 0.2208 | -0.0616 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1110 | 2 | 1027 | 18 | 0.1768 | 0.0847 | 0.1008 | 0.0391 | 0.0072 | 0.1857 | 1.0072 | 0.7863 | 0.2209 | -0.0617 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1111 | 2 | 1028 | 18 | 0.1768 | 0.0847 | 0.1011 | 0.0391 | 0.0072 | 0.1858 | 1.0072 | 0.7851 | 0.222 | -0.062 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1112 | 2 | 1029 | 18 | 0.1768 | 0.0847 | 0.1012 | 0.0391 | 0.0072 | 0.1858 | 1.0072 | 0.7849 | 0.2223 | -0.0621 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1113 | 2 | 1030 | 18 | 0.1768 | 0.0847 | 0.1011 | 0.0391 | 0.0072 | 0.1859 | 1.0072 | 0.7852 | 0.222 | -0.062 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1114 | 2 | 1031 | 18 | 0.1768 | 0.0847 | 0.1011 | 0.0391 | 0.0071 | 0.186 | 1.0071 | 0.7852 | 0.2219 | -0.062 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1115 | 2 | 1032 | 18 | 0.1768 | 0.0847 | 0.101 | 0.0391 | 0.0071 | 0.1861 | 1.0071 | 0.786 | 0.2212 | -0.0619 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1116 | 2 | 1033 | 18 | 0.1768 | 0.0847 | 0.1011 | 0.0391 | 0.0071 | 0.1862 | 1.0071 | 0.7856 | 0.2215 | -0.062 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1117 | 2 | 1034 | 18 | 0.1768 | 0.0847 | 0.1012 | 0.0391 | 0.0071 | 0.1863 | 1.0071 | 0.7854 | 0.2218 | -0.0621 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1118 | 2 | 1035 | 18 | 0.1768 | 0.0847 | 0.1012 | 0.0391 | 0.0071 | 0.1863 | 1.0071 | 0.7851 | 0.222 | -0.0621 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1119 | 2 | 1036 | 18 | 0.1768 | 0.0847 | 0.1014 | 0.0391 | 0.0071 | 0.1864 | 1.0071 | 0.7844 | 0.2227 | -0.0624 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1120 | 2 | 1037 | 18 | 0.1768 | 0.0847 | 0.1015 | 0.0391 | 0.0071 | 0.1865 | 1.0071 | 0.7845 | 0.2227 | -0.0624 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1121 | 2 | 1038 | 18 | 0.1768 | 0.0847 | 0.1015 | 0.0391 | 0.0071 | 0.1866 | 1.0071 | 0.7845 | 0.2226 | -0.0624 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1122 | 2 | 1039 | 18 | 0.1768 | 0.0847 | 0.1012 | 0.0391 | 0.0071 | 0.1867 | 1.0071 | 0.7855 | 0.2216 | -0.0621 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1123 | 2 | 1040 | 18 | 0.1768 | 0.0847 | 0.1013 | 0.0391 | 0.0071 | 0.1868 | 1.0071 | 0.7854 | 0.2218 | -0.0622 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1124 | 2 | 1041 | 18 | 0.1768 | 0.0847 | 0.1015 | 0.0391 | 0.0071 | 0.1868 | 1.0071 | 0.7846 | 0.2225 | -0.0624 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1125 | 2 | 1042 | 18 | 0.1768 | 0.0847 | 0.1015 | 0.0391 | 0.0071 | 0.1869 | 1.0071 | 0.7847 | 0.2224 | -0.0624 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1126 | 2 | 1043 | 18 | 0.1768 | 0.0847 | 0.1013 | 0.0391 | 0.0071 | 0.187 | 1.0071 | 0.7855 | 0.2216 | -0.0622 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1127 | 2 | 1044 | 18 | 0.1768 | 0.0847 | 0.1012 | 0.0391 | 0.0071 | 0.1871 | 1.0071 | 0.7858 | 0.2213 | -0.0621 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1128 | 2 | 1045 | 18 | 0.1768 | 0.0847 | 0.1013 | 0.0391 | 0.0071 | 0.1872 | 1.0071 | 0.7858 | 0.2213 | -0.0622 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1129 | 2 | 1046 | 18 | 0.1768 | 0.0847 | 0.1013 | 0.0391 | 0.0071 | 0.1873 | 1.0071 | 0.786 | 0.2211 | -0.0622 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1130 | 2 | 1047 | 18 | 0.1768 | 0.0847 | 0.1011 | 0.0391 | 0.0071 | 0.1873 | 1.0071 | 0.7867 | 0.2204 | -0.062 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1131 | 2 | 1048 | 18 | 0.1768 | 0.0847 | 0.101 | 0.0391 | 0.0071 | 0.1874 | 1.0071 | 0.787 | 0.2201 | -0.0619 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1132 | 2 | 1049 | 18 | 0.1768 | 0.0847 | 0.1009 | 0.0391 | 0.0071 | 0.1875 | 1.0071 | 0.7875 | 0.2196 | -0.0618 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1133 | 2 | 1050 | 18 | 0.1768 | 0.0847 | 0.1009 | 0.0391 | 0.0071 | 0.1876 | 1.0071 | 0.7875 | 0.2196 | -0.0618 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1134 | 2 | 1051 | 18 | 0.1768 | 0.0847 | 0.1009 | 0.0391 | 0.0071 | 0.1877 | 1.0071 | 0.7878 | 0.2193 | -0.0618 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1135 | 2 | 1052 | 18 | 0.1768 | 0.0847 | 0.1008 | 0.0391 | 0.0071 | 0.1878 | 1.0071 | 0.7882 | 0.2189 | -0.0617 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1136 | 2 | 1053 | 18 | 0.1768 | 0.0847 | 0.1007 | 0.0391 | 0.0071 | 0.1878 | 1.0071 | 0.7887 | 0.2184 | -0.0616 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1137 | 2 | 1054 | 18 | 0.1768 | 0.0847 | 0.1005 | 0.0391 | 0.0071 | 0.1879 | 1.0071 | 0.7896 | 0.2175 | -0.0614 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1138 | 2 | 1055 | 18 | 0.1768 | 0.0847 | 0.1003 | 0.0391 | 0.0071 | 0.188 | 1.0071 | 0.7903 | 0.2168 | -0.0612 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1139 | 2 | 1056 | 18 | 0.1768 | 0.0847 | 0.1002 | 0.0391 | 0.0071 | 0.1881 | 1.0071 | 0.791 | 0.2161 | -0.0611 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1140 | 2 | 1057 | 18 | 0.1768 | 0.0847 | 0.1003 | 0.0391 | 0.0071 | 0.1882 | 1.0071 | 0.7908 | 0.2163 | -0.0612 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1141 | 2 | 1058 | 18 | 0.1768 | 0.0847 | 0.1002 | 0.0391 | 0.0071 | 0.1883 | 1.0071 | 0.7909 | 0.2162 | -0.0611 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1142 | 2 | 1059 | 18 | 0.1768 | 0.0847 | 0.1003 | 0.0391 | 0.007 | 0.1883 | 1.007 | 0.7909 | 0.2161 | -0.0612 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1143 | 2 | 1060 | 18 | 0.1768 | 0.0847 | 0.0999 | 0.0391 | 0.007 | 0.1884 | 1.007 | 0.7923 | 0.2147 | -0.0608 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1144 | 2 | 1061 | 18 | 0.1768 | 0.0847 | 0.0998 | 0.0391 | 0.007 | 0.1885 | 1.007 | 0.7927 | 0.2143 | -0.0607 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1145 | 2 | 1062 | 18 | 0.1768 | 0.0847 | 0.0997 | 0.0391 | 0.007 | 0.1886 | 1.007 | 0.7935 | 0.2136 | -0.0606 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1146 | 2 | 1063 | 18 | 0.1768 | 0.0847 | 0.0997 | 0.0391 | 0.007 | 0.1887 | 1.007 | 0.7933 | 0.2137 | -0.0606 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1147 | 2 | 1064 | 18 | 0.1768 | 0.0847 | 0.0997 | 0.0391 | 0.007 | 0.1887 | 1.007 | 0.7936 | 0.2134 | -0.0606 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1148 | 2 | 1065 | 18 | 0.1768 | 0.0847 | 0.0996 | 0.0391 | 0.007 | 0.1888 | 1.007 | 0.7941 | 0.2129 | -0.0605 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1149 | 2 | 1066 | 18 | 0.1768 | 0.0847 | 0.0994 | 0.0391 | 0.007 | 0.1889 | 1.007 | 0.7948 | 0.2122 | -0.0603 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1150 | 2 | 1067 | 18 | 0.1768 | 0.0847 | 0.0993 | 0.0391 | 0.007 | 0.189 | 1.007 | 0.7954 | 0.2116 | -0.0602 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1151 | 2 | 1068 | 18 | 0.1768 | 0.0847 | 0.0995 | 0.0391 | 0.007 | 0.1891 | 1.007 | 0.7947 | 0.2123 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1152 | 2 | 1069 | 18 | 0.1768 | 0.0847 | 0.0993 | 0.0391 | 0.007 | 0.1892 | 1.007 | 0.7954 | 0.2116 | -0.0602 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1153 | 2 | 1070 | 18 | 0.1768 | 0.0847 | 0.0993 | 0.0391 | 0.007 | 0.1892 | 1.007 | 0.7958 | 0.2112 | -0.0602 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1154 | 2 | 1071 | 18 | 0.1768 | 0.0847 | 0.0993 | 0.0391 | 0.007 | 0.1893 | 1.007 | 0.7957 | 0.2113 | -0.0602 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1155 | 2 | 1072 | 18 | 0.1768 | 0.0847 | 0.0992 | 0.0391 | 0.007 | 0.1894 | 1.007 | 0.7963 | 0.2107 | -0.0601 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1156 | 2 | 1073 | 18 | 0.1768 | 0.0847 | 0.0992 | 0.0391 | 0.007 | 0.1895 | 1.007 | 0.7961 | 0.2109 | -0.0601 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1157 | 2 | 1074 | 18 | 0.1768 | 0.0847 | 0.0993 | 0.0391 | 0.007 | 0.1896 | 1.007 | 0.7959 | 0.2111 | -0.0602 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1158 | 2 | 1075 | 18 | 0.1768 | 0.0847 | 0.0996 | 0.0391 | 0.007 | 0.1897 | 1.007 | 0.795 | 0.212 | -0.0605 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1159 | 2 | 1076 | 18 | 0.1768 | 0.0847 | 0.0997 | 0.0391 | 0.007 | 0.1897 | 1.007 | 0.7944 | 0.2126 | -0.0606 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1160 | 2 | 1077 | 18 | 0.1768 | 0.0847 | 0.0997 | 0.0391 | 0.007 | 0.1898 | 1.007 | 0.7944 | 0.2125 | -0.0606 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1161 | 2 | 1078 | 18 | 0.1768 | 0.0847 | 0.0997 | 0.0391 | 0.007 | 0.1899 | 1.007 | 0.7945 | 0.2124 | -0.0606 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1162 | 2 | 1079 | 18 | 0.1768 | 0.0847 | 0.0999 | 0.0391 | 0.007 | 0.19 | 1.007 | 0.7939 | 0.2131 | -0.0608 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1163 | 2 | 1080 | 18 | 0.1768 | 0.0847 | 0.1002 | 0.0391 | 0.007 | 0.1901 | 1.007 | 0.793 | 0.214 | -0.0611 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1164 | 2 | 1081 | 18 | 0.1768 | 0.0847 | 0.1001 | 0.0391 | 0.007 | 0.1901 | 1.007 | 0.7931 | 0.2138 | -0.0611 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1165 | 2 | 1082 | 18 | 0.1768 | 0.0847 | 0.1004 | 0.0391 | 0.007 | 0.1902 | 1.007 | 0.7921 | 0.2149 | -0.0613 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1166 | 2 | 1083 | 18 | 0.1768 | 0.0847 | 0.1002 | 0.0391 | 0.007 | 0.1903 | 1.007 | 0.7931 | 0.2139 | -0.0611 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1167 | 2 | 1084 | 18 | 0.1768 | 0.0847 | 0.1003 | 0.0391 | 0.007 | 0.1904 | 1.007 | 0.7926 | 0.2143 | -0.0612 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1168 | 2 | 1085 | 18 | 0.1768 | 0.0847 | 0.1003 | 0.0391 | 0.007 | 0.1905 | 1.007 | 0.7928 | 0.2142 | -0.0612 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1169 | 2 | 1086 | 18 | 0.1768 | 0.0847 | 0.1004 | 0.0391 | 0.007 | 0.1906 | 1.007 | 0.7925 | 0.2144 | -0.0613 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1170 | 2 | 1087 | 18 | 0.1768 | 0.0847 | 0.1006 | 0.0391 | 0.0069 | 0.1906 | 1.0069 | 0.7917 | 0.2152 | -0.0615 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1171 | 2 | 1088 | 18 | 0.1768 | 0.0847 | 0.1007 | 0.0391 | 0.0069 | 0.1907 | 1.0069 | 0.7917 | 0.2152 | -0.0616 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1172 | 2 | 1089 | 18 | 0.1768 | 0.0847 | 0.1007 | 0.0391 | 0.0069 | 0.1908 | 1.0069 | 0.7916 | 0.2153 | -0.0616 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1173 | 2 | 1090 | 18 | 0.1768 | 0.0847 | 0.1008 | 0.0391 | 0.0069 | 0.1909 | 1.0069 | 0.7913 | 0.2156 | -0.0617 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1174 | 2 | 1091 | 18 | 0.1768 | 0.0847 | 0.101 | 0.0391 | 0.0069 | 0.191 | 1.0069 | 0.7908 | 0.2161 | -0.0619 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1175 | 2 | 1092 | 18 | 0.1768 | 0.0847 | 0.1012 | 0.0391 | 0.0069 | 0.191 | 1.0069 | 0.7901 | 0.2168 | -0.0621 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1176 | 2 | 1093 | 18 | 0.1768 | 0.0847 | 0.101 | 0.0391 | 0.0069 | 0.1911 | 1.0069 | 0.7906 | 0.2163 | -0.0619 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1177 | 2 | 1094 | 18 | 0.1768 | 0.0847 | 0.1012 | 0.0391 | 0.0069 | 0.1912 | 1.0069 | 0.79 | 0.2169 | -0.0621 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1178 | 2 | 1095 | 18 | 0.1768 | 0.0847 | 0.1015 | 0.0391 | 0.0069 | 0.1913 | 1.0069 | 0.7889 | 0.218 | -0.0624 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1179 | 2 | 1096 | 18 | 0.1768 | 0.0847 | 0.1016 | 0.0391 | 0.0069 | 0.1914 | 1.0069 | 0.7886 | 0.2183 | -0.0625 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1180 | 2 | 1097 | 18 | 0.1768 | 0.0847 | 0.1018 | 0.0391 | 0.0069 | 0.1914 | 1.0069 | 0.7881 | 0.2189 | -0.0627 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1181 | 2 | 1098 | 18 | 0.1768 | 0.0847 | 0.1016 | 0.0391 | 0.0069 | 0.1915 | 1.0069 | 0.789 | 0.2179 | -0.0625 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1182 | 2 | 1099 | 18 | 0.1768 | 0.0847 | 0.1015 | 0.0391 | 0.0069 | 0.1916 | 1.0069 | 0.7894 | 0.2175 | -0.0624 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1183 | 2 | 1100 | 18 | 0.1768 | 0.0847 | 0.1018 | 0.0391 | 0.0069 | 0.1917 | 1.0069 | 0.7882 | 0.2187 | -0.0627 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1184 | 2 | 1101 | 18 | 0.1768 | 0.0847 | 0.1018 | 0.0391 | 0.0069 | 0.1918 | 1.0069 | 0.7885 | 0.2184 | -0.0627 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1185 | 2 | 1102 | 18 | 0.1768 | 0.0847 | 0.1019 | 0.0391 | 0.0069 | 0.1918 | 1.0069 | 0.788 | 0.2189 | -0.0628 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1186 | 2 | 1103 | 18 | 0.1768 | 0.0847 | 0.1021 | 0.0391 | 0.0069 | 0.1919 | 1.0069 | 0.7873 | 0.2196 | -0.063 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1187 | 2 | 1104 | 18 | 0.1768 | 0.0847 | 0.1022 | 0.0391 | 0.0069 | 0.192 | 1.0069 | 0.7869 | 0.2199 | -0.0631 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1188 | 2 | 1105 | 18 | 0.1768 | 0.0847 | 0.1024 | 0.0391 | 0.0069 | 0.1921 | 1.0069 | 0.7863 | 0.2206 | -0.0633 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1189 | 2 | 1106 | 18 | 0.1768 | 0.0847 | 0.1025 | 0.0391 | 0.0069 | 0.1922 | 1.0069 | 0.7862 | 0.2207 | -0.0634 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1190 | 2 | 1107 | 18 | 0.1768 | 0.0847 | 0.1024 | 0.0391 | 0.0069 | 0.1923 | 1.0069 | 0.7868 | 0.2201 | -0.0633 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1191 | 2 | 1108 | 18 | 0.1768 | 0.0847 | 0.1024 | 0.0391 | 0.0069 | 0.1923 | 1.0069 | 0.7865 | 0.2204 | -0.0633 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1192 | 2 | 1109 | 18 | 0.1768 | 0.0847 | 0.1023 | 0.0391 | 0.0069 | 0.1924 | 1.0069 | 0.7871 | 0.2197 | -0.0632 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1193 | 2 | 1110 | 18 | 0.1768 | 0.0847 | 0.1027 | 0.0391 | 0.0069 | 0.1925 | 1.0069 | 0.7858 | 0.2211 | -0.0636 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1194 | 2 | 1111 | 18 | 0.1768 | 0.0847 | 0.103 | 0.0391 | 0.0069 | 0.1926 | 1.0069 | 0.7847 | 0.2222 | -0.0639 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1195 | 2 | 1112 | 18 | 0.1768 | 0.0847 | 0.1031 | 0.0391 | 0.0069 | 0.1927 | 1.0069 | 0.7842 | 0.2227 | -0.064 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1196 | 2 | 1113 | 18 | 0.1768 | 0.0847 | 0.1033 | 0.0391 | 0.0069 | 0.1927 | 1.0069 | 0.7838 | 0.2231 | -0.0642 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1197 | 2 | 1114 | 18 | 0.1768 | 0.0847 | 0.1033 | 0.0391 | 0.0069 | 0.1928 | 1.0069 | 0.7836 | 0.2232 | -0.0642 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1198 | 2 | 1115 | 18 | 0.1768 | 0.0847 | 0.1033 | 0.0391 | 0.0069 | 0.1929 | 1.0069 | 0.7838 | 0.2231 | -0.0642 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1199 | 2 | 1116 | 18 | 0.1768 | 0.0847 | 0.1032 | 0.0391 | 0.0069 | 0.193 | 1.0069 | 0.7844 | 0.2225 | -0.0641 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| current | 1200 | 2 | 1117 | 18 | 0.1768 | 0.0847 | 0.1028 | 0.0391 | 0.0068 | 0.1931 | 1.0068 | 0.7858 | 0.2211 | -0.0637 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1 | 0 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.2068 | 0.0871 | 0.1171 | -0.0026 | 0.1197 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 2 | 1 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.2924 | 0.1232 | 0.7924 | 0.6232 | 0.1692 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 3 | 1 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.3581 | 0.1509 | 1.0248 | 0.8176 | 0.2072 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 4 | 1 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.4135 | 0.1742 | 1.1635 | 0.9242 | 0.2393 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 5 | 1 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.4624 | 0.1948 | 1.2624 | 0.9948 | 0.2675 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 6 | 1 | 1 | 0 | 0.1654 | 0.0697 | 0.0613 | 0.0 | 0.2532 | 0.2134 | 1.2532 | 0.9112 | 0.3421 | -0.0613 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 7 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.1824 | 0.2305 | 0.8111 | 0.9816 | -0.1705 | 0.0141 | - |
| guarded-w2 | 8 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.1949 | 0.2464 | 0.7905 | 1.0203 | -0.2298 | 0.0141 | - |
| guarded-w2 | 9 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.2068 | 0.2614 | 0.8101 | 1.0621 | -0.252 | 0.0141 | - |
| guarded-w2 | 10 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.218 | 0.2755 | 0.8513 | 1.1022 | -0.2509 | 0.0141 | - |
| guarded-w2 | 11 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.2286 | 0.2889 | 0.9247 | 1.1428 | -0.2181 | 0.0141 | - |
| guarded-w2 | 12 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.2388 | 0.3018 | 1.001 | 1.1788 | -0.1778 | 0.0141 | - |
| guarded-w2 | 13 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2485 | 0.1571 | 1.0108 | 0.332 | 0.6788 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 14 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2579 | 0.163 | 0.9741 | 0.3273 | 0.6468 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 15 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2669 | 0.1687 | 0.9785 | 0.332 | 0.6465 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 16 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2757 | 0.1742 | 0.9199 | 0.3221 | 0.5979 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 17 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2842 | 0.1796 | 0.9007 | 0.3211 | 0.5797 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 18 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2924 | 0.1848 | 0.946 | 0.3348 | 0.6112 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 19 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3004 | 0.1899 | 1.0251 | 0.3562 | 0.669 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 20 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3082 | 0.1948 | 1.0647 | 0.3684 | 0.6963 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 21 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3158 | 0.1996 | 1.0212 | 0.1996 | 0.8216 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 22 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3233 | 0.2043 | 1.0286 | 0.2043 | 0.8243 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 23 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3305 | 0.2089 | 1.0359 | 0.2089 | 0.827 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 24 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3377 | 0.2134 | 1.1206 | 0.2134 | 0.9072 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 25 | 1 | 3 | 1 | 0.1654 | 0.0697 | -0.0146 | -0.0723 | 0.2585 | 0.2178 | 1.034 | 0.2178 | 0.8162 | -0.0577 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 26 | 1 | 3 | 1 | 0.1654 | 0.0697 | -0.0146 | -0.0723 | 0.2636 | 0.2221 | 1.0788 | 0.2221 | 0.8567 | -0.0577 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 27 | 1 | 3 | 1 | 0.1654 | 0.0697 | -0.0146 | -0.0723 | 0.2686 | 0.2263 | 1.1196 | 0.42 | 0.6996 | -0.0577 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 28 | 1 | 4 | 1 | 0.1654 | 0.0697 | 0.0036 | -0.0723 | 0.2188 | 0.2305 | 1.2188 | 0.4135 | 0.8053 | -0.0759 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 29 | 1 | 5 | 1 | 0.1654 | 0.0697 | -0.014 | -0.0723 | 0.1856 | 0.2346 | 1.044 | 0.4283 | 0.6158 | -0.0584 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 30 | 1 | 5 | 1 | 0.1654 | 0.0697 | -0.014 | -0.0723 | 0.1888 | 0.2386 | 1.0472 | 0.4323 | 0.6149 | -0.0584 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 31 | 1 | 5 | 1 | 0.1654 | 0.0697 | -0.014 | -0.0723 | 0.1919 | 0.2425 | 1.1177 | 0.4514 | 0.6663 | -0.0584 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 32 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1671 | 0.2464 | 1.0028 | 0.4553 | 0.5475 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 33 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1697 | 0.2502 | 1.0347 | 0.4664 | 0.5683 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 34 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1722 | 0.254 | 1.0574 | 0.4752 | 0.5821 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 35 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1748 | 0.2577 | 1.0835 | 0.4848 | 0.5987 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 36 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1772 | 0.2614 | 1.1473 | 0.5038 | 0.6435 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 37 | 1 | 7 | 1 | 0.1654 | 0.0697 | -0.0089 | -0.0723 | 0.1572 | 0.265 | 1.1572 | 0.4763 | 0.6809 | -0.0635 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 38 | 1 | 8 | 1 | 0.1654 | 0.0697 | 0.0077 | -0.0723 | 0.1416 | 0.2685 | 1.1416 | 0.4437 | 0.6979 | -0.0801 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 39 | 1 | 9 | 1 | 0.1654 | 0.0697 | 0.0072 | -0.0723 | 0.1291 | 0.272 | 1.1291 | 0.4482 | 0.6809 | -0.0795 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 40 | 1 | 10 | 1 | 0.1654 | 0.0697 | 0.008 | -0.0723 | 0.1189 | 0.2755 | 1.1189 | 0.4502 | 0.6687 | -0.0803 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 41 | 1 | 11 | 1 | 0.1654 | 0.0697 | 0.0081 | -0.0723 | 0.1103 | 0.2789 | 1.1103 | 0.4535 | 0.6569 | -0.0804 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 42 | 1 | 12 | 1 | 0.1654 | 0.0697 | 0.0081 | -0.0723 | 0.1031 | 0.2823 | 1.1031 | 0.4568 | 0.6463 | -0.0805 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 43 | 1 | 13 | 1 | 0.1654 | 0.0697 | 0.0088 | -0.0723 | 0.0968 | 0.2856 | 1.0968 | 0.459 | 0.6379 | -0.0811 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 44 | 1 | 14 | 1 | 0.1654 | 0.0697 | 0.0127 | -0.0723 | 0.0914 | 0.2889 | 1.0914 | 0.4556 | 0.6358 | -0.085 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 45 | 1 | 15 | 1 | 0.1654 | 0.0697 | 0.0112 | -0.0723 | 0.0867 | 0.2922 | 1.0867 | 0.4613 | 0.6254 | -0.0836 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 46 | 1 | 16 | 1 | 0.1654 | 0.0697 | 0.009 | -0.0723 | 0.0825 | 0.2954 | 1.0825 | 0.4684 | 0.6141 | -0.0813 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 47 | 1 | 17 | 1 | 0.1654 | 0.0697 | 0.019 | -0.0723 | 0.0788 | 0.2986 | 1.0788 | 0.4557 | 0.6231 | -0.0913 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 48 | 1 | 18 | 1 | 0.1654 | 0.0697 | 0.0281 | -0.0723 | 0.0754 | 0.3018 | 1.0754 | 0.4466 | 0.6288 | -0.1005 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 49 | 1 | 19 | 1 | 0.1654 | 0.0697 | 0.029 | -0.0723 | 0.0724 | 0.3049 | 1.0724 | 0.4486 | 0.6238 | -0.1014 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 50 | 1 | 20 | 1 | 0.1654 | 0.0697 | 0.0288 | -0.0723 | 0.0696 | 0.308 | 1.0696 | 0.452 | 0.6177 | -0.1011 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 51 | 1 | 21 | 1 | 0.1654 | 0.0697 | 0.0289 | -0.0723 | 0.0671 | 0.3111 | 1.0671 | 0.4549 | 0.6122 | -0.1012 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 52 | 1 | 22 | 1 | 0.1654 | 0.0697 | 0.0289 | -0.0723 | 0.0648 | 0.3141 | 1.0648 | 0.458 | 0.6068 | -0.1012 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 53 | 2 | 23 | 1 | 0.1654 | 0.0697 | 0.0218 | -0.0723 | 0.0627 | 0.3171 | 1.0627 | 0.4701 | 0.5926 | -0.0941 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 54 | 2 | 24 | 1 | 0.1654 | 0.0697 | 0.0196 | -0.0723 | 0.0608 | 0.3201 | 1.0608 | 0.4762 | 0.5845 | -0.0919 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 55 | 2 | 25 | 1 | 0.1654 | 0.0697 | 0.0239 | -0.0723 | 0.059 | 0.323 | 1.059 | 0.4733 | 0.5857 | -0.0962 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 56 | 2 | 26 | 1 | 0.1654 | 0.0697 | 0.0186 | -0.0723 | 0.0573 | 0.326 | 1.0573 | 0.4835 | 0.5738 | -0.091 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 57 | 2 | 27 | 1 | 0.1654 | 0.0697 | 0.0165 | -0.0723 | 0.0558 | 0.3289 | 1.0558 | 0.4896 | 0.5661 | -0.0888 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 58 | 2 | 28 | 1 | 0.1654 | 0.0697 | 0.0125 | -0.0723 | 0.0543 | 0.3317 | 1.0543 | 0.4987 | 0.5556 | -0.0848 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 59 | 2 | 29 | 1 | 0.1654 | 0.0697 | 0.0099 | -0.0723 | 0.0529 | 0.3346 | 1.0529 | 0.506 | 0.547 | -0.0822 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 60 | 2 | 30 | 1 | 0.1654 | 0.0697 | 0.0061 | -0.0723 | 0.0517 | 0.3374 | 1.0517 | 0.5156 | 0.536 | -0.0784 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 61 | 2 | 30 | 1 | 0.1654 | 0.0697 | 0.0061 | -0.0723 | 0.0521 | 0.3402 | 1.0521 | 0.5184 | 0.5337 | -0.0784 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 62 | 2 | 31 | 1 | 0.1654 | 0.0697 | 0.0031 | -0.0723 | 0.0509 | 0.343 | 1.0509 | 0.527 | 0.5239 | -0.0754 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 63 | 2 | 32 | 1 | 0.1654 | 0.0697 | 0.0128 | -0.0723 | 0.0497 | 0.3457 | 1.0497 | 0.5122 | 0.5375 | -0.0851 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 64 | 2 | 33 | 1 | 0.1654 | 0.0697 | 0.0128 | -0.0723 | 0.0487 | 0.3485 | 1.0487 | 0.5149 | 0.5337 | -0.0852 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1 | 0 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.2068 | 0.0871 | 0.1171 | -0.0026 | 0.1197 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 2 | 1 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.2924 | 0.1232 | 0.7924 | 0.6232 | 0.1692 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 3 | 1 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.3581 | 0.1509 | 1.0248 | 0.8176 | 0.2072 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 4 | 1 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.4135 | 0.1742 | 1.1635 | 0.9242 | 0.2393 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 5 | 1 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.4624 | 0.1948 | 1.2624 | 0.9948 | 0.2675 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 6 | 1 | 1 | 0 | 0.1654 | 0.0697 | 0.0613 | 0.0 | 0.2532 | 0.2134 | 1.2532 | 0.9112 | 0.3421 | -0.0613 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 7 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.1824 | 0.2305 | 0.8111 | 0.9816 | -0.1705 | 0.0141 | - |
| guarded-w2 | 8 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.1949 | 0.2464 | 0.7905 | 1.0203 | -0.2298 | 0.0141 | - |
| guarded-w2 | 9 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.2068 | 0.2614 | 0.8101 | 1.0621 | -0.252 | 0.0141 | - |
| guarded-w2 | 10 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.218 | 0.2755 | 0.8513 | 1.1022 | -0.2509 | 0.0141 | - |
| guarded-w2 | 11 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.2286 | 0.2889 | 0.9247 | 1.1428 | -0.2181 | 0.0141 | - |
| guarded-w2 | 12 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.2388 | 0.3018 | 1.001 | 1.1788 | -0.1778 | 0.0141 | - |
| guarded-w2 | 13 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2485 | 0.1571 | 1.0108 | 0.332 | 0.6788 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 14 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2579 | 0.163 | 0.9741 | 0.3273 | 0.6468 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 15 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2669 | 0.1687 | 0.9785 | 0.332 | 0.6465 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 16 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2757 | 0.1742 | 0.9199 | 0.3221 | 0.5979 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 17 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2842 | 0.1796 | 0.9007 | 0.3211 | 0.5797 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 18 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2924 | 0.1848 | 0.946 | 0.3348 | 0.6112 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 19 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3004 | 0.1899 | 1.0251 | 0.3562 | 0.669 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 20 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3082 | 0.1948 | 1.0647 | 0.3684 | 0.6963 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 21 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3158 | 0.1996 | 1.0212 | 0.1996 | 0.8216 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 22 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3233 | 0.2043 | 1.0286 | 0.2043 | 0.8243 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 23 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3305 | 0.2089 | 1.0359 | 0.2089 | 0.827 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 24 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3377 | 0.2134 | 1.1206 | 0.2134 | 0.9072 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 25 | 1 | 3 | 1 | 0.1654 | 0.0697 | -0.0146 | -0.0723 | 0.2585 | 0.2178 | 1.034 | 0.2178 | 0.8162 | -0.0577 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 26 | 1 | 3 | 1 | 0.1654 | 0.0697 | -0.0146 | -0.0723 | 0.2636 | 0.2221 | 1.0788 | 0.2221 | 0.8567 | -0.0577 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 27 | 1 | 3 | 1 | 0.1654 | 0.0697 | -0.0146 | -0.0723 | 0.2686 | 0.2263 | 1.1196 | 0.42 | 0.6996 | -0.0577 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 28 | 1 | 4 | 1 | 0.1654 | 0.0697 | 0.0036 | -0.0723 | 0.2188 | 0.2305 | 1.2188 | 0.4135 | 0.8053 | -0.0759 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 29 | 1 | 5 | 1 | 0.1654 | 0.0697 | -0.014 | -0.0723 | 0.1856 | 0.2346 | 1.044 | 0.4283 | 0.6158 | -0.0584 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 30 | 1 | 5 | 1 | 0.1654 | 0.0697 | -0.014 | -0.0723 | 0.1888 | 0.2386 | 1.0472 | 0.4323 | 0.6149 | -0.0584 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 31 | 1 | 5 | 1 | 0.1654 | 0.0697 | -0.014 | -0.0723 | 0.1919 | 0.2425 | 1.1177 | 0.4514 | 0.6663 | -0.0584 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 32 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1671 | 0.2464 | 1.0028 | 0.4553 | 0.5475 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 33 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1697 | 0.2502 | 1.0347 | 0.4664 | 0.5683 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 34 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1722 | 0.254 | 1.0574 | 0.4752 | 0.5821 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 35 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1748 | 0.2577 | 1.0835 | 0.4848 | 0.5987 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 36 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1772 | 0.2614 | 1.1473 | 0.5038 | 0.6435 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 37 | 1 | 7 | 1 | 0.1654 | 0.0697 | -0.0089 | -0.0723 | 0.1572 | 0.265 | 1.1572 | 0.4763 | 0.6809 | -0.0635 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 38 | 1 | 8 | 1 | 0.1654 | 0.0697 | 0.0077 | -0.0723 | 0.1416 | 0.2685 | 1.1416 | 0.4437 | 0.6979 | -0.0801 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 39 | 1 | 9 | 1 | 0.1654 | 0.0697 | 0.0072 | -0.0723 | 0.1291 | 0.272 | 1.1291 | 0.4482 | 0.6809 | -0.0795 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 40 | 1 | 10 | 1 | 0.1654 | 0.0697 | 0.008 | -0.0723 | 0.1189 | 0.2755 | 1.1189 | 0.4502 | 0.6687 | -0.0803 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 41 | 1 | 11 | 1 | 0.1654 | 0.0697 | 0.0081 | -0.0723 | 0.1103 | 0.2789 | 1.1103 | 0.4535 | 0.6569 | -0.0804 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 42 | 1 | 12 | 1 | 0.1654 | 0.0697 | 0.0081 | -0.0723 | 0.1031 | 0.2823 | 1.1031 | 0.4568 | 0.6463 | -0.0805 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 43 | 1 | 13 | 1 | 0.1654 | 0.0697 | 0.0088 | -0.0723 | 0.0968 | 0.2856 | 1.0968 | 0.459 | 0.6379 | -0.0811 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 44 | 1 | 14 | 1 | 0.1654 | 0.0697 | 0.0127 | -0.0723 | 0.0914 | 0.2889 | 1.0914 | 0.4556 | 0.6358 | -0.085 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 45 | 1 | 15 | 1 | 0.1654 | 0.0697 | 0.0112 | -0.0723 | 0.0867 | 0.2922 | 1.0867 | 0.4613 | 0.6254 | -0.0836 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 46 | 1 | 16 | 1 | 0.1654 | 0.0697 | 0.009 | -0.0723 | 0.0825 | 0.2954 | 1.0825 | 0.4684 | 0.6141 | -0.0813 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 47 | 1 | 17 | 1 | 0.1654 | 0.0697 | 0.019 | -0.0723 | 0.0788 | 0.2986 | 1.0788 | 0.4557 | 0.6231 | -0.0913 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 48 | 1 | 18 | 1 | 0.1654 | 0.0697 | 0.0281 | -0.0723 | 0.0754 | 0.3018 | 1.0754 | 0.4466 | 0.6288 | -0.1005 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 49 | 1 | 19 | 1 | 0.1654 | 0.0697 | 0.029 | -0.0723 | 0.0724 | 0.3049 | 1.0724 | 0.4486 | 0.6238 | -0.1014 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 50 | 1 | 20 | 1 | 0.1654 | 0.0697 | 0.0288 | -0.0723 | 0.0696 | 0.308 | 1.0696 | 0.452 | 0.6177 | -0.1011 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 51 | 1 | 21 | 1 | 0.1654 | 0.0697 | 0.0289 | -0.0723 | 0.0671 | 0.3111 | 1.0671 | 0.4549 | 0.6122 | -0.1012 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 52 | 1 | 22 | 1 | 0.1654 | 0.0697 | 0.0289 | -0.0723 | 0.0648 | 0.3141 | 1.0648 | 0.458 | 0.6068 | -0.1012 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 53 | 2 | 23 | 1 | 0.1654 | 0.0697 | 0.0218 | -0.0723 | 0.0627 | 0.3171 | 1.0627 | 0.4701 | 0.5926 | -0.0941 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 54 | 2 | 24 | 1 | 0.1654 | 0.0697 | 0.0196 | -0.0723 | 0.0608 | 0.3201 | 1.0608 | 0.4762 | 0.5845 | -0.0919 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 55 | 2 | 25 | 1 | 0.1654 | 0.0697 | 0.0239 | -0.0723 | 0.059 | 0.323 | 1.059 | 0.4733 | 0.5857 | -0.0962 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 56 | 2 | 26 | 1 | 0.1654 | 0.0697 | 0.0186 | -0.0723 | 0.0573 | 0.326 | 1.0573 | 0.4835 | 0.5738 | -0.091 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 57 | 2 | 27 | 1 | 0.1654 | 0.0697 | 0.0165 | -0.0723 | 0.0558 | 0.3289 | 1.0558 | 0.4896 | 0.5661 | -0.0888 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 58 | 2 | 28 | 1 | 0.1654 | 0.0697 | 0.0125 | -0.0723 | 0.0543 | 0.3317 | 1.0543 | 0.4987 | 0.5556 | -0.0848 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 59 | 2 | 29 | 1 | 0.1654 | 0.0697 | 0.0099 | -0.0723 | 0.0529 | 0.3346 | 1.0529 | 0.506 | 0.547 | -0.0822 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 60 | 2 | 30 | 1 | 0.1654 | 0.0697 | 0.0061 | -0.0723 | 0.0517 | 0.3374 | 1.0517 | 0.5156 | 0.536 | -0.0784 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 61 | 2 | 30 | 1 | 0.1654 | 0.0697 | 0.0061 | -0.0723 | 0.0521 | 0.3402 | 1.0521 | 0.5184 | 0.5337 | -0.0784 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 62 | 2 | 31 | 1 | 0.1654 | 0.0697 | 0.0031 | -0.0723 | 0.0509 | 0.343 | 1.0509 | 0.527 | 0.5239 | -0.0754 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 63 | 2 | 32 | 1 | 0.1654 | 0.0697 | 0.0128 | -0.0723 | 0.0497 | 0.3457 | 1.0497 | 0.5122 | 0.5375 | -0.0851 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 64 | 2 | 33 | 1 | 0.1654 | 0.0697 | 0.0128 | -0.0723 | 0.0487 | 0.3485 | 1.0487 | 0.5149 | 0.5337 | -0.0852 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 65 | 2 | 34 | 1 | 0.1654 | 0.0697 | 0.0154 | -0.0723 | 0.0476 | 0.3512 | 1.0476 | 0.5136 | 0.534 | -0.0877 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 66 | 2 | 35 | 1 | 0.1654 | 0.0697 | 0.0225 | -0.0723 | 0.0467 | 0.3539 | 1.0467 | 0.506 | 0.5407 | -0.0948 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 67 | 2 | 36 | 1 | 0.1654 | 0.0697 | 0.0235 | -0.0723 | 0.0457 | 0.3566 | 1.0457 | 0.5073 | 0.5385 | -0.0958 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 68 | 2 | 37 | 1 | 0.1654 | 0.0697 | 0.0227 | -0.0723 | 0.0449 | 0.3592 | 1.0449 | 0.5109 | 0.5339 | -0.0951 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 69 | 2 | 38 | 1 | 0.1654 | 0.0697 | 0.0198 | -0.0723 | 0.044 | 0.3618 | 1.044 | 0.5176 | 0.5264 | -0.0922 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 70 | 2 | 39 | 1 | 0.1654 | 0.0697 | 0.0182 | -0.0723 | 0.0432 | 0.3644 | 1.0432 | 0.5226 | 0.5206 | -0.0905 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 71 | 2 | 40 | 1 | 0.1654 | 0.0697 | 0.0148 | -0.0723 | 0.0425 | 0.367 | 1.0425 | 0.5303 | 0.5122 | -0.0872 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 72 | 2 | 41 | 1 | 0.1654 | 0.0697 | 0.0118 | -0.0723 | 0.0418 | 0.3696 | 1.0418 | 0.5377 | 0.5041 | -0.0842 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 73 | 2 | 42 | 1 | 0.1654 | 0.0697 | 0.0089 | -0.0723 | 0.0411 | 0.3722 | 1.0411 | 0.5453 | 0.4958 | -0.0812 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 74 | 2 | 43 | 1 | 0.1654 | 0.0697 | 0.0091 | -0.0723 | 0.0404 | 0.3747 | 1.0404 | 0.5475 | 0.4929 | -0.0814 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 75 | 2 | 44 | 1 | 0.1654 | 0.0697 | 0.0074 | -0.0723 | 0.0398 | 0.3772 | 1.0398 | 0.553 | 0.4868 | -0.0797 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 76 | 2 | 45 | 1 | 0.1654 | 0.0697 | 0.0049 | -0.0723 | 0.0392 | 0.3797 | 1.0392 | 0.5602 | 0.479 | -0.0772 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 77 | 2 | 46 | 1 | 0.1654 | 0.0697 | 0.0022 | -0.0723 | 0.0386 | 0.3822 | 1.0386 | 0.568 | 0.4706 | -0.0745 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 78 | 2 | 47 | 1 | 0.1654 | 0.0697 | 0.0006 | -0.0723 | 0.038 | 0.3847 | 1.038 | 0.5738 | 0.4642 | -0.0729 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 79 | 2 | 48 | 1 | 0.1654 | 0.0697 | 0.0002 | -0.0723 | 0.0375 | 0.3872 | 1.0375 | 0.5771 | 0.4604 | -0.0725 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 80 | 2 | 49 | 1 | 0.1654 | 0.0697 | -0.0013 | -0.0723 | 0.037 | 0.3896 | 1.037 | 0.5827 | 0.4543 | -0.0711 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 81 | 2 | 50 | 1 | 0.1654 | 0.0697 | -0.0027 | -0.0723 | 0.0365 | 0.392 | 1.0365 | 0.5883 | 0.4482 | -0.0696 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 82 | 2 | 51 | 1 | 0.1654 | 0.0697 | -0.0017 | -0.0723 | 0.036 | 0.3945 | 1.036 | 0.5886 | 0.4474 | -0.0706 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 83 | 2 | 52 | 1 | 0.1654 | 0.0697 | -0.0034 | -0.0723 | 0.0355 | 0.3969 | 1.0355 | 0.5947 | 0.4408 | -0.0689 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 84 | 2 | 52 | 1 | 0.1654 | 0.0697 | -0.0034 | -0.0723 | 0.0358 | 0.3992 | 1.0358 | 0.5971 | 0.4386 | -0.0689 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 85 | 2 | 53 | 1 | 0.1654 | 0.0697 | -0.0036 | -0.0723 | 0.0353 | 0.4016 | 1.0353 | 0.5999 | 0.4354 | -0.0688 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 86 | 2 | 54 | 1 | 0.1654 | 0.0697 | -0.0043 | -0.0723 | 0.0349 | 0.404 | 1.0349 | 0.6038 | 0.431 | -0.0681 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 87 | 2 | 55 | 1 | 0.1654 | 0.0697 | -0.0059 | -0.0723 | 0.0344 | 0.4063 | 1.0344 | 0.6102 | 0.4242 | -0.0664 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 88 | 2 | 56 | 1 | 0.1654 | 0.0697 | -0.0073 | -0.0723 | 0.034 | 0.4086 | 1.034 | 0.6159 | 0.4181 | -0.065 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 89 | 2 | 57 | 1 | 0.1654 | 0.0697 | -0.0087 | -0.0723 | 0.0336 | 0.4109 | 1.0336 | 0.6217 | 0.4119 | -0.0637 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 90 | 2 | 58 | 1 | 0.1654 | 0.0697 | -0.007 | -0.0723 | 0.0332 | 0.4132 | 1.0332 | 0.6197 | 0.4135 | -0.0654 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 91 | 2 | 59 | 1 | 0.1654 | 0.0697 | -0.0072 | -0.0723 | 0.0329 | 0.4155 | 1.0329 | 0.6226 | 0.4103 | -0.0651 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 92 | 2 | 60 | 1 | 0.1654 | 0.0697 | -0.0081 | -0.0723 | 0.0325 | 0.4178 | 1.0325 | 0.6271 | 0.4055 | -0.0643 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 93 | 2 | 61 | 1 | 0.1654 | 0.0697 | -0.0091 | -0.0723 | 0.0322 | 0.4201 | 1.0322 | 0.6319 | 0.4003 | -0.0633 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 94 | 2 | 61 | 1 | 0.1654 | 0.0697 | -0.0091 | -0.0723 | 0.0323 | 0.4223 | 1.0323 | 0.6342 | 0.3982 | -0.0633 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 95 | 2 | 62 | 1 | 0.1654 | 0.0697 | -0.0102 | -0.0723 | 0.032 | 0.4246 | 1.032 | 0.6396 | 0.3924 | -0.0621 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 96 | 2 | 63 | 1 | 0.1654 | 0.0697 | -0.0085 | -0.0723 | 0.0317 | 0.4268 | 1.0317 | 0.6373 | 0.3944 | -0.0638 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 97 | 2 | 64 | 1 | 0.1654 | 0.0697 | -0.0092 | -0.0723 | 0.0313 | 0.429 | 1.0313 | 0.6412 | 0.3902 | -0.0632 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 98 | 2 | 65 | 1 | 0.1654 | 0.0697 | -0.0096 | -0.0723 | 0.031 | 0.4312 | 1.031 | 0.6445 | 0.3866 | -0.0627 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 99 | 2 | 66 | 1 | 0.1654 | 0.0697 | -0.0111 | -0.0723 | 0.0307 | 0.4334 | 1.0307 | 0.6508 | 0.3799 | -0.0612 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 100 | 2 | 67 | 1 | 0.1654 | 0.0697 | -0.0119 | -0.0723 | 0.0304 | 0.4356 | 1.0304 | 0.6553 | 0.3751 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 101 | 2 | 67 | 1 | 0.1654 | 0.0697 | -0.0119 | -0.0723 | 0.0306 | 0.4378 | 1.0306 | 0.6575 | 0.3731 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 102 | 2 | 67 | 1 | 0.1654 | 0.0697 | -0.0119 | -0.0723 | 0.0307 | 0.4399 | 1.0307 | 0.6596 | 0.3711 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 103 | 2 | 68 | 1 | 0.1654 | 0.0697 | -0.0106 | -0.0723 | 0.0304 | 0.4421 | 1.0304 | 0.6581 | 0.3723 | -0.0617 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 104 | 2 | 69 | 1 | 0.1654 | 0.0697 | -0.0081 | -0.0723 | 0.0301 | 0.4442 | 1.0301 | 0.6537 | 0.3764 | -0.0642 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 105 | 2 | 70 | 1 | 0.1654 | 0.0697 | -0.0079 | -0.0723 | 0.0298 | 0.4464 | 1.0298 | 0.6552 | 0.3746 | -0.0644 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 106 | 2 | 71 | 1 | 0.1654 | 0.0697 | -0.0064 | -0.0723 | 0.0296 | 0.4485 | 1.0296 | 0.6536 | 0.376 | -0.0659 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 107 | 2 | 72 | 1 | 0.1654 | 0.0697 | -0.0056 | -0.0723 | 0.0293 | 0.4506 | 1.0293 | 0.6536 | 0.3757 | -0.0667 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 108 | 2 | 73 | 1 | 0.1654 | 0.0697 | -0.0075 | -0.0723 | 0.029 | 0.4527 | 1.029 | 0.6605 | 0.3685 | -0.0648 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 109 | 2 | 74 | 1 | 0.1654 | 0.0697 | -0.0093 | -0.0723 | 0.0288 | 0.4548 | 1.0288 | 0.6673 | 0.3615 | -0.063 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 110 | 2 | 75 | 1 | 0.1654 | 0.0697 | -0.0092 | -0.0723 | 0.0285 | 0.4569 | 1.0285 | 0.6692 | 0.3594 | -0.0631 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 111 | 2 | 76 | 1 | 0.1654 | 0.0697 | -0.0106 | -0.0723 | 0.0283 | 0.4589 | 1.0283 | 0.6748 | 0.3535 | -0.0618 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 112 | 2 | 77 | 1 | 0.1654 | 0.0697 | -0.0127 | -0.0723 | 0.0281 | 0.461 | 1.0281 | 0.683 | 0.3451 | -0.0596 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 113 | 2 | 78 | 1 | 0.1654 | 0.0697 | -0.0142 | -0.0723 | 0.0278 | 0.463 | 1.0278 | 0.6894 | 0.3384 | -0.0581 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 114 | 2 | 79 | 1 | 0.1654 | 0.0697 | -0.0138 | -0.0723 | 0.0276 | 0.4651 | 1.0276 | 0.6903 | 0.3373 | -0.0585 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 115 | 2 | 80 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.0274 | 0.4671 | 1.0274 | 0.6932 | 0.3342 | -0.0582 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 116 | 2 | 81 | 1 | 0.1654 | 0.0697 | -0.015 | -0.0723 | 0.0272 | 0.4692 | 1.0272 | 0.6979 | 0.3292 | -0.0573 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 117 | 2 | 82 | 1 | 0.1654 | 0.0697 | -0.016 | -0.0723 | 0.0269 | 0.4712 | 1.0269 | 0.7031 | 0.3239 | -0.0563 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 118 | 2 | 83 | 1 | 0.1654 | 0.0697 | -0.0179 | -0.0723 | 0.0267 | 0.4732 | 1.0267 | 0.7112 | 0.3156 | -0.0544 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 119 | 2 | 84 | 1 | 0.1654 | 0.0697 | -0.0179 | -0.0723 | 0.0265 | 0.4752 | 1.0265 | 0.7131 | 0.3134 | -0.0545 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 120 | 2 | 85 | 1 | 0.1654 | 0.0697 | -0.0178 | -0.0723 | 0.0263 | 0.4772 | 1.0263 | 0.715 | 0.3113 | -0.0545 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 121 | 2 | 86 | 1 | 0.1654 | 0.0697 | -0.0165 | -0.0723 | 0.0261 | 0.4792 | 1.0261 | 0.7126 | 0.3135 | -0.0558 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 122 | 2 | 87 | 1 | 0.1654 | 0.0697 | -0.0172 | -0.0723 | 0.026 | 0.4811 | 1.026 | 0.7168 | 0.3092 | -0.0552 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 123 | 2 | 88 | 1 | 0.1654 | 0.0697 | -0.0182 | -0.0723 | 0.0258 | 0.4831 | 1.0258 | 0.7221 | 0.3037 | -0.0542 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 124 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0256 | 0.4851 | 1.0256 | 0.7256 | 0.3 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 125 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0257 | 0.487 | 1.0257 | 0.7276 | 0.2981 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 126 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0258 | 0.489 | 1.0258 | 0.7295 | 0.2963 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 127 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0259 | 0.4909 | 1.0259 | 0.7314 | 0.2945 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 128 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.026 | 0.4928 | 1.026 | 0.7334 | 0.2926 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1 | 0 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.2068 | 0.0871 | 0.1171 | -0.0026 | 0.1197 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 2 | 1 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.2924 | 0.1232 | 0.7924 | 0.6232 | 0.1692 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 3 | 1 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.3581 | 0.1509 | 1.0248 | 0.8176 | 0.2072 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 4 | 1 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.4135 | 0.1742 | 1.1635 | 0.9242 | 0.2393 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 5 | 1 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.4624 | 0.1948 | 1.2624 | 0.9948 | 0.2675 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 6 | 1 | 1 | 0 | 0.1654 | 0.0697 | 0.0613 | 0.0 | 0.2532 | 0.2134 | 1.2532 | 0.9112 | 0.3421 | -0.0613 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 7 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.1824 | 0.2305 | 0.8111 | 0.9816 | -0.1705 | 0.0141 | - |
| guarded-w2 | 8 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.1949 | 0.2464 | 0.7905 | 1.0203 | -0.2298 | 0.0141 | - |
| guarded-w2 | 9 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.2068 | 0.2614 | 0.8101 | 1.0621 | -0.252 | 0.0141 | - |
| guarded-w2 | 10 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.218 | 0.2755 | 0.8513 | 1.1022 | -0.2509 | 0.0141 | - |
| guarded-w2 | 11 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.2286 | 0.2889 | 0.9247 | 1.1428 | -0.2181 | 0.0141 | - |
| guarded-w2 | 12 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.2388 | 0.3018 | 1.001 | 1.1788 | -0.1778 | 0.0141 | - |
| guarded-w2 | 13 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2485 | 0.1571 | 1.0108 | 0.332 | 0.6788 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 14 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2579 | 0.163 | 0.9741 | 0.3273 | 0.6468 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 15 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2669 | 0.1687 | 0.9785 | 0.332 | 0.6465 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 16 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2757 | 0.1742 | 0.9199 | 0.3221 | 0.5979 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 17 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2842 | 0.1796 | 0.9007 | 0.3211 | 0.5797 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 18 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2924 | 0.1848 | 0.946 | 0.3348 | 0.6112 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 19 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3004 | 0.1899 | 1.0251 | 0.3562 | 0.669 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 20 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3082 | 0.1948 | 1.0647 | 0.3684 | 0.6963 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 21 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3158 | 0.1996 | 1.0212 | 0.1996 | 0.8216 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 22 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3233 | 0.2043 | 1.0286 | 0.2043 | 0.8243 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 23 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3305 | 0.2089 | 1.0359 | 0.2089 | 0.827 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 24 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3377 | 0.2134 | 1.1206 | 0.2134 | 0.9072 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 25 | 1 | 3 | 1 | 0.1654 | 0.0697 | -0.0146 | -0.0723 | 0.2585 | 0.2178 | 1.034 | 0.2178 | 0.8162 | -0.0577 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 26 | 1 | 3 | 1 | 0.1654 | 0.0697 | -0.0146 | -0.0723 | 0.2636 | 0.2221 | 1.0788 | 0.2221 | 0.8567 | -0.0577 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 27 | 1 | 3 | 1 | 0.1654 | 0.0697 | -0.0146 | -0.0723 | 0.2686 | 0.2263 | 1.1196 | 0.42 | 0.6996 | -0.0577 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 28 | 1 | 4 | 1 | 0.1654 | 0.0697 | 0.0036 | -0.0723 | 0.2188 | 0.2305 | 1.2188 | 0.4135 | 0.8053 | -0.0759 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 29 | 1 | 5 | 1 | 0.1654 | 0.0697 | -0.014 | -0.0723 | 0.1856 | 0.2346 | 1.044 | 0.4283 | 0.6158 | -0.0584 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 30 | 1 | 5 | 1 | 0.1654 | 0.0697 | -0.014 | -0.0723 | 0.1888 | 0.2386 | 1.0472 | 0.4323 | 0.6149 | -0.0584 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 31 | 1 | 5 | 1 | 0.1654 | 0.0697 | -0.014 | -0.0723 | 0.1919 | 0.2425 | 1.1177 | 0.4514 | 0.6663 | -0.0584 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 32 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1671 | 0.2464 | 1.0028 | 0.4553 | 0.5475 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 33 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1697 | 0.2502 | 1.0347 | 0.4664 | 0.5683 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 34 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1722 | 0.254 | 1.0574 | 0.4752 | 0.5821 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 35 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1748 | 0.2577 | 1.0835 | 0.4848 | 0.5987 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 36 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1772 | 0.2614 | 1.1473 | 0.5038 | 0.6435 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 37 | 1 | 7 | 1 | 0.1654 | 0.0697 | -0.0089 | -0.0723 | 0.1572 | 0.265 | 1.1572 | 0.4763 | 0.6809 | -0.0635 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 38 | 1 | 8 | 1 | 0.1654 | 0.0697 | 0.0077 | -0.0723 | 0.1416 | 0.2685 | 1.1416 | 0.4437 | 0.6979 | -0.0801 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 39 | 1 | 9 | 1 | 0.1654 | 0.0697 | 0.0072 | -0.0723 | 0.1291 | 0.272 | 1.1291 | 0.4482 | 0.6809 | -0.0795 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 40 | 1 | 10 | 1 | 0.1654 | 0.0697 | 0.008 | -0.0723 | 0.1189 | 0.2755 | 1.1189 | 0.4502 | 0.6687 | -0.0803 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 41 | 1 | 11 | 1 | 0.1654 | 0.0697 | 0.0081 | -0.0723 | 0.1103 | 0.2789 | 1.1103 | 0.4535 | 0.6569 | -0.0804 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 42 | 1 | 12 | 1 | 0.1654 | 0.0697 | 0.0081 | -0.0723 | 0.1031 | 0.2823 | 1.1031 | 0.4568 | 0.6463 | -0.0805 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 43 | 1 | 13 | 1 | 0.1654 | 0.0697 | 0.0088 | -0.0723 | 0.0968 | 0.2856 | 1.0968 | 0.459 | 0.6379 | -0.0811 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 44 | 1 | 14 | 1 | 0.1654 | 0.0697 | 0.0127 | -0.0723 | 0.0914 | 0.2889 | 1.0914 | 0.4556 | 0.6358 | -0.085 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 45 | 1 | 15 | 1 | 0.1654 | 0.0697 | 0.0112 | -0.0723 | 0.0867 | 0.2922 | 1.0867 | 0.4613 | 0.6254 | -0.0836 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 46 | 1 | 16 | 1 | 0.1654 | 0.0697 | 0.009 | -0.0723 | 0.0825 | 0.2954 | 1.0825 | 0.4684 | 0.6141 | -0.0813 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 47 | 1 | 17 | 1 | 0.1654 | 0.0697 | 0.019 | -0.0723 | 0.0788 | 0.2986 | 1.0788 | 0.4557 | 0.6231 | -0.0913 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 48 | 1 | 18 | 1 | 0.1654 | 0.0697 | 0.0281 | -0.0723 | 0.0754 | 0.3018 | 1.0754 | 0.4466 | 0.6288 | -0.1005 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 49 | 1 | 19 | 1 | 0.1654 | 0.0697 | 0.029 | -0.0723 | 0.0724 | 0.3049 | 1.0724 | 0.4486 | 0.6238 | -0.1014 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 50 | 1 | 20 | 1 | 0.1654 | 0.0697 | 0.0288 | -0.0723 | 0.0696 | 0.308 | 1.0696 | 0.452 | 0.6177 | -0.1011 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 51 | 1 | 21 | 1 | 0.1654 | 0.0697 | 0.0289 | -0.0723 | 0.0671 | 0.3111 | 1.0671 | 0.4549 | 0.6122 | -0.1012 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 52 | 1 | 22 | 1 | 0.1654 | 0.0697 | 0.0289 | -0.0723 | 0.0648 | 0.3141 | 1.0648 | 0.458 | 0.6068 | -0.1012 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 53 | 2 | 23 | 1 | 0.1654 | 0.0697 | 0.0218 | -0.0723 | 0.0627 | 0.3171 | 1.0627 | 0.4701 | 0.5926 | -0.0941 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 54 | 2 | 24 | 1 | 0.1654 | 0.0697 | 0.0196 | -0.0723 | 0.0608 | 0.3201 | 1.0608 | 0.4762 | 0.5845 | -0.0919 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 55 | 2 | 25 | 1 | 0.1654 | 0.0697 | 0.0239 | -0.0723 | 0.059 | 0.323 | 1.059 | 0.4733 | 0.5857 | -0.0962 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 56 | 2 | 26 | 1 | 0.1654 | 0.0697 | 0.0186 | -0.0723 | 0.0573 | 0.326 | 1.0573 | 0.4835 | 0.5738 | -0.091 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 57 | 2 | 27 | 1 | 0.1654 | 0.0697 | 0.0165 | -0.0723 | 0.0558 | 0.3289 | 1.0558 | 0.4896 | 0.5661 | -0.0888 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 58 | 2 | 28 | 1 | 0.1654 | 0.0697 | 0.0125 | -0.0723 | 0.0543 | 0.3317 | 1.0543 | 0.4987 | 0.5556 | -0.0848 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 59 | 2 | 29 | 1 | 0.1654 | 0.0697 | 0.0099 | -0.0723 | 0.0529 | 0.3346 | 1.0529 | 0.506 | 0.547 | -0.0822 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 60 | 2 | 30 | 1 | 0.1654 | 0.0697 | 0.0061 | -0.0723 | 0.0517 | 0.3374 | 1.0517 | 0.5156 | 0.536 | -0.0784 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 61 | 2 | 30 | 1 | 0.1654 | 0.0697 | 0.0061 | -0.0723 | 0.0521 | 0.3402 | 1.0521 | 0.5184 | 0.5337 | -0.0784 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 62 | 2 | 31 | 1 | 0.1654 | 0.0697 | 0.0031 | -0.0723 | 0.0509 | 0.343 | 1.0509 | 0.527 | 0.5239 | -0.0754 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 63 | 2 | 32 | 1 | 0.1654 | 0.0697 | 0.0128 | -0.0723 | 0.0497 | 0.3457 | 1.0497 | 0.5122 | 0.5375 | -0.0851 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 64 | 2 | 33 | 1 | 0.1654 | 0.0697 | 0.0128 | -0.0723 | 0.0487 | 0.3485 | 1.0487 | 0.5149 | 0.5337 | -0.0852 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 65 | 2 | 34 | 1 | 0.1654 | 0.0697 | 0.0154 | -0.0723 | 0.0476 | 0.3512 | 1.0476 | 0.5136 | 0.534 | -0.0877 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 66 | 2 | 35 | 1 | 0.1654 | 0.0697 | 0.0225 | -0.0723 | 0.0467 | 0.3539 | 1.0467 | 0.506 | 0.5407 | -0.0948 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 67 | 2 | 36 | 1 | 0.1654 | 0.0697 | 0.0235 | -0.0723 | 0.0457 | 0.3566 | 1.0457 | 0.5073 | 0.5385 | -0.0958 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 68 | 2 | 37 | 1 | 0.1654 | 0.0697 | 0.0227 | -0.0723 | 0.0449 | 0.3592 | 1.0449 | 0.5109 | 0.5339 | -0.0951 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 69 | 2 | 38 | 1 | 0.1654 | 0.0697 | 0.0198 | -0.0723 | 0.044 | 0.3618 | 1.044 | 0.5176 | 0.5264 | -0.0922 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 70 | 2 | 39 | 1 | 0.1654 | 0.0697 | 0.0182 | -0.0723 | 0.0432 | 0.3644 | 1.0432 | 0.5226 | 0.5206 | -0.0905 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 71 | 2 | 40 | 1 | 0.1654 | 0.0697 | 0.0148 | -0.0723 | 0.0425 | 0.367 | 1.0425 | 0.5303 | 0.5122 | -0.0872 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 72 | 2 | 41 | 1 | 0.1654 | 0.0697 | 0.0118 | -0.0723 | 0.0418 | 0.3696 | 1.0418 | 0.5377 | 0.5041 | -0.0842 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 73 | 2 | 42 | 1 | 0.1654 | 0.0697 | 0.0089 | -0.0723 | 0.0411 | 0.3722 | 1.0411 | 0.5453 | 0.4958 | -0.0812 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 74 | 2 | 43 | 1 | 0.1654 | 0.0697 | 0.0091 | -0.0723 | 0.0404 | 0.3747 | 1.0404 | 0.5475 | 0.4929 | -0.0814 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 75 | 2 | 44 | 1 | 0.1654 | 0.0697 | 0.0074 | -0.0723 | 0.0398 | 0.3772 | 1.0398 | 0.553 | 0.4868 | -0.0797 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 76 | 2 | 45 | 1 | 0.1654 | 0.0697 | 0.0049 | -0.0723 | 0.0392 | 0.3797 | 1.0392 | 0.5602 | 0.479 | -0.0772 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 77 | 2 | 46 | 1 | 0.1654 | 0.0697 | 0.0022 | -0.0723 | 0.0386 | 0.3822 | 1.0386 | 0.568 | 0.4706 | -0.0745 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 78 | 2 | 47 | 1 | 0.1654 | 0.0697 | 0.0006 | -0.0723 | 0.038 | 0.3847 | 1.038 | 0.5738 | 0.4642 | -0.0729 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 79 | 2 | 48 | 1 | 0.1654 | 0.0697 | 0.0002 | -0.0723 | 0.0375 | 0.3872 | 1.0375 | 0.5771 | 0.4604 | -0.0725 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 80 | 2 | 49 | 1 | 0.1654 | 0.0697 | -0.0013 | -0.0723 | 0.037 | 0.3896 | 1.037 | 0.5827 | 0.4543 | -0.0711 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 81 | 2 | 50 | 1 | 0.1654 | 0.0697 | -0.0027 | -0.0723 | 0.0365 | 0.392 | 1.0365 | 0.5883 | 0.4482 | -0.0696 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 82 | 2 | 51 | 1 | 0.1654 | 0.0697 | -0.0017 | -0.0723 | 0.036 | 0.3945 | 1.036 | 0.5886 | 0.4474 | -0.0706 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 83 | 2 | 52 | 1 | 0.1654 | 0.0697 | -0.0034 | -0.0723 | 0.0355 | 0.3969 | 1.0355 | 0.5947 | 0.4408 | -0.0689 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 84 | 2 | 52 | 1 | 0.1654 | 0.0697 | -0.0034 | -0.0723 | 0.0358 | 0.3992 | 1.0358 | 0.5971 | 0.4386 | -0.0689 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 85 | 2 | 53 | 1 | 0.1654 | 0.0697 | -0.0036 | -0.0723 | 0.0353 | 0.4016 | 1.0353 | 0.5999 | 0.4354 | -0.0688 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 86 | 2 | 54 | 1 | 0.1654 | 0.0697 | -0.0043 | -0.0723 | 0.0349 | 0.404 | 1.0349 | 0.6038 | 0.431 | -0.0681 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 87 | 2 | 55 | 1 | 0.1654 | 0.0697 | -0.0059 | -0.0723 | 0.0344 | 0.4063 | 1.0344 | 0.6102 | 0.4242 | -0.0664 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 88 | 2 | 56 | 1 | 0.1654 | 0.0697 | -0.0073 | -0.0723 | 0.034 | 0.4086 | 1.034 | 0.6159 | 0.4181 | -0.065 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 89 | 2 | 57 | 1 | 0.1654 | 0.0697 | -0.0087 | -0.0723 | 0.0336 | 0.4109 | 1.0336 | 0.6217 | 0.4119 | -0.0637 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 90 | 2 | 58 | 1 | 0.1654 | 0.0697 | -0.007 | -0.0723 | 0.0332 | 0.4132 | 1.0332 | 0.6197 | 0.4135 | -0.0654 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 91 | 2 | 59 | 1 | 0.1654 | 0.0697 | -0.0072 | -0.0723 | 0.0329 | 0.4155 | 1.0329 | 0.6226 | 0.4103 | -0.0651 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 92 | 2 | 60 | 1 | 0.1654 | 0.0697 | -0.0081 | -0.0723 | 0.0325 | 0.4178 | 1.0325 | 0.6271 | 0.4055 | -0.0643 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 93 | 2 | 61 | 1 | 0.1654 | 0.0697 | -0.0091 | -0.0723 | 0.0322 | 0.4201 | 1.0322 | 0.6319 | 0.4003 | -0.0633 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 94 | 2 | 61 | 1 | 0.1654 | 0.0697 | -0.0091 | -0.0723 | 0.0323 | 0.4223 | 1.0323 | 0.6342 | 0.3982 | -0.0633 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 95 | 2 | 62 | 1 | 0.1654 | 0.0697 | -0.0102 | -0.0723 | 0.032 | 0.4246 | 1.032 | 0.6396 | 0.3924 | -0.0621 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 96 | 2 | 63 | 1 | 0.1654 | 0.0697 | -0.0085 | -0.0723 | 0.0317 | 0.4268 | 1.0317 | 0.6373 | 0.3944 | -0.0638 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 97 | 2 | 64 | 1 | 0.1654 | 0.0697 | -0.0092 | -0.0723 | 0.0313 | 0.429 | 1.0313 | 0.6412 | 0.3902 | -0.0632 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 98 | 2 | 65 | 1 | 0.1654 | 0.0697 | -0.0096 | -0.0723 | 0.031 | 0.4312 | 1.031 | 0.6445 | 0.3866 | -0.0627 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 99 | 2 | 66 | 1 | 0.1654 | 0.0697 | -0.0111 | -0.0723 | 0.0307 | 0.4334 | 1.0307 | 0.6508 | 0.3799 | -0.0612 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 100 | 2 | 67 | 1 | 0.1654 | 0.0697 | -0.0119 | -0.0723 | 0.0304 | 0.4356 | 1.0304 | 0.6553 | 0.3751 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 101 | 2 | 67 | 1 | 0.1654 | 0.0697 | -0.0119 | -0.0723 | 0.0306 | 0.4378 | 1.0306 | 0.6575 | 0.3731 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 102 | 2 | 67 | 1 | 0.1654 | 0.0697 | -0.0119 | -0.0723 | 0.0307 | 0.4399 | 1.0307 | 0.6596 | 0.3711 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 103 | 2 | 68 | 1 | 0.1654 | 0.0697 | -0.0106 | -0.0723 | 0.0304 | 0.4421 | 1.0304 | 0.6581 | 0.3723 | -0.0617 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 104 | 2 | 69 | 1 | 0.1654 | 0.0697 | -0.0081 | -0.0723 | 0.0301 | 0.4442 | 1.0301 | 0.6537 | 0.3764 | -0.0642 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 105 | 2 | 70 | 1 | 0.1654 | 0.0697 | -0.0079 | -0.0723 | 0.0298 | 0.4464 | 1.0298 | 0.6552 | 0.3746 | -0.0644 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 106 | 2 | 71 | 1 | 0.1654 | 0.0697 | -0.0064 | -0.0723 | 0.0296 | 0.4485 | 1.0296 | 0.6536 | 0.376 | -0.0659 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 107 | 2 | 72 | 1 | 0.1654 | 0.0697 | -0.0056 | -0.0723 | 0.0293 | 0.4506 | 1.0293 | 0.6536 | 0.3757 | -0.0667 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 108 | 2 | 73 | 1 | 0.1654 | 0.0697 | -0.0075 | -0.0723 | 0.029 | 0.4527 | 1.029 | 0.6605 | 0.3685 | -0.0648 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 109 | 2 | 74 | 1 | 0.1654 | 0.0697 | -0.0093 | -0.0723 | 0.0288 | 0.4548 | 1.0288 | 0.6673 | 0.3615 | -0.063 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 110 | 2 | 75 | 1 | 0.1654 | 0.0697 | -0.0092 | -0.0723 | 0.0285 | 0.4569 | 1.0285 | 0.6692 | 0.3594 | -0.0631 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 111 | 2 | 76 | 1 | 0.1654 | 0.0697 | -0.0106 | -0.0723 | 0.0283 | 0.4589 | 1.0283 | 0.6748 | 0.3535 | -0.0618 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 112 | 2 | 77 | 1 | 0.1654 | 0.0697 | -0.0127 | -0.0723 | 0.0281 | 0.461 | 1.0281 | 0.683 | 0.3451 | -0.0596 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 113 | 2 | 78 | 1 | 0.1654 | 0.0697 | -0.0142 | -0.0723 | 0.0278 | 0.463 | 1.0278 | 0.6894 | 0.3384 | -0.0581 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 114 | 2 | 79 | 1 | 0.1654 | 0.0697 | -0.0138 | -0.0723 | 0.0276 | 0.4651 | 1.0276 | 0.6903 | 0.3373 | -0.0585 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 115 | 2 | 80 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.0274 | 0.4671 | 1.0274 | 0.6932 | 0.3342 | -0.0582 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 116 | 2 | 81 | 1 | 0.1654 | 0.0697 | -0.015 | -0.0723 | 0.0272 | 0.4692 | 1.0272 | 0.6979 | 0.3292 | -0.0573 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 117 | 2 | 82 | 1 | 0.1654 | 0.0697 | -0.016 | -0.0723 | 0.0269 | 0.4712 | 1.0269 | 0.7031 | 0.3239 | -0.0563 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 118 | 2 | 83 | 1 | 0.1654 | 0.0697 | -0.0179 | -0.0723 | 0.0267 | 0.4732 | 1.0267 | 0.7112 | 0.3156 | -0.0544 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 119 | 2 | 84 | 1 | 0.1654 | 0.0697 | -0.0179 | -0.0723 | 0.0265 | 0.4752 | 1.0265 | 0.7131 | 0.3134 | -0.0545 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 120 | 2 | 85 | 1 | 0.1654 | 0.0697 | -0.0178 | -0.0723 | 0.0263 | 0.4772 | 1.0263 | 0.715 | 0.3113 | -0.0545 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 121 | 2 | 86 | 1 | 0.1654 | 0.0697 | -0.0165 | -0.0723 | 0.0261 | 0.4792 | 1.0261 | 0.7126 | 0.3135 | -0.0558 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 122 | 2 | 87 | 1 | 0.1654 | 0.0697 | -0.0172 | -0.0723 | 0.026 | 0.4811 | 1.026 | 0.7168 | 0.3092 | -0.0552 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 123 | 2 | 88 | 1 | 0.1654 | 0.0697 | -0.0182 | -0.0723 | 0.0258 | 0.4831 | 1.0258 | 0.7221 | 0.3037 | -0.0542 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 124 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0256 | 0.4851 | 1.0256 | 0.7256 | 0.3 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 125 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0257 | 0.487 | 1.0257 | 0.7276 | 0.2981 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 126 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0258 | 0.489 | 1.0258 | 0.7295 | 0.2963 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 127 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0259 | 0.4909 | 1.0259 | 0.7314 | 0.2945 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 128 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.026 | 0.4928 | 1.026 | 0.7334 | 0.2926 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 129 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0261 | 0.4947 | 1.0261 | 0.7353 | 0.2908 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 130 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0262 | 0.4967 | 0.9937 | 0.7294 | 0.2643 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 131 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0263 | 0.4986 | 0.9771 | 0.7273 | 0.2498 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 132 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0264 | 0.5005 | 0.9501 | 0.7226 | 0.2274 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 133 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0265 | 0.5024 | 0.9674 | 0.7287 | 0.2387 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 134 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0266 | 0.5042 | 0.9656 | 0.7301 | 0.2355 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 135 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0267 | 0.5061 | 0.9582 | 0.7302 | 0.228 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 136 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0268 | 0.508 | 0.9721 | 0.7354 | 0.2367 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 137 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0269 | 0.5099 | 1.0035 | 0.7448 | 0.2588 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 138 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.027 | 0.5117 | 0.9854 | 0.7422 | 0.2431 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 139 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0271 | 0.5136 | 0.956 | 0.737 | 0.219 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 140 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0272 | 0.5154 | 0.9789 | 0.7443 | 0.2346 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 141 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0273 | 0.5172 | 0.9996 | 0.7511 | 0.2485 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 142 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0274 | 0.5191 | 1.0229 | 0.7585 | 0.2643 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 143 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0275 | 0.5209 | 1.007 | 0.7565 | 0.2505 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 144 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0276 | 0.5227 | 1.0096 | 0.7589 | 0.2507 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 145 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0277 | 0.5245 | 1.0202 | 0.7633 | 0.2569 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 146 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0278 | 0.5263 | 1.0264 | 0.7666 | 0.2599 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 147 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0279 | 0.5281 | 1.0279 | 0.7687 | 0.2592 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 148 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0279 | 0.5299 | 1.0279 | 0.7705 | 0.2575 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 149 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.028 | 0.5317 | 1.028 | 0.7723 | 0.2558 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 150 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0281 | 0.5335 | 1.0138 | 0.7706 | 0.2432 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 151 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0282 | 0.5353 | 1.0041 | 0.77 | 0.2341 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 152 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0283 | 0.537 | 0.9995 | 0.7706 | 0.2288 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 153 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0284 | 0.5388 | 1.0094 | 0.7748 | 0.2346 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 154 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0285 | 0.5406 | 1.0285 | 0.7811 | 0.2474 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 155 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0286 | 0.5423 | 1.018 | 0.7803 | 0.2377 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 156 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0287 | 0.5441 | 1.0128 | 0.7808 | 0.232 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 157 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0288 | 0.5458 | 1.0197 | 0.7842 | 0.2356 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 158 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0289 | 0.5475 | 1.0049 | 0.7823 | 0.2226 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 159 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.029 | 0.5493 | 1.0019 | 0.7833 | 0.2186 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 160 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0291 | 0.551 | 0.9928 | 0.7828 | 0.21 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 161 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0292 | 0.5527 | 0.9828 | 0.7821 | 0.2007 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 162 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0292 | 0.5544 | 0.9762 | 0.7822 | 0.194 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 163 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0293 | 0.5561 | 0.974 | 0.7834 | 0.1906 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 164 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0294 | 0.5578 | 0.973 | 0.7848 | 0.1882 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 165 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0295 | 0.5595 | 0.964 | 0.7843 | 0.1797 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 166 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0296 | 0.5612 | 0.991 | 0.7925 | 0.1985 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 167 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0297 | 0.5629 | 0.9831 | 0.7922 | 0.1908 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 168 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0298 | 0.5646 | 0.9774 | 0.7925 | 0.1848 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 169 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0299 | 0.5663 | 0.9734 | 0.7932 | 0.1802 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 170 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.03 | 0.568 | 0.9513 | 0.7896 | 0.1617 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 171 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.03 | 0.5696 | 0.9442 | 0.7895 | 0.1546 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 172 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0301 | 0.5713 | 0.9407 | 0.7903 | 0.1504 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 173 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0302 | 0.5729 | 0.9425 | 0.7924 | 0.1501 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 174 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0303 | 0.5746 | 0.9337 | 0.7919 | 0.1418 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 175 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0304 | 0.5762 | 0.9282 | 0.7922 | 0.136 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 176 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0305 | 0.5779 | 0.9216 | 0.7922 | 0.1294 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 177 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0306 | 0.5795 | 0.9063 | 0.7902 | 0.1161 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 178 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0307 | 0.5812 | 0.9008 | 0.7905 | 0.1103 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 179 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0307 | 0.5828 | 0.8822 | 0.7876 | 0.0946 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 180 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0308 | 0.5844 | 0.8853 | 0.7899 | 0.0953 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 181 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0309 | 0.586 | 0.8975 | 0.7945 | 0.103 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 182 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.031 | 0.5877 | 0.9213 | 0.8018 | 0.1195 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 183 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0311 | 0.5893 | 0.9214 | 0.8034 | 0.1179 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 184 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0312 | 0.5909 | 0.9214 | 0.805 | 0.1164 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 185 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0312 | 0.5925 | 0.9215 | 0.8066 | 0.1149 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 186 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0313 | 0.5941 | 0.9216 | 0.8082 | 0.1134 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 187 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0314 | 0.5957 | 0.9217 | 0.8098 | 0.1119 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 188 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0315 | 0.5973 | 0.9218 | 0.8114 | 0.1104 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 189 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0316 | 0.5989 | 0.9398 | 0.8173 | 0.1225 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 190 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0317 | 0.6004 | 0.9331 | 0.8173 | 0.1158 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 191 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0318 | 0.602 | 0.9343 | 0.8191 | 0.1152 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 192 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0318 | 0.6036 | 0.9272 | 0.819 | 0.1083 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 193 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0319 | 0.6052 | 0.9282 | 0.8208 | 0.1075 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 194 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.032 | 0.6067 | 0.9295 | 0.8226 | 0.1069 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 195 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0321 | 0.6083 | 0.9294 | 0.8241 | 0.1053 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 196 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0322 | 0.6098 | 0.935 | 0.827 | 0.108 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 197 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0322 | 0.6114 | 0.9352 | 0.8286 | 0.1066 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 198 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0323 | 0.6129 | 0.9307 | 0.829 | 0.1017 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 199 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0324 | 0.6145 | 0.9304 | 0.8305 | 0.0999 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 200 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0325 | 0.616 | 0.9187 | 0.8292 | 0.0895 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 201 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0326 | 0.6176 | 0.9137 | 0.8295 | 0.0842 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 202 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0327 | 0.6191 | 0.9154 | 0.8314 | 0.084 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 203 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0327 | 0.6206 | 0.9133 | 0.8324 | 0.0808 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 204 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0328 | 0.6222 | 0.9189 | 0.8353 | 0.0836 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 205 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0329 | 0.6237 | 0.9215 | 0.8374 | 0.0841 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 206 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.033 | 0.6252 | 0.939 | 0.8431 | 0.0958 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 207 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0331 | 0.6267 | 0.9565 | 0.8488 | 0.1077 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 208 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0331 | 0.6282 | 0.9587 | 0.8509 | 0.1078 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 209 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0332 | 0.6297 | 0.9779 | 0.857 | 0.1209 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 210 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0333 | 0.6312 | 0.9887 | 0.8611 | 0.1276 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 211 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0334 | 0.6327 | 0.9956 | 0.8642 | 0.1314 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 212 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0335 | 0.6342 | 1.0141 | 0.8701 | 0.1439 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 213 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0335 | 0.6357 | 1.0141 | 0.8716 | 0.1425 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 214 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0336 | 0.6372 | 1.0142 | 0.8731 | 0.1411 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 215 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0337 | 0.6387 | 1.0311 | 0.8786 | 0.1525 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 216 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0338 | 0.6402 | 1.0338 | 0.8807 | 0.153 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 217 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0338 | 0.6417 | 1.0338 | 0.8822 | 0.1516 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 218 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0339 | 0.6432 | 1.0339 | 0.8837 | 0.1502 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 219 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.034 | 0.6446 | 1.0322 | 0.8847 | 0.1475 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 220 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0341 | 0.6461 | 1.0313 | 0.886 | 0.1454 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 221 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0342 | 0.6476 | 1.0306 | 0.8872 | 0.1433 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 222 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0342 | 0.649 | 1.0288 | 0.8883 | 0.1406 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 223 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0343 | 0.6505 | 1.0307 | 0.8902 | 0.1406 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 224 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0344 | 0.6519 | 1.0321 | 0.8919 | 0.1401 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 225 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0345 | 0.6534 | 1.0255 | 0.8918 | 0.1337 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 226 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0345 | 0.6548 | 1.0223 | 0.8925 | 0.1299 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 227 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0346 | 0.6563 | 1.0214 | 0.8937 | 0.1277 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 228 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0347 | 0.6577 | 1.0221 | 0.8952 | 0.1268 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 229 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0348 | 0.6592 | 1.0255 | 0.8975 | 0.128 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 230 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0348 | 0.6606 | 1.0281 | 0.8996 | 0.1286 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 231 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0349 | 0.6621 | 1.025 | 0.9002 | 0.1248 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 232 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.035 | 0.6635 | 1.0167 | 0.8996 | 0.1171 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 233 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0351 | 0.6649 | 1.0182 | 0.9014 | 0.1168 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 234 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0351 | 0.6663 | 1.0332 | 0.9064 | 0.1268 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 235 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0352 | 0.6678 | 1.0352 | 0.9083 | 0.1269 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 236 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0353 | 0.6692 | 1.0353 | 0.9097 | 0.1256 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 237 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0354 | 0.6706 | 1.0354 | 0.9111 | 0.1242 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 238 | 1 | 90 | 1 | 0.1654 | 0.0697 | -0.02 | -0.0723 | 0.0351 | 0.672 | 1.0336 | 0.9168 | 0.1168 | -0.0524 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 239 | 1 | 90 | 1 | 0.1654 | 0.0697 | -0.02 | -0.0723 | 0.0351 | 0.6734 | 1.0351 | 0.9186 | 0.1165 | -0.0524 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 240 | 1 | 90 | 1 | 0.1654 | 0.0697 | -0.02 | -0.0723 | 0.0352 | 0.6748 | 1.0352 | 0.92 | 0.1152 | -0.0524 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 241 | 1 | 91 | 1 | 0.1654 | 0.0697 | -0.0198 | -0.0723 | 0.0349 | 0.6762 | 1.0349 | 0.9209 | 0.114 | -0.0525 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 242 | 1 | 92 | 1 | 0.1654 | 0.0697 | -0.0197 | -0.0723 | 0.0346 | 0.6776 | 1.0346 | 0.9218 | 0.1128 | -0.0527 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 243 | 1 | 93 | 1 | 0.1654 | 0.0697 | -0.0197 | -0.0723 | 0.0343 | 0.679 | 1.0343 | 0.9233 | 0.111 | -0.0526 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 244 | 1 | 94 | 1 | 0.1654 | 0.0697 | -0.0201 | -0.0723 | 0.034 | 0.6804 | 1.034 | 0.926 | 0.108 | -0.0522 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 245 | 1 | 95 | 1 | 0.1654 | 0.0697 | -0.0207 | -0.0723 | 0.0337 | 0.6818 | 1.0337 | 0.9296 | 0.1041 | -0.0516 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 246 | 1 | 96 | 1 | 0.1654 | 0.0697 | -0.02 | -0.0723 | 0.0334 | 0.6832 | 1.0334 | 0.9286 | 0.1049 | -0.0523 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 247 | 1 | 97 | 1 | 0.1654 | 0.0697 | -0.0199 | -0.0723 | 0.0332 | 0.6846 | 1.0332 | 0.9294 | 0.1038 | -0.0525 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 248 | 1 | 98 | 1 | 0.1654 | 0.0697 | -0.0207 | -0.0723 | 0.0329 | 0.686 | 1.0329 | 0.9336 | 0.0993 | -0.0517 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 249 | 1 | 99 | 1 | 0.1654 | 0.0697 | -0.021 | -0.0723 | 0.0326 | 0.6874 | 1.0326 | 0.9361 | 0.0966 | -0.0514 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 250 | 1 | 100 | 1 | 0.1654 | 0.0697 | -0.0211 | -0.0723 | 0.0324 | 0.6887 | 1.0324 | 0.9378 | 0.0945 | -0.0513 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 251 | 1 | 101 | 1 | 0.1654 | 0.0697 | -0.0209 | -0.0723 | 0.0321 | 0.6901 | 1.0321 | 0.9386 | 0.0936 | -0.0514 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 252 | 1 | 102 | 1 | 0.1654 | 0.0697 | -0.0211 | -0.0723 | 0.0319 | 0.6915 | 1.0319 | 0.9408 | 0.0911 | -0.0512 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 253 | 1 | 103 | 1 | 0.1654 | 0.0697 | -0.0216 | -0.0723 | 0.0316 | 0.6929 | 1.0316 | 0.944 | 0.0876 | -0.0507 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 254 | 1 | 104 | 1 | 0.1654 | 0.0697 | -0.0203 | -0.0723 | 0.0314 | 0.6942 | 1.0314 | 0.9405 | 0.0909 | -0.0521 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 255 | 1 | 105 | 1 | 0.1654 | 0.0697 | -0.0196 | -0.0723 | 0.0311 | 0.6956 | 1.0311 | 0.9396 | 0.0916 | -0.0527 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 256 | 1 | 106 | 1 | 0.1654 | 0.0697 | -0.0191 | -0.0723 | 0.0309 | 0.697 | 1.0309 | 0.9389 | 0.092 | -0.0533 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 257 | 1 | 107 | 1 | 0.1654 | 0.0697 | -0.0202 | -0.0723 | 0.0307 | 0.6983 | 1.0307 | 0.9443 | 0.0864 | -0.0521 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 258 | 1 | 108 | 1 | 0.1654 | 0.0697 | -0.019 | -0.0723 | 0.0305 | 0.6997 | 1.0305 | 0.9416 | 0.0889 | -0.0533 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 259 | 1 | 109 | 1 | 0.1654 | 0.0697 | -0.018 | -0.0723 | 0.0303 | 0.701 | 1.0303 | 0.9395 | 0.0908 | -0.0543 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 260 | 1 | 110 | 1 | 0.1654 | 0.0697 | -0.0175 | -0.0723 | 0.03 | 0.7024 | 1.03 | 0.9391 | 0.0909 | -0.0548 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 261 | 1 | 111 | 1 | 0.1654 | 0.0697 | -0.0166 | -0.0723 | 0.0298 | 0.7037 | 1.0298 | 0.9375 | 0.0923 | -0.0557 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 262 | 1 | 112 | 1 | 0.1654 | 0.0697 | -0.0166 | -0.0723 | 0.0296 | 0.7051 | 1.0296 | 0.9387 | 0.0909 | -0.0558 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 263 | 1 | 113 | 1 | 0.1654 | 0.0697 | -0.0175 | -0.0723 | 0.0294 | 0.7064 | 1.0294 | 0.9432 | 0.0862 | -0.0548 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 264 | 1 | 114 | 1 | 0.1654 | 0.0697 | -0.0179 | -0.0723 | 0.0292 | 0.7078 | 1.0292 | 0.9457 | 0.0835 | -0.0545 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 265 | 1 | 115 | 1 | 0.1654 | 0.0697 | -0.0181 | -0.0723 | 0.029 | 0.7091 | 1.029 | 0.9477 | 0.0813 | -0.0543 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 266 | 1 | 116 | 1 | 0.1654 | 0.0697 | -0.0172 | -0.0723 | 0.0288 | 0.7104 | 1.0288 | 0.9462 | 0.0826 | -0.0551 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 267 | 1 | 117 | 1 | 0.1654 | 0.0697 | -0.0175 | -0.0723 | 0.0286 | 0.7118 | 1.0286 | 0.9484 | 0.0803 | -0.0549 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 268 | 1 | 118 | 1 | 0.1654 | 0.0697 | -0.0176 | -0.0723 | 0.0284 | 0.7131 | 1.0284 | 0.9503 | 0.0782 | -0.0547 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 269 | 1 | 119 | 1 | 0.1654 | 0.0697 | -0.0174 | -0.0723 | 0.0283 | 0.7144 | 1.0283 | 0.951 | 0.0773 | -0.0549 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 270 | 1 | 120 | 1 | 0.1654 | 0.0697 | -0.0185 | -0.0723 | 0.0281 | 0.7158 | 1.0281 | 0.9558 | 0.0723 | -0.0539 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 271 | 1 | 121 | 1 | 0.1654 | 0.0697 | -0.0179 | -0.0723 | 0.0279 | 0.7171 | 1.0279 | 0.9552 | 0.0727 | -0.0544 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 272 | 1 | 122 | 1 | 0.1654 | 0.0697 | -0.0181 | -0.0723 | 0.0277 | 0.7184 | 1.0277 | 0.957 | 0.0707 | -0.0543 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 273 | 1 | 123 | 1 | 0.1654 | 0.0697 | -0.0169 | -0.0723 | 0.0276 | 0.7197 | 1.0276 | 0.9545 | 0.073 | -0.0554 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 274 | 1 | 124 | 1 | 0.1654 | 0.0697 | -0.0179 | -0.0723 | 0.0274 | 0.721 | 1.0274 | 0.9591 | 0.0683 | -0.0544 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 275 | 1 | 125 | 1 | 0.1654 | 0.0697 | -0.0184 | -0.0723 | 0.0272 | 0.7224 | 1.0272 | 0.9619 | 0.0653 | -0.054 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 276 | 1 | 126 | 1 | 0.1654 | 0.0697 | -0.0185 | -0.0723 | 0.027 | 0.7237 | 1.027 | 0.9639 | 0.0632 | -0.0538 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 277 | 1 | 127 | 1 | 0.1654 | 0.0697 | -0.0173 | -0.0723 | 0.0269 | 0.725 | 1.0269 | 0.9609 | 0.066 | -0.0551 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 278 | 1 | 128 | 1 | 0.1654 | 0.0697 | -0.0165 | -0.0723 | 0.0267 | 0.7263 | 1.0267 | 0.9598 | 0.0669 | -0.0558 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 279 | 1 | 129 | 1 | 0.1654 | 0.0697 | -0.0153 | -0.0723 | 0.0266 | 0.7276 | 1.0266 | 0.9572 | 0.0694 | -0.0571 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 280 | 1 | 130 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.0264 | 0.7289 | 1.0264 | 0.955 | 0.0714 | -0.0582 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 281 | 1 | 131 | 1 | 0.1654 | 0.0697 | -0.0133 | -0.0723 | 0.0263 | 0.7302 | 1.0263 | 0.9538 | 0.0724 | -0.059 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 282 | 2 | 132 | 1 | 0.1654 | 0.0697 | -0.0128 | -0.0723 | 0.0261 | 0.7315 | 1.0261 | 0.9537 | 0.0724 | -0.0595 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 283 | 2 | 133 | 1 | 0.1654 | 0.0697 | -0.0128 | -0.0723 | 0.026 | 0.7328 | 1.026 | 0.955 | 0.071 | -0.0595 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 284 | 2 | 134 | 1 | 0.1654 | 0.0697 | -0.0132 | -0.0723 | 0.0258 | 0.7341 | 1.0258 | 0.9575 | 0.0683 | -0.0591 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 285 | 2 | 135 | 1 | 0.1654 | 0.0697 | -0.0127 | -0.0723 | 0.0257 | 0.7354 | 1.0257 | 0.9572 | 0.0685 | -0.0597 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 286 | 2 | 136 | 1 | 0.1654 | 0.0697 | -0.0123 | -0.0723 | 0.0255 | 0.7367 | 1.0255 | 0.9573 | 0.0682 | -0.0601 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 287 | 2 | 137 | 1 | 0.1654 | 0.0697 | -0.0118 | -0.0723 | 0.0254 | 0.738 | 1.0254 | 0.9573 | 0.0681 | -0.0605 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 288 | 2 | 138 | 1 | 0.1654 | 0.0697 | -0.0116 | -0.0723 | 0.0252 | 0.7392 | 1.0252 | 0.9581 | 0.0672 | -0.0607 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 289 | 2 | 139 | 1 | 0.1654 | 0.0697 | -0.0119 | -0.0723 | 0.0251 | 0.7405 | 1.0251 | 0.9602 | 0.0649 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 290 | 2 | 140 | 1 | 0.1654 | 0.0697 | -0.0119 | -0.0723 | 0.025 | 0.7418 | 1.025 | 0.9613 | 0.0637 | -0.0605 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 291 | 2 | 141 | 1 | 0.1654 | 0.0697 | -0.0119 | -0.0723 | 0.0248 | 0.7431 | 1.0248 | 0.9627 | 0.0621 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 292 | 2 | 142 | 1 | 0.1654 | 0.0697 | -0.0112 | -0.0723 | 0.0247 | 0.7444 | 1.0247 | 0.9621 | 0.0626 | -0.0611 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 293 | 2 | 143 | 1 | 0.1654 | 0.0697 | -0.0107 | -0.0723 | 0.0246 | 0.7456 | 1.0246 | 0.9618 | 0.0627 | -0.0616 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 294 | 2 | 144 | 1 | 0.1654 | 0.0697 | -0.01 | -0.0723 | 0.0245 | 0.7469 | 1.0245 | 0.9612 | 0.0632 | -0.0623 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 295 | 2 | 145 | 1 | 0.1654 | 0.0697 | -0.0094 | -0.0723 | 0.0243 | 0.7482 | 1.0243 | 0.961 | 0.0633 | -0.0629 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 296 | 2 | 146 | 1 | 0.1654 | 0.0697 | -0.0087 | -0.0723 | 0.0242 | 0.7494 | 1.0242 | 0.9604 | 0.0638 | -0.0636 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 297 | 2 | 147 | 1 | 0.1654 | 0.0697 | -0.0086 | -0.0723 | 0.0241 | 0.7507 | 1.0241 | 0.9614 | 0.0627 | -0.0637 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 298 | 2 | 148 | 1 | 0.1654 | 0.0697 | -0.0087 | -0.0723 | 0.024 | 0.752 | 1.024 | 0.9628 | 0.0612 | -0.0637 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 299 | 2 | 149 | 1 | 0.1654 | 0.0697 | -0.0082 | -0.0723 | 0.0238 | 0.7532 | 1.0238 | 0.9628 | 0.0611 | -0.0641 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 300 | 2 | 150 | 1 | 0.1654 | 0.0697 | -0.0075 | -0.0723 | 0.0237 | 0.7545 | 1.0237 | 0.9622 | 0.0615 | -0.0649 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 301 | 2 | 151 | 1 | 0.1654 | 0.0697 | -0.007 | -0.0723 | 0.0236 | 0.7557 | 1.0236 | 0.9623 | 0.0613 | -0.0653 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 302 | 2 | 152 | 1 | 0.1654 | 0.0697 | -0.0075 | -0.0723 | 0.0235 | 0.757 | 1.0235 | 0.9649 | 0.0586 | -0.0648 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 303 | 2 | 153 | 1 | 0.1654 | 0.0697 | -0.0081 | -0.0723 | 0.0234 | 0.7582 | 1.0234 | 0.9675 | 0.0558 | -0.0643 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 304 | 2 | 154 | 1 | 0.1654 | 0.0697 | -0.0083 | -0.0723 | 0.0233 | 0.7595 | 1.0233 | 0.9694 | 0.0538 | -0.064 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 305 | 2 | 155 | 1 | 0.1654 | 0.0697 | -0.0081 | -0.0723 | 0.0231 | 0.7607 | 1.0231 | 0.97 | 0.0531 | -0.0642 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 306 | 2 | 156 | 1 | 0.1654 | 0.0697 | -0.0075 | -0.0723 | 0.023 | 0.762 | 1.023 | 0.9697 | 0.0534 | -0.0649 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 307 | 2 | 157 | 1 | 0.1654 | 0.0697 | -0.0069 | -0.0723 | 0.0229 | 0.7632 | 1.0229 | 0.9696 | 0.0533 | -0.0654 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 308 | 2 | 158 | 1 | 0.1654 | 0.0697 | -0.0065 | -0.0723 | 0.0228 | 0.7645 | 1.0228 | 0.9698 | 0.0531 | -0.0658 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 309 | 2 | 159 | 1 | 0.1654 | 0.0697 | -0.0063 | -0.0723 | 0.0227 | 0.7657 | 1.0227 | 0.9704 | 0.0523 | -0.0661 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 310 | 2 | 160 | 1 | 0.1654 | 0.0697 | -0.006 | -0.0723 | 0.0226 | 0.767 | 1.0226 | 0.971 | 0.0516 | -0.0663 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 311 | 2 | 161 | 1 | 0.1654 | 0.0697 | -0.0057 | -0.0723 | 0.0225 | 0.7682 | 1.0225 | 0.9716 | 0.0509 | -0.0666 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 312 | 2 | 162 | 1 | 0.1654 | 0.0697 | -0.0062 | -0.0723 | 0.0224 | 0.7694 | 1.0224 | 0.9739 | 0.0485 | -0.0662 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 313 | 2 | 163 | 1 | 0.1654 | 0.0697 | -0.0062 | -0.0723 | 0.0223 | 0.7707 | 1.0223 | 0.9752 | 0.0471 | -0.0661 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 314 | 2 | 164 | 1 | 0.1654 | 0.0697 | -0.0061 | -0.0723 | 0.0222 | 0.7719 | 1.0222 | 0.9762 | 0.046 | -0.0662 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 315 | 2 | 165 | 1 | 0.1654 | 0.0697 | -0.0049 | -0.0723 | 0.0221 | 0.7731 | 1.0221 | 0.9744 | 0.0477 | -0.0675 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 316 | 2 | 166 | 1 | 0.1654 | 0.0697 | -0.005 | -0.0723 | 0.022 | 0.7743 | 1.022 | 0.9761 | 0.0459 | -0.0673 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 317 | 2 | 167 | 1 | 0.1654 | 0.0697 | -0.0047 | -0.0723 | 0.0219 | 0.7756 | 1.0219 | 0.9765 | 0.0454 | -0.0676 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 318 | 2 | 168 | 1 | 0.1654 | 0.0697 | -0.0041 | -0.0723 | 0.0218 | 0.7768 | 1.0218 | 0.9764 | 0.0455 | -0.0682 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 319 | 2 | 169 | 1 | 0.1654 | 0.0697 | -0.0044 | -0.0723 | 0.0217 | 0.778 | 1.0217 | 0.9783 | 0.0434 | -0.0679 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 320 | 2 | 170 | 1 | 0.1654 | 0.0697 | -0.0045 | -0.0723 | 0.0216 | 0.7792 | 1.0216 | 0.9796 | 0.042 | -0.0679 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 321 | 2 | 171 | 1 | 0.1654 | 0.0697 | -0.0052 | -0.0723 | 0.0215 | 0.7804 | 1.0215 | 0.9825 | 0.039 | -0.0671 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 322 | 2 | 172 | 1 | 0.1654 | 0.0697 | -0.0037 | -0.0723 | 0.0214 | 0.7817 | 1.0214 | 0.9803 | 0.0412 | -0.0686 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 323 | 2 | 173 | 1 | 0.1654 | 0.0697 | -0.0023 | -0.0723 | 0.0214 | 0.7829 | 1.0214 | 0.9783 | 0.0431 | -0.07 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 324 | 2 | 174 | 1 | 0.1654 | 0.0697 | -0.0016 | -0.0723 | 0.0213 | 0.7841 | 1.0213 | 0.978 | 0.0433 | -0.0707 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 325 | 2 | 175 | 1 | 0.1654 | 0.0697 | -0.001 | -0.0723 | 0.0212 | 0.7853 | 1.0212 | 0.9777 | 0.0435 | -0.0714 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 326 | 2 | 176 | 1 | 0.1654 | 0.0697 | -0.0004 | -0.0723 | 0.0211 | 0.7865 | 1.0211 | 0.9777 | 0.0433 | -0.0719 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 327 | 2 | 177 | 1 | 0.1654 | 0.0697 | 0.0003 | -0.0723 | 0.021 | 0.7877 | 1.021 | 0.9773 | 0.0437 | -0.0727 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 328 | 2 | 178 | 1 | 0.1654 | 0.0697 | 0.0004 | -0.0723 | 0.0209 | 0.7889 | 1.0209 | 0.9785 | 0.0424 | -0.0727 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 329 | 2 | 179 | 1 | 0.1654 | 0.0697 | -0.0001 | -0.0723 | 0.0208 | 0.7901 | 1.0208 | 0.9807 | 0.0402 | -0.0722 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 330 | 2 | 180 | 1 | 0.1654 | 0.0697 | -0.0004 | -0.0723 | 0.0208 | 0.7913 | 1.0208 | 0.9826 | 0.0381 | -0.0719 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 331 | 2 | 181 | 1 | 0.1654 | 0.0697 | -0.0006 | -0.0723 | 0.0207 | 0.7925 | 1.0207 | 0.9841 | 0.0365 | -0.0717 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 332 | 2 | 182 | 1 | 0.1654 | 0.0697 | -0.0007 | -0.0723 | 0.0206 | 0.7937 | 1.0206 | 0.9856 | 0.035 | -0.0716 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 333 | 2 | 183 | 1 | 0.1654 | 0.0697 | -0.0002 | -0.0723 | 0.0205 | 0.7949 | 1.0205 | 0.9856 | 0.0349 | -0.0722 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 334 | 2 | 184 | 1 | 0.1654 | 0.0697 | 0.0003 | -0.0723 | 0.0204 | 0.7961 | 1.0204 | 0.9858 | 0.0347 | -0.0726 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 335 | 2 | 185 | 1 | 0.1654 | 0.0697 | 0.0008 | -0.0723 | 0.0203 | 0.7973 | 1.0203 | 0.986 | 0.0344 | -0.0731 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 336 | 2 | 186 | 1 | 0.1654 | 0.0697 | 0.0016 | -0.0723 | 0.0203 | 0.7985 | 1.0203 | 0.9856 | 0.0347 | -0.0739 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 337 | 2 | 187 | 1 | 0.1654 | 0.0697 | 0.0019 | -0.0723 | 0.0202 | 0.7997 | 1.0202 | 0.986 | 0.0341 | -0.0742 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 338 | 2 | 188 | 1 | 0.1654 | 0.0697 | 0.0022 | -0.0723 | 0.0201 | 0.8008 | 1.0201 | 0.9866 | 0.0335 | -0.0745 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 339 | 2 | 189 | 1 | 0.1654 | 0.0697 | 0.0031 | -0.0723 | 0.02 | 0.802 | 1.02 | 0.9859 | 0.0341 | -0.0755 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 340 | 2 | 190 | 1 | 0.1654 | 0.0697 | 0.0032 | -0.0723 | 0.02 | 0.8032 | 1.02 | 0.9871 | 0.0329 | -0.0755 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 341 | 2 | 191 | 1 | 0.1654 | 0.0697 | 0.0039 | -0.0723 | 0.0199 | 0.8044 | 1.0199 | 0.9868 | 0.0331 | -0.0762 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 342 | 2 | 192 | 1 | 0.1654 | 0.0697 | 0.0037 | -0.0723 | 0.0198 | 0.8056 | 1.0198 | 0.9883 | 0.0315 | -0.0761 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 343 | 2 | 193 | 1 | 0.1654 | 0.0697 | 0.0042 | -0.0723 | 0.0197 | 0.8067 | 1.0197 | 0.9885 | 0.0312 | -0.0766 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 344 | 2 | 194 | 1 | 0.1654 | 0.0697 | 0.005 | -0.0723 | 0.0197 | 0.8079 | 1.0197 | 0.9882 | 0.0314 | -0.0773 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 345 | 2 | 195 | 1 | 0.1654 | 0.0697 | 0.0045 | -0.0723 | 0.0196 | 0.8091 | 1.0196 | 0.9903 | 0.0293 | -0.0769 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 346 | 2 | 196 | 1 | 0.1654 | 0.0697 | 0.0044 | -0.0723 | 0.0195 | 0.8103 | 1.0195 | 0.9917 | 0.0278 | -0.0767 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 347 | 2 | 197 | 1 | 0.1654 | 0.0697 | 0.0051 | -0.0723 | 0.0195 | 0.8114 | 1.0195 | 0.9915 | 0.028 | -0.0774 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 348 | 2 | 198 | 1 | 0.1654 | 0.0697 | 0.0064 | -0.0723 | 0.0194 | 0.8126 | 1.0194 | 0.9903 | 0.0291 | -0.0787 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 349 | 2 | 199 | 1 | 0.1654 | 0.0697 | 0.0066 | -0.0723 | 0.0193 | 0.8138 | 1.0193 | 0.9909 | 0.0284 | -0.079 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 350 | 2 | 200 | 1 | 0.1654 | 0.0697 | 0.0069 | -0.0723 | 0.0192 | 0.8149 | 1.0192 | 0.9917 | 0.0276 | -0.0792 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 351 | 2 | 201 | 1 | 0.1654 | 0.0697 | 0.0077 | -0.0723 | 0.0192 | 0.8161 | 1.0192 | 0.9914 | 0.0278 | -0.08 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 352 | 2 | 202 | 1 | 0.1654 | 0.0697 | 0.0086 | -0.0723 | 0.0191 | 0.8173 | 1.0191 | 0.9908 | 0.0283 | -0.081 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 353 | 2 | 203 | 1 | 0.1654 | 0.0697 | 0.0085 | -0.0723 | 0.019 | 0.8184 | 1.019 | 0.9923 | 0.0267 | -0.0808 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 354 | 2 | 204 | 1 | 0.1654 | 0.0697 | 0.0084 | -0.0723 | 0.019 | 0.8196 | 1.019 | 0.9935 | 0.0255 | -0.0808 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 355 | 2 | 205 | 1 | 0.1654 | 0.0697 | 0.009 | -0.0723 | 0.0189 | 0.8207 | 1.0189 | 0.9937 | 0.0252 | -0.0813 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 356 | 2 | 206 | 1 | 0.1654 | 0.0697 | 0.0098 | -0.0723 | 0.0188 | 0.8219 | 1.0188 | 0.9935 | 0.0254 | -0.0821 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 357 | 2 | 207 | 1 | 0.1654 | 0.0697 | 0.0104 | -0.0723 | 0.0188 | 0.823 | 1.0188 | 0.9936 | 0.0252 | -0.0827 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 358 | 2 | 208 | 1 | 0.1654 | 0.0697 | 0.0103 | -0.0723 | 0.0187 | 0.8242 | 1.0187 | 0.9949 | 0.0238 | -0.0826 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 359 | 2 | 209 | 1 | 0.1654 | 0.0697 | 0.0116 | -0.0723 | 0.0187 | 0.8253 | 1.0187 | 0.9939 | 0.0248 | -0.0839 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 360 | 2 | 210 | 1 | 0.1654 | 0.0697 | 0.0125 | -0.0723 | 0.0186 | 0.8265 | 1.0186 | 0.9934 | 0.0252 | -0.0849 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 361 | 2 | 211 | 1 | 0.1654 | 0.0697 | 0.0141 | -0.0723 | 0.0185 | 0.8276 | 1.0185 | 0.9921 | 0.0264 | -0.0864 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 362 | 2 | 212 | 1 | 0.1654 | 0.0697 | 0.0145 | -0.0723 | 0.0185 | 0.8288 | 1.0185 | 0.9926 | 0.0259 | -0.0868 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 363 | 2 | 213 | 1 | 0.1654 | 0.0697 | 0.0149 | -0.0723 | 0.0184 | 0.8299 | 1.0184 | 0.993 | 0.0254 | -0.0873 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 364 | 2 | 214 | 1 | 0.1654 | 0.0697 | 0.0146 | -0.0723 | 0.0183 | 0.8311 | 1.0183 | 0.9947 | 0.0236 | -0.0869 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 365 | 2 | 215 | 1 | 0.1654 | 0.0697 | 0.0151 | -0.0723 | 0.0183 | 0.8322 | 1.0183 | 0.995 | 0.0233 | -0.0874 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 366 | 2 | 216 | 1 | 0.1654 | 0.0697 | 0.0144 | -0.0723 | 0.0182 | 0.8334 | 1.0182 | 0.9972 | 0.021 | -0.0868 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 367 | 2 | 217 | 1 | 0.1654 | 0.0697 | 0.0157 | -0.0723 | 0.0182 | 0.8345 | 1.0182 | 0.9964 | 0.0217 | -0.088 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 368 | 2 | 218 | 1 | 0.1654 | 0.0697 | 0.0154 | -0.0723 | 0.0181 | 0.8356 | 1.0181 | 0.998 | 0.0201 | -0.0877 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 369 | 2 | 219 | 1 | 0.1654 | 0.0697 | 0.0151 | -0.0723 | 0.0181 | 0.8368 | 1.0181 | 0.9996 | 0.0184 | -0.0874 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 370 | 2 | 220 | 1 | 0.1654 | 0.0697 | 0.017 | -0.0723 | 0.018 | 0.8379 | 1.018 | 0.9979 | 0.0201 | -0.0893 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 371 | 2 | 221 | 1 | 0.1654 | 0.0697 | 0.0166 | -0.0723 | 0.0179 | 0.839 | 1.0179 | 0.9995 | 0.0184 | -0.089 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 372 | 2 | 222 | 1 | 0.1654 | 0.0697 | 0.0162 | -0.0723 | 0.0179 | 0.8402 | 1.0179 | 1.0012 | 0.0166 | -0.0886 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 373 | 2 | 223 | 1 | 0.1654 | 0.0697 | 0.0159 | -0.0723 | 0.0178 | 0.8413 | 1.0178 | 1.0029 | 0.0149 | -0.0882 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 374 | 2 | 224 | 1 | 0.1654 | 0.0697 | 0.0165 | -0.0723 | 0.0178 | 0.8424 | 1.0178 | 1.0031 | 0.0147 | -0.0889 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 375 | 2 | 225 | 1 | 0.1654 | 0.0697 | 0.0172 | -0.0723 | 0.0177 | 0.8435 | 1.0177 | 1.0032 | 0.0145 | -0.0895 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 376 | 2 | 226 | 1 | 0.1654 | 0.0697 | 0.017 | -0.0723 | 0.0177 | 0.8447 | 1.0177 | 1.0046 | 0.013 | -0.0893 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 377 | 2 | 227 | 1 | 0.1654 | 0.0697 | 0.0166 | -0.0723 | 0.0176 | 0.8458 | 1.0176 | 1.0063 | 0.0114 | -0.089 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 378 | 2 | 228 | 1 | 0.1654 | 0.0697 | 0.0161 | -0.0723 | 0.0176 | 0.8469 | 1.0176 | 1.0081 | 0.0094 | -0.0885 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 379 | 2 | 229 | 1 | 0.1654 | 0.0697 | 0.0163 | -0.0723 | 0.0175 | 0.848 | 1.0175 | 1.0091 | 0.0084 | -0.0886 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 380 | 2 | 230 | 1 | 0.1654 | 0.0697 | 0.0161 | -0.0723 | 0.0174 | 0.8491 | 1.0174 | 1.0104 | 0.0071 | -0.0885 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 381 | 2 | 231 | 1 | 0.1654 | 0.0697 | 0.0172 | -0.0723 | 0.0174 | 0.8503 | 1.0174 | 1.0099 | 0.0075 | -0.0895 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 382 | 2 | 232 | 1 | 0.1654 | 0.0697 | 0.0167 | -0.0723 | 0.0173 | 0.8514 | 1.0173 | 1.0117 | 0.0056 | -0.0891 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 383 | 2 | 233 | 1 | 0.1654 | 0.0697 | 0.0165 | -0.0723 | 0.0173 | 0.8525 | 1.0173 | 1.0131 | 0.0042 | -0.0889 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 384 | 2 | 234 | 1 | 0.1654 | 0.0697 | 0.0159 | -0.0723 | 0.0172 | 0.8536 | 1.0172 | 1.0151 | 0.0021 | -0.0883 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1 | 0 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.2068 | 0.0871 | 0.1171 | -0.0026 | 0.1197 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 2 | 1 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.2924 | 0.1232 | 0.7924 | 0.6232 | 0.1692 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 3 | 1 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.3581 | 0.1509 | 1.0248 | 0.8176 | 0.2072 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 4 | 1 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.4135 | 0.1742 | 1.1635 | 0.9242 | 0.2393 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 5 | 1 | 0 | 0 | 0.1654 | 0.0697 | 0.0 | 0.0 | 0.4624 | 0.1948 | 1.2624 | 0.9948 | 0.2675 | 0.0 | move_2_score_gt_move_4 |
| guarded-w2 | 6 | 1 | 1 | 0 | 0.1654 | 0.0697 | 0.0613 | 0.0 | 0.2532 | 0.2134 | 1.2532 | 0.9112 | 0.3421 | -0.0613 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 7 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.1824 | 0.2305 | 0.8111 | 0.9816 | -0.1705 | 0.0141 | - |
| guarded-w2 | 8 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.1949 | 0.2464 | 0.7905 | 1.0203 | -0.2298 | 0.0141 | - |
| guarded-w2 | 9 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.2068 | 0.2614 | 0.8101 | 1.0621 | -0.252 | 0.0141 | - |
| guarded-w2 | 10 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.218 | 0.2755 | 0.8513 | 1.1022 | -0.2509 | 0.0141 | - |
| guarded-w2 | 11 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.2286 | 0.2889 | 0.9247 | 1.1428 | -0.2181 | 0.0141 | - |
| guarded-w2 | 12 | 1 | 2 | 0 | 0.1654 | 0.0697 | -0.0141 | 0.0 | 0.2388 | 0.3018 | 1.001 | 1.1788 | -0.1778 | 0.0141 | - |
| guarded-w2 | 13 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2485 | 0.1571 | 1.0108 | 0.332 | 0.6788 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 14 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2579 | 0.163 | 0.9741 | 0.3273 | 0.6468 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 15 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2669 | 0.1687 | 0.9785 | 0.332 | 0.6465 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 16 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2757 | 0.1742 | 0.9199 | 0.3221 | 0.5979 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 17 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2842 | 0.1796 | 0.9007 | 0.3211 | 0.5797 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 18 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.2924 | 0.1848 | 0.946 | 0.3348 | 0.6112 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 19 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3004 | 0.1899 | 1.0251 | 0.3562 | 0.669 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 20 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3082 | 0.1948 | 1.0647 | 0.3684 | 0.6963 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 21 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3158 | 0.1996 | 1.0212 | 0.1996 | 0.8216 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 22 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3233 | 0.2043 | 1.0286 | 0.2043 | 0.8243 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 23 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3305 | 0.2089 | 1.0359 | 0.2089 | 0.827 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 24 | 1 | 2 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.3377 | 0.2134 | 1.1206 | 0.2134 | 0.9072 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 25 | 1 | 3 | 1 | 0.1654 | 0.0697 | -0.0146 | -0.0723 | 0.2585 | 0.2178 | 1.034 | 0.2178 | 0.8162 | -0.0577 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 26 | 1 | 3 | 1 | 0.1654 | 0.0697 | -0.0146 | -0.0723 | 0.2636 | 0.2221 | 1.0788 | 0.2221 | 0.8567 | -0.0577 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 27 | 1 | 3 | 1 | 0.1654 | 0.0697 | -0.0146 | -0.0723 | 0.2686 | 0.2263 | 1.1196 | 0.42 | 0.6996 | -0.0577 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 28 | 1 | 4 | 1 | 0.1654 | 0.0697 | 0.0036 | -0.0723 | 0.2188 | 0.2305 | 1.2188 | 0.4135 | 0.8053 | -0.0759 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 29 | 1 | 5 | 1 | 0.1654 | 0.0697 | -0.014 | -0.0723 | 0.1856 | 0.2346 | 1.044 | 0.4283 | 0.6158 | -0.0584 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 30 | 1 | 5 | 1 | 0.1654 | 0.0697 | -0.014 | -0.0723 | 0.1888 | 0.2386 | 1.0472 | 0.4323 | 0.6149 | -0.0584 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 31 | 1 | 5 | 1 | 0.1654 | 0.0697 | -0.014 | -0.0723 | 0.1919 | 0.2425 | 1.1177 | 0.4514 | 0.6663 | -0.0584 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 32 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1671 | 0.2464 | 1.0028 | 0.4553 | 0.5475 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 33 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1697 | 0.2502 | 1.0347 | 0.4664 | 0.5683 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 34 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1722 | 0.254 | 1.0574 | 0.4752 | 0.5821 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 35 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1748 | 0.2577 | 1.0835 | 0.4848 | 0.5987 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 36 | 1 | 6 | 1 | 0.1654 | 0.0697 | -0.0213 | -0.0723 | 0.1772 | 0.2614 | 1.1473 | 0.5038 | 0.6435 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 37 | 1 | 7 | 1 | 0.1654 | 0.0697 | -0.0089 | -0.0723 | 0.1572 | 0.265 | 1.1572 | 0.4763 | 0.6809 | -0.0635 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 38 | 1 | 8 | 1 | 0.1654 | 0.0697 | 0.0077 | -0.0723 | 0.1416 | 0.2685 | 1.1416 | 0.4437 | 0.6979 | -0.0801 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 39 | 1 | 9 | 1 | 0.1654 | 0.0697 | 0.0072 | -0.0723 | 0.1291 | 0.272 | 1.1291 | 0.4482 | 0.6809 | -0.0795 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 40 | 1 | 10 | 1 | 0.1654 | 0.0697 | 0.008 | -0.0723 | 0.1189 | 0.2755 | 1.1189 | 0.4502 | 0.6687 | -0.0803 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 41 | 1 | 11 | 1 | 0.1654 | 0.0697 | 0.0081 | -0.0723 | 0.1103 | 0.2789 | 1.1103 | 0.4535 | 0.6569 | -0.0804 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 42 | 1 | 12 | 1 | 0.1654 | 0.0697 | 0.0081 | -0.0723 | 0.1031 | 0.2823 | 1.1031 | 0.4568 | 0.6463 | -0.0805 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 43 | 1 | 13 | 1 | 0.1654 | 0.0697 | 0.0088 | -0.0723 | 0.0968 | 0.2856 | 1.0968 | 0.459 | 0.6379 | -0.0811 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 44 | 1 | 14 | 1 | 0.1654 | 0.0697 | 0.0127 | -0.0723 | 0.0914 | 0.2889 | 1.0914 | 0.4556 | 0.6358 | -0.085 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 45 | 1 | 15 | 1 | 0.1654 | 0.0697 | 0.0112 | -0.0723 | 0.0867 | 0.2922 | 1.0867 | 0.4613 | 0.6254 | -0.0836 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 46 | 1 | 16 | 1 | 0.1654 | 0.0697 | 0.009 | -0.0723 | 0.0825 | 0.2954 | 1.0825 | 0.4684 | 0.6141 | -0.0813 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 47 | 1 | 17 | 1 | 0.1654 | 0.0697 | 0.019 | -0.0723 | 0.0788 | 0.2986 | 1.0788 | 0.4557 | 0.6231 | -0.0913 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 48 | 1 | 18 | 1 | 0.1654 | 0.0697 | 0.0281 | -0.0723 | 0.0754 | 0.3018 | 1.0754 | 0.4466 | 0.6288 | -0.1005 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 49 | 1 | 19 | 1 | 0.1654 | 0.0697 | 0.029 | -0.0723 | 0.0724 | 0.3049 | 1.0724 | 0.4486 | 0.6238 | -0.1014 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 50 | 1 | 20 | 1 | 0.1654 | 0.0697 | 0.0288 | -0.0723 | 0.0696 | 0.308 | 1.0696 | 0.452 | 0.6177 | -0.1011 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 51 | 1 | 21 | 1 | 0.1654 | 0.0697 | 0.0289 | -0.0723 | 0.0671 | 0.3111 | 1.0671 | 0.4549 | 0.6122 | -0.1012 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 52 | 1 | 22 | 1 | 0.1654 | 0.0697 | 0.0289 | -0.0723 | 0.0648 | 0.3141 | 1.0648 | 0.458 | 0.6068 | -0.1012 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 53 | 2 | 23 | 1 | 0.1654 | 0.0697 | 0.0218 | -0.0723 | 0.0627 | 0.3171 | 1.0627 | 0.4701 | 0.5926 | -0.0941 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 54 | 2 | 24 | 1 | 0.1654 | 0.0697 | 0.0196 | -0.0723 | 0.0608 | 0.3201 | 1.0608 | 0.4762 | 0.5845 | -0.0919 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 55 | 2 | 25 | 1 | 0.1654 | 0.0697 | 0.0239 | -0.0723 | 0.059 | 0.323 | 1.059 | 0.4733 | 0.5857 | -0.0962 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 56 | 2 | 26 | 1 | 0.1654 | 0.0697 | 0.0186 | -0.0723 | 0.0573 | 0.326 | 1.0573 | 0.4835 | 0.5738 | -0.091 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 57 | 2 | 27 | 1 | 0.1654 | 0.0697 | 0.0165 | -0.0723 | 0.0558 | 0.3289 | 1.0558 | 0.4896 | 0.5661 | -0.0888 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 58 | 2 | 28 | 1 | 0.1654 | 0.0697 | 0.0125 | -0.0723 | 0.0543 | 0.3317 | 1.0543 | 0.4987 | 0.5556 | -0.0848 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 59 | 2 | 29 | 1 | 0.1654 | 0.0697 | 0.0099 | -0.0723 | 0.0529 | 0.3346 | 1.0529 | 0.506 | 0.547 | -0.0822 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 60 | 2 | 30 | 1 | 0.1654 | 0.0697 | 0.0061 | -0.0723 | 0.0517 | 0.3374 | 1.0517 | 0.5156 | 0.536 | -0.0784 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 61 | 2 | 30 | 1 | 0.1654 | 0.0697 | 0.0061 | -0.0723 | 0.0521 | 0.3402 | 1.0521 | 0.5184 | 0.5337 | -0.0784 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 62 | 2 | 31 | 1 | 0.1654 | 0.0697 | 0.0031 | -0.0723 | 0.0509 | 0.343 | 1.0509 | 0.527 | 0.5239 | -0.0754 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 63 | 2 | 32 | 1 | 0.1654 | 0.0697 | 0.0128 | -0.0723 | 0.0497 | 0.3457 | 1.0497 | 0.5122 | 0.5375 | -0.0851 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 64 | 2 | 33 | 1 | 0.1654 | 0.0697 | 0.0128 | -0.0723 | 0.0487 | 0.3485 | 1.0487 | 0.5149 | 0.5337 | -0.0852 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 65 | 2 | 34 | 1 | 0.1654 | 0.0697 | 0.0154 | -0.0723 | 0.0476 | 0.3512 | 1.0476 | 0.5136 | 0.534 | -0.0877 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 66 | 2 | 35 | 1 | 0.1654 | 0.0697 | 0.0225 | -0.0723 | 0.0467 | 0.3539 | 1.0467 | 0.506 | 0.5407 | -0.0948 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 67 | 2 | 36 | 1 | 0.1654 | 0.0697 | 0.0235 | -0.0723 | 0.0457 | 0.3566 | 1.0457 | 0.5073 | 0.5385 | -0.0958 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 68 | 2 | 37 | 1 | 0.1654 | 0.0697 | 0.0227 | -0.0723 | 0.0449 | 0.3592 | 1.0449 | 0.5109 | 0.5339 | -0.0951 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 69 | 2 | 38 | 1 | 0.1654 | 0.0697 | 0.0198 | -0.0723 | 0.044 | 0.3618 | 1.044 | 0.5176 | 0.5264 | -0.0922 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 70 | 2 | 39 | 1 | 0.1654 | 0.0697 | 0.0182 | -0.0723 | 0.0432 | 0.3644 | 1.0432 | 0.5226 | 0.5206 | -0.0905 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 71 | 2 | 40 | 1 | 0.1654 | 0.0697 | 0.0148 | -0.0723 | 0.0425 | 0.367 | 1.0425 | 0.5303 | 0.5122 | -0.0872 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 72 | 2 | 41 | 1 | 0.1654 | 0.0697 | 0.0118 | -0.0723 | 0.0418 | 0.3696 | 1.0418 | 0.5377 | 0.5041 | -0.0842 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 73 | 2 | 42 | 1 | 0.1654 | 0.0697 | 0.0089 | -0.0723 | 0.0411 | 0.3722 | 1.0411 | 0.5453 | 0.4958 | -0.0812 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 74 | 2 | 43 | 1 | 0.1654 | 0.0697 | 0.0091 | -0.0723 | 0.0404 | 0.3747 | 1.0404 | 0.5475 | 0.4929 | -0.0814 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 75 | 2 | 44 | 1 | 0.1654 | 0.0697 | 0.0074 | -0.0723 | 0.0398 | 0.3772 | 1.0398 | 0.553 | 0.4868 | -0.0797 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 76 | 2 | 45 | 1 | 0.1654 | 0.0697 | 0.0049 | -0.0723 | 0.0392 | 0.3797 | 1.0392 | 0.5602 | 0.479 | -0.0772 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 77 | 2 | 46 | 1 | 0.1654 | 0.0697 | 0.0022 | -0.0723 | 0.0386 | 0.3822 | 1.0386 | 0.568 | 0.4706 | -0.0745 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 78 | 2 | 47 | 1 | 0.1654 | 0.0697 | 0.0006 | -0.0723 | 0.038 | 0.3847 | 1.038 | 0.5738 | 0.4642 | -0.0729 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 79 | 2 | 48 | 1 | 0.1654 | 0.0697 | 0.0002 | -0.0723 | 0.0375 | 0.3872 | 1.0375 | 0.5771 | 0.4604 | -0.0725 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 80 | 2 | 49 | 1 | 0.1654 | 0.0697 | -0.0013 | -0.0723 | 0.037 | 0.3896 | 1.037 | 0.5827 | 0.4543 | -0.0711 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 81 | 2 | 50 | 1 | 0.1654 | 0.0697 | -0.0027 | -0.0723 | 0.0365 | 0.392 | 1.0365 | 0.5883 | 0.4482 | -0.0696 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 82 | 2 | 51 | 1 | 0.1654 | 0.0697 | -0.0017 | -0.0723 | 0.036 | 0.3945 | 1.036 | 0.5886 | 0.4474 | -0.0706 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 83 | 2 | 52 | 1 | 0.1654 | 0.0697 | -0.0034 | -0.0723 | 0.0355 | 0.3969 | 1.0355 | 0.5947 | 0.4408 | -0.0689 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 84 | 2 | 52 | 1 | 0.1654 | 0.0697 | -0.0034 | -0.0723 | 0.0358 | 0.3992 | 1.0358 | 0.5971 | 0.4386 | -0.0689 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 85 | 2 | 53 | 1 | 0.1654 | 0.0697 | -0.0036 | -0.0723 | 0.0353 | 0.4016 | 1.0353 | 0.5999 | 0.4354 | -0.0688 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 86 | 2 | 54 | 1 | 0.1654 | 0.0697 | -0.0043 | -0.0723 | 0.0349 | 0.404 | 1.0349 | 0.6038 | 0.431 | -0.0681 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 87 | 2 | 55 | 1 | 0.1654 | 0.0697 | -0.0059 | -0.0723 | 0.0344 | 0.4063 | 1.0344 | 0.6102 | 0.4242 | -0.0664 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 88 | 2 | 56 | 1 | 0.1654 | 0.0697 | -0.0073 | -0.0723 | 0.034 | 0.4086 | 1.034 | 0.6159 | 0.4181 | -0.065 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 89 | 2 | 57 | 1 | 0.1654 | 0.0697 | -0.0087 | -0.0723 | 0.0336 | 0.4109 | 1.0336 | 0.6217 | 0.4119 | -0.0637 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 90 | 2 | 58 | 1 | 0.1654 | 0.0697 | -0.007 | -0.0723 | 0.0332 | 0.4132 | 1.0332 | 0.6197 | 0.4135 | -0.0654 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 91 | 2 | 59 | 1 | 0.1654 | 0.0697 | -0.0072 | -0.0723 | 0.0329 | 0.4155 | 1.0329 | 0.6226 | 0.4103 | -0.0651 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 92 | 2 | 60 | 1 | 0.1654 | 0.0697 | -0.0081 | -0.0723 | 0.0325 | 0.4178 | 1.0325 | 0.6271 | 0.4055 | -0.0643 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 93 | 2 | 61 | 1 | 0.1654 | 0.0697 | -0.0091 | -0.0723 | 0.0322 | 0.4201 | 1.0322 | 0.6319 | 0.4003 | -0.0633 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 94 | 2 | 61 | 1 | 0.1654 | 0.0697 | -0.0091 | -0.0723 | 0.0323 | 0.4223 | 1.0323 | 0.6342 | 0.3982 | -0.0633 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 95 | 2 | 62 | 1 | 0.1654 | 0.0697 | -0.0102 | -0.0723 | 0.032 | 0.4246 | 1.032 | 0.6396 | 0.3924 | -0.0621 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 96 | 2 | 63 | 1 | 0.1654 | 0.0697 | -0.0085 | -0.0723 | 0.0317 | 0.4268 | 1.0317 | 0.6373 | 0.3944 | -0.0638 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 97 | 2 | 64 | 1 | 0.1654 | 0.0697 | -0.0092 | -0.0723 | 0.0313 | 0.429 | 1.0313 | 0.6412 | 0.3902 | -0.0632 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 98 | 2 | 65 | 1 | 0.1654 | 0.0697 | -0.0096 | -0.0723 | 0.031 | 0.4312 | 1.031 | 0.6445 | 0.3866 | -0.0627 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 99 | 2 | 66 | 1 | 0.1654 | 0.0697 | -0.0111 | -0.0723 | 0.0307 | 0.4334 | 1.0307 | 0.6508 | 0.3799 | -0.0612 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 100 | 2 | 67 | 1 | 0.1654 | 0.0697 | -0.0119 | -0.0723 | 0.0304 | 0.4356 | 1.0304 | 0.6553 | 0.3751 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 101 | 2 | 67 | 1 | 0.1654 | 0.0697 | -0.0119 | -0.0723 | 0.0306 | 0.4378 | 1.0306 | 0.6575 | 0.3731 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 102 | 2 | 67 | 1 | 0.1654 | 0.0697 | -0.0119 | -0.0723 | 0.0307 | 0.4399 | 1.0307 | 0.6596 | 0.3711 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 103 | 2 | 68 | 1 | 0.1654 | 0.0697 | -0.0106 | -0.0723 | 0.0304 | 0.4421 | 1.0304 | 0.6581 | 0.3723 | -0.0617 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 104 | 2 | 69 | 1 | 0.1654 | 0.0697 | -0.0081 | -0.0723 | 0.0301 | 0.4442 | 1.0301 | 0.6537 | 0.3764 | -0.0642 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 105 | 2 | 70 | 1 | 0.1654 | 0.0697 | -0.0079 | -0.0723 | 0.0298 | 0.4464 | 1.0298 | 0.6552 | 0.3746 | -0.0644 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 106 | 2 | 71 | 1 | 0.1654 | 0.0697 | -0.0064 | -0.0723 | 0.0296 | 0.4485 | 1.0296 | 0.6536 | 0.376 | -0.0659 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 107 | 2 | 72 | 1 | 0.1654 | 0.0697 | -0.0056 | -0.0723 | 0.0293 | 0.4506 | 1.0293 | 0.6536 | 0.3757 | -0.0667 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 108 | 2 | 73 | 1 | 0.1654 | 0.0697 | -0.0075 | -0.0723 | 0.029 | 0.4527 | 1.029 | 0.6605 | 0.3685 | -0.0648 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 109 | 2 | 74 | 1 | 0.1654 | 0.0697 | -0.0093 | -0.0723 | 0.0288 | 0.4548 | 1.0288 | 0.6673 | 0.3615 | -0.063 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 110 | 2 | 75 | 1 | 0.1654 | 0.0697 | -0.0092 | -0.0723 | 0.0285 | 0.4569 | 1.0285 | 0.6692 | 0.3594 | -0.0631 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 111 | 2 | 76 | 1 | 0.1654 | 0.0697 | -0.0106 | -0.0723 | 0.0283 | 0.4589 | 1.0283 | 0.6748 | 0.3535 | -0.0618 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 112 | 2 | 77 | 1 | 0.1654 | 0.0697 | -0.0127 | -0.0723 | 0.0281 | 0.461 | 1.0281 | 0.683 | 0.3451 | -0.0596 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 113 | 2 | 78 | 1 | 0.1654 | 0.0697 | -0.0142 | -0.0723 | 0.0278 | 0.463 | 1.0278 | 0.6894 | 0.3384 | -0.0581 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 114 | 2 | 79 | 1 | 0.1654 | 0.0697 | -0.0138 | -0.0723 | 0.0276 | 0.4651 | 1.0276 | 0.6903 | 0.3373 | -0.0585 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 115 | 2 | 80 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.0274 | 0.4671 | 1.0274 | 0.6932 | 0.3342 | -0.0582 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 116 | 2 | 81 | 1 | 0.1654 | 0.0697 | -0.015 | -0.0723 | 0.0272 | 0.4692 | 1.0272 | 0.6979 | 0.3292 | -0.0573 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 117 | 2 | 82 | 1 | 0.1654 | 0.0697 | -0.016 | -0.0723 | 0.0269 | 0.4712 | 1.0269 | 0.7031 | 0.3239 | -0.0563 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 118 | 2 | 83 | 1 | 0.1654 | 0.0697 | -0.0179 | -0.0723 | 0.0267 | 0.4732 | 1.0267 | 0.7112 | 0.3156 | -0.0544 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 119 | 2 | 84 | 1 | 0.1654 | 0.0697 | -0.0179 | -0.0723 | 0.0265 | 0.4752 | 1.0265 | 0.7131 | 0.3134 | -0.0545 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 120 | 2 | 85 | 1 | 0.1654 | 0.0697 | -0.0178 | -0.0723 | 0.0263 | 0.4772 | 1.0263 | 0.715 | 0.3113 | -0.0545 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 121 | 2 | 86 | 1 | 0.1654 | 0.0697 | -0.0165 | -0.0723 | 0.0261 | 0.4792 | 1.0261 | 0.7126 | 0.3135 | -0.0558 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 122 | 2 | 87 | 1 | 0.1654 | 0.0697 | -0.0172 | -0.0723 | 0.026 | 0.4811 | 1.026 | 0.7168 | 0.3092 | -0.0552 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 123 | 2 | 88 | 1 | 0.1654 | 0.0697 | -0.0182 | -0.0723 | 0.0258 | 0.4831 | 1.0258 | 0.7221 | 0.3037 | -0.0542 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 124 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0256 | 0.4851 | 1.0256 | 0.7256 | 0.3 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 125 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0257 | 0.487 | 1.0257 | 0.7276 | 0.2981 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 126 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0258 | 0.489 | 1.0258 | 0.7295 | 0.2963 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 127 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0259 | 0.4909 | 1.0259 | 0.7314 | 0.2945 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 128 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.026 | 0.4928 | 1.026 | 0.7334 | 0.2926 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 129 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0261 | 0.4947 | 1.0261 | 0.7353 | 0.2908 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 130 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0262 | 0.4967 | 0.9937 | 0.7294 | 0.2643 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 131 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0263 | 0.4986 | 0.9771 | 0.7273 | 0.2498 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 132 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0264 | 0.5005 | 0.9501 | 0.7226 | 0.2274 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 133 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0265 | 0.5024 | 0.9674 | 0.7287 | 0.2387 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 134 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0266 | 0.5042 | 0.9656 | 0.7301 | 0.2355 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 135 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0267 | 0.5061 | 0.9582 | 0.7302 | 0.228 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 136 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0268 | 0.508 | 0.9721 | 0.7354 | 0.2367 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 137 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0269 | 0.5099 | 1.0035 | 0.7448 | 0.2588 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 138 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.027 | 0.5117 | 0.9854 | 0.7422 | 0.2431 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 139 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0271 | 0.5136 | 0.956 | 0.737 | 0.219 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 140 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0272 | 0.5154 | 0.9789 | 0.7443 | 0.2346 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 141 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0273 | 0.5172 | 0.9996 | 0.7511 | 0.2485 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 142 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0274 | 0.5191 | 1.0229 | 0.7585 | 0.2643 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 143 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0275 | 0.5209 | 1.007 | 0.7565 | 0.2505 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 144 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0276 | 0.5227 | 1.0096 | 0.7589 | 0.2507 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 145 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0277 | 0.5245 | 1.0202 | 0.7633 | 0.2569 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 146 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0278 | 0.5263 | 1.0264 | 0.7666 | 0.2599 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 147 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0279 | 0.5281 | 1.0279 | 0.7687 | 0.2592 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 148 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0279 | 0.5299 | 1.0279 | 0.7705 | 0.2575 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 149 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.028 | 0.5317 | 1.028 | 0.7723 | 0.2558 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 150 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0281 | 0.5335 | 1.0138 | 0.7706 | 0.2432 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 151 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0282 | 0.5353 | 1.0041 | 0.77 | 0.2341 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 152 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0283 | 0.537 | 0.9995 | 0.7706 | 0.2288 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 153 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0284 | 0.5388 | 1.0094 | 0.7748 | 0.2346 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 154 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0285 | 0.5406 | 1.0285 | 0.7811 | 0.2474 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 155 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0286 | 0.5423 | 1.018 | 0.7803 | 0.2377 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 156 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0287 | 0.5441 | 1.0128 | 0.7808 | 0.232 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 157 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0288 | 0.5458 | 1.0197 | 0.7842 | 0.2356 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 158 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0289 | 0.5475 | 1.0049 | 0.7823 | 0.2226 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 159 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.029 | 0.5493 | 1.0019 | 0.7833 | 0.2186 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 160 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0291 | 0.551 | 0.9928 | 0.7828 | 0.21 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 161 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0292 | 0.5527 | 0.9828 | 0.7821 | 0.2007 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 162 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0292 | 0.5544 | 0.9762 | 0.7822 | 0.194 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 163 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0293 | 0.5561 | 0.974 | 0.7834 | 0.1906 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 164 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0294 | 0.5578 | 0.973 | 0.7848 | 0.1882 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 165 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0295 | 0.5595 | 0.964 | 0.7843 | 0.1797 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 166 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0296 | 0.5612 | 0.991 | 0.7925 | 0.1985 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 167 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0297 | 0.5629 | 0.9831 | 0.7922 | 0.1908 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 168 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0298 | 0.5646 | 0.9774 | 0.7925 | 0.1848 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 169 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0299 | 0.5663 | 0.9734 | 0.7932 | 0.1802 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 170 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.03 | 0.568 | 0.9513 | 0.7896 | 0.1617 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 171 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.03 | 0.5696 | 0.9442 | 0.7895 | 0.1546 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 172 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0301 | 0.5713 | 0.9407 | 0.7903 | 0.1504 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 173 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0302 | 0.5729 | 0.9425 | 0.7924 | 0.1501 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 174 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0303 | 0.5746 | 0.9337 | 0.7919 | 0.1418 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 175 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0304 | 0.5762 | 0.9282 | 0.7922 | 0.136 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 176 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0305 | 0.5779 | 0.9216 | 0.7922 | 0.1294 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 177 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0306 | 0.5795 | 0.9063 | 0.7902 | 0.1161 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 178 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0307 | 0.5812 | 0.9008 | 0.7905 | 0.1103 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 179 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0307 | 0.5828 | 0.8822 | 0.7876 | 0.0946 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 180 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0308 | 0.5844 | 0.8853 | 0.7899 | 0.0953 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 181 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0309 | 0.586 | 0.8975 | 0.7945 | 0.103 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 182 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.031 | 0.5877 | 0.9213 | 0.8018 | 0.1195 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 183 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0311 | 0.5893 | 0.9214 | 0.8034 | 0.1179 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 184 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0312 | 0.5909 | 0.9214 | 0.805 | 0.1164 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 185 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0312 | 0.5925 | 0.9215 | 0.8066 | 0.1149 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 186 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0313 | 0.5941 | 0.9216 | 0.8082 | 0.1134 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 187 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0314 | 0.5957 | 0.9217 | 0.8098 | 0.1119 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 188 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0315 | 0.5973 | 0.9218 | 0.8114 | 0.1104 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 189 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0316 | 0.5989 | 0.9398 | 0.8173 | 0.1225 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 190 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0317 | 0.6004 | 0.9331 | 0.8173 | 0.1158 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 191 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0318 | 0.602 | 0.9343 | 0.8191 | 0.1152 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 192 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0318 | 0.6036 | 0.9272 | 0.819 | 0.1083 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 193 | 2 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0319 | 0.6052 | 0.9282 | 0.8208 | 0.1075 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 194 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.032 | 0.6067 | 0.9295 | 0.8226 | 0.1069 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 195 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0321 | 0.6083 | 0.9294 | 0.8241 | 0.1053 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 196 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0322 | 0.6098 | 0.935 | 0.827 | 0.108 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 197 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0322 | 0.6114 | 0.9352 | 0.8286 | 0.1066 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 198 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0323 | 0.6129 | 0.9307 | 0.829 | 0.1017 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 199 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0324 | 0.6145 | 0.9304 | 0.8305 | 0.0999 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 200 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0325 | 0.616 | 0.9187 | 0.8292 | 0.0895 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 201 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0326 | 0.6176 | 0.9137 | 0.8295 | 0.0842 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 202 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0327 | 0.6191 | 0.9154 | 0.8314 | 0.084 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 203 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0327 | 0.6206 | 0.9133 | 0.8324 | 0.0808 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 204 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0328 | 0.6222 | 0.9189 | 0.8353 | 0.0836 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 205 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0329 | 0.6237 | 0.9215 | 0.8374 | 0.0841 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 206 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.033 | 0.6252 | 0.939 | 0.8431 | 0.0958 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 207 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0331 | 0.6267 | 0.9565 | 0.8488 | 0.1077 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 208 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0331 | 0.6282 | 0.9587 | 0.8509 | 0.1078 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 209 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0332 | 0.6297 | 0.9779 | 0.857 | 0.1209 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 210 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0333 | 0.6312 | 0.9887 | 0.8611 | 0.1276 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 211 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0334 | 0.6327 | 0.9956 | 0.8642 | 0.1314 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 212 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0335 | 0.6342 | 1.0141 | 0.8701 | 0.1439 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 213 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0335 | 0.6357 | 1.0141 | 0.8716 | 0.1425 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 214 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0336 | 0.6372 | 1.0142 | 0.8731 | 0.1411 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 215 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0337 | 0.6387 | 1.0311 | 0.8786 | 0.1525 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 216 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0338 | 0.6402 | 1.0338 | 0.8807 | 0.153 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 217 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0338 | 0.6417 | 1.0338 | 0.8822 | 0.1516 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 218 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0339 | 0.6432 | 1.0339 | 0.8837 | 0.1502 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 219 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.034 | 0.6446 | 1.0322 | 0.8847 | 0.1475 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 220 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0341 | 0.6461 | 1.0313 | 0.886 | 0.1454 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 221 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0342 | 0.6476 | 1.0306 | 0.8872 | 0.1433 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 222 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0342 | 0.649 | 1.0288 | 0.8883 | 0.1406 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 223 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0343 | 0.6505 | 1.0307 | 0.8902 | 0.1406 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 224 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0344 | 0.6519 | 1.0321 | 0.8919 | 0.1401 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 225 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0345 | 0.6534 | 1.0255 | 0.8918 | 0.1337 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 226 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0345 | 0.6548 | 1.0223 | 0.8925 | 0.1299 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 227 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0346 | 0.6563 | 1.0214 | 0.8937 | 0.1277 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 228 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0347 | 0.6577 | 1.0221 | 0.8952 | 0.1268 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 229 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0348 | 0.6592 | 1.0255 | 0.8975 | 0.128 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 230 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0348 | 0.6606 | 1.0281 | 0.8996 | 0.1286 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 231 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0349 | 0.6621 | 1.025 | 0.9002 | 0.1248 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 232 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.035 | 0.6635 | 1.0167 | 0.8996 | 0.1171 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 233 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0351 | 0.6649 | 1.0182 | 0.9014 | 0.1168 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 234 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0351 | 0.6663 | 1.0332 | 0.9064 | 0.1268 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 235 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0352 | 0.6678 | 1.0352 | 0.9083 | 0.1269 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 236 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0353 | 0.6692 | 1.0353 | 0.9097 | 0.1256 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 237 | 1 | 89 | 1 | 0.1654 | 0.0697 | -0.0186 | -0.0723 | 0.0354 | 0.6706 | 1.0354 | 0.9111 | 0.1242 | -0.0537 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 238 | 1 | 90 | 1 | 0.1654 | 0.0697 | -0.02 | -0.0723 | 0.0351 | 0.672 | 1.0336 | 0.9168 | 0.1168 | -0.0524 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 239 | 1 | 90 | 1 | 0.1654 | 0.0697 | -0.02 | -0.0723 | 0.0351 | 0.6734 | 1.0351 | 0.9186 | 0.1165 | -0.0524 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 240 | 1 | 90 | 1 | 0.1654 | 0.0697 | -0.02 | -0.0723 | 0.0352 | 0.6748 | 1.0352 | 0.92 | 0.1152 | -0.0524 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 241 | 1 | 91 | 1 | 0.1654 | 0.0697 | -0.0198 | -0.0723 | 0.0349 | 0.6762 | 1.0349 | 0.9209 | 0.114 | -0.0525 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 242 | 1 | 92 | 1 | 0.1654 | 0.0697 | -0.0197 | -0.0723 | 0.0346 | 0.6776 | 1.0346 | 0.9218 | 0.1128 | -0.0527 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 243 | 1 | 93 | 1 | 0.1654 | 0.0697 | -0.0197 | -0.0723 | 0.0343 | 0.679 | 1.0343 | 0.9233 | 0.111 | -0.0526 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 244 | 1 | 94 | 1 | 0.1654 | 0.0697 | -0.0201 | -0.0723 | 0.034 | 0.6804 | 1.034 | 0.926 | 0.108 | -0.0522 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 245 | 1 | 95 | 1 | 0.1654 | 0.0697 | -0.0207 | -0.0723 | 0.0337 | 0.6818 | 1.0337 | 0.9296 | 0.1041 | -0.0516 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 246 | 1 | 96 | 1 | 0.1654 | 0.0697 | -0.02 | -0.0723 | 0.0334 | 0.6832 | 1.0334 | 0.9286 | 0.1049 | -0.0523 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 247 | 1 | 97 | 1 | 0.1654 | 0.0697 | -0.0199 | -0.0723 | 0.0332 | 0.6846 | 1.0332 | 0.9294 | 0.1038 | -0.0525 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 248 | 1 | 98 | 1 | 0.1654 | 0.0697 | -0.0207 | -0.0723 | 0.0329 | 0.686 | 1.0329 | 0.9336 | 0.0993 | -0.0517 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 249 | 1 | 99 | 1 | 0.1654 | 0.0697 | -0.021 | -0.0723 | 0.0326 | 0.6874 | 1.0326 | 0.9361 | 0.0966 | -0.0514 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 250 | 1 | 100 | 1 | 0.1654 | 0.0697 | -0.0211 | -0.0723 | 0.0324 | 0.6887 | 1.0324 | 0.9378 | 0.0945 | -0.0513 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 251 | 1 | 101 | 1 | 0.1654 | 0.0697 | -0.0209 | -0.0723 | 0.0321 | 0.6901 | 1.0321 | 0.9386 | 0.0936 | -0.0514 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 252 | 1 | 102 | 1 | 0.1654 | 0.0697 | -0.0211 | -0.0723 | 0.0319 | 0.6915 | 1.0319 | 0.9408 | 0.0911 | -0.0512 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 253 | 1 | 103 | 1 | 0.1654 | 0.0697 | -0.0216 | -0.0723 | 0.0316 | 0.6929 | 1.0316 | 0.944 | 0.0876 | -0.0507 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 254 | 1 | 104 | 1 | 0.1654 | 0.0697 | -0.0203 | -0.0723 | 0.0314 | 0.6942 | 1.0314 | 0.9405 | 0.0909 | -0.0521 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 255 | 1 | 105 | 1 | 0.1654 | 0.0697 | -0.0196 | -0.0723 | 0.0311 | 0.6956 | 1.0311 | 0.9396 | 0.0916 | -0.0527 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 256 | 1 | 106 | 1 | 0.1654 | 0.0697 | -0.0191 | -0.0723 | 0.0309 | 0.697 | 1.0309 | 0.9389 | 0.092 | -0.0533 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 257 | 1 | 107 | 1 | 0.1654 | 0.0697 | -0.0202 | -0.0723 | 0.0307 | 0.6983 | 1.0307 | 0.9443 | 0.0864 | -0.0521 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 258 | 1 | 108 | 1 | 0.1654 | 0.0697 | -0.019 | -0.0723 | 0.0305 | 0.6997 | 1.0305 | 0.9416 | 0.0889 | -0.0533 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 259 | 1 | 109 | 1 | 0.1654 | 0.0697 | -0.018 | -0.0723 | 0.0303 | 0.701 | 1.0303 | 0.9395 | 0.0908 | -0.0543 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 260 | 1 | 110 | 1 | 0.1654 | 0.0697 | -0.0175 | -0.0723 | 0.03 | 0.7024 | 1.03 | 0.9391 | 0.0909 | -0.0548 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 261 | 1 | 111 | 1 | 0.1654 | 0.0697 | -0.0166 | -0.0723 | 0.0298 | 0.7037 | 1.0298 | 0.9375 | 0.0923 | -0.0557 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 262 | 1 | 112 | 1 | 0.1654 | 0.0697 | -0.0166 | -0.0723 | 0.0296 | 0.7051 | 1.0296 | 0.9387 | 0.0909 | -0.0558 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 263 | 1 | 113 | 1 | 0.1654 | 0.0697 | -0.0175 | -0.0723 | 0.0294 | 0.7064 | 1.0294 | 0.9432 | 0.0862 | -0.0548 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 264 | 1 | 114 | 1 | 0.1654 | 0.0697 | -0.0179 | -0.0723 | 0.0292 | 0.7078 | 1.0292 | 0.9457 | 0.0835 | -0.0545 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 265 | 1 | 115 | 1 | 0.1654 | 0.0697 | -0.0181 | -0.0723 | 0.029 | 0.7091 | 1.029 | 0.9477 | 0.0813 | -0.0543 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 266 | 1 | 116 | 1 | 0.1654 | 0.0697 | -0.0172 | -0.0723 | 0.0288 | 0.7104 | 1.0288 | 0.9462 | 0.0826 | -0.0551 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 267 | 1 | 117 | 1 | 0.1654 | 0.0697 | -0.0175 | -0.0723 | 0.0286 | 0.7118 | 1.0286 | 0.9484 | 0.0803 | -0.0549 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 268 | 1 | 118 | 1 | 0.1654 | 0.0697 | -0.0176 | -0.0723 | 0.0284 | 0.7131 | 1.0284 | 0.9503 | 0.0782 | -0.0547 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 269 | 1 | 119 | 1 | 0.1654 | 0.0697 | -0.0174 | -0.0723 | 0.0283 | 0.7144 | 1.0283 | 0.951 | 0.0773 | -0.0549 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 270 | 1 | 120 | 1 | 0.1654 | 0.0697 | -0.0185 | -0.0723 | 0.0281 | 0.7158 | 1.0281 | 0.9558 | 0.0723 | -0.0539 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 271 | 1 | 121 | 1 | 0.1654 | 0.0697 | -0.0179 | -0.0723 | 0.0279 | 0.7171 | 1.0279 | 0.9552 | 0.0727 | -0.0544 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 272 | 1 | 122 | 1 | 0.1654 | 0.0697 | -0.0181 | -0.0723 | 0.0277 | 0.7184 | 1.0277 | 0.957 | 0.0707 | -0.0543 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 273 | 1 | 123 | 1 | 0.1654 | 0.0697 | -0.0169 | -0.0723 | 0.0276 | 0.7197 | 1.0276 | 0.9545 | 0.073 | -0.0554 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 274 | 1 | 124 | 1 | 0.1654 | 0.0697 | -0.0179 | -0.0723 | 0.0274 | 0.721 | 1.0274 | 0.9591 | 0.0683 | -0.0544 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 275 | 1 | 125 | 1 | 0.1654 | 0.0697 | -0.0184 | -0.0723 | 0.0272 | 0.7224 | 1.0272 | 0.9619 | 0.0653 | -0.054 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 276 | 1 | 126 | 1 | 0.1654 | 0.0697 | -0.0185 | -0.0723 | 0.027 | 0.7237 | 1.027 | 0.9639 | 0.0632 | -0.0538 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 277 | 1 | 127 | 1 | 0.1654 | 0.0697 | -0.0173 | -0.0723 | 0.0269 | 0.725 | 1.0269 | 0.9609 | 0.066 | -0.0551 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 278 | 1 | 128 | 1 | 0.1654 | 0.0697 | -0.0165 | -0.0723 | 0.0267 | 0.7263 | 1.0267 | 0.9598 | 0.0669 | -0.0558 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 279 | 1 | 129 | 1 | 0.1654 | 0.0697 | -0.0153 | -0.0723 | 0.0266 | 0.7276 | 1.0266 | 0.9572 | 0.0694 | -0.0571 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 280 | 1 | 130 | 1 | 0.1654 | 0.0697 | -0.0141 | -0.0723 | 0.0264 | 0.7289 | 1.0264 | 0.955 | 0.0714 | -0.0582 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 281 | 1 | 131 | 1 | 0.1654 | 0.0697 | -0.0133 | -0.0723 | 0.0263 | 0.7302 | 1.0263 | 0.9538 | 0.0724 | -0.059 | move_2_score_gt_move_4, move_2_q_gt_move_4 |
| guarded-w2 | 282 | 2 | 132 | 1 | 0.1654 | 0.0697 | -0.0128 | -0.0723 | 0.0261 | 0.7315 | 1.0261 | 0.9537 | 0.0724 | -0.0595 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 283 | 2 | 133 | 1 | 0.1654 | 0.0697 | -0.0128 | -0.0723 | 0.026 | 0.7328 | 1.026 | 0.955 | 0.071 | -0.0595 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 284 | 2 | 134 | 1 | 0.1654 | 0.0697 | -0.0132 | -0.0723 | 0.0258 | 0.7341 | 1.0258 | 0.9575 | 0.0683 | -0.0591 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 285 | 2 | 135 | 1 | 0.1654 | 0.0697 | -0.0127 | -0.0723 | 0.0257 | 0.7354 | 1.0257 | 0.9572 | 0.0685 | -0.0597 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 286 | 2 | 136 | 1 | 0.1654 | 0.0697 | -0.0123 | -0.0723 | 0.0255 | 0.7367 | 1.0255 | 0.9573 | 0.0682 | -0.0601 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 287 | 2 | 137 | 1 | 0.1654 | 0.0697 | -0.0118 | -0.0723 | 0.0254 | 0.738 | 1.0254 | 0.9573 | 0.0681 | -0.0605 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 288 | 2 | 138 | 1 | 0.1654 | 0.0697 | -0.0116 | -0.0723 | 0.0252 | 0.7392 | 1.0252 | 0.9581 | 0.0672 | -0.0607 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 289 | 2 | 139 | 1 | 0.1654 | 0.0697 | -0.0119 | -0.0723 | 0.0251 | 0.7405 | 1.0251 | 0.9602 | 0.0649 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 290 | 2 | 140 | 1 | 0.1654 | 0.0697 | -0.0119 | -0.0723 | 0.025 | 0.7418 | 1.025 | 0.9613 | 0.0637 | -0.0605 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 291 | 2 | 141 | 1 | 0.1654 | 0.0697 | -0.0119 | -0.0723 | 0.0248 | 0.7431 | 1.0248 | 0.9627 | 0.0621 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 292 | 2 | 142 | 1 | 0.1654 | 0.0697 | -0.0112 | -0.0723 | 0.0247 | 0.7444 | 1.0247 | 0.9621 | 0.0626 | -0.0611 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 293 | 2 | 143 | 1 | 0.1654 | 0.0697 | -0.0107 | -0.0723 | 0.0246 | 0.7456 | 1.0246 | 0.9618 | 0.0627 | -0.0616 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 294 | 2 | 144 | 1 | 0.1654 | 0.0697 | -0.01 | -0.0723 | 0.0245 | 0.7469 | 1.0245 | 0.9612 | 0.0632 | -0.0623 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 295 | 2 | 145 | 1 | 0.1654 | 0.0697 | -0.0094 | -0.0723 | 0.0243 | 0.7482 | 1.0243 | 0.961 | 0.0633 | -0.0629 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 296 | 2 | 146 | 1 | 0.1654 | 0.0697 | -0.0087 | -0.0723 | 0.0242 | 0.7494 | 1.0242 | 0.9604 | 0.0638 | -0.0636 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 297 | 2 | 147 | 1 | 0.1654 | 0.0697 | -0.0086 | -0.0723 | 0.0241 | 0.7507 | 1.0241 | 0.9614 | 0.0627 | -0.0637 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 298 | 2 | 148 | 1 | 0.1654 | 0.0697 | -0.0087 | -0.0723 | 0.024 | 0.752 | 1.024 | 0.9628 | 0.0612 | -0.0637 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 299 | 2 | 149 | 1 | 0.1654 | 0.0697 | -0.0082 | -0.0723 | 0.0238 | 0.7532 | 1.0238 | 0.9628 | 0.0611 | -0.0641 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 300 | 2 | 150 | 1 | 0.1654 | 0.0697 | -0.0075 | -0.0723 | 0.0237 | 0.7545 | 1.0237 | 0.9622 | 0.0615 | -0.0649 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 301 | 2 | 151 | 1 | 0.1654 | 0.0697 | -0.007 | -0.0723 | 0.0236 | 0.7557 | 1.0236 | 0.9623 | 0.0613 | -0.0653 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 302 | 2 | 152 | 1 | 0.1654 | 0.0697 | -0.0075 | -0.0723 | 0.0235 | 0.757 | 1.0235 | 0.9649 | 0.0586 | -0.0648 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 303 | 2 | 153 | 1 | 0.1654 | 0.0697 | -0.0081 | -0.0723 | 0.0234 | 0.7582 | 1.0234 | 0.9675 | 0.0558 | -0.0643 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 304 | 2 | 154 | 1 | 0.1654 | 0.0697 | -0.0083 | -0.0723 | 0.0233 | 0.7595 | 1.0233 | 0.9694 | 0.0538 | -0.064 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 305 | 2 | 155 | 1 | 0.1654 | 0.0697 | -0.0081 | -0.0723 | 0.0231 | 0.7607 | 1.0231 | 0.97 | 0.0531 | -0.0642 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 306 | 2 | 156 | 1 | 0.1654 | 0.0697 | -0.0075 | -0.0723 | 0.023 | 0.762 | 1.023 | 0.9697 | 0.0534 | -0.0649 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 307 | 2 | 157 | 1 | 0.1654 | 0.0697 | -0.0069 | -0.0723 | 0.0229 | 0.7632 | 1.0229 | 0.9696 | 0.0533 | -0.0654 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 308 | 2 | 158 | 1 | 0.1654 | 0.0697 | -0.0065 | -0.0723 | 0.0228 | 0.7645 | 1.0228 | 0.9698 | 0.0531 | -0.0658 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 309 | 2 | 159 | 1 | 0.1654 | 0.0697 | -0.0063 | -0.0723 | 0.0227 | 0.7657 | 1.0227 | 0.9704 | 0.0523 | -0.0661 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 310 | 2 | 160 | 1 | 0.1654 | 0.0697 | -0.006 | -0.0723 | 0.0226 | 0.767 | 1.0226 | 0.971 | 0.0516 | -0.0663 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 311 | 2 | 161 | 1 | 0.1654 | 0.0697 | -0.0057 | -0.0723 | 0.0225 | 0.7682 | 1.0225 | 0.9716 | 0.0509 | -0.0666 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 312 | 2 | 162 | 1 | 0.1654 | 0.0697 | -0.0062 | -0.0723 | 0.0224 | 0.7694 | 1.0224 | 0.9739 | 0.0485 | -0.0662 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 313 | 2 | 163 | 1 | 0.1654 | 0.0697 | -0.0062 | -0.0723 | 0.0223 | 0.7707 | 1.0223 | 0.9752 | 0.0471 | -0.0661 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 314 | 2 | 164 | 1 | 0.1654 | 0.0697 | -0.0061 | -0.0723 | 0.0222 | 0.7719 | 1.0222 | 0.9762 | 0.046 | -0.0662 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 315 | 2 | 165 | 1 | 0.1654 | 0.0697 | -0.0049 | -0.0723 | 0.0221 | 0.7731 | 1.0221 | 0.9744 | 0.0477 | -0.0675 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 316 | 2 | 166 | 1 | 0.1654 | 0.0697 | -0.005 | -0.0723 | 0.022 | 0.7743 | 1.022 | 0.9761 | 0.0459 | -0.0673 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 317 | 2 | 167 | 1 | 0.1654 | 0.0697 | -0.0047 | -0.0723 | 0.0219 | 0.7756 | 1.0219 | 0.9765 | 0.0454 | -0.0676 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 318 | 2 | 168 | 1 | 0.1654 | 0.0697 | -0.0041 | -0.0723 | 0.0218 | 0.7768 | 1.0218 | 0.9764 | 0.0455 | -0.0682 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 319 | 2 | 169 | 1 | 0.1654 | 0.0697 | -0.0044 | -0.0723 | 0.0217 | 0.778 | 1.0217 | 0.9783 | 0.0434 | -0.0679 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 320 | 2 | 170 | 1 | 0.1654 | 0.0697 | -0.0045 | -0.0723 | 0.0216 | 0.7792 | 1.0216 | 0.9796 | 0.042 | -0.0679 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 321 | 2 | 171 | 1 | 0.1654 | 0.0697 | -0.0052 | -0.0723 | 0.0215 | 0.7804 | 1.0215 | 0.9825 | 0.039 | -0.0671 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 322 | 2 | 172 | 1 | 0.1654 | 0.0697 | -0.0037 | -0.0723 | 0.0214 | 0.7817 | 1.0214 | 0.9803 | 0.0412 | -0.0686 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 323 | 2 | 173 | 1 | 0.1654 | 0.0697 | -0.0023 | -0.0723 | 0.0214 | 0.7829 | 1.0214 | 0.9783 | 0.0431 | -0.07 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 324 | 2 | 174 | 1 | 0.1654 | 0.0697 | -0.0016 | -0.0723 | 0.0213 | 0.7841 | 1.0213 | 0.978 | 0.0433 | -0.0707 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 325 | 2 | 175 | 1 | 0.1654 | 0.0697 | -0.001 | -0.0723 | 0.0212 | 0.7853 | 1.0212 | 0.9777 | 0.0435 | -0.0714 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 326 | 2 | 176 | 1 | 0.1654 | 0.0697 | -0.0004 | -0.0723 | 0.0211 | 0.7865 | 1.0211 | 0.9777 | 0.0433 | -0.0719 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 327 | 2 | 177 | 1 | 0.1654 | 0.0697 | 0.0003 | -0.0723 | 0.021 | 0.7877 | 1.021 | 0.9773 | 0.0437 | -0.0727 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 328 | 2 | 178 | 1 | 0.1654 | 0.0697 | 0.0004 | -0.0723 | 0.0209 | 0.7889 | 1.0209 | 0.9785 | 0.0424 | -0.0727 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 329 | 2 | 179 | 1 | 0.1654 | 0.0697 | -0.0001 | -0.0723 | 0.0208 | 0.7901 | 1.0208 | 0.9807 | 0.0402 | -0.0722 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 330 | 2 | 180 | 1 | 0.1654 | 0.0697 | -0.0004 | -0.0723 | 0.0208 | 0.7913 | 1.0208 | 0.9826 | 0.0381 | -0.0719 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 331 | 2 | 181 | 1 | 0.1654 | 0.0697 | -0.0006 | -0.0723 | 0.0207 | 0.7925 | 1.0207 | 0.9841 | 0.0365 | -0.0717 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 332 | 2 | 182 | 1 | 0.1654 | 0.0697 | -0.0007 | -0.0723 | 0.0206 | 0.7937 | 1.0206 | 0.9856 | 0.035 | -0.0716 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 333 | 2 | 183 | 1 | 0.1654 | 0.0697 | -0.0002 | -0.0723 | 0.0205 | 0.7949 | 1.0205 | 0.9856 | 0.0349 | -0.0722 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 334 | 2 | 184 | 1 | 0.1654 | 0.0697 | 0.0003 | -0.0723 | 0.0204 | 0.7961 | 1.0204 | 0.9858 | 0.0347 | -0.0726 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 335 | 2 | 185 | 1 | 0.1654 | 0.0697 | 0.0008 | -0.0723 | 0.0203 | 0.7973 | 1.0203 | 0.986 | 0.0344 | -0.0731 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 336 | 2 | 186 | 1 | 0.1654 | 0.0697 | 0.0016 | -0.0723 | 0.0203 | 0.7985 | 1.0203 | 0.9856 | 0.0347 | -0.0739 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 337 | 2 | 187 | 1 | 0.1654 | 0.0697 | 0.0019 | -0.0723 | 0.0202 | 0.7997 | 1.0202 | 0.986 | 0.0341 | -0.0742 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 338 | 2 | 188 | 1 | 0.1654 | 0.0697 | 0.0022 | -0.0723 | 0.0201 | 0.8008 | 1.0201 | 0.9866 | 0.0335 | -0.0745 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 339 | 2 | 189 | 1 | 0.1654 | 0.0697 | 0.0031 | -0.0723 | 0.02 | 0.802 | 1.02 | 0.9859 | 0.0341 | -0.0755 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 340 | 2 | 190 | 1 | 0.1654 | 0.0697 | 0.0032 | -0.0723 | 0.02 | 0.8032 | 1.02 | 0.9871 | 0.0329 | -0.0755 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 341 | 2 | 191 | 1 | 0.1654 | 0.0697 | 0.0039 | -0.0723 | 0.0199 | 0.8044 | 1.0199 | 0.9868 | 0.0331 | -0.0762 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 342 | 2 | 192 | 1 | 0.1654 | 0.0697 | 0.0037 | -0.0723 | 0.0198 | 0.8056 | 1.0198 | 0.9883 | 0.0315 | -0.0761 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 343 | 2 | 193 | 1 | 0.1654 | 0.0697 | 0.0042 | -0.0723 | 0.0197 | 0.8067 | 1.0197 | 0.9885 | 0.0312 | -0.0766 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 344 | 2 | 194 | 1 | 0.1654 | 0.0697 | 0.005 | -0.0723 | 0.0197 | 0.8079 | 1.0197 | 0.9882 | 0.0314 | -0.0773 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 345 | 2 | 195 | 1 | 0.1654 | 0.0697 | 0.0045 | -0.0723 | 0.0196 | 0.8091 | 1.0196 | 0.9903 | 0.0293 | -0.0769 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 346 | 2 | 196 | 1 | 0.1654 | 0.0697 | 0.0044 | -0.0723 | 0.0195 | 0.8103 | 1.0195 | 0.9917 | 0.0278 | -0.0767 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 347 | 2 | 197 | 1 | 0.1654 | 0.0697 | 0.0051 | -0.0723 | 0.0195 | 0.8114 | 1.0195 | 0.9915 | 0.028 | -0.0774 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 348 | 2 | 198 | 1 | 0.1654 | 0.0697 | 0.0064 | -0.0723 | 0.0194 | 0.8126 | 1.0194 | 0.9903 | 0.0291 | -0.0787 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 349 | 2 | 199 | 1 | 0.1654 | 0.0697 | 0.0066 | -0.0723 | 0.0193 | 0.8138 | 1.0193 | 0.9909 | 0.0284 | -0.079 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 350 | 2 | 200 | 1 | 0.1654 | 0.0697 | 0.0069 | -0.0723 | 0.0192 | 0.8149 | 1.0192 | 0.9917 | 0.0276 | -0.0792 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 351 | 2 | 201 | 1 | 0.1654 | 0.0697 | 0.0077 | -0.0723 | 0.0192 | 0.8161 | 1.0192 | 0.9914 | 0.0278 | -0.08 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 352 | 2 | 202 | 1 | 0.1654 | 0.0697 | 0.0086 | -0.0723 | 0.0191 | 0.8173 | 1.0191 | 0.9908 | 0.0283 | -0.081 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 353 | 2 | 203 | 1 | 0.1654 | 0.0697 | 0.0085 | -0.0723 | 0.019 | 0.8184 | 1.019 | 0.9923 | 0.0267 | -0.0808 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 354 | 2 | 204 | 1 | 0.1654 | 0.0697 | 0.0084 | -0.0723 | 0.019 | 0.8196 | 1.019 | 0.9935 | 0.0255 | -0.0808 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 355 | 2 | 205 | 1 | 0.1654 | 0.0697 | 0.009 | -0.0723 | 0.0189 | 0.8207 | 1.0189 | 0.9937 | 0.0252 | -0.0813 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 356 | 2 | 206 | 1 | 0.1654 | 0.0697 | 0.0098 | -0.0723 | 0.0188 | 0.8219 | 1.0188 | 0.9935 | 0.0254 | -0.0821 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 357 | 2 | 207 | 1 | 0.1654 | 0.0697 | 0.0104 | -0.0723 | 0.0188 | 0.823 | 1.0188 | 0.9936 | 0.0252 | -0.0827 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 358 | 2 | 208 | 1 | 0.1654 | 0.0697 | 0.0103 | -0.0723 | 0.0187 | 0.8242 | 1.0187 | 0.9949 | 0.0238 | -0.0826 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 359 | 2 | 209 | 1 | 0.1654 | 0.0697 | 0.0116 | -0.0723 | 0.0187 | 0.8253 | 1.0187 | 0.9939 | 0.0248 | -0.0839 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 360 | 2 | 210 | 1 | 0.1654 | 0.0697 | 0.0125 | -0.0723 | 0.0186 | 0.8265 | 1.0186 | 0.9934 | 0.0252 | -0.0849 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 361 | 2 | 211 | 1 | 0.1654 | 0.0697 | 0.0141 | -0.0723 | 0.0185 | 0.8276 | 1.0185 | 0.9921 | 0.0264 | -0.0864 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 362 | 2 | 212 | 1 | 0.1654 | 0.0697 | 0.0145 | -0.0723 | 0.0185 | 0.8288 | 1.0185 | 0.9926 | 0.0259 | -0.0868 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 363 | 2 | 213 | 1 | 0.1654 | 0.0697 | 0.0149 | -0.0723 | 0.0184 | 0.8299 | 1.0184 | 0.993 | 0.0254 | -0.0873 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 364 | 2 | 214 | 1 | 0.1654 | 0.0697 | 0.0146 | -0.0723 | 0.0183 | 0.8311 | 1.0183 | 0.9947 | 0.0236 | -0.0869 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 365 | 2 | 215 | 1 | 0.1654 | 0.0697 | 0.0151 | -0.0723 | 0.0183 | 0.8322 | 1.0183 | 0.995 | 0.0233 | -0.0874 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 366 | 2 | 216 | 1 | 0.1654 | 0.0697 | 0.0144 | -0.0723 | 0.0182 | 0.8334 | 1.0182 | 0.9972 | 0.021 | -0.0868 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 367 | 2 | 217 | 1 | 0.1654 | 0.0697 | 0.0157 | -0.0723 | 0.0182 | 0.8345 | 1.0182 | 0.9964 | 0.0217 | -0.088 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 368 | 2 | 218 | 1 | 0.1654 | 0.0697 | 0.0154 | -0.0723 | 0.0181 | 0.8356 | 1.0181 | 0.998 | 0.0201 | -0.0877 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 369 | 2 | 219 | 1 | 0.1654 | 0.0697 | 0.0151 | -0.0723 | 0.0181 | 0.8368 | 1.0181 | 0.9996 | 0.0184 | -0.0874 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 370 | 2 | 220 | 1 | 0.1654 | 0.0697 | 0.017 | -0.0723 | 0.018 | 0.8379 | 1.018 | 0.9979 | 0.0201 | -0.0893 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 371 | 2 | 221 | 1 | 0.1654 | 0.0697 | 0.0166 | -0.0723 | 0.0179 | 0.839 | 1.0179 | 0.9995 | 0.0184 | -0.089 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 372 | 2 | 222 | 1 | 0.1654 | 0.0697 | 0.0162 | -0.0723 | 0.0179 | 0.8402 | 1.0179 | 1.0012 | 0.0166 | -0.0886 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 373 | 2 | 223 | 1 | 0.1654 | 0.0697 | 0.0159 | -0.0723 | 0.0178 | 0.8413 | 1.0178 | 1.0029 | 0.0149 | -0.0882 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 374 | 2 | 224 | 1 | 0.1654 | 0.0697 | 0.0165 | -0.0723 | 0.0178 | 0.8424 | 1.0178 | 1.0031 | 0.0147 | -0.0889 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 375 | 2 | 225 | 1 | 0.1654 | 0.0697 | 0.0172 | -0.0723 | 0.0177 | 0.8435 | 1.0177 | 1.0032 | 0.0145 | -0.0895 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 376 | 2 | 226 | 1 | 0.1654 | 0.0697 | 0.017 | -0.0723 | 0.0177 | 0.8447 | 1.0177 | 1.0046 | 0.013 | -0.0893 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 377 | 2 | 227 | 1 | 0.1654 | 0.0697 | 0.0166 | -0.0723 | 0.0176 | 0.8458 | 1.0176 | 1.0063 | 0.0114 | -0.089 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 378 | 2 | 228 | 1 | 0.1654 | 0.0697 | 0.0161 | -0.0723 | 0.0176 | 0.8469 | 1.0176 | 1.0081 | 0.0094 | -0.0885 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 379 | 2 | 229 | 1 | 0.1654 | 0.0697 | 0.0163 | -0.0723 | 0.0175 | 0.848 | 1.0175 | 1.0091 | 0.0084 | -0.0886 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 380 | 2 | 230 | 1 | 0.1654 | 0.0697 | 0.0161 | -0.0723 | 0.0174 | 0.8491 | 1.0174 | 1.0104 | 0.0071 | -0.0885 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 381 | 2 | 231 | 1 | 0.1654 | 0.0697 | 0.0172 | -0.0723 | 0.0174 | 0.8503 | 1.0174 | 1.0099 | 0.0075 | -0.0895 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 382 | 2 | 232 | 1 | 0.1654 | 0.0697 | 0.0167 | -0.0723 | 0.0173 | 0.8514 | 1.0173 | 1.0117 | 0.0056 | -0.0891 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 383 | 2 | 233 | 1 | 0.1654 | 0.0697 | 0.0165 | -0.0723 | 0.0173 | 0.8525 | 1.0173 | 1.0131 | 0.0042 | -0.0889 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 384 | 2 | 234 | 1 | 0.1654 | 0.0697 | 0.0159 | -0.0723 | 0.0172 | 0.8536 | 1.0172 | 1.0151 | 0.0021 | -0.0883 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 385 | 2 | 235 | 1 | 0.1654 | 0.0697 | 0.0155 | -0.0723 | 0.0172 | 0.8547 | 1.0172 | 1.0169 | 0.0003 | -0.0878 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 386 | 2 | 236 | 1 | 0.1654 | 0.0697 | 0.0163 | -0.0723 | 0.0171 | 0.8558 | 1.0171 | 1.0167 | 0.0004 | -0.0887 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 387 | 2 | 237 | 1 | 0.1654 | 0.0697 | 0.0169 | -0.0723 | 0.0171 | 0.8569 | 1.0171 | 1.017 | 0.0 | -0.0892 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 388 | 2 | 238 | 1 | 0.1654 | 0.0697 | 0.017 | -0.0723 | 0.017 | 0.858 | 1.017 | 1.018 | -0.0009 | -0.0893 | move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 389 | 2 | 238 | 2 | 0.1654 | 0.0697 | 0.017 | 0.0386 | 0.0171 | 0.5728 | 0.848 | 1.5728 | -0.7247 | 0.0216 | selected_move_2 |
| guarded-w2 | 390 | 2 | 238 | 3 | 0.1654 | 0.0697 | 0.017 | 0.0372 | 0.0171 | 0.4301 | 0.8572 | 1.4301 | -0.5729 | 0.0202 | selected_move_2 |
| guarded-w2 | 391 | 2 | 238 | 4 | 0.1654 | 0.0697 | 0.017 | 0.0242 | 0.0171 | 0.3445 | 0.9538 | 1.3445 | -0.3907 | 0.0072 | selected_move_2 |
| guarded-w2 | 392 | 2 | 238 | 5 | 0.1654 | 0.0697 | 0.017 | 0.0128 | 0.0171 | 0.2875 | 1.0171 | 1.2485 | -0.2314 | -0.0041 | move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 393 | 2 | 238 | 6 | 0.1654 | 0.0697 | 0.017 | 0.0114 | 0.0172 | 0.2467 | 1.0172 | 1.194 | -0.1769 | -0.0056 | move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 394 | 2 | 238 | 7 | 0.1654 | 0.0697 | 0.017 | 0.0071 | 0.0172 | 0.2162 | 1.0172 | 1.123 | -0.1058 | -0.0099 | move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 395 | 2 | 238 | 8 | 0.1654 | 0.0697 | 0.017 | 0.005 | 0.0172 | 0.1924 | 1.0172 | 1.0793 | -0.0621 | -0.012 | move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 396 | 2 | 238 | 9 | 0.1654 | 0.0697 | 0.017 | 0.0043 | 0.0172 | 0.1734 | 1.0172 | 1.0539 | -0.0366 | -0.0127 | move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 397 | 2 | 238 | 10 | 0.1654 | 0.0697 | 0.017 | -0.0016 | 0.0172 | 0.1578 | 1.0172 | 0.9825 | 0.0347 | -0.0186 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 398 | 2 | 239 | 10 | 0.1654 | 0.0697 | 0.0171 | -0.0016 | 0.0172 | 0.158 | 1.0172 | 0.982 | 0.0352 | -0.0187 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 399 | 2 | 240 | 10 | 0.1654 | 0.0697 | 0.0173 | -0.0016 | 0.0171 | 0.1582 | 1.0171 | 0.9807 | 0.0364 | -0.0189 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 400 | 2 | 241 | 10 | 0.1654 | 0.0697 | 0.0174 | -0.0016 | 0.0171 | 0.1584 | 1.0171 | 0.9799 | 0.0371 | -0.019 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 401 | 2 | 242 | 10 | 0.1654 | 0.0697 | 0.0175 | -0.0016 | 0.017 | 0.1586 | 1.017 | 0.9791 | 0.0379 | -0.0192 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 402 | 2 | 243 | 10 | 0.1654 | 0.0697 | 0.0176 | -0.0016 | 0.017 | 0.1588 | 1.017 | 0.9789 | 0.0381 | -0.0192 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 403 | 2 | 244 | 10 | 0.1654 | 0.0697 | 0.0181 | -0.0016 | 0.0169 | 0.159 | 1.0169 | 0.9753 | 0.0416 | -0.0197 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 404 | 2 | 245 | 10 | 0.1654 | 0.0697 | 0.0179 | -0.0016 | 0.0169 | 0.1592 | 1.0169 | 0.9769 | 0.04 | -0.0196 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 405 | 2 | 246 | 10 | 0.1654 | 0.0697 | 0.0183 | -0.0016 | 0.0168 | 0.1594 | 1.0168 | 0.9744 | 0.0425 | -0.0199 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 406 | 2 | 247 | 10 | 0.1654 | 0.0697 | 0.0188 | -0.0016 | 0.0168 | 0.1596 | 1.0168 | 0.9707 | 0.0461 | -0.0204 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 407 | 2 | 248 | 10 | 0.1654 | 0.0697 | 0.0188 | -0.0016 | 0.0168 | 0.1598 | 1.0168 | 0.971 | 0.0458 | -0.0204 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 408 | 2 | 249 | 10 | 0.1654 | 0.0697 | 0.0189 | -0.0016 | 0.0167 | 0.16 | 1.0167 | 0.9702 | 0.0465 | -0.0205 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 409 | 2 | 250 | 10 | 0.1654 | 0.0697 | 0.0194 | -0.0016 | 0.0167 | 0.1602 | 1.0167 | 0.9668 | 0.0499 | -0.021 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 410 | 2 | 251 | 10 | 0.1654 | 0.0697 | 0.019 | -0.0016 | 0.0166 | 0.1604 | 1.0166 | 0.9696 | 0.047 | -0.0207 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 411 | 2 | 252 | 10 | 0.1654 | 0.0697 | 0.0188 | -0.0016 | 0.0166 | 0.1606 | 1.0166 | 0.9714 | 0.0451 | -0.0205 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 412 | 2 | 253 | 10 | 0.1654 | 0.0697 | 0.0193 | -0.0016 | 0.0165 | 0.1608 | 1.0165 | 0.9679 | 0.0486 | -0.021 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 413 | 2 | 254 | 10 | 0.1654 | 0.0697 | 0.0192 | -0.0016 | 0.0165 | 0.161 | 1.0165 | 0.9686 | 0.0479 | -0.0209 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 414 | 2 | 255 | 10 | 0.1654 | 0.0697 | 0.019 | -0.0016 | 0.0164 | 0.1611 | 1.0164 | 0.9705 | 0.046 | -0.0207 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 415 | 2 | 256 | 10 | 0.1654 | 0.0697 | 0.0185 | -0.0016 | 0.0164 | 0.1613 | 1.0164 | 0.9746 | 0.0418 | -0.0201 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 416 | 2 | 257 | 10 | 0.1654 | 0.0697 | 0.0182 | -0.0016 | 0.0163 | 0.1615 | 1.0163 | 0.9769 | 0.0395 | -0.0199 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 417 | 2 | 258 | 10 | 0.1654 | 0.0697 | 0.0179 | -0.0016 | 0.0163 | 0.1617 | 1.0163 | 0.9795 | 0.0368 | -0.0195 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 418 | 2 | 259 | 10 | 0.1654 | 0.0697 | 0.0174 | -0.0016 | 0.0163 | 0.1619 | 1.0163 | 0.9832 | 0.0331 | -0.0191 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 419 | 2 | 260 | 10 | 0.1654 | 0.0697 | 0.0182 | -0.0016 | 0.0162 | 0.1621 | 1.0162 | 0.9775 | 0.0387 | -0.0199 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 420 | 2 | 261 | 10 | 0.1654 | 0.0697 | 0.0183 | -0.0016 | 0.0162 | 0.1623 | 1.0162 | 0.9767 | 0.0395 | -0.02 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 421 | 2 | 262 | 10 | 0.1654 | 0.0697 | 0.0183 | -0.0016 | 0.0161 | 0.1625 | 1.0161 | 0.9775 | 0.0386 | -0.0199 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 422 | 2 | 263 | 10 | 0.1654 | 0.0697 | 0.0179 | -0.0016 | 0.0161 | 0.1627 | 1.0161 | 0.9806 | 0.0354 | -0.0195 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 423 | 2 | 264 | 10 | 0.1654 | 0.0697 | 0.0183 | -0.0016 | 0.016 | 0.1629 | 1.016 | 0.9774 | 0.0387 | -0.02 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 424 | 2 | 265 | 10 | 0.1654 | 0.0697 | 0.018 | -0.0016 | 0.016 | 0.1631 | 1.016 | 0.9798 | 0.0362 | -0.0197 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 425 | 2 | 266 | 10 | 0.1654 | 0.0697 | 0.0179 | -0.0016 | 0.016 | 0.1633 | 1.016 | 0.9808 | 0.0352 | -0.0196 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 426 | 2 | 267 | 10 | 0.1654 | 0.0697 | 0.0183 | -0.0016 | 0.0159 | 0.1635 | 1.0159 | 0.978 | 0.0379 | -0.02 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 427 | 2 | 268 | 10 | 0.1654 | 0.0697 | 0.0189 | -0.0016 | 0.0159 | 0.1637 | 1.0159 | 0.9737 | 0.0422 | -0.0206 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 428 | 2 | 269 | 10 | 0.1654 | 0.0697 | 0.0192 | -0.0016 | 0.0158 | 0.1639 | 1.0158 | 0.9719 | 0.0439 | -0.0208 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 429 | 2 | 270 | 10 | 0.1654 | 0.0697 | 0.0188 | -0.0016 | 0.0158 | 0.164 | 1.0158 | 0.9753 | 0.0405 | -0.0204 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 430 | 2 | 271 | 10 | 0.1654 | 0.0697 | 0.0195 | -0.0016 | 0.0158 | 0.1642 | 1.0158 | 0.9701 | 0.0457 | -0.0211 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 431 | 2 | 272 | 10 | 0.1654 | 0.0697 | 0.02 | -0.0016 | 0.0157 | 0.1644 | 1.0157 | 0.9668 | 0.0489 | -0.0216 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 432 | 2 | 273 | 10 | 0.1654 | 0.0697 | 0.0207 | -0.0016 | 0.0157 | 0.1646 | 1.0157 | 0.9615 | 0.0541 | -0.0223 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 433 | 2 | 274 | 10 | 0.1654 | 0.0697 | 0.0207 | -0.0016 | 0.0156 | 0.1648 | 1.0156 | 0.9618 | 0.0538 | -0.0223 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 434 | 2 | 275 | 10 | 0.1654 | 0.0697 | 0.0207 | -0.0016 | 0.0156 | 0.165 | 1.0156 | 0.9622 | 0.0534 | -0.0223 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 435 | 2 | 276 | 10 | 0.1654 | 0.0697 | 0.0204 | -0.0016 | 0.0156 | 0.1652 | 1.0156 | 0.964 | 0.0516 | -0.0221 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 436 | 2 | 277 | 10 | 0.1654 | 0.0697 | 0.0197 | -0.0016 | 0.0155 | 0.1654 | 1.0155 | 0.9692 | 0.0463 | -0.0214 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 437 | 2 | 278 | 10 | 0.1654 | 0.0697 | 0.0205 | -0.0016 | 0.0155 | 0.1656 | 1.0155 | 0.9642 | 0.0513 | -0.0221 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 438 | 2 | 279 | 10 | 0.1654 | 0.0697 | 0.0205 | -0.0016 | 0.0155 | 0.1658 | 1.0155 | 0.9645 | 0.051 | -0.0221 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 439 | 2 | 280 | 10 | 0.1654 | 0.0697 | 0.0211 | -0.0016 | 0.0154 | 0.1659 | 1.0154 | 0.9598 | 0.0556 | -0.0228 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 440 | 2 | 281 | 10 | 0.1654 | 0.0697 | 0.0209 | -0.0016 | 0.0154 | 0.1661 | 1.0154 | 0.962 | 0.0534 | -0.0225 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 441 | 2 | 282 | 10 | 0.1654 | 0.0697 | 0.0205 | -0.0016 | 0.0153 | 0.1663 | 1.0153 | 0.9645 | 0.0508 | -0.0222 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 442 | 2 | 283 | 10 | 0.1654 | 0.0697 | 0.0205 | -0.0016 | 0.0153 | 0.1665 | 1.0153 | 0.9651 | 0.0502 | -0.0221 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 443 | 2 | 284 | 10 | 0.1654 | 0.0697 | 0.0211 | -0.0016 | 0.0153 | 0.1667 | 1.0153 | 0.961 | 0.0543 | -0.0227 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 444 | 2 | 285 | 10 | 0.1654 | 0.0697 | 0.0207 | -0.0016 | 0.0152 | 0.1669 | 1.0152 | 0.9635 | 0.0517 | -0.0224 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 445 | 2 | 286 | 10 | 0.1654 | 0.0697 | 0.0207 | -0.0016 | 0.0152 | 0.1671 | 1.0152 | 0.9638 | 0.0514 | -0.0224 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 446 | 2 | 287 | 10 | 0.1654 | 0.0697 | 0.0208 | -0.0016 | 0.0152 | 0.1673 | 1.0152 | 0.9632 | 0.0519 | -0.0225 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 447 | 2 | 288 | 10 | 0.1654 | 0.0697 | 0.021 | -0.0016 | 0.0151 | 0.1674 | 1.0151 | 0.9618 | 0.0533 | -0.0227 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 448 | 2 | 289 | 10 | 0.1654 | 0.0697 | 0.0213 | -0.0016 | 0.0151 | 0.1676 | 1.0151 | 0.9606 | 0.0545 | -0.0229 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 449 | 2 | 290 | 10 | 0.1654 | 0.0697 | 0.0211 | -0.0016 | 0.0151 | 0.1678 | 1.0151 | 0.9618 | 0.0532 | -0.0228 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 450 | 2 | 291 | 10 | 0.1654 | 0.0697 | 0.0209 | -0.0016 | 0.015 | 0.168 | 1.015 | 0.9638 | 0.0512 | -0.0225 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 451 | 2 | 292 | 10 | 0.1654 | 0.0697 | 0.0213 | -0.0016 | 0.015 | 0.1682 | 1.015 | 0.9605 | 0.0545 | -0.023 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 452 | 2 | 293 | 10 | 0.1654 | 0.0697 | 0.0207 | -0.0016 | 0.015 | 0.1684 | 1.015 | 0.9652 | 0.0498 | -0.0224 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 453 | 2 | 294 | 10 | 0.1654 | 0.0697 | 0.0202 | -0.0016 | 0.0149 | 0.1686 | 1.0149 | 0.9688 | 0.0461 | -0.0219 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 454 | 2 | 295 | 10 | 0.1654 | 0.0697 | 0.02 | -0.0016 | 0.0149 | 0.1688 | 1.0149 | 0.9704 | 0.0445 | -0.0217 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 455 | 2 | 296 | 10 | 0.1654 | 0.0697 | 0.0204 | -0.0016 | 0.0149 | 0.1689 | 1.0149 | 0.9683 | 0.0466 | -0.022 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 456 | 2 | 297 | 10 | 0.1654 | 0.0697 | 0.0206 | -0.0016 | 0.0148 | 0.1691 | 1.0148 | 0.9669 | 0.0479 | -0.0222 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 457 | 2 | 298 | 10 | 0.1654 | 0.0697 | 0.0207 | -0.0016 | 0.0148 | 0.1693 | 1.0148 | 0.9661 | 0.0487 | -0.0224 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 458 | 2 | 299 | 10 | 0.1654 | 0.0697 | 0.0219 | -0.0016 | 0.0148 | 0.1695 | 1.0148 | 0.9577 | 0.0571 | -0.0236 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 459 | 2 | 300 | 10 | 0.1654 | 0.0697 | 0.0215 | -0.0016 | 0.0147 | 0.1697 | 1.0147 | 0.9607 | 0.0541 | -0.0232 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 460 | 2 | 301 | 10 | 0.1654 | 0.0697 | 0.0222 | -0.0016 | 0.0147 | 0.1699 | 1.0147 | 0.9559 | 0.0588 | -0.0239 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 461 | 2 | 302 | 10 | 0.1654 | 0.0697 | 0.023 | -0.0016 | 0.0147 | 0.17 | 1.0147 | 0.9504 | 0.0642 | -0.0247 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 462 | 2 | 303 | 10 | 0.1654 | 0.0697 | 0.0242 | -0.0016 | 0.0146 | 0.1702 | 1.0146 | 0.9428 | 0.0718 | -0.0258 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 463 | 2 | 304 | 10 | 0.1654 | 0.0697 | 0.024 | -0.0016 | 0.0146 | 0.1704 | 1.0146 | 0.9444 | 0.0702 | -0.0256 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 464 | 2 | 305 | 10 | 0.1654 | 0.0697 | 0.0249 | -0.0016 | 0.0146 | 0.1706 | 1.0146 | 0.9382 | 0.0763 | -0.0265 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 465 | 2 | 306 | 10 | 0.1654 | 0.0697 | 0.0255 | -0.0016 | 0.0145 | 0.1708 | 1.0145 | 0.9345 | 0.08 | -0.0271 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 466 | 2 | 307 | 10 | 0.1654 | 0.0697 | 0.0257 | -0.0016 | 0.0145 | 0.171 | 1.0145 | 0.9334 | 0.0811 | -0.0273 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 467 | 2 | 308 | 10 | 0.1654 | 0.0697 | 0.0259 | -0.0016 | 0.0145 | 0.1712 | 1.0145 | 0.9321 | 0.0824 | -0.0276 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 468 | 2 | 309 | 10 | 0.1654 | 0.0697 | 0.0258 | -0.0016 | 0.0144 | 0.1713 | 1.0144 | 0.9328 | 0.0817 | -0.0275 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 469 | 2 | 310 | 10 | 0.1654 | 0.0697 | 0.0257 | -0.0016 | 0.0144 | 0.1715 | 1.0144 | 0.934 | 0.0804 | -0.0273 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 470 | 2 | 311 | 10 | 0.1654 | 0.0697 | 0.0264 | -0.0016 | 0.0144 | 0.1717 | 1.0144 | 0.9295 | 0.0848 | -0.028 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 471 | 2 | 312 | 10 | 0.1654 | 0.0697 | 0.0267 | -0.0016 | 0.0143 | 0.1719 | 1.0143 | 0.9274 | 0.087 | -0.0284 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 472 | 2 | 313 | 10 | 0.1654 | 0.0697 | 0.0268 | -0.0016 | 0.0143 | 0.1721 | 1.0143 | 0.9274 | 0.0869 | -0.0284 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 473 | 2 | 314 | 10 | 0.1654 | 0.0697 | 0.0268 | -0.0016 | 0.0143 | 0.1722 | 1.0143 | 0.9273 | 0.0869 | -0.0284 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 474 | 2 | 315 | 10 | 0.1654 | 0.0697 | 0.0272 | -0.0016 | 0.0142 | 0.1724 | 1.0142 | 0.9247 | 0.0896 | -0.0289 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 475 | 2 | 316 | 10 | 0.1654 | 0.0697 | 0.0283 | -0.0016 | 0.0142 | 0.1726 | 1.0142 | 0.9178 | 0.0964 | -0.03 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 476 | 2 | 317 | 10 | 0.1654 | 0.0697 | 0.0286 | -0.0016 | 0.0142 | 0.1728 | 1.0142 | 0.9166 | 0.0976 | -0.0302 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 477 | 2 | 318 | 10 | 0.1654 | 0.0697 | 0.0287 | -0.0016 | 0.0142 | 0.173 | 1.0142 | 0.916 | 0.0981 | -0.0303 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 478 | 2 | 319 | 10 | 0.1654 | 0.0697 | 0.0291 | -0.0016 | 0.0141 | 0.1732 | 1.0141 | 0.9137 | 0.1004 | -0.0307 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 479 | 2 | 320 | 10 | 0.1654 | 0.0697 | 0.0291 | -0.0016 | 0.0141 | 0.1733 | 1.0141 | 0.9136 | 0.1005 | -0.0308 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 480 | 2 | 321 | 10 | 0.1654 | 0.0697 | 0.0286 | -0.0016 | 0.0141 | 0.1735 | 1.0141 | 0.9169 | 0.0971 | -0.0303 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 481 | 2 | 322 | 10 | 0.1654 | 0.0697 | 0.0282 | -0.0016 | 0.014 | 0.1737 | 1.014 | 0.9195 | 0.0946 | -0.0299 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 482 | 2 | 323 | 10 | 0.1654 | 0.0697 | 0.028 | -0.0016 | 0.014 | 0.1739 | 1.014 | 0.921 | 0.093 | -0.0297 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 483 | 2 | 324 | 10 | 0.1654 | 0.0697 | 0.0286 | -0.0016 | 0.014 | 0.1741 | 1.014 | 0.9173 | 0.0967 | -0.0303 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 484 | 2 | 325 | 10 | 0.1654 | 0.0697 | 0.0292 | -0.0016 | 0.014 | 0.1742 | 1.014 | 0.9142 | 0.0998 | -0.0308 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 485 | 2 | 326 | 10 | 0.1654 | 0.0697 | 0.0288 | -0.0016 | 0.0139 | 0.1744 | 1.0139 | 0.9164 | 0.0975 | -0.0305 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 486 | 2 | 327 | 10 | 0.1654 | 0.0697 | 0.0287 | -0.0016 | 0.0139 | 0.1746 | 1.0139 | 0.9176 | 0.0963 | -0.0303 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 487 | 2 | 328 | 10 | 0.1654 | 0.0697 | 0.0294 | -0.0016 | 0.0139 | 0.1748 | 1.0139 | 0.9135 | 0.1003 | -0.031 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 488 | 2 | 329 | 10 | 0.1654 | 0.0697 | 0.0289 | -0.0016 | 0.0138 | 0.175 | 1.0138 | 0.9165 | 0.0973 | -0.0306 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 489 | 2 | 330 | 10 | 0.1654 | 0.0697 | 0.0296 | -0.0016 | 0.0138 | 0.1751 | 1.0138 | 0.9123 | 0.1015 | -0.0313 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 490 | 2 | 331 | 10 | 0.1654 | 0.0697 | 0.0302 | -0.0016 | 0.0138 | 0.1753 | 1.0138 | 0.9089 | 0.1049 | -0.0318 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 491 | 2 | 332 | 10 | 0.1654 | 0.0697 | 0.0307 | -0.0016 | 0.0138 | 0.1755 | 1.0138 | 0.9061 | 0.1077 | -0.0323 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 492 | 2 | 333 | 10 | 0.1654 | 0.0697 | 0.0312 | -0.0016 | 0.0137 | 0.1757 | 1.0137 | 0.9033 | 0.1105 | -0.0328 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 493 | 2 | 334 | 10 | 0.1654 | 0.0697 | 0.0318 | -0.0016 | 0.0137 | 0.1759 | 1.0137 | 0.8996 | 0.1141 | -0.0335 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 494 | 2 | 335 | 10 | 0.1654 | 0.0697 | 0.0324 | -0.0016 | 0.0137 | 0.176 | 1.0137 | 0.8965 | 0.1172 | -0.034 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 495 | 2 | 336 | 10 | 0.1654 | 0.0697 | 0.033 | -0.0016 | 0.0137 | 0.1762 | 1.0137 | 0.893 | 0.1206 | -0.0346 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 496 | 2 | 337 | 10 | 0.1654 | 0.0697 | 0.0332 | -0.0016 | 0.0136 | 0.1764 | 1.0136 | 0.892 | 0.1217 | -0.0349 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 497 | 2 | 338 | 10 | 0.1654 | 0.0697 | 0.0335 | -0.0016 | 0.0136 | 0.1766 | 1.0136 | 0.8902 | 0.1234 | -0.0352 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 498 | 2 | 339 | 10 | 0.1654 | 0.0697 | 0.0346 | -0.0016 | 0.0136 | 0.1767 | 1.0136 | 0.8846 | 0.129 | -0.0362 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 499 | 2 | 340 | 10 | 0.1654 | 0.0697 | 0.0349 | -0.0016 | 0.0135 | 0.1769 | 1.0135 | 0.8829 | 0.1306 | -0.0365 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 500 | 2 | 341 | 10 | 0.1654 | 0.0697 | 0.0354 | -0.0016 | 0.0135 | 0.1771 | 1.0135 | 0.8799 | 0.1337 | -0.0371 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 501 | 2 | 342 | 10 | 0.1654 | 0.0697 | 0.0358 | -0.0016 | 0.0135 | 0.1773 | 1.0135 | 0.8779 | 0.1356 | -0.0375 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 502 | 2 | 343 | 10 | 0.1654 | 0.0697 | 0.0363 | -0.0016 | 0.0135 | 0.1775 | 1.0135 | 0.8752 | 0.1382 | -0.038 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 503 | 2 | 344 | 10 | 0.1654 | 0.0697 | 0.036 | -0.0016 | 0.0134 | 0.1776 | 1.0134 | 0.877 | 0.1364 | -0.0377 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 504 | 2 | 345 | 10 | 0.1654 | 0.0697 | 0.0356 | -0.0016 | 0.0134 | 0.1778 | 1.0134 | 0.8798 | 0.1336 | -0.0372 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 505 | 2 | 346 | 10 | 0.1654 | 0.0697 | 0.0351 | -0.0016 | 0.0134 | 0.178 | 1.0134 | 0.8825 | 0.1309 | -0.0368 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 506 | 2 | 347 | 10 | 0.1654 | 0.0697 | 0.0359 | -0.0016 | 0.0134 | 0.1782 | 1.0134 | 0.8783 | 0.1351 | -0.0376 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 507 | 2 | 348 | 10 | 0.1654 | 0.0697 | 0.0364 | -0.0016 | 0.0133 | 0.1783 | 1.0133 | 0.8757 | 0.1377 | -0.0381 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 508 | 2 | 349 | 10 | 0.1654 | 0.0697 | 0.0368 | -0.0016 | 0.0133 | 0.1785 | 1.0133 | 0.8738 | 0.1395 | -0.0384 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 509 | 2 | 350 | 10 | 0.1654 | 0.0697 | 0.0365 | -0.0016 | 0.0133 | 0.1787 | 1.0133 | 0.8753 | 0.138 | -0.0382 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 510 | 2 | 351 | 10 | 0.1654 | 0.0697 | 0.0368 | -0.0016 | 0.0133 | 0.1789 | 1.0133 | 0.8738 | 0.1394 | -0.0385 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 511 | 2 | 352 | 10 | 0.1654 | 0.0697 | 0.0365 | -0.0016 | 0.0132 | 0.179 | 1.0132 | 0.876 | 0.1372 | -0.0381 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 512 | 2 | 353 | 10 | 0.1654 | 0.0697 | 0.0371 | -0.0016 | 0.0132 | 0.1792 | 1.0132 | 0.8728 | 0.1404 | -0.0387 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 513 | 2 | 354 | 10 | 0.1654 | 0.0697 | 0.0373 | -0.0016 | 0.0132 | 0.1794 | 1.0132 | 0.8719 | 0.1413 | -0.0389 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 514 | 2 | 355 | 10 | 0.1654 | 0.0697 | 0.0371 | -0.0016 | 0.0132 | 0.1796 | 1.0132 | 0.8729 | 0.1402 | -0.0388 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 515 | 2 | 356 | 10 | 0.1654 | 0.0697 | 0.037 | -0.0016 | 0.0131 | 0.1797 | 1.0131 | 0.8738 | 0.1393 | -0.0387 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 516 | 2 | 357 | 10 | 0.1654 | 0.0697 | 0.0369 | -0.0016 | 0.0131 | 0.1799 | 1.0131 | 0.8744 | 0.1387 | -0.0386 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 517 | 2 | 358 | 10 | 0.1654 | 0.0697 | 0.0374 | -0.0016 | 0.0131 | 0.1801 | 1.0131 | 0.8718 | 0.1413 | -0.0391 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 518 | 2 | 359 | 10 | 0.1654 | 0.0697 | 0.0378 | -0.0016 | 0.0131 | 0.1803 | 1.0131 | 0.87 | 0.1431 | -0.0394 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 519 | 2 | 360 | 10 | 0.1654 | 0.0697 | 0.0376 | -0.0016 | 0.013 | 0.1804 | 1.013 | 0.8713 | 0.1418 | -0.0392 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 520 | 2 | 361 | 10 | 0.1654 | 0.0697 | 0.038 | -0.0016 | 0.013 | 0.1806 | 1.013 | 0.8694 | 0.1436 | -0.0396 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 521 | 2 | 362 | 10 | 0.1654 | 0.0697 | 0.0382 | -0.0016 | 0.013 | 0.1808 | 1.013 | 0.8682 | 0.1448 | -0.0399 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 522 | 2 | 363 | 10 | 0.1654 | 0.0697 | 0.0389 | -0.0016 | 0.013 | 0.181 | 1.013 | 0.8646 | 0.1484 | -0.0406 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 523 | 2 | 364 | 10 | 0.1654 | 0.0697 | 0.0392 | -0.0016 | 0.013 | 0.1811 | 1.013 | 0.8634 | 0.1495 | -0.0408 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 524 | 2 | 365 | 10 | 0.1654 | 0.0697 | 0.0392 | -0.0016 | 0.0129 | 0.1813 | 1.0129 | 0.8636 | 0.1493 | -0.0408 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 525 | 2 | 366 | 10 | 0.1654 | 0.0697 | 0.0389 | -0.0016 | 0.0129 | 0.1815 | 1.0129 | 0.8654 | 0.1475 | -0.0405 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 526 | 2 | 367 | 10 | 0.1654 | 0.0697 | 0.0392 | -0.0016 | 0.0129 | 0.1816 | 1.0129 | 0.864 | 0.1489 | -0.0408 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 527 | 2 | 368 | 10 | 0.1654 | 0.0697 | 0.0393 | -0.0016 | 0.0129 | 0.1818 | 1.0129 | 0.8634 | 0.1495 | -0.041 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 528 | 2 | 369 | 10 | 0.1654 | 0.0697 | 0.039 | -0.0016 | 0.0128 | 0.182 | 1.0128 | 0.8653 | 0.1475 | -0.0406 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 529 | 2 | 370 | 10 | 0.1654 | 0.0697 | 0.0392 | -0.0016 | 0.0128 | 0.1822 | 1.0128 | 0.8645 | 0.1483 | -0.0408 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 530 | 2 | 371 | 10 | 0.1654 | 0.0697 | 0.0389 | -0.0016 | 0.0128 | 0.1823 | 1.0128 | 0.8664 | 0.1464 | -0.0405 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 531 | 2 | 372 | 10 | 0.1654 | 0.0697 | 0.0388 | -0.0016 | 0.0128 | 0.1825 | 1.0128 | 0.8668 | 0.146 | -0.0405 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 532 | 2 | 373 | 10 | 0.1654 | 0.0697 | 0.0388 | -0.0016 | 0.0128 | 0.1827 | 1.0128 | 0.8669 | 0.1458 | -0.0405 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 533 | 2 | 374 | 10 | 0.1654 | 0.0697 | 0.0395 | -0.0016 | 0.0127 | 0.1828 | 1.0127 | 0.8636 | 0.1492 | -0.0411 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 534 | 2 | 375 | 10 | 0.1654 | 0.0697 | 0.039 | -0.0016 | 0.0127 | 0.183 | 1.0127 | 0.8664 | 0.1463 | -0.0406 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 535 | 2 | 376 | 10 | 0.1654 | 0.0697 | 0.0396 | -0.0016 | 0.0127 | 0.1832 | 1.0127 | 0.8632 | 0.1495 | -0.0413 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 536 | 2 | 377 | 10 | 0.1654 | 0.0697 | 0.0392 | -0.0016 | 0.0127 | 0.1834 | 1.0127 | 0.8659 | 0.1468 | -0.0408 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 537 | 2 | 378 | 10 | 0.1654 | 0.0697 | 0.0396 | -0.0016 | 0.0126 | 0.1835 | 1.0126 | 0.8635 | 0.1492 | -0.0413 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 538 | 2 | 379 | 10 | 0.1654 | 0.0697 | 0.04 | -0.0016 | 0.0126 | 0.1837 | 1.0126 | 0.8619 | 0.1507 | -0.0416 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 539 | 2 | 380 | 10 | 0.1654 | 0.0697 | 0.0409 | -0.0016 | 0.0126 | 0.1839 | 1.0126 | 0.8573 | 0.1553 | -0.0425 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 540 | 2 | 381 | 10 | 0.1654 | 0.0697 | 0.0413 | -0.0016 | 0.0126 | 0.184 | 1.0126 | 0.8552 | 0.1574 | -0.043 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 541 | 2 | 382 | 10 | 0.1654 | 0.0697 | 0.0419 | -0.0016 | 0.0126 | 0.1842 | 1.0126 | 0.8526 | 0.16 | -0.0435 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 542 | 2 | 383 | 10 | 0.1654 | 0.0697 | 0.0424 | -0.0016 | 0.0125 | 0.1844 | 1.0125 | 0.8501 | 0.1624 | -0.044 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 543 | 2 | 384 | 10 | 0.1654 | 0.0697 | 0.0426 | -0.0016 | 0.0125 | 0.1846 | 1.0125 | 0.8493 | 0.1633 | -0.0442 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 544 | 2 | 385 | 10 | 0.1654 | 0.0697 | 0.0432 | -0.0016 | 0.0125 | 0.1847 | 1.0125 | 0.8465 | 0.166 | -0.0448 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 545 | 2 | 386 | 10 | 0.1654 | 0.0697 | 0.0431 | -0.0016 | 0.0125 | 0.1849 | 1.0125 | 0.8469 | 0.1656 | -0.0448 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 546 | 2 | 387 | 10 | 0.1654 | 0.0697 | 0.043 | -0.0016 | 0.0125 | 0.1851 | 1.0125 | 0.8479 | 0.1645 | -0.0446 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 547 | 2 | 388 | 10 | 0.1654 | 0.0697 | 0.0425 | -0.0016 | 0.0124 | 0.1852 | 1.0124 | 0.8505 | 0.1619 | -0.0441 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 548 | 2 | 389 | 10 | 0.1654 | 0.0697 | 0.0432 | -0.0016 | 0.0124 | 0.1854 | 1.0124 | 0.847 | 0.1655 | -0.0449 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 549 | 2 | 390 | 10 | 0.1654 | 0.0697 | 0.0442 | -0.0016 | 0.0124 | 0.1856 | 1.0124 | 0.8423 | 0.1701 | -0.0458 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 550 | 2 | 391 | 10 | 0.1654 | 0.0697 | 0.0445 | -0.0016 | 0.0124 | 0.1857 | 1.0124 | 0.8411 | 0.1712 | -0.0461 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 551 | 2 | 392 | 10 | 0.1654 | 0.0697 | 0.0448 | -0.0016 | 0.0124 | 0.1859 | 1.0124 | 0.8399 | 0.1725 | -0.0464 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 552 | 2 | 393 | 10 | 0.1654 | 0.0697 | 0.0454 | -0.0016 | 0.0123 | 0.1861 | 1.0123 | 0.8369 | 0.1754 | -0.0471 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 553 | 2 | 394 | 10 | 0.1654 | 0.0697 | 0.0455 | -0.0016 | 0.0123 | 0.1862 | 1.0123 | 0.8366 | 0.1757 | -0.0472 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 554 | 2 | 395 | 10 | 0.1654 | 0.0697 | 0.0459 | -0.0016 | 0.0123 | 0.1864 | 1.0123 | 0.8346 | 0.1777 | -0.0476 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 555 | 2 | 396 | 10 | 0.1654 | 0.0697 | 0.046 | -0.0016 | 0.0123 | 0.1866 | 1.0123 | 0.8347 | 0.1776 | -0.0476 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 556 | 2 | 397 | 10 | 0.1654 | 0.0697 | 0.0466 | -0.0016 | 0.0123 | 0.1868 | 1.0123 | 0.8319 | 0.1803 | -0.0482 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 557 | 2 | 398 | 10 | 0.1654 | 0.0697 | 0.0469 | -0.0016 | 0.0122 | 0.1869 | 1.0122 | 0.8304 | 0.1818 | -0.0486 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 558 | 2 | 399 | 10 | 0.1654 | 0.0697 | 0.0469 | -0.0016 | 0.0122 | 0.1871 | 1.0122 | 0.8307 | 0.1816 | -0.0486 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 559 | 2 | 400 | 10 | 0.1654 | 0.0697 | 0.047 | -0.0016 | 0.0122 | 0.1873 | 1.0122 | 0.8305 | 0.1817 | -0.0486 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 560 | 2 | 401 | 10 | 0.1654 | 0.0697 | 0.0469 | -0.0016 | 0.0122 | 0.1874 | 1.0122 | 0.8309 | 0.1812 | -0.0486 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 561 | 2 | 402 | 10 | 0.1654 | 0.0697 | 0.0472 | -0.0016 | 0.0122 | 0.1876 | 1.0122 | 0.8299 | 0.1822 | -0.0488 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 562 | 2 | 403 | 10 | 0.1654 | 0.0697 | 0.0471 | -0.0016 | 0.0121 | 0.1878 | 1.0121 | 0.8306 | 0.1816 | -0.0487 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 563 | 2 | 404 | 10 | 0.1654 | 0.0697 | 0.0468 | -0.0016 | 0.0121 | 0.1879 | 1.0121 | 0.8322 | 0.1799 | -0.0484 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 564 | 2 | 405 | 10 | 0.1654 | 0.0697 | 0.0474 | -0.0016 | 0.0121 | 0.1881 | 1.0121 | 0.8296 | 0.1825 | -0.049 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 565 | 2 | 406 | 10 | 0.1654 | 0.0697 | 0.0475 | -0.0016 | 0.0121 | 0.1883 | 1.0121 | 0.8293 | 0.1828 | -0.0491 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 566 | 2 | 407 | 10 | 0.1654 | 0.0697 | 0.0476 | -0.0016 | 0.0121 | 0.1884 | 1.0121 | 0.8288 | 0.1833 | -0.0493 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 567 | 2 | 408 | 10 | 0.1654 | 0.0697 | 0.0479 | -0.0016 | 0.012 | 0.1886 | 1.012 | 0.8277 | 0.1843 | -0.0495 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 568 | 2 | 409 | 10 | 0.1654 | 0.0697 | 0.0482 | -0.0016 | 0.012 | 0.1888 | 1.012 | 0.8261 | 0.1859 | -0.0499 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 569 | 2 | 410 | 10 | 0.1654 | 0.0697 | 0.0491 | -0.0016 | 0.012 | 0.1889 | 1.012 | 0.8226 | 0.1894 | -0.0507 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 570 | 2 | 411 | 10 | 0.1654 | 0.0697 | 0.0492 | -0.0016 | 0.012 | 0.1891 | 1.012 | 0.822 | 0.19 | -0.0509 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 571 | 2 | 412 | 10 | 0.1654 | 0.0697 | 0.0492 | -0.0016 | 0.012 | 0.1893 | 1.012 | 0.8223 | 0.1897 | -0.0508 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 572 | 2 | 413 | 10 | 0.1654 | 0.0697 | 0.0493 | -0.0016 | 0.0119 | 0.1894 | 1.0119 | 0.8219 | 0.1901 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 573 | 2 | 414 | 10 | 0.1654 | 0.0697 | 0.0492 | -0.0016 | 0.0119 | 0.1896 | 1.0119 | 0.8225 | 0.1894 | -0.0509 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 574 | 2 | 415 | 10 | 0.1654 | 0.0697 | 0.049 | -0.0016 | 0.0119 | 0.1897 | 1.0119 | 0.8236 | 0.1883 | -0.0507 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 575 | 2 | 416 | 10 | 0.1654 | 0.0697 | 0.0488 | -0.0016 | 0.0119 | 0.1899 | 1.0119 | 0.8248 | 0.1871 | -0.0504 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 576 | 2 | 417 | 10 | 0.1654 | 0.0697 | 0.0487 | -0.0016 | 0.0119 | 0.1901 | 1.0119 | 0.8256 | 0.1863 | -0.0503 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 577 | 2 | 418 | 10 | 0.1654 | 0.0697 | 0.0491 | -0.0016 | 0.0119 | 0.1902 | 1.0119 | 0.8235 | 0.1883 | -0.0508 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 578 | 2 | 419 | 10 | 0.1654 | 0.0697 | 0.0495 | -0.0016 | 0.0118 | 0.1904 | 1.0118 | 0.8219 | 0.1899 | -0.0512 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 579 | 2 | 420 | 10 | 0.1654 | 0.0697 | 0.0493 | -0.0016 | 0.0118 | 0.1906 | 1.0118 | 0.823 | 0.1888 | -0.051 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 580 | 2 | 421 | 10 | 0.1654 | 0.0697 | 0.0491 | -0.0016 | 0.0118 | 0.1907 | 1.0118 | 0.8243 | 0.1875 | -0.0507 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 581 | 2 | 422 | 10 | 0.1654 | 0.0697 | 0.0496 | -0.0016 | 0.0118 | 0.1909 | 1.0118 | 0.822 | 0.1898 | -0.0513 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 582 | 2 | 423 | 10 | 0.1654 | 0.0697 | 0.0498 | -0.0016 | 0.0118 | 0.1911 | 1.0118 | 0.8212 | 0.1905 | -0.0515 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 583 | 2 | 424 | 10 | 0.1654 | 0.0697 | 0.0495 | -0.0016 | 0.0117 | 0.1912 | 1.0117 | 0.8228 | 0.189 | -0.0512 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 584 | 2 | 425 | 10 | 0.1654 | 0.0697 | 0.0495 | -0.0016 | 0.0117 | 0.1914 | 1.0117 | 0.8231 | 0.1886 | -0.0511 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 585 | 2 | 426 | 10 | 0.1654 | 0.0697 | 0.0491 | -0.0016 | 0.0117 | 0.1916 | 1.0117 | 0.8251 | 0.1866 | -0.0507 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 586 | 2 | 427 | 10 | 0.1654 | 0.0697 | 0.0487 | -0.0016 | 0.0117 | 0.1917 | 1.0117 | 0.8269 | 0.1848 | -0.0504 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 587 | 2 | 428 | 10 | 0.1654 | 0.0697 | 0.0489 | -0.0016 | 0.0117 | 0.1919 | 1.0117 | 0.8264 | 0.1853 | -0.0505 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 588 | 2 | 429 | 10 | 0.1654 | 0.0697 | 0.0488 | -0.0016 | 0.0117 | 0.192 | 1.0117 | 0.8267 | 0.185 | -0.0505 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 589 | 2 | 430 | 10 | 0.1654 | 0.0697 | 0.0492 | -0.0016 | 0.0116 | 0.1922 | 1.0116 | 0.8253 | 0.1863 | -0.0508 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 590 | 2 | 431 | 10 | 0.1654 | 0.0697 | 0.0489 | -0.0016 | 0.0116 | 0.1924 | 1.0116 | 0.8267 | 0.185 | -0.0506 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 591 | 2 | 432 | 10 | 0.1654 | 0.0697 | 0.0487 | -0.0016 | 0.0116 | 0.1925 | 1.0116 | 0.8276 | 0.184 | -0.0504 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 592 | 2 | 433 | 10 | 0.1654 | 0.0697 | 0.0486 | -0.0016 | 0.0116 | 0.1927 | 1.0116 | 0.8282 | 0.1834 | -0.0503 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 593 | 2 | 434 | 10 | 0.1654 | 0.0697 | 0.0487 | -0.0016 | 0.0116 | 0.1929 | 1.0116 | 0.8279 | 0.1836 | -0.0504 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 594 | 2 | 435 | 10 | 0.1654 | 0.0697 | 0.0487 | -0.0016 | 0.0116 | 0.193 | 1.0116 | 0.8284 | 0.1832 | -0.0503 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 595 | 2 | 436 | 10 | 0.1654 | 0.0697 | 0.0489 | -0.0016 | 0.0115 | 0.1932 | 1.0115 | 0.8277 | 0.1838 | -0.0505 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 596 | 2 | 437 | 10 | 0.1654 | 0.0697 | 0.049 | -0.0016 | 0.0115 | 0.1934 | 1.0115 | 0.8271 | 0.1844 | -0.0507 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 597 | 2 | 438 | 10 | 0.1654 | 0.0697 | 0.0492 | -0.0016 | 0.0115 | 0.1935 | 1.0115 | 0.8267 | 0.1848 | -0.0508 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 598 | 2 | 439 | 10 | 0.1654 | 0.0697 | 0.0495 | -0.0016 | 0.0115 | 0.1937 | 1.0115 | 0.8253 | 0.1861 | -0.0511 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 599 | 2 | 440 | 10 | 0.1654 | 0.0697 | 0.0495 | -0.0016 | 0.0115 | 0.1938 | 1.0115 | 0.8254 | 0.1861 | -0.0512 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 600 | 2 | 441 | 10 | 0.1654 | 0.0697 | 0.0494 | -0.0016 | 0.0115 | 0.194 | 1.0115 | 0.826 | 0.1854 | -0.0511 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 601 | 2 | 442 | 10 | 0.1654 | 0.0697 | 0.0498 | -0.0016 | 0.0114 | 0.1942 | 1.0114 | 0.8244 | 0.187 | -0.0514 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 602 | 2 | 443 | 10 | 0.1654 | 0.0697 | 0.0498 | -0.0016 | 0.0114 | 0.1943 | 1.0114 | 0.8244 | 0.187 | -0.0515 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 603 | 2 | 444 | 10 | 0.1654 | 0.0697 | 0.0497 | -0.0016 | 0.0114 | 0.1945 | 1.0114 | 0.8252 | 0.1862 | -0.0514 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 604 | 2 | 445 | 10 | 0.1654 | 0.0697 | 0.0501 | -0.0016 | 0.0114 | 0.1946 | 1.0114 | 0.8237 | 0.1877 | -0.0517 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 605 | 2 | 446 | 10 | 0.1654 | 0.0697 | 0.0501 | -0.0016 | 0.0114 | 0.1948 | 1.0114 | 0.8238 | 0.1875 | -0.0517 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 606 | 2 | 447 | 10 | 0.1654 | 0.0697 | 0.05 | -0.0016 | 0.0114 | 0.195 | 1.0114 | 0.8244 | 0.187 | -0.0516 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 607 | 2 | 448 | 10 | 0.1654 | 0.0697 | 0.0499 | -0.0016 | 0.0113 | 0.1951 | 1.0113 | 0.8251 | 0.1863 | -0.0515 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 608 | 2 | 449 | 10 | 0.1654 | 0.0697 | 0.0497 | -0.0016 | 0.0113 | 0.1953 | 1.0113 | 0.826 | 0.1853 | -0.0513 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 609 | 2 | 450 | 10 | 0.1654 | 0.0697 | 0.0499 | -0.0016 | 0.0113 | 0.1954 | 1.0113 | 0.8253 | 0.186 | -0.0515 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 610 | 2 | 451 | 10 | 0.1654 | 0.0697 | 0.05 | -0.0016 | 0.0113 | 0.1956 | 1.0113 | 0.8251 | 0.1862 | -0.0516 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 611 | 2 | 452 | 10 | 0.1654 | 0.0697 | 0.0498 | -0.0016 | 0.0113 | 0.1958 | 1.0113 | 0.8261 | 0.1851 | -0.0514 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 612 | 2 | 453 | 10 | 0.1654 | 0.0697 | 0.0496 | -0.0016 | 0.0113 | 0.1959 | 1.0113 | 0.8272 | 0.184 | -0.0512 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 613 | 2 | 454 | 10 | 0.1654 | 0.0697 | 0.0495 | -0.0016 | 0.0113 | 0.1961 | 1.0113 | 0.8275 | 0.1837 | -0.0512 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 614 | 2 | 455 | 10 | 0.1654 | 0.0697 | 0.05 | -0.0016 | 0.0112 | 0.1962 | 1.0112 | 0.8255 | 0.1857 | -0.0517 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 615 | 2 | 456 | 10 | 0.1654 | 0.0697 | 0.0498 | -0.0016 | 0.0112 | 0.1964 | 1.0112 | 0.8265 | 0.1847 | -0.0515 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 616 | 2 | 457 | 10 | 0.1654 | 0.0697 | 0.0501 | -0.0016 | 0.0112 | 0.1966 | 1.0112 | 0.8256 | 0.1856 | -0.0517 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 617 | 2 | 458 | 10 | 0.1654 | 0.0697 | 0.0502 | -0.0016 | 0.0112 | 0.1967 | 1.0112 | 0.8251 | 0.1861 | -0.0519 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 618 | 2 | 459 | 10 | 0.1654 | 0.0697 | 0.0501 | -0.0016 | 0.0112 | 0.1969 | 1.0112 | 0.8257 | 0.1855 | -0.0518 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 619 | 2 | 460 | 10 | 0.1654 | 0.0697 | 0.0504 | -0.0016 | 0.0112 | 0.197 | 1.0112 | 0.8244 | 0.1868 | -0.0521 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 620 | 2 | 461 | 10 | 0.1654 | 0.0697 | 0.0503 | -0.0016 | 0.0111 | 0.1972 | 1.0111 | 0.8251 | 0.1861 | -0.052 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 621 | 2 | 462 | 10 | 0.1654 | 0.0697 | 0.0507 | -0.0016 | 0.0111 | 0.1974 | 1.0111 | 0.8237 | 0.1874 | -0.0523 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 622 | 2 | 463 | 10 | 0.1654 | 0.0697 | 0.0509 | -0.0016 | 0.0111 | 0.1975 | 1.0111 | 0.823 | 0.1881 | -0.0525 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 623 | 2 | 464 | 10 | 0.1654 | 0.0697 | 0.051 | -0.0016 | 0.0111 | 0.1977 | 1.0111 | 0.8227 | 0.1884 | -0.0526 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 624 | 2 | 465 | 10 | 0.1654 | 0.0697 | 0.0511 | -0.0016 | 0.0111 | 0.1978 | 1.0111 | 0.8222 | 0.1889 | -0.0528 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 625 | 2 | 466 | 10 | 0.1654 | 0.0697 | 0.051 | -0.0016 | 0.0111 | 0.198 | 1.0111 | 0.823 | 0.188 | -0.0526 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 626 | 2 | 467 | 10 | 0.1654 | 0.0697 | 0.0511 | -0.0016 | 0.0111 | 0.1982 | 1.0111 | 0.8227 | 0.1884 | -0.0527 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 627 | 2 | 468 | 10 | 0.1654 | 0.0697 | 0.0509 | -0.0016 | 0.011 | 0.1983 | 1.011 | 0.8235 | 0.1875 | -0.0526 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 628 | 2 | 469 | 10 | 0.1654 | 0.0697 | 0.0514 | -0.0016 | 0.011 | 0.1985 | 1.011 | 0.8213 | 0.1897 | -0.0531 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 629 | 2 | 470 | 10 | 0.1654 | 0.0697 | 0.0516 | -0.0016 | 0.011 | 0.1986 | 1.011 | 0.8207 | 0.1903 | -0.0533 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 630 | 2 | 471 | 10 | 0.1654 | 0.0697 | 0.0517 | -0.0016 | 0.011 | 0.1988 | 1.011 | 0.8208 | 0.1902 | -0.0533 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 631 | 2 | 472 | 10 | 0.1654 | 0.0697 | 0.0519 | -0.0016 | 0.011 | 0.1989 | 1.011 | 0.8197 | 0.1913 | -0.0536 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 632 | 2 | 473 | 10 | 0.1654 | 0.0697 | 0.0518 | -0.0016 | 0.011 | 0.1991 | 1.011 | 0.8203 | 0.1907 | -0.0535 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 633 | 2 | 474 | 10 | 0.1654 | 0.0697 | 0.0518 | -0.0016 | 0.011 | 0.1993 | 1.011 | 0.8208 | 0.1902 | -0.0534 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 634 | 2 | 475 | 10 | 0.1654 | 0.0697 | 0.0521 | -0.0016 | 0.0109 | 0.1994 | 1.0109 | 0.8193 | 0.1917 | -0.0538 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 635 | 2 | 476 | 10 | 0.1654 | 0.0697 | 0.0525 | -0.0016 | 0.0109 | 0.1996 | 1.0109 | 0.8178 | 0.1932 | -0.0542 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 636 | 2 | 477 | 10 | 0.1654 | 0.0697 | 0.0526 | -0.0016 | 0.0109 | 0.1997 | 1.0109 | 0.8177 | 0.1932 | -0.0542 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 637 | 2 | 478 | 10 | 0.1654 | 0.0697 | 0.0527 | -0.0016 | 0.0109 | 0.1999 | 1.0109 | 0.8171 | 0.1938 | -0.0544 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 638 | 2 | 479 | 10 | 0.1654 | 0.0697 | 0.053 | -0.0016 | 0.0109 | 0.2 | 1.0109 | 0.8163 | 0.1946 | -0.0546 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 639 | 2 | 480 | 10 | 0.1654 | 0.0697 | 0.0532 | -0.0016 | 0.0109 | 0.2002 | 1.0109 | 0.8153 | 0.1956 | -0.0549 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 640 | 2 | 481 | 10 | 0.1654 | 0.0697 | 0.0529 | -0.0016 | 0.0109 | 0.2004 | 1.0109 | 0.8169 | 0.194 | -0.0545 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 641 | 2 | 482 | 10 | 0.1654 | 0.0697 | 0.0527 | -0.0016 | 0.0108 | 0.2005 | 1.0108 | 0.8179 | 0.1929 | -0.0543 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 642 | 2 | 483 | 10 | 0.1654 | 0.0697 | 0.0529 | -0.0016 | 0.0108 | 0.2007 | 1.0108 | 0.8172 | 0.1937 | -0.0546 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 643 | 2 | 484 | 10 | 0.1654 | 0.0697 | 0.0526 | -0.0016 | 0.0108 | 0.2008 | 1.0108 | 0.8186 | 0.1922 | -0.0543 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 644 | 2 | 485 | 10 | 0.1654 | 0.0697 | 0.0525 | -0.0016 | 0.0108 | 0.201 | 1.0108 | 0.8193 | 0.1915 | -0.0541 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 645 | 2 | 486 | 10 | 0.1654 | 0.0697 | 0.0526 | -0.0016 | 0.0108 | 0.2011 | 1.0108 | 0.8188 | 0.192 | -0.0543 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 646 | 2 | 487 | 10 | 0.1654 | 0.0697 | 0.0523 | -0.0016 | 0.0108 | 0.2013 | 1.0108 | 0.8202 | 0.1905 | -0.054 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 647 | 2 | 488 | 10 | 0.1654 | 0.0697 | 0.0528 | -0.0016 | 0.0108 | 0.2015 | 1.0108 | 0.8185 | 0.1923 | -0.0544 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 648 | 2 | 489 | 10 | 0.1654 | 0.0697 | 0.0525 | -0.0016 | 0.0107 | 0.2016 | 1.0107 | 0.82 | 0.1908 | -0.0541 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 649 | 2 | 490 | 10 | 0.1654 | 0.0697 | 0.0529 | -0.0016 | 0.0107 | 0.2018 | 1.0107 | 0.8181 | 0.1926 | -0.0546 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 650 | 2 | 491 | 10 | 0.1654 | 0.0697 | 0.0533 | -0.0016 | 0.0107 | 0.2019 | 1.0107 | 0.8168 | 0.1939 | -0.0549 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 651 | 2 | 492 | 10 | 0.1654 | 0.0697 | 0.0533 | -0.0016 | 0.0107 | 0.2021 | 1.0107 | 0.8171 | 0.1936 | -0.0549 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 652 | 2 | 493 | 10 | 0.1654 | 0.0697 | 0.053 | -0.0016 | 0.0107 | 0.2022 | 1.0107 | 0.8181 | 0.1926 | -0.0547 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 653 | 2 | 494 | 10 | 0.1654 | 0.0697 | 0.0536 | -0.0016 | 0.0107 | 0.2024 | 1.0107 | 0.8158 | 0.1949 | -0.0553 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 654 | 2 | 495 | 10 | 0.1654 | 0.0697 | 0.054 | -0.0016 | 0.0107 | 0.2025 | 1.0107 | 0.8144 | 0.1963 | -0.0556 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 655 | 2 | 496 | 10 | 0.1654 | 0.0697 | 0.0541 | -0.0016 | 0.0106 | 0.2027 | 1.0106 | 0.8142 | 0.1964 | -0.0557 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 656 | 2 | 497 | 10 | 0.1654 | 0.0697 | 0.0545 | -0.0016 | 0.0106 | 0.2029 | 1.0106 | 0.8126 | 0.198 | -0.0561 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 657 | 2 | 498 | 10 | 0.1654 | 0.0697 | 0.0548 | -0.0016 | 0.0106 | 0.203 | 1.0106 | 0.8114 | 0.1993 | -0.0565 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 658 | 2 | 499 | 10 | 0.1654 | 0.0697 | 0.0549 | -0.0016 | 0.0106 | 0.2032 | 1.0106 | 0.8113 | 0.1993 | -0.0565 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 659 | 2 | 500 | 10 | 0.1654 | 0.0697 | 0.055 | -0.0016 | 0.0106 | 0.2033 | 1.0106 | 0.8109 | 0.1997 | -0.0566 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 660 | 2 | 501 | 10 | 0.1654 | 0.0697 | 0.0546 | -0.0016 | 0.0106 | 0.2035 | 1.0106 | 0.8127 | 0.1979 | -0.0562 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 661 | 2 | 502 | 10 | 0.1654 | 0.0697 | 0.0546 | -0.0016 | 0.0106 | 0.2036 | 1.0106 | 0.8127 | 0.1979 | -0.0563 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 662 | 2 | 503 | 10 | 0.1654 | 0.0697 | 0.0543 | -0.0016 | 0.0106 | 0.2038 | 1.0106 | 0.8141 | 0.1964 | -0.056 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 663 | 2 | 504 | 10 | 0.1654 | 0.0697 | 0.0543 | -0.0016 | 0.0105 | 0.2039 | 1.0105 | 0.8143 | 0.1962 | -0.056 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 664 | 2 | 505 | 10 | 0.1654 | 0.0697 | 0.0541 | -0.0016 | 0.0105 | 0.2041 | 1.0105 | 0.8155 | 0.1951 | -0.0557 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 665 | 2 | 506 | 10 | 0.1654 | 0.0697 | 0.0537 | -0.0016 | 0.0105 | 0.2042 | 1.0105 | 0.8172 | 0.1933 | -0.0554 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 666 | 2 | 507 | 10 | 0.1654 | 0.0697 | 0.0535 | -0.0016 | 0.0105 | 0.2044 | 1.0105 | 0.8183 | 0.1922 | -0.0552 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 667 | 2 | 508 | 10 | 0.1654 | 0.0697 | 0.0534 | -0.0016 | 0.0105 | 0.2045 | 1.0105 | 0.8187 | 0.1918 | -0.0551 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 668 | 2 | 509 | 10 | 0.1654 | 0.0697 | 0.0536 | -0.0016 | 0.0105 | 0.2047 | 1.0105 | 0.818 | 0.1925 | -0.0553 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 669 | 2 | 510 | 10 | 0.1654 | 0.0697 | 0.0537 | -0.0016 | 0.0105 | 0.2049 | 1.0105 | 0.8178 | 0.1926 | -0.0554 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 670 | 2 | 511 | 10 | 0.1654 | 0.0697 | 0.0537 | -0.0016 | 0.0105 | 0.205 | 1.0105 | 0.818 | 0.1924 | -0.0554 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 671 | 2 | 512 | 10 | 0.1654 | 0.0697 | 0.0536 | -0.0016 | 0.0104 | 0.2052 | 1.0104 | 0.8186 | 0.1918 | -0.0553 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 672 | 2 | 512 | 10 | 0.1654 | 0.0697 | 0.0536 | -0.0016 | 0.0104 | 0.2053 | 1.0104 | 0.8188 | 0.1917 | -0.0553 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 673 | 2 | 513 | 10 | 0.1654 | 0.0697 | 0.0534 | -0.0016 | 0.0104 | 0.2055 | 1.0104 | 0.8197 | 0.1908 | -0.0551 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 674 | 2 | 514 | 10 | 0.1654 | 0.0697 | 0.0538 | -0.0016 | 0.0104 | 0.2056 | 1.0104 | 0.8184 | 0.1921 | -0.0554 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 675 | 2 | 515 | 10 | 0.1654 | 0.0697 | 0.0535 | -0.0016 | 0.0104 | 0.2058 | 1.0104 | 0.8199 | 0.1905 | -0.0551 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 676 | 2 | 516 | 10 | 0.1654 | 0.0697 | 0.0533 | -0.0016 | 0.0104 | 0.2059 | 1.0104 | 0.8207 | 0.1897 | -0.0549 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 677 | 2 | 517 | 10 | 0.1654 | 0.0697 | 0.0531 | -0.0016 | 0.0104 | 0.2061 | 1.0104 | 0.8215 | 0.1888 | -0.0548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 678 | 2 | 518 | 10 | 0.1654 | 0.0697 | 0.0531 | -0.0016 | 0.0104 | 0.2062 | 1.0104 | 0.8218 | 0.1885 | -0.0548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 679 | 2 | 519 | 10 | 0.1654 | 0.0697 | 0.0532 | -0.0016 | 0.0104 | 0.2064 | 1.0104 | 0.8216 | 0.1887 | -0.0548 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 680 | 2 | 520 | 10 | 0.1654 | 0.0697 | 0.0532 | -0.0016 | 0.0103 | 0.2065 | 1.0103 | 0.8216 | 0.1888 | -0.0549 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 681 | 2 | 521 | 10 | 0.1654 | 0.0697 | 0.0533 | -0.0016 | 0.0103 | 0.2067 | 1.0103 | 0.8214 | 0.189 | -0.055 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 682 | 2 | 522 | 10 | 0.1654 | 0.0697 | 0.0537 | -0.0016 | 0.0103 | 0.2068 | 1.0103 | 0.8199 | 0.1904 | -0.0553 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 683 | 2 | 523 | 10 | 0.1654 | 0.0697 | 0.0536 | -0.0016 | 0.0103 | 0.207 | 1.0103 | 0.8205 | 0.1898 | -0.0552 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 684 | 2 | 524 | 10 | 0.1654 | 0.0697 | 0.0535 | -0.0016 | 0.0103 | 0.2071 | 1.0103 | 0.8212 | 0.1891 | -0.0551 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 685 | 2 | 525 | 10 | 0.1654 | 0.0697 | 0.0535 | -0.0016 | 0.0103 | 0.2073 | 1.0103 | 0.8212 | 0.1891 | -0.0552 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 686 | 2 | 526 | 10 | 0.1654 | 0.0697 | 0.0536 | -0.0016 | 0.0103 | 0.2074 | 1.0103 | 0.8209 | 0.1894 | -0.0553 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 687 | 2 | 527 | 10 | 0.1654 | 0.0697 | 0.0538 | -0.0016 | 0.0103 | 0.2076 | 1.0103 | 0.8203 | 0.1899 | -0.0554 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 688 | 2 | 528 | 10 | 0.1654 | 0.0697 | 0.0539 | -0.0016 | 0.0103 | 0.2077 | 1.0103 | 0.8198 | 0.1904 | -0.0556 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 689 | 2 | 529 | 10 | 0.1654 | 0.0697 | 0.0538 | -0.0016 | 0.0102 | 0.2079 | 1.0102 | 0.8203 | 0.1899 | -0.0555 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 690 | 2 | 530 | 10 | 0.1654 | 0.0697 | 0.0538 | -0.0016 | 0.0102 | 0.208 | 1.0102 | 0.8209 | 0.1893 | -0.0554 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 691 | 2 | 531 | 10 | 0.1654 | 0.0697 | 0.0536 | -0.0016 | 0.0102 | 0.2082 | 1.0102 | 0.8216 | 0.1886 | -0.0553 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 692 | 2 | 532 | 10 | 0.1654 | 0.0697 | 0.0535 | -0.0016 | 0.0102 | 0.2083 | 1.0102 | 0.8222 | 0.188 | -0.0552 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 693 | 2 | 533 | 10 | 0.1654 | 0.0697 | 0.0535 | -0.0016 | 0.0102 | 0.2085 | 1.0102 | 0.8225 | 0.1877 | -0.0551 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 694 | 2 | 534 | 10 | 0.1654 | 0.0697 | 0.0534 | -0.0016 | 0.0102 | 0.2086 | 1.0102 | 0.8231 | 0.1871 | -0.055 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 695 | 2 | 535 | 10 | 0.1654 | 0.0697 | 0.0533 | -0.0016 | 0.0102 | 0.2088 | 1.0102 | 0.8234 | 0.1868 | -0.055 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 696 | 2 | 536 | 10 | 0.1654 | 0.0697 | 0.0533 | -0.0016 | 0.0102 | 0.2089 | 1.0102 | 0.8237 | 0.1864 | -0.0549 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 697 | 2 | 537 | 10 | 0.1654 | 0.0697 | 0.0531 | -0.0016 | 0.0101 | 0.2091 | 1.0101 | 0.825 | 0.1852 | -0.0547 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 698 | 2 | 538 | 10 | 0.1654 | 0.0697 | 0.053 | -0.0016 | 0.0101 | 0.2092 | 1.0101 | 0.8254 | 0.1848 | -0.0546 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 699 | 2 | 539 | 10 | 0.1654 | 0.0697 | 0.0529 | -0.0016 | 0.0101 | 0.2094 | 1.0101 | 0.8257 | 0.1844 | -0.0546 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 700 | 2 | 540 | 10 | 0.1654 | 0.0697 | 0.0533 | -0.0016 | 0.0101 | 0.2095 | 1.0101 | 0.8243 | 0.1858 | -0.055 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 701 | 2 | 541 | 10 | 0.1654 | 0.0697 | 0.0535 | -0.0016 | 0.0101 | 0.2097 | 1.0101 | 0.8235 | 0.1866 | -0.0552 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 702 | 2 | 542 | 10 | 0.1654 | 0.0697 | 0.0538 | -0.0016 | 0.0101 | 0.2098 | 1.0101 | 0.8223 | 0.1878 | -0.0555 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 703 | 2 | 543 | 10 | 0.1654 | 0.0697 | 0.0539 | -0.0016 | 0.0101 | 0.21 | 1.0101 | 0.8224 | 0.1877 | -0.0555 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 704 | 2 | 544 | 10 | 0.1654 | 0.0697 | 0.0539 | -0.0016 | 0.0101 | 0.2101 | 1.0101 | 0.8223 | 0.1878 | -0.0556 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 705 | 2 | 545 | 10 | 0.1654 | 0.0697 | 0.0544 | -0.0016 | 0.0101 | 0.2103 | 1.0101 | 0.8205 | 0.1896 | -0.056 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 706 | 2 | 546 | 10 | 0.1654 | 0.0697 | 0.0547 | -0.0016 | 0.01 | 0.2104 | 1.01 | 0.8191 | 0.191 | -0.0564 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 707 | 2 | 547 | 10 | 0.1654 | 0.0697 | 0.0552 | -0.0016 | 0.01 | 0.2106 | 1.01 | 0.8173 | 0.1928 | -0.0569 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 708 | 2 | 548 | 10 | 0.1654 | 0.0697 | 0.0549 | -0.0016 | 0.01 | 0.2107 | 1.01 | 0.8185 | 0.1915 | -0.0566 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 709 | 2 | 549 | 10 | 0.1654 | 0.0697 | 0.0549 | -0.0016 | 0.01 | 0.2109 | 1.01 | 0.819 | 0.191 | -0.0565 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 710 | 2 | 550 | 10 | 0.1654 | 0.0697 | 0.0554 | -0.0016 | 0.01 | 0.211 | 1.01 | 0.817 | 0.193 | -0.057 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 711 | 2 | 551 | 10 | 0.1654 | 0.0697 | 0.0553 | -0.0016 | 0.01 | 0.2112 | 1.01 | 0.8173 | 0.1927 | -0.057 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 712 | 2 | 552 | 10 | 0.1654 | 0.0697 | 0.0558 | -0.0016 | 0.01 | 0.2113 | 1.01 | 0.8157 | 0.1943 | -0.0574 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 713 | 2 | 553 | 10 | 0.1654 | 0.0697 | 0.056 | -0.0016 | 0.01 | 0.2115 | 1.01 | 0.8147 | 0.1953 | -0.0577 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 714 | 2 | 554 | 10 | 0.1654 | 0.0697 | 0.0563 | -0.0016 | 0.01 | 0.2116 | 1.01 | 0.8139 | 0.1961 | -0.0579 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 715 | 2 | 555 | 10 | 0.1654 | 0.0697 | 0.0566 | -0.0016 | 0.0099 | 0.2118 | 1.0099 | 0.8128 | 0.1972 | -0.0582 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 716 | 2 | 556 | 10 | 0.1654 | 0.0697 | 0.0566 | -0.0016 | 0.0099 | 0.2119 | 1.0099 | 0.8128 | 0.1972 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 717 | 2 | 557 | 10 | 0.1654 | 0.0697 | 0.0564 | -0.0016 | 0.0099 | 0.2121 | 1.0099 | 0.8137 | 0.1963 | -0.0581 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 718 | 2 | 558 | 10 | 0.1654 | 0.0697 | 0.0563 | -0.0016 | 0.0099 | 0.2122 | 1.0099 | 0.8145 | 0.1954 | -0.0579 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 719 | 2 | 559 | 10 | 0.1654 | 0.0697 | 0.0561 | -0.0016 | 0.0099 | 0.2124 | 1.0099 | 0.8154 | 0.1945 | -0.0577 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 720 | 2 | 560 | 10 | 0.1654 | 0.0697 | 0.0565 | -0.0016 | 0.0099 | 0.2125 | 1.0099 | 0.8137 | 0.1962 | -0.0582 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 721 | 2 | 561 | 10 | 0.1654 | 0.0697 | 0.0567 | -0.0016 | 0.0099 | 0.2127 | 1.0099 | 0.8131 | 0.1967 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 722 | 2 | 562 | 10 | 0.1654 | 0.0697 | 0.0571 | -0.0016 | 0.0099 | 0.2128 | 1.0099 | 0.8116 | 0.1982 | -0.0588 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 723 | 2 | 563 | 10 | 0.1654 | 0.0697 | 0.0575 | -0.0016 | 0.0099 | 0.213 | 1.0099 | 0.8101 | 0.1998 | -0.0592 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 724 | 2 | 564 | 10 | 0.1654 | 0.0697 | 0.0572 | -0.0016 | 0.0098 | 0.2131 | 1.0098 | 0.8116 | 0.1982 | -0.0588 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 725 | 2 | 565 | 10 | 0.1654 | 0.0697 | 0.057 | -0.0016 | 0.0098 | 0.2133 | 1.0098 | 0.8124 | 0.1974 | -0.0587 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 726 | 2 | 566 | 10 | 0.1654 | 0.0697 | 0.0573 | -0.0016 | 0.0098 | 0.2134 | 1.0098 | 0.8115 | 0.1984 | -0.0589 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 727 | 2 | 567 | 10 | 0.1654 | 0.0697 | 0.0569 | -0.0016 | 0.0098 | 0.2135 | 1.0098 | 0.8132 | 0.1966 | -0.0585 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 728 | 2 | 568 | 10 | 0.1654 | 0.0697 | 0.057 | -0.0016 | 0.0098 | 0.2137 | 1.0098 | 0.8131 | 0.1967 | -0.0586 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 729 | 2 | 569 | 10 | 0.1654 | 0.0697 | 0.0574 | -0.0016 | 0.0098 | 0.2138 | 1.0098 | 0.8113 | 0.1985 | -0.0591 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 730 | 2 | 570 | 10 | 0.1654 | 0.0697 | 0.0573 | -0.0016 | 0.0098 | 0.214 | 1.0098 | 0.8121 | 0.1977 | -0.0589 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 731 | 2 | 571 | 10 | 0.1654 | 0.0697 | 0.0571 | -0.0016 | 0.0098 | 0.2141 | 1.0098 | 0.8129 | 0.1968 | -0.0588 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 732 | 2 | 572 | 10 | 0.1654 | 0.0697 | 0.0575 | -0.0016 | 0.0098 | 0.2143 | 1.0098 | 0.8115 | 0.1982 | -0.0591 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 733 | 2 | 573 | 10 | 0.1654 | 0.0697 | 0.0579 | -0.0016 | 0.0098 | 0.2144 | 1.0098 | 0.8099 | 0.1998 | -0.0596 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 734 | 2 | 574 | 10 | 0.1654 | 0.0697 | 0.0578 | -0.0016 | 0.0097 | 0.2146 | 1.0097 | 0.8106 | 0.1991 | -0.0594 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 735 | 2 | 575 | 10 | 0.1654 | 0.0697 | 0.0577 | -0.0016 | 0.0097 | 0.2147 | 1.0097 | 0.8112 | 0.1986 | -0.0593 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 736 | 2 | 576 | 10 | 0.1654 | 0.0697 | 0.0575 | -0.0016 | 0.0097 | 0.2149 | 1.0097 | 0.8122 | 0.1976 | -0.0591 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 737 | 2 | 577 | 10 | 0.1654 | 0.0697 | 0.0572 | -0.0016 | 0.0097 | 0.215 | 1.0097 | 0.8136 | 0.1961 | -0.0588 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 738 | 2 | 578 | 10 | 0.1654 | 0.0697 | 0.0569 | -0.0016 | 0.0097 | 0.2152 | 1.0097 | 0.8147 | 0.195 | -0.0586 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 739 | 2 | 579 | 10 | 0.1654 | 0.0697 | 0.057 | -0.0016 | 0.0097 | 0.2153 | 1.0097 | 0.8144 | 0.1953 | -0.0587 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 740 | 2 | 580 | 10 | 0.1654 | 0.0697 | 0.0571 | -0.0016 | 0.0097 | 0.2154 | 1.0097 | 0.8141 | 0.1956 | -0.0588 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 741 | 2 | 581 | 10 | 0.1654 | 0.0697 | 0.0568 | -0.0016 | 0.0097 | 0.2156 | 1.0097 | 0.8156 | 0.1941 | -0.0585 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 742 | 2 | 582 | 10 | 0.1654 | 0.0697 | 0.057 | -0.0016 | 0.0097 | 0.2157 | 1.0097 | 0.815 | 0.1946 | -0.0586 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 743 | 2 | 583 | 10 | 0.1654 | 0.0697 | 0.0569 | -0.0016 | 0.0097 | 0.2159 | 1.0097 | 0.8153 | 0.1943 | -0.0586 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 744 | 2 | 584 | 10 | 0.1654 | 0.0697 | 0.0572 | -0.0016 | 0.0096 | 0.216 | 1.0096 | 0.8144 | 0.1952 | -0.0589 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 745 | 2 | 585 | 10 | 0.1654 | 0.0697 | 0.0571 | -0.0016 | 0.0096 | 0.2162 | 1.0096 | 0.8148 | 0.1948 | -0.0588 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 746 | 2 | 586 | 10 | 0.1654 | 0.0697 | 0.0571 | -0.0016 | 0.0096 | 0.2163 | 1.0096 | 0.815 | 0.1947 | -0.0588 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 747 | 2 | 587 | 10 | 0.1654 | 0.0697 | 0.057 | -0.0016 | 0.0096 | 0.2165 | 1.0096 | 0.8158 | 0.1938 | -0.0586 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 748 | 2 | 588 | 10 | 0.1654 | 0.0697 | 0.057 | -0.0016 | 0.0096 | 0.2166 | 1.0096 | 0.8157 | 0.1939 | -0.0587 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 749 | 2 | 589 | 10 | 0.1654 | 0.0697 | 0.0573 | -0.0016 | 0.0096 | 0.2168 | 1.0096 | 0.8148 | 0.1948 | -0.0589 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 750 | 2 | 590 | 10 | 0.1654 | 0.0697 | 0.0573 | -0.0016 | 0.0096 | 0.2169 | 1.0096 | 0.8148 | 0.1948 | -0.059 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 751 | 2 | 591 | 10 | 0.1654 | 0.0697 | 0.0571 | -0.0016 | 0.0096 | 0.217 | 1.0096 | 0.8159 | 0.1937 | -0.0587 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 752 | 2 | 592 | 10 | 0.1654 | 0.0697 | 0.0567 | -0.0016 | 0.0096 | 0.2172 | 1.0096 | 0.8175 | 0.1921 | -0.0584 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 753 | 2 | 593 | 10 | 0.1654 | 0.0697 | 0.0564 | -0.0016 | 0.0096 | 0.2173 | 1.0096 | 0.8192 | 0.1904 | -0.058 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 754 | 2 | 594 | 10 | 0.1654 | 0.0697 | 0.0563 | -0.0016 | 0.0095 | 0.2175 | 1.0095 | 0.8195 | 0.19 | -0.058 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 755 | 2 | 595 | 10 | 0.1654 | 0.0697 | 0.0565 | -0.0016 | 0.0095 | 0.2176 | 1.0095 | 0.819 | 0.1905 | -0.0581 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 756 | 2 | 596 | 10 | 0.1654 | 0.0697 | 0.0565 | -0.0016 | 0.0095 | 0.2178 | 1.0095 | 0.819 | 0.1905 | -0.0582 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 757 | 2 | 597 | 10 | 0.1654 | 0.0697 | 0.0564 | -0.0016 | 0.0095 | 0.2179 | 1.0095 | 0.8197 | 0.1899 | -0.058 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 758 | 2 | 598 | 10 | 0.1654 | 0.0697 | 0.0564 | -0.0016 | 0.0095 | 0.2181 | 1.0095 | 0.8199 | 0.1896 | -0.058 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 759 | 2 | 599 | 10 | 0.1654 | 0.0697 | 0.0562 | -0.0016 | 0.0095 | 0.2182 | 1.0095 | 0.8209 | 0.1886 | -0.0578 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 760 | 2 | 600 | 10 | 0.1654 | 0.0697 | 0.0564 | -0.0016 | 0.0095 | 0.2183 | 1.0095 | 0.8201 | 0.1893 | -0.058 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 761 | 2 | 601 | 10 | 0.1654 | 0.0697 | 0.0566 | -0.0016 | 0.0095 | 0.2185 | 1.0095 | 0.8194 | 0.1901 | -0.0582 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 762 | 2 | 602 | 10 | 0.1654 | 0.0697 | 0.0566 | -0.0016 | 0.0095 | 0.2186 | 1.0095 | 0.8197 | 0.1898 | -0.0582 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 763 | 2 | 603 | 10 | 0.1654 | 0.0697 | 0.0564 | -0.0016 | 0.0095 | 0.2188 | 1.0095 | 0.8205 | 0.189 | -0.058 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 764 | 2 | 604 | 10 | 0.1654 | 0.0697 | 0.0563 | -0.0016 | 0.0094 | 0.2189 | 1.0094 | 0.8212 | 0.1883 | -0.0579 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 765 | 2 | 605 | 10 | 0.1654 | 0.0697 | 0.0562 | -0.0016 | 0.0094 | 0.2191 | 1.0094 | 0.8218 | 0.1877 | -0.0578 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 766 | 2 | 606 | 10 | 0.1654 | 0.0697 | 0.0561 | -0.0016 | 0.0094 | 0.2192 | 1.0094 | 0.8222 | 0.1872 | -0.0577 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 767 | 2 | 607 | 10 | 0.1654 | 0.0697 | 0.0558 | -0.0016 | 0.0094 | 0.2193 | 1.0094 | 0.8236 | 0.1858 | -0.0574 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 768 | 2 | 608 | 10 | 0.1654 | 0.0697 | 0.0556 | -0.0016 | 0.0094 | 0.2195 | 1.0094 | 0.8245 | 0.1849 | -0.0572 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 769 | 2 | 609 | 10 | 0.1654 | 0.0697 | 0.0556 | -0.0016 | 0.0094 | 0.2196 | 1.0094 | 0.8247 | 0.1847 | -0.0572 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 770 | 2 | 610 | 10 | 0.1654 | 0.0697 | 0.0554 | -0.0016 | 0.0094 | 0.2198 | 1.0094 | 0.8255 | 0.1839 | -0.0571 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 771 | 2 | 611 | 10 | 0.1654 | 0.0697 | 0.0552 | -0.0016 | 0.0094 | 0.2199 | 1.0094 | 0.8265 | 0.1829 | -0.0569 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 772 | 2 | 612 | 10 | 0.1654 | 0.0697 | 0.0549 | -0.0016 | 0.0094 | 0.2201 | 1.0094 | 0.8281 | 0.1813 | -0.0565 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 773 | 2 | 613 | 10 | 0.1654 | 0.0697 | 0.0549 | -0.0016 | 0.0094 | 0.2202 | 1.0094 | 0.828 | 0.1814 | -0.0566 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 774 | 2 | 614 | 10 | 0.1654 | 0.0697 | 0.055 | -0.0016 | 0.0094 | 0.2203 | 1.0094 | 0.8279 | 0.1814 | -0.0566 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 775 | 2 | 615 | 10 | 0.1654 | 0.0697 | 0.0547 | -0.0016 | 0.0093 | 0.2205 | 1.0093 | 0.8295 | 0.1799 | -0.0563 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 776 | 2 | 616 | 10 | 0.1654 | 0.0697 | 0.055 | -0.0016 | 0.0093 | 0.2206 | 1.0093 | 0.8284 | 0.181 | -0.0566 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 777 | 2 | 617 | 10 | 0.1654 | 0.0697 | 0.0551 | -0.0016 | 0.0093 | 0.2208 | 1.0093 | 0.8278 | 0.1815 | -0.0568 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 778 | 2 | 618 | 10 | 0.1654 | 0.0697 | 0.0553 | -0.0016 | 0.0093 | 0.2209 | 1.0093 | 0.8273 | 0.182 | -0.0569 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 779 | 2 | 619 | 10 | 0.1654 | 0.0697 | 0.0553 | -0.0016 | 0.0093 | 0.2211 | 1.0093 | 0.8274 | 0.182 | -0.0569 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 780 | 2 | 620 | 10 | 0.1654 | 0.0697 | 0.0553 | -0.0016 | 0.0093 | 0.2212 | 1.0093 | 0.8274 | 0.1819 | -0.057 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 781 | 2 | 621 | 10 | 0.1654 | 0.0697 | 0.0554 | -0.0016 | 0.0093 | 0.2213 | 1.0093 | 0.8273 | 0.182 | -0.057 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 782 | 2 | 622 | 10 | 0.1654 | 0.0697 | 0.0557 | -0.0016 | 0.0093 | 0.2215 | 1.0093 | 0.8261 | 0.1832 | -0.0574 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 783 | 2 | 623 | 10 | 0.1654 | 0.0697 | 0.0556 | -0.0016 | 0.0093 | 0.2216 | 1.0093 | 0.8267 | 0.1826 | -0.0572 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 784 | 2 | 624 | 10 | 0.1654 | 0.0697 | 0.0555 | -0.0016 | 0.0093 | 0.2218 | 1.0093 | 0.8273 | 0.182 | -0.0571 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 785 | 2 | 625 | 10 | 0.1654 | 0.0697 | 0.0557 | -0.0016 | 0.0093 | 0.2219 | 1.0093 | 0.8266 | 0.1827 | -0.0573 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 786 | 2 | 626 | 10 | 0.1654 | 0.0697 | 0.0556 | -0.0016 | 0.0092 | 0.222 | 1.0092 | 0.8269 | 0.1823 | -0.0573 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 787 | 2 | 627 | 10 | 0.1654 | 0.0697 | 0.056 | -0.0016 | 0.0092 | 0.2222 | 1.0092 | 0.8256 | 0.1836 | -0.0576 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 788 | 2 | 628 | 10 | 0.1654 | 0.0697 | 0.0557 | -0.0016 | 0.0092 | 0.2223 | 1.0092 | 0.8271 | 0.1822 | -0.0573 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 789 | 2 | 629 | 10 | 0.1654 | 0.0697 | 0.0557 | -0.0016 | 0.0092 | 0.2225 | 1.0092 | 0.8273 | 0.182 | -0.0573 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 790 | 2 | 630 | 10 | 0.1654 | 0.0697 | 0.056 | -0.0016 | 0.0092 | 0.2226 | 1.0092 | 0.8259 | 0.1833 | -0.0577 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 791 | 2 | 631 | 10 | 0.1654 | 0.0697 | 0.0559 | -0.0016 | 0.0092 | 0.2227 | 1.0092 | 0.8264 | 0.1828 | -0.0576 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 792 | 2 | 632 | 10 | 0.1654 | 0.0697 | 0.0563 | -0.0016 | 0.0092 | 0.2229 | 1.0092 | 0.8251 | 0.1841 | -0.0579 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 793 | 2 | 633 | 10 | 0.1654 | 0.0697 | 0.0565 | -0.0016 | 0.0092 | 0.223 | 1.0092 | 0.8243 | 0.1849 | -0.0581 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 794 | 2 | 634 | 10 | 0.1654 | 0.0697 | 0.0567 | -0.0016 | 0.0092 | 0.2232 | 1.0092 | 0.8237 | 0.1855 | -0.0583 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 795 | 2 | 635 | 10 | 0.1654 | 0.0697 | 0.0571 | -0.0016 | 0.0092 | 0.2233 | 1.0092 | 0.8221 | 0.1871 | -0.0588 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 796 | 2 | 636 | 10 | 0.1654 | 0.0697 | 0.0575 | -0.0016 | 0.0092 | 0.2235 | 1.0092 | 0.8208 | 0.1884 | -0.0591 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 797 | 2 | 637 | 10 | 0.1654 | 0.0697 | 0.0577 | -0.0016 | 0.0091 | 0.2236 | 1.0091 | 0.8202 | 0.189 | -0.0593 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 798 | 2 | 638 | 10 | 0.1654 | 0.0697 | 0.0578 | -0.0016 | 0.0091 | 0.2237 | 1.0091 | 0.8197 | 0.1895 | -0.0595 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 799 | 2 | 639 | 10 | 0.1654 | 0.0697 | 0.0581 | -0.0016 | 0.0091 | 0.2239 | 1.0091 | 0.8186 | 0.1905 | -0.0597 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 800 | 2 | 640 | 10 | 0.1654 | 0.0697 | 0.0584 | -0.0016 | 0.0091 | 0.224 | 1.0091 | 0.8177 | 0.1914 | -0.06 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 801 | 2 | 641 | 10 | 0.1654 | 0.0697 | 0.0587 | -0.0016 | 0.0091 | 0.2242 | 1.0091 | 0.8167 | 0.1924 | -0.0603 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 802 | 2 | 642 | 10 | 0.1654 | 0.0697 | 0.0585 | -0.0016 | 0.0091 | 0.2243 | 1.0091 | 0.8176 | 0.1915 | -0.0601 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 803 | 2 | 643 | 10 | 0.1654 | 0.0697 | 0.0585 | -0.0016 | 0.0091 | 0.2244 | 1.0091 | 0.8177 | 0.1914 | -0.0601 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 804 | 2 | 644 | 10 | 0.1654 | 0.0697 | 0.0584 | -0.0016 | 0.0091 | 0.2246 | 1.0091 | 0.8181 | 0.191 | -0.0601 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 805 | 2 | 645 | 10 | 0.1654 | 0.0697 | 0.0584 | -0.0016 | 0.0091 | 0.2247 | 1.0091 | 0.8184 | 0.1906 | -0.06 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 806 | 2 | 646 | 10 | 0.1654 | 0.0697 | 0.0587 | -0.0016 | 0.0091 | 0.2248 | 1.0091 | 0.8174 | 0.1917 | -0.0603 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 807 | 2 | 647 | 10 | 0.1654 | 0.0697 | 0.0586 | -0.0016 | 0.0091 | 0.225 | 1.0091 | 0.8178 | 0.1912 | -0.0602 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 808 | 2 | 648 | 10 | 0.1654 | 0.0697 | 0.0587 | -0.0016 | 0.0091 | 0.2251 | 1.0091 | 0.8176 | 0.1915 | -0.0603 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 809 | 2 | 649 | 10 | 0.1654 | 0.0697 | 0.0587 | -0.0016 | 0.009 | 0.2253 | 1.009 | 0.8176 | 0.1914 | -0.0603 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 810 | 2 | 650 | 10 | 0.1654 | 0.0697 | 0.059 | -0.0016 | 0.009 | 0.2254 | 1.009 | 0.8166 | 0.1925 | -0.0607 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 811 | 2 | 651 | 10 | 0.1654 | 0.0697 | 0.0591 | -0.0016 | 0.009 | 0.2255 | 1.009 | 0.8163 | 0.1928 | -0.0608 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 812 | 2 | 652 | 10 | 0.1654 | 0.0697 | 0.0589 | -0.0016 | 0.009 | 0.2257 | 1.009 | 0.8171 | 0.1919 | -0.0606 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 813 | 2 | 653 | 10 | 0.1654 | 0.0697 | 0.0587 | -0.0016 | 0.009 | 0.2258 | 1.009 | 0.8182 | 0.1908 | -0.0603 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 814 | 2 | 654 | 10 | 0.1654 | 0.0697 | 0.0585 | -0.0016 | 0.009 | 0.226 | 1.009 | 0.8191 | 0.1899 | -0.0602 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 815 | 2 | 655 | 10 | 0.1654 | 0.0697 | 0.0581 | -0.0016 | 0.009 | 0.2261 | 1.009 | 0.8208 | 0.1882 | -0.0598 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 816 | 2 | 656 | 10 | 0.1654 | 0.0697 | 0.0579 | -0.0016 | 0.009 | 0.2262 | 1.009 | 0.8219 | 0.1871 | -0.0595 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 817 | 2 | 657 | 10 | 0.1654 | 0.0697 | 0.0579 | -0.0016 | 0.009 | 0.2264 | 1.009 | 0.8219 | 0.1871 | -0.0596 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 818 | 2 | 658 | 10 | 0.1654 | 0.0697 | 0.0579 | -0.0016 | 0.009 | 0.2265 | 1.009 | 0.822 | 0.187 | -0.0596 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 819 | 2 | 659 | 10 | 0.1654 | 0.0697 | 0.0578 | -0.0016 | 0.009 | 0.2267 | 1.009 | 0.8224 | 0.1865 | -0.0595 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 820 | 2 | 660 | 10 | 0.1654 | 0.0697 | 0.058 | -0.0016 | 0.009 | 0.2268 | 1.009 | 0.8219 | 0.1871 | -0.0597 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 821 | 2 | 661 | 10 | 0.1654 | 0.0697 | 0.0581 | -0.0016 | 0.0089 | 0.2269 | 1.0089 | 0.8216 | 0.1873 | -0.0598 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 822 | 2 | 662 | 10 | 0.1654 | 0.0697 | 0.0583 | -0.0016 | 0.0089 | 0.2271 | 1.0089 | 0.821 | 0.188 | -0.06 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 823 | 2 | 663 | 10 | 0.1654 | 0.0697 | 0.0586 | -0.0016 | 0.0089 | 0.2272 | 1.0089 | 0.8198 | 0.1891 | -0.0603 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 824 | 2 | 664 | 10 | 0.1654 | 0.0697 | 0.059 | -0.0016 | 0.0089 | 0.2273 | 1.0089 | 0.8185 | 0.1904 | -0.0606 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 825 | 2 | 665 | 10 | 0.1654 | 0.0697 | 0.0588 | -0.0016 | 0.0089 | 0.2275 | 1.0089 | 0.8196 | 0.1894 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 826 | 2 | 665 | 10 | 0.1654 | 0.0697 | 0.0588 | -0.0016 | 0.0089 | 0.2276 | 1.0089 | 0.8197 | 0.1892 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 827 | 2 | 666 | 10 | 0.1654 | 0.0697 | 0.0584 | -0.0016 | 0.0089 | 0.2278 | 1.0089 | 0.8212 | 0.1877 | -0.0601 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 828 | 2 | 667 | 10 | 0.1654 | 0.0697 | 0.0588 | -0.0016 | 0.0089 | 0.2279 | 1.0089 | 0.8199 | 0.189 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 829 | 2 | 668 | 10 | 0.1654 | 0.0697 | 0.0585 | -0.0016 | 0.0089 | 0.228 | 1.0089 | 0.8211 | 0.1878 | -0.0602 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 830 | 2 | 669 | 10 | 0.1654 | 0.0697 | 0.0586 | -0.0016 | 0.0089 | 0.2282 | 1.0089 | 0.8208 | 0.1881 | -0.0603 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 831 | 2 | 670 | 10 | 0.1654 | 0.0697 | 0.0586 | -0.0016 | 0.0089 | 0.2283 | 1.0089 | 0.821 | 0.1879 | -0.0603 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 832 | 2 | 671 | 10 | 0.1654 | 0.0697 | 0.0586 | -0.0016 | 0.0089 | 0.2284 | 1.0089 | 0.8214 | 0.1875 | -0.0602 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 833 | 2 | 672 | 10 | 0.1654 | 0.0697 | 0.0584 | -0.0016 | 0.0089 | 0.2286 | 1.0089 | 0.8223 | 0.1866 | -0.06 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 834 | 2 | 673 | 10 | 0.1654 | 0.0697 | 0.0584 | -0.0016 | 0.0089 | 0.2287 | 1.0089 | 0.8224 | 0.1864 | -0.06 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 835 | 2 | 674 | 10 | 0.1654 | 0.0697 | 0.0582 | -0.0016 | 0.0089 | 0.2289 | 1.0089 | 0.8233 | 0.1856 | -0.0598 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 836 | 2 | 675 | 10 | 0.1654 | 0.0697 | 0.0582 | -0.0016 | 0.0088 | 0.229 | 1.0088 | 0.8232 | 0.1856 | -0.0599 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 837 | 2 | 676 | 10 | 0.1654 | 0.0697 | 0.0583 | -0.0016 | 0.0088 | 0.2291 | 1.0088 | 0.8232 | 0.1857 | -0.0599 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 838 | 2 | 677 | 10 | 0.1654 | 0.0697 | 0.0588 | -0.0016 | 0.0088 | 0.2293 | 1.0088 | 0.8214 | 0.1874 | -0.0604 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 839 | 2 | 678 | 10 | 0.1654 | 0.0697 | 0.0591 | -0.0016 | 0.0088 | 0.2294 | 1.0088 | 0.8204 | 0.1885 | -0.0607 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 840 | 2 | 679 | 10 | 0.1654 | 0.0697 | 0.0592 | -0.0016 | 0.0088 | 0.2295 | 1.0088 | 0.82 | 0.1888 | -0.0608 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 841 | 2 | 680 | 10 | 0.1654 | 0.0697 | 0.0595 | -0.0016 | 0.0088 | 0.2297 | 1.0088 | 0.8187 | 0.1901 | -0.0612 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 842 | 2 | 681 | 10 | 0.1654 | 0.0697 | 0.0597 | -0.0016 | 0.0088 | 0.2298 | 1.0088 | 0.8181 | 0.1907 | -0.0614 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 843 | 2 | 682 | 10 | 0.1654 | 0.0697 | 0.0599 | -0.0016 | 0.0088 | 0.23 | 1.0088 | 0.8175 | 0.1913 | -0.0616 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 844 | 2 | 683 | 10 | 0.1654 | 0.0697 | 0.0602 | -0.0016 | 0.0088 | 0.2301 | 1.0088 | 0.8164 | 0.1924 | -0.0619 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 845 | 2 | 684 | 10 | 0.1654 | 0.0697 | 0.0604 | -0.0016 | 0.0088 | 0.2302 | 1.0088 | 0.8157 | 0.1931 | -0.0621 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 846 | 2 | 685 | 10 | 0.1654 | 0.0697 | 0.0608 | -0.0016 | 0.0088 | 0.2304 | 1.0088 | 0.8144 | 0.1943 | -0.0625 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 847 | 2 | 686 | 10 | 0.1654 | 0.0697 | 0.0612 | -0.0016 | 0.0088 | 0.2305 | 1.0088 | 0.8131 | 0.1956 | -0.0628 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 848 | 2 | 687 | 10 | 0.1654 | 0.0697 | 0.0618 | -0.0016 | 0.0088 | 0.2306 | 1.0088 | 0.8109 | 0.1979 | -0.0634 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 849 | 2 | 688 | 10 | 0.1654 | 0.0697 | 0.062 | -0.0016 | 0.0087 | 0.2308 | 1.0087 | 0.8101 | 0.1986 | -0.0637 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 850 | 2 | 689 | 10 | 0.1654 | 0.0697 | 0.0625 | -0.0016 | 0.0087 | 0.2309 | 1.0087 | 0.8085 | 0.2002 | -0.0641 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 851 | 2 | 690 | 10 | 0.1654 | 0.0697 | 0.0626 | -0.0016 | 0.0087 | 0.231 | 1.0087 | 0.8084 | 0.2004 | -0.0642 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 852 | 2 | 691 | 10 | 0.1654 | 0.0697 | 0.0628 | -0.0016 | 0.0087 | 0.2312 | 1.0087 | 0.8075 | 0.2013 | -0.0645 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 853 | 2 | 692 | 10 | 0.1654 | 0.0697 | 0.0633 | -0.0016 | 0.0087 | 0.2313 | 1.0087 | 0.806 | 0.2028 | -0.0649 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 854 | 2 | 693 | 10 | 0.1654 | 0.0697 | 0.0635 | -0.0016 | 0.0087 | 0.2314 | 1.0087 | 0.8051 | 0.2036 | -0.0652 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 855 | 2 | 694 | 10 | 0.1654 | 0.0697 | 0.0638 | -0.0016 | 0.0087 | 0.2316 | 1.0087 | 0.8044 | 0.2043 | -0.0654 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 856 | 2 | 695 | 10 | 0.1654 | 0.0697 | 0.0639 | -0.0016 | 0.0087 | 0.2317 | 1.0087 | 0.804 | 0.2047 | -0.0655 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 857 | 2 | 696 | 10 | 0.1654 | 0.0697 | 0.0644 | -0.0016 | 0.0087 | 0.2319 | 1.0087 | 0.8022 | 0.2065 | -0.0661 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 858 | 2 | 697 | 10 | 0.1654 | 0.0697 | 0.0645 | -0.0016 | 0.0087 | 0.232 | 1.0087 | 0.8022 | 0.2065 | -0.0661 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 859 | 2 | 698 | 10 | 0.1654 | 0.0697 | 0.0648 | -0.0016 | 0.0087 | 0.2321 | 1.0087 | 0.8011 | 0.2076 | -0.0664 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 860 | 2 | 699 | 10 | 0.1654 | 0.0697 | 0.0649 | -0.0016 | 0.0087 | 0.2323 | 1.0087 | 0.801 | 0.2077 | -0.0665 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 861 | 2 | 700 | 10 | 0.1654 | 0.0697 | 0.0649 | -0.0016 | 0.0087 | 0.2324 | 1.0087 | 0.801 | 0.2077 | -0.0665 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 862 | 2 | 701 | 10 | 0.1654 | 0.0697 | 0.065 | -0.0016 | 0.0086 | 0.2325 | 1.0086 | 0.8009 | 0.2078 | -0.0666 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 863 | 2 | 702 | 10 | 0.1654 | 0.0697 | 0.0654 | -0.0016 | 0.0086 | 0.2327 | 1.0086 | 0.7994 | 0.2092 | -0.067 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 864 | 2 | 703 | 10 | 0.1654 | 0.0697 | 0.0657 | -0.0016 | 0.0086 | 0.2328 | 1.0086 | 0.7983 | 0.2104 | -0.0674 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 865 | 2 | 704 | 10 | 0.1654 | 0.0697 | 0.0659 | -0.0016 | 0.0086 | 0.2329 | 1.0086 | 0.798 | 0.2107 | -0.0675 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 866 | 2 | 705 | 10 | 0.1654 | 0.0697 | 0.0661 | -0.0016 | 0.0086 | 0.2331 | 1.0086 | 0.7974 | 0.2112 | -0.0677 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 867 | 2 | 706 | 10 | 0.1654 | 0.0697 | 0.0664 | -0.0016 | 0.0086 | 0.2332 | 1.0086 | 0.7963 | 0.2124 | -0.0681 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 868 | 2 | 707 | 10 | 0.1654 | 0.0697 | 0.0666 | -0.0016 | 0.0086 | 0.2333 | 1.0086 | 0.7958 | 0.2128 | -0.0682 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 869 | 2 | 708 | 10 | 0.1654 | 0.0697 | 0.0665 | -0.0016 | 0.0086 | 0.2335 | 1.0086 | 0.7961 | 0.2125 | -0.0682 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 870 | 2 | 709 | 10 | 0.1654 | 0.0697 | 0.0664 | -0.0016 | 0.0086 | 0.2336 | 1.0086 | 0.7967 | 0.2119 | -0.068 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 871 | 2 | 710 | 10 | 0.1654 | 0.0697 | 0.0666 | -0.0016 | 0.0086 | 0.2337 | 1.0086 | 0.7962 | 0.2124 | -0.0682 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 872 | 2 | 711 | 10 | 0.1654 | 0.0697 | 0.0667 | -0.0016 | 0.0086 | 0.2339 | 1.0086 | 0.796 | 0.2126 | -0.0683 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 873 | 2 | 712 | 10 | 0.1654 | 0.0697 | 0.0664 | -0.0016 | 0.0086 | 0.234 | 1.0086 | 0.7971 | 0.2115 | -0.068 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 874 | 2 | 713 | 10 | 0.1654 | 0.0697 | 0.0661 | -0.0016 | 0.0086 | 0.2341 | 1.0086 | 0.7982 | 0.2103 | -0.0678 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 875 | 2 | 714 | 10 | 0.1654 | 0.0697 | 0.0658 | -0.0016 | 0.0086 | 0.2343 | 1.0086 | 0.7997 | 0.2088 | -0.0674 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 876 | 2 | 715 | 10 | 0.1654 | 0.0697 | 0.0656 | -0.0016 | 0.0085 | 0.2344 | 1.0085 | 0.8003 | 0.2082 | -0.0673 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 877 | 2 | 716 | 10 | 0.1654 | 0.0697 | 0.0657 | -0.0016 | 0.0085 | 0.2345 | 1.0085 | 0.8 | 0.2085 | -0.0674 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 878 | 2 | 717 | 10 | 0.1654 | 0.0697 | 0.0658 | -0.0016 | 0.0085 | 0.2347 | 1.0085 | 0.8001 | 0.2084 | -0.0674 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 879 | 2 | 718 | 10 | 0.1654 | 0.0697 | 0.0659 | -0.0016 | 0.0085 | 0.2348 | 1.0085 | 0.7998 | 0.2087 | -0.0675 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 880 | 2 | 719 | 10 | 0.1654 | 0.0697 | 0.066 | -0.0016 | 0.0085 | 0.2349 | 1.0085 | 0.7995 | 0.209 | -0.0676 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 881 | 2 | 720 | 10 | 0.1654 | 0.0697 | 0.0662 | -0.0016 | 0.0085 | 0.2351 | 1.0085 | 0.7988 | 0.2097 | -0.0679 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 882 | 2 | 721 | 10 | 0.1654 | 0.0697 | 0.0666 | -0.0016 | 0.0085 | 0.2352 | 1.0085 | 0.7977 | 0.2109 | -0.0682 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 883 | 2 | 722 | 10 | 0.1654 | 0.0697 | 0.0667 | -0.0016 | 0.0085 | 0.2353 | 1.0085 | 0.7974 | 0.2111 | -0.0683 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 884 | 2 | 723 | 10 | 0.1654 | 0.0697 | 0.0671 | -0.0016 | 0.0085 | 0.2355 | 1.0085 | 0.7961 | 0.2124 | -0.0687 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 885 | 2 | 724 | 10 | 0.1654 | 0.0697 | 0.0673 | -0.0016 | 0.0085 | 0.2356 | 1.0085 | 0.7954 | 0.2131 | -0.069 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 886 | 2 | 725 | 10 | 0.1654 | 0.0697 | 0.0676 | -0.0016 | 0.0085 | 0.2357 | 1.0085 | 0.7945 | 0.214 | -0.0693 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 887 | 2 | 726 | 10 | 0.1654 | 0.0697 | 0.0676 | -0.0016 | 0.0085 | 0.2359 | 1.0085 | 0.7947 | 0.2138 | -0.0692 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 888 | 2 | 727 | 10 | 0.1654 | 0.0697 | 0.0674 | -0.0016 | 0.0085 | 0.236 | 1.0085 | 0.7955 | 0.213 | -0.069 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 889 | 2 | 728 | 10 | 0.1654 | 0.0697 | 0.0672 | -0.0016 | 0.0085 | 0.2361 | 1.0085 | 0.7962 | 0.2122 | -0.0689 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 890 | 2 | 729 | 10 | 0.1654 | 0.0697 | 0.0676 | -0.0016 | 0.0085 | 0.2363 | 1.0085 | 0.7952 | 0.2133 | -0.0692 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 891 | 2 | 730 | 10 | 0.1654 | 0.0697 | 0.0676 | -0.0016 | 0.0084 | 0.2364 | 1.0084 | 0.7953 | 0.2131 | -0.0692 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 892 | 2 | 731 | 10 | 0.1654 | 0.0697 | 0.0676 | -0.0016 | 0.0084 | 0.2365 | 1.0084 | 0.7954 | 0.213 | -0.0692 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 893 | 2 | 732 | 10 | 0.1654 | 0.0697 | 0.068 | -0.0016 | 0.0084 | 0.2367 | 1.0084 | 0.7939 | 0.2145 | -0.0697 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 894 | 2 | 733 | 10 | 0.1654 | 0.0697 | 0.0682 | -0.0016 | 0.0084 | 0.2368 | 1.0084 | 0.7933 | 0.2151 | -0.0699 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 895 | 2 | 734 | 10 | 0.1654 | 0.0697 | 0.0683 | -0.0016 | 0.0084 | 0.2369 | 1.0084 | 0.7931 | 0.2153 | -0.07 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 896 | 2 | 735 | 10 | 0.1654 | 0.0697 | 0.0685 | -0.0016 | 0.0084 | 0.2371 | 1.0084 | 0.7928 | 0.2156 | -0.0701 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 897 | 2 | 736 | 10 | 0.1654 | 0.0697 | 0.0686 | -0.0016 | 0.0084 | 0.2372 | 1.0084 | 0.7924 | 0.216 | -0.0703 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 898 | 2 | 737 | 10 | 0.1654 | 0.0697 | 0.0689 | -0.0016 | 0.0084 | 0.2373 | 1.0084 | 0.7914 | 0.217 | -0.0706 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 899 | 2 | 738 | 10 | 0.1654 | 0.0697 | 0.0688 | -0.0016 | 0.0084 | 0.2375 | 1.0084 | 0.792 | 0.2164 | -0.0705 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 900 | 2 | 739 | 10 | 0.1654 | 0.0697 | 0.0688 | -0.0016 | 0.0084 | 0.2376 | 1.0084 | 0.7923 | 0.2161 | -0.0704 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 901 | 2 | 740 | 10 | 0.1654 | 0.0697 | 0.069 | -0.0016 | 0.0084 | 0.2377 | 1.0084 | 0.7915 | 0.2169 | -0.0707 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 902 | 2 | 741 | 10 | 0.1654 | 0.0697 | 0.0691 | -0.0016 | 0.0084 | 0.2379 | 1.0084 | 0.7915 | 0.2169 | -0.0707 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 903 | 2 | 742 | 10 | 0.1654 | 0.0697 | 0.0693 | -0.0016 | 0.0084 | 0.238 | 1.0084 | 0.7907 | 0.2177 | -0.071 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 904 | 2 | 743 | 10 | 0.1654 | 0.0697 | 0.0695 | -0.0016 | 0.0084 | 0.2381 | 1.0084 | 0.7902 | 0.2182 | -0.0712 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 905 | 2 | 744 | 10 | 0.1654 | 0.0697 | 0.0693 | -0.0016 | 0.0083 | 0.2383 | 1.0083 | 0.7909 | 0.2175 | -0.071 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 906 | 2 | 745 | 10 | 0.1654 | 0.0697 | 0.0696 | -0.0016 | 0.0083 | 0.2384 | 1.0083 | 0.7901 | 0.2183 | -0.0713 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 907 | 2 | 746 | 10 | 0.1654 | 0.0697 | 0.0699 | -0.0016 | 0.0083 | 0.2385 | 1.0083 | 0.7892 | 0.2191 | -0.0715 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 908 | 2 | 747 | 10 | 0.1654 | 0.0697 | 0.07 | -0.0016 | 0.0083 | 0.2387 | 1.0083 | 0.7889 | 0.2194 | -0.0717 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 909 | 2 | 748 | 10 | 0.1654 | 0.0697 | 0.0703 | -0.0016 | 0.0083 | 0.2388 | 1.0083 | 0.7882 | 0.2202 | -0.0719 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 910 | 2 | 749 | 10 | 0.1654 | 0.0697 | 0.0705 | -0.0016 | 0.0083 | 0.2389 | 1.0083 | 0.7875 | 0.2208 | -0.0722 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 911 | 2 | 750 | 10 | 0.1654 | 0.0697 | 0.0703 | -0.0016 | 0.0083 | 0.239 | 1.0083 | 0.7885 | 0.2199 | -0.0719 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 912 | 2 | 751 | 10 | 0.1654 | 0.0697 | 0.0703 | -0.0016 | 0.0083 | 0.2392 | 1.0083 | 0.7884 | 0.2199 | -0.072 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 913 | 2 | 752 | 10 | 0.1654 | 0.0697 | 0.0706 | -0.0016 | 0.0083 | 0.2393 | 1.0083 | 0.7876 | 0.2207 | -0.0722 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 914 | 2 | 753 | 10 | 0.1654 | 0.0697 | 0.0707 | -0.0016 | 0.0083 | 0.2394 | 1.0083 | 0.7873 | 0.2209 | -0.0724 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 915 | 2 | 754 | 10 | 0.1654 | 0.0697 | 0.071 | -0.0016 | 0.0083 | 0.2396 | 1.0083 | 0.7866 | 0.2217 | -0.0726 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 916 | 2 | 755 | 10 | 0.1654 | 0.0697 | 0.0711 | -0.0016 | 0.0083 | 0.2397 | 1.0083 | 0.7862 | 0.2221 | -0.0728 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 917 | 2 | 756 | 10 | 0.1654 | 0.0697 | 0.0715 | -0.0016 | 0.0083 | 0.2398 | 1.0083 | 0.7852 | 0.2231 | -0.0731 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 918 | 2 | 757 | 10 | 0.1654 | 0.0697 | 0.0716 | -0.0016 | 0.0083 | 0.24 | 1.0083 | 0.7849 | 0.2234 | -0.0732 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 919 | 2 | 758 | 10 | 0.1654 | 0.0697 | 0.0715 | -0.0016 | 0.0083 | 0.2401 | 1.0083 | 0.7854 | 0.2229 | -0.0731 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 920 | 2 | 759 | 10 | 0.1654 | 0.0697 | 0.0717 | -0.0016 | 0.0083 | 0.2402 | 1.0083 | 0.7849 | 0.2234 | -0.0733 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 921 | 2 | 760 | 10 | 0.1654 | 0.0697 | 0.0719 | -0.0016 | 0.0082 | 0.2404 | 1.0082 | 0.7841 | 0.2241 | -0.0736 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 922 | 2 | 761 | 10 | 0.1654 | 0.0697 | 0.0722 | -0.0016 | 0.0082 | 0.2405 | 1.0082 | 0.7835 | 0.2247 | -0.0738 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 923 | 2 | 762 | 10 | 0.1654 | 0.0697 | 0.0725 | -0.0016 | 0.0082 | 0.2406 | 1.0082 | 0.7824 | 0.2258 | -0.0742 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 924 | 2 | 763 | 10 | 0.1654 | 0.0697 | 0.0725 | -0.0016 | 0.0082 | 0.2407 | 1.0082 | 0.7827 | 0.2256 | -0.0741 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 925 | 2 | 764 | 10 | 0.1654 | 0.0697 | 0.0723 | -0.0016 | 0.0082 | 0.2409 | 1.0082 | 0.7834 | 0.2249 | -0.074 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 926 | 2 | 765 | 10 | 0.1654 | 0.0697 | 0.0726 | -0.0016 | 0.0082 | 0.241 | 1.0082 | 0.7826 | 0.2257 | -0.0742 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 927 | 2 | 766 | 10 | 0.1654 | 0.0697 | 0.073 | -0.0016 | 0.0082 | 0.2411 | 1.0082 | 0.7814 | 0.2268 | -0.0746 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 928 | 2 | 767 | 10 | 0.1654 | 0.0697 | 0.0731 | -0.0016 | 0.0082 | 0.2413 | 1.0082 | 0.781 | 0.2272 | -0.0748 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 929 | 2 | 768 | 10 | 0.1654 | 0.0697 | 0.0733 | -0.0016 | 0.0082 | 0.2414 | 1.0082 | 0.7805 | 0.2277 | -0.075 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 930 | 2 | 769 | 10 | 0.1654 | 0.0697 | 0.0736 | -0.0016 | 0.0082 | 0.2415 | 1.0082 | 0.7799 | 0.2283 | -0.0752 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 931 | 2 | 770 | 10 | 0.1654 | 0.0697 | 0.0738 | -0.0016 | 0.0082 | 0.2417 | 1.0082 | 0.7792 | 0.2289 | -0.0754 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 932 | 2 | 771 | 10 | 0.1654 | 0.0697 | 0.0737 | -0.0016 | 0.0082 | 0.2418 | 1.0082 | 0.7797 | 0.2285 | -0.0753 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 933 | 2 | 772 | 10 | 0.1654 | 0.0697 | 0.0737 | -0.0016 | 0.0082 | 0.2419 | 1.0082 | 0.7798 | 0.2284 | -0.0753 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 934 | 2 | 773 | 10 | 0.1654 | 0.0697 | 0.0739 | -0.0016 | 0.0082 | 0.242 | 1.0082 | 0.7792 | 0.229 | -0.0756 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 935 | 2 | 774 | 10 | 0.1654 | 0.0697 | 0.0742 | -0.0016 | 0.0082 | 0.2422 | 1.0082 | 0.7785 | 0.2296 | -0.0758 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 936 | 2 | 775 | 10 | 0.1654 | 0.0697 | 0.0746 | -0.0016 | 0.0082 | 0.2423 | 1.0082 | 0.7773 | 0.2309 | -0.0762 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 937 | 2 | 776 | 10 | 0.1654 | 0.0697 | 0.0745 | -0.0016 | 0.0081 | 0.2424 | 1.0081 | 0.7777 | 0.2304 | -0.0761 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 938 | 2 | 777 | 10 | 0.1654 | 0.0697 | 0.0746 | -0.0016 | 0.0081 | 0.2426 | 1.0081 | 0.7774 | 0.2308 | -0.0763 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 939 | 2 | 778 | 10 | 0.1654 | 0.0697 | 0.0745 | -0.0016 | 0.0081 | 0.2427 | 1.0081 | 0.7779 | 0.2302 | -0.0761 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 940 | 2 | 779 | 10 | 0.1654 | 0.0697 | 0.0748 | -0.0016 | 0.0081 | 0.2428 | 1.0081 | 0.7772 | 0.2309 | -0.0764 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 941 | 2 | 780 | 10 | 0.1654 | 0.0697 | 0.0746 | -0.0016 | 0.0081 | 0.243 | 1.0081 | 0.7778 | 0.2303 | -0.0763 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 942 | 2 | 781 | 10 | 0.1654 | 0.0697 | 0.0748 | -0.0016 | 0.0081 | 0.2431 | 1.0081 | 0.7773 | 0.2308 | -0.0765 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 943 | 2 | 782 | 10 | 0.1654 | 0.0697 | 0.0746 | -0.0016 | 0.0081 | 0.2432 | 1.0081 | 0.778 | 0.2301 | -0.0763 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 944 | 2 | 783 | 10 | 0.1654 | 0.0697 | 0.0747 | -0.0016 | 0.0081 | 0.2433 | 1.0081 | 0.7779 | 0.2302 | -0.0764 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 945 | 2 | 784 | 10 | 0.1654 | 0.0697 | 0.0746 | -0.0016 | 0.0081 | 0.2435 | 1.0081 | 0.7782 | 0.2299 | -0.0763 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 946 | 2 | 785 | 10 | 0.1654 | 0.0697 | 0.0745 | -0.0016 | 0.0081 | 0.2436 | 1.0081 | 0.7788 | 0.2293 | -0.0762 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 947 | 2 | 786 | 10 | 0.1654 | 0.0697 | 0.0744 | -0.0016 | 0.0081 | 0.2437 | 1.0081 | 0.7793 | 0.2288 | -0.076 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 948 | 2 | 787 | 10 | 0.1654 | 0.0697 | 0.0743 | -0.0016 | 0.0081 | 0.2439 | 1.0081 | 0.7798 | 0.2283 | -0.0759 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 949 | 2 | 788 | 10 | 0.1654 | 0.0697 | 0.0742 | -0.0016 | 0.0081 | 0.244 | 1.0081 | 0.7803 | 0.2278 | -0.0758 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 950 | 2 | 789 | 10 | 0.1654 | 0.0697 | 0.0741 | -0.0016 | 0.0081 | 0.2441 | 1.0081 | 0.7806 | 0.2275 | -0.0758 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 951 | 2 | 790 | 10 | 0.1654 | 0.0697 | 0.0739 | -0.0016 | 0.0081 | 0.2442 | 1.0081 | 0.7814 | 0.2266 | -0.0756 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 952 | 2 | 791 | 10 | 0.1654 | 0.0697 | 0.074 | -0.0016 | 0.0081 | 0.2444 | 1.0081 | 0.7814 | 0.2267 | -0.0756 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 953 | 2 | 792 | 10 | 0.1654 | 0.0697 | 0.0737 | -0.0016 | 0.008 | 0.2445 | 1.008 | 0.7825 | 0.2255 | -0.0753 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 954 | 2 | 793 | 10 | 0.1654 | 0.0697 | 0.0738 | -0.0016 | 0.008 | 0.2446 | 1.008 | 0.782 | 0.226 | -0.0755 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 955 | 2 | 794 | 10 | 0.1654 | 0.0697 | 0.0736 | -0.0016 | 0.008 | 0.2448 | 1.008 | 0.783 | 0.225 | -0.0752 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 956 | 2 | 795 | 10 | 0.1654 | 0.0697 | 0.0734 | -0.0016 | 0.008 | 0.2449 | 1.008 | 0.7837 | 0.2244 | -0.0751 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 957 | 2 | 796 | 10 | 0.1654 | 0.0697 | 0.0734 | -0.0016 | 0.008 | 0.245 | 1.008 | 0.7838 | 0.2243 | -0.0751 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 958 | 2 | 797 | 10 | 0.1654 | 0.0697 | 0.0735 | -0.0016 | 0.008 | 0.2451 | 1.008 | 0.7838 | 0.2243 | -0.0751 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 959 | 2 | 798 | 10 | 0.1654 | 0.0697 | 0.0734 | -0.0016 | 0.008 | 0.2453 | 1.008 | 0.7841 | 0.2239 | -0.0751 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 960 | 2 | 799 | 10 | 0.1654 | 0.0697 | 0.0733 | -0.0016 | 0.008 | 0.2454 | 1.008 | 0.7846 | 0.2234 | -0.0749 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 961 | 2 | 800 | 10 | 0.1654 | 0.0697 | 0.0732 | -0.0016 | 0.008 | 0.2455 | 1.008 | 0.7851 | 0.2229 | -0.0748 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 962 | 2 | 801 | 10 | 0.1654 | 0.0697 | 0.0731 | -0.0016 | 0.008 | 0.2456 | 1.008 | 0.7855 | 0.2225 | -0.0747 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 963 | 2 | 802 | 10 | 0.1654 | 0.0697 | 0.0732 | -0.0016 | 0.008 | 0.2458 | 1.008 | 0.7852 | 0.2228 | -0.0749 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 964 | 2 | 803 | 10 | 0.1654 | 0.0697 | 0.0733 | -0.0016 | 0.008 | 0.2459 | 1.008 | 0.785 | 0.223 | -0.075 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 965 | 2 | 804 | 10 | 0.1654 | 0.0697 | 0.0733 | -0.0016 | 0.008 | 0.246 | 1.008 | 0.7854 | 0.2226 | -0.0749 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 966 | 2 | 805 | 10 | 0.1654 | 0.0697 | 0.0732 | -0.0016 | 0.008 | 0.2462 | 1.008 | 0.7857 | 0.2223 | -0.0748 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 967 | 2 | 806 | 10 | 0.1654 | 0.0697 | 0.0734 | -0.0016 | 0.008 | 0.2463 | 1.008 | 0.7851 | 0.2229 | -0.0751 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 968 | 2 | 807 | 10 | 0.1654 | 0.0697 | 0.0735 | -0.0016 | 0.008 | 0.2464 | 1.008 | 0.785 | 0.2229 | -0.0751 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 969 | 2 | 808 | 10 | 0.1654 | 0.0697 | 0.0736 | -0.0016 | 0.008 | 0.2465 | 1.008 | 0.7846 | 0.2234 | -0.0753 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 970 | 2 | 809 | 10 | 0.1654 | 0.0697 | 0.0735 | -0.0016 | 0.008 | 0.2467 | 1.008 | 0.7853 | 0.2227 | -0.0751 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 971 | 2 | 810 | 10 | 0.1654 | 0.0697 | 0.0733 | -0.0016 | 0.0079 | 0.2468 | 1.0079 | 0.7859 | 0.2221 | -0.075 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 972 | 2 | 811 | 10 | 0.1654 | 0.0697 | 0.0734 | -0.0016 | 0.0079 | 0.2469 | 1.0079 | 0.7858 | 0.2221 | -0.075 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 973 | 2 | 812 | 10 | 0.1654 | 0.0697 | 0.0731 | -0.0016 | 0.0079 | 0.247 | 1.0079 | 0.7869 | 0.2211 | -0.0748 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 974 | 2 | 813 | 10 | 0.1654 | 0.0697 | 0.073 | -0.0016 | 0.0079 | 0.2472 | 1.0079 | 0.7875 | 0.2204 | -0.0746 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 975 | 2 | 814 | 10 | 0.1654 | 0.0697 | 0.0728 | -0.0016 | 0.0079 | 0.2473 | 1.0079 | 0.7883 | 0.2197 | -0.0744 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 976 | 2 | 815 | 10 | 0.1654 | 0.0697 | 0.0725 | -0.0016 | 0.0079 | 0.2474 | 1.0079 | 0.7894 | 0.2185 | -0.0741 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 977 | 2 | 816 | 10 | 0.1654 | 0.0697 | 0.0721 | -0.0016 | 0.0079 | 0.2476 | 1.0079 | 0.7907 | 0.2173 | -0.0738 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 978 | 2 | 817 | 10 | 0.1654 | 0.0697 | 0.0722 | -0.0016 | 0.0079 | 0.2477 | 1.0079 | 0.7905 | 0.2174 | -0.0739 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 979 | 2 | 818 | 10 | 0.1654 | 0.0697 | 0.072 | -0.0016 | 0.0079 | 0.2478 | 1.0079 | 0.7912 | 0.2167 | -0.0737 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 980 | 2 | 819 | 10 | 0.1654 | 0.0697 | 0.072 | -0.0016 | 0.0079 | 0.2479 | 1.0079 | 0.7916 | 0.2163 | -0.0736 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 981 | 2 | 820 | 10 | 0.1654 | 0.0697 | 0.072 | -0.0016 | 0.0079 | 0.2481 | 1.0079 | 0.7917 | 0.2162 | -0.0736 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 982 | 2 | 821 | 10 | 0.1654 | 0.0697 | 0.072 | -0.0016 | 0.0079 | 0.2482 | 1.0079 | 0.7918 | 0.216 | -0.0736 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 983 | 2 | 822 | 10 | 0.1654 | 0.0697 | 0.0717 | -0.0016 | 0.0079 | 0.2483 | 1.0079 | 0.7927 | 0.2151 | -0.0734 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 984 | 2 | 823 | 10 | 0.1654 | 0.0697 | 0.0718 | -0.0016 | 0.0079 | 0.2484 | 1.0079 | 0.7927 | 0.2152 | -0.0734 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 985 | 2 | 824 | 10 | 0.1654 | 0.0697 | 0.0719 | -0.0016 | 0.0079 | 0.2486 | 1.0079 | 0.7924 | 0.2155 | -0.0736 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 986 | 2 | 825 | 10 | 0.1654 | 0.0697 | 0.0719 | -0.0016 | 0.0079 | 0.2487 | 1.0079 | 0.7924 | 0.2154 | -0.0736 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 987 | 2 | 826 | 10 | 0.1654 | 0.0697 | 0.0721 | -0.0016 | 0.0079 | 0.2488 | 1.0079 | 0.7919 | 0.2159 | -0.0738 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 988 | 2 | 827 | 10 | 0.1654 | 0.0697 | 0.0724 | -0.0016 | 0.0078 | 0.2489 | 1.0078 | 0.7911 | 0.2168 | -0.0741 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 989 | 2 | 828 | 10 | 0.1654 | 0.0697 | 0.0725 | -0.0016 | 0.0078 | 0.2491 | 1.0078 | 0.791 | 0.2169 | -0.0741 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 990 | 2 | 829 | 10 | 0.1654 | 0.0697 | 0.0725 | -0.0016 | 0.0078 | 0.2492 | 1.0078 | 0.791 | 0.2168 | -0.0742 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 991 | 2 | 830 | 10 | 0.1654 | 0.0697 | 0.0723 | -0.0016 | 0.0078 | 0.2493 | 1.0078 | 0.7918 | 0.216 | -0.0739 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 992 | 2 | 831 | 10 | 0.1654 | 0.0697 | 0.0723 | -0.0016 | 0.0078 | 0.2494 | 1.0078 | 0.7918 | 0.216 | -0.074 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 993 | 2 | 832 | 10 | 0.1654 | 0.0697 | 0.0722 | -0.0016 | 0.0078 | 0.2496 | 1.0078 | 0.7926 | 0.2152 | -0.0738 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 994 | 2 | 833 | 10 | 0.1654 | 0.0697 | 0.072 | -0.0016 | 0.0078 | 0.2497 | 1.0078 | 0.7932 | 0.2146 | -0.0737 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 995 | 2 | 834 | 10 | 0.1654 | 0.0697 | 0.072 | -0.0016 | 0.0078 | 0.2498 | 1.0078 | 0.7935 | 0.2143 | -0.0736 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 996 | 2 | 835 | 10 | 0.1654 | 0.0697 | 0.0721 | -0.0016 | 0.0078 | 0.25 | 1.0078 | 0.7932 | 0.2146 | -0.0737 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 997 | 2 | 836 | 10 | 0.1654 | 0.0697 | 0.0719 | -0.0016 | 0.0078 | 0.2501 | 1.0078 | 0.7939 | 0.2139 | -0.0736 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 998 | 2 | 837 | 10 | 0.1654 | 0.0697 | 0.0718 | -0.0016 | 0.0078 | 0.2502 | 1.0078 | 0.7946 | 0.2132 | -0.0734 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 999 | 2 | 838 | 10 | 0.1654 | 0.0697 | 0.0716 | -0.0016 | 0.0078 | 0.2503 | 1.0078 | 0.7952 | 0.2126 | -0.0732 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1000 | 2 | 839 | 10 | 0.1654 | 0.0697 | 0.0714 | -0.0016 | 0.0078 | 0.2505 | 1.0078 | 0.7959 | 0.2118 | -0.0731 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1001 | 2 | 840 | 10 | 0.1654 | 0.0697 | 0.0711 | -0.0016 | 0.0078 | 0.2506 | 1.0078 | 0.7971 | 0.2106 | -0.0727 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1002 | 2 | 841 | 10 | 0.1654 | 0.0697 | 0.0709 | -0.0016 | 0.0078 | 0.2507 | 1.0078 | 0.7978 | 0.2099 | -0.0726 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1003 | 2 | 842 | 10 | 0.1654 | 0.0697 | 0.0712 | -0.0016 | 0.0078 | 0.2508 | 1.0078 | 0.797 | 0.2108 | -0.0729 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1004 | 2 | 843 | 10 | 0.1654 | 0.0697 | 0.0709 | -0.0016 | 0.0078 | 0.251 | 1.0078 | 0.7981 | 0.2097 | -0.0726 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1005 | 2 | 844 | 10 | 0.1654 | 0.0697 | 0.0706 | -0.0016 | 0.0078 | 0.2511 | 1.0078 | 0.7993 | 0.2085 | -0.0723 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1006 | 2 | 845 | 10 | 0.1654 | 0.0697 | 0.0705 | -0.0016 | 0.0078 | 0.2512 | 1.0078 | 0.8 | 0.2078 | -0.0721 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1007 | 2 | 846 | 10 | 0.1654 | 0.0697 | 0.0704 | -0.0016 | 0.0077 | 0.2513 | 1.0077 | 0.8004 | 0.2074 | -0.072 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1008 | 2 | 847 | 10 | 0.1654 | 0.0697 | 0.0702 | -0.0016 | 0.0077 | 0.2515 | 1.0077 | 0.8012 | 0.2065 | -0.0718 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1009 | 2 | 848 | 10 | 0.1654 | 0.0697 | 0.0701 | -0.0016 | 0.0077 | 0.2516 | 1.0077 | 0.8017 | 0.206 | -0.0717 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1010 | 2 | 849 | 10 | 0.1654 | 0.0697 | 0.0701 | -0.0016 | 0.0077 | 0.2517 | 1.0077 | 0.8017 | 0.206 | -0.0717 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1011 | 2 | 850 | 10 | 0.1654 | 0.0697 | 0.0699 | -0.0016 | 0.0077 | 0.2518 | 1.0077 | 0.8024 | 0.2053 | -0.0716 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1012 | 2 | 851 | 10 | 0.1654 | 0.0697 | 0.0697 | -0.0016 | 0.0077 | 0.252 | 1.0077 | 0.8032 | 0.2045 | -0.0714 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1013 | 2 | 852 | 10 | 0.1654 | 0.0697 | 0.0695 | -0.0016 | 0.0077 | 0.2521 | 1.0077 | 0.8041 | 0.2036 | -0.0712 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1014 | 2 | 853 | 10 | 0.1654 | 0.0697 | 0.0694 | -0.0016 | 0.0077 | 0.2522 | 1.0077 | 0.8048 | 0.2029 | -0.071 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1015 | 2 | 854 | 10 | 0.1654 | 0.0697 | 0.0691 | -0.0016 | 0.0077 | 0.2523 | 1.0077 | 0.8059 | 0.2018 | -0.0707 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1016 | 2 | 855 | 10 | 0.1654 | 0.0697 | 0.069 | -0.0016 | 0.0077 | 0.2524 | 1.0077 | 0.8062 | 0.2015 | -0.0707 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1017 | 2 | 856 | 10 | 0.1654 | 0.0697 | 0.0689 | -0.0016 | 0.0077 | 0.2526 | 1.0077 | 0.8069 | 0.2008 | -0.0705 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1018 | 2 | 857 | 10 | 0.1654 | 0.0697 | 0.069 | -0.0016 | 0.0077 | 0.2527 | 1.0077 | 0.8066 | 0.2011 | -0.0706 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1019 | 2 | 858 | 10 | 0.1654 | 0.0697 | 0.0689 | -0.0016 | 0.0077 | 0.2528 | 1.0077 | 0.8071 | 0.2006 | -0.0705 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1020 | 2 | 859 | 10 | 0.1654 | 0.0697 | 0.0691 | -0.0016 | 0.0077 | 0.2529 | 1.0077 | 0.8065 | 0.2012 | -0.0707 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1021 | 2 | 860 | 10 | 0.1654 | 0.0697 | 0.0688 | -0.0016 | 0.0077 | 0.2531 | 1.0077 | 0.8077 | 0.2 | -0.0704 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1022 | 2 | 861 | 10 | 0.1654 | 0.0697 | 0.0689 | -0.0016 | 0.0077 | 0.2532 | 1.0077 | 0.8075 | 0.2001 | -0.0705 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1023 | 2 | 862 | 10 | 0.1654 | 0.0697 | 0.069 | -0.0016 | 0.0077 | 0.2533 | 1.0077 | 0.8071 | 0.2006 | -0.0707 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1024 | 2 | 863 | 10 | 0.1654 | 0.0697 | 0.0687 | -0.0016 | 0.0077 | 0.2534 | 1.0077 | 0.8083 | 0.1994 | -0.0704 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1025 | 2 | 864 | 10 | 0.1654 | 0.0697 | 0.0684 | -0.0016 | 0.0077 | 0.2536 | 1.0077 | 0.8095 | 0.1982 | -0.0701 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1026 | 2 | 865 | 10 | 0.1654 | 0.0697 | 0.0686 | -0.0016 | 0.0076 | 0.2537 | 1.0076 | 0.809 | 0.1986 | -0.0702 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1027 | 2 | 866 | 10 | 0.1654 | 0.0697 | 0.0688 | -0.0016 | 0.0076 | 0.2538 | 1.0076 | 0.8084 | 0.1992 | -0.0704 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1028 | 2 | 867 | 10 | 0.1654 | 0.0697 | 0.0686 | -0.0016 | 0.0076 | 0.2539 | 1.0076 | 0.8091 | 0.1986 | -0.0703 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1029 | 2 | 868 | 10 | 0.1654 | 0.0697 | 0.0685 | -0.0016 | 0.0076 | 0.2541 | 1.0076 | 0.8097 | 0.1979 | -0.0701 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1030 | 2 | 869 | 10 | 0.1654 | 0.0697 | 0.0683 | -0.0016 | 0.0076 | 0.2542 | 1.0076 | 0.8104 | 0.1972 | -0.07 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1031 | 2 | 870 | 10 | 0.1654 | 0.0697 | 0.0682 | -0.0016 | 0.0076 | 0.2543 | 1.0076 | 0.8111 | 0.1965 | -0.0698 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1032 | 2 | 871 | 10 | 0.1654 | 0.0697 | 0.0683 | -0.0016 | 0.0076 | 0.2544 | 1.0076 | 0.8109 | 0.1967 | -0.0699 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1033 | 2 | 872 | 10 | 0.1654 | 0.0697 | 0.0682 | -0.0016 | 0.0076 | 0.2546 | 1.0076 | 0.8112 | 0.1964 | -0.0699 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1034 | 2 | 873 | 10 | 0.1654 | 0.0697 | 0.0682 | -0.0016 | 0.0076 | 0.2547 | 1.0076 | 0.8112 | 0.1964 | -0.0699 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1035 | 2 | 874 | 10 | 0.1654 | 0.0697 | 0.0681 | -0.0016 | 0.0076 | 0.2548 | 1.0076 | 0.8117 | 0.1959 | -0.0698 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1036 | 2 | 875 | 10 | 0.1654 | 0.0697 | 0.0682 | -0.0016 | 0.0076 | 0.2549 | 1.0076 | 0.8115 | 0.1961 | -0.0699 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1037 | 2 | 876 | 10 | 0.1654 | 0.0697 | 0.068 | -0.0016 | 0.0076 | 0.255 | 1.0076 | 0.8123 | 0.1953 | -0.0697 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1038 | 2 | 877 | 10 | 0.1654 | 0.0697 | 0.068 | -0.0016 | 0.0076 | 0.2552 | 1.0076 | 0.8125 | 0.1951 | -0.0696 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1039 | 2 | 878 | 10 | 0.1654 | 0.0697 | 0.0674 | -0.0016 | 0.0076 | 0.2553 | 1.0076 | 0.8149 | 0.1927 | -0.069 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1040 | 2 | 879 | 10 | 0.1654 | 0.0697 | 0.0672 | -0.0016 | 0.0076 | 0.2554 | 1.0076 | 0.8155 | 0.1921 | -0.0689 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1041 | 2 | 880 | 10 | 0.1654 | 0.0697 | 0.0671 | -0.0016 | 0.0076 | 0.2555 | 1.0076 | 0.8161 | 0.1914 | -0.0687 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1042 | 2 | 881 | 10 | 0.1654 | 0.0697 | 0.0672 | -0.0016 | 0.0076 | 0.2557 | 1.0076 | 0.8159 | 0.1917 | -0.0688 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1043 | 2 | 882 | 10 | 0.1654 | 0.0697 | 0.0673 | -0.0016 | 0.0076 | 0.2558 | 1.0076 | 0.8157 | 0.1919 | -0.0689 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1044 | 2 | 883 | 10 | 0.1654 | 0.0697 | 0.0674 | -0.0016 | 0.0076 | 0.2559 | 1.0076 | 0.8156 | 0.192 | -0.069 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1045 | 2 | 884 | 10 | 0.1654 | 0.0697 | 0.0676 | -0.0016 | 0.0076 | 0.256 | 1.0076 | 0.8149 | 0.1926 | -0.0692 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1046 | 2 | 885 | 10 | 0.1654 | 0.0697 | 0.0673 | -0.0016 | 0.0075 | 0.2561 | 1.0075 | 0.8159 | 0.1917 | -0.069 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1047 | 2 | 886 | 10 | 0.1654 | 0.0697 | 0.0672 | -0.0016 | 0.0075 | 0.2563 | 1.0075 | 0.8166 | 0.1909 | -0.0688 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1048 | 2 | 887 | 10 | 0.1654 | 0.0697 | 0.0674 | -0.0016 | 0.0075 | 0.2564 | 1.0075 | 0.8159 | 0.1917 | -0.069 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1049 | 2 | 888 | 10 | 0.1654 | 0.0697 | 0.0676 | -0.0016 | 0.0075 | 0.2565 | 1.0075 | 0.8154 | 0.1921 | -0.0692 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1050 | 2 | 889 | 10 | 0.1654 | 0.0697 | 0.0673 | -0.0016 | 0.0075 | 0.2566 | 1.0075 | 0.8164 | 0.1912 | -0.069 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1051 | 2 | 890 | 10 | 0.1654 | 0.0697 | 0.0671 | -0.0016 | 0.0075 | 0.2568 | 1.0075 | 0.8172 | 0.1904 | -0.0688 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1052 | 2 | 891 | 10 | 0.1654 | 0.0697 | 0.067 | -0.0016 | 0.0075 | 0.2569 | 1.0075 | 0.8179 | 0.1896 | -0.0686 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1053 | 2 | 892 | 10 | 0.1654 | 0.0697 | 0.0668 | -0.0016 | 0.0075 | 0.257 | 1.0075 | 0.8186 | 0.1889 | -0.0684 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1054 | 2 | 892 | 10 | 0.1654 | 0.0697 | 0.0668 | -0.0016 | 0.0075 | 0.2571 | 1.0075 | 0.8188 | 0.1888 | -0.0684 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1055 | 2 | 893 | 10 | 0.1654 | 0.0697 | 0.0666 | -0.0016 | 0.0075 | 0.2572 | 1.0075 | 0.8195 | 0.188 | -0.0683 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1056 | 2 | 894 | 10 | 0.1654 | 0.0697 | 0.0664 | -0.0016 | 0.0075 | 0.2574 | 1.0075 | 0.8205 | 0.187 | -0.068 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1057 | 2 | 895 | 10 | 0.1654 | 0.0697 | 0.0663 | -0.0016 | 0.0075 | 0.2575 | 1.0075 | 0.8211 | 0.1864 | -0.0679 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1058 | 2 | 896 | 10 | 0.1654 | 0.0697 | 0.0664 | -0.0016 | 0.0075 | 0.2576 | 1.0075 | 0.8208 | 0.1867 | -0.068 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1059 | 2 | 897 | 10 | 0.1654 | 0.0697 | 0.0661 | -0.0016 | 0.0075 | 0.2577 | 1.0075 | 0.8218 | 0.1857 | -0.0678 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1060 | 2 | 898 | 10 | 0.1654 | 0.0697 | 0.0661 | -0.0016 | 0.0075 | 0.2579 | 1.0075 | 0.822 | 0.1855 | -0.0677 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1061 | 2 | 899 | 10 | 0.1654 | 0.0697 | 0.0662 | -0.0016 | 0.0075 | 0.258 | 1.0075 | 0.8217 | 0.1858 | -0.0679 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1062 | 2 | 900 | 10 | 0.1654 | 0.0697 | 0.0661 | -0.0016 | 0.0075 | 0.2581 | 1.0075 | 0.8222 | 0.1853 | -0.0678 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1063 | 2 | 901 | 10 | 0.1654 | 0.0697 | 0.0659 | -0.0016 | 0.0075 | 0.2582 | 1.0075 | 0.8229 | 0.1845 | -0.0676 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1064 | 2 | 902 | 10 | 0.1654 | 0.0697 | 0.0658 | -0.0016 | 0.0075 | 0.2583 | 1.0075 | 0.8236 | 0.1839 | -0.0675 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1065 | 2 | 903 | 10 | 0.1654 | 0.0697 | 0.0661 | -0.0016 | 0.0075 | 0.2585 | 1.0075 | 0.8226 | 0.1848 | -0.0677 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1066 | 2 | 904 | 10 | 0.1654 | 0.0697 | 0.0662 | -0.0016 | 0.0075 | 0.2586 | 1.0075 | 0.8224 | 0.1851 | -0.0678 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1067 | 2 | 905 | 10 | 0.1654 | 0.0697 | 0.0664 | -0.0016 | 0.0075 | 0.2587 | 1.0075 | 0.8218 | 0.1856 | -0.068 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1068 | 2 | 906 | 10 | 0.1654 | 0.0697 | 0.0664 | -0.0016 | 0.0075 | 0.2588 | 1.0075 | 0.8218 | 0.1857 | -0.0681 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1069 | 2 | 907 | 10 | 0.1654 | 0.0697 | 0.0664 | -0.0016 | 0.0074 | 0.2589 | 1.0074 | 0.8219 | 0.1856 | -0.0681 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1070 | 2 | 908 | 10 | 0.1654 | 0.0697 | 0.0666 | -0.0016 | 0.0074 | 0.2591 | 1.0074 | 0.8216 | 0.1859 | -0.0682 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1071 | 2 | 909 | 10 | 0.1654 | 0.0697 | 0.0667 | -0.0016 | 0.0074 | 0.2592 | 1.0074 | 0.8213 | 0.1861 | -0.0683 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1072 | 2 | 910 | 10 | 0.1654 | 0.0697 | 0.0667 | -0.0016 | 0.0074 | 0.2593 | 1.0074 | 0.8215 | 0.1859 | -0.0683 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1073 | 2 | 911 | 10 | 0.1654 | 0.0697 | 0.0667 | -0.0016 | 0.0074 | 0.2594 | 1.0074 | 0.8213 | 0.1861 | -0.0684 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1074 | 2 | 912 | 10 | 0.1654 | 0.0697 | 0.0667 | -0.0016 | 0.0074 | 0.2596 | 1.0074 | 0.8217 | 0.1857 | -0.0683 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1075 | 2 | 913 | 10 | 0.1654 | 0.0697 | 0.0665 | -0.0016 | 0.0074 | 0.2597 | 1.0074 | 0.8223 | 0.1852 | -0.0682 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1076 | 2 | 914 | 10 | 0.1654 | 0.0697 | 0.0665 | -0.0016 | 0.0074 | 0.2598 | 1.0074 | 0.8224 | 0.185 | -0.0682 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1077 | 2 | 915 | 10 | 0.1654 | 0.0697 | 0.0665 | -0.0016 | 0.0074 | 0.2599 | 1.0074 | 0.8228 | 0.1847 | -0.0681 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1078 | 2 | 916 | 10 | 0.1654 | 0.0697 | 0.0663 | -0.0016 | 0.0074 | 0.26 | 1.0074 | 0.8234 | 0.184 | -0.068 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1079 | 2 | 917 | 10 | 0.1654 | 0.0697 | 0.0662 | -0.0016 | 0.0074 | 0.2602 | 1.0074 | 0.824 | 0.1834 | -0.0678 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1080 | 2 | 918 | 10 | 0.1654 | 0.0697 | 0.0662 | -0.0016 | 0.0074 | 0.2603 | 1.0074 | 0.824 | 0.1834 | -0.0679 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1081 | 2 | 919 | 10 | 0.1654 | 0.0697 | 0.0661 | -0.0016 | 0.0074 | 0.2604 | 1.0074 | 0.8244 | 0.183 | -0.0678 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1082 | 2 | 920 | 10 | 0.1654 | 0.0697 | 0.0661 | -0.0016 | 0.0074 | 0.2605 | 1.0074 | 0.8248 | 0.1826 | -0.0677 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1083 | 2 | 921 | 10 | 0.1654 | 0.0697 | 0.0661 | -0.0016 | 0.0074 | 0.2606 | 1.0074 | 0.8246 | 0.1827 | -0.0678 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1084 | 2 | 922 | 10 | 0.1654 | 0.0697 | 0.0662 | -0.0016 | 0.0074 | 0.2608 | 1.0074 | 0.8245 | 0.1828 | -0.0679 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1085 | 2 | 923 | 10 | 0.1654 | 0.0697 | 0.0662 | -0.0016 | 0.0074 | 0.2609 | 1.0074 | 0.8247 | 0.1826 | -0.0678 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1086 | 2 | 924 | 10 | 0.1654 | 0.0697 | 0.0662 | -0.0016 | 0.0074 | 0.261 | 1.0074 | 0.8249 | 0.1825 | -0.0678 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1087 | 2 | 925 | 10 | 0.1654 | 0.0697 | 0.0661 | -0.0016 | 0.0074 | 0.2611 | 1.0074 | 0.8253 | 0.1821 | -0.0677 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1088 | 2 | 925 | 10 | 0.1654 | 0.0697 | 0.0661 | -0.0016 | 0.0074 | 0.2612 | 1.0074 | 0.8254 | 0.182 | -0.0677 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1089 | 2 | 926 | 10 | 0.1654 | 0.0697 | 0.0659 | -0.0016 | 0.0074 | 0.2614 | 1.0074 | 0.8261 | 0.1812 | -0.0676 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1090 | 2 | 927 | 10 | 0.1654 | 0.0697 | 0.066 | -0.0016 | 0.0074 | 0.2615 | 1.0074 | 0.8258 | 0.1815 | -0.0677 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1091 | 2 | 928 | 10 | 0.1654 | 0.0697 | 0.0663 | -0.0016 | 0.0074 | 0.2616 | 1.0074 | 0.8252 | 0.1821 | -0.0679 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1092 | 2 | 929 | 10 | 0.1654 | 0.0697 | 0.0663 | -0.0016 | 0.0073 | 0.2617 | 1.0073 | 0.8251 | 0.1822 | -0.068 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1093 | 2 | 930 | 10 | 0.1654 | 0.0697 | 0.0665 | -0.0016 | 0.0073 | 0.2618 | 1.0073 | 0.8247 | 0.1826 | -0.0681 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1094 | 2 | 931 | 10 | 0.1654 | 0.0697 | 0.0665 | -0.0016 | 0.0073 | 0.262 | 1.0073 | 0.8246 | 0.1828 | -0.0682 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1095 | 2 | 932 | 10 | 0.1654 | 0.0697 | 0.0665 | -0.0016 | 0.0073 | 0.2621 | 1.0073 | 0.8249 | 0.1825 | -0.0681 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1096 | 2 | 933 | 10 | 0.1654 | 0.0697 | 0.0665 | -0.0016 | 0.0073 | 0.2622 | 1.0073 | 0.8248 | 0.1826 | -0.0682 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1097 | 2 | 934 | 10 | 0.1654 | 0.0697 | 0.0666 | -0.0016 | 0.0073 | 0.2623 | 1.0073 | 0.8245 | 0.1828 | -0.0683 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1098 | 2 | 935 | 10 | 0.1654 | 0.0697 | 0.0667 | -0.0016 | 0.0073 | 0.2624 | 1.0073 | 0.8244 | 0.1829 | -0.0684 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1099 | 2 | 936 | 10 | 0.1654 | 0.0697 | 0.0668 | -0.0016 | 0.0073 | 0.2626 | 1.0073 | 0.8242 | 0.1831 | -0.0685 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1100 | 2 | 937 | 10 | 0.1654 | 0.0697 | 0.067 | -0.0016 | 0.0073 | 0.2627 | 1.0073 | 0.8236 | 0.1837 | -0.0686 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1101 | 2 | 938 | 10 | 0.1654 | 0.0697 | 0.0669 | -0.0016 | 0.0073 | 0.2628 | 1.0073 | 0.8239 | 0.1834 | -0.0686 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1102 | 2 | 939 | 10 | 0.1654 | 0.0697 | 0.0671 | -0.0016 | 0.0073 | 0.2629 | 1.0073 | 0.8234 | 0.1839 | -0.0688 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1103 | 2 | 940 | 10 | 0.1654 | 0.0697 | 0.0672 | -0.0016 | 0.0073 | 0.263 | 1.0073 | 0.8231 | 0.1842 | -0.0689 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1104 | 2 | 941 | 10 | 0.1654 | 0.0697 | 0.067 | -0.0016 | 0.0073 | 0.2632 | 1.0073 | 0.8239 | 0.1834 | -0.0687 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1105 | 2 | 942 | 10 | 0.1654 | 0.0697 | 0.0671 | -0.0016 | 0.0073 | 0.2633 | 1.0073 | 0.8239 | 0.1834 | -0.0687 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1106 | 2 | 943 | 10 | 0.1654 | 0.0697 | 0.0669 | -0.0016 | 0.0073 | 0.2634 | 1.0073 | 0.8247 | 0.1826 | -0.0685 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1107 | 2 | 944 | 10 | 0.1654 | 0.0697 | 0.0668 | -0.0016 | 0.0073 | 0.2635 | 1.0073 | 0.8251 | 0.1822 | -0.0685 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1108 | 2 | 945 | 10 | 0.1654 | 0.0697 | 0.0667 | -0.0016 | 0.0073 | 0.2636 | 1.0073 | 0.8256 | 0.1816 | -0.0683 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1109 | 2 | 946 | 10 | 0.1654 | 0.0697 | 0.0668 | -0.0016 | 0.0073 | 0.2637 | 1.0073 | 0.8255 | 0.1818 | -0.0684 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1110 | 2 | 947 | 10 | 0.1654 | 0.0697 | 0.0674 | -0.0016 | 0.0073 | 0.2639 | 1.0073 | 0.8235 | 0.1838 | -0.069 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1111 | 2 | 948 | 10 | 0.1654 | 0.0697 | 0.0673 | -0.0016 | 0.0073 | 0.264 | 1.0073 | 0.8237 | 0.1836 | -0.069 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1112 | 2 | 949 | 10 | 0.1654 | 0.0697 | 0.0674 | -0.0016 | 0.0073 | 0.2641 | 1.0073 | 0.8237 | 0.1836 | -0.069 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1113 | 2 | 950 | 10 | 0.1654 | 0.0697 | 0.0674 | -0.0016 | 0.0073 | 0.2642 | 1.0073 | 0.8236 | 0.1836 | -0.0691 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1114 | 2 | 951 | 10 | 0.1654 | 0.0697 | 0.0676 | -0.0016 | 0.0072 | 0.2643 | 1.0072 | 0.8232 | 0.184 | -0.0692 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1115 | 2 | 952 | 10 | 0.1654 | 0.0697 | 0.0677 | -0.0016 | 0.0072 | 0.2645 | 1.0072 | 0.823 | 0.1842 | -0.0693 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1116 | 2 | 953 | 10 | 0.1654 | 0.0697 | 0.0679 | -0.0016 | 0.0072 | 0.2646 | 1.0072 | 0.8222 | 0.185 | -0.0696 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1117 | 2 | 954 | 10 | 0.1654 | 0.0697 | 0.0681 | -0.0016 | 0.0072 | 0.2647 | 1.0072 | 0.8217 | 0.1855 | -0.0697 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1118 | 2 | 955 | 10 | 0.1654 | 0.0697 | 0.0684 | -0.0016 | 0.0072 | 0.2648 | 1.0072 | 0.8209 | 0.1864 | -0.07 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1119 | 2 | 956 | 10 | 0.1654 | 0.0697 | 0.0686 | -0.0016 | 0.0072 | 0.2649 | 1.0072 | 0.8203 | 0.1869 | -0.0702 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1120 | 2 | 957 | 10 | 0.1654 | 0.0697 | 0.0686 | -0.0016 | 0.0072 | 0.2651 | 1.0072 | 0.8203 | 0.1869 | -0.0702 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1121 | 2 | 958 | 10 | 0.1654 | 0.0697 | 0.0688 | -0.0016 | 0.0072 | 0.2652 | 1.0072 | 0.8197 | 0.1875 | -0.0704 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1122 | 2 | 959 | 10 | 0.1654 | 0.0697 | 0.069 | -0.0016 | 0.0072 | 0.2653 | 1.0072 | 0.819 | 0.1882 | -0.0707 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1123 | 2 | 960 | 10 | 0.1654 | 0.0697 | 0.0692 | -0.0016 | 0.0072 | 0.2654 | 1.0072 | 0.8184 | 0.1888 | -0.0709 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1124 | 2 | 961 | 10 | 0.1654 | 0.0697 | 0.0694 | -0.0016 | 0.0072 | 0.2655 | 1.0072 | 0.818 | 0.1892 | -0.071 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1125 | 2 | 962 | 10 | 0.1654 | 0.0697 | 0.0694 | -0.0016 | 0.0072 | 0.2656 | 1.0072 | 0.8179 | 0.1893 | -0.0711 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1126 | 2 | 963 | 10 | 0.1654 | 0.0697 | 0.0696 | -0.0016 | 0.0072 | 0.2658 | 1.0072 | 0.8177 | 0.1895 | -0.0712 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1127 | 2 | 964 | 10 | 0.1654 | 0.0697 | 0.0697 | -0.0016 | 0.0072 | 0.2659 | 1.0072 | 0.8172 | 0.19 | -0.0714 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1128 | 2 | 965 | 10 | 0.1654 | 0.0697 | 0.0699 | -0.0016 | 0.0072 | 0.266 | 1.0072 | 0.8167 | 0.1904 | -0.0715 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1129 | 2 | 966 | 10 | 0.1654 | 0.0697 | 0.07 | -0.0016 | 0.0072 | 0.2661 | 1.0072 | 0.8164 | 0.1908 | -0.0717 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1130 | 2 | 967 | 10 | 0.1654 | 0.0697 | 0.0702 | -0.0016 | 0.0072 | 0.2662 | 1.0072 | 0.8159 | 0.1913 | -0.0719 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1131 | 2 | 968 | 10 | 0.1654 | 0.0697 | 0.07 | -0.0016 | 0.0072 | 0.2664 | 1.0072 | 0.8166 | 0.1906 | -0.0717 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1132 | 2 | 969 | 10 | 0.1654 | 0.0697 | 0.0703 | -0.0016 | 0.0072 | 0.2665 | 1.0072 | 0.8159 | 0.1913 | -0.0719 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1133 | 2 | 970 | 10 | 0.1654 | 0.0697 | 0.0704 | -0.0016 | 0.0072 | 0.2666 | 1.0072 | 0.8155 | 0.1917 | -0.0721 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1134 | 2 | 971 | 10 | 0.1654 | 0.0697 | 0.0703 | -0.0016 | 0.0072 | 0.2667 | 1.0072 | 0.8161 | 0.191 | -0.0719 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1135 | 2 | 972 | 10 | 0.1654 | 0.0697 | 0.0705 | -0.0016 | 0.0072 | 0.2668 | 1.0072 | 0.8154 | 0.1918 | -0.0722 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1136 | 2 | 973 | 10 | 0.1654 | 0.0697 | 0.0707 | -0.0016 | 0.0072 | 0.2669 | 1.0072 | 0.8149 | 0.1923 | -0.0724 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1137 | 2 | 974 | 10 | 0.1654 | 0.0697 | 0.0709 | -0.0016 | 0.0072 | 0.2671 | 1.0072 | 0.8144 | 0.1927 | -0.0725 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1138 | 2 | 975 | 10 | 0.1654 | 0.0697 | 0.0708 | -0.0016 | 0.0071 | 0.2672 | 1.0071 | 0.8149 | 0.1923 | -0.0724 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1139 | 2 | 976 | 10 | 0.1654 | 0.0697 | 0.0711 | -0.0016 | 0.0071 | 0.2673 | 1.0071 | 0.814 | 0.1932 | -0.0727 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1140 | 2 | 977 | 10 | 0.1654 | 0.0697 | 0.0713 | -0.0016 | 0.0071 | 0.2674 | 1.0071 | 0.8133 | 0.1938 | -0.0729 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1141 | 2 | 978 | 10 | 0.1654 | 0.0697 | 0.0714 | -0.0016 | 0.0071 | 0.2675 | 1.0071 | 0.8131 | 0.194 | -0.073 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1142 | 2 | 979 | 10 | 0.1654 | 0.0697 | 0.0717 | -0.0016 | 0.0071 | 0.2676 | 1.0071 | 0.8124 | 0.1948 | -0.0733 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1143 | 2 | 980 | 10 | 0.1654 | 0.0697 | 0.0719 | -0.0016 | 0.0071 | 0.2678 | 1.0071 | 0.8118 | 0.1954 | -0.0735 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1144 | 2 | 981 | 10 | 0.1654 | 0.0697 | 0.072 | -0.0016 | 0.0071 | 0.2679 | 1.0071 | 0.8115 | 0.1957 | -0.0736 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1145 | 2 | 982 | 10 | 0.1654 | 0.0697 | 0.0722 | -0.0016 | 0.0071 | 0.268 | 1.0071 | 0.8108 | 0.1963 | -0.0739 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1146 | 2 | 983 | 10 | 0.1654 | 0.0697 | 0.0721 | -0.0016 | 0.0071 | 0.2681 | 1.0071 | 0.8113 | 0.1958 | -0.0738 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1147 | 2 | 984 | 10 | 0.1654 | 0.0697 | 0.072 | -0.0016 | 0.0071 | 0.2682 | 1.0071 | 0.8119 | 0.1952 | -0.0736 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1148 | 2 | 985 | 10 | 0.1654 | 0.0697 | 0.0721 | -0.0016 | 0.0071 | 0.2683 | 1.0071 | 0.8114 | 0.1957 | -0.0738 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1149 | 2 | 986 | 10 | 0.1654 | 0.0697 | 0.0721 | -0.0016 | 0.0071 | 0.2685 | 1.0071 | 0.8116 | 0.1955 | -0.0738 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1150 | 2 | 987 | 10 | 0.1654 | 0.0697 | 0.0724 | -0.0016 | 0.0071 | 0.2686 | 1.0071 | 0.8107 | 0.1964 | -0.0741 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1151 | 2 | 988 | 10 | 0.1654 | 0.0697 | 0.0726 | -0.0016 | 0.0071 | 0.2687 | 1.0071 | 0.8102 | 0.1969 | -0.0742 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1152 | 2 | 989 | 10 | 0.1654 | 0.0697 | 0.0727 | -0.0016 | 0.0071 | 0.2688 | 1.0071 | 0.81 | 0.1971 | -0.0743 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1153 | 2 | 990 | 10 | 0.1654 | 0.0697 | 0.0728 | -0.0016 | 0.0071 | 0.2689 | 1.0071 | 0.8099 | 0.1972 | -0.0744 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1154 | 2 | 991 | 10 | 0.1654 | 0.0697 | 0.0727 | -0.0016 | 0.0071 | 0.269 | 1.0071 | 0.8101 | 0.197 | -0.0744 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1155 | 2 | 992 | 10 | 0.1654 | 0.0697 | 0.0726 | -0.0016 | 0.0071 | 0.2692 | 1.0071 | 0.8108 | 0.1963 | -0.0742 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1156 | 2 | 993 | 10 | 0.1654 | 0.0697 | 0.0727 | -0.0016 | 0.0071 | 0.2693 | 1.0071 | 0.8104 | 0.1967 | -0.0744 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1157 | 2 | 994 | 10 | 0.1654 | 0.0697 | 0.0729 | -0.0016 | 0.0071 | 0.2694 | 1.0071 | 0.8099 | 0.1972 | -0.0746 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1158 | 2 | 995 | 10 | 0.1654 | 0.0697 | 0.0732 | -0.0016 | 0.0071 | 0.2695 | 1.0071 | 0.8089 | 0.1982 | -0.0749 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1159 | 2 | 996 | 10 | 0.1654 | 0.0697 | 0.0731 | -0.0016 | 0.0071 | 0.2696 | 1.0071 | 0.8094 | 0.1977 | -0.0748 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1160 | 2 | 997 | 10 | 0.1654 | 0.0697 | 0.0732 | -0.0016 | 0.0071 | 0.2697 | 1.0071 | 0.8093 | 0.1978 | -0.0748 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1161 | 2 | 998 | 10 | 0.1654 | 0.0697 | 0.0735 | -0.0016 | 0.0071 | 0.2699 | 1.0071 | 0.8085 | 0.1985 | -0.0751 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1162 | 2 | 999 | 10 | 0.1654 | 0.0697 | 0.0735 | -0.0016 | 0.007 | 0.27 | 1.007 | 0.8085 | 0.1986 | -0.0751 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1163 | 2 | 1000 | 10 | 0.1654 | 0.0697 | 0.0736 | -0.0016 | 0.007 | 0.2701 | 1.007 | 0.8082 | 0.1988 | -0.0753 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1164 | 2 | 1001 | 10 | 0.1654 | 0.0697 | 0.0735 | -0.0016 | 0.007 | 0.2702 | 1.007 | 0.8086 | 0.1984 | -0.0752 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1165 | 2 | 1002 | 10 | 0.1654 | 0.0697 | 0.0737 | -0.0016 | 0.007 | 0.2703 | 1.007 | 0.8082 | 0.1989 | -0.0754 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1166 | 2 | 1003 | 10 | 0.1654 | 0.0697 | 0.0738 | -0.0016 | 0.007 | 0.2704 | 1.007 | 0.8081 | 0.1989 | -0.0754 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1167 | 2 | 1004 | 10 | 0.1654 | 0.0697 | 0.0739 | -0.0016 | 0.007 | 0.2706 | 1.007 | 0.8077 | 0.1993 | -0.0756 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1168 | 2 | 1005 | 10 | 0.1654 | 0.0697 | 0.0739 | -0.0016 | 0.007 | 0.2707 | 1.007 | 0.808 | 0.199 | -0.0755 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1169 | 2 | 1006 | 10 | 0.1654 | 0.0697 | 0.0741 | -0.0016 | 0.007 | 0.2708 | 1.007 | 0.8072 | 0.1998 | -0.0758 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1170 | 2 | 1007 | 10 | 0.1654 | 0.0697 | 0.0745 | -0.0016 | 0.007 | 0.2709 | 1.007 | 0.8061 | 0.2009 | -0.0761 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1171 | 2 | 1008 | 10 | 0.1654 | 0.0697 | 0.0745 | -0.0016 | 0.007 | 0.271 | 1.007 | 0.8061 | 0.2009 | -0.0762 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1172 | 2 | 1009 | 10 | 0.1654 | 0.0697 | 0.0748 | -0.0016 | 0.007 | 0.2711 | 1.007 | 0.8056 | 0.2015 | -0.0764 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1173 | 2 | 1010 | 10 | 0.1654 | 0.0697 | 0.0747 | -0.0016 | 0.007 | 0.2713 | 1.007 | 0.8058 | 0.2012 | -0.0764 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1174 | 2 | 1011 | 10 | 0.1654 | 0.0697 | 0.0748 | -0.0016 | 0.007 | 0.2714 | 1.007 | 0.8055 | 0.2015 | -0.0765 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1175 | 2 | 1012 | 10 | 0.1654 | 0.0697 | 0.075 | -0.0016 | 0.007 | 0.2715 | 1.007 | 0.8052 | 0.2018 | -0.0766 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1176 | 2 | 1013 | 10 | 0.1654 | 0.0697 | 0.0752 | -0.0016 | 0.007 | 0.2716 | 1.007 | 0.8045 | 0.2024 | -0.0769 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1177 | 2 | 1014 | 10 | 0.1654 | 0.0697 | 0.0753 | -0.0016 | 0.007 | 0.2717 | 1.007 | 0.8044 | 0.2026 | -0.0769 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1178 | 2 | 1015 | 10 | 0.1654 | 0.0697 | 0.0754 | -0.0016 | 0.007 | 0.2718 | 1.007 | 0.8041 | 0.2029 | -0.0771 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1179 | 2 | 1016 | 10 | 0.1654 | 0.0697 | 0.0755 | -0.0016 | 0.007 | 0.2719 | 1.007 | 0.804 | 0.203 | -0.0771 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1180 | 2 | 1017 | 10 | 0.1654 | 0.0697 | 0.0754 | -0.0016 | 0.007 | 0.2721 | 1.007 | 0.8043 | 0.2027 | -0.0771 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1181 | 2 | 1018 | 10 | 0.1654 | 0.0697 | 0.0754 | -0.0016 | 0.007 | 0.2722 | 1.007 | 0.8044 | 0.2025 | -0.0771 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1182 | 2 | 1019 | 10 | 0.1654 | 0.0697 | 0.0755 | -0.0016 | 0.007 | 0.2723 | 1.007 | 0.8043 | 0.2027 | -0.0771 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1183 | 2 | 1020 | 10 | 0.1654 | 0.0697 | 0.0756 | -0.0016 | 0.007 | 0.2724 | 1.007 | 0.8041 | 0.2028 | -0.0772 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1184 | 2 | 1021 | 10 | 0.1654 | 0.0697 | 0.0758 | -0.0016 | 0.007 | 0.2725 | 1.007 | 0.8036 | 0.2033 | -0.0774 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1185 | 2 | 1022 | 10 | 0.1654 | 0.0697 | 0.0759 | -0.0016 | 0.007 | 0.2726 | 1.007 | 0.8032 | 0.2037 | -0.0776 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1186 | 2 | 1023 | 10 | 0.1654 | 0.0697 | 0.0759 | -0.0016 | 0.007 | 0.2728 | 1.007 | 0.8033 | 0.2036 | -0.0776 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1187 | 2 | 1024 | 10 | 0.1654 | 0.0697 | 0.076 | -0.0016 | 0.007 | 0.2729 | 1.007 | 0.8031 | 0.2038 | -0.0777 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1188 | 2 | 1025 | 10 | 0.1654 | 0.0697 | 0.076 | -0.0016 | 0.0069 | 0.273 | 1.0069 | 0.8035 | 0.2034 | -0.0776 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1189 | 2 | 1026 | 10 | 0.1654 | 0.0697 | 0.0762 | -0.0016 | 0.0069 | 0.2731 | 1.0069 | 0.8029 | 0.204 | -0.0778 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1190 | 2 | 1027 | 10 | 0.1654 | 0.0697 | 0.0763 | -0.0016 | 0.0069 | 0.2732 | 1.0069 | 0.8026 | 0.2044 | -0.078 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1191 | 2 | 1028 | 10 | 0.1654 | 0.0697 | 0.0765 | -0.0016 | 0.0069 | 0.2733 | 1.0069 | 0.8022 | 0.2047 | -0.0781 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1192 | 2 | 1029 | 10 | 0.1654 | 0.0697 | 0.0765 | -0.0016 | 0.0069 | 0.2734 | 1.0069 | 0.8023 | 0.2046 | -0.0781 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1193 | 2 | 1030 | 10 | 0.1654 | 0.0697 | 0.0768 | -0.0016 | 0.0069 | 0.2736 | 1.0069 | 0.8013 | 0.2056 | -0.0785 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1194 | 2 | 1031 | 10 | 0.1654 | 0.0697 | 0.077 | -0.0016 | 0.0069 | 0.2737 | 1.0069 | 0.8009 | 0.206 | -0.0786 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1195 | 2 | 1032 | 10 | 0.1654 | 0.0697 | 0.0771 | -0.0016 | 0.0069 | 0.2738 | 1.0069 | 0.8008 | 0.2061 | -0.0787 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1196 | 2 | 1033 | 10 | 0.1654 | 0.0697 | 0.0771 | -0.0016 | 0.0069 | 0.2739 | 1.0069 | 0.8007 | 0.2062 | -0.0788 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1197 | 2 | 1034 | 10 | 0.1654 | 0.0697 | 0.0774 | -0.0016 | 0.0069 | 0.274 | 1.0069 | 0.7999 | 0.207 | -0.0791 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1198 | 2 | 1035 | 10 | 0.1654 | 0.0697 | 0.0776 | -0.0016 | 0.0069 | 0.2741 | 1.0069 | 0.7995 | 0.2074 | -0.0792 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1199 | 2 | 1036 | 10 | 0.1654 | 0.0697 | 0.0778 | -0.0016 | 0.0069 | 0.2742 | 1.0069 | 0.7988 | 0.2081 | -0.0795 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |
| guarded-w2 | 1200 | 2 | 1037 | 10 | 0.1654 | 0.0697 | 0.0781 | -0.0016 | 0.0069 | 0.2744 | 1.0069 | 0.7982 | 0.2087 | -0.0797 | move_2_score_gt_move_4, move_2_q_gt_move_4, selected_move_2 |

## Counterfactual root interventions

| artifact | intervention | simulations | selected_move | selected_is_reference | n_move_2 | n_move_4 | q_move_2 | q_move_4 | score_margin_2_minus_4 | decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current | original | 384 | 2 | false | 328 | 18 | 0.0623 | 0.0391 | 0.0055 | selected_non_reference | diagnostic-only intervention |
| current | original | 1200 | 2 | false | 1117 | 18 | 0.1028 | 0.0391 | 0.2211 | selected_non_reference | diagnostic-only intervention |
| current | root_prior_equalized | 384 | 2 | false | 276 | 70 | 0.0686 | 0.0074 | 0.219 | selected_non_reference | diagnostic-only intervention |
| current | root_prior_equalized | 1200 | 2 | false | 1077 | 70 | 0.0997 | 0.0074 | 0.2547 | selected_non_reference | diagnostic-only intervention |
| current | child_value_override_teacher | 384 | 2 | false | 277 | 68 | 0.0714 | 0.0203 | 0.2004 | selected_non_reference | diagnostic-only intervention |
| current | child_value_override_teacher | 1200 | 2 | false | 1067 | 68 | 0.0999 | 0.0203 | 0.4726 | selected_non_reference | diagnostic-only intervention |
| current | child_value_override_neural_swapped | 384 | 2 | false | 265 | 70 | 0.071 | 0.0081 | 0.252 | selected_non_reference | diagnostic-only intervention |
| current | child_value_override_neural_swapped | 1200 | 2 | false | 1064 | 70 | 0.0995 | 0.0081 | 0.5522 | selected_non_reference | diagnostic-only intervention |
| current | root_q_init_teacher | 384 | 2 | false | 277 | 68 | 0.0714 | 0.0203 | 0.2004 | selected_non_reference | implemented as root-child first-expansion override; clean root-Q preseed hook not present |
| current | root_q_init_teacher | 1200 | 2 | false | 1067 | 68 | 0.0999 | 0.0203 | 0.4726 | selected_non_reference | implemented as root-child first-expansion override; clean root-Q preseed hook not present |
| guarded-w2 | original | 384 | 2 | false | 234 | 1 | 0.0159 | -0.0723 | 0.0021 | selected_non_reference | diagnostic-only intervention |
| guarded-w2 | original | 1200 | 2 | false | 1037 | 10 | 0.0781 | -0.0016 | 0.2087 | selected_non_reference | diagnostic-only intervention |
| guarded-w2 | root_prior_equalized | 384 | 2 | false | 220 | 15 | 0.017 | -0.038 | 0.3027 | selected_non_reference | diagnostic-only intervention |
| guarded-w2 | root_prior_equalized | 1200 | 2 | false | 1034 | 15 | 0.0774 | -0.038 | 0.29 | selected_non_reference | diagnostic-only intervention |
| guarded-w2 | child_value_override_teacher | 384 | 2 | false | 320 | 24 | 0.0311 | -0.0176 | 0.3483 | selected_non_reference | diagnostic-only intervention |
| guarded-w2 | child_value_override_teacher | 1200 | 2 | false | 1130 | 24 | 0.0867 | -0.0176 | 0.4779 | selected_non_reference | diagnostic-only intervention |
| guarded-w2 | child_value_override_neural_swapped | 384 | 1 | false | 108 | 13 | -0.0203 | -0.032 | 0.0851 | selected_non_reference | diagnostic-only intervention |
| guarded-w2 | child_value_override_neural_swapped | 1200 | 2 | false | 920 | 13 | 0.0659 | -0.032 | 0.423 | selected_non_reference | diagnostic-only intervention |
| guarded-w2 | root_q_init_teacher | 384 | 2 | false | 320 | 24 | 0.0311 | -0.0176 | 0.3483 | selected_non_reference | implemented as root-child first-expansion override; clean root-Q preseed hook not present |
| guarded-w2 | root_q_init_teacher | 1200 | 2 | false | 1130 | 24 | 0.0867 | -0.0176 | 0.4779 | selected_non_reference | implemented as root-child first-expansion override; clean root-Q preseed hook not present |

## Interpretation

- Final classification: `teacher_decomposition_disagreement`.
- Current artifact root selected move at the largest traced budget: `2`.
- Teacher child ordering: `move_2_gt_move_4`.
- Current neural child ordering: `move_2_gt_move_4`.
- Current child PUCT ordering: `move_2_gt_move_4`.

## Recommended next action

Recommendation: **rebuild 002 reference with deeper exact/teacher search and do not treat move 4 as settled**.
