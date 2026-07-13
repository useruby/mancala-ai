# AlphaZero-Lite 768:768 Search Variant Diagnostic

- source summary: `/tmp/azlite_terminal_outcome_selfplay_iteration`
- candidate artifact: `/tmp/azlite_terminal_outcome_selfplay_iteration/value_head_only_e2/artifact_value_head_only_e2`
- cases analyzed: `64`

## Variant Aggregate

| Variant | Cand changed rate | Cand teacher rate | Cur teacher rate | Cand-cur teacher | Mean |dV| | Mean |dScore| |
|---|---|---|---|---|---|---|
| default | +1.0000 | +0.3594 | +0.3750 | -0.0156 | +0.1202 | +0.1202 |
| normalize_values | +0.4844 | +0.3281 | +0.2969 | +0.0312 | +0.1119 | +0.1119 |
| value_trust_half | +0.1719 | +0.6406 | +0.7031 | -0.0625 | +0.0866 | +0.0866 |
| normalize_values_plus_value_trust_half | +0.6406 | +0.2500 | +0.2812 | -0.0312 | +0.1037 | +0.1037 |

## Sample Cases

| Game | Ply | Phase | Teacher | Base cur | Base cand | Best variant | Teacher delta |
|---|---|---|---|---|---|---|---|
| 948 | 31 | late | 4 | 4 | 5 | normalize_values | +0.0000 |
| 1004 | 31 | late | 4 | 4 | 5 | normalize_values | +0.0000 |
| 894 | 27 | mid | 1 | 4 | 1 | default | +1.0000 |
| 877 | 15 | mid | 0 | 2 | 0 | default | +1.0000 |
| 947 | 39 | late | 1 | 1 | 3 | normalize_values | +0.0000 |
| 975 | 38 | late | 4 | 0 | 4 | default | +1.0000 |
| 888 | 2 | opening | 4 | 5 | 1 | value_trust_half | +0.0000 |
| 892 | 2 | opening | 1 | 5 | 1 | default | +1.0000 |
| 906 | 2 | opening | 1 | 5 | 1 | default | +1.0000 |
| 913 | 2 | opening | 1 | 5 | 1 | default | +1.0000 |
| 924 | 2 | opening | 1 | 5 | 1 | default | +1.0000 |
| 928 | 2 | opening | 4 | 5 | 1 | value_trust_half | +0.0000 |

