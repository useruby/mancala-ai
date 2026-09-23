# Seed455 Forensic Policy V2 Reclassification

This is an offline reclassification of frozen PR #362 forensic rows. No model
search, training, self-play, or canonical arena was run.

## Inputs

- Frozen forensic report: `.tmp/seed455-exact-root16-canonical/candidate_forensic_suite.json`
- Exact reference: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v2.json`
- Exact reference SHA256: `45aaa5c4e216e19eeff517895c9fc16099dc1406d5a1f233dd632a57c251d626`
- Exact reference schema: `azlite_forensic_references_v2`
- Suite: `incumbent_forensic_suite_v1`, 224 rows
- Exact coverage: 213 solved rows, 11 unresolved rows, including all 24 sparse-endgame rows
- Solved action tables cover every legal action. The retained tier-21 tablebase provenance includes `f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7`.

## Policy Semantics

`forensic_policy_v2_exact_top1` uses exact optimal-action-set membership for
the 213 solved rows. The 11 unresolved rows retain the frozen classic-MCTS
reference move. Regret and blunder retain classic-MCTS child-stat semantics on
all 224 rows; no exact margin values enter legacy thresholds.

| System | Exact rows | Approximate rows | Exact top-1 | Approximate top-1 | Hybrid top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seed48 | 213 | 11 | 0.6995 | 0.5455 | 0.6920 |
| Seed455 | 213 | 11 | 0.7465 | 0.4545 | 0.7321 |

| Bucket | Seed48 exact top-1 | Seed455 exact top-1 | Hybrid delta |
| --- | ---: | ---: | ---: |
| sparse_endgame | 1.0000 | 1.0000 | 0.0000 |
| capture_available | 0.5000 | 0.7083 | +0.2083 |
| overall | 0.6995 | 0.7465 | +0.0402 hybrid |

The prior sparse top-1 delta of `-0.0417` is eliminated because the distinct
selected moves are members of the same exact-optimal action sets.

## Scope Check

The frozen regret and blunder values are unchanged for both systems:

| System | Overall regret / blunder | Sparse regret / blunder | Capture regret / blunder |
| --- | --- | --- | --- |
| Seed48 | 0.0793 / 0.4643 | 0.1087 / 0.2917 | 0.0570 / 0.5833 |
| Seed455 | 0.0789 / 0.4732 | 0.1087 / 0.2917 | 0.0290 / 0.5000 |

With unchanged thresholds, the corrected forensic quality passes. Classification:
`forensic_exact_top1_semantics_removes_seed455_blocker`. Seed455 remains
rejected: this is an evaluation-policy correction, not a promotion decision.
