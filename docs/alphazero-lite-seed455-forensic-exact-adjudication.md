# Seed455 Exact Forensic Adjudication

Primary classification: `forensic_sparse_endgame_outcome_regression_confirmed`.

This is a read-only audit of `seed48-nextgen-s455-default-value`. It does not
train, generate self-play, run canonical arenas, change forensic thresholds, or
change the rejected promotion decision.

Candidate weights: `f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c`.
Incumbent weights: `935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c`.

## Frozen Gate Evidence

The original PR #356 report is preserved at
`.tmp/seed48-nextgen-s455-default-value/runs/seed48-nextgen-s455-default-value-iter1/candidate_forensic_suite.json`.
Its classic-MCTS policy reference used 1,200 simulations and its value teacher
used 1,800. The report did not persist a policy `reference_move`, so its
top-1 fields are the originally recorded `0.0000`; the reconstructed teacher
trace is a separate artifact and was not substituted into the gate evidence.

| Scope | System | Top-1 | Average regret | Blunder rate |
| --- | --- | ---: | ---: | ---: |
| Overall | Seed48 | 0.0000 | 0.0787 | 0.4509 |
| Overall | Seed455 | 0.0000 | 0.0810 | 0.4688 |
| `sparse_endgame` | Seed48 | 0.0000 | 0.1033 | 0.1667 |
| `sparse_endgame` | Seed455 | 0.0000 | 0.1289 | 0.2500 |
| `capture_available` | Seed48 | 0.0000 | 0.0570 | 0.5833 |
| `capture_available` | Seed455 | 0.0000 | 0.0290 | 0.5000 |

| Scope | Top-1 delta / threshold | Regret delta / threshold | Blunder delta / threshold | Result |
| --- | --- | --- | --- | --- |
| Overall | +0.0000 / >= -0.0200 | +0.0023 / <= +0.0200 | +0.0179 / <= +0.0100 | failed: blunder |
| `sparse_endgame` | +0.0000 / >= -0.0300 | +0.0256 / <= +0.0300 | +0.0833 / <= +0.0200 | failed: blunder |
| `capture_available` | +0.0000 / >= -0.0300 | -0.0280 / <= +0.0300 | -0.0833 / <= +0.0200 | passed |

The recorded failure codes remain `forensic_overall_regressed` and
`forensic_bucket_sparse_endgame_regressed`.

## Exact Reference And Coverage

The frozen `azlite_forensic_references_v2` reference was validated against all
224 suite positions. It has 213 exact-solved rows (95.09%) and 11 explicitly
unresolved opening rows, each retained without an MCTS substitute. Every solved
row covers every legal action. All 24 `sparse_endgame` rows are exact-solved.
The reference preserves tier-21 tablebase SHA
`f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7` and native
probe SHA identities per row.

The exact replay used the promotion forensic search semantics: 384 artifact
simulations, `c_puct=1.25`, seed 42. Detailed exact action margins, root-player
W/D/L utilities, tied optimal sets, selected moves, root network values, and
root exact values are in `forensic_exact_evaluation.json`. The reconstructed
1,200/1,800 MCTS trace, including all reference child win rates, is in
`forensic_mcts_reference_replay.json`.

## Sparse Endgame Rows

Exact margin regret is measured from the root player's perspective; W/D/L
regret is 0, 1, or 2. `MCTS reference` is classified against the exact optimal
margin set in the reconstructed trace.

| Position | Exact optimal | Current move | Current WDL regret | Current margin regret | Seed455 move | Seed455 WDL regret | Seed455 margin regret | Old MCTS reference |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sparse_endgame-001 | 4, 5 | 5 | 0 | 0 | 5 | 0 | 0 | 4 (exact optimal) |
| sparse_endgame-002 | 3 | 3 | 0 | 0 | 3 | 0 | 0 | 3 (exact optimal) |
| sparse_endgame-003 | 0, 1, 2 | 1 | 0 | 0 | 1 | 0 | 0 | 0 (exact optimal) |
| sparse_endgame-007 | 0, 1, 2 | 0 | 0 | 0 | 1 | 0 | 0 | 0 (exact optimal) |
| sparse_endgame-008 | 1, 5 | 5 | 0 | 0 | 5 | 0 | 0 | 5 (exact optimal) |
| sparse_endgame-009 | 2 | 2 | 0 | 0 | 2 | 0 | 0 | 5 (same-outcome suboptimal) |
| sparse_endgame-010 | 3 | 3 | 0 | 0 | 3 | 0 | 0 | 3 (exact optimal) |
| sparse_endgame-011 | 0 | 5 | 0 | 6 | 5 | 0 | 6 | 5 (same-outcome suboptimal) |
| sparse_endgame-012 | 0 | 5 | 0 | 4 | 5 | 0 | 4 | 5 (same-outcome suboptimal) |
| sparse_endgame-013 | 5 | 5 | 0 | 0 | 5 | 0 | 0 | 5 (exact optimal) |
| sparse_endgame-014 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 (exact optimal) |
| sparse_endgame-015 | 5 | 5 | 0 | 0 | 5 | 0 | 0 | 5 (exact optimal) |
| sparse_endgame-016 | 3 | 3 | 0 | 0 | 3 | 0 | 0 | 3 (exact optimal) |
| sparse_endgame-017 | 4 | 4 | 0 | 0 | 4 | 0 | 0 | 4 (exact optimal) |
| sparse_endgame-018 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 (exact optimal) |
| sparse_endgame-019 | 5 | 5 | 0 | 0 | 5 | 0 | 0 | 5 (exact optimal) |
| sparse_endgame-020 | 0, 2, 4 | 2 | 0 | 0 | 4 | 0 | 0 | 2 (exact optimal) |
| sparse_endgame-021 | 0, 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 (exact optimal) |
| sparse_endgame-022 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 (exact optimal) |
| sparse_endgame-023 | 4 | 4 | 0 | 0 | 5 | 1 | 6 | 4 (exact optimal) |
| sparse_endgame-024 | 4, 5 | 4 | 0 | 0 | 5 | 0 | 0 | 1 (same-outcome suboptimal) |
| sparse_endgame-025 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 5 (same-outcome suboptimal) |
| sparse_endgame-026 | 5 | 5 | 0 | 0 | 5 | 0 | 0 | 4 (same-outcome suboptimal) |
| sparse_endgame-027 | 5 | 5 | 0 | 0 | 5 | 0 | 0 | 5 (exact optimal) |

`sparse_endgame-023` is the decisive row. The exact player-zero action margins
are `{1: -12, 4: 0, 5: -6}` and the root-player W/D/L utilities are
`{1: -1, 4: 0, 5: -1}`. Seed48 selected 4 (draw); Seed455 selected 5 (loss).
The old MCTS reference also selected 4, so this is not a penalty for rejecting
an imperfect MCTS teacher. Both systems have the same two margin-only errors
(`-011`, `-012`), while only Seed455 adds the outcome error.

| Metric | Seed48 | Seed455 | Delta |
| --- | ---: | ---: | ---: |
| Exact-optimal rate | 0.9167 | 0.8750 | -0.0417 |
| WDL-optimal rate | 1.0000 | 0.9583 | -0.0417 |
| Mean margin regret | 0.4167 | 0.6667 | +0.2500 |
| Median margin regret | 0.0000 | 0.0000 | +0.0000 |
| Max margin regret | 6.0000 | 6.0000 | +0.0000 |
| WDL regression rate | 0.0000 | 0.0417 | +0.0417 |
| Margin-only regression rate | 0.0833 | 0.0833 | +0.0000 |

Across sparse rows, the old MCTS reference is exact-optimal for 18 rows and
same-outcome suboptimal for 6; it is outcome-suboptimal for none. Across all
213 exact-covered forensic rows, the corresponding counts are 114, 75, and 24.

## Overall Exact Evidence

Exact-covered aggregate results are kept separate from the unresolved rows and
from approximate MCTS regret.

| Metric | Seed48 | Seed455 | Delta |
| --- | ---: | ---: | ---: |
| Exact-optimal rate | 0.6901 | 0.7324 | +0.0423 |
| WDL-optimal rate | 0.9296 | 0.9390 | +0.0094 |
| Mean margin regret | 1.9249 | 1.4272 | -0.4977 |
| Median margin regret | 0.0000 | 0.0000 | +0.0000 |
| Max margin regret | 40.0000 | 14.0000 | -26.0000 |
| WDL regression rate | 0.0704 | 0.0610 | -0.0094 |
| Margin-only regression rate | 0.2394 | 0.2066 | -0.0329 |

The original MCTS report has 21 rows where Seed455 regret exceeded Seed48's;
20 are exact-covered and one (`opening_plies_1_8-012`) remains unresolved.
Among those 20 covered rows, exact margin direction is worse/better/equal on
7/9/4 rows and exact W/D/L direction is worse/better/equal on 4/2/14 rows.
Thus the original overall MCTS direction is not a reliable exact aggregate,
but it does include the real sparse outcome regression at `-023`.

## Exact-Solve Planning

No runtime feature is enabled here. Minimum descendant ply at which any state
first reaches each active-pit-stone threshold is recorded below. The decisive
`-023` root has 16 active stones; thresholds 12, 10, and 8 first occur at plies
2, 3, and 3 respectively, so the existing thresholds can affect descendants
but not its root directly. A threshold of 16 is not proposed.

| Position | Root active | <=12 | <=10 | <=8 |
| --- | ---: | ---: | ---: | ---: |
| -001 | 16 | 1 | 1 | 4 |
| -002 | 16 | 2 | 2 | 2 |
| -003 | 16 | 2 | 2 | 2 |
| -007 | 16 | 2 | 2 | 2 |
| -008 | 16 | 3 | 3 | 4 |
| -009 | 14 | 1 | 3 | 3 |
| -010 | 16 | 1 | 3 | 3 |
| -011 | 16 | 3 | 4 | 5 |
| -012 | 16 | 3 | 3 | 5 |
| -013 | 16 | 3 | 4 | 5 |
| -014 | 16 | 1 | 2 | 2 |
| -015 | 16 | 2 | 2 | 3 |
| -016 | 16 | 3 | 3 | 3 |
| -017 | 15 | 2 | 4 | 4 |
| -018 | 15 | 2 | 2 | 3 |
| -019 | 15 | 2 | 3 | 5 |
| -020 | 11 | 1 | 1 | 3 |
| -021 | 9 | 1 | 1 | 2 |
| -022 | 16 | 2 | 2 | 2 |
| -023 | 16 | 2 | 3 | 3 |
| -024 | 16 | 3 | 4 | 4 |
| -025 | 16 | 2 | 2 | 2 |
| -026 | 16 | 3 | 4 | 4 |
| -027 | 16 | 3 | 3 | 4 |

The next experiment should test the already-scaffolded exact-solve thresholds
8, 10, and 12 before considering retraining. Keep Seed48 and do not weaken the
forensic gate.
