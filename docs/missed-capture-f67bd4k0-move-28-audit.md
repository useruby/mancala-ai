# Missed Capture F67BD4K0 Move 28 Audit

## Classification

`fixture_expectation_stale`

The exact KVTB1 result makes fixture move `1` and candidate move `5` equal.
Both are forced losses by six seeds for root player `1`; the fixture's
singleton acceptable set `[1]` is not justified. The fixture is intentionally
unchanged in this audit.

## Reconstructed Position

The regression runner passes the fixture state directly to
`arena.evaluate_artifact_position`, which reconstructs it with
`KalahGame.from_state`.

| Field | Value |
| --- | --- |
| Player-0 pits | `[1, 8, 0, 1, 1, 0]` |
| Player-1 pits | `[1, 3, 5, 0, 0, 1]` |
| Stores (P0/P1) | `20 / 7` |
| Side to move | player `1` |
| Legal moves | `0, 1, 2, 5` |
| Token / move number | `f67bd4k0` / `28` |
| Active pit seeds | `21` |

The full machine-readable states, exact labels, hashes, and commands are in
`docs/data/missed-capture-f67bd4k0-move-28-audit.json`.

## Root Actions

| Move | Immediate behavior | Resulting side | Prior | Visits/share | Q | Exact W/D/L (root P1) | Exact root score | Exact rank |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 0 | no capture, no extra turn | P0 | 0.08782547 | 28 / 7.291667% | -0.70627252 | L | -14 | 3 |
| 1 | capture 9, no extra turn | P0 | 0.20803186 | 127 / 33.072917% | -0.66806832 | L | -6 | 1 |
| 2 | no capture, no extra turn | P0 | 0.33562383 | 66 / 17.187500% | -0.75035860 | L | -14 | 3 |
| 5 | no capture, extra turn | P1 | 0.36851883 | 163 / 42.447917% | -0.68300304 | L | -6 | 1 |

The candidate selected `5`. Raw prior and visit rankings also select `5`;
the final Q ranking selects `1`. Exact play ranks `1` and `5` jointly first,
so the selection is not an exact-outcome error.

## Exact Oracle And Perspective

The existing isolated `native/kalah_v1_tablebase` KVTB1 implementation was
used with the locally generated tier-21 tablebase, SHA-256
`f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7`.
Tier 21 covers every pit distribution at this position's 21 active seeds.
This is exact tablebase evidence, not a deeper-MCTS estimate.

KVTB1 action values are final player-0 pit margins under optimal play. Adding
the root store margin (`20 - 7 = +13`) yields final player-0 margins:
`{0: +14, 1: +6, 2: +14, 5: +6}`. The root player is player 1, so root score
is the negated margin. This explicitly gives moves `1` and `5` the same
root-perspective score, `-6`, and the same W/D/L label, loss.

The native tablebase rules and values were independently checked against the
Python rules and exact solver through tier 8 and sampled through tier 12 in
the existing native-tablebase preflight. The tier-21 artifact itself is not
checked in; its checksum and deterministic command are retained in the JSON
artifact above.

## Reproducibility

The frozen candidate weights SHA-256 is
`935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c`.
The regression configuration is 384 simulations, seed `17`, `c_puct=1.25`,
zero FPU, deterministic root policy, disabled subtree reuse/value
normalization, and zero tactical root bias.

```bash
bash native/kalah_v1_tablebase/build.sh
./.tmp/kalah_v1_tablebase_build/kalah_v1_tablebase probe .tmp/kalah_v1_21.kvtb <<< '{"pits":[1,8,0,1,1,0,1,3,5,0,0,1],"player":1}'
.tmp/venv/bin/python script/ai/check_superhuman_regressions --artifact .tmp/fresh-uniform1200/runs/seed48/uniform1200/fresh-uniform1200-s48-uniform1200-iter1 --simulations 384 --out .tmp/missed_capture_f67bd4k0_move_28-rerun.json
cmp -s .tmp/missed_capture_f67bd4k0_move_28-rerun.json docs/data/alphazero-lite-shadow-canonical-hard-arena/candidate_regression_suite.json
```

The production rerun report SHA-256 is
`62f3ad7a14594a65d4757c98a3f6192b6cd2c1dea99c56e2f455308af8202ee7`,
identical to the committed report. It reproduces the root state and every
recorded search telemetry field exactly.
