# Seed488 completed evaluation: retrospective accounting

This read-only descriptive report uses the registered `openings.jsonl`, verified
`outcomes.jsonl`, per-opening archived game logs and reports, and the already
published `results.json` / `analysis.json`. It does not alter any historical
publication file or recompute the registered bootstrap. Run the accounting and
focused checks from the repository root with:

```bash
.venv-azlite/bin/python -m ml.alphazero_lite.seed488_retrospective --root .
.venv-azlite/bin/python -m pytest -q ml/alphazero_lite/test_seed488_retrospective.py
```

## Reconciliation and registered result

The registered suite has 512 openings, each played once with F at player 0 and
once at player 1: 512 complete seat pairs and 1,024 games. The outcome ledger
and archived logs each account for all 1,024. F recorded **448 wins, 106 draws,
and 470 losses**, for score `(448 + 0.5 × 106) / 1,024 = 0.4892578125`.
There were no 200-ply capped games. The published paired-opening bootstrap is
20,000 resamples (seed 488), one-sided 95% lower bound **0.4755859375** and
descriptive two-sided 95% interval **[0.47314453125, 0.50537109375]**.
The registered advance gate required the lower bound to be strictly greater
than 0.50 and was not passed.

F's score by seat was **0.5537109375** at player 0 (256 wins, 55 draws,
201 losses) and **0.4248046875** at player 1 (192 wins, 51 draws, 269 losses).
The distribution of the 512 paired-opening scores (each game worth 0, 0.5, or
1 point, averaged within opening) was:

| Paired-opening score | Openings |
|---:|---:|
| 0.00 | 32 |
| 0.25 | 37 |
| 0.50 | 388 |
| 0.75 | 31 |
| 1.00 | 24 |

## Archived runtime telemetry

Game lengths ranged from **5 to 61 plies**, with median **33** and mean
**33.31**. Exact-root handoff telemetry was present: **953/1,024 games** had at
least one handoff, totaling **12,876** handoffs (12.57 per game over all games).
For those 953 games, first handoff was recorded at median ply **21** and mean
**20.84**. The other 71 games have no first-handoff ply.

The game records' `normal_puct_time_ms` totals **2,010,197.43 ms** (mean
**1,963.08 ms/game**). `exact_root_solve_time_ms` totals **1,198.08 ms** (mean
**1.17 ms/game**, or **0.093 ms per recorded handoff**). These are sums and
means of the fields as recorded, not independently timed or normalized per
move; they describe this run's recorded telemetry only.

## First-action comparison and interpretation

The two seat-swapped logs per opening record actor-relative first actions for
both arms. Comparing each arm's logged first action between the two games of
the same opening gives F agreement on **95/512** openings (417 disagreements)
and initializer agreement on **88/512** (424 disagreements). The logs do not
provide a reliable action-value/regret measure or a causal account of why a
first action differs. First-action agreement is only a descriptive repeatability
comparison across the seat-swapped runs.

The seat-score gap and frequent seat-swapped first-action disagreement provide
a concrete *candidate for further descriptive investigation*: whether early
action choices or seat-conditioned behavior account for part of the observed
variation. They do not establish a mechanism, treatment effect, or a revised
advance decision. The available artifacts do not report per-decision value
estimates/regret, controlled counterfactual outcomes, or a certified comparison
against seed461's unresolved continuation population. Consequently they do not
support a causal next-step claim or fresh confirmation. Any subgroup observation
here is retrospective and descriptive only.

## Practical conclusion and limits

**Seed479 F did not pass seed488's strength advance gate.** Do not promote it
and do not authorize fresh confirmation. Retain the unresolved seed461 overlap
limitation: the registered exclusion evidence cannot authenticate identities
for 1,129,643 receipt-matched continuation actions. The opening suite's checks
against reconstructable exclusions do not remove that limitation.

Reproduce the primary result with the independent verifier command already
published in `execution-commands.md`; the verifier returned
`verified_complete` before this retrospective work began. The analysis adds no
new game, search, inference, training, or decision rule.
