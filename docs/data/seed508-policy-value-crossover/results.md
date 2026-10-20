# Seed508 exploratory policy/value output crossover

## Result

The preregistered paired-opening contrast is **P minus V = +0.029296875** points
per game (opening score averages its two seat-swapped games). Its 20,000-resample,
seed-508 two-sided 95% percentile interval is **[0.0009765625, 0.05712890625]**.
The interval is wholly above zero, supporting P over V on
this diagnostic suite.

| Composition | Wins | Draws | Losses | Score | Descriptive paired-opening 95% interval |
|---|---:|---:|---:|---:|---:|
| P: E4 policy + initializer value | 493 | 70 | 461 | 0.515625 | [0.49609375, 0.53564453125] |
| V: initializer policy + E4 value | 453 | 90 | 481 | 0.486328125 | [0.4638671875, 0.50830078125] |

These are descriptive standalone scores and do not imply an additive
decomposition. This diagnostic does not promote a checkpoint, reopen seed505's
failed advancement gate, authorize fresh confirmation, or include a 1,200-
simulation evaluation.

The independent ledger-only analysis in `independent-analysis.json` reproduces
the primary mean and interval exactly. All 2,048 games were independently replayed
by the seed508 verifier, including opening prefixes, legal transitions, terminal
sweeps/cap behavior, seat assignment, and stored outcomes.

## Historical comparison and limitations

Seed505's **historical full-E4** result against its initializer was 474 wins,
93 draws, and 457 losses (score **0.50830078125**; descriptive interval
**[0.48583984375, 0.5302734375]**). Its one-sided lower bound was
0.48974609375, so the registered seed505 advancement screen failed. Seed505 is
reported here descriptively only; this crossover is neither an additive
decomposition of that result nor a promotion gate.

Both crossover configurations used all 512 openings from seed488, each in both
seats. This suite had already been examined: it is reused evidence, not a fresh
or historically unseen holdout. Seed505 reported zero E4 opening-root matches
against seed504's training rows, but that does not remove suite reuse. The
unresolved seed461 continuation-population overlap limitation also remains.

## Frozen protocol and source assignments

- P challenger policy output: seed504 E4; challenger value output: frozen
  initializer.
- V challenger policy output: frozen initializer; challenger value output:
  seed504 E4.
- Each source is evaluated with its own complete artifact/trunk; only evaluator
  outputs are composed. No tensors were spliced; no new weights were trained or
  exported for this diagnostic.
- 512 matched openings × two seats = 1,024 games per composition, 2,048 total.
- 128 simulations per side, `c_puct=1.25`, one worker, seed `488 + opening
  index`, `azlite_eval_seed_v2`, 200-ply cap, zero random opening plies, and
  seed488's exact-root and search contracts.
- No game reached the cap (0/2,048); mean game length was 30.936 plies
  (maximum 57).
- Bootstrap: resample 512 opening clusters with replacement, carrying the two
  seat results together; 20,000 replicates, seed 508. A two-sided 95% interval
  wholly above zero supports P; wholly below zero supports V; otherwise the
  comparison is inconclusive.
- Python 3.14.7, NumPy 2.5.2. Exact resolved commands, per-command component
  assignments and hashes, runtime reports, game trajectories, and receipts are
  archived alongside the registration.

The first pre-execution freeze exposed a runner registration-key typo before any
arena invocation. That registration and freeze are preserved in
`superseded-pre-execution-freeze-001/`; the corrected registration and freeze
were completed and verified before the first game. A later verifier-only
portability correction preserves the exact pre-run verifier bytes in
`execution-source-snapshots/` and allows path relocation in copied checkouts.

## Reproduction and verification

From a checkout with the recorded runtime assets provisioned:

```bash
python -m ml.alphazero_lite.verify_seed508_policy_value_crossover --root .
python docs/data/seed508-policy-value-crossover/independent_analysis.py --root .
python docs/data/seed508-policy-value-crossover/make_receipt.py
python -m pytest ml/alphazero_lite/test_seed508_policy_value_crossover.py -q
ruff check ml/alphazero_lite/seed508_policy_value_crossover.py ml/alphazero_lite/verify_seed508_policy_value_crossover.py ml/alphazero_lite/test_seed508_policy_value_crossover.py
```

The verification suite replays all trajectories and validates every committed
complete seat pair, runtime contract, component assignment, and batch receipt.
The corrected resume path validates already committed pairs before skipping
them; an outcome-bearing incomplete batch is rejected and preserved.
`publication-receipt.json` binds the published registrations, raw batches,
outcomes, results, source snapshots, and report files.
