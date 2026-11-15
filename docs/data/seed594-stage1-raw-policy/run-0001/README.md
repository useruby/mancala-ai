# Seed 594 Stage 1 raw-policy qualification — run 0001

**Decision: REJECT.** Exactly one evaluation was executed: Seed 594 deterministic
raw policy versus serving P-E4 on all 423 roots in the existing #570 membership.
This cohort is pre-existing, not a fresh holdout or sealed suite. No rerun is
authorized or performed.

## Registered metrics

| Metric | P-E4 | Seed 594 |
| --- | ---: | ---: |
| Exact-optimal top-action rate | 0.7470449173 | 0.7257683215 |
| Mean exact-optimal prior mass | 0.6581970120 | 0.6097302971 |
| Mean exact regret (seeds) | 1.3049645390 | 1.4515366430 |
| WDL downgrade count / rate | 28 / 0.0661938534 | 31 / 0.0732860520 |

Candidate-minus-P mean regret difference is **+0.1465721040 seeds**. The
registered 20,000-resample paired opening bootstrap (NumPy `default_rng(594)`,
linear quantiles) 95% CI is **[-0.0756501182, 0.3877068558]**.

All three advancement checks fail: the mean difference is not `<= -0.10`, the
CI upper endpoint is not `< 0`, and the candidate WDL downgrade rate is higher.
The registered decision is reject; qualification must not be repeated.

## Evidence and verification order

- `root-results.jsonl`: complete per-root paired predictions and exact-oracle
  metrics for all 423 membership roots.
- `verified-membership-mapping.json`: every expected state hash and exact
  opening-index mapping; this verification completed before aggregate metrics
  were calculated.
- `analysis.json` and `decision.json`: registered aggregates and gate result.
- `manifest.json`: evaluation count, identities, source/evidence hashes,
  authorization status, and complete mapping identity list.
- `../execution-authorization.json`: explicit project-owner authorization,
  waiver of independent human review, and freeze SHA-256 binding. The prior
  ChatGPT technical review is explicitly not represented as independent human
  approval.

Checkpoint identities are unchanged: Seed 594
`7fddf5d6cba7eea4e7b8e06ba89c5b936d28727470934a51c603a739768e9bdd`; serving
P-E4 `2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad`.
The membership SHA-256 is
`94e126af6ea88fb99a4230f84e5cf2a20248daa39ea66d25b68730ea0426d668`.

The evaluation used one CPU thread, Python 3.14.7, NumPy 2.5.2, PyTorch
2.13.0+cpu, and the frozen hash/BLAS thread settings. Synthetic parity passed
for both models before qualification access (three fixtures per model; maximum
absolute probability differences were `1.1920928955078125e-07` for P-E4 and
`5.960464477539063e-08` for Seed 594; illegal-action probabilities were exactly
zero).

The exact authorization-bound preparation freeze used for run 0001 is preserved
as `execution-freeze.json` in this directory and has SHA-256
`d70b3a11c950b331ada5b20d74b0b7d15acaed9eeff56e9b83ef1dffb42bbcb8`. After
execution, CI identified formatter-only changes in the runner and its test; the
current preparation freeze now pins those formatted source bytes. No evaluation
code behavior, checkpoint identity, membership, threshold, or run evidence was
changed, and the original authorization remains bound to the preserved run
freeze.
