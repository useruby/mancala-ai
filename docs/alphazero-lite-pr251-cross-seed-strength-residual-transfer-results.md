# PR #251 Cross-Seed Strength Residual Transfer

**Classification:** `strength_residual_not_transferable`

This frozen intervention evaluated only the four PR #250 checkpoints and raw
positive-minus-negative policy-adapter residuals. It did not train, take an
optimizer step, generate self-play, change MCTS, scale a residual, or promote
a model. The complete machine-readable result is
`/tmp/azlite_pr251_cross_seed_strength_residual_transfer/summary.json`.

## Frozen Contract

| Suite | Seed | SHA-256 | Status |
| --- | ---: | --- | --- |
| D | 4042 | `e9d931031e75d39d188699d0b77ecc91d429c6998d06aada5f04e34b0384b1e2` | CONSUMED |
| E | 5042 | `3e78f5425370bb17ed2280e850a5f93b62cb86d94a0222263d481fcf866def37` | CONSUMED |
| F | 6042 | `8cba0ea407dde34696877d5ee2fe7110cd5610b2392f446df1bc02c1900d7c0e` | CONSUMED |

There were zero duplicate opening states within D/E/F, between D/E/F, against
the canonical suite and consumed A/B/C suites, or against either training
replay. Prefix-overlap checks also found zero overlaps. Candidate model hashes
matched all four PR #250 frozen hashes. Same-seed reconstruction had maximum
adapter error `1.82e-12`; all non-adapter tensors remained byte-identical to
their recipient.

All arenas used ordinary deterministic `1200:1200` PUCT, `c_puct=1.25`, seat
swapping, arena seed 42, and one matched P1-vs-P1 control per suite.

## Residual Geometry

| Metric | Value |
| --- | ---: |
| `||r45||` | .0002534611 |
| `||r46||` | .0002764355 |
| `cosine(r45, r46)` | -.757604 |
| `dot(r45, r46)` | `-5.30820e-08` |
| `||r45|| / ||r46||` | .916891 |
| r45 / full positive delta norm | .112727 |
| r46 / full positive delta norm | .123917 |

The residual weight/bias norms were `.000252822` / `.000017992` for r45 and
`.000275482` / `.000022938` for r46. Thus the discriminative residuals are
small but, unlike the full PR #250 update deltas, strongly anti-aligned.

## Primary Results

| Contrast | D | E | F | Pooled effect / CI95 | Hierarchical CI95 | Positive suites |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| seed45 positive - negative | +.042969 | +.044922 | +.042969 | +.043620 [+.034505, +.053385] | [+.027344, +.060547] | 3/3 |
| seed46 positive - negative | +.042969 | +.044922 | +.042969 | +.043620 [+.034505, +.053385] | [+.027344, +.060547] | 3/3 |
| seed45 cross rescue - negative | .000000 | .000000 | .000000 | .000000 [.000000, .000000] | [.000000, .000000] | 0/3 |
| seed46 cross rescue - negative | .000000 | .000000 | .000000 | .000000 [.000000, .000000] | [.000000, .000000] | 0/3 |
| seed45 positive - cross remove | +.042969 | +.044922 | +.042969 | +.043620 [+.034505, +.053385] | [+.027344, +.060547] | 3/3 |
| seed46 positive - cross remove | +.042969 | +.044922 | +.042969 | +.043620 [+.034505, +.053385] | [+.027344, +.060547] | 3/3 |
| seed45 cross rescue - scrambled | .000000 | .000000 | .000000 | .000000 [.000000, .000000] | [.000000, .000000] | 0/3 |
| seed46 cross rescue - scrambled | .000000 | .000000 | .000000 | .000000 [.000000, .000000] | [.000000, .000000] | 0/3 |

The fresh positive controls replicated. Neither raw cross-rescue passed, while
both cross-removal contrasts passed because removal of the other seed residual
reproduced the matched negative behavior. The scrambled controls also did not
rescue, so this is not evidence for magnitude-only sensitivity.

Cross-rescue was not behaviorally equivalent to the same-seed positive: both
pooled differences were `-.043620`, CI95 `[-.053385, -.034505]`.

## Diagnostics

On a fixed 128-state diagnostic set disjoint from canonical, A/B/C, and D/E/F
openings, all top-1 legal actions agreed. Cross-rescue adapter changes were
`.000276435` (seed45) and `.000253461` (seed46), with mean legal-policy L1
`.000211185` and `.000176138`; maximum legal-logit shifts were `.00102699` and
`.00066440`. The corresponding scrambled controls had much smaller mean L1
(`.000021123` and `.000019489`) but still did not affect paired outcomes.

Cross-remove restored the historical challenger-seat-1, ply-7 divergence
signature: 44, 46, and 44 divergent games in D/E/F, respectively, and each
changed the final outcome. Cross-rescue did not reproduce this signature: the
seed45 transplant had no trajectory divergence; seed46 divergences occurred
at plies 22 and 32 and never changed an outcome.

## Gradient-Difference Audit

| Geometry | Cosine |
| --- | ---: |
| `r45, -dg45` | .819149 |
| `r46, -dg46` | .648863 |
| `r45, -dg46` | -.744065 |
| `r46, -dg45` | -.762879 |
| `dg45, dg46` | -.891146 |

The within-seed residuals are visible in their own negative target-gradient
directions, but their cross-seed gradient-difference relationships are
anti-aligned. Raw positive updates are therefore correlated across replay
seeds while the discriminative component is not a reusable causal direction.

## Next Experiment

Perform state/action-level target-difference attribution rather than additional
parameter transplants.
