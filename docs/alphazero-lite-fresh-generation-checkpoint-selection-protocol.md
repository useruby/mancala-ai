# Fresh-Generation Checkpoint-Selection Protocol

## Purpose

Select a Generation-N candidate by predeclared arena evidence, not by a fixed
training endpoint. This protocol does not alter beta, optimizer, architecture,
MCTS parameters, or `model-artifact/current`.

## Frozen Before Execution

- Parent: the exact promoted Generation-(N-1) artifact and state hash.
- Cumulative baseline: P0, retained as an immutable artifact and weights hash.
- One fresh self-play generation from the parent only.
- Training: existing same-state beta=0.95 policy-head-only configuration.
- Checkpoints: steps `[1, 4, 16, 46]`. No additional checkpoints may be created
  after observing any arena result.
- Arena: canonical 128 openings, two games per opening, seat swapping,
  canonical evaluation seed contract, and 10,000 opening-bootstrap samples.
- Budgets: `384:256` and `1200:1200`; no search configuration changes.

## Evaluation Matrix

For every checkpoint with fit fraction at least `0.25`, run directly:

| Comparison | 384:256 | 1200:1200 |
| --- | --- | --- |
| Candidate vs parent | required | required |
| Candidate vs P0 | required | required |

P1-vs-P0 (or the current parent-vs-P0 relation) is reproduced at both budgets
as a reference control. Candidate-vs-P0 strength must never be inferred by
adding parent-relative effects.

## Selection Gate

A checkpoint is eligible only when all four candidate-vs-parent and
candidate-vs-P0 arena entries are safe. Safety is fixed as either a CI including
zero or a lower 95% bound of at least `-0.03`.

Among eligible checkpoints, select the earliest checkpoint whose
candidate-vs-parent `1200:1200` lower 95% CI is greater than zero. If no
checkpoint meets that gain rule, select none. Do not use training loss, fit
fraction above the minimum, or observed arena magnitude to choose a different
checkpoint.

## Promotion Evidence

The promotion report must include a `candidate_vs_p0` object with both canonical
contexts and their paired effect, bootstrap CI, seat effects, and W/D/L. The
promotion gate is invoked with `--require-cumulative-lineage` and this evidence;
missing or unsafe P0 evidence blocks promotion.
