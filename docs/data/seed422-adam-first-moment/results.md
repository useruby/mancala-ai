# Seed422 Adam first-moment memory ablation

**Decision: retain the baseline and close this fixed β₁ intervention.** The
preregistered advancement rule was not met. This result is conditional on the
frozen #416 A-target replay dataset, initialization, source-row split, batch
order, and the seed422 opening suite. It makes no claim about other datasets or
training orders and authorizes no beta sweep, extra games, checkpoint
selection, or automatic promotion.

## Intervention and training

The only optimizer change was Adam β₁: lane A used `(0.9, 0.999)` and lane B
used `(0.0, 0.999)`. Both used ε `1e-8`, LR `0.001`, zero weight decay, no
scheduler, batch size 512, and four complete epochs. The registered initializer
was the seed455 checkpoint (`7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c`);
the architecture was residual_v3 `[96, 3]` with kalah_v3 inputs, cross-entropy
plus `0.3 × Huber(δ=1)`, and global gradient clipping at 1. The five #416 A
targets were used at weights `1/4/1/8/4`; fresh value targets remained default
and the historical sources remained sharpened. Both lanes completed 1,052
updates with identical exposures and the four frozen #416 permutations.

Lane A reproduced all four published #416 checkpoint identities. Its E4 hash is
`836fc769c62126b649ce9691494719506b88efec0746769c9a589522d08cb2cd`. The
post-A bookkeeping amendment records that the frozen runner completed A's
updates and checkpoints but raised a `KeyError` in the subsequent parameter
drift report because model state-dict names differ from exported checkpoint
names. The four A epoch hashes and permutation hashes match #416 exactly, so
the published #416 A loss and gradient-norm history is bound as the reproduced
history; no A update was repeated. The amendment was frozen before B training.
See `execution-accounting-amendment.json`.

| Lane | β₁ | E4 checkpoint SHA256 | Updates | E4 parameter drift L2 | Score vs seed455 |
|---|---:|---|---:|---:|---:|
| A | 0.9 | `836fc769c62126b649ce9691494719506b88efec0746769c9a589522d08cb2cd` | 1,052 | 6.995381 | 0.482910 |
| B | 0.0 | `78da174747aff7c18d015b78573b6f7bb6ccc7476d74b30c515cc77b3028936b` | 1,052 | 7.850688 | 0.499512 |

## Exclusions and evaluation

The exclusion union starts from the corrected post-execution #415 proof and
the validated #416 v3 union. It adds #416's declared suite and every starting,
intermediate, and terminal state strictly replayed from its complete published
arena trajectories; #418's declared roots and published root/search/oracle
identities; and #420–421 declared roots plus all published `state_sha256`
identities in their instrumented evaluator traces. The final deduplicated union
contains 266,575 identities (`f6713bebfe1ded09ed15e45f17a11566d0012f03fe7719138fdc832bbb1339f7`).
It does not claim reconstruction of unrecorded internal search states. The
512 unique nonterminal openings each have more than 32 active stones and were
strictly replayed; their overlap with the union is zero.

Each candidate played 1,024 new games against the frozen seed455 artifact, for
2,048 total games. Every opening was evaluated in both seats, both sides used
384 simulations, c_puct 1.25, `azlite_eval_seed_v2`, and the unchanged native
root-16 runtime contract. No historical game outcome was reused.

| Candidate | Wins | Draws | Losses | Score | Seat 0 | Seat 1 |
|---|---:|---:|---:|---:|---:|---:|
| A | 446 | 97 | 481 | 0.482910 | 0.266113 | 0.216797 |
| B | 469 | 85 | 470 | 0.499512 | 0.266602 | 0.232910 |

The primary paired B-minus-A opening-average score was **+0.016602**, with a
10,000-resample opening-cluster 95% percentile interval **[-0.009277,
0.042480]** (seed 422). B's opening-cluster score interval was **[0.477539,
0.521484]**. Advancement required B−A ≥0.03 with lower bound >0, and B score
≥0.53 with lower bound >0.5. Neither condition is satisfied; retain the
baseline. Any positive result in a future separately registered confirmation
would remain conditional on its frozen evidence.

## Evidence files

- `registration.json` and `execution-source-snapshots/`: immutable prospective
  protocol and bound implementation snapshots.
- `opening-exclusion-proof.json`, `openings.jsonl`: source-by-source historical
  identity union and new suite.
- `training-results.json`, `execution-accounting-amendment.json`: checkpoint,
  optimizer, exposure, update, loss, norm, drift, and reproduction accounting.
- `runtime-binding.json`: candidate exports, frozen opponent, sidecars, native
  probe, tablebase, and root-16 contract identities.
- `outcome-binding.json`, `outcome-ledger.jsonl`: lane resume/accounting bind
  and complete strict-replayable game records.
- `opening-matrix.json`, `analysis.json`: all opening pairs, summaries,
  bootstrap intervals, and fixed decision.
- Verify with `PYTHONPATH=. python ml/alphazero_lite/verify_seed422_adam_memory.py`.
  The verifier requires neither Torch nor exported/runtime model artifacts.
