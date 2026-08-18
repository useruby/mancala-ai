# AlphaZero-Lite Aggregate Gradient Stability Audit

**Classification:** `inconclusive` (see interpretation below)

**Primary question:** does the harmful 384:256 effect remain once mini-batch noise is averaged out?
**Answer:** **Yes.** All five aggregate-step candidates are harmful at 384:256 (5/5 with upper 95% CI below zero), with a tight spread, so the harm is stable and does not vary materially across replay shards.

## Effective-batch-size curve (Phase C)

| Effective rows | Between-shard median cosine | Within-shard median cosine |
| ---: | ---: | ---: |
| 512 | 0.3963 | 0.4183 |
| 1024 | 0.4796 | 0.5441 |
| 2048 | 0.5778 | 0.6578 |
| 4096 | 0.6309 | 0.7355 |
| 8192 | 0.6699 | 0.8149 |

The between-shard cosine rises strongly (0.40 -> 0.67) as the effective batch grows, so PR #196's ~0.37 is dominated by ordinary 512-row mini-batch stochasticity.

## Variance decomposition (Phase D)

At the largest effective batch (8192 rows), within-shard aggregate updates (median cosine 0.815) exceed between-shard updates (0.670) by **-0.145**. This is a small but real residual between-shard component: between-shard alignment is modestly worse than within-shard, but both are in the "fairly stable" range.

## Shard-mean direction (Phase E)

| Pair | Raw | Clipped | Adam |
| --- | ---: | ---: | ---: |
| S0<->S1 | 0.9385 | 0.9385 | 0.7078 |
| S0<->S2 | 0.9129 | 0.9129 | 0.6781 |
| S0<->S3 | 0.8974 | 0.8974 | 0.6917 |
| S1<->S2 | 0.9055 | 0.9055 | 0.6543 |
| S1<->S3 | 0.8731 | 0.8731 | 0.6481 |
| S2<->S3 | 0.9264 | 0.9264 | 0.7221 |

The raw/clipped shard-mean gradients are highly aligned (~0.91), but the first Adam step (sign-normalized) drops to a median of ~0.685.

## Whole-game bootstrap (Phase F, n=1000)

Median between-shard Adam cosine: 0.6199 (95% CI [0.6147, 0.6185]).

## Canonical arena (Phase I)

| Budget | Candidate | Paired effect | 95% CI |
| --- | --- | ---: | --- |
| 384:256 | aggregate_S0 | -0.0410 | [-0.0684, -0.0156] |
| 384:256 | aggregate_S1 | -0.0234 | [-0.0410, -0.0039] |
| 384:256 | aggregate_S2 | -0.0410 | [-0.0684, -0.0156] |
| 384:256 | aggregate_S3 | -0.0254 | [-0.0488, -0.0039] |
| 384:256 | aggregate_grand_mean | -0.0332 | [-0.0566, -0.0098] |
| 1200:1200 | aggregate_S0 | -0.0566 | [-0.0762, -0.0371] |
| 1200:1200 | aggregate_S1 | -0.0703 | [-0.0898, -0.0508] |
| 1200:1200 | aggregate_S2 | -0.0605 | [-0.0801, -0.0430] |
| 1200:1200 | aggregate_S3 | +0.0000 | [+0.0000, +0.0000] |
| 1200:1200 | aggregate_grand_mean | -0.0566 | [-0.0762, -0.0371] |

## Interpretation

The evidence is a mix of the two hypotheses, dominated by (A):

- **(A) Mini-batch stochasticity is the dominant explanation.** The absolute cosine rises strongly with effective batch size (0.40 -> 0.67), and the raw shard-mean gradient cosine is ~0.91. The harmful effect is stable and consistently harmful once noise is averaged out (5/5 aggregate candidates harmful at 384:256, all CIs below zero).
- **(B) A modest residual between-shard component exists.** Within-shard (0.815) exceeds between-shard (0.670) by ~0.145 at the largest batch, and the shard-mean Adam-update cosine (~0.685) is below the 0.75 "highly aligned" threshold.

No single prespecified classification is cleanly met: `minibatch_noise_explains_low_cosine` (and its harmful variant) require shard-mean Adam cosine >= 0.75 (observed ~0.685) and "between not materially worse than within" (observed a -0.145 gap); `replay_distribution_instability_confirmed` requires the aggregate game effects to vary materially across shards (they do not - the 384:256 effects cluster tightly at -0.023..-0.041). The runner therefore reports `inconclusive`.

The substantive conclusion: PR #196's ~0.37 pairwise Adam cosine is **mostly** mini-batch noise, the aggregate update direction is moderately (not perfectly) stable, and the early destructive shared-trunk effect is a **stable and consistently harmful** direction rather than replay-distribution instability.

Full evidence: `docs/data/alphazero-lite-aggregate-gradient-stability-summary.json`.
