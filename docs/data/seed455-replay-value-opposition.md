# Seed455 complete replay-gradient versus fresh-value diagnostic

## Frozen procedure

This is an offline, forward/backward-only diagnostic using the hash-verified
seed455 checkpoint, original O0 E4 checkpoint (secondary), the five registered
replay sources, original weights and target modes, production train/validation
split, and four deterministic source-row partitions. The frozen source and
configuration manifest is `seed455-replay-value-opposition-manifest.json`; the
complete machine-readable accounting is
`seed455-replay-value-opposition-results.json`.

For row multiplicity `m`, policy weight `q`, and source `s`, each training source
contribution is

`P_s = grad(sum_s(m*q*CE) / global_sum(m*q))`

`V_s = grad(0.3*sum_s(m*Huber) / global_sum(m))`

and `G = sum_s(P_s + V_s)`. For fresh value cohort `C`, `VF_C` is the fresh
source's weighted Huber numerator gradient restricted to C, divided by the
same global `sum(m)` denominator. The cohort restriction is not renormalized.
Within each partition, both gradients use rows from that partition and retain
the original global denominators. Reported source/objective dots sum to
`dot(G, VF_C)`; component norms and all pairwise cross-dots allow the mixture
norm to be reconstructed. All reported groups are projections of the same
complete parameter vector: four existing trunk blocks, shared trunk, policy
head, value head, and all trainable parameters.

## Finding

For the primary seed455 `>32` cohort, the complete-parameter result is
`dot(G,VF_>32)=+0.0022967803`, cosine `+0.04150295`, `||G||=1.16358922`, and
`||VF_>32||=0.04755989`. The four same-partition cosines are `+0.05382272`,
`+0.05828476`, `+0.00094483`, and `+0.05034843`. The frozen rule requires a
cosine at most `-0.05` overall and in at least three partitions. It is not met:
**no robust net-opposition signal**, and no value-target provenance/calibration
audit is recommended by this rule.

On the shared trunk, the corresponding primary dot/cosine are
`+0.0019291512 / +0.03659512`. The residual block 0 dot is negative
(`-0.00074805`), but input projection, residual blocks 1 and 2, and the value
head dots are positive. This is why the primary decision uses the complete
trainable parameter set as specified. Fresh-policy alignment is also reported
as a secondary comparison in the result artifact.

The secondary original O0 E4 `>32` all-parameter cosine is `-0.13011160` and all
four partition cosines are negative, but E4 cannot trigger the frozen follow-up
rule. Its result does not change the seed455 decision.

The full all-parameter source/objective contributions to the primary seed455
dot (their sum is the dot above) are:

| Source/objective | Dot with `VF_>32` |
|---|---:|
| fresh policy | -0.0026357922 |
| fresh weighted value | +0.0061133150 |
| generic bootstrap policy | -0.0016484510 |
| generic bootstrap weighted value | -0.0005967086 |
| opening disagreement policy | +0.0002845475 |
| opening disagreement weighted value | +0.0005124295 |
| random teacher policy | -0.0002134994 |
| random teacher weighted value | -0.0001804580 |
| stability policy | +0.0003578797 |
| stability weighted value | +0.0003035177 |

The four partition sets are disjoint source-row partitions, not independent
games or confidence intervals. A negative dot would mean an infinitesimal raw
gradient-descent step increases the fresh value objective to first order. This
diagnostic is not a reconstruction of minibatch Adam updates and makes no
arena or playing-strength claim. It does not reverse #403 or authorize replay
reweighting. The frozen decision rule was fixed before the final gradient
execution.

The portable verifier recomputes published geometry, cross-dot accounting,
decision, and classification:

```bash
python -m ml.alphazero_lite.verify_replay_value_opposition \
  docs/data/seed455-replay-value-opposition-results.json
```

Focused regression tests:

```bash
python -m pytest ml/alphazero_lite/test_replay_value_opposition.py \
  ml/alphazero_lite/test_replay_gradient_mapping.py -q
```
