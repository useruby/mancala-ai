# seed432 — same-input policy-target compatibility census

This is a read-only data-quality census of the five-source seed461 mixture, using
the registered 1/4/1/8/4 replay multiplicities, `sharpened` policy-target loader
mode, per-source registered value modes, and #416 registration-v3 expanded-row
split. It did not run training, inference, search, or games. Exact input bytes
are 27 little-endian float32 values. Rows are never grouped across partitions.

## Decision

**`target_disagreement_not_material_under_registered_thresholds`**. The rule
requires both mean generalized JS >= 0.05 nats and weighted top-action
disagreement >= 10%, under both primary weight definitions. In training rows
with >32 active stones:

| Weighting | Mean JS (nats) | Weighted top-action disagreement |
|---|---:|---:|
| Replay exposure × policy coefficient | 0.0802718178 | 11.1902005% |
| Equal-input weighting × policy coefficient | 0.0615598109 | 6.7609416% |

The top-action criterion fails under equal-input weighting, so the fixed joint
classification does not indicate material disagreement. These prioritization
thresholds are not significance or strength gates.

## Primary accounting

There are 87,625 compact source rows and 149,448 expanded replay positions.
The frozen split assigns 134,502 positions to training and 14,946 to
validation. Primary >32 training includes 13,738 exact-input groups, of which
2,172 are duplicate groups. Canonical-state grouping yields the same 13,738
groups and no definition disagreement; exact-input groups have no multiple-
canonical-state cases. The group count includes singleton inputs at J=0.
Policy-mass coverage is 53,883 coefficient-exposure units and 29,691
equal-input coefficient units. Primary J decomposes as follows:

| Weighting | Within-source JS | Between-source JS | Sum / total JS |
|---|---:|---:|---:|
| Exposure | 0.0197808031 | 0.0604910146 | 0.0802718178 |
| Equal-input | 0.0090875315 | 0.0524722794 | 0.0615598109 |

Per-source coverage and source-pair top-action rates are in `results.json`.

For context, training 17–32 mean JS is 0.0013129493 (exposure) and 0.0009180599
(equal-input); <=16 is 0.0000836329 and 0.0001313958. Validation >32 is
0.0610640642 and 0.0462035478 respectively; validation 17–32 is below
0.0000012 and <=16 below 0.0000014. Full strata metrics are in `results.json`.

## Interpretation and limitations

For logits `z`, policy cross-entropy is linear in the target: for targets `p_i`
and preserved weights `w_i`, `sum_i w_i CE(p_i,z) = W CE(sum_i w_i p_i/W,z)`.
Differentiating with respect to `z` preserves the same identity, so replacing
same-input targets by their weighted mean while preserving total weight leaves
the aggregate cross-entropy objective and gradient unchanged in exact
arithmetic. This is an equivalence statement, not a proposed remediation.

Target disagreement alone establishes neither incorrect labels nor a cause of
the parent-to-E4 loss increase. Passing the fixed threshold would only support
a focused causal teacher/provenance investigation; it would not authorize
relabeling, filtering, reweighting, or training. The older policy-target
encoding audit used a different dataset and is not pooled here.

`row-accounting.jsonl.gz` preserves one record per compact eligible row,
including source/line, multiplicity, coefficient, partition, stone bucket,
canonical and exact-input identity, original float32 loader target and target
sum. Replay copies affect exposure weights but are not emitted as extra label
rows. `manifest.json` binds protocol, replay and split bytes, loader contract,
execution sources, and row- and group-accounting bytes. No loaded target has
zero policy coefficient; the maximum target-sum deviation is 1.94e-7, with no
row beyond 1e-5. Results include the seed429 normalized-coefficient treatment
as a secondary diagnostic. Run the portable verifier from any
read-only relocated copy with:

```bash
python -m ml.alphazero_lite.verify_seed432 --directory PATH
```
