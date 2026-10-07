# seed433 — retrospective seed432 census correction

This is an append-only retrospective correction to #432. The original #432
source, manifest, row/group evidence, results, and interpretation are preserved.
No training, inference, search, game, relabeling, filtering, reweighting, or
promotion was performed.

## Corrected rule and decision

Groups are exact little-endian float32 network-input bytes, partitioned before
grouping. Within a group, exposure uses replay multiplicity × policy
coefficient; equal-input uses policy coefficient without replay-copy
multiplicity. Exposure groups are outer-weighted by total exposure mass.
Equal-input groups with positive policy mass each receive exactly one unit of
outer weight, independent of compact-row count. Zero-mass groups are excluded
from the denominator and reported separately; positive-mass singleton groups
are included with JS zero. Policy mass and aggregation denominator are distinct
reported fields. The same outer weighting applies to JS, top-action
disagreement, and both terms of source entropy decomposition.

The fixed thresholds remain JS ≥0.05 nats and top-action disagreement ≥10%.
Only original control exposure and equal-input training >32-stone entries
classify the result. Seed429 treatment is secondary diagnostic evidence.

| Control weighting | #432 mean JS | corrected mean JS | #432 top disagreement | corrected top disagreement |
|---|---:|---:|---:|---:|
| Exposure | 0.0802718178 | 0.0802718178 | 11.1902005% | 11.1902005% |
| Equal-input | 0.0615598109 | 0.0083857543 | 6.7609416% | 1.4203943% |

The classification remains **`target_disagreement_not_material_under_registered_thresholds`**:
the corrected equal-input control entry fails both fixed thresholds. #432
treated equal-input coefficient mass as the outer denominator, so compact-row
rich exact-input groups received more influence. Equal-input aggregation now
gives each positive-mass input one vote. Exposure was already mass-weighted and
is unchanged. Classification is restricted to the two control entries;
treatment metrics cannot alter it.

The policy cross-entropy identity remains limited to a fixed logits vector and
preserved total coefficient: target averaging preserves aggregate cross-entropy
and its gradient in exact arithmetic. It does not establish equivalence under
minibatch or optimizer averaging. Target disagreement alone does not prove
incorrect labels or strength harm. If the corrected gate had passed, the only
recommendation would be the already specified causal teacher/provenance
investigation; it is not launched by this correction.

`corrected-results.json` contains all populations and diagnostics.
`correction-receipt.json` binds the original #432 manifest/results and final
correction sources/results. The verifier is invoked with:

```bash
python -m ml.alphazero_lite.verify_seed433 --directory PATH
```
