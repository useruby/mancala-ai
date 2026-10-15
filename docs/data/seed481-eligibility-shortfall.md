# Seed481 eligibility census: confirmed shortfall

## Decision

Stop before inference. The hash-bound seed466 replay bundle contains 851 distinct
fresh-source validation inputs above 32 active stones. The entire seed466
evaluation cohort is explicitly excluded by the seed481 request; those same 851
fresh identities are therefore all historically evaluated and none can enter a
confirmation cohort. The eligible count is 0, a shortfall of 512 identities
against the preregistered minimum. No historical rows were recycled, and no
labels or model predictions were inspected during this eligibility census.

## Reconstructable census evidence

The independently reconstructed seed466 bundle binds manifest SHA-256
`db80f3bdfcc0a126032c1f5a16e78cda74b160d4fa2d80330cec8bed18085b34`.
Its manifest binds the evaluation arrays, including the ordered
`input_identity` (2,607 occurrences; SHA-256
`0ff1892c88065da73eaf896cccd1e413af2a7ff38a820f8d865ec3144a4ded2b`),
`source_ids` (`4ea3110058c801f038afc134b5e195ef2aa0a4ccb3f4ae2b8facbb8aace83e30`),
and occurrence provenance and targets. The committed bundle description and
manifest establish that this is the complete strict-unseen evaluation
membership (`active_stones > 32`), containing 2,607 occurrences and 1,242
identities in total, of which the fresh source contributes 859 occurrences and
851 identities. These membership counts are not inferred from label values.

The exact same fresh membership was the seed466 evaluation cohort and is named
as historical evidence in seed479's bound registration/evidence and seed480's
bound correction; #479 explicitly describes its existing evaluation cohort as
exploratory rather than fresh confirmation. Seed466 independently reconstructs
from the pinned source recipe and authenticates the committed arrays. The
seed480 independent verifier also passed before this census was interpreted.

| Census stage | Distinct identities | Occurrences |
| --- | ---: | ---: |
| Fresh-source validation, >32 active stones | 851 | 859 |
| Excluded: entire seed466 evaluation cohort | 851 | 859 |
| Remaining before other historical/training exclusions | 0 | 0 |
| Eligible after all required exclusions | 0 | 0 |
| Required minimum | 512 | — |
| Shortfall | 512 | — |

Because the mandated seed466 exclusion alone exhausts the available fresh
validation membership, additional exclusions for seed427–480 probes and exact
float32 training-source/partition overlap cannot increase eligibility. No
target, mask, or prediction was used to select or exclude identities. Occurrence
targets and provenance remain untouched in the historical publication.

## Verification

Commands completed before any model evaluation:

```text
./.venv-azlite/bin/python -m ml.alphazero_lite.verify_seed480_value_mse_correction
seed480 independent verification passed

./.venv-azlite/bin/python -m ml.alphazero_lite.verify_seed466_replay_bundle \
  --bundle docs/data/seed466-replay-input-bundle
seed466 independent reconstruction passed
```

Seed481 inference, bootstrap, metric/gate evaluation, and decision were not
performed. A new data-generation protocol is required for a fresh cohort.
