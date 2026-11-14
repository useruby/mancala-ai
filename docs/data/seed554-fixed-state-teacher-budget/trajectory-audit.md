# Seed 554 trajectory audit: C128 vs T512

## Integrity and scope

The trace SHA-256 values match `run-summary.json`: C128
`90c75620ec3d2f6d414f7557d84d9b8d2ed4a99de804acb121cde56aafb1ed8f`; T512
`c064b7bace5bd24ef21028ab04d354d1df28e849b77169c7715820381adae0a6`. All eight
checkpoint files at updates 027, 054, 081, and 108 also match the hashes in that
summary. The derivative verifier records 13,818 matched occurrences and 11,270
paired searches with search replay performed. The existing physical-copy
verification records the artifact snapshot and read-only verifier. The bounded
input authentication explicitly does not claim that the historical #549 full
runtime verifier passed.

## Paired trace observations

Means below are per-update over each 27-update segment (overall is 108 updates).
Clipping counts are updates with clipping active; the recorded post-clip norm is
1.0 on clipped updates.

| Updates | Arm | Policy loss | Value loss | Pre-clip grad norm | Clipped |
| --- | --- | ---: | ---: | ---: | ---: |
| 1–27 | C128 | 0.79780 | 0.16904 | 1.13731 | 14/27 |
|  | T512 | 0.83470 | 0.17028 | 1.30525 | 24/27 |
| 28–54 | C128 | 0.76998 | 0.15566 | 0.93234 | 6/27 |
|  | T512 | 0.78970 | 0.15715 | 1.06526 | 18/27 |
| 55–81 | C128 | 0.75914 | 0.15005 | 0.90880 | 6/27 |
|  | T512 | 0.77371 | 0.15114 | 0.99844 | 14/27 |
| 82–108 | C128 | 0.75117 | 0.14559 | 0.95821 | 9/27 |
|  | T512 | 0.76346 | 0.14639 | 1.03547 | 16/27 |
| 1–108 | C128 | 0.76952 | 0.15509 | 0.98416 | 35/108 |
|  | T512 | 0.79039 | 0.15624 | 1.10110 | 72/108 |

Trace optimizer telemetry supplies before/after optimizer-state hashes and raw
and clipped gradient hashes, not interpretable Adam moment magnitudes. Batch
identity is shared update by update. C128 also matched the production training
primitive in the recorded equivalence gate for all updates.

## Checkpoint parameter comparison

Norms are global L2 norms over listed parameter tensors; difference is the
global L2 norm of T512 minus C128. Shared trunk includes the input and residual
blocks (not the separate policy/value hidden layers).

| Update | Policy C/T | Policy diff | Shared trunk C/T | Trunk diff | Value C/T | Value diff | All-param diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 27 | 4.19336 / 4.19055 | 0.01447 | 27.46382 / 27.46113 | 0.10747 | 2.87070 / 2.87076 | 0.00069 | 0.12077 |
| 54 | 4.19737 / 4.19348 | 0.02297 | 27.46962 / 27.46680 | 0.16585 | 2.87613 / 2.87595 | 0.00158 | 0.18754 |
| 81 | 4.20082 / 4.19684 | 0.02946 | 27.47361 / 27.47137 | 0.21765 | 2.88105 / 2.88058 | 0.00255 | 0.24591 |
| 108 | 4.20469 / 4.20111 | 0.03585 | 27.47864 / 27.47686 | 0.26197 | 2.88550 / 2.88511 | 0.00318 | 0.29775 |

Biases are included in their respective head norms. Separate hidden layers are
not included in the three requested grouped norms; all-param difference spans
every checkpoint tensor. These parameter norms do not measure policy entropy.

## Findings, interpretation, and limitation

The arms already differ by update 27, and the parameter distance grows at every
checkpoint through update 108. The traces likewise show persistently higher
T512 policy loss, pre-clip gradient norm, and clipping frequency in every
segment. Thus the observed divergence accumulated through training; it did not
first appear near the fixed endpoint. The largest checkpoint parameter-distance
component is the shared trunk, while value-head differences remain small.

More frequent T512 clipping is a plausible modifier of its trajectory: 72/108
updates clipped versus 35/108 for C128. This is an association, not evidence
that clipping caused the endpoint difference; both runs use the same clipping
rule and losses/gradients also differ. Shared-trunk interference is also
plausible because policy-label changes feed shared parameters and trunk
differences dominate the grouped parameter distance. These artifacts do not
isolate policy-vs-value gradient contributions or establish interference as a
cause. The endpoint development metrics in `results.md` are fit diagnostics
against fixed existing targets, not evidence of target quality or playing
strength.

**Recommended next action:** keep T512 diagnostic-only and, if the teacher-budget
effect remains decision-relevant, run a preregistered independent-seed paired
replication with the same schedule and telemetry; do not interpret this single
pair as causal evidence for clipping or shared-trunk interference.
