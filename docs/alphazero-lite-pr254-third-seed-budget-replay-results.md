# PR #254 Third Seed Budget Replay

**Classification:** `invariant_failure`

The seed-47 generation and isolated three-lane training completed, but no
primary strength result is reported or promoted.

## Frozen Inputs

- A16 step-16 snapshot: `f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff`
- Initial Adam state: `61d5719e75aae87d7c2ca7ed2c5b01871ac2ea1675a34c4a6c918c783894e8c7`
- P1 checkpoint: `e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9`
- Target evaluator: `8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34`

## Failure

The first sealed M/N/O manifest included canonical and A-I but incorrectly
omitted consumed J/K/L from its exclusion set. A subsequent overlap audit
found the following canonical-opening and prefix overlaps with J/K/L:

| Suite | Overlaps |
| --- | ---: |
| M | 13 |
| N | 14 |
| O | 22 |

M and N arena evaluation had already started before this defect was detected.
They cannot be used for primary classification, and replacement M/N/O suites
would violate the preregistered consumed-suite contract. Consequently this
experiment does not answer the 768-versus-1024 strength question and does not
run conditional semantic or gradient attribution analysis.

The runner now includes J/K/L in both the consumed-state and prefix exclusion
sets before suite selection, so a future independently preregistered run can
enforce the intended contract from the outset.
