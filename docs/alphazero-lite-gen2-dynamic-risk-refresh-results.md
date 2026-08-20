# Gen-2 Dynamic Risk-Mask Refresh Results

**Primary classification:** `dynamic_refresh_insufficient`

**Recommended follow-up:** do not add a more complicated detector; reassess the shared-head constraint

## Fixed Contract

- Initial mask: frozen PR #208 q75 retrospective mask.
- Refresh after optimizer steps: `[4, 16, 32]`.
- Every refresh selects the current candidate-vs-P1 top 25% of unique replay states.

## Mask Refreshes

| Boundary | Protected states | Threshold | Mask SHA256 |
| --- | ---: | ---: | --- |
| initial | 5849 | 0.00000000 | cb36a22880c7fb5a1676c96e6e2a5f2b42d8a7562ca98083642862b6fed47d08 |
| 4 | 5849 | 0.00280967 | 789dae32122a6ec9360d3a71b199249b181ddeebf8120887a5823f7b7c5c7be9 |
| 16 | 5849 | 0.00502940 | 50a1723eac43fabb17a84b9038c76a00389d154cd40dbcf114fe75f8ce751b22 |
| 32 | 5849 | 0.00418826 | 2cb2c1f38bcda5e6bda9be6c0d42b3593b2dfbb0ed0f477af4b5bdc536915b13 |

## Learning And Drift

| Lane | Step | CE(search) | Fit fraction | Unprotected improvement | Current q75 L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| static_q75 | 1 | 1.035568 | 0.0578 | 0.000097 | 0.001025 |
| static_q75 | 4 | 1.035238 | 0.1998 | 0.000344 | 0.002810 |
| static_q75 | 16 | 1.034855 | 0.3643 | 0.000694 | 0.004144 |
| static_q75 | 46 | 1.034088 | 0.6935 | 0.001364 | 0.005812 |
| dynamic_q75 | 1 | 1.035568 | 0.0578 | 0.000097 | 0.001025 |
| dynamic_q75 | 4 | 1.035238 | 0.1998 | 0.000344 | 0.002810 |
| dynamic_q75 | 16 | 1.034690 | 0.4350 | 0.000729 | 0.005029 |
| dynamic_q75 | 46 | 1.034197 | 0.6468 | 0.001225 | 0.006056 |

## Arena Vs P1 (384:256)

| Lane | Step | Effect | 95% CI | Safe |
| --- | ---: | ---: | --- | ---: |
| static_q75 | 16 | -0.0098 | [-0.0195, -0.0020] | True |
| static_q75 | 46 | -0.0273 | [-0.0469, -0.0117] | False |
| dynamic_q75 | 16 | -0.0195 | [-0.0391, -0.0039] | False |
| dynamic_q75 | 46 | -0.0254 | [-0.0430, -0.0098] | False |
