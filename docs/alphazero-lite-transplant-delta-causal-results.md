# AlphaZero-Lite PR #204 Second-Iteration Regression Causal Decomposition

## Executive Summary

- Primary classification: `p1_local_policy_fragility`
- Interpretation: P1 lies in a locally fragile policy region where another policy-head movement in the previously-safe direction pushes the model across a low-budget search boundary, with failure concentrated entirely in the P0 seat.
- Recommended next experiment: measure explicit output-space/search sensitivity around P1 before designing a true projection/trust-region mechanism

## Lineage and Transplant Invariants

| Checkpoint | State Hash | npz SHA-256 | weights.json SHA-256 | Trunk vs P0 | Value Stack vs P0 |
| --- | --- | --- | --- | --- | --- |
| P0 (incumbent) | `d265537d6b63...` | `materialized...` | `8d70e90a684c...` | exact (bit-for-bit) | exact (bit-for-bit) |
| P1 (PR #203 beta_095 step46) | `a86acb54b97c...` | `e4a9c95302c1...` | `77969733ece5...` | exact (bit-for-bit) | exact (bit-for-bit) |
| P2 (PR #204 beta_095 step46) | `336496d5fb33...` | `c1165cf81d1d...` | `7fff5640965a...` | exact (bit-for-bit) | exact (bit-for-bit) |
| control_p1 (P0 + delta_01) | `a86acb54b97c...` | exact P1 match | exact P1 match | exact (bit-for-bit) | exact (bit-for-bit) |
| control_p2 (P1 + delta_12) | `336496d5fb33...` | exact P2 match | exact P2 match | exact (bit-for-bit) | exact (bit-for-bit) |
| X (P0 + delta_12) | `01856e9ac36b...` | `82625f2aa4ba...` | `980875f8a351...` | exact (bit-for-bit) | exact (bit-for-bit) |
| Y (P1 + delta_01) | `73b45f34a3dc...` | `de994358eef9...` | `e809f218f41d...` | exact (bit-for-bit) | exact (bit-for-bit) |

## Policy-Head Delta Alignment

- `||delta_01||_2`: `1.873338e-02`
- `||delta_12||_2`: `1.751139e-02`
- Cosine similarity `cos(delta_01, delta_12)`: `0.960198`
- Projection `proj(delta_12 on delta_01)`: `0.897564`
- Orthogonal residual norm: `4.891285e-03` (27.93% of delta_12)

| Layer | ||delta_01|| | ||delta_12|| | Cosine Similarity |
| --- | ---: | ---: | ---: |
| policy_hidden_layer.weight | 1.816009e-02 | 1.697959e-02 | 0.959971 |
| policy_hidden_layer.bias | 1.632830e-03 | 1.567579e-03 | 0.962444 |
| policy_head.weight | 4.286657e-03 | 3.969821e-03 | 0.964046 |
| policy_head.bias | 3.302358e-04 | 3.540613e-04 | 0.972526 |

## Paired Arena Matches (384:256 context, 128 openings, seat swapped)

| Match | Role | Paired Effect | 95% Bootstrap CI | P0 Seat Effect | P1 Seat Effect | W/D/L | Safe (lower >= -0.030) |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| P1 vs P0 (control) | p1_minus_p0 | -0.0137 | [-0.0293, +0.0039] | -0.0156 | -0.0117 | 392/38/82 | True |
| P2 vs P1 (control) | p2_minus_p1 | -0.0957 | [-0.1270, -0.0645] | -0.1914 | +0.0000 | 354/32/126 | False |
| X vs P0 (Gen-2 delta on P0) | x_minus_p0 | -0.0137 | [-0.0293, +0.0039] | -0.0156 | -0.0117 | 392/38/82 | True |
| Y vs P1 (Gen-1 delta on P1) | y_minus_p1 | -0.0273 | [-0.0469, -0.0117] | -0.0547 | +0.0000 | 382/46/84 | False |

## Opening Failure Attribution (P2 vs P1, 128 Canonical Openings)

- Total canonical openings: 128
- P0-seat losing openings: 30 (indices: `[1, 7, 8, 11, 18, 29, 32, 33, 36, 37, 41, 42, 43, 49, 60, 71, 80, 87, 94, 95, 96, 101, 103, 110, 114, 117, 123, 124, 126, 127]`)
- P0-seat unchanged openings: 98
- P0-seat winning openings: 0
- P1-seat effect across all openings: 0.0000 (0 net wins/losses)

| Metric | Losing Openings (N=30) | Unchanged Openings (N=98) | All Openings (N=128) |
| --- | ---: | ---: | ---: |
| Policy L1 Mean | 0.013549 | 0.011023 | 0.011615 |
| Policy JS Mean | 0.000025 | 0.000018 | 0.000020 |
| Largest Action Prob Delta | 0.006315 | 0.004591 | 0.004995 |
| Parent Prob of Max Action | 0.544868 | 0.499119 | 0.509841 |
| Prior Top-1 Differ Rate | 0.0000 | 0.0000 | 0.0000 |
| Selected Move Changed Rate | 0.0000 | 0.0000 | 0.0000 |
| Visit JS Mean | 0.000051 | 0.000101 | 0.000089 |

## Methodological Note: Adam Scale Invariance and Scalar Beta

For same-state anchor mixing $p_\beta(x) = (1 - \beta) p_{\text{search}}(x) + \beta p_{\text{inc}}(x)$:
- At $\beta = 0.95$, the unnormalized gradient at the initialization point is $g_{0.95} = 0.05 \cdot g_{0.00}$.
- However, the Adam optimizer updates parameters via $m_t / (\sqrt{v_t} + \epsilon)$, which scales invariantly to constant gradient multipliers after the first few steps.
- In both PR #203 and PR #204, the step-1 policy head $L_1$ drift is nearly identical across $\beta=0.00$ and $\beta=0.95$ (~`8.45e-5`).
- Therefore, scalar $\beta$ does not act as a true trust-region step-size limiter under Adam; rather, it shifts the eventual fixed point while allowing comparable per-step parameter movements.

## Classification and Recommended Next Experiment

**Classification:** `p1_local_policy_fragility`

**Evidence:**
1. Delta alignment: `cos(delta_01, delta_12) = 0.9602` (Gen-2 repeats 96.0% of Gen-1 direction).
2. Delta safety on P0: `X (P0 + delta_12)` vs P0 achieves effect `-0.0137` (95% CI [-0.0293, +0.0039]), identical to P1 vs P0.
3. Non-composability on P1: `Y (P1 + delta_01)` vs P1 regresses with effect `-0.0273` (95% CI [-0.0469, -0.0117]), matching the P0-seat failure signature (`p0_effect = -0.0547`).
4. Compounding regression: `P2 (P1 + delta_12)` vs P1 regresses severely (`-0.0957`).

**Recommended Next Experiment:** measure explicit output-space/search sensitivity around P1 before designing a true projection/trust-region mechanism
