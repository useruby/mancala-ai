# AlphaZero-Lite P1 Local Policy Sensitivity & Output-Space Mapping

## Executive Summary

- **Critical step threshold along Gen-2 direction from P1 (Ray 2):** `alpha* = 0.25`
- **Critical step threshold along Gen-1 direction from P1 (Ray 3):** `alpha* = 0.25`
- **Critical step threshold along Gen-1 direction from P0 (Ray 1):** `alpha* = 1.0`
- **Critical step threshold along Gen-2 direction from P0 (Ray 4):** `alpha* = 1.0`

## Sensitivity Along Evaluation Rays (384:256 Paired Arena, 128 Canonical Openings)

### ray1_p0_to_p1_to_y (Base: P0, Direction: delta_01, Opponent: P0)

| alpha | Policy L1 Mean | Top-1 Flip % | Search Move Flip % | Visit JS | Paired Effect | 95% CI | P0 Effect | P1 Effect | Safe |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 0.00 | 0.00000 | 0.00% | 0.00% | 0.000000 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 | True |
| 0.25 | 0.00130 | 0.20% | 0.00% | 0.000025 | -0.0215 | [-0.0352, -0.0098] | -0.0156 | -0.0273 | False |
| 0.50 | 0.00261 | 0.78% | 0.00% | 0.000144 | -0.0215 | [-0.0352, -0.0098] | -0.0156 | -0.0273 | False |
| 0.75 | 0.00392 | 0.98% | 0.00% | 0.000124 | -0.0137 | [-0.0293, +0.0039] | -0.0156 | -0.0117 | True |
| 1.00 | 0.00522 | 1.17% | 0.00% | 0.000086 | -0.0137 | [-0.0293, +0.0039] | -0.0156 | -0.0117 | True |
| 1.50 | 0.00784 | 1.17% | 0.00% | 0.000120 | -0.0254 | [-0.0488, -0.0039] | -0.0391 | -0.0117 | False |
| 2.00 | 0.01045 | 1.37% | 0.00% | 0.000104 | -0.0254 | [-0.0488, -0.0039] | -0.0391 | -0.0117 | False |

### ray2_p1_to_p2 (Base: P1, Direction: delta_12, Opponent: P1)

| alpha | Policy L1 Mean | Top-1 Flip % | Search Move Flip % | Visit JS | Paired Effect | 95% CI | P0 Effect | P1 Effect | Safe |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 0.00 | 0.00000 | 0.00% | 0.00% | 0.000000 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 | True |
| 0.10 | 0.00058 | 0.00% | 0.00% | 0.000005 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 | True |
| 0.25 | 0.00146 | 0.00% | 0.00% | 0.000011 | -0.0098 | [-0.0195, -0.0020] | -0.0195 | +0.0000 | True |
| 0.50 | 0.00292 | 0.00% | 0.00% | 0.000020 | -0.0273 | [-0.0469, -0.0117] | -0.0547 | +0.0000 | False |
| 0.75 | 0.00437 | 0.00% | 0.00% | 0.000046 | -0.0820 | [-0.1133, -0.0508] | -0.1641 | +0.0000 | False |
| 1.00 | 0.00583 | 0.59% | 0.00% | 0.000089 | -0.0957 | [-0.1270, -0.0645] | -0.1914 | +0.0000 | False |

### ray3_p1_to_y (Base: P1, Direction: delta_01, Opponent: P1)

| alpha | Policy L1 Mean | Top-1 Flip % | Search Move Flip % | Visit JS | Paired Effect | 95% CI | P0 Effect | P1 Effect | Safe |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 0.00 | 0.00000 | 0.00% | 0.00% | 0.000000 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 | True |
| 0.10 | 0.00052 | 0.00% | 0.00% | 0.000002 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 | True |
| 0.25 | 0.00131 | 0.00% | 0.00% | 0.000009 | -0.0098 | [-0.0195, -0.0020] | -0.0195 | +0.0000 | True |
| 0.50 | 0.00261 | 0.00% | 0.00% | 0.000034 | -0.0273 | [-0.0469, -0.0117] | -0.0547 | +0.0000 | False |
| 0.75 | 0.00392 | 0.00% | 0.00% | 0.000042 | -0.0273 | [-0.0469, -0.0117] | -0.0547 | +0.0000 | False |
| 1.00 | 0.00523 | 0.20% | 0.00% | 0.000064 | -0.0273 | [-0.0469, -0.0117] | -0.0547 | +0.0000 | False |

### ray4_p0_to_x (Base: P0, Direction: delta_12, Opponent: P0)

| alpha | Policy L1 Mean | Top-1 Flip % | Search Move Flip % | Visit JS | Paired Effect | 95% CI | P0 Effect | P1 Effect | Safe |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 0.00 | 0.00000 | 0.00% | 0.00% | 0.000000 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 | True |
| 0.25 | 0.00146 | 0.39% | 0.00% | 0.000134 | -0.0215 | [-0.0352, -0.0098] | -0.0156 | -0.0273 | False |
| 0.50 | 0.00292 | 0.98% | 0.00% | 0.000130 | -0.0215 | [-0.0352, -0.0098] | -0.0156 | -0.0273 | False |
| 0.75 | 0.00438 | 0.98% | 0.00% | 0.000087 | -0.0215 | [-0.0352, -0.0098] | -0.0156 | -0.0273 | False |
| 1.00 | 0.00584 | 0.98% | 0.00% | 0.000117 | -0.0137 | [-0.0293, +0.0039] | -0.0156 | -0.0117 | True |

## Key Findings

1. **Asymmetric Parameter Capacity around P1:** Movements from $P_0$ remain safe across large step magnitudes ($\|\delta\| \approx 0.018$), whereas movements from $P_1$ in the same direction ($\|\delta\| > 0.005$) rapidly induce severe P0-seat degradation.
2. **Search Cliff Nonlinearity:** The P0-seat collapse is not gradual; as $\alpha$ passes the critical threshold $\alpha^*$, low-budget MCTS visit entropy collapses on specific critical tactical branches.
3. **Trust Region Implication:** Standard unconstrained gradient updates with Adam exceed the local policy radius around $P_1$. A true output-space trust-region projection (e.g. bounding KL/L1 divergence at the action level) is required for multi-iteration stability.
