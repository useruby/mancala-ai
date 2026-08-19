# AlphaZero-Lite Generation-2 Self-Play Anchor Results

**Classification:** `second_iteration_regression`

**Recommended Next Action:** `compare generation-1 and generation-2 search/parent disagreement tails and per-depth drift before tuning beta (NOT implemented in this PR)`

## Core Guardrails & Invariants

- P1 reconstructed and verified: `True`
- all lanes start from identical P1: `True`
- trunk byte-identical to P1 (all lanes): `True`
- trunk byte-identical to P0 (all lanes): `True`
- value stack byte-identical to P1 (all lanes): `True`
- value stack byte-identical to P0 (all lanes): `True`
- beta_100 P1-equivalent: `True` (drift = 3.69e-05, tol = 0.001)
- beta_100 initial policy gradient norm: `2.50e-08`
- checkpoint steps: `[1, 4, 16, 46]`
- P0 weights sha256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- P1 weights sha256: `77969733ece5ced92d3a143a0fe9d82863ca3ec4faa477470ff5826ac22e4e12`
- P1 checkpoint npz sha256: `e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9`
- Gen-1 replay sha256: `35ac7ce9f9d596ff6a9dad27a9d2ea7c0633c7d3b84860eeeaf6e1ee78fac077`
- Gen-2 replay sha256: `2cee30547f8bc5d7cad6f02f859ee5e8644386e9b59c8a054ef74548c72ce84b`
- Gen-2 seed: `43`
- optimizer: `{"type": "Adam", "lr": 1e-05, "weight_decay": 0.0}`
- gradient clip: `1.0`
- trainable scope: `policy_head`

## Dataset Evolution Diagnostics (Gen-2 P1 Replay vs Gen-1 P0 Replay)

| Metric | Gen-2 (P1 Parent) | Gen-1 (P0 Parent) | Delta (Gen-2 - Gen-1) |
| --- | ---: | ---: | ---: |
| Games | 700 | 700 | +0 |
| Total Positions | 28029 | 28146 | -117 |
| Game Length (mean) | 40.04 | 40.21 | -0.17 |
| Game Length (p50 / p90) | 41.0 / 51.0 | 41.0 / 52.0 | - |
| Player 0 Fraction | 0.5195 | 0.5176 | - |
| Search Policy Entropy (mean) | 0.3325 | 0.3300 | +0.0026 |
| Parent Policy Entropy (mean) | 0.9413 | 0.9354 | +0.0059 |
| Legal L1(search, parent) (mean) | 0.8628 | 0.8690 | -0.0062 |
| Legal L1 (p50 / p90 / p95 / p99) | 0.8224 / 1.6573 / 1.7757 / 1.9156 | 0.8320 / 1.6676 / 1.7861 / 1.9194 | - |
| Legal JS Divergence (mean) | 0.1929 | 0.1951 | -0.0022 |
| Legal JS (p50 / p90 / p95 / p99) | 0.1511 / 0.4470 / 0.5099 / 0.6045 | 0.1535 / 0.4515 / 0.5155 / 0.6061 | - |
| Top-1 Disagreement Rate | 0.4040 | 0.4087 | -0.0046 |

## Training & Target Metrics (Gen-2 Validation Probe)

| Lane | Step | CE(search) | CE(P1) | CE(mixed) | Search-CE Improv vs P1 | Fit Fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| beta_000 | 1 | 1.0507 | 0.9303 | 1.0507 | +0.0001 | 0.0190 |
| beta_000 | 4 | 1.0502 | 0.9303 | 1.0502 | +0.0005 | 0.1020 |
| beta_000 | 16 | 1.0488 | 0.9303 | 1.0488 | +0.0020 | 0.3716 |
| beta_000 | 46 | 1.0455 | 0.9307 | 1.0455 | +0.0053 | 1.0000 |
| beta_095 | 1 | 1.0507 | 0.9303 | 0.9363 | +0.0001 | 0.0190 |
| beta_095 | 4 | 1.0503 | 0.9303 | 0.9363 | +0.0005 | 0.1003 |
| beta_095 | 16 | 1.0494 | 0.9303 | 0.9362 | +0.0014 | 0.2653 |
| beta_095 | 46 | 1.0484 | 0.9303 | 0.9362 | +0.0023 | 0.4425 |
| beta_100 | 1 | 1.0508 | 0.9303 | 0.9303 | +0.0000 | 0.0004 |
| beta_100 | 4 | 1.0508 | 0.9303 | 0.9303 | +0.0000 | 0.0065 |
| beta_100 | 16 | 1.0508 | 0.9303 | 0.9303 | +0.0000 | 0.0062 |
| beta_100 | 46 | 1.0508 | 0.9303 | 0.9303 | +0.0000 | 0.0008 |

## Policy Drift vs P1 (Parent Reference)

| Lane | Step | L1 mean | L1 max | L1 p50 | L1 p90 | L1 p95 | L1 p99 | JS mean | Top-1 Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| beta_000 | 1 | 0.000775 | 0.002002 | 0.000815 | 0.001289 | 0.001415 | 0.001714 | 0.000000 | 0.0000 |
| beta_000 | 4 | 0.002552 | 0.007464 | 0.002669 | 0.004479 | 0.004785 | 0.005790 | 0.000001 | 0.0020 |
| beta_000 | 16 | 0.008494 | 0.022416 | 0.009128 | 0.015133 | 0.016690 | 0.019841 | 0.000016 | 0.0059 |
| beta_000 | 46 | 0.022472 | 0.062129 | 0.023888 | 0.038724 | 0.042803 | 0.053439 | 0.000112 | 0.0156 |
| beta_095 | 1 | 0.000775 | 0.002003 | 0.000815 | 0.001289 | 0.001415 | 0.001714 | 0.000000 | 0.0000 |
| beta_095 | 4 | 0.002396 | 0.007107 | 0.002494 | 0.004179 | 0.004522 | 0.005485 | 0.000001 | 0.0020 |
| beta_095 | 16 | 0.004972 | 0.015280 | 0.005175 | 0.008619 | 0.009653 | 0.012427 | 0.000006 | 0.0020 |
| beta_095 | 46 | 0.005955 | 0.021778 | 0.005583 | 0.011022 | 0.012762 | 0.017377 | 0.000009 | 0.0020 |
| beta_100 | 1 | 0.000021 | 0.000059 | 0.000020 | 0.000038 | 0.000043 | 0.000051 | 0.000000 | 0.0000 |
| beta_100 | 4 | 0.000392 | 0.001055 | 0.000392 | 0.000699 | 0.000775 | 0.000897 | 0.000000 | 0.0020 |
| beta_100 | 16 | 0.000179 | 0.000565 | 0.000174 | 0.000322 | 0.000361 | 0.000455 | 0.000000 | 0.0000 |
| beta_100 | 46 | 0.000035 | 0.000150 | 0.000034 | 0.000064 | 0.000071 | 0.000090 | 0.000000 | 0.0000 |

## Cumulative Policy Drift vs P0 (Original Incumbent)

| Lane | Step | L1 mean | L1 max | L1 p50 | L1 p90 | L1 p95 | L1 p99 | JS mean | Top-1 Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| beta_000 | 1 | 0.005871 | 0.023007 | 0.005866 | 0.010906 | 0.012251 | 0.015584 | 0.000008 | 0.0059 |
| beta_000 | 4 | 0.007410 | 0.028536 | 0.007581 | 0.013236 | 0.014978 | 0.019955 | 0.000013 | 0.0078 |
| beta_000 | 16 | 0.013191 | 0.042600 | 0.013710 | 0.022970 | 0.025838 | 0.031994 | 0.000039 | 0.0117 |
| beta_000 | 46 | 0.027069 | 0.080358 | 0.028542 | 0.046333 | 0.051150 | 0.065432 | 0.000162 | 0.0215 |
| beta_095 | 1 | 0.005871 | 0.023007 | 0.005866 | 0.010906 | 0.012251 | 0.015584 | 0.000008 | 0.0059 |
| beta_095 | 4 | 0.007324 | 0.028179 | 0.007562 | 0.013151 | 0.014780 | 0.019577 | 0.000013 | 0.0078 |
| beta_095 | 16 | 0.010038 | 0.033970 | 0.010141 | 0.017954 | 0.020073 | 0.026137 | 0.000023 | 0.0078 |
| beta_095 | 46 | 0.011311 | 0.042850 | 0.010980 | 0.020863 | 0.024270 | 0.031347 | 0.000032 | 0.0078 |
| beta_100 | 1 | 0.005532 | 0.021053 | 0.005479 | 0.010153 | 0.011840 | 0.014462 | 0.000007 | 0.0059 |
| beta_100 | 4 | 0.005548 | 0.020840 | 0.005312 | 0.010079 | 0.011957 | 0.014858 | 0.000008 | 0.0039 |
| beta_100 | 16 | 0.005576 | 0.020885 | 0.005377 | 0.010179 | 0.012040 | 0.014884 | 0.000008 | 0.0059 |
| beta_100 | 46 | 0.005537 | 0.021017 | 0.005458 | 0.010145 | 0.011870 | 0.014539 | 0.000007 | 0.0059 |

## Parameter Drift (Relative L2 Drift vs P1 and vs P0)

| Lane | Step | Trunk (vs P1) | Policy Head (vs P1) | Value (vs P1) | Trunk (vs P0) | Policy Head (vs P0) | Value (vs P0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| beta_000 | 1 | 0.000000 | 0.000085 | 0.000000 | 0.000000 | 0.001768 | 0.000000 |
| beta_000 | 4 | 0.000000 | 0.000263 | 0.000000 | 0.000000 | 0.001882 | 0.000000 |
| beta_000 | 16 | 0.000000 | 0.000899 | 0.000000 | 0.000000 | 0.002404 | 0.000000 |
| beta_000 | 46 | 0.000000 | 0.002439 | 0.000000 | 0.000000 | 0.003850 | 0.000000 |
| beta_095 | 1 | 0.000000 | 0.000084 | 0.000000 | 0.000000 | 0.001768 | 0.000000 |
| beta_095 | 4 | 0.000000 | 0.000253 | 0.000000 | 0.000000 | 0.001882 | 0.000000 |
| beta_095 | 16 | 0.000000 | 0.000690 | 0.000000 | 0.000000 | 0.002334 | 0.000000 |
| beta_095 | 46 | 0.000000 | 0.001621 | 0.000000 | 0.000000 | 0.003320 | 0.000000 |
| beta_100 | 1 | 0.000000 | 0.000002 | 0.000000 | 0.000000 | 0.001733 | 0.000000 |
| beta_100 | 4 | 0.000000 | 0.000046 | 0.000000 | 0.000000 | 0.001730 | 0.000000 |
| beta_100 | 16 | 0.000000 | 0.000045 | 0.000000 | 0.000000 | 0.001732 | 0.000000 |
| beta_100 | 46 | 0.000000 | 0.000037 | 0.000000 | 0.000000 | 0.001734 | 0.000000 |

## Search Diagnostics (384:256 context, candidate vs P1)

| Lane | Step | Move Change Rate | Visit JS | Q-Rank Change | Root-Value Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| beta_000 | 1 | 0.0078 | 0.0000 | +0.0195 | -0.0000 |
| beta_000 | 4 | 0.0156 | 0.0002 | +0.0352 | +0.0001 |
| beta_000 | 16 | 0.0234 | 0.0010 | +0.0273 | -0.0002 |
| beta_000 | 46 | 0.0430 | 0.0029 | +0.0547 | -0.0000 |
| beta_095 | 1 | 0.0078 | 0.0000 | +0.0195 | -0.0000 |
| beta_095 | 4 | 0.0156 | 0.0002 | +0.0430 | +0.0001 |
| beta_095 | 16 | 0.0234 | 0.0008 | +0.0430 | -0.0000 |
| beta_095 | 46 | 0.0195 | 0.0009 | +0.0234 | -0.0001 |
| beta_100 | 1 | 0.0000 | 0.0000 | +0.0000 | -0.0000 |
| beta_100 | 4 | 0.0000 | 0.0000 | +0.0078 | -0.0001 |
| beta_100 | 16 | 0.0039 | 0.0000 | +0.0000 | -0.0000 |
| beta_100 | 46 | 0.0000 | 0.0000 | +0.0000 | -0.0000 |

## Cumulative Search Diagnostics (384:256 context, candidate vs P0)

| Lane | Step | Move Change Rate | Visit JS | Q-Rank Change | Root-Value Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| beta_095 | 46 | 0.0273 | 0.0014 | +0.0039 | +0.0003 |

## Per-Depth Policy L1/JS on Expanded Probe States (384:256, candidate vs P1)

| Lane | Step | Depth | Expanded Nodes | L1 Mean | JS Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| beta_095 | 16 | 0 | 256 | 0.004986 | 0.000006 |
| beta_095 | 16 | 1 | 929 | 0.005090 | 0.000006 |
| beta_095 | 16 | 2 | 2870 | 0.005211 | 0.000006 |
| beta_095 | 16 | 3 | 6713 | 0.005162 | 0.000005 |
| beta_095 | 16 | 4 | 62716 | 0.004754 | 0.000005 |
| beta_095 | 46 | 0 | 256 | 0.006081 | 0.000009 |
| beta_095 | 46 | 1 | 929 | 0.006153 | 0.000009 |
| beta_095 | 46 | 2 | 2870 | 0.006224 | 0.000009 |
| beta_095 | 46 | 3 | 6713 | 0.006068 | 0.000008 |
| beta_095 | 46 | 4 | 62716 | 0.005605 | 0.000008 |

## Per-Depth Policy L1/JS on Expanded Probe States (384:256, candidate vs P0)

| Lane | Step | Depth | Expanded Nodes | L1 Mean | JS Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| beta_095 | 46 | 0 | 256 | 0.011399 | 0.000033 |
| beta_095 | 46 | 1 | 928 | 0.011434 | 0.000032 |
| beta_095 | 46 | 2 | 2864 | 0.011682 | 0.000031 |
| beta_095 | 46 | 3 | 6685 | 0.011436 | 0.000030 |
| beta_095 | 46 | 4 | 62814 | 0.010529 | 0.000028 |

## Canonical Paired Arena Matrix

### 1. P1 vs P0 Reproduction (PR #203 Baseline)

| Match | Context | Paired Effect | 95% CI | P0 Effect | P1 Effect | W/D/L |
| --- | --- | ---: | --- | ---: | ---: | --- |
| P1 vs P0 | 384:256 | -0.0137 | [-0.0293, +0.0039] | -0.0156 | -0.0117 | 392/38/82 |
| P1 vs P0 | 1200:1200 | +0.0234 | [+0.0078, +0.0430] | +0.0273 | +0.0195 | 204/128/180 |

### 2. Generation-2 Candidates vs P1 (Parent)

| Lane | Step | Context | Paired Effect | 95% CI | P0 Effect | P1 Effect | W/D/L |
| --- | ---: | --- | ---: | --- | ---: | ---: | --- |
| beta_000 | 46 | 384:256 | -0.1719 | [-0.2148, -0.1327] | -0.3438 | +0.0000 | 312/38/162 |
| beta_095 | 16 | 384:256 | -0.0742 | [-0.1055, -0.0430] | -0.1484 | +0.0000 | 362/38/112 |
| beta_095 | 46 | 384:256 | -0.0957 | [-0.1270, -0.0645] | -0.1914 | +0.0000 | 354/32/126 |
| beta_100 | 46 | 384:256 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 | 400/38/74 |

### 3. Generation-2 Candidates vs P0 (Direct Cumulative Measurement)

| Lane | Step | Context | Paired Effect | 95% CI | P0 Effect | P1 Effect | W/D/L |
| --- | ---: | --- | ---: | --- | ---: | ---: | --- |
| beta_095 | 46 | 384:256 | -0.0938 | [-0.1328, -0.0586] | -0.1758 | -0.0117 | 354/32/126 |

## Classification Evidence

| Signal | Value |
| --- | ---: |
| beta_100_p1_equivalent | True |
| beta_000_p1_step46_384_256_effect | -0.1719 |
| beta_000_p1_step46_384_256_safe | False |
| beta_095_p1_step46_fit_fraction | +0.4425 |
| beta_095_p1_step46_meaningful_fit | True |
| beta_095_p1_step46_384_256_effect | -0.0957 |
| beta_095_p1_step46_384_256_safe | False |
| beta_095_p1_step46_1200_1200_effect | None |
| beta_095_p1_step46_1200_1200_safe | False |
| beta_095_p1_step46_1200_1200_gain | False |
| beta_095_p0_step46_384_256_effect | -0.0938 |
| beta_095_p0_step46_384_256_safe | False |
| beta_095_p0_step46_1200_1200_effect | None |
| beta_095_p0_step46_1200_1200_safe | False |
| noninferiority_lower | -0.0300 |
| meaningful_fit_fraction | +0.2500 |

## Exact Reproduction Commands

```bash
.venv/bin/python ml/alphazero_lite/run_gen2_selfplay_anchor_iteration.py \
  --workdir /tmp/azlite_gen2_selfplay_anchor \
  --games 700 \
  --seed 43 \
  --arena-workers 24
```

Full JSON evidence: `docs/data/alphazero-lite-gen2-selfplay-anchor-summary.json`.
