# AlphaZero-Lite Fresh Self-Play Anchor Validation Results

**Classification:** `fresh_safe_window_replicated`

**Recommended Next Action:** `test the winning beta in a second AlphaZero-style iteration: promote beta_095 candidate inside experiment workspace, generate new self-play from it, train with same-state beta_095 anchoring to parent, and arena candidate vs parent and original incumbent (NOT implemented in this PR)`

## Core Guardrails & Invariants

- all lanes start from identical incumbent: `True`
- trunk byte-identical to incumbent (all lanes): `True`
- value stack byte-identical to incumbent (all lanes): `True`
- beta_100 incumbent-equivalent: `True` (drift = 4.11e-05, tol = 0.001)
- beta_100 initial policy gradient norm: `1.97e-08`
- checkpoint steps: `[1, 4, 16, 46]`
- incumbent weights sha256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- fresh replay sha256: `35ac7ce9f9d596ff6a9dad27a9d2ea7c0633c7d3b84860eeeaf6e1ee78fac077`
- seed: `42`
- optimizer: `{"type": "Adam", "lr": 1e-05, "weight_decay": 0.0}`
- gradient clip: `1.0`
- trainable scope: `policy_head`

## Dataset-Shift Diagnostics (Fresh Replay vs PR #202 Historical Replay)

| Metric | Fresh Replay | PR #202 Historical Replay | Delta (Fresh - Hist) |
| --- | ---: | ---: | ---: |
| Games | 700 | 700 | +0 |
| Total Positions | 28146 | 27538 | +608 |
| Game Length (mean) | 40.21 | 39.34 | +0.87 |
| Game Length (p50 / p90) | 41.0 / 52.0 | 40.5 / 51.0 | - |
| Player 0 Fraction | 0.5176 | 0.5174 | +0.0002 |
| Search Policy Entropy (mean) | 0.3300 | 0.4250 | -0.0951 |
| Incumbent Policy Entropy (mean) | 0.9354 | 0.9421 | -0.0067 |
| Legal L1 (mean) | 0.8690 | 0.8131 | +0.0559 |
| Legal L1 (p50 / p90 / p95 / p99) | 0.8320 / 1.6676 / 1.7861 / 1.9194 | 0.7374 / 1.6135 / 1.7462 / 1.8970 | - |
| Legal JS Divergence (mean) | 0.1951 | 0.1743 | +0.0208 |
| Legal JS (p50 / p90 / p95 / p99) | 0.1535 / 0.4515 / 0.5155 / 0.6061 | 0.1226 / 0.4255 / 0.4922 / 0.5881 | - |
| Top-1 Disagreement Rate | 0.4087 | 0.4047 | +0.0039 |

## Training & Target Metrics (Fresh Validation Probe)

| Lane | Step | CE(search) | CE(incumbent) | CE(mixed) | Search-CE Improv | Fit Fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| beta_000 | 1 | 1.0879 | 0.9478 | 1.0879 | +0.0001 | 0.0178 |
| beta_000 | 4 | 1.0877 | 0.9478 | 1.0877 | +0.0003 | 0.0906 |
| beta_000 | 16 | 1.0868 | 0.9478 | 1.0868 | +0.0011 | 0.3485 |
| beta_000 | 46 | 1.0848 | 0.9481 | 1.0848 | +0.0031 | 1.0000 |
| beta_095 | 1 | 1.0879 | 0.9478 | 0.9548 | +0.0001 | 0.0178 |
| beta_095 | 4 | 1.0877 | 0.9478 | 0.9548 | +0.0003 | 0.0907 |
| beta_095 | 16 | 1.0870 | 0.9478 | 0.9548 | +0.0010 | 0.3094 |
| beta_095 | 46 | 1.0861 | 0.9478 | 0.9547 | +0.0019 | 0.5941 |
| beta_100 | 1 | 1.0879 | 0.9478 | 0.9478 | +0.0000 | 0.0001 |
| beta_100 | 4 | 1.0879 | 0.9478 | 0.9478 | -0.0000 | -0.0003 |
| beta_100 | 16 | 1.0879 | 0.9478 | 0.9478 | -0.0000 | -0.0006 |
| beta_100 | 46 | 1.0879 | 0.9478 | 0.9478 | +0.0000 | 0.0001 |

## Policy Drift vs Incumbent (Fresh Validation Probe)

| Lane | Step | L1 mean | L1 max | L1 p50 | L1 p90 | L1 p95 | L1 p99 | JS mean | Top-1 Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| beta_000 | 1 | 0.000745 | 0.002095 | 0.000780 | 0.001222 | 0.001343 | 0.001682 | 0.000000 | 0.0020 |
| beta_000 | 4 | 0.002308 | 0.007568 | 0.002398 | 0.003955 | 0.004375 | 0.005465 | 0.000001 | 0.0059 |
| beta_000 | 16 | 0.008105 | 0.027052 | 0.008418 | 0.014407 | 0.015960 | 0.019712 | 0.000015 | 0.0137 |
| beta_000 | 46 | 0.020254 | 0.057772 | 0.020660 | 0.034537 | 0.037311 | 0.045186 | 0.000087 | 0.0254 |
| beta_095 | 1 | 0.000745 | 0.002095 | 0.000780 | 0.001222 | 0.001342 | 0.001682 | 0.000000 | 0.0020 |
| beta_095 | 4 | 0.002126 | 0.007230 | 0.002185 | 0.003636 | 0.004138 | 0.005110 | 0.000001 | 0.0039 |
| beta_095 | 16 | 0.004739 | 0.016406 | 0.004791 | 0.008693 | 0.010022 | 0.011904 | 0.000005 | 0.0098 |
| beta_095 | 46 | 0.005223 | 0.016928 | 0.004897 | 0.009804 | 0.011089 | 0.013844 | 0.000007 | 0.0117 |
| beta_100 | 1 | 0.000017 | 0.000044 | 0.000017 | 0.000028 | 0.000031 | 0.000036 | 0.000000 | 0.0000 |
| beta_100 | 4 | 0.000413 | 0.001081 | 0.000417 | 0.000703 | 0.000807 | 0.001007 | 0.000000 | 0.0020 |
| beta_100 | 16 | 0.000177 | 0.000592 | 0.000162 | 0.000346 | 0.000403 | 0.000507 | 0.000000 | 0.0000 |
| beta_100 | 46 | 0.000052 | 0.000231 | 0.000046 | 0.000100 | 0.000119 | 0.000150 | 0.000000 | 0.0000 |

## Parameter Drift (Relative L2 Drift vs Incumbent)

| Lane | Step | Trunk | Policy Head | Value Stack |
| --- | ---: | ---: | ---: | ---: |
| beta_000 | 1 | 0.000000 | 0.000085 | 0.000000 |
| beta_000 | 4 | 0.000000 | 0.000263 | 0.000000 |
| beta_000 | 16 | 0.000000 | 0.000910 | 0.000000 |
| beta_000 | 46 | 0.000000 | 0.002395 | 0.000000 |
| beta_095 | 1 | 0.000000 | 0.000084 | 0.000000 |
| beta_095 | 4 | 0.000000 | 0.000254 | 0.000000 |
| beta_095 | 16 | 0.000000 | 0.000741 | 0.000000 |
| beta_095 | 46 | 0.000000 | 0.001733 | 0.000000 |
| beta_100 | 1 | 0.000000 | 0.000002 | 0.000000 |
| beta_100 | 4 | 0.000000 | 0.000051 | 0.000000 |
| beta_100 | 16 | 0.000000 | 0.000045 | 0.000000 |
| beta_100 | 46 | 0.000000 | 0.000041 | 0.000000 |

## Search Diagnostics (384:256 context, candidate vs incumbent)

| Lane | Step | Move Change Rate | Visit JS | Q-Rank Change | Root-Value Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| beta_000 | 1 | 0.0000 | 0.0001 | +0.0039 | -0.0000 |
| beta_000 | 4 | 0.0000 | 0.0002 | +0.0000 | -0.0002 |
| beta_000 | 16 | 0.0117 | 0.0005 | +0.0078 | +0.0002 |
| beta_000 | 46 | 0.0195 | 0.0010 | +0.0430 | +0.0006 |
| beta_095 | 1 | 0.0000 | 0.0001 | +0.0039 | -0.0000 |
| beta_095 | 4 | 0.0000 | 0.0002 | +0.0000 | -0.0002 |
| beta_095 | 16 | 0.0117 | 0.0003 | +0.0352 | +0.0001 |
| beta_095 | 46 | 0.0156 | 0.0002 | +0.0430 | -0.0002 |
| beta_100 | 1 | 0.0000 | 0.0000 | +0.0000 | -0.0000 |
| beta_100 | 4 | 0.0000 | 0.0000 | +0.0117 | +0.0000 |
| beta_100 | 16 | 0.0000 | 0.0000 | +0.0039 | -0.0000 |
| beta_100 | 46 | 0.0000 | 0.0000 | +0.0000 | -0.0000 |

## Per-Depth Policy L1/JS on Expanded Probe States (384:256, candidate vs incumbent)

| Lane | Step | Depth | Expanded Nodes | L1 Mean | JS Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| beta_000 | 16 | 0 | 256 | 0.008156 | 0.000015 |
| beta_000 | 16 | 1 | 1005 | 0.008523 | 0.000016 |
| beta_000 | 16 | 2 | 3253 | 0.008571 | 0.000015 |
| beta_000 | 16 | 3 | 7709 | 0.008607 | 0.000015 |
| beta_000 | 16 | 4 | 66506 | 0.008057 | 0.000014 |
| beta_000 | 46 | 0 | 256 | 0.020539 | 0.000090 |
| beta_000 | 46 | 1 | 1005 | 0.021437 | 0.000094 |
| beta_000 | 46 | 2 | 3253 | 0.021588 | 0.000092 |
| beta_000 | 46 | 3 | 7709 | 0.021503 | 0.000091 |
| beta_000 | 46 | 4 | 66506 | 0.019964 | 0.000085 |
| beta_095 | 16 | 0 | 256 | 0.004866 | 0.000005 |
| beta_095 | 16 | 1 | 1005 | 0.005087 | 0.000006 |
| beta_095 | 16 | 2 | 3253 | 0.005037 | 0.000005 |
| beta_095 | 16 | 3 | 7709 | 0.005018 | 0.000005 |
| beta_095 | 16 | 4 | 66506 | 0.004728 | 0.000005 |
| beta_095 | 46 | 0 | 256 | 0.005532 | 0.000007 |
| beta_095 | 46 | 1 | 1005 | 0.005883 | 0.000008 |
| beta_095 | 46 | 2 | 3253 | 0.005808 | 0.000008 |
| beta_095 | 46 | 3 | 7709 | 0.005709 | 0.000007 |
| beta_095 | 46 | 4 | 66506 | 0.005212 | 0.000007 |
| beta_100 | 16 | 0 | 256 | 0.000177 | 0.000000 |
| beta_100 | 16 | 1 | 1005 | 0.000198 | 0.000000 |
| beta_100 | 16 | 2 | 3253 | 0.000198 | 0.000000 |
| beta_100 | 16 | 3 | 7709 | 0.000194 | 0.000000 |
| beta_100 | 16 | 4 | 66506 | 0.000175 | 0.000000 |
| beta_100 | 46 | 0 | 256 | 0.000051 | 0.000000 |
| beta_100 | 46 | 1 | 1005 | 0.000057 | 0.000000 |
| beta_100 | 46 | 2 | 3253 | 0.000057 | 0.000000 |
| beta_100 | 46 | 3 | 7709 | 0.000056 | 0.000000 |
| beta_100 | 46 | 4 | 66506 | 0.000052 | 0.000000 |

## Canonical Paired Arena (Candidate vs Frozen Incumbent)

| Lane | Step | Context | Paired Effect | 95% CI | P0 Effect | P1 Effect | W/D/L |
| --- | ---: | --- | ---: | --- | ---: | ---: | --- |
| beta_000 | 16 | 384:256 | -0.0059 | [-0.0195, +0.0098] | +0.0000 | -0.0117 | 392/46/74 |
| beta_000 | 46 | 384:256 | -0.0605 | [-0.0918, -0.0293] | -0.1094 | -0.0117 | 364/46/102 |
| beta_095 | 16 | 384:256 | -0.0137 | [-0.0234, -0.0039] | +0.0000 | -0.0273 | 388/46/78 |
| beta_095 | 46 | 384:256 | -0.0137 | [-0.0293, +0.0039] | -0.0156 | -0.0117 | 392/38/82 |
| beta_095 | 46 | 1200:1200 | +0.0234 | [+0.0078, +0.0430] | +0.0273 | +0.0195 | 204/128/180 |
| beta_100 | 46 | 384:256 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 | 402/32/78 |

## Classification Evidence

| Signal | Value |
| --- | ---: |
| beta_100_incumbent_equivalent | True |
| beta_000_step46_384_256_effect | -0.0605 |
| beta_000_step46_384_256_safe | False |
| beta_095_step46_384_256_effect | -0.0137 |
| beta_095_step46_384_256_safe | True |
| beta_095_step46_fit_fraction | +0.5941 |
| beta_095_step46_meaningful_fit | True |
| beta_095_step46_1200_1200_effect | +0.0234 |
| beta_095_step46_1200_1200_safe | True |
| noninferiority_lower | -0.0300 |
| meaningful_fit_fraction | +0.2500 |

## Exact Reproduction Commands

```bash
.venv/bin/python ml/alphazero_lite/run_fresh_selfplay_anchor_iteration.py \
  --workdir /tmp/azlite_fresh_selfplay_anchor \
  --games 700 \
  --seed 42 \
  --arena-workers 24
```

Full JSON evidence: `docs/data/alphazero-lite-fresh-selfplay-anchor-summary.json`.
