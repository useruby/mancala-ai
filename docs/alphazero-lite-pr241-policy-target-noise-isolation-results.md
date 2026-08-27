# PR #241 Policy-Target Noise Isolation

**Classification:** `inconclusive`

## Contract

- Frozen `ordinary_onpolicy.jsonl`: SHA-256 `6671e248af4a4c82e1155c798cb7490cd66cd80dc10b203c97d89dced94527f2`; 27,642 rows.
- Exact PR #241 exclusions retained 27,350 rows in their original order.
- Canonical suite SHA-256: `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`.
- Target seed: `sha256([namespace, game_index, move_index, canonical_state_hash])[:16] mod 2**31`. Fresh noisy and denoised targets share this seed.
- Fresh target searches used the A16 step-16 evaluator, 384 simulations, c_puct 1.25, zero FPU, visit-count roots, no subtree reuse, and the original 1.0/0.1 temperature schedule. Only the policy label changed.
- The original lane reproduced bit-identically at steps 1, 4, and 16.

## Target Diagnostics

| Comparison | Mean L1 | P50 / P90 / P99 L1 | Mean JS | Top-1 disagreement | Entropy delta |
| --- | ---: | --- | ---: | ---: | ---: |
| fresh noisy vs fresh denoised | 0.0759 | 0.0000 / 0.3177 / 0.6849 | 0.00819 | 0.0409 | -0.0356 |

The fresh noisy/denoised difference is isolated to roots where PR #241 enabled root noise: mean L1 0.3061, mean JS 0.0330, and top-1 disagreement 0.1648. Roots without action-sampling noise are identical. The fixed 256-root clean-1200 probe put fresh denoised slightly closer than fresh noisy (mean L1 0.3384 vs 0.3508; mean JS 0.0873 vs 0.0896), but both were farther than the original noisy targets.

## Step-16 Training

| Lane | CE(search) | CE(P1) | CE(beta095) | Fit | L1 vs P1 | JS vs P1 | Top-1 | Adapter norm | Delta A16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original noisy | 1.0957 | 0.9455 | 0.9530 | 0.9974 | 0.00268 | 1.57e-06 | 0.00234 | 0.00431 | 0.00196 |
| fresh noisy | 1.0329 | 0.9455 | 0.9498 | 0.9626 | 0.00304 | 2.02e-06 | 0.00249 | 0.00451 | 0.00210 |
| fresh denoised | 0.9565 | 0.9455 | 0.9460 | 0.9811 | 0.00336 | 2.46e-06 | 0.00263 | 0.00464 | 0.00232 |

All step-16 lanes meet the meaningful-fit threshold and retain bit-identical inherited parameters.

## Ordinary-PUCT Arena

All rows use 128 openings, seat swap, seed 42, and 10,000-opening bootstrap intervals.

| Lane, step 16 | 384:256 effect / CI | Safe | 1200:1200 effect / CI | Safe |
| --- | --- | ---: | --- | ---: |
| original noisy | -0.0195 [-0.0391, -0.0039] | No | +0.0410 [+0.0137, +0.0664] | Yes |
| fresh noisy | -0.0195 [-0.0391, -0.0039] | No | +0.0410 [+0.0137, +0.0664] | Yes |
| fresh denoised | -0.0195 [-0.0391, -0.0039] | No | -0.0195 [-0.0391, -0.0039] | No |

The P0 gate was ineligible because no fresh-denoised checkpoint was safe at both budgets.

## Interpretation

Removing root noise from reconstructed policy targets did not improve the shallow-search result, while the high-budget gain observed in the paired fresh-noisy lane disappeared. This does not meet `denoising_sacrifices_high_budget_gain`, which requires shallow safety to improve, nor does it establish `policy_target_noise_not_causal`, because high-budget behavior differs. No candidate is promoted.
