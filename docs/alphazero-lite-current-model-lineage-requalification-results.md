# Current-Model Lineage Requalification

**Classification:** `current_model_canonically_revalidated`

The promoted balanced model `8d70e90a...` robustly improves canonical
`384:256` strength over its direct parent `6ac71425...`. It passes all stated
major-budget robustness limits. Historical seat DS values were not used for
this decision.

## Provenance

- Candidate weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Parent weights SHA256: `6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece`
- Parent source: `b49a4ef:model-artifact/current/{metadata,weights}.json`
- Parent digest reproduced exactly; both artifacts are `residual_v3`, use
  `kalah_v3`, have three residual blocks, and load with `ArtifactEvaluator`.
- Runtime: `azlite_eval_seed_v2`, base seed 42, deterministic root policy,
  tactical root bias 0.0, default c_puct 1.25, and `768:768` c_puct 0.90.
- Each result uses two games per opening and a 10,000-sample unique-opening
  bootstrap. Candidate and matched parent/parent controls used the same suite,
  budgets, runtime, and seed contract.

`Score` columns are candidate challenger score and matched parent/parent
control score. `DS` is only the P0-minus-P1 seat-asymmetry diagnostic. `+ / 0
/ -` counts opening-level paired effects; trajectory fields are candidate game
trajectory counts.

## Medium

| Budget | Candidate score | Control score | Effect 95% CI | P0 / P1 effect | DS | + / 0 / - | Trajectories |
|---|---:|---:|---:|---:|---:|---:|---:|
| 384:256 | 0.791016 | 0.681641 | +0.109375 [+0.080078, +0.138672] | +0.117188 / +0.101562 | -0.105469 | 51 / 73 / 4 | 20 unique, 512 duplicate |
| 768:256 | 0.900391 | 0.818359 | +0.082031 [+0.058594, +0.107422] | +0.125000 / +0.039062 | -0.199219 | 37 / 91 / 0 | 20 unique, 512 duplicate |
| 768:768 | 0.533203 | 0.500000 | +0.033203 [-0.001953, +0.068359] | -0.066406 / +0.132812 | +0.175781 | 38 / 62 / 28 | 17 unique, 512 duplicate |
| 1200:1200 | 0.464844 | 0.500000 | -0.035156 [-0.062500, -0.007812] | +0.019531 / -0.089844 | +0.257812 | 9 / 88 / 31 | 15 unique, 512 duplicate |
| 1200:256 | 0.863281 | 0.847656 | +0.015625 [+0.003906, +0.031250] | +0.000000 / +0.031250 | -0.273438 | 4 / 124 / 0 | 20 unique, 512 duplicate |
| 256:768 | 0.269531 | 0.181641 | +0.087891 [+0.050781, +0.125000] | -0.019531 / +0.195312 | -0.500000 | 33 / 88 / 7 | 20 unique, 512 duplicate |

## Fixed Large

| Budget | Candidate score | Control score | Effect 95% CI | P0 / P1 effect | DS | + / 0 / - | Trajectories |
|---|---:|---:|---:|---:|---:|---:|---:|
| 384:256 | 0.800130 | 0.677734 | +0.122396 [+0.105469, +0.138672] | +0.170573 / +0.074219 | -0.035156 | 172 / 202 / 10 | 22 unique, 1536 duplicate |
| 768:256 | 0.903646 | 0.830729 | +0.072917 [+0.059245, +0.086589] | +0.101562 / +0.044271 | -0.192708 | 95 / 289 / 0 | 22 unique, 1536 duplicate |
| 768:768 | 0.523438 | 0.500000 | +0.023438 [+0.005208, +0.042318] | -0.032552 / +0.079427 | +0.114583 | 85 / 230 / 69 | 19 unique, 1536 duplicate |
| 1200:1200 | 0.481771 | 0.500000 | -0.018229 [-0.031901, -0.003906] | +0.022135 / -0.058594 | +0.192708 | 27 / 292 / 65 | 17 unique, 1536 duplicate |
| 1200:256 | 0.828776 | 0.815755 | +0.013021 [+0.005208, +0.022135] | +0.000000 / +0.026042 | -0.342448 | 10 / 374 / 0 | 22 unique, 1536 duplicate |
| 256:768 | 0.275391 | 0.169271 | +0.106120 [+0.083333, +0.128923] | -0.022135 / +0.234375 | -0.506510 | 134 / 222 / 28 | 22 unique, 1536 duplicate |

Fixed-large `384:256` passes: its effect is +0.122396 and its lower 95% bound
is +0.105469. The largest major-budget regression is `1200:1200` at -0.018229,
which remains inside the -0.03 limit; `768:768` is +0.023438 and `1200:256` is
+0.013021.

## Held-Out Confirmation

Suites 43 through 54 were evaluated at `384:256` (384 openings each). A
hierarchical bootstrap resampled suites and then openings, 10,000 times.

| Mean effect | Hierarchical 95% CI | Worst suite | P0 / P1 effect | Leave-one-suite-out mean range |
|---:|---:|---:|---:|---:|
| +0.123427 | [+0.117730, +0.129340] | +0.113281 (suite 46) | +0.176107 / +0.070747 | [+0.122396, +0.124349] |

All 12 held-out suite effects were positive: 43 +0.123047, 44 +0.120443, 45
+0.129557, 46 +0.113281, 47 +0.131510, 48 +0.115234, 49 +0.125000, 50
+0.123698, 51 +0.125000, 52 +0.134766, 53 +0.118490, and 54 +0.121094.

## Criteria

- Fixed-large `384:256` effect and lower CI are positive: pass.
- Held-out mean and hierarchical lower CI are positive: pass.
- Held-out worst suite is at least -0.03: pass.
- `768:768` is at least -0.05; `1200:1200` and `1200:256` are at least -0.03:
  pass.

The direct-parent comparison resolves the final historical metric blast-radius
action. No training, replay generation, promotion, rollback, runtime tuning,
or new opening suites were performed.
