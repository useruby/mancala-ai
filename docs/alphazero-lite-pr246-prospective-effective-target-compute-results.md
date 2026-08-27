# PR #246 Prospective Effective Target Compute

**Classification:** `prospective_effective_target_compute_replica`

## Frozen Contract

- New ordinary A16 self-play: 700 games, 24 workers, global seed 45, Kalah v3.
- Gameplay: reused-tree ordinary PUCT, 384 simulations, c_puct 1.25, visit-count root policy, FPU zero, unnormalized values.
- Exploration: Dirichlet alpha .3 and epsilon .3 through ply 9; temperature 1.0 through ply 9 then .1.
- Target formula was frozen before generation: `fresh_equiv_simulations = 384 + inherited_child_visit_mass`.
- A16 snapshot SHA-256: `f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff`.
- Initial Adam SHA-256: `61d5719e75aae87d7c2ca7ed2c5b01871ac2ea1675a34c4a6c918c783894e8c7`.
- P1 checkpoint SHA-256: `e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9`.
- Target evaluator SHA-256: `8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34`.
- Generation-config SHA-256: `4cc9aa84fed0c5ab06d1ce9b6620322d8e95e01e28acdef69eeb08f9fc9db148`.

## Invariants

- The three matched RNG clones received exactly identical noisy root priors at every ply.
- Ply 0 had zero inherited child mass and exactly identical reused, fresh-384, and fresh-equivalent targets for all 700 games.
- All three generated replays have the same policy-excluded state/outcome SHA-256: `45e9b0a76879fd0e92f6536107073ecbe16a59832125e74fb6d12fb6230c527c`.
- Every eligible view shares batch-plan SHA-256 `a4e859f1340078e21c9ad2b4e0c04bac5c69c7463d8b8b859f83f79f915382fe`.
- Exclusions were identical: amplified 3, washed 10, PR #220 held-out 4, canonical arena 192, total 209; 27,858 rows remained.
- Every lane independently deep-copied the A16 model and Adam state. The pristine Adam state was unchanged, and a repeated fresh-384 lane was byte-identical.

## Replay Artifacts

| View | Replay SHA-256 |
| --- | --- |
| prospective reused | `245a452f80970485dd9d07dad560e35f04bbccc16f6147e36c98598b7426f106` |
| prospective fresh-384 | `74203f35df72c25d4b02f25605926fb34e36edc114cb0a4e2c2cd8f2771c84c6` |
| prospective fresh-equivalent | `e1bc3b17c165b8aee210fc8267e4477c510abe467f47225c3c4f874f74005bab` |

## Target Geometry

| Comparison | Legal L1 | JS | Top-1 disagreement | Entropy difference |
| --- | ---: | ---: | ---: | ---: |
| reused vs fresh-384 | .241122 | .058496 | .128796 | +.019201 |
| reused vs fresh-equivalent | .002358 | .000240 | .001185 | -.000706 |
| fresh-384 vs fresh-equivalent | .240779 | .058335 | .128222 | -.019907 |

Fresh-equivalent was JS-closer to reused than fresh-384 on 89.66% of eligible positions. The reused/fresh-equivalent geometry remained close across every inherited-mass quartile; their worst quartile mean L1 was .004977 and mean JS .000679.

## Effective Compute

| Metric | Mean | P50 | P90 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh-equivalent target simulations | 1228.42 | 1106 | 2071 | 3301.43 | 5773 |
| Effective target multiplier | 3.1990 | 2.8802 | 5.3932 | 8.5975 | 15.0339 |
| Gameplay plus fresh-equivalent target simulations | 1612.42 | 1490 | 2455 | 3685.43 | 6157 |

The new-batch target budget closely matches PR #246's frozen-replay mean of 1220.08 target simulations and 3.1773 multiplier. The compute accounting above is gameplay 384 plus one fresh-equivalent target search; it does not count the diagnostic fresh-384 fork.

## Step-16 Diagnostics

All lanes meaningfully fit their own targets: step-16 fit fraction was 1.0. Frozen 40+40 ordinary-PUCT diagnostics were unchanged across lanes: amplified rescue .325 and washed new divergence .000.

| Lane | 384:256 effect / CI / W-D-L | 1200:1200 effect / CI / W-D-L |
| --- | --- | --- |
| prospective reused | `-.019531` `[-.039062,-.003906]` 390/38/84 | `+.041016` `[+.013672,+.066406]` 242/70/200 |
| prospective fresh-384 | `-.019531` `[-.039062,-.003906]` 390/38/84 | `-.019531` `[-.039062,-.003906]` 180/132/200 |
| prospective fresh-equivalent | `-.019531` `[-.039062,-.003906]` 390/38/84 | `+.041016` `[+.013672,+.066406]` 242/70/200 |

All arenas used ordinary PUCT, current canonical 128 openings, seat swapping, seed 42, matched P1 controls, and 10,000 opening bootstraps. At 1200:1200, reused and fresh-equivalent each had seat effects p0 `-.039062`, p1 `+.121094`; fresh-384 had p0 `-.039062`, p1 `0`.

## Paired Contrasts

| Contrast | 384:256 | 1200:1200 |
| --- | --- | --- |
| fresh-equivalent minus fresh-384 | `0` `[0,0]` | `+.060547` `[+.042969,+.080078]` |
| reused minus fresh-384 | `0` `[0,0]` | `+.060547` `[+.042969,+.080078]` |
| reused minus fresh-equivalent | `0` `[0,0]` | `0` `[0,0]` |

## Interpretation

The PR #246 mechanism generalizes prospectively. On a new seed-45 self-play batch, a fresh target search with only the inherited visit mass converted into additional target simulations reproduces both the reused target geometry and its 1200:1200 arena gain. No inherited Q values or subtree structure were supplied to either fresh search.

Recommended next: replace inherited-mass-derived target compute with a simpler fixed target-search budget, then identify the cheapest fixed budget that retains the deep-search result.
