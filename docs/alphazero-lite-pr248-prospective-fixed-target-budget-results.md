# PR #249 Prospective Fixed Policy-Target Budget

**Classification:** `target_depth_mechanism_fails_new_batch`

## Frozen Contract

- Generated a new ordinary A16 self-play batch: 700 games, 24 workers, global seed 46, Kalah v3.
- Authoritative gameplay was reused-tree ordinary PUCT: 384 simulations, c_puct 1.25, visit-count root policy, FPU zero, unnormalized values.
- Dirichlet alpha/epsilon were .3/.3 through ply 9; target temperature was 1.0 through ply 9 and .1 afterwards.
- The four views were authoritative reused, fresh-384, fresh-768, and fresh-1024. No other target budget was generated.
- A16 snapshot SHA-256: `f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff`.
- Initial Adam SHA-256: `61d5719e75aae87d7c2ca7ed2c5b01871ac2ea1675a34c4a6c918c783894e8c7`.
- P1 checkpoint SHA-256: `e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9`.
- Target evaluator SHA-256: `8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34`.
- Canonical suite SHA-256: `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`.

## Invariants And Replays

- All matched target searches received identical root priors from clones of the pre-gameplay-search RNG state. Fresh searches did not consume gameplay RNG.
- At ply 0, reused and fresh-384 were exactly equal for all 700 games.
- Every eligible view has policy-excluded state/outcome SHA-256 `7cd216990be013f531df8740c8e24decb124a2254ba0f408ea49f8308f6fc750` and batch-plan SHA-256 `671ac065c7b82c4a04b9201a4cb60b0e0d2ee4c73c390566c800704252a2e748`.
- Replay SHA-256: reused `44b4f0f61e65d50d37f33cb2480c3cf58b9d372642abb1377bba52667422c8f8`; fresh-384 `d0dcae17fdd1f148a8cb1edd9a62caad91b483dad0b4b1e84b15a84682157d4c`; fresh-768 `83afa719a260083902540419b72b30b6a96d7bc63015be2482af442ab7d4baa9`; fresh-1024 `007a72d5c07c15f353ef244cab627198cb7be6b73c43b7a03ea2feb767b1fbb4`.
- Exclusions were identical: amplified 4, washed 3, held-out 2, canonical 202; 211 excluded and 28,217 eligible rows.
- Independent model and Adam copies were used per lane. Pristine Adam was unchanged; repeated fresh-768 was byte-identical; fresh-768 and fresh-1024 hashes were execution-order independent.

## Target Geometry

Values are mean L1 / p50 / p90 / p99 / mean JS / top-1 disagreement / entropy difference. Full phase, legal-count, root-noise, and inherited-mass-quartile strata are in `/tmp/azlite_pr248_prospective_fixed_target_budget/summary.json`.

| Comparison | Metrics |
| --- | --- |
| reused vs fresh-384 | .242335 / .002473 / .996234 / 1.999404 / .059079 / .129390 / +.018539 |
| reused vs fresh-768 | .120885 / .000370 / .275709 / 1.914214 / .021469 / .064217 / -.004262 |
| reused vs fresh-1024 | .111165 / .000142 / .300781 / 1.692619 / .015783 / .059645 / -.015290 |
| fresh-384 vs fresh-768 | .211625 / .002887 / .607270 / 1.983959 / .043408 / .114505 / -.022801 |
| fresh-768 vs fresh-1024 | .097238 / .000238 / .218099 / 1.613044 / .014576 / .053656 / -.011028 |

## Budget Movement

| Transition class | Frequency | Mean two-transition legal L1 |
| --- | ---: | ---: |
| stable | .847680 | .111773 |
| 768_only_flip | .010596 | 1.550879 |
| persistent_flip | .098664 | 1.430506 |
| late_flip | .037814 | 1.263443 |
| other | .005245 | 1.671141 |

## Training And Frozen Diagnostics

All lanes used beta .95, Adam lr 1e-5, `policy_adapter_only`, batch size 512, grad clip 1.0, and the identical batch plan. At step 16 every lane had fit fraction 1.0. The recorded per-step CE values, CE(P1), CE(beta095), policy distance, adapter norm, parameter delta, model SHA, and optimizer SHA are in the summary artifact.

| Lane | Step-16 own CE | CE(P1) | CE(beta095) | L1 vs P1 | JS vs P1 | Adapter norm | Parameter delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reused | 1.086770 | .934817 | .942415 | .003216 | .000002 | .004755 | .002323 |
| fresh-384 | 1.023562 | .934819 | .939256 | .003426 | .000003 | .004850 | .002424 |
| fresh-768 | 1.068478 | .934817 | .941500 | .003117 | .000002 | .004657 | .002258 |
| fresh-1024 | 1.085798 | .934816 | .942365 | .003045 | .000002 | .004631 | .002231 |

Frozen ordinary-PUCT diagnostics were identical for all lanes: amplified rescue rate .300 and washed new-divergence rate .000.

## Canonical Arena

All arenas used ordinary PUCT, current 128 canonical openings, seat swapping, seed 42, matched P1 controls, and 10,000 opening bootstraps.

| Lane | 384:256 effect / W-D-L | 1200:1200 effect / CI / W-D-L |
| --- | --- | --- |
| reused | -.019531 / 390-38-84 | -.019531 / [-.039062,-.003906] / 180-132-200 |
| fresh-384 | -.019531 / 390-38-84 | -.019531 / [-.039062,-.003906] / 180-132-200 |
| fresh-768 | -.019531 / 390-38-84 | -.019531 / [-.039062,-.003906] / 180-132-200 |
| fresh-1024 | -.019531 / 390-38-84 | +.041016 / [+.013672,+.066406] / 242-70-200 |

At 1200:1200, fresh-768 minus fresh-384 was `0`, CI `[0,0]`; reused minus fresh-384 was `0`, CI `[0,0]`; fresh-768 minus fresh-1024 was `-.060547`, CI `[-.080078,-.042969]`.

At 1200:1200, reused, fresh-384, and fresh-768 each had seat effects p0 `-.039062` and p1 `0`; fresh-1024 had p0 `-.039062` and p1 `+.121094`. At 384:256 every lane had p0 `-.039062` and p1 `0`.

## Interpretation

The primary fresh-768 replication criterion failed, and so did the reused positive control. This is therefore Case E under the preregistration, despite fresh-1024 independently matching the historical positive deep-search effect. The seed-45 fixed-768 result and its 768-vs-1024 reversal did not prospectively replicate. No lane is promoted.
