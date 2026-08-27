# PR #241/#244 Optimizer-Isolation Reproduction

**Classification:** `ordinary_reused_target_gain_reproduces`

## Frozen Contract

- No self-play was generated and no replay row or policy target was modified.
- Reused replay SHA-256 / eligible rows: ordinary `6671e248...94527f2` / 27,350; shadow .75 `066e52a5...35bbd039` / 27,597; shadow 1.0 `38dd9e89...e49f1684` / 26,425.
- A16 serialized snapshot SHA-256: `f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff`.
- P1 checkpoint SHA-256: `e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9`.
- Canonical suite SHA-256: `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`.
- Training was unchanged: beta .95, Adam LR `1e-5`, policy-adapter-only, batch size 512, clip 1.0, 16 steps.

## Isolation Invariants

The canonical initial Adam state SHA-256 was `61d5719e75aae87d7c2ca7ed2c5b01871ac2ea1675a34c4a6c918c783894e8c7`. Every lane invocation recorded this same before/after fingerprint. `metrics()` also received its own deep copy and left the pristine state unchanged.

| Check | Result |
| --- | --- |
| ORDER A ordinary, shadow .75, shadow 1.0 vs ORDER B shadow 1.0, ordinary, shadow .75 | Pass: every lane/step model, adapter, optimizer, CE metrics, and fit fraction byte-identical |
| ordinary then ordinary | Pass: steps 1, 4, 16 model and optimizer hashes identical |
| shadow .75 then shadow .75 | Pass: steps 1, 4, 16 model and optimizer hashes identical |
| Pristine Adam state after all training and metrics | Pass: unchanged |

## Clean Metrics

| Lane / step | CE(search) | CE(P1) | CE(beta095) | Fit | L1 vs P1 | JS vs P1 | Top-1 | Adapter norm | Delta A16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ordinary / 1 | 1.095910 | .945449 | .952972 | .6023 | .001857 | 7.57e-7 | .001718 | .002637 | .000160 |
| ordinary / 4 | 1.095859 | .945450 | .952970 | .6944 | .002120 | 9.84e-7 | .001974 | .003047 | .000593 |
| ordinary / 16 | 1.095691 | .945453 | .952965 | 1.0000 | .002901 | 1.84e-6 | .002596 | .004470 | .002072 |
| shadow .75 / 1 | 1.087741 | .933129 | .940860 | .5524 | .001831 | 7.41e-7 | .001232 | .002624 | .000149 |
| shadow .75 / 4 | 1.087677 | .933130 | .940858 | .6481 | .002145 | 1.01e-6 | .001413 | .003071 | .000611 |
| shadow .75 / 16 | 1.087443 | .933135 | .940851 | 1.0000 | .003272 | 2.32e-6 | .002392 | .004722 | .002308 |
| shadow 1.0 / 1 | 1.105322 | .932683 | .941315 | .5486 | .001830 | 7.43e-7 | .001703 | .002627 | .000153 |
| shadow 1.0 / 4 | 1.105264 | .932684 | .941313 | .6449 | .002132 | 1.00e-6 | .002081 | .003071 | .000615 |
| shadow 1.0 / 16 | 1.105052 | .932689 | .941307 | 1.0000 | .003239 | 2.31e-6 | .003065 | .004731 | .002311 |

## Historical Checkpoints

Columns are model SHA match, optimizer SHA match, and adapter maximum absolute difference.

| Lane / step | Historical comparison |
| --- | --- |
| ordinary / 1, 4, 16 | `true`, `true`, `0.0` at every step |
| shadow .75 / 1 | `false`, `false`, `7.487717e-6` |
| shadow .75 / 4 | `false`, `false`, `2.564592e-5` |
| shadow .75 / 16 | `false`, `false`, `6.289248e-5` |
| shadow 1.0 / 1 | `false`, `false`, `6.874361e-6` |
| shadow 1.0 / 4 | `false`, `false`, `2.425768e-5` |
| shadow 1.0 / 16 | `false`, `false`, `5.606542e-5` |

No opening-level historical game records were retained, so an exact clean-versus-historical opening comparison is unavailable. PR #241 retained only arena telemetry.

## Frozen 40+40

All evaluations used ordinary PUCT only, and the roots remain excluded from replay training.

| Step-16 lane | Amplified rescue rate | Washed new-divergence rate |
| --- | ---: | ---: |
| ordinary | .275 | .000 |
| shadow .75 | .325 | .000 |
| shadow 1.0 | .300 | .000 |

## Canonical Arena

Each context used 128 openings, seat swap, seed 42, matched P1-vs-P1 controls, and 10,000 opening bootstraps. W/D/L is candidate-perspective.

Seat effects were `p0=-.019531, p1=0` for steps 1 and 4 at 384:256, and `p0=-.039062, p1=0` for steps 1 and 4 at 1200:1200. At step 16, every 384:256 lane was `p0=-.039062, p1=0`; the shadow lanes remained `p0=-.039062, p1=0` at 1200:1200, while ordinary was `p0=-.039062, p1=+.121094`.

| Lane / step | 384:256 effect, CI, W/D/L | 1200:1200 effect, CI, W/D/L |
| --- | --- | --- |
| ordinary / 1 | -.009766 [-.019531, -.001953], 390/48/74 | -.019531 [-.039062, -.003906], 180/132/200 |
| ordinary / 4 | -.009766 [-.019531, -.001953], 390/48/74 | -.019531 [-.039062, -.003906], 180/132/200 |
| ordinary / 16 | -.019531 [-.039062, -.003906], 390/38/84 | +.041016 [.013672, .066406], 242/70/200 |
| shadow .75 / 1 | -.009766 [-.019531, -.001953], 390/48/74 | -.019531 [-.039062, -.003906], 180/132/200 |
| shadow .75 / 4 | -.009766 [-.019531, -.001953], 390/48/74 | -.019531 [-.039062, -.003906], 180/132/200 |
| shadow .75 / 16 | -.019531 [-.039062, -.003906], 390/38/84 | -.019531 [-.039062, -.003906], 180/132/200 |
| shadow 1.0 / 1 | -.009766 [-.019531, -.001953], 390/48/74 | -.019531 [-.039062, -.003906], 180/132/200 |
| shadow 1.0 / 4 | -.009766 [-.019531, -.001953], 390/48/74 | -.019531 [-.039062, -.003906], 180/132/200 |
| shadow 1.0 / 16 | -.019531 [-.039062, -.003906], 390/38/84 | -.019531 [-.039062, -.003906], 180/132/200 |

No candidate was safe at both contexts, so the P0 gate was not eligible.

## Combined Conclusion

The replay-generation facts are unchanged: ordinary versus P1 state Jaccard is `1.0`; shadow .75 is `.0787`; shadow 1.0 is `.0577`. The clean shadow optimizations meaningfully fit their replay but remain unsafe at 1200 and do not beat clean ordinary. In contrast, clean ordinary is bit-identical to the historical first lane and reproduces its `+0.041015625` 1200:1200 effect, while PR #244's independently initialized fresh-noisy target training remained `-0.01953125`.

**Recommended follow-up:** isolate reused gameplay-tree policy targets versus fresh-tree targets.
