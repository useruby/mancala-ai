# PR #255 Third-Seed Budget Repair Results

**Classification:** `third_seed_no_budget_split`

This completes the frozen seed-47 evaluation from PR #255. It reuses the
committed replay and recovered step-16 candidates, with no retraining, new
target generation, suite replacement, replay change, or arena-setting change.
The first arena cache predated the manifest and was invalidated; all retained
P/Q/R records were rerun under the exact frozen contract after the manifest.

## Frozen Preconditions

`preflight_audit.passed` is `true`. Each P/Q/R suite has 128 unique openings,
zero final-state and prefix overlap with canonical through O, zero mutual
overlap, and zero seed45/46/47 replay-state overlap. The manifest preceded the
first retained arena record.

| File | SHA-256 |
| --- | --- |
| `suite_registry.json` | `6a2ee79b3449dc3718a9e0f98a8f1a6307af7b503da73d8714352713187dc70f` |
| `preflight_audit.json` | `fb36f36daea817252286aeb0efcedc0e3029581db57c0ca716ca5660a85ae87f` |
| `frozen_manifest.json` | `6da0b9f0fa0845c14c6cfbf785dcd72f344ba22464f921dc70dc9dacb991718f` |

| Runtime artifact | SHA-256 |
| --- | --- |
| A16 snapshot | `f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff` |
| Initial Adam | `61d5719e75aae87d7c2ca7ed2c5b01871ac2ea1675a34c4a6c918c783894e8c7` |
| P1 checkpoint | `e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9` |
| Target evaluator | `8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34` |

| Suite | Seed | SHA-256 |
| --- | ---: | --- |
| P | 16042 | `25f8de0e48421e6b112b1bd16425d2f9d645a235694dcd2757d9ae0e9a68e8b6` |
| Q | 17042 | `5ba80e5857c9f52f533c6a3e3b1548a77daebbe45d806d2c76c6b9b425767ef5` |
| R | 18042 | `56766034b7027cccce8492a4cb80ca23f2edec409f2d07b9cd39cb55bfdd106b` |

| Candidate | Full model | Adapter | Optimizer | Artifact weights |
| --- | --- | --- | --- | --- |
| reused | `6c3b49660cb5a42c1b861742ac88bc944f81b657b9954fb5b5041d9cd25de657` | `eea48a24e8092567e8993e11d37b43a4446388bb361d0aabe4c4e4c9a4995a81` | `00a62a73d77b157d3af4f1404da767ba88987d4b9a87bfb22091dd6bdb1d2fcf` | `2bedbed64e34770d053c55a81322a9a8b5999866c59122156378d916bfb29ee8` |
| fresh768 | `c0fc52cfd0c998536457a7cd31a4cf73ec5bbbf80574659d50261d35b18ac582` | `306c5f40f2e7be2ca58846d5bebd99c94ab46ee791d81e0d282fc91b6ba94e16` | `bb110e549a15fe3711c4cea7e04572bc15fe8026e04718725a1096debc4c2391` | `2de3cf04b1bad6ca22200d1c84e2e5af8540ac9ae7a42598d937abf782af65be` |
| fresh1024 | `d8dbed923f24f761e992e00c04258578cf844aa909610c0f395b88e168551ea1` | `cdb8a12cf6a704932575901c2638cb68d840cb680f61026045bfc94053c8640a` | `8718a5464d6887afe0ee3d329fa9cb78259b4de93f631ec494b69972cc660a19` | `51640adfd348f876b89306c59b455c18985ba0b661ba369017c4b6f3cf0e763f` |

## Primary Arena

All P/Q/R arenas used ordinary `1200:1200` PUCT. Each suite includes exactly
one matched P1-vs-P1 control; all effects below are control-adjusted.

| Contrast | P | Q | R | Pooled effect | Opening-bootstrap 95% CI | Hierarchical suite->opening 95% CI | Same-sign suites |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| fresh768 - fresh1024 | .000000 | .000000 | .000000 | .000000 | `[.000000, .000000]` | `[.000000, .000000]` | 3 |

The committed classifier requires a strictly positive pooled lower bound or
strictly negative pooled upper bound, plus at least two same-sign suites.
Neither condition selects a winner at zero, so attribution is skipped.

## Absolute Results

W/D/L is challenger-perspective. Each row reports adjusted effect vs P1,
opening-bootstrap 95% CI, P0 effect, P1 effect, and W/D/L.

| Candidate | Suite | Adjusted effect (95% CI) | P0 | P1 | W/D/L |
| --- | --- | --- | ---: | ---: | --- |
| reused | P | `-.019531 [-.039062, -.003906]` | -.039062 | .000000 | 200/92/220 |
| reused | Q | `-.031250 [-.054688, -.011719]` | -.062500 | .000000 | 180/120/212 |
| reused | R | `-.023438 [-.042969, -.007812]` | -.046875 | .000000 | 194/100/218 |
| fresh768 | P | `.017578 [-.005859, .041016]` | -.039062 | .074219 | 238/54/220 |
| fresh768 | Q | `.025391 [-.005859, .054688]` | -.062500 | .113281 | 238/62/212 |
| fresh768 | R | `.021484 [-.003906, .046875]` | -.046875 | .089844 | 240/54/218 |
| fresh1024 | P | `.017578 [-.005859, .041016]` | -.039062 | .074219 | 238/54/220 |
| fresh1024 | Q | `.025391 [-.005859, .054688]` | -.062500 | .113281 | 238/62/212 |
| fresh1024 | R | `.021484 [-.003906, .046875]` | -.046875 | .089844 | 240/54/218 |

Secondary contrasts are identical for `reused - fresh768` and
`reused - fresh1024`: P/Q/R = `-.037109`/`-.056641`/`-.044922`; pooled effect
`-.046224`, opening-bootstrap 95% CI `[-.056641, -.037109]`, hierarchical 95%
CI `[-.070312, -.025391]`, 3 same-sign suites.

## Three-Seed Conclusion

| Replay seed | Frozen outcome |
| --- | --- |
| seed45 | 768 strong, 1024 weak |
| seed46 | 1024 strong, 768 weak |
| seed47 | `third_seed_no_budget_split` |

Target-budget effects must be treated with replay seed as the experimental
unit. There is no universal target budget conclusion and no attribution result
for seed47. The next science is a multi-replay-seed comparison, closing the
single-seed target-budget mechanism branch rather than selecting a rule from
one replay.
