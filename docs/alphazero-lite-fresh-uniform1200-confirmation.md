# Fresh Uniform1200 Confirmation

Status: complete, no promotion performed.

The frozen parent is `model-artifact/current/weights.json` (`8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`). The inherited normal recipe is 1,600 self-play games, 384 simulations for the first 8 plies and 192 thereafter; its root noise, temperatures, sharpened policy/value targets, fixed replay sources (weights 1:2), four epochs, batch size 512, Huber value loss (weight 0.3), and gradient clip 1.0 are unchanged.

The three mechanically next seeds are 47, 48, and 49. Every matched pair starts from the parent above and uses the same replay and training recipe. `uniform1200` alone uses 1,200 simulations on every self-play move; no rows are matched and no exact/R61/PR #323 states enter training.

Primary evaluation is a 120-game seat-balanced direct arena at an identical 384-simulation evaluation budget for both candidates, followed by the frozen forensic-v2 reference with W/D/L utility. The production gate is recorded unchanged; the outcome-aligned shadow gate is a separate report. The R61 expanded descendant panel is secondary only.

## Results

All six fresh lanes completed from the pinned parent. The regenerated replay inputs preserve the required `4:1:8:4` category weights; they are new SHA-pinned artifacts, not historical-byte-identical inputs.

The matched direct arena (uniform1200 versus control, 384 simulations each) scored `0.5000`, `0.5000`, and `1.0000` for seeds 47, 48, and 49 respectively. The mean effect above draw was `+0.1667`.

Frozen exact forensic-v2 improved in every seed: top-1 agreement changed from `0.5634/0.5493/0.5634` to `0.6620/0.6901/0.6995`; mean regret changed from `2.7230/3.0892/2.8545` to `1.9249/1.9249/1.5023`; and blunder rate changed from `0.4366/0.4507/0.4366` to `0.3380/0.3099/0.3005`.

All six production-gate reports are runtime-valid after correcting forced-seat report aggregation. Uniform1200 classifications were `high_search_breakthrough` (seed 47), `seat_artifact_only` (seed 48), and `standard_budget_breakthrough` (seed 49). Control lanes also showed search-budget classifications, so these gate labels are diagnostic rather than promotion evidence.

Hard classification: `uniform1200_fresh_seed_heterogeneous`. The matched arena has two neutral seeds and one decisive seed; therefore the experiment does not establish a robust promotion case. No artifact was promoted.
