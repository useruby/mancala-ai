# Fresh Uniform1200 Confirmation

Status: pre-registered execution pending. No candidate has been trained, evaluated, or promoted by this change.

The frozen parent is `model-artifact/current/weights.json` (`8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`). The inherited normal recipe is 1,600 self-play games, 384 simulations for the first 8 plies and 192 thereafter; its root noise, temperatures, sharpened policy/value targets, fixed replay sources (weights 1:2), four epochs, batch size 512, Huber value loss (weight 0.3), and gradient clip 1.0 are unchanged.

The three mechanically next seeds are 47, 48, and 49. Every matched pair starts from the parent above and uses the same replay and training recipe. `uniform1200` alone uses 1,200 simulations on every self-play move; no rows are matched and no exact/R61/PR #323 states enter training.

Primary evaluation is a 120-game seat-balanced direct arena at an identical 384-simulation evaluation budget for both candidates, followed by the frozen forensic-v2 reference with W/D/L utility. The production gate is recorded unchanged; the outcome-aligned shadow gate is a separate report. The R61 expanded descendant panel is secondary only.

The runner writes deterministic six-lane manifests and configs. Result artifacts, checkpoint SHAs, arena outcomes, gate reports, and the final hard classification are intentionally absent until the expensive fresh runs complete.
