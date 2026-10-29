# Seed542 current-runtime fixed-update C/D comparison v2

Status: **current-runtime training and development evaluation completed**. This is a new prospective current-runtime production-control experiment, not an exact historical seed504 reproduction.

The original attempt had an update-indexing error. The corrected runner matched production training across 108 diagnostic updates. Current-runtime production did not reproduce historical seed504 E1 bitwise; the cause remains unknown. No attribution is made to PyTorch, threading, or hardware. D was not executed in the prior attempts.

This registration consumes the unchanged seed542 published per-arm one-based schedules and checks each against the merged zero-based schedule view before execution. The training contract is fixed at 108 updates and 54,896 exposures per arm, batches `[512] * 26 + [412]` repeated four times, residual_v3/kalah_v3 (96, 3), legal policy CE + 0.5 × Huber(value, delta=1), fresh Adam (lr 0.0001, betas (0.9,0.999), eps 1e-8), no weight decay or scheduler, and gradient clipping 1.0.

The unmodified seed504 production runner is used as the current-runtime reference. Its initializer path is relocated at runtime to the authenticated, byte-identical seed542 initializer. Non-mutating callbacks record the production update and current direct-C update for exact all-step comparison. Historical seed504 checkpoint differences are descriptive only. D execution status: `True`.

Registration and evidence are in this directory. `execution-binding.json` records numerical settings and resource limits. `production-reference/` contains the production runner outputs and update trace; `C-updates.jsonl` and `D-updates.jsonl` contain row identities and update telemetry; boundary checkpoints are retained at 27, 54, 81, and 108. `development-predictions.jsonl` and `development-metrics.json` cover only the unchanged retained membership and are descriptive; negative D-minus-control loss differences indicate improvement. No development threshold is used.

The intended later composition remains D-final policy/trunk with the frozen initializer value/trunk. Trained value metrics are diagnostics only. No arena, promotion, serving export, or deployment is part of this experiment.
