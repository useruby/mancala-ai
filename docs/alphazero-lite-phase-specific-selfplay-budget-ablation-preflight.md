# Phase-Specific Self-Play Search-Budget Ablation Preflight

Classification: `phase_specific_budget_ablation_preflight_only`.

## Regenerated Baseline

- The historical 38-row replay source could not be recovered from Git.
- The deterministic current audit regenerated a 20-row selected replay, excluding `opening_missed_extra_turn_continuation`, plus 5 corrected guard controls.
- This is a successor baseline, not byte-identical historical lineage.
- Selected replay SHA-256: `787cca93c7db454c2daebe1fdab001e93df0b9f80a1e7005598a0324774c427b`.
- Guard controls SHA-256: `ffac24dc8fb773fddd4dc14f397317d7c5e0a973631ca71adb216b5e68b444a5`.
- Parent weights SHA-256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`.

## Matched Lanes

| lane | normal simulations | opening simulations | opening plies |
| --- | ---: | ---: | ---: |
| `control_384_192` | 192 | 384 | 8 |
| `opening1200_rest192` | 192 | 1200 | 8 |
| `uniform1200` | 1200 | - | - |

All lanes retain 1,600 games, seed 42, sweep `41,42,43`, denoised sharpened targets, tree reuse, root telemetry, selected replay weight 1, and guard-control replay weight 2. The dry-run successfully rendered all three pipelines. The stored config requests 6 workers, but `pipeline.py` globally normalizes worker-capable commands to 24; actual execution must record 24 rather than claim six-worker provenance.

## Safety

The runner invokes only `pipeline.py` under its supplied `/tmp` work directory. It never invokes a promotion gate or promotion command, and its plan explicitly records `promotion.performed: false`.
