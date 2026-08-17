# AlphaZero-Lite Value-Protected Trunk Ablation

**Classification:** `continuation_gate_failed`

- baseline PR191 hash reproduced: `True`
- protected steps completed: `12`
- projection firing rate through step 12: `0.7500`
- continuation gate: `False`
- continuation decision: `prespecified_step_12_arena_gate_failed`

## Step-12 Arena Gate

| Context | Protected - baseline | Protected - current |
| --- | ---: | ---: |
| 384:256 | +0.2051 | -0.2441 |
| 1200:1200 | +0.1230 | -0.0527 |

Full replay snapshots (including optimizer states), real-batch conflict telemetry, frozen-probe metrics, and paired arena evidence are in `docs/data/alphazero-lite-value-protected-trunk-ablation-summary.json`.
