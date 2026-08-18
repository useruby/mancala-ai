# AlphaZero-Lite Policy-Detached Trunk Results

**Classification:** `policy_trunk_gradient_inconclusive_stop`

- Deterministic detached replay through step 12: `True`
- Full-epoch continuation: `False` (`prespecified_continuation_gate_failed`)

## Matched-Current Arena

`policy_trunk_gradient_effect` is joint minus detached. Positive values favor joint; negative values favor detached.

| Step | Context | Heads-current | Detached-current | Joint-current | Joint-detached 95% CI |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | 384:256 | +0.0000 | -0.0137 | -0.0879 | [-0.1074, -0.0410] |
| 1 | 1200:1200 | +0.0098 | -0.0039 | +0.0039 | [+0.0020, +0.0156] |
| 3 | 384:256 | -0.1211 | -0.0195 | -0.0801 | [-0.0938, -0.0273] |
| 3 | 1200:1200 | -0.0508 | +0.0098 | +0.0039 | [-0.0156, +0.0039] |
| 5 | 384:256 | +0.0000 | -0.0117 | -0.0176 | [-0.0234, +0.0098] |
| 5 | 1200:1200 | +0.0137 | +0.0410 | -0.0586 | [-0.1309, -0.0684] |
| 12 | 384:256 | -0.0059 | -0.3242 | -0.2441 | [+0.0273, +0.1328] |
| 12 | 1200:1200 | +0.0039 | +0.0020 | -0.1133 | [-0.1680, -0.0645] |

The step-12 treatment contrast reverses between contexts, so the required persistent detached advantage is not established. No full-epoch continuation was run.

Fixed-step parameter/output attribution, frozen PUCT probe metrics, provenance-bound arena records, and opening-level bootstrap effects are in `docs/data/alphazero-lite-policy-detached-trunk-summary.json`.
