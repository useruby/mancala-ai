# Value-Scale Causal Preflight

**Classification:** `margin_value_semantics_rejected_for_search`

The full aggregate evidence and provenance manifest are in `docs/data/alphazero-lite-value-scale-causal-preflight-summary.json`.

| Independent evaluation domain (n=240) | Current | Margin affine |
| --- | ---: | ---: |
| Normalized-margin MAE | 0.209936 | 0.182972 |

Margin-affine MAE improved by 12.8%, but its primary forced-move causal evidence was decisively negative. The outcome-affine result remains a separate secondary finding: `outcome_value_scale_unresolved`; it cannot suppress the primary rejection. No value-head training is authorized.
