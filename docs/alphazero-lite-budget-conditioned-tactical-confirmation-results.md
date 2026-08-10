# Budget-Conditioned Tactical Confirmation

Classification: `runtime_profile_resolver_or_seed_contract_failed`.

Decision contract SHA256: `749125a3f6a540cc1649a227a773b5003547388e808d31a488dcb25f05f9b0ce`.

| Budget | Mean DS | Median | Worst suite | Hierarchical 95% CI |
|---|---:|---:|---:|---|
| 768:256 | +0.0345 | +0.0342 | +0.0312 | [+0.0292, +0.0397] |
| 1200:1200 | +0.0105 | +0.0107 | +0.0046 | [-0.0011, +0.0217] |
| 1200:256 | +0.0461 | +0.0467 | +0.0420 | [+0.0408, +0.0514] |
| 256:768 | +0.0339 | +0.0334 | +0.0312 | [+0.0288, +0.0391] |

Suite independence audit: `passed`.
Expected-null errors: `1`.

The expected exact null `E_minus_E` at `768:768` on
`heldout_seed49_large` was nonzero. This is classified as
`runtime_profile_resolver_or_seed_contract_failed`; the runtime-profile gate
was not run and the changed-budget estimates are not promotion evidence.
