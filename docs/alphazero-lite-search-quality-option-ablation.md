# Search-Quality Option Ablation Matrix

Config: `ml/alphazero_lite/configs/aggressive_v2_search_quality_local.json`

| Variant | Flags enabled | Arena score | MCTS1200 score | Runtime | Pass/fail | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| parent_q only | `--fpu-mode parent_q`<br>`--root-policy-mode deterministic`<br>`--tactical-root-bias 0.1` | TBD | TBD | TBD | TBD | candidate=`/tmp/azlite_v2_search_quality_local_versions/aggressive-v2-search-quality-local-iter1`; current=`storage/ai/alphazero_lite/current` |
| normalize-values only | `--normalize-values`<br>`--root-policy-mode deterministic`<br>`--tactical-root-bias 0.1` | TBD | TBD | TBD | TBD | candidate=`/tmp/azlite_v2_search_quality_local_versions/aggressive-v2-search-quality-local-iter1`; current=`storage/ai/alphazero_lite/current` |
| reuse-subtree only | `--reuse-subtree`<br>`--root-policy-mode deterministic`<br>`--tactical-root-bias 0.1` | TBD | TBD | TBD | TBD | candidate=`/tmp/azlite_v2_search_quality_local_versions/aggressive-v2-search-quality-local-iter1`; current=`storage/ai/alphazero_lite/current` |
| tactical-root-bias only | `--root-policy-mode deterministic`<br>`--tactical-root-bias 0.1` | TBD | TBD | TBD | TBD | candidate=`/tmp/azlite_v2_search_quality_local_versions/aggressive-v2-search-quality-local-iter1`; current=`storage/ai/alphazero_lite/current` |
| all search-quality flags | `--fpu-mode parent_q`<br>`--reuse-subtree`<br>`--normalize-values`<br>`--root-policy-mode deterministic`<br>`--tactical-root-bias 0.1` | TBD | TBD | TBD | TBD | candidate=`/tmp/azlite_v2_search_quality_local_versions/aggressive-v2-search-quality-local-iter1`; current=`storage/ai/alphazero_lite/current` |
