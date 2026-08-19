# AlphaZero-Lite Frozen-Trunk Head-Isolation Results

**Classification:** `policy_head_accumulation`

- heads_only reproduces PR #199 state hashes: `True`
- all trainable lanes start from identical incumbent: `True`
- policy_head zero trunk change: `True`
- policy_head zero value-stack change: `True`
- value_head zero trunk change: `True`
- value_head zero policy-stack change: `True`
- heads_only zero trunk change: `True`
- checkpoint steps: `[1, 4, 16, 46]`
- current weights sha256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- replay sha256: `7a3f16b145c144531d73982ac2658df312e7fc84bc57412331ad38e9146ea34a`

## Findings

Accumulated policy-head drift is the primary mechanism. The `policy_head` lane reproduces the PR #199 heads-only low-budget failure at 384:256 while the `value_head` lane is substantially safer.

At the final checkpoint (step 46) the 384:256 paired effects are: `heads_only` -0.2598 (P0 -0.5078), `policy_head` -0.1895 (P0 -0.3672), `value_head` +0.0137 (P0 -0.0195).

The `value_head` lane is transiently negative at steps 4, 16 before recovering to neutral or positive at the final checkpoint, so accumulated value drift alone does not cause the terminal low-budget failure; the policy-head drift does.

## Supervised objective and parameter drift (frozen validation probe)

| Lane | Step | Total loss | Policy CE | Value huber | Trunk drift | Policy-head drift | Value-head drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent | 0 | 1.2892 | 1.0831 | 0.3436 | 0.000000 | 0.000000 | 0.000000 |
| heads_only | 1 | 1.2891 | 1.0830 | 0.3436 | 0.000000 | 0.000085 | 0.000098 |
| heads_only | 4 | 1.2887 | 1.0826 | 0.3435 | 0.000000 | 0.000249 | 0.000287 |
| heads_only | 16 | 1.2870 | 1.0812 | 0.3430 | 0.000000 | 0.000901 | 0.001044 |
| heads_only | 46 | 1.2833 | 1.0781 | 0.3420 | 0.000000 | 0.002476 | 0.002978 |
| policy_head | 1 | 1.2891 | 1.0830 | 0.3436 | 0.000000 | 0.000085 | 0.000000 |
| policy_head | 4 | 1.2887 | 1.0826 | 0.3436 | 0.000000 | 0.000249 | 0.000000 |
| policy_head | 16 | 1.2874 | 1.0812 | 0.3436 | 0.000000 | 0.000901 | 0.000000 |
| policy_head | 46 | 1.2842 | 1.0781 | 0.3436 | 0.000000 | 0.002476 | 0.000000 |
| value_head | 1 | 1.2892 | 1.0831 | 0.3436 | 0.000000 | 0.000000 | 0.000098 |
| value_head | 4 | 1.2892 | 1.0831 | 0.3435 | 0.000000 | 0.000000 | 0.000287 |
| value_head | 16 | 1.2889 | 1.0831 | 0.3430 | 0.000000 | 0.000000 | 0.001044 |
| value_head | 46 | 1.2883 | 1.0831 | 0.3420 | 0.000000 | 0.000000 | 0.002978 |
| all | 1 | 1.2863 | 1.0802 | 0.3435 | 0.000092 | 0.000085 | 0.000098 |
| all | 4 | 1.2791 | 1.0732 | 0.3432 | 0.000262 | 0.000244 | 0.000285 |
| all | 16 | 1.2642 | 1.0591 | 0.3419 | 0.000833 | 0.000816 | 0.001055 |
| all | 46 | 1.2458 | 1.0413 | 0.3409 | 0.002055 | 0.002007 | 0.002989 |

## Network-output drift on the frozen probe (versus incumbent)

| Lane | Step | Top-1 change | Legal L1 | Legal JS | Value mean abs delta | Value signed delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent | 0 | 0.0000 | 0.0000 | 0.000000 | 0.0000 | +0.0000 |
| heads_only | 1 | 0.0000 | 0.0006 | 0.000000 | 0.0004 | -0.0001 |
| heads_only | 4 | 0.0020 | 0.0018 | 0.000001 | 0.0018 | +0.0010 |
| heads_only | 16 | 0.0059 | 0.0072 | 0.000012 | 0.0072 | +0.0047 |
| heads_only | 46 | 0.0195 | 0.0203 | 0.000091 | 0.0227 | +0.0199 |
| policy_head | 1 | 0.0000 | 0.0006 | 0.000000 | 0.0000 | +0.0000 |
| policy_head | 4 | 0.0020 | 0.0018 | 0.000001 | 0.0000 | +0.0000 |
| policy_head | 16 | 0.0059 | 0.0072 | 0.000012 | 0.0000 | +0.0000 |
| policy_head | 46 | 0.0195 | 0.0203 | 0.000091 | 0.0000 | +0.0000 |
| value_head | 1 | 0.0000 | 0.0000 | 0.000000 | 0.0004 | -0.0001 |
| value_head | 4 | 0.0000 | 0.0000 | 0.000000 | 0.0018 | +0.0010 |
| value_head | 16 | 0.0000 | 0.0000 | 0.000000 | 0.0072 | +0.0047 |
| value_head | 46 | 0.0000 | 0.0000 | 0.000000 | 0.0227 | +0.0199 |
| all | 1 | 0.0059 | 0.0096 | 0.000026 | 0.0020 | -0.0016 |
| all | 4 | 0.0273 | 0.0314 | 0.000284 | 0.0063 | -0.0030 |
| all | 16 | 0.0664 | 0.0799 | 0.001782 | 0.0197 | -0.0055 |
| all | 46 | 0.1074 | 0.1142 | 0.003622 | 0.0544 | +0.0178 |

## Search diagnostics (384:256 context, versus incumbent)

| Lane | Step | Move change | Visit JS | Q-rank change | Root-value delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| heads_only | 1 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| heads_only | 4 | 0.0078 | 0.0002 | +0.0078 | +0.0001 |
| heads_only | 16 | 0.0312 | 0.0012 | -0.0156 | +0.0006 |
| heads_only | 46 | 0.0352 | 0.0026 | -0.0156 | +0.0011 |
| policy_head | 1 | 0.0039 | 0.0000 | -0.0117 | -0.0000 |
| policy_head | 4 | 0.0039 | 0.0002 | -0.0039 | -0.0001 |
| policy_head | 16 | 0.0156 | 0.0004 | -0.0078 | -0.0001 |
| policy_head | 46 | 0.0273 | 0.0016 | +0.0273 | +0.0001 |
| value_head | 1 | 0.0000 | 0.0000 | +0.0000 | -0.0000 |
| value_head | 4 | 0.0078 | 0.0001 | +0.0156 | +0.0001 |
| value_head | 16 | 0.0273 | 0.0009 | +0.0000 | +0.0005 |
| value_head | 46 | 0.0352 | 0.0009 | -0.0078 | +0.0009 |
| all | 1 | 0.0234 | 0.0010 | -0.0078 | -0.0004 |
| all | 4 | 0.0312 | 0.0035 | +0.0078 | +0.0004 |
| all | 16 | 0.0938 | 0.0089 | +0.0586 | +0.0026 |
| all | 46 | 0.0898 | 0.0141 | +0.0273 | +0.0057 |

## Canonical arena (candidate versus frozen incumbent)

| Lane | Step | Context | Paired effect | 95% CI | P0 effect | P1 effect |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| heads_only | 1 | 384:256 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 |
| heads_only | 1 | 1200:1200 | +0.0098 | [+0.0020, +0.0195] | +0.0000 | +0.0195 |
| heads_only | 4 | 384:256 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 |
| heads_only | 4 | 1200:1200 | +0.0137 | [+0.0039, +0.0254] | +0.0078 | +0.0195 |
| heads_only | 16 | 384:256 | -0.0234 | [-0.0410, -0.0039] | -0.0195 | -0.0273 |
| heads_only | 16 | 1200:1200 | +0.0645 | [+0.0469, +0.0840] | +0.0078 | +0.1211 |
| heads_only | 46 | 384:256 | -0.2598 | [-0.3047, -0.2148] | -0.5078 | -0.0117 |
| heads_only | 46 | 1200:1200 | +0.0312 | [+0.0137, +0.0508] | +0.1211 | -0.0586 |
| policy_head | 1 | 384:256 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 |
| policy_head | 1 | 1200:1200 | +0.0098 | [+0.0020, +0.0195] | +0.0000 | +0.0195 |
| policy_head | 4 | 384:256 | -0.0215 | [-0.0352, -0.0098] | +0.0000 | -0.0430 |
| policy_head | 4 | 1200:1200 | +0.0137 | [+0.0039, +0.0254] | +0.0078 | +0.0195 |
| policy_head | 16 | 384:256 | -0.0098 | [-0.0254, +0.0059] | -0.0352 | +0.0156 |
| policy_head | 16 | 1200:1200 | -0.0078 | [-0.0156, -0.0020] | -0.0156 | +0.0000 |
| policy_head | 46 | 384:256 | -0.1895 | [-0.2324, -0.1465] | -0.3672 | -0.0117 |
| policy_head | 46 | 1200:1200 | -0.0605 | [-0.0801, -0.0430] | -0.1211 | +0.0000 |
| value_head | 1 | 384:256 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 |
| value_head | 1 | 1200:1200 | +0.0098 | [+0.0020, +0.0195] | +0.0000 | +0.0195 |
| value_head | 4 | 384:256 | -0.1211 | [-0.1602, -0.0859] | -0.2422 | +0.0000 |
| value_head | 4 | 1200:1200 | +0.0039 | [+0.0000, +0.0098] | +0.0078 | +0.0000 |
| value_head | 16 | 384:256 | -0.1133 | [-0.1523, -0.0742] | -0.2422 | +0.0156 |
| value_head | 16 | 1200:1200 | -0.0703 | [-0.0898, -0.0508] | -0.0195 | -0.1211 |
| value_head | 46 | 384:256 | +0.0137 | [-0.0059, +0.0352] | -0.0195 | +0.0469 |
| value_head | 46 | 1200:1200 | -0.0605 | [-0.0801, -0.0430] | +0.0195 | -0.1406 |
| all | 1 | 384:256 | -0.0879 | [-0.1230, -0.0547] | -0.1641 | -0.0117 |
| all | 1 | 1200:1200 | +0.0039 | [+0.0000, +0.0098] | +0.0078 | +0.0000 |
| all | 4 | 384:256 | -0.0723 | [-0.1016, -0.0430] | -0.1250 | -0.0195 |
| all | 4 | 1200:1200 | -0.0723 | [-0.0977, -0.0488] | -0.1445 | +0.0000 |
| all | 16 | 384:256 | -0.0898 | [-0.1523, -0.0273] | -0.1680 | -0.0117 |
| all | 16 | 1200:1200 | -0.0527 | [-0.0840, -0.0234] | -0.1523 | +0.0469 |
| all | 46 | 384:256 | -0.0742 | [-0.1230, -0.0254] | -0.1953 | +0.0469 |
| all | 46 | 1200:1200 | -0.1016 | [-0.1445, -0.0586] | -0.2812 | +0.0781 |

## Classification evidence

| Signal | Value |
| --- | ---: |
| material_effect_threshold | -0.0500 |
| primary_context | "384:256" |
| final_step | 46 |
| final_paired_effects | {"heads_only": "-0.2598", "policy_head": "-0.1895", "value_head": "+0.0137"} |
| final_p0_effects | {"heads_only": "-0.5078", "policy_head": "-0.3672", "value_head": "-0.0195"} |
| final_p1_effects | {"heads_only": "-0.0117", "policy_head": "-0.0117", "value_head": "+0.0469"} |
| final_ci_upper_95 | {"heads_only": "-0.2148", "policy_head": "-0.1465", "value_head": "+0.0352"} |
| heads_only_reproduces_failure | True |
| value_head_reproduces_failure | False |
| policy_head_reproduces_failure | True |
| materially_harmful_steps | {"heads_only": ["46"], "policy_head": ["46"], "value_head": ["4", "16"]} |
| p0_failure_concentrated_in_value_head | False |

## Recommended next experiment (not implemented here)

`run a policy-prior/low-budget-search intervention under this same 46-step replay: behavior anchoring against the incumbent policy or constrained prior drift (a trust region on legal-policy divergence from the incumbent)`

## Exact commands

```bash
python ml/alphazero_lite/run_frozen_trunk_head_isolation_ablation.py \
  --pr191-workdir /tmp/azlite_shared_trunk_learning \
  --workdir <workdir> --include-all --arena-workers 24
```

Full evidence: `docs/data/alphazero-lite-frozen-trunk-head-isolation-summary.json`.
