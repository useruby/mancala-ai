# AlphaZero-Lite Frozen-Trunk Same-State Anchor Results

**Classification:** `safe_learning_window_found`

- beta_000 reproduces PR #200 policy_head hashes: `True`
- beta_100 incumbent-equivalent: `True`
- all lanes start from identical incumbent: `True`
- trunk byte-identical to incumbent (all lanes): `True`
- value stack byte-identical to incumbent (all lanes): `True`
- checkpoint steps: `[1, 4, 16, 46]`
- current weights sha256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- replay sha256: `7a3f16b145c144531d73982ac2658df312e7fc84bc57412331ad38e9146ea34a`
- initialization checkpoint sha256: `4cd77f12319935d776b68c5e597b8d399fd68f15ba7ea25b3d66bbe786292ed4`
- seed: `42`
- optimizer: `{"lr": 1e-05, "type": "Adam", "weight_decay": 0.0}`
- gradient clip: `1.0`
- trainable scope: `policy_head`

## Target construction

For every training replay state x: `p_beta(x) = (1-beta) p_search(x) + beta p_inc(x)` with `p_inc(x)` the frozen incumbent legal-masked policy on that exact state (masking identical to the policy loss). Illegal mass zeroed and renormalized over legal moves. beta_000 uses the unmodified search target (byte-identical to PR #200).

## Training / target metrics (frozen validation probe, step 46)

| Lane | CE(search) | CE(incumbent) | CE(mixed) | search-CE improv | fit_fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| beta_000 | 1.0781 | 0.9285 | 1.0781 | +0.0050 | 1.0000 |
| beta_050 | 1.0783 | 0.9284 | 1.0034 | +0.0048 | 0.9463 |
| beta_080 | 1.0791 | 0.9283 | 0.9585 | +0.0040 | 0.7933 |
| beta_095 | 1.0807 | 0.9282 | 0.9358 | +0.0024 | 0.4715 |
| beta_100 | 1.0831 | 0.9281 | 0.9281 | +0.0000 | 0.0001 |
fit_fraction = (search_CE_inc - search_CE_candidate) / (search_CE_inc - search_CE_beta000_step46). Reported `n/a` if the denominator is <= 0.

## Policy drift vs incumbent (frozen validation probe, step 46)

| Lane | L1 mean | L1 max | L1 p50 | L1 p90 | L1 p95 | L1 p99 | JS mean | top-1 change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| beta_000 | 0.020321 | 0.057555 | 0.021490 | 0.035426 | 0.038912 | 0.044751 | 0.000091 | 0.0195 |
| beta_050 | 0.018649 | 0.053406 | 0.019439 | 0.032634 | 0.035535 | 0.041095 | 0.000077 | 0.0176 |
| beta_080 | 0.014086 | 0.041581 | 0.014347 | 0.024793 | 0.027789 | 0.031502 | 0.000044 | 0.0156 |
| beta_095 | 0.005534 | 0.017643 | 0.005238 | 0.010449 | 0.012210 | 0.015122 | 0.000008 | 0.0078 |
| beta_100 | 0.000016 | 0.000081 | 0.000015 | 0.000033 | 0.000037 | 0.000050 | 0.000000 | 0.0000 |

## Policy drift vs search target (frozen validation probe, step 46)

| Lane | L1 mean | L1 max | JS mean | top-1 change |
| --- | ---: | ---: | ---: | ---: |
| beta_000 | 0.832424 | 1.978129 | 0.181688 | 0.4023 |
| beta_050 | 0.832742 | 1.977995 | 0.181777 | 0.4023 |
| beta_080 | 0.833624 | 1.977604 | 0.182027 | 0.4023 |
| beta_095 | 0.835359 | 1.976737 | 0.182510 | 0.3984 |
| beta_100 | 0.836257 | 1.976903 | 0.182852 | 0.4043 |

## Search diagnostics (384:256 context, vs incumbent)

| Lane | Step | Move change | Visit JS | Q-rank change | Root-value delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| beta_000 | 1 | 0.0039 | 0.0000 | -0.0117 | -0.0000 |
| beta_000 | 4 | 0.0039 | 0.0002 | -0.0039 | -0.0001 |
| beta_000 | 16 | 0.0156 | 0.0004 | -0.0078 | -0.0001 |
| beta_000 | 46 | 0.0273 | 0.0016 | +0.0273 | +0.0001 |
| beta_050 | 1 | 0.0039 | 0.0000 | -0.0117 | -0.0000 |
| beta_050 | 4 | 0.0039 | 0.0002 | -0.0039 | -0.0001 |
| beta_050 | 16 | 0.0156 | 0.0004 | -0.0078 | -0.0000 |
| beta_050 | 46 | 0.0312 | 0.0014 | +0.0391 | +0.0002 |
| beta_080 | 1 | 0.0039 | 0.0000 | -0.0117 | -0.0000 |
| beta_080 | 4 | 0.0039 | 0.0002 | -0.0039 | -0.0001 |
| beta_080 | 16 | 0.0117 | 0.0004 | -0.0039 | +0.0000 |
| beta_080 | 46 | 0.0312 | 0.0012 | +0.0117 | +0.0001 |
| beta_095 | 1 | 0.0039 | 0.0000 | -0.0117 | -0.0000 |
| beta_095 | 4 | 0.0039 | 0.0002 | -0.0039 | -0.0001 |
| beta_095 | 16 | 0.0117 | 0.0003 | -0.0156 | -0.0003 |
| beta_095 | 46 | 0.0234 | 0.0004 | -0.0117 | +0.0001 |
| beta_100 | 1 | 0.0000 | 0.0000 | +0.0000 | -0.0000 |
| beta_100 | 4 | 0.0000 | 0.0000 | +0.0000 | +0.0001 |
| beta_100 | 16 | 0.0000 | 0.0000 | +0.0039 | +0.0000 |
| beta_100 | 46 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |

## Per-depth policy L1/JS on expanded probe states (step 46, 384:256, candidate vs incumbent)

| Lane | Depth | Expanded | L1 mean | JS mean |
| --- | ---: | ---: | ---: | ---: |
| beta_000 | 0 | 256 | 0.019738 | 0.000085 |
| beta_000 | 1 | 952 | 0.020183 | 0.000086 |
| beta_000 | 2 | 3005 | 0.021172 | 0.000090 |
| beta_000 | 3 | 7068 | 0.021236 | 0.000090 |
| beta_000 | 4 | 65304 | 0.019678 | 0.000084 |
| beta_050 | 0 | 256 | 0.018102 | 0.000072 |
| beta_050 | 1 | 952 | 0.018515 | 0.000073 |
| beta_050 | 2 | 3005 | 0.019361 | 0.000076 |
| beta_050 | 3 | 7068 | 0.019410 | 0.000075 |
| beta_050 | 4 | 65304 | 0.017989 | 0.000070 |
| beta_080 | 0 | 256 | 0.013645 | 0.000041 |
| beta_080 | 1 | 952 | 0.013954 | 0.000042 |
| beta_080 | 2 | 3005 | 0.014425 | 0.000043 |
| beta_080 | 3 | 7068 | 0.014431 | 0.000043 |
| beta_080 | 4 | 65304 | 0.013381 | 0.000040 |
| beta_095 | 0 | 256 | 0.005395 | 0.000007 |
| beta_095 | 1 | 952 | 0.005623 | 0.000008 |
| beta_095 | 2 | 3005 | 0.005684 | 0.000007 |
| beta_095 | 3 | 7068 | 0.005656 | 0.000007 |
| beta_095 | 4 | 65304 | 0.005213 | 0.000007 |
| beta_100 | 0 | 256 | 0.000017 | 0.000000 |
| beta_100 | 1 | 952 | 0.000017 | 0.000000 |
| beta_100 | 2 | 3005 | 0.000018 | 0.000000 |
| beta_100 | 3 | 7068 | 0.000017 | 0.000000 |
| beta_100 | 4 | 65304 | 0.000016 | 0.000000 |

## Canonical arena (candidate versus frozen incumbent)

| Lane | Step | Context | Paired effect | 95% CI | P0 | P1 | W/D/L |
| --- | ---: | --- | ---: | --- | ---: | ---: | --- |
| beta_000 | 16 | 384:256 | -0.0098 | [-0.0254, +0.0059] | -0.0352 | +0.0156 | 396/34/82 |
| beta_000 | 46 | 384:256 | -0.1895 | [-0.2324, -0.1465] | -0.3672 | -0.0117 | 302/38/172 |
| beta_050 | 16 | 384:256 | -0.0098 | [-0.0254, +0.0059] | -0.0352 | +0.0156 | 396/34/82 |
| beta_050 | 46 | 384:256 | -0.0684 | [-0.0996, -0.0371] | -0.1250 | -0.0117 | 364/38/110 |
| beta_080 | 16 | 384:256 | -0.0234 | [-0.0410, -0.0039] | -0.0352 | -0.0117 | 382/48/82 |
| beta_080 | 46 | 384:256 | -0.1094 | [-0.1484, -0.0703] | -0.1914 | -0.0273 | 346/32/134 |
| beta_095 | 16 | 384:256 | -0.0215 | [-0.0430, -0.0020] | -0.0156 | -0.0273 | 384/46/82 |
| beta_095 | 46 | 384:256 | -0.0137 | [-0.0293, +0.0039] | -0.0156 | -0.0117 | 392/38/82 |
| beta_095 | 46 | 1200:1200 | +0.0039 | [+0.0000, +0.0098] | +0.0078 | +0.0000 | 184/148/180 |
| beta_100 | 16 | 384:256 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 | 402/32/78 |
| beta_100 | 46 | 384:256 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 | 402/32/78 |

## Classification evidence

| Signal | Value |
| --- | ---: |
| beta_000_reproduces_pr200 | True |
| beta_100_incumbent_equivalent | True |
| step46_384_256_effects | {"beta_000": "-0.1895", "beta_050": "-0.0684", "beta_080": "-0.1094", "beta_095": "-0.0137", "beta_100": "+0.0000"} |
| step46_fit_fractions | {"beta_000": "+1.0000", "beta_050": "+0.9463", "beta_080": "+0.7933", "beta_095": "+0.4715", "beta_100": "+0.0001"} |
| arena_safe_lanes_lt1 | ["beta_095"] |
| step46_384_256_monotonic | False |
| meaningful_teacher_fit_lanes_anchored | ["beta_050", "beta_080", "beta_095"] |
| meaningful_teacher_fit_lanes_lt1 | ["beta_000", "beta_050", "beta_080", "beta_095"] |
| safe_and_fit_lanes | ["beta_095"] |
| high_budget_regression | {"beta_095": false} |
| safe_window_lanes | ["beta_095"] |
| noninferiority_lower | -0.03 |
| meaningful_fit_fraction | 0.25 |

## Recommended next experiment (not implemented here)

`test the winning beta in one normal AlphaZero-style iteration with fresh self-play (NOT implemented in this PR)`

## Exact commands

```bash
python ml/alphazero_lite/run_frozen_trunk_same_state_anchor_ablation.py \
  --pr191-workdir /tmp/azlite_shared_trunk_learning \
  --workdir /tmp/azlite_frozen_trunk_same_state_anchor \
  --arena-workers 24
```

Full evidence: `docs/data/alphazero-lite-frozen-trunk-same-state-anchor-summary.json`.
