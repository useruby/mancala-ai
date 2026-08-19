# AlphaZero-Lite Multi-Step Frozen-Trunk Distillation Results

**Classification:** `heads_only_not_useful`

- deterministic reproduction: `True`
- heads-only zero trunk change: `True`
- trainable lanes start from identical incumbent: `True`
- checkpoint steps: `[1, 4, 16, 46]`
- current weights sha256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- replay sha256: `7a3f16b145c144531d73982ac2658df312e7fc84bc57412331ad38e9146ea34a`

## Findings

The frozen-trunk hypothesis is refuted. The `heads_only` lane has zero trunk drift and ~2% policy-output top-1 change (versus ~11% for `all`), yet at the full-pass checkpoint it is *more* game-strength-negative than the full-model lane at 384:256, with the harm concentrated in the P0 seat. Freezing the trunk removes the representation drift but does not prevent the supervised value-head update from degrading low-budget search.

The `all` lane reproduces PR #198's harmful joint-trunk lane byte-for-byte (final state hash `9a15bd97…cc42b87`, trunk delta `0.002055`).

## Supervised objective and drift (frozen validation probe)

| Lane | Step | Total loss | Policy CE | Value huber | Trunk drift | Policy-head drift | Value-head drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| incumbent | 0 | 1.2892 | 1.0831 | 0.3436 | 0.000000 | 0.000000 | 0.000000 |
| all | 1 | 1.2863 | 1.0802 | 0.3435 | 0.000092 | 0.000085 | 0.000098 |
| all | 4 | 1.2791 | 1.0732 | 0.3432 | 0.000262 | 0.000244 | 0.000285 |
| all | 16 | 1.2642 | 1.0591 | 0.3419 | 0.000833 | 0.000816 | 0.001055 |
| all | 46 | 1.2458 | 1.0413 | 0.3409 | 0.002055 | 0.002007 | 0.002989 |
| heads_only | 1 | 1.2891 | 1.0830 | 0.3436 | 0.000000 | 0.000085 | 0.000098 |
| heads_only | 4 | 1.2887 | 1.0826 | 0.3435 | 0.000000 | 0.000249 | 0.000287 |
| heads_only | 16 | 1.2870 | 1.0812 | 0.3430 | 0.000000 | 0.000901 | 0.001044 |
| heads_only | 46 | 1.2833 | 1.0781 | 0.3420 | 0.000000 | 0.002476 | 0.002978 |

## Search diagnostics (384:256 context, versus incumbent)

| Lane | Step | Move change | Visit JS | Q-rank change | Root-value delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 1 | 0.0234 | 0.0010 | -0.0078 | -0.0004 |
| all | 4 | 0.0312 | 0.0035 | +0.0078 | +0.0004 |
| all | 16 | 0.0938 | 0.0089 | +0.0586 | +0.0026 |
| all | 46 | 0.0898 | 0.0141 | +0.0273 | +0.0057 |
| heads_only | 1 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| heads_only | 4 | 0.0078 | 0.0002 | +0.0078 | +0.0001 |
| heads_only | 16 | 0.0312 | 0.0012 | -0.0156 | +0.0006 |
| heads_only | 46 | 0.0352 | 0.0026 | -0.0156 | +0.0011 |

## Canonical arena (candidate versus frozen incumbent)

| Lane | Step | Context | Paired effect | 95% CI | P0 effect | P1 effect |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| all | 1 | 384:256 | -0.0879 | [-0.1230, -0.0547] | -0.1641 | -0.0117 |
| all | 1 | 1200:1200 | +0.0039 | [+0.0000, +0.0098] | +0.0078 | +0.0000 |
| all | 4 | 384:256 | -0.0723 | [-0.1016, -0.0430] | -0.1250 | -0.0195 |
| all | 4 | 1200:1200 | -0.0723 | [-0.0977, -0.0488] | -0.1445 | +0.0000 |
| all | 16 | 384:256 | -0.0898 | [-0.1523, -0.0273] | -0.1680 | -0.0117 |
| all | 16 | 1200:1200 | -0.0527 | [-0.0840, -0.0234] | -0.1523 | +0.0469 |
| all | 46 | 384:256 | -0.0742 | [-0.1230, -0.0254] | -0.1953 | +0.0469 |
| all | 46 | 1200:1200 | -0.1016 | [-0.1445, -0.0586] | -0.2812 | +0.0781 |
| heads_only | 1 | 384:256 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 |
| heads_only | 1 | 1200:1200 | +0.0098 | [+0.0020, +0.0195] | +0.0000 | +0.0195 |
| heads_only | 4 | 384:256 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 |
| heads_only | 4 | 1200:1200 | +0.0137 | [+0.0039, +0.0254] | +0.0078 | +0.0195 |
| heads_only | 16 | 384:256 | -0.0234 | [-0.0410, -0.0039] | -0.0195 | -0.0273 |
| heads_only | 16 | 1200:1200 | +0.0645 | [+0.0469, +0.0840] | +0.0078 | +0.1211 |
| heads_only | 46 | 384:256 | -0.2598 | [-0.3047, -0.2148] | -0.5078 | -0.0117 |
| heads_only | 46 | 1200:1200 | +0.0312 | [+0.0137, +0.0508] | +0.1211 | -0.0586 |

## Classification evidence

| Signal | Value |
| --- | ---: |
| full_harmful_steps | ["1", "4", "16", "46"] |
| heads_safe_where_full_harmful | False |
| heads_nonnegative | True |
| heads_fit_improves | True |
| final_heads_minus_full_ce | 0.0368 |
| material_ce_gap | 0.0100 |

## Next action

`freezing the trunk does not remove the harmful low-budget search effect; the residual harm is driven by the value-head update alone (concentrated in the P0 seat). Test a value-target/search-aware value-head intervention or a dual value representation in a follow-up`

Full evidence: `docs/data/alphazero-lite-frozen-trunk-distillation-summary.json`.
