# PR #245 Subtree-Reuse Mechanism Isolation

**Classification:** `extra_effective_search_budget_explains_reused_gain`

## Frozen Contract

- Ordinary replay SHA-256 / eligible rows: `6671e248af4a4c82e1155c798cb7490cd66cd80dc10b203c97d89dced94527f2` / 27,350.
- A16 serialized snapshot SHA-256: `f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff`.
- P1 checkpoint SHA-256: `e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9`.
- Canonical suite SHA-256: `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`.
- Initial Adam SHA-256: `61d5719e75aae87d7c2ca7ed2c5b01871ac2ea1675a34c4a6c918c783894e8c7`.
- No trajectory, state, outcome, non-policy replay label, hyperparameter, root-noise setting, or evaluation setting changed.

## Reconstruction Invariants

- All 700 original games reconstructed sequentially from the original 24-worker, seed-44 RNG contract.
- Every reconstructed reused policy matched the committed policy, and every sampled move, successor state, game length, and winner matched the replay.
- The three searches received identical root priors from their cloned pre-search RNG state. Ply 0 was identical for reused, fresh-384, and fresh-total in every game.
- `original_reused` reproduced PR #245 model, optimizer, adapter, and CE hashes at steps 1, 4, and 16.
- The pristine optimizer stayed unchanged; a repeated `fresh_matched_384` lane was byte-identical.

## Derived Replays

| View | Replay SHA-256 |
| --- | --- |
| original reused | `3a5e878b953497425c48f3683867179e8ee8095c6215a4092b37e0ae87f056f9` |
| fresh matched 384 | `1d3d3b455680ad61393fb1402c103bc6221e098cb2fb5aad0ee773e4e7d2f41c` |
| fresh matched total | `76c6aadf6a4003b99a3b5aa190c74edabc718ba40398fa195c8bd85330421998` |

Telemetry SHA-256: `ec4440cd718db0e53e149ec69918a78f1459468357d0ebae01f1b72f92acfbf4`.

## Effective Search

| Metric | Value |
| --- | ---: |
| Effective multiplier mean / p50 / p90 / p99 / max | 3.1773 / 2.8464 / 5.3961 / 8.6355 / 15.1979 |
| Matched fresh simulations mean / p50 / p90 / p99 / max | 1220.08 / 1093 / 2072.1 / 3316.04 / 5836 |
| Fresh-total JS closer to original than fresh-384 | 89.91% |
| Fresh-total top-1 newly toward original | 13.10% |

Fresh-total was nearly the original target: mean legal L1 `0.002274`, mean JS `0.000227`, and top-1 disagreement `0.001133`. In contrast, original versus fresh-384 had mean legal L1 `0.242464`, mean JS `0.059374`, and top-1 disagreement `0.131956`.

## Step-16 Arena

All rows use ordinary PUCT, 128 openings, seat swapping, seed 42, matched P1 controls, and 10,000 opening bootstraps.

| Lane | 384:256 effect / CI / W-D-L | 1200:1200 effect / CI / W-D-L |
| --- | --- | --- |
| original reused | `-0.019531` `[-0.039062,-0.003906]` 390/38/84 | `+0.041016` `[+0.013672,+0.066406]` 242/70/200 |
| fresh matched 384 | `-0.019531` `[-0.039062,-0.003906]` 390/38/84 | `-0.019531` `[-0.039062,-0.003906]` 180/132/200 |
| fresh matched total | `-0.019531` `[-0.039062,-0.003906]` 390/38/84 | `+0.041016` `[+0.013672,+0.066406]` 242/70/200 |

At 1200:1200, both original-reused minus fresh-384 and fresh-total minus fresh-384 were `+0.060547` with paired CI `[+0.042969,+0.080078]`. Original-reused minus fresh-total was exactly `0.0` with paired CI `[0.0,0.0]`.

## Interpretation

Fresh matched-total recovers the original target geometry and the complete 1200:1200 gain, while matched-RNG fresh-384 remains negative. Tree reuse therefore helps primarily by contributing already accumulated root visit evidence, rather than residual path-dependent Q or deeper-subtree information.

**Recommended next:** prospectively generate ordinary targets from fresh searches using the equivalent per-root compute budget, without carrying path-dependent reused statistics.
