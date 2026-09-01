# PR #260 Value-Head Refresh Results

**Classification:** `value_refresh_degrades_strength`

## Frozen Contract

- Reused the committed PR #258 ordinary replays for seeds 53 through 62 and
  their exact 16 x 512 single-replay plans. No self-play was generated.
- Every adapter-only step-16 baseline reproduced the committed PR #258 model
  and Adam SHA exactly before its value-only continuation started.
- Each continuation trained only `value_hidden_layer.{weight,bias}` and
  `value_head.{weight,bias}`: 4,705 parameters. The optimizer was a fresh
  isolated Adam at LR `1e-5`, weight decay zero, with 16 fixed batches.
- The terminal row `value` was optimized using the existing `0.6 * Huber`
  objective with delta 1.0. No policy target, policy tensor, or trunk tensor
  was changed.
- The 20 frozen-model set SHA-256 was
  `beb3f9c0451842a1377c21d5bb6f18ec97c34b0426033293ed5afd3f6eb0f8f8`.

## Invariants And Sealing

- Policy adapter, inherited policy, and trunk tensors were bit-identical in
  every baseline/refreshed pair. The maximum PyTorch policy-logit difference
  over all eligible replay rows and the deterministic probe was exactly `0`.
- Artifact round trips passed. Maximum legal-policy error was `2.09e-6` and
  maximum value error was `1.07e-6`, both within the artifact tolerance.
- Held-out Huber improved for 9/10 replay seeds. Seed 62 changed from
  `.311086` to `.311324`; all other seeds improved.
- AB/AC/AD were sealed and consumed with seeds 28042/29042/30042 and SHAs
  `527cc607a3dddbfcaf0203783b58f6b38b9df2fcea2ebc6cbb112d54d7727641`,
  `93786884529f4d75a22a82e5a87074db1f42b5bcb43aaecc42a0abcc28370bcf`, and
  `28b631f74992e81f3227ac3355de6d3cca1ee9724199742039e8331389f98d87`.
  Preflight found zero canonical/final, prefix, replay-state, and mutual
  overlap.

## Primary 1200:1200 Arena

All arenas used the established deterministic ordinary PUCT contract with
`c_puct=1.25`, zero FPU, unnormalized values, seat swaps, arena seed 42, and
matched P1 controls.

| Replay seed | AB | AC | AD | Pooled delta |
| ---: | ---: | ---: | ---: | ---: |
| 53 | -.029297 | -.015625 | -.054688 | -.033203 |
| 54 | -.021484 | -.011719 | -.041016 | -.024740 |
| 55 | -.031250 | -.021484 | -.060547 | -.037760 |
| 56 | -.054688 | -.056641 | -.062500 | -.057943 |
| 57 | +.023438 | +.035156 | +.001953 | +.020182 |
| 58 | -.023438 | -.017578 | -.046875 | -.029297 |
| 59 | +.015625 | +.031250 | -.011719 | +.011719 |
| 60 | -.105469 | -.101562 | -.113281 | -.106771 |
| 61 | +.037109 | +.042969 | +.029297 | +.036458 |
| 62 | +.068359 | +.082031 | +.044922 | +.065104 |

- Mean delta: `-.015625`; median: `-.027018`; SD: `.049919`; range:
  `[-.106771, +.065104]`; positive seeds: `4/10`.
- Hierarchical replay-seed -> suite -> opening 95% CI:
  `[-.047070, +.015820]`.
- The primary success rule fails A, B, C, and D despite satisfying the value
  learning and exact-policy identity requirements.

## Secondary And Search Diagnostics

- Secondary 384:256 mean delta was `+.001888`, with range
  `[-.010417, +.018229]`; the material shallow-harm gate did not trigger.
- On the frozen 256-state diagnostic set, 1200 search move-disagreement was
  1.56%-3.52% and visit-policy JS was `.001801`-.`002397`.
- Across replay seeds, Spearman correlation between 1200 move disagreement and
  arena delta was `+.1168`; for visit JS it was `-.1636`. Search movement did
  not explain strength movement reliably.

## Decision

Short terminal-outcome fitting improves held-out value loss but degrades PUCT
strength on average and has a materially harmful replay-seed tail. Do not
unfreeze or refresh the value branch in production under this recipe. No model
was promoted.
