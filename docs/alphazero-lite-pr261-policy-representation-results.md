# PR #261 Policy Representation Results

**Classification:** `policy_basis_changes_fit_not_strength`

## Frozen Contract

- Reused the committed PR #258 ordinary single-replay evidence for seeds 53
  through 62 and each exact 16 x 512 sample plan. No self-play was generated.
- Every lane started at the exact A16 step-16 state, SHA-256
  `f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff`.
- `trunk_adapter` trained only `policy_adapter.{weight,bias}` and
  `policy_readout` trained only `policy_head.{weight,bias}`. Both scopes had
  exactly 582 trainable parameters and used isolated fresh Adam states at
  LR `1e-5`, zero weight decay, beta `.95`, and grad clip `1.0`.
- Each lane used the same fixed row order for steps 1, 4, and 16. Neither the
  trunk, `policy_hidden_layer`, value branch, nor the opposing policy tensor
  was trainable.
- Initial logits/probabilities/value outputs were bit-identical to A16.
  Trunk/value tensors and every eligible-replay value output remained exactly
  unchanged in all candidates. The value-loss gradient fixture also found zero
  contribution to either trainable policy parameter set.
- Repeated lane and reversed-order reproduction passed for seeds 53 and 60,
  including exact step-16 model and optimizer hashes.
- The frozen 20-model set SHA-256 was
  `5fcf7538a750cf90f0cef8ed98c97cc07acb523ae3714cb64e37102f5de0cf3f`.

## Learning And Sealing

- Held-out beta-.95 CE improved from A16 in all `10/10` policy-readout lanes.
  This confirms the inherited nonlinear policy feature readout fit the reused
  targets under the fixed update budget.
- AE/AF/AG used seeds 31042/32042/33042 and had SHA-256 values
  `21012b3a1eb54f1209de34468390a5f0e4ca123fe1b1b6676f6d3def404e2f05`,
  `f5b09723b807ea820c9338f5ae9c07bcbb82fceeaeaa8db2ae7bb509329eeddd`, and
  `d16a01beaae8af6959523e715c5272e6e6c51fbe21a52d0ff86ef096f780fcb2`.
- All three contained 128 openings. Preflight found zero final-state,
  prefix, replay-state, and mutual-suite overlap. They are consumed in the
  authoritative registry.
- Frozen suite-manifest SHA-256:
  `4c65a14fa7943450512f5e7267e99183f4d42971a44792e994a710432d52d7aa`.
- Frozen preflight SHA-256:
  `cd5c8f132251405d74cb17139c81db651ae7b58ceb408a3f16e72686777a7ef2`.

## Primary 1200:1200 Arena

All arenas used the established deterministic ordinary PUCT contract:
`c_puct=1.25`, zero FPU, unnormalized values, seat swaps, arena seed 42, and
opening-aligned P1-vs-P1 controls. Delta is policy-readout effect minus
trunk-adapter effect.

| Replay seed | AE | AF | AG | Pooled delta |
| ---: | ---: | ---: | ---: | ---: |
| 53 | -.003906 | -.001953 | -.003906 | -.003255 |
| 54 | -.003906 | -.001953 | -.003906 | -.003255 |
| 55 | .000000 | .000000 | .000000 | .000000 |
| 56 | .000000 | .000000 | .000000 | .000000 |
| 57 | .000000 | .000000 | .000000 | .000000 |
| 58 | .000000 | .000000 | .000000 | .000000 |
| 59 | .000000 | .000000 | .000000 | .000000 |
| 60 | +.037109 | +.046875 | +.039062 | +.041016 |
| 61 | .000000 | .000000 | .000000 | .000000 |
| 62 | -.037109 | -.046875 | -.039062 | -.041016 |

- Mean delta: `-.000651`; median: `0`; SD: `.019384`; range:
  `[-.041016, +.041016]`; positive seeds: `1/10`.
- Hierarchical replay-seed -> suite -> opening 95% CI:
  `[-.012695, +.010938]`.
- The success rule fails its positive mean, positive lower CI, 8/10 winner,
  and no-material-negative-seed requirements. It therefore cannot establish
  the nonlinear policy-feature basis as stronger.

## Secondary 384:256 Arena

- Mean delta: `+.003906`; median: `0`; range `[-.015625, +.070312]`; positive
  seeds: `1/10`.
- The material shallow-harm gate did not trigger.

## Decision

The readout basis improves held-out target fit, but that fit improvement did
not translate into robust arena strength under matched 582-parameter updates.
The current bottleneck is therefore not simply raw trunk features versus the
inherited nonlinear policy feature representation.

Recommended next: test `policy_hidden_only` as the next larger-capacity policy
ablation under the same multi-seed replay protocol. No model was promoted.
