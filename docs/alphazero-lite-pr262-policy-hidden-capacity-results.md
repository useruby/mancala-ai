# PR #262 Policy Hidden Capacity Results

**Status:** candidate and suite freeze complete; arena classification pending.

## Frozen Contract

- Reused the PR #258/#261 ordinary single-replay evidence for seeds 53 through
  62, including the fixed 16 x 512 batch plans and exclusion masks. No self-play
  was generated.
- All challengers started from A16 SHA-256
  `f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff`.
- The frozen PR #261 `trunk_adapter` candidates were imported only after their
  recorded step-16 model hashes matched. They were not retrained.
- `policy_hidden_only` trains exactly
  `policy_hidden_layer.{weight,bias}`: `96 * 96 + 96 = 9312` parameters. The
  trunk, policy readout, additive adapter, and value branch remain frozen.
- Hidden candidates use fresh isolated Adam, LR `1e-5`, zero weight decay, beta
  `.95`, no scheduler, and gradient clipping `1.0` for exactly 16 updates.
- The policy-plus-value and policy-only gradients were exactly equal for both
  trainable hidden tensors. State mutation and value-output identity checks
  passed at every checkpoint. Seeds 53 and 60 reproduced the exact step-16
  model and optimizer states under repeated/reversed execution.

## Learning

| Replay seed | Held-out beta-.95 CE, A16 | Step 16 | Improvement |
| ---: | ---: | ---: | ---: |
| 53 | .956926893 | .956890292 | +.000036601 |
| 54 | .945680099 | .945644213 | +.000035886 |
| 55 | .941197054 | .941167578 | +.000029476 |
| 56 | .945270326 | .945239170 | +.000031156 |
| 57 | .946161670 | .946128859 | +.000032811 |
| 58 | .941548222 | .941519420 | +.000028801 |
| 59 | .927028261 | .926998090 | +.000030170 |
| 60 | .928121076 | .928083599 | +.000037477 |
| 61 | .928919776 | .928889204 | +.000030572 |
| 62 | .960337837 | .960295212 | +.000042625 |

- Held-out beta-.95 CE improved in `10/10` replay seeds.
- The combined 20-candidate SHA-256 is
  `48c828b98bc8d906d71fb9c7209be3df0889a33096fe14eccddce45ab18a854b`.
- Per-checkpoint CE, raw-target CE, P1 CE, policy drift, logit RMS, update
  scale, model SHA, optimizer SHA, and hidden-minus-adapter CE telemetry are
  frozen in `/tmp/azlite_pr262_policy_hidden_capacity/frozen_candidates.json`.

## Sealed Suites

- AH/AI/AJ were deterministically selected with seeds 34042/35042/36042.
- Each contains 128 openings. Preflight found zero final-state overlap, prefix
  overlap, replay-state overlap, and mutual overlap.
- Their consumed registry SHA-256 values are respectively
  `c16148b43cb652f2dc28ca4b8e94c67f66da471f3cc36d318e73ce3258483784`,
  `1c1de16cc5c4f16696858b054c07747575301e27ff9308f270dfdc4cfd13579b`, and
  `95b3c2dc333a5411562b1a1aeeccb0e093a1af9f7e6f4aa4b61362301416798d`.

## Pending Evaluation

The sealed candidates must next receive the preregistered ordinary `1200:1200`
and secondary `384:256` arenas plus the frozen 256-state search probe. No
strength classification is made until those sealed evaluations complete.
