# PR #262 Policy Hidden Capacity Results

**Status:** complete. Classification: `policy_hidden_degrades_strength`.

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

## Evaluation Contract

- The candidate audit passed for all 20 fixed artifacts: ten
  `seedXX_trunk_adapter` and ten `seedXX_policy_hidden` state SHA-256 values
  reproduced the frozen aggregate
  `48c828b98bc8d906d71fb9c7209be3df0889a33096fe14eccddce45ab18a854b`.
  Exported artifact paths and SHA-256 values are in
  `docs/data/alphazero-lite-pr262-policy-hidden-capacity-summary.json`.
- AH/AI/AJ exactly matched their sealed SHA-256 values. Each had 128 openings
  and zero consumed-suite, prefix, replay-state, and mutual overlap. The frozen
  suite-manifest and preflight hashes remained respectively
  `ad077c90dc36deafec19512320e6ddfe9077cba33701c5171615b7f35a68ca9e` and
  `6d761626b5b9a17a4efaa6b1f41f806cf4254aabe4ebdc12ce785d216e140747`.
- Both arenas used the preregistered ordinary deterministic contract: `c_puct`
  `1.25`, FPU zero, `normalize_values=false`, seat swap, arena seed 42, and a
  matched P1-vs-P1 control per suite. No candidates or suites were changed.

## Search Probe

The fixed 256 replay states were probed for every seed at `384:256` and
`1200:1200` before arenas. All per-seed selected-move disagreement, visit-policy
JS, root-value shift, selected-child Q-rank change, and unavailable first-
divergence fields are recorded in the machine-readable summary. The probe was
diagnostic only and was not used for selection or classification.

## Primary Arena: 1200:1200

| Replay seed | AH delta | AI delta | AJ delta | Pooled delta | Opening CI95 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 53 | -.003906 | -.007812 | -.001953 | -.004557 | [-.008464, -.001953] |
| 54 | -.003906 | -.007812 | -.001953 | -.004557 | [-.008464, -.001953] |
| 55 | -.003906 | -.007812 | -.001953 | -.004557 | [-.008464, -.001953] |
| 56 | -.007812 | -.013672 | -.009766 | -.010417 | [-.015625, -.005859] |
| 57 | -.046875 | -.048828 | -.042969 | -.046224 | [-.056641, -.038395] |
| 58 | -.003906 | -.007812 | -.001953 | -.004557 | [-.008464, -.001953] |
| 59 | -.046875 | -.048828 | -.042969 | -.046224 | [-.056641, -.038395] |
| 60 | -.003906 | -.007812 | -.001953 | -.004557 | [-.008464, -.001953] |
| 61 | -.046875 | -.048828 | -.042969 | -.046224 | [-.056641, -.038395] |
| 62 | -.046875 | -.048828 | -.042969 | -.046224 | [-.056641, -.038395] |

- Hidden and adapter absolute P1 effects per replay seed are recorded in the
  summary JSON alongside the tabled deltas.
- Mean delta: `-.021810`; median: `-.007487`; SD: `.021088`; range:
  `[-.046224, -.004557]`; positive seeds: `0/10`.
- Hierarchical replay-seed, suite, opening bootstrap CI95:
  `[-.035547, -.009180]`.
- Held-out beta-.95 CE still improved in `10/10` seeds. It did not translate to
  strength: the primary mean was below `-.02` and four replay seeds were below
  `-.02`.

## Secondary Arena: 384:256

- Mean delta: `-.048242`; median: `-.062500`; SD: `.042666`; range:
  `[-.078776, +.062500]`; positive seeds: `1/10`.
- Hierarchical CI95: `[-.070703, -.018750]`. Nine seeds were below `-.02`, so
  the preregistered shallow-harm condition also triggered.

## Classification

`policy_hidden_degrades_strength`

The larger policy-hidden update space improved the held-out supervised metric
but degraded game strength relative to the frozen PR #261 trunk adapter. No
candidate is promoted.

## Recommended Next Experiment

Keep inherited policy tensors frozen; retain additive adapter architecture and
move to prospective iterative AlphaZero training rather than further unfreezing.
