# Joint-Head Arena Failure Attribution

## Outcome

`no_joint_384_256_regression_reproduced`

The corrected paired-seed run did not reproduce the PR #164 primary regression. Per the immediate stop rule, trace, replay coverage, forced continuations, and causal classification were not run.

## Inputs And Verification

| Item | Value |
| --- | --- |
| Current weights SHA256 | `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a` |
| Candidate weights SHA256 | `4dfa06f0301d9daebb3d164275de6d819c92315a578e0810ed422cd8f4d2237c` |
| Replay SHA256 | `101eb0a0e421a9ba48c8e05cde77f00ea9a779eb066d0d5a0c4b79c842d6a8f6` |
| Medium suite SHA256 | `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04` |
| Composition manifest | `0f48c211885177aeaf5cd69ffb18842f0e04fd89722b092b38cd3caa1e6626b3` |
| Composition shards | 48 of 48; every output hash verified |
| Determinism check | Two-opening serial and 16-worker score shards byte-identical |

Profile: deterministic root policy, tactical root bias `0.0`, default `c_puct=1.25`, `768:768` `c_puct=0.90`, no value normalization/interpolation/trust override/root-prior transform. The full run used 16 workers; forced cap and bootstrap configuration were 128 and 1,000 respectively.

## Composition Scores

Values are raw DS for current/current, candidate-policy/current-value, current-policy/candidate-value, and candidate/candidate.

| Budget | Current/current | Policy/current | Current/value | Candidate/candidate |
| --- | ---: | ---: | ---: | ---: |
| 384:256 | 0.01562500 | 0.07031250 | 0.03906250 | 0.05078125 |
| 768:256 | 0.02734375 | -0.02734375 | 0.03515625 | 0.03125000 |
| 768:768 | 0.04687500 | 0.12109375 | 0.05078125 | 0.01953125 |
| 1200:1200 | 0.07812500 | 0.08593750 | 0.10937500 | 0.07031250 |
| 1200:256 | -0.01953125 | -0.01171875 | 0.02734375 | 0.01953125 |
| 256:768 | 0.02734375 | 0.02734375 | 0.08984375 | 0.07421875 |

Signed attribution values are policy-only, value-only, interaction residual, and joint delta versus current/current.

| Budget | Policy | Value | Interaction | Joint |
| --- | ---: | ---: | ---: | ---: |
| 384:256 | 0.05468750 | 0.02343750 | -0.04296875 | 0.03515625 |
| 768:256 | -0.05468750 | 0.00781250 | 0.05078125 | 0.00390625 |
| 768:768 | 0.07421875 | 0.00390625 | -0.10546875 | -0.02734375 |
| 1200:1200 | 0.00781250 | 0.03125000 | -0.04687500 | -0.00781250 |
| 1200:256 | 0.00781250 | 0.04687500 | -0.01562500 | 0.03906250 |
| 256:768 | 0.00000000 | 0.06250000 | -0.01562500 | 0.04687500 |

## Stop Evidence

| Criterion | Required threshold | Observed | Result | Supporting hash |
| --- | --- | ---: | --- | --- |
| Reproduce PR #164 384:256 joint regression | Negative delta equal to `-0.03125` within deterministic tolerance | `+0.03515625` | Fail | `0f48c211885177aeaf5cd69ffb18842f0e04fd89722b092b38cd3caa1e6626b3` |

The current/current 384:256 baseline was `0.015625`, and candidate/candidate was `0.05078125`. The corrected paired-seed candidate delta therefore reverses the expected direction. No trace/cache counts, replay overlap, distance-harm evidence, or neutral/context-matched forced outcomes exist because producing them would violate the stop rule.

## Recommendation

Explain the pre-fix versus paired-seed composition discrepancy before further causal attribution or any model/replay intervention.
