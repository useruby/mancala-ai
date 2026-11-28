# Seed 639 attempt-0002 results

## Hypothesis, estimand, and interpretation

The registered hypothesis was that B512 and S615 have descriptively measurable
legal-masked policy-prior differences on the fixed, authenticated Seed 636
512-state population. The estimand is the descriptive policy difference over
that finite selected-state cohort, including the preregistered high-total-
variation secondary subset. Results are not evidence of relative playing
strength.

Exposure classification remains **known-exposure-screened; inherited P-E4
exposure unresolved**.

## Execution and independent verification

- Owner-merged authorization: PR #642, merged by `useruby` at
  `d63cc1ea6026cdd860fc641de112cf90323c9615`.
- Frozen source commit: `d63cc1ea6026cdd860fc641de112cf90323c9615`.
- Attempt manifest SHA-256:
  `164060dbc000df2882796d5915ca292605b5d750761e032fc8e29ee1389f2a8d`.
- Read-only preflight receipt SHA-256:
  `686d1228efd6ed306639150b6b8727c0c5ec50de720243c053b2e8a1fc24cfa3`.
- Lifecycle execution completed with exit status 0 in 53.8 seconds; no stderr.
- The preflight authenticated all registered inputs, Seed 615 history, the
  Seed 636 archive and member, and all 512 population rows before inference.
  Canonical and production-float32 identity counts were both 512; stone-bin
  counts were 103/103/102/102/102 for 17/18/19/20/21 stones; there were 459
  continuation games and 218 opening clusters.
- Independent verification passed: all 512 states and **1,024 model-state
  evaluations** were reproduced with bitwise-equal complete legal-masked
  probability vectors. Population preflight, summaries, and diagnostic subset
  selection/composition also matched exactly. See
  [`verification.json`](verification.json).

## Primary-population results

| Active pit stones | States | Top-action disagreements | Disagreement rate | Mean TV | Mean entropy delta (S615 − B512, bits) |
|---:|---:|---:|---:|---:|---:|
| 17 | 103 | 3 | 0.029126 | 0.025862 | -0.001045 |
| 18 | 103 | 4 | 0.038835 | 0.022350 | -0.017225 |
| 19 | 102 | 3 | 0.029412 | 0.023531 | 0.002533 |
| 20 | 102 | 6 | 0.058824 | 0.027728 | 0.006233 |
| 21 | 102 | 7 | 0.068627 | 0.022104 | -0.002350 |

Across all 512 states, 23 top actions disagreed (rate 0.044922), mean total
variation was 0.024314, and mean entropy difference was -0.002397 bits. TV
quantiles (0, .25, .5, .75, .9, .95, .99, 1) were
0.000024, 0.005858, 0.018959, 0.035884, 0.054622, 0.065340, 0.101791, and
0.167819. Fixed equal one-fifth-per-bin standardized disagreement was 0.044965;
standardized mean TV was 0.024315 and standardized entropy difference was
-0.002371 bits. Full aggregates are in [`summary.json`](summary.json).

## Secondary diagnostic subset

The frozen descending-TV selection yielded 64 states from 64 distinct
continuation games and 57 opening clusters, satisfying the registered limits of
one state per game and four per opening. It contains 13/12/12/17/10 states in
the 17/18/19/20/21-stone bins. The subset's top-action disagreement rate was
0.15625 and mean TV was 0.070129; this is an enriched, unweighted descriptive
subset, not a replacement or reweighting of the primary cohort.

## Registered criteria and classification

| Criterion | Result |
|---|---|
| Exact authenticated 512-state cohort and identities | Passed |
| Registered stone-bin quotas and provenance | Passed |
| Exactly one B512 and one S615 inference per state | Passed; 1,024 evaluations |
| Legal-masked vectors and independent bitwise reproduction | Passed |
| Deterministic top-action, TV, entropy, aggregate, and standardized summaries | Passed |
| Frozen 64-state secondary subset and coverage/caps | Passed |
| Scope exclusions (no search, exact solving, training, or games) | Satisfied |
| Scientific interpretation | Descriptive policy differences only; no strength claim |

Final classification: **successful infrastructure and verification; descriptive
policy-prior differences observed; no playing-strength conclusion**. The finite
population does not resolve inherited P-E4 exposure. The next scientific
experiment is not authorized by this registration; any follow-up requires its
own reviewed, owner-merged registration.

## Archive

- Archive: `/home/alex/mancala-ai-experiment-archives/seed639-prospective-policy-prior-classification--attempt-0002.tar.zst`
- Size: 187,546 bytes
- SHA-256: `a43dd467bc5f113dcdd471e39426b14018279e1a4008e795e60b445a7cfc2c94`
- Archive sidecar SHA-256:
  `4a53be6f0a2aa231e2e0b52dde648b6fcfdbaab2999c93bf3f2645d84df34231`
- Complete runtime files verified: 526
- Required artifact inventory SHA-256:
  `5af53d4cb272603ac8029ade68768fe33d275e2b4b7cd82d9ccd5995b2951dce`
- Safe archive verification: passed.

Attempt-0001 and all prior preflight operational records remain preserved.
