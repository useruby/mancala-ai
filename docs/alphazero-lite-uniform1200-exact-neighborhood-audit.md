# Uniform1200 Exact Neighborhood Audit

## Status

PR #290 is merged on `main` and classified `midgame_2x2_no_gain`.

This is an evaluation-only artifact. It generated no self-play, ran no training,
did not modify replay, and did not promote a checkpoint.

## Inputs

PR #289's committed machine artifact mechanically supplied 38 consistent
uniform1200 regression anchors and 29 consistent improvement-control anchors.
The improvement anchors are preserved as the registered control set and were
not expanded after inspecting the regression cohort.

## Neighborhood Manifest

The legal Kalah transition engine generated the regression manifest using only
reachable one- and two-action successors. All entries carry
`training_eligible: false`.

| Radius | Canonical non-terminal/terminal states |
| --- | ---: |
| 0-2 total | 1,190 |
| 2 full | 953 |
| 2 sampled | 953 |

The full cohort is below the 10,000-state size guard, so no radius-2 sampling
occurred. Provenance records anchor ID, action path, extra-turn/capture flags,
terminal status, and canonical key.

## Exact Oracle And Checkpoints

The pinned native probe and a deterministic canonical tier-18 tablebase were
provisioned. Under the registered fixed 30-second request timeout, the native
run produced 87 exact solves and 58 explicit timeouts before the audit budget
was stopped. This is far below the registered coverage acceptance thresholds.
Per the registered protocol, the audit did not fall back to MCTS or the Python
solver. Consequently frozen-reference agreement, checkpoint exact-quality
tables, replay coverage, supervision diagnoses, and B/C/D repair diagnostics
remain unavailable rather than inferred.

The machine artifact records the partial coverage, required input hashes, and a
deterministic neighborhood manifest. Checkpoint SHA verification and raw network
evaluation remain gated behind a successful native-oracle run, so no
approximate scores can be mistaken for exact evidence.

## Hard Classification

`forensic_neighborhood_audit_inconclusive`

The lack of the required native exact-solver artifact prevents the acceptance
coverage test and therefore a valid classification among reference mismatch,
stable families, anchor-specific regressions, or heterogeneous mechanisms.

## Recommended Next Experiment

Provision the pinned native probe and canonical tablebase, then rerun this same
frozen neighborhood audit with SHA-verified PR #287/#288 and PR #290 artifacts;
do not train or generate self-play.
