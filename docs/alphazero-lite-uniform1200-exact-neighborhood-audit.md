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

## Completion Oracle Run

Frozen cohort SHA-256: `2ae5376b06e26ec382e31b4ea9bad49a21dadd055891a863c407fde9fbe164ce`.
The committed PR #289, PR #290, and suite input hashes remain
`cdec9522...641a60d`, `0871dcfb...3260234`, and `68506f12...475f4fd`.

The native probe SHA-256 is `c93e3fc532b58ca83da21e69941ff2a2d1a9cf58c161fd32b5b845edcc5c4973`.
The read-verified canonical KVTB1 tier-18 tablebase SHA-256 is
`6e65399387f4b9c13bd83568c60a35fbe474a6bdb8ec868002b783eb0abb02ac`
(172,986,834 bytes; schema 1, revision 2). The original attempts remained at
30 seconds; the single pre-registered completion timeout was 120 seconds with
four independent native workers.

| Radius | Solved | Timeout | Error | Coverage |
| --- | ---: | ---: | ---: | ---: |
| 0 | 30 | 8 | 0 | 78.95% |
| 1 | 164 | 16 | 19 | 82.41% |
| 2 | 54 | 0 | 899 | 5.67% |

The radius-0/1 requirement is 99%; it failed before any further radius-2
campaign. The registered radius-2 requirement is 95% and also does not pass.
The artifact records 248 solved, 24 timed out, 918 error, 0 terminal, and 0
not-attempted state records. The error rows include an explicit local
pre-request dependency crash from the first completion invocation; they are
preserved as attempt history rather than represented as solver labels.

Observed completion-attempt wall durations range from 0 to 120.35 seconds;
the zero-duration rows are the recorded pre-request worker crashes. The
nonzero hard failures reached the fixed 120-second deadline. This is an oracle
coverage/latency blocker, not model evidence.

Per the registered protocol, the audit did not fall back to MCTS or the Python
solver. Frozen-reference agreement, checkpoint exact-quality tables, stable
family classification, replay coverage, supervision diagnoses, and PR #290
B/C/D diagnostics remain incomplete rather than inferred.

Checkpoint SHA verification and raw-network evaluation remain gated behind
oracle coverage, so no approximate score can be mistaken for exact evidence.

## Hard Classification

`forensic_neighborhood_audit_inconclusive`

The blocker is solver/tablebase coverage and solver latency. The reference
test was not run because not every solved radius-0 anchor is available, and
the registered threshold prevents interpreting model-vs-reference regressions.

## Recommended Next Experiment

Make the native exact-oracle completion reliable enough to meet the existing
frozen cohort coverage gates, then rerun this unchanged audit with the required
SHA-verified checkpoint artifacts; do not train or generate self-play.
