# AlphaZero-lite Promotion Provenance Rehearsal

## Classification

`promotion_artifact_contract_gap`

The frozen seed48 artifact is missing candidate-local `arena_report.json`. The
promotion command rejected it before creating the rehearsal target, so no file
under `model-artifact/current` was modified.

## Evidence

- Source candidate: `.tmp/fresh-uniform1200/runs/seed48/uniform1200/fresh-uniform1200-s48-uniform1200-iter1`
- Metadata version: `fresh-uniform1200-s48-uniform1200-iter1`
- Source `weights.json` SHA256: `935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c`
- Source `metadata.json` SHA256: `ce226854584ac2518ddea22caa1f1faa7a2697f8aa96b1ad6d82388b23ca5f51`
- Completed corrected gate: `.tmp/missed_capture_f67bd4k0_move_28-corrected-gate.json`
- Gate result: `passed: true`, `failure_reasons: []`
- Gate-recorded weights SHA256: `935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c`
- Gate-recorded metadata SHA256: `ce226854584ac2518ddea22caa1f1faa7a2697f8aa96b1ad6d82388b23ca5f51`

The identity fields were added to the completed gate record from the frozen
artifact hashes. No gate evaluation was rerun.

## Rehearsal

The requested command used `.tmp/azlite-promotion-rehearsal-current` as its
temporary target. It exited with status 1 before target creation:

```
Missing required file: .tmp/fresh-uniform1200/runs/seed48/uniform1200/fresh-uniform1200-s48-uniform1200-iter1/arena_report.json
```

The target did not exist after rejection. Therefore there are no promoted file
hashes or source-versus-target inference results to report.

The related archived arena report records 70 losses, and the completed gate
records `require_lossless: false`. Consequently the explicitly requested
`--require-lossless --max-losses 0` rehearsal would still reject that evidence
after the missing-artifact contract is corrected.

The normal artifact loader successfully loaded the source. On
`missed_capture_f67bd4k0_move_28`, direct deterministic inference produced:

- Policy: `[0.0878254696726799, 0.20803186297416687, 0.3356238305568695, 0.0, 0.0, 0.3685188293457031]`
- Value: `-0.45103421807289124`
- Selected move: `5`

## Rejection Coverage

- Matching passing gate and candidate: promotion succeeds.
- Passing gate for candidate A applied to candidate B: rejected with `gate_candidate_identity_mismatch`; target unchanged.
- Post-gate `weights.json` mutation: rejected with `gate_candidate_identity_mismatch`; target unchanged.
- Post-gate `metadata.json` mutation: rejected with `gate_candidate_identity_mismatch`; target unchanged.
- Passing gate without identity: rejected with `gate_candidate_identity_mismatch`; target unchanged.
- Failed gate: rejected before target mutation.
