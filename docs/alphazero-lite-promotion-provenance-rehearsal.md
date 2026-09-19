# AlphaZero-lite Promotion Provenance Rehearsal

## Classification

`promotion_contract_unified_rehearsal_passed`

## Policy Contract

`local_promotion_gate.json` is the promotion-policy source of truth. When a
gate report is supplied, `promote_checkpoint.py` rejects caller policy
overrides, requires a passing identity-bound gate, validates the gate's
`min_arena_score`, `min_arena_games`, `require_lossless`, and `max_losses`,
then archives the exact gate-referenced arena report. The promotion wrapper no
longer supplies `--require-lossless --max-losses 0`.

No production workflow independently requires zero losses. Lossless remains an
optional local-gate policy, documented as such in the development guide.

New gate reports record `arena_evidence.path` and `arena_evidence.sha256`.
The completed frozen gate predates that field and was not modified. Its explicit
migration record is
`docs/data/alphazero-lite-promotion-provenance/seed48-corrected-gate-arena-evidence.json`.
That record binds the unchanged gate SHA to the archived report path and SHA;
it was derived by hashing existing files, not by rerunning or recomputing an
evaluation.

## Frozen Evidence

- Candidate: `.tmp/fresh-uniform1200/runs/seed48/uniform1200/fresh-uniform1200-s48-uniform1200-iter1`
- Candidate `weights.json` SHA256: `935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c`
- Candidate `metadata.json` SHA256: `ce226854584ac2518ddea22caa1f1faa7a2697f8aa96b1ad6d82388b23ca5f51`
- Authoritative gate: `.tmp/missed_capture_f67bd4k0_move_28-corrected-gate.json`
- Gate SHA256: `0d0d7176ee062ccb3f16ce3afc2bc72eb25a964a30f87485ba5b32631e69a577`
- Gate result: `passed: true`, `failure_reasons: []`
- Gate `require_lossless`: `false`
- Gate `max_losses`: `0`
- Gate arena report: `docs/data/alphazero-lite-shadow-canonical-hard-arena/candidate_vs_current_arena.json`
- Gate arena SHA256: `26d5709c291bbafd12ec08074d5493a392115b2a5a46ee22863b61dae7abac3a`
- Arena result: 417 wins, 25 draws, 70 losses

No evaluation was rerun. The only evidence transformation was deterministic
materialization: the promotion copied the gate-referenced arena report
byte-for-byte to the temporary artifact.

## Rehearsal

The temporary target was `.tmp/azlite-promotion-rehearsal-current`; the real
`model-artifact/current` was not modified.

| File | Source SHA256 | Temporary target SHA256 |
| --- | --- | --- |
| `weights.json` | `935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c` | `935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c` |
| `metadata.json` | `ce226854584ac2518ddea22caa1f1faa7a2697f8aa96b1ad6d82388b23ca5f51` | `ce226854584ac2518ddea22caa1f1faa7a2697f8aa96b1ad6d82388b23ca5f51` |
| `arena_report.json` | `26d5709c291bbafd12ec08074d5493a392115b2a5a46ee22863b61dae7abac3a` | `26d5709c291bbafd12ec08074d5493a392115b2a5a46ee22863b61dae7abac3a` |

The gate candidate identity matches the promoted weights and metadata. The
migration record's gate SHA matches the authoritative gate, and its arena
identity matches the archived target report.

## Inference Equivalence

Direct deterministic inference on `missed_capture_f67bd4k0_move_28` was exact
between the source and temporary artifact:

- Policy: `[0.0878254696726799, 0.20803186297416687, 0.3356238305568695, 0.0, 0.0, 0.3685188293457031]`
- Value: `-0.45103421807289124`
- Selected raw-policy move: `5`

## Rejection Coverage

- A passing non-lossless gate with arena losses promotes successfully.
- A passing gate claiming `require_lossless: true` while its arena exceeds
  `max_losses` is rejected before target mutation.
- A gate-bound invocation with `--require-lossless` is rejected as a policy
  override; the wrapper supplies no such override.
- An arena report modified after gate creation is rejected by SHA before target
  mutation.
- A different arena reference is rejected before target mutation.
- Candidate identity mismatch and failed gates remain rejected before target
  mutation.
