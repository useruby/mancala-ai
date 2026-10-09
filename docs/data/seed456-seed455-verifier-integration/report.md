# seed456 — seed455 verifier integration verification

This is **post-publication verification** of seed455's published descriptive
classification and measurements. It does not alter the seed455 decision or
historical publications and does not authorize promotion.

## Bound identities

| Artifact | SHA256 |
| --- | --- |
| seed455 verifier `ml/alphazero_lite/verify_seed455_fresh_projection_attribution.py` | `000b5b71ee56bf3370c3bad49e0c488b0de863148f5d4c61d84d319ac1ab6791` |
| seed455 integration test `ml/alphazero_lite/test_seed455_verifier_integration.py` | `cdc98b7a59f0ea6829226ef0eafcf5ca64c52b504c8f4ffaa9d4e88689f05a4a` |
| `freeze-v7.json` | `ab4651ca6a9aa12356356c17e0d2913fad7fb8168fc1420eb328b8544fc48c33` |
| `results.json` | `9365838ff5386f9a696795846e25f2bcaff4ac5c21d9990ce06e3e3925f54c52` |
| seed455 `receipt.json` | `296f772fc8a5e6e4de7d24501da567691023dc13c8c8cb7572ff942dba9f2566` |

## Full-verifier results

The unchanged physical source-and-evidence copy returned `status=valid`,
`row_count=96`, and `fresh_unseen_direction_dominant`. Each negative case ran
`verify(root)` on an isolated disposable copy; outer receipt checksums were
rebound after mutation so the verifier reached its independent semantic
reconstruction. No verifier or model computations were mocked.

| Mutation | Actual semantic rejection |
| --- | --- |
| Cohort identity | `cohort_identity:fresh_unseen_equal_input` |
| Equal-input accounting (exposure-weighted CE ledger) | `pre_ce:B_fresh_unseen_equal_input_0` |
| Archived policy targets | `targets_reconstructed` |
| Archived parameter-state continuity | `trajectory_link:B:1` |
| Saved gradients | `gradient:B_training_guard_0` |
| Saved prediction logits | `prediction:B:training_guard:0:0` |
| Parameter-group slope contributions | `group_contribution:B_fresh_unseen_equal_input_0` |
| Per-step/block totals | `block:B:fresh_unseen_equal_input:all_16:d` |
| Classification | `classification` |

The integration suite also verifies original frozen inputs, source files, and
the complete seed455 publication inventory by SHA256 and modification time
before and after each isolated positive/negative verifier execution.

## Portability and checks

The seed455 physical-copy regression launches the production verifier CLI
with `.venv-azlite/bin/python` from an unrelated working directory, imports
the copied `ml` source using an isolated `PYTHONPATH`, and blocks access to
the original checkout's `docs/data` and historical registered replay paths.
Its publication inventory hashes and modification times match before and
after the CLI run.

Verification completed with no skips:

- New full-verifier integration tests: 10 passed.
- Seed455 helper and physical-copy regressions: 14 passed.
- Combined final seed455 tests: 24 passed.
- `ruff check` passed; `ruff format --check` passed after formatting.
- `script/ai/run_python_quality.sh` passed.

No supplemental verifier was needed: all listed evidence mutations were
rejected by the existing seed455 verifier's independent reconstruction.

## Preserved historical interpretation

This verification preserves seed453's `close_fresh_policy_projection_branch`,
seed454's `archived_trajectory_matches_full_guard_rule`, and seed455's
published descriptive classification and measurements.
