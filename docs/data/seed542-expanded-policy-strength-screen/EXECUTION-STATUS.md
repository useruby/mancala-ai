# Execution status — incomplete, no decision

**Status: stopped at opening 0 before the first ledger commit.** No complete
screen decision, score, bootstrap interval, or advancement result is available.
Frozen registration SHA-256: `b39247e1d28c130da682df654cb7856a56985c79b0df7bb6297c9c65930e862e`.
Execution-freeze SHA-256: `e2d30aa8b1df6658bc3636b487b1bde4e7969e923557c19937a10f46c5a18311`.

## Exact execution failure

The frozen durable runner launched the arena under the registered user-systemd
resource scope. The arena completed opening 0 and wrote its raw two-game file
and arena report. Before appending either game to `outcomes.jsonl`, the runner
called the independent pair verifier. That verifier stopped with:

```text
ValueError: raw_pair_report_receipt_mismatch:0
```

The launch-time command receipt binds the actual argv, suite, components,
runtime, and source hashes, but the runner did not extend it with SHA-256 values
for the game and report files the arena produced. The new verifier correctly
refused the missing report/game bindings. This is a validation failure in the
frozen execution tooling; executable sources were not changed after launch.

The read-only verifier was also run from a physically separate checkout and an
unrelated working directory while the original screen worktree root had mode
`000` and was inaccessible. It stopped earlier at
`verify_seed542_training_root_mismatch:verify_seed542_fixed_update_cd_v2` because
the retained training-verification receipt records `--root .` rather than the
frozen absolute execution root. An external before/after snapshot of 1,221
evidence files confirmed identical bytes and modification times. This is a
second verifier-binding failure; it does not authenticate the raw pair.

## Checks actually completed

- Focused new screen tests: **19 passed**.
- Repository quality hook (`script/ai/run_python_quality.sh`): **passed** all
  configured hooks.
- Arena/composition regression batch: **95 passed, 2 failed**. The failures
  were `OpeningContractTests.test_formatted_delivery_sources_do_not_pass_the_old_launch_binding`
  (`FileNotFoundError` for the absent repo-local
  `.tmp/canonical-reconstruction/medium_eval.jsonl`) and
  `OpeningContractTests.test_publication_snapshots_do_not_weaken_launch_source_hashes`
  (`formatted_delivery_hash_mismatch:ml/alphazero_lite/arena.py`). Neither
  touched the screen publication or changed its executable sources.
- Both seed542 training verifier commands and the supplement passed under the
  documented limits; receipts remain in `training-verification.json`.
- Frozen `registration.json`, imported suite, overlap report, candidate
  artifacts, and all 173 authenticated files rechecked successfully after the
  arena process stopped. The materialized tablebase pointer path is restored.

The read-only verifier did **not** complete successfully, and the opening-0 raw
pair was not independently accepted or entered into the ledger. There is no
advancement decision.

## Preserved evidence and scope

- The exact arena command is `batches/command-0000.json`; SHA-256
  `7444e68ff583f002dd45064b331206fb09082a6b11ded557daf7769d822acb05`.
- The command/arena process receipt is `batches/attempt-0000.json`; SHA-256
  `8c59c74aad53521c90f1c6471d352b0cc422f661404b3811b681c1b16696104a`.
- Raw game trajectories are retained in `batches/games-0000.jsonl`; SHA-256
  `e707d85c3e43e397999616ea805d2a3a5b8fde403d114b743a960cbaff726cd2`.
- Arena report is retained in `batches/report-0000.json`; SHA-256
  `b92575de1882a5ea1b57789960f6e5f33335ec797252cebe5adce2b198676b5f`.
- The registered one-opening input is retained at
  `batches/opening-0000.jsonl`; SHA-256
  `11ed77c47b6abdba2bf5f7de5596ffc6c8d49447ee14b5a319074d6ff7acec7d`.
- `outcomes.jsonl` was not created; there are **0 committed games and 0
  committed pairs**. The pair has not passed independent verification.
- No other opening was launched. There was no cap adjustment, replacement, or
  outcome-bearing retry. The initial separate pre-game tablebase-path failure
  and its receipt remain in `PRELAUNCH-FAILURE-0000.json` and its archive.
- The original checked-in tablebase LFS pointer was restored after execution;
  its SHA-256 is `a82ca3d03265950a23aae8a1088fa86a93db9acea7bca68ea33303292eb0202c`.
  The local materialized runtime copy remains byte-identical to the registered
  tablebase SHA-256 `f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7`.

The runner exited nonzero on pair validation and the durable unit is no longer
running. The existing pre-game import and overlap disclosure remain valid, but
the registered screen has no validated games and no advancement decision. Do
not infer an ordinary draw or a strength regression from this incomplete
attempt. No 1,200-simulation proposal, promotion, deployment, training, or
serving change is authorized.
