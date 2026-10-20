# Seed505 exploratory strength screen

## Result

Seed504 E4 scored **0.50830078125** against its recorded initializer: 474 wins,
93 draws, and 457 losses across 1,024 games. No game reached the 200-ply cap.
The registered primary score is `(wins + 0.5 × draws) / 1,024`.

The 20,000-resample paired-opening percentile bootstrap (seed 505) gives a
one-sided 95% lower bound of **0.48974609375** and a descriptive two-sided 95%
interval of **[0.48583984375, 0.5302734375]**. The lower bound does not exceed
0.50, so the exploratory screen **does not pass**. This does not support
proposing the later 1,200-simulation test under the registered decision rule.
There is no promotion or fresh-confirmation claim.

## Protocol and limitations

- Challenger: seed504 E4, SHA-256
  `2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad`.
- Initializer: SHA-256
  `7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c`.
- Suite: the same 512 opening roots used by seed488, each played with both
  seats. This suite was previously examined; historical seed488 outcomes were
  not used in this comparison. This is not an untouched or historically unseen
  holdout.
- Search: 128 simulations per side, `c_puct=1.25`, one worker, seed
  `488 + parent opening index`, `azlite_eval_seed_v2`, 200-ply cap, zero random
  opening plies, and the unchanged seed488 exact-root/search contracts. Per
  opening resolved commands and settings are recorded in `commands.jsonl`.
- Recorded arena timing over the 512 batch reports: average reported
  `move_time_mean_ms` **5.7146 ms** and average reported
  `move_time_p95_ms` **10.7594 ms**. These are descriptive averages of the
  per-batch report fields, not pooled per-move quantiles.
- E4 opening-root overlap with seed504 training: **0/512** canonical board
  identities and **0/512** complete production float32 inputs. This reports
  only root identity overlap and does not alter the reused-suite limitation.
- The unresolved seed461 continuation-population overlap limitation remains.

## Execution evidence

`registration.json` and `execution-freeze.json` bind the protocol, suite,
runtime artifacts, and command/runtime source dependencies before execution.
After execution, the repository's required Ruff pre-commit hook reformatted the
three new Python files. The exact pre-execution source bytes are retained in
`execution-source-snapshots/` and match the source hashes in the freeze; the
format-only working copies are used for the published verifier and tests.
E4 and initializer exports use the same frozen search-policy sidecar; exported
parameters were checked against their source checkpoints. The paired batch logs
and reports are in `batches/`; outcomes and aggregate analysis are in
`outcomes.jsonl` and `results.json`. Independent replay verification:

```bash
.venv-azlite/bin/python -m ml.alphazero_lite.verify_seed505_strength_screen --root .
```

The result remains exploratory only. A pass would permit only a proposal for a
later 1,200-simulation test; it would not authorize promotion or fresh
confirmation.
