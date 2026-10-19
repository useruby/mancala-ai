# Seed500 512-game self-play corpus: pre-generation manifest

Status: finalized before generation. The original 64-game pilot and its bytes
remain unchanged. This is an exploratory data-integrity corpus only; no training,
candidate evaluation, promotion, or fresh-confirmation claim is authorized.

## Frozen behavior and identities

- Behavior model: `seed48-nextgen-s455-default-value-iter1`.
- `model-artifact/current/weights.json` SHA-256:
  `f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c`.
- `model-artifact/current/search_policy.json` SHA-256:
  `b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d`.
- Materialized checkpoint `.tmp/seed500-current-checkpoint.npz` SHA-256:
  `7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c`.
- Runtime: `puct_exact_root_hybrid`; exact-root final-score-margin through 16
  active stones; network-prior then lowest-index tie break; exact-leaf solving
  disabled. Native probe SHA-256:
  `d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48`;
  tablebase SHA-256:
  `f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7`.
- Generator `ml/alphazero_lite/self_play.py` SHA-256:
  `76718efde7ccfb708f4bb0a4314f812bab746d1364cb75ce5c8b3551b635fc0c`;
  rules `kalah_rules.py` SHA-256:
  `6e5338ef9a85b690f51b6d0d184c8bc0f54c2b28ac084a4a17a88f9f552ad036`;
  loader `train.py` SHA-256:
  `2dcb42f8dff790d5033e2c9a865f3c24eb1fee6b9b37302200dd176d109dad32`;
  pipeline materializer `pipeline.py` SHA-256:
  `ad1c9919e13418c4af53be9856406272bdb4f8ed9a53e7424cc9364364772dfa`.
- Environment at preflight: Python 3.14.7, Linux x86_64 / glibc 2.44,
  NumPy 2.5.2, PyTorch 2.13.0+cu130, Ruff 0.15.13. Relevant installed
  packages include pytest 9.1.1 and pre-commit 4.6.0.
- The frozen pilot validator returned loader acceptance and successful replay;
  its three focused tests passed and focused Ruff passed before generation.

## Commands and partition

All commands use the production CLI and identical generation settings. Each
shard is a separate 64-game invocation and carries its assigned base seed.
Seeds 503000–503005 are training-source shards; 503006–503007 are
development-source shards. The development designation is organizational only
and does not imply historically unseen data. No pilot games are reused.

```bash
.venv-azlite/bin/python -m ml.alphazero_lite.self_play --out docs/data/seed500-selfplay-512/shard-503000.jsonl --games 64 --seed 503000 --checkpoint .tmp/seed500-current-checkpoint.npz --runtime-search-policy-artifact model-artifact/current --input-encoding kalah_v3 --player-mode puct --simulations 128 --c-puct 1.25 --temperature-threshold 10 --temperature 1.0 --temperature-late 0.1 --dirichlet-alpha 0.3 --dirichlet-epsilon 0.25 --schedule-progress-mode none --max-moves 200 --workers 1 --policy-target-mode default --exact-root-policy-target-mode selected_one_hot --policy-target-noise-mode denoised --value-target-mode default --start-state-mode standard_4x6 --write-game-metadata
.venv-azlite/bin/python -m ml.alphazero_lite.self_play --out docs/data/seed500-selfplay-512/shard-503001.jsonl --games 64 --seed 503001 --checkpoint .tmp/seed500-current-checkpoint.npz --runtime-search-policy-artifact model-artifact/current --input-encoding kalah_v3 --player-mode puct --simulations 128 --c-puct 1.25 --temperature-threshold 10 --temperature 1.0 --temperature-late 0.1 --dirichlet-alpha 0.3 --dirichlet-epsilon 0.25 --schedule-progress-mode none --max-moves 200 --workers 1 --policy-target-mode default --exact-root-policy-target-mode selected_one_hot --policy-target-noise-mode denoised --value-target-mode default --start-state-mode standard_4x6 --write-game-metadata
.venv-azlite/bin/python -m ml.alphazero_lite.self_play --out docs/data/seed500-selfplay-512/shard-503002.jsonl --games 64 --seed 503002 --checkpoint .tmp/seed500-current-checkpoint.npz --runtime-search-policy-artifact model-artifact/current --input-encoding kalah_v3 --player-mode puct --simulations 128 --c-puct 1.25 --temperature-threshold 10 --temperature 1.0 --temperature-late 0.1 --dirichlet-alpha 0.3 --dirichlet-epsilon 0.25 --schedule-progress-mode none --max-moves 200 --workers 1 --policy-target-mode default --exact-root-policy-target-mode selected_one_hot --policy-target-noise-mode denoised --value-target-mode default --start-state-mode standard_4x6 --write-game-metadata
.venv-azlite/bin/python -m ml.alphazero_lite.self_play --out docs/data/seed500-selfplay-512/shard-503003.jsonl --games 64 --seed 503003 --checkpoint .tmp/seed500-current-checkpoint.npz --runtime-search-policy-artifact model-artifact/current --input-encoding kalah_v3 --player-mode puct --simulations 128 --c-puct 1.25 --temperature-threshold 10 --temperature 1.0 --temperature-late 0.1 --dirichlet-alpha 0.3 --dirichlet-epsilon 0.25 --schedule-progress-mode none --max-moves 200 --workers 1 --policy-target-mode default --exact-root-policy-target-mode selected_one_hot --policy-target-noise-mode denoised --value-target-mode default --start-state-mode standard_4x6 --write-game-metadata
.venv-azlite/bin/python -m ml.alphazero_lite.self_play --out docs/data/seed500-selfplay-512/shard-503004.jsonl --games 64 --seed 503004 --checkpoint .tmp/seed500-current-checkpoint.npz --runtime-search-policy-artifact model-artifact/current --input-encoding kalah_v3 --player-mode puct --simulations 128 --c-puct 1.25 --temperature-threshold 10 --temperature 1.0 --temperature-late 0.1 --dirichlet-alpha 0.3 --dirichlet-epsilon 0.25 --schedule-progress-mode none --max-moves 200 --workers 1 --policy-target-mode default --exact-root-policy-target-mode selected_one_hot --policy-target-noise-mode denoised --value-target-mode default --start-state-mode standard_4x6 --write-game-metadata
.venv-azlite/bin/python -m ml.alphazero_lite.self_play --out docs/data/seed500-selfplay-512/shard-503005.jsonl --games 64 --seed 503005 --checkpoint .tmp/seed500-current-checkpoint.npz --runtime-search-policy-artifact model-artifact/current --input-encoding kalah_v3 --player-mode puct --simulations 128 --c-puct 1.25 --temperature-threshold 10 --temperature 1.0 --temperature-late 0.1 --dirichlet-alpha 0.3 --dirichlet-epsilon 0.25 --schedule-progress-mode none --max-moves 200 --workers 1 --policy-target-mode default --exact-root-policy-target-mode selected_one_hot --policy-target-noise-mode denoised --value-target-mode default --start-state-mode standard_4x6 --write-game-metadata
.venv-azlite/bin/python -m ml.alphazero_lite.self_play --out docs/data/seed500-selfplay-512/shard-503006.jsonl --games 64 --seed 503006 --checkpoint .tmp/seed500-current-checkpoint.npz --runtime-search-policy-artifact model-artifact/current --input-encoding kalah_v3 --player-mode puct --simulations 128 --c-puct 1.25 --temperature-threshold 10 --temperature 1.0 --temperature-late 0.1 --dirichlet-alpha 0.3 --dirichlet-epsilon 0.25 --schedule-progress-mode none --max-moves 200 --workers 1 --policy-target-mode default --exact-root-policy-target-mode selected_one_hot --policy-target-noise-mode denoised --value-target-mode default --start-state-mode standard_4x6 --write-game-metadata
.venv-azlite/bin/python -m ml.alphazero_lite.self_play --out docs/data/seed500-selfplay-512/shard-503007.jsonl --games 64 --seed 503007 --checkpoint .tmp/seed500-current-checkpoint.npz --runtime-search-policy-artifact model-artifact/current --input-encoding kalah_v3 --player-mode puct --simulations 128 --c-puct 1.25 --temperature-threshold 10 --temperature 1.0 --temperature-late 0.1 --dirichlet-alpha 0.3 --dirichlet-epsilon 0.25 --schedule-progress-mode none --max-moves 200 --workers 1 --policy-target-mode default --exact-root-policy-target-mode selected_one_hot --policy-target-noise-mode denoised --value-target-mode default --start-state-mode standard_4x6 --write-game-metadata
```

Intended: eight shards, 64 games each (512 total); 384 training-source and 128
development-source games. Preserve any file written by an invocation, including
outcome-bearing partial files. Never overwrite or silently regenerate a shard;
do not replace capped or incomplete games.

## Execution record

The manifest and all input identities were checked before generation. An initial
launcher attempt for all eight seeds failed before running Python because
`/usr/bin/time` is absent; no shard file was created. The failure is retained in
the local run log and described in the corpus README. The authorized commands
were then executed sequentially once each using Bash's built-in `time -p`.
Every command exited successfully. Total generator elapsed time was 270.70
seconds. Per-shard elapsed time, row counts, completion status, byte count, and
SHA-256 are recorded in `accounting.json`. The eight outputs were not replaced
or regenerated.
