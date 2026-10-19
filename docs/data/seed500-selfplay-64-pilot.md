# Seed500 64-game self-play data pilot

## Protocol and source

Generated with `ml.alphazero_lite.self_play` from the frozen
`seed48-nextgen-s455-default-value-iter1` behavior model in
`model-artifact/current`. Its weights SHA-256 is
`f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c`; the
runtime search-policy SHA-256 is
`b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d`.
The runtime contract is PUCT with exact-root final-score-margin solving through
16 active stones, network-prior then lowest-index tie-breaking, and exact leaf
solving disabled.

The pilot used 64 standard-start games; seed 500; one worker; PUCT, 128
simulations, `c_puct=1.25`; Dirichlet alpha `0.3`, epsilon `0.25`; temperature
`1.0` through ply 10 then `0.1`; no iteration schedule; 200-ply maximum; default
visit-count policy targets, selected-one-hot exact-root targets, denoised target
noise mode, and default terminal-outcome value targets. This is a new recipe,
distinct from historical seed455 generation. The generated JSONL and temporary
NPZ are preserved locally under `.tmp/`, not committed. JSONL SHA-256:
`5d532603debd431e399b0ec32265a2c6b43064e5b1f7ecb1f0cfca32c7e266f9`.

## Execution

The production generator accepts NPZ checkpoints. The frozen artifact has
`weights.json`, so the existing pipeline materializer was used to create the
temporary checkpoint before invoking the unchanged generator:

```bash
.venv-azlite/bin/python -c 'from pathlib import Path; from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint; p=materialize_weights_json_checkpoint(weights_path=Path("model-artifact/current/weights.json"), out_path=Path(".tmp/seed500-current-checkpoint.npz")); print(p)'
.venv-azlite/bin/python -m ml.alphazero_lite.self_play --out .tmp/seed500-selfplay-64.jsonl --games 64 --seed 500 --checkpoint .tmp/seed500-current-checkpoint.npz --runtime-search-policy-artifact model-artifact/current --input-encoding kalah_v3 --player-mode puct --simulations 128 --c-puct 1.25 --temperature-threshold 10 --temperature 1.0 --temperature-late 0.1 --dirichlet-alpha 0.3 --dirichlet-epsilon 0.25 --schedule-progress-mode none --max-moves 200 --workers 1 --policy-target-mode default --exact-root-policy-target-mode selected_one_hot --policy-target-noise-mode denoised --value-target-mode default --start-state-mode standard_4x6 --write-game-metadata
```

## Checks and interpretation

- 64 games and 2,376 rows; every trajectory replayed legally to terminal state.
- No replay errors, capped games, or incomplete games.
- 800 rows have more than 32 active stones.
- 1,954 unique encoded states (82.2% of rows); 61 unique trajectories.
- Maximum repeated-state count was 64 rows (2.7% of all rows).
- Mean legal-policy target entropy was 0.215 nats; 47.4% of rows had zero
  entropy. There were 1,526 PUCT rows and 850 exact-root rows.
- `train.load_jsonl` accepted the JSONL with default policy and value-target
  modes.

These checks establish pilot-data integrity and basic usability, not model
strength or training benefit. They do not initiate a training ablation. Seed488's
closed F advancement decision remains closed; the known historical seed461
overlap limitation remains. This generated corpus is not an untouched holdout.
