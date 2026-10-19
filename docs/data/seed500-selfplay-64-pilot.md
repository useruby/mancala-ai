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
distinct from historical seed455 generation. Generation occurred during a
planning-only task; it was not prospectively approved or frozen. The generated
JSONL is archived byte-for-byte as `seed500-selfplay-64.jsonl`. Its JSONL SHA-256:
`5d532603debd431e399b0ec32265a2c6b43064e5b1f7ecb1f0cfca32c7e266f9`.

## Execution

The production generator accepts NPZ checkpoints. The frozen artifact has
`weights.json`, so the existing pipeline materializer was used to create the
temporary checkpoint before invoking the unchanged generator:

```bash
.venv-azlite/bin/python -c 'from pathlib import Path; from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint; p=materialize_weights_json_checkpoint(weights_path=Path("model-artifact/current/weights.json"), out_path=Path(".tmp/seed500-current-checkpoint.npz")); print(p)'
.venv-azlite/bin/python -m ml.alphazero_lite.self_play --out .tmp/seed500-selfplay-64.jsonl --games 64 --seed 500 --checkpoint .tmp/seed500-current-checkpoint.npz --runtime-search-policy-artifact model-artifact/current --input-encoding kalah_v3 --player-mode puct --simulations 128 --c-puct 1.25 --temperature-threshold 10 --temperature 1.0 --temperature-late 0.1 --dirichlet-alpha 0.3 --dirichlet-epsilon 0.25 --schedule-progress-mode none --max-moves 200 --workers 1 --policy-target-mode default --exact-root-policy-target-mode selected_one_hot --policy-target-noise-mode denoised --value-target-mode default --start-state-mode standard_4x6 --write-game-metadata
```

## Provenance and validation

Execution-time records in the original pilot note: weights SHA-256
`f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c`, runtime
search-policy SHA-256
`b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d`, native
probe SHA-256 `d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48`,
and tablebase SHA-256
`f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7`. The
runtime contract recorded during execution was `puct_exact_root_hybrid`, exact
root through 16 active stones, exact leaf solving disabled. The materialized
checkpoint identity, generator source identity, and runtime software versions
were not captured at execution time.

Hashes collected now (2026-10-19, from the preserved workspace, not execution
attestations): materialized checkpoint
`.tmp/seed500-current-checkpoint.npz`, SHA-256
`7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c`;
generator source `ml/alphazero_lite/self_play.py`, SHA-256
`76718efde7ccfb708f4bb0a4314f812bab746d1364cb75ce5c8b3551b635fc0c`;
`kalah_rules.py`, SHA-256
`6e5338ef9a85b690f51b6d0d184c8bc0f54c2b28ac084a4a17a88f9f552ad036`.
No original measured generation duration record is available.

Run the read-only report with:

```bash
.venv-azlite/bin/python -m ml.alphazero_lite.validate_seed500_pilot docs/data/seed500-selfplay-64.jsonl
```

It replays complete games through extra turns and the rules engine's terminal
sweep, checks row/action/game/outcome/value consistency, and calls the production
`train.load_jsonl` loader. Canonical-state and encoded-input identities are
reported independently. Its subgroup table separates PUCT from exact-root rows
and partitions active pit stones as `>32`, `17–32`, and `≤16`; each cell reports
row count, repeated encoded inputs, legal-policy entropy, one-hot fraction, and
usable rows.

## Checks and interpretation

- 64 games and 2,376 rows; every trajectory replayed legally to terminal state.
- No replay errors, capped games, or incomplete games.
- 800 rows have more than 32 active stones.
- 1,954 unique encoded states (82.2% of rows); 61 unique trajectories.
- Maximum repeated-state count was 64 rows (2.7% of all rows).
- 1,954 canonical states and 1,954 encoded inputs; 422 repeated rows by either
  identity (equivalent for this corpus).
- The report confirms 1,526 PUCT and 850 exact-root rows. PUCT >32-stone:
  800 rows, 363 repeated inputs, entropy 0.560825 nats, one-hot 11.75%, 800
  usable. PUCT 17–32: 726, 33, 0.084087 nats, 33.88%, 726. Exact-root ≤16:
  850, 26, 0 nats, 100%, 850. The remaining subgroup cells are empty.
- `train.load_jsonl` accepted all 2,376 rows with default policy and value-target
  modes. The report also reconciles 2,376 actions, 64 complete games (48 wins
  for player 0, 12 for player 1, 4 draws), and 937 extra turns.

These checks establish pilot-data integrity and basic usability, not model
strength or training benefit. They do not initiate a training ablation. Seed488's
closed F advancement decision remains closed; the known historical seed461
overlap limitation remains. This generated corpus is not an untouched holdout.
