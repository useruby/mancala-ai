# Seed 604 B2048 training execution report

## Outcome

Training completed: the exact B512 reproduction gate passed at update 512,
training continued with the same live Adam optimizer through update 2,048, and
the independent result verifier passed. This records successful execution and
artifact verification; it does not demonstrate improved playing strength.
No evaluation games were authorized or run.

## Authorization and frozen identities

- Repository commit: `2ec7b7291760cea587936eee3cfa1ce336e5f735`
- Owner approval: Aleksei explicitly authorized this training in the execution
  conversation before the create-only authorization record was written.
- Authorization: `training-authorization.json`, SHA-256
  `9887d00e4f957c2f7bd73a1c937ba5c13500a7e9c66ffccbc8302a219332d7f4`
- Execution freeze SHA-256:
  `205d8de862758d43d139ee29fe9cdf0dd018f73b2fbe4e165a61b329a9d4549f`
- Readiness evidence SHA-256:
  `4d6dafd68a0774d51c68efabb6f1e7a950f98771d3c49f8368fa99211e0c3586`
- Freeze validation attestation SHA-256:
  `1c5a93f1bf3a46d137c25c331f992ad8854f256acd994bf4106373ccb130189e`
- Runtime: CPython 3.14.7, NumPy 2.5.2, PyTorch 2.13.0+cu130; CPU,
  one PyTorch thread, deterministic algorithms enabled, and the registered
  `PYTHONHASHSEED=599`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` settings.

The freeze binds the registered immutable P-E4 parent, Seed 599 B512 checkpoint,
2,048-state membership, training configuration, deterministic schedule, v5
opening cohort evidence, and certified source files by SHA-256. The independent
read-only preflight, bounded-exclusion verifier, opening-cohort verifier,
runtime-certification-v3 verifier, freeze verifier, attestation verifier, and
executor `load_freeze()` all passed before authorization and execution. The
freeze's `source_sha256` map is the exact per-file source identity manifest,
authenticated by the freeze SHA-256 above.

## Execution and verification

- Output: `training-run-B2048/`
- Updates completed: 2,048; fresh Adam was initialized from the immutable P-E4
  checkpoint (the historical B512 checkpoint was not used as the training
  starting point).
- At update 512, the executor compared all 22 named model tensors with the
  immutable Seed 599 B512 checkpoint. Names, shapes, dtypes, and tensor values
  matched exactly; frozen value-network parameters were unchanged; Adam state
  was complete and finite at step 512 for each trainable parameter.
- After passing the gate, the same live Adam optimizer continued through
  updates 513–2,048. The recorded continuity checks and optimizer state at the
  update-513 boundary passed.
- Frozen value-network tensors remained unchanged in the final checkpoint.
- Independent verifier:
  `ml.alphazero_lite.verify_seed604_training_result` passed. It authenticated
  the authorization/freeze, trace and schedule, B512 tensor comparisons, Adam
  continuity, checkpoint structure, frozen tensors, and artifact identities.
  This verifier checks published execution evidence; it is not an independent
  rerun of optimization.

## Result artifact identities

| Artifact | SHA-256 |
|---|---|
| `training-trace.jsonl` | `ffd63c762307e1297367f1d8a5e9b5a093ec78afeb735784eb049b941b7a4480` |
| `b512-reproduction-report.json` | `38bd6b01207f6103579f43b3d931f1888e314991b2d48f45f4eff34a92508491` |
| `seed604-B2048-update-2048.npz` | `e2460325824be6ed5c416022744ffe4f22fe277f7f838fcc47df8732f0a04874` |
| `training-receipt.json` | `d9a19ee701f2abdf5c73064330b826114911a20b24d2f519c0eea49377047435` |

The receipt records the freeze, authorization, registration, membership,
schedule, trace, checkpoint, B512 report, and runtime identities.

Training loss is not evidence of playing-strength improvement. Playing-strength
evaluation remains separately gated and was not performed by this task.
