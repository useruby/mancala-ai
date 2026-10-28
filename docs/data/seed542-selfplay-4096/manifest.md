# Seed542 expanded self-play corpus: pre-generation freeze

Status: generation interrupted after one complete shard and one preserved
partial attempt; training and strength evaluation are not authorized.
This is a data-volume experiment and retains the seed500 behavior model.

## Immutable shard assignment

There are 64 planned shards of 64 games each. Base seeds are assigned in order
542000–542063. Shards 542000–542047 are training-source (3,072 requested
games); shards 542048–542063 are development-source (1,024 requested games).
Each shard is one production `ml.alphazero_lite.self_play` invocation and has a
unique output path. Existing output paths are never overwritten. A failed or
interrupted invocation retains any bytes it produced and is not automatically
rerun.

## Authenticated behavior and generation recipe

The behavior model is `seed48-nextgen-s455-default-value-iter1`, exactly as in
the seed500 manifest. SHA-256: weights
`f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c`, runtime
sidecar `b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d`,
initializer checkpoint `7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c`,
native probe `d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48`,
and tablebase `f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7`.
The initializer bytes are archived as `initializer.npz`; generation used the
byte-identical `.tmp/seed500-current-checkpoint.npz` copy.
Generator, rules, production loader, and pipeline materializer hashes are
respectively `76718efde7ccfb708f4bb0a4314f812bab746d1364cb75ce5c8b3551b635fc0c`,
`6e5338ef9a85b690f51b6d0d184c8bc0f54c2b28ac084a4a17a88f9f552ad036`,
`2dcb42f8dff790d5033e2c9a865f3c24eb1fee6b9b37302200dd176d109dad32`, and
`ad1c9919e13418c4af53be9856406272bdb4f8ed9a53e7424cc9364364772dfa`.
The runtime sidecar selects `puct_exact_root_hybrid`, exact-root threshold 16,
final-score-margin objective, network-prior/lowest-index tie break, and disabled
exact-leaf solving. The model is not the currently served P composition.

All generation settings are seed500-identical: kalah_v3, PUCT, 128 simulations,
c_puct 1.25, temperature threshold 10, temperatures 1.0/0.1, Dirichlet
alpha/epsilon 0.3/0.25, no progress schedule, 200-ply cap, one worker, default
policy/value targets, selected-one-hot exact-root targets, denoised policy
targets, standard 4×6 starts, and game metadata. Exact command template:

```text
/home/alex/Mancala/ai/.tmp/venv/bin/python -m ml.alphazero_lite.self_play --out docs/data/seed542-selfplay-4096/shard-<seed>.jsonl --games 64 --seed <seed> --checkpoint .tmp/seed500-current-checkpoint.npz --runtime-search-policy-artifact model-artifact/current --input-encoding kalah_v3 --player-mode puct --simulations 128 --c-puct 1.25 --temperature-threshold 10 --temperature 1.0 --temperature-late 0.1 --dirichlet-alpha 0.3 --dirichlet-epsilon 0.25 --schedule-progress-mode none --max-moves 200 --workers 1 --policy-target-mode default --exact-root-policy-target-mode selected_one_hot --policy-target-noise-mode denoised --value-target-mode default --start-state-mode standard_4x6 --write-game-metadata
```

Execution is outside the API container in a user systemd scope limited to
CPUQuota=25%, CPUWeight=1, MemoryMax=4G, nice=19, pinned to logical CPU 31.
This is a host-side throttle; the API container's cgroup is not visible in this
host namespace and its own quota has therefore not been measured.

## Environment disclosure

The host Python is 3.14.7, Linux x86_64/glibc 2.44. The available generation
environment is Python 3.14.7, NumPy 2.5.3, PyTorch 2.14.0+cu130. Seed500 used
NumPy 2.5.2 and PyTorch 2.13.0+cu130. These dependency differences are
disclosed; historical environment parity is not claimed. Exact package/runtime
inventory and command outcomes are recorded in `execution-inventory.jsonl`.

## Recovery contract

`execution-inventory.jsonl` contains the complete command, seed, start time,
environment, and finish evidence for seed 542000, and the start record for
542001. The constrained service was explicitly stopped; its end time, elapsed
time, termination, staging-output path, and staging-output hash are preserved in
`execution-interruption.json`. Seed 542001's staging bytes are retained verbatim
and were not promoted to a complete shard or regenerated. The 62 unattempted
shards remain absent. `partial-status.json` reconciles requested, complete,
capped, incomplete, and unstarted game counts. The full-corpus validator,
development membership, and training schedules cannot be completed from this
partial corpus, so training remains unauthorized.

## Execution-source freeze

Before generation, the command runner
`ml/alphazero_lite/run_seed542_generation.py` is frozen at SHA-256
`eeb6293d948e49b987044d0cfd24a0baa032598b602caa4696d2eb5d95f3c73d` and its
focused tests `ml/alphazero_lite/test_run_seed542_generation.py` at
`4acd2583ff0c66ccebd6c536b19a179464a4373fb57c9dc36d2d1a1afcd1c4fe`. Tests
passed (5/5), and Ruff check/format passed before generation. The archived
runner bytes are bound by that digest; dependency versions and actual invocation
records remain explicit in the inventory.
