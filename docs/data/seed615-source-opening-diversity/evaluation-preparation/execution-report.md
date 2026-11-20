# Seed 615 playing-strength evaluation execution report

## Execution identity

- Base commit: `3ce80dd80f7e8117da84f953e306ba9fe80dd878` (merge of PR #617)
- Active execution freeze: `execution-freeze-v4.json`; SHA-256 `ba30f5e3efadf3358db9f03c7b4eba4f69b23e2f95dcb4089723a1fb9664f32b`
- Owner authorization: SHA-256 `5e6664794fe06fbe9aba3ec331ccb628010fa8aa8553a0f84b16b7db722e85be`
- Authorized by GitHub owner `useruby`, who explicitly stated: “I authorize the run.”
- Runtime: CPython 3.14.7, NumPy 2.5.3, PyTorch 2.14.0+cu130, Linux x86-64, CPU only, one PyTorch/search thread per game, deterministic algorithms enabled
- Frozen environment: `PYTHONHASHSEED=615`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=""`
- Resources: 18 pair workers; 18 registered CPUs; 64 GiB registered memory limit; one search worker per game
- Search: 128 simulations per side; `c_puct=1.25`; deterministic root selection; exact-root threshold 16; 200-ply cap

## Authenticated inputs

| Input | SHA-256 |
| --- | --- |
| B512 checkpoint | `bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c` |
| S615 checkpoint | `cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780` |
| Serving P-E4 checkpoint | `2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad` |
| Native exact-root probe | `d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48` |
| Materialized tablebase (709,635,072 bytes) | `f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7` |

The v4 evaluation preflight, independent v4 preflight, independent opening verifier, and independent training-result verifier passed before execution. The preflight authenticated all 1,024 registered openings, all 4,096 assignments, deterministic seeds, candidate/seat ordering, source hashes, training receipt, and the candidate-policy / serving-P-E4-value composition. No game ran during preflight.

## Results and analysis

- Registered games: **4,096 completed** across **1,024 opening clusters**; both candidates played both seats against the same P-E4 opponent.
- Capped games: **0**. Failed, incomplete, or unresolved attempts: **0**.
- Independent game replay and result verification: **passed** for all 4,096 assignments and result artifacts.
- B512 mean seat-averaged score: **0.502197265625**.
- S615 mean seat-averaged score: **0.496337890625**.
- S615 − B512: **−0.005859375**; registered paired 95% opening-cluster bootstrap interval **[−0.01611328125, 0.00439453125]**.
- S615 versus P-E4 one-sided 95% bootstrap lower bound: **0.486572265625**.
- Incremental-improvement gate: **failed** (lower interval endpoint is not strictly above zero).
- Absolute-strength gate: **failed** (lower bound is not strictly above 0.50).
- Registered interpretation: **neither** claim established.
- Independent published-analysis verification, including bootstrap reproduction: **passed**.
- Candidate promotion: **none**; `automatic_promotion` is false.

The complete create-only evidence for every game—trajectory, arena report, exact command, opening prefix, and result receipt—is retained in `results/`. The registered analysis is `analysis.json` (SHA-256 `11d75d26ca18dc6b6168d1c9cb95821d0f49b21a67d7276a828e0bb700158b75`). Independent verification replays game outcomes; it does not claim independent numerical reproduction of game search.
