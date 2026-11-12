# Exact-root native probe reuse benchmark

## Frozen input

This benchmark reuses the published #580 workload only; it launches no games.
The source freeze digest is
`8da846678e89568e17e7bd81270287ef373e1f8acdfcbc892317d08385ad4c36`, and the
published pair/attempt evidence manifest digest is
`7c8d32ced1fd56ae71f261b9dcbc482b5248a62ce825409513bb9ad78aed2c0c`. The
verifier replayed all 256 games and 128 seat-swapped pairs before reconstruction.

`probe-reuse-manifest.json` preserves all 256 game boundaries in assignment
order, including games with no exact-root requests. It contains 3,354 exact-root
requests with pair/game/ply, state, legal actions, archived move, and evaluator
priors for the side to move. It cross-checks all 1,707 retained designated-P
exact-root rows for state, legal actions, selected move, and priors. Manifest
SHA-256: `d62bf636b5c7c10f09d2b7048ee4b109cc74fbe78a43c9ed12a1cef58801513a`.

The manifest binds native probe SHA-256
`d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48` and
tablebase SHA-256
`f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7`. The
execution freeze used one pair at a time, one search worker, CPU quota 1800%,
CPU weight 100, 36 GiB memory limit, CPUs 14-31, and nice 19. Variants alternate
which runs first by pair, starting with per-game.

## Results

| Probe lifetime | Reaped native child CPU | Startup wall | Warmup wall | Query wall |
|---|---:|---:|---:|---:|
| Fresh warmed probe per game | 971.651 s | 0.198 s | 965.180 s | 0.181 s |
| One warmed probe per seat-swapped pair | 485.656 s | 0.097 s | 482.183 s | 0.139 s |

Reusing the probe per pair reduced total reaped native-child CPU by **50.02%**.
There were **zero** between-variant legal-action margin or selected-move
mismatches, zero mismatches against the archived margins where retained, and
zero mismatches against the archived move on any exact-root turn. Per-pair CPU
and full mismatch evidence are in `probe-reuse-benchmark.json`.

The preregistered threshold is met. This supports a narrowly scoped data-
games in a seat-swapped pair. It does not change #580's decision: a probe
efficiency improvement common to both arms provides no evidence of an advantage
against varied opponents. #483 remains closed.

## Reproduction

From the repository root, reconstruct and freeze the requests, then benchmark:

```bash
git lfs pull --include=model-artifact/runtime/kalah_v1_21.kvtb
python -m ml.alphazero_lite.benchmark_exact_root_probe_reuse manifest
systemd-run --user --scope --quiet \
  --property=CPUQuota=1800% --property=CPUWeight=100 \
  --property=MemoryMax=38654705664 --property=AllowedCPUs=14-31 --nice=19 -- \
  taskset -c 14-31 python -m ml.alphazero_lite.benchmark_exact_root_probe_reuse benchmark \
  --manifest docs/data/varied-opponent-coverage-pilot/probe-reuse-manifest.json
python -m unittest ml.alphazero_lite.test_benchmark_exact_root_probe_reuse
```

The verifier recomputes the published evidence digest before building the
manifest. The benchmark command reads only the frozen manifest and local native
runtime inputs; it does not replay or launch games.
