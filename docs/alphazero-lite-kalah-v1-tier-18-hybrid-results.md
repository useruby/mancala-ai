# Kalah V1 Tier-18 Hybrid Results

**Classification:** `native_hybrid_exact_teacher_validation_incomplete`

The canonical tier-18 artifact validated successfully. The native adapter's
tier-offset decoder was corrected and the cited hybrid gate now passes. The
full required hybrid correctness suite has not yet been completed, so
performance testing remains forbidden.

## Validated Artifact

- Indexed states: `172,986,450`
- Edges: `622,751,220` (`290,736,482` same-tier; `332,014,738` lower-tier)
- Payload SHA-256: `9dc63f403bfcfeb5df06eb4ca358b6276cc73fbaccb3a28ce9cebe913dedb66a`
- Complete-file SHA-256: `6e65399387f4b9c13bd83568c60a35fbe474a6bdb8ec868002b783eb0abb02ac`
- File size: `172,986,834` bytes
- Two fresh generations were byte-identical; wall times were 147.37 and 147.46
  seconds, and peak RSS was 611,088 and 611,568 KiB.
- Portable-format checks, tier-8 and tier-12 payload-prefix checks, cited
  positions, exhaustive tier-8 oracle values/actions, the deterministic 10,000
  state tier-9--12 oracle sample, and the 6,000-state tier-13--18 independent
  Bellman sample passed. The latter included both players, extra turns,
  captures, terminal sweeps, and one-sided states.

The complete machine-readable record is
`docs/data/alphazero-lite-kalah-v1-tier18-experiment.json`. The generated
artifact is retained only under `/tmp` and is not committed.

## Corrected Hybrid Gate

With `NATIVE_CANONICAL_KVTB` set to the validated artifact, the corrected
native probe now returns the canonical action values below.

```text
pits=(0,1,1,0,0,0,0,1,1,0,0,0)
stores=(20,20)
player=0
```

```text
actions={1:-4, 2:0}
```

The 12-case hybrid gate and 10,000 reachable-state transition gate also pass.
The bounded adapter suite rejects all 19 portable-format malformed fixtures and
directly compares raw values, offsets, and legal actions against the canonical
reader across tiers 0--18 for both players. The same suites pass under ASan /
UBSan and bounded `NO_TT=1` checks. Fresh-process benchmark and complete hybrid
checks have not been recorded yet. No labels were generated.

## Reproduction

```bash
python -m ml.alphazero_lite.run_kalah_v1_tier18_experiment \
  --output docs/data/alphazero-lite-kalah-v1-tier18-experiment.json
probe=$(bash script/ai/setup_native_mtdf.sh /tmp/girving-kalah-tier18-hybrid)
NATIVE_CANONICAL_KVTB=/tmp/kalah_v1_tier18_mmiqjum_/kalah_v1_18.kvtb \
NATIVE_MTDF_PROBE="$probe" \
python -m unittest ml.alphazero_lite.test_native_mtdf_probe -v
```

## Frozen Feasibility Preflight

The unchanged 96-state preflight completed with seed `271`, three 32-state
buckets, and a 30-second limit per state. It solved `0/32` states in each of
the `17--24`, `25--32`, and `33--40` buckets, so its qualification result is
false. The complete per-state record is
`docs/data/alphazero-lite-kalah-v1-tier18-hybrid-feasibility.json`.

This is the existing Python `ExactKalahSolver` runner, not a tablebase-enabled
native hybrid benchmark; it does not provide the required hybrid tablebase-hit
or native search diagnostics and therefore does not change the incomplete
hybrid classification.
