# Kalah V1 Tier-18 Hybrid Results

**Classification:** `native_hybrid_exact_teacher_incorrect`

The canonical tier-18 artifact validated successfully, but the isolated native
MTD(f) adapter failed its first hybrid correctness gate. Per the preregistered
order, no feasibility performance test or 96-state corpus run was performed.

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

## Hybrid Failure

With `NATIVE_CANONICAL_KVTB` set to the validated artifact, the corrected
native probe returned action value `3` for action `2` of the cited state below.
The canonical tablebase and Python oracle both return `0`.

```text
pits=(0,1,1,0,0,0,0,1,1,0,0,0)
stores=(20,20)
player=0
```

This is a score-orientation or native rank/adapter defect. The 12-case hybrid
gate did not pass, so the required transition, sanitizer, fresh-process,
throughput, warm-cache, and unchanged 96-state feasibility gates were not run.
No labels were generated and no application code was changed.

## Reproduction

```bash
python -m ml.alphazero_lite.run_kalah_v1_tier18_experiment \
  --output docs/data/alphazero-lite-kalah-v1-tier18-experiment.json
probe=$(bash script/ai/setup_native_mtdf.sh /tmp/girving-kalah-tier18-hybrid)
NATIVE_CANONICAL_KVTB=/tmp/kalah_v1_tier18_mmiqjum_/kalah_v1_18.kvtb \
NATIVE_MTDF_PROBE="$probe" \
python -m unittest ml.alphazero_lite.test_native_mtdf_probe -v
```
