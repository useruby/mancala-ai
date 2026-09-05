# Native Hybrid Exact-Teacher Feasibility

Classification: `native_hybrid_exact_teacher_feasible`.

The native MTD(f) probe used the regenerated canonical tier-18 KVTB artifact, not `ExactKalahSolver`, as the measured teacher. The artifact had 172,986,450 states, a 172,986,834-byte file, payload SHA-256 `9dc63f403bfcfeb5df06eb4ca358b6276cc73fbaccb3a28ce9cebe913dedb66a`, complete-file SHA-256 `6e65399387f4b9c13bd83568c60a35fbe474a6bdb8ec868002b783eb0abb02ac`, and zero cycles.

The frozen seed-271 corpus SHA-256 is `fe9c317f6dace7d5c77eb7214db4739643a5b7d3ac0b83a39be885119b459a59`. It contains 96 training-ineligible states: 32 each in the 17-24, 25-32, and 33-40 active-stone buckets. Each request had an independent 30-second limit.

Warm process: 96/96 solved, with all action values and deterministic repeats. Mean runtime was 0.436674 seconds, p50 0.001978 seconds, p90 0.657232 seconds, p95 3.481869 seconds, and total elapsed time 41.920729 seconds. Historical-formula projection is 197,859 labels per 24 hours. Every bucket solved 32/32.

Measured successful-label CPU throughput was 16,415 labels per CPU-hour. There were no failed attempts, so the conservative projection including failed attempts remains 197,859 labels per 24 hours. One tier-18 generation took 147.161051 seconds; including that generation once gives 197,170 labels per 24 hours.

Fresh process: 96/96 solved. Mean runtime was 0.599562 seconds and total elapsed time 57.557986 seconds. Root values, action values, and optimal actions matched the warm process for every state.

Reproduction:

```bash
python -m ml.alphazero_lite.run_kalah_v1_tier18_experiment --output /tmp/kalah-v1-tier18-regeneration.json
artifact=$(python -c 'import json; print(json.load(open("/tmp/kalah-v1-tier18-regeneration.json"))["artifact_path"])')
probe=$(bash script/ai/setup_native_mtdf.sh /tmp/girving-kalah-native)
PYTHONPATH=. python -m ml.alphazero_lite.run_native_hybrid_feasibility --native-probe "$probe" --artifact "$artifact" --output docs/data/alphazero-lite-kalah-v1-tier18-hybrid-warm.json
PYTHONPATH=. python -m ml.alphazero_lite.run_native_hybrid_feasibility --native-probe "$probe" --artifact "$artifact" --fresh --output docs/data/alphazero-lite-kalah-v1-tier18-hybrid-fresh.json
```

Recommend a separate exact-label-production experiment.
