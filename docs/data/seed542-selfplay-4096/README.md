# Seed542 expanded self-play corpus (incomplete)

**Preparation outcome: incomplete; training unauthorized.** Generation was
explicitly stopped under the configured host resource scope. No training,
candidate evaluation, promotion, or deployment was run.

The plan requested 4,096 games in 64 immutable 64-game shards. One shard
(seed 542000) completed: 64 games, 2,216 rows, zero caps, and zero incomplete
games. Its complete JSONL was replay-validated and all 2,216 rows were accepted
by the production loader. It contains 1,798 unique canonical states and 1,798
unique production-float32 inputs.

Seed 542001 was interrupted after 322.35 seconds. Its worker staging file is
preserved at the path and SHA-256 in `execution-interruption.json`: 1,776 rows,
51 complete games, one incomplete game, and 12 games not started. Replaying its
51 complete games validated 1,768 rows; the production loader accepted all
1,776 staged rows. It is not counted as a completed shard. Seeds 542002–542063
were not attempted. Across the plan, 115 games completed, 0 were capped, and
3,981 were incomplete or not started. Per-command timing, hashes, and the
environment delta are in `execution-inventory.jsonl` and
`execution-interruption.json`.

The historical seed500 initializer, runtime sidecar, native probe, tablebase,
generator, rules, loader, and pipeline identities were authenticated against
the original manifest. The available environment differs from the historical
seed500 environment: NumPy 2.5.3 vs 2.5.2 and PyTorch 2.14.0+cu130 vs
2.13.0+cu130. No historical dependency parity is claimed. The host-side
CPUQuota=25%, CPUWeight=1, MemoryMax=4G, nice=19, CPU-31 scope was applied; the
API container's own cgroup was not visible from the host namespace.

The full-corpus accounting, common development cohort, and matched 108-update
training schedules were not prepared because the requested corpus is
incomplete. This data-volume experiment changes no behavior-model identity or
search setting. The later protocol must retain seed504's joint policy/value
objective and final-update-only endpoint. No historical-unseen or
fresh-confirmation claim is made. `ml/alphazero_lite/verify_seed542.py` is
read-only and has a partial-corpus mode that independently replays the complete
shard and the completed games in the preserved staging file, checks production
loader acceptance and archived hashes/counts, and reports this incomplete
status. It does not claim to verify a development cohort or training schedule,
which do not exist for this partial run.
