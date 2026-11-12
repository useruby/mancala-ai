# Varied-opponent P-coverage pilot results

## Verification

The pilot ran from detached `origin/main` at `674a98c902e0f0f23cb1fa510f2e3e26eeac0276` after the registered no-games preflight passed. The reviewed freeze digest was independently supplied and recomputed as:

`8da846678e89568e17e7bd81270287ef373e1f8acdfcbc892317d08385ad4c36`

The registered launch used 18 pair-level workers, one search worker per game, `CPUQuota=1800%`, `CPUWeight=100`, `MemoryMax=38654705664`, CPUs `14-31`, and nice `19`. There was one runner, one completed attempt per pair, and no retries.

Replay verified all **256 frozen assignments** across **128 seat-swapped pairs**. All **256 games completed**; none were capped or incomplete. The 128 attempt records are completed and reconcile to the 128 pair checkpoints. Accepted rows total **1,961 control** and **1,971 treatment**. CPU reconciles as pair-process CPU plus reaped native-child CPU; no started attempt is unaccounted.

| Arm | Games | Accepted P rows | Process CPU (ns) | Native-child CPU (ns) | Total CPU (ns) | Total CPU (hours) |
|---|---:|---:|---:|---:|---:|
| Control | 128 | 1,961 | 279,483,241,840 | 1,065,615,393,000 | 1,345,098,634,840 | 0.373638510 |
| Treatment | 128 | 1,971 | 269,137,154,249 | 1,049,971,886,000 | 1,319,109,040,249 | 0.366419178 |

The process/native split and total are independently reconciled in each raw checkpoint and attempt receipt.

## Distinct eligible P-turn coverage

Eligible rows are accepted designated-P rows with **more than 32 active stones**. The raw eligible-row counts are 402 control and 411 treatment. In this pilot, these occurrences are also distinct within each arm under both identity schemes.

| Identity scheme | Control distinct | Treatment distinct | Shared | Control-only | Treatment-only | Control positions/CPU-hour | Treatment positions/CPU-hour | Treatment/control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Canonical state SHA-256 | 402 | 411 | 289 | 113 | 122 | 1,075.91 | 1,121.67 | 1.0425× |
| Production `kalah_v3` float32 SHA-256 | 402 | 411 | 289 | 113 | 122 | 1,075.91 | 1,121.67 | 1.0425× |

Both identity schemes yield the same counts for this evidence. The separate identities are still counted and reported independently by the verifier.

## Matched-opening opponent outcomes

Outcomes below are for designated P, comparing each treatment opponent over its 32 assigned openings (64 seat-swapped games) against the control P-vs-P results on those exact same 32 openings (also 64 games). The verifier output includes the win/draw/loss comparison for every matched opening.

| Treatment opponent | Treatment P wins | Draws | Losses | Same-opening control P wins | Draws | Losses |
|---|---:|---:|---:|---:|---:|---:|
| initializer | 33 | 3 | 28 | 30 | 2 | 32 |
| T512 | 31 | 3 | 30 | 33 | 2 | 29 |

These outcomes are descriptive pilot accounting, not strength screening.

## Frozen decision gate

The frozen gate requires treatment yield of at least **1.15× control under both identity schemes**. The observed ratio is **1.0425× under each scheme**, so the gate is not met. **Do not propose a larger generation run from this pilot.** The pilot is fully reconciled, so the result is not inconclusive.

## Evidence and read-only verification

Raw atomic pair checkpoints are preserved in `pairs/` (128 files); dispatch/start/completion attempt records are preserved in `attempts/` (128 files). The SHA-256 digest of the sorted path-and-file-hash manifest is `7c8d32ced1fd56ae71f261b9dcbc482b5248a62ce825409513bb9ad78aed2c0c`.

From the repository root, replay all games, reconcile assignments, attempt receipts and CPU, recompute both identity schemes, and print the per-opening matched results with:

```bash
python -m ml.alphazero_lite.verify_varied_opponent_pilot \
  --expected-freeze-sha256 8da846678e89568e17e7bd81270287ef373e1f8acdfcbc892317d08385ad4c36
```

The command is read-only and fails closed on missing, conflicting, or unreconciled evidence. No training, strength screening, sealed-suite access, or serving changes were performed; #483 remains closed.
