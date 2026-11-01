# Controlled self-play evaluator pilot

## Scope and frozen protocol

This exploratory pilot changes only the behavior evaluator used by the
production self-play worker. It reuses the production game, search, target, and
serialization loop. Generation settings are held constant across arms; different
trajectories are expected consequences of the evaluator change. The pilot
measures generation correctness, cost, and data characteristics. It does not
measure improved playing strength and does not authorize training, promotion,
or deployment.

The immutable pre-generation registration and freeze are
`registration.json` and `execution-freeze.json`. Both arms used 64 games, one
worker, the same base seed `542500`, and the production per-game RNG derivation
`(542500 * 1_000_003) + game_index` for indices 0–63. The exact commands and
append-only attempt receipts are in `attempts.jsonl`.

The generation contract is seed542's: 128 PUCT simulations, `c_puct=1.25`,
standard initial state, temperature 1.0 through ply 9 and 0.1 thereafter,
Dirichlet alpha 0.3/epsilon 0.25 while action noise is enabled, no progress
schedule, denoised default visit-count policy targets, default terminal value
targets, selected-one-hot exact-root targets, exact-root threshold 16 active
stones, exact leaf solving disabled, one worker, and a 200-ply cap. All
remaining search options and disabled optional features are explicitly recorded
in the registration and command.

### Arms and identities

- **Control:** complete frozen seed542 initializer, checkpoint SHA-256
  `7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c`.
- **Treatment:** historical seed504 E4 policy through its own complete trunk
  (checkpoint SHA-256
  `2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad`) and
  initializer value through its own complete trunk. No tensor splicing. Policy,
  value, serving release files, sidecars, native probe, and materialized
  tablebase are hash-bound in `registration.json`.

The recorded interpreter was invoked explicitly from this project root:
`/home/alex/Mancala/ai/.tmp/seed542-continuation-prep/.tmp/seed542-recorded-runtime/bin/python`.
It reported Python 3.14.7, NumPy 2.5.3, and PyTorch 2.14.0+cu130; the complete
dependency closure and project-module path are frozen in the registration.
Both arms ran with the same systemd CPU quota/weight, 4 GiB memory limit, CPU 31
affinity, and nice level 19.

## Results

The independent verifier accepted all **128 games** and the production loader
accepted all **4,609 rows**. There were no capped or incomplete games.

| Measure | Control | Treatment |
| --- | ---: | ---: |
| Completed games | 64 | 64 |
| Accepted rows | 2,226 | 2,383 |
| Game length, mean (min–max) | 34.78 (15–50) | 37.23 (14–55) |
| Player 0 wins / player 1 wins / draws | 51 / 10 / 3 | 58 / 4 / 2 |
| PUCT rows / exact-root rows | 1,518 / 708 | 1,545 / 838 |
| Elapsed time | 425.61 s | 712.56 s |
| Throughput | 0.150 games/s | 0.090 games/s |
| Peak sampled process-tree RSS (4 GiB cap) | 2.32 GB | 2.32 GB |

Per-bucket diversity and descriptive target diagnostics:

| Arm | Active stones | Rows | Canonical unique | Production float32 unique | Mean target entropy (nats) | Repeated-state groups | Duplicate-target disagreements |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | >32 | 787 | 466 | 466 | 0.5634 | 71 | 0 |
| Control | 17–32 | 731 | 714 | 714 | 0.0781 | 17 | 0 |
| Control | ≤16 exact-root | 708 | 701 | 701 | 0.0000 | 7 | 0 |
| Treatment | >32 | 793 | 313 | 313 | 0.4275 | 80 | 0 |
| Treatment | 17–32 | 752 | 538 | 538 | 0.0634 | 115 | 0 |
| Treatment | ≤16 exact-root | 838 | 640 | 640 | 0.0000 | 130 | 0 |

The repeated-state disagreement diagnostic groups targets by canonical state
within an arm and reports whether any repeated state received more than one
target. These diagnostics describe the generated data; they do not establish
that policy targets are optimal. Exact-root and PUCT telemetry and target
generation settings were authenticated from the recorded rows and frozen
runtime contract.

`results.json` preserves per-game lengths, outcomes, loader shapes, bucket
summaries, and consumed-state SHA-256 identities in both canonical and
production-float32 form for future exclusion accounting. Seed461 provenance
remains unresolved and was not recovered for this pilot.

Evidence SHA-256 identities:

- Registration: `a893a718c70a7ef4691b0959acc6c228487852fb9c8ce976af4aeb1932d3b375`
- Freeze: `8c7c8d3b87cca8b33955fbac034437ff5c9539e8655428c9a3cfecbcec16f63d`
- Control JSONL / log: `34f7fca49b2efa5129693c19ecd2d5de3c5acf5ba017ec950e7563ed2dd1d86d` /
  `0087c4868f30ca14555d049aea3bda552ef1bfdb3489ad4bea3d3cf777369470`
- Treatment JSONL / log: `cef5913fbda716a610141462a2a60013192f7e2ddd69ad4018733495776c6321` /
  `c6c50e6c74aae7e51ba51b46a16b9f7ca9ffdfa7cebb4fdcb5ac70a0e953987e`
- Verification correction: `541a7f90191017220150cc89275c3b9403f82ca239fa57ba368432ea75930d15`
- Results: `d8e8170956db2cdfc45c0b2f5f70ca09f6706306944b9b89df73c721ce1f8205`

## Verification and correction record

The independent verifier replays actions through the rules engine, including
extra turns and terminal sweep; reconstructs each row's state, action, game
association, outcome, and value target; validates legal normalized targets;
checks search-profile hashes, search telemetry, target settings, and loader
acceptance; and counts completed, capped, and incomplete games. These checks do
not independently prove policy-target optimality.

An append-only verification correction records two issues discovered after
generation: production Dirichlet action noise turns off after ply 9, and the
production worker records the shard base seed on each row while deriving each
game's RNG stream from `global_index`. The correction binds the corrected
verifier, runner, and test hashes, documents the actual shared seed schedule, and binds both
output/log hashes. The original registration, freeze, commands, attempt
records, and outcome-bearing game bytes were preserved; neither arm was rerun.

Validation ran in the working tree and a physical copy while the original
worktree path was inaccessible. Both arms independently replayed and loader
validated in the copy. Verification preserved bytes and nanosecond modification
times for the raw arm outputs, logs, attempt ledger, registration, freeze,
correction receipt, and results. `physical-copy-verification.json` records the
copy verification commands and evidence inventory.

Focused pilot/self-play/runtime checks passed: 157 tests plus 11 subtests before
the append-only verifier corrections, then 12 pilot tests after the final
verifier checks. Ruff and `script/ai/run_python_quality.sh` passed. A broader
seed504 screen regression invocation could not run its fixture-based cases
because the physical checkout does not contain its ignored `batches/` and
`results.json` fixtures; the existing seed542 continuation regression also
expects its already tracked continuation inventory to be absent. Neither
failure involved pilot evidence.

## Reading the evidence

Success means all 128 games completed and independently validate with correct
evaluator routing and loader-accepted targets. It authorizes proposing a larger
controlled corpus experiment only. The findings are exploratory generation
characteristics; they do not establish playing-strength improvement and do not
authorize model training, arena evaluation, promotion, or serving changes.
