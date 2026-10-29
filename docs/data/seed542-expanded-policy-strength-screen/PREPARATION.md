# Seed542 expanded-policy paired strength screen

## Preparation status

This is an exploratory screen reusing seed511's frozen opening suite. It is not
a fresh holdout or fresh confirmation. No historical seed511 games or outcomes
are imported. The unresolved seed461 historical-overlap limitation remains in
force. The original PR #545 preflight record remains unchanged; its historical
full seed511 publication check stopped at the frozen native-source mismatch.
The new import path separately authenticates and replays the reused suite under
the recorded current rules/encoding identities, without weakening or changing
the historical verifier.

The screen was frozen before arena execution. One initial arena process exited
before loading a game because the worktree's tablebase location still contained
the registered Git LFS pointer while the sidecar correctly required the
materialized object. The exact command/attempt receipts and error are preserved
in `batches/prelaunch-failures/opening-0000/` and
`PRELAUNCH-FAILURE-0000.json`. No report, game, trajectory, or outcome was
produced by that attempt. After resolving this concrete runtime-path issue, the
first paired arena invocation produced raw evidence but failed the frozen
independent pair validator before ledger commit. Its raw evidence and exact
validation failure are documented in `EXECUTION-STATUS.md`; there is no
complete-screen result or strength decision.

## Imported suite and overlap disclosure

- Seed511 registration SHA-256:
  `89f04c6e1f774d81a7866d3913dd5ffa990b9e04a58f9a7f066b8f19204445c4`.
- Seed511 execution-freeze registration binding matches that exact hash.
- The imported `openings.jsonl` bytes match both freeze and registration, with
  SHA-256 `27a003322e5469507a508010a9743b40062928140e6a4b9d5628b0290e32c21d`.
- All 1,024 prefixes replayed under `arena_player_relative_v2`; player-relative
  and source absolute moves agree at every ply, including extra turns. Declared
  root states and hashes match; all roots are nonterminal and legal. Both
  canonical and production-float32 input identities are unique (1,024 each).
  Original order and suite bytes are retained exactly.
- Root player orientations are 524 for player 0 and 500 for player 1.
- Rules, input encoding, self-play encoding, and opening-suite dependency
  identities are authenticated in `suite-import-report.json`.

Every registered opening remains in the screen. The overlap files disclose:

| Training corpus | Training rows | Unique canonical / float32 identities | Canonical root overlaps | Float32 root overlaps |
|---|---:|---:|---:|---:|
| Original seed504 | 13,724 | 9,281 / 9,281 | 0 / 1,024 | 0 / 1,024 |
| Expanded seed542 | 109,504 | 55,445 / 55,445 | 12 / 1,024 | 12 / 1,024 |

The per-opening flags are in `overlap-per-opening.jsonl`; source manifests and
all training shard hashes are recorded in `overlap-summary.json`.

## Training and runtime identities

Both seed542 v2 training verifiers, including the supplement, passed again under
the documented CPU quota/weight, memory, affinity, nice, address-space, and
single-thread settings. Their stdout, stderr, exact commands, and return status
are retained in `training-verification.json`.

Candidate routing is fixed as follows:

- Challenger policy: D-final update108, SHA-256
  `6200b04d58f65ee5fb62555c3b32faea4b34206b4c10dbefb22ae9e7ff7b2328`, through
  D-final's own trunk.
- Opponent policy: current serving P's historical seed504 E4, SHA-256
  `2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad`, through
  E4's own trunk.
- Both value outputs: the identical frozen initializer checkpoint, SHA-256
  `7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c`, through
  the initializer's own trunk.

The D-final exporter output is reauthenticated in full. Its metadata, weights,
model, and search-sidecar hashes are respectively `5fb713bb7c6907037ffdff4c93ea47742af836f9187e18ef79f63be96764bce6`,
`6190bf992605400183b8e679f4cf33d103e7a26c5bf73e43906c6d875a128a52`,
`6200b04d58f65ee5fb62555c3b32faea4b34206b4c10dbefb22ae9e7ff7b2328`, and
`b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d`.

E4 and initializer serving identities match the checked-in `policy-value-p`
bundle and deploy release manifest. The native probe is
`d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48`; the
materialized tablebase is
`f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7`.
No checked-in serving artifact or deployment file was modified.

## Frozen arena and analysis contract

The frozen `registration.json` and `execution-freeze.json` bind the suite,
overlap disclosure, component artifacts and routing, runtime assets, training
verifier results, actual executable source bytes, package versions, 1,024
parser-validated arena commands, durable launcher, and analysis rules.

The planned screen is exactly 1,024 openings × two swapped-seat games, with 128
simulations per side, `c_puct=1.25`, one worker, existing deterministic search
options, exact-root threshold 16, final-score-margin objective, highest legal
network prior then lowest move-index tie-break, exact-leaf solving disabled, no
additional random opening plies, and a 200-ply cap. The seed contract is
`azlite_eval_seed_v2`, with base seed `545 + opening_index`.

Pairs are committed only after the read-only verifier validates their raw
reports and trajectories. Complete raw pairs interrupted before ledger commit
are recovered without rerunning; committed pairs are skipped; ambiguous,
partial, capped, or invalid attempts stop the run with their evidence retained.
Advancement requires all 2,048 games valid and a one-sided 95% opening-cluster
bootstrap lower bound strictly above 0.50. It permits proposing a separate
1,200-simulation evaluation only; it does not authorize deployment or promotion.
