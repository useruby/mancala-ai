# Seed484 provenance reconciliation

**Readiness: incomplete; generation and candidate inference remain blocked.**
This addendum leaves all seed483 files and the seed483 confirmation contract
unchanged. `seed484-pr-experiment-provenance.json` is the explicit PR graph;
`seed484-seed467-recovered-identities.json.gz` is the portable, source-bound
conversion; `seed484-extended-probe-union.json.gz` is the independently rebuilt
union extension. `ml/alphazero_lite/seed484_provenance.py` supplies read-only
verification and reconstruction commands.

## PR identifiers and experiment populations

| PR | Experiment seed | Activity | Population effect |
| ---: | ---: | --- | --- |
| 458 | 457 | Preparation correction | No probe launch; registration preparation only. |
| 459 | 457 | Verification correction | Reuses the seed457 registration and population. |
| 460 | 457 | Original execution; verification failed | Consumed the registered seed457 population; the failed verifier does not erase the execution. |
| 461 | 457 | Post-execution correction replay | Reuses the seed457 execution population; no new population. |
| 463 | 462 | Post-execution correction | Replays seed462 evidence bound to its original registration/recovery receipt. |
| 465 | 464 | Audit-packet publication | Packages existing seed464 evidence; no new run or probe population. |
| 467 | 466 | Runtime-reader authentication | Read-only access to the seed466 bundle; no new run or probe population. |

Mappings are checked against concrete changed-source paths and receipt-bearing
artifacts in the JSON graph. Hash-bound registration/receipt files include the
seed457 registration and original receipt, seed457 execution status and
correction receipt, seed462 registration and correction receipt, seed464 packet
manifest, seed466 bundle manifest, and seed467 generation receipt. PRs #458–461
all map to experiment seed457; the separate historical experiment seed461 is
also registered and remains unresolved below. PR #463 maps to seed462, #465 to
seed464, and #467 to seed466. This graph does not infer that similarly numbered
experiments ran or label them empty. The existing 427–480 experiment inventory
is retained separately; seed457 is already classified there as a verified
subset with a 62,330-pair population proof. The matching experiment rows for
seeds 457, 462, 464, and 466 are verified subsets of the same authenticated
seed427/seed466 base population in the seed483 ledger. This closes the mapped
population edges for PRs #458–461, #463, #465, and #467 without counting a
correction or publication as a second population. The apparent numeric entries
458, 459, 460, 463, and 465 have no same-seed registration or PR lineage in the
repository inventory; they are **unregistered / not classified empty**, not
invented population gaps.

## Distinct historic seed461 arena lineage

Seed461 is independently required: its registration and evaluation bindings
register 512-game-per-candidate arenas on the archived opening suites (for
example `seed461-e3-e4-registration.json` and
`seed461-cross-order-e4-average-registration.json`). All 58 recovered game
files are receipt-matched by the seed483 archive verifier.

The archived writer is `ml/alphazero_lite/arena.py` at merged PR #390 commit
`ec87dc4a77dcd9d31ae0e0dcba08c3bf4f20de0a` (SHA-256
`85971b30fa8579da7eac1fb6896d8165c6a8c8f7d4c00e878c34e8f848769554`). Its
loop applies the opening prefix before initializing the recorded trajectory;
the recorded trajectory action is `game.pit_index(move)`, an **absolute pit
index**, and is applied unchanged to the game. No seat canonicalization or
relative-action offset is performed by that writer. The registered prefix is
also absolute: replaying the first seed461 cosine-LR prefix `[4,10,2,0,9,3,7]`
from the standard initial position produces the exact declared opening state.
That declared state is therefore the documented pre-action state for the
trajectory.

The recovered first A game then has trajectory action `8`, but the registered
opening state has `current_player: 0` and legal absolute pits `[0,1,2,4,5]`.
Strict replay rejects it at ply 0. The archived writer’s start-state/action
contract and the recovered row are inconsistent; the verifier does not rotate
seats, subtract an offset, or try alternative replays. Registered seed461
opening roots remain covered by the seed483 union. The uncovered contribution is
the continuation-state population represented by **1,129,643 recorded action
pre-states across 58 streams**; its distinct identity count cannot be asserted
until an authoritative replay representation is recovered.

## Distinct historic seed467 generation lineage

Seed467 is independently required by the generation receipt
`docs/data/alphazero-lite-generations/seed455-nextgen-s467-parent-anchor-w010/generation.json`.
It records the self-play input path, role, expected size (749,547,469 bytes),
and SHA-256
`6f7038dccf282fb9365a031a06f8c5a8af4a2b8a6eda32725fe125b06d654a76`. The
completed run manifest identifies seed467 self-play and the downstream
diagnostic inputs. The original `.tmp` bytes were found and match the receipt
exactly; they were read only and their bytes and mtime were preserved.

The conversion reads all 66,448 source rows, decodes the registered 27-feature
state representation, and records line/game/move provenance. It yields 46,276
distinct canonical/float32 identity pairs. The compressed snapshot is
portable and independently checked for its receipt-bound source hash, row
count, each reconstructed state identity, and each pair; snapshot SHA-256 is
`cfd3e83038b8f93978e449df81f38a39009e79370d3b97983a68f368a84ef16b`. The raw
log was re-hashed against the generation receipt during conversion. The portable
verifier checks the snapshot against that recorded source/receipt identity and
reconstructs every identity pair; it does not require the raw log. The log
itself remains in the local ignored `.tmp` run directory and is not copied into
the PR.

## Rebuilt union and readiness

Seed483’s 110,410-pair union was independently reconstructed from its committed
sources and verified before extension. Seed467 contributes 46,276 distinct
pairs, of which 39,346 are new to that union. The extended union has **149,756
pairs**, pair-set SHA-256
`d7c3d782181af654d56da2d02b091047df9a9cc5f78795b2cb687c9b528dfcf6`. The
additions and overall union are recorded separately in the extension manifest.

Remaining blocker: seed461’s 1,129,643 continuation pre-states are not
identity-reconstructed because the first recorded action contradicts the
registered root under the archived writer contract. Seed461 roots and all
seed467 identities are included in the extended union; the seed461 continuation
population remains explicitly uncovered. The five PR-number confusions are
resolved as provenance edges, not mislabeled as empty experiment contributions.
No generation authorization is granted.

## Verification commands

```bash
python -m ml.alphazero_lite.seed484_provenance --root . --verify-union docs/data/seed484-extended-probe-union.json.gz
python -m ml.alphazero_lite.seed484_provenance --verify-snapshot docs/data/seed484-seed467-recovered-identities.json.gz
```

To repeat source conversion while the receipt-matched original remains
available, use `--reconstruct-seed467 --output PATH`. To rebuild and publish the
extended union, use `--build-union --output PATH`. Neither command launches a
game, search, or model inference.
