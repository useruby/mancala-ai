# Seed485 seed461 arena provenance forensics

**Status: incomplete; generation remains blocked.** The machine-readable
inventory in `seed485-seed461-forensic-inventory.json` records all 58
receipt-matched streams and is rebuilt by the read-only verifier
`ml/alphazero_lite/seed485_forensics.py`. The published seed483 confirmation
rules and the seed483/484 union artifacts are unchanged.

## What is authenticated

For each archive member, the verifier binds the exact game-log SHA-256 to its
evaluation-binding receipt and records the binding path/hash, registration
reference/hash where present, suite digest, candidate record, game rows and
indices, seat-related row fields, prefix/root fields, and current source/version
references. All 58 game-log hashes match. Forty-eight streams resolve their
registration hash to a committed registration file. Forty-six streams have
registered suite digests matching available opening-suite bytes. The remaining
ten streams refer to `seed461-lr-fixed-e4-evidence.json`: it carries game hashes
but no registration digest or suite digest. For each such stream those bindings
are explicitly absent, not borrowed from a neighboring LR-sensitivity run.

The archived PR #390 implementation is reachable at commit
`ec87dc4a77dcd9d31ae0e0dcba08c3bf4f20de0a`; `arena.py` has SHA-256
`85971b30fa8579da7eac1fb6896d8165c6a8c8f7d4c00e878c34e8f848769554`. That
version applies the opening prefix before beginning the trajectory and records
`game.pit_index(move)` as an absolute pit index. This is a competing documented
contract, not a source binding for any of the 58 runs. The committed run
registrations and evaluation bindings bind candidates, runtime, suite and game
receipts, but not the actual arena/rules source bytes used at execution. The
fixed-E4 evidence additionally omits registration and suite receipts entirely.
The inventory therefore reports writer and rules versions as unknown for every
stream. No trajectory has been replayed and no continuation identity has been
added.

## First-failure packet (conditional comparison)

The exact packet is also embedded in the JSON inventory. The first archived
cosine-LR E4 stream is
`/home/alex/Mancala/ai/.tmp/seed461-cosine-lr-ablation/arena-e4/order_38611_A-games.jsonl`,
SHA-256 `6bcf879780dbc21a9502f92333607c9c8c0439766389d2baa2dbcdc6a375a4ea`.
Its row 0 has `game_index=0`, `opening_index=0`, `game_within_opening=0`,
`challenger_player=0`, prefix `[4,10,2,0,9,3,7]`, and trajectory beginning
`8,9,...`. The binding suite SHA-256 is
`41fd667245f6d68848737373074cb5f7494fecce1686a59be8af4a89959d7e2c`, matching
`seed461-cosine-lr-ablation-openings.jsonl`; suite index 0 has the registered
root `{player_pits:[2,6,1,0,3,8], opponent_pits:[6,0,6,1,2,7], player_store:3,
opponent_store:3, current_player:0}`. The row has no separate embedded root;
its embedded prefix is recorded above. Under the current Kalah absolute-pit
orientation, player 0's legal pits from that registered state are `[0,1,2,4,5]`,
so first value 8 is not a legal absolute pit. Under #390's documented behavior,
the prefix is applied and the trajectory begins at this root. That is a concrete
conditional contradiction between the recovered row/root and #390; it does not
establish that #390 governed this execution or authorize a transformed replay.

The inventory preserves recorded per-row game/opening indices and both paired
stream receipts. It does not infer chunk offsets or resume semantics where the
source bindings do not record them. No per-game state hashes are present in the
archived game rows. Prefixes are embedded, roots are supplied by matching suite
bytes where those bytes are available, and the 10 fixed-E4 streams have no
authenticated suite association.

## Remaining exact gap and union

All 58 streams have unknown writer/action and rules bindings; the ten fixed-E4
streams also lack registration and suite bindings. The missing evidence is the
run-specific source snapshot/hash (or equivalent authenticated command/runtime
record) for each execution, plus the missing fixed-E4 registration and suite
receipt. The archived #390 writer and the current rules source are comparison
references only. Uncovered continuation population remains the 1,129,643
recorded trajectory action pre-states; no unique identity count is asserted.
The existing seed484 union remains 149,756 pairs with digest
`d7c3d782181af654d56da2d02b091047df9a9cc5f78795b2cb687c9b528dfcf6`; no
append-only extension is justified. Generation authorization remains false.

## Verification

```bash
python -m ml.alphazero_lite.seed485_forensics --root . --verify docs/data/seed485-seed461-forensic-inventory.json
```

The verifier reconstructs the stream inventory from committed binding files,
registered opening bytes, and the archived logs; it fails on changed/rebound
inventory entries and never chooses an action convention by default.
