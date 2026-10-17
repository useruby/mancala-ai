# Seed483 historical-probe exclusion closure

**Status: incomplete. Generation and candidate inference remain blocked.**

The experiment-by-experiment source inventory, classifications, and exact source
hashes are in `seed483-source-coverage.json`. The compressed
`seed483-historical-probe-identities.json.gz` contains the reconstructed
identities added beyond the seed482 baseline, with source/game/ply provenance.
`seed483-recovered-seed461-arena-games.tar.gz` preserves receipt-matching game
bytes for relocated verification. The read-only reconstruction and verifier are
implemented in `ml/alphazero_lite/seed483_exclusion_closure.py`.

## Union reconstruction

The seed482 baseline is 62,330 canonical/float32 identity pairs, file SHA-256
`129391fc88118a99552029afaafb1880f1daed62476b2d1eafd7e931f621dc60d`, internal
identity `38e97a88041afd3a7f2e580f30155d17fae93f77ddf9267a6e663ca644b758b8`.
Seed483 independently rebuilt that baseline from the seed466 training/evaluation
arrays and the seed427 membership rows; the ordered pair-set digest is
`604d7d177fd0db50c8346b7e54f036c59d4dd4c1109849eaa07eb5b0b59fbdbe`.

Reconstructed additions:

| Evidence | Evaluated/probed occurrences | Distinct pairs | Added to prior union |
| --- | ---: | ---: | ---: |
| Seed429 A/B arena trajectories | 66,497 | 46,270 | 45,776 |
| Seed461 published opening roots | 2,304 | 2,304 | 2,304 |
| **Total reconstructed** | — | — | **48,080** |

The independently rebuilt currently available union contains 110,410 pairs and
has canonical pair-set SHA-256
`74afb0d0a08d1c03d132124089213735843c1eb19497c3af653fdc672cbfc4bd`. The
48,080 additions are separate from the missing/unreconstructed coverage listed
below; this is not a complete historical union. Seed429 sources are
`arena-evidence/A/A-games.jsonl` SHA-256
`0b4a63c05c62bbccaaaecae48e96c0ca9476c857681ad4e508c74c9bbafdc668` and
`arena-evidence/B/B-games.jsonl` SHA-256
`db6914e509b01bd45bf85394f979a46bc5f379d8b1468be9a55cee64ebfcf480`.

## Remaining coverage gaps

The 58 seed461 arena game streams are byte-for-byte receipt-matched and packed
in the recovered archive. Their published trajectories do not currently replay
under the canonical Kalah transition from their registered opening state: for
example, the first A1 game starts from the row's recorded opening, but its first
trajectory action `7` is illegal at ply 0. The opening root identities are
reconstructed; continuation-state identities are not certified. The ledger
records every original `.tmp/...-games.jsonl` path and its expected SHA-256.
Across those streams it records 1,129,643 trajectory actions whose state
identities remain uncovered.

The committed seed467 generation receipt identifies the missing source
`.tmp/seed455-nextgen-s467-parent-anchor-w010/runs/seed455-nextgen-s467-parent-anchor-w010-iter1/self_play.jsonl`, expected SHA-256
`6f7038dccf282fb9365a031a06f8c5a8af4a2b8a6eda32725fe125b06d654a76`, size
749,547,469 bytes. Its self-play states used by fresh validation/training
diagnostics cannot be reconstructed from the committed receipt alone.

No seed-specific committed source artifact was found for 458, 459, 460, 463, or
465. The ledger records the exact namespace searches (`docs/data/seed458-*`,
etc.); no exact source path or expected SHA-256 is recorded in committed
evidence for those numbers, so the missing population cannot be bounded. They
are `uncovered`, not evidence-backed empty contributions. The table also records
source-bound subsets and evidence-backed empty contributions separately. A
coverage label or experiment list alone never passes the generation
authorization function.

## Reproduction and authorization

From any checkout containing the committed seed483 sources and evidence:

```bash
python -m ml.alphazero_lite.seed483_exclusion_closure \
  --root . \
  --output docs/data/seed483-historical-probe-identities.json.gz \
  --ledger-output docs/data/seed483-source-coverage.json
```

The independent verifier reconstructs source bytes, identities, and the union;
it does not rewrite its inputs. The `incomplete` receipt intentionally prevents
authorization. Generation becomes eligible only after every experiment in
427–480 has an independently reconstructed or verified-empty contribution and
the exact final union is rebound and independently verified.
