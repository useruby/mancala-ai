# Future 1,024-opening arena holdback protocol

**Preparation only.** No actual 1,024-opening suite is generated or exposed by
this change, no candidate is evaluated or trained, and serving P is unchanged.
The tests use synthetic states exclusively.

## Exclusion source coverage audit

The starting rules-only builder is the #511 `seed511_fixed_composition_screen`
builder: uniformly random legal prefixes of 4–12 plies, `random.Random(seed)`,
reject terminal states and collisions in either canonical rule state or the
production little-endian float32 model input. Its source inventory is the
reconstructable #482/#483 historic union plus #488, #499, #505, #508 and #500.
The original #483 source ledger (`docs/data/seed483-source-coverage.json`)
records source publications and exact artifact SHA-256 values through #480;
the #511 registration and exclusion source list bind its later inputs.

| Source population / evidence | Receipt/publication and content identity | Canonical / float32 distinct counts | Coverage/gap |
| --- | --- | --- | --- |
| #482 baseline and #483 reconstructed additions | #483 `seed483-source-coverage.json`; baseline gzip SHA `129391fc88118a99552029afaafb1880f1daed62476b2d1eafd7e931f621dc60d`; union pair digest `74afb0d0a08d1c03d132124089213735843c1eb19497c3af653fdc672cbfc4bd` | baseline 62,330 identity pairs; reconstructed union 110,410 pairs | Not a complete historical union. #483 ledger has individual source artifact hashes and classifications. |
| #488, #499, #505, #508 arena and #500 self-play | #511 `registration.json` source hash bindings; #511 replayed published game streams and checked #500 shard hashes against accounting | Counts are in #511 exclusion accounting for the frozen run; not all inputs are independently reconstructable in this PR's clean #560 base without the #511 publication directory | These are known screened populations; their source receipts must accompany any future inventory. |
| #511 screen | #511 v4 publication receipt and raw batch hashes; replayed from its registered suite | 56,329 distinct canonical / 56,329 `kalah_v3` float32 states; 26,837 overlap with #562+#563; 29,492 additions | **Verified archived trajectories.** Reused screen suite is not a fresh holdout. |
| #512, #545, #547, #551, #552, #555 screen trajectories | Final publication/correction receipts, frozen opening suite and outcomes ledger, every exact batch hash; `seed564-seven-source-manifest.json` | #512: 60,399 / 60,399; #545: 50,076 / 50,076; #547: 53,161 / 53,161; #551: 55,015 / 55,015; #552: 51,158 / 51,158; #555: 49,646 / 49,646 canonical / float32 distinct identities | **Verified archived trajectories only**, not a census of each experiment's full training/development source populations. Per-source baseline overlap/additions and pairwise root/continuation overlaps are in `seven-source-count-report.json`. |
| Seven-screen union (#511, #512, #545, #547, #551, #552, #555) | Deterministic occurrence-level provenance indexes, immutable source manifest, read-only absolute-action replay | 207,422 distinct canonical / 207,422 float32 identities; 52,166 overlap with #562+#563; 155,256 additions | **Verified.** The seven sources reuse the same 1,024 opening roots: root union 1,024, zero root additions beyond baseline; continuation union 206,399, with 155,256 additions beyond baseline. Root overlap is reported separately from continuation overlap. Recovered #545 openings 0/1, #547 openings 0/1/2, and #551 opening 1 are each counted once. Failed/superseded attempts are listed separately and excluded from committed games. The verifier uses no Torch, NumPy, model weights, or materialized tablebase. #545 recorded-runtime reproduction remains unverified; its different-runtime historical check is supporting evidence only. |
| #503/#504 archived corpus and policy-training split | #504 `manifest.json` SHA `449506a30cbe04e2cbe34794c0647fad4aa9f74019ee480edf2c67520a9e353a` pinned; it binds all eight #503 shards; #500 accounting and #504 scored results are pinned; split SHA `080b362d334dcce81bbd04a490900ec78483d6c05ad38c88ebb33fd8b8695d89`; membership independently rebuilt and compared occurrence-by-occurrence | 18,281 loader-accepted rows = 13,724 training + 4,557 raw development; 2,671 retained development rows actually scored; 11,896 distinct source-family identities under both schemes | **Verified** by `expanded-training-dev-exclusion-inventory`; raw development and scored-retained cohort kept separate. |
| #543 amended seed542 corpus | Amended source-manifest SHA `63ffa566b929de4bec6fd445768e5161a53807fc2dcc1b170c8240a28be8f616`; #543 final verifier artifact bindings; all listed corpus shards authenticated; development membership checked against authenticated states | 4,096 usable games, 146,348 loader-accepted rows (109,504 training / 36,844 raw development), 15,359 retained development rows; 70,614 distinct identities under both schemes | **Verified**. The recovered 51 games and 13-game supplement are counted once; incomplete and unstarted slots are excluded. |
| #549 matched control/treatment corpora and shared development | Per-arm source manifests are pinned to merged #549; shared membership SHA `186146add16819afe6b42a9137d06fcb0188d4b2d58f7ac5db5120ee003472de` is pinned and bound by #550 registration; occurrence identities rebuilt from authenticated states | 36,984 training/raw-development occurrences across both arms; 20,470 distinct identities under both schemes; 4,738 retained shared-development rows | **Verified**. #550 registration is publication-pinned; #554 training binding and input-authentication roots are pinned, all 12 derivative shard hashes are checked, and decoded state occurrences match the #549 control training multiset. Changed targets, repeated exposures, and checkpoints add no source states. |
| Combined #503/#504, #543, and #549 source families | `docs/data/expanded-training-dev-exclusion-inventory/source-manifest.json`; deterministic provenance index and read-only verifier | 91,082 identities under each scheme; overlap with #562 is 2,037; 89,045 identities are additions beyond #562. Pairwise overlaps and per-family counts are in `count-report.json`. | **Verified for these three source families only.** Cross-family duplicates are deduplicated in the union. |
| #562–#564 published subtotal | #562/#563 authenticated by `archived_screen_identity_seven.verify_seven`; #564 independently replays its seven archived screen sources and rebuilds the union from canonical and production-float32 indexes | **302,402 distinct identities under each scheme** | **Verified independently** before importing #511's older sources. The #503/#504 corpus remains counted once through #563. |
| #511 older exclusions: #482–#484 snapshots, #488/#499/#505/#508 arena records, #500 self-play | `docs/data/seed561-historical-exclusion-inventory-v1/source-manifest.json` pins #511 registration SHA `89f04c6e1f774d81a7866d3913dd5ffa990b9e04a58f9a7f066b8f19204445c4` and all 2,073 `exclusion_source_sha256` inputs; exact-byte authentication passed | Imported #482: 62,330; #483 additions: 48,080; #484/#467 additions: 39,346. Replayed #488: 30,681; #499: 28,344; #505: 31,130; #508: 48,080; #500: 13,363 distinct identities per scheme. Deduplicated union: 254,802; 23,171 overlap #562–#564 and 231,631 are additions under each scheme. | Snapshot identities are imported with published state/provenance. Arena records and #500 input rows are independently reconstructed. #500's 11,896 identities overlap the #563 #503/#504 family and are not counted again. Per-source overlap/additions are in the report. |
| Consolidated #561 historical inventory | `docs/data/seed561-historical-exclusion-inventory-v1/`; deterministic gzip indexes with occurrence-level provenance; verifier `python -m ml.alphazero_lite.seed561_historical_exclusion_inventory` | 534,033 distinct identities under each scheme after deduplicating #562–#564 and the #511 older exclusions | **Verified inventory for known sources only.** Seed461 continuation overlap remains unresolved; other historical populations remain uncovered with unknown counts. The #483 confirmation gate remains closed. |
| Other historical training/development populations and sources outside the seven archived screens above | Publications remain in their corresponding `docs/data/seed*` directories | Counts unknown unless separately authenticated and reconstructed | **Uncovered.** The seven-screen replay verifies only those recorded arena trajectories; it does not certify every historical training/development population. A publication or experiment number alone is not proof of zero overlap. |
| #556 exploratory arena | `seed556-exploratory-arena/screen-0005/verification-receipt.json` SHA `a318933da72b8ed23cb5487fbbd8e57edd0d1d407ba9bc1f8485b0736fcdfbe7`; receipt-bound openings and all indexed game files replayed | 54,373 unique canonical states and 54,373 `kalah_v3` float32 identities, including opening roots and all pre/post-move and terminal states | **Verified** by `seed556-560-exclusion-inventory`; see the separate #503/#504, #543, and #549 coverage rows above. Other historical populations remain uncovered. |
| #557 retrospective diagnostic | Cohort SHA `f62539cc3e1b1843a2bd9296453d51b895a1c5e693bd9edc3b3291f883398243`; #557 README and freeze bind authenticated #556 games | 256 selected, canonical-deduplicated roots (128 per source side); no new games/positions beyond #556 continuations; production-float32 count not separately published | Reached positions sampled from #556 archived games; derived diagnostic evidence, not a separate novel arena population. |
| #558 value-only swap | Cohort SHA `f62539cc3e1b1843a2bd9296453d51b895a1c5e693bd9edc3b3291f883398243` (byte-identical to #557 cohort); README says all 256 roots/oracles reused | 256 roots; 0 newly selected positions relative to #557; no new game trajectories; production-float32 count not separately published | Exact cohort reuse; #558 adds searches, not states. |
| #559 fixed-position feasibility | Count-audit SHA `6f106dc36f36182ee09090defe646901b1bce81c2418ac73aa8b76f775fe9b1e`; cohort SHA `f2e03fb9491474056e7a7cb3ace225cd2310f9069c9e48600d45039ed63158fb`; archive receipt SHA `a318933da72b8ed23cb5487fbbd8e57edd0d1d407ba9bc1f8485b0736fcdfbe7` | From 2,048 #556 games: 1,903 eligible first roots, 1,663 unique canonical states, 295 matching #557 excluded occurrences / 256 unique roots, 1,407 remaining roots, 512 selected (256 per side); float32 identities not separately tallied | Newly selected diagnostics roots, but all are positions within #556 trajectories; do not double-count as independent arena source. Count audit classifies canonical-state identity only. |
| #560 paired first-action diagnostic roots | Cohort SHA `fe24d68aea9a158cb441aa886a60adc26631805c5b1213f14fceb4aa4ce4a254`; freeze binds the authenticated #556 archive | 256 canonical-deduplicated selected roots; roots are a subset of reached #556 positions | Reused starting positions; not an additional root population. |
| #560 forced continuations | `continuations.jsonl` SHA `5c47e4d161077f3b438d737887da35b73ca3dddab7d60db22fed19bd6dd3d8a3`, included in merged #560 commit `8bf3d551`; cohort/freeze hashes authenticated | 512 arms, 313 physical continuations; 16,107 arm-record actions / 9,878 actions replayed once per physical continuation; 9,849 distinct identities under both schemes | **Separate exclusion source.** Its replayed states overlap #556 in 6,121 identities under each scheme; see machine-readable report. |
| #461 continuation population | #483 receipt-matching archive and source ledger; 58 game streams, 1,129,643 trajectory actions | Continuation-state identity counts unavailable/cannot be certified | **Unresolved overlap.** Published roots are included; recorded continuation actions fail canonical replay (first example illegal). No recovery attempt is reopened. |

The #560 continuation census validates each arm from its frozen cohort root,
applies the forced first action, then records every decision state and each
post-move state through terminal scoring; identities are deduplicated using
the production `canonical_identity` and `float32_identity` functions. The
continuation file and cohort hashes above bind the census inputs. The #556–#560
evidence is now separated source-by-source. #557–#560 roots are
diagnostic subsets of #556; #560's forced continuations are separate positions
and are counted independently above. Do not sum diagnostic cohorts as
independent arena populations. The #556/#560 union is 58,101 identities under
each scheme: 6,121 intersect, 48,252 are unique to #556, and 3,728 unique to
#560. Counts above are replay-derived
from the cited byte-bound records; the remaining float32 totals are not inferred
from canonical counts. For #556, the available development overlap is 1/1,024
in both identities, while the recorded T512 and P training overlaps are zero at
the opening roots only.

The #545 standalone index diagnostic was repeated. The earlier failed
invocation selected `.tmp/historical-exclusion-audit` as `--out`, a scratch path
without the committed artifacts; this was not a mismatch in the published
index. Replaying with `--out docs/data/seed564-archived-screen-identity-audit`
passes. Fresh canonical and float32 indexes contain 68,166 rows each and match
the checked-in indexes after decompression with no differing row. No #564 bytes
were changed. #545's recorded-runtime reproduction remains unverified as stated
in its separate report.

This report distinguishes verified counts from unavailable counts; it does not
claim complete historical disjointness. The #483 confirmation gate remains
closed. Missing historical evidence is an explicit block on a fresh-confirmation
claim, regardless of whether a suite screens cleanly against all known sources.
The expanded inventory covers exactly the #503/#504 corpus/split, the amended
#543 seed542 corpus/development membership, and the #549 arm corpora/shared
development membership (with #550/#554 state-reuse checks), in addition to the
previously verified #556–#560 populations. It does not close the remaining
historical evaluation gaps or the unresolved seed461 continuation population.

## Freeze, commitment, and reveal

### Operator rehearsal and private seal commands

The custodian CLI is `python -m ml.alphazero_lite.holdback_operator`. Run
the no-sampling inventory/source/runtime check first:

```bash
python -m ml.alphazero_lite.holdback_operator preflight --root . \
  --output "$HOME/holdback-preflight-seal-01.json"
```

It authenticates the exact #565 source inventory and both identity indexes and
reports the #566 source/runtime bindings, without drawing a seed or calling the
sampler. A passing preflight is not a real seal and does not change the
unresolved historical coverage status.

Run
`initialize` once with a custody directory on a restricted filesystem outside
the repository. It creates the directory as `0700` and writes the immutable
`private-inputs.json` (`0600`) before any sampling. Production initialization
omits `--seed`, draws 256 bits with `secrets.randbits` (OS cryptographic RNG),
and stores the seed plus generator/runtime and inventory pins. The optional
`--seed` argument exists for synthetic fixtures only. Never use a repository
directory for custody.

```bash
python -m ml.alphazero_lite.holdback_operator initialize \
  --root . \
  --custody "$HOME/.local/share/mancala-holdback/seal-01" \
  --size 1024 \
  --preflight-json "$HOME/holdback-preflight-seal-01.json" \
  --manifest docs/data/seed561-historical-exclusion-inventory-v1/source-manifest.json \
  --manifest-sha256 946fc52c75fa2e6c0499811fcb97ff773ea00ec6ec9b6d1bfb3ec1c2466a95ad

python -m ml.alphazero_lite.holdback_operator resume \
  --root . \
  --custody "$HOME/.local/share/mancala-holdback/seal-01" \
  --public-output ./holdback-commitment.json

# After the candidate and serving-P artifacts and all comparison settings are
# frozen, create freeze.json containing their SHA-256 values/settings/rule.
python -m ml.alphazero_lite.holdback_operator freeze \
  --custody "$HOME/.local/share/mancala-holdback/seal-01" \
  --root . \
  --freeze-json ./freeze.json

python -m ml.alphazero_lite.holdback_operator reveal \
  --custody "$HOME/.local/share/mancala-holdback/seal-01" \
  --root . \
  --bundle ./revealed-holdback-bundle

# After the suite has been used, permanently mark it consumed.
python -m ml.alphazero_lite.holdback_operator consume \
  --bundle ./revealed-holdback-bundle
```

On restart, `resume` reads the exact saved seed and inputs, authenticates the
inventory again, and either verifies an existing ordered suite against those
inputs or deterministically creates it. Missing inputs, contradictory suite or
receipt bytes, and an existing public commitment fail closed. It verifies the
private suite and exclusion identities before creating the seed-free public
commitment with exclusive creation. A crash after saving inputs or after suite
write is resumable; an interrupted/partial file is not silently replaced.
Retain the private directory and public commitment separately. Keep the
independent registry digest and lifecycle anchor digest in separate custodian
records.

The exact production generator settings are 1,024 openings, 20,000 total
attempts, one Python `random.Random` MT19937 stream, legal prefix lengths 4–12,
and compact UTF-8 sorted-key JSONL with one LF per opening. The public record
binds suite SHA-256, generator/source/runtime bindings, and the hash-pinned
inventory; it excludes seed and opening bytes. Candidate and serving-P artifact
hashes, arena settings, and analysis rule must be frozen before the private
suite is revealed. Use 18-way opening-pair concurrency, one search worker per
game, and bind registered CPU, memory, and runtime limits in the freeze. Reveal
and consumption are controlled lifecycle events; a consumed suite cannot be
reused. The rehearsal is synthetic only. The seed461 gap remains unresolved,
#483 remains closed, and any later comparison is screened against known
exclusions only.

Production initialization requires a retained preflight JSON record and
recomputes it before invoking the OS RNG. A stale or edited record, changed
runtime/source binding, or inventory mismatch blocks initialization. Explicit
seed injection is available only through the `--synthetic-fixture` test path.
The commitment and candidate-freeze events are independently appended and
anchored at their actual operation times; reveal checks their order, saved
anchor, and the exact already-published public commitment before staging a
portable bundle. Consumption uses a durable pending transaction so restart
replays the exact event, anchor, and registry bytes rather than appending a
second consumption event.

Before any real generation, publish the exact generator version, legal move and
state-identity implementations, 1,024-root sample law, exclusion-inventory
schema and source list, pair execution semantics, and analysis rule. Use a
private cryptographically random seed. The seed and ordered opening list remain
private while the candidate is developed. Generate rules-only openings; reject
terminal prefixes and reject a candidate if either its canonical-state identity
or production-float32 identity appears in the exclusion union or an earlier
selected opening. Do not replace openings based on outcomes.

Publish a commitment over canonical JSON containing the SHA-256 of exact suite
bytes (therefore also binds order and encoding), the complete generator
specification, and the hash-bound exclusion inventory. Public commitment
records are distinct from private custody files: the seed, ordered suite JSONL,
generation receipt, and custody access log remain private until release.
Freeze hashes for both candidate and serving
P artifacts, arena settings (including 1,024 paired openings / 2,048 games,
seat swap, worker count and runtime limits), and the complete analysis rule
before revealing the suite. A verifier must check the commitment and both
identity schemes from a portable bundle. Once revealed and used, append a
consumption receipt bound to the commitment. A consumed suite is never described
as held back in a later comparison.

`ml.alphazero_lite.holdback_arena` implements a read-only bundle verifier and a
deterministic rules-only sampler; it has no arena execution path. It uses one
`random.Random(seed)` MT19937 stream, draws each prefix length with
`randint(4, 12)`, and selects uniformly with `choice` from ascending legal
relative pit indices. The attempt limit is 20,000 total; it never reseeds and
fails on shortfall without publishing a partial result. Serialization is
compact UTF-8 sorted-key JSON, separators `,` and `:`, one LF per row, with
`opening_index`, `opening_contract`, `prefix_moves`,
`source_prefix_moves_absolute`, `state`, and `state_hash` fields under
`arena_player_relative_v2`. Verification rebuilds the exact ordered suite from
the seed, frozen generator, and pinned inventory, without a caller-supplied
pool, then derives canonical and production-float32 identities from
the rule state using `kalah_v3`. Its lifecycle log records commitment,
candidate freeze, reveal, and optional consumption in order. Verification
requires the SHA-256 of an independently retained lifecycle anchor and the
SHA-256 of the custodian's current registry state. Historical anchors remain
valid for audit, but eligibility is granted only when the supplied anchor is
the commitment's current registry anchor and the registry marks it unconsumed.
Thus an older reveal anchor cannot make a suite eligible after a later
consumption record. The trust roots are the separately retained anchor and
current-registry digests, not bundle self-assertions. A consumed suite verifies
as consumed but returns `held_back_eligible: false`.
The future production generator must be separately frozen against the audited
source inventory. A clean known-exclusion screen is readiness for an
**exclusion-screened comparison only**. Before a later 1,024-opening
comparison freeze the generator source/version, RNG and attempt limit,
generator/rules/canonical-identity/float32-identity source hashes, the complete
`self_play.py` and `input_encodings.py` encoder implementation files, and
complete `holdback_arena.py` and `seed482_fresh_confirmation.py` files, plus
Python/NumPy runtime versions, inventory pin, seed custody
procedure, suite digest/order, candidate and
serving-P hashes, pair/seat semantics, pair concurrency, per-game worker count,
CPU/memory/runtime limits, and analysis rule. No real suite is generated or
revealed by this preparation. **Fresh confirmation readiness: blocked**
until historical evidence gaps, especially #461 continuations, are resolved and
the existing #483 gate is explicitly reopened by its authorized process.
