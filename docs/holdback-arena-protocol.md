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

Before any real generation, publish the exact generator version, legal move and
state-identity implementations, 1,024-root sample law, exclusion-inventory
schema and source list, pair execution semantics, and analysis rule. Use a
private cryptographically random seed. The seed and ordered opening list remain
private while the candidate is developed. Generate rules-only openings; reject
terminal prefixes and reject a candidate if either its canonical-state identity
or production-float32 identity appears in the exclusion union or an earlier
selected opening. Do not replace openings based on outcomes.

Publish a commitment over canonical JSON containing the SHA-256 of exact suite
bytes (therefore also binds order and encoding), generator description/version,
and full hash-bound exclusion inventory. Keep the suite bytes and seed in
restricted custody until release. Freeze hashes for both candidate and serving
P artifacts, arena settings (including 1,024 paired openings / 2,048 games,
seat swap, worker count and runtime limits), and the complete analysis rule
before revealing the suite. A verifier must check the commitment and both
identity schemes from a portable bundle. Once revealed and used, append a
consumption receipt bound to the commitment. A consumed suite is never described
as held back in a later comparison.

`ml.alphazero_lite.holdback_arena` implements a read-only bundle verifier and a
deterministic sampler used by synthetic tests; it has no arena execution path.
Verification rebuilds the ordered suite from the bound pool, seed, generator,
and inventory, then derives canonical and production-float32 identities from
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
**exclusion-screened comparison only**. **Fresh confirmation readiness: blocked**
until historical evidence gaps, especially #461 continuations, are resolved and
the existing #483 gate is explicitly reopened by its authorized process.
