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
| #511 screen | #511 v4 registration, suite and outcome publication; exact source hashes in registration | 1,024 roots; historical union counts and replay deltas recorded in registration | Reused screen suite is not a fresh holdout. |
| #512–#555 training/development and arena populations | Published experiment receipts in the corresponding `docs/data/seed*` publications | No consolidated source-bound identity inventory/counts available in the #560 base | **Uncovered pending authenticated replay or evidenced empty contribution.** A publication or experiment number alone is not proof of zero overlap. |
| #556 exploratory arena, #557 diagnostic/physical-copy evidence, #558 value-only swap, #559 fixed-position feasibility, #560 paired first-action diagnostic | Publications/receipts through merged PR #560; inspect each registration, input binding, and game/trajectory artifact before inclusion | Not certified as a single union in the #560 base; no aggregate canonical or float32 identity counts claimed | Must be replayed from receipt-matching source bytes into a source-by-source union before an exclusion-screened suite is authorized. #560 is diagnostic evidence, not confirmation. |
| #461 continuation population | #483 receipt-matching archive and source ledger; 58 game streams, 1,129,643 trajectory actions | Continuation-state identity counts unavailable/cannot be certified | **Unresolved overlap.** Published roots are included; recorded continuation actions fail canonical replay (first example illegal). No recovery attempt is reopened. |

This report distinguishes verified counts from unavailable counts; it does not
claim complete historical disjointness. The #483 confirmation gate remains
closed. Missing historical evidence is an explicit block on a fresh-confirmation
claim, regardless of whether a suite screens cleanly against all known sources.

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
The future production generator must be separately frozen against the audited
source inventory. A clean known-exclusion screen is readiness for an
**exclusion-screened comparison only**. **Fresh confirmation readiness: blocked**
until historical evidence gaps, especially #461 continuations, are resolved and
the existing #483 gate is explicitly reopened by its authorized process.
