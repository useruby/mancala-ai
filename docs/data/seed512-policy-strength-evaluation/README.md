# Update-512 paired root evaluation freeze

This is a preparation freeze for the registered #570 local diagnostic gate.
It authenticates the #571 update-512 checkpoint/export, unchanged P-E4 control,
shared #573 initializer value model, ordered 423-root qualification membership,
#557-derived exclusion identities, #559 oracle records, and #568 runtime search
policy as separate inputs. The shared runtime search-policy file is pinned
independently of the archived policy sidecars copied into the component bundles.
The freeze also hashes the arena search implementation, PUCT implementation,
seed derivation, qualification verifier, rules, encoding, and the evaluator's
search-policy dependencies.

Control and treatment retain their own policy/trunk artifacts. Each evaluator
uses the same separate initializer artifact for both value and value trunk.
Only the policy and shared trunk differ between the arms. Export parity is
tested on two synthetic positions; no qualification root is used for parity
testing.

The prospective evaluation is fixed at 128 simulations, `c_puct=1.25`, one
worker, fresh trees, matched per-root seeds from
`seed557_retrospective.search_seed(canonical_hash(state))`, deterministic root
policy, no subtree reuse, and the exact #568 shared policy. Every result stores
the chosen action, exact final-margin regret, WDL, and matched seed for both
arms. Output is created exclusively and incomplete pairs fail closed.
Evidence carries the selected action alongside its PUCT visit/Q/prior witness.
The read-only verifier replays each frozen arm/root search, checks the recorded
choice and witness against the deterministic PUCT result, validates exact
integer margins/regrets from the authenticated oracle, and then rebuilds the
gate without writing evidence.

The unchanged #570 gate uses treatment-minus-control exact-margin regret paired
by root and clustered by opening, 20,000 resamples from
`numpy.random.default_rng(570)`, and linear 0.025/0.975 quantiles. Advancement
requires mean regret change at most `-0.10`, an interval upper endpoint below
zero, and no increase in WDL downgrade rate. Any other outcome is
`do_not_advance`.

All 423 qualification roots have 17–21 active pit stones. The #568 exact-root
handoff threshold is 16, so no qualification root is routed to exact-root
solving; exact-leaf solving remains disabled. The update-512 export was built
under CPython 3.12.8 / NumPy 2.2.6, as recorded in `freeze.json`.

`freeze.json` is `frozen_pending_review` and explicitly sets
`qualification_authorized=false`. Run the read-only freeze check with:

```bash
python -m ml.alphazero_lite.seed512_policy_strength
```

The `--execute` path requires a separate authorization record bound to the
exact freeze SHA-256 and an independently supplied trusted authorization
SHA-256 from its review record. The evidence records that authorization digest,
and the read-only verifier requires and rechecks the same authorization before
constructing evaluators or replaying searches. Both execution paths apply the
same single-thread CPU setup. An existing output path is rejected before
evaluator construction. Neither path is invoked in this preparation. No arena
games, qualification searches, or sealed-suite access are part of this PR.
Serving remains unchanged; #483 remains closed.
