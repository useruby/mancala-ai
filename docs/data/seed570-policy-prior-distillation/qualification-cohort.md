# Qualification cohort construction

Status: **ready for a separate pre-training freeze**. This artifact is cohort
readiness only; it contains no teacher targets and authorizes no training,
searches, games, or sealed-suite access. #483 remains closed.

`qualification-membership.json` contains the exact ordered 423-root membership.
Its root-list digest is
`94e126af6ea88fb99a4230f84e5cf2a20248daa39ea66d25b68730ea0426d668` (SHA-256
of compact JSON, sorted object keys, over the `roots` array). The 512 #559
roots were reconciled: 89 shared-opening exclusions; zero canonical collisions,
zero `kalah_v3` float32 collisions, zero duplicate identities, and zero
duplicate openings. All 423 surviving roots have distinct openings.

Membership ordering is ascending
`SHA256("seed559:" + canonical_identity)`, with canonical identity as the
secondary key. This is the registered #559 source order, applied after the
frozen exclusion rules. Oracle records are consulted only to verify state
identity and exact legal-action coverage; oracle values do not affect selection
or ranking.

The JSON package pins the #557 cohort, freeze, and correction record; the #559
run-0003 manifest, freeze, oracle records, README and count audits; and the
source verifier/runner bytes. Revision labels for the published corrections
are included in the same package. The verifier recomputes both state identity
schemes from source states, checks source byte bindings and oracle coverage,
reapplies the exclusions and ordering, and rejects any membership mismatch.

Read-only verification (does not rewrite any artifact):

```bash
/home/alex/azlite-venv/bin/python -m ml.alphazero_lite.seed570_qualification_cohort
```

The explicitly separate publisher option is only for reconstructing the
membership package from authenticated source evidence:

```bash
/home/alex/azlite-venv/bin/python -m ml.alphazero_lite.seed570_qualification_cohort --publish
```

Regression tests:

```bash
/home/alex/azlite-venv/bin/python -m pytest ml/alphazero_lite/test_seed570_qualification_cohort.py -q
```
