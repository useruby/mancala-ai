# Seed464 dependency-audit verification supplement

The original report, frozen registration, publication receipt, outcomes, and
classification are unchanged. The attempt remains
`invalid_attempt_dependency_closure_incomplete`; there is no registered
training decision.

## What the packet establishes

The packet manifest records logical/original paths, packet-relative locations,
roles, raw SHA256 identity, archive SHA256, provenance, identity binding source,
and whether identity was prospectively committed or only retrospectively
observed. All archived members are hash-checked before the verifier parses
evidence. Five derivatives are stored gzip-compressed; the verifier hashes both
the archive bytes and the decompressed original JSONL bytes.

The exact seed464 registration and receipt hashes match the requested values.
The receipt binds the original publication members; the registration binds its
12 direct inputs and six source snapshots. The seed435 snapshot commits to the
seed416 registration and two seed427 receipts, and the loader source checks
those identities. The nested seed462 and seed416 source inventories provide
the stated prospective source commitments where present.

The five lane-A derivatives currently match their seed416 registration-v3
`derivative_sha256` commitments. The seed435 loader passes derivative paths to
`load_jsonl_replay` without comparing those digests. Thus the derivative bytes
are prospectively committed and currently agree, but the loader did not enforce
their identities. Recovering their bytes does not establish original runtime
enforcement.

`seed432_policy_target_census.py`, `seed433_census_correction.py`, and
`exact_root_policy_targets.py` remain transitively unbound: no prospective
source digest appears in the reviewed seed464, seed462, or seed416 source
inventories. Their packet hashes are retrospective observations, not newly
created historical commitments. Other helper sources with nested hashes are
reported with their original chain and are not represented as loader-time
checks of the imported current module bytes.

## Limits

The packet establishes the identity/content relationships in the committed
evidence and confirms the recorded descriptive outcome. It does not prove that
the unbound helper source bytes used during the historical execution were the
same as the retrospectively archived files. The missing prospective helper
commitments and absent derivative runtime checks therefore remain blockers.
No result is upgraded to a registered training decision, and no training or
rerun is authorized or performed by this audit verifier.
