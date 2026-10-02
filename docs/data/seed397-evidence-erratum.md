# Erratum: seed397 O0 E1/E4 diagnostic

**Status: protocol-invalid; not confirmatory fresh evidence.** The original
registration, binding, opening suite, reports, game logs, outcome accounting,
and reported arithmetic remain preserved as published. This erratum does not
remove openings or recompute a replacement primary result after observing
outcomes.

The registered complete-exclusion requirement was not met. The suite omits
2,534 identities from the complete exclusion union and collides with 13
historical declared states at zero-based opening indices
18, 107, 136, 137, 170, 262, 297, 336, 350, 385, 404, 503, and 507. The
reproducible identity-level audit, including source-suite attribution and
hashes of each input, is `seed397-evidence-erratum-audit.json`. It also records
the seven state identities shared by the valid #396 suite and invalid #397
suite. Regenerate it with:

```bash
python -m ml.alphazero_lite.audit_seed397_evidence_erratum
```

The former `register`, `run`, and `analyze` commands have been retired. In
particular, their analysis path substituted suite-derived state hashes for
raw consumed-state identities, and the execution path did not validate arena
reports and game evidence before accepting it. The valid original-O0 E1/E4
diagnostic remains #396. Early stopping remains unsupported and seed455
remains incumbent.
