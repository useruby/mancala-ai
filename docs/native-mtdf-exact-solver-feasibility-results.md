# Native MTD(f) Exact Solver Feasibility Results

**Classification:** `native_reference_incompatible`

This was an isolated solver-feasibility audit. It did not train a model,
generate labels, alter production inference, or reopen the previous ML
lineage.

## Source And Licence

- Source: https://github.com/girving/kalah
- Immutable commit: `0a0c00702908f4ba865f2a1baa548ac0e3724950`
- Copyright: Geoffrey Irving, 2009
- Licence: the included three-clause BSD-style licence (upstream does not
  provide an SPDX identifier)

Redistribution and modification are permitted when the copyright notice,
conditions, and disclaimer are retained, and the author's name is not used for
endorsement. The full audit, required notices, source pin, and proposed narrow
patch are retained in `third_party/girving-kalah/UPSTREAM.md` and
`third_party/girving-kalah/compatibility.patch`. No upstream source or build
product is committed.

## Compatibility Audit

The reference has six pits and its serial search uses iterative-deepening
MTD(f), a bounded in-memory transposition table, deterministic ordering, and
an in-memory endgame database. Its board orientation can be adapted.

Its rules implementation captures from an empty opposite pit. Canonical
`kalah_v1` does not. A localized patch added the non-empty-opposite condition
to both player branches, and a separate build-only patch made the 2009 source
compile on GCC 14 (`-std=gnu11 -fgnu89-inline` and an unsequenced hash macro
fix). The patched serial executable and an 8-stone 4-bit endgame database
built successfully. Database SHA-256 was
`43affb05cbd3f3069807b22c40325292b2e1ca3c9225d80f4ecd70d872196cd7`.

The patched native result nevertheless disagreed with the independent PR #271
brute-force oracle on its first tiny probe, with the tablebase disabled:

```text
state: pits=(0,1,1,0,0,0,0,1,1,0,0,0), stores=(20,20), player=0
oracle action values: {1: -4, 2: 0}; root value: 0
native action values: {1: 0, 2: 2}; root value: 2
```

Changing the adapter orientation corrected one action but not the other; the
remaining mismatch persists with endgame lookup disabled. Resolving it would
require altering or replacing native transition/search semantics beyond the
allowed localized rule patch. Therefore the mandatory 100% rule-parity and
12/12 tiny-exactness gates cannot be attempted successfully.

## Decision

No fixed-corpus benchmark, throughput calculation, or per-state labels were
run. Reporting feasibility numbers after a failed correctness gate would be
invalid. Stop exact-teacher work in this repository; no training or production
behavior changed.

## Reproduction

The audit was performed with the following commands; they fetch outside the
repository and do not vendor source:

```bash
git clone --no-checkout https://github.com/girving/kalah.git /tmp/girving-kalah
git -C /tmp/girving-kalah checkout --detach 0a0c00702908f4ba865f2a1baa548ac0e3724950
git -C /tmp/girving-kalah apply --ignore-space-change /path/to/ai/third_party/girving-kalah/compatibility.patch
make -C /tmp/girving-kalah clean serial
printf '4\nc 8\nq\n' | /tmp/girving-kalah/generator /tmp/endgame-8.dat
```
