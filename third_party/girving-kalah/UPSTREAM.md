# Geoffrey Irving Kalah Source Audit

This experiment fetches, rather than vendors, Geoffrey Irving's Kalah solver:

- Repository: https://github.com/girving/kalah
- Pinned commit: `0a0c00702908f4ba865f2a1baa548ac0e3724950`
- Upstream commit subject: `Add a BSD license`
- Licence: BSD 3-Clause (the upstream file is non-SPDX-labelled; its terms are
  the standard three-clause BSD terms)

The upstream `LICENSE` permits redistribution and use in source and binary
forms, with or without modification. Redistributions must retain the copyright
notice, conditions, and disclaimer; binary distributions must reproduce those
notices in accompanying materials; Geoffrey Irving's name may not be used for
endorsement without permission. The fetch procedure copies the upstream
`LICENSE` beside the generated source and this document preserves attribution
and the required notices for this repository's experiment.

No upstream source is committed here. The reproduction commands in
`docs/native-mtdf-exact-solver-feasibility-results.md` check out exactly the
commit above and apply `compatibility.patch` outside the repository.

## Compatibility patch

The patch is deliberately limited to:

1. Remove the obsolete GCC-only `-fnested-functions` option and compile with
   GNU C11 plus GNU89 inline semantics, which the 2009 headers require on
   current GCC.
2. Replace an unsequenced pointer-increment macro in the hash packing check
   with explicit indexed reads. This preserves the intended six-pit range
   check without undefined behaviour under modern optimization.
3. Require a non-empty opposite pit before capture. Upstream captures whenever
   a final seed lands in an empty own pit; canonical `kalah_v1` captures only
   when the opposite pit is non-empty.

The generator and search both compile against the same patched `rules.c`, so
the endgame database and solver use identical canonical transitions. No search
algorithm, pruning rule, move ordering, or tablebase representation is
changed.
