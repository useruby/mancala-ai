# Lagged-Parent Shadow Root-Q

**Classification:** `lagged_parent_shadow_rescues_game_strength`

**Recommended follow-up:** Run one fresh AlphaZero generation where candidate evaluation uses its parent as a lagged shadow-Q search, while training remains unchanged.

## Contracts

- Immutable P0, P1, and A16 artifact hashes matched the established lineage.
- A16 retained the established parent-additive adapter invariant.
- P1 self-shadow reproduced ordinary P1 on all frozen roots.
- Main simulation `t` used only the P1 root-child snapshot captured before shadow simulation `t`.
- Frozen amplified rescue was 39/40 (0.975); new washed-control divergences were 0/40.

## Canonical Arena

| Context | Treatment raw | Matched control raw | Effect | 95% CI | Seat 0 | Seat 1 | W/D/L |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 384:256 | 0.8184 | 0.8184 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 | 400/38/74 |
| 1200:1200 | 0.5000 | 0.5000 | +0.0000 | [+0.0000, +0.0000] | +0.0000 | +0.0000 | 190/132/190 |

Compared with the ordinary PR #214 A16 effects of -0.0098 and -0.0195, exact lagged-parent root-Q shadow search eliminated the observed candidate deficit in both canonical contexts.

## Compute

| Quantity | Value |
| --- | ---: |
| Main simulations per challenger move | Context challenger budget |
| Shadow simulations per challenger move | Same as main |
| Total cost versus ordinary PUCT | Approximately 2x |
