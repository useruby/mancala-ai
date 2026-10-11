# Seed457 occurrence-preserving verification correction

This is a post-execution verification correction; it does not alter the frozen execution, original evidence, or #460 status.

At C step 6, scale 1.0, archived runner KL is `0.032772165412068455`; the original verifier reconstructed `0.03277216548941382`. Compact row `72594` appears at occurrences 343 and 360 with identical inputs but different trial logits (maximum absolute difference `4.76837158203125e-07`). The compact-row scatter/gather path reused occurrence 360's output at occurrence 343, changing one KL term by `3.96008305453277e-08`. The aggregate discrepancy is thereby explained to within `5.94e-18`; reduction-only difference with identical terms is zero.

The correction preserves compact-row pre-policy reference lookup while keeping trial policy and value predictions, masks, terms, and weights occurrence ordered as in the frozen runner. See `reconciliation.json` for the complete reconstruction, fixed gates, classification, and source bindings.
