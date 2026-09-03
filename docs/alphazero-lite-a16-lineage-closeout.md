# A16 Lineage Closeout

**Final classification:** `a16_lineage_closed`

The machine-checkable evidence ledger is
[`docs/data/alphazero-lite-a16-lineage-closeout.json`](data/alphazero-lite-a16-lineage-closeout.json).
It freezes PR #266 at head `91823644a205af601234287e66773d6fae0bb25e`, merge
`d09d6b03101e07a3723835ea3ce26bdf89a70d26`, A16 artifact
`8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34`, and A16
weights `74e160734554dba9ecf4ecf3ac13b3e44526c4f6e7e5e52d9fe8bde0054eb789`.

## Decision

1. PR #264 showed a small one-epoch pure-AZ shallow-budget signal: `+0.038411`
   at 384:384, CI95 `[+0.017969, +0.057812]`.
2. Its primary 1200:1200 effect was not a replacement signal: `-0.007422`,
   CI95 `[-0.024219, +0.013281]`.
3. Frozen PR #266 demonstrated that additional exposure strongly degrades
   1200-budget strength: matched repetition was `-0.0984375`, CI95
   `[-0.126953125, -0.0703125]`, and unique data was `-0.1079427083`, CI95
   `[-0.13828125, -0.076953125]`, versus A16.
4. Approximately five times more distinct replay data did not outperform
   repeated exposure: unique-minus-repeat was `-0.0095052083`, CI95
   `[-0.040625, +0.021875]`.
5. Therefore repetition and insufficient replay diversity are not supported as
   the limiting causes. Incremental A16 fitting is closed.

## Exhausted Continuations

The ledger marks the following branches ineligible. They must not be reopened
under renamed variants:

- More epochs or optimizer steps.
- Larger fixed-parent replay aggregation.
- Target mixing or alternate target-search budgets.
- Isolated head/trunk scopes.
- Anchor-beta variations.
- Another A16 PUCT self-play iteration.
- Residual-v4/move-factorized continuation.
- Tablebase overlays and direct-patch variants already rejected by committed evidence.

The control-EP2, promoted-current PUCT iteration, residual-v4, direct exact
tablebase patch, and tablebase value-overlay records are retained in the ledger
to preserve their negative evidence. None is an A16 continuation candidate.

## Generation-3 Entry Criteria

No new training experiment may consume suites or compute until its written
proposal passes all of these criteria:

1. It introduces an independently justified capability: either supervision
   already demonstrated stronger than A16 at the primary 1200-simulation gate,
   or a materially different model/search interface with a stated
   inference-budget constraint and falsifiable mechanism.
2. It identifies information unavailable to the A16 objective and explains why
   the recorded negative results do not already test the hypothesis.
3. It defines a cheap preflight falsification gate before suite or training
   compute is requested.
4. It freezes parent and artifact identities.
5. It specifies primary 1200-budget evaluation, multi-seed uncertainty, and
   local promotion criteria.

The least-tested surviving idea is not sufficient justification. A Generation-3
proposal requires an independent source of new information.

## Verification

Run `python ml/alphazero_lite/validate_a16_lineage_closeout.py`. The validator
uses only committed files and the standard library; it performs no training,
self-play, arena evaluation, suite consumption, or artifact mutation.
