# Seed 599 evaluation readiness

This preparation is bound to the unchanged `evaluation-contract.json` and
`evaluation-openings.json`. It authenticates both published update-512 training
checkpoints and the registered Seed 552 serving P-E4 runtime bundle. Arm A and
Arm B candidate bundles are policy-only at evaluation time; both use the
unchanged serving P-E4 value evaluator and search-policy sidecar.

The independent checkpoint verification was rerun for both arms. It loaded the
authenticated parent and trained checkpoints and compared each registered
frozen parameter with `torch.equal`. The executor's in-run frozen-parameter
assertion is separately recorded. Numerical training replay was not performed.
The original published verification reports were not modified.

The runner binds 4,096 total games: two seat-swapped games for each of 1,024
openings and each candidate. Each game uses 128 simulations, exact-root
threshold 16, one search worker, deterministic roots, and the remaining
registered settings. It runs at 18-way pair-level parallelism. Completed pairs
are only skipped after independent verification; partial or failed pair
artifacts are preserved and block automatic rerun.

The read-only verifier validates frozen source/runtime/checkpoint identities,
the exact assignments and seat swaps, each legal trajectory and resulting
outcome, search settings and evaluator identities, the full assignment count,
execution accounting, and the registered 20,000-resample paired-opening
bootstrap. It reports A and B seat-averaged scores against P-E4 and, after game
replay, canonical and production-float32 overlap counts versus both frozen
training memberships. Each overlap count includes observed-position and
training-membership denominators; overlap is descriptive only and does not
filter games or affect advancement. The verifier reports all four registered
outcome categories. The primary gate
is B's one-sided 95% lower score bound strictly above 0.50; the paired B-minus-A
95% interval must have a lower endpoint strictly above zero. Held-out policy
cross-entropy remains a secondary diagnostic only.
The 1,353-state held-out cross-entropy is reported for A and B, including
opening-group summaries, in `evaluation-readiness/heldout-policy-cross-entropy.json`.
Observed diagnostics are A `0.697792`, B `0.662917`, and B-minus-A `-0.034876`.
These values do not affect either advancement gate.

The execution freeze is marked `frozen-awaiting-separate-project-owner-authorization`.
The runner requires a separate `evaluation-authorization.json` bound to that
freeze's exact file-byte SHA-256 before it can invoke arena games. No evaluation
games were run in this preparation.

Commands:

```bash
python -m ml.alphazero_lite.seed599_evaluation_readiness --preflight
python -m ml.alphazero_lite.verify_seed599_evaluation --allow-incomplete
```

The frozen execution identity is recorded in
`evaluation-readiness/execution-freeze.json`. Running
`python -m ml.alphazero_lite.seed599_evaluation_readiness --run` currently fails
closed because no separate owner authorization exists.
Execution-freeze SHA-256: `467cb1ecf4acb47013fce9cd9418b3ac6b1d3929752713e9bd029fceabbcea11`.
Superseded pre-execution freeze identities are preserved as
`evaluation-readiness/execution-freeze-superseded-*.json`; each was superseded
before authorization or assignment execution.
