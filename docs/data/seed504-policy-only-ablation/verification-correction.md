# Seed504 policy-only verification correction

This append-only correction documents verification code added after PR #539.
It does not amend the original training registration, screen registration,
execution freeze, snapshots, or arena outcomes. No training or arena games were
run for this correction.

The execution preflight now requires the live checkout bytes for every frozen
executable source. Archived source snapshots remain available only for
historical read-only verification; they cannot authorize a launch. The screen
verifier independently checks the freeze, suite, component and runtime
identities, commands, swapped-seat pairs, raw trajectories, per-pair reports,
ledger receipts, and the published analysis. Its reconstruction does not call
the runner's pair validation or score/bootstrap/decision functions.

The corrected verifier accepts **1,024 pairs / 2,048 games** and reproduces
931 wins, 182 draws, 935 losses; score **0.4990234375**; one-sided 95% lower
bound **0.490478515625**; descriptive two-sided interval
**[0.48876953125, 0.509033203125]**. The registered strict lower-bound rule
therefore yields **no advance**. This is an exploratory reused-suite result,
not fresh confirmation, promotion, or deployment evidence.

The development verifier now independently recomputes and validates every
published T−C policy cross-entropy effect for all four cohorts and both
weighting definitions, rejecting missing, extra, nonfinite, and altered
effects.

See `screen/verification-correction-receipt.json` for source and evidence
hashes, test results, copied-checkout verification, and publication
preservation measurements. The separate seed488 regression suite could not be
validated in this environment because its frozen NumPy version differs from
the available `.tmp/venv`; see the receipt for the observed failure detail.
