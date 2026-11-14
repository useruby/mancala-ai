# #588 retrospective T512 vs P raw-prior diagnostic

**Status: preparation-only; inference is not authorized until this protocol is reviewed.**
This is an exploratory retrospective comparison on the complete, already frozen
#559 cohort—not a fresh qualification or an independent generalization test.
It does not reopen #483.

## Frozen cohort and provenance

All 512 #559 run-0003 roots are retained, one per unique opening. The #559
cohort source, execution freeze, oracle records, and archived search records are
SHA-256 bound in the verifier. Its independent active-pit count is recomputed
from each state and must equal the recorded value in `[17, 21]`. The cohort is
balanced by the original source-side provenance (256 P / 256 T512); this label
is descriptive and does not affect selection. Every legal action must have an
integer exact final-score margin in the archived oracle record.

Historical overlap status is deliberately limited: #559's own construction
excluded #557 and #418 canonical-state matches. #565 verifies a large known
historical inventory under canonical and production float32 identities, but
explicitly leaves Seed 461 continuation overlap unresolved and other historical
populations uncovered. This diagnostic therefore makes **no** claim of
complete historical non-overlap. The roots also derive from previously used
#556 arena continuations and are not independent public positions. Results
cannot establish generalization or improved full-game strength.

## Frozen artifacts and evaluation

The T512 policy/trunk is the exact artifact set pinned by #559's execution
freeze in `seed556-exploratory-arena/screen-0005/runtime-artifacts/T512`:

| File | SHA-256 |
|---|---|
| metadata.json | `b37f3aa75d828b9b9a0743930c73c8e168b159f6b37434eba1d9871e769af1b8` |
| model.npz | `8940cbff36976801e74542f5ee2d6c5caffe0cafd57c7e787cdb375f3bbe485e` |
| search_policy.json | `280c33d88266969da72a7984dabdcfea180bacb5cb690cb759af1ebdbaac65bc` |
| weights.json | `70af73e6f8ee3f4fb76981584c1f2068a4b669300be90bd71cfdd8f71b369997` |

P is the corresponding #559-authenticated serving policy (its four artifact
hashes are frozen in the #559 record and checked by the verifier). Evaluate
exactly one raw policy forward pass per T512/root (512 total) and one P parity
forward pass per root (512 total), with no batching-based semantic changes, no
MCTS, search, value-head use, or native oracle execution. The parity passes
establish that archived P priors match frozen serving P before those archived
values are used in the comparison.
Oracle margins, archived P priors, and model priors are indexed by
player-relative legal actions (indices 0–5). Validate these against the rules
engine's relative legal moves. Convert through the current-player offset only
when validating the cohort's `legal_absolute_actions` field. Do not reinterpret
oracle or prior vector indices as absolute pit indices.

Archived P priors may be reused only after the reviewed runner reproduces them
under the frozen P artifact, model input encoding, legal-action encoding,
masking, normalization, and float32 semantics for all 512 roots within the
registered tolerance (maximum absolute difference `1e-6`, exact legal-action
mask). A parity failure stops the run; it must not be repaired by changing
inputs, tolerances, cohort, or implementation after inference begins. This
parity check does not authorize a second policy comparison or alter the 512
T512 evaluation count. Total planned inference is 1,024 raw policy forwards:
512 P parity forwards followed by exactly 512 T512 forwards.

## Metrics and analysis (pre-registered)

For each root, the exact-optimal action set contains every legal action tied at
the maximum oracle margin. The primary observation is the difference between
T512 and P prior probability mass assigned to that set; the primary statistic
is its unweighted mean over all 512 openings. Secondary outcomes are paired
top-prior exact regret (best legal margin minus the margin of the highest-prior
action; ties use lowest absolute action ID), exact-optimal top-prior selection
rate, and WDL downgrade rate (top-prior action's exact WDL is worse than the
best legal exact WDL, with W/D/L derived from margin sign).

Use 20,000 opening-cluster bootstrap resamples, sampling 512 opening IDs with
replacement and retaining the paired model outcomes for each sampled opening.
Seed: `58820261114`. Quantiles use linear interpolation at 0.025 and 0.975.
Report paired point estimates and two-sided 95% intervals for all metrics.
The deterministic resampling implementation and metric definitions must be
frozen before inference.

The sole advance condition is a positive T512-minus-P primary point estimate,
its 95% interval entirely above zero, and non-worsening mean top-prior exact
regret (T512 minus P `<= 0`). Otherwise stop with no search follow-up proposal.
A pass may justify proposing a separately frozen search diagnostic only; it
does not establish superiority, authorize promotion, or alter serving P.
Stop on any source/hash/membership/legality/oracle/parity/runtime mismatch,
non-finite output, missing root, or count other than 512.

## Read-only preparation verification

```bash
python -m ml.alphazero_lite.seed588_raw_prior_diagnostic
python -m pytest ml/alphazero_lite/test_seed588_raw_prior_diagnostic.py -q
```

These commands authenticate the frozen source package, recompute active pit
stones and legal actions, check oracle completeness and archived P prior shape,
and authenticate policy identities. They do not instantiate evaluators or run
inference. An independent verifier should repeat them from a physical copy of
the repository and compare file hashes before any reviewed execution.

No training, games, MCTS, sealed-suite access, or model promotion is part of
this protocol.
