# T512 raw-policy entropy diagnostic

**Status:** read-only analysis of authenticated #557 and #591 outputs. No new
inference, training, search, games, or oracle calls were run. Seed 588's
registered `do_not_propose_search_followup` decision is unchanged.

## Archived evidence and reproduction

Both archive verifiers passed: `verify_seed557_retrospective` authenticated the
256-root cohort, 768 policy records, full oracle coverage, and model artifacts;
`verify_seed588_raw_prior_result` authenticated the 512-root #591 package,
authorization, freeze, source, artifacts, parity, and metrics.

| Archive | Bound evidence SHA-256 |
|---|---|
| #557 execution freeze | `b0892a24872012486ff164d291913e1799737632a816a40c2547210a458bf36d` |
| #557 `search-records.jsonl` | `6b336dbd73f0ca90620f91737f9fd9a8600e22cf178efb62fd63ec04e5e916e8` |
| #591 `run-manifest.json` | `5b838e273cec56f3e8b8fe2c0a5fcf3db41c4e5180281135d14e597d1a085c67` |
| #591 `p-parity.jsonl` | `dfa400a60db7cdd10a8f0cadc5de47d81841e0197127cf82e1bd8da287f4811b` |
| #591 `t512-predictions.jsonl` | `e08df6e4878af758b5fb3e09014d60d2a1ae36702e7cf6a21edd20ca7a5ede13` |
| Authenticated T512 `weights.json` (both archives) | `70af73e6f8ee3f4fb76981584c1f2068a4b669300be90bd71cfdd8f71b369997` |
| Authenticated serving-P `weights.json` (both archives) | `89dead525a255cff4760867645fd9ce05cb2bbadcf1d72c5ac9089868c4a8136` |

The model weights are the same update-108 T512 and serving-P identities in both
archives. Entropy is Shannon entropy in nats over legal relative actions only;
top action uses maximum prior with lowest legal action index as the tie-break.
Optimal-action mass sums the prior over every action tied at the exact best
margin. Results are unweighted root means.

| Cohort | Roots | Mean H(P) | Mean H(T512) | Top-action agreement | Optimal mass P | Optimal mass T512 | T512−P mass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| #557 | 256 | 0.74040 | 0.87006 | 91.80% | 0.58977 | 0.56025 | −0.02951 |
| #591 | 512 | 0.72238 | 0.84414 | 92.58% | 0.64405 | 0.61273 | −0.03133 |

The exploratory *global* entropy match applies one scalar temperature to every
T512 root, `q(a) ∝ p(a)^(1/τ)` over legal actions, choosing `τ` so mean T512
entropy equals mean P entropy. The fitted temperature uses only archived policy
vectors, not oracle outcomes; the optimal-mass column then descriptively applies
that fixed temperature to the already-authenticated oracle-defined optimal
sets:

| Cohort | Fitted τ | Matched T512 optimal mass | P optimal mass | Top actions changed? |
|---|---:|---:|---:|---|
| #557 | 0.76075 | 0.58674 | 0.58977 | No |
| #591 | 0.77431 | 0.64068 | 0.64405 | No |

This recovers about 90% of the observed optimal-mass gap while preserving every
T512 ranking. It is a retrospective descriptive fit on these cohorts, not a
calibration value for deployment or future evaluation.

The top-action outcome summaries are less separated than optimal-action mass:
the exact-optimal top-action rates were P 65.63% vs T512 62.50% on #557 and
72.85% vs 71.88% on #591. On #591, the paired selection-rate difference was
−0.98 percentage points with the recorded bootstrap interval [−2.93, +0.98]
points. Lower optimal-action *mass* is not proof that T512 chooses inferior
top actions; it mixes ranking with confidence/spread. The high pairwise
top-action agreement also means that rankings do differ on a minority of roots,
so the whole gap cannot be dismissed as temperature alone.

## What the T512 training record shows

The authenticated #554 seed-554 treatment is the source of update-108 T512
(model SHA-256 `8940cbff36976801e74542f5ee2d6c5caffe0cafd57c7e787cdb375f3bbe485e`).
The source-bound `train.py` SHA-256 is
`2dcb42f8dff790d5033e2c9a865f3c24eb1fee6b9b37302200dd176d109dad32`; the
seed-554 runner SHA-256 is
`5e770a2cf6e69d43a460a3a0044532d9aa086ff8762b68417c4b5351c5882925`.
The training-input binding and all 12 control/treatment derivative shard hashes
were rechecked read-only.

- **Targets:** for each unique training state above 16 active stones, replace
  the control's 128-simulation PUCT visit target with the paired 512-simulation
  visit target at temperature 1.0. Targets are denoised (target Dirichlet
  epsilon 0); exact-leaf solving is off. All ≤16-stone targets are copied
  unchanged. The two arms have identical inputs, values, masks, row weights,
  occurrence identities, and schedules; only the high-stone policy labels
  differ. In the 13,818 occurrences, 8,928 policy vectors differ and 1,215
  highest-target actions differ.
- **Target entropy check:** archived T512 targets are *sharper*, not softer:
  mean C128/T512 target entropy is 0.69906/0.62422 nats above 32 stones and
  0.77734/0.67572 nats at 17–32 stones. At ≤16 stones both are unchanged.
- **Replay/schedule:** one fixed pool of 13,818 original #549 control-training
  occurrences, six shards, no added replay source or newly generated self-play.
  Both arms use the identical fixed 108-update schedule and 54,896 total
  exposures (four shuffled passes: 26 batches of 512 plus one of 412 per pass).
- **Objective/regularization:** residual-v3 shared trunk, 27 inputs, 96-wide
  three-block model initialized identically. Legal-mask weighted policy cross
  entropy plus `0.5 ×` Huber value loss (delta 1). Each arm starts with fresh
  Adam (`lr=1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`), no weight decay,
  scheduler, entropy term, or logit/norm penalty; global gradient clipping is
  1.0. The saved T512 trace has clipping active on 72/108 updates. The shared
  trunk lets unchanged value gradients interact with changed policy gradients.
- **Checkpoint selection:** checkpoints are fixed at updates 27, 54, 81, and
  108; update 108 alone is the candidate. No entropy-based checkpoint
  selection occurs. The control arm was production-update-equivalence checked
  before T512 training. T512 development policy CE is slightly worse than C128
  on the same 4,738 held-out occurrences (exposure 0.631068 vs 0.628356;
  equal-input 0.628258 vs 0.625407), while actual-value MSE is nearly equal.

On those already-recorded development predictions, mean entropy is 0.78014 for
T512 and 0.77453 for C128; mean legal-logit standard deviation is 0.92599 vs
0.95490. This independent-of-#557/#559 development comparison supports a
small-spread difference, but does not isolate its cause or show playing
strength. The label-entropy result rules out “T512 learned softer targets” as a
direct explanation. More likely candidates are the different high-stone
ranking/gradient directions interacting with the shared value trunk, frequent
gradient clipping, and a finite fixed update endpoint. There is no explicit
policy-temperature or logit-scale regularizer to constrain the final spread.
These are plausible mechanisms, not causal findings.

## One falsifiable, inexpensive intervention

**Hypothesis:** most of the repeatable T512 entropy excess is a global policy
logit-scale shift induced by the high-stone label change and shared clipped
optimization, rather than a wholesale change in action ranking. The archived
global scalar fits support this: they explain about 90% of the mass gap without
changing top actions, while leaving a small residual ranking-dependent gap.

**One experiment:** test a single post-hoc global T512 policy temperature, with
no weight updates. Before viewing predictions, select 256 new distinct-opening
calibration roots and 256 disjoint validation roots from a separately sourced
self-play archive, balanced across legal-action counts and active-stone strata;
exclude canonical and production-float32 matches to #557/#559. Fit one `τ` on
calibration roots only by matching T512's mean legal-action entropy to serving
P. Freeze that `τ` and all validation membership before evaluation. On the
held-out validation roots, report raw and calibrated entropy gap and top-action
agreement; accept the scale hypothesis only if the absolute mean entropy gap
falls by at least 75% and to ≤0.01 nats, with exact top-action invariance under
the transform. Otherwise reject a single global scale correction. Do not use
#557/#559 outcomes or oracle margins to fit or select `τ`.

This is a policy-calibration diagnostic only—not a search experiment, strength
claim, or promotion path. It does not reopen or alter Seed 588's failed
registered decision.
