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

| #557 policy | Mean entropy (nats) | Mean optimal-action prior mass | Exact-optimal top-action rate |
|---|---:|---:|---:|
| Serving P | 0.74040 | 0.58977 | 65.63% |
| C128 | 0.86574 | 0.55363 | 62.89% |
| T512 | 0.87006 | 0.56025 | 62.50% |

| Paired contrast | Entropy difference | Optimal-mass difference |
|---|---:|---:|
| C128 − P | +0.12534 nats | −0.03614 |
| T512 − C128 | +0.00432 nats | +0.00663 |
| T512 − P | +0.12966 nats | −0.02951 |

Top-action agreement on these same roots was 91.41% for C128 vs P, 95.70% for
T512 vs C128, and 91.80% for T512 vs P. On #591 (512 roots), P/T512 mean
entropy was 0.72238/0.84414 nats, top-action agreement 92.58%, and optimal-action
prior mass 0.64405/0.61273. Thus #591 reproduces the serving-P/T512 contrast,
while #557's three-policy comparison localizes most of that entropy gap to the
C128-versus-P contrast.

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

The controlled 256-root incremental contrast is T512−C128: entropy +0.00432
nats, optimal-action prior mass +0.00663, and top-action agreement 95.70%.
Their exact-optimal top-action rates are 62.89% for C128 and 62.50% for T512.
No interval for these contrasts was registered. Lower optimal-action *mass* is
not proof that a policy chooses inferior top actions; it mixes ranking with
confidence/spread. Top actions do differ on some roots, so the complete gap
cannot be dismissed as temperature alone.

## What the T512 training record shows

The authenticated #554 seed-554 treatment is the source of update-108 T512
(model SHA-256 `8940cbff36976801e74542f5ee2d6c5caffe0cafd57c7e787cdb375f3bbe485e`).
The source-bound `train.py` SHA-256 is
`2dcb42f8dff790d5033e2c9a865f3c24eb1fee6b9b37302200dd176d109dad32`; the
seed-554 runner SHA-256 is
`5e770a2cf6e69d43a460a3a0044532d9aa086ff8762b68417c4b5351c5882925`.
The training-input binding and all 12 control/treatment derivative shard hashes
were rechecked read-only.

The #557 serving P artifact identifies as `seed505-E4`; it is not either
seed-554 training arm. C128 and T512 were both initialized from the same
seed-542 initializer and trained with fresh Adam states on the same data and
schedule. Therefore the large C128−P contrast (+0.12534 nats) mixes different
model lineage/initialization and training/checkpoint histories; it is not an
estimate of the T512-specific teacher-budget effect. The within-seed-554
T512−C128 contrast (+0.00432 nats) is the controlled target-replacement
contrast on #557.

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
0.95490. This comparison is consistent with a small incremental T512 spread
effect on that development set, but neither it nor the #557 paired result
isolates a causal mechanism or shows playing strength. It also cannot explain
the much larger P-versus-C128 difference. The label-entropy result rules out
“T512 learned softer targets” as a direct explanation. Shared trunk/value
gradients, frequent clipping, and selecting only the fixed update-108 endpoint
are possible influences; there is no explicit policy-temperature or logit-scale
regularizer. These are candidates, not causal findings.

## Next existing evidence; defer new-root calibration

Defer the proposed new-root temperature experiment. The most informative
existing evidence to inspect next is the paired #554 update trace and saved
checkpoints at updates 27, 54, 81, and 108:
`control-updates.jsonl`, `treatment-updates.jsonl`, and the corresponding
`control-update-*.npz` / `treatment-update-*.npz`. A read-only comparison of
policy-head weight/bias norms, shared-trunk tensor deltas, and recorded
pre-/post-clipping gradients can show when the C128/T512 trajectories first
separate. Existing development predictions are available only for the
initializer and final checkpoints, so those update checkpoints have not been
shown here to change raw-policy entropy. This inspection needs no forward pass
and would distinguish a late checkpoint effect from an accumulated
training-path effect more directly than another root cohort.

The descriptive temperature alignment is not demonstrated probability
calibration and does not establish playing-strength improvement. No new
experiment is proposed here. Seed 588's failed registered decision is
preserved; this analysis does not authorize search, promotion, serving changes,
or holdout access.
