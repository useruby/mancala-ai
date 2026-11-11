# Registered proposal: policy-prior distillation diagnostic

**Status: registered, not executed.** This is a public development/exploratory
experiment proposal. It is not a candidate freeze, strength claim, serving
recommendation, or authorization to access the sealed suite. No training,
search, or games have been run for this registration.

## Decision from existing evidence

No existing artifact qualifies as stronger than serving P.

- **D432 / #547:** 1,024 paired openings, 2,048 validated games; score
  `0.481689453125`, with one-sided 95% lower bound `0.47021484375`, below the
  strict `>0.50` gate. Its training-budget or development metrics do not
  override this direct strength result.
- **C108 / #552:** 1,024 pairs / 2,048 games; score `0.485107421875`, lower
  bound `0.474853515625`; `do_not_advance_C108`.
- **T512 / #556:** 1,024 pairs / 2,048 games; score `0.497802734375`, lower
  bound `0.486572265625`; `do_not_advance_T512`. Its result against C128 is
  not evidence of superiority to P.
- **#557 diagnostic:** on 256 roots from #556, T512's raw-prior exact-optimal
  rate was 62.50% versus P's 65.63%, and its raw-prior mean regret was 0.1563
  seeds higher (paired interval `[-0.1250, 0.4453]`). After matched 128-simulation
  PUCT, the regret difference was `-0.0078` (interval `[-0.1719, 0.1406]`).
  This is an uncertain raw-prior signal on a selected reached-position cohort,
  not an established full-game cause or strength difference.
- **#558 diagnostic:** substituting D432 value/trunk under P's fixed policy
  changed 9/256 actions but gave zero mean regret improvement; its registered
  improvement rule failed. It does not support a value-only candidate.

These results motivate testing policy-prior quality, while also showing that
the shared search may erase any local gain. They do not justify advancing an
existing model.

## Proposed single change and control

**Training cohort:** the frozen 256-root #557 cohort, whose membership is bound
by the public `cohort.json` and #557 execution freeze. It is used only to form
distillation targets and train the treatment.

**Control:** frozen serving-P E4 policy and trunk, initializer value head and
trunk, evaluated on a separate qualification cohort derived from public #559
`run-0003/cohort-manifest.json` (512 roots from 941 openings, with complete
oracle coverage).

**Treatment:** initialize from exactly the same serving-P components and
fine-tune only the policy head and shared trunk by distilling the normalized
128-simulation PUCT visit distribution on the frozen #557 training roots.
Keep the initializer value head/trunk fixed; do not change architecture, input
encoding, search, or serving runtime. Do not include #557 oracle margins or
evaluation outcomes as training targets.

The hypothesis is that PUCT-corrected targets can improve the raw prior's
selection of useful actions, addressing the specific prior-versus-post-search
gap recorded in #557 without replacing the unchanged search or value component.

## Cohort separation and staged freezes

Before training, construct qualification membership from the public #559 cohort
by excluding, using set-based membership:

1. every opening group (`opening_index`) represented by any #557 training root;
2. every state whose canonical identity **or** production `kalah_v3` float32
   identity collides with any #557 training root; and
3. duplicate qualification roots under either identity scheme, retaining at
   most one root per opening.

Authenticate the #557 and #559 source records, compute both identity schemes
from their states, and publish the exact sorted qualification membership and
digest before training. Require complete exact legal-action oracle coverage
from the existing #559 public records and at least 256 eligible roots from 256
distinct opening groups. If filtering leaves fewer than 256, publish the exact
eligible count and shortfall, mark `stop_shortfall`, and do not train or
evaluate. Do not select replacement roots based on outcomes.

Use two immutable freezes:

- **Training freeze, before training:** bind the #557 membership/hash; serving-P
  teacher policy/trunk and initializer value/trunk hashes; teacher search
  implementation and 128-simulation PUCT settings; target generation
  (`visit_count / sum(visit_count)` over legal moves, zero for unvisited legal
  moves); model inputs; trainable/frozen parameter sets; initialization;
  optimizer, learning rate, batch size, update/epoch schedule, target loss and
  weight, regularization, seed, runtime, and stopping rule. Freeze all these
  values before generating any targets or training. Hash and retain the
  resulting target data. No post-search oracle labels are training targets.
- **Evaluation freeze, after training and before evaluation:** bind exact
  treatment checkpoint/export hashes, unchanged serving-P control hashes,
  qualification membership/hash and both exclusion-index digests, #559 oracle
  identities, evaluation code/runtime, shared search settings, statistic,
  resampling algorithm, seed, resample count, and quantile rule. Record
  treatment artifact identity only after training and before inspecting any
  qualification results.

## Prospective public development gate

Use only the separate, disjoint qualification cohort derived from #559; do not
evaluate on the #557 training roots. Compare control and treatment on the exact
same qualification roots, with fresh trees and the **#568 shared search
contract**: 128 simulations per side,
`c_puct=1.25`, deterministic root policy, native exact-root routing at
threshold 16 with final-score-margin objective, highest-prior then lowest-index
tie break, exact-leaf solving disabled, no subtree reuse, and one search worker
per evaluation. Keep matched per-root search seeds. Training uses only #557
roots; qualification uses only eligible #559 roots after the frozen opening
and two-identity exclusions. No new positions or games are generated.

Primary development statistic: treatment-minus-control exact final-margin
regret, paired by root and clustered by opening. Compute 20,000 paired
opening-cluster bootstrap resamples with replacement using
`numpy.random.default_rng(570)` and linear quantiles (`method="linear"`); report
the two-sided 95% interval at quantiles 0.025 and 0.975. Bind the analysis
implementation and runtime in the evaluation freeze. Advance to a separately
registered public exploratory strength screen only if all conditions hold:

1. mean regret improvement is at least `0.10` seeds (negative treatment minus
   control);
2. the two-sided 95% paired opening-bootstrap interval's upper endpoint is
   strictly below zero; and
3. treatment WDL-downgrade rate is no higher than control.

This local diagnostic gate is for deciding whether the hypothesis merits a
full-game exploratory test; passing it is not qualification as stronger than
P and does not authorize deployment. Any later full-game screen must retain
#568's shared 128-simulation settings and its frozen resource/concurrency
semantics. No settings may be altered to rescue a result.

## Stop rule and scope

If cohort construction yields fewer than 256 eligible roots, stop before
training and record the shortfall. If any development-gate condition fails,
stop this path: record the result as
`do_not_advance`, do not run a full-game screen, do not increase search budget,
and do not iterate against this cohort. If execution or verification is
incomplete, treat the gate as failed for advancement. Preserve serving P.

This registration relies only on public merged repository evidence. It does
not access or disclose private custody, the seed, or ordered openings. The
sealed suite remains untouched; no candidate freeze or reveal is requested.
The unresolved historical coverage limitation remains, and **#483 stays
closed**.
