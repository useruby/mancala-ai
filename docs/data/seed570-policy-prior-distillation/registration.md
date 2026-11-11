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

**Control:** frozen serving-P E4 policy and trunk, initializer value head and
trunk, and the public #557 256-root cohort.

**Treatment:** initialize from exactly the same serving-P components and
fine-tune only the policy head and shared trunk by distilling the normalized
128-simulation PUCT visit distribution on the frozen #557 roots. Keep the
initializer value head/trunk fixed; do not change architecture, input
encoding, search, or serving runtime. Freeze the training seed, optimizer,
schedule, target temperature, and resulting artifact hashes in a separate
pre-training registration before training starts. Do not include #557 oracle
margins or evaluation outcomes as training targets.

The hypothesis is that PUCT-corrected targets can improve the raw prior's
selection of useful actions, addressing the specific prior-versus-post-search
gap recorded in #557 without replacing the unchanged search or value component.

## Prospective public development gate

Use only the already published #557 roots and oracle records for this first
diagnostic. Compare control and treatment on the exact same roots, with fresh
trees and the **#568 shared search contract**: 128 simulations per side,
`c_puct=1.25`, deterministic root policy, native exact-root routing at
threshold 16 with final-score-margin objective, highest-prior then lowest-index
tie break, exact-leaf solving disabled, no subtree reuse, and one search worker
per evaluation. Keep matched per-root search seeds and bind exact model/search
artifact hashes before evaluation. The training uses no new positions beyond
this public cohort.

Primary development statistic: treatment-minus-control exact final-margin
regret, paired by root and clustered by opening. Advance to a separately
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

If any development-gate condition fails, stop this path: record the result as
`do_not_advance`, do not run a full-game screen, do not increase search budget,
and do not iterate against this cohort. If execution or verification is
incomplete, treat the gate as failed for advancement. Preserve serving P.

This registration relies only on public merged repository evidence. It does
not access or disclose private custody, the seed, or ordered openings. The
sealed suite remains untouched; no candidate freeze or reveal is requested.
The unresolved historical coverage limitation remains, and **#483 stays
closed**.
