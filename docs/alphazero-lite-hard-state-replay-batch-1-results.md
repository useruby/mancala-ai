# AlphaZero-lite Hard-State Replay Batch 1 Results

## Outcome

The hard-state replay experiment ran to completion after backfilling mining-compatible row-level forensic artifacts from the batch-1 candidates.

The upstream trigger remained the same as `docs/alphazero-lite-post-merge-batch-1-results.md`: batch 1 found no winning replay/search/value/temperature variant, so the next branch was targeted hard-state replay.

The replay variants did improve holdout hard-state regret and value calibration versus the incumbent, but none improved arena strength over the batch-1 fresh baseline. All three weights tied the same `0.50` arena score, stayed below the `0.55` gate, and failed benchmark promotion checks.

## Artifact Generation

Generated train-only mining artifacts:

- train-only forensic suite: `/tmp/azlite_hard_state_backfill_plan/train_only_forensic_suite.json`
- candidate forensic reports: `/tmp/azlite_hard_state_backfill_plan/*/candidate_forensic_suite.json`
- backfill manifest: `/tmp/azlite_hard_state_backfill_plan/manifest.json`

Guardrails followed:

- did not mine from `ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json`
- excluded holdout canonical states when building the train-only forensic suite
- kept the replay train set separate at `/tmp/azlite_hard_state_replay/post-merge-batch1-hard-state-replay/inputs/hard_state_train_seed42.jsonl`
- used explicit incumbent path `storage/ai/alphazero_lite/current`
- did not promote any model

## Preflight

Initial preflight against the raw batch-1 outputs failed with:

```text
no supported hard-state mining artifacts found
```

After backfilling candidate forensic reports, mining succeeded.

Shared dataset summary:

- mined_state_count: `221`
- selected_top_n_for_labeling: `64`
- label_entropy_mean: `1.672225`
- top_teacher_move_distribution: `0:3, 1:18, 2:12, 3:9, 4:7, 5:15`
- mined_reason_distribution: `large_value_error: 221`

Incumbent hard-state validation baseline:

- policy_top1_agreement: `0.0000`
- average_regret: `0.0933`
- value_calibration_mae: `0.4320`

## Results Table

| variant | hard_state_weight | mined_state_count | label_entropy_mean | top_teacher_move_distribution | hard_state_policy_top1_agreement | hard_state_average_regret | hard_state_value_calibration_mae | arena_score | arena_ci_low | arena_ci_high | unstable_decision | mcts1200_score | benchmark_pass | decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `w1` | `1` | `221` | `1.672225` | `0:3,1:18,2:12,3:9,4:7,5:15` | `0.0000` | `0.0873` | `0.3555` | `0.5000` | `0.4119` | `0.5881` | `true` | `0.6167` | `false` | `reject` | best regret of the three, but no arena gain over fresh batch-1 baseline and worse value calibration than `w2` |
| `w2` | `2` | `221` | `1.672225` | `0:3,1:18,2:12,3:9,4:7,5:15` | `0.0000` | `0.0903` | `0.3405` | `0.5000` | `0.4119` | `0.5881` | `true` | `0.6167` | `false` | `winner_for_rerun` | best value calibration and lowest blunder rate, while tying the arena and MCTS outcomes of the other weights |
| `w4` | `4` | `221` | `1.672225` | `0:3,1:18,2:12,3:9,4:7,5:15` | `0.0000` | `0.0879` | `0.3696` | `0.5000` | `0.4119` | `0.5881` | `true` | `0.6167` | `false` | `reject` | improved over incumbent on hard-state metrics, but trailed `w2` on calibration and showed no arena upside |

## Interpretation

What improved:

- all three weights beat the incumbent holdout on average regret
- all three weights materially improved incumbent value calibration MAE
- `w2` produced the best overall hard-state validation profile

What did not improve:

- policy top-1 agreement stayed `0.0000` for incumbent and all three replay weights
- arena score stayed flat at `0.50` for all three weights
- `MCTS1200` stayed flat at `0.6167` for all three weights and matched the current baseline report
- benchmark promotion checks failed for all three weights because the arena gate failed

Important caveat:

- the mined dataset was dominated by one failure family: `large_value_error`
- this means the replay intervention mostly tested targeted value-calibration repair, not broad tactical disagreement coverage

## W2 Follow-Up

Two kinds of `w2` follow-up reruns were completed:

- label-seed reruns under the original wrapper behavior
- full runtime-seed reruns after fixing `run_hard_state_replay_experiment.py` so the experiment seed propagates into runtime config `seed`, step `--seed` flags, and self-play `--seed-sweep`

| run | label_entropy_mean | top_teacher_move_distribution | hard_state_average_regret | hard_state_value_calibration_mae | arena_score | arena_ci_low | arena_ci_high | unstable_decision | mcts1200_score | benchmark_pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `w2-seed42` | `1.672225` | `0:3,1:18,2:12,3:9,4:7,5:15` | `0.0903` | `0.3405` | `0.5000` | `0.4119` | `0.5881` | `true` | `0.6167` | `false` | hard-state gain over incumbent, no arena lift |
| `w2-seed43` | `1.660454` | `0:1,1:21,2:12,3:6,4:6,5:18` | `0.0943` | `0.3390` | `1.0000` | `0.9690` | `1.0000` | `false` | `0.6167` | `true` | large arena outlier; holdout regret slightly worse than incumbent despite better calibration |
| `w2-seed44` | `1.661044` | `0:3,1:21,2:10,3:8,4:5,5:17` | `0.0902` | `0.3423` | `0.5000` | `0.4119` | `0.5881` | `true` | `0.6167` | `false` | hard-state gain over incumbent, arena back to flat |

Observed pattern from label-seed reruns:

- hard-state value calibration improvement is consistent across all three `w2` runs
- hard-state regret is slightly better than incumbent for seeds `42` and `44`, and slightly worse for seed `43`
- arena is not stable across label-seed reruns: two runs tie baseline at `0.50`, one run jumps to `1.00`

### Corrected Full-Seed Reruns

After fixing seed propagation, the full runtime-seed reruns are:

| run | runtime seed | self_play seed_sweep | hard_state_average_regret | hard_state_value_calibration_mae | arena_score | arena_ci_low | arena_ci_high | unstable_decision | mcts1200_score | benchmark_pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `w2-fullseed43` | `43` | `42,43,44` | `0.0879` | `0.3647` | `0.5000` | `0.4119` | `0.5881` | `true` | `0.5167` | `false` | better regret than incumbent, but no arena gain and weaker MCTS1200 than seed42/seed44 |
| `w2-fullseed44` | `44` | `43,44,45` | `0.0887` | `0.3462` | `0.5000` | `0.4119` | `0.5881` | `true` | `0.6000` | `false` | better regret and calibration than incumbent, but arena still flat and below gate |

Corrected interpretation:

- the earlier `w2-seed43` arena `1.00` result was not representative of a true runtime-seed rerun
- with real seed propagation, `w2` stays flat at arena `0.50` across the observed full-seed reruns
- hard-state validation does improve versus incumbent, but arena strength does not move with it
- this is overfitting risk under the stated decision rules

## Failure-Family Diagnostics

Diagnostic artifacts:

- shared train-only references: `/tmp/azlite_failure_family_diag/train_only_forensic_references.json`
- baseline opening-capture report: `/tmp/azlite_failure_family_diag/aggressive-v3-clone-extend_opening_capture_family_report.json`
- baseline stable-family summary: `/tmp/azlite_failure_family_diag/aggressive-v3-clone-extend_stable_failure_family_summary.json`
- `w2` opening-capture report: `/tmp/azlite_failure_family_diag/exp-v3-replay-bootstrap-w2_opening_capture_family_report.json`
- `w2` stable-family summary: `/tmp/azlite_failure_family_diag/exp-v3-replay-bootstrap-w2_stable_failure_family_summary.json`

Tracked-family comparison from the original batch-1 baseline vs original batch-1 `w2` candidate:

| family | fresh baseline | replay w2 | interpretation |
| --- | --- | --- | --- |
| `capture_available` tracked opening rows | `tracked_rows=4`, `average_regret=0.0058`, `search_flipped_rows=0` | `tracked_rows=4`, `average_regret=0.0193`, `search_flipped_rows=0` | replay `w2` is worse on the tracked opening capture family and search never flips into the correct move |
| `high_imbalance` stable rows | `stable_rows=12`, `average_regret=0.0693`, `blunder_rate_0_20=0.0833` | `stable_rows=12`, `average_regret=0.0684`, `blunder_rate_0_20=0.0833` | effectively unchanged |

Diagnostic readout:

- the mined replay corpus was dominated by `large_value_error`, but the first two tracked failure families do not improve in a meaningful way
- opening capture is especially important because the family report shows candidate prior/search mass still concentrates on the early pits rather than the reference move on the tracked rows
- this supports the conclusion that the replay intervention is mostly calibrating values, not repairing the tactical family that would matter for arena strength

## Recommendation

Next action: **D. abandon hard-state replay and move to failure-family diagnostics**.

Why this is the best next step now:

- full runtime-seed reruns do not show arena improvement
- all replay weights and corrected `w2` follow-ups remain stuck at arena `0.50`
- the mined corpus is dominated by one failure family: `large_value_error`
- the intervention improves holdout calibration faster than it improves playing strength, which is not enough under the guardrails

Diagnostic focus to take next:

- inspect whether the mined states are representative of real arena failures
- split the mined corpus into narrower tactical/value families instead of one replay bucket
- prioritize opening capture and high-imbalance family diagnostics before any more replay weighting sweeps
- check whether the teacher labels are too value-heavy and not action-discriminative enough for arena strength gains

## Opening-Capture Tactical Lane Check

I also ran the built-in opening-capture tactical replay lane as an exploratory confirmation branch using `ml/alphazero_lite/tactical_opening_capture_family_replay.jsonl` with the incumbent path overridden to `storage/ai/alphazero_lite/current`.

Artifacts:

- runtime config: `/tmp/azlite_failure_family_diag/aggressive_v3_tactical_opening_capture_family_runtime.json`
- run root: `/tmp/azlite_v3_tactical_opening_capture_family_batch1_versions/aggressive-v3-tactical-opening-capture-family-batch1-iter1`

Outcome:

- hard-state validation overall: `average_regret=0.0940`, `value_calibration_mae=0.3310`
- capture-available bucket: `average_regret=0.0494`, `value_calibration_mae=0.1212`
- arena: `0.0000` over `120` games

Interpretation:

- the lane sharply improves value calibration on the opening capture bucket
- but it catastrophically regresses arena strength
- that reinforces the broader conclusion from hard-state replay: targeted replay can move family-local metrics without preserving overall strength
- so the next step should stay in diagnostics and data characterization, not more replay-only training branches

## Opening Move-Selection Diagnostic

Focused move-selection summary artifact:

- `/tmp/azlite_failure_family_diag/opening_search_compare/opening_capture_move_selection_summary.json`

Tracked opening-capture rows:

| row | reference_move | fresh baseline searched move | replay w2 searched move | fresh searched reference mass | w2 searched reference mass | fresh searched early mass | w2 searched early mass | readout |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `capture_available-005` | `4` | `1` | `0` | `0.0000` | `0.0000` | `0.9505` | `0.9870` | neither model finds the capture; `w2` becomes even more concentrated on the wrong early pits |
| `capture_available-006` | `2` | `2` | `2` | `0.4609` | `0.3906` | `0.2891` | `0.3281` | both recover the right move after search, but `w2` does so with weaker reference support |
| `capture_available-007` | `1` | `1` | `1` | `0.5547` | `0.5521` | `0.7057` | `0.6693` | effectively unchanged |
| `capture_available-008` | `1` | `1` | `1` | `0.5911` | `0.4661` | `0.8229` | `0.7760` | correct move preserved, but `w2` lowers searched support for it |

Additional repo classifier result:

- the built-in search-interaction classifier surfaced only one row, `sparse_endgame-009`, as `persistent_late_game_weakness`
- it did not classify the tracked opening rows as `search_overrides_prior` or `value_sign_miscalibration`

Opening-family conclusion:

- the main actionable opening capture defect is not that search flips away from a good prior
- it is that the candidate often enters search already pointed at the wrong early-pit family, and search does not recover enough reference-move mass to fix it reliably
- this points toward failure-family characterization and policy-target construction work, not more replay-weight sweeps

## Opening Policy-Target Diagnostic

Policy-target diagnostic artifact:

- `/tmp/azlite_failure_family_diag/opening_capture_policy_target_diagnostic.json`

Key findings:

- tracked failing opening rows require reference moves `{1, 2, 4}` with distribution `1:2, 2:1, 4:1`
- the built-in opening-capture replay artifact contains exactly `3` opening-family rows
- all three of those replay rows teach `teacher_selected_move = 3`
- the replay source row ids are `capture_available-017`, `capture_available-019`, and `capture_available-024`
- there is zero overlap with the tracked failing row ids `capture_available-005` through `capture_available-008`

Mechanism counts on the tracked rows:

- fresh baseline: `correct_family_preserved=2`, `search_rescued_wrong_family=1`, `wrong_family_unrecovered=1`
- replay `w2`: `correct_family_preserved=2`, `search_rescued_wrong_family=1`, `wrong_family_unrecovered=1`

Interpretation:

- the current opening-capture replay target is aimed at a different capture motif than the family that is failing in batch 1
- this explains why the opening-capture tactical lane can improve local capture calibration yet still miss the tracked failing moves and collapse in arena
- the next intervention should be a policy-focused family artifact that explicitly covers the tracked `capture_available-005..008` move targets instead of reusing the existing move-3-only opening capture replay rows

## Tracked Opening Artifact

Built a new policy-focused artifact for the exact tracked failing opening rows:

- artifact: `/tmp/azlite_failure_family_diag/tracked_opening_capture_policy_artifact.jsonl`
- summary: `/tmp/azlite_failure_family_diag/tracked_opening_capture_policy_artifact_summary.json`

Summary:

- `row_count=4`
- `teacher_selected_move_distribution={1: 2, 2: 1, 4: 1}`
- `row_ids=capture_available-005..008`
- `mechanism_counts={correct_family_preserved: 2, search_rescued_wrong_family: 1, wrong_family_unrecovered: 1}`

Why this artifact is different from the existing opening-capture replay rows:

- it directly targets the failing batch-1 opening capture rows instead of unrelated capture motifs
- it preserves exact reference targets from the shared forensic references artifact
- it is policy-focused: the rows carry the teacher policy/child-stats needed to train the model toward moves `1`, `2`, and `4` on the failing family

## Tracked Opening Policy Lane Check

Ran a minimal tactical lane using only the new tracked opening-family artifact:

- runtime config: `/tmp/azlite_failure_family_diag/aggressive_v3_tracked_opening_capture_policy_runtime.json`
- run root: `/tmp/azlite_v3_tracked_opening_capture_policy_batch1_versions/aggressive-v3-tracked-opening-capture-policy-batch1-iter1`

Outcome:

- overall hard-state validation: `average_regret=0.0996`, `value_calibration_mae=0.3720`
- `capture_available` bucket: `average_regret=0.0348`, `value_calibration_mae=0.1450`
- arena: `0.5000` over `120` games

Comparison to the earlier move-3-only opening-capture lane:

- old opening-capture lane arena: `0.0000`
- tracked opening lane arena: `0.5000`

Interpretation:

- targeting the actual failing opening rows removes the catastrophic arena collapse from the old opening-capture lane
- but it still does not produce arena improvement over the incumbent
- and it does not beat the incumbent on capture-bucket regret (`0.0348` vs incumbent `0.0262`)
- the artifact looks directionally safer than the old lane, but still not sufficient as a standalone replay/policy finetune for promotion-relevant strength

## Broader Opening Artifact

Built a broader opening-family-plus-guards artifact by preserving the existing tactical guard composition and replacing the old move-3-only opening rows with the tracked failing opening rows:

- artifact: `/tmp/azlite_failure_family_diag/tracked_opening_capture_family_plus_guards.jsonl`
- summary: `/tmp/azlite_failure_family_diag/tracked_opening_capture_family_plus_guards_summary.json`

Composition:

- `row_count=13`
- replay roles:
  - `capture_protection=1`
  - `capture_preservation=2`
  - `nearby_preservation=6`
  - `opening_capture_family=4`
- tracked opening row ids:
  - `capture_available-005`
  - `capture_available-006`
  - `capture_available-007`
  - `capture_available-008`
- teacher move distribution across the combined artifact:
  - move `0`: `3`
  - move `1`: `4`
  - move `2`: `3`
  - move `3`: `1`
  - move `4`: `2`

Why this is the right next training artifact:

- it keeps the proven guard structure from the existing tactical lane
- it injects the exact failing opening family instead of unrelated move-3-only rows
- it broadens the policy target support enough to reduce the overfitting risk from the 4-row standalone tracked artifact

## Broader Opening Lane Check

Ran a lane on the broader opening-family-plus-guards artifact:

- runtime config: `/tmp/azlite_failure_family_diag/aggressive_v3_tracked_opening_capture_family_plus_guards_runtime.json`
- run root: `/tmp/azlite_v3_tracked_opening_capture_family_plus_guards_batch1_versions/aggressive-v3-tracked-opening-capture-family-plus-guards-batch1-iter1`

Outcome:

- overall hard-state validation: `average_regret=0.0891`, `value_calibration_mae=0.3423`
- `capture_available` bucket: `average_regret=0.0317`, `value_calibration_mae=0.1144`
- arena: `0.0000` over `120` games

Comparison across the three opening-focused branches:

| branch | capture_available regret | capture_available mae | overall regret | overall mae | arena |
| --- | --- | --- | --- | --- | --- |
| move-3-only opening lane | `0.0494` | `0.1212` | `0.0940` | `0.3310` | `0.0000` |
| tracked 4-row lane | `0.0348` | `0.1450` | `0.0996` | `0.3720` | `0.5000` |
| broader tracked+guards lane | `0.0317` | `0.1144` | `0.0891` | `0.3423` | `0.0000` |

Interpretation:

- adding guards improves the opening capture bucket relative to the narrower opening-family lanes
- but it does not preserve arena strength; the broader lane falls back to the same catastrophic `0.00` arena result as the old move-3-only opening lane
- this suggests the failure is not just missing guard coverage; the current family-specific finetune path itself is still too brittle for promotion-oriented training

## Stability Diagnostic

Used the repo’s guard-row arbitration tool on the two most relevant opening-family finetunes:

- tracked-only run arbitration:
  - `/tmp/azlite_failure_family_diag/stability_runs/tracked/final/capture_002_003_search_policy_arbitration.json`
- broader-with-guards run arbitration:
  - `/tmp/azlite_failure_family_diag/stability_runs/broader/final/capture_002_003_search_policy_arbitration.json`

Both runs produced the same classification:

- `classification=state_specific_rule_collision`
- `decision=write_rule_collision_spec`
- follow-up spec: `docs/alphazero-lite-capture-002-003-rule-collision-spec.md`

Shared evidence summary:

- `capture_available-002` and `capture_available-003` diverge on the tactical rule feature `extra_turn_available`
- both finetunes still select the wrong searched move on `capture_available-002`
- both preserve the searched move on `capture_available-003`

Important drift detail:

- tracked-only run on `capture_available-002`:
  - `reference_move_visit_share=0.2005`
  - `selected_move_visit_share=0.5469`
  - `q_margin=0.0168`
- broader-with-guards run on `capture_available-002`:
  - `reference_move_visit_share=0.0964`
  - `selected_move_visit_share=0.6510`
  - `q_margin=-0.0416`

Interpretation:

- the instability is not primarily generic search drift or missing guard rows
- the problem looks like a rule-collision-style local conflict in `capture_available-002`, where the finetuned policy/search stack still amplifies the wrong move on a guard row
- broadening the artifact changed local calibration, but did not fix the guard-row collision and in this case worsened the guard-row visit-share split

## Rule-Collision Follow-Up

Built the row-pair rule-collision diagnostic required by `docs/alphazero-lite-capture-002-003-rule-collision-spec.md`:

- diagnostic artifact: `/tmp/azlite_failure_family_diag/capture_002_003_rule_collision_diagnostic.json`
- markdown note: `docs/alphazero-lite-capture-002-003-rule-collision-note.md`

Key result:

- best explanation: `extra_turn_overvaluation`
- recommendation: `rule_conditioned_policy_artifact_redesign`

Why this is the current best explanation:

- `capture_available-002` reference move `4` is the only compared branch that does **not** grant an extra turn
- both finetuned variants still search-select move `2` on row `002`, and move `2` grants an extra turn
- `capture_available-003` stays stable because its reference move `1` already grants an extra turn
- in the broader-with-guards run, row `002` child Q already favors the reference move `4` (`q_margin=-0.0416` for selected minus reference), but visits still collapse onto move `2` (`0.6510` vs `0.0964`)

Important rule-level implication:

- this is not best explained by missed capture reward on row `002`
- none of the compared reference/selected moves on rows `002` and `003` are capture-legal
- store gain is the same across the compared moves
- the clean structural difference that matches the failure is extra-turn polarity, plus the resulting post-move board shape

Updated conclusion:

- the next useful branch is still non-training by default, but the evidence now points at a concrete redesign target rather than a generic unresolved collision
- if and when another intervention is attempted, it should explicitly separate no-extra-turn opening targets from extra-turn opening targets instead of mixing them into one local family artifact

Follow-up spec for that branch:

- `docs/alphazero-lite-opening-family-rule-conditioned-redesign-spec.md`

## Rule-Conditioned Artifact Redesign

Implemented the non-training artifact redesign from `docs/alphazero-lite-opening-family-rule-conditioned-redesign-spec.md`.

Code changes:

- `ml/alphazero_lite/build_tracked_opening_capture_policy_artifact.py`
- `ml/alphazero_lite/build_tracked_opening_capture_family_plus_guards.py`
- `ml/alphazero_lite/capture_002_003_rule_collision_diagnostic.py` reused as the rules-based move-polarity helper

New replay roles:

- `opening_capture_extra_turn_reference`
- `opening_capture_no_extra_turn_reference`

Generated inspection artifacts:

- tracked-only summary:
  - `/tmp/azlite_failure_family_diag/tracked_opening_capture_policy_artifact_rule_conditioned_summary.json`
- plus-guards summary:
  - `/tmp/azlite_failure_family_diag/tracked_opening_capture_family_plus_guards_rule_conditioned_summary.json`

Tracked-only split summary:

- `row_count=4`
- `opening_capture_extra_turn_reference=3`
- `opening_capture_no_extra_turn_reference=1`
- extra-turn rows: `capture_available-006`, `capture_available-007`, `capture_available-008`
- no-extra-turn row: `capture_available-005`

Plus-guards split summary:

- `row_count=13`
- `guard_row_count=9`
- tracked replay-role counts:
  - `opening_capture_extra_turn_reference=3`
  - `opening_capture_no_extra_turn_reference=1`
- full replay-role counts:
  - `capture_protection=1`
  - `capture_preservation=2`
  - `nearby_preservation=6`
  - `opening_capture_extra_turn_reference=3`
  - `opening_capture_no_extra_turn_reference=1`

Interpretation:

- the redesign now proves that tracked opening rows are no longer collapsed into one undifferentiated opening-family replay role
- the only no-extra-turn tracked opening row is isolated instead of being mixed with the extra-turn rows that dominated the old artifact
- this satisfies the intended non-training exit condition for the redesign branch: the artifact can now be inspected with explicit extra-turn polarity before any future lane is considered

## Rule-Collision Guard Coverage Check

Checked whether the current guard source or redesigned opening-family artifacts explicitly include the rule-collision pair `capture_available-002/003`.

Coverage result:

- `ml/alphazero_lite/tactical_opening_capture_family_replay.jsonl`: does not include `capture_available-002` or `capture_available-003`
- `/tmp/azlite_failure_family_diag/tracked_opening_capture_policy_artifact_rule_conditioned.jsonl`: does not include `capture_available-002` or `capture_available-003`
- `/tmp/azlite_failure_family_diag/tracked_opening_capture_family_plus_guards_rule_conditioned.jsonl`: does not include `capture_available-002` or `capture_available-003`

Implication:

- even after the opening-family redesign, the exact row pair that exposed the extra-turn rule collision still is not explicitly represented in the artifact composition
- so any future lane launched without adding those rows would still be missing the concrete guard pair that motivated the redesign

Built explicit rule-collision guard artifact:

- artifact: `/tmp/azlite_failure_family_diag/capture_002_003_rule_collision_guard_artifact.jsonl`
- summary: `/tmp/azlite_failure_family_diag/capture_002_003_rule_collision_guard_artifact_summary.json`
- builder: `ml/alphazero_lite/build_capture_002_003_rule_collision_guard_artifact.py`

Summary:

- `row_count=2`
- `rule_collision_no_extra_turn_reference_guard=1`
- `rule_collision_extra_turn_reference_guard=1`
- no-extra-turn guard row: `capture_available-002`
- extra-turn guard row: `capture_available-003`

Updated next step:

- before any future training retry, merge the explicit `capture_available-002/003` rule-collision guard artifact with the redesigned rule-conditioned opening-family artifact and inspect the resulting composition
- do not skip that merge step, because the current redesigned opening-family artifact alone still omits the motivating collision pair

## Full Guarded Pre-Lane Composition

Built the merged artifact that combines:

- `/tmp/azlite_failure_family_diag/tracked_opening_capture_family_plus_guards_rule_conditioned.jsonl`
- `/tmp/azlite_failure_family_diag/capture_002_003_rule_collision_guard_artifact.jsonl`

New builder:

- `ml/alphazero_lite/build_rule_conditioned_opening_family_full_guarded_artifact.py`

Generated artifacts:

- merged artifact: `/tmp/azlite_failure_family_diag/rule_conditioned_opening_family_full_guarded_artifact.jsonl`
- merged summary: `/tmp/azlite_failure_family_diag/rule_conditioned_opening_family_full_guarded_artifact_summary.json`

Merged summary:

- `row_count=15`
- replay roles:
  - `capture_protection=1`
  - `capture_preservation=2`
  - `nearby_preservation=6`
  - `opening_capture_extra_turn_reference=3`
  - `opening_capture_no_extra_turn_reference=1`
  - `rule_collision_extra_turn_reference_guard=1`
  - `rule_collision_no_extra_turn_reference_guard=1`
- tracked opening row ids:
  - `capture_available-005`
  - `capture_available-006`
  - `capture_available-007`
  - `capture_available-008`
- rule-collision guard row ids:
  - `capture_available-002`
  - `capture_available-003`

Important inspection result:

- the full pre-lane composition now explicitly contains both the tracked opening family split by extra-turn polarity and the motivating `002/003` collision guard pair
- this is the first artifact composition in the branch that actually contains both pieces at once

Minor summary caveat:

- `row_ids_by_replay_role` in the merged summary is a provenance convenience view, not a deduplicated count check
- one preserved `capture_preservation` row shares the same `source_runs[0].id` as another row, so the role count is the authoritative composition count

Current best next step:

- stop here unless we explicitly choose to launch a new lane from `/tmp/azlite_failure_family_diag/rule_conditioned_opening_family_full_guarded_artifact.jsonl`
- if we do launch one, this merged artifact is now the correct input candidate to evaluate first
