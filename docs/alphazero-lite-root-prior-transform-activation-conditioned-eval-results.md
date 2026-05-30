# AlphaZero-lite Root-Prior Transform Activation-Conditioned Eval Results

## Context

This experiment tests whether the narrow root-prior transform activates in realistic or semi-realistic positions and whether activation changes downstream search or outcome.

- transform: `seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5`
- current artifact: `/home/alex/Mancala/ai/storage/ai/alphazero_lite/current`
- guarded-w2 artifact: `/tmp/azlite_rule_conditioned_opening_full_guarded/rule-conditioned-opening-full-guarded/w2/versions/aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1`
- classification: `transform_too_rare_for_deployment`
- deployment-gate activation count: `6` / `14586`
- deployment-gate activation rate: `0.000411`

## Activation-rate Scan

| source | artifact | states_scanned | activation_count | activation_rate | median_ply | notes |
| --- | --- | --- | --- | --- | --- | --- |
| finished_games_2026_04_06 | current | 93 | 0 | 0.0 | 23 | no_activation |
| finished_games_2026_04_06 | guarded-w2 | 93 | 0 | 0.0 | 23 | no_activation |
| fixtures | current | 6 | 1 | 0.1667 | 9.5 | ok |
| fixtures | guarded-w2 | 6 | 1 | 0.1667 | 9.5 | ok |
| guard_rows | current | 6 | 1 | 0.1667 | 5.5 | known_guard_rows |
| guard_rows | guarded-w2 | 6 | 1 | 0.1667 | 5.5 | known_guard_rows |
| guarded_w2_self_play | guarded-w2 | 12000 | 0 | 0.0 | 19.0 | no_activation |
| random_prefixes | current | 1200 | 3 | 0.0025 | 5.0 | ok |
| random_prefixes | guarded-w2 | 1200 | 3 | 0.0025 | 5.0 | ok |

## Activated-state Dataset Summary

- activated states written: `10`
- activated states path: `/tmp/azlite_root_prior_activation_eval/activated_states.jsonl`
- non-guard activated states: `8`
- current self-play source status: `unavailable_exact_match`

## Paired Search Results

| artifact | simulations | activated_states | changed_selection_count | changed_selection_rate | reference_improvement_count | reference_regression_count | average_prior_mass_shift | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
No paired search run because the activation-rate gate stopped the experiment after Step 1.

## Outcome Rollout Results

| artifact | simulations | changed_states | continuations_per_state | mean_outcome_delta | ci_low | ci_high | positive_count | negative_count | neutral_count | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
No outcome rollout run because the activation-rate gate stopped the experiment after Step 1.

## Optional Activated-state Micro-arena

Not run because the activation-rate gate did not justify continuing to paired search or rollout.

## Interpretation

- classification: `transform_too_rare_for_deployment`
- interpretation: transform too rare for deployment
- gate basis: only `6` activations across `14586` realistic or semi-realistic scanned states, excluding the hand-built fixture and guard sources

## Recommended Next Action

Recommendation: **abandon this transform as a deployment feature; move to input encoding / policy-target audit.**.
