# AlphaZero-lite Capture 002/003 Rule-Collision Spec

## Goal

Diagnose the `state_specific_rule_collision` identified by `capture_002_003_search_policy_arbitration.py` on guard rows `capture_available-002` and `capture_available-003`.

This is a non-training diagnostic branch.

## Trigger

Both opening-family finetune variants produced the same arbitration result:

- `classification=state_specific_rule_collision`
- `decision=write_rule_collision_spec`

Artifacts:

- tracked-only arbitration:
  - `/tmp/azlite_failure_family_diag/stability_runs/tracked/final/capture_002_003_search_policy_arbitration.json`
- broader-with-guards arbitration:
  - `/tmp/azlite_failure_family_diag/stability_runs/broader/final/capture_002_003_search_policy_arbitration.json`

## Observed Failure

Shared evidence:

- `capture_available-002`
  - reference move: `4`
  - searched selected move remains `2`
- `capture_available-003`
  - reference move: `1`
  - searched selected move remains `1`
- differing tactical rule feature:
  - `extra_turn_available`
  - row `002`: `false`
  - row `003`: `true`

Key implication:

- the failing guard row is not just a weaker version of the preserved guard row
- the extra-turn interaction appears to separate the two rows structurally

## Baseline Hypothesis

The finetuned policy/search stack over-generalizes a capture-family preference learned from one local motif, but the learned behavior does not condition sharply enough on the extra-turn rule interaction that distinguishes `capture_available-002` from `capture_available-003`.

Operationally:

- row `003` stays stable because the extra-turn-available branch aligns with the preserved policy/search preference
- row `002` fails because the no-extra-turn branch needs a different tactical preference ordering, but the finetuned artifact still amplifies the wrong move

## Concrete Questions

1. Does the wrong-move amplification on `capture_available-002` originate in prior policy, child Q values, or visit accumulation?
2. Which concrete rule-level features differ between `002` and `003` besides the `extra_turn_available` flag?
3. Is the reference move on `002` underrepresented because the model misses a no-extra-turn capture benefit, or because the competing move is being overvalued?
4. Are the guard rows in the broader artifact too weakly coupled to the tracked opening rows, causing the wrong local preference to survive replay?

## Required Outputs

Produce one diagnostic artifact that compares `capture_available-002` and `capture_available-003` at rule-feature level and one markdown note summarizing the implication.

Artifact requirements:

- include both rows in one payload
- include resolved raw states and legal moves
- include per-move teacher child stats for each row
- include searched visit shares and searched selected moves
- include explicit rule features for the reference move and the searched selected move
- include a side-by-side comparison of:
  - capture legality
  - extra-turn availability
  - store gain
  - landing pit / store landing
  - captured opposite pit
  - post-move empty-pit pattern

Markdown note requirements:

- state whether the collision is best explained by:
  - no-extra-turn capture undervaluation
  - extra-turn overvaluation
  - capture-shape aliasing
  - another rule-conditioned mechanism
- recommend whether the next intervention should be:
  - a rule-conditioned policy artifact redesign
  - a row-pair diagnostic replay artifact
  - a search-selection instrumentation pass

## Guardrails

- do not launch another training lane yet
- do not change architecture
- do not change broad self-play/search/value/temperature settings
- do not promote any model
- keep `storage/ai/alphazero_lite/current` explicit in any artifact-evaluation command

## Exit Criteria

This branch is complete when:

1. the row-pair diagnostic clearly explains why `002` fails while `003` stays stable, or
2. the report explicitly states why the current evidence is still insufficient and what missing telemetry is required next.
