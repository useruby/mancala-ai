# AlphaZero-lite Capture 002/003 Rule-Collision Note

## Outcome

The row-pair diagnostic points to `extra_turn_overvaluation`, not no-extra-turn capture undervaluation.

Artifact:

- `/tmp/azlite_failure_family_diag/capture_002_003_rule_collision_diagnostic.json`

## Why

- `capture_available-002` reference move `4` is a no-extra-turn move.
- both finetuned variants still search-select move `2` on row `002`.
- move `2` is an extra-turn move in both variants.
- `capture_available-003` stays stable because its reference move `1` is already an extra-turn move.
- on row `002`, the broader-with-guards run still collapses visits onto move `2` even though child Q favors the reference move `4`.

This means the failure is not best explained by the model missing some hidden capture reward on row `002`.

Instead, the failure is that the finetuned policy/search stack keeps over-amplifying the extra-turn branch when the correct move on that row is the no-extra-turn branch.

## Rule-Level Comparison

Reference-move split:

- row `002` reference move `4`: `extra_turn_available=false`, `store_gain=1`, `landing_pit=2`
- row `003` reference move `1`: `extra_turn_available=true`, `store_gain=1`, `lands_in_store=true`

Wrong-move split on row `002`:

- reference move `4`: no extra turn, leaves post-move attacking empties `[3,4]`
- selected move `2`: extra turn, leaves post-move attacking empties `[2]`

Shared non-explanation:

- none of the compared moves are capture-legal on these rows
- store gain is `1` across the compared moves

That leaves the extra-turn interaction as the cleanest structural difference that matches the observed divergence.

## Recommendation

Next intervention: `rule_conditioned_policy_artifact_redesign`

Reason:

- a row-pair replay artifact would still be training against the same unresolved rule collision
- the current evidence already isolates the conflict more clearly than a generic search-instrumentation pass
- the redesign should separate opening-family targets by extra-turn polarity so row `002` is not trained under the same local preference shape as row `003`

Practical implication:

- do not launch another broad training lane yet
- first redesign the family artifact so no-extra-turn and extra-turn opening motifs are explicitly disentangled
