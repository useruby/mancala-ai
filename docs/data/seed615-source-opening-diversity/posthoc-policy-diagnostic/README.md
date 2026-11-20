# Seed 615 post-hoc policy-divergence retrospective

Read-only replay and deterministic inference on authenticated historical evidence. This is exploratory and does not alter the registered confirmatory result.

## Findings

- **Identical-position priors:** the registered-opening cohort has 58 top-action disagreements across 1024 states; mean TV is 0.033264 and mean Jensen–Shannon divergence is 0.001859 bits. The deduplicated first-divergence cohort has 209 disagreements across 676 states (mean TV 0.058269; JS 0.005045 bits).
- **Recorded trajectories:** 1333 of 2048 matched opening/seat game pairs were identical; 715 first diverged. Of those, 715 first differing turns were by the candidate and 0 by P-E4.
- **Tactical associations:** among first-divergence states, 283 had an immediate capture available, 251 had an extra turn available, and 178 were within the 16-stone exact-root region. For example, the capture-available stratum's recorded candidate scores were B512 0.507 and S615 0.500; this descriptive selection is not an action-quality or causal comparison.
- **Opening outcomes:** S615 was better on 76 openings, worse on 81, and equal on 867. The frozen aggregate remains B512 0.502197265625, S615 0.496337890625 (difference -0.005859375); paired 95% interval [-0.01611328125, 0.00439453125]; S615 lower bound vs P-E4 0.486572265625; registered interpretation **neither**.
- **Mechanism:** tactical/phase strata and actor-specific outcome summaries are descriptive associations. They do not identify causal effects or establish action quality; available evidence is insufficient to identify a specific mechanism.

Evidence bindings: freeze `ba30f5e3efadf3358db9f03c7b4eba4f69b23e2f95dcb4089723a1fb9664f32b`; B512 `bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c`; S615 `cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780`; P-E4 `2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad`.

Recorded outcome labels `challenger`, `current`, and `draw` are from each game's candidate perspective. `diagnostic.json` contains the full authenticated source-hash map and hashes of all JSONL outputs; each JSONL row carries the same source/checkpoint provenance.

The recorded games were independently verified before reconstruction. Matched games share opening and candidate seat, but their numeric seeds differ because candidate identity is part of the seed contract. First action divergence is therefore not uniquely attributable to the model intervention.

The training intervention increased effective source-opening-group diversity while also changing actual training-state membership. This evaluation therefore compares a changed sampling/allocation strategy; it does not isolate a single diversity statistic.

Policy-prior metrics describe the original network policy distributions on identical states. They do not describe MCTS-selected actions, counterfactual outcomes, or action quality. Tactical indicators use ordinary Kalah consequences only; no new exact solve was run. Post-hoc associations are descriptive, not significance tests or causal evidence. No candidate is selected or promoted.
