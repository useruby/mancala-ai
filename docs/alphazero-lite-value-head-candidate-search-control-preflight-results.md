# AlphaZero-Lite Value-Head Candidate Search-Control Preflight Results

## Artifact Hashes

- current artifact hash: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- value_head_only_e2 artifact hash: `69f56d96e86970a4aca16c23f4215bde04db199baf85f7e58271dc191de1ef8f`

## PR #155 Reproduction Table

- reproduction JSON: `/tmp/azlite_value_head_search_control_preflight/pr155_reproduction.json`

## Lane Definitions And Effective Runtime Profiles

- challenger-only profiles and per-budget c_puct values are in `probes.*.runtime_profile`.

## Search-Control Support Matrix

- `normalize_values`, value-trust schedules, and per-side c_puct are supported; no delta clipping lane was added.

## Search-Aware Probe Table By Budget

- `probes.*.search.by_budget` records change rate, visit KL, and visit-share deltas.

## 768:768 Changed-Case Diagnostic Table

- `probes.*.search.by_budget.768:768` records changed-row |dV|, |dScore|, and teacher rates.

## Medium DS Table

- `suites.medium` uses candidate-minus-current DS orientation.

## Fixed-Large DS Table

- `suites.fixed_large` uses candidate-minus-current DS orientation.

## Held-Out Mean/Worst-Suite Table

- `heldout` contains mean and individual-suite deltas when fixed-large gates pass.

## Bootstrap CIs

- orientation: `candidate_minus_current`.

## P0/P1 Split For 384:256

- `probes.*.search.p0_p1_split_384_256` records the selected-move change split.

## Duplicate Trajectory Count

- per-budget counts are retained in each suite result.

## Runtime Cost

- elapsed seconds: `5412.11`

## Gate Result

- gate run: `False` (this runner does not promote).

## Final Classification

- result: `equal_budget_value_sensitivity_uncontrolled`

- complete machine-readable metrics: `/tmp/azlite_value_head_search_control_preflight/summary_metrics.json`
