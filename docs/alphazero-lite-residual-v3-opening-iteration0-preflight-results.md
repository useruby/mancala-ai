# AlphaZero-Lite Residual_v3 Opening Iteration-0 Preflight

**Classification**: `residual_v3_opening_iteration0_preflight`

## Purpose

This PR adds the first safe, reproducible preflight for an opening-suite-selected multi-iteration residual_v3 lane.

It intentionally does not train, promote, or overwrite `model-artifact/current`.

It also explicitly keeps value-transform plumbing diagnostic-only after PR #142 classified runtime value transforms as `value_transform_semantically_dangerous`.

## Why These States Are Selected

The new runner evaluates canonical opening-suite positions with the current promoted search schedule only:

- `default_c_puct = 1.25`
- override `768:768 -> 0.90`
- `root_policy_mode = deterministic`
- `root_prior_transform = null`
- `tactical_root_bias = 0.0`
- `value_transform = null`

Selection is driven by the current model's own search behavior, not by transforms, promotion heuristics, or seed lottery.

States are retained when the current model is either:

- unstable: the selected move changes across fixed promoted-search settings, especially the `768:256` vs `768:768` schedule boundary
- weak: the root search remains low-confidence under the promoted schedule, measured by low top-visit share, low top-2 margin, or high entropy

This keeps iteration-0 focused on positions where search-time behavior is visibly brittle under the exact runtime schedule we now care about.

## What The Iteration-0 Candidate Will Train On

The preflight emits:

- `iteration0_selected_positions.jsonl`: deterministic selected opening-suite states plus search diagnostics per budget setting
- `iteration0_training_manifest.json`: reproducible manifest with suite hashes, current artifact hash, search profile hash, selection criteria, distributions, and dedup fingerprints

The next PR should train a residual_v3 candidate only from these selected states after generating the actual supervised targets for them. This PR stops one step earlier so the dataset boundary is frozen and reviewable before any training run happens.

## Guardrails

The runner enforces:

- residual_v3 only
- no tablebase overlay
- no value transform
- no root-prior transform
- no seed-lottery promotion
- no overwrite or promotion of `model-artifact/current`

## Next-PR Evaluation And Promotion Gates

The follow-up training PR should keep the same promoted deterministic search schedule and use fixed, non-lottery evaluation gates:

1. Train residual_v3 only from the preflight-selected iteration-0 dataset.
2. Evaluate against `model-artifact/current` on the deterministic opening-suite seat-aware benchmark.
3. Require improvement on the primary opening-sensitive budgets: `384:256` and `768:768`.
4. Require non-regression on higher-budget checks: `1200:1200` and `1200:256`.
5. Run the standard deterministic promotion gate only if those fixed gates pass.

No transform-assisted strength path should be used in that PR. Value-transform plumbing remains diagnostic infrastructure only.
