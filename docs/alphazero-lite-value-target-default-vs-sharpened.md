# Default Versus Sharpened Value Targets

Classification: `default_value_targets_improve_calibration_and_search`.

This preregistered, non-promoting experiment compares the six PR #351
`fresh_w1` sharpened-target models with six treatments that relabel only the
corresponding fresh self-play rows to canonical final outcomes. Historical
replay remains byte-pinned and unchanged at weights `1,4,1,8,4`.

The runner validates all source, parent, fixed-replay, baseline, and arena
hashes before writing a deterministic derivative. It fails with
`value_target_relabel_integrity_failed` if any field other than `value` and
`value_target_mode` changes, and with
`value_target_source_provenance_unavailable` if fewer than five registered
sources validate. Execution additionally requires all six paired sources.

Each treatment started from seed48, used seed 443 and exactly 1084 optimiser
updates, then ran the exact calibration, one-ply ranking, raw-policy,
MCTS-384, diagnostic arena, and superhuman-regression evaluations. No
self-play, canonical promotion holdout, selection, or promotion was run.

## Results

All six source hashes validated. Every derivative passed the field-level
integrity audit; only `value` and `value_target_mode` changed. The resulting
default labels are approximately 92% unit magnitude, compared with roughly
47% mean absolute sharpened target magnitude for source 401. Per-source target
diagnostics, SHA256s, row counts, and field audits are in
`relabel_manifest.json`.

| Source | Sharpened value MAE | Default value MAE | Delta | Sharpened MCTS mass | Default MCTS mass | Delta | Arena delta | Regressions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 401 | 0.4248 | 0.2529 | -0.1719 | 0.7204 | 0.7252 | +0.0048 | +0.0781 | pass |
| 407 | 0.4119 | 0.2680 | -0.1439 | 0.6848 | 0.6929 | +0.0081 | +0.2324 | pass |
| 413 | 0.4282 | 0.2694 | -0.1588 | 0.6968 | 0.7002 | +0.0034 | +0.1621 | pass |
| 419 | 0.4161 | 0.2647 | -0.1514 | 0.6994 | 0.7065 | +0.0070 | -0.1328 | pass |
| 443 | 0.3914 | 0.2212 | -0.1703 | 0.6890 | 0.7016 | +0.0126 | -0.0723 | pass |
| 449 | 0.4359 | 0.2911 | -0.1449 | 0.6889 | 0.6921 | +0.0032 | +0.2617 | pass |

| Metric | Sharpened | Default | Paired delta | 95% CI |
| --- | ---: | ---: | ---: | --- |
| Root MAE | 0.4181 | 0.2612 | -0.1568 | [-0.1660, -0.1483] |
| Sign accuracy | 0.8725 | 0.8767 | +0.0042 | [-0.0042, +0.0133] |
| One-ply ranking regret | 1.9367 | 1.7450 | -0.1917 | [-0.4283, +0.0367] |
| Raw optimal mass | 0.6232 | 0.6247 | +0.0015 | [-0.0008, +0.0039] |
| Raw expected regret | 2.3005 | 2.2892 | -0.0112 | [-0.0293, +0.0064] |
| MCTS optimal mass | 0.6965 | 0.7031 | +0.0065 | [+0.0042, +0.0094] |
| MCTS expected regret | 1.6385 | 1.5327 | -0.1058 | [-0.1175, -0.0918] |
| Arena score | 0.4814 | 0.5697 | +0.0882 | [-0.0283, +0.1999] |

The raw-policy effects are near zero, while value calibration and both
search-quality effects improve with paired confidence intervals excluding
zero. The diagnostic arena is positive overall (+0.0882), and every treatment
passed the current superhuman regression suite.

The preregistered next experiment is one new forward AlphaZero challenger from
seed48 using fresh 1600-game uniform1200 self-play with
`value_target_mode=default` and the otherwise unchanged production recipe.
This experiment itself produces no promotion candidate.

Artifacts are recorded in
`docs/data/alphazero-lite-value-target-default-vs-sharpened/`.
