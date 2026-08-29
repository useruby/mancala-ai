# PR #252 Phase Target-Delta Attribution

**Classification:** `target_delta_semantics_align_without_phase_sufficiency`

This sealed experiment used only the four frozen PR #250 policy replays. It
did not generate self-play, run target MCTS, alter target budgets, beta,
optimizer, learning rate, architecture, or evaluation PUCT, and did not
promote a model. Suites G/H/I are consumed.

## Invariants

| Seed | Positive replay SHA | Negative replay SHA | Gradient cosine | Relative L2 error |
| --- | --- | --- | ---: | ---: |
| 45 | `aed8a767...d48b5a5` | `7bc74369...85092f` | .999999999999 | 1.63e-6 |
| 46 | `007a72d5...1fbb4` | `83afa719...4baa9` | .999999999997 | 2.75e-6 |

The full positive and negative step-16 models reproduced their required
frozen hashes. Both replay pairs had identical non-policy rows and exclusion
plans; decoded `kalah_v3` states re-encoded exactly within float tolerance.

## Sealed Suites

| Suite | Seed | SHA-256 |
| --- | ---: | --- |
| G | 7042 | `ec331fc5672d0af95083443620ede6aac68755280ba375a9727bdd781918b216` |
| H | 8042 | `24e4ae8cb9ab336959f243c1dedd903201497d94e04fbc99ddaeed8894c58682` |
| I | 9042 | `5c8265a4c387c1a2f40ddc492da0435db548b98e02692582ccbf4b119f621b8e` |

## Primary 1200:1200

Both full positive controls replicated: `+.042969`, 95% CI
`[+.033854, +.052734]`, positive in 3/3 suites.

| Seed | Opening-positive minus negative | Postopening-positive minus negative |
| --- | --- | --- |
| 45 | `+.042969` CI `[+.033854, +.052734]`, 3/3 | `+.042969` CI `[+.033854, +.052734]`, 3/3 |
| 46 | `.000000` CI `[.000000, .000000]`, 0/3 | `.000000` CI `[.000000, .000000]`, 0/3 |

For seed 45, both hybrids were exactly equivalent to full positive. For seed
46, neither hybrid reproduced the full positive. Thus neither phase is
sufficient across both replay seeds. The secondary 384:256 result was zero
for every contrast and is descriptive only.

## Attribution

Top 1% / 5% / 10% of rows carried 25.62% / 61.26% / 77.63% of absolute
gradient attribution for seed 45, and 28.11% / 63.86% / 79.41% for seed 46.
Opening-early produced most positive attribution: 71.0% for seed 45 and
49.9% for seed 46. But post-opening rows held more absolute attribution:
67.1% and 69.5%, respectively.

Semantic signed-attribution alignment was strong: phase cosine `.9407`, pit
index `.9808`, extra-turn/capture `.9968`, immediate-score `.9973`, and legal
count `.9218`; all Spearman correlations were at least `.90`.

Positive target mass moved mostly toward ordinary moves rather than captures:
87.2% / 90.0% toward non-captures and 82.6% / 68.7% toward non-extra-turn
actions for seeds 45/46.

## Telemetry

Seed 45 hybrids reproduced the challenger-seat-1 ply-7 signature in all
G/H/I suites. Seed 46 hybrids had no trajectory divergence from negative.

## Interpretation

The two replay seeds do not support a common temporal sufficiency result at
the preregistered ply-10 split. Their action-semantic attribution is instead
strongly aligned despite their different phase-hybrid behavior.

Recommended next: causal action-category target surgery, without reusing
G/H/I for model selection.
