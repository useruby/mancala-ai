# PR #253 Semantic Receiver Target Surgery

**Classification:** `semantic_causality_is_seed_specific`

This sealed experiment consumed the exact aligned PR #252 replay pairs and
only J/K/L for primary classification. It did not generate self-play, run
target MCTS, change target budgets, beta, optimizer, learning rate, model, or
evaluation PUCT, and did not promote a model. J/K/L are consumed.

## Invariants

The frozen replay hashes, non-policy row alignment, and PR #252 exclusions
reproduced exactly. The seed45/seed46 full positive and negative step-16
models reproduced their required historical hashes. All hybrid target rows
were normalized without post-hoc renormalization, nonnegative, zero on illegal
actions, and satisfied `ordinary + tactical == positive + negative`.

| Seed | Ordinary/full norm | Tactical/full norm | Gradient decomposition relative L2 error |
| --- | ---: | ---: | ---: |
| 45 | 1.11354 | .71577 | 2.27e-6 |
| 46 | 1.06562 | .86706 | 1.83e-6 |

## Receiver Mass

| Seed | Ordinary | Tactical | Capture-only | Extra-turn-only | Capture + extra-turn |
| --- | ---: | ---: | ---: | ---: | ---: |
| 45 | 69.83% | 30.17% | 12.78% | 17.40% | 0.00% |
| 46 | 58.65% | 41.35% | 10.01% | 31.34% | 0.00% |

Ordinary receivers received the majority of positive target movement in both
replay pairs. Alpha is reported only for rows with nonzero moved mass.

| Seed | Mean | p10 | p25 | p50 | p75 | p90 | alpha = 0 | alpha = 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 45 | .7079 | .0000 | .2000 | 1.0000 | 1.0000 | 1.0000 | 15.63% | 59.17% |
| 46 | .5822 | .0000 | .0000 | 1.0000 | 1.0000 | 1.0000 | 30.73% | 52.79% |

## Sealed Suites

| Suite | Seed | SHA-256 |
| --- | ---: | --- |
| J | 10042 | `7c0b097e18949a2d0ac5657116c565f3b9c0bb942ae69c9741dcd328915bf98b` |
| K | 11042 | `64725ca85c6657eeefbbc329bc8a03f762bbbb7b615ff4fefeb4356fa158e1fd` |
| L | 12042 | `147ff9d503975641da2ad366e1df93aa07a296190f850ed0514ecbf903b43ac0` |

The suite construction excluded canonical and A-I keys and prefixes, each
other, and both replay-state sets.

## Primary 1200:1200

Both positive controls replicated on all suites:

| Seed | Positive minus negative |
| --- | --- |
| 45 | `+.044922`, 95% CI `[+.035807, +.054688]`, 3/3 |
| 46 | `+.044922`, 95% CI `[+.035807, +.054688]`, 3/3 |

| Seed | Ordinary-positive minus negative | Tactical-positive minus negative |
| --- | --- | --- |
| 45 | `+.044922`, CI `[+.035807, +.054688]`, 3/3 | `+.044922`, CI `[+.035807, +.054688]`, 3/3 |
| 46 | `.000000`, CI `[.000000, .000000]`, 0/3 | `.000000`, CI `[.000000, .000000]`, 0/3 |

For seed45, both semantic lanes were exactly equal to full positive. For
seed46, both semantic lanes were exactly equal to negative and were
non-equivalent to positive (`-.044922`, CI `[-.054688, -.035807]`). Neither
semantic destination category is sufficient across both independent replay
seeds.

## Telemetry

Seed45 semantic lanes had the established challenger-seat-1, ply-7 first
divergence phenotype. Seed46 had no trajectory divergence from negative for
either semantic lane. The secondary 384:256 result was zero for all contrasts
and is descriptive only.

## Interpretation

Although ordinary actions receive most positive target mass and the semantic
gradient decompositions are valid, their strength effect is not common across
the two replays. Aggregate action semantics therefore do not establish a
universal causal target category.

Recommended next: add a third independent replay seed before deriving a
semantic training rule. No model was promoted.
