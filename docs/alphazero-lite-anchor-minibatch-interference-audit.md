# Anchor Minibatch Interference Audit

Inherited classification: `anchor_forgetting_replay_optimizer_interaction`.

## Baseline Reproduction

| cell | delta | reference/trace SHA identical | top action correct |
| --- | ---: | --- | --- |
| R61-T61 | -0.2289 | True | False |
| R61-T63 | 0.1592 | True | True |
| R62-T61 | 0.0837 | True | True |
| R62-T63 | -0.2500 | True | False |

## Step Concentration

| cell | harmful | protective | worst-1 | worst-5 | worst-10 | mechanism |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| R61-T61 | 184 | 181 | 3.04% | 11.07% | 17.54% | diffuse |
| R61-T63 | 194 | 192 | 2.87% | 9.69% | 16.17% | diffuse |
| R62-T61 | 182 | 183 | 3.43% | 11.07% | 17.89% | diffuse |
| R62-T63 | 181 | 177 | 3.48% | 12.50% | 20.94% | diffuse |

## Content And Order

- R61-T61: harmful structural-neighbor enrichment versus neutral `0.645889369397345`; protective `0.8701439228931236`.
- R61-T63: harmful structural-neighbor enrichment versus neutral `0.8990110258992434`; protective `0.8882032082994779`.
- R62-T61: harmful structural-neighbor enrichment versus neutral `1.249045465225564`; protective `0.9967760946460184`.
- R62-T63: harmful structural-neighbor enrichment versus neutral `1.1929823180391423`; protective `1.105649038703432`.
- R61 order comparison: first material divergence at step `1`, effect-sign-flip canonical rows `40334`.
- R62 order comparison: first material divergence at step `2`, effect-sign-flip canonical rows `40237`.

## Gradient And Safety

Policy-head and final shared-layer anchor-gradient cosines are recorded before every applied update in the machine-readable trace. Probe gradients use `torch.autograd.grad`, leave `.grad` and Adam state untouched, and reference/trace checkpoint bytes matched for every cell.

Frozen-set evaluations are recorded for each final checkpoint. No local order-swap was run because no small high-impact block met the spike criterion.

## Classification

`anchor_interference_diffuse_optimization`

Exactly one next experiment: test one frozen-replay optimizer-stability intervention (reduced LR) under R61 only; do not edit replay.
