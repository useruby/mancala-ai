# Seed461 lower-learning-rate sensitivity

**Decision:** reject_lr_0.0005_retain_lr_0.001. A paired order regressed by more than 0.05 and between-order spread increased

Primary mean B−A opening-score difference: **0.0635** (95% shared-opening cluster bootstrap interval 0.0459 to 0.0811).

| Order seed | A score (LR 0.001) | B score (LR 0.0005) | B−A | A/B selected epoch |
|---:|---:|---:|---:|:---:|
| 38411 | 0.4814 | 0.6016 | +0.1201 | E4/E2 |
| 38412 | 0.5996 | 0.7598 | +0.1602 | E1/E4 |
| 38413 | 0.4980 | 0.6016 | +0.1035 | E4/E4 |
| 38414 | 0.5410 | 0.4551 | -0.0859 | E2/E4 |
| 38415 | 0.5410 | 0.5605 | +0.0195 | E3/E2 |

Between-order score range: A **0.1182**, B **0.3047**. Stability improvement: no.

Validation total-loss trajectories (epochs 1–4):

| Order | Arm | Selected | Validation total loss by epoch |
|---:|:---:|:---:|:---|
| 38411 | A | E4 | 1.0251, 1.0278, 1.0178, 1.0148 |
| 38411 | B | E2 | 1.0106, 1.0080, 1.0138, 1.0082 |
| 38412 | A | E1 | 1.0206, 1.0308, 1.0385, 1.0237 |
| 38412 | B | E4 | 1.0121, 1.0097, 1.0114, 1.0079 |
| 38413 | A | E4 | 1.0417, 1.0227, 1.0244, 1.0123 |
| 38413 | B | E4 | 1.0133, 1.0070, 1.0126, 1.0057 |
| 38414 | A | E2 | 1.0202, 1.0191, 1.0343, 1.0293 |
| 38414 | B | E4 | 1.0080, 1.0104, 1.0127, 1.0054 |
| 38415 | A | E3 | 1.0239, 1.0272, 1.0227, 1.0243 |
| 38415 | B | E2 | 1.0133, 1.0068, 1.0069, 1.0132 |

Inference is conditional on these five order seeds and this frozen dataset. A favorable result would require another dataset/generation replication before production adoption.
Detailed hash-bound provenance and per-game file hashes are in `seed461-lr-sensitivity-results.json`; raw artifacts are under `.tmp/seed461-lr-sensitivity/`.
