# Seed504 fixed four-epoch learning diagnostic

This is development evidence from the archived seed500 corpus. It makes no
historical-unseen, fresh-confirmation, promotion, or strength-test claim. E4 is
the sole candidate; the screen is exploratory and passing would authorize only
a proposal for a later strength test.

## Frozen inputs and membership

- Training: shards 503000–503005, unit source weights, all original rows and
  multiplicities, production `default` policy/value targets.
- Development: shards 503006–503007; removed a row if either its first 15
  production float32 features (canonical board) or complete 27-feature
  production float32 input matched any training row.
- Training: 13,724 rows / 9,281 unique identities. Original development:
  4,557 rows / 3,654 identities. Filtered development: 2,671 rows / 2,615
  identities; 1,886 rows excluded. Remaining canonical and full-input overlap
  are both zero.
- Filtered development active stones: >32: 548, 17–32: 992, ≤16: 1,131.
  Training active stones: >32: 4,705, 17–32: 4,432, ≤16: 4,587.
- Float32 accounting matches PR #503 exactly: 2,671 filtered rows and 2,615
  unique filtered identities. No accounting difference was found.
- `.tmp/seed500-current-checkpoint.npz` was hashed directly before execution;
  SHA-256 `7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c`.

`split.json` contains ordered per-shard source-line membership for all training,
original development, and filtered development rows, subgroup counts, and the
membership digest. `manifest.json` was written prospectively before training;
it binds shard hashes, initializer, split, deterministic permutations, runner,
production trainer, protocol, and runtime versions.

## Protocol and outcomes

Production `train.train` and `train.load_jsonl_replay` were used with
`residual_v3/kalah_v3`, four epochs, batch size 512, Adam (lr 0.0001, betas
(0.9, 0.999), epsilon 1e-8, weight decay 0), constant learning rate, seed
504, policy CE + 0.5 × Huber (delta 1), exact-root policy coefficient 1, and
gradient norm clipping 1.0. Each epoch processed 13,724 examples in 27 updates.
There were 108 updates total; every epoch checkpoint is retained. E4 is the
sole candidate, with no development-based selection or tuning.

| Epoch | Policy CE | Value Huber | Total objective | Updates |
|---|---:|---:|---:|---:|
| E1 | 0.72575221 | 0.18029188 | 0.81589814 | 27 |
| E2 | 0.68265505 | 0.16499831 | 0.76515420 | 27 |
| E3 | 0.66417453 | 0.16115804 | 0.74475356 | 27 |
| E4 | 0.65134573 | 0.15890623 | 0.73079885 | 27 |

### Development metrics

Each cell is `policy CE / actual-value MSE`. Exposure weighting averages rows;
equal-exact-input weighting averages within each 27-float32 input first, then
weights unique inputs equally. Distinct targets for repeated inputs are retained
when averaging.

| Checkpoint | Cohort | Exposure | Equal exact input |
|---|---|---:|---:|
| Initializer | pooled | 0.753460 / 0.389799 | 0.755281 / 0.393617 |
| Initializer | >32 | 0.948154 / 0.782262 | 0.954791 / 0.793685 |
| Initializer | 17–32 | 0.822509 / 0.469851 | 0.825354 / 0.478463 |
| Initializer | ≤16 | 0.598562 / 0.129426 | 0.600207 / 0.130856 |
| E1 | pooled | 0.682182 / 0.326608 | 0.683157 / 0.331145 |
| E1 | >32 | 0.917978 / 0.560889 | 0.921633 / 0.571449 |
| E1 | 17–32 | 0.737893 / 0.407239 | 0.741862 / 0.415957 |
| E1 | ≤16 | 0.519069 / 0.142371 | 0.519489 / 0.144008 |
| E2 | pooled | 0.669297 / 0.321874 | 0.669560 / 0.326737 |
| E2 | >32 | 0.906500 / 0.530799 | 0.909733 / 0.541261 |
| E2 | 17–32 | 0.725924 / 0.405696 | 0.729990 / 0.414985 |
| E2 | ≤16 | 0.504698 / 0.147124 | 0.503595 / 0.148823 |
| E3 | pooled | 0.658331 / 0.315745 | 0.658808 / 0.320513 |
| E3 | >32 | 0.899251 / 0.525875 | 0.903387 / 0.536154 |
| E3 | 17–32 | 0.714934 / 0.400519 | 0.719189 / 0.409785 |
| E3 | ≤16 | 0.491952 / 0.139576 | 0.490801 / 0.141185 |
| E4 | pooled | 0.655992 / 0.312565 | 0.656697 / 0.317272 |
| E4 | >32 | 0.901285 / 0.524258 | 0.906067 / 0.534429 |
| E4 | 17–32 | 0.712685 / 0.398886 | 0.717427 / 0.408156 |
| E4 | ≤16 | 0.487415 / 0.134283 | 0.486120 / 0.135831 |

Checkpoint SHA-256: E1 `6187b0e702a2118579c82fc0950398babe255e95f1f64a4f695d52b3e56144c3`,
E2 `d1ef62ad445ad33a532a9a5efa8aadefe5657235af1706fdba4b1c1c8579d796`, E3
`e049d2a7e6702b68cf4bf2d230f5966f86e4dfc8e4ed928a534b61f7d7804842`, E4
`2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad`.
The ordered epoch permutation file SHA-256 is
`1115b156abd646fe3ba4483c18465b9323c926427a7305a026e95309982589eb`.

E4 passes the stated exploratory screen: >32 CE improves versus the initializer
by 0.046870 exposure-weighted and 0.048724 equal-input-weighted (both exceed
0.005). Pooled actual-value MSE decreases by 0.077233 exposure-weighted and
0.076345 equal-input-weighted (both within the +0.01 limit). This is development
evidence only and does not promote the checkpoint.

## Reproduction and files

Run from the repository root:

```bash
.venv-azlite/bin/python -m pytest ml/alphazero_lite/test_run_seed504.py -q
.venv-azlite/bin/python -m ml.alphazero_lite.verify_seed504
```

- `manifest.json`: prospective protocol and input/source identities.
- `split.json`: full row membership and overlap/accounting evidence.
- `permutations.json`: four recorded deterministic minibatch orders.
- `epochs.json`: per-epoch objectives, examples, gradient norms, and updates.
- `seed504_epoch1.npz` … `seed504_epoch4.npz`: saved production checkpoints.
- `results.json`: initializer/E1–E4 checkpoint hashes and predictions on every
  identical filtered development row, along with metrics. The evaluator applies
  production legality masks and computes actual-value MSE against original
  targets.
