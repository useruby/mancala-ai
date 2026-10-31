# Seed542 432-update training-budget experiment

## Result

The 108-update control passed. The first 108 D update records matched the registered seed542 v2 evidence, including every compared update field and post-update parameter state; update 108 is bitwise identical to the registered checkpoint (`6200b04d58f65ee5fb62555c3b32faea4b34206b4c10dbefb22ae9e7ff7b2328`). Training continued using the same in-memory model and Adam optimizer through update 432. Update 432 is the sole candidate checkpoint; updates 108 and 216 are diagnostic checkpoints.

No arena evaluation, serving change, or advancement decision is part of this experiment. Lower development loss is descriptive evidence only; a strength claim requires a subsequent arena screen.

## Frozen schedule and training contract

`registration.json` and `schedule.json` were written before training and hash-bind the finalized runner, independent verifier, publication workflow, tests, and training dependencies. The registration preserves the initializer, corpus, architecture, legal-policy cross-entropy plus 0.5 Huber value loss, Adam settings, clipping, and numerical runtime. Only the budget changed.

| Phase | Updates | Exposures | Batch sizes |
| --- | ---: | ---: | --- |
| Registered seed542 D schedule | 1–108 | 54,896 | four repetitions of 26 × 512 and 1 × 412 |
| Finish unused tail of seed-542 permutation | 109–215 | 54,608 | 106 × 512 and 1 × 336 |
| Deterministic shuffled passes (seeds 543 then 544) | 216–432 | 111,104 | 217 × 512 |
| **Total** | **1–432** | **220,608** | |

The first 108 updates cover 54,896 distinct rows. Updates 109–215 consume the remaining 54,608 rows of that same seed-542 permutation exactly once, before the seed-543 pass begins. Pass seeds and row order are included in `schedule.json`. Adam moments were retained in the live optimizer across all 432 updates.

## Fixed development checkpoints

Development membership is unchanged (`development-membership.json` SHA-256 `feeed97c7666a69603620e4e59575fdf02aa206e9e570cde937aa360bd37505b`). Each cell is **policy CE / actual-value MSE**. `Exposure` weights each retained row equally; `Equal-input` averages duplicate production float32 inputs first, then weights each distinct input equally. Full precision, pooled results, row counts, predictions, and checkpoint hashes are in `development-metrics.json` and `development-predictions.jsonl`.

| Stone bucket | Checkpoint | Exposure | Equal-input |
| --- | ---: | ---: | ---: |
| >32 | 108 | 0.9669415 / 0.7894428 | 0.9637882 / 0.7809665 |
| >32 | 216 | 0.9640544 / 0.7825235 | 0.9616046 / 0.7733995 |
| >32 | 432 | 0.9651034 / 0.7738234 | 0.9624977 / 0.7641476 |
| 17–32 | 108 | 0.7465860 / 0.4174258 | 0.7475832 / 0.4164104 |
| 17–32 | 216 | 0.7396551 / 0.3998162 | 0.7408194 / 0.3990908 |
| 17–32 | 432 | 0.7339994 / 0.3931337 | 0.7351935 / 0.3923272 |
| ≤16 | 108 | 0.5079532 / 0.1503156 | 0.5094473 / 0.1515030 |
| ≤16 | 216 | 0.4857817 / 0.1451440 | 0.4866240 / 0.1468238 |
| ≤16 | 432 | 0.4770726 / 0.1417413 | 0.4779496 / 0.1438932 |
| Pooled | 108 | 0.6808275 / 0.3647482 | 0.6806251 / 0.3623839 |
| Pooled | 216 | 0.6680225 / 0.3542680 | 0.6676963 / 0.3521199 |
| Pooled | 432 | 0.6621740 / 0.3486581 | 0.6618320 / 0.3465976 |

## Checkpoints and verification

| Update | SHA-256 |
| ---: | --- |
| 108 | `6200b04d58f65ee5fb62555c3b32faea4b34206b4c10dbefb22ae9e7ff7b2328` |
| 216 | `0fe04f18735e67a50175f430fb63db5f54155fbfe8799e00de5188929d1233eb` |
| 432 | `4555cba73d2c908d1eca829836f361b226ccc1072be28affd816b2a240123163` |

Independent training replay and development reconciliation passed in the working tree and from a separate physical copy. The verifier independently replayed all 432 optimizer steps and matched the registered first-108 trace and endpoint checkpoint, all three saved checkpoints, and all development prediction/metric rows. The physical copy's evidence directory was byte-for-byte equal to this directory; checkpoint inodes differ, confirming separate physical files.

The pre-freeze tests passed (4 tests): exact schedule coverage and exposure accounting; synthetic sequential updates retain one Adam instance and moments; a changed registered control field is identified; and schedule tampering is rejected. The test command was also rerun after publication: `python -m pytest ml/alphazero_lite/test_seed542_training_budget_432.py -q` (4 passed). Ruff passed for the four experiment source files.

The independent verifier command is:

```bash
python -m ml.alphazero_lite.verify_seed542_training_budget_432 \
  --out docs/data/seed542-training-budget-432
```
