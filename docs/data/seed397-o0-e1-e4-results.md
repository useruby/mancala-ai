# Seed397 O0 E1/E4 paired comparison

## Registration and recovery

The suite was registered before games in `seed397-o0-e1-e4-registration.json`:
512 unique openings selected with seed 397, fully replayed under the arena
player-relative opening contract, at most eight plies, nonterminal, and with pit
sum above 32. `seed397-o0-e1-e4-exclusions.json` records the complete verified
#394/#395 exclusion union (including seeds 395 and 396), plus the original
seed461 replay inputs. The selected suite has zero overlap with that union.

The original execution checkout contained both O0 checkpoints and runtime
exports. `seed461-o0-checkpoint-recovery-provenance.json` records the inspected
paths, expected and actual identities, all E1–E4 checkpoint hashes, runtime
contract, original training-record hash, and Python/Torch/NumPy/platform
environment. No reproduction or replacement checkpoint selection was needed.
E1 and E4 checkpoint and model hashes equal their published bindings exactly.

## Evaluation

E1 and E4 each played 1,024 games against the frozen seed455 artifact on the
same 512 openings, with both seats represented once per opening. The arena
used 384 simulations per side, c_puct 1.25, base seed 397, the registered
`azlite_eval_seed_v2` seed contract, and the unchanged native runtime. No
extensions were run. The evaluation binding and raw per-game accounting are
published alongside the paired opening matrix and this report.

| Checkpoint | Wins | Draws | Losses | Games | Score | Seat 0 score | Seat 1 score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 | 458 | 103 | 463 | 1,024 | 0.4976 | 0.5488 | 0.4463 |
| E4 | 472 | 102 | 450 | 1,024 | 0.5107 | 0.5410 | 0.4805 |

## Preregistered analysis and classification

The 10,000-resample opening-cluster percentile bootstrap used seed 397. Each
opening's score averages its two seats, and the E1−E4 contrast uses the same
opening clusters for both checkpoints.

- **Useful early gain:** E1 score 0.4976, 95% interval [0.4756, 0.5190]. The
  registered requirements (score ≥ 0.55 and lower bound > 0.50) both fail.
- **Subsequent weakening:** paired mean E1−E4 = −0.0132, 95% interval
  [−0.0405, 0.0132]. The registered requirements (mean ≥ 0.03 and lower bound
  > 0) both fail.

The conclusions are separate: **no useful early gain; no evidence of
subsequent weakening**. No promotion or outcome-dependent extensions were
performed. `seed397-o0-e1-e4-results.json` contains the exact intervals and
full paired per-opening score vectors for reproduction.
