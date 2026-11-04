# Seed553: fixed-state teacher-search budget pilot

## Scope and interpretation

This pilot compares denoised PUCT policy targets at 128 and 512 simulations on
the same 256 roots, using the complete frozen seed542 initializer through the
same `CheckpointEvaluator` path used to generate #549 control data. It measures
policy-target changes and compute cost only. It does not establish target
quality or playing strength, and its planning classification does not authorize
training automatically.

The cohort is selected deterministically from #549 control's training
partition—not an unseen evaluation cohort—with exactly 128 nonterminal roots
above 32 active pit stones and 128 roots with 17–32. Every root is unique under
both canonical-state and production-float32 identities. The cohort records the
source shard/game/move/row and row ID, state, legal relative actions, both
identities, and exact production-float32 encoded input. Re-encoding preserves
the input bytes. The production loader accepted every control training shard;
all 256 selected inputs were bitwise-equal to the loader's float32 tensors.

## Frozen paired protocol

Each root receives two independent searches, with a fresh tree and RNG for each
search and the same seed `553000 + cohort_index`. Budget order alternates by
cohort index. Both searches use `c_puct=1.25`, FPU zero, no Dirichlet noise,
subtree reuse, value normalization, or tactical bias. The existing denoised
teacher's remaining search options are retained. Legal root visits are converted
with the existing helper at temperature 1.0. This diagnostic target temperature
is explicitly distinct from the historical generation schedule (1.0 through
ply 9, then 0.1).

All roots exceed 16 active pit stones. The run enters ordinary `PUCT` directly;
exact-root dispatch and exact-leaf solving are disabled. The fail-closed PUCT
regression patches native exact-root construction and tablebase lookup methods
to raise. Actual PUCT computation passed without invoking them, including a
≤16-stone leaf evaluated by the ordinary evaluator. The historical tablebase
check is therefore outside this pilot's executed dependency closure.

The frozen protocol, exact dependencies, numerical runtime, input authentication,
seed schedule, and planning rule are bound in `registration.json`,
`execution-freeze.json`, and `input-authentication.json`. Full source code hashes
and the recorded Python 3.14.7 / NumPy 2.5.3 / PyTorch 2.14.0+cu130 environment
are included. The complete 37-package closure matches #549's recorded
generation environment. Searches ran under CPU quota 25%, CPU weight 1, CPU 31,
nice 19, and a 4 GiB memory limit.

### Bounded #549 authentication note

All six control training shard hashes, #549's registration, the
post-generation receipt and all ten products, training occurrence identities,
source manifest, initializer bytes, and selected source rows are authenticated.
The full historical #549 verifier **did not pass in this checkout**: it stopped
at `runtime_asset_changed:model-artifact/runtime/kalah_v1_21.kvtb`, expected
SHA-256 `f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7`,
observed `a82ca3d03265950a23aae8a1088fa86a93db9acea7bca68ea33303292eb0202c`.
The exact command and identities remain in
`docs/data/seed553-prerequisite-failure.json`; neither the historical asset nor
historical verifier was changed. This limitation is not hidden or represented as
a successful full #549 verification.

## Paired results

| Cohort | Roots | Highest-visit action differs | Median TV | Mean TV | Bitwise-identical targets |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 256 | 31 | 0.0908203 | 0.1310501 | 6 |
| >32 stones | 128 | 23 | 0.1201172 | 0.1635895 | 3 |
| 17–32 stones | 128 | 8 | 0.0683594 | 0.0985107 | 3 |

TV is half the L1 distance between the two temperature-1.0 legal-action target
distributions. Highest-visit ties resolve to the lowest legal relative action
index.

| Cohort | Budget | Mean target entropy (nats) | Mean support | Search seconds | Evaluator calls | Configured / actual root visits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 128 | 0.757040 | 4.46875 | 24.8539 | 31,666 | 32,768 / 32,768 |
| Overall | 512 | 0.634273 | 4.46875 | 121.7197 | 123,053 | 131,072 / 131,072 |
| >32 stones | 128 | 0.755147 | 4.71875 | 12.3990 | 15,879 | 16,384 / 16,384 |
| >32 stones | 512 | 0.613729 | 4.71875 | 66.4398 | 62,572 | 65,536 / 65,536 |
| 17–32 stones | 128 | 0.758933 | 4.21875 | 12.4549 | 15,787 | 16,384 / 16,384 |
| 17–32 stones | 512 | 0.654817 | 4.21875 | 55.2798 | 60,481 | 65,536 / 65,536 |

Measured search seconds cover PUCT search execution and exclude profiling or
tracing. The relabeling cost estimate multiplies the cohort's mean search time
by the #549 control training population of 13,818 rows: approximately 1,341.5
seconds at 128 simulations and 6,570.0 seconds at 512. These are estimates, not
full-corpus measurements.

**Classification: `propose_fixed_state_budget_training_ablation`.** The frozen
gate was at least 13/256 highest-visit actions differing **or** median TV at
least 0.05. The observed 31 disagreements and 0.0908 median TV meet both
conditions. This only proposes a future training ablation; it does not claim
512-simulation targets are better and does not automatically authorize training.

## Prior findings and limitations

- #548's 64-game-per-arm evaluator pilot measured generation correctness, cost,
  and data characteristics—not strength or target optimality. Its corrections
  clarified action-noise turnoff after ply 9 and per-game RNG derivation.
- #550 compares training on different source-evaluator populations, which
  changes states, targets, outcomes, and occurrence frequencies; it does not
  isolate policy-target quality.
- #551's reused-suite C-versus-T screen scored 0.48218 and classified
  `do_not_advance_source_evaluator_branch`. The seed461 continuation-overlap
  limitation remained unresolved; non-advancement does not establish universal
  source-evaluator inferiority.
- #552's reused-suite C108-versus-serving-P screen scored 0.48511 with a
  one-sided lower bound of 0.47485, below its strict 0.50 threshold. No
  1,200-simulation follow-up ran, and that result does not imply C-versus-T
  strength.

Seed553 keeps those findings separate from this same-state target-change
diagnostic. It generated no games, changed no replay targets, trained no model,
ran no arena, and changed no serving or deployment asset.

## Verification and receipts

The independent read-only verifier reconstructs the selected cohort directly
from #549 training shards, replays all 512 searches from the frozen initializer,
rebuilds targets and metrics, authenticates attempt and execution bindings, and
checks that evidence bytes and nanosecond mtimes are unchanged. It does not call
publisher analysis functions. Append-only verifier corrections preserve the
original registration, cohort, search rows, timings, and results. Correction 01
fixes aggregate reconciliation of the separate resource observation; correction
02 checks alternating budget-first order; correction 03 makes resume validate
each saved row semantically before skipping it; correction 04 validates the
ordered correction chain; correction 05 extends rebound-mutation resume tests to
assert unchanged bytes and mtimes for all output files.

Run with the recorded environment:

```bash
/home/alex/Mancala/ai/.tmp/seed542-continuation-prep/.tmp/seed542-recorded-runtime/bin/python \
  -m ml.alphazero_lite.verify_seed553_fixed_state_budget
```

The physical-copy verifier checks a copied evidence and source bundle from an
unrelated working directory while the original worktree path is hidden. The
latest receipt is `physical-copy-verification.json`. Previous passing receipts
and append-only supersession bindings are preserved with numbered suffixes.
