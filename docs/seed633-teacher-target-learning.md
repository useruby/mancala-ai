# Seed 633 — teacher-target learning and transfer diagnostic

**Decision: `training_local_improvement_only`.** This is a descriptive policy-
teacher alignment analysis, not a playing-strength result or a causal estimate
of opening diversity.

## Protocol and authentication

The initial plan used historical PyTorch training checkpoints directly. Before
inference, this diagnostic was amended to use Seed 615's already-published
NumPy evaluation exports and `arena.ArtifactEvaluator`. B512 and S615 were run
on the same frozen state rows. P-E4 was omitted: this comparison does not need a
parent-policy baseline, and the inherited parent-training lineage remains
unresolved. Original teacher targets were used unchanged. Inference was
CPU-only; there were no optimizer updates, MCTS, solver calls, new games, or
training.

Seed 627's snapshot files are tracked on `main` (not omitted by sparse checkout).
Their file bytes match the committed snapshot manifest and their rows were
independently reconstructed from the authenticated membership records and
training traces:

| Artifact | SHA-256 | Rows |
| --- | --- | ---: |
| Seed 627 manifest | `e7a6af69f72780ca5ca79f9fe8214b9d347b6c6c0fb7cfea87b0617e3a16d3f3` | — |
| B512 snapshot | `de7650a5d4f0d7a9dec11adb3d100689381a4b1539ec0c9749a0f82feda28d59` | 2,048 |
| S615 snapshot | `8ed6f059d491fa4a6bfd90c942d6e02d05ef1b6a82db65969402ad4351a084fd` | 2,048 |
| Protected-holdout snapshot | `4716976a654ecba4aadccfa336d0dc3c5fdbd354ac27786848a06485d6062610` | 1,353 |

The Seed 599 B512, Seed 615 training-result/freeze, Seed 615 opening-diversity,
and Seed 622 provenance verifiers passed. The Seed 615 evaluation result
verifier and the #619 post-hoc diagnostic authentication passed in the original
Seed 615 evidence worktree. The #619 check authenticated all 4,096 existing
evaluation records; it generated no games. The older Seed 627 bundle verifier
expects its recorded PyTorch 2.13 runtime and therefore does not pass its
runtime-equality check under the Seed 615 evaluation runtime. This diagnostic
instead independently reconstructed the snapshot contents and checked their
committed byte hashes; the new Seed 633 verifier repeats that reconstruction.

The active Seed 615 evaluation freeze is
`ba30f5e3efadf3358db9f03c7b4eba4f69b23e2f95dcb4089723a1fb9664f32b`. Its
registered evaluation runtime is CPython 3.14.7, NumPy 2.5.3, CPU-only, with
one-thread environment settings. PyTorch 2.14.0+cu130 was inspected for runtime
identity and thread count only; the model forward passes used NumPy.
`ml/alphazero_lite/arena.py` is byte-identical to the freeze-bound source
`c16a668e8f90bd87b59dbc08f6a1ddb3b5ffc912e21cc0564efaceba37e03fe2`.

| Model | Checkpoint / `model.npz` SHA-256 | `weights.json` SHA-256 | `metadata.json` SHA-256 |
| --- | --- | --- | --- |
| B512 | `bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c` | `d3effd234fe657fd8be3e3896153441d75ba338e83e83b66d7541936b3c01ccd` | `510d4d5056d1b68ddd83eed89a4e8da13e9305d2fa39a46eefcebcf13334dce6` |
| S615 | `cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780` | `8f36140461390d317e7e5bcf604357ca65fa1b605bd1e9562bf923b840e13433` | `24e42998824856d0242e04ff8a98431b139375e0bc4540d759373ccb335983b3` |

These hashes are bound by the evaluation freeze and its independent verifier.
The run freeze also records the full input SHA-256 map, evaluation source map,
runtime, metric definitions, and cohort digest. No converted checkpoint was
used.

## Frozen cohort

The selection is all unique authenticated Seed 627 states, sorted by canonical
identity, with no performance-conditioned filtering. Canonical-state and
production-float32 identities were independently reconstructed. Duplicate
training identities had matching targets, legal actions, source-opening labels,
phase, and teacher evidence. The four membership classes are:

| Membership | States | Source-opening groups |
| --- | ---: | ---: |
| Shared B512/S615 training membership | 928 | 150 |
| B512-only membership | 1,120 | 146 |
| S615-only membership | 1,120 | 150 |
| Outside both (Seed 599 protected holdout) | 1,353 | 38 |

The total is 4,521 states in 188 source-opening groups. The held-out groups are
disjoint from the B512/S615 source-opening groups. This is a verified Seed 599
source split, **not** proof of complete independence from P-E4: inherited
P-E4 ancestry is unresolved, and Seed 624 `game_id`/`opening_identity` cannot be
fully crosswalked to these historical source identifiers. Identity
non-overlap is not treated as a complete training-exposure proof.

Targets comprise 3,040 `artifact_puct` records and 1,481
`exact_root_tablebase` records. The former are treated only as recorded search-
teacher targets. Exact-optimal action mass/agreement is emitted only for records
with authenticated exact action margins and exact-optimal-action sets; tied
optimal actions are retained as a set.

## Results

All model comparisons below are paired on identical rows. Negative S615-minus-
B512 cross-entropy or KL means S615 is closer to the recorded teacher target.
The overall source-opening-cluster bootstrap uses 20,000 replicates, RNG seed
633, and linear 2.5%/97.5% quantiles. These are descriptive stability
intervals, not independent-population inference.

| Cohort | Rows | B512 mean CE (nats) | S615 mean CE (nats) | S615 − B512 CE | 95% source-opening bootstrap interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| All states | 4,521 | 0.649684 | 0.648825 | −0.000859 | [−0.003838, 0.001771] |
| Shared training states | 928 | 0.732811 | 0.729264 | −0.003547 | [−0.007991, 0.000438] |
| B512-only states | 1,120 | 0.595073 | 0.619839 | +0.024765 | [0.020266, 0.029195] |
| S615-only states | 1,120 | 0.619431 | 0.595231 | −0.024200 | [−0.028714, −0.019913] |
| Outside both / protected holdout | 1,353 | 0.662917 | 0.662011 | −0.000905 | [−0.008516, 0.004929] |

The complementary exclusive-membership effects are the clearest pattern: S615
aligns better on its own exclusive states, while B512 aligns better on B512's
exclusive states. The held-out estimate is near zero, and its descriptive
interval includes zero. The pooled mean difference is also near zero with an
interval including zero.

| Stratification | Rows | S615 − B512 CE (nats) | Interpretation |
| --- | ---: | ---: | --- |
| Phase `above_32` | 1,029 | +0.002033 | Slightly worse S615 alignment |
| Phase `17_to_32` | 2,011 | −0.001802 | Slightly better S615 alignment |
| Phase `at_most_16` | 1,481 | −0.001589 | Slightly better S615 alignment |
| `artifact_puct` teacher | 3,040 | −0.000504 | Very small improvement on search-teacher targets |
| `exact_root_tablebase` teacher | 1,481 | −0.001589 | Small CE improvement; exact-optimal action agreement 0.88049 vs 0.88184, optimal-action mass 0.84159 vs 0.83971 (B512 vs S615) |

Other all-state means: teacher top-action agreement was 0.77616 for B512 and
0.77969 for S615; legal-action mass before masking was 0.63247 and 0.63283;
legal-policy entropy was 0.67125 and 0.66982 nats; model confidence was 0.72415
and 0.72504. Probability-mass summaries can differ from one by tiny float32
roundoff when all six actions are legal. The full per-state data preserves
unrounded values. Teacher-to-policy KL has the same paired delta as
cross-entropy because each row uses the same teacher target entropy for both
models.

The descriptive entropy strata are recorded in the summary. One-hot exact-root
targets create a large atom at zero target entropy, so pooled quantile cuts can
leave an entropy bin empty; bins are not treated as inferential subgroups.

## Answers and research decision

1. **Does S615 align better overall?** Its mean CE is lower by 0.000859 nats on
   the full paired cohort, but the cluster interval includes zero. This is not
   a material pooled improvement.
2. **Are improvements concentrated in S615 membership?** Yes. S615 is better
   by 0.024200 nats on S615-only states, while it is worse by 0.024765 nats on
   B512-only states. Shared states show only a small, interval-overlapping
   change.
3. **Is there held-out source-group improvement?** No clear evidence. The
   protected-holdout estimate is −0.000905 nats with interval
   [−0.008516, 0.004929]. Even a directional result would remain descriptive
   because full P-E4 exposure is unresolved.
4. **Do phase or teacher types show worse alignment?** S615 is slightly worse
   in the `above_32` phase and slightly better in the other two phases. Mean CE
   changes are small for both teacher types; artifact-PUCT targets are not
   called exact-optimal. Some move-count/entropy strata vary in direction, and
   they are descriptive only.
5. **Next experiment:** investigate the reciprocal training-membership pattern
   before increasing source-opening diversity again. A focused follow-up should
   preserve the parent, total updates, state count, and teacher-target
   composition while varying a predeclared opening-group sampling/allocation
   rule, with source groups independently cleared against parent ancestry.

**Final classification: `training_local_improvement_only`.** The observed
benefit is concentrated on S615's exclusive membership and is offset by poorer
alignment on B512-only states; the study finds no clear out-of-membership
improvement. This does not establish a causal effect of diversity, training
independence, or playing strength. Membership, source allocation, and target
composition changed together.

## Evidence and verification

Successful, final-source frozen run:
`docs/data/seed633-teacher-target-learning/attempt-0004/`.

| Evidence | SHA-256 |
| --- | --- |
| Freeze | `72e847696338777dd278ee676471033e11dd2345869f4f7b7cce46632bd74719` |
| Frozen cohort JSONL | `303e6c38feff9868133e84a80b92421e26da480dbea36dd3a37a8f70d33e914d` |
| Per-state metric JSONL | `ebba7af50e154ff6e31e3afce3ee325da915295ac11f8a10a73f7a516dd2cccb` |
| Machine summary | `3be38d7f930f3f5fd06e750f6bf02c0d2dcb9edb453e1e3a244cf9b8b5bf8634` |
| Independent verification evidence | `34aa59dea2547c81aba439aeaef2fb8e8dc56d307f2dbda686d2fb2b45aba639` |

The independent verifier reconstructs the cohort from the historical source
memberships and snapshots, verifies artifact and input digests, independently
recomputes policy probabilities and all per-state metrics in NumPy, then
rebuilds subgroup aggregates and bootstrap results. Its five synthetic tests
reject substituted model artifacts, changed teacher targets, invalid legal
masks, altered membership, and illegal target mass. Verification status is
`verified` for 4,521 rows; no playing-strength claim is made.

Attempt 1 is retained at the top-level data directory. It completed per-state
inference but failed during summary aggregation because boolean agreement arrays
were subtracted; it wrote no metric dataset or summary. Attempt 2 completed and
was independently verified, but the repository's required Ruff formatter later
changed source bytes bound by that freeze. Attempt 3 completed after formatting
and was independently verified, then the verifier was strengthened to
reconstruct teacher entropy and legal-action count per state. The intermediate
attempt outputs remain in the local workspace and are not included in the
published bundle. Attempt 4 was frozen after both changes, rerun, and
independently verified; it is authoritative. The published failure record is
`docs/data/seed633-teacher-target-learning/attempt-0001-failure.json`.

Reproduce the successful bundle in the pinned Seed 615 evaluation environment
(using a new create-only attempt directory):

```sh
PYTHONHASHSEED=615 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  python -m ml.alphazero_lite.seed633_teacher_target_diagnostic \
  --freeze --output docs/data/seed633-teacher-target-learning/attempt-0005
PYTHONHASHSEED=615 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  python -m ml.alphazero_lite.seed633_teacher_target_diagnostic \
  --run --output docs/data/seed633-teacher-target-learning/attempt-0005
PYTHONHASHSEED=615 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  python -m ml.alphazero_lite.verify_seed633_teacher_target_diagnostic \
  --output docs/data/seed633-teacher-target-learning/attempt-0005
```
