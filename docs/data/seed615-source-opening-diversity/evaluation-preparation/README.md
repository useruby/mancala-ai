# Seed 615 playing-strength evaluation preparation

This directory freezes the already registered comparison between immutable B512
and trained S615 against unchanged serving P-E4. Preparation authenticates
inputs and creates candidate runtime artifacts; it does not grant evaluation
authorization. The registered openings and assignments remain at their original
paths and are consumed in-place.

## Preparation commands

Use the pinned evaluation environment (`CPython 3.14.7`, NumPy 2.5.3, PyTorch
2.14.0) and the environment variables bound into `execution-freeze-v3.json`.

```sh
python -m ml.alphazero_lite.seed615_evaluation --preflight
python -m ml.alphazero_lite.seed615_evaluation --prepare-runtime-artifacts
python -m ml.alphazero_lite.seed615_evaluation --publish-freeze
python -m ml.alphazero_lite.verify_seed615_evaluation --preflight
```

Preflight is read-only. Runtime exports are create-only and preserve the exact
checkpoint `model.npz` bytes. The synthetic composition probe checks both
players and extra turns. Candidate policy uses the candidate's full trunk and
policy head; candidate value uses the full serving P-E4 model, as does the
opponent's policy and value.

## Registered contract

The registered opening and assignment JSONL files are not regenerated, sorted,
or rewritten. Their canonical JSON digests and actual file-byte digests are
recorded separately in the freeze. The independently reconstructed 4,096
assignment rows must match the source file exactly, including game order,
candidate seat, and deterministic seed.

Search is 128 simulations per side, `c_puct=1.25`, one worker per game,
deterministic zero-temperature roots, zero FPU, no noise, no subtree reuse, no
value normalization, zero tactical bias, 16-stone exact-root solving by final
score margin, registered network-prior/move-index tie breaking, no exact-leaf
solving, and a 200-ply cap. Opening pairs are scheduled with 18-way pair
parallelism and a 64 GiB resource limit.

The analysis is an opening-cluster paired bootstrap with 20,000 resamples and
`numpy.random.default_rng(615)`. The primary S615-minus-B512 score interval uses
linear 2.5%/97.5% quantiles and passes only when its lower endpoint is strictly
above zero. The separate S615-versus-P-E4 lower bound uses the linear 5%
quantile and passes only when strictly above 0.50. Incremental improvement
without absolute strength above P-E4 does not authorize automatic promotion.

## Authorization and execution policy

The initial local draft `execution-freeze.json` (SHA-256
`181592beb7ad73978f42f508c0c91b155cf9725b7f6fe583334228ad7ba92218`) was
published before the implementation audit completed. It is retained unchanged
as superseded evidence. The intermediate v2 freeze (SHA-256
`ef977b567c133ce78090c019633b604dcf966132398a5ef135b57c90f202278c`) is also
retained unchanged as superseded evidence. The final reviewed, source-bound
freeze is `execution-freeze-v3.json` with paired
`readiness-evidence-v3.json`. The reviewed freeze is immutable, create-only,
and explicitly records
`evaluation_authorized: false` and `games_started: false`. The preflight
verifier rejects authorization records, result directories, and analysis
artifacts. Execution requires a separately created owner authorization bound
to the exact freeze bytes. Game attempts and result ledgers are create-only;
partial attempts are forensic evidence and stop resume until reconciled. No
retry changes the registered seed or replaces an earlier attempt. Analysis is
available only after the independent verifier accepts all 4,096 results.

Execution entry point (not used during preparation):

```sh
python -m ml.alphazero_lite.seed615_evaluation --run
python -m ml.alphazero_lite.verify_seed615_evaluation --post-execution
python -m ml.alphazero_lite.seed615_evaluation --analyze
```

## Historical Seed 604 audit and evidence retention

The established Seed 604 evaluation was inspected in
`docs/data/seed604-replay-optimization/evaluation-preparation/`. Its freeze
SHA-256 is `3812d07445d61d91afcf2a47b536b0b93cf915774f51a77f17af4aade37a47aa`;
its independent verification record SHA-256 is
`7abfc8a24b1fb7695627eb76cb462930ad50483ab8df2014c45d108e21fd676a`. The
original worktree's independent verifier passed with 4,096 completed games,
4,096 attempts, zero capped games, and an authorized status. The historic
authorization hash is
`a6c322f04df29c0a32d52a9030e7a4105a91063520fefa104136d4dd1bb886d9`; the
assignment manifest and analysis hashes are
`bcbccf36657092f4320d86afaf7da7b475db059f3172259157c4f9d0298e65c7` and
`4c5437bd1d885524da850e4f1a6d605d0c389e06c1ccb9a61359694c3ccb4dfa`.

Seed 604's freeze, authorization, cohort, runtime artifacts, all attempt
directories, full results, analysis, and independent verification record are
retained at their historical paths. Its stored commands contain absolute paths
to the original Seed 604 worktree; consequently the verifier passes in that
worktree but is not relocatable to a different checkout path without modifying
historical evidence. This Seed 615 implementation preserves the recorded
commands and artifact paths and verifies the submitted ledger independently.

The wider `./.tmp` tree contains numerous linked historical experiment
worktrees, source snapshots, checkpoints, datasets, runtime environments, and
test outputs (approximately 450 GiB at audit time). These are not disposable
cache outputs: some are linked worktrees and some may be unique experimental
evidence. No such pre-existing files were removed. No disposable Seed 615
temporary output remains: the only generated outputs are the two requested
runtime-artifact bundles and immutable freeze/readiness evidence, all retained
here. No registered game, result, or bootstrap artifact has been generated.
