# Seed 594 — Stage 1 raw-policy preparation

Status: **prepared, pending independent review; no qualification execution
authorized**. This package compares deterministic raw policy actions from Seed
594 with frozen serving P-E4 at the existing 423 #570 qualification roots.
The cohort is the pre-existing #570 membership; it is not described here as a
fresh holdout or as sealed. This preparation did not change its membership,
the Seed 594 checkpoint, the exact target distribution, training configuration,
or the registered decision gate.

## Access-order disclosure

During initial artifact discovery, the public #570 `qualification-membership.json`
file was opened before the required synthetic parity check. This was an
historical access-order deviation and is preserved as an explicit field in the
execution freeze. During this initial discovery, no qualification roots were
fed to either model and no qualification outcomes or metrics were examined.
After that discovery, synthetic export/inference parity was completed before
any further qualification package or oracle inspection. No qualification
inference, qualification outcome analysis, PUCT search, game, or sealed-suite
access was performed. The existing cohort was not selected or altered in this
preparation and is not characterized as fresh or sealed.

## Synthetic-only parity

The frozen implementation uses `train.load_checkpoint_into_model` for both
P-E4 and Seed 594 and `train.checkpoint_from_model` to export the loaded
parameters. It compares PyTorch forward probabilities with the existing
`arena.ArtifactEvaluator` NumPy implementation on three literal synthetic
non-terminal states. Both models use `kalah_v3` encoding, the exact legal mask
over player-relative actions `0..5`, and lowest-index deterministic
argmax tie-breaking. No qualification root is read by the parity function.

Recorded parity (Python 3.14.7, NumPy 2.5.2, PyTorch 2.13.0+cpu):

| Model | Synthetic forwards | Maximum absolute probability difference | Illegal-action masks |
| --- | ---: | ---: | --- |
| P-E4 | 3 | `1.1920928955078125e-07` | exact zero |
| Seed 594 | 3 | `5.960464477539063e-08` | exact zero |

Tolerance: `1e-6`. Seed 594's authenticated checkpoint SHA-256 is
`7fddf5d6cba7eea4e7b8e06ba89c5b936d28727470934a51c603a739768e9bdd`; P-E4's
is `2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad`.

## Identity-only cohort/oracle authentication

After synthetic parity passed, the existing #570 verifier authenticated the
membership and its complete-action oracle coverage. It reported 423 membership
roots, 89 shared-opening rejections, and zero canonical-identity, float32,
duplicate-identity, or duplicate-opening rejections. This is source identity
and coverage verification only; no predictions, exact regret, WDL rates, or
qualification outcome summaries were computed. Existing membership digest:
`94e126af6ea88fb99a4230f84e5cf2a20248daa39ea66d25b68730ea0426d668`.

The authenticated #559 run-0003 source identities are pinned in
`execution-freeze.json`, including the cohort manifest and complete exact-action
oracle JSONL. The verifier checked exact legal-action coverage for each retained
membership root; oracle values were not used for any selection or analysis.

## Frozen Stage 1 procedure and gate

The authenticated #559 native oracle records each action margin from its
requested `perspective_player`; the source records use the root side to move.
Exact regret is `best legal action margin - chosen action margin` in that
acting-player perspective. Equal policy logits select the lowest legal
player-relative action index. WDL downgrade is the rate at which the selected
action's exact outcome rank (`win=1`, `draw=0`, `loss=-1`, by acting-player
perspective margin) is below the best legal outcome rank. Candidate and P rates
are compared on the same root set.

Each root records the full exact-optimal action set, whether each model's
deterministic top action is exact-optimal, and each model's prior mass over all
exact-optimal actions. Aggregate reports include both models' exact-optimal
top-action rates, mean exact-optimal prior mass, mean exact regret, and WDL
downgrade counts/rates. The original regret-difference bootstrap and three-part
advancement gate remain unchanged.

`verify_paired_result_membership` checks all 423 result state hashes against
their exact #570 opening indices and rejects missing, substituted, duplicate,
or mismapped rows. `verify_results_against_authenticated_membership` first
re-verifies the frozen #570 package and then returns the complete checked hash
list and opening-to-state mapping for inclusion with published results.

The paired CI resamples the 423 opening clusters with replacement using
`numpy.random.default_rng(594)`, 20,000 resamples, and NumPy `quantile(method="linear")`
at `[0.025, 0.975]`. There is one retained root per opening. The advancement
gate is unchanged:

1. Mean candidate-minus-P exact regret is `<= -0.10` seeds.
2. The paired 95% opening-bootstrap CI upper endpoint is `< 0`.
3. Candidate WDL downgrade rate does not exceed P's rate.

All checks are required to keep/advance; missing or duplicate results fail
closed. `evaluate_membership_roots` is the frozen one-forward-per-model runner;
it requires a review-bound authorization whose SHA-256 names this exact
execution freeze. No qualification predictions or results are included in
this package.

## Verification

```bash
PYTHONHASHSEED=571 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/alex/azlite-venv/bin/python -m pytest \
  ml/alphazero_lite/test_seed594_stage1_raw_policy.py -q

PYTHONHASHSEED=571 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/alex/azlite-venv/bin/python -c \
  'from ml.alphazero_lite.seed594_stage1_raw_policy import verify_synthetic_export_parity; print(verify_synthetic_export_parity())'

PYTHONHASHSEED=571 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/alex/azlite-venv/bin/python -m \
  ml.alphazero_lite.seed570_qualification_cohort
```

The final command is read-only identity/coverage authentication. Qualification
execution remains a separate, once-only action after independent review and
authorization. Freeze verification, which enforces parity-before-membership
ordering, is:

```bash
PYTHONHASHSEED=571 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/alex/azlite-venv/bin/python -c \
  'from ml.alphazero_lite.seed594_stage1_raw_policy import verify_preparation_freeze; print(verify_preparation_freeze())'
```

This preparation PR does not authorize execution.
