# #571 policy distillation: preparation runbook

Status: prepared and frozen; **do not run the real target/training command yet**.
This PR does not generate targets, train, play games, access a sealed suite, or
change serving. #483 remains closed. The output checkpoint must receive a
separate freeze before any qualification.

## Authentication and data boundary

The read-only #557 verifier is `python -m
ml.alphazero_lite.verify_seed557_retrospective`. The #571 read-only cohort
verifier is `python -m ml.alphazero_lite.seed570_qualification_cohort
--membership docs/data/seed570-policy-prior-distillation/qualification-membership.json`.
It verifies 423-root membership digest
`94e126af6ea88fb99a4230f84e5cf2a20248daa39ea66d25b68730ea0426d668`.
That membership is identity-only for this training job. Training inputs are
exactly the 256 #557 roots authenticated by the #557 verifier. Qualification
states, oracle labels, and oracle margins are never read by target generation,
training, or checkpoint selection.

The policy/trunk teacher is frozen serving-P E4; the separate value model is
the authenticated #557 initializer. Their exact four-file identities are in
`training-configuration.json`. The teacher uses fresh 128-simulation PUCT
searches and the registered #570/#557 settings. Targets normalize visits over
legal actions; illegal actions and legal actions with zero visits remain zero.

## Synthetic tests

Run `python -m pytest ml/alphazero_lite/test_seed571_policy_distillation.py`.
The fixtures exercise pre-operation input rejection, legal-mask visit targets,
and isolation of trainable trunk/policy parameters from frozen value parameters.
They do not invoke real searches or train on real inputs.

## Later run procedure

The preparation runner is intentionally fail-closed for real runs. A later
execution PR must bind the finalized runner and this configuration's exact bytes
in a new immutable pre-training execution freeze, confirm the pinned runtime,
and change the freeze status to `authorized-for-pretraining`. Do not edit a
frozen runner/config in place. After that separate freeze has been reviewed, run
from the repository root with a new, nonexistent output directory:

```bash
PYTHONHASHSEED=571 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m ml.alphazero_lite.seed571_policy_distillation \
  --run --output-dir docs/data/seed571-policy-prior-distillation/run-0001
```

The run must retain, under that run directory:

1. `teacher-visits.jsonl`: all 256 root identities, legal actions, complete
   six-action integer visit vectors, search seed, and search settings;
2. `targets.sha256`: digest of canonical target records and their root order;
3. `training-trace.jsonl`: every one of 512 update numbers, batch root IDs,
   loss, gradient norm, and determinism/runtime identity;
4. `seed571-policy-distillation-update-512.npz` and
   `checkpoint-identity.json`: file digest, tensor names/shapes/dtypes, parent
   identities, and proof that frozen value parameters were unchanged.

No interim checkpoint may be selected. Any failed precondition, integrity
check, nonfinite value, changed frozen tensor, or incomplete trace stops the
run and preserves its partial evidence; it must not trigger a retry with altered
settings.
