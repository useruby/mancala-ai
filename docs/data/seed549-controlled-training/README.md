# Seed549 controlled training comparison

This prospective experiment compares the source self-play evaluators. Their
effects include different states, policy/value targets, outcomes, and occurrence
frequencies; this does **not** isolate policy-target quality alone. Training
occurrence multiplicity, production-loader target handling, and policy-loss
coefficients are retained without deduplication, rebalancing, target softening,
or additional data.

## Frozen contract

Control trains on the initializer-generated seed549 training population;
treatment trains on the serving-P-generated population. Both start from the
same seed542 initializer (SHA-256
`7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c`) and use
their unchanged 108-update, 54,896-exposure schedules (`[512] * 26 + [412]`
repeated four times). The architecture is residual_v3/kalah_v3, hidden `(96,3)`,
27 inputs. The objective is legal policy CE plus 0.5 times Huber value loss
(delta 1), with a fresh Adam per arm (lr `1e-4`, betas `(0.9,0.999)`, epsilon
`1e-8`, no decay/scheduler) and gradient clipping at 1.0. Adam moments persist
through all updates within each arm.

## Reproduction

From the repository root, use the frozen numerical environment and resource
scope recorded in `registration.json`:

```bash
systemd-run --user --scope -p CPUQuota=25% -p CPUWeight=1 \
  -p MemoryMax=4294967296 -p AllowedCPUs=30 -- \
  nice -n 19 prlimit --as=17179869184 -- taskset -c 30 env \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  BLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /home/alex/Mancala/ai/.tmp/seed542-continuation-prep/.tmp/seed542-recorded-runtime/bin/python \
  -m ml.alphazero_lite.seed549_training freeze

# After freeze and required checks:
systemd-run --user --scope -p CPUQuota=25% -p CPUWeight=1 \
  -p MemoryMax=4294967296 -p AllowedCPUs=30 -- \
  nice -n 19 prlimit --as=17179869184 -- taskset -c 30 env \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  BLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /home/alex/Mancala/ai/.tmp/seed542-continuation-prep/.tmp/seed542-recorded-runtime/bin/python \
  -m ml.alphazero_lite.seed549_training execute

# One read-only command; for a physical copy pass both --root and --out:
systemd-run --user --scope -p CPUQuota=25% -p CPUWeight=1 \
  -p MemoryMax=4294967296 -p AllowedCPUs=30 -- \
  nice -n 19 prlimit --as=17179869184 -- taskset -c 30 env \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  BLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /home/alex/Mancala/ai/.tmp/seed542-continuation-prep/.tmp/seed542-recorded-runtime/bin/python \
  -m ml.alphazero_lite.verify_seed549_training \
  --root /path/to/physical-copy \
  --out /path/to/physical-copy/docs/data/seed549-controlled-training
```

Training and publication evidence is in this directory. `*-updates.jsonl`
records each scheduled occurrence batch, loader tensors, objective components,
raw/clipped gradient identities and norms, Adam state identities, and
before/after parameter identities. Checkpoints are retained after updates 27,
54, 81, and 108. The control has a current-runtime production-equivalence gate;
treatment is not started if any control update differs. Update 108 treatment is
the sole candidate.

`development-predictions.jsonl` binds every endpoint prediction to each retained
development occurrence, including occurrence-specific targets and exact
float32-input identity. `development-results.json` reports policy CE and
actual-value MSE for pooled, `>32`, `17–32`, and `<=16` cohorts, by source
evaluator and combined, under exposure and equal-input weighting. Effects are
T−C and each trained endpoint minus initializer. These findings are descriptive
only and do not authorize advancement. No new generation, arena games, serving
export, promotion, or deployment is part of this comparison.
