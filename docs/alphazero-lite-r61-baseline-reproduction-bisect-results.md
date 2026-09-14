# R61 Baseline Reproduction Bisect

Inherited classification: `parameter_drift_baseline_not_reproduced`.

## Environment

- Python: `3.14.7 (main, Aug 10 2026, 07:46:56) [GCC 16.1.1 20260728]`
- NumPy: `2.5.3`; PyTorch: `2.14.0+cu130`
- Platform: `Linux-7.1.9-arch1-2-x86_64-with-glibc2.44`; architecture: `x86_64`
- CPU threads: `32`; torch intra/inter-op: `24/24`; deterministic algorithms: `False`
- OMP/MKL/OpenBLAS/PYTHONHASHSEED: `{'MKL_NUM_THREADS': None, 'OMP_NUM_THREADS': None, 'OPENBLAS_NUM_THREADS': None, 'PYTHONHASHSEED': None}`

## Configurations

Every cell uses CPU residual_v3, Adam, LR 0.001, B512, four epochs, clip 1.0, Huber delta 1.0, value weight 0.3, no scheduler, and replay weights 1/1/2.

| variant | anchor | raw gradients | snapshots | Adam/subspace |
| --- | --- | --- | --- | --- |
| V0 | False | False | False | False |
| V1 | True | True | False | False |
| V2 | False | False | False | False |
| V3 | True | False | False | False |
| V4 | True | True | False | False |
| V5 | True | True | True | False |
| V6 | True | True | True | True |
| V7 | True | True | True | True |

## Input And RNG Parity

All V0-V7 cells had identical replay `x`, `p`, `v`, and `replay_indexes` shape/dtype/byte hashes; initial model state and anchor policy hashes; and Python, NumPy, and Torch CPU RNG hashes for each seed. The JSON artifact retains each value.

## Matrix

| seed | variant | delta | top | best epoch | first batch divergence | first parameter divergence |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T61 | V0 | -0.126246 | 3 | 4 | None | None |
| T61 | V1 | -0.126246 | 3 | 4 | None | None |
| T61 | V2 | -0.126246 | 3 | 4 | None | None |
| T61 | V3 | -0.126246 | 3 | 4 | None | None |
| T61 | V4 | -0.126246 | 3 | 4 | None | None |
| T61 | V5 | -0.126246 | 3 | 4 | None | None |
| T61 | V6 | -0.126246 | 3 | 4 | None | None |
| T61 | V7 | -0.126246 | 3 | 4 | None | None |
| T63 | V0 | 0.154806 | 0 | 4 | None | None |
| T63 | V1 | 0.154806 | 0 | 4 | None | None |
| T63 | V2 | 0.154806 | 0 | 4 | None | None |
| T63 | V3 | 0.154806 | 0 | 4 | None | None |
| T63 | V4 | 0.154806 | 0 | 4 | None | None |
| T63 | V5 | 0.154806 | 0 | 4 | None | None |
| T63 | V6 | 0.154806 | 0 | 4 | None | None |
| T63 | V7 | 0.154806 | 0 | 4 | None | None |

## Epoch And Validation

| seed | variant | raw epoch-end anchor masses | validation totals by epoch |
| --- | --- | --- | --- |
| T61 | V0 | 0.1439, 0.3589, 0.2627, 0.2737 | 0.845674, 0.845255, 0.829294, 0.806239 |
| T61 | V1 | 0.1439, 0.3589, 0.2627, 0.2737 | 0.845674, 0.845255, 0.829294, 0.806239 |
| T61 | V2 | 0.1439, 0.3589, 0.2627, 0.2737 | 0.845674, 0.845255, 0.829294, 0.806239 |
| T61 | V3 | 0.1439, 0.3589, 0.2627, 0.2737 | 0.845674, 0.845255, 0.829294, 0.806239 |
| T61 | V4 | 0.1439, 0.3589, 0.2627, 0.2737 | 0.845674, 0.845255, 0.829294, 0.806239 |
| T61 | V5 | 0.1439, 0.3589, 0.2627, 0.2737 | 0.845674, 0.845255, 0.829294, 0.806239 |
| T61 | V6 | 0.1439, 0.3589, 0.2627, 0.2737 | 0.845674, 0.845255, 0.829294, 0.806239 |
| T61 | V7 | 0.1439, 0.3589, 0.2627, 0.2737 | 0.845674, 0.845255, 0.829294, 0.806239 |
| T63 | V0 | 0.4201, 0.5390, 0.2314, 0.5548 | 0.838463, 0.813169, 0.814293, 0.806136 |
| T63 | V1 | 0.4201, 0.5390, 0.2314, 0.5548 | 0.838463, 0.813169, 0.814293, 0.806136 |
| T63 | V2 | 0.4201, 0.5390, 0.2314, 0.5548 | 0.838463, 0.813169, 0.814293, 0.806136 |
| T63 | V3 | 0.4201, 0.5390, 0.2314, 0.5548 | 0.838463, 0.813169, 0.814293, 0.806136 |
| T63 | V4 | 0.4201, 0.5390, 0.2314, 0.5548 | 0.838463, 0.813169, 0.814293, 0.806136 |
| T63 | V5 | 0.4201, 0.5390, 0.2314, 0.5548 | 0.838463, 0.813169, 0.814293, 0.806136 |
| T63 | V6 | 0.4201, 0.5390, 0.2314, 0.5548 | 0.838463, 0.813169, 0.814293, 0.806136 |
| T63 | V7 | 0.4201, 0.5390, 0.2314, 0.5548 | 0.838463, 0.813169, 0.814293, 0.806136 |

## Provenance

- `train.py` blob: `8c9e733673767b005c131a7351610844a6555f42`
- `run_anchor_minibatch_interference_audit.py` blob: `3704c6c8d638eefbb3c9189e387dcad9c27bb08f`
- Historical local commits: PR #307 `e83c263193b0c08f928eb22883859ae719e577b6`, PR #308 `6084fd9a446afd5a78ba40d2185d6a805b05d1b7`.
- The relevant post-#308 `train.py` diff is callback/telemetry plumbing only; no replay split, legal-mask, permutation, mode, validation, best-state, clipping, or Adam initialization semantic change was found.

## Classification

`baseline_reproduction_unexplained`

Exactly one next action: perform an optimizer-step numerical parity audit at the first differing parameter tensor.

The JSON artifact contains the full environment manifest, input/model/RNG hashes, epoch permutations and step batches, parameter hashes, raw epoch anchors, validation losses, and checkpoint identities.
