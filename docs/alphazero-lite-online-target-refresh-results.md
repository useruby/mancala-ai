# Online Current-Candidate Search-Target Refresh

**Classification:** `dynamic_targets_are_effectively_static`

**Recommended follow-up:** Do not pursue selector refresh merely to force a positive result.

## Invariants And Calibration

- Immutable P1, A16, replay, selector, and static-cache hashes matched PR #233.
- In-memory A16 ordinary and shadow PUCT reproduced the cached targets.
- Targets were generated from and fingerprinted against the pre-update candidate.
- Frozen global auxiliary weight: `0.021055896346340041`; maximum prospective raw ratio: `23.74631751`.
- No runtime weighted auxiliary/primary ratio exceeded 2.0.

## Checkpoint Metrics

| Lane | Step | CE(search) | CE(P1) | CE(beta095) | Fit | L1 | JS | Top-1 | Movement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_continue | 1 | 1.100517 | 0.947238 | 0.954902 | 0.3462 | 0.001860 | 7.57e-07 | 0.0020 | 0.002631 |
| baseline_continue | 4 | 1.100463 | 0.947239 | 0.954900 | 0.4034 | 0.002157 | 1.02e-06 | 0.0022 | 0.003057 |
| baseline_continue | 16 | 1.100276 | 0.947243 | 0.954895 | 0.6013 | 0.003115 | 2.11e-06 | 0.0028 | 0.004595 |
| online_ordinary | 1 | 1.100516 | 0.947238 | 0.954902 | 0.3466 | 0.001861 | 7.58e-07 | 0.0020 | 0.002635 |
| online_ordinary | 4 | 1.100457 | 0.947239 | 0.954900 | 0.4090 | 0.002184 | 1.04e-06 | 0.0022 | 0.003104 |
| online_ordinary | 16 | 1.100233 | 0.947245 | 0.954894 | 0.6467 | 0.003342 | 2.44e-06 | 0.0030 | 0.004978 |
| online_shadow | 1 | 1.100516 | 0.947238 | 0.954902 | 0.3466 | 0.001861 | 7.58e-07 | 0.0020 | 0.002635 |
| online_shadow | 4 | 1.100457 | 0.947239 | 0.954900 | 0.4090 | 0.002184 | 1.04e-06 | 0.0022 | 0.003105 |
| online_shadow | 16 | 1.100233 | 0.947245 | 0.954894 | 0.6466 | 0.003339 | 2.44e-06 | 0.0030 | 0.004979 |
| static_shadow | 1 | 1.100516 | 0.947238 | 0.954902 | 0.3466 | 0.001861 | 7.58e-07 | 0.0020 | 0.002635 |
| static_shadow | 4 | 1.100457 | 0.947239 | 0.954900 | 0.4090 | 0.002184 | 1.04e-06 | 0.0022 | 0.003105 |
| static_shadow | 16 | 1.100233 | 0.947245 | 0.954894 | 0.6464 | 0.003337 | 2.44e-06 | 0.0030 | 0.004977 |

## Target Drift

At online-shadow step 16, drift from the PR #233 static shadow cache was mean L1 `0.00125000` and mean JS `1.922e-06` with zero top-1 disagreement.
At online-ordinary step 16, drift from static shadow was mean L1 `0.03343099` and mean JS `9.562e-04`. Online-shadow versus online-ordinary JS was `9.414e-04`.

## Ordinary-PUCT Evaluation

All meaningful checkpoints were evaluated with ordinary PUCT only. At step 16, every lane had frozen rescue `0.30`, new-divergence `0.0`, and the same unsafe 1200:1200 paired effect `-0.01953125` with 95% CI `[-0.0390625, -0.00390625]`.

No checkpoint was safe at both budgets; therefore no P0 gate was eligible.

## Interpretation

Refreshing the shadow target against the current candidate produced only extremely small shadow-target movement and did not alter ordinary-PUCT strength or frozen-root rescue relative to the static lane. The primary online-shadow comparison therefore does not support target staleness as the failure mechanism.
