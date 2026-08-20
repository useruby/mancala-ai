# Gen-2 Selective Risk-Target Results

**Primary classification:** `static_mask_risk_migrates`

**Recommended follow-up:** test dynamic periodic mask refresh during training

## Frozen Masks

- q75: `0.0084936395`; q90: `0.0110329449`
- Unique states: `23395`; q75 protected: `5849`

## Training Metrics

| Lane | Step | CE(search) | CE(P1) | CE(target) | Fit fraction | Unprotected improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| beta095 | 1 | 1.035547 | 0.934207 | 0.939274 | 0.0669 | 0.000156 |
| beta095 | 4 | 1.035109 | 0.934211 | 0.939256 | 0.2550 | 0.000594 |
| beta095 | 16 | 1.034240 | 0.934227 | 0.939228 | 0.6285 | 0.001463 |
| beta095 | 46 | 1.033375 | 0.934240 | 0.939196 | 1.0000 | 0.002328 |
| risk_q90 | 1 | 1.035562 | 0.934207 | 0.938739 | 0.0606 | 0.000119 |
| risk_q90 | 4 | 1.035188 | 0.934210 | 0.938727 | 0.2213 | 0.000448 |
| risk_q90 | 16 | 1.034498 | 0.934220 | 0.938709 | 0.5177 | 0.001066 |
| risk_q90 | 46 | 1.033686 | 0.934231 | 0.938685 | 0.8665 | 0.001823 |
| risk_q75 | 1 | 1.035568 | 0.934207 | 0.937885 | 0.0578 | 0.000097 |
| risk_q75 | 4 | 1.035238 | 0.934210 | 0.937879 | 0.1998 | 0.000344 |
| risk_q75 | 16 | 1.034855 | 0.934214 | 0.937870 | 0.3643 | 0.000694 |
| risk_q75 | 46 | 1.034088 | 0.934222 | 0.937853 | 0.6935 | 0.001364 |
| matched_random25 | 1 | 1.035576 | 0.934207 | 0.938017 | 0.0546 | 0.000130 |
| matched_random25 | 4 | 1.035164 | 0.934210 | 0.938005 | 0.2316 | 0.000552 |
| matched_random25 | 16 | 1.034474 | 0.934221 | 0.937988 | 0.5277 | 0.001266 |
| matched_random25 | 46 | 1.033785 | 0.934229 | 0.937969 | 0.8237 | 0.001974 |

## Protected-State Leakage (Step 46)

| Lane | Protected L1 mean | Protected L1 p95 | Protected L1 p99 | CE(candidate, P1) | Direction vs P2-P1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| beta095 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | +0.0000 |
| risk_q90 | 0.010083 | 0.013385 | 0.015902 | 1.028250 | +0.9816 |
| risk_q75 | 0.006533 | 0.009923 | 0.011626 | 1.107262 | +0.9390 |
| matched_random25 | 0.004708 | 0.010270 | 0.012739 | 0.936406 | +0.9851 |

## Risk Migration (Step 46)

| Lane | Current q75 | Top-25% Jaccard vs frozen q75 | Originally protected in current top-25% | New unprotected above frozen q75 |
| --- | ---: | ---: | ---: | ---: |
| beta095 | 0.008494 | 1.0000 | 0.0000 | 5849 |
| risk_q90 | 0.007357 | 0.6771 | 0.3705 | 1496 |
| risk_q75 | 0.005812 | 0.4726 | 0.6418 | 91 |
| matched_random25 | 0.006869 | 0.8128 | 0.2406 | 2299 |

## Deterministic Search Probes vs P1

| Lane | Step | Move changes | Visit JS | Q-rank changes | Root-value delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| beta095 | 16 | 0.0156 | 0.000248 | +0.0078 | +0.000240 |
| beta095 | 46 | 0.0117 | 0.000298 | +0.0312 | +0.000035 |
| risk_q90 | 16 | 0.0078 | 0.000274 | +0.0000 | +0.000154 |
| risk_q90 | 46 | 0.0039 | 0.000270 | +0.0117 | -0.000027 |
| risk_q75 | 16 | 0.0039 | 0.000283 | +0.0000 | +0.000151 |
| risk_q75 | 46 | 0.0039 | 0.000221 | -0.0078 | +0.000118 |
| matched_random25 | 16 | 0.0078 | 0.000341 | +0.0117 | +0.000250 |
| matched_random25 | 46 | 0.0156 | 0.000244 | +0.0352 | +0.000002 |

## Frozen q75 Tail Override on risk_q75

| Step | Expanded nodes | Replacement nodes | Replacement fraction |
| ---: | ---: | ---: | ---: |
| 16 | 77364 | 95 | 0.0012 |
| 46 | 77371 | 2451 | 0.0317 |

## Arena

| Lane | Step | Effect vs P1 | 95% CI | Safe |
| --- | ---: | ---: | --- | ---: |
| beta095 | 46 | -0.0957 | [-0.1270, -0.0645] | False |
| risk_q90 | 16 | -0.0195 | [-0.0391, -0.0039] | False |
| risk_q90 | 46 | -0.0742 | [-0.1055, -0.0430] | False |
| risk_q75 | 16 | -0.0098 | [-0.0195, -0.0020] | True |
| risk_q75 | 46 | -0.0273 | [-0.0469, -0.0117] | False |
| matched_random25 | 46 | -0.0820 | [-0.1133, -0.0508] | False |
