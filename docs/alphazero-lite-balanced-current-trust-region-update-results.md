# AlphaZero-Lite Balanced Current Trust-Region Update Results

**Date**: 2026-06-26

**Classification**: `update_format_still_harmful`

## Inputs

- Current artifact weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Current source checkpoint SHA256: `18dacb5ef38602a77abf3927dc8748ac167c2ca91237385ec010ab27a123adc9`

## Dataset Build

- Validated disagreement rows: `2000`
- Behavior anchor rows: `8000`
- Probe-state composition: `{"broad_anchor": 4000, "disagreement": 2000, "stability": 2000}`

## Probe Metrics

| Candidate | PUCT agree | Cand||PUCT KL | Stability preserve | Broad preserve | Anchor KL | Broad changed | Aborted |
|---|---|---|---|---|---|---|---|
| balanced_current_ref | +0.0000 | +1.5117 | +1.0000 | +1.0000 | +0.0000 | +0.0000 | False |
| trust_region_kl1_e1 | +0.0345 | +1.5093 | +1.0000 | +0.9828 | +0.0002 | +0.0173 | False |
| trust_region_kl1_e2 | +0.0340 | +1.5076 | +1.0000 | +0.9798 | +0.0003 | +0.0203 | False |
| trust_region_kl4_e1 | +0.0320 | +1.5191 | +1.0000 | +0.9792 | +0.0004 | +0.0208 | False |
| trust_region_kl4_e2 | +0.0375 | +1.5231 | +1.0000 | +0.9765 | +0.0007 | +0.0235 | True |
| trust_region_kl8_e1 | +0.0315 | +1.5219 | +1.0000 | +0.9775 | +0.0005 | +0.0225 | True |
| trust_region_kl8_e2 | +0.0375 | +1.5278 | +1.0000 | +0.9732 | +0.0009 | +0.0267 | True |

## Aborted Candidates

| Candidate | Reasons |
|---|---|
| trust_region_kl4_e2 | mean anchor KL exceeds 2x best lower-KL lane |
| trust_region_kl8_e1 | mean anchor KL exceeds 2x best lower-KL lane |
| trust_region_kl8_e2 | mean anchor KL exceeds 2x best lower-KL lane |

## Training Losses And Artifacts

| Candidate | Policy loss | Value loss | Behavior loss | Total loss | Validation loss | Checkpoint SHA256 | Artifact weights SHA256 | Delta norm |
|---|---|---|---|---|---|---|---|---|
| balanced_current_ref | n/a | n/a | n/a | n/a | n/a | 18dacb5ef38602a77abf3927dc8748ac167c2ca91237385ec010ab27a123adc9 | 8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a | +0.0000 |
| trust_region_kl1_e1 | +1.1855 | +0.2183 | +1.1700 | +2.4210 | +2.4985 | d7bf03cc782d4f1df88b94190c36ff5a53d2d1d7c1e2f31c2106467e65f8f894 | ac720f88831fcd33ebaaa4cc460ae6b5bf8f2e7aad0d0372d5342987c9d4f60d | +0.0257 |
| trust_region_kl1_e2 | +1.1859 | +0.2184 | +1.1691 | +2.4205 | +2.4979 | 95130cdda604dab82df15bb61b60b5a9eb89ce691995624745bd7904969431af | f43fb1a3a4f668fe2329651be42261c4fc1cef7488272a117572ee5aafe1fc07 | +0.0451 |
| trust_region_kl4_e1 | +1.1868 | +0.2183 | +1.1690 | +5.9282 | +5.9805 | 80c1b6044130d31d5aea965578e884e30b786f5fc4a77139ce2e3cb4c9b61b16 | 45fb77e4c92ba330377d84b79d5844c73e53557ccfbd1aa41185732736b99e8a | +0.0413 |
| trust_region_kl4_e2 | +1.1891 | +0.2184 | +1.1667 | +5.9213 | +5.9764 | ad16de32888c06abfbdcfb0339e95c35988074eeb34d896f4976a4aa094dde59 | 06df61216bb3e3096b5d7f482f2fa0a15468fb57c57661d66826471ab3a3c6e4 | +0.0745 |
| trust_region_kl8_e1 | +1.1872 | +0.2183 | +1.1688 | +10.6030 | +10.6194 | 98314e2878fe2bb6db78907ec61f2a01217024ec25f48306f6e544983781285f | f38342a692e7317fb28b448db6ea3741465c8f1fbc1936bdb15b7076468ae1b6 | +0.0480 |
| trust_region_kl8_e2 | +1.1902 | +0.2184 | +1.1662 | +10.5852 | +10.6081 | 5e7295ec5c180b1348b7f6243a2c280a6ea1adf89ee77a0d9eb1b6de8c0f1f8e | 61022c314be94247876e86e65c1964866b39c851e2fc381e4c68315ce4ae9b8e | +0.0868 |

## Fixed Large DS Table

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| balanced_current_ref | -0.5456 | -0.4648 | -0.2943 | -0.1380 | -0.1797 | -0.4648 |
| trust_region_kl1_e1 | -0.6732 | -0.5026 | -0.1667 | -0.3060 | -0.2904 | -0.4388 |
| trust_region_kl1_e2 | -0.6693 | -0.4388 | -0.2943 | -0.1953 | -0.2643 | -0.3880 |
| trust_region_kl4_e1 | -0.8424 | -0.4388 | -0.4505 | -0.2773 | -0.2643 | -0.7878 |

## Held-Out Mean/Worst DS Table

| Candidate | Held-out mean 384:256 | Held-out worst-suite 384:256 |
|---|---|---|
| balanced_current_ref | -0.5488 | -0.5664 |
| trust_region_kl1_e1 | -0.6916 | -0.7096 |
| trust_region_kl1_e2 | -0.6901 | -0.7057 |
| trust_region_kl4_e1 | -0.8596 | -0.8828 |

## Bootstrap CI Table

| Comparison | Mean | Lower 95% | Upper 95% |
|---|---|---|---|
| balanced_current_ref_minus_balanced_current_ref_384_256 | +0.0000 | +0.0000 | +0.0000 |
| balanced_current_ref_minus_balanced_current_ref_768_768 | +0.0000 | +0.0000 | +0.0000 |
| balanced_current_ref_minus_balanced_current_ref_1200_1200 | +0.0000 | +0.0000 | +0.0000 |
| balanced_current_ref_minus_balanced_current_ref_1200_256 | +0.0000 | +0.0000 | +0.0000 |
| trust_region_kl1_e1_minus_balanced_current_ref_384_256 | -0.1406 | -0.1540 | -0.1280 |
| trust_region_kl1_e1_minus_balanced_current_ref_768_768 | +0.1406 | +0.1280 | +0.1540 |
| trust_region_kl1_e1_minus_balanced_current_ref_1200_1200 | -0.1685 | -0.1838 | -0.1536 |
| trust_region_kl1_e1_minus_balanced_current_ref_1200_256 | -0.1088 | -0.1166 | -0.1010 |
| trust_region_kl1_e2_minus_balanced_current_ref_384_256 | -0.1388 | -0.1535 | -0.1248 |
| trust_region_kl1_e2_minus_balanced_current_ref_768_768 | +0.0000 | +0.0000 | +0.0000 |
| trust_region_kl1_e2_minus_balanced_current_ref_1200_1200 | -0.0625 | -0.0759 | -0.0499 |
| trust_region_kl1_e2_minus_balanced_current_ref_1200_256 | -0.0841 | -0.0911 | -0.0770 |
| trust_region_kl4_e1_minus_balanced_current_ref_384_256 | -0.3088 | -0.3263 | -0.2920 |
| trust_region_kl4_e1_minus_balanced_current_ref_768_768 | -0.1553 | -0.1685 | -0.1425 |
| trust_region_kl4_e1_minus_balanced_current_ref_1200_1200 | -0.1373 | -0.1479 | -0.1270 |
| trust_region_kl4_e1_minus_balanced_current_ref_1200_256 | -0.0869 | -0.0969 | -0.0770 |

## P0/P1 Split At 384:256

| Candidate | Mean P0 | Mean P1 | Gap | Mean duplicates |
|---|---|---|---|---|
| balanced_current_ref | +0.4107 | +0.9591 | +0.5484 | +1536.0000 |
| trust_region_kl1_e1 | +0.2225 | +0.9115 | +0.6890 | +1536.0000 |
| trust_region_kl1_e2 | +0.2472 | +0.9343 | +0.6871 | +1536.0000 |
| trust_region_kl4_e1 | +0.0772 | +0.9343 | +0.8571 | +1536.0000 |

## Gate

- balanced_current_ref: `high_search_breakthrough`
- trust_region_kl1_e1: `not_run`
- trust_region_kl1_e2: `not_run`
- trust_region_kl4_e1: `not_run`
- trust_region_kl4_e2: `not_run`
- trust_region_kl8_e1: `not_run`
- trust_region_kl8_e2: `not_run`
