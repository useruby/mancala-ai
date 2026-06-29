# AlphaZero-Lite Post-Schedule Pairwise PUCT Update Results

**Date**: 2026-06-29

**Classification**: `pairwise_update_too_weak`

## Inputs

- Current artifact weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Current source checkpoint SHA256: `18dacb5ef38602a77abf3927dc8748ac167c2ca91237385ec010ab27a123adc9`

## Promoted Search Schedule Confirmation

- Schedule manifest: `{"default_c_puct": 1.25, "overrides": {"768:768": 0.9}}`
- Root policy mode: `deterministic`
- Tactical root bias: `0.0`

## Dataset Build

- Pairwise target row count: `1130`
- Anchor row count: `8000`
- Probe composition: `{"broad_anchor": 4000, "pairwise": 1130, "stability": 2000}`

## Pairwise Loss Implementation Summary

- `train.py` now accepts optional pairwise target files, pairwise loss weight, and margin.
- Pairwise loss: `softplus(margin - (logit[puct_move] - logit[raw_move]))`.
- Behavior anchors remain policy-only and keep the current distribution via cross-entropy.
- Training remains restricted to `trainable_scope=policy_head`, `residual_v3`, `kalah_v3`, `hidden_sizes=96,3`.

## Probe Metrics Table

| Candidate | Pairwise success | Margin improve | Stability preserve | Broad changed | Anchor KL | PUCT agree | Aborted |
|---|---|---|---|---|---|---|---|
| current_ref | +0.0000 | -0.6908 | +1.0000 | +0.0000 | +0.0000 | +0.5391 | False |
| pairwise_margin005_kl4_e1 | +0.0035 | -0.6881 | +0.9995 | +0.0060 | +0.0000 | +0.5415 | True |
| pairwise_margin005_kl4_e2 | +0.0053 | -0.6855 | +0.9995 | +0.0120 | +0.0000 | +0.5435 | True |
| pairwise_margin010_kl4_e1 | +0.0035 | -0.6881 | +0.9995 | +0.0060 | +0.0000 | +0.5415 | True |
| pairwise_margin010_kl4_e2 | +0.0053 | -0.6855 | +0.9995 | +0.0120 | +0.0000 | +0.5435 | True |
| pairwise_margin010_kl8_e1 | +0.0027 | -0.6885 | +1.0000 | +0.0045 | +0.0000 | +0.5421 | True |
| pairwise_margin010_kl8_e2 | +0.0044 | -0.6863 | +0.9995 | +0.0090 | +0.0000 | +0.5425 | True |

## Aborted-Candidate Table

| Candidate | Reasons |
|---|---|
| pairwise_margin005_kl4_e1 | pairwise success gain < +3 percentage points |
| pairwise_margin005_kl4_e2 | pairwise success gain < +3 percentage points; anchor KL materially exceeds best lower-drift lane |
| pairwise_margin010_kl4_e1 | pairwise success gain < +3 percentage points |
| pairwise_margin010_kl4_e2 | pairwise success gain < +3 percentage points; anchor KL materially exceeds best lower-drift lane |
| pairwise_margin010_kl8_e1 | pairwise success gain < +3 percentage points |
| pairwise_margin010_kl8_e2 | pairwise success gain < +3 percentage points; anchor KL materially exceeds best lower-drift lane |

## Training Losses

| Candidate | Pairwise loss | Behavior loss | Total loss | Validation loss | Checkpoint SHA256 | Artifact weights SHA256 | Delta norm |
|---|---|---|---|---|---|---|---|
| current_ref | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| pairwise_margin005_kl4_e1 | +1.3770 | +1.3983 | +6.9700 | +7.0300 | cf160dd89c089f9813da1846c416d1f49de1df5ce47f58d2dd6b486e8090d45c | 0e88efdd7090f926f10774b08c659a8f017f02e1166f96ae66e6d90250ee4cd8 | +0.0017 |
| pairwise_margin005_kl4_e2 | +1.3746 | +1.3957 | +6.9573 | +7.0280 | db7081d5aa74c1b661b4f9975d39151af53dd6b728f6f4a64f2dad99fdd5149c | 72d760d823068c6591021bfce77a8f78b30e17022d39940645eb2744244c3a1e | +0.0033 |
| pairwise_margin010_kl4_e1 | +1.4124 | +1.3983 | +7.0054 | +7.0659 | a90872476abb83c5834cf59fcb2a7aaf4a1abaf5bb8855a9753fe47e53f3ca7f | 7fab341601685fc2a5dc088bcd968abb7a020f06cd7ded8170ae2d67e462dc95 | +0.0017 |
| pairwise_margin010_kl4_e2 | +1.4099 | +1.3957 | +6.9926 | +7.0640 | 6482ce437799307fb641fbf1c3fbc38ad32ded09a6b66b724ee4c47be165b8c8 | 911a1415e2b4ee5fcd399b55cd1cf376c580613664ebc19fee88809fb6d1f4be | +0.0033 |
| pairwise_margin010_kl8_e1 | +1.4124 | +1.3982 | +12.5984 | +12.6517 | ab0d5ffdde366312f3598eac80ce6fd831e399ddc82f4bf3e4f9d29b8b659bc2 | 4fe537bd8a2b76c817c501bb2e45357929b20d805ef01b90c37765690d8251f6 | +0.0017 |
| pairwise_margin010_kl8_e2 | +1.4103 | +1.3956 | +12.5752 | +12.6496 | f5a0352172619670dba2d939946475c5939448c389bd52d40334ac0afce1b495 | 64d470ebfce81d53de91844e5c42e7e1a8d9e0ae28a3f072b2bb7afa065e2c20 | +0.0033 |

## Fixed Large DS Table

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| current_ref | -0.3099 | -0.4049 | +0.6849 | +0.2812 | -0.1159 | -0.4049 |

## Held-Out Mean/Worst-Suite Table

| Candidate | Held-out mean 384:256 | Held-out worst-suite 384:256 |
|---|---|---|
| current_ref | -0.2993 | -0.3255 |

## Bootstrap CI

| Comparison | Mean | Lower 95% | Upper 95% |
|---|---|---|---|

## P0/P1 Split For 384:256

| Candidate | Mean P0 | Mean P1 | Gap | Mean duplicates |
|---|---|---|---|---|
| current_ref | +0.6157 | +0.9240 | +0.3084 | +1536.0000 |

## Gate Classification

- current_ref: `high_search_breakthrough`
- pairwise_margin005_kl4_e1: `not_run`
- pairwise_margin005_kl4_e2: `not_run`
- pairwise_margin010_kl4_e1: `not_run`
- pairwise_margin010_kl4_e2: `not_run`
- pairwise_margin010_kl8_e1: `not_run`
- pairwise_margin010_kl8_e2: `not_run`
