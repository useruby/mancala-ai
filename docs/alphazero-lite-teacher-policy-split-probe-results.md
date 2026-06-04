# AlphaZero-lite Teacher Policy Split Probe Results

## 1. Context

- No arena was run.
- No model was promoted.
- Active references stayed read-only at `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Proposed patch bundle from PR #69 is not applied.
- PR #69 cleanup audit concluded: `teacher_policy_architecture_needed`.
- Key reason: `incumbent_proxy_disagreement` has 6 Classic-teacher eligible targets, but PUCT teacher disagrees on 7 rows.
- This probe tests whether teacher-policy split modeling can learn both teacher targets without cross-teacher interference.

- Output: `/tmp/azlite_teacher_policy_split_probe/teacher_policy_split_probe_summary.json`

## 2. Why PR #69 points to teacher-policy architecture

- PR #69 showed that even after full family cleanup, the incumbent_proxy_disagreement family still has incompatible Classic and PUCT teacher labels.
- Patch-only cleanup is insufficient because the teacher disagreement is architectural, not a reference quality issue.
- 3 families become trainable after patch, but only at 4 rows each — diagnostic-only scale.
- Opening branch remains excluded.
- Proposed patch bundle has 23 validated non-mutating entries with `do_not_auto_apply=true`.

## 3. Bucket reconstruction

| row_id | bucket | teacher_id | preferred_move | active_reference_move | legal | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-003 | excluded | excluded | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-007 | puct_teacher | artifact_puct | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-008 | classic_teacher | classic_mcts | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-009 | puct_teacher | artifact_puct | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-010 | excluded | excluded | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-012 | excluded | excluded | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-014 | classic_teacher | classic_mcts | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-018 | excluded | excluded | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-020 | excluded | excluded | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-021 | puct_teacher | artifact_puct | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-022 | classic_teacher | classic_mcts | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-023 | excluded | excluded | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-024 | classic_teacher | classic_mcts | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-025 | classic_teacher | classic_mcts | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-026 | puct_teacher | artifact_puct | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-027 | excluded | excluded | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-028 | puct_teacher | artifact_puct | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-029 | excluded | excluded | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-032 | puct_teacher | artifact_puct | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-033 | puct_teacher | artifact_puct | ? | ? | True | ok | ok |
| incumbent_proxy_disagreement-035 | classic_teacher | classic_mcts | ? | ? | True | ok | ok |

## 4. Variant definitions

| variant | architecture_or_objective | data | epochs | status | notes |
| --- | --- | --- | --- | --- | --- |
| single_head_mixed_replay | single-head MLP, mixed Classic+PUCT targets | classic + puct target rows | 1,2,4 | pending | Negative control |
| classic_only_replay | single-head MLP, Classic targets only | classic target rows | 1,2,4 | pending | Reproduce PR #48 Classic gains |
| puct_only_replay | single-head MLP, PUCT targets only | puct target rows | 1,2,4 | pending | Check PUCT-only damage to Classic |
| teacher_conditioned_probe | single-head MLP + teacher_id extra input features (one-hot) | classic + puct target rows with teacher conditioning | 1,2,4 | pending | Experimental: teacher_id as 2 extra input features |

## 5. Current baseline

| row_id | bucket | selected_384 | selected_1200 | ref_visit_share_384 | ref_visit_share_1200 | preferred_policy_prob | preferred_policy_rank | entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-003 | excluded | 4 | 5 | 0.5807 | 0.2175 | - | 2 | 1.0129 |
| incumbent_proxy_disagreement-007 | puct | 5 | 3 | 0.0052 | 0.0025 | - | 3 | 1.7160 |
| incumbent_proxy_disagreement-008 | classic | 2 | 2 | 0.9297 | 0.9708 | - | 1 | 1.7794 |
| incumbent_proxy_disagreement-009 | puct | 3 | 3 | 0.0026 | 0.0008 | - | 2 | 1.0292 |
| incumbent_proxy_disagreement-010 | excluded | 5 | 5 | 0.0417 | 0.1183 | - | 2 | 1.4633 |
| incumbent_proxy_disagreement-012 | excluded | 5 | 5 | 0.0911 | 0.0633 | - | 3 | 2.0064 |
| incumbent_proxy_disagreement-014 | classic | 5 | 5 | 0.0026 | 0.3392 | - | 2 | 1.0084 |
| incumbent_proxy_disagreement-018 | excluded | 5 | 3 | 0.0182 | 0.0108 | - | 2 | 1.8967 |
| incumbent_proxy_disagreement-020 | excluded | 5 | 5 | 0.0312 | 0.1308 | - | 4 | 2.2844 |
| incumbent_proxy_disagreement-021 | puct | 5 | 5 | 0.0078 | 0.0042 | - | 1 | 1.9341 |
| incumbent_proxy_disagreement-022 | classic | 5 | 5 | 0.0052 | 0.0033 | - | 4 | 1.9200 |
| incumbent_proxy_disagreement-023 | excluded | 1 | 1 | 0.0130 | 0.0050 | - | 4 | 1.9091 |
| incumbent_proxy_disagreement-024 | classic | 1 | 1 | 0.0104 | 0.0067 | - | 5 | 2.1013 |
| incumbent_proxy_disagreement-025 | classic | 4 | 4 | 0.0000 | 0.2650 | - | 5 | 0.3681 |
| incumbent_proxy_disagreement-026 | puct | 5 | 5 | 0.9922 | 0.9650 | - | 1 | 0.5458 |
| incumbent_proxy_disagreement-027 | excluded | 5 | 5 | 0.3542 | 0.1333 | - | 2 | 1.0666 |
| incumbent_proxy_disagreement-028 | puct | 5 | 5 | 0.9323 | 0.9250 | - | 1 | 1.5685 |
| incumbent_proxy_disagreement-029 | excluded | 4 | 4 | 0.9323 | 0.9767 | - | 1 | 1.1501 |
| incumbent_proxy_disagreement-032 | puct | 0 | 1 | 0.0859 | 0.0275 | - | 4 | 1.7833 |
| incumbent_proxy_disagreement-033 | puct | 0 | 0 | 0.2578 | 0.0825 | - | 2 | 1.7962 |
| incumbent_proxy_disagreement-035 | classic | 4 | 4 | 0.0104 | 0.0058 | - | 3 | 1.7848 |

## 6. Classic bucket results

| variant | epoch | row_id | preferred_move | selected_384 | selected_1200 | preferred_visit_share_384 | preferred_visit_share_1200 | preferred_policy_prob | improved_vs_current | damaged_vs_current | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_head_mixed_replay | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9531 | 0.9783 | 0.6787 | true | false | improved |
| single_head_mixed_replay | 1 | incumbent_proxy_disagreement-014 | 4 | 5 | 5 | 0.0026 | 0.0300 | 0.0716 | false | true | damaged |
| single_head_mixed_replay | 1 | incumbent_proxy_disagreement-022 | 3 | 2 | 3 | 0.1536 | 0.5600 | 0.1366 | true | false | improved |
| single_head_mixed_replay | 1 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.8802 | 0.9533 | 0.4296 | true | false | improved |
| single_head_mixed_replay | 1 | incumbent_proxy_disagreement-025 | 2 | 2 | 2 | 0.7318 | 0.7225 | 0.1863 | true | false | improved |
| single_head_mixed_replay | 1 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9583 | 0.9758 | 0.4993 | true | false | improved |
| single_head_mixed_replay | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9453 | 0.9758 | 0.7334 | true | false | improved |
| single_head_mixed_replay | 2 | incumbent_proxy_disagreement-014 | 4 | 5 | 5 | 0.0391 | 0.1058 | 0.1232 | true | true | improved,damaged |
| single_head_mixed_replay | 2 | incumbent_proxy_disagreement-022 | 3 | 2 | 3 | 0.1693 | 0.4317 | 0.2797 | true | false | improved |
| single_head_mixed_replay | 2 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9427 | 0.9383 | 0.6753 | true | false | improved |
| single_head_mixed_replay | 2 | incumbent_proxy_disagreement-025 | 2 | 2 | 2 | 0.5521 | 0.7142 | 0.3710 | true | false | improved |
| single_head_mixed_replay | 2 | incumbent_proxy_disagreement-035 | 3 | 3 | 4 | 0.9922 | 0.3442 | 0.8333 | true | false | improved |
| single_head_mixed_replay | 4 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9167 | 0.9708 | 0.5836 | false | false | no_change |
| single_head_mixed_replay | 4 | incumbent_proxy_disagreement-014 | 4 | 4 | 4 | 0.7552 | 0.4933 | 0.2764 | true | false | improved |
| single_head_mixed_replay | 4 | incumbent_proxy_disagreement-022 | 3 | 3 | 3 | 0.8333 | 0.9292 | 0.4996 | true | false | improved |
| single_head_mixed_replay | 4 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9635 | 0.9792 | 0.8549 | true | false | improved |
| single_head_mixed_replay | 4 | incumbent_proxy_disagreement-025 | 2 | 2 | 2 | 0.7266 | 0.7283 | 0.7111 | true | false | improved |
| single_head_mixed_replay | 4 | incumbent_proxy_disagreement-035 | 3 | 4 | 3 | 0.4661 | 0.6358 | 0.7821 | true | false | improved |
| classic_only_replay | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9609 | 0.9833 | 0.7443 | true | false | improved |
| classic_only_replay | 1 | incumbent_proxy_disagreement-014 | 4 | 5 | 1 | 0.0521 | 0.0167 | 0.2273 | true | true | improved,damaged |
| classic_only_replay | 1 | incumbent_proxy_disagreement-022 | 3 | 3 | 3 | 0.5755 | 0.4275 | 0.2307 | true | false | improved |
| classic_only_replay | 1 | incumbent_proxy_disagreement-024 | 2 | 3 | 3 | 0.0234 | 0.0108 | 0.1506 | true | false | improved |
| classic_only_replay | 1 | incumbent_proxy_disagreement-025 | 2 | 2 | 2 | 0.6927 | 0.4875 | 0.3184 | true | false | improved |
| classic_only_replay | 1 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9714 | 0.9867 | 0.6608 | true | false | improved |
| classic_only_replay | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9219 | 0.9700 | 0.7722 | false | false | no_change |
| classic_only_replay | 2 | incumbent_proxy_disagreement-014 | 4 | 2 | 2 | 0.1406 | 0.0467 | 0.1690 | true | true | improved,damaged |
| classic_only_replay | 2 | incumbent_proxy_disagreement-022 | 3 | 2 | 2 | 0.0964 | 0.0425 | 0.3153 | true | false | improved |
| classic_only_replay | 2 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.9089 | 0.9625 | 0.2418 | true | false | improved |
| classic_only_replay | 2 | incumbent_proxy_disagreement-025 | 2 | 2 | 2 | 0.9896 | 0.9967 | 0.9461 | true | false | improved |
| classic_only_replay | 2 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9922 | 0.6367 | 0.8957 | true | false | improved |
| classic_only_replay | 4 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.8906 | 0.9417 | 0.6760 | false | false | no_change |
| classic_only_replay | 4 | incumbent_proxy_disagreement-014 | 4 | 4 | 4 | 0.6953 | 0.5542 | 0.5643 | true | false | improved |
| classic_only_replay | 4 | incumbent_proxy_disagreement-022 | 3 | 2 | 2 | 0.3438 | 0.1142 | 0.7739 | true | false | improved |
| classic_only_replay | 4 | incumbent_proxy_disagreement-024 | 2 | 2 | 2 | 0.4141 | 0.8008 | 0.1533 | true | false | improved |
| classic_only_replay | 4 | incumbent_proxy_disagreement-025 | 2 | 2 | 2 | 0.8802 | 0.7242 | 0.7210 | true | false | improved |
| classic_only_replay | 4 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9922 | 0.9867 | 0.9221 | true | false | improved |
| puct_only_replay | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.9427 | 0.9733 | 0.4013 | true | false | improved |
| puct_only_replay | 1 | incumbent_proxy_disagreement-014 | 4 | 1 | 1 | 0.0026 | 0.0008 | 0.0235 | false | true | damaged |
| puct_only_replay | 1 | incumbent_proxy_disagreement-022 | 3 | 2 | 2 | 0.0078 | 0.0025 | 0.0861 | true | false | improved |
| puct_only_replay | 1 | incumbent_proxy_disagreement-024 | 2 | 1 | 1 | 0.0026 | 0.0008 | 0.0169 | false | false | no_change |
| puct_only_replay | 1 | incumbent_proxy_disagreement-025 | 2 | 2 | 2 | 0.3958 | 0.8000 | 0.0230 | true | false | improved |
| puct_only_replay | 1 | incumbent_proxy_disagreement-035 | 3 | 4 | 4 | 0.0078 | 0.0033 | 0.0743 | false | false | no_change |
| puct_only_replay | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.8333 | 0.9400 | 0.2076 | false | true | damaged |
| puct_only_replay | 2 | incumbent_proxy_disagreement-014 | 4 | 1 | 1 | 0.0026 | 0.0008 | 0.0244 | false | true | damaged |
| puct_only_replay | 2 | incumbent_proxy_disagreement-022 | 3 | 1 | 2 | 0.1641 | 0.0717 | 0.1687 | true | false | improved |
| puct_only_replay | 2 | incumbent_proxy_disagreement-024 | 2 | 1 | 1 | 0.0026 | 0.0008 | 0.0152 | false | false | no_change |
| puct_only_replay | 2 | incumbent_proxy_disagreement-025 | 2 | 1 | 3 | 0.0026 | 0.2833 | 0.0174 | true | false | improved |
| puct_only_replay | 2 | incumbent_proxy_disagreement-035 | 3 | 5 | 1 | 0.0130 | 0.0058 | 0.1531 | true | false | improved |
| puct_only_replay | 4 | incumbent_proxy_disagreement-008 | 2 | 3 | 3 | 0.0182 | 0.0067 | 0.1137 | false | true | damaged |
| puct_only_replay | 4 | incumbent_proxy_disagreement-014 | 4 | 5 | 5 | 0.0026 | 0.0008 | 0.0120 | false | true | damaged |
| puct_only_replay | 4 | incumbent_proxy_disagreement-022 | 3 | 5 | 5 | 0.3698 | 0.1283 | 0.3250 | true | false | improved |
| puct_only_replay | 4 | incumbent_proxy_disagreement-024 | 2 | 1 | 1 | 0.0026 | 0.0008 | 0.0204 | false | false | no_change |
| puct_only_replay | 4 | incumbent_proxy_disagreement-025 | 2 | 3 | 3 | 0.0026 | 0.0008 | 0.0303 | true | true | improved,damaged |
| puct_only_replay | 4 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.9349 | 0.9708 | 0.4459 | true | false | improved |
| teacher_conditioned_probe | 1 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.6875 | 0.9000 | 0.2352 | false | true | damaged |
| teacher_conditioned_probe | 1 | incumbent_proxy_disagreement-014 | 4 | 4 | 1 | 0.6094 | 0.2183 | 0.1953 | true | true | improved,damaged |
| teacher_conditioned_probe | 1 | incumbent_proxy_disagreement-022 | 3 | 1 | 1 | 0.1927 | 0.0617 | 0.2190 | true | false | improved |
| teacher_conditioned_probe | 1 | incumbent_proxy_disagreement-024 | 2 | 3 | 3 | 0.0911 | 0.0292 | 0.1913 | true | false | improved |
| teacher_conditioned_probe | 1 | incumbent_proxy_disagreement-025 | 2 | 4 | 4 | 0.2057 | 0.1150 | 0.1954 | true | true | improved,damaged |
| teacher_conditioned_probe | 1 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.8021 | 0.9367 | 0.2715 | true | false | improved |
| teacher_conditioned_probe | 2 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.6328 | 0.8742 | 0.2378 | false | true | damaged |
| teacher_conditioned_probe | 2 | incumbent_proxy_disagreement-014 | 4 | 5 | 5 | 0.1484 | 0.0475 | 0.1910 | true | true | improved,damaged |
| teacher_conditioned_probe | 2 | incumbent_proxy_disagreement-022 | 3 | 5 | 5 | 0.1510 | 0.0483 | 0.2251 | true | false | improved |
| teacher_conditioned_probe | 2 | incumbent_proxy_disagreement-024 | 2 | 0 | 0 | 0.0234 | 0.0075 | 0.1972 | true | false | improved |
| teacher_conditioned_probe | 2 | incumbent_proxy_disagreement-025 | 2 | 3 | 4 | 0.0833 | 0.0275 | 0.2005 | true | true | improved,damaged |
| teacher_conditioned_probe | 2 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.4870 | 0.8150 | 0.2807 | true | false | improved |
| teacher_conditioned_probe | 4 | incumbent_proxy_disagreement-008 | 2 | 2 | 2 | 0.7266 | 0.8675 | 0.2435 | false | true | damaged |
| teacher_conditioned_probe | 4 | incumbent_proxy_disagreement-014 | 4 | 4 | 4 | 0.5885 | 0.8608 | 0.1811 | true | false | improved |
| teacher_conditioned_probe | 4 | incumbent_proxy_disagreement-022 | 3 | 5 | 5 | 0.1432 | 0.3525 | 0.2387 | true | false | improved |
| teacher_conditioned_probe | 4 | incumbent_proxy_disagreement-024 | 2 | 0 | 0 | 0.0312 | 0.0100 | 0.2102 | true | false | improved |
| teacher_conditioned_probe | 4 | incumbent_proxy_disagreement-025 | 2 | 4 | 4 | 0.0964 | 0.0308 | 0.2120 | true | true | improved,damaged |
| teacher_conditioned_probe | 4 | incumbent_proxy_disagreement-035 | 3 | 3 | 3 | 0.3906 | 0.7075 | 0.3037 | true | false | improved |

## 7. PUCT bucket results

| variant | epoch | row_id | preferred_move | selected_384 | selected_1200 | preferred_visit_share_384 | preferred_visit_share_1200 | preferred_policy_prob | improved_vs_current | damaged_vs_current | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_head_mixed_replay | 1 | incumbent_proxy_disagreement-007 | 3 | 3 | 2 | 0.4036 | 0.1633 | 0.1043 | false | true | damaged |
| single_head_mixed_replay | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.6875 | 0.5650 | 0.3583 | false | true | damaged |
| single_head_mixed_replay | 1 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.1302 | 0.0450 | 0.2074 | false | true | damaged |
| single_head_mixed_replay | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.7812 | 0.5533 | 0.6471 | false | true | damaged |
| single_head_mixed_replay | 1 | incumbent_proxy_disagreement-028 | 5 | 5 | 2 | 0.8255 | 0.2892 | 0.4179 | false | true | damaged |
| single_head_mixed_replay | 1 | incumbent_proxy_disagreement-032 | 1 | 0 | 0 | 0.1693 | 0.3025 | 0.2346 | false | true | damaged |
| single_head_mixed_replay | 1 | incumbent_proxy_disagreement-033 | 0 | 1 | 0 | 0.3281 | 0.5133 | 0.5401 | false | true | damaged |
| single_head_mixed_replay | 2 | incumbent_proxy_disagreement-007 | 3 | 3 | 3 | 0.6302 | 0.5958 | 0.1171 | true | false | improved |
| single_head_mixed_replay | 2 | incumbent_proxy_disagreement-009 | 3 | 5 | 3 | 0.3307 | 0.7033 | 0.6355 | false | true | damaged |
| single_head_mixed_replay | 2 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0495 | 0.0158 | 0.0849 | false | true | damaged |
| single_head_mixed_replay | 2 | incumbent_proxy_disagreement-026 | 5 | 2 | 0 | 0.2057 | 0.0708 | 0.4725 | false | true | damaged |
| single_head_mixed_replay | 2 | incumbent_proxy_disagreement-028 | 5 | 4 | 4 | 0.4375 | 0.2958 | 0.2710 | false | true | damaged |
| single_head_mixed_replay | 2 | incumbent_proxy_disagreement-032 | 1 | 2 | 0 | 0.3099 | 0.1108 | 0.3439 | true | true | improved,damaged |
| single_head_mixed_replay | 2 | incumbent_proxy_disagreement-033 | 0 | 1 | 1 | 0.1693 | 0.0742 | 0.6832 | false | true | damaged |
| single_head_mixed_replay | 4 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | 0.2240 | 0.0958 | 0.1622 | false | true | damaged |
| single_head_mixed_replay | 4 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.8854 | 0.8467 | 0.6099 | false | true | damaged |
| single_head_mixed_replay | 4 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.2135 | 0.0683 | 0.2918 | false | true | damaged |
| single_head_mixed_replay | 4 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.9922 | 0.5833 | 0.9130 | false | true | damaged |
| single_head_mixed_replay | 4 | incumbent_proxy_disagreement-028 | 5 | 5 | 3 | 0.9531 | 0.4117 | 0.7685 | false | true | damaged |
| single_head_mixed_replay | 4 | incumbent_proxy_disagreement-032 | 1 | 1 | 1 | 0.5781 | 0.8217 | 0.4044 | true | false | improved |
| single_head_mixed_replay | 4 | incumbent_proxy_disagreement-033 | 0 | 0 | 0 | 0.8021 | 0.5567 | 0.7666 | true | true | improved,damaged |
| classic_only_replay | 1 | incumbent_proxy_disagreement-007 | 3 | 0 | 0 | 0.0651 | 0.0208 | 0.0992 | false | true | damaged |
| classic_only_replay | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.8125 | 0.9375 | 0.6279 | false | false | no_change |
| classic_only_replay | 1 | incumbent_proxy_disagreement-021 | 5 | 2 | 3 | 0.2995 | 0.1083 | 0.0839 | false | true | damaged |
| classic_only_replay | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.8594 | 0.7292 | 0.4800 | false | true | damaged |
| classic_only_replay | 1 | incumbent_proxy_disagreement-028 | 5 | 5 | 2 | 0.5573 | 0.3458 | 0.2016 | false | true | damaged |
| classic_only_replay | 1 | incumbent_proxy_disagreement-032 | 1 | 2 | 2 | 0.0026 | 0.0008 | 0.0359 | false | true | damaged |
| classic_only_replay | 1 | incumbent_proxy_disagreement-033 | 0 | 2 | 3 | 0.0312 | 0.0100 | 0.1329 | false | true | damaged |
| classic_only_replay | 2 | incumbent_proxy_disagreement-007 | 3 | 0 | 0 | 0.2266 | 0.0725 | 0.0694 | false | true | damaged |
| classic_only_replay | 2 | incumbent_proxy_disagreement-009 | 3 | 1 | 1 | 0.4245 | 0.1900 | 0.8002 | false | true | damaged |
| classic_only_replay | 2 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0008 | 0.0232 | false | true | damaged |
| classic_only_replay | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.5312 | 0.8217 | 0.1358 | false | true | damaged |
| classic_only_replay | 2 | incumbent_proxy_disagreement-028 | 5 | 4 | 2 | 0.0026 | 0.3242 | 0.0351 | false | true | damaged |
| classic_only_replay | 2 | incumbent_proxy_disagreement-032 | 1 | 2 | 2 | 0.0026 | 0.1500 | 0.0518 | false | true | damaged |
| classic_only_replay | 2 | incumbent_proxy_disagreement-033 | 0 | 2 | 2 | 0.0026 | 0.0008 | 0.0352 | false | true | damaged |
| classic_only_replay | 4 | incumbent_proxy_disagreement-007 | 3 | 2 | 0 | 0.2344 | 0.0767 | 0.1299 | false | true | damaged |
| classic_only_replay | 4 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.9922 | 0.9900 | 0.8998 | true | false | improved |
| classic_only_replay | 4 | incumbent_proxy_disagreement-021 | 5 | 3 | 3 | 0.0026 | 0.0008 | 0.0110 | false | true | damaged |
| classic_only_replay | 4 | incumbent_proxy_disagreement-026 | 5 | 2 | 2 | 0.0026 | 0.3050 | 0.0774 | false | true | damaged |
| classic_only_replay | 4 | incumbent_proxy_disagreement-028 | 5 | 4 | 4 | 0.0026 | 0.0008 | 0.0141 | false | true | damaged |
| classic_only_replay | 4 | incumbent_proxy_disagreement-032 | 1 | 2 | 2 | 0.0026 | 0.0233 | 0.0612 | false | true | damaged |
| classic_only_replay | 4 | incumbent_proxy_disagreement-033 | 0 | 2 | 2 | 0.1146 | 0.0508 | 0.0982 | false | true | damaged |
| puct_only_replay | 1 | incumbent_proxy_disagreement-007 | 3 | 5 | 5 | 0.2995 | 0.0958 | 0.2941 | false | true | damaged |
| puct_only_replay | 1 | incumbent_proxy_disagreement-009 | 3 | 1 | 3 | 0.2630 | 0.7175 | 0.1919 | false | true | damaged |
| puct_only_replay | 1 | incumbent_proxy_disagreement-021 | 5 | 3 | 5 | 0.3177 | 0.5100 | 0.2205 | false | true | damaged |
| puct_only_replay | 1 | incumbent_proxy_disagreement-026 | 5 | 2 | 2 | 0.3490 | 0.1200 | 0.7549 | false | true | damaged |
| puct_only_replay | 1 | incumbent_proxy_disagreement-028 | 5 | 1 | 1 | 0.3464 | 0.1108 | 0.5382 | false | true | damaged |
| puct_only_replay | 1 | incumbent_proxy_disagreement-032 | 1 | 2 | 2 | 0.1797 | 0.1525 | 0.7932 | false | true | damaged |
| puct_only_replay | 1 | incumbent_proxy_disagreement-033 | 0 | 1 | 1 | 0.0833 | 0.0267 | 0.1783 | false | true | damaged |
| puct_only_replay | 2 | incumbent_proxy_disagreement-007 | 3 | 3 | 3 | 0.9531 | 0.6658 | 0.6025 | true | false | improved |
| puct_only_replay | 2 | incumbent_proxy_disagreement-009 | 3 | 5 | 3 | 0.0729 | 0.5167 | 0.3815 | false | true | damaged |
| puct_only_replay | 2 | incumbent_proxy_disagreement-021 | 5 | 5 | 5 | 0.5339 | 0.4817 | 0.2855 | false | true | damaged |
| puct_only_replay | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 3 | 0.6354 | 0.2033 | 0.8553 | false | true | damaged |
| puct_only_replay | 2 | incumbent_proxy_disagreement-028 | 5 | 5 | 3 | 0.6172 | 0.1975 | 0.7155 | false | true | damaged |
| puct_only_replay | 2 | incumbent_proxy_disagreement-032 | 1 | 1 | 3 | 0.3672 | 0.3717 | 0.8385 | true | true | improved,damaged |
| puct_only_replay | 2 | incumbent_proxy_disagreement-033 | 0 | 1 | 1 | 0.0911 | 0.0300 | 0.1611 | false | true | damaged |
| puct_only_replay | 4 | incumbent_proxy_disagreement-007 | 3 | 3 | 3 | 0.9505 | 0.9833 | 0.7618 | true | false | improved |
| puct_only_replay | 4 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.7839 | 0.6058 | 0.7012 | false | true | damaged |
| puct_only_replay | 4 | incumbent_proxy_disagreement-021 | 5 | 5 | 5 | 0.7240 | 0.6833 | 0.5747 | false | true | damaged |
| puct_only_replay | 4 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.9896 | 0.9608 | 0.9350 | false | false | no_change |
| puct_only_replay | 4 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.9870 | 0.6125 | 0.8951 | true | true | improved,damaged |
| puct_only_replay | 4 | incumbent_proxy_disagreement-032 | 1 | 1 | 3 | 0.9245 | 0.3958 | 0.8552 | true | true | improved,damaged |
| puct_only_replay | 4 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | 0.0651 | 0.0550 | 0.1133 | false | true | damaged |
| teacher_conditioned_probe | 1 | incumbent_proxy_disagreement-007 | 3 | 2 | 2 | 0.1068 | 0.0342 | 0.2670 | false | true | damaged |
| teacher_conditioned_probe | 1 | incumbent_proxy_disagreement-009 | 3 | 3 | 1 | 0.4479 | 0.2525 | 0.2688 | false | true | damaged |
| teacher_conditioned_probe | 1 | incumbent_proxy_disagreement-021 | 5 | 1 | 2 | 0.0391 | 0.0125 | 0.2153 | false | true | damaged |
| teacher_conditioned_probe | 1 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.7552 | 0.9167 | 0.2241 | false | true | damaged |
| teacher_conditioned_probe | 1 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.6406 | 0.8825 | 0.1886 | false | true | damaged |
| teacher_conditioned_probe | 1 | incumbent_proxy_disagreement-032 | 1 | 3 | 3 | 0.0130 | 0.0133 | 0.1995 | false | true | damaged |
| teacher_conditioned_probe | 1 | incumbent_proxy_disagreement-033 | 0 | 2 | 2 | 0.0417 | 0.0142 | 0.1947 | false | true | damaged |
| teacher_conditioned_probe | 2 | incumbent_proxy_disagreement-007 | 3 | 5 | 5 | 0.0443 | 0.0142 | 0.2729 | false | true | damaged |
| teacher_conditioned_probe | 2 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.8568 | 0.9533 | 0.2781 | false | false | no_change |
| teacher_conditioned_probe | 2 | incumbent_proxy_disagreement-021 | 5 | 5 | 5 | 0.7812 | 0.9300 | 0.2185 | false | true | damaged |
| teacher_conditioned_probe | 2 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.6979 | 0.8958 | 0.2312 | false | true | damaged |
| teacher_conditioned_probe | 2 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.6016 | 0.8725 | 0.1956 | false | true | damaged |
| teacher_conditioned_probe | 2 | incumbent_proxy_disagreement-032 | 1 | 3 | 3 | 0.0547 | 0.0217 | 0.1956 | false | true | damaged |
| teacher_conditioned_probe | 2 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | 0.0443 | 0.0142 | 0.1928 | false | true | damaged |
| teacher_conditioned_probe | 4 | incumbent_proxy_disagreement-007 | 3 | 5 | 5 | 0.0443 | 0.0142 | 0.2855 | false | true | damaged |
| teacher_conditioned_probe | 4 | incumbent_proxy_disagreement-009 | 3 | 3 | 3 | 0.8594 | 0.9542 | 0.2970 | false | false | no_change |
| teacher_conditioned_probe | 4 | incumbent_proxy_disagreement-021 | 5 | 5 | 5 | 0.7891 | 0.9308 | 0.2274 | false | true | damaged |
| teacher_conditioned_probe | 4 | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.7109 | 0.9075 | 0.2483 | false | true | damaged |
| teacher_conditioned_probe | 4 | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.5729 | 0.8633 | 0.2130 | false | true | damaged |
| teacher_conditioned_probe | 4 | incumbent_proxy_disagreement-032 | 1 | 3 | 0 | 0.0599 | 0.0225 | 0.1878 | false | true | damaged |
| teacher_conditioned_probe | 4 | incumbent_proxy_disagreement-033 | 0 | 3 | 3 | 0.0417 | 0.0133 | 0.1861 | false | true | damaged |

## 8. Cross-teacher interference analysis

| variant | epoch | classic_gain | puct_gain | classic_damage | puct_damage | excluded_drift | teacher_separation | interference_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_head_mixed_replay | 4 | 5 | 2 | 0 | 6 | 4 | 1 | 5 |
| classic_only_replay | 4 | 5 | 1 | 0 | 6 | 3 | 0 | 4 |
| puct_only_replay | 4 | 3 | 3 | 3 | 5 | 2 | -2 | 4 |
| teacher_conditioned_probe | 4 | 5 | 0 | 2 | 6 | 2 | -3 | 3 |

## 9. Decision

| classification | supporting_evidence | rejected_alternatives | next_action |
| --- | --- | --- | --- |
| teacher_policy_conflict_not_representable_by_small_probe | Mixed replay interference_score=5, all variants show conflict | no variant resolved the teacher conflict | stop training on incumbent_proxy; use teacher-policy split only for evaluation, and improve mining/scoring from fresh positions |

## 10. Optional gradient check

- Gradient check was computed at initialization.
- Strongest negative policy-gradient cosine pairs (threshold < -0.02): 10
  - `incumbent_proxy_disagreement-014` vs `incumbent_proxy_disagreement-021`: cosine=-0.749159
  - `incumbent_proxy_disagreement-022` vs `incumbent_proxy_disagreement-021`: cosine=-0.735593
  - `incumbent_proxy_disagreement-014` vs `incumbent_proxy_disagreement-028`: cosine=-0.551857
  - `incumbent_proxy_disagreement-022` vs `incumbent_proxy_disagreement-028`: cosine=-0.487305
  - `incumbent_proxy_disagreement-035` vs `incumbent_proxy_disagreement-026`: cosine=-0.321776

## 11. Interpretation

- Primary hypothesis test: teacher-aware diagnostic objective may preserve both local behaviors better than single-head replay.
- Classification: `teacher_policy_conflict_not_representable_by_small_probe`.
- Next action: stop training on incumbent_proxy; use teacher-policy split only for evaluation, and improve mining/scoring from fresh positions

## 12. Exactly one recommended next action

Recommendation: **stop training on incumbent_proxy; use teacher-policy split only for evaluation, and improve mining/scoring from fresh positions**

### Acceptance criteria

- No arena was run.
- No model was promoted.
- Active references were not mutated.
- Proposed patch bundle from PR #69 was not applied.
- Excluded rows were never training targets.
- Single-head mixed replay was treated as a diagnostic baseline, not a candidate.
- Teacher-conditioned variants were experimental only.

