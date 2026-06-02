# AlphaZero-lite Incumbent Proxy Disagreement Reference Adjudication Results

## Context

- Focused suspicious rows came from `/tmp/azlite_incumbent_proxy_value_backup_audit/incumbent_proxy_value_backup_audit_summary.json`.
- During the adjudication run, corrected references were treated as read-only inputs from `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- This checked-in report predates the later approved fixture update for `incumbent_proxy_disagreement-021` that is also included in this PR.
- No training, arena, promotion, or fixture mutation was performed.

## Focus rows

| row_id | corrected_reference_move | current_selected_1200 | classic_majority_move | majority_fraction | stable | adjudication | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-007 | 0 | 3 | 0 | 1.0000 | true | reference_upheld | deep classic root search keeps the corrected reference |
| incumbent_proxy_disagreement-009 | 0 | 3 | 0 | 1.0000 | true | reference_upheld | deep classic root search keeps the corrected reference |
| incumbent_proxy_disagreement-021 | 3 | 5 | 2 | 1.0000 | true | reference_overturned | deep classic root search consistently prefers a different move |
| incumbent_proxy_disagreement-023 | 2 | 1 | 2 | 1.0000 | true | reference_upheld | deep classic root search keeps the corrected reference |
| incumbent_proxy_disagreement-024 | 2 | 1 | 2 | 1.0000 | true | reference_upheld | deep classic root search keeps the corrected reference |

## Deep Classic MCTS rows

| row_id | budget | seed | selected_move | top_q_move | corrected_reference_q | selected_q | selected_minus_reference_q_margin | corrected_reference_visits | selected_visits | selected_is_reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-007 | 5000 | 11 | 0 | 0 | 0.9003 | 0.9003 | 0.0000 | 3551 | 3551 | true |
| incumbent_proxy_disagreement-007 | 5000 | 23 | 0 | 0 | 0.9013 | 0.9013 | 0.0000 | 3568 | 3568 | true |
| incumbent_proxy_disagreement-007 | 5000 | 37 | 0 | 0 | 0.9032 | 0.9032 | 0.0000 | 3565 | 3565 | true |
| incumbent_proxy_disagreement-007 | 5000 | 42 | 0 | 0 | 0.9030 | 0.9030 | 0.0000 | 3712 | 3712 | true |
| incumbent_proxy_disagreement-007 | 5000 | 101 | 0 | 0 | 0.9079 | 0.9079 | 0.0000 | 3603 | 3603 | true |
| incumbent_proxy_disagreement-007 | 10000 | 11 | 0 | 0 | 0.9041 | 0.9041 | 0.0000 | 7987 | 7987 | true |
| incumbent_proxy_disagreement-007 | 10000 | 23 | 0 | 0 | 0.9075 | 0.9075 | 0.0000 | 7997 | 7997 | true |
| incumbent_proxy_disagreement-007 | 10000 | 37 | 0 | 0 | 0.9071 | 0.9071 | 0.0000 | 7945 | 7945 | true |
| incumbent_proxy_disagreement-007 | 10000 | 42 | 0 | 0 | 0.9061 | 0.9061 | 0.0000 | 7840 | 7840 | true |
| incumbent_proxy_disagreement-007 | 10000 | 101 | 0 | 0 | 0.9053 | 0.9053 | 0.0000 | 7899 | 7899 | true |
| incumbent_proxy_disagreement-007 | 20000 | 11 | 0 | 0 | 0.9126 | 0.9126 | 0.0000 | 17522 | 17522 | true |
| incumbent_proxy_disagreement-007 | 20000 | 23 | 0 | 0 | 0.9154 | 0.9154 | 0.0000 | 17475 | 17475 | true |
| incumbent_proxy_disagreement-007 | 20000 | 37 | 0 | 0 | 0.9159 | 0.9159 | 0.0000 | 17282 | 17282 | true |
| incumbent_proxy_disagreement-007 | 20000 | 42 | 0 | 0 | 0.9172 | 0.9172 | 0.0000 | 17382 | 17382 | true |
| incumbent_proxy_disagreement-007 | 20000 | 101 | 0 | 0 | 0.9135 | 0.9135 | 0.0000 | 17471 | 17471 | true |
| incumbent_proxy_disagreement-009 | 5000 | 11 | 0 | 0 | 0.7698 | 0.7698 | 0.0000 | 3237 | 3237 | true |
| incumbent_proxy_disagreement-009 | 5000 | 23 | 0 | 0 | 0.7752 | 0.7752 | 0.0000 | 3558 | 3558 | true |
| incumbent_proxy_disagreement-009 | 5000 | 37 | 0 | 0 | 0.7815 | 0.7815 | 0.0000 | 3510 | 3510 | true |
| incumbent_proxy_disagreement-009 | 5000 | 42 | 0 | 0 | 0.7766 | 0.7766 | 0.0000 | 3442 | 3442 | true |
| incumbent_proxy_disagreement-009 | 5000 | 101 | 0 | 0 | 0.7768 | 0.7768 | 0.0000 | 3670 | 3670 | true |
| incumbent_proxy_disagreement-009 | 10000 | 11 | 0 | 0 | 0.8491 | 0.8491 | 0.0000 | 8237 | 8237 | true |
| incumbent_proxy_disagreement-009 | 10000 | 23 | 0 | 0 | 0.8448 | 0.8448 | 0.0000 | 8555 | 8555 | true |
| incumbent_proxy_disagreement-009 | 10000 | 37 | 0 | 0 | 0.8518 | 0.8518 | 0.0000 | 8510 | 8510 | true |
| incumbent_proxy_disagreement-009 | 10000 | 42 | 0 | 0 | 0.8467 | 0.8467 | 0.0000 | 8442 | 8442 | true |
| incumbent_proxy_disagreement-009 | 10000 | 101 | 0 | 0 | 0.8434 | 0.8434 | 0.0000 | 8670 | 8670 | true |
| incumbent_proxy_disagreement-009 | 20000 | 11 | 0 | 0 | 0.8967 | 0.8967 | 0.0000 | 18237 | 18237 | true |
| incumbent_proxy_disagreement-009 | 20000 | 23 | 0 | 0 | 0.8953 | 0.8953 | 0.0000 | 18555 | 18555 | true |
| incumbent_proxy_disagreement-009 | 20000 | 37 | 0 | 0 | 0.8974 | 0.8974 | 0.0000 | 18510 | 18510 | true |
| incumbent_proxy_disagreement-009 | 20000 | 42 | 0 | 0 | 0.8955 | 0.8955 | 0.0000 | 18442 | 18442 | true |
| incumbent_proxy_disagreement-009 | 20000 | 101 | 0 | 0 | 0.8946 | 0.8946 | 0.0000 | 18670 | 18670 | true |
| incumbent_proxy_disagreement-021 | 5000 | 11 | 3 | 3 | 0.4720 | 0.4720 | 0.0000 | 2407 | 2407 | true |
| incumbent_proxy_disagreement-021 | 5000 | 23 | 2 | 2 | 0.4042 | 0.6759 | 0.2717 | 767 | 3036 | false |
| incumbent_proxy_disagreement-021 | 5000 | 37 | 2 | 2 | 0.3963 | 0.6648 | 0.2685 | 747 | 2840 | false |
| incumbent_proxy_disagreement-021 | 5000 | 42 | 2 | 2 | 0.4866 | 0.6495 | 0.1630 | 1303 | 2605 | false |
| incumbent_proxy_disagreement-021 | 5000 | 101 | 2 | 2 | 0.3644 | 0.6823 | 0.3179 | 568 | 3274 | false |
| incumbent_proxy_disagreement-021 | 10000 | 11 | 2 | 2 | 0.4720 | 0.7410 | 0.2690 | 2407 | 5799 | false |
| incumbent_proxy_disagreement-021 | 10000 | 23 | 2 | 2 | 0.4042 | 0.7865 | 0.3823 | 767 | 8036 | false |
| incumbent_proxy_disagreement-021 | 10000 | 37 | 2 | 2 | 0.3963 | 0.7736 | 0.3773 | 747 | 7840 | false |
| incumbent_proxy_disagreement-021 | 10000 | 42 | 2 | 2 | 0.4866 | 0.7728 | 0.2862 | 1303 | 7605 | false |
| incumbent_proxy_disagreement-021 | 10000 | 101 | 2 | 2 | 0.3644 | 0.7764 | 0.4120 | 568 | 8274 | false |
| incumbent_proxy_disagreement-021 | 20000 | 11 | 2 | 2 | 0.4720 | 0.8352 | 0.3632 | 2407 | 15799 | false |
| incumbent_proxy_disagreement-021 | 20000 | 23 | 2 | 2 | 0.4042 | 0.8442 | 0.4400 | 767 | 18036 | false |
| incumbent_proxy_disagreement-021 | 20000 | 37 | 2 | 2 | 0.3963 | 0.8360 | 0.4398 | 747 | 17840 | false |
| incumbent_proxy_disagreement-021 | 20000 | 42 | 2 | 2 | 0.4866 | 0.8389 | 0.3523 | 1303 | 17605 | false |
| incumbent_proxy_disagreement-021 | 20000 | 101 | 2 | 2 | 0.3644 | 0.8355 | 0.4711 | 568 | 18274 | false |
| incumbent_proxy_disagreement-023 | 5000 | 11 | 2 | 2 | 0.3373 | 0.3373 | 0.0000 | 4225 | 4225 | true |
| incumbent_proxy_disagreement-023 | 5000 | 23 | 2 | 2 | 0.3733 | 0.3733 | 0.0000 | 4720 | 4720 | true |
| incumbent_proxy_disagreement-023 | 5000 | 37 | 2 | 2 | 0.3465 | 0.3465 | 0.0000 | 4485 | 4485 | true |
| incumbent_proxy_disagreement-023 | 5000 | 42 | 2 | 2 | 0.3408 | 0.3408 | 0.0000 | 4665 | 4665 | true |
| incumbent_proxy_disagreement-023 | 5000 | 101 | 2 | 2 | 0.3789 | 0.3789 | 0.0000 | 4642 | 4642 | true |
| incumbent_proxy_disagreement-023 | 10000 | 11 | 2 | 2 | 0.5143 | 0.5143 | 0.0000 | 9225 | 9225 | true |
| incumbent_proxy_disagreement-023 | 10000 | 23 | 2 | 2 | 0.5336 | 0.5336 | 0.0000 | 9720 | 9720 | true |
| incumbent_proxy_disagreement-023 | 10000 | 37 | 2 | 2 | 0.5514 | 0.5514 | 0.0000 | 9485 | 9485 | true |
| incumbent_proxy_disagreement-023 | 10000 | 42 | 2 | 2 | 0.4646 | 0.4646 | 0.0000 | 9665 | 9665 | true |
| incumbent_proxy_disagreement-023 | 10000 | 101 | 2 | 2 | 0.5430 | 0.5430 | 0.0000 | 9642 | 9642 | true |
| incumbent_proxy_disagreement-023 | 20000 | 11 | 2 | 2 | 0.6670 | 0.6670 | 0.0000 | 19225 | 19225 | true |
| incumbent_proxy_disagreement-023 | 20000 | 23 | 2 | 2 | 0.6684 | 0.6684 | 0.0000 | 19720 | 19720 | true |
| incumbent_proxy_disagreement-023 | 20000 | 37 | 2 | 2 | 0.6844 | 0.6844 | 0.0000 | 19485 | 19485 | true |
| incumbent_proxy_disagreement-023 | 20000 | 42 | 2 | 2 | 0.6313 | 0.6313 | 0.0000 | 19665 | 19665 | true |
| incumbent_proxy_disagreement-023 | 20000 | 101 | 2 | 2 | 0.6789 | 0.6789 | 0.0000 | 19642 | 19642 | true |
| incumbent_proxy_disagreement-024 | 5000 | 11 | 2 | 2 | 0.3175 | 0.3175 | 0.0000 | 4576 | 4576 | true |
| incumbent_proxy_disagreement-024 | 5000 | 23 | 2 | 2 | 0.3521 | 0.3521 | 0.0000 | 4516 | 4516 | true |
| incumbent_proxy_disagreement-024 | 5000 | 37 | 2 | 2 | 0.3598 | 0.3598 | 0.0000 | 4603 | 4603 | true |
| incumbent_proxy_disagreement-024 | 5000 | 42 | 2 | 2 | 0.3764 | 0.3764 | 0.0000 | 4617 | 4617 | true |
| incumbent_proxy_disagreement-024 | 5000 | 101 | 2 | 2 | 0.3585 | 0.3585 | 0.0000 | 4541 | 4541 | true |
| incumbent_proxy_disagreement-024 | 10000 | 11 | 2 | 2 | 0.5382 | 0.5382 | 0.0000 | 9576 | 9576 | true |
| incumbent_proxy_disagreement-024 | 10000 | 23 | 2 | 2 | 0.5587 | 0.5587 | 0.0000 | 9516 | 9516 | true |
| incumbent_proxy_disagreement-024 | 10000 | 37 | 2 | 2 | 0.5701 | 0.5701 | 0.0000 | 9603 | 9603 | true |
| incumbent_proxy_disagreement-024 | 10000 | 42 | 2 | 2 | 0.5660 | 0.5660 | 0.0000 | 9617 | 9617 | true |
| incumbent_proxy_disagreement-024 | 10000 | 101 | 2 | 2 | 0.5508 | 0.5508 | 0.0000 | 9541 | 9541 | true |
| incumbent_proxy_disagreement-024 | 20000 | 11 | 2 | 2 | 0.6619 | 0.6619 | 0.0000 | 19576 | 19576 | true |
| incumbent_proxy_disagreement-024 | 20000 | 23 | 2 | 2 | 0.6728 | 0.6728 | 0.0000 | 19516 | 19516 | true |
| incumbent_proxy_disagreement-024 | 20000 | 37 | 2 | 2 | 0.7144 | 0.7144 | 0.0000 | 19603 | 19603 | true |
| incumbent_proxy_disagreement-024 | 20000 | 42 | 2 | 2 | 0.7107 | 0.7107 | 0.0000 | 19617 | 19617 | true |
| incumbent_proxy_disagreement-024 | 20000 | 101 | 2 | 2 | 0.6678 | 0.6678 | 0.0000 | 19541 | 19541 | true |

## Suggested non-mutating patch

| row_id | old_reference_move | suggested_reference_move | status | notes |
| --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-007 | 0 | 0 | reference_upheld | deep classic root search keeps the corrected reference |
| incumbent_proxy_disagreement-009 | 0 | 0 | reference_upheld | deep classic root search keeps the corrected reference |
| incumbent_proxy_disagreement-021 | 3 | 2 | reference_overturned | deep classic root search consistently prefers a different move |
| incumbent_proxy_disagreement-023 | 2 | 2 | reference_upheld | deep classic root search keeps the corrected reference |
| incumbent_proxy_disagreement-024 | 2 | 2 | reference_upheld | deep classic root search keeps the corrected reference |

## Exactly one recommended next action

Recommendation: **prepare a non-mutating candidate reference patch for the overturned incumbent_proxy_disagreement rows, then review it before any training.**
