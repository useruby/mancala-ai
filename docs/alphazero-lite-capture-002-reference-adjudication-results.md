# AlphaZero-lite Capture 002 Reference Adjudication Results

## Context

- No training, no arena, no promotion, and no model-artifact changes were run.
- Summary artifact: `/tmp/azlite_capture_002_reference_adjudication/capture_002_reference_adjudication_summary.json`.
- Candidate reference artifact: `/tmp/azlite_capture_002_reference_adjudication/incumbent_forensic_suite_v1_references_adjudicated.json`.
- Patch artifact: `-`.

## Why this audit was needed

- PR #30 showed that the current root search, child-afterstate ordering, and direct child PUCT ordering all preferred move `2` over old reference move `4` on `capture_available-002`.
- Earlier policy-target work still treated `002` as a model/search failure against the older move-`4` label.
- This run checks whether `002` is a true failure or a bad/unstable reference target.

## Row and move consequence summary

| row_id | state_source | legal_moves | old_reference_move | current_known_selected_move | old_reference_state_matches_row | notes |
| --- | --- | --- | --- | --- | --- | --- |
| capture_available-002 | fixture | `[0, 1, 2, 3, 4]` | 4 | 2 | false | old_reference_state_mismatch |
| capture_available-003 | fixture | `[0, 1, 2, 3, 4]` | 1 | 2 | false | old_reference_state_mismatch |
| capture_available-006 | old_reference_suite_fallback | `[0, 1, 2, 3, 4]` | 2 | 2 | true | row_loaded_from_old_suite_fallback |
| capture_available-007 | fixture | `[0, 1, 2, 3, 4]` | 1 | 2 | false | old_reference_state_mismatch |
| capture_available-008 | fixture | `[0, 1, 2, 3, 4]` | 1 | 1 | false | old_reference_state_mismatch |

| row_id | move | gives_extra_turn | produces_capture | capture_count | immediate_store_delta | side_to_move_after | game_over_after_move | remaining_seeds_after_move | child_tablebase_solvable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capture_available-002 | 0 | false | true | 2 | 2 | 0 | false | 44 | false |
| capture_available-002 | 1 | false | true | 2 | 2 | 0 | false | 44 | false |
| capture_available-002 | 2 | true | false | 0 | 1 | 1 | false | 45 | false |
| capture_available-002 | 3 | false | false | 0 | 1 | 0 | false | 45 | false |
| capture_available-002 | 4 | false | false | 0 | 1 | 0 | false | 45 | false |
| capture_available-003 | 0 | false | true | 2 | 2 | 0 | false | 44 | false |
| capture_available-003 | 1 | true | false | 0 | 1 | 1 | false | 45 | false |
| capture_available-003 | 2 | true | false | 0 | 1 | 1 | false | 45 | false |
| capture_available-003 | 3 | false | false | 0 | 1 | 0 | false | 45 | false |
| capture_available-003 | 4 | false | false | 0 | 1 | 0 | false | 45 | false |
| capture_available-006 | 0 | false | true | 2 | 2 | 0 | false | 43 | false |
| capture_available-006 | 1 | false | true | 2 | 2 | 0 | false | 43 | false |
| capture_available-006 | 2 | true | false | 0 | 1 | 1 | false | 44 | false |
| capture_available-006 | 3 | false | false | 0 | 1 | 0 | false | 44 | false |
| capture_available-006 | 4 | false | false | 0 | 1 | 0 | false | 44 | false |
| capture_available-007 | 0 | false | true | 2 | 2 | 0 | false | 44 | false |
| capture_available-007 | 1 | true | false | 0 | 1 | 1 | false | 45 | false |
| capture_available-007 | 2 | true | false | 0 | 1 | 1 | false | 45 | false |
| capture_available-007 | 3 | false | false | 0 | 1 | 0 | false | 45 | false |
| capture_available-007 | 4 | false | false | 0 | 1 | 0 | false | 45 | false |
| capture_available-008 | 0 | false | true | 2 | 2 | 0 | false | 44 | false |
| capture_available-008 | 1 | true | false | 0 | 1 | 1 | false | 45 | false |
| capture_available-008 | 2 | false | false | 0 | 1 | 0 | false | 45 | false |
| capture_available-008 | 3 | false | false | 0 | 1 | 0 | false | 45 | false |
| capture_available-008 | 4 | false | false | 0 | 1 | 0 | false | 45 | false |

For `capture_available-002`:
- Move `2` child state: `{"current_player": 1, "opponent_pits": [5, 4, 0, 5, 5, 1], "opponent_store": 2, "player_pits": [1, 0, 7, 6, 6, 5], "player_store": 1}`
- Move `4` child state: `{"current_player": 0, "opponent_pits": [5, 4, 4, 4, 0, 1], "opponent_store": 2, "player_pits": [2, 1, 7, 6, 6, 5], "player_store": 1}`
- Move `2` perspective conversion: `identity`
- Move `4` perspective conversion: `sign_flip`

## Root ClassicMCTS multi-budget/multi-seed adjudication

| row_id | budget | seed | selected_move | old_reference_move | disputed_move | visits_move_2 | visits_move_4 | value_move_2 | value_move_4 | value_4_minus_2 | top1_margin | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capture_available-002 | 1200 | 11 | 2 | 4 | 2 | 405 | 267 | 0.6444 | 0.5581 | -0.0864 | 0.075 | old_reference_state_mismatch |
| capture_available-002 | 1200 | 23 | 2 | 4 | 2 | 457 | 212 | 0.6477 | 0.4811 | -0.1666 | 0.1413 | old_reference_state_mismatch |
| capture_available-002 | 1200 | 37 | 2 | 4 | 2 | 499 | 167 | 0.6713 | 0.4192 | -0.2522 | 0.1568 | old_reference_state_mismatch |
| capture_available-002 | 1200 | 42 | 2 | 4 | 2 | 485 | 224 | 0.6701 | 0.5045 | -0.1656 | 0.1656 | old_reference_state_mismatch |
| capture_available-002 | 1200 | 101 | 2 | 4 | 2 | 486 | 193 | 0.6399 | 0.4301 | -0.2099 | 0.1359 | old_reference_state_mismatch |
| capture_available-002 | 1200 | 202 | 2 | 4 | 2 | 527 | 200 | 0.6679 | 0.465 | -0.2029 | 0.2029 | old_reference_state_mismatch |
| capture_available-002 | 1200 | 303 | 2 | 4 | 2 | 475 | 259 | 0.6674 | 0.5405 | -0.1268 | 0.1268 | old_reference_state_mismatch |
| capture_available-003 | 1200 | 11 | 2 | 1 | 2 | 430 | 98 | 0.586 | 0.1837 | -0.4024 | 0.0913 | old_reference_state_mismatch |
| capture_available-003 | 1200 | 23 | 2 | 1 | 2 | 598 | 123 | 0.6204 | 0.2439 | -0.3765 | 0.2274 | old_reference_state_mismatch |
| capture_available-003 | 1200 | 37 | 2 | 1 | 2 | 463 | 141 | 0.635 | 0.3404 | -0.2946 | 0.0953 | old_reference_state_mismatch |
| capture_available-003 | 1200 | 42 | 2 | 1 | 2 | 583 | 104 | 0.6329 | 0.2019 | -0.431 | 0.2268 | old_reference_state_mismatch |
| capture_available-003 | 1200 | 101 | 2 | 1 | 2 | 568 | 73 | 0.6408 | 0.0548 | -0.5861 | 0.1915 | old_reference_state_mismatch |
| capture_available-003 | 1200 | 202 | 2 | 1 | 2 | 471 | 122 | 0.6582 | 0.3115 | -0.3467 | 0.1454 | old_reference_state_mismatch |
| capture_available-003 | 1200 | 303 | 2 | 1 | 2 | 391 | 121 | 0.5806 | 0.2727 | -0.3078 | 0.0587 | old_reference_state_mismatch |
| capture_available-006 | 1200 | 11 | 2 | 2 | 2 | 649 | 122 | 0.453 | 0.0574 | -0.3956 | 0.202 | - |
| capture_available-006 | 1200 | 23 | 2 | 2 | 2 | 670 | 128 | 0.4612 | 0.0703 | -0.3909 | 0.2144 | - |
| capture_available-006 | 1200 | 37 | 2 | 2 | 2 | 690 | 82 | 0.442 | -0.1098 | -0.5518 | 0.2208 | - |
| capture_available-006 | 1200 | 42 | 2 | 2 | 2 | 606 | 76 | 0.4125 | -0.1579 | -0.5704 | 0.1616 | - |
| capture_available-006 | 1200 | 101 | 2 | 2 | 2 | 520 | 137 | 0.4 | 0.0876 | -0.3124 | 0.109 | - |
| capture_available-006 | 1200 | 202 | 2 | 2 | 2 | 613 | 109 | 0.4388 | 0.0183 | -0.4205 | 0.1569 | - |
| capture_available-006 | 1200 | 303 | 2 | 2 | 2 | 582 | 145 | 0.4175 | 0.1034 | -0.3141 | 0.158 | - |
| capture_available-007 | 1200 | 11 | 2 | 1 | 2 | 443 | 222 | 0.6501 | 0.5 | -0.1501 | 0.1176 | old_reference_state_mismatch |
| capture_available-007 | 1200 | 23 | 2 | 1 | 2 | 330 | 282 | 0.5727 | 0.5426 | -0.0302 | 0.0302 | old_reference_state_mismatch |
| capture_available-007 | 1200 | 37 | 2 | 1 | 2 | 470 | 232 | 0.6638 | 0.5129 | -0.1509 | 0.1509 | old_reference_state_mismatch |
| capture_available-007 | 1200 | 42 | 2 | 1 | 2 | 433 | 200 | 0.6166 | 0.45 | -0.1666 | 0.1166 | old_reference_state_mismatch |
| capture_available-007 | 1200 | 101 | 2 | 1 | 2 | 454 | 191 | 0.6718 | 0.4817 | -0.1901 | 0.1184 | old_reference_state_mismatch |
| capture_available-007 | 1200 | 202 | 2 | 1 | 2 | 391 | 276 | 0.6138 | 0.5399 | -0.074 | 0.074 | old_reference_state_mismatch |
| capture_available-007 | 1200 | 303 | 2 | 1 | 2 | 398 | 195 | 0.6457 | 0.4821 | -0.1637 | 0.0723 | old_reference_state_mismatch |
| capture_available-008 | 1200 | 11 | 1 | 1 | 2 | 174 | 224 | 0.2874 | 0.3527 | 0.0653 | 0.1553 | old_reference_state_mismatch |
| capture_available-008 | 1200 | 23 | 3 | 1 | 2 | 174 | 249 | 0.2471 | 0.3454 | 0.0983 | 0.0172 | old_reference_state_mismatch |
| capture_available-008 | 1200 | 37 | 2 | 1 | 2 | 292 | 257 | 0.3938 | 0.3619 | -0.032 | 0.024 | old_reference_state_mismatch |
| capture_available-008 | 1200 | 42 | 1 | 1 | 2 | 158 | 209 | 0.2848 | 0.3636 | 0.0788 | 0.1908 | old_reference_state_mismatch |
| capture_available-008 | 1200 | 101 | 1 | 1 | 2 | 143 | 304 | 0.1678 | 0.3717 | 0.2039 | 0.0429 | old_reference_state_mismatch |
| capture_available-008 | 1200 | 202 | 4 | 1 | 2 | 221 | 298 | 0.2851 | 0.3557 | 0.0706 | 0.0082 | old_reference_state_mismatch |
| capture_available-008 | 1200 | 303 | 1 | 1 | 2 | 238 | 250 | 0.3529 | 0.368 | 0.0151 | 0.0547 | old_reference_state_mismatch |
| capture_available-002 | 2400 | 11 | 2 | 4 | 2 | 1015 | 494 | 0.6404 | 0.5324 | -0.108 | 0.103 | old_reference_state_mismatch |
| capture_available-002 | 2400 | 23 | 2 | 4 | 2 | 1197 | 400 | 0.6642 | 0.495 | -0.1692 | 0.1582 | old_reference_state_mismatch |
| capture_available-002 | 2400 | 37 | 2 | 4 | 2 | 1174 | 389 | 0.6431 | 0.4704 | -0.1727 | 0.1465 | old_reference_state_mismatch |
| capture_available-002 | 2400 | 42 | 2 | 4 | 2 | 1151 | 479 | 0.656 | 0.5198 | -0.1361 | 0.1361 | old_reference_state_mismatch |
| capture_available-002 | 2400 | 101 | 2 | 4 | 2 | 1239 | 388 | 0.6481 | 0.4665 | -0.1816 | 0.1744 | old_reference_state_mismatch |
| capture_available-002 | 2400 | 202 | 2 | 4 | 2 | 1203 | 379 | 0.6459 | 0.467 | -0.1789 | 0.1447 | old_reference_state_mismatch |
| capture_available-002 | 2400 | 303 | 2 | 4 | 2 | 1260 | 368 | 0.6563 | 0.4674 | -0.189 | 0.1623 | old_reference_state_mismatch |
| capture_available-003 | 2400 | 11 | 2 | 1 | 2 | 1143 | 143 | 0.6212 | 0.1958 | -0.4254 | 0.1044 | old_reference_state_mismatch |
| capture_available-003 | 2400 | 23 | 2 | 1 | 2 | 1164 | 146 | 0.6512 | 0.226 | -0.4252 | 0.0995 | old_reference_state_mismatch |
| capture_available-003 | 2400 | 37 | 2 | 1 | 2 | 1258 | 191 | 0.6542 | 0.3037 | -0.3505 | 0.1426 | old_reference_state_mismatch |
| capture_available-003 | 2400 | 42 | 2 | 1 | 2 | 1248 | 149 | 0.637 | 0.2013 | -0.4357 | 0.1489 | old_reference_state_mismatch |
| capture_available-003 | 2400 | 101 | 2 | 1 | 2 | 1254 | 100 | 0.6443 | 0.07 | -0.5743 | 0.1443 | old_reference_state_mismatch |
| capture_available-003 | 2400 | 202 | 2 | 1 | 2 | 1200 | 172 | 0.6658 | 0.2849 | -0.3809 | 0.1501 | old_reference_state_mismatch |
| capture_available-003 | 2400 | 303 | 2 | 1 | 2 | 897 | 205 | 0.6098 | 0.3171 | -0.2927 | 0.0551 | old_reference_state_mismatch |
| capture_available-006 | 2400 | 11 | 2 | 2 | 2 | 1515 | 161 | 0.4554 | 0.0311 | -0.4244 | 0.1676 | - |
| capture_available-006 | 2400 | 23 | 2 | 2 | 2 | 1630 | 241 | 0.4466 | 0.1328 | -0.3138 | 0.2498 | - |
| capture_available-006 | 2400 | 37 | 2 | 2 | 2 | 1640 | 121 | 0.4427 | -0.0826 | -0.5253 | 0.2284 | - |
| capture_available-006 | 2400 | 42 | 2 | 2 | 2 | 1548 | 131 | 0.4587 | -0.0382 | -0.4968 | 0.1771 | - |
| capture_available-006 | 2400 | 101 | 2 | 2 | 2 | 1393 | 191 | 0.4465 | 0.0838 | -0.3627 | 0.1507 | - |
| capture_available-006 | 2400 | 202 | 2 | 2 | 2 | 1476 | 164 | 0.4573 | 0.0427 | -0.4146 | 0.1539 | - |
| capture_available-006 | 2400 | 303 | 2 | 2 | 2 | 1525 | 168 | 0.4485 | 0.0357 | -0.4128 | 0.1795 | - |
| capture_available-007 | 2400 | 11 | 2 | 1 | 2 | 1087 | 376 | 0.6385 | 0.4707 | -0.1677 | 0.1571 | old_reference_state_mismatch |
| capture_available-007 | 2400 | 23 | 2 | 1 | 2 | 814 | 420 | 0.6081 | 0.4976 | -0.1105 | 0.0522 | old_reference_state_mismatch |
| capture_available-007 | 2400 | 37 | 2 | 1 | 2 | 998 | 335 | 0.6513 | 0.4687 | -0.1826 | 0.0916 | old_reference_state_mismatch |
| capture_available-007 | 2400 | 42 | 2 | 1 | 2 | 804 | 402 | 0.6107 | 0.505 | -0.1057 | 0.0582 | old_reference_state_mismatch |
| capture_available-007 | 2400 | 101 | 2 | 1 | 2 | 885 | 432 | 0.6565 | 0.5394 | -0.1171 | 0.0953 | old_reference_state_mismatch |
| capture_available-007 | 2400 | 202 | 2 | 1 | 2 | 830 | 429 | 0.6398 | 0.5315 | -0.1083 | 0.0681 | old_reference_state_mismatch |
| capture_available-007 | 2400 | 303 | 2 | 1 | 2 | 768 | 417 | 0.6172 | 0.5132 | -0.104 | 0.0445 | old_reference_state_mismatch |
| capture_available-008 | 2400 | 11 | 1 | 1 | 2 | 300 | 377 | 0.3 | 0.3475 | 0.0475 | 0.177 | old_reference_state_mismatch |
| capture_available-008 | 2400 | 23 | 1 | 1 | 2 | 325 | 345 | 0.3015 | 0.3188 | 0.0173 | 0.0832 | old_reference_state_mismatch |
| capture_available-008 | 2400 | 37 | 1 | 1 | 2 | 421 | 400 | 0.3729 | 0.3625 | -0.0104 | 0.1122 | old_reference_state_mismatch |
| capture_available-008 | 2400 | 42 | 1 | 1 | 2 | 311 | 380 | 0.3087 | 0.3526 | 0.0439 | 0.1471 | old_reference_state_mismatch |
| capture_available-008 | 2400 | 101 | 1 | 1 | 2 | 167 | 421 | 0.1078 | 0.3397 | 0.2319 | 0.119 | old_reference_state_mismatch |
| capture_available-008 | 2400 | 202 | 1 | 1 | 2 | 412 | 428 | 0.318 | 0.3318 | 0.0138 | 0.1257 | old_reference_state_mismatch |
| capture_available-008 | 2400 | 303 | 1 | 1 | 2 | 471 | 403 | 0.3822 | 0.3524 | -0.0298 | 0.111 | old_reference_state_mismatch |
| capture_available-002 | 5000 | 11 | 2 | 4 | 2 | 2921 | 752 | 0.6402 | 0.4907 | -0.1495 | 0.1254 | old_reference_state_mismatch |
| capture_available-002 | 5000 | 23 | 2 | 4 | 2 | 2898 | 830 | 0.6277 | 0.4928 | -0.1349 | 0.1349 | old_reference_state_mismatch |
| capture_available-002 | 5000 | 37 | 2 | 4 | 2 | 3050 | 673 | 0.6462 | 0.477 | -0.1693 | 0.1469 | old_reference_state_mismatch |
| capture_available-002 | 5000 | 42 | 2 | 4 | 2 | 3087 | 786 | 0.6385 | 0.4924 | -0.1461 | 0.1461 | old_reference_state_mismatch |
| capture_available-002 | 5000 | 101 | 2 | 4 | 2 | 3167 | 590 | 0.6422 | 0.4475 | -0.1948 | 0.1579 | old_reference_state_mismatch |
| capture_available-002 | 5000 | 202 | 2 | 4 | 2 | 2912 | 871 | 0.6459 | 0.5178 | -0.1282 | 0.1282 | old_reference_state_mismatch |
| capture_available-002 | 5000 | 303 | 2 | 4 | 2 | 3062 | 722 | 0.6401 | 0.4806 | -0.1595 | 0.1498 | old_reference_state_mismatch |
| capture_available-003 | 5000 | 11 | 2 | 1 | 2 | 2939 | 229 | 0.6359 | 0.2402 | -0.3958 | 0.1248 | old_reference_state_mismatch |
| capture_available-003 | 5000 | 23 | 2 | 1 | 2 | 3088 | 209 | 0.6483 | 0.2249 | -0.4234 | 0.1284 | old_reference_state_mismatch |
| capture_available-003 | 5000 | 37 | 2 | 1 | 2 | 3024 | 289 | 0.6518 | 0.3149 | -0.3369 | 0.139 | old_reference_state_mismatch |
| capture_available-003 | 5000 | 42 | 2 | 1 | 2 | 2998 | 182 | 0.6381 | 0.1703 | -0.4678 | 0.12 | old_reference_state_mismatch |
| capture_available-003 | 5000 | 101 | 2 | 1 | 2 | 3129 | 156 | 0.6552 | 0.141 | -0.5141 | 0.1452 | old_reference_state_mismatch |
| capture_available-003 | 5000 | 202 | 2 | 1 | 2 | 2836 | 221 | 0.6548 | 0.2489 | -0.4059 | 0.1121 | old_reference_state_mismatch |
| capture_available-003 | 5000 | 303 | 2 | 1 | 2 | 2448 | 242 | 0.6262 | 0.2603 | -0.3659 | 0.0881 | old_reference_state_mismatch |
| capture_available-006 | 5000 | 11 | 2 | 2 | 2 | 3661 | 228 | 0.4698 | 0.057 | -0.4128 | 0.1681 | - |
| capture_available-006 | 5000 | 23 | 2 | 2 | 2 | 3798 | 325 | 0.4518 | 0.1262 | -0.3257 | 0.2105 | - |
| capture_available-006 | 5000 | 37 | 2 | 2 | 2 | 3843 | 166 | 0.4627 | -0.0482 | -0.5109 | 0.214 | - |
| capture_available-006 | 5000 | 42 | 2 | 2 | 2 | 3651 | 198 | 0.456 | 0.0051 | -0.451 | 0.1705 | - |
| capture_available-006 | 5000 | 101 | 2 | 2 | 2 | 3546 | 316 | 0.4509 | 0.1234 | -0.3275 | 0.1672 | - |
| capture_available-006 | 5000 | 202 | 2 | 2 | 2 | 3263 | 232 | 0.471 | 0.0733 | -0.3978 | 0.0963 | - |
| capture_available-006 | 5000 | 303 | 2 | 2 | 2 | 3800 | 223 | 0.4632 | 0.0404 | -0.4228 | 0.2085 | - |
| capture_available-007 | 5000 | 11 | 2 | 1 | 2 | 2435 | 704 | 0.6333 | 0.4886 | -0.1446 | 0.0992 | old_reference_state_mismatch |
| capture_available-007 | 5000 | 23 | 2 | 1 | 2 | 2323 | 582 | 0.6177 | 0.4467 | -0.171 | 0.0814 | old_reference_state_mismatch |
| capture_available-007 | 5000 | 37 | 2 | 1 | 2 | 2560 | 651 | 0.643 | 0.4823 | -0.1606 | 0.1075 | old_reference_state_mismatch |
| capture_available-007 | 5000 | 42 | 2 | 1 | 2 | 2011 | 818 | 0.5957 | 0.4914 | -0.1043 | 0.0622 | old_reference_state_mismatch |
| capture_available-007 | 5000 | 101 | 2 | 1 | 2 | 2451 | 635 | 0.645 | 0.4819 | -0.1632 | 0.0926 | old_reference_state_mismatch |
| capture_available-007 | 5000 | 202 | 2 | 1 | 2 | 2169 | 685 | 0.6432 | 0.5036 | -0.1395 | 0.0829 | old_reference_state_mismatch |
| capture_available-007 | 5000 | 303 | 2 | 1 | 2 | 2315 | 630 | 0.6251 | 0.4667 | -0.1584 | 0.0867 | old_reference_state_mismatch |
| capture_available-008 | 5000 | 11 | 1 | 1 | 2 | 468 | 488 | 0.2906 | 0.2971 | 0.0065 | 0.0651 | old_reference_state_mismatch |
| capture_available-008 | 5000 | 23 | 1 | 1 | 2 | 485 | 466 | 0.2887 | 0.2833 | -0.0054 | 0.1104 | old_reference_state_mismatch |
| capture_available-008 | 5000 | 37 | 1 | 1 | 2 | 1007 | 553 | 0.4211 | 0.3291 | -0.0919 | 0.0838 | old_reference_state_mismatch |
| capture_available-008 | 5000 | 42 | 1 | 1 | 2 | 869 | 511 | 0.3901 | 0.3053 | -0.0848 | 0.1188 | old_reference_state_mismatch |
| capture_available-008 | 5000 | 101 | 1 | 1 | 2 | 521 | 574 | 0.2975 | 0.3136 | 0.0161 | 0.0952 | old_reference_state_mismatch |
| capture_available-008 | 5000 | 202 | 1 | 1 | 2 | 1045 | 562 | 0.4105 | 0.3132 | -0.0974 | 0.0855 | old_reference_state_mismatch |
| capture_available-008 | 5000 | 303 | 1 | 1 | 2 | 903 | 588 | 0.3843 | 0.318 | -0.0662 | 0.1106 | old_reference_state_mismatch |
| capture_available-002 | 10000 | 11 | 2 | 4 | 2 | 7382 | 939 | 0.6636 | 0.4803 | -0.1833 | 0.1554 | old_reference_state_mismatch |
| capture_available-002 | 10000 | 23 | 2 | 4 | 2 | 7548 | 988 | 0.6595 | 0.4798 | -0.1798 | 0.1798 | old_reference_state_mismatch |
| capture_available-002 | 10000 | 37 | 2 | 4 | 2 | 7538 | 1044 | 0.6597 | 0.4875 | -0.1722 | 0.1722 | old_reference_state_mismatch |
| capture_available-002 | 10000 | 42 | 2 | 4 | 2 | 7351 | 1286 | 0.652 | 0.5062 | -0.1458 | 0.1458 | old_reference_state_mismatch |
| capture_available-002 | 10000 | 101 | 2 | 4 | 2 | 7837 | 707 | 0.6668 | 0.4356 | -0.2312 | 0.1939 | old_reference_state_mismatch |
| capture_available-002 | 10000 | 202 | 2 | 4 | 2 | 7526 | 1080 | 0.675 | 0.5148 | -0.1602 | 0.1602 | old_reference_state_mismatch |
| capture_available-002 | 10000 | 303 | 2 | 4 | 2 | 7508 | 1148 | 0.6582 | 0.5017 | -0.1565 | 0.1565 | old_reference_state_mismatch |
| capture_available-003 | 10000 | 11 | 2 | 1 | 2 | 7571 | 254 | 0.667 | 0.2244 | -0.4426 | 0.1652 | old_reference_state_mismatch |
| capture_available-003 | 10000 | 23 | 2 | 1 | 2 | 7486 | 238 | 0.6644 | 0.2059 | -0.4586 | 0.146 | old_reference_state_mismatch |
| capture_available-003 | 10000 | 37 | 2 | 1 | 2 | 7399 | 332 | 0.6724 | 0.2982 | -0.3742 | 0.1676 | old_reference_state_mismatch |
| capture_available-003 | 10000 | 42 | 2 | 1 | 2 | 7327 | 210 | 0.6652 | 0.1714 | -0.4938 | 0.1493 | old_reference_state_mismatch |
| capture_available-003 | 10000 | 101 | 2 | 1 | 2 | 7364 | 238 | 0.6639 | 0.2017 | -0.4622 | 0.1567 | old_reference_state_mismatch |
| capture_available-003 | 10000 | 202 | 2 | 1 | 2 | 6965 | 238 | 0.6695 | 0.2143 | -0.4552 | 0.1244 | old_reference_state_mismatch |
| capture_available-003 | 10000 | 303 | 2 | 1 | 2 | 6699 | 268 | 0.6571 | 0.2351 | -0.422 | 0.1235 | old_reference_state_mismatch |
| capture_available-006 | 10000 | 11 | 2 | 2 | 2 | 8611 | 243 | 0.53 | 0.0576 | -0.4724 | 0.2336 | - |
| capture_available-006 | 10000 | 23 | 2 | 2 | 2 | 8154 | 358 | 0.5113 | 0.1285 | -0.3828 | 0.2407 | - |
| capture_available-006 | 10000 | 37 | 2 | 2 | 2 | 8336 | 179 | 0.508 | -0.0615 | -0.5695 | 0.2529 | - |
| capture_available-006 | 10000 | 42 | 2 | 2 | 2 | 8608 | 212 | 0.5103 | 0.0094 | -0.5009 | 0.2266 | - |
| capture_available-006 | 10000 | 101 | 2 | 2 | 2 | 7151 | 342 | 0.4973 | 0.1082 | -0.3891 | 0.1295 | - |
| capture_available-006 | 10000 | 202 | 2 | 2 | 2 | 7619 | 324 | 0.5229 | 0.1111 | -0.4118 | 0.1501 | - |
| capture_available-006 | 10000 | 303 | 2 | 2 | 2 | 8207 | 228 | 0.5048 | 0.0175 | -0.4873 | 0.1809 | - |
| capture_available-007 | 10000 | 11 | 2 | 1 | 2 | 5371 | 1950 | 0.6338 | 0.5559 | -0.0779 | 0.0779 | old_reference_state_mismatch |
| capture_available-007 | 10000 | 23 | 2 | 1 | 2 | 5736 | 827 | 0.6295 | 0.4426 | -0.187 | 0.0916 | old_reference_state_mismatch |
| capture_available-007 | 10000 | 37 | 2 | 1 | 2 | 5746 | 1091 | 0.6357 | 0.4867 | -0.149 | 0.1053 | old_reference_state_mismatch |
| capture_available-007 | 10000 | 42 | 2 | 1 | 2 | 5566 | 1284 | 0.629 | 0.5031 | -0.1259 | 0.1089 | old_reference_state_mismatch |
| capture_available-007 | 10000 | 101 | 2 | 1 | 2 | 5566 | 1331 | 0.6392 | 0.5199 | -0.1193 | 0.0969 | old_reference_state_mismatch |
| capture_available-007 | 10000 | 202 | 2 | 1 | 2 | 4748 | 1566 | 0.6361 | 0.5434 | -0.0926 | 0.0696 | old_reference_state_mismatch |
| capture_available-007 | 10000 | 303 | 2 | 1 | 2 | 5732 | 941 | 0.6336 | 0.4655 | -0.1682 | 0.1012 | old_reference_state_mismatch |
| capture_available-008 | 10000 | 11 | 1 | 1 | 2 | 1932 | 524 | 0.4524 | 0.2653 | -0.1871 | 0.0545 | old_reference_state_mismatch |
| capture_available-008 | 10000 | 23 | 1 | 1 | 2 | 1268 | 495 | 0.4014 | 0.2566 | -0.1449 | 0.0209 | old_reference_state_mismatch |
| capture_available-008 | 10000 | 37 | 1 | 1 | 2 | 2284 | 596 | 0.4733 | 0.3037 | -0.1696 | 0.0706 | old_reference_state_mismatch |
| capture_available-008 | 10000 | 42 | 1 | 1 | 2 | 1234 | 810 | 0.4028 | 0.3457 | -0.0571 | 0.1454 | old_reference_state_mismatch |
| capture_available-008 | 10000 | 101 | 1 | 1 | 2 | 557 | 611 | 0.2926 | 0.3093 | 0.0167 | 0.1819 | old_reference_state_mismatch |
| capture_available-008 | 10000 | 202 | 1 | 1 | 2 | 1117 | 637 | 0.393 | 0.3093 | -0.0838 | 0.1807 | old_reference_state_mismatch |
| capture_available-008 | 10000 | 303 | 1 | 1 | 2 | 2013 | 685 | 0.4476 | 0.3109 | -0.1366 | 0.0894 | old_reference_state_mismatch |
| capture_available-002 | 30000 | 11 | 2 | 4 | 2 | 27148 | 1002 | 0.738 | 0.4691 | -0.2689 | 0.2345 | - |
| capture_available-002 | 30000 | 23 | 2 | 4 | 2 | 27261 | 1200 | 0.7268 | 0.49 | -0.2368 | 0.2368 | - |
| capture_available-002 | 30000 | 37 | 2 | 4 | 2 | 27170 | 1286 | 0.7331 | 0.5 | -0.2331 | 0.2331 | - |
| capture_available-002 | 30000 | 42 | 2 | 4 | 2 | 27296 | 1287 | 0.738 | 0.5051 | -0.233 | 0.233 | - |
| capture_available-002 | 30000 | 101 | 2 | 4 | 2 | 27545 | 809 | 0.7324 | 0.4363 | -0.296 | 0.257 | - |
| capture_available-002 | 30000 | 202 | 2 | 4 | 2 | 27008 | 1313 | 0.7256 | 0.5065 | -0.2192 | 0.2192 | - |
| capture_available-002 | 30000 | 303 | 2 | 4 | 2 | 27276 | 1211 | 0.7343 | 0.4979 | -0.2363 | 0.2363 | - |
| capture_available-003 | 30000 | 11 | 2 | 1 | 2 | 27105 | 268 | 0.714 | 0.2015 | -0.5125 | 0.2059 | - |
| capture_available-003 | 30000 | 23 | 2 | 1 | 2 | 27085 | 309 | 0.7116 | 0.2395 | -0.4722 | 0.2021 | - |
| capture_available-003 | 30000 | 37 | 2 | 1 | 2 | 25390 | 369 | 0.7087 | 0.2818 | -0.4269 | 0.1329 | - |
| capture_available-003 | 30000 | 42 | 2 | 1 | 2 | 23884 | 248 | 0.6941 | 0.1694 | -0.5248 | 0.0954 | - |
| capture_available-003 | 30000 | 101 | 2 | 1 | 2 | 26661 | 276 | 0.7015 | 0.2029 | -0.4986 | 0.1817 | - |
| capture_available-003 | 30000 | 202 | 2 | 1 | 2 | 25690 | 282 | 0.7046 | 0.2128 | -0.4918 | 0.1562 | - |
| capture_available-003 | 30000 | 303 | 2 | 1 | 2 | 25845 | 289 | 0.7093 | 0.2145 | -0.4947 | 0.1721 | - |
| capture_available-006 | 30000 | 11 | 2 | 2 | 2 | 28497 | 281 | 0.5985 | 0.0747 | -0.5238 | 0.2996 | - |
| capture_available-006 | 30000 | 23 | 2 | 2 | 2 | 28152 | 358 | 0.595 | 0.1285 | -0.4665 | 0.3244 | - |
| capture_available-006 | 30000 | 37 | 2 | 2 | 2 | 28336 | 179 | 0.594 | -0.0615 | -0.6555 | 0.3388 | - |
| capture_available-006 | 30000 | 42 | 2 | 2 | 2 | 28427 | 256 | 0.5843 | 0.0352 | -0.5492 | 0.3005 | - |
| capture_available-006 | 30000 | 101 | 2 | 2 | 2 | 27151 | 342 | 0.5931 | 0.1082 | -0.4849 | 0.2253 | - |
| capture_available-006 | 30000 | 202 | 2 | 2 | 2 | 27616 | 324 | 0.6002 | 0.1111 | -0.489 | 0.2274 | - |
| capture_available-006 | 30000 | 303 | 2 | 2 | 2 | 28203 | 228 | 0.5987 | 0.0175 | -0.5811 | 0.2748 | - |
| capture_available-007 | 30000 | 11 | 2 | 1 | 2 | 21720 | 3105 | 0.6955 | 0.5626 | -0.1328 | 0.1257 | - |
| capture_available-007 | 30000 | 23 | 2 | 1 | 2 | 22073 | 1559 | 0.6836 | 0.4869 | -0.1967 | 0.1139 | - |
| capture_available-007 | 30000 | 37 | 2 | 1 | 2 | 23037 | 1959 | 0.7022 | 0.5181 | -0.1841 | 0.1592 | - |
| capture_available-007 | 30000 | 42 | 2 | 1 | 2 | 21115 | 4455 | 0.6767 | 0.5796 | -0.0971 | 0.0971 | - |
| capture_available-007 | 30000 | 101 | 2 | 1 | 2 | 22864 | 2454 | 0.6903 | 0.5359 | -0.1544 | 0.1533 | - |
| capture_available-007 | 30000 | 202 | 2 | 1 | 2 | 20859 | 2687 | 0.6751 | 0.5408 | -0.1344 | 0.107 | - |
| capture_available-007 | 30000 | 303 | 2 | 1 | 2 | 22807 | 1772 | 0.6835 | 0.5 | -0.1835 | 0.1457 | - |
| capture_available-008 | 30000 | 11 | 1 | 1 | 2 | 4784 | 525 | 0.5173 | 0.2629 | -0.2545 | 0.1703 | - |
| capture_available-008 | 30000 | 23 | 1 | 1 | 2 | 1274 | 497 | 0.4003 | 0.2535 | -0.1468 | 0.2053 | - |
| capture_available-008 | 30000 | 37 | 1 | 1 | 2 | 2284 | 596 | 0.4733 | 0.3037 | -0.1696 | 0.2181 | - |
| capture_available-008 | 30000 | 42 | 1 | 1 | 2 | 1234 | 810 | 0.4028 | 0.3457 | -0.0571 | 0.3215 | - |
| capture_available-008 | 30000 | 101 | 1 | 1 | 2 | 557 | 611 | 0.2926 | 0.3093 | 0.0167 | 0.3226 | - |
| capture_available-008 | 30000 | 202 | 1 | 1 | 2 | 1117 | 637 | 0.393 | 0.3093 | -0.0838 | 0.3136 | - |
| capture_available-008 | 30000 | 303 | 1 | 1 | 2 | 2013 | 685 | 0.4476 | 0.3109 | -0.1366 | 0.2651 | - |

- 30000-budget status: `30000 included: projected root ClassicMCTS runtime 48.1s`.

Stability summary:

| row_id | budget | seeds | observed_reference_moves | majority_move | majority_fraction | old_reference_move | stable | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capture_available-002 | 1200 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 4 | true | old_target_invalid |
| capture_available-002 | 2400 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 4 | true | old_target_invalid |
| capture_available-002 | 5000 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 4 | true | old_target_invalid |
| capture_available-002 | 10000 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 4 | true | old_target_invalid |
| capture_available-002 | 30000 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 4 | true | old_target_invalid |
| capture_available-003 | 1200 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 1 | true | old_target_invalid |
| capture_available-003 | 2400 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 1 | true | old_target_invalid |
| capture_available-003 | 5000 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 1 | true | old_target_invalid |
| capture_available-003 | 10000 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 1 | true | old_target_invalid |
| capture_available-003 | 30000 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 1 | true | old_target_invalid |
| capture_available-006 | 1200 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 2 | true | old_target_still_valid |
| capture_available-006 | 2400 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 2 | true | old_target_still_valid |
| capture_available-006 | 5000 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 2 | true | old_target_still_valid |
| capture_available-006 | 10000 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 2 | true | old_target_still_valid |
| capture_available-006 | 30000 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 2 | true | old_target_still_valid |
| capture_available-007 | 1200 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 1 | true | old_target_invalid |
| capture_available-007 | 2400 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 1 | true | old_target_invalid |
| capture_available-007 | 5000 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 1 | true | old_target_invalid |
| capture_available-007 | 10000 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 1 | true | old_target_invalid |
| capture_available-007 | 30000 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 1 | true | old_target_invalid |
| capture_available-008 | 1200 | `[11, 23, 37, 42, 101, 202, 303]` | `[1, 2, 3, 4]` | 1 | 0.5714 | 1 | false | reference_unstable |
| capture_available-008 | 2400 | `[11, 23, 37, 42, 101, 202, 303]` | `[1]` | 1 | 1.0 | 1 | true | old_target_still_valid |
| capture_available-008 | 5000 | `[11, 23, 37, 42, 101, 202, 303]` | `[1]` | 1 | 1.0 | 1 | true | old_target_still_valid |
| capture_available-008 | 10000 | `[11, 23, 37, 42, 101, 202, 303]` | `[1]` | 1 | 1.0 | 1 | true | old_target_still_valid |
| capture_available-008 | 30000 | `[11, 23, 37, 42, 101, 202, 303]` | `[1]` | 1 | 1.0 | 1 | true | old_target_still_valid |

## Child-afterstate adjudication

| child_from_move | budget | seed | child_value_raw | child_value_root_perspective | child_selected_move | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 1200 | 11 | 0.6483 | 0.6483 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 1200 | 23 | 0.6733 | 0.6733 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 1200 | 37 | 0.6342 | 0.6342 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 1200 | 42 | 0.6258 | 0.6258 | 3 | root-perspective child ClassicMCTS adjudication |
| 2 | 1200 | 101 | 0.6392 | 0.6392 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 1200 | 202 | 0.6275 | 0.6275 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 1200 | 303 | 0.655 | 0.655 | 5 | root-perspective child ClassicMCTS adjudication |
| 4 | 1200 | 11 | 0.2683 | -0.2683 | 4 | root-perspective child ClassicMCTS adjudication |
| 4 | 1200 | 23 | 0.1833 | -0.1833 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 1200 | 37 | 0.185 | -0.185 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 1200 | 42 | 0.1775 | -0.1775 | 4 | root-perspective child ClassicMCTS adjudication |
| 4 | 1200 | 101 | 0.1433 | -0.1433 | 4 | root-perspective child ClassicMCTS adjudication |
| 4 | 1200 | 202 | 0.19 | -0.19 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 1200 | 303 | 0.2125 | -0.2125 | 3 | root-perspective child ClassicMCTS adjudication |
| 2 | 2400 | 11 | 0.6458 | 0.6458 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 2400 | 23 | 0.6267 | 0.6267 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 2400 | 37 | 0.6338 | 0.6338 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 2400 | 42 | 0.6458 | 0.6458 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 2400 | 101 | 0.6408 | 0.6408 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 2400 | 202 | 0.6333 | 0.6333 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 2400 | 303 | 0.6483 | 0.6483 | 5 | root-perspective child ClassicMCTS adjudication |
| 4 | 2400 | 11 | 0.4067 | -0.4067 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 2400 | 23 | 0.3783 | -0.3783 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 2400 | 37 | 0.3838 | -0.3838 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 2400 | 42 | 0.3629 | -0.3629 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 2400 | 101 | 0.3483 | -0.3483 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 2400 | 202 | 0.3558 | -0.3558 | 4 | root-perspective child ClassicMCTS adjudication |
| 4 | 2400 | 303 | 0.3908 | -0.3908 | 2 | root-perspective child ClassicMCTS adjudication |
| 2 | 5000 | 11 | 0.647 | 0.647 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 5000 | 23 | 0.6448 | 0.6448 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 5000 | 37 | 0.635 | 0.635 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 5000 | 42 | 0.6496 | 0.6496 | 3 | root-perspective child ClassicMCTS adjudication |
| 2 | 5000 | 101 | 0.6434 | 0.6434 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 5000 | 202 | 0.6402 | 0.6402 | 3 | root-perspective child ClassicMCTS adjudication |
| 2 | 5000 | 303 | 0.6372 | 0.6372 | 5 | root-perspective child ClassicMCTS adjudication |
| 4 | 5000 | 11 | 0.5262 | -0.5262 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 5000 | 23 | 0.5328 | -0.5328 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 5000 | 37 | 0.531 | -0.531 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 5000 | 42 | 0.5142 | -0.5142 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 5000 | 101 | 0.5082 | -0.5082 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 5000 | 202 | 0.5058 | -0.5058 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 5000 | 303 | 0.5224 | -0.5224 | 2 | root-perspective child ClassicMCTS adjudication |
| 2 | 10000 | 11 | 0.6742 | 0.6742 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 10000 | 23 | 0.669 | 0.669 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 10000 | 37 | 0.6584 | 0.6584 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 10000 | 42 | 0.6709 | 0.6709 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 10000 | 101 | 0.6575 | 0.6575 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 10000 | 202 | 0.6495 | 0.6495 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 10000 | 303 | 0.6654 | 0.6654 | 5 | root-perspective child ClassicMCTS adjudication |
| 4 | 10000 | 11 | 0.6378 | -0.6378 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 10000 | 23 | 0.6477 | -0.6477 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 10000 | 37 | 0.6627 | -0.6627 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 10000 | 42 | 0.6308 | -0.6308 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 10000 | 101 | 0.6267 | -0.6267 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 10000 | 202 | 0.6256 | -0.6256 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 10000 | 303 | 0.6565 | -0.6565 | 2 | root-perspective child ClassicMCTS adjudication |
| 2 | 30000 | 11 | 0.7489 | 0.7489 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 30000 | 23 | 0.7492 | 0.7492 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 30000 | 37 | 0.7538 | 0.7538 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 30000 | 42 | 0.752 | 0.752 | 3 | root-perspective child ClassicMCTS adjudication |
| 2 | 30000 | 101 | 0.7428 | 0.7428 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 30000 | 202 | 0.7549 | 0.7549 | 5 | root-perspective child ClassicMCTS adjudication |
| 2 | 30000 | 303 | 0.7476 | 0.7476 | 5 | root-perspective child ClassicMCTS adjudication |
| 4 | 30000 | 11 | 0.7779 | -0.7779 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 30000 | 23 | 0.7854 | -0.7854 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 30000 | 37 | 0.7982 | -0.7982 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 30000 | 42 | 0.7729 | -0.7729 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 30000 | 101 | 0.7726 | -0.7726 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 30000 | 202 | 0.7726 | -0.7726 | 2 | root-perspective child ClassicMCTS adjudication |
| 4 | 30000 | 303 | 0.7952 | -0.7952 | 2 | root-perspective child ClassicMCTS adjudication |

Child stability:

| budget | seeds | observed_move2_root_values | observed_move4_root_values | ordering | stable | decomposition_agrees_with_root |
| --- | --- | --- | --- | --- | --- | --- |
| 1200 | `[11, 23, 37, 42, 101, 202, 303]` | `[0.6483, 0.6733, 0.6342, 0.6258, 0.6392, 0.6275, 0.655]` | `[-0.2683, -0.1833, -0.185, -0.1775, -0.1433, -0.19, -0.2125]` | move_2_gt_move_4 | true | true |
| 2400 | `[11, 23, 37, 42, 101, 202, 303]` | `[0.6458, 0.6267, 0.6338, 0.6458, 0.6408, 0.6333, 0.6483]` | `[-0.4067, -0.3783, -0.3838, -0.3629, -0.3483, -0.3558, -0.3908]` | move_2_gt_move_4 | true | true |
| 5000 | `[11, 23, 37, 42, 101, 202, 303]` | `[0.647, 0.6448, 0.635, 0.6496, 0.6434, 0.6402, 0.6372]` | `[-0.5262, -0.5328, -0.531, -0.5142, -0.5082, -0.5058, -0.5224]` | move_2_gt_move_4 | true | true |
| 10000 | `[11, 23, 37, 42, 101, 202, 303]` | `[0.6742, 0.669, 0.6584, 0.6709, 0.6575, 0.6495, 0.6654]` | `[-0.6378, -0.6477, -0.6627, -0.6308, -0.6267, -0.6256, -0.6565]` | move_2_gt_move_4 | true | true |
| 30000 | `[11, 23, 37, 42, 101, 202, 303]` | `[0.7489, 0.7492, 0.7538, 0.752, 0.7428, 0.7549, 0.7476]` | `[-0.7779, -0.7854, -0.7982, -0.7729, -0.7726, -0.7726, -0.7952]` | move_2_gt_move_4 | true | true |

## Tablebase availability

| state_label | tablebase_available | remaining_seeds | root_perspective_value | notes |
| --- | --- | --- | --- | --- |
| capture_available-002 | false | 46 | - | root state |
| capture_available-003 | false | 46 | - | root state |
| capture_available-006 | false | 45 | - | root state |
| capture_available-007 | false | 46 | - | root state |
| capture_available-008 | false | 46 | - | root state |
| capture_available-002 child_after_move_2 | false | 45 | - | 002 child afterstate |
| capture_available-002 child_after_move_4 | false | 45 | - | 002 child afterstate |

## PUCT/artifact teacher comparison

| row_id | budget | seed | selected_move | old_reference_move | disputed_move | visits_move_2 | visits_move_4 | value_move_2 | value_move_4 | value_4_minus_2 | top1_margin | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capture_available-002 | 384 | 11 | 2 | 4 | 2 | 328 | 18 | 0.0623 | 0.0391 | -0.0232 | 0.0232 | current artifact deterministic PUCT |
| capture_available-002 | 384 | 23 | 2 | 4 | 2 | 328 | 18 | 0.0623 | 0.0391 | -0.0232 | 0.0232 | current artifact deterministic PUCT |
| capture_available-002 | 384 | 37 | 2 | 4 | 2 | 328 | 18 | 0.0623 | 0.0391 | -0.0232 | 0.0232 | current artifact deterministic PUCT |
| capture_available-002 | 384 | 42 | 2 | 4 | 2 | 328 | 18 | 0.0623 | 0.0391 | -0.0232 | 0.0232 | current artifact deterministic PUCT |
| capture_available-002 | 384 | 101 | 2 | 4 | 2 | 328 | 18 | 0.0623 | 0.0391 | -0.0232 | 0.0232 | current artifact deterministic PUCT |
| capture_available-002 | 384 | 202 | 2 | 4 | 2 | 328 | 18 | 0.0623 | 0.0391 | -0.0232 | 0.0232 | current artifact deterministic PUCT |
| capture_available-002 | 384 | 303 | 2 | 4 | 2 | 328 | 18 | 0.0623 | 0.0391 | -0.0232 | 0.0232 | current artifact deterministic PUCT |
| capture_available-003 | 384 | 11 | 2 | 1 | 2 | 302 | 14 | 0.1134 | 0.0611 | -0.0522 | 0.0522 | current artifact deterministic PUCT |
| capture_available-003 | 384 | 23 | 2 | 1 | 2 | 302 | 14 | 0.1134 | 0.0611 | -0.0522 | 0.0522 | current artifact deterministic PUCT |
| capture_available-003 | 384 | 37 | 2 | 1 | 2 | 302 | 14 | 0.1134 | 0.0611 | -0.0522 | 0.0522 | current artifact deterministic PUCT |
| capture_available-003 | 384 | 42 | 2 | 1 | 2 | 302 | 14 | 0.1134 | 0.0611 | -0.0522 | 0.0522 | current artifact deterministic PUCT |
| capture_available-003 | 384 | 101 | 2 | 1 | 2 | 302 | 14 | 0.1134 | 0.0611 | -0.0522 | 0.0522 | current artifact deterministic PUCT |
| capture_available-003 | 384 | 202 | 2 | 1 | 2 | 302 | 14 | 0.1134 | 0.0611 | -0.0522 | 0.0522 | current artifact deterministic PUCT |
| capture_available-003 | 384 | 303 | 2 | 1 | 2 | 302 | 14 | 0.1134 | 0.0611 | -0.0522 | 0.0522 | current artifact deterministic PUCT |
| capture_available-006 | 384 | 11 | 2 | 2 | 2 | 328 | 36 | 0.0095 | 0.004 | -0.0055 | 0.0055 | current artifact deterministic PUCT |
| capture_available-006 | 384 | 23 | 2 | 2 | 2 | 328 | 36 | 0.0095 | 0.004 | -0.0055 | 0.0055 | current artifact deterministic PUCT |
| capture_available-006 | 384 | 37 | 2 | 2 | 2 | 328 | 36 | 0.0095 | 0.004 | -0.0055 | 0.0055 | current artifact deterministic PUCT |
| capture_available-006 | 384 | 42 | 2 | 2 | 2 | 328 | 36 | 0.0095 | 0.004 | -0.0055 | 0.0055 | current artifact deterministic PUCT |
| capture_available-006 | 384 | 101 | 2 | 2 | 2 | 328 | 36 | 0.0095 | 0.004 | -0.0055 | 0.0055 | current artifact deterministic PUCT |
| capture_available-006 | 384 | 202 | 2 | 2 | 2 | 328 | 36 | 0.0095 | 0.004 | -0.0055 | 0.0055 | current artifact deterministic PUCT |
| capture_available-006 | 384 | 303 | 2 | 2 | 2 | 328 | 36 | 0.0095 | 0.004 | -0.0055 | 0.0055 | current artifact deterministic PUCT |
| capture_available-007 | 384 | 11 | 2 | 1 | 2 | 329 | 3 | 0.107 | -0.1216 | -0.2286 | 0.0776 | current artifact deterministic PUCT |
| capture_available-007 | 384 | 23 | 2 | 1 | 2 | 329 | 3 | 0.107 | -0.1216 | -0.2286 | 0.0776 | current artifact deterministic PUCT |
| capture_available-007 | 384 | 37 | 2 | 1 | 2 | 329 | 3 | 0.107 | -0.1216 | -0.2286 | 0.0776 | current artifact deterministic PUCT |
| capture_available-007 | 384 | 42 | 2 | 1 | 2 | 329 | 3 | 0.107 | -0.1216 | -0.2286 | 0.0776 | current artifact deterministic PUCT |
| capture_available-007 | 384 | 101 | 2 | 1 | 2 | 329 | 3 | 0.107 | -0.1216 | -0.2286 | 0.0776 | current artifact deterministic PUCT |
| capture_available-007 | 384 | 202 | 2 | 1 | 2 | 329 | 3 | 0.107 | -0.1216 | -0.2286 | 0.0776 | current artifact deterministic PUCT |
| capture_available-007 | 384 | 303 | 2 | 1 | 2 | 329 | 3 | 0.107 | -0.1216 | -0.2286 | 0.0776 | current artifact deterministic PUCT |
| capture_available-008 | 384 | 11 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0318 | current artifact deterministic PUCT |
| capture_available-008 | 384 | 23 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0318 | current artifact deterministic PUCT |
| capture_available-008 | 384 | 37 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0318 | current artifact deterministic PUCT |
| capture_available-008 | 384 | 42 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0318 | current artifact deterministic PUCT |
| capture_available-008 | 384 | 101 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0318 | current artifact deterministic PUCT |
| capture_available-008 | 384 | 202 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0318 | current artifact deterministic PUCT |
| capture_available-008 | 384 | 303 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0318 | current artifact deterministic PUCT |
| capture_available-002 | 1200 | 11 | 2 | 4 | 2 | 1117 | 18 | 0.1028 | 0.0391 | -0.0637 | 0.0481 | current artifact deterministic PUCT |
| capture_available-002 | 1200 | 23 | 2 | 4 | 2 | 1117 | 18 | 0.1028 | 0.0391 | -0.0637 | 0.0481 | current artifact deterministic PUCT |
| capture_available-002 | 1200 | 37 | 2 | 4 | 2 | 1117 | 18 | 0.1028 | 0.0391 | -0.0637 | 0.0481 | current artifact deterministic PUCT |
| capture_available-002 | 1200 | 42 | 2 | 4 | 2 | 1117 | 18 | 0.1028 | 0.0391 | -0.0637 | 0.0481 | current artifact deterministic PUCT |
| capture_available-002 | 1200 | 101 | 2 | 4 | 2 | 1117 | 18 | 0.1028 | 0.0391 | -0.0637 | 0.0481 | current artifact deterministic PUCT |
| capture_available-002 | 1200 | 202 | 2 | 4 | 2 | 1117 | 18 | 0.1028 | 0.0391 | -0.0637 | 0.0481 | current artifact deterministic PUCT |
| capture_available-002 | 1200 | 303 | 2 | 4 | 2 | 1117 | 18 | 0.1028 | 0.0391 | -0.0637 | 0.0481 | current artifact deterministic PUCT |
| capture_available-003 | 1200 | 11 | 2 | 1 | 2 | 1116 | 14 | 0.1323 | 0.0611 | -0.0712 | 0.0712 | current artifact deterministic PUCT |
| capture_available-003 | 1200 | 23 | 2 | 1 | 2 | 1116 | 14 | 0.1323 | 0.0611 | -0.0712 | 0.0712 | current artifact deterministic PUCT |
| capture_available-003 | 1200 | 37 | 2 | 1 | 2 | 1116 | 14 | 0.1323 | 0.0611 | -0.0712 | 0.0712 | current artifact deterministic PUCT |
| capture_available-003 | 1200 | 42 | 2 | 1 | 2 | 1116 | 14 | 0.1323 | 0.0611 | -0.0712 | 0.0712 | current artifact deterministic PUCT |
| capture_available-003 | 1200 | 101 | 2 | 1 | 2 | 1116 | 14 | 0.1323 | 0.0611 | -0.0712 | 0.0712 | current artifact deterministic PUCT |
| capture_available-003 | 1200 | 202 | 2 | 1 | 2 | 1116 | 14 | 0.1323 | 0.0611 | -0.0712 | 0.0712 | current artifact deterministic PUCT |
| capture_available-003 | 1200 | 303 | 2 | 1 | 2 | 1116 | 14 | 0.1323 | 0.0611 | -0.0712 | 0.0712 | current artifact deterministic PUCT |
| capture_available-006 | 1200 | 11 | 2 | 2 | 2 | 1116 | 38 | 0.0282 | -0.0057 | -0.0339 | 0.0275 | current artifact deterministic PUCT |
| capture_available-006 | 1200 | 23 | 2 | 2 | 2 | 1116 | 38 | 0.0282 | -0.0057 | -0.0339 | 0.0275 | current artifact deterministic PUCT |
| capture_available-006 | 1200 | 37 | 2 | 2 | 2 | 1116 | 38 | 0.0282 | -0.0057 | -0.0339 | 0.0275 | current artifact deterministic PUCT |
| capture_available-006 | 1200 | 42 | 2 | 2 | 2 | 1116 | 38 | 0.0282 | -0.0057 | -0.0339 | 0.0275 | current artifact deterministic PUCT |
| capture_available-006 | 1200 | 101 | 2 | 2 | 2 | 1116 | 38 | 0.0282 | -0.0057 | -0.0339 | 0.0275 | current artifact deterministic PUCT |
| capture_available-006 | 1200 | 202 | 2 | 2 | 2 | 1116 | 38 | 0.0282 | -0.0057 | -0.0339 | 0.0275 | current artifact deterministic PUCT |
| capture_available-006 | 1200 | 303 | 2 | 2 | 2 | 1116 | 38 | 0.0282 | -0.0057 | -0.0339 | 0.0275 | current artifact deterministic PUCT |
| capture_available-007 | 1200 | 11 | 2 | 1 | 2 | 1137 | 4 | 0.1161 | -0.086 | -0.2021 | 0.0867 | current artifact deterministic PUCT |
| capture_available-007 | 1200 | 23 | 2 | 1 | 2 | 1137 | 4 | 0.1161 | -0.086 | -0.2021 | 0.0867 | current artifact deterministic PUCT |
| capture_available-007 | 1200 | 37 | 2 | 1 | 2 | 1137 | 4 | 0.1161 | -0.086 | -0.2021 | 0.0867 | current artifact deterministic PUCT |
| capture_available-007 | 1200 | 42 | 2 | 1 | 2 | 1137 | 4 | 0.1161 | -0.086 | -0.2021 | 0.0867 | current artifact deterministic PUCT |
| capture_available-007 | 1200 | 101 | 2 | 1 | 2 | 1137 | 4 | 0.1161 | -0.086 | -0.2021 | 0.0867 | current artifact deterministic PUCT |
| capture_available-007 | 1200 | 202 | 2 | 1 | 2 | 1137 | 4 | 0.1161 | -0.086 | -0.2021 | 0.0867 | current artifact deterministic PUCT |
| capture_available-007 | 1200 | 303 | 2 | 1 | 2 | 1137 | 4 | 0.1161 | -0.086 | -0.2021 | 0.0867 | current artifact deterministic PUCT |
| capture_available-008 | 1200 | 11 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0759 | current artifact deterministic PUCT |
| capture_available-008 | 1200 | 23 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0759 | current artifact deterministic PUCT |
| capture_available-008 | 1200 | 37 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0759 | current artifact deterministic PUCT |
| capture_available-008 | 1200 | 42 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0759 | current artifact deterministic PUCT |
| capture_available-008 | 1200 | 101 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0759 | current artifact deterministic PUCT |
| capture_available-008 | 1200 | 202 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0759 | current artifact deterministic PUCT |
| capture_available-008 | 1200 | 303 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0759 | current artifact deterministic PUCT |
| capture_available-002 | 2400 | 11 | 2 | 4 | 2 | 2317 | 18 | 0.1385 | 0.0391 | -0.0994 | 0.0838 | current artifact deterministic PUCT |
| capture_available-002 | 2400 | 23 | 2 | 4 | 2 | 2317 | 18 | 0.1385 | 0.0391 | -0.0994 | 0.0838 | current artifact deterministic PUCT |
| capture_available-002 | 2400 | 37 | 2 | 4 | 2 | 2317 | 18 | 0.1385 | 0.0391 | -0.0994 | 0.0838 | current artifact deterministic PUCT |
| capture_available-002 | 2400 | 42 | 2 | 4 | 2 | 2317 | 18 | 0.1385 | 0.0391 | -0.0994 | 0.0838 | current artifact deterministic PUCT |
| capture_available-002 | 2400 | 101 | 2 | 4 | 2 | 2317 | 18 | 0.1385 | 0.0391 | -0.0994 | 0.0838 | current artifact deterministic PUCT |
| capture_available-002 | 2400 | 202 | 2 | 4 | 2 | 2317 | 18 | 0.1385 | 0.0391 | -0.0994 | 0.0838 | current artifact deterministic PUCT |
| capture_available-002 | 2400 | 303 | 2 | 4 | 2 | 2317 | 18 | 0.1385 | 0.0391 | -0.0994 | 0.0838 | current artifact deterministic PUCT |
| capture_available-003 | 2400 | 11 | 2 | 1 | 2 | 2310 | 14 | 0.1195 | 0.0611 | -0.0584 | 0.058 | current artifact deterministic PUCT |
| capture_available-003 | 2400 | 23 | 2 | 1 | 2 | 2310 | 14 | 0.1195 | 0.0611 | -0.0584 | 0.058 | current artifact deterministic PUCT |
| capture_available-003 | 2400 | 37 | 2 | 1 | 2 | 2310 | 14 | 0.1195 | 0.0611 | -0.0584 | 0.058 | current artifact deterministic PUCT |
| capture_available-003 | 2400 | 42 | 2 | 1 | 2 | 2310 | 14 | 0.1195 | 0.0611 | -0.0584 | 0.058 | current artifact deterministic PUCT |
| capture_available-003 | 2400 | 101 | 2 | 1 | 2 | 2310 | 14 | 0.1195 | 0.0611 | -0.0584 | 0.058 | current artifact deterministic PUCT |
| capture_available-003 | 2400 | 202 | 2 | 1 | 2 | 2310 | 14 | 0.1195 | 0.0611 | -0.0584 | 0.058 | current artifact deterministic PUCT |
| capture_available-003 | 2400 | 303 | 2 | 1 | 2 | 2310 | 14 | 0.1195 | 0.0611 | -0.0584 | 0.058 | current artifact deterministic PUCT |
| capture_available-006 | 2400 | 11 | 2 | 2 | 2 | 2301 | 38 | 0.0443 | -0.0057 | -0.0499 | 0.0499 | current artifact deterministic PUCT |
| capture_available-006 | 2400 | 23 | 2 | 2 | 2 | 2301 | 38 | 0.0443 | -0.0057 | -0.0499 | 0.0499 | current artifact deterministic PUCT |
| capture_available-006 | 2400 | 37 | 2 | 2 | 2 | 2301 | 38 | 0.0443 | -0.0057 | -0.0499 | 0.0499 | current artifact deterministic PUCT |
| capture_available-006 | 2400 | 42 | 2 | 2 | 2 | 2301 | 38 | 0.0443 | -0.0057 | -0.0499 | 0.0499 | current artifact deterministic PUCT |
| capture_available-006 | 2400 | 101 | 2 | 2 | 2 | 2301 | 38 | 0.0443 | -0.0057 | -0.0499 | 0.0499 | current artifact deterministic PUCT |
| capture_available-006 | 2400 | 202 | 2 | 2 | 2 | 2301 | 38 | 0.0443 | -0.0057 | -0.0499 | 0.0499 | current artifact deterministic PUCT |
| capture_available-006 | 2400 | 303 | 2 | 2 | 2 | 2301 | 38 | 0.0443 | -0.0057 | -0.0499 | 0.0499 | current artifact deterministic PUCT |
| capture_available-007 | 2400 | 11 | 2 | 1 | 2 | 2328 | 6 | 0.1327 | -0.0839 | -0.2166 | 0.1146 | current artifact deterministic PUCT |
| capture_available-007 | 2400 | 23 | 2 | 1 | 2 | 2328 | 6 | 0.1327 | -0.0839 | -0.2166 | 0.1146 | current artifact deterministic PUCT |
| capture_available-007 | 2400 | 37 | 2 | 1 | 2 | 2328 | 6 | 0.1327 | -0.0839 | -0.2166 | 0.1146 | current artifact deterministic PUCT |
| capture_available-007 | 2400 | 42 | 2 | 1 | 2 | 2328 | 6 | 0.1327 | -0.0839 | -0.2166 | 0.1146 | current artifact deterministic PUCT |
| capture_available-007 | 2400 | 101 | 2 | 1 | 2 | 2328 | 6 | 0.1327 | -0.0839 | -0.2166 | 0.1146 | current artifact deterministic PUCT |
| capture_available-007 | 2400 | 202 | 2 | 1 | 2 | 2328 | 6 | 0.1327 | -0.0839 | -0.2166 | 0.1146 | current artifact deterministic PUCT |
| capture_available-007 | 2400 | 303 | 2 | 1 | 2 | 2328 | 6 | 0.1327 | -0.0839 | -0.2166 | 0.1146 | current artifact deterministic PUCT |
| capture_available-008 | 2400 | 11 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0566 | current artifact deterministic PUCT |
| capture_available-008 | 2400 | 23 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0566 | current artifact deterministic PUCT |
| capture_available-008 | 2400 | 37 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0566 | current artifact deterministic PUCT |
| capture_available-008 | 2400 | 42 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0566 | current artifact deterministic PUCT |
| capture_available-008 | 2400 | 101 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0566 | current artifact deterministic PUCT |
| capture_available-008 | 2400 | 202 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0566 | current artifact deterministic PUCT |
| capture_available-008 | 2400 | 303 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.0566 | current artifact deterministic PUCT |
| capture_available-002 | 5000 | 11 | 2 | 4 | 2 | 4904 | 18 | 0.1613 | 0.0391 | -0.1222 | 0.1066 | current artifact deterministic PUCT |
| capture_available-002 | 5000 | 23 | 2 | 4 | 2 | 4904 | 18 | 0.1613 | 0.0391 | -0.1222 | 0.1066 | current artifact deterministic PUCT |
| capture_available-002 | 5000 | 37 | 2 | 4 | 2 | 4904 | 18 | 0.1613 | 0.0391 | -0.1222 | 0.1066 | current artifact deterministic PUCT |
| capture_available-002 | 5000 | 42 | 2 | 4 | 2 | 4904 | 18 | 0.1613 | 0.0391 | -0.1222 | 0.1066 | current artifact deterministic PUCT |
| capture_available-002 | 5000 | 101 | 2 | 4 | 2 | 4904 | 18 | 0.1613 | 0.0391 | -0.1222 | 0.1066 | current artifact deterministic PUCT |
| capture_available-002 | 5000 | 202 | 2 | 4 | 2 | 4904 | 18 | 0.1613 | 0.0391 | -0.1222 | 0.1066 | current artifact deterministic PUCT |
| capture_available-002 | 5000 | 303 | 2 | 4 | 2 | 4904 | 18 | 0.1613 | 0.0391 | -0.1222 | 0.1066 | current artifact deterministic PUCT |
| capture_available-003 | 5000 | 11 | 2 | 1 | 2 | 4897 | 15 | 0.1571 | 0.0408 | -0.1163 | 0.1144 | current artifact deterministic PUCT |
| capture_available-003 | 5000 | 23 | 2 | 1 | 2 | 4897 | 15 | 0.1571 | 0.0408 | -0.1163 | 0.1144 | current artifact deterministic PUCT |
| capture_available-003 | 5000 | 37 | 2 | 1 | 2 | 4897 | 15 | 0.1571 | 0.0408 | -0.1163 | 0.1144 | current artifact deterministic PUCT |
| capture_available-003 | 5000 | 42 | 2 | 1 | 2 | 4897 | 15 | 0.1571 | 0.0408 | -0.1163 | 0.1144 | current artifact deterministic PUCT |
| capture_available-003 | 5000 | 101 | 2 | 1 | 2 | 4897 | 15 | 0.1571 | 0.0408 | -0.1163 | 0.1144 | current artifact deterministic PUCT |
| capture_available-003 | 5000 | 202 | 2 | 1 | 2 | 4897 | 15 | 0.1571 | 0.0408 | -0.1163 | 0.1144 | current artifact deterministic PUCT |
| capture_available-003 | 5000 | 303 | 2 | 1 | 2 | 4897 | 15 | 0.1571 | 0.0408 | -0.1163 | 0.1144 | current artifact deterministic PUCT |
| capture_available-006 | 5000 | 11 | 2 | 2 | 2 | 4864 | 38 | 0.0335 | -0.0057 | -0.0392 | 0.0392 | current artifact deterministic PUCT |
| capture_available-006 | 5000 | 23 | 2 | 2 | 2 | 4864 | 38 | 0.0335 | -0.0057 | -0.0392 | 0.0392 | current artifact deterministic PUCT |
| capture_available-006 | 5000 | 37 | 2 | 2 | 2 | 4864 | 38 | 0.0335 | -0.0057 | -0.0392 | 0.0392 | current artifact deterministic PUCT |
| capture_available-006 | 5000 | 42 | 2 | 2 | 2 | 4864 | 38 | 0.0335 | -0.0057 | -0.0392 | 0.0392 | current artifact deterministic PUCT |
| capture_available-006 | 5000 | 101 | 2 | 2 | 2 | 4864 | 38 | 0.0335 | -0.0057 | -0.0392 | 0.0392 | current artifact deterministic PUCT |
| capture_available-006 | 5000 | 202 | 2 | 2 | 2 | 4864 | 38 | 0.0335 | -0.0057 | -0.0392 | 0.0392 | current artifact deterministic PUCT |
| capture_available-006 | 5000 | 303 | 2 | 2 | 2 | 4864 | 38 | 0.0335 | -0.0057 | -0.0392 | 0.0392 | current artifact deterministic PUCT |
| capture_available-007 | 5000 | 11 | 2 | 1 | 2 | 4911 | 9 | 0.1628 | -0.0478 | -0.2105 | 0.1538 | current artifact deterministic PUCT |
| capture_available-007 | 5000 | 23 | 2 | 1 | 2 | 4911 | 9 | 0.1628 | -0.0478 | -0.2105 | 0.1538 | current artifact deterministic PUCT |
| capture_available-007 | 5000 | 37 | 2 | 1 | 2 | 4911 | 9 | 0.1628 | -0.0478 | -0.2105 | 0.1538 | current artifact deterministic PUCT |
| capture_available-007 | 5000 | 42 | 2 | 1 | 2 | 4911 | 9 | 0.1628 | -0.0478 | -0.2105 | 0.1538 | current artifact deterministic PUCT |
| capture_available-007 | 5000 | 101 | 2 | 1 | 2 | 4911 | 9 | 0.1628 | -0.0478 | -0.2105 | 0.1538 | current artifact deterministic PUCT |
| capture_available-007 | 5000 | 202 | 2 | 1 | 2 | 4911 | 9 | 0.1628 | -0.0478 | -0.2105 | 0.1538 | current artifact deterministic PUCT |
| capture_available-007 | 5000 | 303 | 2 | 1 | 2 | 4911 | 9 | 0.1628 | -0.0478 | -0.2105 | 0.1538 | current artifact deterministic PUCT |
| capture_available-008 | 5000 | 11 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.059 | current artifact deterministic PUCT |
| capture_available-008 | 5000 | 23 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.059 | current artifact deterministic PUCT |
| capture_available-008 | 5000 | 37 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.059 | current artifact deterministic PUCT |
| capture_available-008 | 5000 | 42 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.059 | current artifact deterministic PUCT |
| capture_available-008 | 5000 | 101 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.059 | current artifact deterministic PUCT |
| capture_available-008 | 5000 | 202 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.059 | current artifact deterministic PUCT |
| capture_available-008 | 5000 | 303 | 1 | 1 | 2 | 17 | 13 | -0.0238 | -0.0183 | 0.0055 | 0.059 | current artifact deterministic PUCT |

PUCT stability:

| row_id | budget | seeds | observed_reference_moves | majority_move | majority_fraction | old_reference_move | stable | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capture_available-002 | 384 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 4 | true | old_target_invalid |
| capture_available-002 | 1200 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 4 | true | old_target_invalid |
| capture_available-002 | 2400 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 4 | true | old_target_invalid |
| capture_available-002 | 5000 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 4 | true | old_target_invalid |
| capture_available-003 | 384 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 1 | true | old_target_invalid |
| capture_available-003 | 1200 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 1 | true | old_target_invalid |
| capture_available-003 | 2400 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 1 | true | old_target_invalid |
| capture_available-003 | 5000 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 1 | true | old_target_invalid |
| capture_available-006 | 384 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 2 | true | old_target_still_valid |
| capture_available-006 | 1200 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 2 | true | old_target_still_valid |
| capture_available-006 | 2400 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 2 | true | old_target_still_valid |
| capture_available-006 | 5000 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 2 | true | old_target_still_valid |
| capture_available-007 | 384 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 1 | true | old_target_invalid |
| capture_available-007 | 1200 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 1 | true | old_target_invalid |
| capture_available-007 | 2400 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 1 | true | old_target_invalid |
| capture_available-007 | 5000 | `[11, 23, 37, 42, 101, 202, 303]` | `[2]` | 2 | 1.0 | 1 | true | old_target_invalid |
| capture_available-008 | 384 | `[11, 23, 37, 42, 101, 202, 303]` | `[1]` | 1 | 1.0 | 1 | true | old_target_still_valid |
| capture_available-008 | 1200 | `[11, 23, 37, 42, 101, 202, 303]` | `[1]` | 1 | 1.0 | 1 | true | old_target_still_valid |
| capture_available-008 | 2400 | `[11, 23, 37, 42, 101, 202, 303]` | `[1]` | 1 | 1.0 | 1 | true | old_target_still_valid |
| capture_available-008 | 5000 | `[11, 23, 37, 42, 101, 202, 303]` | `[1]` | 1 | 1.0 | 1 | true | old_target_still_valid |

## Candidate reference artifact

- Artifact scope: `audited_subset`.
- Included row ids: `["capture_available-002", "capture_available-003", "capture_available-007", "capture_available-008"]`.
- Missing requested incumbent rows: `["capture_available-006"]`.

## Old vs adjudicated reference comparison

| row_id | old_reference_move | adjudicated_reference_move | reference_unstable | observed_reference_moves | decision | notes |
| --- | --- | --- | --- | --- | --- | --- |
| capture_available-002 | 4 | 2 | false | `[2]` | old_target_invalid | old_reference_state_mismatch, reference_should_flip_to_move_2 |
| capture_available-003 | 1 | 2 | false | `[2]` | old_target_invalid | old_reference_state_mismatch |
| capture_available-006 | 2 | 2 | false | `[2]` | old_target_still_valid | - |
| capture_available-007 | 1 | 2 | false | `[2]` | old_target_invalid | old_reference_state_mismatch |
| capture_available-008 | 1 | 1 | false | `[1]` | old_target_still_valid | old_reference_state_mismatch |

## Interpretation

- `capture_available-002` classification: `reference_should_flip_to_move_2`.
- Highest-budget ClassicMCTS majority move for `002`: `2`.
- Highest-budget PUCT majority move for `002`: `2`.
- Highest-budget child-afterstate ordering for `002`: `move_2_gt_move_4`.

## Exactly One Recommended Next Action

Recommendation: **update forensic reference artifacts and remove capture_available-002 from model-failure guard status, then rerun prior local diagnostics with corrected reference labels**.
