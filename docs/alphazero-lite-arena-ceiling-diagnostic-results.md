# AlphaZero-Lite Arena Ceiling Diagnostic Results

**Date:** 2026-06-08

## Classification

**SEARCH_BUDGET_LIMITING, SEAT_OR_OPENING_ARTIFACT**

## Artifacts

| Artifact | Path | SHA256 |
|----------|------|--------|
| current | model-artifact/current | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| iter0_reference | /tmp/azlite_iterative_random_replay/iter0_candidate_artifact | `0bbeaa9c6954ea147b084cdb3e68355887df1b0ab34c1fbc952b7988fdddc3bd` |
| iter1_continue_no_new_data | /tmp/azlite_iterative_random_replay/iter1_continue_no_new_data/artifact | `a4a86dceb064a2be63d186785e9e3e30e5e5bed52bd4be2fb878d42edc45a5a9` |
| iter1_candidate_random_replay | /tmp/azlite_iterative_random_replay/iter1_candidate_random_replay/artifact | `777b25d4c5a601ff4b3cdc3a750550f2fee2756da296ad15c300b0d749b61b44` |

## Arena Results by Candidate and Budget Pair

| Candidate | Chall Sims | Curr Sims | Games | Wins | Losses | Draws | Score | CI95 Lower | CI95 Upper | P0 Score | P1 Score | Margin Mean | Game Len Mean | Dup Traj | Move Time ms | Move Time p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iter0_reference | 128 | 128 | 120 | 0 | 120 | 0 | 0.0000 | 0.0000 | 0.0310 | 0.0000 | 0.0000 | -9.2 | 37.3 | 15 | 8.5 | 11.8 |
| iter0_reference | 256 | 256 | 120 | 60 | 60 | 0 | 0.5000 | 0.4119 | 0.5881 | 1.0000 | 0.0000 | -3.9 | 41.1 | 15 | 23.4 | 32.9 |
| iter0_reference (ext) | 256 | 256 | 240 | 120 | 120 | 0 | 0.5000 | 0.4372 | 0.5628 | — | — | — | — | — | 23.3 | 32.8 |
| iter0_reference | 384 | 384 | 120 | 60 | 60 | 0 | 0.5000 | 0.4119 | 0.5881 | 1.0000 | 0.0000 | -7.7 | 49.3 | 15 | 33.3 | 49.7 |
| iter0_reference (ext) | 384 | 384 | 240 | 120 | 120 | 0 | 0.5000 | 0.4372 | 0.5628 | — | — | — | — | — | 33.3 | 49.9 |
| iter0_reference | 768 | 768 | 120 | 60 | 60 | 0 | 0.5000 | 0.4119 | 0.5881 | 1.0000 | 0.0000 | -3.5 | 34.7 | 15 | 70.4 | 103.5 |
| iter0_reference (ext) | 768 | 768 | 240 | 120 | 120 | 0 | 0.5000 | 0.4372 | 0.5628 | — | — | — | — | — | 70.8 | 103.8 |
| iter0_reference | 1200 | 1200 | 120 | 120 | 0 | 0 | 1.0000 | 0.9690 | 1.0000 | 1.0000 | 1.0000 | 12.5 | 41.7 | 15 | 112.6 | 165.2 |
| iter0_reference | 384 | 256 | 120 | 60 | 60 | 0 | 0.5000 | 0.4119 | 0.5881 | 1.0000 | 0.0000 | -4.9 | 38.9 | 15 | 29.5 | 46.2 |
| iter0_reference (ext) | 384 | 256 | 240 | 120 | 120 | 0 | 0.5000 | 0.4372 | 0.5628 | — | — | — | — | — | 29.5 | 46.2 |
| iter0_reference | 768 | 256 | 120 | 60 | 60 | 0 | 0.5000 | 0.4119 | 0.5881 | 1.0000 | 0.0000 | 3.1 | 31.7 | 15 | 51.7 | 94.7 |
| iter0_reference (ext) | 768 | 256 | 240 | 120 | 120 | 0 | 0.5000 | 0.4372 | 0.5628 | — | — | — | — | — | 51.4 | 94.1 |
| iter0_reference | 1200 | 256 | 120 | 60 | 60 | 0 | 0.5000 | 0.4119 | 0.5881 | 1.0000 | 0.0000 | 1.2 | 34.6 | 15 | 66.2 | 148.2 |
| iter0_reference (ext) | 1200 | 256 | 240 | 120 | 120 | 0 | 0.5000 | 0.4372 | 0.5628 | — | — | — | — | — | 66.1 | 148.2 |
| iter0_reference | 256 | 768 | 120 | 60 | 60 | 0 | 0.5000 | 0.4119 | 0.5881 | 1.0000 | 0.0000 | -6.7 | 36.0 | 15 | 51.9 | 102.2 |
| iter0_reference (ext) | 256 | 768 | 240 | 120 | 120 | 0 | 0.5000 | 0.4372 | 0.5628 | — | — | — | — | — | 52.3 | 103.2 |
| iter1_continue_no_new_data | 128 | 128 | 120 | 0 | 60 | 60 | 0.2500 | 0.1811 | 0.3344 | 0.5000 | 0.0000 | -8.5 | 36.0 | 15 | 10.9 | 16.1 |
| iter1_continue_no_new_data | 256 | 256 | 120 | 0 | 120 | 0 | 0.0000 | 0.0000 | 0.0310 | 0.0000 | 0.0000 | -15.9 | 37.1 | 15 | 21.4 | 32.5 |
| iter1_continue_no_new_data | 384 | 384 | 120 | 60 | 60 | 0 | 0.5000 | 0.4119 | 0.5881 | 1.0000 | 0.0000 | -6.8 | 45.5 | 15 | 33.4 | 50.2 |
| iter1_continue_no_new_data (ext) | 384 | 384 | 240 | 120 | 120 | 0 | 0.5000 | 0.4372 | 0.5628 | — | — | — | — | — | 33.4 | 50.2 |
| iter1_continue_no_new_data | 768 | 768 | 120 | 60 | 60 | 0 | 0.5000 | 0.4119 | 0.5881 | 1.0000 | 0.0000 | -10.9 | 39.3 | 15 | 65.3 | 102.6 |
| iter1_continue_no_new_data (ext) | 768 | 768 | 240 | 120 | 120 | 0 | 0.5000 | 0.4372 | 0.5628 | — | — | — | — | — | 65.4 | 103.0 |
| iter1_continue_no_new_data | 1200 | 1200 | 120 | 60 | 0 | 60 | 0.7500 | 0.6656 | 0.8189 | 1.0000 | 0.5000 | 2.1 | 40.7 | 15 | 102.8 | 164.6 |
| iter1_continue_no_new_data | 384 | 256 | 120 | 60 | 60 | 0 | 0.5000 | 0.4119 | 0.5881 | 1.0000 | 0.0000 | -6.8 | 38.6 | 15 | 28.2 | 46.5 |
| iter1_continue_no_new_data (ext) | 384 | 256 | 240 | 120 | 120 | 0 | 0.5000 | 0.4372 | 0.5628 | — | — | — | — | — | 28.3 | 46.6 |
| iter1_continue_no_new_data | 768 | 256 | 120 | 60 | 60 | 0 | 0.5000 | 0.4119 | 0.5881 | 1.0000 | 0.0000 | -6.1 | 35.3 | 15 | 42.9 | 94.0 |
| iter1_continue_no_new_data (ext) | 768 | 256 | 240 | 120 | 120 | 0 | 0.5000 | 0.4372 | 0.5628 | — | — | — | — | — | 43.3 | 94.7 |
| iter1_continue_no_new_data | 1200 | 256 | 120 | 120 | 0 | 0 | 1.0000 | 0.9690 | 1.0000 | 1.0000 | 1.0000 | 10.1 | 26.3 | 15 | 59.7 | 151.5 |
| iter1_continue_no_new_data | 256 | 768 | 120 | 0 | 120 | 0 | 0.0000 | 0.0000 | 0.0310 | 0.0000 | 0.0000 | -17.1 | 34.1 | 15 | 50.8 | 102.7 |
| iter1_candidate_random_replay | 128 | 128 | 120 | 60 | 60 | 0 | 0.5000 | 0.4119 | 0.5881 | 1.0000 | 0.0000 | -0.3 | 32.7 | 15 | 12.7 | 16.2 |
| iter1_candidate_random_replay (ext) | 128 | 128 | 240 | 120 | 120 | 0 | 0.5000 | 0.4372 | 0.5628 | — | — | — | — | — | 12.7 | 16.2 |
| iter1_candidate_random_replay | 256 | 256 | 120 | 0 | 120 | 0 | 0.0000 | 0.0000 | 0.0310 | 0.0000 | 0.0000 | -13.5 | 42.3 | 15 | 22.5 | 32.7 |
| iter1_candidate_random_replay | 384 | 384 | 120 | 60 | 60 | 0 | 0.5000 | 0.4119 | 0.5881 | 1.0000 | 0.0000 | -5.7 | 40.7 | 15 | 36.4 | 50.1 |
| iter1_candidate_random_replay (ext) | 384 | 384 | 240 | 120 | 120 | 0 | 0.5000 | 0.4372 | 0.5628 | — | — | — | — | — | 36.5 | 50.1 |
| iter1_candidate_random_replay | 768 | 768 | 120 | 60 | 60 | 0 | 0.5000 | 0.4119 | 0.5881 | 1.0000 | 0.0000 | -2.4 | 42.2 | 15 | 67.4 | 102.0 |
| iter1_candidate_random_replay (ext) | 768 | 768 | 240 | 120 | 120 | 0 | 0.5000 | 0.4372 | 0.5628 | — | — | — | — | — | 67.8 | 102.7 |
| iter1_candidate_random_replay | 1200 | 1200 | 120 | 60 | 60 | 0 | 0.5000 | 0.4119 | 0.5881 | 1.0000 | 0.0000 | -2.4 | 32.9 | 15 | 104.6 | 163.8 |
| iter1_candidate_random_replay (ext) | 1200 | 1200 | 240 | 120 | 120 | 0 | 0.5000 | 0.4372 | 0.5628 | — | — | — | — | — | 105.0 | 164.9 |
| iter1_candidate_random_replay | 384 | 256 | 120 | 60 | 60 | 0 | 0.5000 | 0.4119 | 0.5881 | 1.0000 | 0.0000 | -6.8 | 38.6 | 15 | 28.4 | 46.5 |
| iter1_candidate_random_replay (ext) | 384 | 256 | 240 | 120 | 120 | 0 | 0.5000 | 0.4372 | 0.5628 | — | — | — | — | — | 28.4 | 46.6 |
| iter1_candidate_random_replay | 768 | 256 | 120 | 60 | 60 | 0 | 0.5000 | 0.4119 | 0.5881 | 1.0000 | 0.0000 | -0.7 | 33.7 | 15 | 47.6 | 95.0 |
| iter1_candidate_random_replay (ext) | 768 | 256 | 240 | 120 | 120 | 0 | 0.5000 | 0.4372 | 0.5628 | — | — | — | — | — | 47.6 | 95.0 |
| iter1_candidate_random_replay | 1200 | 256 | 120 | 60 | 60 | 0 | 0.5000 | 0.4119 | 0.5881 | 1.0000 | 0.0000 | -3.3 | 34.3 | 15 | 66.1 | 149.9 |
| iter1_candidate_random_replay (ext) | 1200 | 256 | 240 | 120 | 120 | 0 | 0.5000 | 0.4372 | 0.5628 | — | — | — | — | — | 66.8 | 151.6 |
| iter1_candidate_random_replay | 256 | 768 | 120 | 0 | 120 | 0 | 0.0000 | 0.0000 | 0.0310 | 0.0000 | 0.0000 | -16.8 | 39.0 | 15 | 46.6 | 102.2 |

## Forced Seat-Split Results

| Candidate | Chall Sims | Curr Sims | Starts | Games | Wins | Losses | Draws | Score |
|---|---|---|---|---|---|---|---|---|
| iter0_reference | 128 | 128 | 0 | 120 | 0 | 120 | 0 | 0.0000 |
| iter0_reference | 128 | 128 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter0_reference | 256 | 256 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter0_reference | 256 | 256 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter0_reference | 384 | 384 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter0_reference | 384 | 384 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter0_reference | 768 | 768 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter0_reference | 768 | 768 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter0_reference | 1200 | 1200 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter0_reference | 1200 | 1200 | 1 | 120 | 120 | 0 | 0 | 1.0000 |
| iter0_reference | 384 | 256 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter0_reference | 384 | 256 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter0_reference | 768 | 256 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter0_reference | 768 | 256 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter0_reference | 1200 | 256 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter0_reference | 1200 | 256 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter0_reference | 256 | 768 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter0_reference | 256 | 768 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_continue_no_new_data | 128 | 128 | 0 | 120 | 0 | 0 | 120 | 0.5000 |
| iter1_continue_no_new_data | 128 | 128 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_continue_no_new_data | 256 | 256 | 0 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_continue_no_new_data | 256 | 256 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_continue_no_new_data | 384 | 384 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter1_continue_no_new_data | 384 | 384 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_continue_no_new_data | 768 | 768 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter1_continue_no_new_data | 768 | 768 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_continue_no_new_data | 1200 | 1200 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter1_continue_no_new_data | 1200 | 1200 | 1 | 120 | 0 | 0 | 120 | 0.5000 |
| iter1_continue_no_new_data | 384 | 256 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter1_continue_no_new_data | 384 | 256 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_continue_no_new_data | 768 | 256 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter1_continue_no_new_data | 768 | 256 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_continue_no_new_data | 1200 | 256 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter1_continue_no_new_data | 1200 | 256 | 1 | 120 | 120 | 0 | 0 | 1.0000 |
| iter1_continue_no_new_data | 256 | 768 | 0 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_continue_no_new_data | 256 | 768 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_candidate_random_replay | 128 | 128 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter1_candidate_random_replay | 128 | 128 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_candidate_random_replay | 256 | 256 | 0 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_candidate_random_replay | 256 | 256 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_candidate_random_replay | 384 | 384 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter1_candidate_random_replay | 384 | 384 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_candidate_random_replay | 768 | 768 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter1_candidate_random_replay | 768 | 768 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_candidate_random_replay | 1200 | 1200 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter1_candidate_random_replay | 1200 | 1200 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_candidate_random_replay | 384 | 256 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter1_candidate_random_replay | 384 | 256 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_candidate_random_replay | 768 | 256 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter1_candidate_random_replay | 768 | 256 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_candidate_random_replay | 1200 | 256 | 0 | 120 | 120 | 0 | 0 | 1.0000 |
| iter1_candidate_random_replay | 1200 | 256 | 1 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_candidate_random_replay | 256 | 768 | 0 | 120 | 0 | 120 | 0 | 0.0000 |
| iter1_candidate_random_replay | 256 | 768 | 1 | 120 | 0 | 120 | 0 | 0.0000 |

## First Move Distributions

| Candidate | Chall Sims | Curr Sims | Challenger First Moves | Current First Moves |
|---|---|---|---|---|
| iter0_reference | 128 | 128 | {"1": 8, "2": 7} | {"2": 8, "1": 7} |
| iter0_reference | 256 | 256 | {"1": 8, "2": 7} | {"2": 8, "1": 7} |
| iter0_reference | 384 | 384 | {"2": 8, "1": 7} | {"1": 8, "2": 7} |
| iter0_reference | 768 | 768 | {"1": 8, "2": 7} | {"2": 8, "1": 7} |
| iter0_reference | 1200 | 1200 | {"1": 8, "2": 7} | {"2": 8, "1": 7} |
| iter0_reference | 384 | 256 | {"1": 8, "2": 7} | {"2": 8, "1": 7} |
| iter0_reference | 768 | 256 | {"1": 8, "2": 7} | {"2": 8, "1": 7} |
| iter0_reference | 1200 | 256 | {"2": 8, "1": 7} | {"1": 8, "2": 7} |
| iter0_reference | 256 | 768 | {"1": 8, "2": 7} | {"2": 8, "1": 7} |
| iter1_continue_no_new_data | 128 | 128 | {"1": 8, "2": 7} | {"2": 8, "1": 7} |
| iter1_continue_no_new_data | 256 | 256 | {"2": 8, "5": 7} | {"1": 8, "2": 7} |
| iter1_continue_no_new_data | 384 | 384 | {"2": 8, "1": 7} | {"1": 8, "2": 7} |
| iter1_continue_no_new_data | 768 | 768 | {"5": 8, "2": 7} | {"2": 8, "1": 7} |
| iter1_continue_no_new_data | 1200 | 1200 | {"2": 8, "1": 7} | {"1": 8, "2": 7} |
| iter1_continue_no_new_data | 384 | 256 | {"2": 8, "1": 7} | {"1": 8, "2": 7} |
| iter1_continue_no_new_data | 768 | 256 | {"5": 8, "2": 7} | {"2": 8, "1": 7} |
| iter1_continue_no_new_data | 1200 | 256 | {"2": 8, "1": 7} | {"1": 8, "2": 7} |
| iter1_continue_no_new_data | 256 | 768 | {"5": 8, "2": 7} | {"2": 8, "1": 7} |
| iter1_candidate_random_replay | 128 | 128 | {"1": 8, "2": 7} | {"2": 8, "1": 7} |
| iter1_candidate_random_replay | 256 | 256 | {"5": 8, "2": 7} | {"2": 8, "1": 7} |
| iter1_candidate_random_replay | 384 | 384 | {"2": 8, "1": 7} | {"1": 8, "2": 7} |
| iter1_candidate_random_replay | 768 | 768 | {"5": 8, "2": 7} | {"2": 8, "1": 7} |
| iter1_candidate_random_replay | 1200 | 1200 | {"2": 8, "5": 7} | {"1": 8, "2": 7} |
| iter1_candidate_random_replay | 384 | 256 | {"2": 8, "1": 7} | {"1": 8, "2": 7} |
| iter1_candidate_random_replay | 768 | 256 | {"2": 8, "5": 7} | {"1": 8, "2": 7} |
| iter1_candidate_random_replay | 1200 | 256 | {"5": 8, "2": 7} | {"2": 8, "1": 7} |
| iter1_candidate_random_replay | 256 | 768 | {"2": 8, "5": 7} | {"1": 8, "2": 7} |

## Raw Policy Diagnostic

| Metric | iter0_reference | iter1_continue_no_new_data | iter1_candidate_random_replay |
|---|---|---|---|
| candidate_classic_mcts_top_move_agreement | 0.4820 | 0.4860 | 0.4980 |
| candidate_current_top_move_agreement | 0.6520 | 0.6140 | 0.6200 |
| candidate_policy_entropy_mean | 0.8805 | 0.8339 | 0.8600 |
| candidate_value_mean | 0.0165 | 0.0354 | -0.0533 |
| candidate_value_std | 0.4299 | 0.4811 | 0.4339 |
| current_classic_mcts_top_move_agreement | 0.4780 | 0.4780 | 0.4780 |
| current_policy_entropy_mean | 0.7350 | 0.7350 | 0.7350 |
| current_value_mean | 0.0530 | 0.0530 | 0.0530 |
| current_value_std | 0.3915 | 0.3915 | 0.3915 |
| sampled_states | 500 | 500 | 500 |

## Classification Rationale

Classification: **search_budget_limiting, seat_or_opening_artifact**

### Candidate-vs-Current Score by Search Budget (challenger:current)

Candidate | 128:128 | 256:256 | 384:384 | 768:768 | 1200:1200 | 384:256 | 768:256 | 1200:256 | 256:768
---|---|---|---|---|---|---|---|---|---
| iter0_reference | 0.0000 | 0.5000 | 0.5000 | 0.5000 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| iter1_candidate_random_replay | 0.5000 | 0.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.0000 |
| iter1_continue_no_new_data | 0.2500 | 0.0000 | 0.5000 | 0.5000 | 0.7500 | 0.5000 | 0.5000 | 1.0000 | 0.0000 |

