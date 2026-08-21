# PR #214 PUCT Divergence

**Classification:** `distributed_prior_boundary_crossings`

**Recommended follow-up:** Build a search-margin-weighted policy sensitivity metric across replay states.

## First Game Divergences

| Opening | Seat | Game ply | Player | Treatment | Control | Outcome |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0 | 2 | 0 | 3 | 1 | current |
| 1 | 0 | 2 | 0 | 3 | 1 | current |
| 8 | 0 | 2 | 0 | 3 | 1 | current |
| 8 | 0 | 2 | 0 | 3 | 1 | current |
| 33 | 1 | 7 | 1 | 8 | 6 | challenger |
| 33 | 1 | 7 | 1 | 8 | 6 | challenger |
| 36 | 1 | 7 | 1 | 8 | 6 | challenger |
| 36 | 1 | 7 | 1 | 8 | 6 | challenger |
| 49 | 1 | 7 | 1 | 8 | 6 | challenger |
| 49 | 1 | 7 | 1 | 8 | 6 | challenger |
| 60 | 1 | 7 | 1 | 8 | 6 | challenger |
| 60 | 1 | 7 | 1 | 8 | 6 | challenger |
| 71 | 0 | 2 | 0 | 3 | 1 | current |
| 71 | 0 | 2 | 0 | 3 | 1 | current |
| 80 | 0 | 2 | 0 | 3 | 1 | current |
| 80 | 0 | 2 | 0 | 3 | 1 | current |
| 101 | 1 | 7 | 1 | 8 | 6 | challenger |
| 101 | 1 | 7 | 1 | 8 | 6 | challenger |
| 110 | 1 | 7 | 1 | 8 | 6 | challenger |
| 110 | 1 | 7 | 1 | 8 | 6 | challenger |
| 123 | 0 | 2 | 0 | 3 | 1 | current |
| 123 | 0 | 2 | 0 | 3 | 1 | current |
| 126 | 1 | 7 | 1 | 8 | 6 | challenger |
| 126 | 1 | 7 | 1 | 8 | 6 | challenger |

## First Search Divergences

| State hash | First simulation | Depth | A16 action | P1 action | U-only | Prefix-8 root |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `6d7a71e6007e3943a223024aa1515659887ecb275806402f645a5b6367f942fe` | 28 | 0 | 5 | 3 | True | 3 |
| `cd6293ed266fb1db26224cb9208d2494bd9f35be6d37526c55b2a64437e98d8c` | 50 | 1 | 5 | 2 | True | 2 |

The JSON summary retains the complete PUCT decomposition, continuous checkpoint snapshots, and progressive interventions.

## PR #219

Step-46 artifacts were unavailable locally. Recreating them would require prohibited training, so corroboration was not run.
