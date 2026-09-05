# Kalah V1 Native Tablebase Preflight Results

**Classification:** `canonical_tablebase_feasible`

The committed raw result is `docs/data/alphazero-lite-kalah-v1-native-tablebase-preflight-full.json`. This isolated C++17 experiment does not integrate lookup into production search, generate an 18-stone teacher, or start training.

## Correctness

| Gate | Threshold | Observed | Status |
| --- | --- | --- | --- |
| Rank/unrank | exhaustive through tier 10 | 1,293,292 indexed states | pass |
| Transition parity | exhaustive through tier 8 | 251,940 states; 604,656 legal actions | pass |
| Root/action oracle | exhaustive through tier 8 | 251,940 roots; 604,656 actions | pass |
| Store/action order | exhaustive through tier 8 | 251,940 roots, two store offsets | pass |
| Independent oracle | 10,000 fixed queries, seed 277 | 10,000 | pass |
| Cited regressions | both positions | both exact action maps | pass |
| Determinism | fresh tiers 8 and 12 | byte-identical | pass |
| Portable format | reject malformed inputs | 19 fixtures rejected | pass |

Tier 8 produced payload SHA-256 `441341d8825a19ab6126e8194f7045ec36f869173e788bcccce544cfff68ef94` and complete-file SHA-256 `94ee7c4698bb4b5aa94ce80a1d6900f46a5221c5686f8aa44e2e2cf5e9d3c5cf`. Tier 12 produced payload SHA-256 `d4728322ad902d504a700caef6b74b819453163420e7417f9227e279ad69595a` and complete-file SHA-256 `f3ff687c9aa7309d93d1ea2647d8e8e21a8ded43e2e943b8511d5a0a8b4ba452`. All generation runs recorded zero cycles.

## Scalability

Each generation enforced a 30-minute wall/CPU cap, 8 GiB address-space cap, 8 GiB temporary-disk cap, and two complete-file bytes per indexed state.

| Tier | Indexed states | Wall time | Peak RSS | Output bytes | Edges | File SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 251,940 | 0.085 s | 4,280 KiB | 252,164 | 604,656 | `94ee7c4698bb4b5aa94ce80a1d6900f46a5221c5686f8aa44e2e2cf5e9d3c5cf` |
| 10 | 1,293,292 | 0.564 s | 6,324 KiB | 1,293,548 | 3,527,160 | `75088e978b7586d79767de4b16097dc81089644da207e41ee65a6cad64d28884` |
| 12 | 5,408,312 | 2.972 s | 21,928 KiB | 5,408,600 | 16,224,936 | `f3ff687c9aa7309d93d1ea2647d8e8e21a8ded43e2e943b8511d5a0a8b4ba452` |
| 14 | 19,315,400 | 12.605 s | 70,464 KiB | 19,315,720 | 62,403,600 | `a1e7aebe6ed4f35de28466f8a91004fba36b9b55b31c55c5edecba32f42028ea` |

The native in-process fixed seed-277 corpus hash is `cce36676004502e358a5d683f7d37a82d2cd528117225f3865a491c23e7ee92a`. It loaded the tier-14 file in 112.6 ms, used 129,892 KiB RSS, and completed 100,000 warm lookups at 8.86 million/s, with 87 ns median and 211 ns p95 latency. These pass the 1 ms median/p95 and 100,000/s registered thresholds.

## Tier-18 Projection

Using the tier-14 measurements, cumulative-state ratio 8.9559, and the required 2x safety factor projects 225.8 seconds generation, 1,262,135 KiB peak RSS, and 345,978,632 output bytes. These are below the 8-hour, 16 GiB, and 4 GiB limits respectively. The raw result contains the measured inputs and thresholds.

A separate PR may now evaluate the 18-stone teacher. It must retain this experiment's isolation until that PR has its own review and validation.
