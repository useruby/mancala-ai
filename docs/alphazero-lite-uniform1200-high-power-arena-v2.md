# Uniform1200 High-Power Paired Arena

Inherited PR #325 classification: `uniform1200_six_seed_exact_gain_arena_uncertain`.

Hard classification: `uniform1200_high_power_strength_confirmed`.

## Frozen Design

- Opening suite: 256 model-independent four-ply prefixes, SHA256 `c4f74fe141cae1ea1aebe0fb97865f96864172cdd806ed9d7984184c6c8d9b9d`, canonical uniqueness 256/256.
- V1 failed before games: 233/256 canonical states (91.02%). V2 enumerates all legal four-ply prefixes, groups by canonical resulting-state hash, retains the lexicographically smallest prefix, then ranks states by stable hash of `uniform1200_high_power_unique_v2`, seed `90417`, and state hash without replacement.
- Four-ply population: 942 legal prefixes and 942 canonical states; multiplicity min/median/p90/max 1/1.0/1.0/1.
- V1/V2 canonical-state overlap: 76.
- Checkpoint preflight: all 12 frozen control/uniform1200 weight SHA256 values match the PR #326 aggregate.
- Arena: 2 seat-swapped games/opening, 384/384 simulations, c_puct 1.25, deterministic PUCT, zero FPU, seed `90417`, contract `azlite_eval_seed_v1`, 24 workers.

## Per-Seed Results

| Seed | W/D/L | Pair score | Pair CI95 | Median pair | Seat P0/P1 | Margin mean/median | Trajectories unique/duplicate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 47 | 304/47/161 | 0.6396 | [0.6074, 0.6709] | 0.5000 | 0.6973/0.5820 | 3.41/4.00 | 511/2 |
| 48 | 333/32/147 | 0.6816 | [0.6494, 0.7139] | 0.5000 | 0.7695/0.5938 | 4.38/6.00 | 512/0 |
| 49 | 319/35/158 | 0.6572 | [0.6240, 0.6904] | 0.5000 | 0.7207/0.5938 | 4.21/6.00 | 512/0 |
| 50 | 307/34/171 | 0.6328 | [0.6006, 0.6650] | 0.5000 | 0.7129/0.5527 | 4.09/6.00 | 512/0 |
| 51 | 300/39/173 | 0.6240 | [0.5889, 0.6582] | 0.5000 | 0.6973/0.5508 | 2.62/4.00 | 512/0 |
| 52 | 307/37/168 | 0.6357 | [0.6035, 0.6680] | 0.5000 | 0.7266/0.5449 | 3.67/4.00 | 512/0 |

## Aggregate

Effects: +0.1396, +0.1816, +0.1572, +0.1328, +0.1240, +0.1357. Mean `+0.1452`, median `+0.1377`; positive/neutral/negative `6/0/0`.
Hierarchical 95% CI for grand score: `[0.6258, 0.6660]`.
Repeated strength openings: `63` IDs `[1, 2, 7, 9, 16, 21, 27, 28, 30, 34, 35, 42, 43, 48, 49, 52, 54, 55, 61, 63, 70, 83, 84, 85, 89, 107, 108, 117, 126, 131, 134, 135, 136, 143, 144, 148, 150, 151, 154, 167, 173, 176, 190, 191, 200, 201, 208, 211, 213, 214, 215, 222, 232, 233, 234, 235, 240, 243, 245, 246, 247, 249, 255]`. Repeated weakness openings: `1` IDs `[204]`.

## Frozen Exact Context

PR #325 exact W/D/L improved in all six matched pairs; no true win-to-loss regression repeated at 4/6; the outcome-aligned shadow gate passed 6/6. These exact results were reused, not rerun. PR #326 did not execute arena games because V1 preflight failed.

## Next Experiment

Run ONE promotion-candidate confirmation using a pre-registered candidate-selection rule over the six frozen uniform1200 checkpoints against the unchanged incumbent with the full production battery.
