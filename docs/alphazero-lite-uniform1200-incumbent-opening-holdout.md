# Uniform1200 Incumbent Opening Holdout

Inherited classifications: PR #327 `uniform1200_high_power_strength_confirmed`; PR #328 `uniform1200_promotion_arena_failed`.

Candidate SHA256: `935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c`. Incumbent SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a` (`azlite-balanced-w8s4-policy-head-e1`).

The immutable PR #327 suite SHA256 was verified as `c4f74fe141cae1ea1aebe0fb97865f96864172cdd806ed9d7984184c6c8d9b9d`. Its 256 hashes were excluded from the mechanically enumerated 942 four-ply canonical states, leaving 686; selected-suite overlap is 0/256. The new suite SHA256 is `5d1f5982c990d00bce0a57161c1ae710bed6b8ced831ac2c1551af5217d2716c`. It ranks the remainder by stable hash of `uniform1200_incumbent_holdout_unique_v1`, seed 42, and canonical state hash.

Arena semantics exactly preserve PR #328: candidate/incumbent simulations 384/256, c_puct 1.25, zero FPU, no subtree reuse or normalization, deterministic root policy, zero tactical bias and root temperature, no root-prior transform, no opening cache, no value transform, base seed 42, seed contract `azlite_eval_seed_v2`.

## Results

- W/D/L: 443/19/50 across 512 games; raw score `0.8838`.
- Candidate P0 W/D/L: {'wins': 225, 'draws': 9, 'losses': 22}; P1: {'wins': 218, 'draws': 10, 'losses': 28}. Scores P0/P1: `0.8965`/`0.8711`.
- Mean/median opening-pair score: `0.8838`/`1.0000`; paired bootstrap CI95 `[0.8564, 0.9102]`.
- Mean/median candidate stone margin: `11.06`/`10.00`. Unique/duplicate trajectories: 512/0 (0.0000).
- Opening pairs: candidate wins both `193`, incumbent wins both `3`, one win each `41`, one/both draws `19`.
- Candidate/neutral/incumbent-favored openings: 209/41/6. Seat-pair balance 0/0.5/1/1.5/2: {'0': 3, '0.5': 3, '1': 41, '1.5': 16, '2': 193}.
- Existing exact-reference coverage: `5` holdout openings; no new tablebase was generated.
- Difficulty descriptors, candidate-favored/neutral/incumbent-favored: counts `209`/`41`/`6`; capture available `17`/`5`/`0`; extra turn available `111`/`21`/`4`. Full current-player, legal-action, store, and pit-stone distributions are frozen in the aggregate JSON.
- Threshold margin: `+0.3338`. Point estimate exceeds 0.55: `True`; exceeds 0.50: `True`; CI lower bound above 0.50: `True`.

Compared descriptively with frozen PR #328 start-state W/D/L 60/0/60, score 0.5000 and threshold margin -0.05, diversity gain is `+0.3838`.

Frozen W/D/L shadow safety remains valid: candidate/incumbent top1 0.9296/0.8732, policy mass 0.4078/0.3577, regret 0.1174/0.2207, blunder rate 0.0704/0.1268, win-to-draw 5/5, win-to-loss 10/20.

## Classification

`production_start_state_arena_disagreement_confirmed`

Next experiment: Audit the production arena prefilter against canonical-unique suites across frozen historical pairs with known strength ordering.
