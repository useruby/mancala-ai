# Seed542 update-432 128-simulation strength screen

## Registered decision

The update-432 candidate did not pass the registered strength gate against current serving P.

| Measure | Result |
| --- | ---: |
| Validated openings / seat-paired games | 1,024 / 2,048 |
| Challenger wins / draws / losses | 889 / 195 / 964 |
| Score | 0.481689453125 |
| One-sided 95% lower bound | 0.47021484375 |
| Descriptive two-sided 95% interval | [0.4677734375, 0.4951171875] |
| Advancement rule | all 2,048 games validate and lower bound strictly > 0.50 |
| Decision | **do not advance** |

Update 432 remains the sole candidate. Update 216 was not evaluated. No serving model change, promotion, or deployment is authorized.

## Composition and protocol

- Challenger: update-432 policy and its own trunk from `docs/data/seed542-training-budget-432/D-update-432.npz`, with the frozen initializer value and its own trunk.
- Opponent: current serving P composition, E4 policy and E4 trunk with initializer value and initializer trunk.
- Reused suite: the exact 1,024-opening seed511 v4 suite, SHA-256 `27a003322e5469507a508010a9743b40062928140e6a4b9d5628b0290e32c21d`. No seed511 game outcomes were imported.
- Search: 128 simulations per side, `c_puct=1.25`, one worker, `azlite_eval_seed_v2`, seed `545 + opening_index`, no random opening plies, deterministic root policy, FPU zero, no subtree reuse, no value normalization, no tactical root bias, 200-ply cap, exact-root threshold 16 with `final_score_margin`, highest legal prior then lowest move-index tie break, exact leaf solve disabled.
- Analysis: 20,000 opening-cluster bootstrap resamples, seed 545, replacement with multiplicity, linear quantiles, one-sided lower quantile 0.05. The cluster is the mean of the two seat-swapped game points.

## Reuse and overlap disclosures

This is an exploratory reused-suite screen, not fresh confirmation or a new holdout. The suite was replayed and compared with registered training data: seed504 original training overlap is **0/1,024** roots; expanded seed542 training overlap is **12/1,024** roots, by both canonical state and production float32 identity. The historical seed461 overlap limitation remains unresolved. The seed511 full-publication check previously stopped at its frozen native-source mismatch; this screen reused its frozen suite only, not its games or outcomes.

## Recovery and verification

Two pre-game preparation attempts stopped before freezing because of a missing local tablebase materialization and then an output-path mismatch; neither launched an arena game. The first frozen arena runner wrote complete raw opening-0 evidence, then stopped before ledger commit because it did not create the completion receipt before commit. Continuation v4 and v5 exposed the same receipt-order issue for openings 1 and 2 and also stopped after preserving each complete raw pair. All three pairs were independently replayed, receipt-bound, and committed without rerunning. Append-only continuation freezes v4, v5, and v6 preserve these recovery steps; v6 froze the validated four-game prefix and raw opening-2 pair before proceeding with openings 3–1023. `attempt-disposition.json` summarizes the attempts.

The independent verifier replayed all 2,048 game trajectories, reconciled every pair and ledger entry, recomputed the registered bootstrap, and passed from the original output and a separate physical copy. Evidence bytes and mtimes remained unchanged during verification. The physical copy's game-file inode differs from the original.

Key identities:

- Original screen registration SHA-256: `8541fef485ac915cbf2b32ac7b46f49e43e9fa867fb8411822594eaa824059f6`
- Original execution freeze SHA-256: `65286df075d662232a439e7288285fd31d34a9cfc5f4d5d5aa2b6b4006cd60ab`
- Continuation-v6 registration SHA-256: `62d51231c933241c6fb08083abd10f417fe30af66872c7e69118be07b86d99e9`
- Continuation-v6 freeze SHA-256: `240d92c0ee6db0ae01d292cb310e8db1dbc9223b1e81fdfbe0a8b3ec8f693a5c`
- Complete ledger SHA-256: `8cb6f4523c9832d6c79a15a363548ad989d9da4c145e3729017c9be29cd3ddaf`
- Results SHA-256: `a8c607345275eb602c747d3253cafb26c0a256a65049792082a600cb71bfaec5`
- Per-opening results SHA-256: `afa2a10f021157caf871d5924e76d40a0d9313727e8defc0ae0a61ffb21b04eb`

The training-budget publication is [PR #546](https://github.com/useruby/mancala-ai/pull/546). This strength screen is a separate stacked change and leaves serving artifacts untouched.
