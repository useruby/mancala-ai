# Seed557 retrospective policy/search diagnostic

This read-only retrospective evaluates T512 update 108 (#555), C128 update 108 (#555), and frozen serving P from #556 on fixed roots replayed from the authenticated #556 game archive. It ran no games or training. The 256-root cohort was frozen before any policy evaluation.

## Cohort and execution

- Source: merge commit `f38838e6ffe20a44c1e8c161a9daed0f1243f0ae`.
- All `batches/games-0000.jsonl` through `games-1023.jsonl` match the per-file SHA-256 values in #556's `verification-receipt.json`. The indexed filename supplies the global opening index; `game_row.opening_index` remains pair-local.
- Each registered opening's absolute prefix was replayed from the standard initial position and checked against its registered state/hash. Absolute trajectory moves were then replayed from that position.
- Eligibility is the first nonterminal position in each game with 17–21 active pit stones and at least two legal moves. Candidate roots were deduplicated by canonical state hash, ranked by `SHA256("seed557:" + state_hash)`, capped at one per opening, and alternated by source side (T512/P) where possible.
- There were 1,903 eligible game positions in 996 openings, 1,663 distinct states, and no cohort shortfall. The frozen 256 roots comprise 128 positions with T512 to move and 128 with P to move.
- Each policy uses its own trunk/policy with the same frozen initializer value network. Every root/policy search used a fresh tree, 128 ordinary PUCT simulations, `c_puct=1.25`, #556's deterministic root settings, one worker, and a matched per-root seed. Exact values were queried separately and never supplied to search.
- The authenticated native tablebase returned margins for every legal action at all 256 roots. Any coverage gap is a verifier failure.

`cohort.json` records opening, source side/seat, ply, absolute prefix, game-local index, source archive path/line, source record hash, and batch hash. `search-records.jsonl` contains priors and selected PUCT actions; `oracle-records.jsonl` contains complete per-action exact margins; `root-metrics.jsonl` contains the reconstructed regret outcomes. `execution-freeze.json` binds the cohort, archive hashes, assets, and execution contract.

## Results

Primary estimand is paired T512 minus P mean final-margin regret after PUCT; positive favors P (T512 left more seeds on the table). Secondary is T512 minus C128.

| Policy | Prior exact-optimal | PUCT exact-optimal | Prior WDL downgrade | PUCT WDL downgrade | PUCT improved prior | PUCT worsened prior | Mean post-PUCT regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| T512 | 62.50% | 76.95% | 12.11% | 4.30% | 17.58% | 4.30% | 1.1406 |
| C128 | 62.89% | 75.78% | 12.11% | 4.69% | 16.80% | 4.69% | 1.2031 |
| P | 65.63% | 78.13% | 10.94% | 4.30% | 15.23% | 2.73% | 1.1484 |

- **Primary, T512 − P after PUCT:** −0.0078 seeds; paired opening-bootstrap 95% interval [−0.1719, 0.1406] (256 openings).
- T512 − P raw-prior mean regret: +0.1563 seeds; paired 95% interval [−0.1250, 0.4453]. The raw exact-optimal point rates also favor P (65.63% vs 62.50%).
- **Secondary, T512 − C128 after PUCT:** −0.0625 seeds; paired opening-bootstrap 95% interval [−0.1797, 0.0313]. Raw-prior regret difference is +0.0781 seeds; interval [−0.0859, 0.2969].

The point estimates show a small disadvantage for T512 in raw priors, while the post-search contrast is near zero and its paired uncertainty includes both directions. **The measured gap is unresolved after search; this cohort suggests a raw-prior tendency that is itself uncertain.** These results describe positions reached in #556 only; they cannot establish why the arena result occurred or how policies compare across full games. #556's `do_not_advance_T512` decision is unchanged.

## Verification

Run `python -m ml.alphazero_lite.verify_seed557_retrospective`. The verifier uses the committed #556 game archive and published diagnostic files to reconstruct the opening prefixes, eligible states, frozen cohort, legal-action oracle coverage, per-root metrics, and aggregate metrics. It does not open the original execution or tablebase paths. The integration tests also exercise both player perspectives and tied optimal actions.
