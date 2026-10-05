# Seed416 policy-target softening ablation

Inference is conditional on this dataset and training seed. The A and B lanes were trained from the same frozen seed455 initialization; B applies the square-root transform to stored ordinary sharpened policy distributions. This is not recovery of original visit counts.

- Games: 2,048 (1,024 per lane), 512 opening clusters, both seats.
- Paired B-minus-A score: 0.021484 (95% percentile interval -0.001953, 0.043945).
- B score against frozen seed455: 0.524414 (95% percentile interval 0.503906, 0.544434).
- Decision: **retain_baseline**.

The registered thresholds nominate only a separate confirmation; they do not authorize promotion. The complete ledger, per-opening matrix, analysis, registrations, bindings, and portable verifier accompany this report.

## Derivative audit

A preserves all 87,625 source rows and 149,448 weighted replay occurrences unchanged. B changes 64,573 rows; explicitly tagged exact-root targets remain identical. Entropy deltas are mean B-minus-source nats. Phase follows the source `move_index` schedule (early `<10`, mid `<30`, late `>=30`); each row is also available with its source bucket under `registration-v3.json`.

| Source | Phase | Rows | B changed | Mean entropy delta (nats) |
| --- | --- | ---: | ---: | ---: |
| fresh | early | 16,000 | 15,999 | 0.547965 |
| fresh | mid | 31,455 | 27,688 | 0.095282 |
| fresh | late | 19,278 | 2,824 | 0.007886 |
| generic bootstrap | early | 5,430 | 5,430 | 0.473967 |
| generic bootstrap | mid | 7,317 | 6,007 | 0.277540 |
| generic bootstrap | late | 1,194 | 409 | 0.057674 |
| random teacher | early | 239 | 239 | 0.442563 |
| random teacher | mid | 2,006 | 1,730 | 0.240358 |
| random teacher | late | 706 | 305 | 0.031274 |
| opening disagreement | early | 563 | 563 | 0.343426 |
| opening disagreement | mid | 892 | 892 | 0.379244 |
| opening disagreement | late | 545 | 545 | 0.332920 |
| stability | early | 1,039 | 1,039 | 0.440341 |
| stability | mid | 636 | 615 | 0.117191 |
| stability | late | 325 | 288 | 0.044384 |

An initial baseline initialization parity preflight under the superseded registration compared raw network priors with the runtime's legal-masked policy and failed; the exact exported weight hash matched the frozen seed455 runtime. The corrected legal-masked check passed before training. After A's fixed 1,024 games completed, strict replay validation exposed that arena outcome trajectories store absolute pit indices; an accounting-only correction was recorded before B launched. No games were added or rerun, and candidate/search/runtime/seed/analysis settings did not change. Both events are retained in the supersession and accounting-amendment receipts. The reported result remains conditional on this dataset and training seed.

## Execution accounting amendment

The A lane completed under the fixed evaluation contract. Its first post-run validator treated absolute pit indices in the arena trajectory ledger as relative actions and rejected the valid ledger. After observing the A summary (467 wins, 461 losses, 96 draws; score 0.502930), an accounting-only amendment corrected strict trajectory replay before launching B. No A games were extended or rerun; the game-generation command, both already-bound candidates, suite, seeds, native root-16 runtime, simulations, thresholds, and analysis remained fixed. The amendment and the A report/ledger hashes are published in `post-first-lane-accounting-amendment.json`. The analysis is therefore reported conditionally on this dataset and training seed, with this post-A validation correction disclosed.
