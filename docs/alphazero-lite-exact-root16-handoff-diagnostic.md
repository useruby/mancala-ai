# Exact Root16 Handoff Diagnostic

Classification: `exact_root16_repairs_seed455_without_search_regression`.

This diagnostic uses `EndgameTablebase.MAX_SOLVED_SEEDS == 16` as the fixed
`exact_root_solve_threshold`. The decisive `sparse_endgame-023` root has 16
active pit stones. No alternate threshold was tested.

The treatment is a direct root handoff, not exact leaf-value injection:

```text
PUCT while active pit stones > 16
exact minimax final-score-margin root decision at <= 16
```

At an eligible root, every legal action is applied to a clone and evaluated
with `EndgameTablebase.final_margin` from the root player's perspective. The
maximum margin wins; equal-margin actions use highest legal network prior and
then the lowest move index. Ineligible roots do not invoke the helper or the
network, so normal PUCT remains unchanged.

## Solver Check

Artifact bytes were verified before evaluation:

| Model | SHA-256 |
| --- | --- |
| seed48 incumbent | `935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c` |
| seed455 candidate | `f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c` |

`sparse_endgame-023` was solved as required:

| Move | Exact final margin | WDL |
| --- | ---: | --- |
| 1 | -12 | loss |
| 4 | 0 | draw |
| 5 | -6 | loss |

The exact-optimal set is `[4]`. Seed455's network priors were approximately
`[move 1: 0.3341, move 4: 0.0805, move 5: 0.5853]`; direct root handoff still
selected move 4 and executed zero PUCT simulations.

The complete 24-row sparse-endgame suite was evaluated through the direct
root solver with one worker-local tablebase cache. Every selected action was
in its exact-optimal set: exact-optimal rate `1.0`, WDL-optimal rate `1.0`,
mean and maximum final-margin regret `0`.

## Runtime

The existing documented operational limits are 200 ms arena mean and 350 ms
arena p95 (`docs/alphazero-lite-ml-handoff.md`). The direct Python tablebase
does not meet them:

| Measure | Sparse 24-root worker-local run |
| --- | ---: |
| Median root solve | about 3.9 s |
| p90 root solve | about 40.0 s |
| p95 root solve | about 46.2 s |
| Maximum root solve | 46.4 s |
| Cache entries after 24 roots | 41,460,858 margins |
| Seed455 decisive root cold latency | 28.5 s |
| Seed455 decisive root warm latency after related cache population | 381 ms |

The Python backend is not operationally viable. The native follow-up below
replaces it for the subsequent PR340, forensic, and diagnostic-arena phases.

## Native Follow-Up

The worker-local `NativeExactRootTablebase` adapter now uses the validated
KVTB probe artifact directly. Its single root-action query reproduces
`sparse_endgame-023` exactly: `{1: -12, 4: 0, 5: -6}`, selects move 4, and
has one solver call rather than one Python recursive solve per action.

The native process takes about 3.7 s to memory-load the local tier-21 artifact
on worker startup. The arena prewarms that process before games begin; the
subsequent decisive root took 0.49 ms. Across the 24 sparse roots after the
initial cold query, median/p90/p95 probe latency was 0.018/0.103/0.161 ms.
This removes the per-move runtime blocker while keeping identical exact
final-margin semantics.

## Native A/B Results

The reconstructed 128-opening medium suite exactly matched the frozen SHA256
`57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04` using
the documented deterministic opening-suite command.

The frozen PR340 corpus had 110 eligible roots and 90 normal-PUCT roots for
each model. Both seed48 and seed455 had zero mean and maximum exact regret on
eligible roots. All 90 noneligible roots matched normal PUCT on raw priors,
selected move, visits, child statistics, and root telemetry.

The forensic suite had 24 eligible and 200 noneligible roots for each model.
Both models had exact-optimal rate 1.0 and zero regret on eligible roots; all
200 noneligible rows were byte-identical to baseline.

The equal-policy non-promotion arena used seed455 versus seed48 with both
sides at 384 simulations outside the handoff, deterministic roots, 128
openings, 256 seat-balanced games, and one worker. Seed455 scored `0.6484`
(166 wins, 90 losses, 0 draws; Wilson 95% CI `[0.5881, 0.7043]`). It recorded
209 games with at least one handoff, 3,204 handoffs total, median first
handoff ply 26, and 12.52 handoffs per game. Native root solves used 325 ms
total versus 215.4 seconds for normal PUCT. Arena mean/p95 move time was
23.22/40.03 ms, within the documented 200/350 ms limits.
