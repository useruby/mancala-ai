# AlphaZero-Lite Control EP2 Best-of-N Seed Harvest Results

**Date**: 2026-06-13
**Classification**: `control_ep2_canonical_best_confirmed` + `seed_lottery_exhausted`
**Schema**: `azlite_control_ep2_seed_replication_v1`

## Summary

Ran the 2-epoch LR=3e-5 continuation recipe from `iter0_reference` across seed `42` plus 24 new seeds (`100` through `123`), selected strictly by medium opening-suite seat-aware `384:256` DS, then re-checked the top 5 seeds plus canonical `control_ep2` on the large suite.

**Result**: no seed beat canonical `control_ep2` beyond the 0.12 DS large-suite noise threshold. The best new seed on large `384:256` was `seed_121`, but it still finished **0.0312 DS worse** than canonical `control_ep2` and slightly regressed `1200:1200` (`-0.0312` vs `0.0000`).

The medium-stage ranking was not stable. Canonical `control_ep2` ranked only 8th on medium `384:256`, but rose to **1st** on large `384:256`. The same exact model evaluated twice on medium (`control_ep2` and exact-match `seed_42`) differed by **0.1273 DS** at `384:256`, which is larger than many medium-stage gaps. The seed harvest did not uncover an improved continuation; it mostly demonstrated that medium-only ranking is too noisy to dethrone canonical `control_ep2`.

## Classification

### Overall: `control_ep2_canonical_best_confirmed` + `seed_lottery_exhausted`

| Criterion | Evidence |
|-----------|----------|
| Large-suite `384:256` improvement by at least `+0.12` DS | No seed improved on canonical `control_ep2` at all. Best new seed was `seed_121` at `-0.1562` vs `control_ep2` `-0.1250` (`-0.0312` delta). |
| Preserve or improve `1200:1200` | No new seed improved `1200:1200` on large. Best tie was `seed_103` at `0.0000`, but it badly regressed `384:256` to `-0.3750`. |
| No material `256:768` regression | Some seeds improved `256:768`, but none paired that with a real `384:256` win. |
| Beat canonical on medium and large ranking | Did not happen. `control_ep2` was medium rank 8 but large rank 1. |
| Broad seed search still promising | Not supported. After 24 new seeds, the large-suite winner remained canonical `control_ep2`. |

## Artifact Lineage

| Artifact | Path | SHA256 |
|----------|------|--------|
| current production artifact | `model-artifact/current` | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| iter0 reference checkpoint | `/tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz` | `c2430b56e7a68386343579aeac3610f4c88e1c35904e39179902e437fef36c18` |
| canonical control_ep2 checkpoint | `/tmp/azlite_control_continuation_sweep/lr3e5/checkpoint_epoch2.npz` | `619376dba87b968cc85bfc401cc230aca64e444dc3570216b0630c2c29a388f9` |
| canonical control_ep2 artifact | `/tmp/azlite_control_ep2_seed_harvest/control_ep2_eval_only/artifact_control_ep2` | `34b3697f95c3c63e092838d10668ba1361dab2a68a0bfe12305a294c45f765ad` |

## Dataset And Training Configuration

Only the requested two replay files were used:

| Dataset | Path | Weight |
|---------|------|--------|
| generic bootstrap | `/tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl` | 4 |
| random teacher replay | `/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl` | 1 |

All replicate lanes used: `residual_v3` (`96,3`), `kalah_v3`, `LR=3e-5`, `epochs=2`, `batch_size=512`, `value_loss=huber`, `value_loss_weight=0.3`, `grad_clip=1.0`, `policy_target_mode=sharpened`, `value_target_mode=sharpened`, `lr_scheduler=none`, init from `iter0_reference`.

Stage 1 medium eval used `/tmp/azlite_opening_suite/medium_eval.jsonl` (`suite_size=128`).

Stage 2 large eval used `/tmp/azlite_opening_suite/large_eval.jsonl` (`suite_size=384`).

## Stage 1 Selection

Top 5 seeds by medium-suite `384:256` DS were:

1. `seed_103` at `+0.0909`
2. `seed_112` at `0.0000`
3. `seed_121` at `0.0000`
4. `seed_101` at `-0.0545`
5. `seed_105` at `-0.1818`

Canonical `control_ep2` ranked 8th on medium at `-0.2727`.

## Training And Medium Eval Results

`control_ep2` is eval-only, so training losses are reported through its exact rerun `seed_42`.

| Candidate | Seed | Ckpt SHA | Artifact SHA | Delta vs iter0 | Delta vs EP2 | Policy Loss | Value Loss | Val Loss | Medium 384:256 | Medium 1200:1200 | Medium 256:768 |
|-----------|------|----------|--------------|----------------|--------------|-------------|------------|----------|----------------|------------------|----------------|
| control_ep2 | — | `619376db` | `34b3697f` | 0.090479 | 0.000000 | — | — | — | -0.2727 | -0.3636 | -0.1818 |
| seed_42 | 42 | `619376db` | `34b3697f` | 0.090479 | 0.000000 | 0.998653 | 0.245369 | 1.179735 | -0.4000 | -0.3636 | 0.0000 |
| seed_100 | 100 | `99e00d76` | `92801402` | 0.102686 | 0.065536 | 1.003443 | 0.245833 | 1.124197 | -0.2727 | 0.0909 | 0.0909 |
| seed_101 | 101 | `972a4bdf` | `84701e24` | 0.101663 | 0.065216 | 1.007111 | 0.244177 | 1.076592 | -0.0545 | -0.4545 | 0.0000 |
| seed_102 | 102 | `b4c0ba77` | `2aab5c2d` | 0.101664 | 0.066578 | 1.008425 | 0.245929 | 1.079058 | -0.4545 | -0.3091 | 0.0000 |
| seed_103 | 103 | `60ea71f7` | `55621030` | 0.100022 | 0.065033 | 1.008992 | 0.245251 | 1.075132 | 0.0909 | -0.2727 | -0.1818 |
| seed_104 | 104 | `96778819` | `e638c21a` | 0.100600 | 0.064463 | 1.009691 | 0.246902 | 1.064685 | -0.5455 | -0.4545 | -0.1818 |
| seed_105 | 105 | `8bf63f6e` | `717abf1e` | 0.101896 | 0.067418 | 1.009426 | 0.244512 | 1.075675 | -0.1818 | -0.1636 | -0.1818 |
| seed_106 | 106 | `036cd461` | `4f2a4d92` | 0.102722 | 0.068276 | 1.006778 | 0.246121 | 1.093680 | -0.6364 | -0.5455 | -0.0909 |
| seed_107 | 107 | `24cef3d7` | `8a850cd3` | 0.100449 | 0.066340 | 1.008213 | 0.246032 | 1.081393 | -0.7273 | -0.0909 | 0.0000 |
| seed_108 | 108 | `83e2e984` | `073932a3` | 0.103237 | 0.068585 | 0.995936 | 0.258381 | 1.119358 | -0.2727 | -0.0909 | -0.2000 |
| seed_109 | 109 | `b80b89a3` | `4dda89dd` | 0.100599 | 0.066198 | 1.006754 | 0.246333 | 1.092314 | -0.2727 | -0.5455 | 0.1818 |
| seed_110 | 110 | `afc28318` | `a124d677` | 0.100431 | 0.066021 | 1.000464 | 0.246676 | 1.102912 | -0.4545 | -0.4545 | -0.0909 |
| seed_111 | 111 | `f4deef38` | `c7b33f7c` | 0.101891 | 0.067001 | 1.007130 | 0.246393 | 1.093939 | -1.0000 | -0.1818 | -0.1818 |
| seed_112 | 112 | `55706dac` | `8f486617` | 0.099793 | 0.066836 | 1.005319 | 0.246150 | 1.109462 | 0.0000 | -0.1818 | -0.1818 |
| seed_113 | 113 | `680bcc6f` | `2f709fc2` | 0.101763 | 0.066830 | 1.003258 | 0.245563 | 1.126566 | -0.5455 | -0.2727 | 0.0000 |
| seed_114 | 114 | `a3e5f0ba` | `73b8ae4f` | 0.100124 | 0.065652 | 1.009353 | 0.245408 | 1.073994 | -0.1818 | -0.0909 | 0.0000 |
| seed_115 | 115 | `5fd5f8af` | `5fba241f` | 0.101403 | 0.067479 | 1.001797 | 0.245751 | 1.125282 | -0.1818 | 0.0909 | 0.0000 |
| seed_116 | 116 | `d3e59b9f` | `54a6c367` | 0.100155 | 0.065653 | 1.008555 | 0.245565 | 1.078377 | -0.2727 | 0.1818 | 0.0000 |
| seed_117 | 117 | `e6885c1f` | `e0fdd348` | 0.100833 | 0.066592 | 1.009379 | 0.245547 | 1.071391 | -0.5455 | -0.2727 | 0.0000 |
| seed_118 | 118 | `479d49bf` | `0c9d2b3c` | 0.102989 | 0.065949 | 1.007578 | 0.246253 | 1.084613 | -0.5455 | -0.2727 | -0.1818 |
| seed_119 | 119 | `d8de7e2c` | `2021a6ed` | 0.098548 | 0.064588 | 1.006605 | 0.245435 | 1.099369 | -0.7273 | -0.3636 | 0.0000 |
| seed_120 | 120 | `3363acb9` | `c51dba70` | 0.101200 | 0.064509 | 1.008216 | 0.246566 | 1.078510 | -0.3636 | -0.4545 | -0.1818 |
| seed_121 | 121 | `5582ef1b` | `3cd012f2` | 0.102372 | 0.066689 | 1.004624 | 0.246157 | 1.110858 | 0.0000 | -0.5455 | -0.1818 |
| seed_122 | 122 | `43190f18` | `3fcf149e` | 0.101414 | 0.065802 | 1.008147 | 0.245881 | 1.083494 | -0.3636 | 0.2545 | 0.2727 |
| seed_123 | 123 | `56487e19` | `6c97f3b3` | 0.101147 | 0.066865 | 1.007450 | 0.245479 | 1.088395 | -0.9091 | -0.1818 | -0.1818 |

## Stage 2 Large Eval Results

Large eval covered canonical `control_ep2` plus the top 5 medium seeds.

| Candidate | Seed | Large 384:256 | Large 768:256 | Large 768:768 | Large 1200:1200 | Large 256:768 |
|-----------|------|----------------|---------------|---------------|-----------------|---------------|
| control_ep2 | — | **-0.1250** | **0.0000** | -0.1875 | **0.0000** | -0.1250 |
| seed_103 | 103 | -0.3750 | -0.1875 | -0.1875 | **0.0000** | -0.2500 |
| seed_112 | 112 | -0.6562 | -0.3750 | **0.0000** | -0.3750 | -0.0625 |
| seed_121 | 121 | -0.1562 | -0.2500 | -0.1875 | -0.0312 | **-0.0625** |
| seed_101 | 101 | -0.5625 | -0.1250 | -0.0625 | -0.0312 | -0.1250 |
| seed_105 | 105 | -0.5625 | -0.1250 | -0.1875 | -0.1250 | **-0.0625** |

Large-stage best new seed was `seed_121`, but it did **not** clear the acceptance rule:

| Candidate | Delta vs control 384:256 | Delta vs control 1200:1200 | Delta vs control 256:768 | Meets `improved_seed_candidate`? |
|-----------|---------------------------|----------------------------|--------------------------|-------------------------------|
| seed_103 | -0.2500 | 0.0000 | -0.1250 | No |
| seed_112 | -0.5312 | -0.3750 | +0.0625 | No |
| seed_121 | -0.0312 | -0.0312 | +0.0625 | No |
| seed_101 | -0.4375 | -0.0312 | 0.0000 | No |
| seed_105 | -0.4375 | -0.1250 | +0.0625 | No |

No seed improved large-suite `384:256` by even `+0.01`, much less the required `+0.12` noise margin.

## Ranking Stability

Ranking by medium `384:256` did not survive the large suite.

| Candidate | Medium Rank | Large Rank | Note |
|-----------|-------------|------------|------|
| control_ep2 | 8 | **1** | Canonical recovered to first on large. |
| seed_103 | **1** | 3 | Stage-1 winner lost 0.2500 DS vs canonical at large `384:256`. |
| seed_112 | 2 | 6 | Sharp collapse from medium shortlist to last on large. |
| seed_121 | 3 | 2 | Best new seed on large, still behind canonical. |
| seed_101 | 4 | 4 | Stayed mid-pack, never threatened canonical. |
| seed_105 | 5 | 5 | Stayed mid-pack, never threatened canonical. |

The strongest noise signal in the whole run is the exact-match rerun:

| Same Model Pair | Checkpoint Match | Medium 384:256 | Medium 1200:1200 | Medium 256:768 |
|-----------------|------------------|----------------|------------------|----------------|
| `control_ep2` vs `seed_42` | exact SHA256 match | `-0.2727` vs `-0.4000` | `-0.3636` vs `-0.3636` | `-0.1818` vs `0.0000` |

That is enough to treat medium-only differences as unstable unless they survive the large suite.

## Default Opening Gate

Requested gate results for canonical `control_ep2` and the best new seed (`seed_121`):

| Candidate | Classification | Standard Alt Score | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 256:768 DS |
|-----------|----------------|--------------------|------------|------------|------------|---------------|------------|
| control_ep2 | `high_search_breakthrough` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| seed_121 | `high_search_breakthrough` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |

`seed_121` preserved the default-opening gate signature, but that was not enough to overcome its worse large-suite opening-suite ranking.

## Primary Findings

1. Exact seed-42 reproducibility still holds. `seed_42` reproduced canonical `control_ep2` exactly at both checkpoint and artifact SHA256.
2. The broader seed harvest did not find an improvement. None of the 24 new seeds beat canonical `control_ep2` on large `384:256`, and none satisfied the `+0.12` DS margin requirement.
3. Medium ranking was too noisy to be trusted on its own. Canonical `control_ep2` moved from medium rank 8 to large rank 1, and the exact same model (`seed_42`) produced materially different medium DS.
4. Large-suite behavior confirms canonical control remains the best checkpoint. `seed_121` was the closest challenger, but still lost on `384:256` and slightly regressed `1200:1200`.
5. Validation loss did not identify the best arena seed. The best validation losses in the harvest (`seed_104`, `seed_117`, `seed_114`) were not the best opening-suite seeds, which is exactly why selection stayed arena-first.

## Runner Notes

Two small runner fixes were needed to execute the requested harvest correctly:

| File | Change |
|------|--------|
| `ml/alphazero_lite/run_control_ep2_seed_replication.py` | Added `--large-eval-top-n` so stage 2 can evaluate top 5 seeds plus canonical `control_ep2`. |
| `ml/alphazero_lite/run_control_ep2_seed_replication.py` | Removed the outer benchmark subprocess timeout that was incorrectly killing long medium-suite sweeps before completion. |
| `ml/alphazero_lite/run_control_ep2_seed_replication.py` | Updated gate target selection to prefer the best replicate from the final large-stage ranking for future runs. |

## Verification

| Check | Result |
|-------|--------|
| Harvest driver | PASS: completed and wrote `/tmp/azlite_control_ep2_seed_harvest/control_ep2_seed_replication_report.json` |
| Unit tests | `Ran 2336 tests in 102.055s`; `FAILED (failures=1, errors=8, skipped=3)`; matches pre-existing baseline from prior control docs |
| Ruff | PASS: `ruff check ml/alphazero_lite script/ai` |
| Driver lint after timeout/gate patches | PASS: `ruff check ml/alphazero_lite/run_control_ep2_seed_replication.py` |

## Guardrails

| Guardrail | Status |
|-----------|--------|
| No promotion | PASS |
| No overwrite of `model-artifact/current` | PASS |
| No train beyond 2 epochs | PASS |
| No LR change from `3e-5` | PASS |
| No added data | PASS |
| No architecture change | PASS |
| No validation-loss-based selection | PASS |

## Artifacts

| Artifact | Path |
|----------|------|
| Harvest workdir | `/tmp/azlite_control_ep2_seed_harvest` |
| Training metric capture | `/tmp/azlite_control_ep2_seed_harvest/training_metrics.json` |
| Sweep report | `/tmp/azlite_control_ep2_seed_harvest/control_ep2_seed_replication_report.json` |
| Medium eval report | `/tmp/azlite_control_ep2_seed_harvest/eval_medium/temperature_benchmark_report.json` |
| Large eval report | `/tmp/azlite_control_ep2_seed_harvest/eval_large/temperature_benchmark_report.json` |
| Gate reports | `/tmp/azlite_control_ep2_seed_harvest/eval_gate/*.json` |
