# Phase-Specific Self-Play Search-Budget Ablation Results

Classification: `uniform1200_dominates_phase_specific_budget`.

## Scope

- This is a regenerated successor baseline, not the lost 38-row historical replay lineage.
- Every lane initialized from incumbent weights SHA-256 `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`.
- Each lane used 1,600 games, seed 42, seed sweep `41,42,43`, denoised sharpened targets, tree reuse, and the same fixed replays at weights 1 and 2.
- The pipeline normalized the requested 6 workers to 24 in every actual self-play command. This preserves between-lane matching but means the result is not six-worker provenance.
- No artifact was promoted.

## Fixed Replay Sources

| source | rows | SHA-256 | weight |
| --- | ---: | --- | ---: |
| selected regenerated artifact | 20 | `787cca93c7db454c2daebe1fdab001e93df0b9f80a1e7005598a0324774c427b` | 1 |
| corrected guard controls | 5 | `ffac24dc8fb773fddd4dc14f397317d7c5e0a973631ca71adb216b5e68b444a5` | 2 |

## Completed Pipelines

Effective simulation counts describe the persisted self-play rows. Opening rows are exactly 1,600 games times the eight-ply phase horizon.

| lane | effective normal / opening simulations | replay rows | effective opening rows | effective normal rows | replay SHA-256 | checkpoint SHA-256 | exported weights SHA-256 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `control_384_192` | 192 / 384 | 63,959 | 12,800 | 51,159 | `a8b9e6cf5b453db49b695e474c64a8432e5fc59f688cbdbcde3559e342ea73a8` | `cee3a843334f716b7dd1d1661c0018675127e5bea5d941cff6c676103307ea4e` | `afb961d72aa2548afae006b924344462f10c1a4d49f5e9ed505b85e57ca23ac1` |
| `opening1200_rest192` | 192 / 1200 | 64,928 | 12,800 | 52,128 | `10dcfc501821a16e22010ffb5955005199180cb4d78c7f0997567858f8bc6829` | `51b7256011548842fa353f796b318b336caa2dd960ce38349e80f5addfdf4d60` | `660c37407859f1296b4a3bb3c78380533a9382911f4052c24f5b8a2e6fe8675d` |
| `uniform1200` | 1200 / 1200 | 71,106 | 71,106 | 0 | `d197bd7171838935001ad1a6900dfa0436ffbcd10063d653cf4b9e11e83c803c` | `263418dfe9fbd48b8decc384f975289366d10cdddd67d854776c30eaa59fe86a` | `8671edd37a5b7d3b7a708c0f1ddf11a35e9560bf9debfa1b9ff2dce6c3a10615` |

## Arena

The seat-aware benchmark used the frozen 128-opening medium suite (SHA-256 `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`), deterministic roots, seed 42, two games per opening and both forced challenger seats. Effects are paired against an equal current-control run; intervals are opening-bootstrap 95% CIs.

| lane | `384:256` paired effect [95% CI] | `1200:1200` paired effect [95% CI] |
| --- | --- | --- |
| `control_384_192` | -0.254 [-0.307, -0.201] | -0.057 [-0.119, +0.006] |
| `opening1200_rest192` | -0.006 [-0.066, +0.053] | +0.148 [+0.086, +0.209] |
| `uniform1200` | +0.053 [-0.012, +0.117] | +0.406 [+0.363, +0.445] |

Benchmark report SHA-256: `1450be3e98822efc1dc15d4ff257fcddcbff367577d4af61fbfcee1e1b8a9c7d`.

## Interpretation

Higher opening search is helpful: the phase-specific lane is positive under equal 1200 evaluation. Uniform 1200 is substantially stronger, however, and its CI excludes both no effect and the phase-specific effect's range. The evidence therefore supports raising the self-play budget throughout the game rather than only in the first eight plies. This single-suite, one-seed ablation is not a promotion result; no artifact was promoted.
