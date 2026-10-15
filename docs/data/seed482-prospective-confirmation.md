# Seed482 prospective fresh confirmation preparation

**Status: protocol and fixture pipeline only. No real games, searches, candidate
inference, training, runtime export, promotion, or strength claim were performed.**

## Historical source trace and limits

The seed466 replay bundle was independently authenticated by its manifest
(`db80f3bdfcc0a126032c1f5a16e78cda74b160d4fa2d80330cec8bed18085b34`), with
`training.npz` SHA-256
`2e3b503b4e5d64c40d2bca0b44560bfd9f380c2f57606ad8c2db949a1a9c93ae` and
`evaluation.npz` SHA-256
`72950845b76cb0cb842d9ba8065591978469e74c9ac51cd1c0f76c120158b364`. The
seed466 reconstruction recipe describes `kalah_v3` (27 float32 features),
stored sharpened policy targets (without further sharpening), and registered
value-mode semantics. It authenticates data, not the process that generated the
original fresh games.

The historical generation receipt is
`docs/data/alphazero-lite-generations/seed455-nextgen-s461-default-value-root16/generation.json`,
SHA-256 `fbd81375854c43427bbae44a653f253d0238cc54cff12607d3abd3d4f7008e98`.
It records 1,600 games; seed sweep 460,461,462; 1,200 simulations; `kalah_v3`;
temperatures 1.1 / 0.15 after ply 12; Dirichlet alpha/epsilon 0.3/0.3;
sharpened, denoised policy targets; default value targets; six workers; subtree
reuse; linear schedule; exact-root threshold 16 and exact leaf solve disabled.
The receipt binds the historical output bytes at SHA-256
`97bc0b031eb041a4582e67651b0f39d63d04e8a884c6e955e217b7bbb29dc3ab`, but it
does not provide the original generator source snapshot or a complete
hash-bound invocation/runtime closure. In particular, the receipt does not
authenticate all action-selection, target-construction, or worker-order details
as executed. The current checkout's `ml/alphazero_lite/self_play.py` is not
asserted to be the exact historical implementation.

The behavior model and supported search configuration are available in this
checkout and are hash-bound prospectively:

| Artifact | SHA-256 |
| --- | --- |
| `model-artifact/current/weights.json` (seed455 parent behavior weights) | `f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c` |
| `model-artifact/current/metadata.json` | `8b9f4d02395271accd5accb5cde20ec6ddc4ecd63ded192b0b84d9ecf6c649ac` |
| `model-artifact/current/search_policy.json` | `b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d` |
| native probe `model-artifact/runtime/kalah_v1_tablebase` | `d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48` |
| tablebase `model-artifact/runtime/kalah_v1_21.kvtb` | `f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7` |
| current prospective generator `ml/alphazero_lite/self_play.py` | `76718efde7ccfb708f4bb0a4314f812bab746d1364cb75ce5c8b3551b635fc0c` |
| current prospective runner `ml/alphazero_lite/pipeline.py` | `ad1c9919e13418c4af53be9856406272bdb4f8ed9a53e7424cc9364364772dfa` |

The policy is PUCT with exact-root hybrid enabled at 16 stones, final-score
margin objective, highest legal network prior then lowest move index for ties,
and exact leaf solving disabled. This identifies the *available* seed455 parent
behavior model and configuration; it does not cure missing historical generator
provenance.

The frozen runtime is CPython 3.14.7, NumPy 2.5.2, Linux
`7.2.6-arch2-1-x86_64-with-glibc2.44`; inference is float32. The prospective
per-game Python RNG seed is exactly
`game_seed*1_000_003 + global_index + worker_id*9_973`; each single-game process
uses global index 0 and worker 0.

## Distribution decision

Seed482 is explicitly a **new prospective confirmation distribution**, not a
reproduction or relabeling of historical fresh-source sampling. It preserves
recoverable historical settings listed in the receipt above. It prospectively
sets standard 4x6 starts, serial one-game jobs, a fixed consecutive per-game
seed schedule `482000000 + game_index`, linear schedule progress, and deterministic
fixed game-index processing. Every game is complete and strictly replayable.
There is no random opening selection or model/candidate-conditioned position
selection. The first position in ply order with >32 active stones and both
identities absent from exclusions and earlier accepted states is selected; at
most one position is accepted per game. Stop at 512 states or after 2,048
complete games. A shortfall is reported as-is.

Selection reads replayed states and identity exclusions only. It does not read
policy targets, values, game winner, model predictions, losses, or candidate
endpoints. Labels are attached only after the position is determined: the
stored denoised sharpened teacher policy distribution and terminal outcome from
the selected state's player perspective (+1/0/-1). In the prospective denoised
target path, an independent second 1,200-simulation PUCT search uses no
Dirichlet noise and a seed drawn from the gameplay RNG. Its legal visit
distribution is temperature-scaled and squared/renormalized:
`q(a)=z(a)^2 / sum_legal(z^2)`. Gameplay uses Dirichlet alpha/epsilon 0.3/0.3
before ply 12 and disables it from ply 12 onward. Default value mode is the
unshaped terminal outcome, not the intermediate search value. Target values are
validated but cannot influence selection.

**Distribution change:** independently seeded serial games, standard starts,
one accepted position per game, the specified seed schedule, the first-eligible
position rule, and seed455 parent model pinning replace historical multiworker
1,600-game generation whose exact implementation closure is not available.
This measures generalization to this newly defined distribution, not to the
historical fresh-source sampling distribution.

Settings absent from the historical receipt are prospectively fixed in the
protocol: c_puct 1.25; zero FPU; unnormalized values; visit-count root policy;
zero tactical root bias and root temperature; Dirichlet gameplay exploration
with alpha/epsilon 0.3/0.3; one worker; linear schedule with iteration 1 of 1;
standard 4x6 start; fixed consecutive game seeds. These are explicit new
choices, not recovered historical values. Seed482 binds the present checkout's
self-play/rules/encoding/search helper source hashes in `PROTOCOL` and verifies
them with `runtime-preflight`.

## Exclusions and identities

Each excluded source state is represented by both a canonical-state SHA-256 and
the SHA-256 of the exact little-endian float32 `kalah_v3` 27-feature input.
Exclusion manifests bind all reconstructed identities and source byte hashes.
Seed466 training inputs are excluded across the complete source pool (therefore
including both frozen training and validation partitions); the complete seed466
evaluation cohort is excluded, including all 851 fresh-source identities.
Historical probe/evaluation source evidence must also be included as identity
records. The fixture verifier checks both identity forms and rejects either
kind of overlap. The seed466 corpus alone cannot prove non-overlap with the
additional historical probe universe; the manifest builder must bind those
source records before a real cohort is authorized.

## Confirmation analysis freeze

Retained contrasts: F versus initializer and F versus M for fresh policy
cross-entropy; both exposure and equal-input weighting. Retained gates from
seed479/480: fresh CE improvement at least 0.001 for both contrasts under both
weightings; pooled policy CE regression versus initializer at most 0.005 under
both weightings; pooled value-head MSE regression versus initializer at most
0.01 under both weightings; and full fresh-training policy CE must decrease.
All gates must pass to advance; otherwise close the full-population direction
branch. The seed481 shortfall report contains no bootstrap replicate count, RNG
seed, or confidence level. These settings are not recoverable from the inspected
authoritative seed481 artifact. They must be added to the frozen registration
before any generation/evaluation; the bootstrap sampling unit is one independent
game, resampled with replacement, with the single selected position carried
within each sampled game. No bootstrap was performed here.

Initializer identity is seed479/480's frozen
`7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c`. The
authenticated seed479 trajectory file SHA-256 is
`2b78bb032d4bd2c384661bd63391abcd8eb66544a1dee8201aabadaff693aa32`; the
concatenated step-16 post-parameter identities are M
`c283db6c4c33a01de5991d8f2a1f56463a9d047c2b7523f4280512f35d8d5ae2` and F
`29f9a687dcdd8f39ca09bb7cf8fe6ba7ea268f4fdabc6b2af04f4d93a06402f3`. Neither
endpoint is a generation, state-selection, or label source.

The inspected authoritative seed481 shortfall report does not record bootstrap
replicate count, seed, confidence level, or interval construction. These cannot
truthfully be claimed as retained/recovered settings. Seed482 prospectively
freezes 10,000 game-cluster bootstrap replicates, RNG seed 482, and two-sided
95% percentile intervals for reporting uncertainty; these intervals are
descriptive and retained point-estimate gates decide. This prospective choice
is frozen now, before any games or inference.

## Commands

The `runtime-preflight` command checks the behavior-model/runtime and current
generator-source closure identities. The `build-exclusions` command reconstructs
identities from authenticated seed466 arrays and any explicitly supplied
historical JSONL/JSON/NPZ evidence; those extra source files are SHA-256 bound.
The committed `seed482-exclusion-manifest.json.gz` binds 62,330 identity pairs
from the complete seed466 compact training pool and seed427 validation membership
(including seed466's complete 2,607-occurrence evaluation cohort / 1,242
identities). Its uncompressed manifest identity is
`38e97a88041afd3a7f2e580f30155d17fae93f77ddf9267a6e663ca644b758b8`; compressed
file SHA-256 is `129391fc88118a99552029faafb1880f1daed62476b2d1eafd7e931f621dc60d`.
This evidence does not establish complete identity coverage of every separate
seed427–480 historical probe. A consolidated reconstructed ledger covering that
probe universe is not present in the inspected artifacts. The required missing
artifact is `docs/data/seed482-historical-probe-ledger.jsonl`: per-probe identity
records reconstructed from the underlying seed427–480 evidence, with canonical
state and exact float32 input identities plus provenance sufficient to map every
record to the probe that evaluated it. No file or hash for that ledger was found.
Real generation is therefore explicitly blocked until the ledger is reviewed and
its exact path/hash is frozen; fixture
generation, replay, verification, and protocol testing remain available.
The `fixture-run` command is strictly for synthetic/fixture records. It does not
invoke a search backend. The verifier is read-only:

```bash
python -m ml.alphazero_lite.seed482_fresh_confirmation fixture-run \
  --games fixtures.jsonl --exclusions exclusions.json --output fixture-output
python -m ml.alphazero_lite.seed482_fresh_confirmation verify \
  --games fixtures.jsonl --exclusions exclusions.json --output fixture-output
python -m ml.alphazero_lite.seed482_fresh_confirmation runtime-preflight --root .
python -m ml.alphazero_lite.seed482_fresh_confirmation build-exclusions \
  --training docs/data/seed466-replay-input-bundle/training.npz \
  --evaluation docs/data/seed466-replay-input-bundle/evaluation.npz \
  --historical-input path/to/reconstructed-probe-identities.jsonl \
  --output exclusions.json
```

`generate --root . --exclusions exclusions.json.gz --output run-dir` is the
real-search entry point. It verifies frozen behavior/runtime/source hashes and
refuses to start until the exclusion manifest reconstructs the complete
seed427–480 historical probe ledger. That prerequisite is currently absent, so
this command is blocked in the committed preparation state.

The separately gated evaluation command accepts only an externally verified,
complete 512-state cohort receipt whose exact bytes are hash-pinned, and requires
the separately registered initializer/M/F parameter identities. This PR does
not implement or execute candidate inference.

After successful complete generation, `freeze` runs the independent read-only
verifier and writes a distinct `seed482-frozen-cohort-receipt-v1`. Only that
verified receipt can pass `evaluation-gate`.
