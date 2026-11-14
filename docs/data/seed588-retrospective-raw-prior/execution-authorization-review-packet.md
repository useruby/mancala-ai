# Seed 588 execution-authorization review packet

**Status: authorization requested; inference has not been authorized or run.**

This packet requests an independent, explicit authorization for the already
registered retrospective diagnostic. Approval of merged PR #589 is not the
execution authorization. No authorization JSON or trusted authorization digest
is included or asserted here.

## Reviewed source and executable freeze

- Repository: `useruby/mancala-ai`
- Merged readiness change: PR [#589](https://github.com/useruby/mancala-ai/pull/589)
- Merged commit: `4591bf4ce0429fafce467bfbace6182e359f1bc9`
- Executable freeze: `docs/data/seed588-retrospective-raw-prior/execution-freeze-v2.json`
- Executable freeze SHA-256: `fea0ba1e910184db2717228e238d26bd55c7b1f95427b1b24026a9bf1714001c`
- Freeze status: `frozen-pending-review-no-inference`
- Protocol SHA-256: `d86a69bd62eb00e4583808bb54588c0783754817b5ea0167aa09e4098e954b2a`
- Preparation-freeze SHA-256: `a45e432ce17a46ca7f30980ed3031a0010e3a3049e6fb60ed3e8e5a938aed8fd`

The read-only source verifier and executable-freeze verifier passed in the
frozen runtime on 2026-11-14. Both report `inference_performed: false`. The
source verifier authenticated 512 unique roots/openings, complete oracle and
archived-P coverage for all 512, and active-pit counts from 17 through 21.

## Frozen implementation/dependency SHA-256 map

| Path | SHA-256 |
|---|---|
| `ml/alphazero_lite/run_seed588_raw_prior.py` | `c04c9823c70f36688030716f47cbbae08578fff88e1a497769e5261431ee6361` |
| `ml/alphazero_lite/seed588_raw_prior_diagnostic.py` | `4fb060a83e185008a479696c90535d8cef87e07dae5afe80f88a908021caf695` |
| `ml/alphazero_lite/seed588_raw_prior_analysis.py` | `69aabdaa87c71786abe9248344009a05d36bb1e7b90257b4798c1919d5103237` |
| `ml/alphazero_lite/verify_seed588_raw_prior_result.py` | `fd37d585fcace79b09d3d64a02f5cff660418c9446d25c0c7cf8226dfdc09fef` |
| `ml/alphazero_lite/test_seed588_raw_prior_execution.py` | `04287e7fdc47eb8f22ec5450ace553291a7f1164f557fc6e7db16a914c41ddeb` |
| `ml/alphazero_lite/kalah_rules.py` | `6e5338ef9a85b690f51b6d0d184c8bc0f54c2b28ac084a4a17a88f9f552ad036` |
| `ml/alphazero_lite/arena.py` | `c16a668e8f90bd87b59dbc08f6a1ddb3b5ffc912e21cc0564efaceba37e03fe2` |
| `ml/alphazero_lite/self_play.py` | `76718efde7ccfb708f4bb0a4314f812bab746d1364cb75ce5c8b3551b635fc0c` |
| `ml/alphazero_lite/input_encodings.py` | `0a4f56ed60f0c2ed6b93a673dbc0850a966a907912dc26edd092bfd1ac209dbe` |
| `ml/alphazero_lite/seed557_retrospective.py` | `78dbc90f3339a41f932fd07594237fd7a90606251db5e0c49b0070b0586cc3f2` |
| `docs/data/seed588-retrospective-raw-prior/README.md` | `d86a69bd62eb00e4583808bb54588c0783754817b5ea0167aa09e4098e954b2a` |
| `docs/data/seed588-retrospective-raw-prior/execution-freeze.json` | `a45e432ce17a46ca7f30980ed3031a0010e3a3049e6fb60ed3e8e5a938aed8fd` |

## Pinned model artifact identities

Paths are under
`docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts/`.

| Model | File | SHA-256 |
|---|---|---|
| T512 | `T512/metadata.json` | `b37f3aa75d828b9b9a0743930c73c8e168b159f6b37434eba1d9871e769af1b8` |
| T512 | `T512/model.npz` | `8940cbff36976801e74542f5ee2d6c5caffe0cafd57c7e787cdb375f3bbe485e` |
| T512 | `T512/search_policy.json` | `280c33d88266969da72a7984dabdcfea180bacb5cb690cb759af1ebdbaac65bc` |
| T512 | `T512/weights.json` | `70af73e6f8ee3f4fb76981584c1f2068a4b669300be90bd71cfdd8f71b369997` |
| P | `P-E4/metadata.json` | `a9d133043c64ca41e3d77354f4ccf3403ab5f94286608f01589387159ac0258c` |
| P | `P-E4/model.npz` | `2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad` |
| P | `P-E4/search_policy.json` | `280c33d88266969da72a7984dabdcfea180bacb5cb690cb759af1ebdbaac65bc` |
| P | `P-E4/weights.json` | `89dead525a255cff4760867645fd9ce05cb2bbadcf1d72c5ac9089868c4a8136` |

## Frozen source evidence SHA-256

The evidence is `docs/data/seed559-fixed-position-feasibility/run-0003/`.

| File | SHA-256 |
|---|---|
| `cohort-manifest.json` | `f2e03fb9491474056e7a7cb3ace225cd2310f9069c9e48600d45039ed63158fb` |
| `execution-freeze.json` | `3b1ab386c33534130c7e15df024f21ef75359ccd00d453e4c6aea104c8181927` |
| `oracle-records.jsonl` | `83bec1bb348cbc6d1183112363f27d0612ee204a48d14ee5dbb7530e9b5fb8c2` |
| `search-records.jsonl` | `b2653c41fe34328386ae432a3e019568eba9df7c5dc82a596a18917e8c8fb12e` |

The cohort has 512 unique roots/openings; oracle coverage and archived P priors
are complete for every root. Historical overlap is verified as excluding #557
and #418 canonical-state matches; Seed 461 continuations and other uncovered
historical populations remain unresolved. This is not an independent
generalization sample and cannot support a full-game-strength claim.

## Runtime and execution contract

Frozen runtime, observed successfully in the isolated environment:

```text
Python 3.12.8
NumPy 2.2.6
CPU
PYTHONHASHSEED=588
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
execution_threads=1
```

Inference uses the NumPy `arena.ArtifactEvaluator` and authenticated `weights.json`;
PyTorch is not required by the executable v2 path. Upon a valid independent
authorization, the exact permitted sequence is:

1. Run exactly 512 P raw-prior forwards; require exact legal-action masks and
   maximum absolute prior difference at most `1e-6` on every root.
2. Only after all P parity checks pass, run exactly 512 T512 raw-prior forwards.
3. Compute the preregistered paired metrics and 20,000 opening-cluster bootstrap
   resamples (seed `58820261114`, 2.5%/97.5% linear quantiles).
4. Exclusively create
   `docs/data/seed588-retrospective-raw-prior/run-0001/` with the five frozen
   evidence files and verify from a separate physical copy against independently
   recorded run-manifest and authorization SHA-256 digests.

No MCTS/PUCT, search, arena games, training, sealed-suite access, or model
promotion is permitted. This remains an exploratory retrospective evaluation;
it does not reopen #483.

## Authorization requested

An independent reviewer must decide whether to authorize this exact frozen
scope, then supply a separate JSON record conforming to
`seed588-reviewed-execution-authorization-v1` with status
`reviewed_and_authorized`, reviewer identity, review URL, exact executable
freeze SHA-256, complete implementation/dependency map, source and artifact
bindings, and exact runtime contract. The reviewer must provide the record's
SHA-256 independently to the executor. The executor will validate the record
and digest against all frozen bindings before any model forward pass. A GitHub
PR review or approval alone is not this authorization.

Until the valid authorization JSON and independently trusted digest are
received and pass executor validation, **no inference is authorized**.
