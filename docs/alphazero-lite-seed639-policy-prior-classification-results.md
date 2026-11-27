# Seed 639 prospective policy-prior classification — blocked attempt

## Outcome

The registered attempt did not reach inference. It exited after 1.616 seconds with status `failed` while authenticating the historical Seed 615 evaluation inputs. The Seed 615 verifier imports `ml.alphazero_lite.train`, which requires PyTorch; the frozen runtime does not contain it (`ModuleNotFoundError: No module named 'torch'`). No model was loaded and no policy forward pass was performed.

This is an infrastructure failure, not a scientific result. The registration forbids retries after an authentication/runtime failure, so attempt `attempt-0001` is preserved as terminal and will not be retried or altered. A future run requires a reviewed registration amendment specifying the missing dependency and a new attempt freeze.

## Authorization and freeze

- Registration PR: [#639](https://github.com/useruby/mancala-ai/pull/639), merged by `useruby` into `main` at `5f4dc090bf57ce1757c4695313cfe28749600771`.
- Attempt: `attempt-0001`; frozen source commit: `5f4dc090bf57ce1757c4695313cfe28749600771`.
- Freeze manifest SHA-256: `acb10005043bd5b180236d6c27fb8fe4b16298ed3223690d6215a0fbbe4801a4`.
- Lifecycle verification: passed for the failed terminal record, frozen source, input identities, and attempt inventory.
- Domain inference/reproduction verifier: not run, because there are no completed state records to verify and the registered run stopped before model loading.
- Registered Seed 636 inventory SHA-256: `6c6b1f680b2177cb67c79a9a642216703f160131034fbf8fe86e8130910ed077`.
- Exposure classification remains `known-exposure-screened; inherited P-E4 exposure unresolved`.

## Observations and requested statistics

No population states were evaluated, so complete-population statistics, per-bin statistics, standardized summaries, and high-TV subset composition are **not available**. In particular, this attempt produced no disagreement rate, total-variation values or quantiles, policy entropy summaries, or selected diagnostic subset. It did not alter or regenerate the primary population.

| Registered measure | Attempt result |
|---|---|
| Population states authenticated for inference | 0 (authentication stopped before completion) |
| Model-state forwards | 0 of 1,024 |
| Overall/per-bin top-action disagreement | Not produced |
| Overall/per-bin TV mean and quantiles | Not produced |
| Fixed one-fifth-bin standardized summaries | Not produced |
| Policy entropy and entropy-difference summaries | Not produced |
| Top-64 high-TV subset and composition | Not produced |

## Acceptance criteria

All scientific acceptance criteria remain unassessed. Infrastructure verification establishes only that the failed attempt and its registered inputs were recorded and archived. No conclusion about B512/S615 policy differences or playing strength is supported.

## Archive and evidence

The complete failed attempt was archived and independently verified. Archive URI/path, byte size, SHA-256, and inventory digest are recorded in [`seed639-attempt-0001-archive-reference.json`](data/seed639-prospective-policy-prior-classification/attempt-0001/seed639-attempt-0001-archive-reference.json). The archive contains the immutable freeze, start and failed terminal receipts, runtime identity, failure receipt, and stdout/stderr evidence. The archive sidecar records all member hashes.

The frozen input/runtime receipt confirms Python 3.14.7 and NumPy 2.5.3. The blocker is the undeclared/unavailable PyTorch dependency imported by the registered Seed 615 historical verifier; no frozen parameter was changed.

## Limitations and next step

This attempt supplies no scientific observations. Before any prospective execution can proceed, the project must register and review the required historical-verifier dependency and runtime provisioning, then freeze a new authorized attempt. This report does not authorize that work.
