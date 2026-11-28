# Seed 639 historical-context validation

This is an operational validation receipt for the corrective historical-runtime
change. It is separate from the official attempt-0002 preflight receipt and does
not freeze or execute the Seed 639 attempt.

## Result

- Historical authentication: **passed** using the original Seed 615 interpreter
  and source checkout.
- Verified recorded games: **4,096**, with zero capped games.
- Verified model identities: B512
  `bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c`, S615
  `cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780`, and
  P-E4 `2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad`.
- Historical Seed 615 freeze SHA-256:
  `ba30f5e3efadf3358db9f03c7b4eba4f69b23e2f95dcb4089723a1fb9664f32b`.
- Imported historical project modules: **50**; all origins resolve under
  `/home/alex/Mancala/ai-seed615-authorized-eval`. SHA-256 identities for all
  observed imported modules are included in the machine-readable evidence.
- Seed 636 archive reading: passed; `population/population.json` was present in
  the authenticated archive inventory.
- The production `authenticate_seed615_in_subprocess` launcher also passed with
  its sanitized child environment: 4,096 results, 50 historical module origins,
  and 50 source identities. Captured subprocess stdout SHA-256:
  `01b9ed7138b45f408bbfb6f9b365f5f2ee2f59d0b115c5bf4234431cbb4ea1ee`.

## Runtime

- `sys.executable`:
  `/home/alex/tmp/mancala-ai-venv/bin/python` (launched by that absolute path).
- Python `3.14.7`, NumPy `2.5.3`, PyTorch `2.14.0+cu130`.
- CPU-only; CUDA unavailable; PyTorch threads `1`; deterministic algorithms
  enabled.
- Registered thread environment: `PYTHONHASHSEED=615`, `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=""`.
- Installed package inventory: 36 distributions; exact inventory is bound by
  `historical-verifier-runtime-lock.txt` (SHA-256
  `99a376de59cfd861f9be09c34a16526e57f1c0177eb2a5186444e936a09347ec`).
- Original Seed 615 requirements lock SHA-256:
  `5950f891ea746be6058af5fae4c414ac0aca872bec064a7d28c383f795e2fa79`.

## Evidence and disposition

- Complete JSON receipt:
  [`seed639-historical-context-validation.json`](seed639-historical-context-validation.json)
- Receipt SHA-256:
  `01b9ed7138b45f408bbfb6f9b365f5f2ee2f59d0b115c5bf4234431cbb4ea1ee`
- The earlier launch-environment and repository-context failures remain
  preserved outside the official attempt receipt at
  `/home/alex/Mancala/ai/.tmp/seed639-attempt-0002-preflight-failure.json` and
  `/home/alex/Mancala/ai/.tmp/seed639-attempt-0002-preflight-failure-02.json`.
- The later executable/inventory investigation is preserved at
  `/home/alex/Mancala/ai/.tmp/seed639-historical-context-validation-failure.json`.
- No historical checkout source, historical evidence, or frozen artifact was
  modified. No official preflight receipt, attempt freeze, inference, MCTS,
  solving, training, or evaluation game was run.
