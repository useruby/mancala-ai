# Seed398 paired first-action continuation diagnostic

## Scope and execution correction

This retrospective diagnostic uses the exact 64 states and recorded SS/FF 384-simulation actions in the #405 registration. Each branch forces one recorded root action; all later moves use the frozen seed455 artifact. Budgets are 1,536 (primary) and 384 (secondary), for exactly 256 native-backend continuation games.

The initial ten outcomes were produced with the Python exact-root fallback because `evaluate_artifact_position` skips artifact sidecar loading when an explicit exact-root threshold is provided. Those rows remain intact at `outcomes.jsonl` and are archived byte-for-byte under `fallback-archive/`; they are excluded from all final evidence. `execution-amendment-v1.json` records the runtime correction, original and executed source hashes, reason, and ten pre-existing outcomes. Amendments v2–v4 append later source-formatting hashes without changing experiment logic or results. The corrected run uses `native-outcomes.jsonl` only.

The corrected runner verifies the artifact sidecar, probe and tablebase hashes before execution, keeps one `NativeExactRootTablebase` process/tablebase warm for its lifetime, passes it explicitly to each position evaluation, checks one native solver call at each eligible (≤16-stone) root, and closes the adapter in `finally`. The native exact-root contract optimizes final score margin and resolves equal margins by highest legal network prior, then lowest move index. Exact-leaf solving remains disabled.

The native run completed under the transient user systemd service `seed398-paired-first-action.service` with one ledger writer. Wall time was 6m 9.8s (6m 13.156s CPU); per-game recorded elapsed time ranged from 0.085 to 4.145 seconds. The final ledger contains 256 unique validated cases and 3,324 reported native exact-root solver calls. Resume validates every existing trajectory and terminal score before skipping its case.

## Frozen analysis

- Primary mean paired delta (FF score − SS score), 1,536 budget: **−0.109375**.
- Paired bootstrap over 64 distinct openings, 10,000 resamples, seed 406: **95% interval [−0.2109375, −0.015625]**.
- Secondary mean paired delta, 384 budget: **−0.046875**.
- Registered classification: **SS-action advantage**.

Descriptive #405 groups (not used to choose states or treatments): 32 states where SS/FF selected the same 1,536-simulation action; 17 meeting #405's persistent-disagreement-with-margin description.

These are retrospective, reference-policy-dependent action comparisons. They do not establish minimax quality, overall FF strength, or promotion eligibility.

## Published evidence and verification

- `registration.json`: original immutable registration, containing the exact states, actions, reference bindings, seed protocol, budgets, options and analysis rules.
- `execution-amendment-v1.json` through `execution-amendment-v4.json`: append-only execution corrections and source-hash chain.
- `native-outcomes.jsonl`: all 256 validated terminal outcomes with complete legal trajectories, seeds, relative and absolute actions, root scores, final store margins, and exact-root telemetry.
- `analysis.json`: paired state-by-state score/margin matrix, primary/secondary estimates, bootstrap interval and classification.
- `fallback-archive/outcomes.python-fallback.jsonl`: the ten nonconforming preliminary outcomes, isolated from final evidence.
- `native-run.log`: retained service journal.

From the repository root, run:

```bash
PYTHONPATH=. python -m ml.alphazero_lite.verify_seed398_paired_first_action
```
