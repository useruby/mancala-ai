# Seed 414 FF384 vs FF1536 first-action confirmation

Fresh, preregistered 128-state diagnostic of first-action quality under two frozen continuation references. It is not an overall-strength or promotion result. No historical outcomes were reused.

- Decision: **confirmation_criteria_not_met**.
- Mean paired gain: **0.023438**.
- Paired bootstrap 95% interval (10,000 state-level resamples, seed 414): **[0.001953, 0.048828]**.
- Reference gains: seed455 **0.027344**; original O0 E4 **0.019531**.
- First-action change rate: **12.500%** (16/128).
- Mean root-search time: **0.140s**; physical continuation count: **288**.
- Mean store-margin changes: {"original_O0_E4": 0.140625, "seed455": 0.421875}.

Evidence files include the registration, source snapshots, extended exclusion proof, 128 E4 root probes, complete physical trajectories, 512 logical aliases/cases, and paired matrix.

Verify without loading model artifacts or invoking searches:

```sh
PYTHONPATH=. python -m ml.alphazero_lite.verify_seed414_root_budget_confirmation
```
