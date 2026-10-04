# Seed398 first-action diagnostic under original O0 E4

Retrospective reference-dependence diagnostic. It uses the exact 64 registered #405 states and #406's recorded SS/FF 384-search first actions. After the forced move, both players use the original O0 E4 runtime artifact. There are 256 games: 64 cases × two first actions × two continuation budgets.

## Frozen configuration

- E4 checkpoint SHA256: `4bc05f8284d5adb5beac5090dbf2c09686c84831131cf3ceede606587ec127f4`.
- Continuation budgets: 1,536 primary and 384 secondary; `c_puct=1.25`; #406 search options and seed contexts; base seed 406.
- Root-16 native solver; final-score-margin objective; prior-based tie rule; exact-leaf solving disabled.
- Complete trajectories and the state-wise paired matrix are in `outcomes.jsonl` and `analysis.json`.

## Results

- Primary 1,536 mean FF-minus-SS score: **-0.10156250**; paired bootstrap 95% interval **[-0.22656250, 0.02343750]**.
- Secondary 384 mean: **-0.10156250**.
- Descriptive E4 delta minus #406 seed455 delta: **0.00781250**, paired bootstrap interval **[-0.13281250, 0.14062500]**.
- Preregistered conclusion: **Reference robustness unresolved**.

The interaction is descriptive. This retrospective diagnostic supports no training, state reselection, promotion, minimax-quality, or overall-strength claim.

Verify the public ledger without model artifacts or native searches:

```sh
PYTHONPATH=. python -m ml.alphazero_lite.verify_seed398_e4_reference_diagnostic
```
