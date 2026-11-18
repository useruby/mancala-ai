# Seed 604 replay-coverage optimization preparation

**Final status:** The registered Seed 604 playing-strength evaluation is complete and closed. Both preregistered gates failed; neither checkpoint is promoted. See [frozen evaluation results](evaluation-preparation/evaluation-results.md), the [independent verification record](evaluation-preparation/independent-verification.json), and the preserved game attempts and results in `evaluation-preparation/results/`.

Seed 604 asks whether 2,048 updates on Seed 599 Arm B's exact 2,048-state replay dataset can establish both absolute playing strength and an incremental benefit over the immutable B512 checkpoint. The frozen proposal and artifact identities are in `registration.json`.

The preparation module in `ml/alphazero_lite/seed604_replay_optimization.py` reproduces the registered deterministic schedule. Its first 512 batches must equal Seed 599 B's authenticated schedule. Before updates 513–2048 are allowed, a guarded executor must start from the authenticated P-E4 parent with a fresh Adam optimizer, reproduce B512 at the tensor level, and abort if any tensor differs. Training from the B512 weights with a reset optimizer is prohibited.

At preparation time, the read-only preflight authenticated the P-E4 parent bytes, Seed 599 B512 checkpoint, frozen B configuration, training freeze, 2,048-state membership and target validity, published 512-batch schedule, and Seed 599 evaluation contract/opening bytes. The membership identity is `95ecc180e28945702e08815de7a5210e1bce036afdaf76d194bbd774589a4516`; preflight independently computed it from the canonical membership JSON and required agreement with both `training-B.json` and this registration. On the reviewed worktree at that time, preflight passed with 2,048 states, the B512 schedule prefix reproduced, and no Seed 604 training/evaluation freeze or authorization files present.

## Historical pre-execution requirements (superseded)

The registered evaluation design requires 1,024 new opening groups and 4,096 games, but the cohort is not yet frozen. Before evaluation authorization, prepare and authenticate the opening manifest against Seed 599 openings and all applicable excluded training/evaluation state identities; freeze its new seed, identities, execution sources, runtime and PUCT settings. Seed 599 outcomes must not enter confirmatory inference or checkpoint selection. Both candidates use the unchanged P-E4 value evaluator.

The primary gate is B2048's one-sided 95% paired-opening-bootstrap lower bound versus P-E4 strictly above 0.50. The secondary gate is the paired two-sided 95% interval for B2048 minus B512 with lower endpoint strictly above zero. The four interpretations are registered in `registration.json`.

At the time this preparation document was written, training and evaluation required separate owner-authorization files bound to separate execution-freeze byte identities. Those gates were subsequently satisfied for the recorded B2048 training and the evaluation freeze cited in the final results. The final evaluation did not modify Seed 599 evidence and did not promote either model.

Run the synthetic contract tests with:

```bash
python -m unittest ml.alphazero_lite.test_seed604_replay_optimization
```

The current synthetic suite passes five tests. Ruff lint and format checks pass for both Seed 604 Python modules, and `compileall` passes. GitHub CI is run on the PR update.
