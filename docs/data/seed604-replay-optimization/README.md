# Seed 604 replay-coverage optimization preparation

Seed 604 asks whether 2,048 updates on Seed 599 Arm B's exact 2,048-state replay dataset can establish both absolute playing strength and an incremental benefit over the immutable B512 checkpoint. The frozen proposal and artifact identities are in `registration.json`.

The preparation module in `ml/alphazero_lite/seed604_replay_optimization.py` reproduces the registered deterministic schedule. Its first 512 batches must equal Seed 599 B's authenticated schedule. Before updates 513–2048 are allowed, a guarded executor must start from the authenticated P-E4 parent with a fresh Adam optimizer, reproduce B512 at the tensor level, and abort if any tensor differs. Training from the B512 weights with a reset optimizer is prohibited.

## Evaluation registration remains a prerequisite

The registered evaluation design requires 1,024 new opening groups and 4,096 games, but the cohort is not yet frozen. Before evaluation authorization, prepare and authenticate the opening manifest against Seed 599 openings and all applicable excluded training/evaluation state identities; freeze its new seed, identities, execution sources, runtime and PUCT settings. Seed 599 outcomes must not enter confirmatory inference or checkpoint selection. Both candidates use the unchanged P-E4 value evaluator.

The primary gate is B2048's one-sided 95% paired-opening-bootstrap lower bound versus P-E4 strictly above 0.50. The secondary gate is the paired two-sided 95% interval for B2048 minus B512 with lower endpoint strictly above zero. The four interpretations are registered in `registration.json`.

Training and evaluation have separate owner-authorization files bound to separate execution-freeze byte identities. Neither authorization is supplied. This PR performs preparation and synthetic integrity checks only; it does not train, run games, modify Seed 599 evidence, or promote a model.

Run the synthetic contract tests with:

```bash
python -m unittest ml.alphazero_lite.test_seed604_replay_optimization
```
