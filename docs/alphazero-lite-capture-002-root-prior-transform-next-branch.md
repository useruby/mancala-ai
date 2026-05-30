# AlphaZero-lite Capture 002 Root-Prior Transform Next Branch

## Context

The first generalized transform ablation classified the branch as `extra_turn_tradeoff`: broad extra-turn damping could help `capture_available-002`, but it risked suppressing legitimate extra-turn reference rows.

The immediate next branch was therefore to test more precise move-feature conditioning, still using only legal-move consequences and without row-specific logic.

## Narrow Follow-up

Follow-up artifact:

- `/tmp/azlite_capture_002_root_prior_transform_followup/root_prior_transform_followup_summary.json`

Best narrow transform:

- `seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5`

Definition:

- only activate when the legal move set contains:
  - at least one extra-turn move with `seed_count=4`
  - at least two no-extra-turn capture moves
  - at least one no-extra-turn non-capture move with `seed_count=5`
  - exactly four legal moves
- in that context only, multiply the matching extra-turn move prior by `0.10`, then renormalize

## Local Validation

Guarded `w2` follow-up results:

- `capture_available-002` at `384`: selected reference move `4`, reference visit share `0.8385`
- `capture_available-002` at `1200`: selected reference move `4`, reference visit share `0.9383`
- `capture_available-003` at `384`: selected reference move `1`, reference visit share `0.9401`
- `capture_available-003` at `1200`: selected reference move `1`, reference visit share `0.9433`
- nearby guard rows `006`, `007`, and `008` remained on their reference moves at `384` and `1200`
- row `005` was unchanged and already non-reference under the guarded `w2` baseline

This means the narrow transform clears the intended local `002`/`003` contract and avoids the original broad damping tradeoff on the checked opening-capture family.

## Arena Validation

Command run:

```bash
"/home/alex/Mancala/ai/.venv/bin/python" -m ml.alphazero_lite.arena --challenger "/tmp/azlite_rule_conditioned_opening_full_guarded/rule-conditioned-opening-full-guarded/w2/versions/aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1" --current "storage/ai/alphazero_lite/current" --games 40 --challenger-simulations 640 --current-simulations 256 --workers 4 --min-score 0.55 --out "/tmp/azlite_capture_002_root_prior_transform_followup/arena_seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5.json" --root-prior-transform seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 --fpu-mode parent_q --root-policy-mode deterministic --tactical-root-bias 0.1 --reuse-subtree --normalize-values
```

Arena result:

- score: `1.0`
- record: `40-0-0`
- report: `/tmp/azlite_capture_002_root_prior_transform_followup/arena_seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5.json`

## Tooling Note

The repo `mcts1200_baseline.py` path is a ClassicMCTS comparison harness and explicitly records PUCT search flags only as provenance while ignoring them during execution.

So it is not a valid vehicle for root-prior-transform validation of the challenger artifact, and I did not use it to claim transform efficacy.

## Recommendation

Recommendation: **add `seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5` as an explicit optional MCTS evaluation mode and use arena-style PUCT validation for broader rollout, not the current ClassicMCTS1200 harness**.
