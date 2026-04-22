# AlphaZero-lite Hard-State Mining

## Command

Use `ml/alphazero_lite/mine_hard_states.py` to mine hard states from completed JSON reports.

## Inputs

- forensic-style disagreement and value-error reports
- challenger loss reports vs `current`
- challenger loss reports vs `MCTS1200`

## Output Row Schema

- `canonical_state`
- `state`
- `side_to_move`
- `legal_moves`
- `selection_reasons`
- `source_artifacts`
- `source_runs`
- `priority_score`
- `priority_breakdown`
- `consequence`
- `metadata`
