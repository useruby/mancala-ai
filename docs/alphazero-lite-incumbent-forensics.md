# AlphaZero-lite Incumbent Forensics

## Question

Where does `current` still outperform the strongest available recent challenger, and is the gap mostly policy choice, value estimation, opening quality, or endgame quality?

## Inputs

- Suite: `ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json` (`224` positions)
- Current artifact: `model-artifact/current`
- Challenger artifact: `storage/ai/alphazero_lite/current_pre_unify`
- Teacher reference: classic MCTS policy `1200`, value `1800`
- Note: paths below are repository-relative examples; replace them if your artifact storage or output directory differs.
- Command:

```bash
.venv/bin/python ml/alphazero_lite/run_forensic_suite.py \
  --suite ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json \
  --current-artifact model-artifact/current \
  --challenger-artifact storage/ai/alphazero_lite/current_pre_unify \
  --mcts-simulations 1200 \
  --teacher-simulations 1800 \
  --out artifacts/incumbent_forensics.json
```

## Findings

- Overall, the challenger slightly leads on teacher top-1 agreement (`0.4643` vs `0.4330`) and value calibration MAE (`0.4381` vs `0.4488`), but it gives back that gain with worse average regret (`0.0968` vs `0.0889`) and weak tactical buckets.
- Opening (`opening_plies_1_8`): challenger is modestly better on top-1 agreement (`0.4167` vs `0.3542`) and regret (`0.0567` vs `0.0630`), so openings are not the main reason `current` survives.
- Tactical captures (`capture_available`): `current` is much better aligned (`0.5417` vs `0.3333` top-1 agreement) and lower regret (`0.0262` vs `0.0633`), making this the clearest tactical incumbent edge.
- Sparse endgame (`sparse_endgame`): challenger wins on both policy and regret (`0.6667` vs `0.5417` top-1 agreement, `0.1298` vs `0.1560` regret), so endgame quality is not where `current` is protecting the gate.
- Proxy disagreement bucket (`incumbent_proxy_disagreement`): this pinned proxy bucket now clearly favors the challenger on policy (`0.6562` vs `0.2812` top-1 agreement) and regret (`0.0144` vs `0.0406`). That means it is useful as a deterministic incumbent-style stress slice, but it should not be treated as direct evidence of real challenger/current promotion-failure states.
- Largest shared weakness is `high_imbalance`: both systems are poor, but `current` is less bad on regret (`0.2021` vs `0.2231`), suggesting the next lane should target skewed-material tactical conversion rather than openings.

## Recommendation

Keep `current` as the incumbent. The next training lane should target `capture_available` and `high_imbalance` move quality first, while preserving the challenger's endgame gains in `sparse_endgame`. Use `incumbent_proxy_disagreement` only as a deterministic stress slice, not as the primary explanation for failed promotion, until we add a real challenger-vs-incumbent disagreement builder.
