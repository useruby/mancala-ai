# Seed455 fresh-versus-historical replay gradient alignment

This offline, forward/backward-only diagnostic measures source contributions to
the cohort-average mixture objective at frozen seed455. It does not reproduce
the original minibatch sequence or Adam updates, and it is not evidence of
playing-strength improvement.

## Frozen artifacts

- Full source/split/config/code manifest: `seed455-fresh-historical-replay-gradient-alignment-manifest.json`
- Detailed source, cohort, partition, block, and objective results: `seed455-fresh-historical-replay-gradient-alignment.json`
- Portable metric/decision verifier:
  `ml/alphazero_lite/verify_fresh_historical_replay_gradient_alignment.py`

The frozen manifest binds the seed455 checkpoint, original O0 E4 checkpoint,
all five registered replay source hashes and weights, target modes, recovery
receipt and utility, production split seed and fraction, code identities, and
the deterministic partition rule. Four disjoint partitions keep every repeated
copy of a source row together; they measure consistency, not independent games
or statistical confidence.

## Objective and mapping

Compact rows, weighted replay positions, and production split positions are
kept as distinct index spaces. Split positions are mapped through production
`replay_indexes` before row metadata is selected. For multiplicity `m_i`,
policy-loss weight `q_i`, and source `s`, each source contribution is

`sum_s(m_i*q_i*CE_i)/sum_all(m_i*q_i) + 0.3*sum_s(m_i*Huber_i)/sum_all(m_i)`.

Legal masks, policy cross-entropy, and smooth-L1 Huber loss (`delta=1.0`) use
production primitives. No optimizer step, search, arena game, export, or
promotion was run. Focused loader/mapping/decomposition regressions are in
`test_replay_gradient_mapping.py`.

## Primary finding

Seed455's primary `>32` shared-trunk cohort has a **positive** fresh/history
dot product (`+0.1849565`) and retained fresh projection `1.28224`; policy-only
is also positive (`+0.2095468`, projection `1.33762`). None of the four
partitions meets the negative-dot/projection rule. The result therefore does
**not support a later replay-reweighting experiment** under the registered
decision rule. Original O0 E4 is secondary and likewise has positive
fresh/history alignment (dot `+0.2383555`, retained projection `1.44793`).

These are observational gradient diagnostics, not playing-strength results.
A positive diagnostic, if observed in a different audit, would only motivate a
future separately registered strength experiment; it would not establish a
better model.

Verify the published summary and manifest with:

```bash
python -m ml.alphazero_lite.verify_fresh_historical_replay_gradient_alignment \
  docs/data/seed455-fresh-historical-replay-gradient-alignment.json
```
