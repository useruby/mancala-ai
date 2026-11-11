# Update-512 policy-strength qualification results

The authorized 423-root qualification ran from revision
`3521a6a3f718ad2bc3a2a5f62023210fe094cd1f` with the frozen 128-simulation
search settings, matched per-root seeds, separate shared initializer value model,
and 423 matched pairs (846 search records). The evidence is preserved in
`qualification-evidence.json`.

## Input verification

- Freeze SHA-256: `3631361ba748511d2f4b532df808982da9c0559b0a71bf5b8627d68916f56096`
- Reviewed authorization SHA-256:
  `cbd127dee72d0500ad5010dd8faeaa9f5a5dc7586c706e78a8af3c8e02314559`
- Runtime: CPython 3.12.8, NumPy 2.2.6, PyTorch 2.7.1+cpu
- Read-only freeze check: `freeze_verified`, 423 roots
- Separate-process evidence replay: `verified`, 423 roots

## Registered gate

- Treatment-minus-control mean regret: **−0.0709219858 seeds**
- Paired opening-cluster bootstrap 95% interval: **[−0.1843971631, +0.0283687943]**
- Control WDL downgrade rate: **2.1277%** (9/423)
- Treatment WDL downgrade rate: **2.3641%** (10/423)
- Decision: **`do_not_advance`**

Advancement required mean regret improvement of at least 0.10 seeds, an interval
upper endpoint below zero, and no increase in WDL downgrade rate. None of those
three conditions is satisfied. This qualification does not authorize proceeding
to the separate public exploratory strength screen.

The verifier used 20,000 opening-cluster bootstrap resamples, seed 570, and
linear 0.025/0.975 quantiles. No arena games, retraining, sealed-suite access, or
serving changes were performed. #483 remains closed.
