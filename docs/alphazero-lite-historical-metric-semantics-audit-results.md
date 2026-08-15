# Historical Evaluation-Metric Blast-Radius Audit

**Classification:** `historical_metric_blast_radius_closed`

`seat_asymmetry_ds = P0 challenger score - P1 challenger score`. It is a
seat/budget asymmetry diagnostic, not candidate strength. Candidate strength is
the paired, per-opening mean of candidate score minus matched current/current
control score, bootstrapped over unique openings (10,000 samples).

## Result

| Item | Count |
|---|---:|
| Decision-bearing historical experiments inspected | 8 |
| Already using a canonical paired metric | 3 |
| Affected by historical seat DS | 4 |
| Recomputed from retained records without games | 2 |
| Game reruns required | 0 |
| Decisions changed | 0 |
| Candidate artifacts reopened | 0 |

## Recomputed Decisions

The retained D384/D1200 candidate records and matched current controls reproduce
the PR #185 path-equivalence results exactly at the primary `384:256` budget.

| Experiment | Historical metric | Historical value | Paired effect | 95% opening CI | Original -> corrected | Impact |
|---|---|---:|---:|---|---|---|
| PR #182 D384 | seat asymmetry DS | -0.316406 | +0.025391 | [-0.009766, +0.062500] | reject -> reject | decision unchanged |
| PR #182 D1200 | seat asymmetry DS | -0.246094 | +0.000000 | [-0.033203, +0.031250] | reject -> reject | decision unchanged |

Both point estimates fail the original robustness requirement of a positive
lower bound. Neither is reopened merely because its point estimate is
non-negative.

## Historical Inventory

`docs/data/historical_metric_inventory.json` records every decision-bearing path,
runner, result document, retained provenance, control availability, and recovery
status. The two old DS-based promotions have only aggregate historical reports
remaining. Their individual games are unrecoverable, but their effects are
superseded by subsequent current-lineage and `azlite_eval_seed_v2` canonical
evidence; they are not retroactively invalidated. Tactical/c_puct decisions were
independently revalidated under the canonical contract and require no rerun.

No model training, replay generation, promotion, runtime tuning, or arena games
were run for this audit.

## Lineage Action

No historical candidate was reopened. Policy-head-only teacher-budget tuning is
permanently closed. The next ML branch should target value/representation work,
not another policy-target variation.
