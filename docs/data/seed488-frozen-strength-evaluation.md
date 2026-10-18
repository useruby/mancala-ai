# Seed488 frozen direct-strength evaluation

**Preparation only. No arena games/search or training has been run. Synthetic
export and policy/value inference fixtures were tested against the frozen
parameter states; no runtime arm artifacts or game outputs were produced. The
seed483 fresh-confirmation gate remains closed.** Seed479 F and the
seed480 correction remain promising exploratory evidence; this prospective arena
is an exploratory direct strength evaluation, not fresh confirmation and not an
independently untouched holdout.

## Frozen comparison

Compare the exact step-16 seed479 F endpoint against the seed479 initializer.
Both are exported as the same `residual_v3 (96,3)` / `kalah_v3` model artifacts,
with identical artifact-export semantics and a shared runtime search-policy
sidecar. Before runtime export, synthetic fixtures round-trip both frozen
parameter states through the checkpoint exporter and require exact equality of
masked policy outputs and value outputs. The focused synthetic test is
`ml/alphazero_lite/test_seed488_strength_evaluation.py`.

The machine-readable registration binds initializer SHA-256
`7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c`, seed479
trajectory SHA-256
`2b78bb032d4bd2c384661bd63391abcd8eb66544a1dee8201aabadaff693aa32`, and F
endpoint float32 parameter-state SHA-256
`29f9a687dcdd8f39ca09bb7cf8fe6ba7ea268f4fdabc6b2af04f4d93a06402f3`. It also
binds arena/rules/search/export source identities, the native exact-root probe,
tablebase, and runtime policy sidecar.

## Suite, exclusions, and limitation

`seed488-frozen-strength-evaluation/openings.jsonl` contains 512 deterministic
legal opening roots generated with `random.Random(488)` from 4–12-ply prefixes.
The generator rejects root identities colliding in either canonical or float32
identity with the reconstructable seed482 base exclusion set and seed483 union,
and rejects duplicate roots. Each opening is played twice, once with F seated as
player 0 and once as player 1: 1,024 total games. Arena base seed is
`488 + opening_index`, under `azlite_eval_seed_v2`.

The suite is checked against all currently reconstructable training/evaluation
and historical-probe identities. **Non-overlap with the unresolved seed461
continuation population cannot be certified.** Seed461 opening roots that are
reconstructable are included in the exclusion union, but the 1,129,643
receipt-matched continuation actions have no authenticated state identities.
This experiment therefore makes no fresh-confirmation or untouched-holdout
claim, and it does not alter seed483’s confirmation rules or authorize generation.

## Runtime and analysis freeze

Both arms use 1,200 simulations per move, `c_puct=1.25`, one worker, a 200-ply
cap, no arm-specific transforms, and the same native exact-root behavior
(threshold 16; final-score-margin objective; highest legal network prior then
lowest move-index tie rule; exact leaf solve disabled). Search options are frozen
to FPU `zero`, subtree reuse off, value normalization off, deterministic root
policy, zero tactical root bias, zero root temperature, no root-prior transform,
and no random opening plies. Both sides receive the exact same options. All material runtime
source and executable/tablebase identities are recorded in
`seed488-frozen-strength-evaluation/registration.json`.

Primary score is `(F wins + 0.5 × draws) / 1,024`. The uncertainty interval
resamples 512 opening clusters with replacement, carrying both seat games
together, for 20,000 replicates with fixed seed 488. Advance only if the
one-sided 95% percentile lower score bound is strictly above 0.50. The two-sided
95% percentile interval is descriptive. This experiment alone cannot promote a
candidate.

The conservative simulation budget is at most 245,760,000 simulations
(`1,024 × 200 × 1,200`), plus exact-root probes. Wall time depends on game lengths
and native runtime throughput; no duration is claimed from an unrun evaluation.
The resumable runner writes one paired opening batch atomically, validates
completed seat pairs, skips completed openings, and publishes a receipt-bound
outcome ledger. It requires an explicit `--execute` after protocol review and is
not launched by preparation or verification commands.

## Readiness and commands

Prepared assets:

- `seed488-frozen-strength-evaluation/registration.json` — frozen settings and
  source/runtime bindings; `fresh_confirmation_authorized` is false.
- `seed488-frozen-strength-evaluation/openings.jsonl` — fixed suite and digest.
- `seed488-frozen-strength-evaluation/receipt.json` — preparation receipt only;
  it does not claim execution.

Read-only readiness verification:

```bash
python -m ml.alphazero_lite.verify_seed488_strength_evaluation
```

Focused synthetic tests:

```bash
python -m pytest ml/alphazero_lite/test_seed488_strength_evaluation.py -q
```

After review, an operator may explicitly start/resume the separate exploratory
evaluation with `python -m ml.alphazero_lite.seed488_strength_evaluation
--execute`. No confirmation gate is bypassed and this command is not run as part
of the preparation PR.
