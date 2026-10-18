# Seed488 frozen direct-strength evaluation — superseding preparation

**This preparation-only correction supersedes PR #487's registration and suite
before any outcomes existed. The original registration, opening suite, and
preparation receipt are preserved byte-for-byte as `registration-pr487.json`,
`openings-pr487.jsonl`, and `receipt-pr487.json`; `supersession.json` binds their
SHA-256 digests and mtimes. No arena games/search or training has been run. Synthetic
export and policy/value inference fixtures were tested against the frozen
parameter states; no runtime arm artifacts or game outputs were produced. The
seed483 fresh-confirmation gate remains closed.** Seed479 F and the
seed480 correction remain promising exploratory evidence; this prospective arena
is an exploratory direct strength evaluation, not fresh confirmation and not an
independently untouched holdout.

The batch-identity correction also supersedes the pre-execution freeze published
by PR #488. Its registration, opening suite, and receipt are preserved as
`registration-pre-batch-identity.json`, `openings-pre-batch-identity.jsonl`, and
`receipt-pre-batch-identity.json`; the updated `supersession.json` records their
SHA-256 digests and original mtimes. Each arena call receives a one-entry JSONL
whose SHA-256 is passed as `--suite-sha256`. The 512-opening parent digest remains
registered separately and is carried with the parent `opening_index` and
`state_hash` in each batch runtime contract and outcome provenance. The arena's
local opening index is 0; the parent index is the row's index in the frozen
512-entry suite. The registered base seed remains `488 + parent_opening_index`,
so each single-entry batch is prospectively and deterministically tied to its
parent opening.

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
identity with the complete seed484 reconstructable exclusion union,
and rejects duplicate roots. Each opening is played twice, once with F seated as
player 0 and once as player 1: 1,024 total games. Arena base seed is
`488 + parent_opening_index`, under `azlite_eval_seed_v2`. Because each arena
invocation receives one opening, arena-local opening index is always 0; it must
not be confused with the parent suite index used for seed derivation and outcome
identity.

The suite is checked against the independently reconstructed **149,756-pair**
seed484 union (pair-set SHA-256
`d7c3d782181af654d56da2d02b091047df9a9cc5f78795b2cb687c9b528dfcf6`; file
SHA-256 `f419d615724f8f741fb4a58bb060901d4359e4f975ca64bfa184da4e892c10cd`).
The seed484 verifier independently reconstructed and verified this union.
**Non-overlap with the unresolved seed461
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
The runner validates runtime/source bindings and both frozen endpoint exports
before launch and on resume. It revalidates committed pairs before skipping
them. Prefixes and trajectories use the bound player-relative v2 convention and
`KalahGame` rules, including extra turns. The independent verifier replays each
trajectory and derives winner, margin, seat attribution, and F score. A game
ending before 200 plies uses terminal-swept stores; at the 200-ply cap the game
stops without sweeping and uses captured stores at that point. Interrupted
uncommitted game/report files are rejected rather than rerun. Checkpoint and
artifact verification is read-only; it compares exported arrays, weights,
metadata, and sidecars directly to frozen endpoint states. Superseded preparation
files are authenticated by registered SHA-256 digests; historical mtimes remain
provenance metadata and need not survive a checkout. Each verification snapshots
preserved files' current bytes and mtimes immediately before checking and
requires both unchanged on success or failure.

The runner requires explicit `--execute` after protocol review and is not
launched by preparation or verification commands. Do not execute it before the
correction is reviewed.

The portability correction publishes a superseding pre-execution registration
and receipt while preserving the PR #487, PR #488, and PR #489 preparation
versions and the PR #489 supersession record byte-for-byte; their recorded
mtimes remain provenance. No outcomes existed when this freeze was created.

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
