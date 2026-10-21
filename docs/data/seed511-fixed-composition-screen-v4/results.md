# Seed511 fixed-composition exploratory screen

## Result

The registered P composition scored **0.51904296875** against the complete
frozen initializer: 965 wins, 196 draws, and 887 losses in 2,048 games. The
20,000-resample, seed-511 paired-opening bootstrap gave a one-sided 95% lower
bound of **0.5068359375** and a descriptive two-sided 95% interval of
**[0.50439453125, 0.53369140625]**. The registered screen criterion passed
because the lower bound is greater than 0.50.

This is a **new exploratory suite**, not an untouched or fresh-confirmation
holdout. Passing only permits proposing a later 1,200-simulation test. It does
not promote a checkpoint, establish fresh confirmation, or reopen seed505's
failed gate. No training, checkpoint selection, tensor splicing, or promotion
was performed.

## Frozen identities and protocol

- Challenger P policy output: seed504 E4, evaluated with its own complete
  artifact/trunk.
- Challenger P value output: the frozen initializer, evaluated with its own
  complete artifact/trunk.
- Opponent: the complete frozen initializer artifact.
- Bound checkpoint hashes: E4 `2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad`;
  initializer `7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c`.
- 1,024 rules-only roots generated with seed511 and the seed488 prefix-length
  law (`randint(4, 12)`), each played in both seats: exactly 2,048 games.
- Search/runtime: 128 simulations per side, `c_puct=1.25`, one worker,
  `azlite_eval_seed_v2`, base seed `511 + opening index`, zero random opening
  plies, exact-root threshold 16, `final_score_margin` objective, highest legal
  prior then lowest move-index tie break, exact leaf solve disabled, 200-ply
  cap.
- No games reached the cap; mean length was 31.740234375 plies and maximum
  length was 56.
- Score: `(wins + 0.5 × draws) / 2,048`. Bootstrap resampled 1,024 opening
  clusters with both seat results kept together, 20,000 replicates, seed 511.

`registration.json` and `execution-freeze.json` bind the exact suite, code,
runtime, artifact, environment, analysis and input hashes. `commands.jsonl`
records all resolved invocations, component assignments, runtime settings and
hashes. `publication-receipt.json` binds the published output files.

## Exclusions and limits

Each prospective root was excluded if its canonical-state identity **or** its
production float32 input identity matched any included source, and duplicate
roots were rejected by either identity. The registration publishes hashes for
every included source file. The reconstructed union consists of:

| Included source | Evidence used |
|---|---|
| Existing reconstructable historical exclusion set | Seed482 exclusion manifest, seed483 reconstructed identities, and seed484 extended union; 149,756 bound identity pairs |
| Seed500 pilot and corpus | Pilot rows plus all eight original seed500 shards, each checked against `accounting.json`; 20,657 input rows |
| Seed488 | Replayed opening prefixes and all 1,024 game trajectories |
| Seed499 | Replayed opening prefixes and all 1,024 game trajectories |
| Seed505 | Replayed opening prefixes and all 1,024 game trajectories |
| Seed508 | Replayed all 2,048 games from its P and V batch game files |

After source union and de-duplication, the screened source identities numbered
254,802 canonical and 254,802 production-float32 identities. The seed461
continuation replay gap remains disclosed; its recovery was not reopened and no
complete historical-exclusion claim is made.

## Checkout and LFS recovery

The evaluation worktree is based on #510 (`8f5063df571e521363afe4c01e6eb431aca32d3d`)
and includes seed503 merge `50f42ef63e72d297b66c5f9dbaf52c0bf8f9e1f6`; sparse
checkout was disabled. The initial apparent absence of seed500 shards came
from a non-recursive listing: all eight paths are committed and present. Their
individual hashes match the archive accounting; corpus replay validated 512
complete games and 18,281 decision rows; the production loader accepted all
rows and the recorded accounting agrees.

The #510 worktree had the registered 134-byte LFS pointer for the exact-root
tablebase because LFS smudge/filter configuration was disabled. Its shared LFS
cache contained the exact object with the registered SHA256/OID
`f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7`. The runner
verified and temporarily hard-linked this object at its registered path for
arena execution, then restored the pointer. The tree is unchanged by this
materialization step.

Three pre-execution attempts exposed setup defects (runtime LFS materialization,
runtime path routing, then an unsupported `random_opening_plies` search override).
Each stopped before a game completed; no outcomes were used to select or replace
roots. The final v4 registration passed preflight and froze the same
deterministically regenerated suite before inference or games. The unsuccessful
pre-execution files remain outside this results publication.

## Independent verification

The seed511 verifier independently replayed all 2,048 trajectories from the
frozen roots, checked legal transitions, seat assignments, terminal results,
runtime reports, command receipts, and component/runtime identities, then
recomputed the paired-opening bootstrap directly from the ledger. It reproduced
the published counts, score, interval and pass decision. Synthetic controls
also cover command routing, interruption/resume preservation and semantic
corruption rejection. The corrected seed508 verification tests pass as an
additional regression check.

The repository's commit hook applied Ruff formatting to the three seed511
Python files after the games had completed. `execution-source-snapshots/`
preserves the exact pre-format source and test bytes whose hashes are bound in
the immutable registration. Their hashes were recovered from the staged index
and match every registered digest. The final verifier checks these snapshots
when the formatted working files no longer match the frozen byte digests; the
suite, commands, artifacts, environment, and game data were not changed.

Run from the repository root with the registered environment:

```bash
python -m ml.alphazero_lite.verify_seed511_fixed_composition_screen
python -m pytest ml/alphazero_lite/test_seed511_fixed_composition_screen.py \
  ml/alphazero_lite/test_seed508_policy_value_crossover.py \
  ml/alphazero_lite/test_seed508_verification_supplement_v2.py -q
ruff check ml/alphazero_lite/seed511_fixed_composition_screen.py \
  ml/alphazero_lite/verify_seed511_fixed_composition_screen.py \
  ml/alphazero_lite/test_seed511_fixed_composition_screen.py
```
