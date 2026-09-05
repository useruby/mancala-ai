# Kalah V1 Tier-18 Hybrid Experiment Plan

## Scope

This is an isolated diagnostic experiment. It generates and validates a canonical
`KVTB1` tier-18 tablebase for use only by a native MTD(f) probe. It does not
alter production search, train a model, create training-eligible labels, or use
the incompatible Geoffrey Irving endgame database.

## Preregistered Order

1. Run tablebase correctness gates before any scalability or performance gate.
2. Generate tier 18 twice in separate temporary directories under eight-hour,
   16-GiB RSS, and 4-GiB output limits. Require byte-identical payload and
   complete-file hashes, sizes, state counts, and edge counts.
3. Validate portable format, validated tier-8 and tier-12 prefixes, exhaustive
   tier-8 oracle values/actions, the deterministic 10,000 tier-9--12 sample,
   and deterministic stratified tier-13--18 Bellman checks through an
   independent Python decoder.
4. Only after those gates pass, load the artifact in the isolated native probe
   and run hybrid correctness gates before the unchanged 96-state corpus.
5. Classify using the priority specified by the experiment request. All corpus
   records remain `training_eligible: false`.

## Representation

The tablebase stores one signed byte per canonical storeless state, indexed by
composition rank and current player. Its value is the eventual player-zero
margin from active stones. A complete position is evaluated as
`store_0 - store_1 + tablebase_storeless_value`.

## Reproduction

Run `python -m ml.alphazero_lite.run_kalah_v1_tier18_experiment` after the
native adapter is available. The command records compiler, platform, generator
revision, exact invocation, resource measurements, checksums, and validation
results without retaining a generated `.kvtb` in the repository.
