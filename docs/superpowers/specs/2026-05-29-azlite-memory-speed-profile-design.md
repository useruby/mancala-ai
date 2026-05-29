# AlphaZero-lite Memory Speed Profile Design

## Goal

Use the spare memory on `mancala-ai` to reduce wall-clock time for generation-heavy AlphaZero-lite runs, starting with checkpoint self-play and bootstrap dataset generation.

## Problem

`mancala-ai` has ample free RAM and `/tmp` tmpfs space, but the current superhuman workflow primarily exploits CPU parallelism. The active phase2 config already enables some reuse behavior and sets `--evaluator-cache-size 50000` for checkpoint self-play, but memory-backed speed knobs are not expressed as a clear runtime profile and are not applied consistently across the generation workflows we care about.

The first optimization pass should focus on the highest-leverage workloads:

- `ml/alphazero_lite/self_play.py`
- `ml/alphazero_lite/generate_bootstrap_dataset.py`

Training data format changes are a larger follow-on project and should stay out of this pass.

## Approaches Considered

### 1. Generation-focused memory profile

Promote existing knobs into a reusable high-memory runtime profile for self-play and bootstrap generation.

Pros:

- builds on flags and behavior the code already supports
- low risk of functional regression
- directly targets the longest parts of the superhuman loop

Cons:

- speedup depends on workload characteristics and cache hit rates
- still requires measurement on `mancala-ai` to choose final values

### 2. Tmpfs-first artifact placement

Force more intermediate outputs and shards onto `/tmp` tmpfs.

Pros:

- simple conceptually
- can reduce filesystem overhead for shard-heavy runs

Cons:

- expected benefit is smaller because many configs already write versions to `/tmp`
- does not directly address evaluator/search recomputation costs

### 3. Training loader rewrite

Introduce a binary replay format or preprocessed cached dataset for `train.py`.

Pros:

- could reduce JSONL parsing and replay duplication overhead

Cons:

- broader scope than needed for the first pass
- touches training data interfaces and downstream workflows
- less aligned with the immediate superhuman bottleneck

## Recommended Approach

Implement approach 1: a reusable memory-speed profile for generation-heavy local runs.

This profile should tune existing self-play and bootstrap flags instead of inventing a new training data path. The design should keep the changes small, auditable, and reversible while making it easy to run a more memory-aggressive configuration on machines like `mancala-ai`.

## Scope

This change will:

- add a reusable generation-oriented memory-speed profile for AlphaZero-lite runtime/config command materialization
- cover checkpoint self-play and bootstrap dataset generation command paths
- preserve or enable existing reuse/cache flags where they are already valid
- expose the effective settings clearly enough that run artifacts can be audited afterward

This change will not:

- redesign `train.py` data loading or introduce a binary replay cache
- change arena evaluation behavior in this first pass
- change model architecture, search budget, or promotion thresholds
- claim strength improvements without separate experiments

## Design

### Runtime profile boundary

Keep the behavior in one reusable profile layer rather than embedding more server-specific constants directly into individual configs.

The profile should be applied where runtime configs or command lists are already normalized. That keeps the feature aligned with the recent shared worker-default work: command rewriting stays centralized instead of spreading new special cases through individual scripts.

### Self-play behavior

For checkpoint-backed self-play commands:

- preserve `--tree-reuse-enabled`
- set or raise `--evaluator-cache-size` through the profile
- leave non-checkpoint self-play commands unchanged unless there is an explicit reason to cache

The initial profile should treat the current `50000` cache size as a baseline, not a guaranteed final value. The implementation should make it possible to raise that value for high-memory local runs without forcing the same default everywhere.

### Bootstrap behavior

For `generate_bootstrap_dataset.py` commands:

- preserve `--tree-reuse-enabled` for PUCT generation
- enable `--teacher-search-reuse` only when all of the following are true:
  - `--teacher-mode puct`
  - `--position-selection-mode hybrid_teacher`
  - the flag is not already set explicitly

This keeps the optimization tied to the path that actually reuses deeper teacher search state and avoids silently changing classic MCTS behavior or unrelated bootstrap modes.

### Auditability

The effective memory-speed settings should be visible in generated commands or runtime-config output so a finished run can answer:

- whether the memory-speed profile was active
- what evaluator cache size was used
- whether bootstrap teacher search reuse was enabled

The first pass does not need a separate telemetry file if the command materialization already makes the applied flags obvious.

## Error Handling

- Do not override explicit per-command flags with contradictory values unless the selected profile is documented to do so.
- Do not add `--teacher-search-reuse` when bootstrap mode cannot use it safely.
- If a command path is outside the supported allowlist, leave it unchanged.

## Testing

Add targeted regression coverage around the command-building or runtime-config normalization layer:

- self-play profile applies the intended evaluator cache setting for checkpoint commands
- self-play profile does not alter non-checkpoint commands unexpectedly
- bootstrap profile enables `--teacher-search-reuse` only for `puct` + `hybrid_teacher`
- bootstrap profile leaves explicit overrides intact

If an existing dry-run or runtime-config test path already covers these command lists, extend that coverage instead of creating a separate test harness.

## Success Criteria

- high-memory local generation runs can be configured through one reusable profile path rather than ad hoc config edits
- checkpoint self-play receives the intended evaluator cache setting through that profile
- eligible bootstrap hybrid-teacher runs receive teacher search reuse through that profile
- targeted tests pass
- resulting commands are easy to inspect after a dry-run or launched run

## Follow-on Work

If the first pass shows limited speedup, the next design candidates are:

- tmpfs placement cleanup for hot intermediates
- training replay loading optimization
- arena-specific caching for repeated evaluation workflows
