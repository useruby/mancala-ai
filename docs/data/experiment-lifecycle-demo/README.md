# Experiment lifecycle synthetic demonstration

The unprivileged regression test
`ml.alphazero_lite.test_experiment_lifecycle.ExperimentLifecycleTests.test_success_archive_publication_and_safe_cleanup`
is the executable demonstration. It creates a temporary Git repository and a
separate experiment worktree, commits a registration at a pinned base, freezes
an attempt with an authenticated fixture input and deterministic command, runs
the command, verifies its output/log identities, creates and independently
extracts a `.tar.zst` (or `.tar.gz` fallback) archive, prepares publication
files, verifies dirty-worktree cleanup refusal, merges the experiment branch,
and removes the now-clean worktree only after archive verification. The
companion tests preserve timeout/launch/disk failures, interrupted process
cleanup, post-freeze source/HEAD/file-and-directory-input changes, directory
model artifacts, two historical attempts across source and publication commits,
archive/reference tampering and inventory drift, duplicate archive, and duplicate
experiment/attempt identity behavior. Prelaunch authentication rejects modified
dependencies; archive verification records input-mismatch failures as
unverified while preserving their runtime evidence. Cleanup compares the full
runtime inventory to its attempt-bound archive and committed publication anchor.

Reproduce from the repository root with:

```sh
python -m unittest ml.alphazero_lite.test_experiment_lifecycle
```

The fixture writes a small JSON value (`{"answer":42}`) only. It performs no
Mancala game generation, model inference, training, MCTS, exact solving, or
evaluation. The test's temporary worktree and archive are deleted by its
temporary-directory cleanup; this document records the test procedure and
observed pass, not a scientific result or persistent research artifact.

**Classification:** lifecycle integration demonstrated in an unprivileged
synthetic fixture; no scientific outcome, experiment authorization, or
production readiness claim.
