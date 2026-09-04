# Kalah V1 Native Tablebase Preflight Plan

## Scope

This diagnostic-only experiment evaluates an isolated C++17 `kalah_v1` tablebase. It does not modify `EndgameTablebase`, search, self-play, artifacts, registries, training, or the Geoffrey Irving compatibility implementation. Generated databases are written only beneath a validated temporary directory.

## Preregistered Gates

1. Exhaustive dense rank/unrank through 10 active stones, including both players and one-sided states.
2. Exhaustive transition parity through 8 stones against `KalahGame` and `ExactState`.
3. Store-offset and action-order invariance through 8 stones.
4. Exhaustive native values and action values through 8 stones against `ExactKalahSolver`.
5. Deterministic byte-identical 8- and 12-stone outputs.
6. Only after all correctness gates pass, generate 8, 10, 12, and 14 cumulative tiers with 30-minute, 8-GiB RSS, and 8-GiB temporary-disk caps.
7. Stop at the first correctness or resource failure. Classification priority is: incorrect, recurrence blocked, budget exceeded, correct but not scalable, feasible.

## Fixed Representation

For pits `p` and player `q`, store `F(p, q)`, the eventual player-zero margin contributed by all active stones. A complete position returns `store_0 - store_1 + F(p, q)`. Values are signed `int8`; for tiers through 20, `F` is in `[-20, 20]`, while `-128` is the explicit uninitialized sentinel.

## File And Execution Rules

The file format is little-endian `KVTB1`, schema 1, with `kalah_v1`, tier metadata, generator revision, generation parameters, and SHA-256 payload checksum. The generator records all measurements and hashes in JSON results. No generated `.kvtb` file is committed.
