# Seed512 examined exploratory follow-up

## Result

P scored **0.525634765625** against the complete frozen initializer: **947
wins, 259 draws, and 842 losses** in 2,048 games. The 20,000-resample,
seed-512 paired-opening bootstrap gave a one-sided 95% lower bound of
**0.514892578125** and a descriptive two-sided 95% interval of
**[0.5126953125, 0.538818359375]**. The registered primary gate passed because
the lower bound exceeds 0.50.

The matched per-opening difference from seed511 (1,200 minus 128 simulations)
was **0.006591796875**. Its descriptive paired-opening bootstrap interval was
**[-0.01318359375, 0.0263671875]**. Both seats remained together in each
resampled opening. This is secondary descriptive analysis and does not replace
the standalone primary gate or pool historical games into the score.

## Examined exploratory status and decision limits

The seed511 suite and its 128-simulation outcomes had already been examined
before seed512. This follow-up and its outcomes are also examined. It is
**exploratory, not fresh confirmation**. A pass supports proposing a subsequent
deployment and evaluation proposal only; it does not authorize promotion. No
training, checkpoint selection, tensor splicing, candidate changes, or
promotion occurred. Seed505's separate failed gate remains failed. The
seed461 historical overlap/replay limitation remains disclosed and unresolved.

## Frozen candidate and protocol

- Candidate P policy output: seed504 E4, with its own trunk.
- Candidate P value output: the frozen initializer, with its own trunk.
- Opponent: complete frozen initializer artifact.
- Exact seed511 checkpoint, artifact and sidecar identities are bound in
  `registration.json`; no artifact contents were changed.
- Exactly the same 1,024 frozen seed511 opening roots, in both seats: 2,048
  **new** games. The suite SHA256 is
  `27a003322e5469507a508010a9743b40062928140e6a4b9d5628b0290e32c21d`.
- 1,200 simulations per side; `c_puct=1.25`; one worker;
  `azlite_eval_seed_v2`; base seed `511 + opening index`; 200-ply cap; zero
  random-opening plies; unchanged seed511 search options and exact-root
  behavior.
- The seed511 environment's recorded Python and dependency versions were
  verified. Execution used that environment on CPU (`torch.cuda.is_available()`
  was false); inference backend and registered game/search settings were not
  changed.
- Mean game length was **34.16796875** plies, maximum **66**, and no game
  reached the cap.

## Execution and verification

The first registered pair (opening 0, both seats) completed in **11.5286
seconds**. At that point, the simple estimate for 1,023 remaining pairs was
**11,793.76 seconds** (about 3 h 16 min). The completed 1,024 pairs consumed
**10,965.18 seconds** in aggregate (about 3 h 03 min), with mean pair execution
time **10.7082 seconds**. The first pair was retained in the final outcomes.

`commands.jsonl` records each resolved invocation, assignments, and freeze
hashes. Each pair's raw game JSONL, report, and opening are kept under `batches/`;
`outcomes.jsonl` links each ledger row to those file hashes. The independent
verifier replayed every trajectory, checked runtime and seat contracts, and
recomputed the primary and matched bootstrap analyses. The publication receipt
binds the complete result bundle.

See `registration.json`, `execution-freeze.json`, `first-pair-timing.json`,
`verification.json`, and `publication-receipt.json` for machine-checkable
protocol, runtime, and integrity records.
