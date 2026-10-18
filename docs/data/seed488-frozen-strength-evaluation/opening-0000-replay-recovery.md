# Seed488 opening-0000 replay recovery

The preserved arena pair in `resume-corrected/` was not replayable by the
PR #492-bound verifier: that verifier treated trajectory entries as
player-relative pits. The frozen arena writer serializes each selected pit as
an absolute pit index in `[0, 11]`; its opening prefix remains player-relative.
For example, absolute pit `11` is player 1's relative pit `5`.

`ml/alphazero_lite/verify_seed488_opening_replay.py` is a separate, read-only
recovery verifier. It authenticates the existing freeze, original preparation
receipt, failed-attempt evidence, frozen runtime identities, exported artifact
manifest, parent-suite opening, exact batch bytes, actual report, and two-seat
game log. It replays each absolute pit index through `KalahGame`, checks seat,
trajectory length, terminal state, captured-store margin, winner, first moves,
report settings, and report tallies. It snapshots preserved historical and
failure evidence bytes and mtimes and requires them unchanged on success or
failure. It writes no outcomes or receipts and starts no games.

Read-only invocation:

```bash
python -m ml.alphazero_lite.verify_seed488_opening_replay
```

Recovery verification of the preserved pair:

- Status: `opening_0000_pair_replay_verified`
- Parent suite SHA-256: `53841cad6c5d34355cd7d6533a0fa435f4164260fd40a2da760a3d78c784cff8`
- One-opening batch SHA-256: `4dee384ce68692d75ce8b5e0d1ca67cafefc7f87330aeb9005938f6e55a13c7b`
- Raw game log SHA-256: `34f4d32b0abc864cace899bd3b69b3ef58624be3078e972dbe2a52f08a9a3d1d`
- Actual arena report SHA-256: `d22e47c1ce979fd8067bf9a7148ca637b314817aea5dbfd94aa843b3e6559901`
- Both seats replayed for 33 plies; each replay derived margin 0 and winner `draw`, matching the actual report.

The raw opening, game log, and report remain unchanged. No outcome has been
committed, and no later opening was run. This replay correction alone does not
authorize resuming the frozen runner: its already-bound commit validator still
uses the old relative-index interpretation. Any runner integration requires a
separately reviewed correction before further execution. This remains
exploratory evidence with the unresolved seed461 overlap limitation and does
not establish fresh confirmation, promotion, or production replacement.
