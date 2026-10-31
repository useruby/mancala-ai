# Opening-0 recovery and continuation status

This is an append-only continuation of `EXECUTION-STATUS.md`. The original
commands, raw games, report, registration, and pre-execution freeze remain
unchanged. The initial execution failure record also remains unchanged.

## Opening 0 recovered

The verifier defect was corrected by adding a separate output-completion
receipt; the original `batches/command-0000.json` was not modified. The new
`batches/completion-0000.json` binds that command hash, the exact one-opening
suite hash, and the hashes of the saved raw games and report. The corrected
read-only verifier independently replayed the raw trajectories for both seats,
reconciled the report settings and outcomes, and accepted the complete pair.

Opening 0 was committed to `outcomes.jsonl` and preserved separately in
`opening0-ledger.jsonl`. It was not rerun. Both files contain exactly the same
two validated ledger rows; their common SHA-256 is
`c52bee090106e87ec4748978d501f38892c440f81469a48f17540f1d15a84892`. The
completion receipt SHA-256 is
`f2d3417fd2f23e081f48748f347d199a3f1d22c763cb711e48fd435b2137da09`. The prior
`EXECUTION-STATUS.md` described the state before this recovery and is retained
as the original failure record.

## Training verifier receipt and continuation freeze

The two read-only seed542 v2 verifiers, including the supplement, were rerun
under the documented limits with absolute `--root` and `--out` paths. Their
corrected command/cwd/status record is `training-verification-v2.json`; the
original receipt remains unchanged.

The prospective continuation is frozen for openings 1–1023 only:

- Continuation registration SHA-256:
  `524571a4b885c8b23dd92706490246c273608eba6a8efb07ff2242997ea45c8f`.
- Continuation freeze SHA-256:
  `5dcaee1a9af8d99fcf9db7cc129ec045e9064e9189f46ad2aae22359a4b77495`.
- 1,023 remaining arena commands were validated by the actual arena parser.
- The continuation source binding authenticates the current executable files;
  the initial freeze and source snapshots remain separate historical evidence.

At the time of this v2 status update, the remaining openings had not started.
The screen was incomplete, with two validated games committed and no aggregate
analysis or advancement decision. The separate 1,200-simulation evaluation
remains unauthorized.

## Opening 1 recovery and continuation v3

The v2 continuation launched opening 1 and preserved both raw games and its
arena report. Independent replay showed the arena emits `opening_index: 0`
inside each one-entry suite invocation; the globally registered opening index
is bound by the command, per-opening input hash, seed, and ledger. The verifier
was corrected to validate that local arena index while retaining the global
opening binding. Opening 1 was then independently validated and committed from
the saved raw pair without rerunning it. Its original command, report, game
file, and v2 continuation freeze remain unchanged; its separate completion
receipt and ledger are in `batches/completion-0001.json` and
`opening1-ledger.jsonl`.

Continuation v2 is preserved as historical execution binding for opening 1.
Continuation v3 was frozen before further games, covering openings 2–1023 and
binding the corrected current executable bytes, v2 source snapshots, both
validated pair ledgers, and all remaining parser-validated commands. At this
point four games (two complete seat-swapped pairs) were validated and committed;
the remaining 1,022 openings had not started. The later final status follows.

## Final continuation, independent verification, and decision

The service subsequently completed all openings 2–1023. All 1,024 pairs / 2,048
games passed the independent no-analysis completeness verifier, first from the
original output and then from a physical copy with the original worktree path
inaccessible. The pre-analysis screen files matched byte-for-byte and by mtime
after the physical-copy run. See `POSTRUN-VERIFICATION-RESULTS.md` for verifier
identity, frozen hashes, and command details.

After both gates passed, the registered result was calculated: score
`0.486572265625`, one-sided 95% lower bound `0.4765625`, and decision **do not
advance** under the strict `> 0.50` rule. No separate evaluation or training was
started.
