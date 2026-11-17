# Seed 604 opening-cohort recovery audit

The first local generation attempt is classified as
`incomplete_provenance_not_publishable`. Its saved registration records generator
SHA-256 `9227717501d21874942a03ea553b7a0c044dfe8ad3be2d95687e2ec7061ad821`,
while the untracked generator present at audit time hashes to
`2e82cb170702c566a7a27f3f444d91dd9587ac7b0be09918fb39abb00bb2e094`. The
generator source was edited after its manifest was produced, and no execution-time
source snapshot or commit binding exists. The old manifest is not copied into or
used by this recovery. The original worktree and its incomplete artifacts were
left unchanged.

Recovery starts at merged PR #606 (`07ad18318919f03c92fe49cf5e337f108ee7c930`)
in a separate clean Git worktree. The fresh attempt below uses a generator and
independent verifier committed before generation, with source and rules hashes
bound into its registration. An initial isolated `-v2` attempt was also
invalidated and preserved without changes: its generated report marked
verification as passed before the independent verifier had been run. It is not
used as evidence. The `-v3` attempt was also invalidated:
Python loaded a stale cached generator bytecode, and the verifier rejected the
manifest because the execution-time generator digest did not match the committed
source. It remains untouched and is not used. The final fresh attempt uses a
new `-v4` destination, with the exact cache disabled and bytecode writes
disabled for generation and verification.
