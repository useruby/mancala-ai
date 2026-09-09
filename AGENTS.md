# AGENTS

## PR Creation

- When a user asks to create a PR, first inspect `git status --short`, `git diff` for intended files, and `git log --oneline -10`.
- Stage only the intended files for the task. Do not include unrelated untracked paths or user work.
- Create a focused commit with a concise message that matches recent repo style.
- Push the working branch and open the PR with `gh pr create` against `main` unless the user says otherwise.

## Waiting For Review

- After requesting review, wait for review comments before making follow-up changes unless the user asks for more work immediately.
- To check review state, inspect both the PR view and the requested-reviewers API when needed:

```bash
gh pr view PR_NUMBER --json reviewRequests,reviews,latestReviews
gh api repos/useruby/mancala-ai/pulls/PR_NUMBER/requested_reviewers
```

## Addressing Review Comments

- When review comments appear, automatically address them unless the user asks to review them first or a comment is ambiguous.
- Before editing, read the exact review comments and inspect the referenced code carefully.
- Make the smallest correct fix that resolves the feedback.
- Run the relevant verification for the changed code.
- Commit the fixes, push the branch, and reply on the PR with a concise summary of what was addressed when appropriate.
- If a review comment is unclear, conflicting, or would require a product decision, stop and ask the user one short clarifying question.

## Temp Files and Scratch Space

- NEVER write experiment artifacts, venvs, datasets, or model outputs to `/tmp`.
  `/tmp` is a tmpfs mount with a per-user quota — filling it breaks developer tools
  (opencode/Bun fails to start with `Failed to open library ... libopentui-*.so`
  because it cannot extract its native lib).
- Use the repo-local `./.tmp/` directory (gitignored) for run artifacts instead.
- If a tool needs temp space outside the repo, use `~/tmp` and export
  `TMPDIR=$HOME/tmp` and `BUN_TMPDIR=$HOME/tmp`.
- When tools start failing with "Disk quota exceeded", check `du -sh /tmp/*`
  and delete stale run dirs whose results are already recorded elsewhere.
  Orphaned `/tmp/.9adb*.so` extracts and `pyright-*`/`pymp-*` caches are always safe to delete.
