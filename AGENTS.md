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
