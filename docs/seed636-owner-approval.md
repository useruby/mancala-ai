# Seed 636 — merged-registration authorization

## Authorization contract

For Seed 636, merging the registration PR into `main` in
`useruby/mancala-ai` by repository owner `useruby` is sufficient authorization
for the registered population-generation scope. No separate owner comment,
approval JSON, copied hash, or authorization reference is required.

The registration PR must contain `.experiment/registration.json` and explicitly
bind:

- the Seed 625 finite-cohort estimand amendment;
- the Seed 635 worktree-based execution contract;
- `known-exposure-screened; inherited P-E4 exposure unresolved`; and
- `model-independent-generation-and-screening-only`.

The registration also pins protocol-document SHA-256 values, inputs, generation
parameters, resource budgets, scientific acceptance criteria, and stop rules.
Its merge authorizes population generation and membership screening only. It
does not authorize inference, MCTS, exact solving, tablebase probes, training,
candidate promotion, or later benchmark stages.

## Automatic authorization verification

The Seed 636 execution entry point obtains GitHub PR associations for its exact
source commit and fetches the associated PR metadata. It accepts exactly one PR
only when the PR is merged into this repository's `main` by `useruby`, its merge
commit is the attempt source commit, and the committed
`.experiment/registration.json` bytes at that merge commit exactly match the
worktree registration. It checks the required decision list, scope, protocol
blobs, scientific criteria, and registered resource budgets from the committed
Git objects. The independent verifier repeats the GitHub metadata and Git-object
checks.

The generator automatically records the PR number, merge commit, merged-by
identity, merge timestamp, registration digest, repository, scope, and approved
decisions in the generated authorization evidence. No authorization record is
manually authored or committed before execution. The attempt still requires the
ordinary clean-source lifecycle freeze, authenticated inputs, immutable attempt
records, independent verification, and artifact archival.

The project owner merges the PR explicitly containing both this contract and
the registration. No automatic merge is permitted. The ordinary Seed 635
worktree lifecycle remains in effect, and the Seed 629 production launch guard
is unchanged.
