# Proposed amendment: Seed 623–631 execution requirements

**Status:** approved for the Seed 636 registered population-generation scope
when its registration PR is merged into `main` by `useruby`. It does not change a
production launch guard or alter archived Seed 623–631 evidence.

## Proposed replacement

Replace any interpretation of the Seed 623–631 readiness contract that makes a
separate privileged supervisor qualification, delegated cgroup subtree, XFS
project-quota filesystem, disposable VM, root collector, or test signing key a
prerequisite for ordinary experiment execution with the common procedure in
[`experiment-execution-protocol.md`](experiment-execution-protocol.md).

After the experiment's registration PR is merged by the project owner, each
experiment remains subject to:

- the Seed 623 scientific preregistration, population definition, known-exposure
  limitation and all relevant authorization gates;
- exact authentication of checkpoint, rules, tablebase, cohort, inputs and
  executable identities using existing domain verifiers;
- frozen configuration, random seeds, worker count, CPU/GPU/device, expected
  runtime, disk budget, command, outputs and acceptance/stop criteria;
- resource limits appropriate to that experiment and runtime, using existing
  experiment-specific controls and checkpoints;
- preservation of every attempt and its failure/recovery history;
- complete exact-oracle action coverage and independent solver agreement where
  required by the scientific protocol;
- independent reproduction of scientific calculations from archived inputs and
  outputs; and
- a separate owner-merged registration PR for each newly gated scientific
  stage, plus independent scientific review where specified.

The Seed 623 requirements for the 512-state cohort, 17–21 stone domain, five
equal-weight bins, cluster caps, known-exposure screening, all-legal-action
oracle values, calibration gates, 128-search-simulation budget, single search
worker, and stop conditions are scientific/reproducibility requirements and
remain intact. The ordinary experiment process records and verifies them; it
does not waive them.

## Legacy implementation disposition

Keep Seeds 629–631 source, tests, and historical readiness reports unchanged as
historical implementations and evidence. Do not run or remove them as a side
effect of this proposal. Subsequent experiment registrations may use the common
worktree-based protocol rather than the privileged Seed 631 qualification
matrix. The Seed 624 generation path requires a frozen, started lifecycle
attempt and a project-owner-merged registration PR bound to the registration's
exact Git objects.

In particular, the Seed 629 `run_authorized()` launch guard remains
unconditional. This protocol neither calls it nor bypasses it. A future request
to change that production guard requires explicit owner approval, a separately
reviewed code change, and an explicit decision on how production launches are
authorized. A general worktree utility cannot itself authorize a protected
production workload.

The privileged Seed 629 execution path remains disabled and is superseded for
ordinary research experiments by the Seed 635 worktree lifecycle. Its
unconditional guard remains intact for its historical protected workload.

## Current execution contract

All experiment worktree, registration PR, independent review, owner-merge
authorization, freeze, execution, verification, archive, results publication,
and cleanup requirements are defined by the mandatory
[`experiment-execution-protocol.md`](experiment-execution-protocol.md). This
amendment adds Seed 623 scientific requirements only; it does not define a
separate approval-comment mechanism or weaken the common lifecycle.

The Seed 636 registration PR approves the finite-cohort estimand, this
worktree-based execution contract, the inherited P-E4 limitation, and the
model-independent generation/screening scope. Later Seed 623 inference,
calibration, exact solving, or search work requires its own registration and
owner-merged authorization under the common protocol. This amendment makes no
scientific result claim and does not qualify a production supervisor.
