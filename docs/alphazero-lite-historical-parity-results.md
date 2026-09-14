# Historical Parity Result

Inherited PR #313 classification: `baseline_reproduction_unexplained`.

## Same Runtime Check

The available shared interpreter was `/home/alex/Mancala/ai/.venv/bin/python`: Python 3.14.7, NumPy 2.4.6, and PyTorch 2.12.0+cu130. It is not the PR #313 interpreter recorded as NumPy 2.5.3 and PyTorch 2.14.0+cu130, so it cannot answer the requested current-runtime question.

H307 `e83c263193b0c08f928eb22883859ae719e577b6`, H308 `6084fd9a446afd5a78ba40d2185d6a805b05d1b7`, and current `b7a4ff8f6909f637b51afb72ef57148a31ce85ab` used isolated worktrees and SHA-verified G0, R61, the frozen set, and both fixed replay sources.

| cell | final checkpoint SHA-256 |
| --- | --- |
| H307 T61 | `50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de` |
| H308 T61 | `50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de` |
| CURRENT T61 | `50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de` |
| H308 T63 | `bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324` |
| CURRENT T63 | `bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324` |

The H308 runner was invoked through its own historical `run_cell` implementation, not ported into current code. H307 was run through its historical train command. No replay was regenerated or mutated, no self-play was run, and no checkpoint was promoted.

## Interpretation

PR #313's `-0.126246` is its raw epoch-4 in-memory anchor metric. Its saved T61 checkpoint is byte-identical to the historical H307/H308 checkpoint. The registered historical `-0.2289` is a checkpoint-evaluation metric, so these are not yet like-for-like final-state measurements.

Repository history provides only broad requirements (`numpy>=1.26,<3`, `torch>=2.3,<3`) and CI Python 3.12. No lockfile, `.python-version`, virtualenv metadata, package freeze, or retained historical runtime metadata supports reconstruction of the PR #313 NumPy/PyTorch environment.

## Classification

`historical_parity_audit_inconclusive`

Exactly one next action: restore the exact PR #313 Python 3.14.7/NumPy 2.5.3/PyTorch 2.14.0+cu130 environment from trustworthy provenance, then rerun untouched H308 T61/T63 and checkpoint-based evaluation.
