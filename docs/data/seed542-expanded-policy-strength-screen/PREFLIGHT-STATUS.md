# Seed542 expanded-policy strength screen — preflight status

**Status: blocked before arena execution; no games were run.** This directory is a
preflight record, not a completed screen or a strength result. The planned screen
remains exploratory and reused: its intended opening suite is seed511's exact
1,024-opening suite. No opening was selected, removed, or replaced based on an
outcome. There is no fresh-holdout claim, and seed461's historical-overlap
limitation remains unresolved.

## Training evidence checks

On the isolated detached worktree at merged PR #544 (`d9e4e098a6b9013cbc59c32b4bfce1420b29ac7d`), both seed542 v2 read-only training verifiers passed. They were run under the registration's user systemd scope and limits: CPUQuota=25%, CPUWeight=1, MemoryMax=4,294,967,296 bytes, AllowedCPUs=30, nice 19, address-space limit 17,179,869,184 bytes, and one thread for each configured numerical library.

Base verifier result: `verified_complete`; 108 updates replayed; C matched the current-runtime production trace; D executed; 108 registered inputs authenticated; all evidence bytes and modification times remained unchanged. Supplement result: `supplement_verified`; C and production boundary checkpoints at updates 27, 54, 81, and 108 independently matched; duplicated development effects reconciled; evidence remained read-only. Both verifier commands used `verify_seed542_fixed_update_cd_v2` and `verify_seed542_fixed_update_cd_v2_supplement` with `--root` set to this isolated checkout, the registered v2 output directory, and the checked-in `kalah_v1_21.kvtb` tablebase.

## Candidate export staged for preflight

The established exporter exported `docs/data/seed542-fixed-update-cd-v2/D-update-108.npz` to `runtime-artifacts/D-final/` as residual_v3 / kalah_v3, version `seed542-D-final-update108-policy-and-trunk`. The checkpoint and exported `model.npz` SHA-256 are both `6200b04d58f65ee5fb62555c3b32faea4b34206b4c10dbefb22ae9e7ff7b2328`; exported `weights.json` is `6190bf992605400183b8e679f4cf33d103e7a26c5bf73e43906c6d875a128a52`; metadata is `5fb713bb7c6907037ffdff4c93ea47742af836f9187e18ef79f63be96764bce6`. The frozen initializer checkpoint is SHA-256 `7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c`.

This export is only partial preflight evidence; it is not authenticated as a complete arena runtime artifact set. Policy/value component assignments, serving release reconciliation, sidecar binding, and native probe/tablebase runtime binding were not frozen for the new screen.

## Suite validation blocker

The registered seed511 suite bytes are present and have the expected SHA-256:
`docs/data/seed511-fixed-composition-screen-v4/openings.jsonl` —
`27a003322e5469507a508010a9743b40062928140e6a4b9d5628b0290e32c21d`.
Its registration SHA-256 is `89f04c6e1f774d81a7866d3913dd5ffa990b9e04a58f9a7f066b8f19204445c4`.

The checked-in seed511 verifier was run from the isolated worktree and stopped
in `seed511_fixed_composition_screen.verify_frozen_inputs` with:
`ValueError: frozen_source_changed:ml/alphazero_lite/native_exact_root_tablebase.py`.
The seed511 registration expects that source hash to be
`49fe76e8dc0e827db3181397dee7bfbcf3b1b35234757be1ac441ea09eefab6a`; the
checked-out #544 source is `bf241fa772efd122355b0706fff9035ee40f6cb9ac913fdd829444f359d62f15`.
Therefore suite replay, identity/eligibility/uniqueness checks, and overlap
measurement for this screen are not claimed complete. The source difference
must be resolved by freezing and independently validating the actual corrected
runtime sources before any games.

The initially attempted v2 verifier invocations without the registered systemd
scope failed at the resource-limit contract (`v2_verifier_resource_limit_mismatch`).
They were subsequently rerun in the documented constrained scope and passed as
recorded above. No training or arena command was run.

## Decision and remaining status

There is no result, score, interval, advancement decision, or inference about
whether D-final is stronger in play. No game ledger, raw reports, trajectories,
or overlap report exists for this screen. Pair-level recovery, parser preflight,
runner/verifier semantic mutation tests, physical-copy verification, and the
arena protocol were not started. Do not infer a draw or a failed strength gate
from this blocked preflight. No promotion, deployment, or 1,200-simulation
evaluation is authorized by this status.
