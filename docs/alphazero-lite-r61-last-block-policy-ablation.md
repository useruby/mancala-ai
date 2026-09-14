# R61 Last-Block Policy Ablation

## Status

`last_block_policy_ablation_inconclusive`

The SHA-verified PR #315 full-scope and PR #316 heads-only checkpoint directories are not present in this checkout, so the two new train-only lanes were not run. No self-play, replay mutation, search change, or promotion occurred.

Exactly one next experiment: recover the SHA-verified inherited PR #315/#316 artifacts and execute this same R61 `last_block_policy` ablation.

## Inherited Results

- PR #315: `anchor_drift_shared_trunk_primary`.
- PR #316: `heads_only_no_anchor_rescue`.

## Scope Manifest

The runner mechanically verifies a three-block `residual_v3` model and derives the final-block prefix as `residual_layers.{len(model.residual_layers) - 1}.`.

- Trainable: final residual block, `policy_hidden_layer.*`, `policy_head.*`.
- Frozen: `input_layer.*`, earlier residual blocks, `value_hidden_layer.*`, `value_head.*`.

## Artifact SHAs

- G0: `4cd77f12319935d776b68c5e597b8d399fd68f15ba7ea25b3d66bbe786292ed4`.
- Frozen evaluation set: `5f86367fdb75af02294a1ace1c08dc2938919f0a5f0419376bdf89d127b6257a`.
- Inherited checkpoint SHA verification is implemented but could not run because those checkpoint files are absent.

## Pending Measurements

The executable runner records exact frozen-tensor equality after every epoch and restored-best selection; final-block/policy movement; G0 and epoch 1--4 legal-normalized anchor trajectories; final-block drift cosines against full T61/T63; frozen cluster/control metrics and forgotten/repaired counts; validation and frozen-set value metrics; outcome-aligned exact-forensic metrics; and the fixed lightweight arena including Wilson interval and effect.

No measurement values are reported before the required artifacts are verified.
