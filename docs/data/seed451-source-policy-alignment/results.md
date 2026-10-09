# Seed451 source-specific policy-gradient alignment

Retrospective frozen-checkpoint diagnostic; seed447–450 outcomes were known.

**Classification:** `mixed_or_checkpoint_dependent_alignment`

Historical policy opposition: **false** across both checkpoints. Per-checkpoint U_F equal-input opposition/alignment: `{"T_final": {"equal_input_U_F_cosines": {"G_PF": 0.6599360441398906, "G_PH": 0.2525361894575257, "G_T": 0.7583120906460623}, "historical_policy_opposition": false, "joint_first_order_alignment_positive": true}, "initializer": {"equal_input_U_F_cosines": {"G_PF": 0.24237881938259626, "G_PH": -0.2475005222155044, "G_T": -0.007171159939138528}, "historical_policy_opposition": true, "joint_first_order_alignment_positive": false}}`. Joint-objective positive first-order alignment at both checkpoints: **false**.

This descriptive first-order diagnostic neither explains finite-step seed447 behavior causally nor authorizes replay-weight changes or follow-on training.
