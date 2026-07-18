# Canonical Runtime Profile Revalidation

Classification: `runtime_profile_interaction_detected`.

Artifact SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`. Seed contract: `azlite_eval_seed_v1`; base seed: `42`.

Seed identity ledgers contain only canonical seed inputs and derived identities; outcome ledgers contain moves, visits, cache execution, trajectories, and profile hashes. Detailed ledgers remain in the workdir.

## Profiles

```json
{
  "legacy_tactical_no_schedule": {
    "name": "legacy_tactical_no_schedule",
    "default_c_puct": 1.25,
    "c_puct_schedule": {},
    "tactical_root_bias": 0.1,
    "root_policy_mode": "deterministic",
    "normalize_values": false,
    "value_transform": null,
    "root_prior_transform": null,
    "hash": "d3f1218d81c847c4de3348b45eda61817995a76bb615d7908fcb182c2e2d8445"
  },
  "no_tactical_no_schedule": {
    "name": "no_tactical_no_schedule",
    "default_c_puct": 1.25,
    "c_puct_schedule": {},
    "tactical_root_bias": 0.0,
    "root_policy_mode": "deterministic",
    "normalize_values": false,
    "value_transform": null,
    "root_prior_transform": null,
    "hash": "8fbf9f1d206ac656e3927f6e1b98bb02b9b2e9f48222b352f4e51d61077e435e"
  },
  "legacy_tactical_with_schedule": {
    "name": "legacy_tactical_with_schedule",
    "default_c_puct": 1.25,
    "c_puct_schedule": {
      "768:768": 0.9
    },
    "tactical_root_bias": 0.1,
    "root_policy_mode": "deterministic",
    "normalize_values": false,
    "value_transform": null,
    "root_prior_transform": null,
    "hash": "c3e43ed7e28d44b4a2328189a2264c883711146a0acc4a407049e6a4d476759d"
  },
  "current_promoted_profile": {
    "name": "current_promoted_profile",
    "default_c_puct": 1.25,
    "c_puct_schedule": {
      "768:768": 0.9
    },
    "tactical_root_bias": 0.0,
    "root_policy_mode": "deterministic",
    "normalize_values": false,
    "value_transform": null,
    "root_prior_transform": null,
    "hash": "934b462967acf516ee1a355b24df452d5b8ea1b5f5abf595ca1ffd8b15776d4e"
  }
}
```

## Results

Aggregate medium, fixed-large, held-out, P0/P1, latency, paired opening-cluster CI, and factorial contrast tables are in `docs/data/alphazero-lite-canonical-runtime-profile-revalidation-summary.json`.
