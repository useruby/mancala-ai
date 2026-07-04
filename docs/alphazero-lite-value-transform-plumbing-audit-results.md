# AlphaZero-Lite Value Transform Plumbing Audit Results

**Classification**: `value_transform_semantically_dangerous`

## Decision Split

- Plumbing is active: extreme transforms change leaf backups, child Q, visit distributions, and root moves.
- The diagnostic path is fixed: `evaluate_artifact_position()` now forwards `value_transform` into PUCT, and the helper compares searched root-Q when available.
- Runtime risk remains: challenger-only smoke shows unstable asymmetric DS shifts under transformed runtime play.

## Artifact Hash

- Current weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`

## Promoted Search Schedule Confirmation

- Runtime profile: `{"artifact": "model-artifact/current", "c_puct_schedule": {"default_c_puct": 1.25, "overrides": {"768:768": 0.9}}, "default_c_puct": 1.25, "root_policy_mode": "deterministic", "root_prior_transform": null, "search_mode": "full", "tactical_root_bias": 0.0}`

## Static Plumbing Audit

- Search profile includes value_transform: `True`
- Opening-suite benchmark and seat-aware gate share the same arena PUCT path: `True`
- Leaf values are transformed before backup: `True`
- Terminal outcomes transformed: `False`

## Transform Definitions And Hashes

| Transform | Diagnostic only | Fitted | Hash |
|---|---|---|---|
| identity_ref | False | False | b4cd027da13708e3e07064b28da5f283e609afae41f9cb0acfca2b7c6d8ab4c7 |
| zero_value | True | False | e324f59c55c87bd1c15620a9d1ec053fdd3d62f1ef95fc913369aab1d624140e |
| negate_value | True | False | 8abb8f7c8f4059ab628bb2783e5cf9547a5414b43120b5d7e499e5c5d2d2da9d |
| sign_only | True | False | f1311e23761c0d73a1403587300eaee84102c64bb5e254b33de2d6e72e4c0e36 |
| amplify_value_2x | True | False | b1a9acb205ea5d9c99ee5475dd660ffde6725d72f4e1186fab1b5fb9c85b8fbc |
| opening_zero_value | True | False | 33cb760d7b1365bbbe32ea64e709d658b667a7368fade1d10280027ff8e52460 |
| opening_negate_value | True | False | d3cb296a876ad42eaade434fd8f6a50df85b614e31f4ed5fc67d459534746fce |
| phase_isotonic | False | True | 3b21ed7e51ccca8d09e3958321b8b435531281a898b4b8131818a2f4f5486902 |

## Root-Q Sensitivity Table

| Transform | Move change | Mean abs root value delta | Mean abs child-Q delta | Visit KL |
|---|---|---|---|---|
| amplify_value_2x | 0.1581 | 0.0793 | 0.0893 | 0.1038 |
| negate_value | 0.5420 | 0.1615 | 0.1746 | 0.7902 |
| opening_negate_value | 0.3815 | 0.0552 | 0.0659 | 0.4465 |
| opening_zero_value | 0.2314 | 0.0358 | 0.0380 | 0.1443 |
| phase_isotonic | 0.1387 | 0.0430 | 0.0559 | 0.0791 |
| sign_only | 0.3209 | 0.2395 | 0.3645 | 0.4359 |
| zero_value | 0.3403 | 0.0936 | 0.0961 | 0.2900 |

## Child-Q And Visit Telemetry Examples

```json
[
  {
    "transform": "phase_isotonic",
    "state_hash": "00325b714cf0a8b4a7abc37bde2121880e9ad08c03df3cb3438f94aa4b916f5e",
    "budget_label": "1200:1200",
    "selected_move": 5,
    "root_value_estimate": -0.9770036230615805,
    "root_evaluation_raw_value": -0.2980455756187439,
    "root_evaluation_transformed_value": -0.41257833328357973,
    "selection_breakdown": {
      "fpu_mode": "zero",
      "value_trust_multiplier": 1.0,
      "parent_q_value": -0.9770036230615805,
      "selected_move": 5,
      "reference_move": 5,
      "reference_move_kind": "highest_prior_move",
      "highest_prior_move": 5,
      "policy_top_move": 5,
      "visit_top_move": 5,
      "q_top_move": 3,
      "next_simulation_move": 2,
      "next_simulation_move_kind": "highest current PUCT selection score using deterministic telemetry ordering",
      "moves": [
        {
          "move": 2,
          "prior": 0.34418511390686035,
          "visit_count": 345,
          "q_value": -0.9840296156269691,
          "selection_q_value": -0.9840296156269691,
          "q_component": -0.9840296156269691,
          "u_component": 0.04307414049823436,
          "selection_score": -0.9409554751287348,
          "used_fpu": false,
          "fpu_value": null
        },
        {
          "move": 3,
          "prior": 0.08718501031398773,
          "visit_count": 213,
          "q_value": -0.9586346766072629,
          "selection_q_value": -0.9586346766072629,
          "q_component": -0.9586346766072629,
          "u_component": 0.01764122284138357,
          "selection_score": -0.9409934537658793,
          "used_fpu": false,
          "fpu_value": null
        },
        {
          "move": 5,
          "prior": 0.5686298608779907,
          "visit_count": 642,
          "q_value": -0.9793223429365185,
          "selection_q_value": -0.9793223429365185,
          "q_component": -0.9793223429365185,
          "u_component": 0.03829299415791222,
          "selection_score": -0.9410293487786063,
          "used_fpu": false,
          "fpu_value": null
        }
      ]
    }
  },
  {
    "transform": "zero_value",
    "state_hash": "00325b714cf0a8b4a7abc37bde2121880e9ad08c03df3cb3438f94aa4b916f5e",
    "budget_label": "1200:1200",
    "selected_move": 5,
    "root_value_estimate": -0.9325,
    "root_evaluation_raw_value": -0.2980455756187439,
    "root_evaluation_transformed_value": 0.0,
    "selection_breakdown": {
      "fpu_mode": "zero",
      "value_trust_multiplier": 1.0,
      "parent_q_value": -0.9325,
      "selected_move": 5,
      "reference_move": 5,
      "reference_move_kind": "highest_prior_move",
      "highest_prior_move": 5,
      "policy_top_move": 5,
      "visit_top_move": 5,
      "q_top_move": 3,
      "next_simulation_move": 5,
      "next_simulation_move_kind": "highest current PUCT selection score using deterministic telemetry ordering",
      "moves": [
        {
          "move": 2,
          "prior": 0.34418511390686035,
          "visit_count": 308,
          "q_value": -0.9448051948051948,
          "selection_q_value": -0.9448051948051948,
          "q_component": -0.9448051948051948,
          "u_component": 0.04823188547698734,
          "selection_score": -0.8965733093282074,
          "used_fpu": false,
          "fpu_value": null
        },
        {
          "move": 3,
          "prior": 0.08718501031398773,
          "visit_count": 346,
          "q_value": -0.9075144508670521,
          "selection_q_value": -0.9075144508670521,
          "q_component": -0.9075144508670521,
          "u_component": 0.010879601406501682,
          "selection_score": -0.8966348494605504,
          "used_fpu": false,
          "fpu_value": null
        },
        {
          "move": 5,
          "prior": 0.5686298608779907,
          "visit_count": 546,
          "q_value": -0.9413919413919414,
          "selection_q_value": -0.9413919413919414,
          "q_component": -0.9413919413919414,
          "u_component": 0.04501351964083648,
          "selection_score": -0.8963784217511049,
          "used_fpu": false,
          "fpu_value": null
        }
      ]
    }
  }
]
```

## Medium Smoke DS Table

- Calibration-state input had no serialized `state` payloads, so the root probe backfilled non-opening states with deterministic traced continuations from the suite seeds.
- Smoke evaluation now applies transforms to the challenger path only; the current side remains on identity.

| Lane | 384:256 | 768:768 | 1200:1200 | 1200:256 |
|---|---|---|---|---|
| identity_ref | -0.2344 | +0.6484 | +0.3906 | -0.1367 |
| negate_value | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| opening_negate_value | +0.0742 | +0.0000 | -0.1016 | +0.3008 |
| opening_zero_value | -0.4961 | -0.1758 | -0.1367 | -0.3438 |
| phase_isotonic | -0.3164 | +0.0742 | +0.0859 | -0.2500 |
| zero_value | +0.1406 | +0.0000 | -0.0273 | +0.0039 |

## Recommendation

- Strongest fitted transform audited: `phase_isotonic`.
- Final classification: `value_transform_semantically_dangerous`.
