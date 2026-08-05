import unittest

from ml.alphazero_lite.evaluation_seed_contract import (
    SEED_CONTRACT_VERSION,
    SEED_IDENTITY_FIELDS,
    derive_search_seed,
    search_configuration_ledger_record,
    stable_hash,
    search_seed_context,
    verify_provenance_ledgers,
)


class EvaluationSeedContractTests(unittest.TestCase):
    def context(self, **overrides):
        value = {
            "base_seed": 42,
            "suite_sha256": "suite",
            "budget_pair": "384:256",
            "opening_index": 3,
            "opening_state_hash": "opening",
            "challenger_player": 0,
            "game_within_opening": 0,
            "ply": 7,
            "canonical_current_state_hash": "state",
            "acting_role": "challenger",
            "simulations": 384,
            "effective_c_puct": 1.25,
        }
        return {**value, **overrides}

    def test_same_context_reproduces_exactly(self):
        self.assertEqual(
            derive_search_seed(**self.context()), derive_search_seed(**self.context())
        )

    def test_model_and_execution_metadata_are_not_seed_inputs(self):
        seed, context_hash = derive_search_seed(**self.context())
        self.assertEqual((seed, context_hash), derive_search_seed(**self.context()))
        self.assertNotIn("artifact_path", SEED_IDENTITY_FIELDS)
        self.assertNotIn("artifact_hash", SEED_IDENTITY_FIELDS)
        self.assertNotIn("candidate_label", SEED_IDENTITY_FIELDS)
        self.assertNotIn("worker_id", SEED_IDENTITY_FIELDS)

    def test_base_seed_and_state_change_seed(self):
        first = derive_search_seed(**self.context())
        self.assertNotEqual(first, derive_search_seed(**self.context(base_seed=43)))
        self.assertNotEqual(
            first,
            derive_search_seed(**self.context(canonical_current_state_hash="other")),
        )

    def test_hash_is_canonical_and_cryptographic_shape(self):
        self.assertEqual(stable_hash({"b": 2, "a": 1}), stable_hash({"a": 1, "b": 2}))
        self.assertEqual(64, len(stable_hash(SEED_CONTRACT_VERSION)))

    def test_canonical_serialization_has_no_model_identifier(self):
        context = search_seed_context(**self.context())
        serialized = str(context)
        self.assertNotIn("artifact", serialized)
        self.assertNotIn("candidate", serialized)
        self.assertNotIn("composition", serialized)

    def test_treatments_do_not_change_v2_seed_identity(self):
        baseline = self.context()
        lower_cpuct = self.context(effective_c_puct=0.90)
        larger_budget = self.context(simulations=768, budget_pair="768:768")
        self.assertEqual(
            derive_search_seed(**baseline), derive_search_seed(**lower_cpuct)
        )
        self.assertEqual(
            derive_search_seed(**baseline), derive_search_seed(**larger_budget)
        )
        seed, context_hash = derive_search_seed(**baseline)
        first = search_configuration_ledger_record(
            seed_context_hash=context_hash,
            simulations=256,
            effective_c_puct=1.25,
            tactical_root_bias=0.10,
            runtime_profile_hash="a",
            budget_pair="256:256",
            artifact_hash="artifact-a",
        )
        second = search_configuration_ledger_record(
            seed_context_hash=context_hash,
            simulations=768,
            effective_c_puct=0.90,
            tactical_root_bias=0.00,
            runtime_profile_hash="renamed-profile",
            budget_pair="768:768",
            artifact_hash="artifact-b",
        )
        self.assertNotEqual(
            first["search_configuration_hash"], second["search_configuration_hash"]
        )
        self.assertEqual(seed, derive_search_seed(**self.context())[0])

    def test_state_changes_v2_seed_and_mixed_contracts_are_rejected(self):
        first = seed_identity = {
            **search_seed_context(**self.context()),
            "seed_context_hash": derive_search_seed(**self.context())[1],
            "derived_search_seed": derive_search_seed(**self.context())[0],
        }
        other = self.context(canonical_current_state_hash="other")
        self.assertNotEqual(
            first["derived_search_seed"], derive_search_seed(**other)[0]
        )
        v1 = {
            **search_seed_context(
                **self.context(), contract_version="azlite_eval_seed_v1"
            ),
            "seed_context_hash": derive_search_seed(
                **self.context(), contract_version="azlite_eval_seed_v1"
            )[1],
            "derived_search_seed": derive_search_seed(
                **self.context(), contract_version="azlite_eval_seed_v1"
            )[0],
        }
        with self.assertRaises(ValueError):
            verify_provenance_ledgers(
                seed_identity_ledger=[seed_identity, v1],
                search_configuration_ledger=[],
                search_outcome_ledger=[],
            )
