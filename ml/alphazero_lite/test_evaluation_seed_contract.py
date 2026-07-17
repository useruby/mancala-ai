import unittest

from ml.alphazero_lite.evaluation_seed_contract import (
    SEED_CONTRACT_VERSION,
    SEED_IDENTITY_FIELDS,
    derive_search_seed,
    stable_hash,
    search_seed_context,
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
