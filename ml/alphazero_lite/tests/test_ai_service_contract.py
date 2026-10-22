"""Portable, executable API v1 contract tests."""

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from ml.alphazero_lite.serving import (
    MoveRequest,
    MoveResponse,
    State,
    canonical_state_bytes,
    from_kalah_game,
    state_sha256,
    to_kalah_game,
    validate_response,
    validate_state,
)

ROOT = Path(__file__).parents[3]
CONTRACT = json.loads((ROOT / "docs/ai-service/api-v1.json").read_text())
FIXTURES = json.loads((ROOT / "docs/ai-service/fixtures/v1.json").read_text())
SCHEMAS = CONTRACT["components"]["schemas"]


def schema_validator(name):
    """JSON Schema validation intentionally has standard JSON numeric semantics."""
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    registry = Registry().with_resource(
        "https://contract.invalid/api-v1.json",
        Resource.from_contents(CONTRACT, default_specification=DRAFT202012),
    )
    schema = {
        "$ref": f"https://contract.invalid/api-v1.json#/components/schemas/{name}"
    }
    return Draft202012Validator(schema, registry=registry)


class AiServiceContractTest(unittest.TestCase):
    def test_examples_validate_with_standard_schema_and_strict_models(self):
        self.assertEqual(CONTRACT["openapi"], "3.1.0")
        request_validator = schema_validator("MoveRequest")
        response_validator = schema_validator("MoveResponse")
        for raw in FIXTURES["valid_requests"]:
            request_validator.validate(raw)
            MoveRequest.model_validate(raw)
            validate_state(raw["state"])
        for raw in FIXTURES["valid_responses"]:
            response_validator.validate(raw)
            MoveResponse.model_validate(raw)

    def test_standard_integer_semantics_differ_from_runtime_strictness(self):
        raw = dict(FIXTURES["valid_requests"][0], seed=1.0)
        schema_validator("MoveRequest").validate(raw)
        with self.assertRaises(ValidationError):
            MoveRequest.model_validate(raw)
        for bad in (True, "1", 1.0, -1, 2**32):
            with self.subTest(value=bad), self.assertRaises(ValidationError):
                MoveRequest.model_validate(
                    dict(FIXTURES["valid_requests"][0], seed=bad)
                )

    def test_state_semantics_and_invalid_fixture_states(self):
        for case in FIXTURES["invalid_states"]:
            with (
                self.subTest(case=case["id"]),
                self.assertRaises((ValueError, ValidationError)),
            ):
                validate_state(case["state"])
        for bad in (True, "0", 1.0, -1, 2):
            raw = dict(FIXTURES["valid_requests"][0]["state"], current_player=bad)
            with self.assertRaises((ValueError, ValidationError)):
                validate_state(raw)

    def test_canonical_bytes_vectors_and_dictionary_order(self):
        for vector in FIXTURES["canonical_state_vectors"]:
            state = vector["state"]
            self.assertEqual(
                canonical_state_bytes(state).decode(), vector["canonical_utf8"]
            )
            self.assertEqual(state_sha256(state), vector["sha256"])
            reordered = dict(reversed(list(state.items())))
            self.assertEqual(
                canonical_state_bytes(reordered), canonical_state_bytes(state)
            )

    def test_conversion_is_fixed_orientation_and_defensive(self):
        raw = FIXTURES["valid_requests"][1]["state"]
        game = to_kalah_game(raw)
        self.assertEqual(game.to_state()["player_pits"], raw["player0_pits"])
        self.assertEqual(game.to_state()["opponent_pits"], raw["player1_pits"])
        self.assertEqual(
            from_kalah_game(game).model_dump(), State.model_validate(raw).model_dump()
        )
        game.pits[0] = 99
        self.assertEqual(raw["player0_pits"][0], 4)
        self.assertEqual(to_kalah_game(raw).pits[0], 4)

    def test_contextual_response_validation(self):
        requests = FIXTURES["valid_requests"]
        responses = FIXTURES["valid_responses"]
        for req, res in zip(requests, responses, strict=False):
            if req["request_id"] == "fixture-terminal":
                continue
            validate_response(req, res)
        req = requests[0]
        res = responses[0]
        for key, value in (
            ("request_id", "wrong"),
            ("game_revision", 8),
            ("state_sha256", "0" * 64),
        ):
            changed = dict(res, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_response(req, changed)
        illegal = dict(res, result={"kind": "move", "move": 4})
        with self.assertRaises(ValueError):
            validate_response(req, illegal)
        self.assertEqual(state_sha256(req["state"]), res["state_sha256"])

    def test_terminal_state_requires_zero_simulations_and_consistent_reason(self):
        terminal_request = {
            "request_id": "terminal",
            "game_revision": 0,
            "state": FIXTURES["terminal_state"],
            "profile": "balanced@v1",
            "seed": 0,
        }
        base = FIXTURES["valid_responses"][3]
        response = dict(
            base,
            request_id="terminal",
            game_revision=0,
            state_sha256=state_sha256(terminal_request["state"]),
        )
        validate_response(terminal_request, response)
        for change in (
            {"completed_simulations": 1},
            {"termination_reason": "completed"},
            {"result": {"kind": "move", "move": 0}},
        ):
            with (
                self.subTest(change=change),
                self.assertRaises((ValueError, ValidationError)),
            ):
                validate_response(terminal_request, dict(response, **change))


if __name__ == "__main__":
    unittest.main()
