"""Portable v1 contract checks; no model/runtime dependencies required."""

import hashlib
import json
import re
import unittest
from pathlib import Path

from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).parents[3]
CONTRACT = json.loads((ROOT / "docs/ai-service/api-v1.json").read_text())
FIXTURES = json.loads(
    (ROOT / "docs/ai-service/fixtures/v1.json").read_text(encoding="utf-8")
)
SCHEMAS = CONTRACT["components"]["schemas"]


def schema_valid(value, schema):
    """Small OpenAPI/JSON-Schema subset validator for schemas used here."""
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[1]
        return schema_valid(value, SCHEMAS[name])
    if "oneOf" in schema:
        return sum(schema_valid(value, option) for option in schema["oneOf"]) == 1
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return False
        if any(key not in value for key in schema.get("required", [])):
            return False
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            return False
        return all(
            key not in properties or schema_valid(item, properties[key])
            for key, item in value.items()
        ) and all(
            value.get(key) in prop["enum"]
            for key, prop in properties.items()
            if key in value and "enum" in prop
        )
    if kind == "array":
        return (
            isinstance(value, list)
            and len(value) >= schema.get("minItems", 0)
            and len(value) <= schema.get("maxItems", 2**53)
            and all(schema_valid(item, schema["items"]) for item in value)
        )
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        return (
            value >= schema.get("minimum", -(2**53))
            and value <= schema.get("maximum", 2**53 - 1)
            and ("enum" not in schema or value in schema["enum"])
        )
    if isinstance(kind, list):
        return any(schema_valid(value, {**schema, "type": entry}) for entry in kind)
    if kind == "string":
        return (
            isinstance(value, str)
            and len(value) >= schema.get("minLength", 0)
            and len(value) <= schema.get("maxLength", 2**53)
            and ("pattern" not in schema or re.fullmatch(schema["pattern"], value))
            and ("enum" not in schema or value in schema["enum"])
            and ("const" not in schema or value == schema["const"])
        )
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if "const" in schema:
        return value == schema["const"]
    return True


def valid_game_state(state):
    """Validate the semantic conservation and terminal representation rules."""
    if not schema_valid(state, SCHEMAS["State"]):
        return False
    stones = sum(state["player0_pits"] + state["player1_pits"])
    stones += state["player0_store"] + state["player1_store"]
    if stones != 48:
        return False
    terminal = all(
        stone == 0 for stone in state["player0_pits"] + state["player1_pits"]
    )
    current_pits = state[f"player{state['current_player']}_pits"]
    return (terminal and state["player0_store"] + state["player1_store"] == 48) or (
        not terminal and any(current_pits)
    )


class AiServiceContractTest(unittest.TestCase):
    def test_openapi_is_31_and_fixtures_validate_against_contract(self):
        self.assertEqual(CONTRACT["openapi"], "3.1.0")
        request_schema = SCHEMAS["MoveRequest"]
        response_schema = SCHEMAS["MoveResponse"]
        for request in FIXTURES["valid_requests"]:
            self.assertTrue(schema_valid(request, request_schema), request)
            self.assertTrue(valid_game_state(request["state"]), request)
        for response in FIXTURES["valid_responses"]:
            self.assertTrue(schema_valid(response, response_schema), response)

    def test_schema_rejects_bad_shapes_types_and_unknown_properties(self):
        valid = FIXTURES["valid_requests"][0]
        for change in (
            {"game_revision": True},
            {"extra": "not allowed"},
            {"seed": 1.0},
            {"state": {**valid["state"], "player0_pits": [1]}},
            {"profile": "balanced@v1", "search_options": {"deadline": 1}},
        ):
            with self.subTest(change=change):
                self.assertFalse(
                    schema_valid({**valid, **change}, SCHEMAS["MoveRequest"])
                )
        self.assertFalse(schema_valid(True, SCHEMAS["Seeds"]))

    def test_fixture_states_pass_or_fail_semantics_as_declared(self):
        self.assertTrue(valid_game_state(FIXTURES["valid_requests"][0]["state"]))
        self.assertTrue(valid_game_state(FIXTURES["valid_requests"][1]["state"]))
        self.assertTrue(valid_game_state(FIXTURES["terminal_state"]))
        for case in FIXTURES["invalid_states"]:
            with self.subTest(case=case["id"]):
                self.assertFalse(valid_game_state(case["state"]))

    def test_successful_move_examples_are_legal_under_repository_rules(self):
        for request, response in zip(
            FIXTURES["valid_requests"], FIXTURES["valid_responses"][:3], strict=True
        ):
            state = request["state"]
            self.assertTrue(valid_game_state(state))
            game = KalahGame(
                pits=state["player0_pits"] + state["player1_pits"],
                captured_seeds=[state["player0_store"], state["player1_store"]],
                current_player=state["current_player"],
            )
            move = response["result"]["move"]
            self.assertIn(move, game.possible_moves())
        extra_state = FIXTURES["valid_requests"][2]["state"]
        self.assertTrue(valid_game_state(extra_state))
        extra = KalahGame(
            pits=extra_state["player0_pits"] + extra_state["player1_pits"],
            captured_seeds=[extra_state["player0_store"], extra_state["player1_store"]],
            current_player=extra_state["current_player"],
        )
        self.assertTrue(extra.move(extra.pit_index(5)))
        self.assertEqual(extra.current_player, 0)
        self.assertEqual(extra.captured_seeds[0], 21)
        terminal = KalahGame(pits=[0] * 12, captured_seeds=[20, 28], current_player=0)
        self.assertTrue(terminal.over())

    def test_literal_canonical_bytes_and_independent_sha256_vectors(self):
        for vector in FIXTURES["canonical_state_vectors"]:
            state = vector["state"]
            canonical = (
                "{"
                + ",".join(
                    f"{json.dumps(key)}:{json.dumps(state[key], separators=(',', ':'))}"
                    for key in (
                        "player0_pits",
                        "player1_pits",
                        "player0_store",
                        "player1_store",
                        "current_player",
                    )
                )
                + "}"
            ).encode("utf-8")
            self.assertEqual(canonical, vector["canonical_utf8"].encode("utf-8"))
            self.assertEqual(hashlib.sha256(canonical).hexdigest(), vector["sha256"])

    def test_fixtures_cover_identity_mismatches_and_illegal_move(self):
        for request, response in zip(
            FIXTURES["valid_requests"], FIXTURES["valid_responses"][:3], strict=True
        ):
            self.assertEqual(request["request_id"], response["request_id"])
            self.assertEqual(request["game_revision"], response["game_revision"])
        for case in FIXTURES["invalid_results"]:
            if case["id"] == "illegal-empty-pit":
                state = FIXTURES["valid_requests"][case["request_state_index"]]["state"]
                game = KalahGame(
                    pits=state["player0_pits"] + state["player1_pits"],
                    captured_seeds=[state["player0_store"], state["player1_store"]],
                    current_player=state["current_player"],
                )
                self.assertNotIn(
                    case["response"]["result"]["move"], game.possible_moves()
                )
            else:
                if case["id"] == "incorrect-request-identity":
                    self.assertNotEqual(case["request_id"], case["response_request_id"])
                else:
                    self.assertEqual(case["request_id"], case["response_request_id"])
                self.assertNotEqual(case["request_revision"], case["response_revision"])


if __name__ == "__main__":
    unittest.main()
