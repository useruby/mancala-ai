import hashlib
import json
from collections.abc import Mapping


SCHEMA = "azlite_opening_cache_v1"
DEFAULT_OPENING_GATE = {"max_ply": 10, "min_stones_in_pits": 36}


def canonical_payload(state):
    canonical_payload = {
        "current_player": int(state["current_player"]),
        "player_pits": [int(value) for value in state["player_pits"]],
        "opponent_pits": [int(value) for value in state["opponent_pits"]],
        "player_store": int(state["player_store"]),
        "opponent_store": int(state["opponent_store"]),
    }
    return canonical_payload


def canonical_payload_json(state):
    encoded = json.dumps(canonical_payload(state), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return encoded


def canonical_key(state):
    encoded = canonical_payload_json(state)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def stones_in_pits(state):
    return sum(int(value) for value in state["player_pits"]) + sum(int(value) for value in state["opponent_pits"])


def state_qualifies_for_opening_cache(state, *, ply, opening_gate=None):
    gate = dict(DEFAULT_OPENING_GATE)
    if opening_gate:
        gate.update(opening_gate)

    if int(ply) > int(gate["max_ply"]):
        return False

    if stones_in_pits(state) < int(gate["min_stones_in_pits"]):
        return False

    return True


class OpeningCache:
    def __init__(self, payload):
        self.opening_gate = dict(payload.get("opening_gate", {}))
        self.search_profile = dict(payload.get("search_profile", {}))
        self._entries_by_key = dict(payload["entries"])

    def lookup(self, state, *, ply):
        if not state_qualifies_for_opening_cache(state, ply=ply, opening_gate=self.opening_gate):
            return None

        return self._entries_by_key.get(canonical_key(state))


def normalize_opening_gate(opening_gate):
    if opening_gate is None:
        return {}
    if not isinstance(opening_gate, Mapping):
        raise ValueError("Opening cache opening_gate must be a mapping")

    normalized_gate = dict(opening_gate)
    for field in DEFAULT_OPENING_GATE:
        if field not in normalized_gate:
            continue
        try:
            normalized_gate[field] = int(normalized_gate[field])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Opening cache opening_gate.{field} must be numeric") from error

    return normalized_gate


def load_opening_cache(payload):
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported opening cache schema: {payload.get('schema')}")

    entries_by_key = payload.get("entries")
    if not isinstance(entries_by_key, dict):
        raise ValueError("Opening cache entries must be a mapping keyed by canonical state")

    normalized_entries = {}
    for provided_key, entry in entries_by_key.items():
        state = entry["state"]
        hashed_key = canonical_key(state)
        legacy_key = canonical_payload_json(state)

        if provided_key not in (hashed_key, legacy_key):
            raise ValueError("Opening cache entry key does not match entry state")

        if hashed_key in normalized_entries:
            raise ValueError("Opening cache contains duplicate normalized keys")

        normalized_entries[hashed_key] = entry

    normalized_payload = dict(payload)
    normalized_payload["opening_gate"] = normalize_opening_gate(payload.get("opening_gate"))
    normalized_payload["entries"] = normalized_entries
    return OpeningCache(normalized_payload)
