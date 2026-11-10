"""Deterministic, commit/reveal protocol for a future opening-paired arena.

This module never launches games. Generation is parameterized by a supplied
candidate pool; production use must use the frozen rules-only generator.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import inspect
import platform
from pathlib import Path
from typing import Any
import zlib

from ml.alphazero_lite.seed482_fresh_confirmation import (
    canonical_identity,
    float32_identity,
)
from ml.alphazero_lite.kalah_rules import KalahGame

INITIAL_STATE = {
    "player_pits": [4] * 6,
    "opponent_pits": [4] * 6,
    "player_store": 0,
    "opponent_store": 0,
    "current_player": 0,
}
RULES_GENERATOR = "rules_only_uniform_prefix_v1"
RULES_ATTEMPT_LIMIT = 20_000
RULES_GENERATOR_SCHEMA = "rules_only_uniform_prefix_v1"

INVENTORY_REFERENCE_SCHEMA = "seed561_holdback_inventory_reference_v1"
INVENTORY_MANIFEST_SCHEMA = "seed561_historical_exclusion_inventory_v1"
# Updated to the committed source-manifest digest when the v1 indexes publish.
TRUSTED_HISTORICAL_INVENTORY_MANIFEST_SHA256 = (
    "946fc52c75fa2e6c0499811fcb97ff773ea00ec6ec9b6d1bfb3ec1c2466a95ad"
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identities(row: dict[str, Any]) -> tuple[str, str]:
    """Derive both identities from rule state and the registered Kalah encoder."""
    state = row["state"]
    return canonical_identity(state), float32_identity(state)


def validate_inventory(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    result = {"canonical": set(), "float32": set()}
    for row in rows:
        can, f32 = identities(row)
        if can in result["canonical"] or f32 in result["float32"]:
            raise ValueError("duplicate_exclusion_identity")
        result["canonical"].add(can)
        result["float32"].add(f32)
    return result


def validate_identity_sets(
    identities_by_scheme: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Validate and copy precomputed identity sets for both required schemes."""
    if set(identities_by_scheme) != {"canonical", "float32"}:
        raise ValueError("inventory_identity_schemes_invalid")
    result: dict[str, set[str]] = {}
    for scheme in ("canonical", "float32"):
        values = identities_by_scheme[scheme]
        if not isinstance(values, set) or any(
            not isinstance(value, str) or len(value) != 64 for value in values
        ):
            raise ValueError(f"inventory_identity_set_invalid:{scheme}")
        result[scheme] = set(values)
    return result


def _bundle_file(bundle: Path, relative: str) -> Path:
    """Resolve a bundle-relative path while preventing traversal outside it."""
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("inventory_path_must_be_relative")
    root = bundle.resolve()
    resolved = (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("inventory_path_escapes_bundle")
    return resolved


def load_pinned_identity_indexes(
    bundle: Path,
    inventory_reference: dict[str, Any],
    trusted_manifest_sha256: str,
) -> dict[str, set[str]]:
    """Load both published indexes only when their manifest has an external pin.

    Gzip headers and compression settings are intentionally not identity
    evidence: each index is authenticated by the SHA-256 of its decompressed
    deterministic JSONL, whose digest is itself bound by the trusted manifest.
    """
    if inventory_reference.get("schema") != INVENTORY_REFERENCE_SCHEMA:
        raise ValueError("inventory_reference_schema_mismatch")
    if inventory_reference.get("manifest_sha256") != trusted_manifest_sha256:
        raise ValueError("untrusted_inventory_manifest")
    manifest_path = _bundle_file(
        bundle, str(inventory_reference.get("manifest_path", ""))
    )
    manifest_bytes = manifest_path.read_bytes()
    if sha256(manifest_bytes) != trusted_manifest_sha256:
        raise ValueError("inventory_manifest_bytes_changed")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != INVENTORY_MANIFEST_SCHEMA:
        raise ValueError("inventory_manifest_schema_mismatch")
    if manifest.get("identity_schemes") != [
        "canonical_state_sha256",
        "kalah_v3_float32_sha256",
    ]:
        raise ValueError("inventory_manifest_identity_schemes_invalid")
    expected_schemes = {"canonical", "kalah_v3_float32"}
    index_manifest = manifest.get("indexes")
    if not isinstance(index_manifest, dict) or set(index_manifest) != expected_schemes:
        raise ValueError("inventory_manifest_indexes_invalid")

    result: dict[str, set[str]] = {}
    for scheme in ("canonical", "kalah_v3_float32"):
        binding = index_manifest[scheme]
        index_path = _bundle_file(manifest_path.parent, str(binding.get("path", "")))
        identities: set[str] = set()
        occurrence_count = 0
        content_digest = hashlib.sha256()
        try:
            with gzip.open(index_path, "rb") as stream:
                for line in stream:
                    content_digest.update(line)
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    value = row.get("identity")
                    if (
                        not isinstance(value, str)
                        or len(value) != 64
                        or any(char not in "0123456789abcdef" for char in value)
                        or not isinstance(row.get("source"), str)
                        or "provenance" not in row
                    ):
                        raise ValueError("inventory_index_row_invalid")
                    identities.add(value)
                    occurrence_count += 1
        except (
            OSError,
            EOFError,
            zlib.error,
            json.JSONDecodeError,
            AttributeError,
        ) as exc:
            raise ValueError(f"inventory_index_jsonl_invalid:{scheme}") from exc
        if content_digest.hexdigest() != binding.get("sha256_jsonl"):
            raise ValueError(f"inventory_index_content_mismatch:{scheme}")
        if occurrence_count != binding.get("occurrence_rows"):
            raise ValueError(f"inventory_index_row_count_mismatch:{scheme}")
        if len(identities) != binding.get("distinct_identities"):
            raise ValueError(f"inventory_index_identity_count_mismatch:{scheme}")
        result["canonical" if scheme == "canonical" else "float32"] = identities
    return result


def build_suite(
    pool: list[dict[str, Any]],
    inventory: list[dict[str, Any]] | dict[str, set[str]],
    seed: int,
    size: int,
) -> list[dict[str, Any]]:
    import random

    excluded = (
        validate_identity_sets(inventory)
        if isinstance(inventory, dict)
        else validate_inventory(inventory)
    )
    rng = random.Random(seed)
    candidates = list(pool)
    rng.shuffle(candidates)
    chosen: list[dict[str, Any]] = []
    seen = {key: set(values) for key, values in excluded.items()}
    for row in candidates:
        can, f32 = identities(row)
        if can in seen["canonical"] or f32 in seen["float32"]:
            continue
        chosen.append(row)
        seen["canonical"].add(can)
        seen["float32"].add(f32)
        if len(chosen) == size:
            break
    if len(chosen) != size:
        raise ValueError("eligible_pool_shortfall")
    return chosen


def build_rules_only_suite(
    inventory: dict[str, set[str]],
    seed: int,
    size: int,
    *,
    attempt_limit: int = RULES_ATTEMPT_LIMIT,
) -> list[dict[str, Any]]:
    """Reproduce #511's uniform 4–12-ply legal-prefix sampling law.

    Python's ``random.Random`` MT19937 is seeded once; each attempt draws a
    length with ``randint(4, 12)`` and then ``choice`` from ascending legal
    relative pit indices. No retries are reseeded.
    """
    import random

    excluded = validate_identity_sets(inventory)
    rng = random.Random(seed)
    seen = {key: set(values) for key, values in excluded.items()}
    result: list[dict[str, Any]] = []
    for _ in range(attempt_limit):
        game = KalahGame.from_state(INITIAL_STATE)
        length = rng.randint(4, 12)
        absolute: list[int] = []
        relative: list[int] = []
        for _ply in range(length):
            legal = game.possible_moves()
            if not legal or game.over():
                break
            move = rng.choice(legal)
            relative.append(move)
            absolute_move = game.pit_index(move)
            absolute.append(absolute_move)
            if not game.move(absolute_move):
                raise ValueError("generated_prefix_move_rejected")
        if len(absolute) != length or game.over():
            continue
        state = game.to_state()
        can, f32 = canonical_identity(state), float32_identity(state)
        if can in seen["canonical"] or f32 in seen["float32"]:
            continue
        index = len(result)
        result.append(
            {
                "opening_index": index,
                "opening_contract": "arena_player_relative_v2",
                "prefix_moves": relative,
                "source_prefix_moves_absolute": absolute,
                "state": state,
                "state_hash": sha256(canonical_bytes(state)),
            }
        )
        seen["canonical"].add(can)
        seen["float32"].add(f32)
        if len(result) == size:
            return result
    raise ValueError(f"rules_only_opening_shortfall:{len(result)}")


def suite_bytes(suite: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in suite)


def commitment(
    suite: bytes, generator: dict[str, Any], inventory: dict[str, Any]
) -> str:
    public_generator = public_generator_spec(generator)
    return sha256(
        canonical_bytes(
            {
                "suite_sha256": sha256(suite),
                "generator": public_generator,
                "exclusion_inventory": inventory,
            }
        )
    )


def _source_sha256(function: Any) -> str:
    return sha256(inspect.getsource(function).encode())


def rules_generator_spec(
    size: int, attempt_limit: int = RULES_ATTEMPT_LIMIT
) -> dict[str, Any]:
    """Return the public, seed-free generator identity bound into commitments."""
    if size <= 0 or attempt_limit != RULES_ATTEMPT_LIMIT:
        raise ValueError("rules_generator_parameters_invalid")
    from ml.alphazero_lite.seed482_fresh_confirmation import (
        canonical_identity as canonical_fn,
        float32_identity as float32_fn,
    )
    from ml.alphazero_lite.self_play import encode_state
    from ml.alphazero_lite.input_encodings import feature_count_for
    import numpy as np

    rules_path = Path(inspect.getfile(KalahGame))
    encoder_path = Path(inspect.getfile(encode_state))
    encoding_contract_path = Path(inspect.getfile(feature_count_for))
    return {
        "name": RULES_GENERATOR,
        "schema": RULES_GENERATOR_SCHEMA,
        "size": size,
        "attempt_limit": attempt_limit,
        "rng": "python.random.Random/MT19937",
        "python_runtime": platform.python_version(),
        "numpy_runtime": np.__version__,
        "initial_state_sha256": sha256(canonical_bytes(INITIAL_STATE)),
        "generator_source_sha256": _source_sha256(build_rules_only_suite),
        "rules_source_sha256": sha256(rules_path.read_bytes()),
        "canonical_identity_source_sha256": _source_sha256(canonical_fn),
        "float32_identity_source_sha256": _source_sha256(float32_fn),
        "kalah_v3_encoder_source_sha256": _source_sha256(encode_state),
        "encoder_implementation_files_sha256": {
            "self_play.py": sha256(encoder_path.read_bytes()),
            "input_encodings.py": sha256(encoding_contract_path.read_bytes()),
        },
    }


def public_generator_spec(generator: dict[str, Any]) -> dict[str, Any]:
    """Remove only the revealed seed, preserving the exact committed spec."""
    return {key: value for key, value in generator.items() if key != "seed"}


def public_commitment_record(
    suite_digest: str, generator: dict[str, Any], inventory: dict[str, Any]
) -> dict[str, Any]:
    """Create the seed-free public commitment payload for a sealed suite."""
    if "seed" in generator:
        raise ValueError("public_generator_must_not_expose_seed")
    size = generator.get("size")
    attempt_limit = generator.get("attempt_limit")
    if not isinstance(size, int) or not isinstance(attempt_limit, int):
        raise ValueError("public_generator_spec_invalid")
    expected_generator = rules_generator_spec(size, attempt_limit)
    if generator != expected_generator:
        raise ValueError("public_generator_spec_invalid")
    bound = {
        "suite_sha256": suite_digest,
        "generator": generator,
        "exclusion_inventory": inventory,
    }
    return {
        "schema": "holdback_public_commitment_v1",
        **bound,
        "commitment": sha256(canonical_bytes(bound)),
    }


def verify_public_commitment_record(record: dict[str, Any]) -> None:
    """Validate a pre-reveal public record without opening private custody files."""
    if record.get("schema") != "holdback_public_commitment_v1":
        raise ValueError("public_commitment_schema_invalid")
    if "seed" in record.get("generator", {}):
        raise ValueError("public_commitment_seed_exposed")
    expected = public_commitment_record(
        record.get("suite_sha256", ""),
        record.get("generator", {}),
        record.get("exclusion_inventory", {}),
    )
    if expected != record:
        raise ValueError("public_commitment_record_mismatch")


def lifecycle_event(
    kind: str, payload: dict[str, Any], previous: str
) -> dict[str, Any]:
    event = {"kind": kind, "previous": previous, "payload": payload}
    event["event_sha256"] = sha256(canonical_bytes(event))
    return event


def event_log_bytes(events: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(event) + b"\n" for event in events)


def verify(
    bundle: Path,
    trusted_anchor_sha256: str,
    trusted_registry_sha256: str,
    *,
    anchor_name: str = "anchor.json",
    trusted_inventory_manifest_sha256: str = TRUSTED_HISTORICAL_INVENTORY_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Verify a revealed bundle against an independently retained event-log hash.

    The trusted anchor is delivered out-of-band by the commitment custodian; a
    hash stored only inside the bundle would not establish event order.
    """
    manifest = json.loads((bundle / "manifest.json").read_text())
    suite = (bundle / "suite.jsonl").read_bytes()
    if sha256(suite) != manifest["suite_sha256"]:
        raise ValueError("suite_bytes_changed")
    rows = [json.loads(line) for line in suite.splitlines() if line]
    if sha256(suite_bytes(rows)) != manifest["suite_sha256"]:
        raise ValueError("suite_order_or_encoding_changed")
    if len(rows) != manifest["generator"].get("size"):
        raise ValueError("suite_declared_size_mismatch")
    inventory_path = bundle / manifest["inventory_path"]
    inventory_bytes = inventory_path.read_bytes()
    if sha256(inventory_bytes) != manifest["inventory_sha256"]:
        raise ValueError("inventory_bytes_changed")
    inventory_doc = json.loads(inventory_bytes)
    if (
        commitment(suite, manifest["generator"], inventory_doc)
        != manifest["commitment"]
    ):
        raise ValueError("commitment_mismatch")
    if not trusted_inventory_manifest_sha256:
        raise ValueError("trusted_inventory_manifest_pin_required")
    excluded = load_pinned_identity_indexes(
        bundle, inventory_doc, trusted_inventory_manifest_sha256
    )
    generator = manifest["generator"]
    if generator.get("name") == RULES_GENERATOR:
        seed = generator.get("seed")
        size = generator.get("size")
        attempt_limit = generator.get("attempt_limit")
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or not isinstance(attempt_limit, int)
            or isinstance(attempt_limit, bool)
        ):
            raise ValueError("rules_generator_spec_invalid")
        expected_spec = rules_generator_spec(size, attempt_limit)
        if public_generator_spec(generator) != expected_spec:
            raise ValueError("rules_generator_spec_invalid")
        rebuilt = build_rules_only_suite(
            excluded,
            seed,
            generator["size"],
            attempt_limit=generator["attempt_limit"],
        )
    else:
        if generator.get("pool_sha256") != manifest["pool_sha256"]:
            raise ValueError("generator_pool_binding_mismatch")
        pool_bytes = (bundle / manifest["pool_path"]).read_bytes()
        if sha256(pool_bytes) != manifest["pool_sha256"]:
            raise ValueError("synthetic_pool_bytes_changed")
        pool = [json.loads(line) for line in pool_bytes.splitlines() if line]
        rebuilt = build_suite(pool, excluded, generator["seed"], generator["size"])
    if suite_bytes(rebuilt) != suite:
        raise ValueError("suite_rebuild_mismatch")
    seen = {"canonical": set(), "float32": set()}
    for row in rows:
        can, f32 = identities(row)
        if can in excluded["canonical"] or f32 in excluded["float32"]:
            raise ValueError("suite_exclusion_overlap")
        if can in seen["canonical"] or f32 in seen["float32"]:
            raise ValueError("suite_duplicate_identity")
        seen["canonical"].add(can)
        seen["float32"].add(f32)
    anchor_bytes = (bundle / anchor_name).read_bytes()
    if sha256(anchor_bytes) != trusted_anchor_sha256:
        raise ValueError("untrusted_lifecycle_anchor")
    anchor = json.loads(anchor_bytes)
    event_bytes = (bundle / anchor["lifecycle_path"]).read_bytes()
    if anchor.get("lifecycle_sha256") != sha256(event_bytes):
        raise ValueError("anchor_lifecycle_mismatch")
    events = [json.loads(line) for line in event_bytes.splitlines() if line]
    expected_kinds = ["commitment", "candidate_freeze", "reveal"]
    if len(events) == 4:
        expected_kinds.append("consumption")
    if [event["kind"] for event in events] != expected_kinds:
        raise ValueError("lifecycle_event_order")
    previous = "0" * 64
    for event in events:
        expected = lifecycle_event(event["kind"], event["payload"], previous)
        if event != expected:
            raise ValueError("lifecycle_chain_invalid")
        previous = event["event_sha256"]
    checkpoints = anchor.get("checkpoints", [])
    if len(checkpoints) != len(events) or any(
        point.get("event_sha256") != event["event_sha256"]
        for point, event in zip(checkpoints, events, strict=True)
    ):
        raise ValueError("anchor_checkpoint_mismatch")
    times = [
        dt.datetime.fromisoformat(point["anchored_at_utc"]) for point in checkpoints
    ]
    if any(time.utcoffset() != dt.timedelta(0) for time in times):
        raise ValueError("anchor_timestamp_not_utc")
    if any(left >= right for left, right in zip(times, times[1:])):
        raise ValueError("anchor_checkpoint_order")
    commit_event, freeze_event, reveal_event = [
        event["payload"] for event in events[:3]
    ]
    if commit_event.get("commitment") != manifest["commitment"]:
        raise ValueError("lifecycle_commitment_mismatch")
    freeze = freeze_event.get("freeze", {})
    if freeze_event.get("commitment") != manifest["commitment"] or not all(
        freeze.get(key)
        for key in (
            "candidate_artifact_sha256",
            "serving_p_artifact_sha256",
            "arena_settings",
            "analysis_rule",
        )
    ):
        raise ValueError("candidate_freeze_not_bound")
    if reveal_event.get("commitment") != manifest["commitment"]:
        raise ValueError("reveal_commitment_mismatch")
    if reveal_event.get("suite_sha256") != manifest["suite_sha256"]:
        raise ValueError("reveal_suite_mismatch")
    rules_mode = generator.get("name") == RULES_GENERATOR
    if reveal_event.get("seed") != generator["seed"] or (
        not rules_mode and reveal_event.get("pool_sha256") != manifest["pool_sha256"]
    ):
        raise ValueError("reveal_generator_inputs_mismatch")
    consumed = len(events) == 4
    if consumed and events[3]["payload"].get("commitment") != manifest["commitment"]:
        raise ValueError("consumption_commitment_mismatch")
    registry_bytes = (bundle / "custodian-state.json").read_bytes()
    if sha256(registry_bytes) != trusted_registry_sha256:
        raise ValueError("untrusted_custodian_registry")
    registry = json.loads(registry_bytes)
    if registry.get("commitment") != manifest["commitment"]:
        raise ValueError("custodian_commitment_mismatch")
    anchor_is_current = registry.get("latest_anchor_sha256") == trusted_anchor_sha256
    if anchor_is_current and registry.get("consumed") != consumed:
        raise ValueError("custodian_state_lifecycle_mismatch")
    currently_consumed = bool(registry.get("consumed"))
    return {
        "status": "verified",
        "held_back_eligible": anchor_is_current and not currently_consumed,
        "consumed": currently_consumed,
        "verified_lifecycle_consumed": consumed,
        "anchor_is_current": anchor_is_current,
        "openings": len(rows),
        "commitment": manifest["commitment"],
        "canonical_identities": len(seen["canonical"]),
        "float32_identities": len(seen["float32"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--trusted-anchor-sha256",
        required=True,
        help="SHA-256 of anchor.json obtained independently from its custodian",
    )
    parser.add_argument(
        "--trusted-registry-sha256",
        required=True,
        help="SHA-256 of current custodian-state.json obtained independently",
    )
    parser.add_argument(
        "--trusted-inventory-manifest-sha256",
        default=TRUSTED_HISTORICAL_INVENTORY_MANIFEST_SHA256,
        help="Independently trusted digest of the published historical inventory manifest",
    )
    parser.add_argument("--anchor-name", default="anchor.json")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                args.bundle,
                args.trusted_anchor_sha256,
                args.trusted_registry_sha256,
                anchor_name=args.anchor_name,
                trusted_inventory_manifest_sha256=(
                    args.trusted_inventory_manifest_sha256
                ),
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
