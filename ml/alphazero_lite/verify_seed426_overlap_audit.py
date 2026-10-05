"""Read-only, standard-library verifier for the seed426 overlap publication."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from ml.alphazero_lite.seed426_overlap_analysis import (  # noqa: E402
    ACCOUNTING_DEFINITIONS,
    DECISION_RULES,
    IDENTITY_DEFINITIONS,
    audit,
    sha256,
)


def _decode(encoded: list[object]) -> dict[str, object]:
    if len(encoded) != 27:
        raise ValueError("encoder_identity_length_mismatch")

    def stones(start: int, count: int) -> list[int]:
        values = []
        for i in range(start, start + count):
            value = float(encoded[i]) * 48
            rounded = round(value)
            if abs(value - rounded) > 1e-5 or rounded < 0:
                raise ValueError("encoder_identity_decode_mismatch")
            values.append(int(rounded))
        return values

    player = round(float(encoded[14]))
    if player not in (0, 1) or abs(float(encoded[14]) - player) > 1e-5:
        raise ValueError("encoder_identity_player_mismatch")
    return {
        "player_pits": stones(0, 6),
        "opponent_pits": stones(6, 6),
        "player_store": stones(12, 1)[0],
        "opponent_store": stones(13, 1)[0],
        "current_player": int(player),
    }


def _canonical(state: dict[str, object]) -> str:
    identity = (
        tuple(state["player_pits"]),
        tuple(state["opponent_pits"]),
        state["player_store"],
        state["opponent_store"],
        state["current_player"],
    )
    return json.dumps(identity, separators=(",", ":"))


def _f32(values: list[object]) -> bytes:
    return struct.pack("<" + "f" * len(values), *(float(value) for value in values))


def verify(root: Path) -> None:
    data = root / "docs/data/seed426-canonical-overlap"
    manifest_path = data / "manifest.json"
    result_path = data / "results.json"
    evidence_path = data / "row-accounting.jsonl.gz"
    for path, reason in (
        (manifest_path, "missing_manifest"),
        (result_path, "missing_results"),
        (evidence_path, "missing_row_accounting"),
    ):
        if not path.is_file():
            raise ValueError(reason)
    manifest = json.loads(manifest_path.read_text())
    results = json.loads(result_path.read_text())
    if manifest.get("identity_definitions") != IDENTITY_DEFINITIONS:
        raise ValueError("manifest_identity_definition_mismatch")
    if manifest.get("accounting") != ACCOUNTING_DEFINITIONS:
        raise ValueError("manifest_accounting_definition_mismatch")
    if manifest.get("decision_rules") != DECISION_RULES:
        raise ValueError("manifest_decision_rule_mismatch")
    if not manifest.get("chronology", {}).get("amendment"):
        raise ValueError("manifest_chronology_missing")
    registration_path = (
        root / "docs/data/seed416-policy-target-softening/registration-v3.json"
    )
    if not registration_path.is_file():
        raise ValueError("missing_registration")
    registration_bytes = registration_path.read_bytes()
    registration = json.loads(registration_bytes)
    if sha256(registration_bytes) != manifest["registrations"]["seed416-v3"]:
        raise ValueError("registration_hash_mismatch")
    split_path = root / registration["training"]["source_row_split"]["path"]
    if not split_path.is_file():
        raise ValueError("missing_frozen_split")
    split_raw = split_path.read_bytes()
    if sha256(split_raw) != registration["training"]["source_row_split"]["sha256"]:
        raise ValueError("registered_split_hash_mismatch")
    if sha256(split_raw) != manifest["registrations"]["source_split"]:
        raise ValueError("manifest_split_hash_mismatch")
    split = json.loads(gzip.decompress(split_raw))
    train_positions = split["train_positions"]
    valid_positions = split["validation_positions"]
    if len(set(train_positions)) != len(train_positions) or len(
        set(valid_positions)
    ) != len(valid_positions):
        raise ValueError("split_duplicate_position")
    if set(train_positions) & set(valid_positions):
        raise ValueError("split_partition_overlap")
    int64_hash = hashlib.sha256()
    int64_hash.update(
        b"".join(struct.pack("<q", int(value)) for value in train_positions)
    )
    if (
        int64_hash.hexdigest()
        != registration["training"]["source_row_split"]["train_positions_sha256"]
    ):
        raise ValueError("train_position_identity_mismatch")
    int64_hash = hashlib.sha256()
    int64_hash.update(
        b"".join(struct.pack("<q", int(value)) for value in valid_positions)
    )
    if (
        int64_hash.hexdigest()
        != registration["training"]["source_row_split"]["validation_positions_sha256"]
    ):
        raise ValueError("validation_position_identity_mismatch")

    source_specs = registration["replays"]
    if [spec["weight"] for spec in source_specs] != [1, 4, 1, 8, 4]:
        raise ValueError("registered_weights_mismatch")
    if [
        (spec["name"], spec["sha256"], spec["weight"], spec["value_target_mode"])
        for spec in source_specs
    ] != [
        (spec["name"], spec["sha256"], spec["weight"], spec["value_target_mode"])
        for spec in manifest["sources"]
    ]:
        raise ValueError("manifest_source_registration_mismatch")
    if any(
        spec.get("policy_target_mode") != "sharpened"
        or spec.get("eligibility") != "no exclusion buckets"
        for spec in manifest["sources"]
    ):
        raise ValueError("manifest_loader_contract_mismatch")
    if (
        manifest["registrations"]["frozen_train"]
        != registration["source_hashes"]["ml/alphazero_lite/train.py"]
    ):
        raise ValueError("frozen_loader_registration_hash_mismatch")
    if (
        manifest["registrations"]["kalah_v3_encoder"]
        != registration["source_hashes"]["ml/alphazero_lite/self_play.py"]
    ):
        raise ValueError("encoder_registration_hash_mismatch")

    expected: list[dict[str, object]] = []
    target_checks = {}
    loaded_x_digest = hashlib.sha256()
    loaded_policy_digest = hashlib.sha256()
    loaded_value_digest = hashlib.sha256()
    compact = 0
    weighted = 0
    for spec in source_specs:
        source_path = data / "sources" / f"{spec['name']}.jsonl.gz"
        if not source_path.is_file():
            raise ValueError(f"missing_replay_snapshot:{spec['name']}")
        raw = gzip.decompress(source_path.read_bytes())
        if sha256(raw) != spec["sha256"]:
            raise ValueError(f"replay_snapshot_hash_mismatch:{spec['name']}")
        rows = []
        policy_digest = hashlib.sha256()
        value_digest = hashlib.sha256()
        for raw_line, line in enumerate(raw.splitlines(), 1):
            row = json.loads(line)
            policy_mode = row.get(
                "policy_target_actual_mode", row.get("policy_target_mode")
            )
            if policy_mode not in {
                "sharpened",
                "exact_root_one_hot",
                "exact_root_optimal_set_uniform",
            }:
                raise ValueError(
                    f"target_mode_declaration_mismatch:{spec['name']}:{raw_line}"
                )
            value_mode = row.get("value_target_mode")
            if value_mode is None:
                if spec["value_target_mode"] != "default":
                    raise ValueError(
                        f"target_mode_declaration_mismatch:{spec['name']}:{raw_line}"
                    )
            elif value_mode != spec["value_target_mode"]:
                raise ValueError(
                    f"target_mode_declaration_mismatch:{spec['name']}:{raw_line}"
                )
            encoded = row["state"]
            state = _decode(encoded)
            input_bytes = _f32(encoded)
            if not isinstance(row.get("policy"), list) or len(row["policy"]) != 6:
                raise ValueError(f"target_shape_mismatch:{spec['name']}:{raw_line}")
            policy_values = [float(value) for value in row["policy"]]
            value_value = float(row["value"])
            if not all(
                math.isfinite(value) for value in policy_values
            ) or not math.isfinite(value_value):
                raise ValueError(f"target_nonfinite:{spec['name']}:{raw_line}")
            if not -1.0 <= value_value <= 1.0:
                raise ValueError(f"target_value_out_of_range:{spec['name']}:{raw_line}")
            policy_digest.update(_f32(policy_values))
            value_digest.update(_f32([value_value]))
            loaded_x_digest.update(input_bytes)
            loaded_policy_digest.update(_f32(policy_values))
            loaded_value_digest.update(_f32([value_value]))
            # The encoder ID is independent: decode the full base state, then
            # require registered input bytes to round-trip to the same bytes.
            canonical = _canonical(state)
            active = sum(state["player_pits"]) + sum(state["opponent_pits"])
            rows.append(
                {
                    "source": spec["name"],
                    "raw_line": raw_line,
                    "compact_row": compact,
                    "active_stones": active,
                    "canonical_identity": canonical,
                    "input_identity": input_bytes.hex(),
                }
            )
            compact += 1
        target_checks[spec["name"]] = {
            "eligible_rows": len(rows),
            "policy_float32_sha256": policy_digest.hexdigest(),
            "value_float32_sha256": value_digest.hexdigest(),
        }
        expected.extend(rows)

    train_set, valid_set = set(train_positions), set(valid_positions)
    if train_set | valid_set != set(
        range(sum(len(s) for s in [train_positions, valid_positions]))
    ):
        raise ValueError("split_partition_incomplete")
    by_source = {
        spec["name"]: [row for row in expected if row["source"] == spec["name"]]
        for spec in source_specs
    }
    expanded = []
    for spec in source_specs:
        for copy in range(spec["weight"]):
            for row in by_source[spec["name"]]:
                expanded.append(
                    {
                        **row,
                        "copy": copy,
                        "weighted_position": weighted,
                        "partition": "train" if weighted in train_set else "validation",
                        "weight": 1,
                    }
                )
                weighted += 1
    source_row_sets = split["train_source_rows"], split["validation_source_rows"]
    if set(source_row_sets[0]) & set(source_row_sets[1]):
        raise ValueError("source_row_partition_overlap")
    if set(source_row_sets[0]) | set(source_row_sets[1]) != set(range(compact)):
        raise ValueError("source_row_partition_incomplete")
    mapped_train = {
        row["compact_row"] for row in expanded if row["partition"] == "train"
    }
    mapped_valid = {
        row["compact_row"] for row in expanded if row["partition"] == "validation"
    }
    if mapped_train != set(source_row_sets[0]) or mapped_valid != set(
        source_row_sets[1]
    ):
        raise ValueError("source_row_membership_mismatch")

    with gzip.open(evidence_path, "rt", encoding="utf-8") as handle:
        published = [json.loads(line) for line in handle]
    if published != expanded:
        raise ValueError("row_mapping_mismatch")
    census = audit(expanded)
    if census != results["identity_census"]:
        raise ValueError("published_accounting_mismatch")
    if (
        results["weighted_positions"] != weighted
        or results["eligible_source_rows"] != compact
    ):
        raise ValueError("published_count_mismatch")
    source_accounting = {}
    for spec in source_specs:
        source_rows = [row for row in expected if row["source"] == spec["name"]]
        source_positions = [row for row in expanded if row["source"] == spec["name"]]
        train_source_rows = {
            row["compact_row"]
            for row in source_positions
            if row["partition"] == "train"
        }
        valid_source_rows = {
            row["compact_row"]
            for row in source_positions
            if row["partition"] == "validation"
        }
        source_accounting[spec["name"]] = {
            "weight": spec["weight"],
            "eligible_source_rows": len(source_rows),
            "weighted_positions": len(source_positions),
            "train_source_rows": len(train_source_rows),
            "validation_source_rows": len(valid_source_rows),
            "train_weighted_positions": sum(
                row["partition"] == "train" for row in source_positions
            ),
            "validation_weighted_positions": sum(
                row["partition"] == "validation" for row in source_positions
            ),
        }
    if source_accounting != results["source_accounting"]:
        raise ValueError("source_partition_accounting_mismatch")
    if target_checks != results["loader_parity"]["source_targets"]:
        raise ValueError("loaded_target_accounting_mismatch")
    loader_parity = results["loader_parity"]
    if loader_parity["loaded_x_shape"] != [compact, 27]:
        raise ValueError("loaded_state_shape_mismatch")
    if (
        loader_parity["loaded_x_float32_sha256"] != loaded_x_digest.hexdigest()
        or loader_parity["loaded_policy_float32_sha256"]
        != loaded_policy_digest.hexdigest()
        or loader_parity["loaded_value_float32_sha256"]
        != loaded_value_digest.hexdigest()
    ):
        raise ValueError("loaded_target_accounting_mismatch")
    if results["eligibility"] != {
        "frozen_call_exclude_buckets": None,
        "raw_jsonl_lines": compact,
        "eligible_compact_rows": compact,
        "skipped_raw_lines": [],
        "rule": "the frozen load_jsonl_replay call supplies no exclusion buckets; every parsed raw row must validate and load",
    }:
        raise ValueError("eligibility_accounting_mismatch")
    if not math.isclose(
        sum(
            pattern["shared_validation_positions"]
            for pattern in census["canonical"]["additive_membership_patterns"]
        ),
        census["canonical"]["validation_overlap_denominators"]["weighted_positions"][
            "numerator"
        ],
    ):
        raise ValueError("source_attribution_not_additive")
    source_totals = census["canonical"]["additive_source_attribution"].values()
    if (
        sum(item["validation_positions"] for item in source_totals)
        != census["canonical"]["validation_overlap_denominators"]["weighted_positions"][
            "numerator"
        ]
    ):
        raise ValueError("source_attribution_not_additive")
    if (
        sum(item["identities"] for item in source_totals)
        != census["canonical"]["intersection_unique"]
    ):
        raise ValueError("source_identity_attribution_not_additive")
    expected_decision = (
        "canonical_state_overlap_detected"
        if census["canonical"]["intersection_unique"]
        else "canonical_state_disjoint"
    )
    if results["decision"] != expected_decision:
        raise ValueError("decision_rule_mismatch")
    high_fraction = census["canonical"]["buckets"][">32"][
        "validation_overlap_denominators"
    ]["weighted_positions"]["fraction"]
    expected_recommendation = (
        "separately_authorized_validation_metric_correction_using_identity_disjoint_holdout"
        if high_fraction >= 0.05
        else "close_diagnostic_without_training_or_replay_changes"
    )
    if results["recommendation"] != expected_recommendation:
        raise ValueError("recommendation_rule_mismatch")
    for relative, expected_hash in manifest["execution_source_snapshots"].items():
        snapshot = data / relative
        if not snapshot.is_file() or sha256(snapshot.read_bytes()) != expected_hash:
            raise ValueError(f"execution_source_snapshot_mismatch:{relative}")
    for name, expected_hash in manifest["execution_sources"].items():
        source = (
            root
            / "ml/alphazero_lite"
            / (
                "fresh_p1_adapter_teacher_audit.py"
                if name == "state_decoder"
                else f"{name}.py"
            )
        )
        if name == "frozen_train":
            source = root / "ml/alphazero_lite/train.py"
        elif name == "kalah_v3_encoder":
            source = root / "ml/alphazero_lite/self_play.py"
        elif name == "analysis":
            source = root / "ml/alphazero_lite/seed426_overlap_analysis.py"
        elif name == "runner":
            source = root / "ml/alphazero_lite/run_seed426_overlap_audit.py"
        elif name == "verifier":
            source = root / "ml/alphazero_lite/verify_seed426_overlap_audit.py"
        if not source.is_file() or sha256(source.read_bytes()) != expected_hash:
            raise ValueError(f"execution_source_hash_mismatch:{name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    verify(parser.parse_args().root)
