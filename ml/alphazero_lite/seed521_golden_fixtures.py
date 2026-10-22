"""Extract and verify archived seed521 evaluator golden fixtures (no inference)."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed522-golden-fixtures"
ORIENTATION = {
    "player_pits": "fixed player-0 pits, indices 0..5 in sowing order",
    "opponent_pits": "fixed player-1 pits, indices 0..5 in sowing order",
    "player_store": "fixed player-0 store",
    "opponent_store": "fixed player-1 store",
    "current_player": "the acting side; it does not change field orientation",
    "policy_indices": "indices 0..5 are relative to current_player; index 0 is that side's first pit",
}
OUTPUT_ENCODING = (
    "policy_f32_hex is six little-endian IEEE-754 float32 values; "
    "value_f64_hex is one little-endian IEEE-754 float64"
)
SOURCES = {
    "seed514_cohort": "docs/data/seed514-operational-benchmark-v2/cohort.json",
    "seed515_provenance": "docs/data/seed515-independent-verification/seed515-provenance-supplement.json",
    "seed518_outputs": "docs/data/seed518-selective-head/parity-outputs.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build(root: Path = ROOT, output: Path | None = None) -> dict[str, Any]:
    """Join only archived states/provenance/output bytes and publish fixture bytes."""
    output = output or root / "docs/data/seed521-golden-fixtures"
    cohort = _json(root / SOURCES["seed514_cohort"])
    provenance = _json(root / SOURCES["seed515_provenance"])
    outputs = _json(root / SOURCES["seed518_outputs"])
    if not (len(cohort) == len(provenance) == len(outputs) == 64):
        raise ValueError("seed521_source_case_count")
    by_state = {item["state_sha256"]: item for item in provenance}
    cases = []
    for index, (state, saved) in enumerate(zip(cohort, outputs, strict=True)):
        identity = state["state_sha256"]
        source = by_state.get(identity)
        if source is None or saved.get("state_sha256") != identity:
            raise ValueError(f"seed521_state_join:{index}")
        archived = saved["evaluator_outputs"]
        reference, selective = archived["reference_P"], archived["selective_P"]
        if reference != selective:
            raise ValueError(f"seed521_output_disagreement:{identity}")
        cases.append(
            {
                "case_id": f"cohort-{index:02d}",
                "kind": "archived_nonterminal",
                "state_sha256": identity,
                "state": state["state"],
                "cohort_bucket": state["bucket"],
                "selection_order_sha256": state["selection_order_sha256"],
                "source": {
                    key: source[key]
                    for key in (
                        "batch_relative_path",
                        "batch_sha256",
                        "record_index",
                        "game_index",
                        "opening_index",
                        "continuation_ply",
                        "total_ply",
                        "matching_occurrences",
                    )
                },
                "policy_f32_hex": reference["policy_f32_hex"],
                "value_f64_hex": reference["value_f64_hex"],
            }
        )
    for player in (0, 1):
        state = {
            "current_player": player,
            "player_pits": [0] * 6,
            "opponent_pits": [0] * 6,
            "player_store": 48,
            "opponent_store": 0,
        }
        cases.append(
            {
                "case_id": f"terminal-player-{player}",
                "kind": "explicit_terminal_semantics",
                "terminal_player": player,
                "state": state,
                "policy_f32_hex": "00000000" * 6,
                "value_f64_hex": "0000000000000000",
            }
        )
    fixture_bytes = (
        json.dumps(
            {
                "schema": "seed522-golden-fixtures-v2",
                "case_count": 66,
                "field_orientation": ORIENTATION,
                "output_encoding": OUTPUT_ENCODING,
                "cases": cases,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode()
    if output.exists():
        raise ValueError(f"seed521_output_exists:{output}")
    output.mkdir(parents=True)
    fixture_path = output / "fixtures.json"
    fixture_path.write_bytes(fixture_bytes)
    index = {
        "schema": "seed522-golden-fixture-index-v2",
        "sources": {
            name: {"path": relative, "sha256": sha256(root / relative)}
            for name, relative in SOURCES.items()
        },
        "fixtures": {
            "path": "fixtures.json",
            "sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        },
    }
    (output / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index


def verify(root: Path = ROOT, fixture_dir: Path | None = None) -> None:
    """Verify source binding, complete frozen order, distinct states and outputs."""
    fixture_dir = fixture_dir or root / "docs/data/seed522-golden-fixtures"
    index = _json(fixture_dir / "index.json")
    if index.get("schema") != "seed522-golden-fixture-index-v2":
        raise ValueError("seed522_index_schema")
    if set(index.get("sources", {})) != set(SOURCES):
        raise ValueError("seed522_index_sources")
    for name, relative in SOURCES.items():
        binding = index["sources"][name]
        if binding["path"] != relative or sha256(root / relative) != binding["sha256"]:
            raise ValueError(f"seed521_source_hash:{name}")
    if index.get("fixtures", {}).get("path") != "fixtures.json":
        raise ValueError("seed522_fixture_path")
    fixture_path = fixture_dir / "fixtures.json"
    if sha256(fixture_path) != index["fixtures"]["sha256"]:
        raise ValueError("seed521_fixture_hash")
    data = _json(fixture_path)
    cases = data.get("cases")
    if data.get("schema") != "seed522-golden-fixtures-v2":
        raise ValueError("seed522_fixture_schema")
    if data.get("field_orientation") != ORIENTATION:
        raise ValueError("seed522_orientation")
    if not isinstance(cases, list) or len(cases) != 66 or data.get("case_count") != 66:
        raise ValueError("seed522_case_count")
    ids = [case.get("case_id") for case in cases]
    if (
        len(set(ids)) != 66
        or ids[:64] != [f"cohort-{i:02d}" for i in range(64)]
        or ids[64:] != ["terminal-player-0", "terminal-player-1"]
    ):
        raise ValueError("seed522_case_id_order_or_duplicate")
    if len({case.get("state_sha256") for case in cases[:64]}) != 64:
        raise ValueError("seed522_state_duplicate")
    if data.get("output_encoding") != OUTPUT_ENCODING:
        raise ValueError("seed522_output_encoding")
    cohort = _json(root / SOURCES["seed514_cohort"])
    provenance = {
        row["state_sha256"]: row for row in _json(root / SOURCES["seed515_provenance"])
    }
    outputs = {
        row["state_sha256"]: row["evaluator_outputs"]
        for row in _json(root / SOURCES["seed518_outputs"])
    }
    # Each provenance locator points at an authenticated archived batch, and
    # each published field must equal the deterministic reference locator.
    batch_hashes: dict[str, str] = {}
    from .kalah_rules import KalahGame

    for row in provenance.values():
        rel = row["batch_relative_path"]
        digest = row["batch_sha256"]
        if rel not in batch_hashes:
            batch_path = root / rel
            if sha256(batch_path) != digest:
                raise ValueError("seed522_provenance_batch_hash")
            batch_hashes[rel] = digest
    for case, source_state in zip(cases[:64], cohort, strict=True):
        identity = source_state["state_sha256"]
        if (
            case.get("state_sha256") != identity
            or case.get("state") != source_state["state"]
            or case.get("cohort_bucket") != source_state["bucket"]
            or case.get("selection_order_sha256")
            != source_state["selection_order_sha256"]
        ):
            raise ValueError("seed521_fixture_archived_state_mismatch")
        evidence = provenance.get(identity)
        expected_locators = (
            "batch_relative_path",
            "batch_sha256",
            "record_index",
            "game_index",
            "opening_index",
            "continuation_ply",
            "total_ply",
            "matching_occurrences",
        )
        if (
            evidence is None
            or any(
                case.get("source", {}).get(key) != evidence.get(key)
                for key in expected_locators
            )
            or set(case.get("source", {})) != set(expected_locators)
        ):
            raise ValueError("seed522_provenance_locator_mismatch")
        locator = {
            key: evidence[key]
            for key in expected_locators
            if key != "matching_occurrences"
        }
        if locator not in evidence.get("occurrences", []):
            raise ValueError("seed522_provenance_occurrence_reference")
        if evidence.get("matching_occurrences") != len(evidence.get("occurrences", [])):
            raise ValueError("seed522_provenance_occurrence_count")
        if case.get("kind") != "archived_nonterminal":
            raise ValueError("seed522_case_kind")
        if KalahGame.from_state(case["state"]).over():
            raise ValueError("seed522_archived_case_terminal")
        if evidence.get("reference_semantics") != (
            "deterministic selection-order reference reconstructed from the frozen rule; "
            "it does not assert which duplicate trajectory occurrence was historically consumed"
        ):
            raise ValueError("seed522_provenance_semantics")
        archived = outputs.get(identity)
        if archived is None or archived["reference_P"] != archived["selective_P"]:
            raise ValueError("seed522_archived_output_disagreement")
        if (
            case.get("policy_f32_hex") != archived["reference_P"]["policy_f32_hex"]
            or case.get("value_f64_hex") != archived["reference_P"]["value_f64_hex"]
        ):
            raise ValueError("seed522_archived_output_mismatch")
    for case in cases[:64]:
        try:
            policy = bytes.fromhex(case["policy_f32_hex"])
            value = bytes.fromhex(case["value_f64_hex"])
            numbers = struct.unpack("<6f", policy) + struct.unpack("<d", value)
            if (
                len(policy) != 24
                or len(value) != 8
                or not all(float(x) == float(x) for x in numbers)
            ):
                raise ValueError
        except (TypeError, ValueError, struct.error) as exc:
            raise ValueError("seed522_output_encoding_invalid") from exc
    for player, case in enumerate(cases[64:]):
        state = case.get("state", {})
        if (
            case.get("kind") != "explicit_terminal_semantics"
            or case.get("terminal_player") != player
            or state
            != {
                "current_player": player,
                "player_pits": [0] * 6,
                "opponent_pits": [0] * 6,
                "player_store": 48,
                "opponent_store": 0,
            }
            or state.get("current_player") != player
            or KalahGame.from_state(state).possible_moves() != []
            or not KalahGame.from_state(state).over()
            or case.get("policy_f32_hex") != "00000000" * 6
            or case.get("value_f64_hex") != "0000000000000000"
        ):
            raise ValueError("seed521_terminal_fixture_invalid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    generate = subparsers.add_parser(
        "generate", help="write a new bundle (never overwrite)"
    )
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--repository-root", type=Path, default=ROOT)
    check = subparsers.add_parser("verify", help="verify an existing bundle read-only")
    check.add_argument(
        "--fixture-root", type=Path, default=ROOT / "docs/data/seed522-golden-fixtures"
    )
    check.add_argument("--repository-root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "generate":
        print(json.dumps(build(args.repository_root, args.output), sort_keys=True))
    else:
        verify(
            getattr(args, "repository_root", ROOT),
            getattr(args, "fixture_root", ROOT / "docs/data/seed522-golden-fixtures"),
        )
        print("seed522 golden fixtures verified")


if __name__ == "__main__":
    main()
