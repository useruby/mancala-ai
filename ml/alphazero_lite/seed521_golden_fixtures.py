"""Extract and verify archived seed521 evaluator golden fixtures (no inference)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed521-golden-fixtures"
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
                "schema": "seed521-golden-fixtures-v1",
                "case_count": 66,
                "field_orientation": {
                    "player_pits": "six pits for current_player, indices 0..5 in sowing order",
                    "opponent_pits": "six pits for other player, indices 0..5 in sowing order",
                    "stores": "player_store belongs to current_player; opponent_store to other player",
                },
                "output_encoding": "policy_f32_hex is six little-endian IEEE-754 float32 values; value_f64_hex is one little-endian IEEE-754 float64",
                "cases": cases,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode()
    output.mkdir(parents=True, exist_ok=True)
    fixture_path = output / "fixtures.json"
    fixture_path.write_bytes(fixture_bytes)
    index = {
        "schema": "seed521-golden-fixture-index-v1",
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
    fixture_dir = fixture_dir or root / "docs/data/seed521-golden-fixtures"
    index = _json(fixture_dir / "index.json")
    for name, relative in SOURCES.items():
        binding = index["sources"][name]
        if binding["path"] != relative or sha256(root / relative) != binding["sha256"]:
            raise ValueError(f"seed521_source_hash:{name}")
    fixture_path = fixture_dir / index["fixtures"]["path"]
    if sha256(fixture_path) != index["fixtures"]["sha256"]:
        raise ValueError("seed521_fixture_hash")
    data = _json(fixture_path)
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != 66 or data.get("case_count") != 66:
        raise ValueError("seed521_fixture_case_count")
    ids = [case.get("case_id") for case in cases]
    if (
        len(set(ids)) != 66
        or ids[:64] != [f"cohort-{i:02d}" for i in range(64)]
        or ids[64:] != ["terminal-player-0", "terminal-player-1"]
    ):
        raise ValueError("seed521_fixture_order_or_duplicate")
    if len({case.get("state_sha256") for case in cases[:64]}) != 64:
        raise ValueError("seed521_fixture_state_duplicate")
    cohort = _json(root / SOURCES["seed514_cohort"])
    provenance = {
        row["state_sha256"]: row for row in _json(root / SOURCES["seed515_provenance"])
    }
    outputs = {
        row["state_sha256"]: row["evaluator_outputs"]
        for row in _json(root / SOURCES["seed518_outputs"])
    }
    for case, source_state in zip(cases[:64], cohort, strict=True):
        identity = source_state["state_sha256"]
        if (
            case.get("state_sha256") != identity
            or case.get("state") != source_state["state"]
        ):
            raise ValueError("seed521_fixture_archived_state_mismatch")
        evidence = provenance.get(identity)
        if (
            evidence is None
            or case.get("source", {}).get("batch_sha256") != evidence["batch_sha256"]
        ):
            raise ValueError("seed521_fixture_provenance_mismatch")
        archived = outputs.get(identity)
        if archived is None or archived["reference_P"] != archived["selective_P"]:
            raise ValueError("seed521_fixture_archived_output_disagreement")
        if (
            case.get("policy_f32_hex") != archived["reference_P"]["policy_f32_hex"]
            or case.get("value_f64_hex") != archived["reference_P"]["value_f64_hex"]
        ):
            raise ValueError("seed521_fixture_archived_output_mismatch")
    for case in cases[:64]:
        if (
            len(bytes.fromhex(case["policy_f32_hex"])) != 24
            or len(bytes.fromhex(case["value_f64_hex"])) != 8
        ):
            raise ValueError("seed521_fixture_output_width")
    for player, case in enumerate(cases[64:]):
        if (
            case.get("terminal_player") != player
            or case.get("policy_f32_hex") != "00000000" * 6
            or case.get("value_f64_hex") != "0000000000000000"
        ):
            raise ValueError("seed521_terminal_fixture_invalid")


if __name__ == "__main__":
    build()
    verify()
