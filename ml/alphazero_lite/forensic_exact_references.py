"""Exact-aware forensic reference v2 construction and validation.

This module deliberately consumes historical labels before attempting any native
work.  It never substitutes an MCTS move for an unresolved exact root.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ml.alphazero_lite.forensic_suite import load_suite


SCHEMA = "azlite_forensic_references_v2"
ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_LABELS = (
    (
        "pr291_292",
        ROOT / "docs/data/alphazero-lite-uniform1200-exact-neighborhood-audit.json",
        "cohort",
        "oracle_status",
    ),
    (
        "tier19",
        ROOT
        / "docs/data/alphazero-lite-uniform1200-tier19-unresolved-feasibility.json",
        "rows",
        "status",
    ),
    (
        "tier20",
        ROOT
        / "docs/data/alphazero-lite-uniform1200-tier20-unresolved-feasibility.json",
        "rows",
        "status",
    ),
    (
        "tier21",
        ROOT / "docs/data/alphazero-lite-uniform1200-tier21-gate.json",
        "rows",
        "status",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(row: dict[str, Any]) -> str:
    return str(row.get("canonical_state_key", row.get("canonical_state", "")))


def historical_labels() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Return successful labels using the frozen PR #291--#296 precedence."""
    labels: dict[str, dict[str, Any]] = {}
    input_shas: dict[str, str] = {}
    for source, path, collection, status_key in HISTORICAL_LABELS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        input_shas[str(path.relative_to(ROOT))] = sha256_file(path)
        for raw in payload[collection]:
            if raw.get(status_key) not in {"exact_solved", "exact"}:
                continue
            key = _canonical(raw)
            if not key or key in labels:
                continue
            labels[key] = {**raw, "_source": source}
    return labels, input_shas


def exact_regret(row: dict[str, Any], selected_move: int | None) -> float | None:
    """Return root-player regret from native player-0 margins, or unavailable."""
    if row.get("exact_status") != "exact_solved" or selected_move is None:
        return None
    values = {int(key): int(value) for key, value in row["exact_action_values"].items()}
    if int(selected_move) not in values:
        return None
    if int(row["state"]["current_player"]) == 0:
        return float(max(values.values()) - values[int(selected_move)])
    return float(values[int(selected_move)] - min(values.values()))


def exact_row(
    position: Any,
    label: dict[str, Any] | None,
    *,
    native_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = native_result or label
    row: dict[str, Any] = {
        "id": position.id,
        "canonical_state": position.canonical_key,
        "state": position.state,
        "legal_moves": list(position.legal_moves),
        "oracle_kind": "exact",
        "exact_status": "unresolved",
        "exact_root_value": None,
        "exact_action_values": None,
        "exact_optimal_actions": None,
        "oracle_provenance": {
            "source": None,
            "native_probe_sha256": None,
            "tablebase_sha256": None,
            "tablebase_tier": None,
        },
    }
    if (
        result is None
        or result.get("status", result.get("oracle_status")) != "exact_solved"
    ):
        if result is not None:
            row["oracle_provenance"]["source"] = result.get("_source", "native_tier21")
            row["unresolved_reason"] = result.get(
                "failure_reason", result.get("status")
            )
        return row
    values = {
        str(int(action)): int(value)
        for action, value in result["exact_action_values"].items()
    }
    row.update(
        {
            "exact_status": "exact_solved",
            "exact_root_value": float(result["exact_root_value"]),
            "exact_action_values": dict(
                sorted(values.items(), key=lambda item: int(item[0]))
            ),
            "exact_optimal_actions": sorted(
                int(action) for action in result["exact_optimal_actions"]
            ),
            "oracle_provenance": {
                "source": result.get("_source", "native_tier21"),
                "native_probe_sha256": result.get("probe_sha256"),
                "tablebase_sha256": result.get("tablebase_sha256"),
                "tablebase_tier": result.get("tablebase_tier"),
            },
        }
    )
    return row


def build_artifact(
    suite_path: Path, native_results: Iterable[dict[str, Any]] = ()
) -> dict[str, Any]:
    labels, input_shas = historical_labels()
    native_by_key = {_canonical(row): row for row in native_results}
    rows = [
        exact_row(
            position,
            labels.get(position.canonical_key),
            native_result=native_by_key.get(position.canonical_key),
        )
        for position in load_suite(suite_path)
    ]
    return {
        "schema": SCHEMA,
        "suite_path": str(suite_path),
        "input_sha256": input_shas,
        "precedence": [item[0] for item in HISTORICAL_LABELS],
        "rows": rows,
    }


def validate_v2(payload: dict[str, Any], suite_path: Path) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append(f"reference artifact schema must be {SCHEMA}")
    suite = {position.canonical_key: position for position in load_suite(suite_path)}
    seen: set[str] = set()
    for index, row in enumerate(payload.get("rows", [])):
        key = str(row.get("canonical_state", ""))
        if key in seen:
            errors.append(f"duplicate canonical_state in reference artifact: {key}")
        seen.add(key)
        position = suite.get(key)
        if position is None or row.get("state") != getattr(position, "state", None):
            errors.append(f"row {index} does not match suite canonical state")
            continue
        if row.get("legal_moves") != list(position.legal_moves):
            errors.append(f"row {row.get('id', index)} legal_moves do not match suite")
        solved = row.get("exact_status") == "exact_solved"
        if solved:
            values = row.get("exact_action_values")
            optimal = row.get("exact_optimal_actions")
            if not isinstance(values, dict) or set(map(int, values)) != set(
                position.legal_moves
            ):
                errors.append(
                    f"row {row.get('id', index)} exact action values do not cover legal moves"
                )
            if (
                not isinstance(optimal, list)
                or not optimal
                or not set(optimal) <= set(position.legal_moves)
            ):
                errors.append(
                    f"row {row.get('id', index)} exact optimal actions are invalid"
                )
        elif row.get("exact_status") != "unresolved":
            errors.append(f"row {row.get('id', index)} has invalid exact_status")
        elif any(
            row.get(field) is not None
            for field in (
                "exact_root_value",
                "exact_action_values",
                "exact_optimal_actions",
            )
        ):
            errors.append(
                f"row {row.get('id', index)} unresolved exact row must not manufacture labels"
            )
    if set(suite) != seen:
        errors.append("reference artifact does not cover every suite row")
    return errors
