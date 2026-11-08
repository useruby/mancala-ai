"""Tests for the seed556 concurrent completion-order ledger correction."""

from __future__ import annotations

import pytest
import hashlib
import json
import sys
from pathlib import Path

from ml.alphazero_lite import seed556_exploratory_arena as protocol
from ml.alphazero_lite.verify_seed556_completion_order_correction import (
    normalize_completion_order_ledger,
    validate_report_profile,
)


def test_completion_order_normalizes_to_indexed_opening_seat_order() -> None:
    rows = [
        {"opening_index": opening, "game_position": seat, "value": opening * 2 + seat}
        for opening in range(protocol.OPENINGS)
        for seat in (0, 1)
    ]
    completion_order = rows[::2] + rows[1::2]
    assert normalize_completion_order_ledger(completion_order) == rows


def test_completion_order_rejects_receipt_rebound_duplicate_and_missing_seats() -> None:
    rows = [
        {"opening_index": opening, "game_position": seat}
        for opening in range(protocol.OPENINGS)
        for seat in (0, 1)
    ]
    duplicate = [*rows[:-1], rows[0]]
    with pytest.raises(ValueError, match="duplicate_ledger_opening_seat"):
        normalize_completion_order_ledger(duplicate)
    with pytest.raises(ValueError, match="coverage_mismatch"):
        normalize_completion_order_ledger(rows[:-1])


def _report_fixture(tmp_path: Path):
    root = tmp_path / "root"
    output = root / "screen"
    artifact_root = output / "runtime-artifacts"
    assignments = {
        "T512": artifact_root / "T512",
        "P-E4": artifact_root / "P-E4",
        "initializer": artifact_root / "initializer",
    }
    options = json.dumps(protocol.SEARCH_OPTIONS, sort_keys=True, separators=(",", ":"))
    probe = "model-artifact/runtime/probe"
    tablebase = str(tmp_path / "tablebase.kvtb")
    argv = [
        sys.executable,
        "-m",
        "ml.alphazero_lite.seed556_arena_entrypoint",
        "--challenger",
        str(assignments["T512"]),
        "--current",
        str(assignments["P-E4"]),
        "--challenger-policy-artifact",
        str(assignments["T512"]),
        "--challenger-value-artifact",
        str(assignments["initializer"]),
        "--current-policy-artifact",
        str(assignments["P-E4"]),
        "--current-value-artifact",
        str(assignments["initializer"]),
        "--challenger-simulations",
        "128",
        "--current-simulations",
        "128",
        "--workers",
        "1",
        "--games",
        "2",
        "--c-puct",
        "1.25",
        "--challenger-search-options-json",
        options,
        "--current-search-options-json",
        options,
        "--seed",
        "545",
        "--seed-contract",
        "azlite_eval_seed_v2",
        "--max-moves",
        "200",
        "--random-opening-plies",
        "0",
        "--exact-root-solve-threshold",
        "16",
        "--exact-root-native-probe",
        probe,
        "--exact-root-tablebase",
        tablebase,
    ]
    registration = {
        "execution_root": str(root),
        "execution_output": str(output),
        "runtime": {"executable": sys.executable},
    }
    p_sidecar = output / "runtime-artifacts/P-E4/search_policy.json"
    p_sidecar.parent.mkdir(parents=True)
    p_sidecar.write_text(json.dumps({"tablebase": {"path": "../tablebase.kvtb"}}))
    preflight = {
        "exact_root_native_probe": str((root / probe).resolve()),
        "exact_root_native_probe_sha256": "a" * 64,
        "exact_root_tablebase": tablebase,
        "exact_root_tablebase_sha256": "b" * 64,
    }
    component_files = {"P-E4": {"search_policy.json": "c" * 64}}
    profile = {
        "version": "v1",
        "kind": "arena_eval",
        "player_mode": "puct",
        "simulations": 128,
        "c_puct": 1.25,
        "search_options": protocol.SEARCH_OPTIONS,
        "challenger_simulations": 128,
        "current_simulations": 128,
        "exact_root_solve_enabled": True,
        "exact_root_solve_threshold": 16,
        "exact_root_solver": "native_kvtb_root_action_probe_v1",
        "exact_root_solver_max_solved_seeds": 16,
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve_mode": "disabled",
        "runtime_search_policy_mode": "puct_exact_root_hybrid",
        "exact_root_native_probe": str((root / probe).resolve()),
        "exact_root_native_probe_sha256": preflight["exact_root_native_probe_sha256"],
        "exact_root_tablebase": str(root / "../tablebase.kvtb"),
        "exact_root_tablebase_sha256": preflight["exact_root_tablebase_sha256"],
        "runtime_search_policy_sha256": component_files["P-E4"]["search_policy.json"],
        "challenger_runtime_policy_inherited": False,
    }
    profile["hash"] = hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    notes = {
        "search_profile": profile,
        "search_profile_hash": profile["hash"],
        "challenger_path": str(assignments["T512"]),
        "current_path": str(assignments["P-E4"]),
        "challenger_simulations": 128,
        "current_simulations": 128,
        "workers_requested": 1,
        "workers_used": 1,
        "seed": 545,
        "search_options": protocol.SEARCH_OPTIONS,
        "exact_root_native_probe_sha256": preflight["exact_root_native_probe_sha256"],
        "exact_root_tablebase_sha256": preflight["exact_root_tablebase_sha256"],
        "runtime_search_policy_sha256": component_files["P-E4"]["search_policy.json"],
    }
    report = {"notes": notes}
    command = {"argv": argv}
    return report, command, registration, preflight, component_files


def _rebind_report_profile(report: dict) -> None:
    profile = report["notes"]["search_profile"]
    unsigned = {key: value for key, value in profile.items() if key != "hash"}
    profile["hash"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report["notes"]["search_profile_hash"] = profile["hash"]


def test_report_profile_reconciles_search_and_asset_identities(tmp_path: Path) -> None:
    values = _report_fixture(tmp_path)
    validate_report_profile(*values, 0)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("simulations", 32, "profile_mismatch:0:simulations"),
        (
            "exact_root_native_probe_sha256",
            "d" * 64,
            "profile_mismatch:0:exact_root_native_probe_sha256",
        ),
    ],
)
def test_rebound_report_profile_mutations_are_rejected(
    tmp_path: Path, field: str, value, error: str
) -> None:
    report, *rest = _report_fixture(tmp_path)
    report["notes"]["search_profile"][field] = value
    if field == "exact_root_native_probe_sha256":
        report["notes"][field] = value
    _rebind_report_profile(report)
    with pytest.raises(ValueError, match=error):
        validate_report_profile(report, *rest, 0)
