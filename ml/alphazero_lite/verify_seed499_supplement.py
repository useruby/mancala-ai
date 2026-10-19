"""Read-only semantic supplement to the seed499 publication verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed488_strength_evaluation as seed488
from ml.alphazero_lite.seed499_budget_analysis import analyze, load_rows
from ml.alphazero_lite.seed499_budget_comparison import HISTORICAL, OUT, _sha
from ml.alphazero_lite.verify_seed499_publication import verify as verify_publication
from ml.alphazero_lite.verify_seed488_strength_evaluation import _replay_game

_ANALYSIS_FIELDS = (
    "winner",
    "game_length",
    "normal_puct_time_ms",
    "exact_root_solve_time_ms",
)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _audit_dependencies(root: Path, registration: dict[str, Any]) -> dict[str, Any]:
    historical = root / HISTORICAL
    receipt = json.loads((historical / "execution-receipt.json").read_text())
    _require(
        json.loads((historical / "publication-receipt.json").read_text()).get(
            "execution-receipt.json"
        )
        == _sha(historical / "execution-receipt.json"),
        "historical_execution_receipt_binding",
    )
    freeze_path = historical / "execution-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    _require(
        receipt.get("execution-freeze.json") == _sha(freeze_path)
        and freeze.get("registration_sha256")
        == registration["models"]["seed488_registration_sha256"],
        "historical_execution_freeze_binding",
    )
    builder = "ml/alphazero_lite/seed488_strength_evaluation.py"
    historically_frozen_builder = freeze.get("source_sha256", {}).get(builder)
    current_builder = _sha(root / builder)
    identities = registration["runtime"]["identities_sha256"]
    historical_registration = json.loads((historical / "registration.json").read_text())
    historical_identities = historical_registration["runtime"]["identities_sha256"]
    rules = "ml/alphazero_lite/kalah_rules.py"
    arena = "ml/alphazero_lite/arena.py"
    for path in (arena, rules):
        _require(
            historical_identities.get(path) == identities.get(path)
            and _sha(root / path) == identities[path],
            f"dependency_identity:{path}",
        )
    return {
        "builder": {
            "identity": builder,
            "seed488_prospective_sha256": historically_frozen_builder,
            "current_checkout_sha256": current_builder,
            "seed499_prospective_binding": False,
            "finding": (
                "The seed488 execution freeze binds an earlier builder digest; "
                "that digest differs from current checkout bytes, and the seed499 "
                "registration does not separately bind this imported module. "
                "The builder identity used by seed499 execution is unauthenticated."
            ),
        },
        "arena": {
            "identity": arena,
            "sha256": identities[arena],
            "prospective_binding": "seed488 and seed499 pre-execution runtime registrations",
        },
        "rules": {
            "identity": rules,
            "sha256": identities[rules],
            "prospective_binding": "seed488 and seed499 pre-execution runtime registrations",
        },
        "unbound_execution_identity": (
            "No process transcript or executable/environment binary digest authenticates "
            "the runtime process; archived reports omit the requested move cap, so "
            "the actually executed max-moves option cannot be established. Registered "
            "Python/NumPy version strings are not a process identity."
        ),
    }


def _check_frozen_command(
    root: Path,
    index: int,
    opening: dict[str, Any],
    identity: dict[str, Any],
    historical_registration: dict[str, Any],
    registration: dict[str, Any],
    opening_path: Path,
    report_path: Path,
    game_path: Path,
) -> None:
    command = seed488.build_arena_command(
        historical_registration,
        root / OUT / "runtime-artifacts",
        opening_path,
        report_path,
        game_path,
        identity,
    )
    for flag in ("--challenger-simulations", "--current-simulations"):
        command[command.index(flag) + 1] = str(
            registration["runtime"]["simulations_per_side"]
        )
    args = dict(
        zip(
            command[command.index("--challenger") :][::2],
            command[command.index("--challenger") + 1 :][::2],
        )
    )
    expected = {
        "--games": str(registration["games"] // registration["opening_suite"]["count"]),
        "--games-per-opening": str(
            registration["games"] // registration["opening_suite"]["count"]
        ),
        "--workers": str(registration["runtime"]["workers"]),
        "--challenger-simulations": str(
            registration["runtime"]["simulations_per_side"]
        ),
        "--current-simulations": str(registration["runtime"]["simulations_per_side"]),
        "--c-puct": str(registration["runtime"]["c_puct"]),
        "--seed-contract": "azlite_eval_seed_v2",
        "--random-opening-plies": str(
            registration["runtime"]["search_options"]["random_opening_plies"]
        ),
        "--max-moves": str(registration["runtime"]["max_moves"]),
        "--seed": str(488 + index),
        "--suite-sha256": identity["batch_suite_sha256"],
    }
    _require(
        all(args.get(flag) == value for flag, value in expected.items()),
        f"frozen_command_contract:{index}",
    )
    search_options = seed488.arena_search_overrides(historical_registration)
    for flag in ("--challenger-search-options-json", "--current-search-options-json"):
        _require(
            json.loads(args[flag]) == search_options,
            f"frozen_command_search_options:{index}",
        )


def _validate_batch(
    root: Path,
    index: int,
    opening: dict[str, Any],
    pair: list[dict[str, Any]],
    registration: dict[str, Any],
    historical_registration: dict[str, Any],
    *,
    check_runtime_identities: bool = True,
) -> list[dict[str, Any]]:
    out = root / OUT
    batch = out / "batches"
    opening_path = batch / f"opening-{index:04d}.jsonl"
    game_path = batch / f"games-{index:04d}.jsonl"
    report_path = batch / f"report-{index:04d}.json"
    opening_bytes = seed488.canonical_json(opening) + b"\n"
    _require(opening_path.read_bytes() == opening_bytes, f"opening_bytes:{index}")
    opening_digest = _sha(opening_path)
    identity = seed488.batch_opening_identity(
        historical_registration, opening, opening_path
    )
    _check_frozen_command(
        root,
        index,
        opening,
        identity,
        historical_registration,
        registration,
        opening_path,
        report_path,
        game_path,
    )
    runtime = registration["runtime"]
    expected_options = seed488.arena_effective_search_options(historical_registration)
    artifacts = json.loads((out / "runtime-artifacts.json").read_text())
    expected_artifacts = json.loads(
        (root / HISTORICAL / "runtime-artifacts.json").read_text()
    )
    if check_runtime_identities:
        _require(artifacts == expected_artifacts, "runtime_artifact_manifest_mismatch")
    policy_path = root / "model-artifact/current/search_policy.json"
    policy = json.loads(policy_path.read_text())
    games = load_rows(game_path)
    report = json.loads(report_path.read_text())
    notes = report.get("notes", {})
    profile = notes.get("search_profile", {})
    identities = runtime["identities_sha256"]

    _require(len(pair) == 2 and len(games) == 2, f"batch_pair_count:{index}")
    _require(
        {int(row["F_player"]) for row in pair} == {0, 1}
        and {int(game["challenger_player"]) for game in games} == {0, 1},
        f"batch_seats:{index}",
    )
    _require(
        len({row["opening_state_hash"] for row in pair}) == 1
        and pair[0]["opening_state_hash"] == opening["state_hash"],
        f"ledger_opening_association:{index}",
    )
    _require(
        notes.get("suite_sha256") == identity["batch_suite_sha256"]
        and identity["parent_suite_sha256"]
        == historical_registration["opening_suite"]["opening_suite_sha256"]
        and identity["parent_entry_state_hash"] == opening["state_hash"],
        f"suite_association:{index}",
    )

    search_checks = {
        "challenger_simulations": runtime["simulations_per_side"],
        "current_simulations": runtime["simulations_per_side"],
        "workers_requested": runtime["workers"],
        "workers_used": runtime["workers"],
        "base_seed": 488 + index,
        "seed": 488 + index,
        "seed_contract": "azlite_eval_seed_v2",
        "random_opening_plies": runtime["search_options"]["random_opening_plies"],
    }
    for field, expected in search_checks.items():
        _require(notes.get(field) == expected, f"report_{field}:{index}")
    _require(
        profile.get("c_puct") == runtime["c_puct"]
        and profile.get("search_options") == expected_options
        and notes.get("search_options") == expected_options,
        f"search_options:{index}",
    )
    _require(
        profile.get("version") == "v1"
        and profile.get("kind") == "arena_eval"
        and profile.get("player_mode") == "puct"
        and profile.get("exact_root_solver_max_solved_seeds") == 16
        and profile.get("exact_solve_value_mode") == "wdl"
        and profile.get("exact_solve_semantics") == "network_priors_exact_leaf_value"
        and profile.get("challenger_runtime_policy_inherited") is False,
        f"search_profile_contract:{index}",
    )
    _require(
        report.get("schema") == "arena_v1"
        and report.get("games_played") == 2
        and all(int(game["game_length"]) <= runtime["max_moves"] for game in games),
        f"report_game_budget:{index}",
    )

    # Runtime files and copied sidecar artifacts are compared to frozen
    # registration records, not inferred from their current contents.
    if check_runtime_identities:
        for path, digest in identities.items():
            _require(_sha(root / path) == digest, f"runtime_identity:{path}")
    _require(
        profile.get("runtime_search_policy_sha256") == _sha(policy_path)
        and profile.get("exact_root_tablebase_sha256") == policy["tablebase"]["sha256"]
        and profile.get("exact_root_native_probe_sha256")
        == policy["native_probe"]["sha256"],
        f"report_runtime_identity:{index}",
    )
    _require(
        profile.get("exact_root_solve_enabled") is policy["exact_root_enabled"]
        and profile.get("exact_root_solve_threshold") == policy["exact_root_threshold"]
        and profile.get("exact_root_objective") == policy["exact_root_objective"]
        and profile.get("exact_root_tie_rule") == policy["exact_root_tie_rule"]
        and profile.get("exact_leaf_solve_mode") == policy["exact_leaf_solve"]
        and profile.get("exact_solve_stone_threshold") is None,
        f"exact_root_contract:{index}",
    )
    _require(
        profile.get("exact_root_solver") == policy["solver_implementation_identity"]
        and profile.get("runtime_search_policy_mode") == policy["mode"],
        f"exact_root_implementation:{index}",
    )
    if check_runtime_identities:
        for arm, files in artifacts.items():
            for name, digest in files.items():
                _require(
                    _sha(out / "runtime-artifacts" / arm / name) == digest,
                    f"artifact_identity:{arm}/{name}",
                )

    totals = {"wins": 0, "losses": 0, "draws": 0}
    for row in pair:
        position = int(row["batch_game_position"])
        _require(position in (0, 1), f"batch_position:{index}")
        game = games[position]
        _require(
            int(game["challenger_player"]) == int(row["F_player"])
            and int(game["game_index"]) == position
            and int(game["opening_index"]) == 0,
            f"ledger_game_association:{index}:{position}",
        )
        _require(
            row.get("batch_game_sha256") == _sha(game_path)
            and row.get("batch_report_sha256") == _sha(report_path)
            and row.get("batch_opening_sha256") == opening_digest,
            f"batch_receipt:{index}:{position}",
        )
        _require(
            row.get("opening_prefix_moves") == opening["prefix_moves"]
            and row.get("opening_applied_prefix_length")
            == len(opening["prefix_moves"]),
            f"ledger_opening_prefix:{index}:{position}",
        )
        _replay_game(opening, game, int(row["F_player"]))
        for field in _ANALYSIS_FIELDS:
            _require(
                row.get(field) == game.get(field),
                f"ledger_game_mismatch:{index}:{position}:{field}",
            )
        _require(
            row.get("winner") in ("challenger", "current", "draw"),
            f"winner_label:{index}:{position}",
        )
        if row["winner"] == "challenger":
            totals["wins"] += 1
        elif row["winner"] == "current":
            totals["losses"] += 1
        else:
            totals["draws"] += 1
    for name, value in totals.items():
        _require(report.get(name) == value, f"report_totals:{index}:{name}")
    expected_score = (totals["wins"] + totals["draws"] * 0.5) / 2
    _require(report.get("score") == expected_score, f"report_score:{index}")
    return games


def verify(root: Path) -> dict[str, Any]:
    """Run the frozen publication check, then add independent semantics."""
    root = root.resolve()
    publication = verify_publication(root)
    out = root / OUT
    registration = json.loads((out / "registration.json").read_text())
    historical_registration = json.loads(
        (root / HISTORICAL / "registration.json").read_text()
    )
    expected_artifacts = json.loads(
        (root / HISTORICAL / "runtime-artifacts.json").read_text()
    )
    actual_artifacts = json.loads((out / "runtime-artifacts.json").read_text())
    _require(
        actual_artifacts == expected_artifacts, "runtime_artifact_manifest_mismatch"
    )
    for arm, files in expected_artifacts.items():
        for name, digest in files.items():
            _require(
                _sha(out / "runtime-artifacts" / arm / name) == digest,
                f"artifact_identity:{arm}/{name}",
            )
    for path, digest in registration["runtime"]["identities_sha256"].items():
        _require(_sha(root / path) == digest, f"runtime_identity:{path}")
    openings = load_rows(out / "openings.jsonl")
    rows = load_rows(out / "outcomes.jsonl")
    by_opening: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_opening.setdefault(int(row["opening_index"]), []).append(row)
    for index, opening in enumerate(openings):
        _validate_batch(
            root,
            index,
            opening,
            by_opening[index],
            registration,
            historical_registration,
            check_runtime_identities=False,
        )
    recomputed = analyze(rows, load_rows(root / HISTORICAL / "outcomes.jsonl"))
    _require(
        recomputed == json.loads((out / "analysis.json").read_text()),
        "supplement_analysis_mismatch",
    )
    receipt_path = out / "verification-supplement-receipt.json"
    supplement_receipt = json.loads(receipt_path.read_text())
    supplemental_files = {
        "supplement": root
        / "docs/data/seed499-budget-comparison/verification-supplement.md",
        "verifier": Path(__file__).resolve(),
        "tests": root / "ml/alphazero_lite/test_seed499_supplement.py",
    }
    for key, path in supplemental_files.items():
        _require(
            _sha(path) == supplement_receipt.get("sha256", {}).get(key),
            f"supplement_receipt_hash:{key}",
        )
    dependency_audit = _audit_dependencies(root, registration)
    return {
        "status": "verified_complete_with_semantic_supplement",
        "publication": publication,
        "batches_checked": len(openings),
        "ledger_games_reconciled": len(rows),
        "analysis": recomputed,
        "dependency_audit": dependency_audit,
        "execution_provenance_limitation": (
            "The exact imported seed488 command-builder identity for seed499 was "
            "not prospectively authenticated: the seed488 freeze binds an earlier "
            "digest that differs from current checkout, while the seed499 freeze "
            "does not bind the imported module. Arena and rules identities were "
            "prospectively authenticated in the seed488 registration and seed499 "
            "runtime registration. Archived outputs are not a trusted process "
            "transcript and cannot prove which builder/runtime bytes the execution "
            "process loaded or the requested move-cap option. No retrospective freeze "
            "or provenance claim is made."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
