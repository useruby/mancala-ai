"""Read-only correction for unordered concurrent ledger append order.

The frozen 18-way executor appends each independently validated pair as it
finishes. Therefore the raw ledger order is completion order, while semantic
reconstruction is indexed by opening and seat. This verifier preserves and
checks raw ledger bytes without sorting or rewriting them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed556_exploratory_arena as protocol
from ml.alphazero_lite import verify_seed551_controlled_policy_strength as gamecheck
from ml.alphazero_lite import verify_seed552_c108_serving_p as prior_verify


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def normalize_completion_order_ledger(
    ledger: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Index the append-only ledger and reject duplicate/missing seat keys."""
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for row in ledger:
        key = (int(row["opening_index"]), int(row["game_position"]))
        if key in indexed:
            raise ValueError("seed556_duplicate_ledger_opening_seat")
        indexed[key] = row
    expected = {
        (opening, seat) for opening in range(protocol.OPENINGS) for seat in (0, 1)
    }
    if indexed.keys() != expected:
        raise ValueError("seed556_ledger_opening_seat_coverage_mismatch")
    return [
        indexed[(opening, seat)]
        for opening in range(protocol.OPENINGS)
        for seat in (0, 1)
    ]


def validate_report_profile(
    report: dict[str, Any],
    command: dict[str, Any],
    registration: dict[str, Any],
    policy_preflight: dict[str, Any],
    component_files: dict[str, dict[str, str]],
    output: Path,
    opening_index: int,
) -> None:
    """Bind reported search semantics and runtime asset identities to launch."""
    argv = command.get("argv", [])
    if (
        len(argv) < 3
        or argv[0] != registration.get("runtime", {}).get("executable")
        or argv[1:3] != ["-m", "ml.alphazero_lite.seed556_arena_entrypoint"]
    ):
        raise ValueError(f"seed556_report_runtime_command_mismatch:{opening_index}")

    def arg(flag: str) -> str:
        if flag not in argv or argv.index(flag) + 1 >= len(argv):
            raise ValueError(
                f"seed556_report_command_argument_missing:{opening_index}:{flag}"
            )
        return argv[argv.index(flag) + 1]

    challenger = Path(arg("--challenger-policy-artifact"))
    opponent = Path(arg("--current-policy-artifact"))
    challenger_evaluator = Path(arg("--challenger"))
    opponent_evaluator = Path(arg("--current"))
    challenger_value = Path(arg("--challenger-value-artifact"))
    opponent_value = Path(arg("--current-value-artifact"))
    seed = int(arg("--seed"))
    sims_t = int(arg("--challenger-simulations"))
    sims_p = int(arg("--current-simulations"))
    c_puct = float(arg("--c-puct"))
    options_t = json.loads(arg("--challenger-search-options-json"))
    options_p = json.loads(arg("--current-search-options-json"))
    probe = Path(arg("--exact-root-native-probe"))
    tablebase = Path(arg("--exact-root-tablebase"))
    threshold = int(arg("--exact-root-solve-threshold"))
    if (
        seed != protocol.SEED + opening_index
        or sims_t != protocol.SIMULATIONS
        or sims_p != protocol.SIMULATIONS
        or c_puct != 1.25
        or options_t != protocol.SEARCH_OPTIONS
        or options_p != protocol.SEARCH_OPTIONS
        or arg("--workers") != "1"
        or arg("--games") != "2"
        or arg("--max-moves") != "200"
        or arg("--random-opening-plies") != "0"
        or arg("--seed-contract") != "azlite_eval_seed_v2"
        or threshold != 16
        or challenger
        != Path(registration["execution_output"]) / "runtime-artifacts/T512"
        or opponent != Path(registration["execution_output"]) / "runtime-artifacts/P-E4"
        or challenger_evaluator != challenger
        or opponent_evaluator != opponent
        or challenger_value
        != Path(registration["execution_output"]) / "runtime-artifacts/initializer"
        or opponent_value != challenger_value
    ):
        raise ValueError(
            f"seed556_report_frozen_command_semantics_mismatch:{opening_index}"
        )

    profile = report.get("notes", {}).get("search_profile")
    notes = report.get("notes", {})
    if not isinstance(profile, dict):
        raise ValueError(f"seed556_report_search_profile_missing:{opening_index}")
    expected_probe = policy_preflight["exact_root_native_probe_sha256"]
    expected_tablebase = policy_preflight["exact_root_tablebase_sha256"]
    expected_sidecar = component_files["P-E4"]["search_policy.json"]
    policy_sidecar = json.loads(
        (output / "runtime-artifacts/P-E4/search_policy.json").read_text()
    )
    expected_profile_tablebase = str(
        Path(registration["execution_root"]) / policy_sidecar["tablebase"]["path"]
    )
    if (Path(registration["execution_root"]) / probe).resolve() != Path(
        policy_preflight["exact_root_native_probe"]
    ).resolve() or tablebase.resolve() != Path(
        policy_preflight["exact_root_tablebase"]
    ).resolve():
        raise ValueError(f"seed556_report_command_asset_path_mismatch:{opening_index}")
    profile_fields = {
        "version": "v1",
        "kind": "arena_eval",
        "player_mode": "puct",
        "simulations": protocol.SIMULATIONS,
        "c_puct": 1.25,
        "search_options": protocol.SEARCH_OPTIONS,
        "challenger_simulations": protocol.SIMULATIONS,
        "current_simulations": protocol.SIMULATIONS,
        "exact_root_solve_enabled": True,
        "exact_root_solve_threshold": 16,
        "exact_root_solver": "native_kvtb_root_action_probe_v1",
        "exact_root_solver_max_solved_seeds": 16,
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve_mode": "disabled",
        "runtime_search_policy_mode": "puct_exact_root_hybrid",
        "exact_root_native_probe": str(
            (Path(registration["execution_root"]) / probe).resolve()
        ),
        "exact_root_native_probe_sha256": expected_probe,
        "exact_root_tablebase": expected_profile_tablebase,
        "exact_root_tablebase_sha256": expected_tablebase,
        "runtime_search_policy_sha256": expected_sidecar,
        "challenger_runtime_policy_inherited": False,
    }
    for key, expected in profile_fields.items():
        if profile.get(key) != expected:
            raise ValueError(f"seed556_report_profile_mismatch:{opening_index}:{key}")
    if (
        notes.get("challenger_path") != str(challenger)
        or notes.get("current_path") != str(opponent)
        or notes.get("challenger_simulations") != protocol.SIMULATIONS
        or notes.get("current_simulations") != protocol.SIMULATIONS
        or notes.get("workers_requested") != 1
        or notes.get("workers_used") != 1
        or notes.get("seed") != seed
        or notes.get("search_options") != protocol.SEARCH_OPTIONS
        or notes.get("exact_root_native_probe_sha256") != expected_probe
        or notes.get("exact_root_tablebase_sha256") != expected_tablebase
        or notes.get("runtime_search_policy_sha256") != expected_sidecar
    ):
        raise ValueError(f"seed556_report_runtime_identity_mismatch:{opening_index}")
    unsigned = {key: value for key, value in profile.items() if key != "hash"}
    expected_profile_hash = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        profile.get("hash") != expected_profile_hash
        or notes.get("search_profile_hash") != expected_profile_hash
    ):
        raise ValueError(f"seed556_report_profile_hash_mismatch:{opening_index}")


def verify(
    root: Path,
    output: Path,
    *,
    materialized_tablebase: Path | None = None,
    native_probe: Path | None = None,
    require_published_receipt: bool = True,
) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    registration = json.loads((output / "registration.json").read_text())
    freeze = json.loads((output / "execution-freeze.json").read_text())
    if (
        freeze.get("registration_sha256")
        != protocol.sha256(output / "registration.json")
        or freeze.get("source_sha256") != registration.get("source_sha256")
        or freeze.get("command_preflight_sha256")
        != protocol.sha256(output / "command-preflight.jsonl")
        or protocol.sha256(output / "openings.jsonl") != protocol.OPENINGS_SHA256
    ):
        raise ValueError("seed556_completion_order_freeze_mismatch")
    published_hashes: dict[str, str] = {}
    published_receipt_path = output / "verification-receipt.json"
    published_verification = output / "verification.json"
    if require_published_receipt:
        published_receipt = json.loads(published_receipt_path.read_text())
        published_hashes = published_receipt.get("evidence_sha256", {})
        if (
            published_receipt.get("verification_sha256")
            != protocol.sha256(published_verification)
            or published_receipt.get("verified_games") != protocol.GAMES
            or published_receipt.get("verified_pairs") != protocol.OPENINGS
        ):
            raise ValueError("seed556_published_verification_receipt_mismatch")
        # The executor's final progress marker is written after the initial
        # verification receipt. Every underlying command/game/report/input asset,
        # including the search preflight, remains bound by the receipt.
        for relative, expected_sha256 in published_hashes.items():
            if relative in {
                "execution-progress.json",
                "physical-copy-verification.json",
                "verification.json",
                "verification-receipt.json",
            }:
                continue
            path = (output / relative).resolve()
            if output not in path.parents or protocol.sha256(path) != expected_sha256:
                raise ValueError(f"seed556_published_evidence_hash_mismatch:{relative}")
    for relative, expected_sha256 in registration.get("source_sha256", {}).items():
        snapshot = output / "execution-source-snapshots" / relative
        source = snapshot if snapshot.is_file() else root / relative
        if protocol.sha256(source) != expected_sha256:
            raise ValueError(f"seed556_frozen_source_rehash_mismatch:{relative}")
    for name, expected_files in registration.get("components", {}).items():
        component_dir = output / "runtime-artifacts" / name
        actual_files = {
            path.name: protocol.sha256(path)
            for path in sorted(component_dir.iterdir())
            if path.is_file()
        }
        if actual_files != expected_files:
            raise ValueError(f"seed556_exported_component_rehash_mismatch:{name}")
    if protocol.sha256(root / protocol.TABLEBASE_PATH) != registration.get(
        "tablebase_lfs_pointer_sha256"
    ):
        raise ValueError("seed556_tablebase_lfs_pointer_rehash_mismatch")
    policy_preflight = json.loads(
        (output / "runtime-policy-preflight.json").read_text()
    )
    registered_preflight_sha = registration.get("runtime_policy_preflight_sha256")
    receipt_preflight_sha = published_hashes.get("runtime-policy-preflight.json")
    actual_preflight_sha = protocol.sha256(output / "runtime-policy-preflight.json")
    if (
        registered_preflight_sha is not None
        and registered_preflight_sha != actual_preflight_sha
    ) or (require_published_receipt and receipt_preflight_sha != actual_preflight_sha):
        raise ValueError("seed556_runtime_policy_preflight_binding_mismatch")
    component_files = registration["components"]
    materialized_tablebase = materialized_tablebase or Path(
        policy_preflight["exact_root_tablebase"]
    )
    native_probe = native_probe or Path(policy_preflight["exact_root_native_probe"])
    if (
        protocol.sha256(materialized_tablebase)
        != registration["tablebase_materialized_sha256"]
        or materialized_tablebase.stat().st_size
        != registration["tablebase_materialized_size"]
    ):
        raise ValueError("seed556_materialized_tablebase_rehash_mismatch")
    if (
        protocol.sha256(native_probe)
        != policy_preflight["exact_root_native_probe_sha256"]
    ):
        raise ValueError("seed556_native_probe_rehash_mismatch")
    commands = _jsonl(output / "command-preflight.jsonl")
    ledger = _jsonl(output / "outcomes.jsonl")
    if len(commands) != protocol.OPENINGS or len(ledger) != protocol.GAMES:
        raise ValueError("seed556_completion_order_evidence_incomplete")
    raw_openings = (
        (root / protocol.OPENINGS_PATH).read_bytes().splitlines(keepends=True)
    )
    openings = gamecheck.replay_suite(root)
    reconstructed: list[dict[str, Any]] = []
    batch = output / "batches"
    registered_output = Path(registration["execution_output"])
    for index in range(protocol.OPENINGS):
        prefix_path = batch / f"opening-{index:04d}.jsonl"
        command_path = batch / f"command-{index:04d}.json"
        games_path = batch / f"games-{index:04d}.jsonl"
        report_path = batch / f"report-{index:04d}.json"
        completion_path = batch / f"completion-{index:04d}.json"
        if prefix_path.read_bytes() != raw_openings[index]:
            raise ValueError(f"seed556_indexed_opening_mismatch:{index}")
        command = json.loads(command_path.read_text())
        if (
            command != commands[index]
            or command.get("component_assignments") != protocol.ASSIGNMENTS
        ):
            raise ValueError(f"seed556_frozen_command_mismatch:{index}")
        argv = command["argv"]
        required = {
            "--challenger-policy-artifact": registered_output
            / "runtime-artifacts/T512",
            "--challenger-value-artifact": registered_output
            / "runtime-artifacts/initializer",
            "--current-policy-artifact": registered_output / "runtime-artifacts/P-E4",
            "--current-value-artifact": registered_output
            / "runtime-artifacts/initializer",
        }
        for flag, expected in required.items():
            if Path(argv[argv.index(flag) + 1]) != expected:
                raise ValueError(f"seed556_command_component_mismatch:{index}:{flag}")
        attempt_path = batch / f"attempt-{index:04d}-00.json"
        attempt = json.loads(attempt_path.read_text())
        if attempt.get("returncode") != 0:
            raise ValueError(f"seed556_attempt_failed:{index}")
        games = _jsonl(games_path)
        if len(games) != 2:
            raise ValueError(f"seed556_pair_game_count_mismatch:{index}")
        pair = []
        for seat, game in enumerate(games):
            gamecheck._validate_arena_local_opening_index(game)
            if (
                game.get("game_index") != seat
                or game.get("game_within_opening") != seat
                or game.get("challenger_player") != seat
            ):
                raise ValueError(f"seed556_seat_pairing_mismatch:{index}:{seat}")
            result, margin, capped = prior_verify._outcome(openings[index], game)
            result = result.removeprefix("capped_")
            outcome = {"C": "T512", "P": "P", "draw": "draw"}[result]
            pair.append(
                {
                    "opening_index": index,
                    "game_position": seat,
                    "outcome": outcome,
                    "margin": margin,
                    "capped_nonterminal": capped,
                    "games_sha256": protocol.sha256(games_path),
                    "report_sha256": protocol.sha256(report_path),
                    "command_sha256": protocol.sha256(command_path),
                }
            )
        report = json.loads(report_path.read_text())
        validate_report_profile(
            report,
            command,
            registration,
            policy_preflight,
            component_files,
            output,
            index,
        )
        if (
            report.get("schema") != "arena_v1"
            or report.get("games_played") != 2
            or report.get("wins") != sum(g.get("winner") == "challenger" for g in games)
            or report.get("losses") != sum(g.get("winner") == "current" for g in games)
            or report.get("draws") != sum(g.get("winner") == "draw" for g in games)
        ):
            raise ValueError(f"seed556_report_reconciliation_mismatch:{index}")
        completion = json.loads(completion_path.read_text())
        if (
            completion.get("opening_index") != index
            or completion.get("outcomes") != pair
            or completion.get("command_sha256") != protocol.sha256(command_path)
        ):
            raise ValueError(f"seed556_completion_receipt_mismatch:{index}")
        reconstructed.extend(pair)
    ordered_ledger = normalize_completion_order_ledger(ledger)
    if ordered_ledger != reconstructed:
        raise ValueError("seed556_outcome_ledger_reconstruction_mismatch")
    analysis, scores = protocol.analyze(reconstructed)
    saved_analysis = json.loads((output / "analysis.json").read_text())
    saved_scores = _jsonl(output / "per-opening-scores.jsonl")
    if analysis != saved_analysis or scores != saved_scores:
        raise ValueError("seed556_registered_bootstrap_reconstruction_mismatch")
    return {
        "status": "verified_complete_with_completion_order_correction",
        "correction": "ledger rows validated in completion order and compared after indexed normalization; raw ledger unchanged",
        "validated_pairs": protocol.OPENINGS,
        "validated_games": protocol.GAMES,
        "component_artifacts_rehashed": True,
        "repository_lfs_pointer_rehashed": True,
        "native_probe_rehashed": True,
        "frozen_execution_sources_rehashed": True,
        "execution_source_snapshot_count": len(registration.get("source_sha256", {})),
        "materialized_tablebase_sha256": registration["tablebase_materialized_sha256"],
        "materialized_tablebase_size_bytes": registration[
            "tablebase_materialized_size"
        ],
        "all_games_valid": True,
        "all_games_terminal": analysis["capped_nonterminal_games"] == 0,
        "analysis": analysis,
    }
