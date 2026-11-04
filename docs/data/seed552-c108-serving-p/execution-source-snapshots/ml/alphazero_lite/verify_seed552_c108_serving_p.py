"""Read-only independent semantic verifier for seed552 C108 versus P."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import verify_seed551_controlled_policy_strength as prior_verifier
from ml.alphazero_lite import seed552_c108_serving_p as protocol


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate_command_semantics(
    command: dict[str, Any], index: int, out: Path, opening_sha: str,
    runtime_identity: dict[str, Any],
) -> None:
    """Check command meaning independently of any receipt/hash binding."""
    argv = command.get("argv", [])
    if len(argv) < 3 or argv[0] != runtime_identity.get("executable") or argv[1:3] != ["-m", "ml.alphazero_lite.seed552_arena_entrypoint"]:
        raise ValueError("seed552_command_runtime_executable_mismatch")
    if command.get("opening_index") != index or command.get("suite_row_sha256") != opening_sha:
        raise ValueError("seed552_command_opening_binding_mismatch")
    if command.get("component_assignments") != protocol.ASSIGNMENTS:
        raise ValueError("seed552_command_component_assignment_mismatch")
    if command.get("runtime_identity") != runtime_identity:
        raise ValueError("seed552_command_runtime_identity_mismatch")
    if not argv or "--seed" not in argv or argv[argv.index("--seed") + 1] != str(protocol.SEED + index):
        raise ValueError("seed552_command_seed_mismatch")
    expected_flags = {
        "--challenger-policy-artifact": out / "runtime-artifacts/C108",
        "--challenger-value-artifact": out / "runtime-artifacts/initializer",
        "--current-policy-artifact": out / "runtime-artifacts/P-E4",
        "--current-value-artifact": out / "runtime-artifacts/initializer",
    }
    for flag, expected in expected_flags.items():
        if flag not in argv or Path(argv[argv.index(flag) + 1]) != expected:
            raise ValueError(f"seed552_command_component_path_mismatch:{flag}")
    options = json.dumps(protocol.SEARCH_OPTIONS, sort_keys=True, separators=(",", ":"))
    for flag in ("--challenger-search-options-json", "--current-search-options-json"):
        if flag not in argv or argv[argv.index(flag) + 1] != options:
            raise ValueError(f"seed552_command_search_options_mismatch:{flag}")
    if argv[argv.index("--exact-root-tablebase") + 1] != runtime_identity.get("tablebase_path"):
        raise ValueError("seed552_command_tablebase_runtime_path_mismatch")


def compare_analysis_binding(saved: dict[str, Any], rebuilt: dict[str, Any],
                             saved_rows: list[dict[str, Any]], rebuilt_rows: list[dict[str, Any]]) -> None:
    if saved != rebuilt:
        raise ValueError("seed552_published_analysis_reconstruction_mismatch")
    if saved_rows != rebuilt_rows:
        raise ValueError("seed552_published_per_opening_reconstruction_mismatch")


def verify_preflight(root: Path, out: Path) -> dict[str, Any]:
    root, out = root.resolve(), out.resolve()
    registration_path = out / "registration.json"
    freeze_path = out / "execution-freeze.json"
    registration = json.loads(registration_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    if (
        registration.get("schema") != "seed552_c108_serving_p_v1"
        or registration.get("status") != "prospective_candidate_and_rules_frozen_before_games"
        or freeze.get("registration_sha256") != protocol.sha256(registration_path)
        or freeze.get("openings_sha256") != protocol.sha256(out / "openings.jsonl")
        or protocol.sha256(out / "openings.jsonl") != protocol.OPENINGS_SHA256
        or registration.get("suite_sha256") != protocol.OPENINGS_SHA256
        or freeze.get("tablebase_object_sha256") != protocol.TABLEBASE_OBJECT_SHA256
        or freeze.get("tablebase_object_size_bytes") != protocol.TABLEBASE_OBJECT_SIZE
        or freeze.get("tablebase_pointer_sha256") != protocol.HISTORICAL_TABLEBASE_POINTER_SHA256
        or freeze.get("tablebase_repository_object_sha256") != protocol.TABLEBASE_OBJECT_SHA256
        or freeze.get("tablebase_repository_object_size_bytes") != protocol.TABLEBASE_OBJECT_SIZE
        or freeze.get("tablebase_pointer_git_commit") != protocol.HISTORICAL_TABLEBASE_POINTER_COMMIT
    ):
        raise ValueError("seed552_freeze_or_suite_binding_mismatch")
    if (
        freeze.get("source_sha256") != registration.get("execution_sources")
        or freeze.get("analysis") != registration.get("analysis")
        or freeze.get("arena") != registration.get("arena")
        or freeze.get("runtime") != registration.get("runtime")
        or freeze.get("durable_execution_command") != registration.get("durable_execution_command")
    ):
        raise ValueError("seed552_freeze_registration_semantic_mismatch")
    receipts = registration.get("historical_verification_receipts", {})
    expected_receipt_names = (
        "seed550-training-verification.json",
        "seed551-publication-verification.json",
    )
    if set(receipts) != set(expected_receipt_names):
        raise ValueError("seed552_historical_verification_receipt_set_mismatch")
    attempt_path = out / "historical-verifier-attempts.json"
    attempts = json.loads(attempt_path.read_text())
    if protocol.sha256(attempt_path) != registration.get("historical_verifier_attempts_sha256") or attempts.get("schema") != "seed552_historical_verifier_attempts_v1" or [row.get("result") for row in attempts.get("attempts", [])] != ["failed_seed551_freeze_binding_mismatch", "failed_seed551_authenticated_input_changed:model-artifact/runtime/kalah_v1_21.kvtb", "verified_complete"]:
        raise ValueError("seed552_historical_verifier_attempt_history_mismatch")
    for filename in expected_receipt_names:
        if protocol.sha256(out / "historical-authentication" / filename) != receipts[filename]:
            raise ValueError(f"seed552_historical_verification_receipt_hash_mismatch:{filename}")
    training_receipt = json.loads((out / "historical-authentication/seed550-training-verification.json").read_text())
    publication_receipt = json.loads((out / "historical-authentication/seed551-publication-verification.json").read_text())
    if training_receipt.get("status") != "verified" or training_receipt.get("training_updates_per_arm") != 108 or not training_receipt.get("bytes_and_mtimes_unchanged"):
        raise ValueError("seed552_seed550_training_receipt_semantic_mismatch")
    if publication_receipt.get("status") != "verified_complete" or publication_receipt.get("verified_games") != 2048 or not publication_receipt.get("read_only_bytes_and_mtimes_unchanged") or publication_receipt.get("analysis", {}).get("decision") != "do_not_advance_source_evaluator_branch":
        raise ValueError("seed552_seed551_publication_receipt_semantic_mismatch")
    runtime = json.loads((out / "runtime-authentication.json").read_text())
    if (
        runtime.get("component_assignments") != protocol.ASSIGNMENTS
        or registration.get("component_assignments") != protocol.ASSIGNMENTS
    ):
        raise ValueError("seed552_component_assignment_mismatch")
    components = runtime["components"]
    for name, digest in (("C108", protocol.C_SHA256), ("P-E4", protocol.P_SHA256), ("initializer", protocol.INITIALIZER_SHA256)):
        component = components[name]
        if component["checkpoint_sha256"] != digest or component["files"].get("model.npz") != digest:
            raise ValueError(f"seed552_component_identity_mismatch:{name}")
        artifact = root / component["artifact_dir"]
        if any(protocol.sha256(artifact / filename) != value for filename, value in component["files"].items()):
            raise ValueError(f"seed552_component_file_changed:{name}")
    tb = runtime.get("tablebase", {})
    if (
        tb.get("historical_pointer_sha256") != freeze["tablebase_pointer_sha256"]
        or tb.get("repository_object_sha256") != protocol.TABLEBASE_OBJECT_SHA256
        or tb.get("materialized_object_sha256") != protocol.TABLEBASE_OBJECT_SHA256
        or tb.get("materialized_object_size_bytes") != protocol.TABLEBASE_OBJECT_SIZE
        or not Path(tb.get("resolved_runtime_path", "")).is_absolute()
    ):
        raise ValueError("seed552_tablebase_identity_mismatch")
    for path in (root / tb["repository_path"], Path(tb["resolved_runtime_path"])):
        if not path.is_file() or path.stat().st_size != protocol.TABLEBASE_OBJECT_SIZE or protocol.sha256(path) != protocol.TABLEBASE_OBJECT_SHA256:
            raise ValueError("seed552_materialized_tablebase_bytes_mismatch")
    if protocol.sha256(out / "runtime-authentication.json") != registration.get("runtime_authentication_sha256"):
        raise ValueError("seed552_runtime_authentication_binding_mismatch")
    authenticated = registration["authenticated_file_sha256"]
    for relative, digest in authenticated.items():
        if protocol.sha256(root / relative) != digest:
            raise ValueError(f"seed552_authenticated_root_file_changed:{relative}")
    if (
        protocol.sha256(out / "README.md") != registration.get("readme_sha256")
        or protocol.sha256(out / "overlap-report.json") != registration.get("opening_overlap_report_sha256")
        or protocol.sha256(out / "overlap-per-opening.jsonl") != registration.get("opening_overlap_rows_sha256")
        or protocol.sha256(out / "overlap-limitations.json") != registration.get("opening_overlap_note_sha256")
        or protocol.sha256(out / "runtime-policy-preflight.json") != registration.get("runtime_policy_preflight_sha256")
    ):
        raise ValueError("seed552_registered_preflight_artifact_mismatch")
    execution_attempt_path = out / "execution-attempts.json"
    execution_attempts = json.loads(execution_attempt_path.read_text())
    if (
        protocol.sha256(execution_attempt_path) != registration.get("execution_attempts_sha256")
        or execution_attempts.get("schema") != "seed552_pre_game_execution_attempts_v1"
        or len(execution_attempts.get("attempts", [])) != 2
        or any(row.get("failure") != "seed551_registered_runtime_contract_mismatch" for row in execution_attempts["attempts"])
        or any(row.get("arena_games_started") is not False for row in execution_attempts["attempts"])
        or any(row.get("observed_runtime", {}).get("cpu_affinity") != list(range(32)) for row in execution_attempts["attempts"])
    ):
        raise ValueError("seed552_pre_game_attempt_record_mismatch")
    expected_overlap = protocol.opening_overlaps(root, prior_verifier.replay_suite(root))
    expected_summary = {key: value for key, value in expected_overlap.items() if key != "per_opening"}
    if json.loads((out / "overlap-report.json").read_text()) != expected_summary:
        raise ValueError("seed552_overlap_summary_reconstruction_mismatch")
    if jsonl(out / "overlap-per-opening.jsonl") != expected_overlap["per_opening"]:
        raise ValueError("seed552_overlap_rows_reconstruction_mismatch")
    for relative, digest in freeze["source_sha256"].items():
        if protocol.sha256(root / relative) != digest:
            raise ValueError(f"seed552_frozen_source_changed:{relative}")
    c_checkpoint = root / protocol.C_CHECKPOINT
    p_checkpoint = root / protocol.P_CHECKPOINT
    initializer = root / protocol.INITIALIZER
    if any((protocol.sha256(path) != expected) for path, expected in (
        (c_checkpoint, protocol.C_SHA256), (p_checkpoint, protocol.P_SHA256),
        (initializer, protocol.INITIALIZER_SHA256),
    )):
        raise ValueError("seed552_source_checkpoint_identity_mismatch")
    from ml.alphazero_lite.verify_seed542_expanded_policy_strength_screen import _serving_identity

    release = json.loads((root / "deploy/serving-release.json").read_text())
    expected_release_ids = release["bundles"]["policy-value-p"]["expected_artifact_sha256"]
    actual_release_ids = [
        _serving_identity(out / "runtime-artifacts/initializer"),
        _serving_identity(out / "runtime-artifacts/P-E4"),
    ]
    if actual_release_ids != expected_release_ids or actual_release_ids != runtime.get("serving_release_component_artifact_identities"):
        raise ValueError("seed552_serving_P_component_identity_mismatch")
    actual_policy_ids = [
        protocol.sha256(out / "runtime-artifacts/initializer/search_policy.json"),
        protocol.sha256(out / "runtime-artifacts/P-E4/search_policy.json"),
    ]
    if actual_policy_ids != release["bundles"]["policy-value-p"]["expected_policy_sha256"] or actual_policy_ids != runtime.get("serving_release_component_policy_identities"):
        raise ValueError("seed552_serving_P_sidecar_identity_mismatch")
    source = (root / protocol.OPENINGS_PATH).read_bytes()
    copied = (out / "openings.jsonl").read_bytes()
    if source != copied or hashlib.sha256(source).hexdigest() != protocol.OPENINGS_SHA256:
        raise ValueError("seed552_suite_copy_changed")
    source_rows = source.splitlines(keepends=True)
    commands = jsonl(out / "command-preflight.jsonl")
    if len(commands) != protocol.OPENINGS or protocol.sha256(out / "command-preflight.jsonl") != freeze.get("command_preflight_sha256"):
        raise ValueError("seed552_command_preflight_binding_mismatch")
    frozen_runtime = registration.get("runtime")
    for i, command in enumerate(commands):
        opening = out / "batches" / f"opening-{i:04d}.jsonl"
        if opening.read_bytes() != source_rows[i] or command.get("opening_index") != i or command.get("suite_row_sha256") != protocol.sha256(opening):
            raise ValueError(f"seed552_opening_row_or_command_index_mismatch:{i}")
        validate_command_semantics(command, i, out, protocol.sha256(opening), frozen_runtime)
    return {"registration": registration, "freeze": freeze, "runtime": runtime, "commands": commands}


def _outcome(opening: dict[str, Any], row: dict[str, Any]) -> tuple[str, int, bool]:
    result, margin, capped = prior_verifier._game_outcome(opening, row)
    # The seed551 helper labels its challenger T; seed552's challenger is C108.
    label = "C" if result == "T" else "P" if result == "C" else "draw"
    if capped:
        label = f"capped_{label}"
    return label, margin, capped


def validate_pair(root: Path, out: Path, index: int) -> list[dict[str, Any]]:
    root, out = root.resolve(), out.resolve()
    openings = prior_verifier.replay_suite(root)
    batch = out / "batches"
    opening_path = batch / f"opening-{index:04d}.jsonl"
    expected_row = (root / protocol.OPENINGS_PATH).read_bytes().splitlines(keepends=True)[index]
    if opening_path.read_bytes() != expected_row:
        raise ValueError(f"seed552_played_opening_not_frozen_row:{index}")
    command = json.loads((batch / f"command-{index:04d}.json").read_text())
    frozen_commands = jsonl(out / "command-preflight.jsonl")
    if command != frozen_commands[index]:
        raise ValueError(f"seed552_played_command_not_frozen:{index}")
    games_path = batch / f"games-{index:04d}.jsonl"
    report_path = batch / f"report-{index:04d}.json"
    games = jsonl(games_path)
    if len(games) != 2:
        raise ValueError(f"seed552_pair_game_count_mismatch:{index}")
    outcomes = []
    for seat, game in enumerate(games):
        prior_verifier._validate_arena_local_opening_index(game)
        if game.get("game_index") != seat or game.get("game_within_opening") != seat or game.get("challenger_player") != seat:
            raise ValueError(f"seed552_seat_pairing_mismatch:{index}:{seat}")
        result, margin, capped = _outcome(openings[index], game)
        outcomes.append({
            "opening_index": index, "game_position": seat, "challenger_player": seat,
            "outcome": result, "margin": margin, "capped_nonterminal": capped,
            "game_sha256": protocol.sha256(games_path), "report_sha256": protocol.sha256(report_path),
            "command_sha256": protocol.sha256(batch / f"command-{index:04d}.json"),
        })
    report = json.loads(report_path.read_text())
    if report.get("schema") != "arena_v1" or report.get("games_played") != 2:
        raise ValueError(f"seed552_report_schema_or_count_mismatch:{index}")
    if report.get("wins") != sum(row["winner"] == "challenger" for row in games) or report.get("losses") != sum(row["winner"] == "current" for row in games) or report.get("draws") != sum(row["winner"] == "draw" for row in games):
        raise ValueError(f"seed552_report_outcome_accounting_mismatch:{index}")
    notes = report.get("notes", {})
    profile = notes.get("search_profile", {})
    runtime = json.loads((out / "runtime-authentication.json").read_text())
    expected_tb = runtime["tablebase"]["materialized_object_sha256"]
    expected_probe = runtime["native_probe"]["sha256"]
    if any(container.get("exact_root_tablebase_sha256") != expected_tb or container.get("exact_root_native_probe_sha256") != expected_probe for container in (notes, profile)):
        raise ValueError(f"seed552_raw_report_tablebase_identity_mismatch:{index}")
    expected_settings = {
        "simulations": 128, "challenger_simulations": 128, "current_simulations": 128,
        "c_puct": 1.25, "exact_root_solve_enabled": True,
        "exact_root_solve_threshold": 16, "exact_root_objective": "final_score_margin",
        "exact_leaf_solve_mode": "disabled", "search_options": protocol.SEARCH_OPTIONS,
        "exact_root_solver": "native_kvtb_root_action_probe_v1",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
    }
    if any(profile.get(key) != value for key, value in expected_settings.items()):
        raise ValueError(f"seed552_raw_report_search_profile_mismatch:{index}")
    expected_runtime_policy = {
        "runtime_search_policy_mode": "puct_exact_root_hybrid",
        "runtime_search_policy_sha256": protocol.sha256(out / "runtime-artifacts/P-E4/search_policy.json"),
        "challenger_runtime_policy_inherited": False,
    }
    if any(profile.get(key) != value for key, value in expected_runtime_policy.items()):
        raise ValueError(f"seed552_raw_report_runtime_policy_profile_mismatch:{index}")
    if notes.get("search_options") != protocol.SEARCH_OPTIONS or notes.get("workers_requested") != 1 or notes.get("workers_used") != 1 or notes.get("random_opening_plies") != 0 or notes.get("seed_contract") != "azlite_eval_seed_v2" or notes.get("base_seed") != protocol.SEED + index:
        raise ValueError(f"seed552_raw_report_notes_runtime_mismatch:{index}")
    if notes.get("exact_root_solve_threshold") != 16 or notes.get("exact_root_objective") != "final_score_margin" or notes.get("exact_leaf_solve_mode") != "disabled" or notes.get("exact_root_tie_rule") != "highest_legal_network_prior_then_lowest_move_index":
        raise ValueError(f"seed552_raw_report_notes_exact_root_mismatch:{index}")
    for key, value in {**expected_runtime_policy, **{
        "exact_root_native_probe_sha256": expected_probe,
        "exact_root_tablebase_sha256": expected_tb,
    }}.items():
        if notes.get(key) != value:
            raise ValueError(f"seed552_raw_report_notes_identity_mismatch:{index}:{key}")
    if not str(notes.get("challenger_path", "")).endswith("/runtime-artifacts/C108") or not str(notes.get("current_path", "")).endswith("/runtime-artifacts/P-E4"):
        raise ValueError(f"seed552_raw_report_component_paths_mismatch:{index}")
    frozen_command = jsonl(out / "command-preflight.jsonl")[index]
    identity_expected = {
        "opening_index": index,
        "seed": protocol.SEED + index,
        "command_preflight_entry_sha256": hashlib.sha256(protocol.stable_json(frozen_command)).hexdigest(),
        "component_assignments": protocol.ASSIGNMENTS,
        "runtime_authentication_sha256": protocol.sha256(out / "runtime-authentication.json"),
        "tablebase_object_sha256": expected_tb,
        "tablebase_object_size_bytes": protocol.TABLEBASE_OBJECT_SIZE,
        "tablebase_resolved_runtime_path": runtime["tablebase"]["resolved_runtime_path"],
        "native_probe_sha256": expected_probe,
        "registered_runtime": frozen_command["runtime_identity"],
    }
    for container in (notes, profile):
        identity = container.get("seed552_execution_identity", {})
        if any(identity.get(key) != value for key, value in identity_expected.items()):
            raise ValueError(f"seed552_raw_report_execution_identity_mismatch:{index}")
        if identity.get("observed_runtime") != frozen_command["runtime_identity"].get("observed"):
            raise ValueError(f"seed552_raw_report_observed_runtime_mismatch:{index}")
    unhashed_profile = {key: value for key, value in profile.items() if key != "hash"}
    expected_profile_hash = hashlib.sha256(
        json.dumps(unhashed_profile, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    if profile.get("hash") != expected_profile_hash or notes.get("search_profile_hash") != expected_profile_hash:
        raise ValueError(f"seed552_raw_report_profile_hash_mismatch:{index}")
    return outcomes


def reconstruct_analysis(ledger: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rebuild the preregistered cluster bootstrap independently of publisher code."""
    if len(ledger) != protocol.GAMES:
        raise ValueError("seed552_independent_analysis_requires_2048_rows")
    points = {"C": 1.0, "draw": 0.5, "P": 0.0,
              "capped_C": 1.0, "capped_draw": 0.5, "capped_P": 0.0}
    grouped: dict[int, dict[int, dict[str, Any]]] = {i: {} for i in range(protocol.OPENINGS)}
    terminals = {"C": 0, "draw": 0, "P": 0}
    capped = 0
    for row in ledger:
        opening, seat = int(row["opening_index"]), int(row["game_position"])
        if opening not in grouped or seat not in (0, 1) or seat in grouped[opening]:
            raise ValueError("seed552_independent_duplicate_ledger_coordinate")
        outcome = row["outcome"]
        if outcome not in points:
            raise ValueError("seed552_independent_outcome_invalid")
        if row.get("capped_nonterminal"):
            if not outcome.startswith("capped_"):
                raise ValueError("seed552_capped_game_not_separately_labeled")
            capped += 1
        else:
            if outcome.startswith("capped_"):
                raise ValueError("seed552_terminal_game_mislabeled_capped")
            terminals[outcome] += 1
        grouped[opening][seat] = row
    if any(set(pair) != {0, 1} for pair in grouped.values()):
        raise ValueError("seed552_independent_missing_seat_pair")
    clusters = np.asarray([
        (points[grouped[i][0]["outcome"]] + points[grouped[i][1]["outcome"]]) / 2
        for i in range(protocol.OPENINGS)
    ], dtype=np.float64)
    rng = np.random.default_rng(protocol.BOOTSTRAP_SEED)
    indices = rng.integers(0, protocol.OPENINGS, size=(protocol.BOOTSTRAP_RESAMPLES, protocol.OPENINGS))
    distribution = clusters[indices].mean(axis=1)
    lower = float(np.quantile(distribution, 0.05, method="linear"))
    interval = [float(np.quantile(distribution, q, method="linear")) for q in (0.025, 0.975)]
    valid = capped == 0
    decision = (
        "advance_to_separate_1200_simulation_screen" if valid and lower > 0.5
        else "no_valid_strength_decision" if not valid else "do_not_advance_C108"
    )
    result = {
        "status": "complete", "games": protocol.GAMES, "opening_pairs": protocol.OPENINGS,
        "terminal_wins": terminals["C"], "terminal_draws": terminals["draw"],
        "terminal_losses": terminals["P"], "capped_nonterminal_games": capped,
        "capped_nonterminal_prevents_valid_decision": bool(capped),
        "score": float(clusters.mean()), "bootstrap_resamples": protocol.BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": protocol.BOOTSTRAP_SEED,
        "sampling": "opening clusters with replacement; duplicate selections retained",
        "quantile_method": "NumPy linear", "one_sided_95_lower": lower,
        "descriptive_95_interval": interval, "decision": decision,
        "strength_decision_valid": valid, "promotion_or_deployment_authorized": False,
    }
    openings = [
        {"opening_index": i, "seat0_C_score": points[grouped[i][0]["outcome"]],
         "seat1_C_score": points[grouped[i][1]["outcome"]],
         "opening_cluster_score": float(clusters[i]),
         "capped_nonterminal_games": sum(bool(grouped[i][s].get("capped_nonterminal")) for s in (0, 1))}
        for i in range(protocol.OPENINGS)
    ]
    return result, openings


def verify(root: Path, out: Path) -> dict[str, Any]:
    preflight = verify_preflight(root, out)
    openings = prior_verifier.replay_suite(root)
    ledger = jsonl(out / "outcomes.jsonl") if (out / "outcomes.jsonl").is_file() else []
    committed: dict[int, list[dict[str, Any]]] = {}
    for index in range(protocol.OPENINGS):
        marker = out / "batches" / f"completion-{index:04d}.json"
        if marker.exists():
            expected = validate_pair(root, out, index)
            receipt = json.loads(marker.read_text())
            pair_receipts = [r for r in ledger if r["opening_index"] == index]
            if len(pair_receipts) != 2 or pair_receipts != expected:
                raise ValueError(f"seed552_ledger_pair_reconstruction_mismatch:{index}")
            if receipt.get("command_sha256") != protocol.sha256(out / "batches" / f"command-{index:04d}.json"):
                raise ValueError(f"seed552_completion_command_binding_mismatch:{index}")
            batch = out / "batches"
            for name, expected in (
                ("games_sha256", protocol.sha256(batch / f"games-{index:04d}.jsonl")),
                ("report_sha256", protocol.sha256(batch / f"report-{index:04d}.json")),
                ("opening_sha256", protocol.sha256(batch / f"opening-{index:04d}.jsonl")),
                ("registration_sha256", protocol.sha256(out / "registration.json")),
            ):
                if receipt.get(name) != expected:
                    raise ValueError(f"seed552_completion_evidence_binding_mismatch:{index}:{name}")
            if receipt.get("outcomes") != expected:
                raise ValueError(f"seed552_completion_outcomes_mismatch:{index}")
            committed[index] = expected
        else:
            batch = out / "batches"
            if any((batch / f"{kind}-{index:04d}.{suffix}").exists() for kind, suffix in (("command", "json"), ("games", "jsonl"), ("report", "json"))):
                raise ValueError(f"seed552_uncommitted_raw_attempt_preserved:{index}")
    if len(ledger) != 2 * len(committed):
        raise ValueError("seed552_ledger_accounting_mismatch")
    if len(ledger) != protocol.GAMES:
        return {"status": "incomplete", "validated_games": len(ledger), "valid_strength_decision": False}
    result, opening_scores = reconstruct_analysis(ledger)
    published = json.loads((out / "analysis.json").read_text())
    published_openings = jsonl(out / "per-opening-scores.jsonl")
    compare_analysis_binding(published, result, published_openings, opening_scores)
    result = {
        "status": "verified_complete", "preflight": {"status": "verified", "commands": len(preflight["commands"])},
        "validated_games": len(ledger), "validated_pairs": len(committed),
        "analysis": result, "per_opening_scores": opening_scores,
        "all_suite_rows_retained": len(openings) == protocol.OPENINGS,
    }
    receipt_path = out / "verification-receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        receipt_paths = (
            "registration.json", "execution-freeze.json", "command-preflight.jsonl",
            "runtime-authentication.json", "openings.jsonl", "outcomes.jsonl",
            "analysis.json", "per-opening-scores.jsonl",
        )
        expected_hashes = {relative: protocol.sha256(out / relative) for relative in receipt_paths}
        saved_result = json.loads((out / "verification.json").read_text())
        progress = json.loads((out / "execution-progress.json").read_text())
        if (
            receipt.get("schema") != "seed552_read_only_verification_receipt_v1"
            or receipt.get("status") != result["status"]
            or receipt.get("verified_games") != len(ledger)
            or receipt.get("verified_pairs") != len(committed)
            or receipt.get("evidence_sha256") != expected_hashes
            or receipt.get("result_sha256") != protocol.sha256(out / "verification.json")
            or saved_result != result
            or progress.get("status") != "complete_verified"
            or progress.get("committed_games") != protocol.GAMES
            or progress.get("committed_opening_pairs") != protocol.OPENINGS
        ):
            raise ValueError("seed552_verification_receipt_reconstruction_mismatch")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=protocol.ROOT)
    parser.add_argument("--out", type=Path, default=protocol.OUT)
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else args.root / args.out
    print(json.dumps(verify(args.root, out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
