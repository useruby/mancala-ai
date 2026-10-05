"""Portable, read-only public verifier for the corrected seed422 publication.

Run from any directory with ``PYTHONPATH=. python
ml/alphazero_lite/verify_seed422_publication.py --root REPOSITORY``. The root
defaults to the repository containing this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed422_adam_memory_analysis import (
    analyze as analyze_original,
    replay_opening,
    validate_ledger,
)
from ml.alphazero_lite.seed422_corrected_analysis import analyze
from ml.alphazero_lite.seed422_public_exclusions import build_union

DATA_RELATIVE = Path("docs/data/seed422-adam-first-moment")
RAW_LEDGER_SHA256 = {
    "A": "ff1b48dea6a41d959372e0c048155f81602406b2cb7631b58733b4b3abdc7f26",
    "B": "0a51168195f0a2b56f04e54879e9273d600e1e8ad6e8587166adb0d9c2d4bd33",
}
ARENA_REPORT_SHA256 = {
    "A": "c9ef246f9b79e0e8088a5643f9129c081099b80b3d433d7ff32daf79d9abb5a0",
    "B": "530bc5b1a01ad8a177b166db5b4390e372adde1448287ab300fd4c8574fdc0a2",
}
DECISION = "retain_baseline_close_fixed_beta1_intervention"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_supplemental_receipt(root: Path, data: Path) -> None:
    receipt_path = data / "supplemental-verification-receipt.json"
    receipt = read_json(receipt_path)
    _require(
        receipt["bound_correction_receipt_sha256"]
        == sha256(data / "correction-receipt.json"),
        "supplemental_receipt_correction_binding_mismatch",
    )
    for relative, expected in receipt["source_sha256"].items():
        _require(
            sha256(root / relative) == expected,
            f"supplemental_source_hash_mismatch:{relative}",
        )
    for relative, expected in receipt["evidence_sha256"].items():
        _require(
            sha256(data / relative) == expected,
            f"supplemental_evidence_hash_mismatch:{relative}",
        )


def _verify_sources(root: Path, data: Path) -> dict[str, Any]:
    registration = read_json(data / "registration.json")
    declared = registration.pop("registration_sha256")
    _require(
        canonical_sha(registration) == declared, "registration_canonical_hash_mismatch"
    )
    registration["registration_sha256"] = declared

    snapshot_root = data / registration["source_snapshots"]
    checked = {}
    for relative, expected in registration["source_hashes"].items():
        live = root / relative
        snapshot = snapshot_root / relative
        _require(live.is_file(), f"registered_source_missing:{relative}")
        _require(
            sha256(live) == expected, f"registered_source_hash_mismatch:{relative}"
        )
        _require(
            snapshot.is_file() and sha256(snapshot) == expected,
            f"registered_source_snapshot_hash_mismatch:{relative}",
        )
        checked[relative] = expected
    return registration


def _verify_exclusions(
    root: Path, data: Path, registration: dict[str, Any]
) -> set[str]:
    proof_path = data / "opening-exclusion-proof.json"
    proof = read_json(proof_path)
    identities, rebuilt = build_union(root)
    _require(proof == rebuilt, "historical_exclusion_union_rebuild_mismatch")
    _require(
        registration["exclusion_proof"]["sha256"] == sha256(proof_path),
        "registration_exclusion_proof_hash_mismatch",
    )
    suite_path = data / "openings.jsonl"
    suite = read_jsonl(suite_path)
    _require(
        registration["suite"]["sha256"] == sha256(suite_path),
        "registered_opening_suite_hash_mismatch",
    )
    suite_ids = {canonical_sha(replay_opening(row)) for row in suite}
    _require(
        len(suite) == 512 and len(suite_ids) == 512,
        "suite_count_or_unique_identity_failure",
    )
    _require(
        not suite_ids.intersection(identities), "suite_overlaps_historical_exclusions"
    )
    return identities


def _verify_training(
    root: Path, data: Path, registration: dict[str, Any]
) -> dict[str, Any]:
    training_path = data / "training-results.json"
    training = read_json(training_path)
    amendment_path = data / "execution-accounting-amendment.json"
    amendment = read_json(amendment_path)
    registration_path = data / "registration.json"
    registration_sha = sha256(registration_path)
    _require(
        training["registration_sha256"] == registration_sha,
        "training_registration_hash_mismatch",
    )
    _require(
        training["execution_amendment_sha256"] == sha256(amendment_path),
        "training_amendment_hash_mismatch",
    )
    _require(
        amendment["registration_sha256"] == registration_sha,
        "amendment_registration_identity_mismatch",
    )
    _require(
        amendment["A_epoch_checkpoint_hashes"] == training["lanes"]["A"]["epochs"],
        "amendment_A_checkpoint_binding_mismatch",
    )
    recovery = amendment["supplemental_execution_source"]
    recovery_snapshot = data / recovery["snapshot"]
    _require(
        sha256(recovery_snapshot) == recovery["snapshot_sha256"] == recovery["sha256"],
        "amendment_recovery_snapshot_hash_mismatch",
    )
    _require(
        sha256(root / recovery["path"]) == recovery["sha256"],
        "amendment_recovery_source_hash_mismatch",
    )
    settings = registration["training"]
    intervention = registration["intervention"]
    _require(
        intervention["A"]["optimizer"] == intervention["B"]["optimizer"] == "Adam",
        "registered_optimizer_mismatch",
    )
    _require(
        intervention["common"]["scheduler"] == "none",
        "registered_scheduler_mismatch",
    )
    _require(
        settings["architecture"]
        == {
            "hidden_sizes": [96, 3],
            "input_encoding": "kalah_v3",
            "model": "residual_v3",
        },
        "registered_architecture_mismatch",
    )
    _require(
        settings["gradient_clip_global_norm"] == 1 and settings["huber_delta"] == 1,
        "registered_loss_or_clipping_mismatch",
    )
    _require(
        [row["weight"] for row in settings["replays"]] == [1, 4, 1, 8, 4],
        "registered_replay_weights_mismatch",
    )
    _require(
        settings["fresh_value_target_mode"] == "default"
        and settings["historical_value_target_mode"] == "sharpened",
        "registered_target_mode_mismatch",
    )
    _require(
        settings["initializer_seed"] == 416, "registered_initializer_seed_mismatch"
    )
    permutation_file = root / settings["epoch_permutations"]["path"]
    _require(
        sha256(permutation_file) == settings["epoch_permutations"]["sha256"],
        "registered_permutation_file_hash_mismatch",
    )
    historical_source = root / amendment["A_history_source"]["path"]
    _require(
        sha256(historical_source) == amendment["A_history_source"]["sha256"],
        "A_history_source_hash_mismatch",
    )
    historic_training = read_json(historical_source)
    _require(
        training["optimizer_updates_per_lane"]
        == settings["expected_updates_per_lane"]
        == 1052,
        "optimizer_update_count_mismatch",
    )
    _require(
        settings["batch_size"] == 512 and settings["epochs"] == 4,
        "training_plan_mismatch",
    )
    for lane, betas in (("A", [0.9, 0.999]), ("B", [0.0, 0.999])):
        result = training["lanes"][lane]
        _require(
            result["initialization_sha256"] == settings["initializer"]["sha256"],
            f"initializer_identity_mismatch:{lane}",
        )
        common = intervention["common"]
        registered_betas = intervention[lane]["betas"]
        _require(
            betas == registered_betas, f"registered_optimizer_betas_mismatch:{lane}"
        )
        _require(
            result["optimizer_parameter_group"]
            == {
                "betas": registered_betas,
                "eps": common["eps"],
                "lr": common["lr"],
                "weight_decay": common["weight_decay"],
            },
            f"optimizer_settings_mismatch:{lane}",
        )
        _require(
            result["optimizer_updates"] == 1052,
            f"optimizer_update_count_mismatch:{lane}",
        )
        _require(
            set(result["epochs"]) == {"1", "2", "3", "4"},
            f"epoch_identity_count_mismatch:{lane}",
        )
        _require(
            result["permutations"] == settings["epoch_permutations"]["epoch_sha256"],
            f"registered_batch_plan_mismatch:{lane}",
        )
        _require(
            all(row["optimizer_updates"] == 263 for row in result["history"]),
            f"epoch_update_accounting_mismatch:{lane}",
        )
    _require(
        training["lanes"]["A"]["epochs"] == amendment["A_epoch_checkpoint_hashes"],
        "A_reproduced_epoch_mismatch",
    )
    _require(
        training["lanes"]["A"]["epochs"] == historic_training["lanes"]["A"]["epochs"],
        "A_historical_epoch_reproduction_mismatch",
    )
    _require(
        training["lanes"]["A"]["permutations"]
        == historic_training["lanes"]["A"]["permutations"],
        "A_historical_batch_plan_mismatch",
    )
    _require(
        training["lanes"]["A"]["epochs"]["4"]
        == settings["expected_A_E4_checkpoint_sha256"],
        "A_E4_identity_mismatch",
    )
    _require(
        training["lanes"]["A"]["permutations"]
        == training["lanes"]["B"]["permutations"],
        "lane_batch_plan_mismatch",
    )
    _require(
        training["lanes"]["A"]["multiplicity_sha256"]
        == training["lanes"]["B"]["multiplicity_sha256"],
        "lane_exposure_mismatch",
    )
    return training


def _verify_runtime(
    data: Path, registration: dict[str, Any], training: dict[str, Any]
) -> dict[str, Any]:
    path = data / "runtime-binding.json"
    binding = read_json(path)
    evaluation = registration["evaluation"]
    _require(
        binding["registration_sha256"] == sha256(data / "registration.json"),
        "runtime_registration_binding_mismatch",
    )
    _require(
        binding["training_sha256"] == sha256(data / "training-results.json"),
        "runtime_training_binding_mismatch",
    )
    _require(
        binding["suite_sha256"] == sha256(data / "openings.jsonl"),
        "runtime_suite_binding_mismatch",
    )
    for field in ("native_probe", "runtime_policy", "tablebase"):
        _require(
            binding[
                {
                    "native_probe": "native_probe_sha256",
                    "runtime_policy": "runtime_policy_sha256",
                    "tablebase": "tablebase_sha256",
                }[field]
            ]
            == evaluation["runtime_hashes"][field],
            f"registered_runtime_identity_mismatch:{field}",
        )
    _require(
        binding["runtime_contract"]["exact_root_solve_threshold"] == 16,
        "root16_threshold_mismatch",
    )
    contract = binding["runtime_contract"]
    _require(
        contract["exact_root_objective"] == "final_score_margin",
        "root16_objective_mismatch",
    )
    _require(
        contract["exact_root_tie_rule"]
        == "highest_legal_network_prior_then_lowest_move_index",
        "root16_tie_rule_mismatch",
    )
    _require(
        contract["exact_leaf_solve_mode"] == "disabled", "exact_leaf_mode_mismatch"
    )
    _require(
        contract["exact_root_solve_threshold"]
        == evaluation["native_root_solve_threshold"],
        "registered_root16_mismatch",
    )
    _require(
        contract["exact_root_native_probe_sha256"] == binding["native_probe_sha256"],
        "native_probe_identity_mismatch",
    )
    _require(
        contract["exact_root_tablebase_sha256"] == binding["tablebase_sha256"],
        "tablebase_identity_mismatch",
    )
    _require(
        contract["runtime_search_policy_sha256"] == binding["runtime_policy_sha256"],
        "runtime_policy_identity_mismatch",
    )
    _require(
        binding["opponent"]["files"] == registration["evaluation"]["opponent_files"],
        "frozen_opponent_identity_mismatch",
    )
    for lane in ("A", "B"):
        candidate = binding["candidates"][lane]
        e4 = training["lanes"][lane]["epochs"]["4"]
        _require(
            candidate["checkpoint_sha256"] == e4,
            f"candidate_E4_identity_mismatch:{lane}",
        )
        _require(
            candidate["artifact_files"]["model.npz"] == e4,
            f"candidate_model_identity_mismatch:{lane}",
        )
        _require(
            candidate["artifact_files"]["search_policy.json"]
            == binding["runtime_policy_sha256"],
            f"candidate_sidecar_identity_mismatch:{lane}",
        )
        _require(
            candidate["runtime_contract"] == contract,
            f"candidate_runtime_contract_mismatch:{lane}",
        )
    return binding


def _verify_outcomes(
    data: Path, binding: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outcome_path = data / "outcome-binding.json"
    outcome = read_json(outcome_path)
    _require(
        outcome["runtime_binding_sha256"] == sha256(data / "runtime-binding.json"),
        "outcome_runtime_binding_mismatch",
    )
    suite = read_jsonl(data / "openings.jsonl")
    rows = read_jsonl(data / "outcome-ledger.jsonl")
    validate_ledger(rows, suite)
    _require(len(rows) == 2048, "arena_game_count_mismatch")
    _require(
        binding["suite_sha256"] == sha256(data / "openings.jsonl"),
        "outcome_suite_identity_mismatch",
    )
    for lane in ("A", "B"):
        entry = outcome["lanes"][lane]
        lane_rows = [row for row in rows if row["lane"] == lane]
        reconstructed = "".join(
            json.dumps(row["game"]) + "\n" for row in lane_rows
        ).encode("utf-8")
        raw_hash = hashlib.sha256(reconstructed).hexdigest()
        _require(
            len(lane_rows) == 1024 and entry["games"] == 1024,
            f"arena_report_game_count_mismatch:{lane}",
        )
        _require(
            raw_hash == RAW_LEDGER_SHA256[lane] == entry["games_sha256"],
            f"raw_game_ledger_hash_mismatch:{lane}",
        )
        report = entry["report"]
        _require(
            entry["report_sha256"] == ARENA_REPORT_SHA256[lane],
            f"arena_report_identity_mismatch:{lane}",
        )
        _require(
            report["games"] == report["games_played"] == 1024,
            f"arena_report_count_mismatch:{lane}",
        )
        wins = sum(row["game"]["winner"] == "challenger" for row in lane_rows)
        draws = sum(row["game"]["winner"] == "draw" for row in lane_rows)
        losses = sum(row["game"]["winner"] == "current" for row in lane_rows)
        _require(
            (report["wins"], report["draws"], report["losses"])
            == (wins, draws, losses),
            f"arena_report_outcome_accounting_mismatch:{lane}",
        )
        _require(
            report["score"] == (wins + draws / 2) / 1024,
            f"arena_report_score_mismatch:{lane}",
        )
        _require(
            report["notes"]["challenger_simulations"]
            == report["notes"]["current_simulations"]
            == 384,
            f"arena_report_simulations_mismatch:{lane}",
        )
        profile = report["notes"]["search_profile"]
        _require(
            profile["kind"] == "arena_eval"
            and profile["simulations"] == 384
            and profile["c_puct"] == 1.25,
            f"arena_report_search_profile_mismatch:{lane}",
        )
        _require(
            profile["search_options"]
            == {
                "fpu_mode": "zero",
                "normalize_values": False,
                "reuse_subtree": False,
                "root_policy_mode": "deterministic",
                "root_temperature": 0.0,
                "tactical_root_bias": 0.0,
            },
            f"arena_report_search_settings_mismatch:{lane}",
        )
        _require(
            report["notes"]["seed"] == 422
            and report["notes"]["seed_contract"] == "azlite_eval_seed_v2",
            f"arena_report_seed_mismatch:{lane}",
        )
        _require(
            report["notes"]["exact_root_solve_threshold"] == 16,
            f"arena_report_root16_mismatch:{lane}",
        )
        _require(
            report["notes"]["runtime_search_policy_sha256"]
            == binding["runtime_policy_sha256"],
            f"arena_report_runtime_identity_mismatch:{lane}",
        )
        _require(
            report["notes"]["exact_root_native_probe_sha256"]
            == binding["native_probe_sha256"],
            f"arena_report_native_probe_mismatch:{lane}",
        )
        _require(
            report["notes"]["exact_root_tablebase_sha256"]
            == binding["tablebase_sha256"],
            f"arena_report_tablebase_mismatch:{lane}",
        )
        _require(
            report["notes"]["suite_sha256"] == binding["suite_sha256"],
            f"arena_report_suite_mismatch:{lane}",
        )
    return rows, outcome


def verify(root: Path | str | None = None) -> dict[str, Any]:
    """Verify the immutable publication rooted at ``root`` without writing."""
    repository = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    data = repository / DATA_RELATIVE
    receipt = read_json(data / "correction-receipt.json")
    _verify_supplemental_receipt(repository, data)
    registration = _verify_sources(repository, data)
    for relative, digest in receipt["original_evidence_sha256"].items():
        _require(
            sha256(data / relative) == digest,
            f"original_evidence_hash_mismatch:{relative}",
        )
    for relative, digest in receipt["registered_source_sha256"].items():
        _require(
            sha256(data / "execution-source-snapshots" / relative) == digest,
            f"registered_source_snapshot_hash_mismatch:{relative}",
        )
    for relative, digest in receipt["corrected_artifacts"].items():
        _require(
            sha256(data / relative) == digest,
            f"corrected_artifact_hash_mismatch:{relative}",
        )
    for relative, digest in receipt["correction_modules"].items():
        _require(
            sha256(repository / relative) == digest,
            f"correction_module_hash_mismatch:{relative}",
        )

    excluded = _verify_exclusions(repository, data, registration)
    training = _verify_training(repository, data, registration)
    binding = _verify_runtime(data, registration, training)
    rows, _outcome = _verify_outcomes(data, binding)
    analysis = analyze(rows)
    corrected = read_json(data / "corrected-analysis.json")
    _require(analysis == corrected, "corrected_analysis_mismatch")
    original = analyze_original(rows)
    published_original = read_json(data / "analysis.json")
    for field in (
        "paired_delta",
        "paired_delta_ci95",
        "B_opening_cluster_score_ci95",
        "bootstrap_samples",
        "bootstrap_seed",
        "cluster_count",
        "opening_matrix",
        "thresholds",
        "decision",
    ):
        _require(
            original[field] == published_original[field],
            f"primary_analysis_changed:{field}",
        )
    _require(analysis["decision"] == DECISION, "scientific_decision_changed")
    _require(
        read_json(data / "opening-matrix.json") == original["opening_matrix"],
        "opening_matrix_mismatch",
    )
    _require(
        read_json(data / "corrected-opening-matrix.json") == analysis["opening_matrix"],
        "corrected_opening_matrix_mismatch",
    )
    for lane, expected in {
        "A": {"0": 0.5322265625, "1": 0.43359375},
        "B": {"0": 0.533203125, "1": 0.4658203125},
    }.items():
        _require(
            analysis["lanes"][lane]["seat_scores"] == expected,
            f"corrected_seat_score_mismatch:{lane}",
        )
    return {
        "verified": True,
        "repository_root": str(repository),
        "excluded_identity_count": len(excluded),
        "suite_count": 512,
        "game_count": len(rows),
        "seat_scores": {
            lane: analysis["lanes"][lane]["seat_scores"] for lane in ("A", "B")
        },
        "primary_estimates_unchanged": True,
        "decision": analysis["decision"],
        "torch_required": False,
        "runtime_artifacts_required": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="repository root (defaults to the module-relative root)",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
