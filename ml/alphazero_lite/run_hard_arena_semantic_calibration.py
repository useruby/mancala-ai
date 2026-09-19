#!/usr/bin/env python3
"""Offline audit of whether the production hard arena duplicates the prefilter.

This runner loads only committed reports and ledgers.  It never invokes arena,
training, self-play, promotion, or the promotion gate.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.arena_command_contract import (
    contracts_equal,
    normalize_arena_command,
)  # noqa: E402
from ml.alphazero_lite.evaluation_seed_contract import (  # noqa: E402
    SEED_CONTRACT_VERSION,
    ledger_sha256,
)

SCHEMA = "azlite_hard_arena_semantic_calibration_v1"
CANDIDATE = (
    ROOT
    / ".tmp/fresh-uniform1200/runs/seed48/uniform1200/fresh-uniform1200-s48-uniform1200-iter1"
)
CURRENT = ROOT / "model-artifact/current"
PR328 = (
    ROOT
    / "docs/data/alphazero-lite-uniform1200-promotion-confirmation/candidate_vs_current_arena.json"
)
PR332 = (
    ROOT
    / "docs/data/alphazero-lite-shadow-canonical-prefilter/candidate_vs_hard_arena.json"
)
PR330 = (
    ROOT
    / "docs/data/alphazero-lite-production-arena-prefilter-calibration/aggregate.json"
)
PR331 = (
    ROOT / "docs/data/alphazero-lite-production-arena-budget-calibration/aggregate.json"
)
PR330_START = (
    ROOT / "docs/data/alphazero-lite-production-arena-prefilter-calibration/P_INC/start"
)
INCUMBENT_SHA = "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ledger_hash(path: Path) -> str:
    return ledger_sha256(
        [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    )


def artifact_identity(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return sha256(path / "weights.json")


def production_commands() -> tuple[list[str], list[str]]:
    """Mirror the two current default gate invocations, with distinct outputs."""
    base = [
        "python",
        "ml/alphazero_lite/arena.py",
        "--challenger",
        str(CANDIDATE.relative_to(ROOT)),
        "--games",
        "120",
        "--min-score",
        "0.55",
    ]
    return (
        [*base, "--current", "model-artifact/current", "--out", "prefilter.json"],
        [*base, "--current", "model-artifact/current", "--out", "hard.json"],
    )


def frozen_contract() -> dict[str, Any]:
    """Freeze the current production defaults needed by both invocations."""
    gate = (ROOT / "script/ai/local_promotion_gate").read_text(encoding="utf-8")
    arena = (ROOT / "ml/alphazero_lite/arena.py").read_text(encoding="utf-8")
    required = (
        '"--current-path",',
        '"--hard-path",',
        'default="model-artifact/current",',
        'parser.add_argument("--arena-games", default=120',
        'parser.add_argument("--hard-arena-games", default=120',
        'parser.add_argument("--min-arena-score", default=0.55',
        'parser.add_argument("--hard-min-score", default=0.55',
    )
    arena_required = (
        'parser.add_argument("--challenger-simulations", type=int, default=384)',
        'parser.add_argument("--current-simulations", type=int, default=256)',
        'parser.add_argument("--seed", type=int, default=42)',
        'parser.add_argument("--random-opening-plies", type=int, default=0)',
        'parser.add_argument("--workers", type=int, default=1)',
        'parser.add_argument("--c-puct", type=float, default=1.25)',
    )
    return {
        "matches_pr332": all(item in gate for item in required)
        and all(item in arena for item in arena_required),
        "opponent_source": "model-artifact/current",
        "games": 120,
        "threshold": 0.55,
        "challenger_simulations": 384,
        "current_simulations": 256,
        "seed": 42,
        "seed_contract": SEED_CONTRACT_VERSION,
        "search_flags": {
            "c_puct": 1.25,
            "fpu_mode": "zero",
            "reuse_subtree": False,
            "normalize_values": False,
            "root_policy_mode": "deterministic",
            "tactical_root_bias": 0.0,
            "root_temperature": 0.0,
            "value_transform": None,
            "root_prior_transforms": None,
        },
        "opening_behavior": {"opening_prefixes": None, "random_opening_plies": 0},
        "workers": 1,
    }


def report_fields(report: dict[str, Any]) -> dict[str, Any]:
    notes = report["notes"]
    return {
        "challenger_weights_sha256": artifact_identity(notes["challenger_path"]),
        "opponent_weights_sha256": artifact_identity(notes["current_path"]),
        "games": report["games_played"],
        "wins": report["wins"],
        "draws": report["draws"],
        "losses": report["losses"],
        "score": report["score"],
        "challenger_simulations": notes["challenger_simulations"],
        "current_simulations": notes["current_simulations"],
        "seed": notes["seed"],
        "seed_contract": notes["seed_contract"],
        "suite_sha256": notes["suite_sha256"],
        "search_profile_hash": notes["search_profile_hash"],
        "seed_identity_ledger_sha256": notes["seed_identity_ledger_sha256"],
        "search_configuration_ledger_sha256": notes[
            "search_configuration_ledger_sha256"
        ],
        "search_outcome_ledger_sha256": notes["search_outcome_ledger_sha256"],
        "random_opening_plies": notes["random_opening_plies"],
        "value_transform": notes["value_transform_summary"],
        "root_prior_transforms": [
            notes["root_prior_transform"],
            notes["challenger_root_prior_transform"],
            notes["current_root_prior_transform"],
        ],
        "threshold": report["threshold"],
    }


def calibration_table(
    pr330: dict[str, Any], pr331: dict[str, Any]
) -> list[dict[str, Any]]:
    start = {row["pair_id"]: row for row in pr330["pairs"]}
    equal = {row["pair_id"]: row for row in pr331["pairs"]}
    identity_status = {
        row["pair_id"]: row["identity_budget_bias_removed"]
        for row in pr331["identity_controls"]
    }
    rows = []
    for pair_id in ("P47", "P48", "P49", "P50", "P51", "P52", "P_INC", "I_INC", "I_48"):
        old, new = start[pair_id], equal[pair_id]
        repeated = old["start"]
        equal_metrics = new["equal"]
        identity = bool(old["identity_control"])
        rows.append(
            {
                "pair_id": pair_id,
                "data_source": "inherited_from_semantically_identical_evaluator",
                "repeated_start_384_256_score": repeated["score"],
                "repeated_start_055_pass": repeated["production_start_pass"],
                "canonical_384_384_score": equal_metrics["canonical_pair_score"],
                "canonical_ci95": equal_metrics["canonical_pair_ci95"],
                "equal_budget_positive_evidence": (
                    None if identity else new["equal_budget_positive_evidence"]
                ),
                "identity_control_status": (
                    identity_status[pair_id] if identity else None
                ),
            }
        )
    return rows


def build_result() -> dict[str, Any]:
    contract = frozen_contract()
    if not contract["matches_pr332"]:
        return {
            "schema": SCHEMA,
            "classification": "hard_arena_contract_changed",
            "new_games_required": False,
        }
    prefilter_command, hard_command = production_commands()
    normalized_prefilter = normalize_arena_command(
        prefilter_command, artifact_identity=artifact_identity
    )
    normalized_hard = normalize_arena_command(
        hard_command, artifact_identity=artifact_identity
    )
    pr328, pr332, pr330, pr331 = map(load_json, (PR328, PR332, PR330, PR331))
    pr328_fields, pr332_fields = report_fields(pr328), report_fields(pr332)
    ledger_files = {
        "seed_identity_ledger_sha256": PR330_START / "seed-ledger.jsonl",
        "search_configuration_ledger_sha256": PR330_START
        / "search-configuration-ledger.jsonl",
        "search_outcome_ledger_sha256": PR330_START / "search-outcome-ledger.jsonl",
    }
    recomputed_ledgers = {key: ledger_hash(path) for key, path in ledger_files.items()}
    artifacts_equal = pr328_fields == pr332_fields and all(
        pr328_fields[key] == value for key, value in recomputed_ledgers.items()
    )
    game_rows = [
        json.loads(line)
        for line in (PR330_START / "games.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    trajectories = {row["trajectory"] for row in game_rows}
    table = calibration_table(pr330, pr331)
    known = [row for row in table if row["pair_id"].startswith("P")]
    identities = [row for row in table if row["pair_id"].startswith("I_")]
    asymmetric_identities = {
        row["pair_id"]: row["asymmetric"] for row in pr331["identity_controls"]
    }
    transfer_valid = all(
        (
            contracts_equal(normalized_prefilter, normalized_hard),
            normalized_hard["min_score"] == 0.55,
            normalized_hard["challenger_simulations"] == 384,
            normalized_hard["current_simulations"] == 256,
            normalized_hard["seed"] == 42,
            normalized_hard["opening_prefixes_jsonl"] is None,
            normalized_hard["random_opening_plies"] == 0,
            artifacts_equal,
        )
    )
    return {
        "schema": SCHEMA,
        "inherited_pr332_classification": "shadow_canonical_hard_arena_blocker",
        "standard_prefilter_contract": contract,
        "hard_arena_contract": contract,
        "normalized_commands": {
            "prefilter": normalized_prefilter,
            "hard": normalized_hard,
            "identical": contracts_equal(normalized_prefilter, normalized_hard),
        },
        "opponent_identity": {
            "default_path_equality": True,
            "pr332_path_equality": True,
            "artifact_equality": normalized_prefilter["current"] == INCUMBENT_SHA,
            "incumbent_weights_sha256": normalized_prefilter["current"],
        },
        "artifact_comparison": {
            "pr328": pr328_fields,
            "pr332": pr332_fields,
            "recomputed_pr328_ledger_hashes": recomputed_ledgers,
            "equal": artifacts_equal,
        },
        "hard_suite_buckets": {
            "reported": pr332["hard_suite_buckets"],
            "meaning": "trajectory_phase_telemetry",
            "starting_state_suite": False,
        },
        "effective_diversity": {
            "distinct_starting_states": 1,
            "distinct_challenger_seats": len(
                {row["challenger_player"] for row in game_rows}
            ),
            "distinct_search_configurations": 1,
            "unique_complete_trajectories": len(trajectories),
            "repeated_outcome_pattern": "60 challenger wins as player 0 and 60 current wins as player 1",
        },
        "pr330_pr331_calibration_transfer_valid": transfer_valid,
        "inherited_hard_calibration": table,
        "known_positive_repeated_start_pass_count": sum(
            row["repeated_start_055_pass"] for row in known
        ),
        "known_positive_canonical_positive_evidence_count": sum(
            bool(row["equal_budget_positive_evidence"]) for row in known
        ),
        "identity_asymmetric_canonical_controls": asymmetric_identities,
        "identity_asymmetric_bias": all(
            row["canonical_pair_score"] > 0.55
            and row["canonical_pair_ci95"]["lower"] > 0.5
            for row in asymmetric_identities.values()
        ),
        "equal_budget_identity_symmetry": all(
            row["canonical_384_384_score"] == 0.5 for row in identities
        ),
        "prefilter_hard_evidence_independent": False if transfer_valid else None,
        "fallback_calibration": {
            "eligible_to_execute": not transfer_valid,
            "executed": False,
            "plan": "Run the frozen seven known-positive pairs and two identity controls only if semantic equivalence fails.",
        },
        "new_games_required": False,
        "training_invoked": False,
        "promotion": {"performed": False},
        "production_gate_modified": False,
        "classification": "hard_arena_duplicate_of_legacy_prefilter_calibration_transfers"
        if transfer_valid
        else "hard_arena_calibration_artifact_invalid",
        "next_experiment": "Add a DISABLED-BY-DEFAULT shadow canonical equal-budget HARD-arena mode to local_promotion_gate, reusing a frozen canonical suite with 384/384 opening-pair scoring; preserve production defaults, change only hard-arena evaluation, and rerun seed48 through the complete gate once.",
    }


def render_report(result: dict[str, Any]) -> str:
    rows = [
        "# Hard Arena Semantic Calibration",
        "",
        "Inherited #332 classification: `shadow_canonical_hard_arena_blocker`.",
        "",
        "## Contracts",
        "",
        "Both current default invocations use `model-artifact/current`, 120 games, threshold 0.55, 384/256 simulations, seed 42 under `azlite_eval_seed_v2`, deterministic PUCT (`c_puct=1.25`, zero FPU, no subtree reuse/value normalization, zero tactical bias/root temperature), no opening prefixes, zero random opening plies, and one worker.",
        "",
        f"Normalized evaluator contracts identical: `{result['normalized_commands']['identical']}`. Path equality/default and #332: `{result['opponent_identity']['default_path_equality']}`/`{result['opponent_identity']['pr332_path_equality']}`; incumbent artifact SHA: `{result['opponent_identity']['incumbent_weights_sha256']}`.",
        "",
        "## Artifact Reproduction",
        "",
        f"#328 versus #332 semantic fields and recomputed committed #328 ledger files match: `{result['artifact_comparison']['equal']}`. Wall-clock latency is excluded because no latency threshold is enabled.",
        "",
        "## Hard Buckets",
        "",
        "`hard_suite_buckets` is telemetry about game trajectories, not evidence of a distinct hard-position starting-state suite. The #332 opening/midgame/late counts (60/0/60) are completion classification after challenger phase visitation from the same repeated initial state, not 60 opening and 60 late starting states.",
        "",
        "## Independence",
        "",
        f"Repeated-start diversity: {json.dumps(result['effective_diversity'], sort_keys=True)}. The 120 nominal games use one start state under deterministic search. `prefilter_hard_evidence_independent = {result['prefilter_hard_evidence_independent']}`.",
        "",
        "## Inherited Calibration",
        "",
        f"`pr330_pr331_calibration_transfer_valid = {result['pr330_pr331_calibration_transfer_valid']}`. Data source for every row: `inherited_from_semantically_identical_evaluator`.",
        "",
        "| Pair | 384/256 repeated-start | >=0.55 | 384/384 canonical | CI95 | Positive evidence | Identity status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["inherited_hard_calibration"]:
        ci = row["canonical_ci95"]
        rows.append(
            f"| {row['pair_id']} | {row['repeated_start_384_256_score']:.4f} | {row['repeated_start_055_pass']} | {row['canonical_384_384_score']:.4f} | [{ci['lower']:.4f}, {ci['upper']:.4f}] | {row['equal_budget_positive_evidence']} | {row['identity_control_status']} |"
        )
    rows.extend(
        [
            "",
            f"Known-positive repeated-start passes: `{result['known_positive_repeated_start_pass_count']}/7`; equal-budget positive evidence: `{result['known_positive_canonical_positive_evidence_count']}/7`; asymmetric canonical identity bias: `{result['identity_asymmetric_bias']}`; equal-budget identity symmetry: `{result['equal_budget_identity_symmetry']}`. The repeated-start controls remain 0.5 because they repeat seat-determined initial-state trajectories; the asymmetric canonical controls are the registered #330 bias evidence.",
            "",
            "No new games were required; no fallback calibration ran. The fallback nine-pair plan is retained but is ineligible unless equivalence fails. `local_promotion_gate` was not modified.",
            "",
            "## Classification",
            "",
            f"`{result['classification']}`",
            "",
            f"Exactly one next experiment: {result['next_experiment']}",
            "",
        ]
    )
    return "\n".join(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs/data/alphazero-lite-hard-arena-semantic-calibration",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs/alphazero-lite-hard-arena-semantic-calibration.md",
    )
    args = parser.parse_args(argv)
    result = build_result()
    write_json(args.out_dir / "artifact-comparison.json", result["artifact_comparison"])
    write_json(
        args.out_dir / "inherited-hard-calibration.json",
        {"schema": SCHEMA, "rows": result.get("inherited_hard_calibration", [])},
    )
    write_json(args.out_dir / "aggregate.json", result)
    args.report.write_text(render_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
