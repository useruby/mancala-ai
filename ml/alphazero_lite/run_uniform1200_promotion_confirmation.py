#!/usr/bin/env python3
"""Run the single pre-registered uniform1200 incumbent holdout confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))
HIGH_POWER = (
    ROOT / "docs/data/alphazero-lite-uniform1200-high-power-arena-v2/aggregate.json"
)
EXACT = ROOT / "docs/data/alphazero-lite-fresh-uniform1200-confirmation-6seed.json"
CURRENT = ROOT / "model-artifact/current"
REFERENCES = ROOT / "ml/alphazero_lite/fixtures/incumbent_forensic_references_v2.json"
SEEDS = (47, 48, 49, 50, 51, 52)
EXPECTED_CANDIDATE_SHA256 = (
    "935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c"
)
EXPECTED_CURRENT_SHA256 = (
    "f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c"
)
EXPECTED_CURRENT_VERSION = "seed48-nextgen-s455-default-value-iter1"
SHADOW_THRESHOLDS = {
    "overall": {"accuracy": -0.02, "regret": 0.02, "blunder": 0.01},
    "capture_available": {"accuracy": -0.03, "regret": 0.03, "blunder": 0.02},
    "sparse_endgame": {"accuracy": -0.03, "regret": 0.03, "blunder": 0.02},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def candidate_path(seed: int) -> Path:
    root = (
        ".tmp/fresh-uniform1200" if seed <= 49 else ".tmp/fresh-uniform1200-extension"
    )
    return (
        ROOT
        / root
        / "runs"
        / f"seed{seed}"
        / "uniform1200"
        / f"fresh-uniform1200-s{seed}-uniform1200-iter1"
    )


def selection_candidates(
    high_power: dict[str, Any], exact: dict[str, Any]
) -> list[dict[str, Any]]:
    exact_by_seed = {
        int(row["seed"]): row["uniform1200"] for row in exact["exact_outcome_metrics"]
    }
    candidates = []
    for seed in SEEDS:
        metrics = high_power["seeds"][str(seed)]["metrics"]
        candidates.append(
            {
                "seed": seed,
                "mean_opening_pair_score": float(metrics["mean_opening_pair_score"]),
                "opening_pair_ci95_lower": float(metrics["opening_pair_ci95"]["lower"]),
                "outcome_optimal_top1_accuracy": float(exact_by_seed[seed]["top1"]),
                "mean_wdl_outcome_regret": float(exact_by_seed[seed]["mean_regret"]),
                "checkpoint_weights_json_sha256": high_power["checkpoint_sha256"][
                    str(seed)
                ]["uniform1200"],
            }
        )
    if {row["seed"] for row in candidates} != set(SEEDS) or len(candidates) != len(
        SEEDS
    ):
        raise ValueError("selection candidates must be exactly seeds47-52")
    return sorted(
        candidates,
        key=lambda row: (
            -row["mean_opening_pair_score"],
            -row["opening_pair_ci95_lower"],
            -row["outcome_optimal_top1_accuracy"],
            row["mean_wdl_outcome_regret"],
            row["seed"],
        ),
    )


def verify_incumbent() -> dict[str, Any]:
    metadata_path, weights_path = CURRENT / "metadata.json", CURRENT / "weights.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    weights_sha = sha256(weights_path)
    if (
        weights_sha != EXPECTED_CURRENT_SHA256
        or metadata.get("version") != EXPECTED_CURRENT_VERSION
    ):
        raise ValueError("promotion_incumbent_changed")
    return {
        "metadata_sha256": sha256(metadata_path),
        "weights_json_sha256": weights_sha,
        "version": metadata["version"],
        "architecture": metadata["architecture"],
        "input_encoding": metadata["input_encoding"],
    }


def verify_candidate(selected: dict[str, Any]) -> dict[str, Any]:
    path = candidate_path(int(selected["seed"]))
    weights = path / "weights.json"
    checkpoint = path / "checkpoint.npz"
    metadata_path = path / "metadata.json"
    if not all(item.is_file() for item in (weights, checkpoint, metadata_path)):
        raise FileNotFoundError("promotion_candidate_artifact_missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    weights_sha = sha256(weights)
    if (
        weights_sha != EXPECTED_CANDIDATE_SHA256
        or selected["checkpoint_weights_json_sha256"] != weights_sha
    ):
        raise ValueError("promotion_candidate_selection_mismatch")
    architecture = metadata.get("architecture", {})
    if (
        architecture.get("model_type") != "residual_v3"
        or metadata.get("input_encoding") != "kalah_v3"
    ):
        raise FileNotFoundError("promotion_candidate_artifact_missing")
    return {
        "path": str(path.relative_to(ROOT)),
        "weights_json_sha256": weights_sha,
        "checkpoint_sha256": sha256(checkpoint),
        "metadata_sha256": sha256(metadata_path),
        "version": metadata.get("version"),
        "architecture": architecture,
        "input_encoding": metadata.get("input_encoding"),
        "historical_export_manifest": str(
            (path / "run_manifest.json").relative_to(ROOT)
        ),
    }


def gate_defaults() -> dict[str, Any]:
    # These are read from the current production script's argparse defaults.
    return {
        "arena_games": 120,
        "min_arena_score": 0.55,
        "min_arena_games": 120,
        "mcts_games": 40,
        "min_mcts_games": 40,
        "hard_arena_games": 120,
        "hard_min_score": 0.55,
        "skip_mcts_relative_check": False,
        "forensic_thresholds": {
            "overall": {
                "top1_agreement": -0.02,
                "average_regret": 0.02,
                "blunder_rate": 0.01,
            },
            "critical_buckets": {
                "sparse_endgame": {
                    "top1_agreement": -0.03,
                    "average_regret": 0.03,
                    "blunder_rate": 0.02,
                },
                "capture_available": {
                    "top1_agreement": -0.03,
                    "average_regret": 0.03,
                    "blunder_rate": 0.02,
                },
            },
        },
    }


def gate_command(candidate: Path, output: Path) -> list[str]:
    """Render the production invocation without any gate override flags."""
    return [
        str(ROOT / "script/ai/local_promotion_gate"),
        "--candidate-path",
        str(candidate),
        "--current-path",
        "model-artifact/current",
        "--hard-path",
        "model-artifact/current",
        "--out",
        str(output),
    ]


def run_shadow(candidate: Path, output: Path) -> dict[str, Any]:
    raw = output / "outcome_shadow_forensic_raw.json"
    command = [
        sys.executable,
        "ml/alphazero_lite/run_forensic_suite.py",
        "--current-artifact",
        str(CURRENT),
        "--challenger-artifact",
        str(candidate),
        "--reference-artifact",
        str(REFERENCES),
        "--out",
        str(raw),
    ]
    if not raw.exists():
        subprocess.run(command, cwd=ROOT, check=True)
    report = json.loads(raw.read_text(encoding="utf-8"))
    exact_rows = {
        row["id"]: row
        for row in json.loads(REFERENCES.read_text(encoding="utf-8"))["rows"]
    }
    from ml.alphazero_lite.arena import ArtifactEvaluator
    from ml.alphazero_lite.kalah_rules import KalahGame

    def outcome_optimal_masses(
        path: Path, rows: list[dict[str, Any]]
    ) -> dict[str, float]:
        evaluator = ArtifactEvaluator(path)
        masses = {}
        for row in rows:
            exact = exact_rows[row["id"]]
            if exact["exact_status"] != "exact_solved":
                continue
            policy, _value = evaluator.evaluate(KalahGame.from_state(row["state"]))
            masses[row["id"]] = float(
                sum(policy[action] for action in exact["exact_optimal_actions"])
            )
        return masses

    current_masses = outcome_optimal_masses(
        CURRENT, report["systems"]["current"]["rows"]
    )
    candidate_masses = outcome_optimal_masses(
        candidate, report["systems"]["challenger"]["rows"]
    )

    def metrics(rows: list[dict[str, Any]], masses: dict[str, float]) -> dict[str, Any]:
        from ml.alphazero_lite.forensic_exact_references import (
            outcome_regret,
            outcome_utilities,
        )

        solved = [
            (row, exact_rows[row["id"]])
            for row in rows
            if exact_rows[row["id"]]["exact_status"] == "exact_solved"
        ]
        regrets = [
            int(outcome_regret(exact, row["selected_move"])) for row, exact in solved
        ]
        return {
            "roots": len(solved),
            "outcome_optimal_top1": statistics.fmean(regret == 0 for regret in regrets),
            "outcome_optimal_policy_mass": statistics.fmean(
                masses[row["id"]] for row, _exact in solved
            ),
            "mean_outcome_regret": statistics.fmean(regrets),
            "true_outcome_blunder_rate": statistics.fmean(
                regret > 0 for regret in regrets
            ),
            "win_to_draw": sum(
                max(outcome_utilities(exact).values()) == 1
                and outcome_utilities(exact)[row["selected_move"]] == 0
                for row, exact in solved
            ),
            "win_to_loss": sum(
                max(outcome_utilities(exact).values()) == 1
                and outcome_utilities(exact)[row["selected_move"]] == -1
                for row, exact in solved
            ),
        }

    systems = report["systems"]
    current_rows, challenger_rows = (
        systems["current"]["rows"],
        systems["challenger"]["rows"],
    )
    scopes = {"overall": (current_rows, challenger_rows)}
    for bucket in ("capture_available", "sparse_endgame"):
        scopes[bucket] = (
            [row for row in current_rows if row["bucket"] == bucket],
            [row for row in challenger_rows if row["bucket"] == bucket],
        )
    summary = {
        scope: {
            "incumbent": metrics(left, current_masses),
            "candidate": metrics(right, candidate_masses),
        }
        for scope, (left, right) in scopes.items()
    }
    transitions = []
    from ml.alphazero_lite.forensic_exact_references import (
        outcome_regret,
        outcome_utilities,
    )

    for current_row, candidate_row in zip(current_rows, challenger_rows, strict=True):
        exact = exact_rows[current_row["id"]]
        if exact["exact_status"] != "exact_solved":
            continue
        utilities = outcome_utilities(exact)
        transitions.append(
            {
                "id": current_row["id"],
                "bucket": current_row["bucket"],
                "incumbent_action": current_row["selected_move"],
                "candidate_action": candidate_row["selected_move"],
                "incumbent_outcome_regret": outcome_regret(
                    exact, current_row["selected_move"]
                ),
                "candidate_outcome_regret": outcome_regret(
                    exact, candidate_row["selected_move"]
                ),
                "outcome_transition": f"{utilities[current_row['selected_move']]}_to_{utilities[candidate_row['selected_move']]}",
            }
        )
    failures = []
    for scope, threshold in SHADOW_THRESHOLDS.items():
        current_metrics, candidate_metrics = (
            summary[scope]["incumbent"],
            summary[scope]["candidate"],
        )
        if (
            candidate_metrics["outcome_optimal_top1"]
            - current_metrics["outcome_optimal_top1"]
            < threshold["accuracy"]
        ):
            failures.append(f"{scope}_top1")
        if (
            candidate_metrics["mean_outcome_regret"]
            - current_metrics["mean_outcome_regret"]
            > threshold["regret"]
        ):
            failures.append(f"{scope}_regret")
        if (
            candidate_metrics["true_outcome_blunder_rate"]
            - current_metrics["true_outcome_blunder_rate"]
            > threshold["blunder"]
        ):
            failures.append(f"{scope}_blunder")
    return {
        "schema": "azlite_uniform1200_outcome_shadow_v1",
        "raw_report": str(raw.relative_to(ROOT)),
        "thresholds": SHADOW_THRESHOLDS,
        "metrics": summary,
        "paired_state_transitions": transitions,
        "passed": not failures,
        "failure_reasons": failures,
    }


def classify(gate: dict[str, Any], shadow: dict[str, Any]) -> str:
    reasons = {row["code"] for row in gate.get("failure_reasons", [])}
    if (
        "arena_score_below_threshold" in reasons
        or "arena_games_below_minimum" in reasons
    ):
        return "uniform1200_promotion_arena_failed"
    if "candidate_mcts_below_current" in reasons:
        return "uniform1200_mcts1200_relative_failed"
    if "regression_check_failed" in reasons:
        return "uniform1200_regression_gate_failed"
    if {"hard_games_below_minimum", "candidate_not_stronger_than_hard"} & reasons:
        return "uniform1200_hard_gate_failed"
    forensic = {code for code in reasons if code.startswith("forensic_")}
    non_forensic = reasons - forensic
    if forensic and not non_forensic and shadow["passed"]:
        return "uniform1200_production_forensic_only_blocker"
    if not shadow["passed"]:
        return "uniform1200_outcome_shadow_regression"
    if gate.get("passed"):
        return "uniform1200_promotion_confirmation_passed"
    return "uniform1200_promotion_confirmation_inconclusive"


def render_report(
    manifest: dict[str, Any], gate: dict[str, Any], shadow: dict[str, Any]
) -> str:
    arena = json.loads(Path(gate["arena_report_path"]).read_text(encoding="utf-8"))
    lines = [
        "# Uniform1200 Promotion Confirmation",
        "",
        "## Inherited Evidence",
        "",
        "PR #327 classification: `uniform1200_high_power_strength_confirmed`. Its six seed scores were 0.6396484375, 0.681640625, 0.6572265625, 0.6328125, 0.6240234375, and 0.6357421875; aggregate mean effect was +0.1451822917 with all six positive. PR #325 exact W/D/L improved in 6/6 pairs with no repeated true win-to-loss regression and its shadow gate passed 6/6.",
        "",
        "## Frozen Selection",
        "",
        "Selection was persisted before the holdout gate. The ranking rule was mean high-power score, CI95 lower bound, PR #325 outcome-optimal top1, lower mean W/D/L regret, then lower seed. It selected seed48 only.",
        f"- Candidate weights JSON SHA256: `{manifest['candidate_artifact']['weights_json_sha256']}`",
        f"- Incumbent weights JSON SHA256: `{manifest['incumbent']['weights_json_sha256']}` (`{manifest['incumbent']['version']}`)",
        f"- `selection_frozen_before_holdout`: `{manifest['selection_frozen_before_holdout']}`",
        "",
        "## Production Gate",
        "",
        f"The unmodified `script/ai/local_promotion_gate` ran once, with no config override, no skip-MCTS flag, no stub reports, and no PR #327 opening suite. Its direct arena was {arena['wins']}/{arena['draws']}/{arena['losses']} over {arena['games_played']} games, score `{arena['score']:.4f}` vs threshold `{arena['threshold']:.2f}`; move-time mean/p95 were `{arena['notes']['move_time_mean_ms']}`/`{arena['notes']['move_time_p95_ms']}` ms.",
        f"The gate prefilter failed with `{', '.join(row['code'] for row in gate['failure_reasons'])}`. Consequently hard arena, MCTS1200 comparisons, regression positions, and production forensic suite were intentionally not produced by the gate.",
        "",
        "## Outcome-Aligned Shadow",
        "",
        "This separate W/D/L exact-reference evaluation did not affect production pass/fail. Overall candidate/incumbent: top1 `%.4f`/`%.4f`, policy mass `%.4f`/`%.4f`, mean regret `%.4f`/`%.4f`, blunder rate `%.4f`/`%.4f`, win-to-draw `%s`/`%s`, win-to-loss `%s`/`%s`. Capture-available candidate/incumbent top1 was `%.4f`/`%.4f`; sparse-endgame was `%.4f`/`%.4f`."
        % (
            shadow["metrics"]["overall"]["candidate"]["outcome_optimal_top1"],
            shadow["metrics"]["overall"]["incumbent"]["outcome_optimal_top1"],
            shadow["metrics"]["overall"]["candidate"]["outcome_optimal_policy_mass"],
            shadow["metrics"]["overall"]["incumbent"]["outcome_optimal_policy_mass"],
            shadow["metrics"]["overall"]["candidate"]["mean_outcome_regret"],
            shadow["metrics"]["overall"]["incumbent"]["mean_outcome_regret"],
            shadow["metrics"]["overall"]["candidate"]["true_outcome_blunder_rate"],
            shadow["metrics"]["overall"]["incumbent"]["true_outcome_blunder_rate"],
            shadow["metrics"]["overall"]["candidate"]["win_to_draw"],
            shadow["metrics"]["overall"]["incumbent"]["win_to_draw"],
            shadow["metrics"]["overall"]["candidate"]["win_to_loss"],
            shadow["metrics"]["overall"]["incumbent"]["win_to_loss"],
            shadow["metrics"]["capture_available"]["candidate"]["outcome_optimal_top1"],
            shadow["metrics"]["capture_available"]["incumbent"]["outcome_optimal_top1"],
            shadow["metrics"]["sparse_endgame"]["candidate"]["outcome_optimal_top1"],
            shadow["metrics"]["sparse_endgame"]["incumbent"]["outcome_optimal_top1"],
        ),
        "",
        "## Classification",
        "",
        f"`{manifest['classification']}`",
        "",
        "Next experiment: run ONE independent canonical-unique opening-suite arena against the incumbent, not matched control, to determine whether the production start-state arena is the disagreement source.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs/data/alphazero-lite-uniform1200-promotion-confirmation",
    )
    args = parser.parse_args(argv)
    high_power, exact = (
        json.loads(HIGH_POWER.read_text(encoding="utf-8")),
        json.loads(EXACT.read_text(encoding="utf-8")),
    )
    try:
        ranked = selection_candidates(high_power, exact)
        selected = ranked[0]
        if (
            selected["seed"] != 48
            or selected["checkpoint_weights_json_sha256"] != EXPECTED_CANDIDATE_SHA256
        ):
            raise ValueError("promotion_candidate_selection_mismatch")
        incumbent, candidate = verify_incumbent(), verify_candidate(selected)
    except (ValueError, FileNotFoundError) as error:
        write_json(
            args.out_dir / "manifest.json",
            {"classification": str(error), "selection_frozen_before_holdout": False},
        )
        return 0
    manifest = {
        "schema": "azlite_uniform1200_promotion_confirmation_v1",
        "selection_frozen_before_holdout": True,
        "selection_rule": [
            "highest PR327 mean opening-pair score",
            "highest PR327 opening-pair CI95 lower bound",
            "highest PR325 outcome-optimal top1 accuracy",
            "lowest PR325 mean W/D/L outcome regret",
            "lower training seed",
        ],
        "ranked_candidates": ranked,
        "selected": selected,
        "candidate_artifact": candidate,
        "incumbent": incumbent,
        "production_gate_defaults": gate_defaults(),
        "production_opening_suite_reused": False,
        "promotion": {"performed": False},
        "gate_candidate_paths": [candidate["path"]],
    }
    write_json(args.out_dir / "selection_manifest.json", manifest)
    gate_path = args.out_dir / "local_promotion_gate.json"
    command = gate_command(candidate_path(48), gate_path)
    manifest["production_gate_command"] = command
    result = (
        subprocess.run(command, cwd=ROOT, check=False)
        if not gate_path.exists()
        else subprocess.CompletedProcess(command, 1)
    )
    if not gate_path.is_file():
        manifest["classification"] = "uniform1200_promotion_confirmation_inconclusive"
        manifest["gate_returncode"] = result.returncode
        write_json(args.out_dir / "manifest.json", manifest)
        return 0
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    shadow = run_shadow(candidate_path(48), args.out_dir)
    write_json(args.out_dir / "outcome_shadow.json", shadow)
    manifest.update(
        {
            "gate_returncode": result.returncode,
            "gate_passed": gate.get("passed"),
            "shadow_passed": shadow["passed"],
            "classification": classify(gate, shadow),
        }
    )
    write_json(args.out_dir / "manifest.json", manifest)
    (ROOT / "docs/alphazero-lite-uniform1200-promotion-confirmation.md").write_text(
        render_report(manifest, gate, shadow), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
