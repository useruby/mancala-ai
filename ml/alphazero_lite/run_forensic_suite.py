#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.forensic_suite import ForensicPosition, canonical_state_key, centered_value_from_probability, load_suite, summarize_bucket_matrix, summarize_system

if os.environ.get("AZLITE_FORENSIC_SUITE_STUB") != "1":
    from ml.alphazero_lite.arena import ArtifactEvaluator, build_eval_search_options, evaluate_artifact_position
    from ml.alphazero_lite.classic_mcts import MCTS
    from ml.alphazero_lite.kalah_rules import KalahGame
else:
    ArtifactEvaluator = None
    build_eval_search_options = lambda: {}  # noqa: E731
    evaluate_artifact_position = None
    MCTS = None
    KalahGame = None


DEFAULT_SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
SHARED_REFERENCE_SCHEMA = "azlite_forensic_references_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default=str(DEFAULT_SUITE_PATH))
    parser.add_argument("--current-artifact", required=True)
    parser.add_argument("--challenger-artifact", required=True)
    parser.add_argument("--reference-artifact")
    parser.add_argument("--mcts-simulations", type=int, default=1200)
    parser.add_argument("--teacher-simulations", type=int, default=0)
    parser.add_argument("--artifact-simulations", type=int, default=384)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _teacher_value(child_stats: list[dict]) -> float | None:
    total_visits = sum(int(child.get("visits", 0)) for child in child_stats)
    if total_visits <= 0:
        return None
    weighted_sum = sum(float(child.get("win_rate", 0.0)) * int(child.get("visits", 0)) for child in child_stats)
    return weighted_sum / total_visits


def _regret(child_stats: list[dict], selected_move: int | None) -> float | None:
    if not child_stats or selected_move is None:
        return None
    best = max(float(child.get("win_rate", 0.0)) for child in child_stats)
    chosen = next((float(child.get("win_rate", 0.0)) for child in child_stats if child.get("move") == selected_move), best)
    return max(0.0, best - chosen)


def _has_reference_child_stats(reference: dict[str, Any]) -> bool:
    return bool(_reference_child_stats(reference))


def _mark_regret_unavailable(summary: dict[str, Any]) -> dict[str, Any]:
    summary["overall"]["average_regret"] = None
    summary["overall"]["blunder_rate"] = None
    for bucket_summary in summary["buckets"].values():
        bucket_summary["average_regret"] = None
        bucket_summary["blunder_rate"] = None
    return summary


def _clear_regret_summary(summary: dict[str, Any]) -> dict[str, Any]:
    summary["average_regret"] = None
    summary["blunder_rate"] = None
    return summary


def _clear_top1_summary(summary: dict[str, Any]) -> dict[str, Any]:
    summary["top1_agreement"] = None
    return summary


def _regret_available(rows: list[dict[str, Any]]) -> bool:
    return any(row.get("regret") is not None for row in rows)


def _top1_available(rows: list[dict[str, Any]]) -> bool:
    return any(row.get("agrees_top1") is not None for row in rows)


def _apply_sparse_regret_summaries(summary: dict[str, Any]) -> dict[str, Any]:
    if not _regret_available(summary["rows"]):
        _clear_regret_summary(summary["overall"])

    for bucket, bucket_summary in summary["buckets"].items():
        bucket_rows = [row for row in summary["rows"] if row["bucket"] == bucket]
        if not _regret_available(bucket_rows):
            _clear_regret_summary(bucket_summary)
    return summary


def _apply_sparse_top1_summaries(summary: dict[str, Any]) -> dict[str, Any]:
    available_rows = [row for row in summary["rows"] if row.get("agrees_top1") is not None]
    if not available_rows:
        _clear_top1_summary(summary["overall"])
    else:
        agreements = sum(1 for row in available_rows if row["agrees_top1"])
        summary["overall"]["top1_agreement"] = round(agreements / len(available_rows), 4)

    for bucket, bucket_summary in summary["buckets"].items():
        bucket_rows = [row for row in summary["rows"] if row["bucket"] == bucket and row.get("agrees_top1") is not None]
        if not bucket_rows:
            _clear_top1_summary(bucket_summary)
            continue
        agreements = sum(1 for row in bucket_rows if row["agrees_top1"])
        bucket_summary["top1_agreement"] = round(agreements / len(bucket_rows), 4)
    return summary


def _apply_sparse_bucket_matrix(matrix: dict[str, dict[str, Any]], system_rows: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    for bucket, bucket_summary in matrix.items():
        for system_name, system_summary in bucket_summary["systems"].items():
            rows = [row for row in system_rows[system_name] if row["bucket"] == bucket]
            if not _regret_available(rows):
                _clear_regret_summary(system_summary)
    return matrix


def _apply_sparse_top1_bucket_matrix(matrix: dict[str, dict[str, Any]], system_rows: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    for bucket, bucket_summary in matrix.items():
        for system_name, system_summary in bucket_summary["systems"].items():
            rows = [
                row
                for row in system_rows[system_name]
                if row["bucket"] == bucket and row.get("agrees_top1") is not None
            ]
            if not rows:
                _clear_top1_summary(system_summary)
                continue
            agreements = sum(1 for row in rows if row["agrees_top1"])
            system_summary["top1_agreement"] = round(agreements / len(rows), 4)
    return matrix


def _load_shared_references(reference_artifact_path: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    artifact = json.loads(Path(reference_artifact_path).read_text(encoding="utf-8"))
    if artifact.get("schema") != SHARED_REFERENCE_SCHEMA:
        raise SystemExit(
            f"shared reference artifact must use schema {SHARED_REFERENCE_SCHEMA}"
        )

    rows = artifact.get("rows")
    if not isinstance(rows, list):
        raise SystemExit("reference artifact must contain a rows list")

    references_by_canonical_state: dict[str, dict[str, Any]] = {}
    required_fields = {"id", "canonical_state", "state", "reference_move", "teacher_value", "reference_unstable", "observed_reference_moves", "seed_samples"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"shared reference artifact row {index} must be an object")

        missing_fields = sorted(field for field in required_fields if field not in row)
        if missing_fields:
            raise SystemExit(
                "shared reference artifact row "
                f"{index} missing required fields: {', '.join(missing_fields)}"
            )

        canonical_state = str(row["canonical_state"])
        derived_canonical_state = canonical_state_key(row["state"])
        if derived_canonical_state != canonical_state:
            raise SystemExit(
                "shared reference artifact row "
                f"{index} canonical_state does not match state"
            )
        if canonical_state in references_by_canonical_state:
            raise SystemExit(
                "shared reference artifact row "
                f"{index} has duplicate canonical_state"
            )
        references_by_canonical_state[canonical_state] = dict(row)
    return artifact, references_by_canonical_state


def _reference_move(reference: dict[str, Any]) -> int | None:
    if reference.get("reference_move") is not None:
        return int(reference["reference_move"])
    return None


def _reference_child_stats(reference: dict[str, Any]) -> list[dict[str, Any]]:
    child_stats = reference.get("child_stats")
    return child_stats if isinstance(child_stats, list) else []


def _stub_reference(index: int, policy_simulations: int, value_simulations: int) -> dict:
    if index == 0:
        policy = {
            "selected_move": 0,
            "child_stats": [
                {"move": 0, "visits": policy_simulations, "win_rate": 0.9},
                {"move": 1, "visits": policy_simulations, "win_rate": 0.1},
            ],
        }
        teacher_probability = (value_simulations - 1) / 20.0
    else:
        policy = {
            "selected_move": 4,
            "child_stats": [
                {"move": 1, "visits": policy_simulations, "win_rate": 0.1},
                {"move": 4, "visits": policy_simulations, "win_rate": 0.5},
            ],
        }
        teacher_probability = (value_simulations - 9) / 20.0
    teacher_probability = max(0.0, min(1.0, teacher_probability))
    return {
        **policy,
        "teacher_value": centered_value_from_probability(teacher_probability),
    }


def _stub_system(system_name: str, index: int, legal_moves: tuple[int, ...]) -> dict:
    current_cycle = [
        {"selected_move": 0, "value": 0.2},
        {"selected_move": 1, "value": -0.2},
        {"selected_move": 2, "value": 0.0},
    ]
    challenger_cycle = [
        {"selected_move": 1, "value": -0.4},
        {"selected_move": 4, "value": -0.2},
        {"selected_move": 3, "value": -0.1},
    ]
    rows = current_cycle if system_name == "current" else challenger_cycle
    system = dict(rows[index % len(rows)])
    if system["selected_move"] not in legal_moves:
        system["selected_move"] = legal_moves[index % len(legal_moves)]
    return {**system, "policy": [], "child_stats": [], "visits": []}


def run_reference(state: dict, policy_simulations: int, value_simulations: int, seed: int, index: int) -> dict:
    if os.environ.get("AZLITE_FORENSIC_SUITE_STUB") == "1":
        return _stub_reference(index, policy_simulations, value_simulations)

    search = MCTS(KalahGame.from_state(state), simulations=policy_simulations, seed=seed)
    summary = search.root_summary()

    if value_simulations == policy_simulations:
        teacher_probability = _teacher_value(summary["child_stats"])
    else:
        teacher_search = MCTS(KalahGame.from_state(state), simulations=value_simulations, seed=seed + 50_000)
        teacher_probability = _teacher_value(teacher_search.root_summary()["child_stats"])

    summary["teacher_value"] = None if teacher_probability is None else centered_value_from_probability(teacher_probability)
    return summary


def build_row(*, position: ForensicPosition, reference: dict, system: dict) -> dict:
    reference_move = _reference_move(reference)
    teacher_value = reference.get("teacher_value")
    system_value = float(system["value"])
    regret = _regret(_reference_child_stats(reference), system["selected_move"])
    agrees_top1 = None if reference_move is None else system["selected_move"] == reference_move
    row = {
        "id": position.id,
        "state": position.state,
        "canonical_state": position.canonical_key,
        "side_to_move": position.side_to_move,
        "legal_moves": list(position.legal_moves),
        "phase": position.phase,
        "bucket": position.bucket,
        "tags": list(position.tags),
        "source": position.source,
        "reference_move": reference_move,
        "selected_move": system["selected_move"],
        "agrees_top1": agrees_top1,
        "regret": None if regret is None else round(regret, 4),
        "teacher_value": None if teacher_value is None else round(float(teacher_value), 4),
        "system_value": round(system_value, 4),
        "value_error": None if teacher_value is None else round(abs(system_value - float(teacher_value)), 4),
    }
    for field_name in ("reference_unstable", "observed_reference_moves", "seed_samples"):
        if field_name in reference:
            row[field_name] = reference[field_name]
    return row


def main() -> None:
    args = parse_args()
    reference_artifact_path = getattr(args, "reference_artifact", None)
    suite_path = Path(args.suite)
    out_path = Path(args.out)
    suite = load_suite(suite_path)
    search_options = build_eval_search_options()
    value_reference_simulations = int(args.teacher_simulations) if int(args.teacher_simulations) > 0 else int(args.mcts_simulations)
    stub_mode = os.environ.get("AZLITE_FORENSIC_SUITE_STUB") == "1"

    if stub_mode:
        print("warning: stub mode enabled for forensic suite output", file=sys.stderr)

    shared_reference_artifact = None
    if reference_artifact_path:
        shared_reference_artifact, references_by_canonical_state = _load_shared_references(reference_artifact_path)
        references = []
        for position in suite:
            reference = references_by_canonical_state.get(position.canonical_key)
            if reference is None:
                raise SystemExit(f"missing shared reference row for canonical state: {position.id}")
            references.append(reference)
    else:
        references = [
            run_reference(position.state, int(args.mcts_simulations), value_reference_simulations, args.seed + index, index)
            for index, position in enumerate(suite)
        ]

    systems = {
        "current": str(Path(args.current_artifact)),
        "challenger": str(Path(args.challenger_artifact)),
    }
    evaluators = None
    if not stub_mode:
        evaluators = {
            system_name: ArtifactEvaluator(Path(artifact_path)) for system_name, artifact_path in systems.items()
        }

    system_rows: dict[str, list[dict]] = {}
    for system_name, artifact_path in systems.items():
        rows: list[dict] = []
        for index, position in enumerate(suite):
            if stub_mode:
                system = _stub_system(system_name, index, position.legal_moves)
            else:
                system = evaluate_artifact_position(
                    artifact_path=artifact_path,
                    evaluator=None if evaluators is None else evaluators[system_name],
                    state=position.state,
                    simulations=args.artifact_simulations,
                    seed=args.seed + 1000 + index,
                    c_puct=args.c_puct,
                    search_options=search_options,
                )
            rows.append(
                build_row(
                    position=position,
                    reference=references[index],
                    system=system,
                )
            )
        system_rows[system_name] = rows

    report = {
        "schema": "azlite_forensic_suite_v1",
        "stub": stub_mode,
        "suite_path": str(suite_path),
        "positions": len(suite),
        "reference": {
            "kind": "shared_artifact",
            "artifact_path": str(Path(reference_artifact_path)),
            "shared_reference": None if shared_reference_artifact is None else shared_reference_artifact.get("reference"),
        }
        if reference_artifact_path
        else {
            "kind": "classic_mcts",
            "policy_reference": {
                "kind": "classic_mcts",
                "simulations": int(args.mcts_simulations),
            },
            "value_reference": {
                "kind": "classic_mcts",
                "simulations": value_reference_simulations,
            },
        },
        "systems": {
            system_name: {
                "artifact_path": artifact_path,
                **(
                    _apply_sparse_regret_summaries(summarize_system(system_rows[system_name]))
                    if reference_artifact_path and not all(_has_reference_child_stats(reference) for reference in references)
                    else summarize_system(system_rows[system_name])
                ),
            }
            for system_name, artifact_path in systems.items()
        },
        "buckets": summarize_bucket_matrix(system_rows),
    }

    if reference_artifact_path:
        for system_summary in report["systems"].values():
            _apply_sparse_top1_summaries(system_summary)

    if reference_artifact_path and not all(_has_reference_child_stats(reference) for reference in references):
        report["buckets"] = _apply_sparse_bucket_matrix(report["buckets"], system_rows)

    if reference_artifact_path:
        report["buckets"] = _apply_sparse_top1_bucket_matrix(report["buckets"], system_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote forensic report to {out_path}")


if __name__ == "__main__":
    main()
