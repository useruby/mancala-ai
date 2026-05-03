#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.forensic_suite import ForensicPosition, centered_value_from_probability, load_suite, summarize_bucket_matrix, summarize_system

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default=str(DEFAULT_SUITE_PATH))
    parser.add_argument("--current-artifact", required=True)
    parser.add_argument("--challenger-artifact", required=True)
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


def _regret(child_stats: list[dict], selected_move: int | None) -> float:
    if not child_stats or selected_move is None:
        return 0.0
    best = max(float(child.get("win_rate", 0.0)) for child in child_stats)
    chosen = next((float(child.get("win_rate", 0.0)) for child in child_stats if child.get("move") == selected_move), best)
    return max(0.0, best - chosen)


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
    teacher_value = reference.get("teacher_value")
    system_value = float(system["value"])
    return {
        "id": position.id,
        "state": position.state,
        "side_to_move": position.side_to_move,
        "legal_moves": list(position.legal_moves),
        "phase": position.phase,
        "bucket": position.bucket,
        "tags": list(position.tags),
        "source": position.source,
        "reference_move": reference["selected_move"],
        "selected_move": system["selected_move"],
        "agrees_top1": system["selected_move"] == reference["selected_move"],
        "regret": round(_regret(reference["child_stats"], system["selected_move"]), 4),
        "teacher_value": None if teacher_value is None else round(float(teacher_value), 4),
        "system_value": round(system_value, 4),
        "value_error": None if teacher_value is None else round(abs(system_value - float(teacher_value)), 4),
    }


def main() -> None:
    args = parse_args()
    suite_path = Path(args.suite)
    out_path = Path(args.out)
    suite = load_suite(suite_path)
    search_options = build_eval_search_options()
    value_reference_simulations = int(args.teacher_simulations) if int(args.teacher_simulations) > 0 else int(args.mcts_simulations)
    stub_mode = os.environ.get("AZLITE_FORENSIC_SUITE_STUB") == "1"

    if stub_mode:
        print("warning: stub mode enabled for forensic suite output", file=sys.stderr)

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
                **summarize_system(system_rows[system_name]),
            }
            for system_name, artifact_path in systems.items()
        },
        "buckets": summarize_bucket_matrix(system_rows),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote forensic report to {out_path}")


if __name__ == "__main__":
    main()
