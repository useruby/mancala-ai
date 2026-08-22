"""Offline reconstruction and metrics for paired PUCT selection traces."""

from __future__ import annotations

from typing import Any

import numpy as np


def visit_js(left: list[float], right: list[float]) -> float:
    """Jensen-Shannon divergence of non-negative visit vectors."""
    p, q = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    p, q = p / p.sum(), q / q.sum()
    midpoint = (p + q) / 2.0
    return float(
        0.5 * np.sum(p * np.log(np.maximum(p, 1e-12) / np.maximum(midpoint, 1e-12)))
        + 0.5 * np.sum(q * np.log(np.maximum(q, 1e-12) / np.maximum(midpoint, 1e-12)))
    )


def _rank(values: dict[int, float]) -> tuple[int, ...]:
    return tuple(sorted(values, key=lambda move: (-values[move], move)))


def reconstruct_root_trajectory(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover root child statistics after each trace record.

    An unvisited child has ``value_sum=0`` and ``q_value=0``. This is both the
    live PUCT summary convention and the convention used in paired Q vectors.
    """
    if not trace or not trace[0]["selection_path"]:
        raise ValueError("trace must contain a root selection")
    root = trace[0]["selection_path"][0]
    actions = tuple(sorted(int(move) for move in root["legal_moves"]))
    priors = {int(child["move"]): float(child["prior"]) for child in root["children"]}
    visits = {move: 0 for move in actions}
    values = {move: 0.0 for move in actions}
    trajectory = []
    for record in trace:
        path = record["selection_path"]
        if not path:
            raise ValueError("trace record has no root selection")
        action = int(path[0]["chosen_move"])
        if action not in visits:
            raise ValueError("root action is not legal")
        visits[action] += 1
        values[action] += float(record["backed_up_value"])
        q_values = {
            move: values[move] / visits[move] if visits[move] else 0.0
            for move in actions
        }
        visit_leader = max(actions, key=lambda move: (visits[move], -move))
        deterministic_move = max(
            actions,
            key=lambda move: (visits[move], q_values[move], priors[move], -move),
        )
        ordered_visits = sorted(visits.values(), reverse=True)
        trajectory.append(
            {
                "simulation": int(record["simulation_index"]),
                "actions": list(actions),
                "visit_count": dict(visits),
                "value_sum": dict(values),
                "q_value": q_values,
                "visit_distribution": [visits[move] for move in actions],
                "deterministic_move": int(deterministic_move),
                "visit_leader": int(visit_leader),
                "q_ranking": list(_rank(q_values)),
                "best_q_action": int(_rank(q_values)[0]),
                "top1_top2_visit_margin": float(
                    ordered_visits[0] - ordered_visits[1]
                    if len(ordered_visits) > 1
                    else ordered_visits[0]
                ),
            }
        )
    return trajectory


def validate_final_root_trajectory(
    trajectory: list[dict[str, Any]], summary: dict[str, Any]
) -> bool:
    """Check the reconstructed final child visits/Qs against live PUCT output."""
    final = trajectory[-1]
    children = {int(child["move"]): child for child in summary["child_stats"]}
    return (
        int(final["deterministic_move"]) == int(summary["selected_move"])
        and set(final["actions"]) == set(children)
        and all(
            int(final["visit_count"][move]) == int(children[move]["visits"])
            and abs(float(final["q_value"][move]) - float(children[move]["q_value"]))
            <= 1e-12
            for move in children
        )
    )


def _common_prefix(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    depth = 0
    for a, b in zip(left, right):
        if int(a["chosen_move"]) != int(b["chosen_move"]):
            break
        depth += 1
    return depth


def _run(values: list[bool]) -> dict[str, Any]:
    runs, current, longest = [], 0, 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    first = next((index + 1 for index, value in enumerate(values) if value), None)
    final_persistent_start = None
    if values and values[-1]:
        final_persistent_start = len(values)
        while final_persistent_start > 0 and values[final_persistent_start - 1]:
            final_persistent_start -= 1
        final_persistent_start += 1
    return {
        "first_relative_simulation": first,
        "first_final_persistent_relative_simulation": final_persistent_start,
        "disappears_again": bool(
            first is not None and any(not value for value in values[first:])
        ),
        "longest_consecutive_run": longest,
        "fraction_remaining": float(np.mean(values)) if values else 0.0,
    }


def paired_postdivergence_metrics(
    p1_trace: list[dict[str, Any]],
    a16_trace: list[dict[str, Any]],
    p1_trajectory: list[dict[str, Any]],
    a16_trajectory: list[dict[str, Any]],
    first_simulation: int,
) -> dict[str, Any]:
    """Calculate fixed post-divergence metrics from continuous paired traces."""
    start = first_simulation - 1
    samples = []
    for index in range(start, len(p1_trace)):
        p, a = p1_trace[index], a16_trace[index]
        pt, at = p1_trajectory[index], a16_trajectory[index]
        actions = pt["actions"]
        q_deltas = [abs(at["q_value"][move] - pt["q_value"][move]) for move in actions]
        path_equal = [int(item["chosen_move"]) for item in p["selection_path"]] == [
            int(item["chosen_move"]) for item in a["selection_path"]
        ]
        samples.append(
            {
                "simulation": index + 1,
                "path_differs": not path_equal,
                "common_prefix_depth": _common_prefix(
                    p["selection_path"], a["selection_path"]
                ),
                "leaf_hash_mismatch": p["selected_leaf_state_hash"]
                != a["selected_leaf_state_hash"],
                "terminal_mismatch": bool(p["terminal_leaf"])
                != bool(a["terminal_leaf"]),
                "backup_difference": float(a["backed_up_value"] - p["backed_up_value"]),
                "backup_absolute_difference": abs(
                    float(a["backed_up_value"] - p["backed_up_value"])
                ),
                "backup_opposite_sign": bool(
                    float(a["backed_up_value"]) * float(p["backed_up_value"]) < 0.0
                ),
                "root_q_l1": float(sum(q_deltas)),
                "root_q_max": float(max(q_deltas)),
                "q_rank_disagreement": at["q_ranking"] != pt["q_ranking"],
                "best_q_disagreement": at["best_q_action"] != pt["best_q_action"],
                "candidate_selected_root_q_difference": float(
                    at["q_value"][at["deterministic_move"]]
                    - pt["q_value"][at["deterministic_move"]]
                ),
                "visit_js": visit_js(
                    at["visit_distribution"], pt["visit_distribution"]
                ),
                "visit_l1": float(
                    np.abs(
                        np.asarray(at["visit_distribution"])
                        - np.asarray(pt["visit_distribution"])
                    ).sum()
                ),
                "visit_leader_disagreement": at["visit_leader"] != pt["visit_leader"],
                "root_move_disagreement": at["deterministic_move"]
                != pt["deterministic_move"],
                "visit_margin_difference": float(
                    at["top1_top2_visit_margin"] - pt["top1_top2_visit_margin"]
                ),
            }
        )

    cumulative_abs, squared_sum = 0.0, 0.0
    for index, sample in enumerate(samples, start=1):
        cumulative_abs += sample["backup_absolute_difference"]
        squared_sum += sample["backup_difference"] ** 2
        sample["backup_cumulative_absolute_difference"] = cumulative_abs
        sample["backup_rms_difference"] = float(np.sqrt(squared_sum / index))

    def auc(name: str, width: int, *, absolute: bool = False) -> float:
        values = [float(sample[name]) for sample in samples[:width]]
        return float(sum(abs(value) for value in values) if absolute else sum(values))

    def fraction(name: str, width: int) -> float:
        return float(np.mean([sample[name] for sample in samples[:width]]))

    windows = {}
    for offset in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256):
        sample = samples[min(offset, len(samples) - 1)]
        windows[str(offset)] = sample
    return {
        "early_metrics": {
            "backup_gap_auc_32": auc("backup_difference", 32, absolute=True),
            "q_divergence_auc_32": auc("root_q_l1", 32),
            "visit_js_auc_32": auc("visit_js", 32),
            "path_divergence_fraction_32": fraction("path_differs", 32),
            "backup_gap_auc_64": auc("backup_difference", 64, absolute=True),
            "q_divergence_auc_64": auc("root_q_l1", 64),
            "visit_js_auc_64": auc("visit_js", 64),
            "path_divergence_fraction_64": fraction("path_differs", 64),
        },
        "windows": windows,
        "at_384": samples[384 - first_simulation]
        if first_simulation <= 384 <= len(p1_trace)
        else None,
        "lead_lag": {
            "path_divergence": _run([sample["path_differs"] for sample in samples]),
            "backup_value_difference": _run(
                [abs(sample["backup_difference"]) > 1e-12 for sample in samples]
            ),
            "q_ranking_difference": _run(
                [sample["q_rank_disagreement"] for sample in samples]
            ),
            "visit_leader_difference": _run(
                [sample["visit_leader_disagreement"] for sample in samples]
            ),
            "root_move_difference": _run(
                [sample["root_move_disagreement"] for sample in samples]
            ),
        },
    }
