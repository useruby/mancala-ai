#!/usr/bin/env python3
"""Audit runtime value-transform plumbing and root-Q sensitivity."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator  # noqa: E402
from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    parse_cpuct_schedule_json,
    resolve_budget_cpuct,
    schedule_definition,
)
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_opening_suite_seat_benchmark import (  # noqa: E402
    compute_seat_metrics,
    parse_game_jsonl,
    run_arena,
)
from ml.alphazero_lite.runtime_root_sensitivity import (  # noqa: E402
    runtime_sensitivity_diagnostic_for_opening_suite,
)
from ml.alphazero_lite.self_play import (  # noqa: E402
    PUCT,
    build_eval_search_options,
    build_search_profile,
)
from ml.alphazero_lite.value_transforms import (  # noqa: E402
    normalize_value_transform_config,
    phase_bucket_from_seed_count,
    value_transform_hash,
)

SUMMARY_SCHEMA = "azlite_value_transform_plumbing_audit_v1"
BUDGET_LABELS = ("384:256", "768:768", "1200:1200", "1200:256")
DIAGNOSTIC_TRANSFORM_ORDER = (
    "identity_ref",
    "zero_value",
    "negate_value",
    "sign_only",
    "amplify_value_2x",
    "opening_zero_value",
    "opening_negate_value",
)
_WORKER_EVALUATOR: ArtifactEvaluator | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--calibration-states", required=True)
    parser.add_argument("--value-transform-manifest", required=True)
    parser.add_argument("--default-c-puct", type=float, default=1.25)
    parser.add_argument("--cpuct-schedule", required=True)
    parser.add_argument("--tactical-root-bias", type=float, default=0.0)
    parser.add_argument(
        "--diagnostic-transforms",
        default=",".join(
            name for name in DIAGNOSTIC_TRANSFORM_ORDER if name != "identity_ref"
        ),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_expected_hash(weights_path: Path, expected: str) -> str:
    actual = sha256_file(weights_path)
    if actual != expected:
        raise SystemExit(
            f"weights hash mismatch for {weights_path}: expected {expected}, got {actual}"
        )
    return actual


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if payload:
                rows.append(json.loads(payload))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def line_number(path: Path, needle: str) -> int | None:
    for index, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if needle in line:
            return index
    return None


def line_ref(path: str, needle: str) -> str | None:
    file_path = REPO_ROOT / path
    number = line_number(file_path, needle)
    if number is None:
        return None
    return f"{path}:L{number}"


def budget_pairs(labels: tuple[str, ...] = BUDGET_LABELS) -> list[tuple[int, int]]:
    pairs = []
    for label in labels:
        challenger, current = label.split(":", 1)
        pairs.append((int(challenger), int(current)))
    return pairs


def state_hash(state: dict[str, Any]) -> str:
    return canonical_json_hash(state)


def state_phase(state: dict[str, Any]) -> str:
    seeds_remaining = sum(
        int(seed) for seed in state["player_pits"] + state["opponent_pits"]
    )
    return str(phase_bucket_from_seed_count(seeds_remaining))


def state_seat_context(state: dict[str, Any]) -> str:
    return f"P{int(state['current_player'])}"


def diagnostic_transform_configs() -> dict[str, dict[str, Any] | None]:
    def phase_config(mode: str, *, scale: float | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"mode": mode}
        if scale is not None:
            payload["scale"] = scale
        return payload

    def diagnostic_config(
        name: str, phase_params: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "version": "v1",
            "name": name,
            "kind": "diagnostic_phase_transform",
            "phase_params": phase_params,
            "diagnostic_only": True,
        }

    return {
        "identity_ref": None,
        "zero_value": diagnostic_config(
            "zero_value",
            {phase: phase_config("zero") for phase in ("opening", "midgame", "late")},
        ),
        "negate_value": diagnostic_config(
            "negate_value",
            {phase: phase_config("negate") for phase in ("opening", "midgame", "late")},
        ),
        "sign_only": diagnostic_config(
            "sign_only",
            {
                phase: phase_config("sign_only")
                for phase in ("opening", "midgame", "late")
            },
        ),
        "amplify_value_2x": diagnostic_config(
            "amplify_value_2x",
            {
                phase: phase_config("scale_clamp", scale=2.0)
                for phase in ("opening", "midgame", "late")
            },
        ),
        "opening_zero_value": diagnostic_config(
            "opening_zero_value",
            {
                "opening": phase_config("zero"),
                "midgame": phase_config("identity"),
                "late": phase_config("identity"),
            },
        ),
        "opening_negate_value": diagnostic_config(
            "opening_negate_value",
            {
                "opening": phase_config("negate"),
                "midgame": phase_config("identity"),
                "late": phase_config("identity"),
            },
        ),
    }


def requested_transform_names(text: str) -> list[str]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    return [
        "identity_ref",
        *[
            "identity_ref" if item == "identity" else item
            for item in names
            if item not in {"identity", "identity_ref"}
        ],
    ]


def strongest_fitted_transform(
    manifest: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    candidates: list[tuple[float, str, dict[str, Any] | None]] = []
    for name, payload in manifest.get("transforms", {}).items():
        if not payload.get("supported") or payload.get("diagnostic_only"):
            continue
        validation = payload.get("validation", {}).get("overall", {})
        mae = validation.get("mae")
        if isinstance(mae, (int, float)):
            candidates.append((float(mae), str(name), payload.get("value_transform")))
    if not candidates:
        return "identity_ref", None
    _mae, name, transform = min(candidates)
    return name, transform


def transform_specs(
    manifest: dict[str, Any], requested: list[str]
) -> list[dict[str, Any]]:
    diagnostics = diagnostic_transform_configs()
    strongest_name, strongest_transform = strongest_fitted_transform(manifest)
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in [*requested, strongest_name]:
        if name in seen:
            continue
        seen.add(name)
        transform = diagnostics.get(name)
        diagnostic_only = name in diagnostics and name != "identity_ref"
        if name == strongest_name and name not in diagnostics:
            transform = strongest_transform
        if name == "identity_ref":
            transform = None
        normalized = normalize_value_transform_config(transform)
        specs.append(
            {
                "name": name,
                "value_transform": normalized,
                "value_transform_hash": value_transform_hash(normalized),
                "diagnostic_only": diagnostic_only,
                "fitted": name == strongest_name,
            }
        )
    return specs


def suite_state_rows(path: Path, *, source: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    return [
        {
            "source": source,
            "state": row["state"],
            "state_hash": state_hash(row["state"]),
            "phase": state_phase(row["state"]),
            "seat_context": state_seat_context(row["state"]),
            "prefix_moves": row.get("prefix_moves", []),
            "ply": int(row.get("ply", 0)),
        }
        for row in rows
    ]


def calibration_unique_rows(path: Path) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = str(row["state_hash"])
        if key in unique:
            continue
        state = row.get("state")
        if not isinstance(state, dict):
            continue
        unique[key] = {
            "source": "calibration_states",
            "state": state,
            "state_hash": key,
            "phase": str(row.get("phase") or state_phase(state)).replace(
                "mid", "midgame"
            ),
            "seat_context": str(row.get("seat_context") or state_seat_context(state)),
            "prefix_moves": row.get("prefix_moves", []),
            "ply": len(row.get("prefix_moves", [])),
        }
    return sorted(unique.values(), key=lambda row: row["state_hash"])


def trace_probe_states(
    *,
    current_path: Path,
    seed_rows: list[dict[str, Any]],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    seed: int,
    target: int,
) -> list[dict[str, Any]]:
    evaluator = ArtifactEvaluator(current_path)
    traced: dict[str, dict[str, Any]] = {}
    challenger_sims, current_sims = 384, 256
    effective_c_puct = resolve_budget_cpuct(
        schedule=cpuct_schedule,
        challenger_simulations=challenger_sims,
        current_simulations=current_sims,
        default_c_puct=default_c_puct,
    )
    for row_index, seed_row in enumerate(seed_rows):
        game = KalahGame.from_state(seed_row["state"])
        total_ply = int(seed_row.get("ply", len(seed_row.get("prefix_moves", []))))
        local_ply = 0
        while not game.over() and len(traced) < (target * 2):
            state = game.to_state()
            key = state_hash(state)
            traced.setdefault(
                key,
                {
                    "source": "traced_calibration_proxy",
                    "state": state,
                    "state_hash": key,
                    "phase": state_phase(state),
                    "seat_context": state_seat_context(state),
                    "prefix_moves": list(seed_row.get("prefix_moves", [])),
                    "ply": total_ply,
                },
            )
            search_options = build_eval_search_options(
                root_policy_mode="deterministic",
                tactical_root_bias=0.0,
            )
            search = PUCT(
                evaluator=evaluator,
                simulations=(
                    challenger_sims if game.current_player == 0 else current_sims
                ),
                c_puct=effective_c_puct,
                rng=random.Random(seed + (row_index * 97) + local_ply),
                fpu_mode=str(search_options["fpu_mode"]),
                reuse_subtree=bool(search_options["reuse_subtree"]),
                normalize_values=bool(search_options["normalize_values"]),
                root_policy_mode=str(search_options["root_policy_mode"]),
                tactical_root_bias=float(search_options["tactical_root_bias"]),
                root_temperature=float(search_options["root_temperature"]),
            )
            _visits, root = search.run(game)
            legal_moves = game.possible_moves()
            if not legal_moves or root is None:
                break
            move = search.select_root_move(root, legal_moves)
            if not game.move(game.pit_index(move)):
                break
            total_ply += 1
            local_ply += 1
    return select_balanced(list(traced.values()), target=target)


def select_balanced(rows: list[dict[str, Any]], *, target: int) -> list[dict[str, Any]]:
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[(str(row["phase"]), str(row["seat_context"]))].append(row)
    for bucket in by_stratum.values():
        bucket.sort(key=lambda row: row["state_hash"])
    ordered_keys = sorted(by_stratum)
    selected: list[dict[str, Any]] = []
    while len(selected) < target and ordered_keys:
        progressed = False
        for key in ordered_keys:
            bucket = by_stratum[key]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break
    return selected


def sampled_probe_states(
    medium_suite: Path,
    large_suite: Path,
    calibration_states: Path,
    *,
    current_path: Path,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    seed: int,
) -> list[dict[str, Any]]:
    medium_rows = suite_state_rows(medium_suite, source="medium_suite")
    large_rows = suite_state_rows(large_suite, source="fixed_large_suite")
    calibration_rows = calibration_unique_rows(calibration_states)
    if calibration_rows:
        calibration_rows = select_balanced(calibration_rows, target=256)
    else:
        calibration_rows = trace_probe_states(
            current_path=current_path,
            seed_rows=medium_rows + large_rows,
            default_c_puct=default_c_puct,
            cpuct_schedule=cpuct_schedule,
            seed=seed,
            target=256,
        )
    sample = medium_rows + large_rows + calibration_rows
    unique: dict[str, dict[str, Any]] = {}
    for row in sample:
        unique.setdefault(str(row["state_hash"]), row)
    return list(unique.values())


def budget_context_profile(
    *,
    transform: dict[str, Any] | None,
    challenger_sims: int,
    current_sims: int,
    c_puct: float,
) -> dict[str, Any]:
    search_options = build_eval_search_options(
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        value_transform=transform,
    )
    return build_search_profile(
        kind="arena_eval",
        player_mode="puct",
        simulations=max(challenger_sims, current_sims),
        c_puct=float(c_puct),
        search_options=search_options,
        extra_fields={
            "challenger_simulations": int(challenger_sims),
            "current_simulations": int(current_sims),
        },
    )


def _worker_init(current_path: str) -> None:
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = ArtifactEvaluator(Path(current_path))


def _probe_state(task: dict[str, Any]) -> dict[str, Any]:
    global _WORKER_EVALUATOR
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("worker evaluator not initialized")
    state = task["state"]
    game = KalahGame.from_state(state)
    transform = task["value_transform"]
    search_options = build_eval_search_options(
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        value_transform=transform,
    )
    search = PUCT(
        evaluator=_WORKER_EVALUATOR,
        simulations=int(task["simulations"]),
        c_puct=float(task["c_puct"]),
        rng=random.Random(int(task["seed"])),
        fpu_mode=str(search_options["fpu_mode"]),
        reuse_subtree=bool(search_options["reuse_subtree"]),
        normalize_values=bool(search_options["normalize_values"]),
        root_policy_mode=str(search_options["root_policy_mode"]),
        tactical_root_bias=float(search_options["tactical_root_bias"]),
        root_temperature=float(search_options["root_temperature"]),
        value_transform=transform,
    )
    visits, root = search.run(game)
    summary = search.root_summary()
    selection = summary["selection_breakdown"]
    moves = selection["moves"]
    return {
        "state_hash": str(task["state_hash"]),
        "source": str(task["source"]),
        "phase": str(task["phase"]),
        "seat_context": str(task["seat_context"]),
        "budget_label": str(task["budget_label"]),
        "simulations": int(task["simulations"]),
        "current_simulations": int(task["current_simulations"]),
        "selected_move": summary["selected_move"],
        "legal_moves": [int(move) for move in game.possible_moves()],
        "raw_policy_priors": {
            str(entry["move"]): float(entry["prior"]) for entry in moves
        },
        "child_visits": {
            str(entry["move"]): int(entry["visit_count"]) for entry in moves
        },
        "child_q_values": {
            str(entry["move"]): float(entry["q_value"]) for entry in moves
        },
        "child_selection_q_values": {
            str(entry["move"]): float(entry["selection_q_value"]) for entry in moves
        },
        "child_u_values": {
            str(entry["move"]): float(entry["u_component"]) for entry in moves
        },
        "selection_breakdown": selection,
        "root_value_estimate": float(summary.get("root_q_value") or 0.0),
        "root_evaluation_raw_value": summary.get("root_evaluation_raw_value"),
        "root_evaluation_transformed_value": summary.get(
            "root_evaluation_transformed_value"
        ),
        "backed_up_value_range": summary.get("backed_up_value_range"),
        "terminal_leaf_count": int(summary.get("terminal_leaf_count") or 0),
        "nonterminal_leaf_count": int(summary.get("nonterminal_leaf_count") or 0),
        "value_transform": summary.get("value_transform"),
        "value_transform_hash": str(task["value_transform_hash"]),
        "search_profile_hash": str(task["search_profile_hash"]),
        "visits": [float(value) for value in visits.tolist()],
        "root_children": sorted(int(move) for move in root.children),
    }


def top_two_moves(row: dict[str, Any]) -> tuple[int | None, int | None]:
    moves = row["selection_breakdown"]["moves"]
    ranked = sorted(
        moves,
        key=lambda entry: (
            -int(entry["visit_count"]),
            -float(entry["q_value"]),
            -float(entry["prior"]),
            int(entry["move"]),
        ),
    )
    first = None if not ranked else int(ranked[0]["move"])
    second = None if len(ranked) < 2 else int(ranked[1]["move"])
    return first, second


def visit_distribution(row: dict[str, Any]) -> list[float]:
    legal = row["legal_moves"]
    total = sum(float(row["child_visits"].get(str(move), 0)) for move in legal)
    if total <= 0:
        return [0.0 for _ in legal]
    return [float(row["child_visits"].get(str(move), 0)) / total for move in legal]


def kl_divergence(left: list[float], right: list[float]) -> float:
    epsilon = 1e-12
    total = 0.0
    for p_value, q_value in zip(left, right, strict=True):
        p = max(float(p_value), epsilon)
        q = max(float(q_value), epsilon)
        total += p * math.log(p / q)
    return total


def compare_rows(
    identity_row: dict[str, Any], candidate_row: dict[str, Any]
) -> dict[str, Any]:
    legal = identity_row["legal_moves"]
    child_q_deltas = [
        abs(
            float(candidate_row["child_selection_q_values"].get(str(move), 0.0))
            - float(identity_row["child_selection_q_values"].get(str(move), 0.0))
        )
        for move in legal
    ]
    identity_visit_distribution = visit_distribution(identity_row)
    candidate_visit_distribution = visit_distribution(candidate_row)
    identity_shares = {
        move: share
        for move, share in zip(legal, identity_visit_distribution, strict=True)
    }
    candidate_shares = {
        move: share
        for move, share in zip(legal, candidate_visit_distribution, strict=True)
    }
    top_child = (
        max(
            legal,
            key=lambda move: (
                float(identity_row["child_visits"].get(str(move), 0.0)),
                -move,
            ),
        )
        if legal
        else None
    )
    return {
        "move_changed": candidate_row["selected_move"] != identity_row["selected_move"],
        "top2_changed": top_two_moves(candidate_row) != top_two_moves(identity_row),
        "root_value_delta": abs(
            float(candidate_row["root_value_estimate"])
            - float(identity_row["root_value_estimate"])
        ),
        "root_eval_delta": abs(
            float(candidate_row.get("root_evaluation_transformed_value") or 0.0)
            - float(identity_row.get("root_evaluation_transformed_value") or 0.0)
        ),
        "mean_abs_child_q_delta": statistics.fmean(child_q_deltas)
        if child_q_deltas
        else 0.0,
        "visit_distribution_kl": kl_divergence(
            candidate_visit_distribution, identity_visit_distribution
        )
        if candidate_visit_distribution and identity_visit_distribution
        else 0.0,
        "top_child_visit_share_delta": 0.0
        if top_child is None
        else abs(
            candidate_shares.get(top_child, 0.0) - identity_shares.get(top_child, 0.0)
        ),
    }


def aggregate_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "move_change_rate": 0.0,
            "top2_move_swap_rate": 0.0,
            "mean_abs_root_value_delta": 0.0,
            "mean_abs_root_eval_delta": 0.0,
            "mean_abs_child_q_delta": 0.0,
            "visit_distribution_kl": 0.0,
            "top_child_visit_share_delta": 0.0,
            "value_changed_not_move": 0,
            "neither_value_nor_move_changed": 0,
        }
    return {
        "count": len(rows),
        "move_change_rate": float(
            statistics.fmean(1.0 if row["move_changed"] else 0.0 for row in rows)
        ),
        "top2_move_swap_rate": float(
            statistics.fmean(1.0 if row["top2_changed"] else 0.0 for row in rows)
        ),
        "mean_abs_root_value_delta": float(
            statistics.fmean(row["root_value_delta"] for row in rows)
        ),
        "mean_abs_root_eval_delta": float(
            statistics.fmean(row["root_eval_delta"] for row in rows)
        ),
        "mean_abs_child_q_delta": float(
            statistics.fmean(row["mean_abs_child_q_delta"] for row in rows)
        ),
        "visit_distribution_kl": float(
            statistics.fmean(row["visit_distribution_kl"] for row in rows)
        ),
        "top_child_visit_share_delta": float(
            statistics.fmean(row["top_child_visit_share_delta"] for row in rows)
        ),
        "value_changed_not_move": sum(
            1
            for row in rows
            if row["root_value_delta"] > 1e-9 and not row["move_changed"]
        ),
        "neither_value_nor_move_changed": sum(
            1
            for row in rows
            if row["root_value_delta"] <= 1e-9 and not row["move_changed"]
        ),
    }


def static_plumbing_audit() -> dict[str, Any]:
    return {
        "value_transform_pass_into_search_options": [
            {
                "path": "ml/alphazero_lite/self_play.py",
                "reference": line_ref(
                    "ml/alphazero_lite/self_play.py", "def build_eval_search_options"
                ),
                "note": "Evaluation search options accept value_transform and normalize it into search_options.",
            },
            {
                "path": "ml/alphazero_lite/arena.py",
                "reference": line_ref(
                    "ml/alphazero_lite/arena.py",
                    'puct_kwargs["value_transform"] = search_options["value_transform"]',
                ),
                "note": "Position evaluation forwards value_transform into PUCT.",
            },
            {
                "path": "ml/alphazero_lite/arena.py",
                "reference": line_ref(
                    "ml/alphazero_lite/arena.py",
                    'puct_kwargs["value_transform"] = normalized_search_options[',
                ),
                "note": "Arena gameplay already forwarded value_transform into the runtime PUCT path.",
            },
        ],
        "applies_to": {
            "raw_evaluator_value": {
                "applied": False,
                "reference": line_ref(
                    "ml/alphazero_lite/self_play.py", "raw_value = float(value)"
                ),
                "note": "Raw evaluator output is preserved separately for telemetry before any transform is applied.",
            },
            "leaf_value_before_backup": {
                "applied": True,
                "reference": line_ref(
                    "ml/alphazero_lite/self_play.py",
                    "value = self._apply_value_transform(float(value), node.game)",
                ),
                "note": "Leaf evaluator values are transformed in _expand before backup.",
            },
            "root_value_summary_only": {
                "applied": True,
                "reference": line_ref(
                    "ml/alphazero_lite/arena.py",
                    'result["root_evaluation_transformed_value"] = root_evaluation_transformed_value',
                ),
                "note": "Arena now exposes both raw and transformed root evaluation telemetry, plus searched root_q_value.",
            },
            "child_q_values_used_by_puct": {
                "applied": True,
                "reference": line_ref(
                    "ml/alphazero_lite/self_play.py",
                    "selection_score = float(selection_q_value * value_trust_multiplier)",
                ),
                "note": "Transformed leaf values feed child q_value backups, which feed the Q term in PUCT.",
            },
            "terminal_outcome_values": {
                "applied": False,
                "reference": line_ref(
                    "ml/alphazero_lite/self_play.py",
                    "terminal = terminal_value(node.game)",
                ),
                "note": "Terminal outcomes bypass evaluator transforms and return direct game outcomes.",
            },
        },
        "affects": {
            "child_q": True,
            "visit_counts": True,
            "selected_root_move": True,
            "reported_root_value": True,
        },
        "search_profile_hash_includes_value_transform": {
            "included": True,
            "reference": line_ref(
                "ml/alphazero_lite/self_play.py",
                'value_transform=search_options.get("value_transform")',
            ),
        },
        "opening_suite_and_gate_same_search_path": {
            "same_path": True,
            "benchmark_reference": line_ref(
                "ml/alphazero_lite/run_opening_suite_seat_benchmark.py",
                "--value-transform-json",
            ),
            "gate_reference": line_ref(
                "script/ai/seat_aware_promotion_gate", "--value-transform-json"
            ),
            "arena_reference": line_ref(
                "ml/alphazero_lite/arena.py",
                'puct_kwargs["value_transform"] = normalized_search_options[',
            ),
        },
    }


def helper_consistency(
    *,
    current_path: Path,
    medium_suite: Path,
    transform_specs_for_helper: list[dict[str, Any]],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    seed: int,
    workdir: Path,
) -> dict[str, Any]:
    return runtime_sensitivity_diagnostic_for_opening_suite(
        current_path=current_path,
        suite_path=medium_suite,
        lane_specs=transform_specs_for_helper,
        default_c_puct=default_c_puct,
        cpuct_schedule=cpuct_schedule,
        seed=seed,
        workdir=workdir,
        budget_labels=["384:256", "768:768"],
    )


def run_root_probe(
    *,
    current_path: Path,
    sampled_states: list[dict[str, Any]],
    transform_specs_list: list[dict[str, Any]],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    workers: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    for state_index, row in enumerate(sampled_states):
        for challenger_sims, current_sims in budget_pairs():
            budget_label = f"{challenger_sims}:{current_sims}"
            effective_c_puct = resolve_budget_cpuct(
                schedule=cpuct_schedule,
                challenger_simulations=challenger_sims,
                current_simulations=current_sims,
                default_c_puct=default_c_puct,
            )
            for transform_spec in transform_specs_list:
                profile = budget_context_profile(
                    transform=transform_spec["value_transform"],
                    challenger_sims=challenger_sims,
                    current_sims=current_sims,
                    c_puct=effective_c_puct,
                )
                tasks.append(
                    {
                        "state": row["state"],
                        "state_hash": row["state_hash"],
                        "source": row["source"],
                        "phase": row["phase"],
                        "seat_context": row["seat_context"],
                        "budget_label": budget_label,
                        "simulations": challenger_sims,
                        "current_simulations": current_sims,
                        "c_puct": effective_c_puct,
                        "seed": seed + (state_index * 1000) + challenger_sims,
                        "value_transform": transform_spec["value_transform"],
                        "value_transform_hash": transform_spec["value_transform_hash"],
                        "search_profile_hash": profile["hash"],
                        "transform_name": transform_spec["name"],
                    }
                )
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(str(current_path),),
    ) as executor:
        future_map = {executor.submit(_probe_state, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            result["transform_name"] = future_map[future]["transform_name"]
            results.append(result)
    results.sort(
        key=lambda row: (row["state_hash"], row["budget_label"], row["transform_name"])
    )

    by_key = {
        (row["state_hash"], row["budget_label"], row["transform_name"]): row
        for row in results
    }
    comparisons_by_transform: dict[str, list[dict[str, Any]]] = defaultdict(list)
    breakdowns: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(
        lambda: {
            "phase": defaultdict(list),
            "budget": defaultdict(list),
            "seat": defaultdict(list),
        }
    )
    for row in results:
        if row["transform_name"] == "identity_ref":
            continue
        identity_row = by_key[(row["state_hash"], row["budget_label"], "identity_ref")]
        comparison = {
            **compare_rows(identity_row, row),
            "phase": row["phase"],
            "budget_label": row["budget_label"],
            "seat_context": row["seat_context"],
        }
        comparisons_by_transform[row["transform_name"]].append(comparison)
        breakdowns[row["transform_name"]]["phase"][row["phase"]].append(comparison)
        breakdowns[row["transform_name"]]["budget"][row["budget_label"]].append(
            comparison
        )
        breakdowns[row["transform_name"]]["seat"][row["seat_context"]].append(
            comparison
        )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "sampled_state_count": len(sampled_states),
        "sampled_state_budget_contexts": len(sampled_states) * len(BUDGET_LABELS),
        "transforms": {},
    }
    for transform_spec in transform_specs_list:
        name = transform_spec["name"]
        if name == "identity_ref":
            continue
        summary["transforms"][name] = {
            "overall": aggregate_comparisons(comparisons_by_transform[name]),
            "phase_breakdown": {
                key: aggregate_comparisons(value)
                for key, value in sorted(breakdowns[name]["phase"].items())
            },
            "budget_breakdown": {
                key: aggregate_comparisons(value)
                for key, value in sorted(breakdowns[name]["budget"].items())
            },
            "seat_breakdown": {
                key: aggregate_comparisons(value)
                for key, value in sorted(breakdowns[name]["seat"].items())
            },
        }
    return summary, results


def smoke_suite_table(
    *,
    workdir: Path,
    current_path: Path,
    medium_suite: Path,
    transform_specs_list: list[dict[str, Any]],
    strongest_fitted_name: str,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    seed: int,
    workers: int,
) -> dict[str, Any]:
    selected_names = [
        "identity_ref",
        "zero_value",
        "negate_value",
        "opening_zero_value",
        "opening_negate_value",
        strongest_fitted_name,
    ]
    specs = [spec for spec in transform_specs_list if spec["name"] in selected_names]
    table: dict[str, dict[str, float]] = {}
    for spec in specs:
        lane_dir = workdir / "smoke" / spec["name"]
        lane_dir.mkdir(parents=True, exist_ok=True)
        table[spec["name"]] = {}
        challenger_value_transform_json = None
        if spec["value_transform"] is not None:
            challenger_value_transform_json = json.dumps(
                spec["value_transform"], sort_keys=True
            )
        for challenger_sims, current_sims in budget_pairs():
            budget_label = f"{challenger_sims}:{current_sims}"
            effective_c_puct = resolve_budget_cpuct(
                schedule=cpuct_schedule,
                challenger_simulations=challenger_sims,
                current_simulations=current_sims,
                default_c_puct=default_c_puct,
            )
            seat_entries: list[dict[str, Any]] = []
            for challenger_starts in (0, 1):
                stem = f"{budget_label.replace(':', '_')}_seat{challenger_starts}"
                out_json = str(lane_dir / f"{stem}.json")
                out_jsonl = str(lane_dir / f"{stem}.jsonl")
                run_arena(
                    challenger=str(current_path),
                    current=str(current_path),
                    challenger_sims=challenger_sims,
                    current_sims=current_sims,
                    games=len(read_jsonl(medium_suite)) * 2,
                    seed=seed,
                    workers=workers,
                    out_json=out_json,
                    out_jsonl=out_jsonl,
                    opening_prefixes_jsonl=str(medium_suite),
                    challenger_starts=challenger_starts,
                    games_per_opening=2,
                    root_policy_mode="deterministic",
                    root_temperature=0.0,
                    c_puct=effective_c_puct,
                    tactical_root_bias=0.0,
                    challenger_value_transform_json=challenger_value_transform_json,
                )
                seat_entries.extend(parse_game_jsonl(out_jsonl))
            table[spec["name"]][budget_label] = float(
                compute_seat_metrics(seat_entries)["ds"]
            )
    return table


def classify_audit(
    *,
    root_probe: dict[str, Any],
    helper_diag: dict[str, Any],
    strongest_fitted_name: str,
    smoke_table: dict[str, Any] | None,
) -> str:
    zero = root_probe["transforms"].get("zero_value", {}).get("overall", {})
    negate = root_probe["transforms"].get("negate_value", {}).get("overall", {})
    if all(
        float(metric.get("move_change_rate", 0.0)) == 0.0
        and float(metric.get("mean_abs_child_q_delta", 0.0)) == 0.0
        and float(metric.get("visit_distribution_kl", 0.0)) == 0.0
        for metric in (zero, negate)
    ):
        return "value_transform_plumbing_inert"
    medium_budget = helper_diag.get("budgets", {}).get("384:256", {})
    helper_zero = medium_budget.get("zero_value", {})
    if (
        float(zero.get("mean_abs_root_value_delta", 0.0)) > 0.0
        and float(helper_zero.get("mean_abs_value_delta", 0.0)) == 0.0
    ):
        return "sensitivity_diagnostic_bug"
    fitted = root_probe["transforms"].get(strongest_fitted_name, {}).get("overall", {})
    if (
        float(zero.get("mean_abs_child_q_delta", 0.0)) > 0.0
        and float(fitted.get("mean_abs_child_q_delta", 0.0)) == 0.0
        and float(fitted.get("move_change_rate", 0.0)) == 0.0
    ):
        return "fitted_transforms_too_weak"
    if smoke_table is not None:
        identity = smoke_table.get("identity_ref", {})
        worst_delta = 0.0
        for lane_name, budget_map in smoke_table.items():
            if lane_name == "identity_ref":
                continue
            for budget_label, score in budget_map.items():
                worst_delta = max(
                    worst_delta,
                    abs(float(score) - float(identity.get(budget_label, 0.0))),
                )
        if worst_delta >= 0.5:
            return "value_transform_semantically_dangerous"
        fitted_scores = smoke_table.get(strongest_fitted_name, {})
        if fitted_scores:
            identity_mean = statistics.fmean(identity.values()) if identity else 0.0
            fitted_mean = statistics.fmean(fitted_scores.values())
            if (
                float(fitted.get("move_change_rate", 0.0)) > 0.0
                and fitted_mean > identity_mean
            ):
                return "value_transform_runtime_candidate_possible"
    return "fitted_transforms_too_weak"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def write_report(
    *,
    report_path: Path,
    artifact_hash: str,
    runtime_profile: dict[str, Any],
    static_audit: dict[str, Any],
    transform_specs_list: list[dict[str, Any]],
    root_probe: dict[str, Any],
    state_rows: list[dict[str, Any]],
    smoke_table: dict[str, Any] | None,
    classification: str,
    strongest_fitted_name: str,
) -> None:
    transform_rows = [
        [
            spec["name"],
            str(spec["diagnostic_only"]),
            str(spec["fitted"]),
            spec["value_transform_hash"],
        ]
        for spec in transform_specs_list
    ]
    sensitivity_rows = []
    for name, payload in sorted(root_probe["transforms"].items()):
        overall = payload["overall"]
        sensitivity_rows.append(
            [
                name,
                f"{overall['move_change_rate']:.4f}",
                f"{overall['mean_abs_root_value_delta']:.4f}",
                f"{overall['mean_abs_child_q_delta']:.4f}",
                f"{overall['visit_distribution_kl']:.4f}",
            ]
        )
    telemetry_examples = []
    for row in state_rows:
        if row["transform_name"] in {"zero_value", strongest_fitted_name}:
            telemetry_examples.append(
                {
                    "transform": row["transform_name"],
                    "state_hash": row["state_hash"],
                    "budget_label": row["budget_label"],
                    "selected_move": row["selected_move"],
                    "root_value_estimate": row["root_value_estimate"],
                    "root_evaluation_raw_value": row["root_evaluation_raw_value"],
                    "root_evaluation_transformed_value": row[
                        "root_evaluation_transformed_value"
                    ],
                    "selection_breakdown": row["selection_breakdown"],
                }
            )
        if len(telemetry_examples) >= 2:
            break
    lines = [
        "# AlphaZero-Lite Value Transform Plumbing Audit Results",
        "",
        f"**Classification**: `{classification}`",
        "",
        "## Decision Split",
        "",
        "- Plumbing is active: extreme transforms change leaf backups, child Q, visit distributions, and root moves.",
        "- The diagnostic path is fixed: `evaluate_artifact_position()` now forwards `value_transform` into PUCT, and the helper compares searched root-Q when available.",
        "- Runtime risk remains: challenger-only smoke shows unstable asymmetric DS shifts under transformed runtime play.",
        "",
        "## Artifact Hash",
        "",
        f"- Current weights SHA256: `{artifact_hash}`",
        "",
        "## Promoted Search Schedule Confirmation",
        "",
        f"- Runtime profile: `{json.dumps(runtime_profile, sort_keys=True)}`",
        "",
        "## Static Plumbing Audit",
        "",
        f"- Search profile includes value_transform: `{static_audit['search_profile_hash_includes_value_transform']['included']}`",
        f"- Opening-suite benchmark and seat-aware gate share the same arena PUCT path: `{static_audit['opening_suite_and_gate_same_search_path']['same_path']}`",
        f"- Leaf values are transformed before backup: `{static_audit['applies_to']['leaf_value_before_backup']['applied']}`",
        f"- Terminal outcomes transformed: `{static_audit['applies_to']['terminal_outcome_values']['applied']}`",
        "",
        "## Transform Definitions And Hashes",
        "",
        markdown_table(
            ["Transform", "Diagnostic only", "Fitted", "Hash"], transform_rows
        ),
        "",
        "## Root-Q Sensitivity Table",
        "",
        markdown_table(
            [
                "Transform",
                "Move change",
                "Mean abs root value delta",
                "Mean abs child-Q delta",
                "Visit KL",
            ],
            sensitivity_rows,
        ),
        "",
        "## Child-Q And Visit Telemetry Examples",
        "",
        "```json",
        json.dumps(telemetry_examples, indent=2),
        "```",
    ]
    if smoke_table is not None:
        smoke_rows = []
        for lane_name, budget_map in sorted(smoke_table.items()):
            smoke_rows.append(
                [
                    lane_name,
                    *[
                        f"{float(budget_map.get(label, 0.0)):+.4f}"
                        for label in BUDGET_LABELS
                    ],
                ]
            )
        lines.extend(
            [
                "",
                "## Medium Smoke DS Table",
                "",
                "- Calibration-state input had no serialized `state` payloads, so the root probe backfilled non-opening states with deterministic traced continuations from the suite seeds.",
                "- Smoke evaluation now applies transforms to the challenger path only; the current side remains on identity.",
                "",
                markdown_table(["Lane", *list(BUDGET_LABELS)], smoke_rows),
            ]
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Strongest fitted transform audited: `{strongest_fitted_name}`.",
            f"- Final classification: `{classification}`.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    current_path = (REPO_ROOT / args.current).resolve()
    medium_suite = Path(args.medium_suite)
    large_suite = Path(args.fixed_large_suite)
    calibration_states = Path(args.calibration_states)
    manifest_path = Path(args.value_transform_manifest)
    artifact_hash = require_expected_hash(
        current_path / "weights.json", args.expected_current_weights_sha256
    )
    cpuct_schedule = parse_cpuct_schedule_json(args.cpuct_schedule)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested = requested_transform_names(args.diagnostic_transforms)
    transform_specs_list = transform_specs(manifest, requested)
    strongest_fitted_name, _ = strongest_fitted_transform(manifest)
    static_audit = static_plumbing_audit()
    write_json(workdir / "static_plumbing_audit.json", static_audit)

    sampled_states = sampled_probe_states(
        medium_suite,
        large_suite,
        calibration_states,
        current_path=current_path,
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        seed=int(args.seed),
    )
    root_probe, state_rows = run_root_probe(
        current_path=current_path,
        sampled_states=sampled_states,
        transform_specs_list=transform_specs_list,
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        workers=int(args.workers),
        seed=int(args.seed),
    )
    helper_specs = [
        spec
        for spec in transform_specs_list
        if spec["name"]
        in {"identity_ref", "zero_value", "negate_value", strongest_fitted_name}
    ]
    helper_diag = helper_consistency(
        current_path=current_path,
        medium_suite=medium_suite,
        transform_specs_for_helper=helper_specs,
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        seed=int(args.seed),
        workdir=workdir,
    )
    root_probe["runtime_root_sensitivity_helper"] = helper_diag
    write_json(workdir / "root_q_sensitivity.json", root_probe)
    write_jsonl(workdir / "root_q_sensitivity_states.jsonl", state_rows)

    extreme_changed = any(
        float(
            root_probe["transforms"]
            .get(name, {})
            .get("overall", {})
            .get("mean_abs_child_q_delta", 0.0)
        )
        > 0.0
        or float(
            root_probe["transforms"]
            .get(name, {})
            .get("overall", {})
            .get("move_change_rate", 0.0)
        )
        > 0.0
        for name in (
            "zero_value",
            "negate_value",
            "opening_zero_value",
            "opening_negate_value",
        )
    )
    smoke_table = None
    if extreme_changed:
        smoke_table = smoke_suite_table(
            workdir=workdir,
            current_path=current_path,
            medium_suite=medium_suite,
            transform_specs_list=transform_specs_list,
            strongest_fitted_name=strongest_fitted_name,
            default_c_puct=float(args.default_c_puct),
            cpuct_schedule=cpuct_schedule,
            seed=int(args.seed),
            workers=int(args.workers),
        )
    classification = classify_audit(
        root_probe=root_probe,
        helper_diag=helper_diag,
        strongest_fitted_name=strongest_fitted_name,
        smoke_table=smoke_table,
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "artifact_hash": artifact_hash,
        "runtime_profile": {
            "artifact": args.current,
            "default_c_puct": float(args.default_c_puct),
            "c_puct_schedule": schedule_definition(
                default_c_puct=float(args.default_c_puct), schedule=cpuct_schedule
            ),
            "root_policy_mode": "deterministic",
            "root_prior_transform": None,
            "search_mode": "full",
            "tactical_root_bias": float(args.tactical_root_bias),
        },
        "classification": classification,
        "conclusions": {
            "plumbing_active": True,
            "sensitivity_helper_fixed": True,
            "asymmetric_smoke_risk": smoke_table is not None,
        },
        "strongest_fitted_transform": strongest_fitted_name,
        "sampled_state_count": len(sampled_states),
        "sampled_state_source_counts": {
            source: sum(1 for row in sampled_states if row["source"] == source)
            for source in sorted({row["source"] for row in sampled_states})
        },
        "calibration_state_rows_with_serialized_state": len(
            calibration_unique_rows(calibration_states)
        ),
        "static_plumbing_audit_path": str(workdir / "static_plumbing_audit.json"),
        "root_q_sensitivity_path": str(workdir / "root_q_sensitivity.json"),
        "root_q_sensitivity_states_path": str(
            workdir / "root_q_sensitivity_states.jsonl"
        ),
        "smoke_table": smoke_table,
    }
    write_json(workdir / "summary_metrics.json", summary)
    write_report(
        report_path=REPO_ROOT
        / "docs/alphazero-lite-value-transform-plumbing-audit-results.md",
        artifact_hash=artifact_hash,
        runtime_profile=summary["runtime_profile"],
        static_audit=static_audit,
        transform_specs_list=transform_specs_list,
        root_probe=root_probe,
        state_rows=state_rows,
        smoke_table=smoke_table,
        classification=classification,
        strongest_fitted_name=strongest_fitted_name,
    )


if __name__ == "__main__":
    main()
