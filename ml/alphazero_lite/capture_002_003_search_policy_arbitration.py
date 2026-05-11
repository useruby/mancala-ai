from __future__ import annotations

import argparse
import json
import math
from numbers import Real
from pathlib import Path


SCHEMA = "azlite_capture_002_003_search_policy_arbitration_v1"
ROW_IDS = ["capture_available-002", "capture_available-003"]
CLASSIFICATION_LABELS = [
    "policy_prior_gap",
    "value_backup_gap",
    "search_amplification_gap",
    "state_specific_rule_collision",
    "unresolved",
]
CLASSIFICATION_DECISIONS = {
    "policy_prior_gap": "write_search_adjustment_spec",
    "value_backup_gap": "write_value_backup_followup_spec",
    "search_amplification_gap": "write_search_adjustment_spec",
    "state_specific_rule_collision": "write_rule_collision_spec",
    "unresolved": "stop_unresolved",
}


def default_out_path(*, rebalanced_run_dir: Path) -> Path:
    return rebalanced_run_dir / "final" / "capture_002_003_search_policy_arbitration.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def raw_state_from_canonical_state(canonical_state: str | None) -> dict | None:
    if not isinstance(canonical_state, str) or not canonical_state:
        return None
    try:
        parsed = json.loads(canonical_state)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def resolve_full_state(*, row: dict) -> dict:
    parsed_canonical_state = raw_state_from_canonical_state(row.get("canonical_state")) or {}
    embedded_state = row.get("state") if isinstance(row.get("state"), dict) else {}

    resolved = dict(parsed_canonical_state)
    if "current_player" in embedded_state or "current_player" in parsed_canonical_state:
        resolved["current_player"] = embedded_state.get(
            "current_player",
            parsed_canonical_state.get("current_player", 0),
        )

    for key in ("player_pits", "opponent_pits"):
        embedded_value = embedded_state.get(key)
        canonical_value = parsed_canonical_state.get(key)
        if key in embedded_state:
            resolved[key] = list(embedded_value) if isinstance(embedded_value, (list, tuple)) else embedded_value
        elif isinstance(canonical_value, (list, tuple)):
            resolved[key] = list(canonical_value)

    for key, value in embedded_state.items():
        resolved.setdefault(key, value)

    return resolved


def _state_source_has_required_board_shape(state: object) -> bool:
    if not isinstance(state, dict):
        return False

    current_player = state.get("current_player")
    if not _is_integer_value(current_player):
        return False

    player_pits = state.get("player_pits")
    opponent_pits = state.get("opponent_pits")
    if not isinstance(player_pits, (list, tuple)) or not isinstance(opponent_pits, (list, tuple)):
        return False

    if not player_pits or not opponent_pits or len(player_pits) != len(opponent_pits):
        return False

    for pits in (player_pits, opponent_pits):
        if any(not _is_integer_value(seeds) or seeds < 0 for seeds in pits):
            return False

    return True


def validated_diagnostic_state(*, row: dict) -> dict:
    resolved_state = resolve_full_state(row=row)

    if not _state_source_has_required_board_shape(resolved_state):
        row_id = row.get("id", "<unknown>")
        raise ValueError(
            f"row {row_id} must provide a usable state from embedded state or canonical_state with "
            "current_player, player_pits, and opponent_pits"
        )

    return resolved_state


def _simulate_reference_move(*, player_pits: list[int], opponent_pits: list[int], reference_move: int) -> dict:
    if not (0 <= reference_move < len(player_pits)):
        return {"extra_turn_available": False, "capture_legal": False}

    seeds = player_pits[reference_move]
    if seeds <= 0:
        return {"extra_turn_available": False, "capture_legal": False}

    player_after = list(player_pits)
    opponent_after = list(opponent_pits)
    player_after[reference_move] = 0

    track = [
        *[("player", pit_index) for pit_index in range(len(player_after))],
        ("store", None),
        *[("opponent", pit_index) for pit_index in reversed(range(len(opponent_after)))],
    ]
    position = reference_move
    last_side = None
    last_index = None

    for _ in range(seeds):
        position = (position + 1) % len(track)
        last_side, last_index = track[position]
        if last_side == "player":
            player_after[last_index] += 1
        elif last_side == "opponent":
            opponent_after[last_index] += 1

    extra_turn_available = last_side == "store"
    capture_legal = False
    if last_side == "player" and last_index is not None and player_after[last_index] == 1:
        opposite_index = len(opponent_after) - 1 - last_index
        capture_legal = 0 <= opposite_index < len(opponent_after) and opponent_after[opposite_index] > 0

    return {
        "extra_turn_available": extra_turn_available,
        "capture_legal": capture_legal,
    }


def probe_artifact_position(**kwargs) -> dict:
    from ml.alphazero_lite.search_interaction_diagnostic import probe_artifact_position as _probe

    return _probe(**kwargs)


def load_arena_module():
    import importlib

    return importlib.import_module("ml.alphazero_lite.arena")


def load_selected_artifact(*, run_dir: Path) -> dict:
    manifest_path = run_dir / "selection" / "selection_manifest.json"
    manifest = load_json(manifest_path)

    selected_target = manifest.get("selected_target")
    selected_artifact = manifest.get("selected_artifact")
    if isinstance(selected_target, str) and selected_target:
        return {
            "path": selected_target,
            "provenance_source": "selection_manifest.selected_target",
            "selected_target": selected_target,
            "selected_artifact": (
                selected_artifact
                if isinstance(selected_artifact, str) and selected_artifact
                else None
            ),
        }

    if isinstance(selected_artifact, str) and selected_artifact:
        return {
            "path": selected_artifact,
            "provenance_source": "selection_manifest.selected_artifact",
            "selected_target": None,
            "selected_artifact": selected_artifact,
        }

    raise ValueError(f"selected artifact could not be resolved for {run_dir}")


def resolve_rows(*, rows_by_id: dict[str, dict]) -> list[dict]:
    resolved = []
    for row_id in ROW_IDS:
        row = rows_by_id.get(row_id)
        if row is None:
            raise ValueError(f"missing required row id: {row_id}")

        missing_fields = [
            field
            for field in ["id", "canonical_state", "legal_moves", "reference_move", "state"]
            if row.get(field) is None
        ]
        if missing_fields:
            raise ValueError(
                f"row {row_id} missing required fields: {', '.join(missing_fields)}"
            )

        if row["id"] != row_id:
            raise ValueError(f"row {row_id} has mismatched id: {row['id']}")

        if not isinstance(row["legal_moves"], list):
            raise ValueError(f"row {row_id} legal_moves must be a list")

        if not row["legal_moves"]:
            raise ValueError(f"row {row_id} has empty legal_moves")

        if any(not _is_integer_value(move) for move in row["legal_moves"]):
            raise ValueError(f"row {row_id} legal_moves entries must be integers")

        if any(move < 0 for move in row["legal_moves"]):
            raise ValueError(f"row {row_id} legal_moves entries must be non-negative")

        _validate_move_id(
            value=row["reference_move"],
            field_name=f"row {row_id} reference_move",
        )

        if row["reference_move"] not in row["legal_moves"]:
            raise ValueError(f"row {row_id} reference_move must be present in legal_moves")

        resolved.append(row)

    return resolved


def _distribution(values: list[float], legal_moves: list[int]) -> dict[str, float] | None:
    if (
        not values
        or any(move >= len(values) for move in legal_moves)
        or any(values[move] is None for move in legal_moves)
    ):
        return None
    return {str(move): round(float(values[move]), 4) for move in legal_moves}


def _is_numeric_value(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _is_integer_value(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_move_id(*, value: object, field_name: str) -> None:
    if not _is_integer_value(value):
        raise ValueError(f"{field_name} must be an integer")

    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validated_optional_numeric_sequence(*, value: object, field_name: str) -> list[float | None]:
    if value is None:
        return []

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple")

    for index, item in enumerate(value):
        if item is not None and not _is_numeric_value(item):
            raise ValueError(f"{field_name} entries must be numeric or None (index {index})")

    return list(value)


def _validated_optional_numeric_scalar(*, value: object, field_name: str) -> float | None:
    if value is None:
        return None

    if not _is_numeric_value(value):
        raise ValueError(f"{field_name} must be numeric")

    return float(value)


def _validated_child_stats_container(value: object) -> list[dict]:
    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError("child_stats must be a list of dicts")

    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"child_stats entries must be dicts (index {index})")

    return value


def _child_stat_maps(
    child_stats: list[dict],
    legal_moves: list[int],
) -> tuple[dict[str, int] | None, dict[str, float] | None, list[dict] | None]:
    if not child_stats:
        return None, None, None

    required_fields = ["move", "visits", "q_value"]
    seen_moves: set[int] = set()
    for index, item in enumerate(child_stats):
        if not isinstance(item, dict):
            raise ValueError(f"child_stats entries must be dicts (index {index})")

        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            raise ValueError(
                "child_stats "
                f"entry at index {index} missing required fields: {', '.join(missing_fields)}"
            )

        _validate_move_id(
            value=item["move"],
            field_name=f"child_stats entry at index {index} move",
        )

        if item["move"] not in legal_moves:
            raise ValueError(
                f"child_stats entry at index {index} move must be present in legal_moves"
            )

        if item["move"] in seen_moves:
            return None, None, None

        seen_moves.add(int(item["move"]))

        if not _is_integer_value(item["visits"]):
            raise ValueError(f"child_stats entry at index {index} visits must be an integer")

        if item["visits"] < 0:
            raise ValueError(f"child_stats entry at index {index} visits must be non-negative")

        if not _is_numeric_value(item["q_value"]):
            raise ValueError(f"child_stats entry at index {index} q_value must be numeric")

    missing_moves = [move for move in legal_moves if move not in seen_moves]
    if missing_moves:
        return None, None, None

    visits = {str(item["move"]): int(item["visits"]) for item in child_stats}
    q_values = {
        str(item["move"]): round(float(item["q_value"]), 4) for item in child_stats
    }
    normalized = [
        {
            "move": int(item["move"]),
            "visits": int(item["visits"]),
            "q_value": round(float(item["q_value"]), 4),
        }
        for item in child_stats
    ]
    return visits, q_values, normalized


def build_row_views(*, row: dict, probe_summary: dict) -> dict:
    legal_moves = row["legal_moves"]
    reference_key = str(row["reference_move"])
    selected_move = probe_summary.get("selected_move")
    if selected_move is not None:
        _validate_move_id(value=selected_move, field_name="selected_move")

        if selected_move not in legal_moves:
            raise ValueError("selected_move must be present in legal_moves")

    selected_key = None if selected_move is None else str(selected_move)
    raw_policy = _distribution(
        _validated_optional_numeric_sequence(
            value=probe_summary.get("policy"), field_name="policy"
        ),
        legal_moves,
    )
    search_distribution = _distribution(
        _validated_optional_numeric_sequence(
            value=probe_summary.get("visits"), field_name="visits"
        ),
        legal_moves,
    )
    _, per_move_q_values, child_stats = _child_stat_maps(
        _validated_child_stats_container(probe_summary.get("child_stats")),
        legal_moves,
    )

    policy_top_move = (
        None if raw_policy is None else max(legal_moves, key=lambda move: raw_policy[str(move)])
    )
    reference_policy = None if raw_policy is None else raw_policy.get(reference_key, 0.0)
    selected_policy = (
        None
        if raw_policy is None or selected_key is None
        else raw_policy.get(selected_key, 0.0)
    )
    q_reference = None if per_move_q_values is None else per_move_q_values.get(reference_key)
    q_selected = (
        None
        if per_move_q_values is None or selected_key is None
        else per_move_q_values.get(selected_key)
    )

    missing_fields = []
    if per_move_q_values is None:
        missing_fields.extend(
            [
                "value_view.per_child_q_values",
                "value_view.reference_move_q_value",
                "value_view.selected_move_q_value",
                "value_view.selected_minus_reference_q_margin",
                "search_view.child_stats",
            ]
        )
        if probe_summary.get("child_stats"):
            missing_fields.append("search_view.child_stats_partial")

    if raw_policy is None:
        missing_fields.append("policy_view.raw_policy_distribution")

    probe_value = _validated_optional_numeric_scalar(
        value=probe_summary.get("value"), field_name="value"
    )
    root_value_estimate = None if probe_value is None else round(probe_value, 4)
    if root_value_estimate is None:
        missing_fields.append("value_view.root_value_estimate")

    total_visits = None if search_distribution is None else sum(search_distribution.values())
    reference_move_visit_share = None
    selected_move_visit_share = None
    if search_distribution is None:
        missing_fields.extend(
            [
                "search_view.visit_distribution",
                "search_view.reference_move_visit_share",
            ]
        )
        if selected_key is not None:
            missing_fields.append("search_view.selected_move_visit_share")
    elif total_visits <= 0:
        missing_fields.extend(
            [
                "search_view.reference_move_visit_share",
                "search_view.selected_move_visit_share",
            ]
        )
    else:
        reference_move_visit_share = round(
            search_distribution.get(reference_key, 0.0) / total_visits, 4
        )
        selected_move_visit_share = (
            None
            if selected_key is None
            else round(search_distribution.get(selected_key, 0.0) / total_visits, 4)
        )

    return {
        "row_id": row["id"],
        "canonical_state": row["canonical_state"],
        "legal_moves": list(legal_moves),
        "reference_move": row["reference_move"],
        "policy_view": {
            "raw_policy_distribution": raw_policy,
            "top_move": policy_top_move,
            "reference_move_probability": reference_policy,
            "selected_minus_reference_margin": (
                None
                if selected_policy is None
                else round(selected_policy - reference_policy, 4)
            ),
        },
        "value_view": {
            "root_value_estimate": root_value_estimate,
            "per_child_q_values": per_move_q_values,
            "reference_move_q_value": q_reference,
            "selected_move_q_value": q_selected,
            "selected_minus_reference_q_margin": (
                None
                if q_reference is None or q_selected is None
                else round(q_selected - q_reference, 4)
            ),
        },
        "search_view": {
            "searched_selected_move": selected_move,
            "visit_distribution": search_distribution,
            "reference_move_visit_share": reference_move_visit_share,
            "selected_move_visit_share": selected_move_visit_share,
            "child_stats": child_stats,
            "child_stats_complete": child_stats is not None,
            "missing_fields": missing_fields,
        },
    }


def compute_rule_features(*, row: dict) -> dict:
    state = validated_diagnostic_state(row=row)
    reference_move = int(row["reference_move"])
    current_player = int(state.get("current_player", 0))
    player_pits = list(state.get("player_pits") or [])
    opponent_pits = list(state.get("opponent_pits") or [])

    if current_player == 0:
        side_to_move_pits = player_pits
        defending_pits = opponent_pits
        starvation_target_pits = opponent_pits
    else:
        side_to_move_pits = opponent_pits
        defending_pits = player_pits
        starvation_target_pits = player_pits

    move_features = _simulate_reference_move(
        player_pits=side_to_move_pits,
        opponent_pits=defending_pits,
        reference_move=reference_move,
    )

    non_empty_opponent_pits = sum(1 for seeds_in_pit in starvation_target_pits if seeds_in_pit > 0)
    starvation_risk = non_empty_opponent_pits <= 2
    starvation_shape = "sparse" if starvation_risk else "distributed"

    return {
        "side_to_move": current_player,
        "capture_legal": move_features["capture_legal"],
        "extra_turn_available": move_features["extra_turn_available"],
        "starvation_shape": starvation_shape,
        "starvation_risk": starvation_risk,
    }


def build_rows_payload(
    *,
    selected_artifact: dict,
    rows: list[dict],
    simulations: int,
    seed: int,
    c_puct: float,
    search_options: dict,
) -> dict[str, dict]:
    payload = {}
    probe_modes = [
        ("policy_only", "policy_only"),
        ("value_only", "value_only"),
        ("full_search", "full"),
    ]
    for row in rows:
        resolved_state = validated_diagnostic_state(row=row)
        probe_views = {}
        for probe_key, ablation_mode in probe_modes:
            probe_summary = probe_artifact_position(
                artifact_path=selected_artifact["path"],
                state=resolved_state,
                simulations=simulations,
                seed=seed,
                c_puct=c_puct,
                search_options=search_options,
                ablation_mode=ablation_mode,
            )
            probe_views[probe_key] = build_row_views(row=row, probe_summary=probe_summary)

        payload[row["id"]] = {
            **probe_views["full_search"],
            "probe_views": probe_views,
            "rule_features": compute_rule_features(row=row),
        }
    return payload


def build_paired_comparison(*, rows: dict[str, dict]) -> dict:
    row_002 = rows["capture_available-002"]
    row_003 = rows["capture_available-003"]
    return {
        "row_ids": list(ROW_IDS),
        "reference_moves": {
            row_id: rows[row_id]["reference_move"] for row_id in ROW_IDS
        },
        "policy_top_moves": {
            row_id: rows[row_id]["policy_view"]["top_move"] for row_id in ROW_IDS
        },
        "searched_selected_moves": {
            row_id: rows[row_id]["search_view"]["searched_selected_move"]
            for row_id in ROW_IDS
        },
        "q_margins": {
            row_id: rows[row_id]["value_view"]["selected_minus_reference_q_margin"]
            for row_id in ROW_IDS
        },
        "visit_shares": {
            row_id: {
                "reference": rows[row_id]["search_view"]["reference_move_visit_share"],
                "selected": rows[row_id]["search_view"]["selected_move_visit_share"],
            }
            for row_id in ROW_IDS
        },
        "rule_feature_differences": {
            key: [
                row_002.get("rule_features", {}).get(key),
                row_003.get("rule_features", {}).get(key),
            ]
            for key in {
                **row_002.get("rule_features", {}),
                **row_003.get("rule_features", {}),
            }
            if row_002.get("rule_features", {}).get(key)
            != row_003.get("rule_features", {}).get(key)
        },
        "missing_child_stats": {
            row_id: list(rows[row_id]["search_view"].get("missing_fields") or [])
            for row_id in ROW_IDS
        },
    }


def classify_paired_comparison(*, comparison: dict) -> dict:
    reference_moves = comparison["reference_moves"]
    top_moves = comparison["policy_top_moves"]
    q_margins = comparison["q_margins"]
    visit_shares = comparison["visit_shares"]
    searched_selected_moves = comparison["searched_selected_moves"]
    rule_differences = comparison["rule_feature_differences"]
    reference_move_002 = reference_moves["capture_available-002"]
    reference_move_003 = reference_moves["capture_available-003"]
    top_move_002 = top_moves["capture_available-002"]
    top_move_003 = top_moves["capture_available-003"]
    q_margin_002 = q_margins["capture_available-002"]
    q_margin_003 = q_margins["capture_available-003"]
    searched_selected_move_002 = searched_selected_moves["capture_available-002"]
    searched_selected_move_003 = searched_selected_moves["capture_available-003"]
    visit_share_002_reference = visit_shares["capture_available-002"]["reference"]
    visit_share_002_selected = visit_shares["capture_available-002"]["selected"]
    visit_share_003_reference = visit_shares["capture_available-003"]["reference"]
    visit_share_003_selected = visit_shares["capture_available-003"]["selected"]

    search_amplification_inputs = (
        searched_selected_move_002,
        visit_share_002_reference,
        visit_share_002_selected,
        searched_selected_move_003,
        visit_share_003_reference,
        visit_share_003_selected,
    )

    if (
        top_move_002 is not None
        and top_move_003 is not None
        and top_move_002 != reference_move_002
        and top_move_003 == reference_move_003
    ):
        return {
            "classification": "policy_prior_gap",
            "evidence_summary": "002 policy top move misses the reference move while 003 policy top move matches it.",
        }

    if (
        top_move_002 is not None
        and q_margin_002 is not None
        and q_margin_003 is not None
        and top_move_002 == reference_move_002
        and q_margin_002 > 0.15
        and q_margin_003 <= 0.05
    ):
        return {
            "classification": "value_backup_gap",
            "evidence_summary": "002 policy supports the reference move, but 002 child Q values favor the wrong move much more strongly than 003.",
        }

    if (
        top_move_002 == reference_move_002
        and ((q_margin_002 or 0.0) <= 0.05)
        and any(value is None for value in search_amplification_inputs)
    ):
        return {
            "classification": "unresolved",
            "evidence_summary": "Required search-amplification inputs are missing, so the paired evidence cannot isolate a supported failure mechanism.",
        }

    if (
        top_move_002 is not None
        and q_margin_002 is not None
        and top_move_002 == reference_move_002
        and q_margin_002 <= 0.05
        and all(value is not None for value in search_amplification_inputs)
        and searched_selected_move_002 != reference_move_002
        and visit_share_002_selected > visit_share_002_reference + 0.15
        and searched_selected_move_003 == reference_move_003
        and visit_share_003_selected <= visit_share_003_reference + 0.15
    ):
        return {
            "classification": "search_amplification_gap",
            "evidence_summary": "002 policy and Q values are near neutral, but search visits amplify the wrong move into the final choice while 003 does not.",
        }

    if rule_differences:
        return {
            "classification": "state_specific_rule_collision",
            "evidence_summary": f"002 and 003 diverge on tactical rule features: {sorted(rule_differences)}.",
        }

    return {
        "classification": "unresolved",
        "evidence_summary": "Required data is present, but the paired evidence does not isolate a supported failure mechanism.",
    }


def decision_for_classification(classification: str) -> str:
    return CLASSIFICATION_DECISIONS[classification]


def build_payload(
    *,
    selected_artifact: dict,
    source_artifacts: dict,
    settings: dict,
    rows: dict[str, dict],
) -> dict:
    paired_comparison = build_paired_comparison(rows=rows)
    classification = classify_paired_comparison(comparison=paired_comparison)
    return {
        "schema": SCHEMA,
        "selected_artifact": selected_artifact,
        "source_artifacts": source_artifacts,
        "settings": settings,
        "rows": rows,
        "paired_comparison": paired_comparison,
        "classification": classification,
        "decision": decision_for_classification(classification["classification"]),
    }


def load_rows_from_run(*, run_dir: Path) -> dict[str, dict]:
    forensics = load_json(run_dir / "final" / "selected_candidate_forensics.json")
    rows = ((forensics.get("systems") or {}).get("current") or {}).get("rows") or []
    rows_by_id = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"forensic rows entries must be dicts (index {index})")

        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"forensic row at index {index} must contain non-empty id")

        if row_id in rows_by_id:
            raise ValueError(f"duplicate forensic row id: {row_id}")

        rows_by_id[row_id] = row

    return rows_by_id


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose search policy arbitration for capture_available-002 and capture_available-003"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=384)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--c-puct", type=float, default=1.25)
    args = parser.parse_args(argv)

    if args.simulations <= 0:
        parser.error("--simulations must be > 0")

    if not math.isfinite(args.c_puct):
        parser.error("--c-puct must be finite")

    if args.c_puct <= 0:
        parser.error("--c-puct must be > 0")

    return args


def validate_base_config(*, base_config: Path) -> None:
    if not base_config.is_file():
        raise ValueError(f"base config must exist and be readable: {base_config}")

    try:
        base_config.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"base config must exist and be readable: {base_config}") from error


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_base_config(base_config=args.base_config)
    arena = load_arena_module()
    selected_artifact = load_selected_artifact(run_dir=args.run_dir)
    rows_by_id = load_rows_from_run(run_dir=args.run_dir)
    rows = resolve_rows(rows_by_id=rows_by_id)
    search_options = dict(arena.build_eval_search_options())
    search_options["reuse_subtree"] = False
    settings = {
        "base_config_path": str(args.base_config),
        "row_ids": list(ROW_IDS),
        "search_settings": {
            "c_puct": args.c_puct,
            **search_options,
        },
        "seeds": [args.seed, args.seed],
        "simulation_count": args.simulations,
    }
    payload_rows = build_rows_payload(
        selected_artifact=selected_artifact,
        rows=rows,
        simulations=args.simulations,
        seed=args.seed,
        c_puct=args.c_puct,
        search_options=search_options,
    )
    payload = build_payload(
        selected_artifact=selected_artifact,
        source_artifacts={"base_config": str(args.base_config)},
        settings=settings,
        rows=payload_rows,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact_path": str(args.out),
                "schema": SCHEMA,
                "decision": payload["decision"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
