#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.build_train_only_forensic_suite_from_selfplay import decode_state
from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.corrected_guard_kill_gate import (
    DEFAULT_FALLBACK_REFERENCE_ARTIFACT,
    DEFAULT_REFERENCE_ARTIFACT,
)
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state
from ml.alphazero_lite.self_play import (
    PUCT,
    build_eval_search_options,
    build_policy_target_from_distribution,
    build_search_options,
    policy_from_visits,
    value_from_classic_mcts_root,
    visits_from_classic_mcts_root,
)


DEFAULT_CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
DEFAULT_CONFIG_PATH = Path(
    "ml/alphazero_lite/configs/exp_v3_guard_safe_opening_selected_w1.json"
)
DEFAULT_SELF_PLAY_PATH = Path(
    "/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_versions/"
    "exp-v3-guard-safe-opening-selected-w1-iter1/self_play.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path("/tmp/azlite_opening_017_puct_target_generation_audit")
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_ROOT / "puct_target_generation_audit_summary.json"
DEFAULT_REPORT_PATH = Path(
    "docs/alphazero-lite-opening-017-puct-target-generation-audit-results.md"
)
SCHEMA = "azlite_opening_017_puct_target_generation_audit_v1"

AUDIT_ROW_IDS = (
    "opening_plies_1_8-017",
    "capture_available-002",
    "capture_available-003",
    "capture_available-006",
    "capture_available-007",
    "capture_available-008",
)
SEEDS = (41, 42, 43, 101, 202, 303)
SIMULATIONS = 192
HIGHER_SIMULATIONS = (384, 1200)
CLASSIC_BUDGETS = (1200, 2400)
C_PUCT = 1.25
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.3
TEMPERATURE = 1.1
TEMPERATURE_LATE = 0.15
TEMPERATURE_THRESHOLD = 12
POLICY_MASS_WEAK_THRESHOLD = 0.4
REPRODUCTION_MASS_TOLERANCE = 0.12


@dataclass(frozen=True)
class VariantSpec:
    name: str
    simulations: int
    dirichlet_epsilon: float
    policy_target_mode: str
    player_mode: str = "puct"
    search_options_kind: str = "self_play"
    note: str = ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--current-artifact", type=Path, default=DEFAULT_CURRENT_ARTIFACT
    )
    parser.add_argument("--self-play", type=Path, default=DEFAULT_SELF_PLAY_PATH)
    parser.add_argument(
        "--reference-artifact", type=Path, default=DEFAULT_REFERENCE_ARTIFACT
    )
    parser.add_argument(
        "--fallback-reference-artifact",
        type=Path,
        default=DEFAULT_FALLBACK_REFERENCE_ARTIFACT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object row in {path}")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def merged_reference_rows(
    reference_artifact_path: Path, fallback_reference_artifact_path: Path
) -> dict[str, dict[str, Any]]:
    primary = load_json(reference_artifact_path)
    fallback = load_json(fallback_reference_artifact_path)
    rows: dict[str, dict[str, Any]] = {}
    for row in list(fallback.get("rows") or []):
        row_id = row.get("id")
        if isinstance(row_id, str) and row_id:
            rows[row_id] = {**row, "reference_artifact_kind": "fallback"}
    for row in list(primary.get("rows") or []):
        row_id = row.get("id")
        if isinstance(row_id, str) and row_id:
            rows[row_id] = {**row, "reference_artifact_kind": "primary"}
    missing = [row_id for row_id in AUDIT_ROW_IDS if row_id not in rows]
    if missing:
        raise FileNotFoundError(
            "missing corrected reference rows: " + ", ".join(sorted(missing))
        )
    return rows


def top_policy_move(policy: list[float], legal_moves: list[int]) -> int | None:
    if not legal_moves:
        return None
    return min(legal_moves, key=lambda move: (-float(policy[move]), move))


def policy_entropy(policy: list[float], legal_moves: list[int]) -> float:
    entropy = 0.0
    for move in legal_moves:
        probability = float(policy[move])
        if probability > 0.0:
            entropy -= probability * math.log(probability, 2)
    return round(entropy, 4)


def rounded_distribution(policy: list[float]) -> list[float]:
    return [round(float(value), 6) for value in policy]


def rounded_move_map(values: dict[int, float]) -> dict[str, float]:
    return {str(move): round(float(value), 6) for move, value in sorted(values.items())}


def temperature_for_ply(ply: int) -> float:
    return TEMPERATURE if int(ply) < TEMPERATURE_THRESHOLD else TEMPERATURE_LATE


def dominant_ply(counter: Counter[int]) -> int:
    if not counter:
        return 0
    return min(counter.items(), key=lambda item: (-item[1], item[0]))[0]


def search_options_for(kind: str) -> dict[str, Any]:
    if kind == "eval_control":
        return dict(build_eval_search_options())
    if kind == "self_play":
        return dict(build_search_options(reuse_subtree=True))
    raise ValueError(f"unsupported search options kind: {kind}")


def variant_specs() -> list[VariantSpec]:
    return [
        VariantSpec(
            name="baseline_self_play_target",
            simulations=SIMULATIONS,
            dirichlet_epsilon=DIRICHLET_EPSILON,
            policy_target_mode="sharpened",
            search_options_kind="self_play",
            note="tree_reuse_not_applicable_for_isolated_root",
        ),
        VariantSpec(
            name="no_dirichlet",
            simulations=SIMULATIONS,
            dirichlet_epsilon=0.0,
            policy_target_mode="sharpened",
            search_options_kind="self_play",
            note="tree_reuse_not_applicable_for_isolated_root",
        ),
        VariantSpec(
            name="unsharpened_policy_target",
            simulations=SIMULATIONS,
            dirichlet_epsilon=DIRICHLET_EPSILON,
            policy_target_mode="default",
            search_options_kind="self_play",
            note="tree_reuse_not_applicable_for_isolated_root",
        ),
        VariantSpec(
            name="no_dirichlet_unsharpened",
            simulations=SIMULATIONS,
            dirichlet_epsilon=0.0,
            policy_target_mode="default",
            search_options_kind="self_play",
            note="tree_reuse_not_applicable_for_isolated_root",
        ),
        VariantSpec(
            name="higher_sims_384",
            simulations=384,
            dirichlet_epsilon=0.0,
            policy_target_mode="default",
            search_options_kind="self_play",
            note="tree_reuse_not_applicable_for_isolated_root",
        ),
        VariantSpec(
            name="higher_sims_1200",
            simulations=1200,
            dirichlet_epsilon=0.0,
            policy_target_mode="default",
            search_options_kind="self_play",
            note="tree_reuse_not_applicable_for_isolated_root",
        ),
        VariantSpec(
            name="eval_search_control",
            simulations=384,
            dirichlet_epsilon=0.0,
            policy_target_mode="default",
            search_options_kind="eval_control",
            note="build_eval_search_options control",
        ),
        VariantSpec(
            name="classic_teacher_control_1200",
            simulations=1200,
            dirichlet_epsilon=0.0,
            policy_target_mode="default",
            player_mode="classic_mcts",
            note="classic_teacher_control budget 1200",
        ),
        VariantSpec(
            name="classic_teacher_control_2400",
            simulations=2400,
            dirichlet_epsilon=0.0,
            policy_target_mode="default",
            player_mode="classic_mcts",
            note="classic_teacher_control budget 2400",
        ),
    ]


def reference_teacher_policy(reference_row: dict[str, Any]) -> list[float]:
    policy = [0.0] * 6
    child_stats = list(reference_row.get("child_stats") or [])
    total = sum(int(child.get("visits", 0)) for child in child_stats)
    if total <= 0:
        return policy
    for child in child_stats:
        move = int(child["move"])
        policy[move] = float(child["visits"]) / float(total)
    return policy


def extract_audited_rows(
    *, reference_rows: dict[str, dict[str, Any]], self_play_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    canonical_to_row_id = {
        str(reference_rows[row_id]["canonical_state"]): row_id
        for row_id in AUDIT_ROW_IDS
    }
    aggregates: dict[str, dict[str, Any]] = {
        row_id: {
            "count": 0,
            "policy_sum": [0.0] * 6,
            "move_indices": Counter(),
            "players": Counter(),
            "teacher_sources": Counter(),
            "policy_target_modes": Counter(),
            "policy_target_actual_modes": Counter(),
            "search_profile_hashes": Counter(),
            "teacher_search_profile_hashes": Counter(),
            "teacher_root_summary_rows": 0,
            "line_numbers": [],
        }
        for row_id in AUDIT_ROW_IDS
    }
    canonical_mismatches: dict[str, str] = {}

    for line_number, row in enumerate(self_play_rows, start=1):
        raw_state = decode_state(list(row["state"]))
        canonical = canonical_state_key(raw_state)
        row_id = canonical_to_row_id.get(canonical)
        if row_id is None:
            continue
        reference_canonical = str(reference_rows[row_id]["canonical_state"])
        if canonical != reference_canonical:
            canonical_mismatches[row_id] = "decoded_self_play_state_mismatch"
            continue
        aggregate = aggregates[row_id]
        aggregate["count"] += 1
        aggregate["line_numbers"].append(line_number)
        policy = list(row.get("policy") or [])
        if len(policy) != 6:
            raise ValueError(
                f"unexpected policy width for {row_id} at line {line_number}"
            )
        for move, value in enumerate(policy):
            aggregate["policy_sum"][move] += float(value)
        move_index = row.get("move_index")
        if isinstance(move_index, int) and not isinstance(move_index, bool):
            aggregate["move_indices"][int(move_index)] += 1
        player = row.get("player")
        if isinstance(player, int) and not isinstance(player, bool):
            aggregate["players"][int(player)] += 1
        teacher_source = row.get("teacher_source")
        if isinstance(teacher_source, str) and teacher_source:
            aggregate["teacher_sources"][teacher_source] += 1
        policy_target_mode = row.get("policy_target_mode")
        if isinstance(policy_target_mode, str) and policy_target_mode:
            aggregate["policy_target_modes"][policy_target_mode] += 1
        policy_target_actual_mode = row.get("policy_target_actual_mode")
        if isinstance(policy_target_actual_mode, str) and policy_target_actual_mode:
            aggregate["policy_target_actual_modes"][policy_target_actual_mode] += 1
        search_profile_hash = row.get("search_profile_hash")
        if isinstance(search_profile_hash, str) and search_profile_hash:
            aggregate["search_profile_hashes"][search_profile_hash] += 1
        teacher_search_profile_hash = row.get("teacher_search_profile_hash")
        if isinstance(teacher_search_profile_hash, str) and teacher_search_profile_hash:
            aggregate["teacher_search_profile_hashes"][teacher_search_profile_hash] += 1
        if isinstance(row.get("teacher_root_summary"), dict):
            aggregate["teacher_root_summary_rows"] += 1

    audited_rows: list[dict[str, Any]] = []
    audited_row_map: dict[str, dict[str, Any]] = {}
    for row_id in AUDIT_ROW_IDS:
        reference_row = reference_rows.get(row_id)
        if reference_row is None:
            raise FileNotFoundError(f"missing corrected reference row: {row_id}")
        state = dict(reference_row.get("state") or {})
        if not state:
            raise FileNotFoundError(f"missing corrected reference state: {row_id}")
        game = KalahGame.from_state(state)
        legal_moves = [int(move) for move in game.possible_moves()]
        corrected_reference_move = int(reference_row["reference_move"])
        if corrected_reference_move not in legal_moves:
            raise ValueError(
                f"corrected reference move {corrected_reference_move} is illegal for {row_id}"
            )
        aggregate = aggregates[row_id]
        averaged_policy = None
        pr36_top_target_move = None
        pr36_corrected_reference_mass = None
        if int(aggregate["count"]) > 0:
            averaged_policy = [
                float(value) / float(aggregate["count"])
                for value in aggregate["policy_sum"]
            ]
            pr36_top_target_move = top_policy_move(averaged_policy, legal_moves)
            pr36_corrected_reference_mass = round(
                float(averaged_policy[corrected_reference_move]), 4
            )
        move_consequences = []
        for move in legal_moves:
            consequence = move_consequence_for_state(state, move)
            move_consequences.append(
                {
                    "move": int(move),
                    "gives_extra_turn": bool(consequence["gives_extra_turn"]),
                    "produces_capture": bool(consequence["produces_capture"]),
                    "capture_count": int(consequence["capture_count"]),
                    "immediate_store_delta": int(consequence["store_delta_immediate"]),
                    "side_to_move_after": int(consequence["resulting_side_to_move"]),
                    "game_over_after_move": bool(consequence["game_over_after_move"]),
                }
            )
        notes: list[str] = []
        if canonical_mismatches.get(row_id):
            notes.append(canonical_mismatches[row_id])
        if aggregate["teacher_root_summary_rows"] <= 0:
            notes.append("pr36_rows_do_not_store_teacher_root_summary")
        if aggregate["count"] <= 0:
            notes.append("no_pr36_self_play_match")
        audited_row = {
            "row_id": row_id,
            "canonical_state_hash": str(reference_row["canonical_state"]),
            "corrected_reference_move": corrected_reference_move,
            "legal_moves": legal_moves,
            "per_move_consequences": move_consequences,
            "pr36_self_play_count": int(aggregate["count"]),
            "pr36_averaged_policy_target": None
            if averaged_policy is None
            else [round(float(value), 4) for value in averaged_policy],
            "pr36_top_target_move": pr36_top_target_move,
            "pr36_corrected_reference_mass": pr36_corrected_reference_mass,
            "reference_source": str(reference_row.get("reference_source") or "unknown"),
            "reference_artifact_kind": str(
                reference_row.get("reference_artifact_kind") or "unknown"
            ),
            "dominant_ply": dominant_ply(aggregate["move_indices"]),
            "pr36_context": {
                "move_index_counts": {
                    str(key): int(value)
                    for key, value in sorted(aggregate["move_indices"].items())
                },
                "player_counts": {
                    str(key): int(value)
                    for key, value in sorted(aggregate["players"].items())
                },
                "teacher_source_counts": {
                    str(key): int(value)
                    for key, value in sorted(aggregate["teacher_sources"].items())
                },
                "policy_target_mode_counts": {
                    str(key): int(value)
                    for key, value in sorted(aggregate["policy_target_modes"].items())
                },
                "policy_target_actual_mode_counts": {
                    str(key): int(value)
                    for key, value in sorted(
                        aggregate["policy_target_actual_modes"].items()
                    )
                },
                "search_profile_hashes": {
                    str(key): int(value)
                    for key, value in sorted(aggregate["search_profile_hashes"].items())
                },
                "teacher_search_profile_hashes": {
                    str(key): int(value)
                    for key, value in sorted(
                        aggregate["teacher_search_profile_hashes"].items()
                    )
                },
                "teacher_root_summary_rows": int(
                    aggregate["teacher_root_summary_rows"]
                ),
                "row_canonicalization_exact_match": bool(
                    aggregate["count"] <= 0 or row_id not in canonical_mismatches
                ),
                "root_visit_counts_stored": False,
                "policy_target_generated_before_temperature_sampling": True,
                "tree_reuse_enabled_in_search_profile": True,
            },
            "notes": notes,
        }
        audited_rows.append(audited_row)
        audited_row_map[row_id] = audited_row
    return audited_rows, audited_row_map, canonical_to_row_id


def run_puct_probe(
    *,
    evaluator: ArtifactEvaluator,
    state: dict[str, Any],
    seed: int,
    simulations: int,
    corrected_reference_move: int,
    temperature: float,
    dirichlet_epsilon: float,
    policy_target_mode: str,
    search_options: dict[str, Any],
) -> dict[str, Any]:
    game = KalahGame.from_state(state)
    legal_moves = [int(move) for move in game.possible_moves()]
    search = PUCT(
        evaluator=evaluator,
        simulations=int(simulations),
        c_puct=float(C_PUCT),
        rng=random.Random(int(seed)),
        fpu_mode=str(search_options["fpu_mode"]),
        reuse_subtree=bool(search_options["reuse_subtree"]),
        normalize_values=bool(search_options["normalize_values"]),
        root_policy_mode=str(search_options["root_policy_mode"]),
        tactical_root_bias=float(search_options["tactical_root_bias"]),
        value_trust_schedule=search_options.get("value_trust_schedule"),
    )
    visits, root = search.run(
        game,
        dirichlet_alpha=(DIRICHLET_ALPHA if float(dirichlet_epsilon) > 0.0 else None),
        dirichlet_epsilon=float(dirichlet_epsilon),
    )
    root_summary = search.root_summary()
    raw_policy = policy_from_visits(
        visits, legal_moves=legal_moves, temperature=temperature
    )
    target_policy = build_policy_target_from_distribution(
        raw_policy, mode=policy_target_mode
    )
    selection_breakdown = dict(root_summary.get("selection_breakdown") or {})
    selection_entries = list(selection_breakdown.get("moves") or [])
    entry_by_move = {int(entry["move"]): entry for entry in selection_entries}
    root_prior = {
        int(move): float(entry_by_move.get(move, {}).get("prior", 0.0))
        for move in legal_moves
    }
    root_q = {
        int(move): float(entry_by_move.get(move, {}).get("q_value", 0.0))
        for move in legal_moves
    }
    root_u = {
        int(move): float(entry_by_move.get(move, {}).get("u_component", 0.0))
        for move in legal_moves
    }
    raw_top_move = top_policy_move(raw_policy, legal_moves)
    target_top_move = top_policy_move(target_policy, legal_moves)
    notes = ["tree_reuse_not_applicable_for_isolated_root"]
    if raw_top_move != target_top_move:
        notes.append("sharpening_changes_top_move")
    return {
        "raw_visit_counts": [int(value) for value in visits.tolist()],
        "unsharpened_visit_policy": rounded_distribution(raw_policy),
        "policy_target": rounded_distribution(target_policy),
        "top_move_before_sharpening": raw_top_move,
        "top_move_after_sharpening": target_top_move,
        "corrected_reference_mass_before_sharpening": round(
            float(raw_policy[corrected_reference_move]), 6
        ),
        "corrected_reference_mass_after_sharpening": round(
            float(target_policy[corrected_reference_move]), 6
        ),
        "sharpening_changes_top_move": bool(raw_top_move != target_top_move),
        "search_value": round(float(root.q_value), 6),
        "root_prior_per_move": rounded_move_map(root_prior),
        "root_q_per_move": rounded_move_map(root_q),
        "root_u_per_move": rounded_move_map(root_u),
        "root_selected_move": selection_breakdown.get("selected_move"),
        "selection_breakdown": selection_entries,
        "visit_snapshots": list(root_summary.get("visit_snapshots") or []),
        "root_prior_telemetry": root_summary.get("root_prior_telemetry"),
        "notes": notes,
    }


def run_classic_probe(
    *,
    state: dict[str, Any],
    seed: int,
    simulations: int,
    corrected_reference_move: int,
    temperature: float,
) -> dict[str, Any]:
    game = KalahGame.from_state(state)
    legal_moves = [int(move) for move in game.possible_moves()]
    search = ClassicMCTS(game.clone(), simulations=int(simulations), seed=int(seed))
    root = search.search_root()
    visits = np.asarray(visits_from_classic_mcts_root(root), dtype=np.float32)
    raw_policy = policy_from_visits(
        visits, legal_moves=legal_moves, temperature=temperature
    )
    selected_move = search.root_summary().get("selected_move")
    return {
        "raw_visit_counts": [int(value) for value in visits.tolist()],
        "unsharpened_visit_policy": rounded_distribution(raw_policy),
        "policy_target": rounded_distribution(raw_policy),
        "top_move_before_sharpening": top_policy_move(raw_policy, legal_moves),
        "top_move_after_sharpening": top_policy_move(raw_policy, legal_moves),
        "corrected_reference_mass_before_sharpening": round(
            float(raw_policy[corrected_reference_move]), 6
        ),
        "corrected_reference_mass_after_sharpening": round(
            float(raw_policy[corrected_reference_move]), 6
        ),
        "sharpening_changes_top_move": False,
        "search_value": round(float(value_from_classic_mcts_root(root)), 6),
        "root_prior_per_move": None,
        "root_q_per_move": None,
        "root_u_per_move": None,
        "root_selected_move": selected_move,
        "selection_breakdown": list(search.root_summary().get("child_stats") or []),
        "visit_snapshots": [],
        "root_prior_telemetry": None,
        "notes": ["classic_teacher_control"],
    }


def run_variant_for_row(
    *,
    evaluator: ArtifactEvaluator,
    row: dict[str, Any],
    spec: VariantSpec,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    temperature = temperature_for_ply(int(row["dominant_ply"]))
    search_options = None
    if spec.player_mode == "puct":
        search_options = search_options_for(spec.search_options_kind)
    for seed in SEEDS:
        if spec.player_mode == "classic_mcts":
            probe = run_classic_probe(
                state=dict(row["state"]),
                seed=seed,
                simulations=spec.simulations,
                corrected_reference_move=int(row["corrected_reference_move"]),
                temperature=temperature,
            )
        else:
            assert search_options is not None
            probe = run_puct_probe(
                evaluator=evaluator,
                state=dict(row["state"]),
                seed=seed,
                simulations=spec.simulations,
                corrected_reference_move=int(row["corrected_reference_move"]),
                temperature=temperature,
                dirichlet_epsilon=spec.dirichlet_epsilon,
                policy_target_mode=spec.policy_target_mode,
                search_options=search_options,
            )
        pr36_top = row.get("pr36_top_target_move")
        corrected_move = int(row["corrected_reference_move"])
        pr36_mass = row.get("pr36_corrected_reference_mass")
        reproduced = False
        if pr36_top is not None:
            if int(pr36_top) != corrected_move:
                reproduced = int(probe["top_move_after_sharpening"]) == int(pr36_top)
            elif pr36_mass is not None:
                reproduced = (
                    int(probe["top_move_after_sharpening"]) == corrected_move
                    and abs(
                        float(probe["corrected_reference_mass_after_sharpening"])
                        - float(pr36_mass)
                    )
                    <= REPRODUCTION_MASS_TOLERANCE
                )
        results.append(
            {
                "row_id": row["row_id"],
                "seed": int(seed),
                "simulations": int(spec.simulations),
                "dirichlet_epsilon": float(spec.dirichlet_epsilon),
                "policy_target_mode": spec.policy_target_mode,
                "player_mode": spec.player_mode,
                "search_options_kind": spec.search_options_kind,
                "temperature": round(float(temperature), 4),
                "top_target_move": probe["top_move_after_sharpening"],
                "corrected_reference_mass_raw": probe[
                    "corrected_reference_mass_before_sharpening"
                ],
                "corrected_reference_mass_sharpened": probe[
                    "corrected_reference_mass_after_sharpening"
                ],
                "top_move_before_sharpening": probe["top_move_before_sharpening"],
                "top_move_after_sharpening": probe["top_move_after_sharpening"],
                "sharpening_changes_top_move": probe["sharpening_changes_top_move"],
                "search_value": probe["search_value"],
                "reproduces_pr36_bad_target": bool(reproduced),
                "root_prior_per_move": probe["root_prior_per_move"],
                "root_q_per_move": probe["root_q_per_move"],
                "root_u_per_move": probe["root_u_per_move"],
                "raw_visit_counts": probe["raw_visit_counts"],
                "unsharpened_visit_policy": probe["unsharpened_visit_policy"],
                "policy_target": probe["policy_target"],
                "root_selected_move": probe["root_selected_move"],
                "selection_breakdown": probe["selection_breakdown"],
                "visit_snapshots": probe["visit_snapshots"],
                "root_prior_telemetry": probe["root_prior_telemetry"],
                "notes": ", ".join([spec.note, *probe["notes"]]).strip(", "),
            }
        )
    return results


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def majority_move(moves: list[int | None]) -> int | None:
    filtered = [int(move) for move in moves if move is not None]
    if not filtered:
        return None
    counts = Counter(filtered)
    return min(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def summarize_variant_runs(
    *, row: dict[str, Any], variant_name: str, runs: list[dict[str, Any]]
) -> dict[str, Any]:
    corrected_move = int(row["corrected_reference_move"])
    top_hits = [run for run in runs if int(run["top_target_move"]) == corrected_move]
    masses = [float(run["corrected_reference_mass_sharpened"]) for run in runs]
    entropies = [
        policy_entropy(run["policy_target"], row["legal_moves"]) for run in runs
    ]
    diagnosis = "selected_target_matches_corrected_reference"
    if len(top_hits) <= 0:
        diagnosis = "selected_target_misses_corrected_reference"
    elif len(top_hits) < len(runs):
        diagnosis = "selected_target_unstable_across_seeds"
    return {
        "row_id": row["row_id"],
        "variant": variant_name,
        "simulations": int(runs[0]["simulations"]),
        "seeds": [int(run["seed"]) for run in runs],
        "dirichlet_epsilon": float(runs[0]["dirichlet_epsilon"]),
        "policy_target_mode": str(runs[0]["policy_target_mode"]),
        "corrected_reference_top_rate": round(
            float(len(top_hits)) / float(len(runs)), 4
        ),
        "average_corrected_reference_mass": round(mean(masses), 4),
        "majority_top_move": majority_move([run["top_target_move"] for run in runs]),
        "target_entropy": round(mean(entropies), 4),
        "selected_target_matches_corrected_reference": bool(
            majority_move([run["top_target_move"] for run in runs]) == corrected_move
        ),
        "diagnosis": diagnosis,
    }


def is_problem_row(row: dict[str, Any]) -> bool:
    top_move = row.get("pr36_top_target_move")
    mass = row.get("pr36_corrected_reference_mass")
    corrected_move = int(row["corrected_reference_move"])
    if top_move is None or mass is None:
        return False
    return int(top_move) != corrected_move or float(mass) < POLICY_MASS_WEAK_THRESHOLD


def build_isolated_vs_pr36_comparison(
    *,
    audited_rows: list[dict[str, Any]],
    variant_summaries: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for row in audited_rows:
        row_id = row["row_id"]
        corrected_move = int(row["corrected_reference_move"])
        baseline = variant_summaries[row_id]["baseline_self_play_target"]
        no_dirichlet = variant_summaries[row_id]["no_dirichlet"]
        no_dirichlet_unsharpened = variant_summaries[row_id]["no_dirichlet_unsharpened"]
        higher_384 = variant_summaries[row_id]["higher_sims_384"]
        higher_1200 = variant_summaries[row_id]["higher_sims_1200"]
        eval_control = variant_summaries[row_id]["eval_search_control"]
        classic_1200 = variant_summaries[row_id]["classic_teacher_control_1200"]
        classic_2400 = variant_summaries[row_id]["classic_teacher_control_2400"]
        pr36_top = row.get("pr36_top_target_move")
        pr36_mass = row.get("pr36_corrected_reference_mass")
        baseline_reproduces_bad = bool(
            pr36_top is not None
            and int(pr36_top) != corrected_move
            and baseline["majority_top_move"] == int(pr36_top)
        )
        classification = "inconsistent_or_unreproduced"
        notes: list[str] = []
        if not is_problem_row(row):
            notes.append("pr36_control_row_or_reference_supported")
        elif baseline_reproduces_bad:
            classification = "isolated_reproduces_bad_target"
        elif baseline["majority_top_move"] == corrected_move:
            classification = "bad_only_in_full_trajectory"
            notes.extend(
                [
                    "pr36_rows_store_no_teacher_root_summary",
                    "root_visit_counts_not_stored_in_self_play_row",
                    "policy_target_built_before_temperature_sampling",
                    "canonical_state_matches_reference_exactly",
                ]
            )
        elif no_dirichlet["majority_top_move"] == corrected_move:
            classification = "bad_only_with_dirichlet"
        elif (
            no_dirichlet["majority_top_move"] != corrected_move
            and no_dirichlet_unsharpened["majority_top_move"] == corrected_move
        ):
            classification = "bad_only_after_sharpening"
        elif no_dirichlet_unsharpened["majority_top_move"] != corrected_move and (
            higher_384["majority_top_move"] == corrected_move
            or higher_1200["majority_top_move"] == corrected_move
        ):
            classification = "bad_due_to_low_sims"
        elif (
            higher_1200["majority_top_move"] != corrected_move
            and eval_control["majority_top_move"] != corrected_move
            and classic_1200["majority_top_move"] != corrected_move
            and classic_2400["majority_top_move"] != corrected_move
        ):
            classification = "bad_due_to_puct_teacher_disagreement"
        notes.extend(
            [
                f"pr36_top_target_move={pr36_top}",
                f"pr36_corrected_reference_mass={pr36_mass}",
                f"baseline_majority_top_move={baseline['majority_top_move']}",
                f"baseline_avg_corrected_reference_mass={baseline['average_corrected_reference_mass']}",
            ]
        )
        comparisons.append(
            {
                "row_id": row_id,
                "classification": classification,
                "pr36_top_target_move": pr36_top,
                "pr36_corrected_reference_mass": pr36_mass,
                "isolated_baseline_majority_top_move": baseline["majority_top_move"],
                "isolated_baseline_average_corrected_reference_mass": baseline[
                    "average_corrected_reference_mass"
                ],
                "baseline_reproduces_pr36_bad_target": baseline_reproduces_bad,
                "notes": notes,
            }
        )
    return comparisons


def summarize_fix_candidate(
    *,
    name: str,
    rows: list[dict[str, Any]],
    per_row_summaries: dict[str, dict[str, Any]],
    implementation_risk: str,
    notes: str,
    recommended: bool,
) -> dict[str, Any]:
    total_rows = len(rows)
    top_hits = sum(
        1
        for row in rows
        if per_row_summaries[row["row_id"]]["majority_top_move"]
        == int(row["corrected_reference_move"])
    )
    avg_mass = mean(
        [
            float(per_row_summaries[row["row_id"]]["average_corrected_reference_mass"])
            for row in rows
        ]
    )
    avg_entropy = mean(
        [float(per_row_summaries[row["row_id"]]["target_entropy"]) for row in rows]
    )
    regressions: list[str] = []
    for row in rows:
        summary = per_row_summaries[row["row_id"]]
        pr36_top = row.get("pr36_top_target_move")
        corrected_move = int(row["corrected_reference_move"])
        if (
            pr36_top == corrected_move
            and summary["majority_top_move"] != corrected_move
        ):
            regressions.append(f"{row['row_id']}: loses corrected top move")
        pr36_mass = row.get("pr36_corrected_reference_mass")
        if pr36_mass is not None and (
            float(summary["average_corrected_reference_mass"]) + 0.05 < float(pr36_mass)
        ):
            regressions.append(f"{row['row_id']}: corrected mass drops")
    return {
        "fix_candidate": name,
        "rows_evaluated": total_rows,
        "corrected_reference_top_rate": round(float(top_hits) / float(total_rows), 4)
        if total_rows > 0
        else 0.0,
        "average_corrected_reference_mass": round(avg_mass, 4),
        "target_entropy": round(avg_entropy, 4),
        "regressions": regressions,
        "implementation_risk": implementation_risk,
        "recommended": bool(recommended),
        "notes": notes,
    }


def choose_decision(
    *,
    audited_rows: list[dict[str, Any]],
    variant_summaries: dict[str, dict[str, dict[str, Any]]],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    problem_rows = [row for row in audited_rows if is_problem_row(row)]
    problem_count = len(problem_rows)

    def fixed_count(variant_name: str) -> int:
        return sum(
            1
            for row in problem_rows
            if variant_summaries[row["row_id"]][variant_name]["majority_top_move"]
            == int(row["corrected_reference_move"])
        )

    baseline_correct = sum(
        1
        for row in problem_rows
        if variant_summaries[row["row_id"]]["baseline_self_play_target"][
            "majority_top_move"
        ]
        == int(row["corrected_reference_move"])
    )

    no_dirichlet_fixed = fixed_count("no_dirichlet")
    unsharpened_fixed = fixed_count("unsharpened_policy_target")
    higher_384_fixed = fixed_count("higher_sims_384")
    higher_1200_fixed = fixed_count("higher_sims_1200")
    teacher_fallback_fixed = problem_count
    high_sims_still_bad = sum(
        1
        for row in problem_rows
        if variant_summaries[row["row_id"]]["higher_sims_1200"]["majority_top_move"]
        != int(row["corrected_reference_move"])
    )

    if problem_count > 0 and baseline_correct > (problem_count / 2.0):
        return {
            "classification": "trajectory_or_tree_reuse_target_bug",
            "next_action": "instrument full self-play trajectory target capture, especially tree reuse and canonicalization.",
            "notes": "Isolated-root baseline stays corrected on most PR #36 problem rows, so the bad rows are more consistent with trajectory-scoped behavior than isolated-root target generation.",
        }
    if problem_count > 0 and no_dirichlet_fixed > (problem_count / 2.0):
        return {
            "classification": "dirichlet_noise_leaks_into_targets",
            "next_action": "modify self-play target generation so training targets are built from de-noised or separate no-noise search while exploration noise remains only for action sampling.",
            "notes": f"Removing Dirichlet noise restores the corrected reference as the majority target on {no_dirichlet_fixed}/{problem_count} problem rows.",
        }
    if problem_count > 0 and unsharpened_fixed > (problem_count / 2.0):
        return {
            "classification": "target_sharpening_amplifies_wrong_visits",
            "next_action": "test unsharpened opening targets or phase-aware sharpening.",
            "notes": f"Leaving targets unsharpened restores the corrected reference as the majority target on {unsharpened_fixed}/{problem_count} problem rows.",
        }
    higher_sim_fixed = max(higher_384_fixed, higher_1200_fixed)
    if problem_count > 0 and higher_sim_fixed > (problem_count / 2.0):
        chosen_budget = 384 if higher_384_fixed >= higher_1200_fixed else 1200
        return {
            "classification": "low_simulation_target_noise",
            "next_action": "run a small self-play target-quality lane with 384 or 1200 sims for opening positions only.",
            "notes": f"Raising the no-noise unsharpened budget to {chosen_budget} restores the corrected reference as the majority target on {higher_sim_fixed}/{problem_count} problem rows.",
        }
    if problem_count > 0 and teacher_fallback_fixed > (problem_count / 2.0):
        return {
            "classification": "forensic_teacher_fallback_needed",
            "next_action": "add a corrected-reference teacher fallback for exact forensic-state hits during self-play data generation.",
            "notes": "The corrected forensic teacher target resolves every audited problem row, while the search-based variants do not fix a majority.",
        }
    if problem_count > 0 and high_sims_still_bad > (problem_count / 2.0):
        return {
            "classification": "puct_teacher_disagreement",
            "next_action": "align self-play teacher with corrected reference teacher or exclude those rows from self-play target training.",
            "notes": f"Even at 1200 simulations with no Dirichlet noise and no sharpening, PUCT still disagrees with the corrected reference on {high_sims_still_bad}/{problem_count} problem rows.",
        }
    return {
        "classification": "target_generation_inconclusive",
        "next_action": "full trace of one PR #36 game path through opening_plies_1_8-017 to target writeout.",
        "notes": "The isolated-root ablations did not isolate a single dominant cause across the audited rows.",
    }


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-lite Opening 017 PUCT Target Generation Audit Results",
        "",
        "## 1. Context",
        "",
        "- PR #38 concluded the opening-017 failure chain is `self_play_target_generation_broken` and recommended auditing PUCT target generation instead of adding more replay weight.",
        "- This audit stayed diagnostic-only: no training, no arena, no promotion, no new corrective replay artifact, and no replay-weight sweep.",
        "- The audit reused corrected references from `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`, the current artifact under `storage/ai/alphazero_lite/current`, and PR #36 self-play from `/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_versions/exp-v3-guard-safe-opening-selected-w1-iter1/self_play.jsonl`.",
        "",
        "## 2. Why Replay Anchoring Failed",
        "",
        "- PR #38 already showed the corrective artifact statically matched the intended rows, so the failure was not a stale-reference or duplicate-conflict issue.",
        "- The corrected patch still failed the corrected guard kill gate at epoch 4 for every tested replay weight, which means the underlying self-play targets remained strong enough to reintroduce the bad branch.",
        "- That shifts the audit from replay composition to how self-play PUCT constructed the original targets for `opening_plies_1_8-017` and its descendants.",
        "",
        "## 3. Audited Rows and Corrected References",
        "",
        "| row_id | corrected_reference_move | pr36_self_play_count | pr36_top_target_move | pr36_corrected_reference_mass | legal_moves | notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["audited_rows"]:
        lines.append(
            f"| {row['row_id']} | {row['corrected_reference_move']} | {row['pr36_self_play_count']} | {row['pr36_top_target_move'] if row['pr36_top_target_move'] is not None else '-'} | {format_float(row['pr36_corrected_reference_mass'])} | `{json.dumps(row['legal_moves'])}` | {', '.join(row['notes']) if row['notes'] else 'ok'} |"
        )
    lines.extend(
        [
            "",
            "## 4. PR #36 Self-Play Target Reproduction",
            "",
            "| row_id | seed | simulations | dirichlet_epsilon | policy_target_mode | top_target_move | corrected_reference_mass_raw | corrected_reference_mass_sharpened | search_value | reproduces_pr36_bad_target | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["baseline_reproduction_rows"]:
        lines.append(
            f"| {row['row_id']} | {row['seed']} | {row['simulations']} | {format_float(row['dirichlet_epsilon'])} | {row['policy_target_mode']} | {row['top_target_move']} | {format_float(row['corrected_reference_mass_raw'])} | {format_float(row['corrected_reference_mass_sharpened'])} | {format_float(row['search_value'])} | {str(bool(row['reproduces_pr36_bad_target'])).lower()} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 5. Target-Generation Ablation Matrix",
            "",
            "| row_id | variant | simulations | seeds | dirichlet_epsilon | policy_target_mode | corrected_reference_top_rate | average_corrected_reference_mass | majority_top_move | target_entropy | diagnosis |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["ablation_rows"]:
        seed_text = ",".join(str(seed) for seed in row["seeds"])
        lines.append(
            f"| {row['row_id']} | {row['variant']} | {row['simulations']} | `{seed_text}` | {format_float(row['dirichlet_epsilon'])} | {row['policy_target_mode']} | {format_float(row['corrected_reference_top_rate'])} | {format_float(row['average_corrected_reference_mass'])} | {row['majority_top_move'] if row['majority_top_move'] is not None else '-'} | {format_float(row['target_entropy'])} | {row['diagnosis']} |"
        )
    lines.extend(
        [
            "",
            "## 6. Isolated-Root vs Full-Trajectory Comparison",
            "",
        ]
    )
    for row in summary["isolated_vs_pr36_rows"]:
        lines.append(
            f"- `{row['row_id']}`: `{row['classification']}`. Notes: {', '.join(row['notes'])}."
        )
    lines.extend(
        [
            "",
            "## 7. Diagnostic Target-Generation Fix Candidates",
            "",
            "| fix_candidate | rows_evaluated | corrected_reference_top_rate | average_corrected_reference_mass | regressions | implementation_risk | recommended | notes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["fix_candidate_rows"]:
        regressions = "-" if not row["regressions"] else "; ".join(row["regressions"])
        lines.append(
            f"| {row['fix_candidate']} | {row['rows_evaluated']} | {format_float(row['corrected_reference_top_rate'])} | {format_float(row['average_corrected_reference_mass'])} | {regressions} | {row['implementation_risk']} | {str(bool(row['recommended'])).lower()} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 8. Interpretation",
            "",
            f"- Classification: `{summary['decision']['classification']}`.",
            f"- Evidence: {summary['decision']['notes']}",
            f"- PR #36 self-play row canonicalization against corrected references: `{str(bool(summary['integrity']['canonical_state_match'])).lower()}`.",
            f"- PR #36 row-level root telemetry available for audited rows: `{str(bool(summary['integrity']['teacher_root_summary_present'])).lower()}`.",
            "",
            "## 9. Exactly One Recommended Next Action",
            "",
            f"Recommendation: **{summary['decision']['next_action']}**",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    current_artifact = resolve_path(root, args.current_artifact)
    self_play_path = resolve_path(root, args.self_play)
    reference_artifact = resolve_path(root, args.reference_artifact)
    fallback_reference_artifact = resolve_path(root, args.fallback_reference_artifact)
    output_root = resolve_path(root, args.output_root)
    summary_path = resolve_path(root, args.summary_path)
    report_path = resolve_path(root, args.report_path)

    if not self_play_path.exists():
        raise FileNotFoundError(f"missing PR #36 self-play artifact: {self_play_path}")
    if not current_artifact.exists():
        raise FileNotFoundError(f"missing current artifact: {current_artifact}")

    output_root.mkdir(parents=True, exist_ok=True)

    config = load_json(resolve_path(root, args.config))
    reference_rows = merged_reference_rows(
        reference_artifact, fallback_reference_artifact
    )
    self_play_rows = read_jsonl(self_play_path)
    audited_rows, audited_row_map, _canonical_map = extract_audited_rows(
        reference_rows=reference_rows,
        self_play_rows=self_play_rows,
    )
    for row in audited_rows:
        row["state"] = dict(reference_rows[row["row_id"]]["state"])

    evaluator = ArtifactEvaluator(current_artifact)
    variant_runs: dict[str, dict[str, list[dict[str, Any]]]] = {
        row_id: {} for row_id in AUDIT_ROW_IDS
    }
    for spec in variant_specs():
        for row in audited_rows:
            variant_runs[row["row_id"]][spec.name] = run_variant_for_row(
                evaluator=evaluator,
                row=row,
                spec=spec,
            )

    variant_summaries: dict[str, dict[str, dict[str, Any]]] = {
        row_id: {} for row_id in AUDIT_ROW_IDS
    }
    ablation_rows: list[dict[str, Any]] = []
    baseline_reproduction_rows: list[dict[str, Any]] = []
    for row in audited_rows:
        row_id = row["row_id"]
        for spec in variant_specs():
            summary = summarize_variant_runs(
                row=row,
                variant_name=spec.name,
                runs=variant_runs[row_id][spec.name],
            )
            variant_summaries[row_id][spec.name] = summary
            ablation_rows.append(summary)
            if spec.name == "baseline_self_play_target":
                for run in variant_runs[row_id][spec.name]:
                    baseline_reproduction_rows.append(run)

    isolated_vs_pr36_rows = build_isolated_vs_pr36_comparison(
        audited_rows=audited_rows,
        variant_summaries=variant_summaries,
    )

    no_noise_candidate = {
        row["row_id"]: variant_summaries[row["row_id"]]["no_dirichlet"]
        for row in audited_rows
    }
    unsharpened_candidate = {
        row["row_id"]: variant_summaries[row["row_id"]]["unsharpened_policy_target"]
        for row in audited_rows
    }
    higher_sim_variant = "higher_sims_384"
    higher_384_top_rate = mean(
        [
            variant_summaries[row["row_id"]]["higher_sims_384"][
                "corrected_reference_top_rate"
            ]
            for row in audited_rows
        ]
    )
    higher_1200_top_rate = mean(
        [
            variant_summaries[row["row_id"]]["higher_sims_1200"][
                "corrected_reference_top_rate"
            ]
            for row in audited_rows
        ]
    )
    if higher_1200_top_rate > higher_384_top_rate:
        higher_sim_variant = "higher_sims_1200"
    higher_sims_candidate = {
        row["row_id"]: variant_summaries[row["row_id"]][higher_sim_variant]
        for row in audited_rows
    }
    teacher_candidate = {}
    for row in audited_rows:
        reference_row = reference_rows[row["row_id"]]
        teacher_policy = reference_teacher_policy(reference_row)
        teacher_candidate[row["row_id"]] = {
            "majority_top_move": top_policy_move(teacher_policy, row["legal_moves"]),
            "average_corrected_reference_mass": round(
                float(teacher_policy[int(row["corrected_reference_move"])]), 4
            ),
            "target_entropy": policy_entropy(teacher_policy, row["legal_moves"]),
        }

    decision = choose_decision(
        audited_rows=audited_rows,
        variant_summaries=variant_summaries,
        comparisons=isolated_vs_pr36_rows,
    )
    recommended_candidate = {
        "dirichlet_noise_leaks_into_targets": "no_noise_targets",
        "target_sharpening_amplifies_wrong_visits": "unsharpened_opening_targets",
        "low_simulation_target_noise": "min_sims_for_opening_targets",
        "forensic_teacher_fallback_needed": "teacher_fallback_for_corrected_forensic_hits",
    }.get(decision["classification"])
    fix_candidate_rows = [
        summarize_fix_candidate(
            name="no_noise_targets",
            rows=audited_rows,
            per_row_summaries=no_noise_candidate,
            implementation_risk="medium",
            notes="Uses the no-Dirichlet isolated-root target while leaving exploration-only noise available for action sampling.",
            recommended=recommended_candidate == "no_noise_targets",
        ),
        summarize_fix_candidate(
            name="unsharpened_opening_targets",
            rows=audited_rows,
            per_row_summaries=unsharpened_candidate,
            implementation_risk="low",
            notes="Uses default visit-policy targets for opening rows instead of squared sharpening.",
            recommended=recommended_candidate == "unsharpened_opening_targets",
        ),
        summarize_fix_candidate(
            name="min_sims_for_opening_targets",
            rows=audited_rows,
            per_row_summaries=higher_sims_candidate,
            implementation_risk="medium",
            notes=f"Diagnostic summary uses `{higher_sim_variant}` as the stronger opening-only minimum-simulation candidate.",
            recommended=recommended_candidate == "min_sims_for_opening_targets",
        ),
        summarize_fix_candidate(
            name="teacher_fallback_for_corrected_forensic_hits",
            rows=audited_rows,
            per_row_summaries=teacher_candidate,
            implementation_risk="medium",
            notes="Uses the corrected forensic teacher policy when self-play hits an exact audited forensic state.",
            recommended=recommended_candidate
            == "teacher_fallback_for_corrected_forensic_hits",
        ),
    ]

    summary = {
        "schema": SCHEMA,
        "config": config,
        "audited_rows": [
            {key: value for key, value in row.items() if key != "state"}
            for row in audited_rows
        ],
        "baseline_reproduction_rows": baseline_reproduction_rows,
        "ablation_rows": ablation_rows,
        "variant_runs": variant_runs,
        "isolated_vs_pr36_rows": isolated_vs_pr36_rows,
        "fix_candidate_rows": fix_candidate_rows,
        "decision": decision,
        "integrity": {
            "canonical_state_match": all(
                bool(row["pr36_context"]["row_canonicalization_exact_match"])
                for row in audited_rows
            ),
            "teacher_root_summary_present": any(
                int(row["pr36_context"]["teacher_root_summary_rows"]) > 0
                for row in audited_rows
            ),
            "missing_inputs": [],
        },
        "constraints": {
            "training_run": False,
            "arena_run": False,
            "promotion_run": False,
            "corrective_artifact_created": False,
            "replay_weight_sweep_run": False,
        },
        "paths": {
            "current_artifact": str(current_artifact),
            "self_play": str(self_play_path),
            "reference_artifact": str(reference_artifact),
            "fallback_reference_artifact": str(fallback_reference_artifact),
        },
    }
    write_json(summary_path, summary)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "report_path": str(report_path),
                "classification": decision["classification"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
