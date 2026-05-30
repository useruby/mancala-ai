#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.capture_002_003_rule_collision_diagnostic import (
    ROW_IDS,
    simulate_move_rule_features,
)
from ml.alphazero_lite.capture_002_003_search_policy_arbitration import (
    build_row_views,
    probe_artifact_position,
    validated_diagnostic_state,
)
from ml.alphazero_lite.pipeline import (
    build_step_command,
    render_command,
    resolve_step_command,
)


DEFAULT_ARTIFACT_PATH = (
    "/tmp/azlite_failure_family_diag/"
    "rule_conditioned_opening_family_full_guarded_artifact.jsonl"
)
DEFAULT_BASE_CONFIG = (
    "ml/alphazero_lite/configs/aggressive_v3_targeted_hard_state_replay.json"
)
DEFAULT_CURRENT_PATH = "storage/ai/alphazero_lite/current"
DEFAULT_OUTPUT_ROOT = "/tmp/azlite_rule_conditioned_opening_full_guarded"
DEFAULT_REFERENCE_ARTIFACT = (
    "ml/alphazero_lite/fixtures/incumbent_train_only_forensic_references_v1.json"
)
DEFAULT_WEIGHTS = (1, 2)
ARTIFACT_POLICY_TARGET_MODE = "sharpened"
ARTIFACT_VALUE_TARGET_MODE = "sharpened"
PRE_GATE_STEP_NAMES = ["self_play", "perspective_audit", "train", "export_artifact"]
EVAL_STEP_NAMES = [
    "hard_state_validation",
    "arena_confirm_report",
    "mcts1200_baseline_report",
    "current_mcts1200_baseline_report",
    "benchmark_contract",
]
TOP_K_GAMES = 60
GATE_SIMULATIONS = 384
GATE_SEED = 17
GATE_C_PUCT = 1.25
MATERIAL_DEGRADE_MARGIN = 0.05


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def python_bin(root: Path) -> str:
    candidate = root / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifact-path", default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--current-path", default=DEFAULT_CURRENT_PATH)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument(
        "--weights", default=",".join(str(weight) for weight in DEFAULT_WEIGHTS)
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_weights(raw_value: str) -> list[int]:
    weights = [int(value.strip()) for value in raw_value.split(",") if value.strip()]
    if not weights:
        raise SystemExit("--weights must contain at least one positive integer")
    if any(weight <= 0 for weight in weights):
        raise SystemExit("--weights must contain only positive integers")
    return weights


def _rewrite_command_seed_flags(
    command: list[object], *, original_seed: int, new_seed: int
) -> list[object]:
    rewritten = list(command)
    if "--seed" in rewritten:
        seed_index = rewritten.index("--seed")
        if seed_index + 1 < len(rewritten):
            rewritten[seed_index + 1] = str(new_seed)

    if "--seed-sweep" in rewritten:
        sweep_index = rewritten.index("--seed-sweep")
        if sweep_index + 1 < len(rewritten):
            sweep_values = [
                int(value.strip())
                for value in str(rewritten[sweep_index + 1]).split(",")
                if value.strip()
            ]
            if sweep_values:
                rewritten[sweep_index + 1] = ",".join(
                    str(new_seed + (value - original_seed)) for value in sweep_values
                )
    return rewritten


def apply_runtime_seed(config: dict, *, seed: int) -> dict:
    rewritten = json.loads(json.dumps(config))
    original_seed = int(rewritten.get("seed", seed))
    rewritten["seed"] = int(seed)
    for step in rewritten.get("steps", []):
        if not isinstance(step, dict):
            continue
        command = step.get("command")
        if not isinstance(command, list):
            continue
        step["command"] = _rewrite_command_seed_flags(
            command,
            original_seed=original_seed,
            new_seed=int(seed),
        )
    return rewritten


def replace_flag_value(command: list[object], *, flag: str, value: str) -> list[object]:
    rewritten = list(command)
    if flag in rewritten:
        index = rewritten.index(flag)
        if index + 1 < len(rewritten):
            rewritten[index + 1] = value
            return rewritten
    rewritten.extend([flag, value])
    return rewritten


def align_target_modes(config: dict) -> dict:
    rewritten = json.loads(json.dumps(config))
    for step in rewritten.get("steps", []):
        if not isinstance(step, dict):
            continue
        command = step.get("command")
        if not isinstance(command, list):
            continue
        if step.get("name") in {"self_play", "train"}:
            command = replace_flag_value(
                command,
                flag="--policy-target-mode",
                value=ARTIFACT_POLICY_TARGET_MODE,
            )
            command = replace_flag_value(
                command,
                flag="--value-target-mode",
                value=ARTIFACT_VALUE_TARGET_MODE,
            )
            step["command"] = command
    return rewritten


def resolve_parent_checkpoint(parent_artifact_path: Path) -> Path:
    checkpoint_candidate = parent_artifact_path / "checkpoint.npz"
    if checkpoint_candidate.exists():
        return checkpoint_candidate
    model_candidate = parent_artifact_path / "model.npz"
    if model_candidate.exists():
        return model_candidate
    weights_candidate = parent_artifact_path / "weights.json"
    if weights_candidate.exists():
        return weights_candidate
    raise SystemExit(
        "parent artifact must contain checkpoint.npz, model.npz, or weights.json: "
        f"{parent_artifact_path}"
    )


def candidate_dir_for_config(config: dict) -> Path:
    start_iteration = int(config.get("start_iteration", 1))
    iterations = int(config.get("iterations", 1))
    final_iteration = start_iteration + iterations - 1
    return (
        Path(str(config["versions_dir"])) / f"{config['run_id']}-iter{final_iteration}"
    )


def run_command(
    command: list[str],
    *,
    cwd: Path,
    dry_run: bool,
    log_path: Path | None = None,
) -> dict:
    if dry_run:
        result = {
            "command": command,
            "cwd": str(cwd),
            "returncode": None,
            "dry_run": True,
        }
        if log_path is not None:
            write_json(log_path, result)
        return result

    completed = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=False
    )
    result = {
        "command": command,
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if completed.returncode != 0:
        raise SystemExit(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}"
        )
    return result


def row_map_from_reference(reference_artifact: dict) -> dict[str, dict]:
    rows = reference_artifact.get("rows")
    if not isinstance(rows, list):
        raise SystemExit("reference artifact must contain a rows list")
    return {str(row["id"]): row for row in rows}


def step_map(config: dict) -> dict[str, dict]:
    return {
        str(step["name"]): step
        for step in config.get("steps", [])
        if isinstance(step, dict) and isinstance(step.get("name"), str)
    }


def render_step(
    *,
    config: dict,
    step_name: str,
    iter_dir: Path,
    current_path: str,
    parent_model_dir: Path,
    parent_checkpoint: Path,
    replay_data: str,
    replay_weights: str,
) -> list[str]:
    root = repo_root()
    steps = step_map(config)
    step = steps.get(step_name)
    if step is None:
        raise SystemExit(f"missing required step in runtime config: {step_name}")
    command = build_step_command(step)
    if not isinstance(command, list) or not command:
        raise SystemExit(f"step command must be a non-empty list: {step_name}")
    rendered = render_command(
        command,
        iteration=int(config.get("start_iteration", 1)),
        iter_dir=iter_dir,
        run_id=str(config["run_id"]),
        versions_dir=Path(str(config["versions_dir"])),
        current_path=current_path,
        parent_model_dir=parent_model_dir,
        parent_checkpoint=parent_checkpoint,
        replay_data=replay_data,
        replay_weights=replay_weights,
        hard_state_validation_path=str(config.get("hard_state_validation_path", "")),
    )
    return resolve_step_command(rendered, repo_root=root)


def build_probe_row(reference_row: dict) -> dict:
    return {
        "id": str(reference_row["id"]),
        "canonical_state": str(reference_row["canonical_state"]),
        "legal_moves": [
            int(child["move"]) for child in list(reference_row["child_stats"])
        ],
        "reference_move": int(reference_row["reference_move"]),
        "state": dict(reference_row["state"]),
    }


def evaluate_gate_row(
    *,
    artifact_path: Path,
    reference_row: dict,
    search_options: dict,
) -> dict:
    probe_row = build_probe_row(reference_row)
    state = validated_diagnostic_state(row=probe_row)
    summary = probe_artifact_position(
        artifact_path=str(artifact_path),
        state=state,
        simulations=GATE_SIMULATIONS,
        seed=GATE_SEED,
        c_puct=GATE_C_PUCT,
        search_options=search_options,
        ablation_mode="full",
    )
    return build_row_views(row=probe_row, probe_summary=summary)


def extract_row_metrics(*, row_view: dict, reference_move: int) -> dict:
    policy_view = row_view.get("policy_view") or {}
    value_view = row_view.get("value_view") or {}
    search_view = row_view.get("search_view") or {}
    visit_distribution = search_view.get("visit_distribution") or {}
    selected_move = search_view.get("searched_selected_move")

    return {
        "reference_move": reference_move,
        "searched_selected_move": selected_move,
        "reference_move_visit_share": search_view.get("reference_move_visit_share"),
        "selected_move_visit_share": search_view.get("selected_move_visit_share"),
        "selected_minus_reference_q_margin": value_view.get(
            "selected_minus_reference_q_margin"
        ),
        "policy_reference_probability": policy_view.get("reference_move_probability"),
        "policy_selected_minus_reference_margin": policy_view.get(
            "selected_minus_reference_margin"
        ),
        "visit_distribution": visit_distribution,
        "missing_fields": list(search_view.get("missing_fields") or []),
    }


def highest_wrong_extra_turn_move(
    *,
    state: dict,
    legal_moves: list[int],
    reference_move: int,
    visit_distribution: dict,
) -> tuple[int | None, float | None]:
    total_visits = sum(float(value) for value in visit_distribution.values())
    if total_visits <= 0.0:
        return None, None
    best_move = None
    best_share = None
    for move in legal_moves:
        if move == reference_move:
            continue
        features = simulate_move_rule_features(state=state, move=move)
        if not features["extra_turn_available"]:
            continue
        visit_share = visit_distribution.get(str(move))
        if visit_share is None:
            continue
        visit_share = float(visit_share) / total_visits
        if best_share is None or visit_share > best_share:
            best_move = move
            best_share = visit_share
    return best_move, best_share


def gate_row_payload(
    *,
    row_id: str,
    reference_row: dict,
    incumbent_view: dict,
    candidate_view: dict,
) -> dict:
    reference_move = int(reference_row["reference_move"])
    legal_moves = [int(child["move"]) for child in list(reference_row["child_stats"])]
    state = dict(reference_row["state"])
    incumbent_metrics = extract_row_metrics(
        row_view=incumbent_view, reference_move=reference_move
    )
    candidate_metrics = extract_row_metrics(
        row_view=candidate_view, reference_move=reference_move
    )

    wrong_extra_turn_move, wrong_extra_turn_visit_share = highest_wrong_extra_turn_move(
        state=state,
        legal_moves=legal_moves,
        reference_move=reference_move,
        visit_distribution=candidate_metrics["visit_distribution"],
    )

    required_fields = [
        candidate_metrics["searched_selected_move"],
        candidate_metrics["reference_move_visit_share"],
        candidate_metrics["selected_move_visit_share"],
        candidate_metrics["selected_minus_reference_q_margin"],
        candidate_metrics["policy_reference_probability"],
        candidate_metrics["policy_selected_minus_reference_margin"],
        incumbent_metrics["reference_move_visit_share"],
    ]
    pass_reasons: list[str] = []
    fail_reasons: list[str] = []

    if any(value is None for value in required_fields):
        fail_reasons.append("required_fields_missing")
    if candidate_metrics["missing_fields"]:
        fail_reasons.append("probe_missing_fields")

    candidate_reference_share = candidate_metrics["reference_move_visit_share"]
    incumbent_reference_share = incumbent_metrics["reference_move_visit_share"]
    if candidate_reference_share is not None:
        candidate_reference_share = float(candidate_reference_share)
    if incumbent_reference_share is not None:
        incumbent_reference_share = float(incumbent_reference_share)

    if row_id == "capture_available-002":
        if candidate_metrics["searched_selected_move"] != reference_move:
            fail_reasons.append("row_002_reference_move_not_selected")
        if (
            candidate_reference_share is not None
            and incumbent_reference_share is not None
            and candidate_reference_share <= incumbent_reference_share
        ):
            fail_reasons.append("row_002_reference_visit_share_not_improved")
        if (
            wrong_extra_turn_visit_share is not None
            and candidate_reference_share is not None
            and float(wrong_extra_turn_visit_share) > candidate_reference_share
        ):
            fail_reasons.append("row_002_wrong_extra_turn_still_dominates")
        if not fail_reasons:
            pass_reasons.append("row_002_fixed")

    if row_id == "capture_available-003":
        if candidate_metrics["searched_selected_move"] != reference_move:
            fail_reasons.append("row_003_reference_move_not_selected")
        if (
            candidate_reference_share is not None
            and incumbent_reference_share is not None
            and candidate_reference_share
            < incumbent_reference_share - MATERIAL_DEGRADE_MARGIN
        ):
            fail_reasons.append("row_003_reference_visit_share_materially_degraded")
        if not fail_reasons:
            pass_reasons.append("row_003_stable")

    passed = not fail_reasons
    return {
        "row_id": row_id,
        "reference_move": reference_move,
        "wrong_extra_turn_move": wrong_extra_turn_move,
        "wrong_extra_turn_visit_share": wrong_extra_turn_visit_share,
        "incumbent": {
            key: value
            for key, value in incumbent_metrics.items()
            if key != "visit_distribution"
        },
        "candidate": {
            key: value
            for key, value in candidate_metrics.items()
            if key != "visit_distribution"
        },
        "pass": passed,
        "pass_fail_reason": ",".join(pass_reasons if passed else fail_reasons),
    }


def run_local_gate(
    *,
    current_path: Path,
    candidate_path: Path,
    reference_artifact_path: Path,
    out_path: Path,
    search_options_overrides: dict | None = None,
    dry_run: bool,
) -> dict:
    if dry_run:
        payload = {
            "schema": "azlite_rule_conditioned_opening_local_gate_v1",
            "current_path": str(current_path),
            "candidate_path": str(candidate_path),
            "reference_artifact_path": str(reference_artifact_path),
            "settings": {
                "simulations": GATE_SIMULATIONS,
                "seed": GATE_SEED,
                "c_puct": GATE_C_PUCT,
                "material_degrade_margin": MATERIAL_DEGRADE_MARGIN,
                "search_options_overrides": search_options_overrides or {},
            },
            "rows": {},
            "pass": False,
            "decision": "dry_run",
            "reason": "gate_not_executed",
        }
        write_json(out_path, payload)
        return payload

    reference_rows = row_map_from_reference(load_json(reference_artifact_path))
    arena_module = __import__("ml.alphazero_lite.arena", fromlist=["dummy"])
    search_options = dict(arena_module.build_eval_search_options())
    if search_options_overrides:
        search_options.update(search_options_overrides)
    else:
        search_options["reuse_subtree"] = False
    rows = {}
    for row_id in ROW_IDS:
        reference_row = reference_rows[row_id]
        incumbent_view = evaluate_gate_row(
            artifact_path=current_path,
            reference_row=reference_row,
            search_options=search_options,
        )
        candidate_view = evaluate_gate_row(
            artifact_path=candidate_path,
            reference_row=reference_row,
            search_options=search_options,
        )
        rows[row_id] = gate_row_payload(
            row_id=row_id,
            reference_row=reference_row,
            incumbent_view=incumbent_view,
            candidate_view=candidate_view,
        )

    gate_pass = all(bool(rows[row_id]["pass"]) for row_id in ROW_IDS)
    if (
        rows["capture_available-002"]["candidate"]["searched_selected_move"]
        != rows["capture_available-002"]["reference_move"]
    ):
        reason = "row_002_local_rule_collision_persists"
    elif not rows["capture_available-003"]["pass"]:
        reason = "row_003_regressed"
    else:
        reason = "row_pair_gate_passed" if gate_pass else "required_fields_missing"

    payload = {
        "schema": "azlite_rule_conditioned_opening_local_gate_v1",
        "current_path": str(current_path),
        "candidate_path": str(candidate_path),
        "reference_artifact_path": str(reference_artifact_path),
        "settings": {
            "simulations": GATE_SIMULATIONS,
            "seed": GATE_SEED,
            "c_puct": GATE_C_PUCT,
            "material_degrade_margin": MATERIAL_DEGRADE_MARGIN,
            "search_options_overrides": search_options_overrides or {},
        },
        "rows": rows,
        "pass": gate_pass,
        "decision": "pass" if gate_pass else "reject_local_gate",
        "reason": reason,
    }
    write_json(out_path, payload)
    return payload


def arena_score(report: dict) -> float | None:
    if "score" in report:
        return float(report["score"])
    games_played = int(report.get("games_played", 0))
    if games_played <= 0:
        return None
    wins = int(report.get("wins", 0))
    draws = int(report.get("draws", 0))
    return (wins + (0.5 * draws)) / games_played


def mcts_score(report: dict) -> float | None:
    if "score" in report:
        return float(report["score"])
    games = int(report.get("games", 0))
    if games <= 0:
        return None
    wins = int(report.get("az_wins", 0))
    draws = int(report.get("draws", 0))
    return (wins + (0.5 * draws)) / games


def benchmark_pass(report: dict) -> bool | None:
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        return None
    statuses = [check.get("passed") for check in checks if isinstance(check, dict)]
    if not statuses:
        return None
    return all(bool(status) for status in statuses)


def top_k_checkpoints_present(iter_dir: Path) -> bool:
    return (
        any(iter_dir.glob("checkpoint.top*.npz"))
        or (iter_dir / "checkpoint.npz").exists()
    )


def build_variant_configs(
    *,
    base_config: dict,
    variant_root: Path,
    artifact_path: Path,
    weight: int,
    seed: int,
    current_path: str,
) -> tuple[dict, Path, dict, Path]:
    runtime_config = align_target_modes(apply_runtime_seed(base_config, seed=seed))
    runtime_config["run_id"] = (
        f"{runtime_config['run_id']}-rule-conditioned-opening-full-guarded-w{weight}"
    )
    runtime_config["versions_dir"] = str(variant_root / "versions")
    runtime_config["current_path"] = current_path
    runtime_config["parent_artifact_path"] = current_path
    fixed_replay_sources = list(runtime_config.get("fixed_replay_sources", []))
    fixed_replay_sources.append({"path": str(artifact_path), "weight": int(weight)})
    runtime_config["fixed_replay_sources"] = fixed_replay_sources

    pre_gate_config = json.loads(json.dumps(runtime_config))
    pre_gate_config["steps"] = [
        step
        for step in pre_gate_config.get("steps", [])
        if isinstance(step, dict) and step.get("name") in PRE_GATE_STEP_NAMES
    ]

    runtime_config_path = variant_root / "runtime_config.json"
    pre_gate_config_path = variant_root / "pre_gate_runtime_config.json"
    write_json(runtime_config_path, runtime_config)
    write_json(pre_gate_config_path, pre_gate_config)
    return runtime_config, runtime_config_path, pre_gate_config, pre_gate_config_path


def decide_variant_outcome(variant: dict) -> str:
    gate = variant.get("row_pair_gate") or {}
    if gate.get("decision") == "reject_local_gate":
        return "reject_local_gate"

    arena = variant.get("arena_report") or {}
    arena_value = arena.get("score")
    if arena_value is None:
        return "gate_passed_eval_missing"
    if float(arena_value) < 0.45:
        return "reject_overfit_brittle"
    if float(arena_value) <= 0.5:
        return "diagnostic_win_no_promotion"
    return "promising_rerun_two_more_seeds"


def final_next_action(variants: list[dict]) -> str:
    gate_passers = [
        variant
        for variant in variants
        if (variant.get("row_pair_gate") or {}).get("pass")
    ]
    if not gate_passers:
        return "Search-prior control next; stop replay lanes because both weights failed the 002/003 local gate."

    improved = [
        variant
        for variant in gate_passers
        if (variant.get("arena_report") or {}).get("score") is not None
        and float((variant.get("arena_report") or {})["score"]) > 0.5
    ]
    if improved:
        best = max(
            improved,
            key=lambda variant: float((variant.get("arena_report") or {})["score"]),
        )
        return f"Rerun replay weight {best['replay_weight']} with two more full runtime seeds."

    collapsed = [
        variant
        for variant in gate_passers
        if (variant.get("arena_report") or {}).get("score") is not None
        and float((variant.get("arena_report") or {})["score"]) < 0.45
    ]
    if collapsed:
        return "Reject the lane as overfit/brittle; do not continue this replay branch."

    best = gate_passers[0]
    return (
        "Treat the surviving weight as a diagnostic win only and next test whether the fix "
        "generalizes to more no-extra-turn opening rows."
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    python = python_bin(root)

    artifact_path = resolve_path(root, args.artifact_path)
    artifact_summary_path = artifact_path.with_name(
        artifact_path.stem + "_summary.json"
    )
    current_path = resolve_path(root, args.current_path)
    base_config_path = resolve_path(root, args.base_config)
    reference_artifact_path = resolve_path(root, DEFAULT_REFERENCE_ARTIFACT)
    output_root = resolve_path(root, args.output_root)
    run_root = output_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)

    if not artifact_path.exists():
        raise SystemExit(f"artifact path does not exist: {artifact_path}")
    if not artifact_summary_path.exists():
        raise SystemExit(
            f"artifact summary path does not exist: {artifact_summary_path}"
        )
    if not current_path.exists():
        raise SystemExit(f"current path does not exist: {current_path}")
    if not reference_artifact_path.exists():
        raise SystemExit(
            f"reference artifact path does not exist: {reference_artifact_path}"
        )

    base_config = load_json(base_config_path)
    artifact_summary = load_json(artifact_summary_path)
    weights = parse_weights(args.weights)
    manifest = {
        "schema": "azlite_rule_conditioned_opening_full_guarded_experiment_v1",
        "run_id": args.run_id,
        "artifact_path": str(artifact_path),
        "artifact_summary_path": str(artifact_summary_path),
        "base_config": str(base_config_path),
        "current_path": str(current_path),
        "seed": int(args.seed),
        "weights": weights,
        "variants": [],
        "dry_run": bool(args.dry_run),
    }

    for weight in weights:
        variant_root = run_root / f"w{weight}"
        variant_root.mkdir(parents=True, exist_ok=True)
        runtime_config, runtime_config_path, _pre_gate_config, pre_gate_config_path = (
            build_variant_configs(
                base_config=base_config,
                variant_root=variant_root,
                artifact_path=artifact_path,
                weight=weight,
                seed=args.seed,
                current_path=str(current_path),
            )
        )

        candidate_dir = candidate_dir_for_config(runtime_config)
        parent_checkpoint = resolve_parent_checkpoint(current_path)
        replay_data = str(artifact_path)
        replay_weights = str(weight)
        variant_payload = {
            "variant": f"w{weight}",
            "replay_weight": weight,
            "runtime_config_path": str(runtime_config_path),
            "pre_gate_runtime_config_path": str(pre_gate_config_path),
            "candidate_dir": str(candidate_dir),
            "candidate_artifact_path": str(candidate_dir),
            "artifact_row_count": artifact_summary.get("row_count"),
            "has_002_guard": "capture_available-002"
            in list(artifact_summary.get("rule_collision_guard_row_ids") or []),
            "has_003_guard": "capture_available-003"
            in list(artifact_summary.get("rule_collision_guard_row_ids") or []),
        }

        pipeline_log = variant_root / "pipeline_pre_gate_command.json"
        run_command(
            [
                python,
                str(root / "ml/alphazero_lite/pipeline.py"),
                "--config",
                str(pre_gate_config_path),
            ],
            cwd=root,
            dry_run=args.dry_run,
            log_path=pipeline_log,
        )

        gate_path = candidate_dir / "capture_002_003_rule_conditioned_gate.json"
        gate_payload = run_local_gate(
            current_path=current_path,
            candidate_path=candidate_dir,
            reference_artifact_path=reference_artifact_path,
            out_path=gate_path,
            dry_run=args.dry_run,
        )
        variant_payload["row_pair_gate_path"] = str(gate_path)
        variant_payload["row_pair_gate"] = gate_payload

        if gate_payload.get("decision") == "reject_local_gate":
            variant_payload["decision"] = "reject_local_gate"
            manifest["variants"].append(variant_payload)
            continue

        step_logs_dir = variant_root / "eval_logs"
        for step_name in EVAL_STEP_NAMES:
            command = render_step(
                config=runtime_config,
                step_name=step_name,
                iter_dir=candidate_dir,
                current_path=str(current_path),
                parent_model_dir=current_path,
                parent_checkpoint=parent_checkpoint,
                replay_data=replay_data,
                replay_weights=replay_weights,
            )
            run_command(
                command,
                cwd=root,
                dry_run=args.dry_run,
                log_path=step_logs_dir / f"{step_name}.json",
            )

        if top_k_checkpoints_present(candidate_dir):
            top_k_out = candidate_dir / "top_k_evaluation_summary.json"
            run_command(
                [
                    python,
                    str(root / "ml/alphazero_lite/evaluate_top_k_checkpoints.py"),
                    "--iter-dir",
                    str(candidate_dir),
                    "--current-path",
                    str(current_path),
                    "--games",
                    str(TOP_K_GAMES),
                    "--out",
                    str(top_k_out),
                ],
                cwd=root,
                dry_run=args.dry_run,
                log_path=step_logs_dir / "top_k_checkpoints.json",
            )
            variant_payload["top_k_summary_path"] = str(top_k_out)

        if not args.dry_run:
            hard_state_report = load_json(candidate_dir / "hard_state_validation.json")
            arena_report = load_json(candidate_dir / "arena_report.json")
            mcts_report = load_json(candidate_dir / "mcts1200_report.json")
            benchmark_report = load_json(candidate_dir / "benchmark_report.json")
            current_mcts_report = load_json(
                candidate_dir / "current_mcts1200_report.json"
            )
            variant_payload["hard_state_validation"] = {
                "average_regret": hard_state_report.get("average_regret"),
                "value_calibration_mae": hard_state_report.get("value_calibration_mae"),
                "report_path": str(candidate_dir / "hard_state_validation.json"),
            }
            variant_payload["arena_report"] = {
                "score": arena_score(arena_report),
                "ci_low": (arena_report.get("confidence_interval_95") or {}).get(
                    "lower"
                ),
                "ci_high": (arena_report.get("confidence_interval_95") or {}).get(
                    "upper"
                ),
                "unstable_decision": arena_report.get("unstable_decision"),
                "report_path": str(candidate_dir / "arena_report.json"),
            }
            variant_payload["mcts1200_report"] = {
                "score": mcts_score(mcts_report),
                "report_path": str(candidate_dir / "mcts1200_report.json"),
            }
            variant_payload["current_mcts1200_report"] = {
                "score": mcts_score(current_mcts_report),
                "report_path": str(candidate_dir / "current_mcts1200_report.json"),
            }
            variant_payload["benchmark_report"] = {
                "passed": benchmark_pass(benchmark_report),
                "report_path": str(candidate_dir / "benchmark_report.json"),
            }

            promising = (
                variant_payload["arena_report"]["score"] is not None
                and float(variant_payload["arena_report"]["score"]) > 0.5
            )
            if promising:
                local_gate_out = output_root / f"{args.run_id}-local-promotion.json"
                run_command(
                    [
                        str(root / "script/ai/local_promotion_gate"),
                        "--candidate-path",
                        str(candidate_dir),
                        "--current-path",
                        str(current_path),
                        "--hard-path",
                        str(current_path),
                        "--config-path",
                        str(runtime_config_path),
                        "--out",
                        str(local_gate_out),
                    ],
                    cwd=root,
                    dry_run=False,
                    log_path=step_logs_dir / "local_promotion_gate.json",
                )
                variant_payload["local_promotion_gate_path"] = str(local_gate_out)

        variant_payload["decision"] = decide_variant_outcome(variant_payload)
        manifest["variants"].append(variant_payload)

    manifest["recommended_next_action"] = final_next_action(manifest["variants"])
    manifest_path = run_root / "experiment_summary.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps({"summary_path": str(manifest_path), "dry_run": bool(args.dry_run)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
