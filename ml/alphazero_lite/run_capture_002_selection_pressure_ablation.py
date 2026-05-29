#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite import capture_002_selection_score_trace
from ml.alphazero_lite import capture_002_trace_capture
from ml.alphazero_lite.capture_002_003_search_policy_arbitration import (
    build_row_views,
    probe_artifact_position,
    validated_diagnostic_state,
)
from ml.alphazero_lite.run_capture_002_trace_pair_harmonization import (
    build_pair_projected_trace_capture_artifact,
    pair_project_trace_points,
)


DEFAULT_SOURCE_TRACE_CAPTURE_PATH = (
    "/tmp/azlite_capture_002_prior_pressure_from_search_control/"
    "capture-002-prior-pressure/capture_002_trace_capture.json"
)
DEFAULT_OUTPUT_ROOT = "/tmp/azlite_capture_002_selection_pressure_ablation"
SCHEMA = "azlite_capture_002_selection_pressure_ablation_v1"
ROW_ID = "capture_available-002"
PRESSURE_DECISION = "write_002_selection_pressure_ablation_spec"

VARIANT_SPECS = (
    {
        "name": "baseline_full",
        "label": "Baseline Full",
        "ablation_mode": "full",
        "search_options_overrides": {},
    },
    {
        "name": "value_only",
        "label": "Value Only",
        "ablation_mode": "value_only",
        "search_options_overrides": {},
    },
    {
        "name": "policy_only",
        "label": "Policy Only",
        "ablation_mode": "policy_only",
        "search_options_overrides": {},
    },
    {
        "name": "full_fpu_zero",
        "label": "Full FPU Zero",
        "ablation_mode": "full",
        "search_options_overrides": {"fpu_mode": "zero"},
    },
    {
        "name": "full_root_visit_count",
        "label": "Full Root Visit Count",
        "ablation_mode": "full",
        "search_options_overrides": {
            "root_policy_mode": "visit_count",
            "tactical_root_bias": 0.0,
        },
    },
    {
        "name": "full_quality_off",
        "label": "Full Quality Off",
        "ablation_mode": "full",
        "search_options_overrides": {
            "fpu_mode": "zero",
            "reuse_subtree": False,
            "normalize_values": False,
            "root_policy_mode": "visit_count",
            "tactical_root_bias": 0.0,
        },
    },
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--source-trace-capture-artifact", default=DEFAULT_SOURCE_TRACE_CAPTURE_PATH
    )
    parser.add_argument("--candidate-path", default=None)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _source_row(trace_capture_artifact: dict) -> dict:
    source_payload = (
        trace_capture_artifact.get("source_shared_drift_artifact") or {}
    ).get("source_payload") or {}
    row = ((source_payload.get("rows") or {}).get(ROW_ID)) or {}
    if not isinstance(row, dict) or not row:
        raise SystemExit(
            "source trace capture artifact must embed source_shared_drift row 002"
        )
    return row


def _source_settings(trace_capture_artifact: dict) -> dict:
    source_payload = (
        trace_capture_artifact.get("source_shared_drift_artifact") or {}
    ).get("source_payload") or {}
    settings = source_payload.get("settings") or {}
    if not isinstance(settings, dict):
        raise SystemExit("source trace capture artifact must embed source settings")
    return settings


def _candidate_path(trace_capture_artifact: dict, override: str | None) -> str:
    if override:
        return override
    selected_artifact = (
        trace_capture_artifact.get("source_shared_drift_artifact") or {}
    ).get("selected_artifact") or {}
    path = selected_artifact.get("path")
    if not isinstance(path, str) or not path:
        raise SystemExit("candidate path missing from source trace capture artifact")
    return path


def _probe_row(trace_capture_artifact: dict) -> dict:
    row_context = trace_capture_artifact.get("row_context") or {}
    return {
        "id": ROW_ID,
        "canonical_state": row_context.get("canonical_state"),
        "legal_moves": copy.deepcopy(row_context.get("legal_moves")),
        "reference_move": row_context.get("reference_move"),
    }


def _seed(settings: dict) -> int:
    seed = settings.get("seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        return seed
    seeds = settings.get("seeds")
    if isinstance(seeds, list) and seeds and isinstance(seeds[0], int):
        return int(seeds[0])
    return 17


def _simulation_count(settings: dict) -> int:
    simulation_count = settings.get("simulation_count")
    if isinstance(simulation_count, int) and not isinstance(simulation_count, bool):
        return simulation_count
    return 384


def _c_puct(settings: dict) -> float:
    search_settings = settings.get("search_settings") or {}
    c_puct = search_settings.get("c_puct")
    if isinstance(c_puct, (int, float)) and not isinstance(c_puct, bool):
        return float(c_puct)
    return 1.25


def _base_search_options(settings: dict) -> dict:
    search_settings = settings.get("search_settings") or {}
    return {
        "fpu_mode": str(search_settings.get("fpu_mode", "parent_q")),
        "reuse_subtree": bool(search_settings.get("reuse_subtree", True)),
        "normalize_values": bool(search_settings.get("normalize_values", True)),
        "root_policy_mode": str(
            search_settings.get("root_policy_mode", "deterministic")
        ),
        "tactical_root_bias": float(search_settings.get("tactical_root_bias", 0.1)),
    }


def _snapshot_simulation(trace_point: dict | None) -> float | None:
    if not isinstance(trace_point, dict):
        return None
    value = trace_point.get("simulation")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _variant_outcome(
    *,
    variant_spec: dict,
    trace_capture_artifact: dict,
    candidate_path: str,
    variant_dir: Path,
) -> dict:
    settings = _source_settings(trace_capture_artifact)
    probe_row = _probe_row(trace_capture_artifact)
    state = validated_diagnostic_state(row=probe_row)
    search_options = _base_search_options(settings)
    search_options.update(variant_spec["search_options_overrides"])

    probe_summary = probe_artifact_position(
        artifact_path=candidate_path,
        state=state,
        simulations=_simulation_count(settings),
        seed=_seed(settings),
        c_puct=_c_puct(settings),
        search_options=search_options,
        ablation_mode=variant_spec["ablation_mode"],
    )

    row_view = build_row_views(row=probe_row, probe_summary=probe_summary)

    synthetic_trace_capture = copy.deepcopy(trace_capture_artifact)
    synthetic_trace_capture["rerun_trace"] = {
        "trace_origin": "rerun",
        "trace_points": copy.deepcopy(probe_summary.get("visit_snapshots") or []),
        "insufficiency_reasons": [],
    }

    projected_trace_points, projection_summary = pair_project_trace_points(
        synthetic_trace_capture
    )

    projected_trace_capture_path = (
        variant_dir / "capture_002_pair_projected_trace_capture.json"
    )
    projected_trace_capture = build_pair_projected_trace_capture_artifact(
        synthetic_trace_capture,
        projected_trace_points=projected_trace_points,
        projection_summary=projection_summary,
        out_path=projected_trace_capture_path,
    )
    write_json(projected_trace_capture_path, projected_trace_capture)
    projected_trace_capture["artifact_write_summary"]["trace_capture_sha256"] = (
        capture_002_trace_capture.sha256_file(projected_trace_capture_path)
    )

    regenerated_shared_drift = None
    regenerated_shared_drift_path = (
        variant_dir / "capture_002_pair_projected_shared_drift.json"
    )
    if not projected_trace_capture["insufficiency_reasons"]:
        regenerated_shared_drift = (
            capture_002_trace_capture.build_regenerated_shared_drift_artifact(
                projected_trace_capture
            )
        )

    default_trace = None
    relaxed_trace = None
    if regenerated_shared_drift is not None:
        write_json(regenerated_shared_drift_path, regenerated_shared_drift)
        projected_trace_capture["artifact_write_summary"][
            "regenerated_shared_drift_written"
        ] = True
        projected_trace_capture["artifact_write_summary"][
            "regenerated_shared_drift_path"
        ] = str(regenerated_shared_drift_path)
        projected_trace_capture["artifact_write_summary"][
            "regenerated_shared_drift_sha256"
        ] = capture_002_trace_capture.sha256_file(regenerated_shared_drift_path)

        loaded_source = capture_002_selection_score_trace.load_source_shared_drift_artifact_document(
            regenerated_shared_drift,
            artifact_path=str(regenerated_shared_drift_path),
        )
        default_trace = capture_002_selection_score_trace.build_payload(loaded_source)
        relaxed_trace = capture_002_selection_score_trace.build_payload(
            loaded_source,
            material_visit_share_margin=0.04,
        )
        write_json(
            variant_dir / "capture_002_selection_score_trace.json", default_trace
        )
        write_json(
            variant_dir / "capture_002_selection_score_trace_relaxed.json",
            relaxed_trace,
        )
    else:
        projected_trace_capture["artifact_write_summary"][
            "regenerated_shared_drift_skip_reason"
        ] = "pair_projected_trace_not_downstream_ready"

    write_json(projected_trace_capture_path, projected_trace_capture)

    default_decision = None if default_trace is None else default_trace.get("decision")
    relaxed_decision = None if relaxed_trace is None else relaxed_trace.get("decision")
    pressure_relieved = bool(
        default_trace is not None
        and relaxed_trace is not None
        and default_decision != PRESSURE_DECISION
        and relaxed_decision != PRESSURE_DECISION
    )

    return {
        "variant": variant_spec["name"],
        "label": variant_spec["label"],
        "ablation_mode": variant_spec["ablation_mode"],
        "search_options": copy.deepcopy(search_options),
        "search_view": row_view["search_view"],
        "value_view": row_view["value_view"],
        "policy_view": row_view["policy_view"],
        "pair_projection_summary": projection_summary,
        "pressure_relieved": pressure_relieved,
        "default_trace": None
        if default_trace is None
        else {
            "trace_origin": default_trace.get("trace_origin"),
            "classification": (default_trace.get("classification") or {}).get(
                "classification"
            ),
            "decision": default_trace.get("decision"),
            "first_selected_selection_score_overtake_simulation": _snapshot_simulation(
                default_trace.get("first_selected_selection_score_overtake_snapshot")
            ),
            "first_selected_material_visit_share_simulation": _snapshot_simulation(
                default_trace.get("first_selected_material_visit_share_snapshot")
            ),
            "first_selected_meaningful_q_support_simulation": _snapshot_simulation(
                default_trace.get("first_selected_meaningful_q_support_snapshot")
            ),
            "final_selected_minus_reference_selection_score": default_trace.get(
                "final_selected_minus_reference_selection_score"
            ),
            "final_selected_minus_reference_q": default_trace.get(
                "final_selected_minus_reference_q"
            ),
            "final_selected_minus_reference_visit_share": default_trace.get(
                "final_selected_minus_reference_visit_share"
            ),
        },
        "relaxed_trace": None
        if relaxed_trace is None
        else {
            "trace_origin": relaxed_trace.get("trace_origin"),
            "classification": (relaxed_trace.get("classification") or {}).get(
                "classification"
            ),
            "decision": relaxed_trace.get("decision"),
            "first_selected_selection_score_overtake_simulation": _snapshot_simulation(
                relaxed_trace.get("first_selected_selection_score_overtake_snapshot")
            ),
            "first_selected_material_visit_share_simulation": _snapshot_simulation(
                relaxed_trace.get("first_selected_material_visit_share_snapshot")
            ),
            "first_selected_meaningful_q_support_simulation": _snapshot_simulation(
                relaxed_trace.get("first_selected_meaningful_q_support_snapshot")
            ),
        },
        "paths": {
            "pair_projected_trace_capture": str(projected_trace_capture_path),
            "pair_projected_shared_drift": None
            if regenerated_shared_drift is None
            else str(regenerated_shared_drift_path),
            "default_trace": None
            if default_trace is None
            else str(variant_dir / "capture_002_selection_score_trace.json"),
            "relaxed_trace": None
            if relaxed_trace is None
            else str(variant_dir / "capture_002_selection_score_trace_relaxed.json"),
        },
    }


def _top_level_classification(variants: list[dict]) -> tuple[str, str, str]:
    relieved = [variant for variant in variants if variant["pressure_relieved"]]
    if relieved:
        labels = ", ".join(variant["variant"] for variant in relieved)
        return (
            "selection_pressure_variant_sensitive",
            "write_002_search_pressure_variant_followup_spec",
            f"Selection-score pressure is relieved by at least one search variant: {labels}.",
        )
    return (
        "selection_pressure_persists",
        "stop_002_selection_pressure_ablation_inconclusive",
        "Selection-score pressure persists across the tested search variants.",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    source_trace_capture_path = resolve_path(root, args.source_trace_capture_artifact)
    out_root = resolve_path(root, args.output_root) / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    source_trace_capture = load_json(source_trace_capture_path)
    candidate_path = _candidate_path(source_trace_capture, args.candidate_path)

    variants = []
    for variant_spec in VARIANT_SPECS:
        variant_dir = out_root / variant_spec["name"]
        variants.append(
            _variant_outcome(
                variant_spec=variant_spec,
                trace_capture_artifact=source_trace_capture,
                candidate_path=candidate_path,
                variant_dir=variant_dir,
            )
        )

    classification, decision, evidence_summary = _top_level_classification(variants)
    summary = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "source_trace_capture_artifact": str(source_trace_capture_path),
        "candidate_path": candidate_path,
        "row_id": ROW_ID,
        "classification": {
            "classification": classification,
            "evidence_summary": evidence_summary,
        },
        "decision": decision,
        "variants": variants,
    }
    summary_path = out_root / "capture_002_selection_pressure_ablation_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary_path": str(summary_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
