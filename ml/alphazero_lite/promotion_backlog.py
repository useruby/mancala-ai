"""Manifest helpers for hard-bot promotion backlog workflows."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "azlite_hard_bot_backlog_v1"
STAGES = {
    "promotion_gate",
    "forensic_suite",
    "multi_seed_confirmation",
    "readiness_report",
}


def required_artifact_keys() -> list[str]:
    return [
        "promotion_gate_report",
        "forensic_report",
        "multi_seed_summary",
    ]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def worktree_workspace_root(root: Path) -> Path | None:
    if root.parent.name != ".worktrees":
        return None
    return root.parent.parent


def python_executable() -> str:
    root = repo_root()
    candidates = [root / ".venv/bin/python"]
    workspace_root = worktree_workspace_root(root)
    if workspace_root is not None:
        candidates.append(workspace_root / ".venv/bin/python")
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return sys.executable


def normalize_manifest(manifest: dict) -> dict:
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema')}")

    artifacts = manifest.setdefault("artifacts", {})
    if not artifacts.get("promotion_gate_report") and artifacts.get("promotion_gate"):
        artifacts["promotion_gate_report"] = artifacts["promotion_gate"]

    manifest.setdefault("steps", {})
    manifest.setdefault("multi_seed_confirmation", {"runs": []})
    return manifest


def load_manifest(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return normalize_manifest(manifest)


def write_manifest(manifest_path: Path, manifest: dict) -> dict:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = iso_now()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def initialize_manifest(
    *,
    manifest_path: Path,
    campaign_id: str,
    candidate_path: Path,
    config_path: Path | None,
    parent_artifact_path: Path | None,
) -> dict:
    now = iso_now()
    manifest = {
        "schema": SCHEMA,
        "manifest_path": str(manifest_path),
        "campaign_id": campaign_id,
        "candidate": {
            "artifact_path": str(candidate_path),
            "config_path": None if config_path is None else str(config_path),
            "parent_artifact_path": None if parent_artifact_path is None else str(parent_artifact_path),
        },
        "inputs": {},
        "artifacts": {},
        "steps": {},
        "multi_seed_confirmation": {"runs": []},
        "readiness": {
            "state": "needs_runs",
            "passed": False,
            "failure_reasons": [],
            "missing_evidence": [],
            "stale_artifacts": [],
            "last_evaluated_at": None,
        },
        "created_at": now,
    }
    return write_manifest(manifest_path, manifest)


def stage_command(*, stage: str, manifest: dict) -> list[str]:
    if stage not in STAGES:
        raise ValueError(f"unsupported stage: {stage}")

    manifest_path = manifest.get("manifest_path")
    candidate = manifest.get("candidate", {})
    candidate_path = candidate.get("artifact_path")
    config_path = candidate.get("config_path")
    parent_artifact_path = candidate.get("parent_artifact_path") or "model-artifact/current"
    output_root = str(Path(manifest_path).parent) if manifest_path else "."

    if stage == "promotion_gate":
        command = [
            "script/ai/local_promotion_gate",
            "--candidate-path",
            candidate_path,
            "--out",
            str(Path(output_root) / "promotion_gate.json"),
        ]
        if config_path:
            command.extend(["--config-path", config_path])
        return command

    if stage == "forensic_suite":
        return [
            python_executable(),
            "ml/alphazero_lite/run_forensic_suite.py",
            "--suite",
            "ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json",
            "--current-artifact",
            parent_artifact_path,
            "--challenger-artifact",
            candidate_path,
            "--mcts-simulations",
            "1200",
            "--teacher-simulations",
            "1800",
            "--out",
            str(Path(output_root) / "forensic_report.json"),
        ]

    if stage == "multi_seed_confirmation":
        return [
            python_executable(),
            "script/ai/model_robustness_confirmation",
            "--base-config",
            config_path or "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json",
            "--parent-artifact",
            parent_artifact_path,
            "--current-path",
            candidate_path,
            "--output-root",
            str(Path(output_root) / "robustness_confirmation"),
        ]

    return [
        "script/ai/report_hard_bot_promotion_readiness",
        "--manifest-path",
        manifest_path,
    ]


def record_step_result(
    *,
    manifest_path: Path,
    step_name: str,
    status: str,
    command: list[str],
    outputs: dict[str, str],
    failure_summary: str | None,
) -> dict:
    manifest = load_manifest(manifest_path)
    manifest["steps"][step_name] = {
        "status": status,
        "command": command,
        "outputs": outputs,
        "failure_summary": failure_summary,
        "updated_at": iso_now(),
    }
    return write_manifest(manifest_path, manifest)


def record_artifact(manifest_path: Path, *, key: str, value: str) -> dict:
    manifest = load_manifest(manifest_path)
    manifest.setdefault("artifacts", {})[key] = value
    return write_manifest(manifest_path, manifest)


def record_multi_seed_runs(manifest_path: Path, *, seeds: list[int], summary_path: str, passed: bool) -> dict:
    manifest = load_manifest(manifest_path)
    manifest["multi_seed_confirmation"] = {
        "runs": [{"seed": seed, "passed": passed} for seed in seeds],
        "summary": {"status": "passed" if passed else "failed"},
        "summary_path": summary_path,
    }
    return write_manifest(manifest_path, manifest)


def promotion_failures_indicate_runtime_failure(failure_reasons: list[dict]) -> bool:
    for reason in failure_reasons:
        code = str(reason.get("code", ""))
        if "runtime" in code or "move_time" in code:
            return True

    return False


def inspect_json_artifact(path: str | None, *, name: str) -> tuple[dict | None, str | None]:
    if not path:
        return None, f"{name}_missing"

    artifact_path = Path(path)
    if not artifact_path.exists():
        return None, f"{name}_missing"

    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, f"{name}_malformed"

    if not isinstance(payload, dict):
        return None, f"{name}_unusable"

    return payload, None


def validate_promotion_gate_report(payload: dict | None) -> str | None:
    if payload is None:
        return None

    if not isinstance(payload.get("passed"), bool):
        return "promotion_gate_report_unusable"

    failure_reasons = payload.get("failure_reasons")
    if not isinstance(failure_reasons, list):
        return "promotion_gate_report_unusable"

    if any(not isinstance(reason, dict) for reason in failure_reasons):
        return "promotion_gate_report_unusable"

    return None


def validate_forensic_report(payload: dict | None) -> str | None:
    if payload is None:
        return None

    if payload.get("schema") == "azlite_forensic_suite_v1":
        systems = payload.get("systems")
        if not isinstance(systems, dict):
            return "forensic_report_unusable"

        for system_name in ("current", "challenger"):
            system_payload = systems.get(system_name)
            if not isinstance(system_payload, dict):
                return "forensic_report_unusable"
            overall = system_payload.get("overall")
            if not isinstance(overall, dict):
                return "forensic_report_unusable"
            if not isinstance(overall.get("top1_agreement"), (int, float)):
                return "forensic_report_unusable"
            if not isinstance(overall.get("average_regret"), (int, float)):
                return "forensic_report_unusable"
            if not isinstance(overall.get("value_calibration_mae"), (int, float)):
                return "forensic_report_unusable"

        return None

    forensic_quality = payload.get("forensic_quality")
    if not isinstance(forensic_quality, dict):
        return "forensic_report_unusable"

    if not isinstance(forensic_quality.get("passed"), bool):
        return "forensic_report_unusable"

    return None


def forensic_report_passed(payload: dict | None) -> bool | None:
    if payload is None:
        return None

    if payload.get("schema") == "azlite_forensic_suite_v1":
        challenger_overall = payload["systems"]["challenger"]["overall"]
        current_overall = payload["systems"]["current"]["overall"]
        return (
            float(challenger_overall["top1_agreement"]) >= float(current_overall["top1_agreement"])
            and float(challenger_overall["average_regret"]) <= float(current_overall["average_regret"])
            and float(challenger_overall["value_calibration_mae"]) <= float(current_overall["value_calibration_mae"])
        )

    forensic_quality = payload.get("forensic_quality", {})
    return bool(forensic_quality.get("passed"))


def validate_multi_seed_summary_report(payload: dict | None) -> str | None:
    if payload is None:
        return None

    if not isinstance(payload.get("passed"), bool):
        return "multi_seed_summary_unusable"

    return None


def deduplicate_reasons(reasons: list[str]) -> list[str]:
    return list(dict.fromkeys(reasons))


def evaluate_readiness(*, manifest_path: Path) -> dict:
    manifest = load_manifest(manifest_path)
    artifacts = manifest.get("artifacts", {})
    missing_evidence = [key for key in required_artifact_keys() if not artifacts.get(key)]

    readiness = {
        "state": "needs_runs",
        "passed": False,
        "failure_reasons": [],
        "missing_evidence": missing_evidence,
        "stale_artifacts": [],
        "last_evaluated_at": iso_now(),
    }

    if missing_evidence:
        manifest["readiness"] = readiness
        write_manifest(manifest_path, manifest)
        return readiness

    steps = manifest.get("steps", {})
    multi_seed_confirmation = manifest.get("multi_seed_confirmation", {})
    runs = multi_seed_confirmation.get("runs", [])
    summary = multi_seed_confirmation.get("summary", {})

    promotion_gate_report, promotion_report_issue = inspect_json_artifact(
        artifacts.get("promotion_gate_report"),
        name="promotion_gate_report",
    )
    forensic_report, forensic_report_issue = inspect_json_artifact(
        artifacts.get("forensic_report"),
        name="forensic_report",
    )
    multi_seed_summary_report, multi_seed_summary_issue = inspect_json_artifact(
        artifacts.get("multi_seed_summary"),
        name="multi_seed_summary",
    )

    blocking_failures: list[str] = []
    investigation_failures: list[str] = []

    if promotion_report_issue:
        investigation_failures.append(promotion_report_issue)
    if forensic_report_issue:
        investigation_failures.append(forensic_report_issue)
    if multi_seed_summary_issue:
        investigation_failures.append(multi_seed_summary_issue)

    promotion_shape_issue = validate_promotion_gate_report(promotion_gate_report)
    forensic_shape_issue = validate_forensic_report(forensic_report)
    multi_seed_shape_issue = validate_multi_seed_summary_report(multi_seed_summary_report)

    if promotion_shape_issue:
        investigation_failures.append(promotion_shape_issue)
        promotion_gate_report = None
    if forensic_shape_issue:
        investigation_failures.append(forensic_shape_issue)
        forensic_report = None
    if multi_seed_shape_issue:
        investigation_failures.append(multi_seed_shape_issue)
        multi_seed_summary_report = None

    promotion_failure_reasons = [] if promotion_gate_report is None else promotion_gate_report.get("failure_reasons", [])
    forensic_passed = forensic_report_passed(forensic_report)

    promotion_step_status = steps.get("promotion_gate", {}).get("status")
    forensic_step_status = steps.get("forensic_suite", {}).get("status")
    multi_seed_step_status = steps.get("multi_seed_confirmation", {}).get("status")

    if promotion_step_status != "completed":
        investigation_failures.append("promotion_gate_incomplete")
    elif promotion_gate_report is not None and promotion_gate_report.get("passed") is False:
        blocking_failures.append("promotion_gate_failed")
        if promotion_failures_indicate_runtime_failure(promotion_failure_reasons):
            blocking_failures.append("runtime_failed")

    if forensic_step_status != "completed":
        investigation_failures.append("forensic_gate_incomplete")
    elif forensic_passed is False:
        blocking_failures.append("forensic_gate_failed")

    if multi_seed_step_status != "completed":
        investigation_failures.append("multi_seed_incomplete")

    run_count = len(runs)
    summary_status = summary.get("status")
    multi_seed_report_passed = None if multi_seed_summary_report is None else multi_seed_summary_report.get("passed")
    if multi_seed_summary_report is not None:
        manifest_multi_seed_passed = summary_status == "passed"
        if multi_seed_report_passed != manifest_multi_seed_passed:
            investigation_failures.append("multi_seed_summary_inconsistent")

    if summary_status == "failed" or any(run.get("passed") is False for run in runs):
        blocking_failures.append("multi_seed_failed")
    elif summary_status == "passed" and run_count < 3:
        investigation_failures.append("multi_seed_incomplete")
    elif summary_status not in {"passed", "failed"}:
        investigation_failures.append("multi_seed_incomplete")

    blocking_failures = deduplicate_reasons(blocking_failures)
    investigation_failures = deduplicate_reasons(investigation_failures)

    if blocking_failures:
        readiness["state"] = "not_ready"
        readiness["failure_reasons"] = blocking_failures
    elif investigation_failures:
        readiness["state"] = "needs_investigation"
        readiness["failure_reasons"] = investigation_failures
    else:
        readiness["state"] = "promotion_candidate"
        readiness["passed"] = True

    manifest["readiness"] = readiness
    write_manifest(manifest_path, manifest)
    return readiness
