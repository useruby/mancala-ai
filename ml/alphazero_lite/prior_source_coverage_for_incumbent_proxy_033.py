from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "azlite_prior_source_coverage_for_incumbent_proxy_033_v1"
ARTIFACT_PINNED_ABLATION_SCHEMA = "azlite_artifact_pinned_value_trust_ablation_v1"
SEARCH_INTERACTION_SCHEMA = "azlite_search_interaction_diagnostic_v1"
TARGET_ROW_ID = "incumbent_proxy_disagreement-033"
GUARD_ROW_IDS = ["capture_available-002", "capture_available-003"]
CONTEXT_ROW_IDS = ["incumbent_proxy_disagreement-031"]
ROW_IDS = [TARGET_ROW_ID, *GUARD_ROW_IDS, *CONTEXT_ROW_IDS]
EXPECTED_ROW_BUCKETS = {
    TARGET_ROW_ID: "incumbent_proxy_disagreement",
    "incumbent_proxy_disagreement-031": "incumbent_proxy_disagreement",
    "capture_available-002": "capture_available",
    "capture_available-003": "capture_available",
}
REPLAY_MEMBERSHIP_FIELDS = ["row_id", "source_row_id", "id", "stable_id"]
CLOSE_DECISION = "close_033_source_coverage_variant"
WRITE_CANDIDATE_SPEC_DECISION = "write_targeted_source_coverage_candidate_spec"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_schema(payload: dict[str, Any], expected_schema: str, path: Path) -> None:
    actual_schema = payload.get("schema")
    if actual_schema != expected_schema:
        raise ValueError(f"unexpected schema for {path}: {actual_schema}")


def replay_memberships(
    path: Path, canonical_states: dict[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    memberships = {
        row_id: {"present": False, "count": 0, "matched_by": [], "line_numbers": []}
        for row_id in ROW_IDS
    }
    if not path.exists():
        raise FileNotFoundError(path)
    canonical_states = canonical_states or {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            for row_id in ROW_IDS:
                matched_by = [
                    field
                    for field in REPLAY_MEMBERSHIP_FIELDS
                    if payload.get(field) == row_id
                ]
                row_canonical_states = canonical_states.get(row_id)
                if row_canonical_states is not None and not isinstance(
                    row_canonical_states, list
                ):
                    row_canonical_states = [row_canonical_states]
                if (
                    row_canonical_states
                    and payload.get("canonical_state") in row_canonical_states
                ):
                    matched_by.append("canonical_state")
                if matched_by:
                    memberships[row_id]["present"] = True
                    memberships[row_id]["count"] += 1
                    memberships[row_id]["line_numbers"].append(line_number)
                    for field in matched_by:
                        if field not in memberships[row_id]["matched_by"]:
                            memberships[row_id]["matched_by"].append(field)
                    expected_bucket = EXPECTED_ROW_BUCKETS[row_id]
                    actual_bucket = payload.get("bucket")
                    if actual_bucket is not None and actual_bucket != expected_bucket:
                        memberships[row_id]["classification"] = "misclassified"
                        evidence = memberships[row_id].setdefault(
                            "classification_evidence", []
                        )
                        for field in matched_by:
                            evidence.append(
                                {
                                    "expected_bucket": expected_bucket,
                                    "actual_bucket": actual_bucket,
                                    "line_number": line_number,
                                    "matched_by": field,
                                }
                            )
    return memberships


def selected_candidate_forensics_path(
    search_interaction_payload: dict[str, Any],
) -> Path | None:
    rebalanced_run_dir = search_interaction_payload.get("rebalanced_run_dir")
    if not rebalanced_run_dir:
        return None
    path = Path(rebalanced_run_dir) / "final" / "selected_candidate_forensics.json"
    return path if path.exists() else None


def canonical_states_from_forensics(path: Path) -> dict[str, str]:
    payload = load_json(path)
    states: dict[str, str] = {}
    for system in (payload.get("systems") or {}).values():
        for row in system.get("rows") or []:
            row_id = row.get("id") or row.get("row_id")
            canonical_state = row.get("canonical_state")
            if row_id in ROW_IDS and canonical_state is not None:
                states[row_id] = canonical_state
    return states


def load_canonical_state_provenance(
    *, candidate_payload: dict[str, Any], rebalance_payload: dict[str, Any]
) -> dict[str, Any]:
    lane_states: dict[str, dict[str, str]] = {}
    paths: dict[str, str | None] = {}
    for lane, payload in [
        ("candidate_search_interaction", candidate_payload),
        ("rebalance_search_interaction", rebalance_payload),
    ]:
        path = selected_candidate_forensics_path(payload)
        paths[lane] = str(path) if path else None
        lane_states[lane] = canonical_states_from_forensics(path) if path else {}

    states: dict[str, str] = {}
    sources: dict[str, str] = {}
    row_states: dict[str, dict[str, str | None]] = {}
    conflicts: dict[str, dict[str, str]] = {}
    for row_id in ROW_IDS:
        candidate_state = lane_states["candidate_search_interaction"].get(row_id)
        rebalance_state = lane_states["rebalance_search_interaction"].get(row_id)
        row_states[row_id] = {
            "candidate_search_interaction": candidate_state,
            "rebalance_search_interaction": rebalance_state,
        }
        if (
            candidate_state is not None
            and rebalance_state is not None
            and candidate_state != rebalance_state
        ):
            conflicts[row_id] = {
                "candidate_search_interaction": candidate_state,
                "rebalance_search_interaction": rebalance_state,
            }
        for lane in ["candidate_search_interaction", "rebalance_search_interaction"]:
            canonical_state = lane_states[lane].get(row_id)
            if canonical_state is not None:
                states[row_id] = canonical_state
                sources[row_id] = lane
                break

    return {
        "states": states,
        "sources": sources,
        "row_states": row_states,
        "conflicts": conflicts,
        "paths": paths,
        "missing_row_ids": [row_id for row_id in ROW_IDS if row_id not in states],
    }


def load_source_artifacts(
    *,
    artifact_pinned_ablation: Path,
    candidate_search_interaction: Path,
    rebalance_search_interaction: Path,
    replay_source: Path,
) -> dict[str, Any]:
    pinned_payload = load_json(artifact_pinned_ablation)
    validate_schema(
        payload=pinned_payload,
        expected_schema=ARTIFACT_PINNED_ABLATION_SCHEMA,
        path=artifact_pinned_ablation,
    )
    candidate_payload = load_json(candidate_search_interaction)
    validate_schema(
        payload=candidate_payload,
        expected_schema=SEARCH_INTERACTION_SCHEMA,
        path=candidate_search_interaction,
    )
    rebalance_payload = load_json(rebalance_search_interaction)
    validate_schema(
        payload=rebalance_payload,
        expected_schema=SEARCH_INTERACTION_SCHEMA,
        path=rebalance_search_interaction,
    )
    canonical_state_provenance = load_canonical_state_provenance(
        candidate_payload=candidate_payload,
        rebalance_payload=rebalance_payload,
    )
    return {
        "artifact_pinned_ablation": {
            "path": str(artifact_pinned_ablation),
            "payload": pinned_payload,
        },
        "candidate_search_interaction": {
            "path": str(candidate_search_interaction),
            "payload": candidate_payload,
        },
        "rebalance_search_interaction": {
            "path": str(rebalance_search_interaction),
            "payload": rebalance_payload,
        },
        "canonical_state_provenance": canonical_state_provenance,
        "replay_source": {
            "path": str(replay_source),
            "memberships": replay_memberships(
                replay_source,
                canonical_states={
                    row_id: [
                        state for state in row_states.values() if state is not None
                    ]
                    for row_id, row_states in (
                        canonical_state_provenance.get("row_states") or {}
                    ).items()
                },
            ),
        },
    }


def row_role(row_id: str) -> str:
    if row_id == TARGET_ROW_ID:
        return "target"
    if row_id in GUARD_ROW_IDS:
        return "guard"
    return "context"


def _search_row(
    source_artifacts: dict[str, Any], artifact_key: str, row_id: str
) -> dict[str, Any] | None:
    payload = source_artifacts.get(artifact_key, {}).get("payload") or {}
    return (payload.get("rows") or {}).get(row_id)


def _pinned_row(source_artifacts: dict[str, Any], row_id: str) -> dict[str, Any]:
    payload = source_artifacts["artifact_pinned_ablation"]["payload"]
    return (payload.get("rows") or {}).get(row_id) or {}


def _full_default(pinned_row: dict[str, Any]) -> dict[str, Any]:
    return (pinned_row.get("configs") or {}).get("full_default") or {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def build_row_entry(*, source_artifacts: dict[str, Any], row_id: str) -> dict[str, Any]:
    pinned_row = _pinned_row(source_artifacts, row_id)
    candidate_row = _search_row(
        source_artifacts, "candidate_search_interaction", row_id
    )
    rebalance_row = _search_row(
        source_artifacts, "rebalance_search_interaction", row_id
    )
    default_config = _full_default(pinned_row)
    replay_membership = (
        source_artifacts.get("replay_source", {}).get("memberships") or {}
    ).get(row_id)
    canonical_state_provenance = (
        source_artifacts.get("canonical_state_provenance") or {}
    )
    canonical_state = (canonical_state_provenance.get("states") or {}).get(row_id)
    reference_values = [
        value
        for value in [
            pinned_row.get("reference_move"),
            (candidate_row or {}).get("reference_move"),
            (rebalance_row or {}).get("reference_move"),
        ]
        if value is not None
    ]
    teacher_values = [
        value
        for value in [
            (candidate_row or {}).get("teacher_value"),
            (rebalance_row or {}).get("teacher_value"),
        ]
        if value is not None
    ]
    notes = []
    if candidate_row is None:
        notes.append("candidate search interaction row unavailable")
    if rebalance_row is None:
        notes.append("rebalance search interaction row unavailable")
    if replay_membership is None:
        notes.append("replay source membership unavailable")
    if canonical_state is None:
        notes.append("canonical state unavailable")
    row_canonical_states = (canonical_state_provenance.get("row_states") or {}).get(
        row_id
    ) or {}
    canonical_state_conflict = row_id in (
        canonical_state_provenance.get("conflicts") or {}
    )
    if canonical_state_conflict:
        notes.append(
            "canonical state conflict between candidate and rebalance forensics"
        )

    return {
        "row_id": row_id,
        "row_role": row_role(row_id),
        "canonical_state": canonical_state,
        "canonical_state_provenance": {
            "source": (canonical_state_provenance.get("sources") or {}).get(row_id),
            "candidate_canonical_state": row_canonical_states.get(
                "candidate_search_interaction"
            ),
            "rebalance_canonical_state": row_canonical_states.get(
                "rebalance_search_interaction"
            ),
            "conflict": canonical_state_conflict,
            "paths": canonical_state_provenance.get("paths"),
        },
        "reference_move": _first_present(
            pinned_row.get("reference_move"),
            (candidate_row or {}).get("reference_move"),
            (rebalance_row or {}).get("reference_move"),
        ),
        "teacher_value": _first_present(
            (candidate_row or {}).get("teacher_value"),
            (rebalance_row or {}).get("teacher_value"),
        ),
        "full_default_selected_move": default_config.get("selected_move"),
        "passes_reference": default_config.get("passes_reference"),
        "bucket": _first_present(
            (candidate_row or {}).get("bucket"), (rebalance_row or {}).get("bucket")
        ),
        "phase": _first_present(
            (candidate_row or {}).get("phase"), (rebalance_row or {}).get("phase")
        ),
        "row_state_provenance": {
            "candidate_search_interaction": candidate_row is not None,
            "rebalance_search_interaction": rebalance_row is not None,
        },
        "source_fixture_provenance": {
            "artifact_pinned_ablation": pinned_row if pinned_row else None,
        },
        "forensic_row_provenance": {
            "candidate_search_interaction": candidate_row,
            "rebalance_search_interaction": rebalance_row,
        },
        "opening_family_provenance": {
            "candidate_opening_row": (candidate_row or {}).get("opening_family")
            if candidate_row
            else None,
            "rebalance_opening_row": (rebalance_row or {}).get("opening_family")
            if rebalance_row
            else None,
        },
        "replay_source_membership": replay_membership,
        "reference_values": reference_values,
        "teacher_values": teacher_values,
        "selected_artifact_paths": {
            "artifact_pinned_ablation": source_artifacts["artifact_pinned_ablation"][
                "path"
            ],
            "candidate_search_interaction": source_artifacts[
                "candidate_search_interaction"
            ]["path"],
            "rebalance_search_interaction": source_artifacts[
                "rebalance_search_interaction"
            ]["path"],
        },
        "notes": notes,
    }


def classify_source_gap(rows: dict[str, dict[str, Any]]) -> str:
    target = rows[TARGET_ROW_ID]
    membership = target.get("replay_source_membership") or {}
    if membership.get("present") is False:
        return "absent_from_source_construction"
    if membership.get("classification") == "misclassified":
        return "misclassified_in_source_construction"
    if membership.get("count", 1) == 0:
        return "absent_from_source_construction"
    if membership.get("count", 1) > 0 and membership.get("count", 1) < min(
        (rows[row_id].get("replay_source_membership") or {}).get("count", 1)
        for row_id in GUARD_ROW_IDS
    ):
        return "underrepresented_in_source_construction"
    if (
        len(set(target.get("reference_values") or [])) > 1
        or len(set(target.get("teacher_values") or [])) > 1
    ):
        return "reference_or_teacher_drift_detected"
    return "no_concrete_source_gap_found"


def guard_rows_clean(rows: dict[str, dict[str, Any]]) -> bool:
    for row_id in GUARD_ROW_IDS:
        row = rows[row_id]
        membership = row.get("replay_source_membership") or {}
        if not membership.get("present"):
            return False
        if membership.get("count", 0) <= 0:
            return False
        if membership.get("classification") == "misclassified":
            return False
        if len(set(row.get("reference_values") or [])) > 1:
            return False
        if len(set(row.get("teacher_values") or [])) > 1:
            return False
    return True


def decision_for(*, classification: str, guards_clean: bool) -> str:
    if (
        classification
        in {"absent_from_source_construction", "misclassified_in_source_construction"}
        and guards_clean
    ):
        return WRITE_CANDIDATE_SPEC_DECISION
    return CLOSE_DECISION


def build_payload(*, source_artifacts: dict[str, Any]) -> dict[str, Any]:
    rows = {
        row_id: build_row_entry(source_artifacts=source_artifacts, row_id=row_id)
        for row_id in ROW_IDS
    }
    classification = classify_source_gap(rows)
    guards_clean = guard_rows_clean(rows)
    decision = decision_for(classification=classification, guards_clean=guards_clean)
    return {
        "schema": SCHEMA,
        "target_row_id": TARGET_ROW_ID,
        "guard_row_ids": list(GUARD_ROW_IDS),
        "context_row_ids": list(CONTEXT_ROW_IDS),
        "source_artifacts": {
            key: value.get("path")
            for key, value in source_artifacts.items()
            if isinstance(value, dict) and "path" in value
        },
        "canonical_state_provenance": source_artifacts.get(
            "canonical_state_provenance"
        ),
        "probe_settings": (
            source_artifacts["artifact_pinned_ablation"]["payload"].get(
                "probe_settings"
            )
            or None
        ),
        "rows": rows,
        "summary": {
            "decision": decision,
            "source_gap_classification": classification,
            "future_candidate_justified": decision == WRITE_CANDIDATE_SPEC_DECISION,
            "guards_clean": guards_clean,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose prior/source coverage for incumbent_proxy_disagreement-033"
    )
    parser.add_argument("--artifact-pinned-ablation", type=Path, required=True)
    parser.add_argument("--candidate-search-interaction", type=Path, required=True)
    parser.add_argument("--rebalance-search-interaction", type=Path, required=True)
    parser.add_argument("--replay-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_artifacts = load_source_artifacts(
        artifact_pinned_ablation=args.artifact_pinned_ablation,
        candidate_search_interaction=args.candidate_search_interaction,
        rebalance_search_interaction=args.rebalance_search_interaction,
        replay_source=args.replay_source,
    )
    payload = build_payload(source_artifacts=source_artifacts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "artifact_path": str(args.out),
                "schema": SCHEMA,
                "decision": payload["summary"]["decision"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
