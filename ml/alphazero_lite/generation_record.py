#!/usr/bin/env python3
"""Canonical, versioned provenance records for AlphaZero-lite generations.

This deliberately records stage summaries and immutable artifact references, not
large evaluator telemetry. Detailed domain reports remain their own artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ml.alphazero_lite.export_artifact import sha256_file

SCHEMA = "azlite_generation_record_v1"
STATUSES = {
    "planned",
    "self_play_complete",
    "training_complete",
    "candidate_ready",
    "evaluated",
    "rejected",
    "promotion_ready",
    "promoted",
    "invalid",
}
NOT_RUN = {"status": "not_run", "reason": None}


class GenerationRecordError(ValueError):
    """Raised when a generation record is malformed or internally inconsistent."""


def artifact_ref(
    role: str,
    path: str | Path,
    *,
    schema: str | None = None,
    sha256: str | None = None,
    required: bool = False,
) -> dict[str, Any]:
    """Return the shared artifact reference shape, hashing a local file when present."""
    value = Path(path)
    exists = value.is_file()
    return {
        "role": role,
        "path": str(path),
        "sha256": sha256 or (sha256_file(value) if exists else None),
        "size_bytes": value.stat().st_size if exists else None,
        "schema": schema,
        "required": bool(required),
    }


def new_record(
    *,
    generation_id: str,
    parent: dict[str, Any] | None = None,
    status: str = "planned",
    notes: list[str] | None = None,
) -> dict[str, Any]:
    if status not in STATUSES:
        raise GenerationRecordError(f"unsupported generation status: {status}")
    return {
        "schema": SCHEMA,
        "generation_id": generation_id,
        "status": status,
        "parent": parent
        or {"version": None, "weights_sha256": None, "metadata_sha256": None},
        "self_play": {
            "status": "not_run",
            "config": {},
            "seeds": [],
            "artifact": None,
            "compute": {},
        },
        "replay": {"sources": []},
        "training": {
            "status": "not_run",
            "config": {},
            "seed": None,
            "optimizer_updates": None,
            "checkpoint_policy": None,
            "metrics": {},
            "compute": {},
        },
        "candidate": {
            "version": None,
            "checkpoint": None,
            "weights": None,
            "metadata": None,
        },
        "diagnostics": {
            name: dict(NOT_RUN)
            for name in (
                "policy",
                "value",
                "exact_oracle",
                "regressions",
                "diagnostic_arena",
            )
        },
        "canonical_evaluation": {
            "prefilter": dict(NOT_RUN),
            "hard_arena": dict(NOT_RUN),
            "downstream": dict(NOT_RUN),
        },
        "promotion": {
            "decision": "not_evaluated",
            "failure_reasons": [],
            "gate_report": None,
        },
        "compute": {},
        "artifacts": [],
        "provenance_notes": notes or [],
    }


def attach_artifact(record: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact.get("role") or not artifact.get("path"):
        raise GenerationRecordError("artifact references require role and path")
    record["artifacts"] = [
        item for item in record["artifacts"] if item["role"] != artifact["role"]
    ]
    record["artifacts"].append(artifact)
    record["artifacts"].sort(key=lambda item: item["role"])
    return record


def record_self_play(
    record: dict[str, Any],
    *,
    config: dict[str, Any],
    seeds: list[int],
    artifact: dict[str, Any],
    compute: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record["self_play"] = {
        "status": "complete",
        "config": config,
        "seeds": [int(seed) for seed in seeds],
        "artifact": artifact,
        "compute": compute or {},
    }
    attach_artifact(record, artifact)
    record["status"] = "self_play_complete"
    return record


def record_training(
    record: dict[str, Any],
    *,
    config: dict[str, Any],
    seed: int,
    metrics: dict[str, Any] | None = None,
    optimizer_updates: int | None = None,
    checkpoint_policy: str | None = None,
    compute: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record["training"] = {
        "status": "complete",
        "config": config,
        "seed": int(seed),
        "optimizer_updates": optimizer_updates,
        "checkpoint_policy": checkpoint_policy,
        "metrics": metrics or {},
        "compute": compute or {},
    }
    record["status"] = "training_complete"
    return record


def record_candidate(
    record: dict[str, Any],
    *,
    version: str,
    checkpoint: dict[str, Any],
    weights: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    record["candidate"] = {
        "version": version,
        "checkpoint": checkpoint,
        "weights": weights,
        "metadata": metadata,
    }
    for artifact in (checkpoint, weights, metadata):
        attach_artifact(record, artifact)
    record["status"] = "candidate_ready"
    return record


def attach_diagnostic(
    record: dict[str, Any],
    name: str,
    *,
    tool: str,
    config: dict[str, Any],
    summary: dict[str, Any],
    artifact: dict[str, Any] | None = None,
    status: str = "complete",
) -> dict[str, Any]:
    # Named standard slots preserve discoverability; arbitrary names remain forward-compatible.
    record["diagnostics"][name] = {
        "status": status,
        "tool": tool,
        "config": config,
        "summary": summary,
        "artifact": artifact,
    }
    if artifact:
        attach_artifact(record, artifact)
    return record


def attach_evaluation(
    record: dict[str, Any],
    name: str,
    *,
    status: str,
    tool: str | None = None,
    config: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record["canonical_evaluation"][name] = {
        "status": status,
        "tool": tool,
        "config": config or {},
        "summary": summary or {},
        "artifact": artifact,
    }
    if artifact:
        attach_artifact(record, artifact)
    return record


def record_promotion(
    record: dict[str, Any],
    *,
    decision: str,
    failure_reasons: list[Any] | None = None,
    gate_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if decision not in {"not_evaluated", "passed", "rejected", "promoted"}:
        raise GenerationRecordError(f"unsupported promotion decision: {decision}")
    record["promotion"] = {
        "decision": decision,
        "failure_reasons": failure_reasons or [],
        "gate_report": gate_report,
    }
    if gate_report:
        attach_artifact(record, gate_report)
    if decision == "promoted":
        record["status"] = "promoted"
    elif decision == "rejected":
        record["status"] = "rejected"
    elif decision == "passed":
        record["status"] = "promotion_ready"
    return record


def load_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    validate(record, base_dir=path.parent)
    return record


def write_record(path: Path, record: dict[str, Any]) -> None:
    validate(record, base_dir=path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(encoded)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def validate(record: dict[str, Any], *, base_dir: Path | None = None) -> None:
    if record.get("schema") != SCHEMA:
        raise GenerationRecordError(
            f"unsupported generation schema: {record.get('schema')}"
        )
    if not isinstance(record.get("generation_id"), str) or not record["generation_id"]:
        raise GenerationRecordError("generation_id is required")
    if record.get("status") not in STATUSES:
        raise GenerationRecordError("invalid generation status")
    for artifact in record.get("artifacts", []):
        if not artifact.get("sha256") and artifact.get("required"):
            raise GenerationRecordError(
                f"required artifact lacks SHA256: {artifact.get('role')}"
            )
        local_path = Path(artifact["path"])
        # Pipeline artifacts use repository-relative paths while records live under
        # docs/. Resolve an existing working-directory path before the record path.
        if not local_path.is_absolute() and not local_path.exists() and base_dir:
            local_path = base_dir / local_path
        if (
            local_path.is_file()
            and artifact.get("sha256")
            and sha256_file(local_path) != artifact["sha256"]
        ):
            raise GenerationRecordError(f"artifact SHA256 mismatch: {artifact['path']}")
        if (
            artifact.get("required")
            and local_path.exists() is False
            and base_dir is not None
        ):
            raise GenerationRecordError(
                f"required artifact missing: {artifact['path']}"
            )
    trained = record["status"] not in {"planned", "invalid"}
    if trained and not record["parent"].get("weights_sha256"):
        raise GenerationRecordError("trained descendants require parent weights_sha256")
    self_play = record["self_play"]
    if self_play["status"] == "complete" and (
        not self_play["seeds"] or not self_play["config"] or not self_play["artifact"]
    ):
        raise GenerationRecordError(
            "completed self-play requires config, seeds, and artifact"
        )
    training = record["training"]
    if training["status"] == "complete" and (
        training["seed"] is None or not training["config"]
    ):
        raise GenerationRecordError("completed training requires seed and config")
    if record["status"] in {
        "candidate_ready",
        "evaluated",
        "rejected",
        "promotion_ready",
        "promoted",
    }:
        candidate = record["candidate"]
        if not candidate.get("version") or not all(
            candidate.get(name) and candidate[name].get("sha256")
            for name in ("weights", "metadata")
        ):
            raise GenerationRecordError(
                "candidate-ready records require complete candidate identity"
            )
    decision = record["promotion"]["decision"]
    if record["status"] == "promoted" and decision != "promoted":
        raise GenerationRecordError("promoted records require promoted decision")
    if decision == "rejected" and not record["promotion"]["failure_reasons"]:
        raise GenerationRecordError("rejected records require failure reasons")
    gate = record["promotion"].get("gate_report")
    if (
        gate
        and record["candidate"].get("weights")
        and gate.get("candidate_weights_sha256")
        not in (None, record["candidate"]["weights"]["sha256"])
    ):
        raise GenerationRecordError(
            "gate candidate identity does not match record candidate"
        )


def migrate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") != "azlite_run_manifest_v1":
        raise GenerationRecordError("expected azlite_run_manifest_v1")
    parent = {
        "version": manifest.get("parent_version"),
        "weights_sha256": None,
        "metadata_sha256": None,
    }
    record = new_record(
        generation_id=f"{manifest['run_id']}-iter{manifest['iteration']}",
        parent=parent,
        notes=[
            "Imported from azlite_run_manifest_v1; hashes were not recorded by the legacy manifest."
        ],
    )
    record["legacy_manifest"] = manifest
    return record


def validate_index(path: Path) -> None:
    index = json.loads(path.read_text(encoding="utf-8"))
    if index.get("schema") != "azlite_generation_index_v1":
        raise GenerationRecordError("unsupported generation index schema")
    identifiers: set[str] = set()
    for row in index.get("generations", []):
        generation_id = row.get("generation_id")
        if not isinstance(generation_id, str) or generation_id in identifiers:
            raise GenerationRecordError(
                "generation index contains duplicate or invalid generation_id"
            )
        identifiers.add(generation_id)
        record_path = path.parents[3] / row["record"]
        record = load_record(record_path)
        if record["generation_id"] != generation_id:
            raise GenerationRecordError("generation index id does not match record")


def merge_gate_report(record: dict[str, Any], report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidate = report.get("candidate_identity") or {}
    candidate_weights = candidate.get("weights_json_sha256")
    if candidate_weights:
        report_ref = artifact_ref(
            "gate_report", report_path, schema=report.get("schema"), required=True
        )
        report_ref["candidate_weights_sha256"] = candidate_weights
    else:
        report_ref = artifact_ref(
            "gate_report", report_path, schema=report.get("schema"), required=True
        )
    # Canonical gate attempts are immutable evidence.  Keep prior attempts when a
    # frozen candidate is evaluated under a new runtime policy.
    attempts = record.setdefault("canonical_evaluation_attempts", [])
    if not attempts and record.get("promotion", {}).get("gate_report"):
        attempts.append(
            {
                "gate_report": record["promotion"]["gate_report"],
                "canonical_evaluation": record.get("canonical_evaluation", {}),
            }
        )
    attempt = {
        "gate_report": report_ref,
        "runtime_search_policy": report.get("runtime_search_policy", {}),
        "classification": report.get("shadow_classification"),
        "canonical_evaluation": {},
    }
    attempts.append(attempt)
    arena_path = report.get("arena_report_path")
    if arena_path:
        prefilter_score = report.get("arena_score")
        prefilter_threshold = report.get("min_arena_score")
        attach_evaluation(
            record,
            "prefilter",
            status="complete",
            tool="local_promotion_gate",
            summary={
                "score": prefilter_score,
                "passed": (
                    None
                    if prefilter_score is None or prefilter_threshold is None
                    else float(prefilter_score) >= float(prefilter_threshold)
                ),
            },
            artifact=artifact_ref("canonical_prefilter", arena_path, schema="arena_v1"),
        )
        attempt["canonical_evaluation"]["prefilter"] = record["canonical_evaluation"][
            "prefilter"
        ]
    hard_path = report.get("hard_report_path")
    if hard_path:
        attach_evaluation(
            record,
            "hard_arena",
            status="complete",
            tool="local_promotion_gate",
            summary={
                "score": report.get("hard_score"),
                "passed": float(report["hard_score"])
                >= float(report.get("hard_min_score", 0.0)),
            },
            artifact=artifact_ref("canonical_hard_arena", hard_path, schema="arena_v1"),
        )
        attempt["canonical_evaluation"]["hard_arena"] = record["canonical_evaluation"][
            "hard_arena"
        ]
    elif report.get("passed") is False:
        attach_evaluation(
            record,
            "hard_arena",
            status="not_run",
            tool="local_promotion_gate",
            summary={"reason": "prefilter_failed"},
        )
        attempt["canonical_evaluation"]["hard_arena"] = record["canonical_evaluation"][
            "hard_arena"
        ]
    decision = "passed" if report.get("passed") else "rejected"
    return record_promotion(
        record,
        decision=decision,
        failure_reasons=report.get("failure_reasons", []),
        gate_report=report_ref,
    )


def command_options(command: list[str]) -> dict[str, Any]:
    """Serialize ordinary ``--flag value`` pipeline command options for provenance."""
    options: dict[str, Any] = {}
    index = 0
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            index += 1
            continue
        key = token[2:].replace("-", "_")
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            options[key] = command[index + 1]
            index += 2
        else:
            options[key] = True
            index += 1
    return options


def record_pipeline_step(
    record: dict[str, Any], step: dict[str, Any]
) -> dict[str, Any]:
    """Apply a normal pipeline step result without encoding experiment-specific logic."""
    if step.get("status") != "completed":
        return record
    command = step.get("command", [])
    if not isinstance(command, list):
        return record
    options = command_options(command)
    name = step.get("name")
    duration = {"wall_clock_duration_seconds": step.get("duration_s")}
    if name == "self_play" and options.get("out"):
        seeds = [
            int(value)
            for value in str(options.get("seed_sweep", options.get("seed", ""))).split(
                ","
            )
            if value
        ]
        output = artifact_ref(
            "self_play",
            options["out"],
            schema="azlite_self_play_jsonl_v1",
            required=True,
        )
        record_self_play(
            record, config=options, seeds=seeds, artifact=output, compute=duration
        )
    elif name == "train" and "seed" in options:
        record_training(
            record,
            config=options,
            seed=int(options["seed"]),
            checkpoint_policy=options.get(
                "final_checkpoint", "production_default_best_validation"
            ),
            compute=duration,
        )
        paths = str(options.get("data_files", options.get("data", ""))).split(",")
        weights = str(options.get("replay_weights", "")).split(",")
        record["replay"] = {
            "sources": [
                {
                    "artifact": artifact_ref(
                        "replay_source", path, schema="azlite_self_play_jsonl_v1"
                    ),
                    "weight": int(weights[index])
                    if index < len(weights) and weights[index]
                    else 1,
                }
                for index, path in enumerate(paths)
                if path
            ]
        }
    elif name == "export_artifact" and options.get("out_dir"):
        output = Path(options["out_dir"])
        record_candidate(
            record,
            version=str(options.get("version")),
            checkpoint=artifact_ref(
                "checkpoint", options["checkpoint"], schema="npz", required=True
            ),
            weights=artifact_ref(
                "weights",
                output / "weights.json",
                schema="azlite_model_weights_v1",
                required=True,
            ),
            metadata=artifact_ref(
                "metadata",
                output / "metadata.json",
                schema="azlite_model_v1",
                required=True,
            ),
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage canonical AlphaZero-lite generation records"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--out", type=Path, required=True)
    init.add_argument("--generation-id", required=True)
    validate_command = subparsers.add_parser("validate")
    validate_command.add_argument("record", type=Path)
    show = subparsers.add_parser("show")
    show.add_argument("record", type=Path)
    migrate = subparsers.add_parser("migrate-manifest")
    migrate.add_argument("manifest", type=Path)
    migrate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "init":
        write_record(args.out, new_record(generation_id=args.generation_id))
    elif args.command == "validate":
        load_record(args.record)
    elif args.command == "show":
        print(json.dumps(load_record(args.record), indent=2, sort_keys=True))
    else:
        write_record(
            args.out,
            migrate_manifest(json.loads(args.manifest.read_text(encoding="utf-8"))),
        )


if __name__ == "__main__":
    main()
