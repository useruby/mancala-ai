"""Independent read-only verification of the completed seed542 amendment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import train, validate_seed500_pilot
from ml.alphazero_lite import run_seed542_amended_generation as runner
from ml.alphazero_lite import seed542_amendment
from ml.alphazero_lite import verify_seed542 as partial_verifier

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "docs/data/seed542-selfplay-4096"
PUBLISHED = {
    "accounting.json": "amended-accounting.json",
    "development-membership.json": "development-membership.json",
    "training-protocol.json": "training-protocol.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def current_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def current_installed_packages() -> list[str]:
    packages = (
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name") and dist.metadata["Name"].lower() != "pip"
    )
    return sorted(packages, key=str.casefold)


def verify_recorded_environment(
    amendment: dict[str, Any],
    environment: dict[str, str] | None = None,
    installed_packages: list[str] | None = None,
) -> None:
    environment = environment or current_environment()
    installed_packages = installed_packages or current_installed_packages()
    if environment != amendment["environment"]:
        raise ValueError("amended_runtime_environment_mismatch")
    if installed_packages != amendment["interpreter"]["installed_packages"]:
        raise ValueError("amended_runtime_dependency_set_mismatch")


def verify_runtime_assets(
    root: Path,
    amendment: dict[str, Any],
    *,
    tablebase_path: Path | None = None,
) -> None:
    for name, asset in amendment["assets"].items():
        path = (
            tablebase_path
            if name == "tablebase" and tablebase_path is not None
            else root / asset["path"]
        )
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or sha256(path) != asset["sha256"]:
            raise ValueError(f"amended_runtime_asset_sha256_mismatch:{name}")
        if (
            asset.get("size_bytes") is not None
            and path.stat().st_size != asset["size_bytes"]
        ):
            raise ValueError(f"amended_runtime_asset_size_mismatch:{name}")


def _read_partition_rows(
    sources: list[dict[str, Any]], role: str
) -> tuple[list[Path], list[dict[str, Any]], dict[str, Any]]:
    paths: list[Path] = []
    raw_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    offset = 0
    identities: list[str] = []
    for source in sources:
        if source["role"] != role:
            continue
        path = ROOT / source["path"]
        data = path.read_bytes()
        source_digest = hashlib.sha256(data).hexdigest()
        if source_digest != source["sha256"]:
            raise ValueError(f"amended_source_sha256_mismatch:{source['path']}")
        lines = data.splitlines()
        rows_in_file = []
        for line_number, line in enumerate(lines, 1):
            row = json.loads(line)
            identity = f"{source['path']}:{line_number}"
            row.update(
                {
                    "_identity": identity,
                    "_path": path,
                    "_line": line_number,
                    "_source_sha256": source_digest,
                    "_line_sha256": hashlib.sha256(line).hexdigest(),
                }
            )
            rows_in_file.append(row)
            identities.append(identity)
        game_count = int(source["games"])
        game_ids = sorted({int(row["game_index"]) for row in rows_in_file})
        if game_ids != list(range(game_count)):
            raise ValueError(f"amended_source_game_indices_mismatch:{source['path']}")
        if {int(row["game_seed"]) for row in rows_in_file} != {
            int(source["source_seed"])
        }:
            raise ValueError(f"amended_source_seed_mismatch:{source['path']}")
        if any(
            row.get("game_completed") is not True or row.get("game_capped") is True
            for row in rows_in_file
        ):
            raise ValueError(f"amended_source_has_noncomplete_row:{source['path']}")
        paths.append(path)
        raw_rows.extend(rows_in_file)
        replay_rows.extend(
            {**row, "game_index": int(row["game_index"]) + offset}
            for row in rows_in_file
        )
        offset += game_count
    if not paths:
        raise ValueError(f"amended_source_role_empty:{role}")
    replay = validate_seed500_pilot._validate_and_account(replay_rows)
    x, policies, values = train.load_jsonl_replay(
        paths,
        weights=[1] * len(paths),
        policy_target_mode="default",
        value_target_mode="default",
    )[:3]
    if (
        x.shape != (len(raw_rows), 27)
        or policies.shape != (len(raw_rows), 6)
        or values.shape != (len(raw_rows), 1)
    ):
        raise ValueError(f"amended_production_loader_shape_mismatch:{role}")
    expected_x = np.asarray([row["state"] for row in raw_rows], dtype=np.float32)
    expected_p = np.asarray([row["policy"] for row in raw_rows], dtype=np.float32)
    expected_v = np.asarray([[row["value"]] for row in raw_rows], dtype=np.float32)
    if not (
        np.array_equal(x, expected_x)
        and np.array_equal(policies, expected_p)
        and np.array_equal(values, expected_v)
    ):
        raise ValueError(f"amended_production_loader_value_mismatch:{role}")
    replay.update(
        {
            "loader_accepted_rows": len(x),
            "loader_rejected_rows": 0,
            "loader_rejection_reasons": {},
            "row_identity_count": len(raw_rows),
            "row_identity_sha256": hashlib.sha256(
                "\n".join(identities).encode()
            ).hexdigest(),
            "target_mode_counts": {
                name: dict(Counter(str(row.get(name, "missing")) for row in raw_rows))
                for name in (
                    "policy_target_mode",
                    "policy_target_actual_mode",
                    "value_target_mode",
                )
            },
            "source_rows_preserved_with_multiplicity": True,
        }
    )
    return paths, raw_rows, replay


def _canonical_key(row: dict[str, Any]) -> tuple[int, ...]:
    state = np.asarray(row["state"], dtype=np.float32)
    board = np.rint(state[:14] * np.float32(48)).astype(np.int64)
    return (*map(int, board), int(round(float(state[14]))))


def _input_key(row: dict[str, Any]) -> bytes:
    return np.asarray(row["state"], dtype=np.float32).tobytes()


def _membership(
    development_rows: list[dict[str, Any]], training_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    seed504_manifest = _read_json(ROOT / "docs/data/seed504/manifest.json")
    seed504_sources = seed504_manifest["sources"]["training"]
    old_rows: list[dict[str, Any]] = []
    for source in seed504_sources:
        path = ROOT / source["path"]
        if sha256(path) != source["sha256"]:
            raise ValueError(
                f"seed504_training_source_sha256_mismatch:{source['path']}"
            )
        for line in path.read_text(encoding="utf-8").splitlines():
            old_rows.append(json.loads(line))
    if len(old_rows) != 13724:
        raise ValueError(f"seed504_training_population_mismatch:{len(old_rows)}")
    canonical_training = {_canonical_key(row) for row in [*old_rows, *training_rows]}
    input_training = {_input_key(row) for row in [*old_rows, *training_rows]}
    membership = []
    included = []
    for row in development_rows:
        canonical_state = _canonical_key(row)
        encoded = _input_key(row)
        reasons = []
        if canonical_state in canonical_training:
            reasons.append("canonical_state_overlap")
        if encoded in input_training:
            reasons.append("production_float32_input_overlap")
        record = {
            "row_id": row["_identity"],
            "source_sha256": row["_source_sha256"],
            "line_sha256": row["_line_sha256"],
            "canonical_sha256": hashlib.sha256(
                json.dumps(canonical_state, separators=(",", ":")).encode()
            ).hexdigest(),
            "production_float32_input_sha256": hashlib.sha256(encoded).hexdigest(),
            "included": not reasons,
            "exclusion_reasons": reasons,
        }
        membership.append(record)
        if not reasons:
            included.append((canonical_state, encoded))
    return {
        "scope": "separation only from seed504 training and the amended 3,072-game training population",
        "membership": membership,
        "membership_sha256": hashlib.sha256(canonical(membership)).hexdigest(),
        "counts": {
            "development_source_rows": len(development_rows),
            "included_rows": len(included),
            "excluded_rows": len(development_rows) - len(included),
            "excluded_canonical": sum(
                "canonical_state_overlap" in row["exclusion_reasons"]
                for row in membership
            ),
            "excluded_float32": sum(
                "production_float32_input_overlap" in row["exclusion_reasons"]
                for row in membership
            ),
            "remaining_canonical_overlap": len(
                {key for key, _ in included} & canonical_training
            ),
            "remaining_float32_overlap": len(
                {key for _, key in included} & input_training
            ),
            "included_canonical_unique": len({key for key, _ in included}),
            "included_float32_unique": len({key for _, key in included}),
        },
    }


def _schedule_summary(schedule: list[dict[str, Any]]) -> dict[str, Any]:
    exposures = Counter(row for update in schedule for row in update["rows"])
    data = json.dumps(schedule, separators=(",", ":")).encode()
    return {
        "updates": len(schedule),
        "exposures": sum(exposures.values()),
        "unique_rows_covered": len(exposures),
        "maximum_row_exposure": max(exposures.values()),
        "exposure_counts": dict(sorted(exposures.items())),
        "schedule": schedule,
        "schedule_sha256": hashlib.sha256(data).hexdigest(),
    }


def _schedules(training_rows: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = _read_json(ROOT / "docs/data/seed504/manifest.json")
    permutation_path = ROOT / "docs/data/seed504/permutations.json"
    if sha256(permutation_path) != manifest["permutations_sha256"]:
        raise ValueError("seed504_frozen_permutations_sha256_mismatch")
    permutations = _read_json(permutation_path)
    old_sources = manifest["sources"]["training"]
    control_ids: list[str] = []
    for source in old_sources:
        path = ROOT / source["path"]
        rows = path.read_text(encoding="utf-8").splitlines()
        control_ids.extend(
            f"{source['path']}:{line_number}" for line_number, _ in enumerate(rows, 1)
        )
    if (
        len(control_ids) != 13724
        or len(permutations) != 4
        or any(len(order) != 13724 for order in permutations)
    ):
        raise ValueError("seed504_frozen_permutation_shape_mismatch")
    control_indices = [index for order in permutations for index in order]
    batch_lengths = [([512] * 26 + [412]) for _ in range(4)]
    lengths = [length for block in batch_lengths for length in block]
    if len(lengths) != 108 or sum(lengths) != 54896:
        raise ValueError("matched_schedule_batch_sequence_mismatch")
    if len(training_rows) < sum(lengths):
        raise ValueError("expanded_training_population_too_small")
    expanded_ids = [row["_identity"] for row in training_rows]
    generator = torch.Generator(device="cpu").manual_seed(542)
    expanded_indices = torch.randperm(len(expanded_ids), generator=generator).tolist()
    control_schedule = []
    expanded_schedule = []
    offset = 0
    for update, size in enumerate(lengths, 1):
        control_schedule.append(
            {
                "update": update,
                "batch_size": size,
                "rows": [
                    control_ids[i] for i in control_indices[offset : offset + size]
                ],
            }
        )
        expanded_schedule.append(
            {
                "update": update,
                "batch_size": size,
                "rows": [
                    expanded_ids[i] for i in expanded_indices[offset : offset + size]
                ],
            }
        )
        offset += size
    protocol = {
        "status": "frozen prospective; NOT EXECUTED",
        "comparison": {
            "C": "original seed504 training data",
            "D": "new 3,072-game training-source data",
        },
        "updates_per_arm": 108,
        "batch_size_sequence": lengths,
        "sampling": "deterministic permutation without replacement; no replacement sampling",
        "expanded_schedule_seed": 542,
        "schedule_runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "initializer_sha256": "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c",
        "architecture": "residual_v3/kalah_v3 (96, 3), 27 inputs",
        "objective": "legal policy CE + 0.5 * Huber(value, delta=1); joint policy/value",
        "optimizer": "Adam lr=0.0001 betas=(0.9,0.999) epsilon=1e-8 weight_decay=0; gradient clip norm=1; no scheduler",
        "control_permutations": "original frozen seed504 permutations",
        "expanded_data_description": "fixed-update experiment, not four complete passes over the expanded dataset",
        "candidate_endpoint": "final update only",
        "development_metrics": "descriptive only",
        "later_strength_comparison": "candidate policy plus frozen initializer value versus current P",
    }
    return {
        "protocol": protocol,
        "control": _schedule_summary(control_schedule),
        "expanded": _schedule_summary(expanded_schedule),
        "updates": [
            {
                "update": index,
                "batch_size": lengths[index],
                "C": control_schedule[index]["rows"],
                "D": expanded_schedule[index]["rows"],
            }
            for index in range(108)
        ],
    }


def _rebuild_artifacts_independently(
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    _, training_rows, training_report = _read_partition_rows(
        manifest["sources"], "training"
    )
    _, development_rows, development_report = _read_partition_rows(
        manifest["sources"], "development"
    )
    if (
        sum(int(s["games"]) for s in manifest["sources"] if s["role"] == "training")
        != 3072
    ):
        raise ValueError("amended_training_game_total_mismatch")
    if (
        sum(int(s["games"]) for s in manifest["sources"] if s["role"] == "development")
        != 1024
    ):
        raise ValueError("amended_development_game_total_mismatch")
    membership = _membership(development_rows, training_rows)
    protocol = _schedules(training_rows)
    original = _read_json(CORPUS / "accounting.json")
    source_map = {str(s["source_seed"]): s for s in manifest["sources"]}
    recovered = source_map["542001"]
    full_original_shards = sum(
        source["games"] == 64 and source["source_seed"] < 542064
        for source in manifest["sources"]
    )
    original_plan = {
        "requested_games": 4096,
        "completed_games": full_original_shards * 64 + recovered["games"],
        "capped_games": 0,
        "incomplete_games": 1,
        "unstarted_games": 12,
        "completed_original_shards": full_original_shards,
        "interrupted_original_shards": 1,
        "interrupted_seed542001_completed_games": 51,
        "interrupted_seed542001_incomplete_games": 1,
        "interrupted_seed542001_unstarted_games": 12,
        "interrupted_shard_counted_as_completed": False,
    }
    report = {
        "requested_games": 4109,
        "planned_slots": 4109,
        "original_interrupted_attempt": {
            "requested_games": original["requested_games"],
            "completed_games": original["game_status"]["completed"],
            "capped_games": original["game_status"]["capped"],
            "incomplete_or_not_started_games": original["game_status"][
                "incomplete_or_not_started"
            ],
            "completed_shards": original["shard_status"]["complete"],
            "interrupted_shards": original["shard_status"]["partial"],
            "unattempted_shards": original["shard_status"]["not_started"],
        },
        "original_plan": original_plan,
        "supplementary_plan": {
            "seed": 542064,
            "requested_games": 13,
            "completed_games": 13,
            "source": "training",
        },
        "usable_corpus_games": 4096,
        "game_status": {
            "complete": training_report["games"] + development_report["games"],
            "capped": 0,
            "incomplete_or_not_started": 13,
        },
        "source_manifest": manifest,
        "training_corpus": training_report,
        "development_source": development_report,
        "development_membership": membership,
        "training_protocol": protocol,
    }
    return {
        "accounting.json": report,
        "development-membership.json": membership,
        "training-protocol.json": protocol,
    }


def _verify_amendment_and_inputs(
    *, tablebase_path: Path | None = None
) -> dict[str, Any]:
    amendment_path = CORPUS / "continuation-amendment.json"
    amendment_bytes = amendment_path.read_bytes()
    if sha256(amendment_path) != runner.AMENDMENT_SHA256:
        raise ValueError("amended_continuation_amendment_sha256_mismatch")
    amendment = json.loads(amendment_bytes)
    if amendment.get("status") != "frozen continuation amendment":
        raise ValueError("amended_continuation_amendment_status_mismatch")
    verify_recorded_environment(amendment)
    for relative, expected in amendment["source_sha256"].items():
        if runner.bound_source_digest(relative) != expected:
            raise ValueError(f"amended_source_sha256_mismatch:{relative}")
    verify_runtime_assets(ROOT, amendment, tablebase_path=tablebase_path)
    preserved_files = {
        "docs/data/seed542-selfplay-4096/manifest.md": "original_manifest_sha256",
        "docs/data/seed542-selfplay-4096/continuation-freeze.json": "continuation_freeze_sha256",
        "docs/data/seed542-selfplay-4096/execution-interruption.json": "interruption_record_sha256",
        "docs/data/seed542-selfplay-4096/execution-inventory.jsonl": "execution_inventory_sha256",
        "docs/data/seed542-selfplay-4096/accounting.json": "original_accounting_sha256",
        "docs/data/seed542-selfplay-4096/partial-status.json": "original_partial_status_sha256",
        "docs/data/seed542-selfplay-4096/shard-542000.jsonl": "seed542000_sha256",
        "docs/data/seed542-selfplay-4096/shard-542000.log": "seed542000_log_sha256",
        "docs/data/seed542-selfplay-4096/shard-542001.log": "seed542001_log_sha256",
        "docs/data/seed542-selfplay-4096/azlite-self-play-shards-etoy6j_i/worker_0.jsonl": "staging_sha256",
    }
    for relative, key in preserved_files.items():
        if sha256(ROOT / relative) != amendment["preserved_evidence"][key]:
            raise ValueError(f"amended_preserved_evidence_changed:{relative}")
    interruption = _read_json(CORPUS / "execution-interruption.json")
    recovered = amendment["recovered_derivative"]
    extraction = seed542_amendment.verify_extraction(
        ROOT / interruption["staging_output"]["path"],
        ROOT / recovered["path"],
        ROOT / recovered["provenance_path"],
        source_path_label=recovered["original_source_path"],
    )
    if (
        extraction["complete_games"] != 51
        or extraction["rows"] != 1768
        or extraction["derivative_sha256"] != recovered["sha256"]
        or extraction["provenance_sha256"] != recovered["provenance_sha256"]
    ):
        raise ValueError("amended_recovered_derivative_reconciliation_failed")
    return amendment


def _verify_execution_events(amendment: dict[str, Any]) -> None:
    partial_verifier.verify_inventory(require_complete=True)
    events = partial_verifier._read_jsonl(CORPUS / "continuation-inventory.jsonl")
    partial_verifier.verify_amended_commands(events, amendment, require_complete=True)
    for seed in range(542002, 542065):
        seed_events = [event for event in events if int(event.get("seed", -1)) == seed]
        starts = [event for event in seed_events if event.get("event") == "started"]
        finishes = [event for event in seed_events if event.get("event") == "finished"]
        if len(starts) != 1 or len(finishes) != 1:
            raise ValueError(f"amended_execution_event_multiplicity:{seed}")
        if events.index(starts[0]) >= events.index(finishes[0]):
            raise ValueError(f"amended_execution_event_order:{seed}")
        finished = finishes[0]
        log_path = CORPUS / f"shard-{seed}.log"
        if not log_path.is_file() or sha256(log_path) != finished.get("log_sha256"):
            raise ValueError(f"amended_execution_log_mismatch:{seed}")
    for seed, state in ((542000, "completed"), (542001, "interrupted")):
        skips = [
            event
            for event in events
            if event.get("event") == "amendment_skip_preserved"
            and int(event.get("seed", -1)) == seed
        ]
        if len(skips) != 1 or skips[0].get("state") != state:
            raise ValueError(f"amended_preserved_seed_skip_mismatch:{seed}")
    if any(
        event.get("event") in {"started", "finished", "interrupted"}
        and int(event.get("seed", -1)) == 542001
        for event in events
    ):
        raise ValueError("amended_runner_retried_interrupted_seed542001")
    service = _read_json(CORPUS / "execution-service.json")
    finished = [event for event in events if event.get("event") == "finished"]
    if (
        service.get("unit") != "seed542-amended-generation.service"
        or service.get("result") != "success"
        or service.get("exit_status") != 0
        or service.get("finished_invocations") != 63
        or service.get("completed_games") != 3981
        or service.get("capped_games") != 0
        or service.get("incomplete_or_unstarted_games") != 0
        or len(finished) != 63
        or service.get("resource_scope", {}).get("cpu_quota")
        != amendment["resource_scope"]["cpu_quota"]
        or service.get("resource_scope", {}).get("cpu_weight")
        != amendment["resource_scope"]["cpu_weight"]
        or service.get("resource_scope", {}).get("memory_max_bytes")
        != amendment["resource_scope"]["memory_max_bytes"]
        or service.get("resource_scope", {}).get("nice")
        != amendment["resource_scope"]["nice"]
        or service.get("resource_scope", {}).get("allowed_cpus")
        != amendment["resource_scope"]["cpu_affinity"]
    ):
        raise ValueError("amended_execution_service_receipt_mismatch")


def compare_published_artifacts(
    corpus: Path, rebuilt: dict[str, dict[str, Any]]
) -> dict[str, str]:
    verified = {}
    for source_name, published_name in PUBLISHED.items():
        path = corpus / published_name
        archived = _read_json(path)
        if canonical(archived) != canonical(rebuilt[source_name]):
            raise ValueError(f"amended_published_artifact_mismatch:{published_name}")
        verified[published_name] = sha256(path)
    return verified


def verify(
    *, root: Path | None = None, tablebase_path: Path | None = None
) -> dict[str, str]:
    """Rebuild games, loader rows, accounting, membership, and schedules read-only."""
    global ROOT, CORPUS
    previous = (
        ROOT,
        CORPUS,
        runner.ROOT,
        partial_verifier.ROOT,
        partial_verifier.CORPUS,
    )
    if root is not None:
        ROOT = root.resolve()
        CORPUS = ROOT / "docs/data/seed542-selfplay-4096"
        runner.ROOT = ROOT
        partial_verifier.ROOT = ROOT
        partial_verifier.CORPUS = CORPUS
    if tablebase_path is not None and not tablebase_path.is_absolute():
        tablebase_path = ROOT / tablebase_path
    try:
        amendment = _verify_amendment_and_inputs(tablebase_path=tablebase_path)
        _verify_execution_events(amendment)
        manifest_path = CORPUS / "source-manifest.json"
        archived_manifest = _read_json(manifest_path)
        rebuilt_manifest = seed542_amendment.build_source_manifest(ROOT, CORPUS)
        if canonical(archived_manifest) != canonical(rebuilt_manifest):
            raise ValueError("amended_source_manifest_mismatch")
        rebuilt = _rebuild_artifacts_independently(archived_manifest)
        return {
            "continuation-amendment.json": sha256(
                CORPUS / "continuation-amendment.json"
            ),
            "execution-service.json": sha256(CORPUS / "execution-service.json"),
            "source-manifest.json": sha256(manifest_path),
            **compare_published_artifacts(CORPUS, rebuilt),
        }
    finally:
        ROOT, CORPUS, runner.ROOT, partial_verifier.ROOT, partial_verifier.CORPUS = (
            previous
        )


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tablebase", type=Path)
    parser.add_argument(
        "--asset-check-only",
        action="store_true",
        help="authenticate checkout-relative runtime assets without rebuilding corpus reports",
    )
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    if arguments.asset_check_only:
        amendment_path = (
            root / "docs/data/seed542-selfplay-4096/continuation-amendment.json"
        )
        amendment_bytes = amendment_path.read_bytes()
        if hashlib.sha256(amendment_bytes).hexdigest() != runner.AMENDMENT_SHA256:
            raise ValueError("amended_continuation_amendment_sha256_mismatch")
        amendment = json.loads(amendment_bytes)
        verify_recorded_environment(amendment)
        tablebase_path = arguments.tablebase
        if tablebase_path is not None and not tablebase_path.is_absolute():
            tablebase_path = root / tablebase_path
        verify_runtime_assets(root, amendment, tablebase_path=tablebase_path)
        result = {"status": "verified_runtime_assets"}
    else:
        result = {
            "status": "verified_complete",
            **verify(root=root, tablebase_path=arguments.tablebase),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
