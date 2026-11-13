"""Run a frozen, paired #580 subset through the original and v2 pair workers."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ml.alphazero_lite import run_varied_opponent_pilot_v2 as v2
from ml.alphazero_lite import varied_opponent_pilot as pilot

ROOT = Path(__file__).resolve().parents[2]
RELATIVE = "ml/alphazero_lite/validate_varied_opponent_probe_reuse.py"
BASE_FREEZE = "docs/data/varied-opponent-coverage-pilot/freeze.json"
OUTPUT_ROOT = "docs/data/seed583-probe-reuse-generation-validation"
OPENING_SELECTION = {
    "initializer": ["opening-0000", "opening-0003", "opening-0004", "opening-0005"],
    "T512": ["opening-0001", "opening-0002", "opening-0007", "opening-0008"],
}
RESOURCE_LIMITS = {
    "pair_concurrency": 1,
    "workers_per_game": 1,
    "cpu_quota": pilot.CPU_QUOTA,
    "cpu_weight": pilot.CPU_WEIGHT,
    "memory_max_bytes": pilot.MEMORY_MAX_BYTES,
    "allowed_cpus": pilot.ALLOWED_CPUS,
    "nice": pilot.NICE,
    "torch_intraop_threads": 1,
    "torch_interop_threads": 1,
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_body(freeze: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in freeze.items() if key != "freeze_sha256"}


def build_freeze(source_root: Path = ROOT) -> dict[str, Any]:
    """Select eight treatment openings and their exact matched control pairs."""
    original = json.loads((source_root / BASE_FREEZE).read_text())
    assignments = original["assignments"]
    treatment_by_opening: dict[str, list[dict[str, Any]]] = {}
    control_by_opening: dict[str, list[dict[str, Any]]] = {}
    for item in assignments:
        destination = control_by_opening if item["arm"] == "control" else treatment_by_opening
        destination.setdefault(item["opening_id"], []).append(item)
    selected: list[dict[str, Any]] = []
    for opponent, opening_ids in OPENING_SELECTION.items():
        for opening_id in opening_ids:
            treatments = treatment_by_opening[opening_id]
            if any(item["opponent"] != opponent for item in treatments):
                raise ValueError(f"selected_opening_opponent_mismatch:{opening_id}")
            selected.extend(control_by_opening[opening_id])
            selected.extend(treatments)
    selected.sort(key=lambda item: item["pair_id"])
    jobs = pair_jobs(selected)
    counts = {
        "control": sum(job["game_specs"][0]["arm"] == "control" for job in jobs),
        "initializer": sum(
            job["game_specs"][0]["opponent"] == "initializer" for job in jobs
        ),
        "T512": sum(job["game_specs"][0]["opponent"] == "T512" for job in jobs),
    }
    if counts != {"control": 8, "initializer": 4, "T512": 4}:
        raise ValueError("subset_arm_counts_mismatch")

    artifacts = source_root / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts"
    effective = pilot.effective_runtime_policy(artifacts)
    assets = {
        name: {"path": value["path"], "sha256": value["sha256"]}
        for name, value in effective["runtime_assets"].items()
    }
    freeze: dict[str, Any] = {
        "schema": "probe_reuse_generation_validation_freeze_v1",
        "parent_freeze": {"path": BASE_FREEZE, "sha256": original["freeze_sha256"]},
        "subset_rule": "four initializer and four T512 treatment openings, each with its original matched control pair",
        "selected_openings": OPENING_SELECTION,
        "pairs": jobs,
        "pair_count": 16,
        "game_count_across_both_runners": 64,
        "runner_order": "alternate_first_runner_by_sorted_pair_id_starting_v1",
        "runner_sources": {
            "v1": "ml/alphazero_lite/run_varied_opponent_pilot.py",
            "v2": "ml/alphazero_lite/run_varied_opponent_pilot_v2.py",
            "pilot": "ml/alphazero_lite/varied_opponent_pilot.py",
            "harness": RELATIVE,
        },
        "runner_sha256": {
            relative: digest(source_root / relative)
            for relative in [
                "ml/alphazero_lite/run_varied_opponent_pilot.py",
                "ml/alphazero_lite/run_varied_opponent_pilot_v2.py",
                "ml/alphazero_lite/varied_opponent_pilot.py",
                RELATIVE,
            ]
        },
        "runtime_assets": assets,
        "runtime_limits": RESOURCE_LIMITS,
        "cpu_accounting": "pair_process_cpu_time_ns plus reaped native_child_cpu_time_ns; wall time reported separately",
        "adoption_gate": {
            "zero_output_mismatches": True,
            "all_attempts_reconciled": True,
            "minimum_total_pair_cpu_reduction": 0.20,
        },
    }
    freeze["freeze_sha256"] = hashlib.sha256(canonical(freeze)).hexdigest()
    return freeze


def pair_jobs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        grouped.setdefault(str(spec["pair_id"]), []).append(spec)
    jobs = []
    for pair_id, members in sorted(grouped.items()):
        members.sort(key=lambda spec: int(spec["designated_p_seat"]))
        pilot.validate_game_specs(members)
        if len(members) != 2:
            raise ValueError(f"pair_size_invalid:{pair_id}")
        jobs.append({"game_specs": members})
    return jobs


def verify_freeze(freeze: dict[str, Any], source_root: Path = ROOT) -> None:
    if hashlib.sha256(canonical(freeze_body(freeze))).hexdigest() != freeze.get("freeze_sha256"):
        raise ValueError("validation_freeze_digest_mismatch")
    if len(freeze.get("pairs", [])) != 16 or sum(
        len(job["game_specs"]) for job in freeze["pairs"]
    ) * 2 != 64:
        raise ValueError("validation_freeze_size_mismatch")
    for relative, expected in freeze["runner_sha256"].items():
        if digest(source_root / relative) != expected:
            raise ValueError(f"validation_source_hash_mismatch:{relative}")
    if freeze["runtime_limits"] != RESOURCE_LIMITS:
        raise ValueError("validation_resource_binding_mismatch")


def preflight(freeze: dict[str, Any], artifacts: Path, source_root: Path = ROOT) -> dict[str, Any]:
    verify_freeze(freeze, source_root)
    jobs = freeze["pairs"]
    specs = [spec for job in jobs for spec in job["game_specs"]]
    pilot.validate_game_specs(specs)
    if len(specs) * 2 != 64:
        raise ValueError("preflight_game_count_mismatch")
    pilot.verify_components(artifacts)
    effective = pilot.effective_runtime_policy(artifacts)
    observed = {name: value["sha256"] for name, value in effective["runtime_assets"].items()}
    expected = {name: value["sha256"] for name, value in freeze["runtime_assets"].items()}
    if observed != expected:
        raise ValueError("preflight_runtime_asset_mismatch")
    check = (
        "from ml.alphazero_lite.varied_opponent_pilot import verify_live_resource_limits; "
        f"verify_live_resource_limits({{key: value for key, value in {RESOURCE_LIMITS!r}.items() "
        "if key != 'pair_concurrency'})"
    )
    scope_argv = pilot.systemd_scope_command(
        ["taskset", "-c", pilot.ALLOWED_CPUS, sys.executable, "-c", check]
    )
    subprocess.run(scope_argv, check=True, capture_output=True, text=True)
    return {
        "status": "preflight_passed_no_games_launched",
        "pairs": len(jobs),
        "games_across_both_runners": 64,
        "arm_counts": {"control": 8, "initializer": 4, "T512": 4},
        "resource_limits": RESOURCE_LIMITS,
        "asset_hashes": observed,
        "systemd_scope_probe": "passed",
        "games_launched": 0,
    }


def _run_one(variant: str, job: dict[str, Any], artifacts: Path, root: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    pair_id = str(job["game_specs"][0]["pair_id"])
    variant_root = root / variant
    checkpoint_dir = variant_root / "checkpoints" / pair_id
    attempt_dir = variant_root / "attempts" / pair_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    committed, pending = pilot.load_pair_checkpoints(checkpoint_dir, [job])
    committed_by_id = {str(item["pair_id"]): item for item in committed}
    numbers = pilot.reconcile_pair_attempts(attempt_dir, [job], committed_by_id, checkpoint_dir)
    if not pending:
        return committed_by_id[pair_id]
    number = numbers[pair_id]
    attempt = pilot.pair_attempt_path(attempt_dir, pair_id, number)
    pilot.write_json_atomic(attempt, {"pair_id": pair_id, "status": "dispatched", "assignment": job})
    wall_start = time.monotonic_ns()
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as pool:
        if variant == "v1":
            future = pool.submit(pilot._play_pair_attempt, job, str(artifacts), attempt)
        elif variant == "v2":
            identity = v2.execution_identity({
                "freeze_sha256": freeze["freeze_sha256"],
                "runner_source": v2.RUNNER_RELATIVE_PATH,
                "runner_sha256": freeze["runner_sha256"][v2.RUNNER_RELATIVE_PATH],
                "probe_lifetime": v2.PAIR_LIFETIME,
            })
            future = pool.submit(v2._run_attempt, job, str(artifacts), attempt, identity)
        else:
            raise ValueError(f"unknown_runner:{variant}")
        result = future.result()
    wall_ns = time.monotonic_ns() - wall_start
    result["validation_wall_time_ns"] = wall_ns
    if variant == "v1":
        checkpoint = pilot.commit_verified_pair(checkpoint_dir, job, result, attempt_number=number)
        receipt = {"pair_id": pair_id, "status": "completed", "assignment": job,
                   "checkpoint_sha256": digest(checkpoint), "pair_cpu_time_ns": result["pair_cpu_time_ns"]}
    else:
        identity = result["execution_identity"]
        checkpoint = v2.commit_v2_pair(checkpoint_dir, job, result, identity=identity, attempt_number=number)
        receipt = {"pair_id": pair_id, "status": "completed", "assignment": job,
                   "execution_identity": identity, "checkpoint_sha256": digest(checkpoint),
                   "pair_cpu_time_ns": result["pair_cpu_time_ns"]}
    pilot.write_json_atomic(attempt, receipt)
    return result


def execute(freeze_path: Path, artifacts: Path, output_root: Path, expected_digest: str) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("freeze_sha256") != expected_digest:
        raise ValueError("external_validation_freeze_digest_mismatch")
    verify_freeze(freeze)
    pilot.verify_live_resource_limits(
        {key: value for key, value in RESOURCE_LIMITS.items() if key != "pair_concurrency"}
    )
    pilot.verify_components(artifacts)
    results: dict[str, dict[str, Any]] = {"v1": {}, "v2": {}}
    order = []
    for job, variant in execution_plan(freeze["pairs"]):
        pair_id = str(job["game_specs"][0]["pair_id"])
        result = _run_one(variant, job, artifacts, output_root, freeze)
        results[variant][pair_id] = result
        order.append({"pair_id": pair_id, "variant": variant})
    report = {"freeze_sha256": expected_digest, "execution_order": order, "results": results}
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "execution.json"
    pilot.write_json_atomic(report_path, report)
    return report


def execution_plan(jobs: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    """Return back-to-back runner execution, alternating first runner by pair."""
    plan = []
    for index, job in enumerate(jobs):
        pair_id = str(job["game_specs"][0]["pair_id"])
        first, second = ("v1", "v2") if index % 2 == 0 else ("v2", "v1")
        plan.extend(((job, first), (job, second)))
        if str(job["game_specs"][0]["pair_id"]) != pair_id:
            raise ValueError("execution_plan_pair_id_changed")
    return plan


def semantic_game(game: dict[str, Any]) -> dict[str, Any]:
    """Retain semantic settings and training rows, omit timing/runner metadata."""
    ignored = {"cpu_time_ns", "runner_metadata", "timing_metadata"}
    return {key: value for key, value in game.items() if key not in ignored}


def verify_output_pair(v1_result: dict[str, Any], v2_result: dict[str, Any]) -> list[dict[str, Any]]:
    differences = []
    for index, (left, right) in enumerate(zip(v1_result["games"], v2_result["games"], strict=True)):
        if semantic_game(left) != semantic_game(right):
            differences.append({"game_index": index, "v1_game_id": left["spec"]["game_id"], "mismatch": "trajectory_outcome_or_training_rows"})
    return differences


def verify_evidence(freeze: dict[str, Any], output_root: Path) -> dict[str, Any]:
    differences = []
    accounting = {name: {"pair_process_cpu_time_ns": 0, "native_child_cpu_time_ns": 0,
                         "pair_cpu_time_ns": 0, "wall_time_ns": 0} for name in ("v1", "v2")}
    for job in freeze["pairs"]:
        pair_id = str(job["game_specs"][0]["pair_id"])
        pair_results = {}
        for variant in ("v1", "v2"):
            base = output_root / variant
            checkpoint_dir = base / "checkpoints" / pair_id
            attempt_dir = base / "attempts" / pair_id
            cp = pilot.pair_checkpoint_path(checkpoint_dir, pair_id)
            if not cp.is_file():
                raise ValueError(f"missing_checkpoint:{variant}:{pair_id}")
            payload = json.loads(cp.read_text())
            if payload.get("assignment") != job:
                raise ValueError(f"checkpoint_assignment_mismatch:{variant}:{pair_id}")
            producing_attempt = payload.get("producing_attempt")
            record_path = pilot.pair_attempt_path(attempt_dir, pair_id, producing_attempt)
            if not record_path.is_file():
                raise ValueError(f"attempt_receipt_count_mismatch:{variant}:{pair_id}")
            receipt = json.loads(record_path.read_text())
            if (receipt.get("status") != "completed" or receipt.get("checkpoint_sha256") != digest(cp)
                    or receipt.get("pair_cpu_time_ns") != payload["result"]["pair_cpu_time_ns"]):
                raise ValueError(f"attempt_receipt_unreconciled:{variant}:{pair_id}")
            result = payload["result"]
            pair_results[variant] = result
            summary = accounting[variant]
            summary["pair_process_cpu_time_ns"] += result["pair_process_cpu_time_ns"]
            summary["native_child_cpu_time_ns"] += result["native_child_cpu_time_ns"]
            summary["pair_cpu_time_ns"] += result["pair_cpu_time_ns"]
            summary["wall_time_ns"] += result.get("validation_wall_time_ns", 0)
        differences.extend({"pair_id": pair_id, **mismatch} for mismatch in verify_output_pair(pair_results["v1"], pair_results["v2"]))
    for variant in ("v1", "v2"):
        variant_root = output_root / variant
        for job in freeze["pairs"]:
            pair_id = str(job["game_specs"][0]["pair_id"])
            checkpoint_dir = variant_root / "checkpoints" / pair_id
            attempt_dir = variant_root / "attempts" / pair_id
            committed, pending = pilot.load_pair_checkpoints(checkpoint_dir, [job])
            if pending or len(committed) != 1:
                raise ValueError(f"checkpoint_set_incomplete:{variant}:{pair_id}")
            pilot.reconcile_pair_attempts(
                attempt_dir,
                [job],
                {str(result["pair_id"]): result for result in committed},
                checkpoint_dir,
            )
    v1_cpu, v2_cpu = accounting["v1"]["pair_cpu_time_ns"], accounting["v2"]["pair_cpu_time_ns"]
    reduction = (v1_cpu - v2_cpu) / v1_cpu if v1_cpu else None
    return {"pair_count": 16, "game_count_per_runner": 32, "output_mismatches": differences,
            "attempts_reconciled": True, "accounting": accounting, "v2_total_pair_cpu_reduction": reduction,
            "adoption_gate_passed": not differences and reduction is not None and reduction >= 0.20}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--output", type=Path, required=True)
    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument("--freeze", type=Path, required=True)
    preflight_parser.add_argument("--expected-freeze-sha256", required=True)
    preflight_parser.add_argument("--artifacts", type=Path, default=Path("docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts"))
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--freeze", type=Path, required=True)
    execute_parser.add_argument("--expected-freeze-sha256", required=True)
    execute_parser.add_argument("--artifacts", type=Path, default=Path("docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts"))
    execute_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--freeze", type=Path, required=True)
    verify_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = build_freeze()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(result["freeze_sha256"])
    elif args.command == "preflight":
        freeze = json.loads(args.freeze.read_text())
        if freeze["freeze_sha256"] != args.expected_freeze_sha256:
            raise ValueError("external_validation_freeze_digest_mismatch")
        print(json.dumps(preflight(freeze, args.artifacts), indent=2, sort_keys=True))
    elif args.command == "execute":
        print(json.dumps(execute(args.freeze, args.artifacts, args.output_root, args.expected_freeze_sha256), indent=2, sort_keys=True))
    else:
        freeze = json.loads(args.freeze.read_text())
        report = verify_evidence(freeze, args.output_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        if report["output_mismatches"] or not report["attempts_reconciled"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
