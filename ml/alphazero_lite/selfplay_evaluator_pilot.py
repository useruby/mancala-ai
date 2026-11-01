"""Controlled seed542-settings self-play evaluator pilot.

This module changes only evaluator construction around the production worker.
The game/search/target-generation implementation remains ``self_play``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import arena, self_play

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/selfplay-evaluator-pilot")
BASE = Path("docs/data/seed542-selfplay-4096")
P_ARTIFACT = Path(
    "docs/data/seed504-policy-only-ablation/screen/runtime-artifacts/P-E4"
)
VALUE_ARTIFACT = Path(
    "docs/data/seed504-policy-only-ablation/screen/runtime-artifacts/initializer"
)
P_CHECKPOINT = Path("docs/data/seed504/seed504_epoch4.npz")
INITIALIZER = BASE / "initializer.npz"
PROBE = Path("model-artifact/runtime/kalah_v1_tablebase")
TABLEBASE = Path("model-artifact/runtime/kalah_v1_21.kvtb")
ARTIFACT_FILES = ("model.npz", "weights.json", "metadata.json", "search_policy.json")
SOURCE_FILES = tuple(
    path.relative_to(ROOT).as_posix()
    for path in sorted((ROOT / "ml/alphazero_lite").glob("*.py"))
)
RUNTIME_FILES = (str(PROBE), str(TABLEBASE))
SERVING_FILES = tuple(
    f"model-artifact/current/{name}"
    for name in (
        "weights.json",
        "metadata.json",
        "search_policy.json",
        "arena_report.json",
    )
)
GAMES = 64
SEED = 542500
SIMULATIONS = 128
RESOURCE_SCOPE = {
    "cpu_quota": "25%",
    "cpu_weight": 1,
    "memory_max_bytes": 4294967296,
    "nice": 19,
    "cpu_affinity": "31",
}
RESOURCE_LAUNCHER = [
    "systemd-run",
    "--user",
    "--scope",
    "--quiet",
    "-p",
    "CPUQuota=25%",
    "-p",
    "CPUWeight=1",
    "-p",
    "MemoryMax=4294967296",
    "--",
    "taskset",
    "-c",
    "31",
    "nice",
    "-n",
    "19",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def process_tree_rss_bytes(root_pid: int) -> int:
    """Sample resident memory for a command and its currently visible descendants."""
    parents: dict[int, int] = {}
    rss: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat[stat.rfind(")") + 2 :].split()
            parents[int(entry.name)] = int(fields[1])
            status = (entry / "status").read_text()
            match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.MULTILINE)
            rss[int(entry.name)] = int(match.group(1)) * 1024 if match else 0
        except (OSError, ValueError, IndexError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        before = len(descendants)
        descendants.update(
            pid for pid, parent in parents.items() if parent in descendants
        )
        changed = len(descendants) != before
    return sum(rss.get(pid, 0) for pid in descendants)


def atomic_json(path: Path, value: Any) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def artifact_hashes(path: Path) -> dict[str, str]:
    return {name: sha(path / name) for name in ARTIFACT_FILES}


def authenticate_component(root: Path, component: dict[str, Any]) -> None:
    if sha(root / component["checkpoint"]) != component["checkpoint_sha256"]:
        raise ValueError("pilot_component_checkpoint_changed")
    if artifact_hashes(root / component["artifact"]) != component["files"]:
        raise ValueError("pilot_component_artifact_changed")


def authenticate_files(root: Path, expected: dict[str, str]) -> None:
    for relative, digest in expected.items():
        if sha(root / relative) != digest:
            raise ValueError(f"pilot_release_file_changed:{relative}")


def prepare(root: Path = ROOT) -> dict[str, Any]:
    """Authenticate components and freeze every generation/analysis input."""
    root = root.resolve()
    out = root / OUT
    if (out / "registration.json").exists() or (out / "attempts.jsonl").exists():
        raise FileExistsError("pilot_registration_already_exists")
    out.mkdir(parents=True, exist_ok=True)
    components = {
        "historical_policy": {
            "artifact": str(P_ARTIFACT),
            "checkpoint": str(P_CHECKPOINT),
            "checkpoint_sha256": sha(root / P_CHECKPOINT),
            "files": artifact_hashes(root / P_ARTIFACT),
            "trunk": "complete seed504 E4 artifact; no tensor splicing",
        },
        "initializer": {
            "artifact": str(VALUE_ARTIFACT),
            "checkpoint": str(INITIALIZER),
            "checkpoint_sha256": sha(root / INITIALIZER),
            "files": artifact_hashes(root / VALUE_ARTIFACT),
            "trunk": "complete initializer artifact; no tensor splicing",
        },
    }
    initializer_sha = "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c"
    if sha(root / INITIALIZER) != initializer_sha:
        raise ValueError("seed542_initializer_authentication_failed")
    if (
        sha(root / P_CHECKPOINT)
        != "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad"
    ):
        raise ValueError("seed504_e4_checkpoint_authentication_failed")
    if (
        sha(root / P_ARTIFACT / "search_policy.json")
        != "b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d"
    ):
        raise ValueError("serving_search_policy_authentication_failed")
    sources = {relative: sha(root / relative) for relative in SOURCE_FILES}
    runtime = {relative: sha(root / relative) for relative in RUNTIME_FILES}
    serving_files = {relative: sha(root / relative) for relative in SERVING_FILES}
    import torch

    deps = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    command_base = [
        sys.executable,
        "-m",
        "ml.alphazero_lite.selfplay_evaluator_pilot",
        "execute-arm",
        "--arm",
        "{arm}",
        "--out",
        "{output}",
        "--games",
        str(GAMES),
        "--seed",
        str(SEED),
        "--checkpoint",
        str(INITIALIZER),
        "--runtime-search-policy-artifact",
        "model-artifact/current",
        "--input-encoding",
        "kalah_v3",
        "--player-mode",
        "puct",
        "--simulations",
        str(SIMULATIONS),
        "--c-puct",
        "1.25",
        "--temperature-threshold",
        "10",
        "--temperature",
        "1.0",
        "--temperature-late",
        "0.1",
        "--dirichlet-alpha",
        "0.3",
        "--dirichlet-epsilon",
        "0.25",
        "--schedule-progress-mode",
        "none",
        "--max-moves",
        "200",
        "--workers",
        "1",
        "--policy-target-mode",
        "default",
        "--exact-root-policy-target-mode",
        "selected_one_hot",
        "--policy-target-noise-mode",
        "denoised",
        "--value-target-mode",
        "default",
        "--start-state-mode",
        "standard_4x6",
        "--write-game-metadata",
        "--fpu-mode",
        "zero",
        "--root-policy-mode",
        "visit_count",
        "--tactical-root-bias",
        "0.0",
        "--root-temperature",
        "0.0",
        "--evaluator-cache-size",
        "0",
        "--opening-min-simulations",
        "0",
        "--opening-min-simulations-plies",
        "0",
        "--value-transform-json",
        "",
        "--shadow-q-weight",
        "0.0",
    ]
    registration = {
        "schema": "selfplay-evaluator-pilot-v1",
        "status": "frozen_before_generation",
        "interpretation": "exploratory generation comparison only; no playing-strength claim; seed461 provenance limitation unresolved",
        "arms": {
            "control": {
                "evaluator": "complete seed542 frozen initializer",
                "checkpoint": str(INITIALIZER),
            },
            "treatment": {
                "policy": "historical seed504 E4 through its own trunk",
                "value": "initializer through its own trunk",
                "tensor_splicing": False,
                "policy_artifact": str(P_ARTIFACT),
                "value_artifact": str(VALUE_ARTIFACT),
                "initializer_checkpoint": str(INITIALIZER),
            },
        },
        "components": components,
        "serving_release_files": serving_files,
        "generation": {
            "games_per_arm": GAMES,
            "total_games": 128,
            "seed_schedule": list(range(SEED, SEED + GAMES)),
            "same_seed_schedule_both_arms": True,
            "simulations": SIMULATIONS,
            "c_puct": 1.25,
            "start_state_mode": "standard_4x6",
            "temperature_threshold": 10,
            "temperature": 1.0,
            "temperature_late": 0.1,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
            "schedule_progress_mode": "none",
            "policy_target_mode": "default",
            "exact_root_policy_target_mode": "selected_one_hot",
            "policy_target_noise_mode": "denoised",
            "value_target_mode": "default",
            "exact_root_threshold": 16,
            "exact_leaf_solve": "disabled",
            "workers": 1,
            "max_moves": 200,
            "search_options": {
                "fpu_mode": "zero",
                "reuse_subtree": False,
                "normalize_values": False,
                "root_policy_mode": "visit_count",
                "tactical_root_bias": 0.0,
                "root_temperature": 0.0,
            },
            "remaining_search_options": {
                "value_trust_schedule": None,
                "value_transform": None,
                "tree_reuse_enabled": False,
                "evaluator_cache_size": 0,
                "opening_min_simulations": 0,
                "opening_min_simulations_plies": 0,
                "opponent_pool_config": None,
                "opening_cache": None,
                "shadow_checkpoint": None,
                "shadow_q_weight": 0.0,
            },
            "command_template": command_base,
        },
        "resource_scope": RESOURCE_SCOPE,
        "resource_launcher": RESOURCE_LAUNCHER,
        "runtime": {
            "interpreter": sys.executable,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pip_freeze": deps,
            "project_module": str(Path(self_play.__file__).resolve()),
        },
        "source_sha256": sources,
        "runtime_asset_sha256": runtime,
        "analysis": {
            "stone_buckets": {">32": [33, 48], "17-32": [17, 32], "<=16": [0, 16]},
            "identities": ["canonical state", "production float32 encoded input"],
            "policy_diagnostics": ["target entropy", "duplicate-target disagreement"],
            "telemetry_claim": "authenticates recorded search telemetry and target-generation settings; does not independently prove policy-target optimality",
        },
    }
    atomic_json(out / "registration.json", registration)
    atomic_json(
        out / "execution-freeze.json",
        {
            "status": "frozen_before_generation",
            "registration_sha256": sha(out / "registration.json"),
        },
    )
    return registration


def preflight(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    reg_path = out / "registration.json"
    registration = json.loads(reg_path.read_text())
    freeze = json.loads((out / "execution-freeze.json").read_text())
    if freeze.get("registration_sha256") != sha(reg_path):
        raise ValueError("pilot_freeze_registration_mismatch")
    correction_path = out / "verification-correction.json"
    correction = (
        json.loads(correction_path.read_text()) if correction_path.is_file() else None
    )
    if correction is not None and (
        correction.get("status") != "append_only_verification_correction"
        or correction.get("frozen_registration_sha256") != sha(reg_path)
    ):
        raise ValueError("pilot_verification_correction_binding_mismatch")
    for relative, digest in registration["source_sha256"].items():
        actual = sha(root / relative)
        corrected = (correction or {}).get("source_corrections", {}).get(relative)
        correction_matches = corrected == {
            "frozen_sha256": digest,
            "corrected_sha256": actual,
        }
        if actual != digest and not correction_matches:
            raise ValueError(f"pilot_source_changed:{relative}")
    for relative, digest in registration["runtime_asset_sha256"].items():
        if sha(root / relative) != digest:
            raise ValueError(f"pilot_runtime_asset_changed:{relative}")
    authenticate_files(root, registration["serving_release_files"])
    for component in registration["components"].values():
        authenticate_component(root, component)
    if sys.executable != registration["runtime"]["interpreter"]:
        raise ValueError("pilot_interpreter_changed")
    if sys.version != registration["runtime"]["python"]:
        raise ValueError("pilot_python_runtime_changed")
    import numpy as active_numpy
    import torch

    if (
        active_numpy.__version__ != registration["runtime"]["numpy"]
        or torch.__version__ != registration["runtime"]["torch"]
    ):
        raise ValueError("pilot_numerical_environment_changed")
    if (
        platform.platform() != registration["runtime"]["platform"]
        or Path(self_play.__file__).resolve() != root / "ml/alphazero_lite/self_play.py"
    ):
        raise ValueError("pilot_runtime_identity_changed")
    installed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if installed != registration["runtime"]["pip_freeze"]:
        raise ValueError("pilot_dependency_closure_changed")
    return registration


def authenticate_worker_command(
    registration: dict[str, Any], arm: str, worker_args: list[str], root: Path = ROOT
) -> None:
    if arm not in {"control", "treatment"}:
        raise ValueError("pilot_unknown_arm")
    expected = list(registration["generation"]["command_template"])[6:]
    expected[expected.index("{output}")] = str(OUT / f"{arm}.jsonl")
    if worker_args != expected:
        raise ValueError("pilot_worker_command_mismatch")


def authenticate_completed_output(
    arm: str,
    events: list[dict[str, Any]],
    data_path: Path,
    log_path: Path,
    *,
    verify_corpus,
) -> bool:
    """Return true only for a receipt-bound, independently complete prior arm."""
    if not events:
        return False
    if [event.get("event") for event in events] != ["started", "finished"]:
        raise RuntimeError(f"pilot_refuses_prior_incomplete_attempt:{arm}")
    receipt = events[-1]
    if (
        not data_path.is_file()
        or not log_path.is_file()
        or sha(data_path) != receipt.get("output_sha256")
        or sha(log_path) != receipt.get("log_sha256")
    ):
        raise ValueError(f"pilot_prior_output_receipt_mismatch:{arm}")
    report = verify_corpus(data_path, arm=arm, expected_seed=SEED)
    if report["games"] != {"completed": GAMES, "capped": 0, "incomplete": 0}:
        raise ValueError(f"pilot_prior_output_not_complete:{arm}")
    return True


_ARM = ""


def _evaluator_factory(checkpoint_path: Path, *, input_encoding: str, cache_size: int):
    if _ARM == "control":
        return self_play.CheckpointEvaluator(
            checkpoint_path, input_encoding=input_encoding, cache_size=cache_size
        )
    if _ARM != "treatment":
        raise RuntimeError("pilot_arm_not_configured")
    policy = arena.ArtifactEvaluator(ROOT / P_ARTIFACT)
    value = arena.ArtifactEvaluator(ROOT / VALUE_ARTIFACT)
    return arena.ComposedArtifactEvaluator(
        policy, value, policy_source="current", value_source="candidate"
    )


def execute_arm(argv: list[str] | None = None) -> None:
    global _ARM
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("control", "treatment"), required=True)
    args, worker_args = parser.parse_known_args(argv)
    _ARM = args.arm
    registration = preflight()
    authenticate_worker_command(registration, _ARM, worker_args)
    # The evaluator factory is intentionally experiment-local; fork preserves
    # this process-local adapter in the production worker subprocess.
    import multiprocessing

    multiprocessing.set_start_method("fork", force=True)
    ordinary_factory = self_play.build_checkpoint_evaluator
    self_play.build_checkpoint_evaluator = _evaluator_factory
    try:
        sys.argv = ["self_play", *worker_args]
        self_play.main()
    finally:
        self_play.build_checkpoint_evaluator = ordinary_factory


def execute(root: Path = ROOT) -> dict[str, Any]:
    """Run each frozen arm once, retaining every attempted output and log."""
    root = root.resolve()
    registration = preflight(root)
    out = root / OUT
    ledger = out / "attempts.jsonl"
    records = []
    skipped = []
    for arm in ("control", "treatment"):
        prior = (
            [
                json.loads(line)
                for line in ledger.read_text().splitlines()
                if line.strip()
            ]
            if ledger.exists()
            else []
        )
        arm_events = [event for event in prior if event.get("arm") == arm]
        data_path = out / f"{arm}.jsonl"
        log_path = out / f"{arm}.log"
        if arm_events:
            from ml.alphazero_lite.verify_selfplay_evaluator_pilot import verify_shard

            if authenticate_completed_output(
                arm, arm_events, data_path, log_path, verify_corpus=verify_shard
            ):
                skipped.append(arm)
                continue
        if data_path.exists() or log_path.exists():
            raise FileExistsError(f"pilot_refuses_existing_attempt:{arm}")
        command = [
            *registration["resource_launcher"],
            *registration["generation"]["command_template"],
        ]
        command[command.index("{arm}")] = arm
        command[command.index("{output}")] = str(OUT / f"{arm}.jsonl")
        event = {
            "event": "started",
            "arm": arm,
            "argv": command,
            "registration_sha256": sha(out / "registration.json"),
            "output": str(data_path.relative_to(root)),
            "log": str(log_path.relative_to(root)),
            "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        started = time.monotonic()
        peak_rss = 0
        process: subprocess.Popen | None = None
        with log_path.open("xb") as log:
            try:
                process = subprocess.Popen(
                    command, cwd=root, stdout=log, stderr=subprocess.STDOUT
                )
                while process.poll() is None:
                    peak_rss = max(peak_rss, process_tree_rss_bytes(process.pid))
                    time.sleep(0.1)
                return_code = process.returncode
            except BaseException as error:
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait()
                interrupted = {
                    **event,
                    "event": "interrupted",
                    "ended_at": datetime.datetime.now(datetime.UTC).isoformat(),
                    "error": repr(error),
                    "output_sha256": sha(data_path) if data_path.is_file() else None,
                    "log_sha256": sha(log_path),
                    "output_bytes": data_path.stat().st_size
                    if data_path.is_file()
                    else 0,
                }
                with ledger.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(interrupted, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                raise
        finished = {
            **event,
            "event": "finished",
            "ended_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "exit_status": return_code,
            "output_sha256": sha(data_path) if data_path.is_file() else None,
            "log_sha256": sha(log_path),
            "output_bytes": data_path.stat().st_size if data_path.is_file() else 0,
            "peak_process_tree_rss_bytes": peak_rss,
            "memory_limit_bytes": RESOURCE_SCOPE["memory_max_bytes"],
        }
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(finished, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        records.append(finished)
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    return {"attempts": records, "completed_outputs_skipped": skipped}


def verify_attempts(root: Path = ROOT) -> dict[str, Any]:
    """Authenticate exact commands, attempt accounting, and immutable evidence."""
    root = root.resolve()
    registration = preflight(root)
    out = root / OUT
    ledger_path = out / "attempts.jsonl"
    rows = (
        [
            json.loads(line)
            for line in ledger_path.read_text().splitlines()
            if line.strip()
        ]
        if ledger_path.is_file()
        else []
    )
    attempted: dict[str, list[dict[str, Any]]] = {"control": [], "treatment": []}
    for row in rows:
        arm = row.get("arm")
        if arm not in attempted:
            raise ValueError("attempt_unknown_arm")
        attempted[arm].append(row)
        command = [
            *registration["resource_launcher"],
            *registration["generation"]["command_template"],
        ]
        command[command.index("{arm}")] = arm
        command[command.index("{output}")] = str(OUT / f"{arm}.jsonl")
        if row.get("argv") != command or row.get("registration_sha256") != sha(
            out / "registration.json"
        ):
            raise ValueError(f"attempt_command_or_registration_mismatch:{arm}")
    result = {}
    for arm, events in attempted.items():
        if len(events) not in (0, 1, 2) or (
            len(events) == 2
            and [e["event"] for e in events]
            not in (["started", "finished"], ["started", "interrupted"])
        ):
            raise ValueError(f"attempt_event_accounting_mismatch:{arm}")
        evidence = out / f"{arm}.jsonl"
        log = out / f"{arm}.log"
        terminal = events[-1]["event"] if events else "unattempted"
        if terminal == "unattempted" and (evidence.exists() or log.exists()):
            raise ValueError(f"orphan_attempt_evidence:{arm}")
        if terminal == "finished":
            receipt = events[-1]
            if (
                not evidence.is_file()
                or not log.is_file()
                or sha(evidence) != receipt.get("output_sha256")
                or sha(log) != receipt.get("log_sha256")
            ):
                raise ValueError(f"attempt_evidence_hash_mismatch:{arm}")
        if terminal == "interrupted":
            receipt = events[-1]
            if evidence.is_file() and sha(evidence) != receipt.get("output_sha256"):
                raise ValueError(f"interrupted_evidence_hash_mismatch:{arm}")
            if log.is_file() and sha(log) != receipt.get("log_sha256"):
                raise ValueError(f"interrupted_log_hash_mismatch:{arm}")
        result[arm] = {
            "state": terminal,
            "output_exists": evidence.is_file(),
            "log_exists": log.is_file(),
        }
    return {"attempts": result, "registration_sha256": sha(out / "registration.json")}


def analyze(root: Path = ROOT) -> dict[str, Any]:
    """Produce bounded descriptive generation findings from verified evidence."""
    from ml.alphazero_lite.verify_selfplay_evaluator_pilot import verify_shard

    root = root.resolve()
    attempts = verify_attempts(root)
    out = root / OUT
    ledger = [
        json.loads(line)
        for line in (out / "attempts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    reports = {}
    for arm in ("control", "treatment"):
        path = out / f"{arm}.jsonl"
        if path.is_file():
            reports[arm] = verify_shard(path, arm=arm, expected_seed=SEED)
    by_arm = {}
    for arm, report in reports.items():
        terminal = [
            event
            for event in ledger
            if event.get("arm") == arm
            and event.get("event") in {"finished", "interrupted"}
        ]
        event = terminal[-1] if terminal else {}
        durations = float(event.get("elapsed_seconds", 0.0))
        games = report["games"]
        exact_rows = sum(
            value["rows"]
            for name, value in report["stone_buckets"].items()
            if name == "≤16"
        )
        total_rows = report["rows"]
        by_arm[arm] = {
            **report,
            "throughput_games_per_second": games["completed"] / durations
            if durations > 0
            else 0.0,
            "elapsed_seconds": durations,
            "peak_process_tree_rss_bytes": int(
                event.get("peak_process_tree_rss_bytes", 0)
            ),
            "memory_limit_bytes": int(
                event.get("memory_limit_bytes", RESOURCE_SCOPE["memory_max_bytes"])
            ),
            "exact_root_rows": exact_rows,
            "puct_rows": total_rows - exact_rows,
        }
    successful = len(by_arm) == 2 and all(
        report["games"] == {"completed": GAMES, "capped": 0, "incomplete": 0}
        and report["loader"] == "accepted"
        for report in by_arm.values()
    )
    result = {
        "status": "pilot_success" if successful else "pilot_incomplete_or_invalid",
        "success_criteria_met": successful,
        "attempt_accounting": attempts["attempts"],
        "arms": by_arm,
        "interpretation": "exploratory generation correctness, cost, and data characteristics only; evaluator changes explain differing trajectories; no improved-playing-strength conclusion",
        "future_use": "success authorizes proposing a larger controlled corpus experiment only; no training, promotion, or strength claim",
        "seed461_provenance": "unresolved; not recovered for this pilot",
    }
    atomic_json(out / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "preflight",
            "execute-arm",
            "execute",
            "verify-attempts",
            "analyze",
        ),
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--arm", choices=("control", "treatment"))
    args, unknown = parser.parse_known_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.root), indent=2, sort_keys=True))
    elif args.action == "preflight":
        print(json.dumps(preflight(args.root), indent=2, sort_keys=True))
    elif args.action == "execute":
        print(json.dumps(execute(args.root), indent=2, sort_keys=True))
    elif args.action == "verify-attempts":
        print(json.dumps(verify_attempts(args.root), indent=2, sort_keys=True))
    elif args.action == "analyze":
        print(json.dumps(analyze(args.root), indent=2, sort_keys=True))
    else:
        execute_arm(["--arm", str(args.arm), *unknown])


if __name__ == "__main__":
    main()
