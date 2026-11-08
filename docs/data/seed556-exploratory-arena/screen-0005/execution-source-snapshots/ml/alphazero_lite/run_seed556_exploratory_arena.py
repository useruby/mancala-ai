"""Freeze and execute the exploratory T512-versus-serving-P arena."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ml.alphazero_lite import arena
from ml.alphazero_lite import seed556_exploratory_arena as protocol
from ml.alphazero_lite import run_seed551_controlled_policy_strength as runtime

ROOT = protocol.ROOT
OUT = ROOT / protocol.OUT
MATERIALIZED_TABLEBASE = Path("/home/alex/Mancala/ai/.tmp/seed542-screen-runtime/kalah_v1_21.kvtb")


def multicore_observed_runtime() -> dict:
    observed = runtime.observed_runtime()
    return observed


def assert_multicore_runtime(expected: dict | None = None) -> dict:
    observed = multicore_observed_runtime()
    fixed = {
        "cpu_affinity": list(range(14, 32)),
        "nice": 19,
        "address_space_rlimit": [68_719_476_736, 68_719_476_736],
        "environment": {name: "1" for name in observed["environment"]},
        "torch_threads": 1,
        "torch_interop_threads": 1,
    }
    limits = observed["cgroup_limits"]
    if (any(observed.get(k) != v for k, v in fixed.items())
            or limits.get("cpu.max") != "1800000 100000"
            or limits.get("cpu.weight") != "100"
            or limits.get("memory.max") != "38654705664"
            or limits.get("cpuset.cpus.effective") not in {"14-31", None}
            or (expected is not None and observed != expected)):
        raise ValueError("seed556_multicore_runtime_contract_mismatch")
    return observed


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(pending, path)


def prepare(root: Path = ROOT, output: Path = OUT) -> dict:
    """Authenticate inputs, compose both sides, and freeze every launch command."""
    import torch

    root, output = root.resolve(), output.resolve()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    observed = assert_multicore_runtime()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("seed556_output_must_be_empty")
    raw, openings = protocol.load_openings(root)
    for path, digest in ((protocol.T512_CHECKPOINT, protocol.T512_SHA256),
                         (protocol.P_CHECKPOINT, protocol.P_SHA256),
                         (protocol.INITIALIZER, protocol.INITIALIZER_SHA256)):
        if protocol.sha256(root / path) != digest:
            raise ValueError(f"seed556_checkpoint_identity_mismatch:{path}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "openings.jsonl").write_bytes(raw)
    batch = output / "batches"
    batch.mkdir()
    for index, line in enumerate(raw.splitlines(keepends=True)):
        (batch / f"opening-{index:04d}.jsonl").write_bytes(line)
    artifacts = output / "runtime-artifacts"
    artifacts.mkdir()
    sources = {
        "T512": root / "docs/data/seed555-policy-strength-screen/runtime-artifacts/T512",
        "P-E4": root / "docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4",
        "initializer": root / "docs/data/seed555-policy-strength-screen/runtime-artifacts/initializer",
    }
    hashes = {"T512": protocol.T512_SHA256, "P-E4": protocol.P_SHA256,
              "initializer": protocol.INITIALIZER_SHA256}
    component_files = {}
    for name, source in sources.items():
        if protocol.sha256(source / "model.npz") != hashes[name]:
            raise ValueError(f"seed556_exported_component_identity_mismatch:{name}")
        shutil.copytree(source, artifacts / name)
        sidecar = artifacts / name / "search_policy.json"
        if sidecar.is_file():
            policy = json.loads(sidecar.read_text())
            policy["tablebase"]["path"] = os.path.relpath(MATERIALIZED_TABLEBASE, root)
            sidecar.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
        component_files[name] = {p.name: protocol.sha256(p) for p in sorted((artifacts / name).iterdir()) if p.is_file()}
    challenger_policy = arena.ArtifactEvaluator(artifacts / "T512")
    opponent_policy = arena.ArtifactEvaluator(artifacts / "P-E4")
    value = arena.ArtifactEvaluator(artifacts / "initializer")
    composed_t = arena.ComposedArtifactEvaluator(value, challenger_policy,
        policy_source="candidate", value_source="current")
    composed_p = arena.ComposedArtifactEvaluator(value, opponent_policy,
        policy_source="candidate", value_source="current")
    if composed_t.candidate_evaluator is not challenger_policy or composed_t.current_evaluator is not value or composed_p.candidate_evaluator is not opponent_policy or composed_p.current_evaluator is not value:
        raise ValueError("seed556_composed_evaluator_preflight_failed")
    if protocol.sha256(MATERIALIZED_TABLEBASE) != protocol.TABLEBASE_SHA256 or MATERIALIZED_TABLEBASE.stat().st_size != protocol.TABLEBASE_SIZE:
        raise ValueError("seed556_materialized_tablebase_identity_mismatch")
    search_preflight = arena.resolve_strength_comparison_runtime_contract(
        current_artifact=artifacts / "P-E4",
        challenger_artifact=artifacts / "T512",
        exact_root_threshold=16,
        native_probe=root / protocol.NATIVE_PROBE,
        tablebase=MATERIALIZED_TABLEBASE,
    )
    if not search_preflight or search_preflight.get("exact_root_solve_threshold") != 16 or search_preflight.get("exact_root_objective") != "final_score_margin" or search_preflight.get("exact_leaf_solve_mode") != "disabled":
        raise ValueError("seed556_native_runtime_preflight_failed")
    write_json(output / "runtime-policy-preflight.json", search_preflight)
    overlap = protocol.overlap_report(root, openings)
    write_json(output / "overlap-report.json", {k: v for k, v in overlap.items() if k != "per_opening"})
    (output / "overlap-per-opening.jsonl").write_bytes(b"".join(protocol.stable_json(row) + b"\n" for row in overlap["per_opening"]))
    write_json(output / "overlap-limitations.json", {
        "reused_seed511_v4_suite": True,
        "seed461_continuation_overlap_unresolved": True,
        "exploratory": True,
        "historical_games_imported": False,
    })
    registration = {
        "schema": "seed556_t512_vs_serving_p_v1",
        "status": "prospective_rules_frozen_before_games",
        "suite_sha256": protocol.OPENINGS_SHA256,
        "new_games": protocol.GAMES,
        "component_assignments": protocol.ASSIGNMENTS,
        "components": component_files,
        "runtime": {
            **runtime.runtime_settings(),
            "observed": observed,
            "torch_threads": 1,
            "torch_interop_threads": 1,
            "resource_limits": {
                "cpu_quota": "1800%", "cpu_weight": "100",
                "memory_max": "38654705664", "allowed_cpus": "14-31",
                "nice": 19, "address_space_limit": "68719476736",
                "workers_per_game": 1,
                "parallel_opening_pairs": protocol.PAIR_CONCURRENCY,
            },
        },
        "source_sha256": {p: protocol.sha256(root / p) for p in (
            "ml/alphazero_lite/arena.py",
            "ml/alphazero_lite/seed556_exploratory_arena.py",
            "ml/alphazero_lite/run_seed556_exploratory_arena.py",
            "ml/alphazero_lite/seed556_arena_entrypoint.py",
            "ml/alphazero_lite/verify_seed556_exploratory_arena.py",
            "ml/alphazero_lite/test_seed556_exploratory_arena.py",
            "ml/alphazero_lite/verify_seed552_c108_serving_p.py",
            "ml/alphazero_lite/verify_seed555_attempt_ledger.py",
        )},
        "tablebase_lfs_pointer_sha256": protocol.sha256(root / protocol.TABLEBASE_PATH),
        "tablebase_materialized_sha256": protocol.sha256(MATERIALIZED_TABLEBASE),
        "tablebase_materialized_size": MATERIALIZED_TABLEBASE.stat().st_size,
        "overlap_sha256": protocol.sha256(output / "overlap-report.json"),
        "overlap_rows_sha256": protocol.sha256(output / "overlap-per-opening.jsonl"),
        "execution_root": str(root),
        "execution_output": str(output),
        "prerequisite_publications": {
            "seed555_verification_sha256": protocol.sha256(root / "docs/data/seed555-policy-strength-screen/verification.json"),
            "seed555_corrected_attempt_verifier_sha256": protocol.sha256(root / "ml/alphazero_lite/verify_seed555_attempt_ledger.py"),
            "seed552_verification_sha256": protocol.sha256(root / "docs/data/seed552-c108-serving-p/verification.json"),
        },
    }
    write_json(output / "registration.json", registration)
    freeze = {"registration_sha256": protocol.sha256(output / "registration.json"),
              "component_assignments": protocol.ASSIGNMENTS,
              "source_sha256": {p: protocol.sha256(root / p) for p in (
                  "ml/alphazero_lite/arena.py",
                  "ml/alphazero_lite/seed556_exploratory_arena.py",
        "ml/alphazero_lite/run_seed556_exploratory_arena.py",
                  "ml/alphazero_lite/seed556_arena_entrypoint.py",
                  "ml/alphazero_lite/verify_seed556_exploratory_arena.py",
                  "ml/alphazero_lite/test_seed556_exploratory_arena.py",
                  "ml/alphazero_lite/verify_seed552_c108_serving_p.py",
                  "ml/alphazero_lite/verify_seed555_attempt_ledger.py")}}
    commands = [build_command(root, output, i, freeze) for i in range(protocol.OPENINGS)]
    (output / "command-preflight.jsonl").write_bytes(b"".join(protocol.stable_json(c) + b"\n" for c in commands))
    freeze["command_preflight_sha256"] = protocol.sha256(output / "command-preflight.jsonl")
    write_json(output / "execution-freeze.json", freeze)
    return {"status": "prepared_frozen", "commands": len(commands), "opening_pairs": protocol.OPENINGS}


def validate_pair(root: Path, output: Path, index: int) -> list[dict]:
    """Replay one saved pair against the indexed opening and frozen seat order."""
    from ml.alphazero_lite import verify_seed552_c108_serving_p as predecessor
    from ml.alphazero_lite import verify_seed551_controlled_policy_strength as gamecheck

    expected_openings = gamecheck.replay_suite(root)
    batch = output / "batches"
    command_path = batch / f"command-{index:04d}.json"
    commands = [json.loads(line) for line in (output / "command-preflight.jsonl").read_text().splitlines()]
    command = json.loads(command_path.read_text())
    if command != commands[index] or command.get("component_assignments") != protocol.ASSIGNMENTS:
        raise ValueError(f"seed556_frozen_command_mismatch:{index}")
    if (batch / f"opening-{index:04d}.jsonl").read_bytes() != (root / protocol.OPENINGS_PATH).read_bytes().splitlines(keepends=True)[index]:
        raise ValueError(f"seed556_opening_input_mismatch:{index}")
    games_path = batch / f"games-{index:04d}.jsonl"
    report_path = batch / f"report-{index:04d}.json"
    games = [json.loads(line) for line in games_path.read_text().splitlines()]
    if len(games) != 2:
        raise ValueError(f"seed556_pair_count_mismatch:{index}")
    outcomes = []
    for seat, game in enumerate(games):
        gamecheck._validate_arena_local_opening_index(game)
        if game.get("game_index") != seat or game.get("game_within_opening") != seat or game.get("challenger_player") != seat:
            raise ValueError(f"seed556_seat_pair_mismatch:{index}:{seat}")
        result, margin, capped = predecessor._outcome(expected_openings[index], game)
        raw_result = result.removeprefix("capped_")
        label = {"C": "T512", "P": "P", "draw": "draw"}[raw_result]
        outcomes.append({"opening_index": index, "game_position": seat,
                         "outcome": label, "margin": margin,
                         "capped_nonterminal": capped})
    report = json.loads(report_path.read_text())
    if report.get("schema") != "arena_v1" or report.get("games_played") != 2:
        raise ValueError(f"seed556_report_invalid:{index}")
    if report.get("wins") != sum(row.get("winner") == "challenger" for row in games) or report.get("losses") != sum(row.get("winner") == "current" for row in games) or report.get("draws") != sum(row.get("winner") == "draw" for row in games):
        raise ValueError(f"seed556_report_outcome_mismatch:{index}")
    for row in outcomes:
        row.update({"games_sha256": protocol.sha256(games_path),
                    "report_sha256": protocol.sha256(report_path),
                    "command_sha256": protocol.sha256(command_path)})
    return outcomes


def execute(root: Path = ROOT, output: Path = OUT) -> None:
    """Resume a frozen run; prevalidate every extant attempt before writing."""
    import torch
    root, output = root.resolve(), output.resolve()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    registration = json.loads((output / "registration.json").read_text())
    freeze = json.loads((output / "execution-freeze.json").read_text())
    if freeze.get("registration_sha256") != protocol.sha256(output / "registration.json") or freeze.get("command_preflight_sha256") != protocol.sha256(output / "command-preflight.jsonl"):
        raise ValueError("seed556_freeze_or_command_binding_mismatch")
    observed = assert_multicore_runtime(registration["runtime"]["observed"])
    if observed != registration["runtime"]["observed"]:
        raise ValueError("seed556_live_runtime_mismatch")
    batch = output / "batches"
    commands = [json.loads(line) for line in (output / "command-preflight.jsonl").read_text().splitlines()]
    # Validate all prior evidence before the first mutation or subprocess launch.
    committed: dict[int, list[dict]] = {}
    recovered: set[int] = set()
    for index in range(protocol.OPENINGS):
        completion = batch / f"completion-{index:04d}.json"
        action = recovery_action(batch, index, set())
        if action == "recover" or completion.exists():
            if not completion.exists():
                committed[index] = validate_pair(root, output, index)
                recovered.add(index)
            else:
                rows = validate_pair(root, output, index)
                saved = json.loads(completion.read_text())
                if saved.get("outcomes") != rows:
                    raise ValueError(f"seed556_completion_receipt_mismatch:{index}")
                committed[index] = rows
    outcomes_path = output / "outcomes.jsonl"
    ledger_rows = [committed[i][seat] for i in sorted(committed) for seat in (0, 1)]
    tmp = outcomes_path.with_name(".outcomes.reconcile")
    tmp.write_bytes(b"".join(protocol.stable_json(row) + b"\n" for row in ledger_rows))
    os.replace(tmp, outcomes_path)
    for index in sorted(committed):
        if not (batch / f"completion-{index:04d}.json").exists():
            pair = committed[index]
            write_json(batch / f"completion-{index:04d}.json", {"opening_index": index, "command_sha256": protocol.sha256(batch / f"command-{index:04d}.json"), "outcomes": pair})
        if index in recovered and not (batch / f"recovery-{index:04d}.json").exists():
            write_json(batch / f"recovery-{index:04d}.json", {
                "status": "recovered_valid_saved_pair_without_replay",
                "opening_index": index,
                "command_sha256": protocol.sha256(batch / f"command-{index:04d}.json"),
                "games_sha256": protocol.sha256(batch / f"games-{index:04d}.jsonl"),
                "report_sha256": protocol.sha256(batch / f"report-{index:04d}.json"),
            })

    def play(index: int):
        command = commands[index]
        if recovery_action(batch, index, set()) != "execute":
            raise ValueError(f"seed556_conflicting_resume_evidence:{index}")
        write_json(batch / f"command-{index:04d}.json", command)
        result = invoke(command, root=root, output=output)
        write_json(batch / f"attempt-{index:04d}-00.json", {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError(f"seed556_arena_failed_preserved:{index}")
        return index, validate_pair(root, output, index)

    pending = [i for i in range(protocol.OPENINGS) if i not in committed]
    for start in range(0, len(pending), protocol.PAIR_CONCURRENCY):
        group = pending[start : start + protocol.PAIR_CONCURRENCY]
        with ThreadPoolExecutor(max_workers=protocol.PAIR_CONCURRENCY) as pool:
            futures = {pool.submit(play, i): i for i in group}
            for future in as_completed(futures):
                index, pair = future.result()
                committed[index] = pair
                write_json(batch / f"completion-{index:04d}.json", {
                    "opening_index": index,
                    "command_sha256": protocol.sha256(batch / f"command-{index:04d}.json"),
                    "outcomes": pair,
                })
                with outcomes_path.open("ab") as stream:
                    stream.write(b"".join(protocol.stable_json(row) + b"\n" for row in pair))
                    stream.flush()
                    os.fsync(stream.fileno())
                write_json(output / "execution-progress.json", {
                    "status": "running", "committed_opening_pairs": len(committed),
                    "committed_games": len(committed) * 2,
                })
    if sum((batch / f"completion-{i:04d}.json").is_file() for i in range(protocol.OPENINGS)) == protocol.OPENINGS:
        from ml.alphazero_lite import verify_seed556_exploratory_arena as verifier
        all_rows = [json.loads(line) for line in outcomes_path.read_text().splitlines()]
        analysis, scores = protocol.analyze(all_rows)
        write_json(output / "analysis.json", analysis)
        (output / "per-opening-scores.jsonl").write_bytes(b"".join(protocol.stable_json(row) + b"\n" for row in scores))
        verified = verifier.verify(root, output)
        write_json(output / "verification.json", verified)
        write_json(output / "verification-receipt.json", {
            "status": verified["status"],
            "verified_games": verified["validated_games"],
            "verified_pairs": verified["validated_pairs"],
            "verification_sha256": protocol.sha256(output / "verification.json"),
        })
        write_json(output / "execution-progress.json", {"status": "complete_verified", "committed_opening_pairs": protocol.OPENINGS, "committed_games": protocol.GAMES})


def durable_command(root: Path = ROOT, output: Path = OUT) -> list[str]:
    return ["systemd-run", "--user", "--unit=seed556-exploratory-arena-18way", "--collect",
            f"--property=WorkingDirectory={root}", "--property=CPUQuota=1800%", "--property=CPUWeight=100",
            "--property=MemoryMax=38654705664", "--property=AllowedCPUs=14-31", "--property=Nice=19",
            "--property=LimitAS=68719476736", f"--setenv=PYTHONPATH={root}",
            "--setenv=OMP_NUM_THREADS=1", "--setenv=MKL_NUM_THREADS=1", "--setenv=OPENBLAS_NUM_THREADS=1",
            "--setenv=BLAS_NUM_THREADS=1", "--setenv=NUMEXPR_NUM_THREADS=1", "--", "taskset", "-c", "14-31",
            sys.executable, "-m", "ml.alphazero_lite.run_seed556_exploratory_arena", "--execute",
            "--root", str(root), "--out", str(output)]


def build_command(root: Path, output: Path, index: int, freeze: dict) -> dict:
    batch = output / "batches"
    art = output / "runtime-artifacts"
    options = json.dumps(protocol.SEARCH_OPTIONS, sort_keys=True, separators=(",", ":"))
    argv = [
        sys.executable, "-m", "ml.alphazero_lite.seed556_arena_entrypoint",
        "--challenger", str(art / "T512"), "--current", str(art / "P-E4"),
        "--challenger-policy-artifact", str(art / "T512"),
        "--challenger-value-artifact", str(art / "initializer"),
        "--current-policy-artifact", str(art / "P-E4"),
        "--current-value-artifact", str(art / "initializer"),
        "--games", "2", "--games-per-opening", "2", "--workers", "1",
        "--challenger-simulations", "128", "--current-simulations", "128",
        "--c-puct", "1.25", "--seed-contract", "azlite_eval_seed_v2",
        "--challenger-search-options-json", options,
        "--current-search-options-json", options,
        "--random-opening-plies", "0", "--max-moves", "200",
        "--exact-root-solve-threshold", "16",
        "--exact-root-native-probe", protocol.NATIVE_PROBE,
        "--exact-root-tablebase", str(MATERIALIZED_TABLEBASE),
        "--seed", str(protocol.SEED + index),
        "--opening-prefixes-jsonl", str(batch / f"opening-{index:04d}.jsonl"),
        "--suite-sha256", protocol.sha256(batch / f"opening-{index:04d}.jsonl"),
        "--game-jsonl", str(batch / f"games-{index:04d}.jsonl"),
        "--out", str(batch / f"report-{index:04d}.json"),
    ]
    if freeze.get("component_assignments") != protocol.ASSIGNMENTS:
        raise ValueError("seed556_frozen_component_assignment_mismatch")
    saved = sys.argv
    try:
        sys.argv = ["arena.py", *argv[3:]]
        parsed = arena.parse_args()
    finally:
        sys.argv = saved
    paths = {
        "--challenger-policy-artifact": art / "T512",
        "--challenger-value-artifact": art / "initializer",
        "--current-policy-artifact": art / "P-E4",
        "--current-value-artifact": art / "initializer",
    }
    for flag, expected in paths.items():
        if Path(argv[argv.index(flag) + 1]) != expected:
            raise ValueError(f"seed556_component_selection_mismatch:{flag}")
    if parsed.seed != protocol.SEED + index or parsed.games != 2:
        raise ValueError(f"seed556_arena_arguments_mismatch:{index}")
    return {
        "opening_index": index,
        "argv": argv,
        "component_assignments": protocol.ASSIGNMENTS,
        "registration_sha256": freeze["registration_sha256"],
    }


def invoke(command: dict, *, root: Path, output: Path, subprocess_run=None):
    """Validate matchup semantics before any evidence write or child launch."""
    if command.get("component_assignments") != protocol.ASSIGNMENTS:
        raise ValueError("seed556_command_component_assignment_mismatch")
    argv = command.get("argv", [])
    expected = {
        "--challenger-policy-artifact": output / "runtime-artifacts/T512",
        "--challenger-value-artifact": output / "runtime-artifacts/initializer",
        "--current-policy-artifact": output / "runtime-artifacts/P-E4",
        "--current-value-artifact": output / "runtime-artifacts/initializer",
    }
    for flag, path in expected.items():
        if flag not in argv or Path(argv[argv.index(flag) + 1]) != path:
            raise ValueError(f"seed556_command_component_selection_mismatch:{flag}")
    runner = subprocess_run or subprocess.run
    return runner(argv, cwd=root, check=False, capture_output=True, text=True)


def recovery_action(batch: Path, index: int, committed: set[int]) -> str:
    if index in committed:
        return "skip"
    games = batch / f"games-{index:04d}.jsonl"
    report = batch / f"report-{index:04d}.json"
    command = batch / f"command-{index:04d}.json"
    if any(p.exists() for p in (games, report, command)):
        if games.is_file() and report.is_file() and command.is_file():
            return "recover"
        raise ValueError(f"seed556_incomplete_prior_attempt_preserved:{index}")
    return "execute"


def finalize(output: Path, verifier) -> dict:
    """Finalize reconstructed evidence without requiring publication receipts."""
    ledger = [json.loads(line) for line in (output / "outcomes.jsonl").read_text().splitlines() if line]
    analysis, scores = protocol.analyze(ledger)
    write_json(output / "analysis.json", analysis)
    (output / "per-opening-scores.jsonl").write_text(
        "".join(protocol.stable_json(row).decode() + "\n" for row in scores)
    )
    result = verifier(output)
    write_json(output / "verification.json", result)
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.out if args.out.is_absolute() else root / args.out
    if args.prepare:
        print(json.dumps(prepare(root, output), indent=2, sort_keys=True))
    else:
        execute(root, output)


if __name__ == "__main__":
    main()
