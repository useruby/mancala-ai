"""Frozen matched self-play corpus orchestration and derivative verification.

The production self-play worker and the pilot's evaluator adapter are reused;
this module adds shard accounting, durable receipts, population derivation, and
read-only end-to-end verification for seed549.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import selfplay_evaluator_pilot as pilot
from ml.alphazero_lite.verify_selfplay_evaluator_pilot import (
    expected_search_profile,
    verify_shard,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed549-matched-selfplay")
BASE_SEED = 549000
SHARDS = 8
GAMES_PER_SHARD = 64
TRAIN_SHARDS = set(range(6))
BATCHES = [512] * 26 + [412]
UPDATES = 108
EXPOSURES = sum(BATCHES) * 4


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def write_json(path: Path, value: Any) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source_paths(root: Path) -> list[str]:
    return [
        *(
            path.relative_to(root).as_posix()
            for path in sorted((root / "ml/alphazero_lite").glob("*.py"))
        ),
        "ml/alphazero_lite/test_seed549_corpus.py",
    ]


def completed_receipt_verified(events: list[dict[str, Any]], final: Path) -> bool:
    """A prior shard is skippable only with complete publication receipts."""
    return (
        [event.get("event") for event in events] == ["started", "finished", "published"]
        and final.is_file()
        and digest(final) == events[-1].get("output_sha256")
    )


def make_command(root: Path, arm: str, shard: int, output: Path) -> list[str]:
    seed = BASE_SEED + shard
    output_arg = (
        output.relative_to(root).as_posix()
        if output.is_absolute()
        else output.as_posix()
    )
    return [
        sys.executable,
        "-m",
        "ml.alphazero_lite.seed549_corpus",
        "worker",
        "--arm",
        arm,
        "--shard",
        str(shard),
        "--out",
        output_arg,
        "--games",
        str(GAMES_PER_SHARD),
        "--seed",
        str(seed),
        "--checkpoint",
        "docs/data/seed542-selfplay-4096/initializer.npz",
        "--runtime-search-policy-artifact",
        "model-artifact/current",
        "--input-encoding",
        "kalah_v3",
        "--player-mode",
        "puct",
        "--simulations",
        "128",
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


def prepare(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("seed549_output_directory_not_empty")
    pilot_reg = json.loads(
        (root / "docs/data/selfplay-evaluator-pilot/registration.json").read_text()
    )
    for component in pilot_reg["components"].values():
        pilot.authenticate_component(root, component)
    pilot.authenticate_files(root, pilot_reg["serving_release_files"])
    runtime = pilot_reg["runtime_asset_sha256"]
    for path, expected in runtime.items():
        if digest(root / path) != expected:
            raise ValueError(f"runtime_asset_identity_mismatch:{path}")
    import torch

    deps = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    source_sha = {p: digest(root / p) for p in source_paths(root)}
    registration: dict[str, Any] = {
        "schema": "seed549-matched-corpus-v1",
        "state": "frozen_before_generation",
        "claim": "generation characteristics and valid training inputs only; no strength claim",
        "arms": {
            "control": "complete frozen seed542 initializer",
            "treatment": "serving-P E4 policy/trunk plus initializer value/trunk",
        },
        "components": pilot_reg["components"],
        "serving_release_files": pilot_reg["serving_release_files"],
        "runtime_asset_sha256": runtime,
        "generation": {
            "shards_per_arm": SHARDS,
            "games_per_shard": GAMES_PER_SHARD,
            "games_per_arm": SHARDS * GAMES_PER_SHARD,
            "base_seeds": list(range(BASE_SEED, BASE_SEED + SHARDS)),
            "game_rng_derivation": "(shard_base_seed * 1_000_003) + game_index (production worker)",
            "train_shards": sorted(TRAIN_SHARDS),
            "development_shards": [6, 7],
            "train_games_per_arm": len(TRAIN_SHARDS) * GAMES_PER_SHARD,
            "development_games_per_arm": 2 * GAMES_PER_SHARD,
            "pilot_settings": pilot_reg["generation"],
            "worker_count": 1,
            "command_template": make_command(
                root, "{arm}", 0, OUT / "{arm}" / "shard-0.jsonl"
            ),
        },
        "training_schedule": {
            "updates": UPDATES,
            "batch_sizes": BATCHES * 4,
            "exposures": EXPOSURES,
            "pass_seed_rule": "base seed 549000 + pass index",
            "without_replacement_within_pass": True,
        },
        "runtime": {
            "interpreter": sys.executable,
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "dependencies": deps,
            "project_modules": {
                name: str((root / name).resolve()) for name in source_sha
            },
        },
        "resource_scope": pilot_reg["resource_scope"],
        "resource_launcher": pilot_reg["resource_launcher"],
        "source_sha256": source_sha,
    }
    write_json(out / "registration.json", registration)
    write_json(
        out / "execution-freeze.json",
        {
            "state": "frozen_before_generation",
            "registration_sha256": digest(out / "registration.json"),
        },
    )
    return registration


def preflight(root: Path = ROOT) -> dict[str, Any]:
    out = root.resolve() / OUT
    registration_path = out / "registration.json"
    registration = json.loads(registration_path.read_text())
    freeze = json.loads((out / "execution-freeze.json").read_text())
    if freeze.get("registration_sha256") != digest(registration_path):
        raise ValueError("registration_freeze_mismatch")
    expected_sources = dict(registration["source_sha256"])
    correction_path = out / "execution-correction.json"
    correction_freeze_path = out / "execution-freeze-correction.json"
    if correction_path.is_file():
        correction = json.loads(correction_path.read_text())
        if correction.get("registration_sha256") != digest(registration_path):
            raise ValueError("execution_correction_registration_mismatch")
        relative = "ml/alphazero_lite/seed549_corpus.py"
        if correction.get("previous_sha256") != expected_sources.get(relative):
            raise ValueError("execution_correction_parent_source_mismatch")
        expected_sources[relative] = correction["corrected_sha256"]
        correction_freeze = json.loads(correction_freeze_path.read_text())
        if correction_freeze.get("registration_sha256") != digest(
            registration_path
        ) or correction_freeze.get("correction_sha256") != digest(correction_path):
            raise ValueError("execution_correction_freeze_mismatch")
    for p, expected in expected_sources.items():
        if digest(root / p) != expected:
            raise ValueError(f"frozen_source_changed:{p}")
    for p, expected in registration["runtime_asset_sha256"].items():
        if digest(root / p) != expected:
            raise ValueError(f"runtime_asset_changed:{p}")
    pilot.authenticate_files(root, registration["serving_release_files"])
    for component in registration["components"].values():
        pilot.authenticate_component(root, component)
    import torch

    if (
        sys.executable != registration["runtime"]["interpreter"]
        or sys.version != registration["runtime"]["python"]
        or np.__version__ != registration["runtime"]["numpy"]
        or torch.__version__ != registration["runtime"]["torch"]
    ):
        raise ValueError("runtime_mismatch")
    return registration


def record_zero_output_retry_correction(root: Path = ROOT) -> dict[str, Any]:
    """Bind a narrow runner fix after a pre-worker CLI failure with zero output."""
    root = root.resolve()
    out = root / OUT
    ledger = read_jsonl(out / "attempts.jsonl")
    attempt = [e for e in ledger if e.get("arm") == "control" and e.get("shard") == 0]
    if [e.get("event") for e in attempt] != ["started", "finished"]:
        raise RuntimeError("retry_correction_requires_one_closed_attempt")
    if attempt[-1].get("exit_status") != 2 or attempt[-1].get("output_bytes") != 0:
        raise RuntimeError("retry_correction_requires_cli_failure_with_zero_output")
    path = out / "execution-correction.json"
    if path.exists():
        raise FileExistsError("execution_correction_already_exists")
    reg_path = out / "registration.json"
    relative = "ml/alphazero_lite/seed549_corpus.py"
    correction = {
        "schema": "seed549-execution-correction-v1",
        "registration_sha256": digest(reg_path),
        "reason": "worker adapter omitted --out and production self_play argparse exited 2 before writing a row",
        "failed_attempt": {
            "arm": "control",
            "shard": 0,
            "seed": BASE_SEED,
            "exit_status": 2,
            "output_bytes": 0,
            "log_sha256": attempt[-1]["log_sha256"],
        },
        "source_path": relative,
        "previous_sha256": json.loads(reg_path.read_text())["source_sha256"][relative],
        "corrected_sha256": digest(root / relative),
        "retry_policy": "explicit retry of the same planned shard/seed only when prior output is zero bytes; partial or interrupted outputs are terminal",
    }
    write_json(path, correction)
    write_json(
        out / "execution-freeze-correction.json",
        {"registration_sha256": digest(reg_path), "correction_sha256": digest(path)},
    )
    return correction


def run_worker(arm: str, worker_args: list[str]) -> None:
    if arm not in {"control", "treatment"}:
        raise ValueError("unknown_arm")
    # Authenticate the user's exact shard command against registration.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args, actual = parser.parse_known_args(worker_args)
    preflight()
    if args.arm != arm or args.shard not in range(SHARDS):
        raise ValueError("worker_identity_mismatch")
    expected = make_command(ROOT, arm, args.shard, args.out)
    # command starts with interpreter/module/worker and metadata; compare worker args.
    expected_worker = expected[4:]
    supplied = [
        "--arm",
        arm,
        "--shard",
        str(args.shard),
        "--out",
        args.out.as_posix(),
        *actual,
    ]
    if supplied != expected_worker:
        raise ValueError("worker_command_mismatch")
    from ml.alphazero_lite import self_play

    pilot._ARM = arm
    import multiprocessing

    multiprocessing.set_start_method("fork", force=True)
    original = self_play.build_checkpoint_evaluator
    self_play.build_checkpoint_evaluator = pilot._evaluator_factory
    try:
        sys.argv = ["self_play", *expected_worker[4:]]
        self_play.main()
    finally:
        self_play.build_checkpoint_evaluator = original


def shard_path(root: Path, arm: str, shard: int) -> Path:
    return root / OUT / arm / f"shard-{shard:03d}.jsonl"


def execute(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    reg = preflight(root)
    out = root / OUT
    ledger = out / "attempts.jsonl"
    completed: list[str] = []
    # fixed sequence is arm-major then ascending shard; never run shards concurrently.
    for arm in ("control", "treatment"):
        (out / arm).mkdir(exist_ok=True)
        for shard in range(SHARDS):
            seed = BASE_SEED + shard
            final = shard_path(root, arm, shard)
            events = [
                e
                for e in (read_jsonl(ledger) if ledger.exists() else [])
                if e.get("arm") == arm and e.get("shard") == shard
            ]
            retry_indices = [
                i
                for i, event in enumerate(events)
                if event["event"] == "retry_authorized"
            ]
            attempt_events = (
                events[retry_indices[-1] + 1 :] if retry_indices else events
            )
            if attempt_events:
                if [e["event"] for e in attempt_events] not in (
                    ["started", "finished", "published"],
                    ["started", "finished"],
                    ["started", "interrupted"],
                ):
                    raise RuntimeError(
                        f"seed549_preserving_ambiguous_attempt:{arm}:{shard}"
                    )
                if (
                    attempt_events[-1].get("event") in {"finished", "published"}
                    and attempt_events[-1].get("exit_status", 0) == 0
                ):
                    if not completed_receipt_verified(attempt_events, final):
                        if attempt_events[-1]["event"] != "finished":
                            raise RuntimeError(
                                f"seed549_preserving_ambiguous_attempt:{arm}:{shard}"
                            )
                        raise ValueError(
                            f"completed_shard_receipt_mismatch:{arm}:{shard}"
                        )
                    report = verify_shard(
                        final,
                        arm=f"{arm}/{shard}",
                        expected_seed=seed,
                        expected_game_count=GAMES_PER_SHARD,
                    )
                    if report["games"] != {
                        "completed": GAMES_PER_SHARD,
                        "capped": 0,
                        "incomplete": 0,
                    }:
                        raise ValueError(f"completed_shard_invalid:{arm}:{shard}")
                    continue
                if attempt_events[-1].get("event") in {"started", "interrupted"}:
                    raise RuntimeError(
                        f"seed549_preserving_ambiguous_attempt:{arm}:{shard}"
                    )
            elif events and events[-1].get("event") == "finished":
                if (
                    events[-1].get("exit_status") == 0
                    or events[-1].get("output_bytes") != 0
                ):
                    raise RuntimeError(
                        f"seed549_refuses_failed_or_partial_retry:{arm}:{shard}"
                    )
                if not (out / "execution-correction.json").is_file():
                    raise RuntimeError("seed549_retry_requires_execution_correction")
                append_jsonl(
                    ledger,
                    {
                        "event": "retry_authorized",
                        "arm": arm,
                        "shard": shard,
                        "prior_attempt_output_bytes": 0,
                        "prior_attempt_exit_status": events[-1].get("exit_status"),
                        "correction_sha256": digest(out / "execution-correction.json"),
                        "disposition": "retry same planned shard/seed; prior attempt produced no game rows",
                    },
                )
                events = read_jsonl(ledger)
                events = [
                    e for e in events if e.get("arm") == arm and e.get("shard") == shard
                ]
            elif events and events[-1].get("event") != "retry_authorized":
                raise RuntimeError(
                    f"seed549_preserving_ambiguous_attempt:{arm}:{shard}"
                )
            if final.exists():
                raise FileExistsError(f"unreceipted_output_preserved:{final}")
            attempt = sum(e.get("event") == "started" for e in events)
            output = out / "attempts" / f"{arm}-{shard:03d}-attempt-{attempt:02d}.jsonl"
            output.parent.mkdir(exist_ok=True)
            log = output.with_suffix(".log")
            cmd = make_command(root, arm, shard, output.relative_to(root))
            start = {
                "event": "started",
                "arm": arm,
                "shard": shard,
                "seed": seed,
                "partition": "training" if shard in TRAIN_SHARDS else "development",
                "argv": cmd,
                "registration_sha256": digest(out / "registration.json"),
                "output": output.relative_to(root).as_posix(),
                "log": log.relative_to(root).as_posix(),
                "started_at": time.time(),
            }
            append_jsonl(ledger, start)
            start_time = time.monotonic()
            peak = 0
            with log.open("xb") as stream:
                command = [*reg["resource_launcher"], *cmd]
                process = subprocess.Popen(
                    command, cwd=root, stdout=stream, stderr=subprocess.STDOUT
                )
                try:
                    while process.poll() is None:
                        peak = max(peak, pilot.process_tree_rss_bytes(process.pid))
                        time.sleep(1)
                    status = process.returncode
                except BaseException as exc:
                    process.terminate()
                    process.wait()
                    append_jsonl(
                        ledger,
                        {
                            **start,
                            "event": "interrupted",
                            "error": repr(exc),
                            "exit_status": process.returncode,
                            "output_sha256": digest(output)
                            if output.exists()
                            else None,
                            "log_sha256": digest(log),
                        },
                    )
                    raise
            receipt = {
                **start,
                "event": "finished",
                "exit_status": status,
                "elapsed_seconds": time.monotonic() - start_time,
                "peak_sampled_process_tree_rss_bytes": peak,
                "output_sha256": digest(output) if output.exists() else None,
                "log_sha256": digest(log),
                "output_bytes": output.stat().st_size if output.exists() else 0,
            }
            append_jsonl(ledger, receipt)
            if status:
                raise subprocess.CalledProcessError(status, command)
            report = verify_shard(
                output,
                arm=f"{arm}/{shard}",
                expected_seed=seed,
                expected_game_count=GAMES_PER_SHARD,
            )
            if report["games"] != {
                "completed": GAMES_PER_SHARD,
                "capped": 0,
                "incomplete": 0,
            }:
                raise ValueError(f"new_shard_incomplete:{arm}:{shard}")
            # Only promote a verified shard to its stable corpus path.
            final.parent.mkdir(exist_ok=True)
            output.replace(final)
            receipt["published_output"] = final.relative_to(root).as_posix()
            receipt["published_output_sha256"] = digest(final)
            append_jsonl(
                ledger,
                {
                    "event": "published",
                    "arm": arm,
                    "shard": shard,
                    "output": receipt["published_output"],
                    "output_sha256": receipt["published_output_sha256"],
                },
            )
            completed.append(f"{arm}/{shard}")
    return {"completed_this_run": completed}


def identity(row: dict[str, Any]) -> tuple[str, str]:
    canonical_state = row.get("_canonical_state")
    if canonical_state is None:
        raise ValueError("state_identity_requires_replayed_state")
    # The row state is the production float32 vector; exact encoded bytes are canonicalized.
    encoded = np.asarray(row["state"], dtype=np.float32).tobytes().hex()
    return hashlib.sha256(canonical_state.encode()).hexdigest(), hashlib.sha256(
        bytes.fromhex(encoded)
    ).hexdigest()


def annotate_replayed_states(rows: list[dict[str, Any]]) -> None:
    from collections import defaultdict
    from ml.alphazero_lite.kalah_rules import KalahGame

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["game_index"])].append(row)
    for game_rows in grouped.values():
        sequence = sorted(game_rows, key=lambda r: int(r["move_index"]))
        if not sequence:
            continue
        game = KalahGame.from_state(sequence[0]["game_start_state"])
        moves = sequence[0]["game_moves"]
        if len(moves) != len(sequence):
            raise ValueError("trajectory_row_count_mismatch")
        for row, move in zip(sequence, moves, strict=True):
            state = game.to_state()
            canonical_tuple = tuple(
                state["player_pits"]
                + state["opponent_pits"]
                + [
                    state["player_store"],
                    state["opponent_store"],
                    state["current_player"],
                ]
            )
            row["_canonical_state"] = json.dumps(canonical_tuple, separators=(",", ":"))
            row["_active_stones"] = sum(state["player_pits"] + state["opponent_pits"])
            if not game.move(game.pit_index(int(move))):
                raise ValueError("trajectory_replay_failed_during_derivative_build")


def build_schedule(row_count: int, seed: int = BASE_SEED) -> dict[str, Any]:
    if row_count <= 0:
        raise ValueError("training_population_must_be_nonempty")
    order: list[int] = []
    boundaries = []
    while len(order) < EXPOSURES:
        pass_number = len(boundaries)
        permutation = (
            np.random.default_rng(seed + pass_number).permutation(row_count).tolist()
        )
        start = len(order)
        order.extend(permutation[: min(len(permutation), EXPOSURES - start)])
        boundaries.append(
            {
                "pass": pass_number,
                "permutation_seed": seed + pass_number,
                "start_exposure": start,
                "end_exposure_exclusive": len(order),
            }
        )
    batches, cursor = [], 0
    for update, size in enumerate(BATCHES * 4, 1):
        batches.append(
            {
                "update": update,
                "start_exposure": cursor,
                "end_exposure_exclusive": cursor + size,
                "row_occurrence_indices": order[cursor : cursor + size],
            }
        )
        cursor += size
    return {
        "updates": UPDATES,
        "exposures": len(order),
        "pass_boundaries": boundaries,
        "batches": batches,
        "row_order_sha256": hashlib.sha256(
            np.asarray(order, dtype="<u8").tobytes()
        ).hexdigest(),
    }


def development_exclusion_reasons(
    canonical_id: str, float32_id: str, training_by_arm: dict[str, set[str]]
) -> list[str]:
    reasons = []
    for arm, ids in sorted(training_by_arm.items()):
        if canonical_id in ids:
            reasons.append(f"canonical_overlap:{arm}")
        if float32_id in ids:
            reasons.append(f"float32_overlap:{arm}")
    return reasons


def build_derivatives(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    preflight(root)
    out = root / OUT
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    manifests: dict[str, Any] = {}
    memberships: dict[str, dict[str, set[str]]] = {}
    for arm in ("control", "treatment"):
        train_rows, dev_rows, manifest_shards = [], [], []
        ids = {"training": set(), "development": set()}
        for shard in range(SHARDS):
            path = shard_path(root, arm, shard)
            report = verify_shard(
                path,
                arm=f"{arm}/{shard}",
                expected_seed=BASE_SEED + shard,
                expected_game_count=GAMES_PER_SHARD,
            )
            rows = read_jsonl(path)
            annotate_replayed_states(rows)
            part = "training" if shard in TRAIN_SHARDS else "development"
            for row in rows:
                row["_seed549_source"] = {
                    "arm": arm,
                    "shard": shard,
                    "game_index": row["game_index"],
                    "move_index": row["move_index"],
                }
                (train_rows if part == "training" else dev_rows).append(row)
                ids[part].add(identity(row)[0])
                ids[part].add(identity(row)[1])
            manifest_shards.append(
                {
                    "shard": shard,
                    "partition": part,
                    "path": path.relative_to(root).as_posix(),
                    "sha256": digest(path),
                    "rows": len(rows),
                    "games": report["games"],
                }
            )
        arm_rows[arm] = train_rows + dev_rows
        memberships[arm] = ids
        manifests[arm] = {
            "arm": arm,
            "shards": manifest_shards,
            "training_rows": sum(1 for r in train_rows),
            "development_rows": sum(1 for r in dev_rows),
        }
        write_json(out / f"{arm}-source-manifest.json", manifests[arm])
        for part, rows in (("training", train_rows), ("development", dev_rows)):
            with (out / f"{arm}-{part}-rows.jsonl").open(
                "x", encoding="utf-8"
            ) as stream:
                for row in rows:
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    # Shared descriptive development pool with training exclusions by either identity.
    train_identities: dict[str, set[str]] = {arm: set() for arm in arm_rows}
    for arm in arm_rows:
        for row in arm_rows[arm]:
            source = row["_seed549_source"]
            if source["shard"] in TRAIN_SHARDS:
                train_identities[arm].update(identity(row))
    common_dev: list[dict[str, Any]] = []
    for arm in arm_rows:
        for row in arm_rows[arm]:
            source = row["_seed549_source"]
            if source["shard"] not in {6, 7}:
                continue
            canonical_id, float_id = identity(row)
            reasons = []
            reasons = development_exclusion_reasons(
                canonical_id, float_id, train_identities
            )
            common_dev.append(
                {
                    "arm": arm,
                    "source": source,
                    "canonical_sha256": canonical_id,
                    "float32_sha256": float_id,
                    "included": not reasons,
                    "exclusion_reasons": reasons,
                    "policy": row["policy"],
                    "value": row.get("value"),
                    "active_stones": row["_active_stones"],
                }
            )
    with (out / "shared-development-population.jsonl").open(
        "x", encoding="utf-8"
    ) as stream:
        for row in common_dev:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    # Schedules index occurrences (not deduplicated states), each pass shuffled without replacement.
    schedules = {}
    for arm in arm_rows:
        rows = read_jsonl(out / f"{arm}-training-rows.jsonl")
        schedule = {"arm": arm, **build_schedule(len(rows))}
        schedules[arm] = schedule
        write_json(out / f"{arm}-training-schedule.json", schedule)
    receipt = {
        "schema": "seed549-derivatives-v1",
        "generation_registration_sha256": digest(out / "registration.json"),
        "manifests": {
            arm: digest(out / f"{arm}-source-manifest.json") for arm in manifests
        },
        "training_rows": {
            arm: digest(out / f"{arm}-training-rows.jsonl") for arm in arm_rows
        },
        "development_rows": {
            arm: digest(out / f"{arm}-development-rows.jsonl") for arm in arm_rows
        },
        "shared_development": digest(out / "shared-development-population.jsonl"),
        "schedules": {
            arm: digest(out / f"{arm}-training-schedule.json") for arm in arm_rows
        },
        "schedule_exposures": {arm: schedules[arm]["exposures"] for arm in arm_rows},
        "status": "derivatives_built",
    }
    write_json(out / "post-generation-receipt.json", receipt)
    return receipt


def verify(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    reg = preflight(root)
    out = root / OUT
    ledger = read_jsonl(out / "attempts.jsonl")
    profiles = expected_search_profile(
        reg,
        json.loads((root / "model-artifact/current/search_policy.json").read_text()),
    )
    reports = {}
    for arm in ("control", "treatment"):
        shard_reports = []
        for shard in range(SHARDS):
            path = shard_path(root, arm, shard)
            report = verify_shard(
                path,
                arm=f"{arm}/{shard}",
                expected_seed=BASE_SEED + shard,
                expected_game_count=GAMES_PER_SHARD,
                expected_profile=profiles,
            )
            if report["games"] != {
                "completed": GAMES_PER_SHARD,
                "capped": 0,
                "incomplete": 0,
            }:
                raise ValueError(f"shard_accounting_invalid:{arm}:{shard}")
            shard_reports.append(report)
        reports[arm] = {
            "games": sum(r["games"]["completed"] for r in shard_reports),
            "rows": sum(r["rows"] for r in shard_reports),
            "lengths": [length for r in shard_reports for length in r["lengths"]],
            "unique_canonical": sum(0 for _ in []),
            "shards": shard_reports,
        }
        # Loader accepts each combined occurrence file with default targets.
        accepted = __import__(
            "ml.alphazero_lite.train", fromlist=["load_jsonl"]
        ).load_jsonl(out / f"{arm}-training-rows.jsonl")
        reports[arm]["loader_accepted_training_rows"] = int(accepted[0].shape[0])
    receipt = json.loads((out / "post-generation-receipt.json").read_text())
    for p, expected in {
        **receipt["manifests"],
        **receipt["training_rows"],
        **receipt["development_rows"],
        **receipt["schedules"],
        "shared-development-population.jsonl": receipt["shared_development"],
    }.items():
        path = out / p if "/" not in p else out / p
        if digest(path) != expected:
            raise ValueError(f"derivative_hash_mismatch:{p}")
    return {
        "status": "verified",
        "arms": reports,
        "attempt_records": len(ledger),
        "receipt_sha256": digest(out / "post-generation-receipt.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "preflight",
            "authorize-retry",
            "execute",
            "worker",
            "build-derivatives",
            "verify",
        ),
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--arm", choices=("control", "treatment"))
    args, unknown = parser.parse_known_args()
    if args.action == "prepare":
        result = prepare(args.root)
    elif args.action == "preflight":
        result = preflight(args.root)
    elif args.action == "authorize-retry":
        result = record_zero_output_retry_correction(args.root)
    elif args.action == "execute":
        result = execute(args.root)
    elif args.action == "build-derivatives":
        result = build_derivatives(args.root)
    elif args.action == "verify":
        result = verify(args.root)
    elif args.action == "worker":
        if args.arm is None:
            parser.error("worker requires --arm")
        run_worker(args.arm, ["--arm", args.arm, *unknown])
        return
    else:
        raise AssertionError(args.action)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
