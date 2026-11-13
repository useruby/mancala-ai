"""Versioned #580 data-generation runner with one warmed probe per pair."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from ml.alphazero_lite import arena, run_varied_opponent_pilot as frozen_runner
from ml.alphazero_lite import varied_opponent_pilot as pilot
from ml.alphazero_lite.kalah_rules import KalahGame

RUNNER_RELATIVE_PATH = "ml/alphazero_lite/run_varied_opponent_pilot_v2.py"
RUN_IDENTITY_SCHEMA = "varied_opponent_pair_run_identity_v2"
PAIR_LIFETIME = "one_fresh_warmed_probe_per_seat_swapped_pair"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_v2_freeze(
    path: Path, source_root: Path, *, expected_digest: str
) -> dict[str, Any]:
    """Authenticate the versioned execution contract and its runner source."""
    freeze = json.loads(path.read_text())
    recorded_digest = freeze.pop("freeze_sha256", None)
    actual_digest = hashlib.sha256(canonical_json(freeze)).hexdigest()
    freeze["freeze_sha256"] = recorded_digest
    if recorded_digest != actual_digest:
        raise ValueError("pilot_v2_freeze_digest_mismatch")
    if actual_digest != expected_digest:
        raise ValueError("pilot_v2_freeze_external_digest_mismatch")
    if freeze.get("schema") != "varied_opponent_pilot_execution_freeze_v2":
        raise ValueError("pilot_v2_freeze_schema_mismatch")
    if freeze.get("runner_source") != RUNNER_RELATIVE_PATH:
        raise ValueError("pilot_v2_runner_source_binding_mismatch")
    if sha256(source_root / RUNNER_RELATIVE_PATH) != freeze.get("runner_sha256"):
        raise ValueError("pilot_v2_runner_source_hash_mismatch")
    if freeze.get("probe_lifetime") != PAIR_LIFETIME:
        raise ValueError("pilot_v2_probe_lifetime_mismatch")
    parent = freeze.get("parent_freeze")
    if not isinstance(parent, dict) or not parent.get("path"):
        raise ValueError("pilot_v2_parent_freeze_binding_missing")
    parent_path = source_root / parent["path"]
    base_freeze = frozen_runner.load_freeze(
        parent_path, expected_digest=str(parent.get("sha256", ""))
    )
    frozen_runner.verify_execution_sources(base_freeze, source_root)
    if freeze.get("runtime_assets") != {
        name: {"sha256": value["sha256"]}
        for name, value in json.loads(
            (
                source_root
                / "docs/data/varied-opponent-coverage-pilot/probe-reuse-manifest.json"
            ).read_text()
        )["runtime"].items()
    }:
        raise ValueError("pilot_v2_runtime_binding_mismatch")
    return freeze


def execution_identity(freeze: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": RUN_IDENTITY_SCHEMA,
        "freeze_sha256": freeze["freeze_sha256"],
        "runner_source": freeze["runner_source"],
        "runner_sha256": freeze["runner_sha256"],
        "probe_lifetime": freeze["probe_lifetime"],
        "checkpoint_schema": "varied_opponent_pair_checkpoint_v1",
    }


def bind_run_directory(
    checkpoint_dir: Path,
    attempt_dir: Path,
    identity: dict[str, Any],
) -> Path:
    """Reject pre-v2 evidence before creating a run marker or dispatching work."""
    marker = checkpoint_dir / "run-identity.v2"
    files = [*checkpoint_dir.glob("*.json"), *attempt_dir.glob("*.json")]
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("pilot_v2_run_identity_conflict")
    elif files:
        raise ValueError("pilot_v2_unbound_existing_run_evidence")
    else:
        pilot.write_json_atomic(marker, identity)
    for path in [*checkpoint_dir.glob("*.json")]:
        if path == marker:
            continue
        payload = json.loads(path.read_text())
        if (
            payload.get("execution_identity") != identity
            or payload.get("result", {}).get("execution_identity") != identity
        ):
            raise ValueError("pilot_v2_checkpoint_identity_mismatch")
    for path in attempt_dir.glob("*.json"):
        if json.loads(path.read_text()).get("execution_identity") != identity:
            raise ValueError("pilot_v2_attempt_identity_mismatch")
    return marker


@contextlib.contextmanager
def warmed_probe(
    binary: str | Path,
    tablebase: str | Path,
    *,
    probe_factory: Any | None = None,
):
    """Own, warm, close, and reap one exact-root probe for a lifecycle unit."""
    probe = (probe_factory or arena.NativeExactRootTablebase)(
        binary, tablebase, warm_on_start=False
    )
    try:
        probe.warm()
        yield probe
    finally:
        probe.close()


def play_pair_with_probe(
    job: dict[str, Any],
    evaluators: pilot.PilotEvaluators,
    *,
    probe_factory: Any | None = None,
    max_moves: int = pilot.MAX_MOVES,
    accounting_start: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Run both seat-swapped games against one warmed, pair-owned probe."""
    specs = job["game_specs"]
    pilot.validate_game_specs(specs)
    policy = evaluators.runtime_policy
    assets = policy["runtime_assets"]
    threshold = int(policy["effective_search_policy"]["exact_root_threshold"])
    process_start, child_start = accounting_start or (
        time.process_time_ns(),
        pilot.reaped_child_cpu_ns(),
    )
    games = []
    with warmed_probe(
        assets["native_probe"]["path"],
        assets["tablebase"]["path"],
        probe_factory=probe_factory,
    ) as probe:
        exact_root = exact_root_callback(probe, threshold)
        for spec in specs:
            opponent = None
            if spec["arm"] == "treatment":
                opponent = (
                    evaluators.initializer
                    if spec["opponent"] == "initializer"
                    else evaluators.t512
                )
            games.append(
                pilot.play_game(
                    spec,
                    p_evaluator=evaluators.p,
                    opponent_evaluator=opponent,
                    search_turn=lambda game, evaluator, rng, ply, retain: (
                        pilot.production_search_turn(
                            game,
                            evaluator,
                            rng,
                            ply,
                            retain,
                            exact_root=exact_root,
                        )
                    ),
                    max_moves=max_moves,
                )
            )
    child_cpu = pilot.reaped_child_cpu_ns() - child_start
    if child_cpu < 0:
        raise ValueError("reaped_child_cpu_counter_regressed")
    process_cpu = time.process_time_ns() - process_start
    result = {
        "pair_id": str(specs[0]["pair_id"]),
        "arm": str(specs[0]["arm"]),
        "games": games,
        "pair_process_cpu_time_ns": process_cpu,
        "native_child_cpu_time_ns": child_cpu,
        "pair_cpu_time_ns": process_cpu + child_cpu,
    }
    pilot.verify_pair_result(job, result)
    return result


def exact_root_callback(probe: Any, threshold: int):
    """Build the production exact-root callback for a warmed native probe."""

    def exact_root(game: KalahGame, evaluator: Any) -> Any:
        return arena.exact_root_decision(
            game, evaluator, tablebase=probe, threshold=threshold
        )

    return exact_root


def replay_frozen_requests(
    manifest: dict[str, Any],
    binary: Path,
    tablebase: Path,
    *,
    probe_factory: Any | None = None,
) -> dict[str, Any]:
    """Replay frozen roots through v2's exact-root callback and compare lifetimes."""
    expected_runtime = manifest["runtime"]
    if sha256(binary) != expected_runtime["native_probe"]["sha256"]:
        raise ValueError("pilot_v2_replay_probe_hash_mismatch")
    if sha256(tablebase) != expected_runtime["tablebase"]["sha256"]:
        raise ValueError("pilot_v2_replay_tablebase_hash_mismatch")
    boundaries = manifest["game_boundaries"]
    requests = manifest["requests"]
    if (
        len(boundaries) != 256
        or sum(int(item["requests"]) for item in boundaries) != 3354
    ):
        raise ValueError("pilot_v2_replay_boundary_contract_mismatch")
    by_game: dict[str, list[dict[str, Any]]] = {}
    for request in requests:
        by_game.setdefault(str(request["game_id"]), []).append(request)
    boundary_counts = {
        str(item["game_id"]): int(item["requests"]) for item in boundaries
    }
    if (
        {game_id for game_id, count in boundary_counts.items() if count > 0}
        != set(by_game)
        or any(
            len(by_game.get(game_id, [])) != count
            for game_id, count in boundary_counts.items()
        )
        or len(requests) != 3354
    ):
        raise ValueError("pilot_v2_replay_request_boundary_mismatch")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for boundary in boundaries:
        grouped.setdefault(str(boundary["pair_id"]), []).append(boundary)
    factory = probe_factory or arena.NativeExactRootTablebase
    threshold = 16
    baseline: dict[str, dict[str, Any]] = {}
    reused: dict[str, dict[str, Any]] = {}
    for pair_id, pair_boundaries in grouped.items():
        pair_requests = [
            request
            for boundary in pair_boundaries
            for request in by_game.get(str(boundary["game_id"]), [])
        ]
        for boundary in pair_boundaries:
            with warmed_probe(binary, tablebase, probe_factory=factory) as probe:
                callback = exact_root_callback(probe, threshold)
                for request in by_game.get(str(boundary["game_id"]), []):
                    baseline[_request_key(request)] = _replay_decision(
                        callback, request
                    )
        with warmed_probe(binary, tablebase, probe_factory=factory) as probe:
            callback = exact_root_callback(probe, threshold)
            for request in pair_requests:
                reused[_request_key(request)] = _replay_decision(callback, request)
    mismatch = []
    archived_margin_mismatch = []
    archived_move_mismatch = []
    for request in requests:
        key = _request_key(request)
        left, right = baseline[key], reused[key]
        if left != right:
            mismatch.append({"request": key, "per_game": left, "per_pair": right})
        if request["archived_exact_row"]:
            for lifetime, result in (("per_game", left), ("per_pair", right)):
                if result["action_margins"] != request["archived_action_margins"]:
                    archived_margin_mismatch.append(
                        {"request": key, "lifetime": lifetime}
                    )
        for lifetime, result in (("per_game", left), ("per_pair", right)):
            if result["selected_move"] != request["archived_move"]:
                archived_move_mismatch.append({"request": key, "lifetime": lifetime})
    if len(grouped) != 128 or any(len(items) != 2 for items in grouped.values()):
        raise ValueError("pilot_v2_replay_pair_boundary_mismatch")
    return {
        "schema": "varied_opponent_v2_frozen_request_replay_v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "requests": len(requests),
        "game_boundaries": len(boundaries),
        "pairs": len(grouped),
        "per_game_probe_count": len(boundaries),
        "per_pair_probe_count": len(grouped),
        "decision_mismatches": mismatch,
        "archived_margin_mismatches": archived_margin_mismatch,
        "archived_move_mismatches": archived_move_mismatch,
        "passed": not mismatch
        and not archived_margin_mismatch
        and not archived_move_mismatch,
    }


def _request_key(request: dict[str, Any]) -> str:
    return str(request["game_id"]) + f":{int(request['ply'])}"


def _replay_decision(callback: Any, request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    game = KalahGame.from_state(request["state"])

    class FrozenPriorEvaluator:
        def evaluate(self, _game: KalahGame):
            return np.asarray(request["tie_break_priors"], dtype=np.float64), 0.0

    decision = callback(game, FrozenPriorEvaluator())
    if decision is None:
        raise ValueError("pilot_v2_replay_root_not_exact_eligible")
    return {
        "action_margins": {
            str(move): int(decision.action_margins[move])
            for move in sorted(decision.action_margins)
        },
        "selected_move": int(decision.selected_move),
    }


def _run_attempt(
    job: dict[str, Any],
    artifacts: str,
    attempt_path: Path,
    identity: dict[str, Any],
) -> dict[str, Any]:
    pilot.write_json_atomic(
        attempt_path,
        {
            "pair_id": str(job["game_specs"][0]["pair_id"]),
            "status": "started",
            "assignment": job,
            "execution_identity": identity,
        },
    )
    accounting_start = (time.process_time_ns(), pilot.reaped_child_cpu_ns())
    pilot._initialize_pair_process(artifacts)
    if pilot._PROCESS_EVALUATORS is None:
        raise RuntimeError("pair_process_evaluators_not_initialized")
    result = play_pair_with_probe(
        job, pilot._PROCESS_EVALUATORS, accounting_start=accounting_start
    )
    result["execution_identity"] = identity
    return result


def run_checkpointed_pairs(
    jobs: list[dict[str, Any]],
    artifacts: Path,
    checkpoint_dir: Path,
    attempt_dir: Path,
    identity: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run and atomically commit complete pairs, failing closed on interrupted attempts."""
    bind_run_directory(checkpoint_dir, attempt_dir, identity)
    committed, pending = pilot.load_pair_checkpoints(checkpoint_dir, jobs)
    committed_by_id = {str(result["pair_id"]): result for result in committed}
    attempt_dir.mkdir(parents=True, exist_ok=True)
    numbers = pilot.reconcile_pair_attempts(
        attempt_dir, jobs, committed_by_id, checkpoint_dir
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=pilot.PAIR_CONCURRENCY, max_tasks_per_child=1
    ) as pool:
        future_jobs = {}
        for job in pending:
            pair_id = str(job["game_specs"][0]["pair_id"])
            path = pilot.pair_attempt_path(attempt_dir, pair_id, numbers[pair_id])
            pilot.write_json_atomic(
                path,
                {
                    "pair_id": pair_id,
                    "status": "dispatched",
                    "assignment": job,
                    "execution_identity": identity,
                },
            )
            future_jobs[
                pool.submit(_run_attempt, job, str(artifacts), path, identity)
            ] = (job, path)
        for future in concurrent.futures.as_completed(future_jobs):
            job, path = future_jobs[future]
            result = future.result()
            checkpoint = commit_v2_pair(
                checkpoint_dir,
                job,
                result,
                identity=identity,
                attempt_number=numbers[str(result["pair_id"])],
            )
            pilot.write_json_atomic(
                path,
                {
                    "pair_id": result["pair_id"],
                    "status": "completed",
                    "assignment": job,
                    "execution_identity": identity,
                    "checkpoint_sha256": pilot.sha256(checkpoint),
                    "pair_cpu_time_ns": result["pair_cpu_time_ns"],
                },
            )
            committed.append(result)
    return committed


def commit_v2_pair(
    checkpoint_dir: Path,
    job: dict[str, Any],
    result: dict[str, Any],
    *,
    identity: dict[str, Any],
    attempt_number: int,
) -> Path:
    """Atomically commit complete pair evidence bound to the v2 run identity."""
    pilot.verify_pair_result(job, result)
    if result.get("execution_identity") != identity:
        raise ValueError("pilot_v2_result_identity_mismatch")
    specs = job["game_specs"]
    path = pilot.pair_checkpoint_path(checkpoint_dir, str(specs[0]["pair_id"]))
    payload = {
        "schema": "varied_opponent_pair_checkpoint_v1",
        "assignment": json.loads(json.dumps(job, sort_keys=True)),
        "producing_attempt": attempt_number,
        "execution_identity": identity,
        "result": result,
    }
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError("pair_checkpoint_conflict")
        return path
    pilot.write_json_atomic(path, payload)
    return path


def main() -> None:
    if sys.argv[1:2] == ["replay"]:
        replay_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze-v2",
        type=Path,
        default=Path("docs/data/varied-opponent-pilot-v2/freeze.json"),
    )
    parser.add_argument("--expected-freeze-v2-sha256", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(
            "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts"
        ),
    )
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[2]
    freeze = load_v2_freeze(
        args.freeze_v2,
        source_root,
        expected_digest=args.expected_freeze_v2_sha256,
    )
    frozen_runner.verify_resource_bindings(freeze)
    base_freeze = frozen_runner.load_freeze(
        source_root / freeze["parent_freeze"]["path"],
        expected_digest=freeze["parent_freeze"]["sha256"],
    )
    jobs = frozen_runner.pair_jobs(base_freeze)
    if len(jobs) != 128:
        raise ValueError("pilot_freeze_pair_count_mismatch")
    pilot.verify_components(args.artifacts)
    effective_runtime = pilot.effective_runtime_policy(args.artifacts)
    observed_assets = {
        name: {"sha256": asset["sha256"]}
        for name, asset in effective_runtime["runtime_assets"].items()
    }
    if observed_assets != freeze["runtime_assets"]:
        raise ValueError("pilot_v2_effective_runtime_binding_mismatch")
    completed = run_checkpointed_pairs(
        jobs,
        args.artifacts,
        args.checkpoint_dir,
        args.checkpoint_dir.parent / "attempts",
        execution_identity(freeze),
    )
    print(
        json.dumps(
            {
                "pairs": len(completed),
                "accounting": pilot.summarize_pair_accounting(completed),
            },
            indent=2,
            sort_keys=True,
        )
    )


def replay_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Replay the frozen #581 requests through v2."
    )
    parser.add_argument(
        "--freeze-v2",
        type=Path,
        default=Path("docs/data/varied-opponent-pilot-v2/freeze.json"),
    )
    parser.add_argument("--expected-freeze-v2-sha256", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "docs/data/varied-opponent-coverage-pilot/probe-reuse-manifest.json"
        ),
    )
    parser.add_argument(
        "--probe", type=Path, default=Path("model-artifact/runtime/kalah_v1_tablebase")
    )
    parser.add_argument(
        "--tablebase",
        type=Path,
        default=Path("model-artifact/runtime/kalah_v1_21.kvtb"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source_root = Path(__file__).resolve().parents[2]
    freeze = load_v2_freeze(
        args.freeze_v2,
        source_root,
        expected_digest=args.expected_freeze_v2_sha256,
    )
    manifest = json.loads(args.manifest.read_text())
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual_digest = hashlib.sha256(canonical_json(body)).hexdigest()
    if actual_digest != manifest.get("manifest_sha256") or actual_digest != freeze.get(
        "request_manifest_sha256"
    ):
        raise ValueError("pilot_v2_replay_manifest_digest_mismatch")
    result = replay_frozen_requests(manifest, args.probe, args.tablebase)
    result["execution_identity"] = execution_identity(freeze)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
