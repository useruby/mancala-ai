"""Versioned #580 data-generation runner with one warmed probe per pair."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any

from ml.alphazero_lite import arena, run_varied_opponent_pilot as frozen_runner
from ml.alphazero_lite import varied_opponent_pilot as pilot
from ml.alphazero_lite.kalah_rules import KalahGame


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
    probe = (probe_factory or arena.NativeExactRootTablebase)(
        assets["native_probe"]["path"],
        assets["tablebase"]["path"],
        warm_on_start=False,
    )
    try:
        probe.warm()

        def exact_root(game: KalahGame, evaluator: Any) -> Any:
            return arena.exact_root_decision(
                game, evaluator, tablebase=probe, threshold=threshold
            )

        games = []
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
    finally:
        probe.close()
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


def _run_attempt(
    job: dict[str, Any], artifacts: str, attempt_path: Path
) -> dict[str, Any]:
    pilot.write_json_atomic(
        attempt_path,
        {
            "pair_id": str(job["game_specs"][0]["pair_id"]),
            "status": "started",
            "assignment": job,
        },
    )
    accounting_start = (time.process_time_ns(), pilot.reaped_child_cpu_ns())
    pilot._initialize_pair_process(artifacts)
    if pilot._PROCESS_EVALUATORS is None:
        raise RuntimeError("pair_process_evaluators_not_initialized")
    return play_pair_with_probe(
        job, pilot._PROCESS_EVALUATORS, accounting_start=accounting_start
    )


def run_checkpointed_pairs(
    jobs: list[dict[str, Any]], artifacts: Path, checkpoint_dir: Path, attempt_dir: Path
) -> list[dict[str, Any]]:
    """Run and atomically commit complete pairs, failing closed on interrupted attempts."""
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
                path, {"pair_id": pair_id, "status": "dispatched", "assignment": job}
            )
            future_jobs[pool.submit(_run_attempt, job, str(artifacts), path)] = (
                job,
                path,
            )
        for future in concurrent.futures.as_completed(future_jobs):
            job, path = future_jobs[future]
            result = future.result()
            checkpoint = pilot.commit_verified_pair(
                checkpoint_dir,
                job,
                result,
                attempt_number=numbers[str(result["pair_id"])],
            )
            pilot.write_json_atomic(
                path,
                {
                    "pair_id": result["pair_id"],
                    "status": "completed",
                    "assignment": job,
                    "checkpoint_sha256": pilot.sha256(checkpoint),
                    "pair_cpu_time_ns": result["pair_cpu_time_ns"],
                },
            )
            committed.append(result)
    return committed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
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
    freeze = frozen_runner.load_freeze(
        args.freeze, expected_digest=args.expected_freeze_sha256
    )
    frozen_runner.verify_execution_sources(freeze, Path(__file__).resolve().parents[2])
    frozen_runner.verify_resource_bindings(freeze)
    jobs = frozen_runner.pair_jobs(freeze)
    if len(jobs) != 128:
        raise ValueError("pilot_freeze_pair_count_mismatch")
    pilot.verify_components(args.artifacts)
    pilot.effective_runtime_policy(args.artifacts)
    completed = run_checkpointed_pairs(
        jobs,
        args.artifacts,
        args.checkpoint_dir,
        args.checkpoint_dir.parent / "attempts",
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


if __name__ == "__main__":
    main()
