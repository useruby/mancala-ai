"""Reconstruct #580 exact-root requests and benchmark native probe lifetimes.

This is an experiment-local benchmark. It does not launch games or alter the
production runner. Run manifest mode first, inspect its freeze, then run bench.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import run_varied_opponent_pilot as runner
from ml.alphazero_lite import varied_opponent_pilot as pilot
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.native_exact_root_tablebase import NativeExactRootTablebase

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/varied-opponent-coverage-pilot"
FREEZE_SHA = "8da846678e89568e17e7bd81270287ef373e1f8acdfcbc892317d08385ad4c36"
EVIDENCE_SHA = "7c8d32ced1fd56ae71f261b9dcbc482b5248a62ce825409513bb9ad78aed2c0c"
MANIFEST_SHA = "d5bf560e2932b58c643cce872c17d4608bbb358cc9e7372979b898d1228302af"


def _child_cpu_ns() -> int:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return int(round((usage.ru_utime + usage.ru_stime) * 1_000_000_000))


def _archive() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    freeze = runner.load_freeze(DATA / "freeze.json", expected_digest=FREEZE_SHA)
    runner.verify_execution_sources(freeze, ROOT)
    jobs = runner.pair_jobs(freeze)
    committed, pending = pilot.load_pair_checkpoints(DATA / "pairs", jobs)
    if pending or len(committed) != 128:
        raise ValueError("checkpoint_coverage_mismatch")
    expected_evidence = EVIDENCE_SHA
    verify = __import__(
        "ml.alphazero_lite.verify_varied_opponent_pilot", fromlist=["verify"]
    )
    verify.verify(ROOT, FREEZE_SHA, expected_evidence)
    games = {
        game["spec"]["game_id"]: game for pair in committed for game in pair["games"]
    }
    if len(games) != 256:
        raise ValueError("game_count_mismatch")
    jobs_by_pair = {job["game_specs"][0]["pair_id"]: job for job in jobs}
    published_pair_order = list(
        dict.fromkeys(str(spec["pair_id"]) for spec in freeze["assignments"])
    )
    return freeze, [
        {
            "pair_id": pair_id,
            "game_specs": jobs_by_pair[pair_id]["game_specs"],
            "games": [
                games[spec["game_id"]] for spec in jobs_by_pair[pair_id]["game_specs"]
            ],
        }
        for pair_id in published_pair_order
    ]


def _evaluator(
    evaluators: pilot.PilotEvaluators, spec: dict[str, Any], player: int
) -> Any:
    opponent = None
    if spec["arm"] == "treatment":
        opponent = (
            evaluators.initializer
            if spec["opponent"] == "initializer"
            else evaluators.t512
        )
    return pilot.evaluator_for_player(
        player=player,
        designated_p_seat=int(spec["designated_p_seat"]),
        arm=str(spec["arm"]),
        p_evaluator=evaluators.p,
        opponent_evaluator=opponent,
    )


def build_manifest(artifact_root: Path) -> dict[str, Any]:
    freeze, pairs = _archive()
    evaluators = pilot.load_evaluators(artifact_root)
    requests: list[dict[str, Any]] = []
    game_boundaries = []
    archived_crosschecks = 0
    for pair in pairs:
        for game_record in pair["games"]:
            spec = game_record["spec"]
            rows = {int(row["move_index"]): row for row in game_record["rows"]}
            game_requests = 0
            for step in game_record["trajectory"]:
                state = step["state"]
                game = KalahGame.from_state(state)
                if sum(game.pits) > pilot.EXACT_ROOT_THRESHOLD:
                    continue
                legal = [int(move) for move in game.possible_moves()]
                evaluator = _evaluator(evaluators, spec, int(game.current_player))
                priors, _value = evaluator.evaluate(game)
                priors = [float(value) for value in np.asarray(priors).tolist()]
                row = rows.get(int(step["ply"]))
                archived = (
                    row is not None
                    and row.get("teacher_source") == "exact_root_tablebase"
                )
                if row is not None and not archived:
                    raise ValueError("eligible_retained_row_not_exact_root")
                if archived:
                    if row["canonical_state"] != state or row["legal_moves"] != legal:
                        raise ValueError("archived_root_input_mismatch")
                    stored = [float(value) for value in row["exact_network_priors"]]
                    if not np.allclose(priors, stored, rtol=0.0, atol=1e-7):
                        raise ValueError("archived_prior_reconstruction_mismatch")
                    if int(step["move"]) != int(row["exact_selected_action"]):
                        raise ValueError("archived_move_mismatch")
                    archived_crosschecks += 1
                requests.append(
                    {
                        "pair_id": pair["pair_id"],
                        "game_id": spec["game_id"],
                        "ply": int(step["ply"]),
                        "state": state,
                        "legal_moves": legal,
                        "archived_move": int(step["move"]),
                        "tie_break_priors": priors,
                        "archived_exact_row": archived,
                        "archived_action_margins": row.get("exact_action_margins")
                        if archived
                        else None,
                        "archived_selected_move": row.get("exact_selected_action")
                        if archived
                        else None,
                    }
                )
                game_requests += 1
            game_boundaries.append(
                {
                    "pair_id": pair["pair_id"],
                    "game_id": spec["game_id"],
                    "requests": game_requests,
                }
            )
    body = {
        "schema": "exact_root_probe_reuse_manifest_v1",
        "freeze_sha256": FREEZE_SHA,
        "evidence_manifest_sha256": EVIDENCE_SHA,
        "runtime": {
            name: {"sha256": asset["sha256"]}
            for name, asset in evaluators.runtime_policy["runtime_assets"].items()
        },
        "execution_freeze": {
            "pair_concurrency": 1,
            "workers_per_game": 1,
            "cpu_quota": "1800%",
            "cpu_weight": "100",
            "memory_max_bytes": 38654705664,
            "allowed_cpus": "14-31",
            "nice": 19,
            "variant_order": "alternate_per_pair_starting_per_game",
            "lifetime": {
                "per_game": "one_fresh_warmed_probe_per_game",
                "per_pair": "one_fresh_warmed_probe_per_seat_swapped_pair",
            },
        },
        "game_count": len(game_boundaries),
        "pair_count": len(pairs),
        "pair_order": [pair["pair_id"] for pair in pairs],
        "archived_exact_row_crosschecks": archived_crosschecks,
        "game_boundaries": game_boundaries,
        "requests": requests,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return body


def _decide(probe: NativeExactRootTablebase, request: dict[str, Any]) -> dict[str, Any]:
    game = KalahGame.from_state(request["state"])
    margins = probe.root_action_margins(game, int(game.current_player))
    legal_moves = {int(move) for move in request["legal_moves"]}
    observed_moves = {int(move) for move in margins}
    if observed_moves != legal_moves:
        raise ValueError(
            "native_probe_legal_action_coverage_mismatch:"
            f"missing={sorted(legal_moves - observed_moves)}:"
            f"unexpected={sorted(observed_moves - legal_moves)}"
        )
    priors = request["tie_break_priors"]
    optimal = [
        action for action, margin in margins.items() if margin == max(margins.values())
    ]
    selected = max(optimal, key=lambda action: (priors[action], -action))
    return {
        "action_margins": {
            str(action): int(margins[action]) for action in sorted(margins)
        },
        "selected_move": int(selected),
    }


def _run_lifetime(
    requests: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    binary: Path,
    tablebase: Path,
    lifetime: str,
) -> dict[str, Any]:
    outputs: dict[str, dict[str, Any]] = {}
    timings = {"startup_wall_ns": 0, "warmup_wall_ns": 0, "query_wall_ns": 0}
    child_before = _child_cpu_ns()
    group_cpu: dict[str, int] = {}
    by_game: dict[str, list[dict[str, Any]]] = {}
    for request in requests:
        by_game.setdefault(str(request["game_id"]), []).append(request)
    for group in groups:
        ids = [str(game_id) for game_id in group["game_ids"]]
        if lifetime == "per_game":
            units = [[game_id] for game_id in ids]
        else:
            units = [ids]
        for unit in units:
            before_cpu = _child_cpu_ns()
            started = time.perf_counter_ns()
            probe = NativeExactRootTablebase(binary, tablebase, warm_on_start=False)
            timings["startup_wall_ns"] += time.perf_counter_ns() - started
            try:
                started = time.perf_counter_ns()
                probe.warm()
                timings["warmup_wall_ns"] += time.perf_counter_ns() - started
                for game_id in unit:
                    for request in by_game.get(game_id, []):
                        started = time.perf_counter_ns()
                        outputs[request["game_id"] + f":{request['ply']}"] = _decide(
                            probe, request
                        )
                        timings["query_wall_ns"] += time.perf_counter_ns() - started
            finally:
                probe.close()
            spent = _child_cpu_ns() - before_cpu
            for game_id in unit:
                group_cpu[game_id] = spent if len(unit) == 1 else 0
            if len(unit) > 1:
                group_cpu["pair:" + str(group["pair_id"])] = spent
    return {
        "outputs": outputs,
        "timings": timings,
        "native_child_cpu_ns": _child_cpu_ns() - child_before,
        "group_native_child_cpu_ns": group_cpu,
    }


def benchmark(
    manifest: dict[str, Any], binary: Path, tablebase: Path
) -> dict[str, Any]:
    requests = manifest["requests"]
    boundaries = manifest["game_boundaries"]
    pairs: dict[str, list[dict[str, Any]]] = {}
    for boundary in boundaries:
        pairs.setdefault(str(boundary["pair_id"]), []).append(boundary)
    pair_groups = [
        {"pair_id": pair_id, "game_ids": [item["game_id"] for item in items]}
        for pair_id, items in pairs.items()
    ]
    baseline_cpu = treatment_cpu = 0
    base_outputs: dict[str, Any] = {}
    treatment_outputs: dict[str, Any] = {}
    totals = {
        name: {"startup_wall_ns": 0, "warmup_wall_ns": 0, "query_wall_ns": 0}
        for name in ("per_game", "per_pair")
    }
    pair_cpu: dict[str, dict[str, int]] = {}
    for index, group in enumerate(pair_groups):
        order = ("per_game", "per_pair") if index % 2 == 0 else ("per_pair", "per_game")
        for lifetime in order:
            subset = [
                request
                for request in requests
                if request["pair_id"] == group["pair_id"]
            ]
            run = _run_lifetime(subset, [group], binary, tablebase, lifetime)
            totals[lifetime] = {
                key: totals[lifetime][key] + value
                for key, value in run["timings"].items()
            }
            if lifetime == "per_game":
                baseline_cpu += run["native_child_cpu_ns"]
                base_outputs.update(run["outputs"])
                pair_cpu.setdefault(str(group["pair_id"]), {})[lifetime] = run[
                    "native_child_cpu_ns"
                ]
            else:
                treatment_cpu += run["native_child_cpu_ns"]
                treatment_outputs.update(run["outputs"])
                pair_cpu.setdefault(str(group["pair_id"]), {})[lifetime] = run[
                    "native_child_cpu_ns"
                ]
    mismatches = []
    archived_margin_mismatches = []
    archived_move_mismatches = []
    for request in requests:
        key = str(request["game_id"]) + f":{request['ply']}"
        left, right = base_outputs[key], treatment_outputs[key]
        expected_moves = {str(int(move)) for move in request["legal_moves"]}
        for lifetime, result in (("per_game", left), ("per_pair", right)):
            if set(result["action_margins"]) != expected_moves:
                raise ValueError(
                    f"benchmark_legal_action_coverage_mismatch:{lifetime}:{key}"
                )
        if left != right:
            mismatches.append(
                {
                    "game_id": request["game_id"],
                    "ply": request["ply"],
                    "per_game": left,
                    "per_pair": right,
                }
            )
        for lifetime, result in (("per_game", left), ("per_pair", right)):
            if (
                request["archived_exact_row"]
                and result["action_margins"] != request["archived_action_margins"]
            ):
                archived_margin_mismatches.append(
                    {
                        "lifetime": lifetime,
                        "game_id": request["game_id"],
                        "ply": request["ply"],
                    }
                )
            if result["selected_move"] != request["archived_move"]:
                archived_move_mismatches.append(
                    {
                        "lifetime": lifetime,
                        "game_id": request["game_id"],
                        "ply": request["ply"],
                        "selected": result["selected_move"],
                        "archived": request["archived_move"],
                    }
                )
    saving = 1.0 - treatment_cpu / baseline_cpu if baseline_cpu else None
    return {
        "schema": "exact_root_probe_reuse_benchmark_v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "requests": len(requests),
        "decision_mismatches": mismatches,
        "archived_margin_mismatches": archived_margin_mismatches,
        "archived_move_mismatches": archived_move_mismatches,
        "native_child_cpu_ns": {
            "per_game": baseline_cpu,
            "per_pair": treatment_cpu,
            "reduction_fraction": saving,
        },
        "per_pair_native_child_cpu_ns": pair_cpu,
        "wall_times": totals,
        "threshold_passed": not mismatches
        and not archived_margin_mismatches
        and not archived_move_mismatches
        and saving is not None
        and saving >= 0.20,
    }


def validate_frozen_benchmark(
    manifest: dict[str, Any], binary: Path, tablebase: Path
) -> dict[str, Any]:
    """Authenticate workload/runtime hashes and enforce the frozen scope."""
    expected_digest = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual_digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("request_manifest_digest_mismatch")
    if actual_digest != MANIFEST_SHA:
        raise ValueError("request_manifest_not_registered")
    if manifest.get("freeze_sha256") != FREEZE_SHA:
        raise ValueError("source_freeze_digest_mismatch")
    if manifest.get("evidence_manifest_sha256") != EVIDENCE_SHA:
        raise ValueError("source_evidence_digest_mismatch")
    runtime = manifest["runtime"]
    if pilot.sha256(binary) != runtime["native_probe"]["sha256"]:
        raise ValueError("native_probe_hash_mismatch")
    if pilot.sha256(tablebase) != runtime["tablebase"]["sha256"]:
        raise ValueError("tablebase_hash_mismatch")
    observed = pilot.verify_live_resource_limits(manifest["execution_freeze"])
    return {"manifest_sha256": actual_digest, "live_resources": observed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("manifest", "benchmark"))
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=ROOT
        / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts",
    )
    parser.add_argument(
        "--manifest", type=Path, default=DATA / "probe-reuse-manifest.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help="required for manifest mode; writes a separate manifest file",
    )
    parser.add_argument(
        "--probe",
        type=Path,
        default=ROOT / "model-artifact/runtime/kalah_v1_tablebase",
    )
    parser.add_argument(
        "--tablebase",
        type=Path,
        default=ROOT / "model-artifact/runtime/kalah_v1_21.kvtb",
    )
    args = parser.parse_args()
    output_path = args.manifest_output if args.mode == "manifest" else args.output
    if output_path is None:
        parser.error(
            "manifest mode requires --manifest-output"
            if args.mode == "manifest"
            else "benchmark mode requires --output"
        )
    if output_path.exists() or output_path.is_symlink():
        parser.error(f"output path already exists: {output_path}")
    if args.mode == "manifest":
        result = build_manifest(args.artifacts)
        with args.manifest_output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        manifest = json.loads(args.manifest.read_text())
        scope = validate_frozen_benchmark(manifest, args.probe, args.tablebase)
        result = benchmark(manifest, args.probe, args.tablebase)
        result["resource_scope"] = scope["live_resources"]
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"requests", "game_boundaries"}
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
