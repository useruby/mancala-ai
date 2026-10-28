"""Read-only verifier for seed542 corpus, membership, and training schedules."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from collections import Counter, defaultdict
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "docs/data/seed542-selfplay-4096"
sys.path.insert(0, str(ROOT))

from ml.alphazero_lite import prepare_seed542_experiment as rebuild  # noqa: E402
from ml.alphazero_lite import train, validate_seed500_pilot  # noqa: E402


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _all_inventory_events() -> list[dict[str, Any]]:
    events = []
    for path in (
        CORPUS / "execution-inventory.jsonl",
        CORPUS / "continuation-inventory.jsonl",
    ):
        if path.is_file():
            events.extend(_read_jsonl(path))
    return events


def verify_inventory() -> list[dict[str, Any]]:
    events = _all_inventory_events()
    finished_events = [event for event in events if event["event"] == "finished"]
    finished: dict[int, dict[str, Any]] = {}
    for event in finished_events:
        seed = int(event["seed"])
        if seed in finished:
            raise ValueError(f"inventory_duplicate_finished_event:{seed}")
        finished[seed] = event
    if not set(finished).issubset(set(range(542000, 542064))):
        raise ValueError("inventory_finished_seed_out_of_range")
    for seed, event in finished.items():
        path = CORPUS / f"shard-{seed}.jsonl"
        if event.get("exit_status") != 0 or not path.is_file():
            raise ValueError(f"inventory_shard_incomplete:{seed}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            event.get("output_sha256") != digest
            or event.get("output_bytes") != path.stat().st_size
        ):
            raise ValueError(f"inventory_shard_hash_mismatch:{seed}")
    return events


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _target_modes(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        key: dict(Counter(str(row.get(key, "missing")) for row in rows))
        for key in (
            "policy_target_mode",
            "policy_target_actual_mode",
            "value_target_mode",
        )
    }


def _cohort_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    seen: dict[tuple[str, str], set[bytes]] = defaultdict(set)
    buckets = {">32": ">32", "17–32": "17-32", "≤16": "<=16"}
    for teacher in ("PUCT", "exact-root"):
        for bucket in buckets:
            stats[(teacher, bucket)] = {
                "rows": 0,
                "duplicate_rows": 0,
                "entropy_sum": 0.0,
                "one_hot_rows": 0,
            }
    for row in rows:
        encoded = rebuild.input_key(row)
        state = np.asarray(row["state"], dtype=np.float32)
        active = int(round(float(state[:12].sum(dtype=np.float32)) * 48))
        bucket = ">32" if active > 32 else "17–32" if active >= 17 else "≤16"
        teacher = "exact-root" if active <= 16 else "PUCT"
        key = (teacher, bucket)
        cell = stats[key]
        cell["rows"] += 1
        cell["duplicate_rows"] += int(encoded in seen[key])
        seen[key].add(encoded)
        policy = np.asarray(row["policy"], dtype=float)
        cell["entropy_sum"] += -sum(
            float(value) * math.log(float(value)) for value in policy if value > 0
        )
        cell["one_hot_rows"] += int(np.count_nonzero(policy > 1e-8) == 1)
    return {
        f"{teacher}_{buckets[bucket]}": {
            **(
                {
                    "rows": 0,
                }
                if not cell["rows"]
                else {
                    "rows": cell["rows"],
                    "duplicate_rows": cell["duplicate_rows"],
                    "mean_legal_policy_entropy_nats": cell["entropy_sum"]
                    / cell["rows"],
                    "one_hot_fraction": cell["one_hot_rows"] / cell["rows"],
                }
            ),
        }
        for (teacher, bucket), cell in stats.items()
    }


def _cohorts_from_replay(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    buckets = {">32": ">32", "17–32": "17-32", "≤16": "<=16"}
    return {
        f"{teacher}_{buckets[bucket]}": {
            **(
                {"rows": 0}
                if not cell["rows"]
                else {
                    key: value
                    for key, value in cell.items()
                    if key
                    in {
                        "rows",
                        "duplicate_rows",
                        "mean_legal_policy_entropy_nats",
                        "one_hot_fraction",
                    }
                }
            )
        }
        for teacher, by_bucket in report[
            "subgroups_by_teacher_and_active_stones"
        ].items()
        for bucket, cell in by_bucket.items()
    }


def _game_completion(rows: list[dict[str, Any]], requested: int) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["game_index"])].append(row)
    if sorted(grouped) != list(range(len(grouped))) or len(grouped) > requested:
        raise ValueError("partial_game_index_accounting_mismatch")
    completed: list[int] = []
    capped: list[int] = []
    incomplete: list[int] = []
    for game_index, game_rows in sorted(grouped.items()):
        first = game_rows[0]
        ordered = sorted(game_rows, key=lambda row: int(row["move_index"]))
        expected_length = len(first.get("game_moves", []))
        is_capped = any(row.get("game_capped") is True for row in ordered)
        is_complete = (
            len(ordered) == expected_length
            and [int(row["move_index"]) for row in ordered]
            == list(range(expected_length))
            and all(row.get("game_completed") is True for row in ordered)
        )
        if is_capped:
            capped.append(game_index)
        elif is_complete:
            completed.append(game_index)
        else:
            incomplete.append(game_index)
    return {
        "completed_indices": completed,
        "capped_indices": capped,
        "incomplete_indices": incomplete,
        "unstarted_indices": list(range(len(grouped), requested)),
    }


def _verify_completed_shard(
    summary: dict[str, Any],
) -> tuple[dict[str, Any], set[bytes]]:
    seed = int(summary["seed"])
    path = CORPUS / f"shard-{seed}.jsonl"
    rows = _read_jsonl(path)
    report = validate_seed500_pilot.validate_shards([path])
    loader_rows = train.load_jsonl(
        path, policy_target_mode="default", value_target_mode="default"
    )[0]
    unique_inputs = {rebuild.input_key(row) for row in rows}
    actual = {
        "seed": seed,
        "source": "training" if seed < 542048 else "development",
        "requested_games": 64,
        "completed_games": report["games"],
        "capped_games": report["game_status"]["capped"],
        "incomplete_games": report["game_status"]["incomplete"],
        "rows": report["rows"],
        "loader_accepted_rows": int(loader_rows.shape[0]),
        "canonical_unique_states": report["canonical_unique_states"],
        "canonical_duplicate_rows": report["canonical_duplicate_rows"],
        "production_float32_unique_inputs": report["encoded_input_unique_states"],
        "production_float32_duplicate_rows": report["encoded_input_duplicate_rows"],
        "teacher_and_active_stone_cohorts": _cohorts_from_replay(report),
        "target_mode_counts": _target_modes(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "replay_validation": "passed",
        "production_loader": "accepted",
    }
    event = next(
        event
        for inventory in (
            CORPUS / "execution-inventory.jsonl",
            CORPUS / "continuation-inventory.jsonl",
        )
        if inventory.is_file()
        for event in _read_jsonl(inventory)
        if int(event.get("seed", -1)) == seed
        and event.get("event") == "finished"
        and event.get("exit_status") == 0
    )
    actual["elapsed_seconds"] = event["elapsed_seconds"]
    log_path = CORPUS / f"shard-{seed}.log"
    if (
        not log_path.is_file()
        or event.get("output_bytes") != path.stat().st_size
        or event.get("log_sha256") != hashlib.sha256(log_path.read_bytes()).hexdigest()
    ):
        raise ValueError(f"completed_shard_execution_evidence_mismatch:{seed}")
    if not np.isclose(
        actual["elapsed_seconds"], summary["elapsed_seconds"], rtol=0, atol=1e-9
    ):
        raise ValueError(f"completed_shard_elapsed_mismatch:{seed}")
    for key, value in actual.items():
        if key == "elapsed_seconds":
            continue
        if canonical_json(summary.get(key)) != canonical_json(value):
            raise ValueError(f"completed_shard_accounting_mismatch:{seed}:{key}")
    if report["rows"] != report["loader_accepted_rows"]:
        raise ValueError(f"completed_shard_loader_reconciliation:{seed}")
    return actual, report, unique_inputs


def verify_partial() -> dict[str, str]:
    accounting_path = CORPUS / "accounting.json"
    accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
    interruption = json.loads(
        (CORPUS / "execution-interruption.json").read_text(encoding="utf-8")
    )
    status = json.loads((CORPUS / "partial-status.json").read_text(encoding="utf-8"))
    events = verify_inventory()
    if not accounting["completed_shards"]:
        raise ValueError("partial_accounting_has_no_completed_shards")
    completed_shard_reports = []
    seen_inputs: set[bytes] = set()
    for shard_summary in accounting["completed_shards"]:
        actual, report, shard_inputs = _verify_completed_shard(shard_summary)
        completed_shard_reports.append((actual, report, shard_summary))
        new_inputs = len(shard_inputs - seen_inputs)
        seen_inputs.update(shard_inputs)
        growth = shard_summary["diversity_growth"]
        if growth != {
            "new_unique_inputs": new_inputs,
            "cumulative_unique_inputs": len(seen_inputs),
        }:
            raise ValueError(
                f"completed_shard_diversity_growth_mismatch:{shard_summary['seed']}"
            )
    staged = CORPUS / interruption["staging_output"]["path"].removeprefix(
        "docs/data/seed542-selfplay-4096/"
    )
    if (
        hashlib.sha256(staged.read_bytes()).hexdigest()
        != interruption["staging_output"]["sha256"]
    ):
        raise ValueError("partial_staging_hash_mismatch")
    rows = _read_jsonl(staged)
    all_worker_staging = sorted(CORPUS.glob("azlite-self-play-shards-*/worker_0.jsonl"))
    worker_seeds = {
        int(row["game_seed"])
        for worker_path in all_worker_staging
        for row in _read_jsonl(worker_path)
    }
    if worker_seeds != {int(interruption["seed"])} or staged not in all_worker_staging:
        raise ValueError("staged_output_seed_inventory_mismatch")
    completion = _game_completion(rows, int(interruption["requested_games"]))
    complete_indices = set(completion["completed_indices"])
    complete_rows = [row for row in rows if int(row["game_index"]) in complete_indices]
    partial_report = validate_seed500_pilot._validate_and_account(complete_rows)
    x, _, _ = train.load_jsonl(
        staged, policy_target_mode="default", value_target_mode="default"
    )
    partial_expected = accounting["partial_shards"][0]
    canonical_keys = {rebuild.canonical_key(row) for row in rows}
    input_keys = {rebuild.input_key(row) for row in rows}
    if (
        len(rows) != partial_expected["staged_rows"]
        or len(canonical_keys) != partial_expected["canonical_unique_staged_inputs"]
        or len(input_keys) != partial_expected["float32_unique_staged_inputs"]
    ):
        raise ValueError("partial_staged_row_accounting_mismatch")
    staged_cohorts = _cohort_stats(rows)
    published_staged_cohorts = {
        cohort: {
            key: value
            for key, value in stats.items()
            if key
            in partial_expected["teacher_and_active_stone_cohorts_of_staged_rows"][
                cohort
            ]
        }
        for cohort, stats in staged_cohorts.items()
    }
    staged_actual = {
        "seed": int(interruption["seed"]),
        "source": "training" if int(interruption["seed"]) < 542048 else "development",
        "requested_games": int(interruption["requested_games"]),
        "completed_games_in_staging": len(completion["completed_indices"]),
        "capped_games_in_staging": len(completion["capped_indices"]),
        "incomplete_games_in_staging": len(completion["incomplete_indices"]),
        "not_started_games": len(completion["unstarted_indices"]),
        "staged_rows": len(rows),
        "loader_accepted_staged_rows": int(x.shape[0]),
        "canonical_unique_staged_inputs": len(canonical_keys),
        "canonical_duplicate_staged_rows": len(rows) - len(canonical_keys),
        "float32_unique_staged_inputs": len(input_keys),
        "float32_duplicate_staged_rows": len(rows) - len(input_keys),
        "complete_game_rows_replayed": len(complete_rows),
        "canonical_unique_states_of_complete_games": partial_report[
            "canonical_unique_states"
        ],
        "float32_unique_inputs_of_complete_games": partial_report[
            "encoded_input_unique_states"
        ],
        "teacher_and_active_stone_cohorts_of_staged_rows": published_staged_cohorts,
        "target_mode_counts_of_staged_rows": {
            f"policy_{policy}_actual_{actual}_value_{value}": count
            for (policy, actual, value), count in Counter(
                (
                    str(row.get("policy_target_mode", "missing")),
                    str(row.get("policy_target_actual_mode", "missing")),
                    str(row.get("value_target_mode", "missing")),
                )
                for row in rows
            ).items()
        },
    }
    for key, value in staged_actual.items():
        if canonical_json(partial_expected.get(key)) != canonical_json(value):
            raise ValueError(f"partial_shard_accounting_mismatch:{key}")
    for key in ("completed_games", "incomplete_game_indices", "unstarted_game_indices"):
        derived = {
            "completed_games": completion["completed_indices"],
            "incomplete_game_indices": completion["incomplete_indices"],
            "unstarted_game_indices": completion["unstarted_indices"],
        }[key]
        if interruption["staging_output"].get(key) != (
            len(derived) if key == "completed_games" else derived
        ):
            raise ValueError(f"interruption_record_game_status_mismatch:{key}")
    if (
        interruption["staging_output"].get("rows") != len(rows)
        or interruption["staging_output"].get("bytes") != staged.stat().st_size
        or interruption["staging_output"].get("production_loader_accepted_rows")
        != int(x.shape[0])
        or {row.get("game_seed") for row in rows} != {int(interruption["seed"])}
    ):
        raise ValueError("interruption_record_staged_data_mismatch")
    if status["status"] != "incomplete; training unauthorized":
        raise ValueError("partial_status_mismatch")
    finished_seeds = {int(item["seed"]) for item, _, _ in completed_shard_reports}
    interrupted_seeds = {int(interruption["seed"])}
    all_seeds = set(range(542000, 542064))
    if len(interrupted_seeds) != 1 or finished_seeds & interrupted_seeds:
        raise ValueError("completed_interrupted_seed_overlap")
    unattempted_seeds = sorted(all_seeds - finished_seeds - interrupted_seeds)
    if accounting["unattempted_seeds"] != unattempted_seeds:
        raise ValueError("unattempted_seed_set_mismatch")
    attempted_seeds = {
        int(event["seed"])
        for event in events
        if event.get("event") in {"started", "finished", "interrupted"}
    }
    if attempted_seeds != finished_seeds | interrupted_seeds:
        raise ValueError("execution_inventory_attempt_set_mismatch")
    for seed in unattempted_seeds:
        if (CORPUS / f"shard-{seed}.jsonl").exists() or (
            CORPUS / f"shard-{seed}.log"
        ).exists():
            raise ValueError(f"unattempted_seed_has_output:{seed}")
    completed_games = sum(
        int(report["games"]) for _, report, _ in completed_shard_reports
    ) + len(completion["completed_indices"])
    capped_games = sum(
        int(report["game_status"]["capped"]) for _, report, _ in completed_shard_reports
    ) + len(completion["capped_indices"])
    incomplete_games = len(completion["incomplete_indices"])
    unstarted_games = len(completion["unstarted_indices"]) + 64 * len(unattempted_seeds)
    requested_games = 64 * (
        len(finished_seeds) + len(interrupted_seeds) + len(unattempted_seeds)
    )
    aggregate = {
        "requested_games": requested_games,
        "completed": completed_games,
        "capped": capped_games,
        "incomplete_or_not_started": incomplete_games + unstarted_games,
    }
    expected_aggregate = {
        "requested_games": accounting["requested_games"],
        **accounting["game_status"],
    }
    if canonical_json(aggregate) != canonical_json(expected_aggregate):
        raise ValueError("aggregate_game_accounting_mismatch")
    expected_shards = {
        "requested": 64,
        "complete": len(finished_seeds),
        "partial": len(interrupted_seeds),
        "not_started": len(unattempted_seeds),
    }
    if canonical_json(accounting["shard_status"]) != canonical_json(expected_shards):
        raise ValueError("aggregate_shard_accounting_mismatch")
    if (
        status["requested_games"] != requested_games
        or status["completed_games"] != completed_games
        or status["capped_games"] != capped_games
        or status["incomplete_or_not_started_games"]
        != incomplete_games + unstarted_games
        or status["shards_completed"] != len(finished_seeds)
        or status["shards_partial"] != len(interrupted_seeds)
        or status["shards_not_started"] != len(unattempted_seeds)
    ):
        raise ValueError("partial_status_aggregate_mismatch")
    for source, seeds in (
        ("training_source", [seed for seed in all_seeds if seed < 542048]),
        ("development_source", [seed for seed in all_seeds if seed >= 542048]),
    ):
        source_completed = sum(
            int(report["games"])
            for _, report, summary in completed_shard_reports
            if int(summary["seed"]) in seeds
        ) + (
            len(completion["completed_indices"])
            if int(interruption["seed"]) in seeds
            else 0
        )
        source_expected = status[source]
        if (
            source_expected["requested_games"] != 64 * len(seeds)
            or source_expected["completed_games"] != source_completed
            or source_expected.get("partial_seed")
            != (
                int(interruption["seed"])
                if int(interruption["seed"]) in seeds
                else None
            )
        ):
            raise ValueError(f"partial_source_accounting_mismatch:{source}")
    completed_status = status["completed_shard"]
    for key, derived in {
        "games_requested": 64,
        "games_complete": sum(
            int(report["games"]) for _, report, _ in completed_shard_reports
        ),
        "games_capped": sum(
            int(report["game_status"]["capped"])
            for _, report, _ in completed_shard_reports
        ),
        "games_incomplete": sum(
            int(report["game_status"]["incomplete"])
            for _, report, _ in completed_shard_reports
        ),
        "rows": sum(int(report["rows"]) for _, report, _ in completed_shard_reports),
        "loader_accepted_rows": sum(
            int(report["loader_accepted_rows"])
            for _, report, _ in completed_shard_reports
        ),
        "canonical_unique_states": sum(
            int(report["canonical_unique_states"])
            for _, report, _ in completed_shard_reports
        ),
        "production_float32_unique_inputs": sum(
            int(report["encoded_input_unique_states"])
            for _, report, _ in completed_shard_reports
        ),
        "sha256": completed_shard_reports[0][0]["sha256"],
        "generation_elapsed_seconds": completed_shard_reports[0][0]["elapsed_seconds"],
        "validation": "64 games replayed; production loader accepted",
    }.items():
        if completed_status.get(key) != derived:
            raise ValueError(f"partial_status_completed_shard_mismatch:{key}")
    if int(interruption["seed"]) != int(status["preserved_partial_attempt"]["seed"]):
        raise ValueError("partial_status_interruption_seed_mismatch")
    partial_status = status["preserved_partial_attempt"]
    for key, derived in {
        "requested_games": interruption["requested_games"],
        "complete_games_in_staging": len(completion["completed_indices"]),
        "capped_games_in_staging": len(completion["capped_indices"]),
        "incomplete_games_in_staging": len(completion["incomplete_indices"]),
        "not_started_games": len(completion["unstarted_indices"]),
        "staged_rows": len(rows),
        "complete_game_rows_replayed": len(complete_rows),
        "complete_game_canonical_unique_states": partial_report[
            "canonical_unique_states"
        ],
        "complete_game_float32_unique_inputs": partial_report[
            "encoded_input_unique_states"
        ],
        "production_loader_accepted_staged_rows": int(x.shape[0]),
        "path": interruption["staging_output"]["path"],
        "bytes": staged.stat().st_size,
        "sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
    }.items():
        if partial_status.get(key) != derived:
            raise ValueError(f"partial_status_staging_mismatch:{key}")
    return {
        "accounting.json": hashlib.sha256(accounting_path.read_bytes()).hexdigest(),
        "partial-status.json": hashlib.sha256(
            (CORPUS / "partial-status.json").read_bytes()
        ).hexdigest(),
        "completed-shard.sha256": completed_shard_reports[-1][1]["corpus_sha256"],
        "partial-staging.sha256": interruption["staging_output"]["sha256"],
    }


def verify() -> dict[str, str]:
    accounting = json.loads((CORPUS / "accounting.json").read_text(encoding="utf-8"))
    if accounting.get("status", "").startswith("partial corpus"):
        return verify_partial()
    verify_inventory()
    expected = rebuild.build_artifacts()
    verified = {}
    for name, rebuilt in expected.items():
        path = CORPUS / name
        archived = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json(archived) != canonical_json(rebuilt):
            raise ValueError(f"archived_evidence_mismatch:{name}")
        verified[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return verified


def main() -> None:
    for name, digest in verify().items():
        print(f"{name} sha256={digest}")


if __name__ == "__main__":
    main()
