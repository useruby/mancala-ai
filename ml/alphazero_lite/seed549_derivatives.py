"""Build and independently verify seed549 training/development derivatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed549_corpus as generation
from ml.alphazero_lite.self_play import encode_state
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.verify_selfplay_evaluator_pilot import (
    expected_search_profile,
    verify_shard,
)

ROOT = generation.ROOT
OUT = generation.OUT
ARMS = ("control", "treatment")
BUCKETS = ((">32", 33, 48), ("17-32", 17, 32), ("<=16", 0, 16))


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        + b"\n"
        for row in rows
    )


def canonical_state_hash(state: dict[str, Any]) -> str:
    fields = tuple(
        state["player_pits"]
        + state["opponent_pits"]
        + [state["player_store"], state["opponent_store"], state["current_player"]]
    )
    return sha_bytes(json.dumps(fields, separators=(",", ":")).encode())


def encoded_state_hash(state: dict[str, Any]) -> str:
    encoded = np.asarray(
        encode_state(state, input_encoding="kalah_v3"), dtype=np.float32
    )
    return sha_bytes(encoded.tobytes())


def replay_annotate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["game_index"])].append(row)
    annotations: dict[tuple[int, int], tuple[str, str, int]] = {}
    for game_index, game_rows in grouped.items():
        sequence = sorted(game_rows, key=lambda row: int(row["move_index"]))
        game = KalahGame.from_state(sequence[0]["game_start_state"])
        moves = sequence[0]["game_moves"]
        if len(moves) != len(sequence):
            raise ValueError("derivative_game_row_count_mismatch")
        for row, move in zip(sequence, moves, strict=True):
            state = game.to_state()
            key = (game_index, int(row["move_index"]))
            annotations[key] = (
                canonical_state_hash(state),
                encoded_state_hash(state),
                sum(state["player_pits"] + state["opponent_pits"]),
            )
            if not game.move(game.pit_index(int(move))):
                raise ValueError("derivative_trajectory_replay_failed")
        if not game.over():
            raise ValueError("derivative_game_not_terminal")
    result = []
    for row_number, row in enumerate(rows):
        item = dict(row)
        canonical_id, encoded_id, stones = annotations[
            (int(row["game_index"]), int(row["move_index"]))
        ]
        item["_canonical_sha256"] = canonical_id
        item["_float32_sha256"] = encoded_id
        item["_active_stones"] = stones
        item["_source_row_number"] = row_number
        result.append(item)
    return result


def source_row_identity(arm: str, shard: int, row: dict[str, Any]) -> dict[str, Any]:
    source_path = (OUT / arm / f"shard-{shard:03d}.jsonl").as_posix()
    row_number = int(row["_source_row_number"])
    row_id = sha_bytes(
        f"seed549:{arm}:{shard}:{row['game_index']}:{row['move_index']}".encode()
    )
    return {
        "row_id": row_id,
        "source": {
            "arm": arm,
            "shard": shard,
            "game_index": int(row["game_index"]),
            "move_index": int(row["move_index"]),
            "row_number": row_number,
            "path": source_path,
        },
        "canonical_state_sha256": row["_canonical_sha256"],
        "production_float32_sha256": row["_float32_sha256"],
        "active_stones": row["_active_stones"],
        "policy_target": row["policy"],
        "value_target": row["value"],
    }


def build_schedule(
    row_ids: list[str], *, seed: int = generation.BASE_SEED
) -> dict[str, Any]:
    if not row_ids:
        raise ValueError("empty_training_rows")
    total = generation.EXPOSURES
    order: list[int] = []
    passes = []
    while len(order) < total:
        pass_index = len(passes)
        permutation = (
            np.random.default_rng(seed + pass_index).permutation(len(row_ids)).tolist()
        )
        start = len(order)
        take = min(len(permutation), total - start)
        order.extend(permutation[:take])
        passes.append(
            {
                "pass_index": pass_index,
                "seed": seed + pass_index,
                "start_exposure": start,
                "end_exposure_exclusive": len(order),
                "source_occurrences": len(row_ids),
                "without_replacement": True,
            }
        )
    batches = []
    cursor = 0
    batch_sizes = generation.BATCHES * 4
    if len(batch_sizes) != generation.UPDATES or sum(batch_sizes) != total:
        raise ValueError("registered_training_batch_accounting_invalid")
    for update, size in enumerate(batch_sizes, start=1):
        end = cursor + size
        batches.append(
            {
                "update": update,
                "start_exposure": cursor,
                "end_exposure_exclusive": end,
                "row_occurrence_indices": order[cursor:end],
            }
        )
        cursor = end
    return {
        "updates": generation.UPDATES,
        "exposures": total,
        "row_occurrence_count": len(row_ids),
        "pass_rule": "numpy default_rng(seed + pass_index).permutation; no replacement within pass",
        "passes": passes,
        "batches": batches,
        "row_id_sha256": sha_bytes("\n".join(row_ids).encode()),
        "exposure_order_sha256": sha_bytes(np.asarray(order, dtype="<u8").tobytes()),
    }


def exclusion_reasons(
    canonical_id: str,
    encoded_id: str,
    training_ids: dict[str, tuple[set[str], set[str]]],
) -> list[str]:
    reasons = []
    for arm in ARMS:
        canonical_set, encoded_set = training_ids[arm]
        if canonical_id in canonical_set:
            reasons.append(f"canonical_overlap:{arm}")
        if encoded_id in encoded_set:
            reasons.append(f"production_float32_overlap:{arm}")
    return reasons


def reconcile_product(name: str, expected: bytes, actual: bytes) -> None:
    if actual != expected:
        raise ValueError(f"published_derivative_mismatch:{name}")


def validate_attempt_command(
    root: Path, event: dict[str, Any], registration_hash: str
) -> None:
    if event.get("registration_sha256") != registration_hash:
        raise ValueError("attempt_registration_mismatch")
    expected = generation.make_command(
        root, event["arm"], int(event["shard"]), Path(event["output"])
    )
    if event.get("argv") != expected:
        raise ValueError("attempt_command_mismatch")


def _all_derivative_products(root: Path) -> dict[str, bytes]:
    out = root / OUT
    registration = json.loads((out / "registration.json").read_text())
    profile_registration = {
        **registration,
        "generation": registration["generation"]["pilot_settings"],
    }
    runtime_policy = json.loads(
        (root / "model-artifact/current/search_policy.json").read_text()
    )
    profile = expected_search_profile(profile_registration, runtime_policy)
    products: dict[str, bytes] = {}
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    arm_reports: dict[str, list[dict[str, Any]]] = {}
    identity_rows: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        all_rows = []
        reports = []
        shard_manifest = []
        for shard in range(generation.SHARDS):
            relative = OUT / arm / f"shard-{shard:03d}.jsonl"
            path = root / relative
            report = verify_shard(
                path,
                arm=f"{arm}/{shard}",
                expected_seed=generation.BASE_SEED + shard,
                expected_game_count=generation.GAMES_PER_SHARD,
                expected_profile=profile,
            )
            if report["games"] != {"completed": 64, "capped": 0, "incomplete": 0}:
                raise ValueError(f"shard_game_accounting_mismatch:{arm}:{shard}")
            rows = replay_annotate(generation.read_jsonl(path))
            reports.append(report)
            all_rows.extend((shard, row) for row in rows)
            shard_manifest.append(
                {
                    "shard": shard,
                    "seed": generation.BASE_SEED + shard,
                    "partition": "training"
                    if shard in generation.TRAIN_SHARDS
                    else "development",
                    "path": relative.as_posix(),
                    "sha256": sha_file(path),
                    "rows": len(rows),
                    "games": report["games"],
                }
            )
        arm_rows[arm] = all_rows
        arm_reports[arm] = reports
        training = [
            source_row_identity(arm, shard, row)
            for shard, row in all_rows
            if shard in generation.TRAIN_SHARDS
        ]
        development = [
            source_row_identity(arm, shard, row)
            for shard, row in all_rows
            if shard in {6, 7}
        ]
        identity_rows[arm] = training + development
        for part, part_rows in (("training", training), ("development", development)):
            products[f"{arm}-{part}-row-identities.jsonl"] = jsonl_bytes(part_rows)
        products[f"{arm}-source-manifest.json"] = json_bytes(
            {
                "schema": "seed549-arm-source-manifest-v1",
                "arm": arm,
                "games": 512,
                "training_games": 384,
                "development_games": 128,
                "training_rows": len(training),
                "development_rows": len(development),
                "shards": shard_manifest,
                "training_identity_sha256": sha_bytes(jsonl_bytes(training)),
                "development_identity_sha256": sha_bytes(jsonl_bytes(development)),
            }
        )

    training_ids: dict[str, tuple[set[str], set[str]]] = {}
    for arm in ARMS:
        source_rows = [
            row for shard, row in arm_rows[arm] if shard in generation.TRAIN_SHARDS
        ]
        training_ids[arm] = (
            {row["_canonical_sha256"] for row in source_rows},
            {row["_float32_sha256"] for row in source_rows},
        )
    common_rows = []
    cohort = {
        arm: {bucket[0]: {"source_rows": 0, "included_rows": 0} for bucket in BUCKETS}
        for arm in ARMS
    }
    for arm in ARMS:
        for shard, row in arm_rows[arm]:
            if shard not in {6, 7}:
                continue
            item = source_row_identity(arm, shard, row)
            canonical_id = item["canonical_state_sha256"]
            float_id = item["production_float32_sha256"]
            reasons = exclusion_reasons(canonical_id, float_id, training_ids)
            stones = int(item["active_stones"])
            bucket = next(name for name, low, high in BUCKETS if low <= stones <= high)
            cohort[arm][bucket]["source_rows"] += 1
            if not reasons:
                cohort[arm][bucket]["included_rows"] += 1
            common_rows.append(
                {
                    **item,
                    "source_evaluator": arm,
                    "development_shard": shard,
                    "included": not reasons,
                    "exclusion_reasons": reasons,
                    "stone_bucket": bucket,
                }
            )
    products["shared-development-membership.jsonl"] = jsonl_bytes(common_rows)

    schedules = {}
    for arm in ARMS:
        training_identity_rows = [
            json.loads(line)
            for line in products[f"{arm}-training-row-identities.jsonl"]
            .decode()
            .splitlines()
        ]
        schedule = build_schedule([row["row_id"] for row in training_identity_rows])
        schedules[arm] = schedule
        products[f"{arm}-training-schedule.json"] = json_bytes(schedule)

    # Existing pilot and evaluation-suite overlaps are reported descriptively.
    pilot_results = json.loads(
        (root / "docs/data/selfplay-evaluator-pilot/results.json").read_text()
    )
    pilot_ids = {
        kind: set().union(
            *(
                set(pilot_results["arms"][arm]["consumed_state_identities"][kind])
                for arm in ARMS
            )
        )
        for kind in ("canonical", "production_float32")
    }
    evaluation_roots = {}
    for name in (
        "seed511-fixed-composition-screen-v4",
        "seed512-fixed-composition-evaluation",
    ):
        opening_path = root / "docs/data" / name / "openings.jsonl"
        states = [
            json.loads(line)["state"]
            for line in opening_path.read_text().splitlines()
            if line.strip()
        ]
        evaluation_roots[name] = {
            "canonical": {canonical_state_hash(state) for state in states},
            "production_float32": {encoded_state_hash(state) for state in states},
        }

    elapsed = {
        arm: sum(
            float(event.get("elapsed_seconds", 0))
            for event in generation.read_jsonl(out / "attempts.jsonl")
            if event.get("arm") == arm
            and event.get("event") == "finished"
            and event.get("exit_status") == 0
        )
        for arm in ARMS
    }
    results = {
        "schema": "seed549-descriptive-results-v1",
        "status": "complete",
        "interpretation": "exploratory generation characteristics and valid input preparation only; concentration or entropy differences are not evidence of improved quality or stronger play",
        "seed461_overlap": "unresolved; recovery outside this task",
        "arms": {},
        "common_development": {
            "source_rows": len(common_rows),
            "included_rows": sum(row["included"] for row in common_rows),
            "excluded_rows": sum(not row["included"] for row in common_rows),
            "cohort_sizes_by_source_evaluator_and_stone_bucket": cohort,
            "known_pilot_overlap": {},
            "known_evaluation_root_overlap": {},
        },
    }
    for arm in ARMS:
        reports = arm_reports[arm]
        rows = [row for _, row in arm_rows[arm]]
        canonical = {row["_canonical_sha256"] for row in rows}
        encoded = {row["_float32_sha256"] for row in rows}
        games = []
        for shard, report in enumerate(reports):
            lengths = report["game_lengths"]["per_completed_game"]
            raw = generation.read_jsonl(root / OUT / arm / f"shard-{shard:03d}.jsonl")
            for game_index, length in enumerate(lengths):
                game_rows = [row for row in raw if int(row["game_index"]) == game_index]
                unique_high = len(
                    {
                        row["_canonical_sha256"]
                        for row in replay_annotate(game_rows)
                        if row["_active_stones"] > 32
                    }
                )
                games.append(
                    {
                        "shard": shard,
                        "game_index": game_index,
                        "length": length,
                        "unique_over_32_stone_identities": unique_high,
                    }
                )
        high_unique = len(
            {row["_canonical_sha256"] for row in rows if row["_active_stones"] > 32}
        )
        attempt_rows = [
            event
            for event in generation.read_jsonl(out / "attempts.jsonl")
            if event.get("arm") == arm
            and event.get("event") == "finished"
            and event.get("exit_status") == 0
        ]
        memory = max(
            (
                int(event.get("peak_sampled_process_tree_rss_bytes", 0))
                for event in attempt_rows
            ),
            default=0,
        )
        bucket_rows: dict[str, list[dict[str, Any]]] = {
            name: [] for name, _, _ in BUCKETS
        }
        for row in rows:
            bucket = next(
                name
                for name, low, high in BUCKETS
                if low <= row["_active_stones"] <= high
            )
            bucket_rows[bucket].append(row)
        exact = len(bucket_rows["<=16"])
        bucket_diagnostics = {}
        for name, bucket_content in bucket_rows.items():
            targets: dict[str, list[tuple[float, ...]]] = defaultdict(list)
            for row in bucket_content:
                targets[row["_canonical_sha256"]].append(
                    tuple(float(x) for x in row["policy"])
                )
            duplicate_groups = [group for group in targets.values() if len(group) > 1]
            entropies = [
                -sum(
                    float(value) * math.log(float(value))
                    for value in row["policy"]
                    if value > 0
                )
                for row in bucket_content
            ]
            bucket_diagnostics[name] = {
                "rows": len(bucket_content),
                "canonical_unique": len(
                    {row["_canonical_sha256"] for row in bucket_content}
                ),
                "production_float32_unique": len(
                    {row["_float32_sha256"] for row in bucket_content}
                ),
                "mean_target_entropy_nats": float(np.mean(entropies))
                if entropies
                else 0.0,
                "duplicate_target_groups": len(duplicate_groups),
                "duplicate_target_disagreement_states": sum(
                    len(set(group)) > 1 for group in duplicate_groups
                ),
                "one_hot_rows": sum(
                    sum(value > 0 for value in row["policy"]) == 1
                    for row in bucket_content
                ),
            }
        results["arms"][arm] = {
            "completed_games": len(games),
            "accepted_rows": len(rows),
            "game_lengths": {
                "mean": float(np.mean([g["length"] for g in games])),
                "min": min(g["length"] for g in games),
                "max": max(g["length"] for g in games),
                "per_game": games,
            },
            "elapsed_seconds": elapsed[arm],
            "games_per_second": len(games) / elapsed[arm],
            "peak_sampled_memory_bytes": memory,
            "memory_limit_bytes": registration["resource_scope"]["memory_max_bytes"],
            "exact_root_rows": exact,
            "puct_rows": len(rows) - exact,
            "canonical_unique": len(canonical),
            "production_float32_unique": len(encoded),
            "unique_over_32_stone_identities": high_unique,
            "unique_over_32_stone_identities_per_game": high_unique / len(games),
            "unique_over_32_stone_identities_per_generation_second": high_unique
            / elapsed[arm],
            "stone_bucket_diagnostics": bucket_diagnostics,
            "attempts": [
                {
                    "shard": event["shard"],
                    "elapsed_seconds": event["elapsed_seconds"],
                    "peak_sampled_process_tree_rss_bytes": event[
                        "peak_sampled_process_tree_rss_bytes"
                    ],
                    "output_sha256": event["output_sha256"],
                }
                for event in attempt_rows
            ],
        }
        dev_rows = [row for row in common_rows if row["source_evaluator"] == arm]
        results["common_development"]["known_pilot_overlap"][arm] = {
            "canonical_rows": sum(
                row["canonical_state_sha256"] in pilot_ids["canonical"]
                for row in dev_rows
            ),
            "float32_rows": sum(
                row["production_float32_sha256"] in pilot_ids["production_float32"]
                for row in dev_rows
            ),
        }
        results["common_development"]["known_evaluation_root_overlap"][arm] = {
            name: {
                "canonical_rows": sum(
                    row["canonical_state_sha256"] in identities["canonical"]
                    for row in dev_rows
                ),
                "float32_rows": sum(
                    row["production_float32_sha256"] in identities["production_float32"]
                    for row in dev_rows
                ),
                "scope": "registered evaluation opening roots only",
            }
            for name, identities in evaluation_roots.items()
        }

    products["results.json"] = json_bytes(results)
    return products


def build(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    generation.preflight(root)
    out = root / OUT
    products = _all_derivative_products(root)
    for name, payload in products.items():
        path = out / name
        if path.exists():
            raise FileExistsError(f"refuse_derivative_overwrite:{name}")
        path.write_bytes(payload)
    receipt = {
        "schema": "seed549-post-generation-receipt-v1",
        "generation_registration_sha256": sha_file(out / "registration.json"),
        "generation_correction_sha256": sha_file(out / "execution-correction.json"),
        "derivative_algorithm_sha256": sha_file(Path(__file__)),
        "derivative_test_sha256": sha_file(
            Path(__file__).with_name("test_seed549_derivatives.py")
        ),
        "products": {
            name: sha_bytes(payload) for name, payload in sorted(products.items())
        },
        "dependencies": json.loads((out / "registration.json").read_text())["runtime"][
            "dependencies"
        ],
        "derivatives_complete": True,
        "training_executed": False,
        "arena_evaluation_executed": False,
    }
    receipt_path = out / "post-generation-receipt.json"
    if receipt_path.exists():
        raise FileExistsError("refuse_post_generation_receipt_overwrite")
    receipt_path.write_bytes(json_bytes(receipt))
    return receipt


def verify(root: Path = ROOT) -> dict[str, Any]:
    """Read-only end-to-end replay, loader, attempt, derivative, and receipt check."""
    root = root.resolve()
    generation.preflight(root)
    out = root / OUT
    receipt_path = out / "post-generation-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    correction_paths = sorted(out.glob("post-generation-verification-correction*.json"))
    if receipt["generation_registration_sha256"] != sha_file(out / "registration.json"):
        raise ValueError("post_generation_registration_mismatch")
    if receipt["generation_correction_sha256"] != sha_file(
        out / "execution-correction.json"
    ):
        raise ValueError("post_generation_correction_mismatch")
    if correction_paths:
        known_hashes = {sha_file(receipt_path)}
        ordered_corrections = sorted(
            correction_paths, key=lambda path: path.stat().st_mtime_ns
        )
        last_correction = None
        for correction_path in ordered_corrections:
            correction = json.loads(correction_path.read_text())
            parent_hash = correction.get("parent_receipt_sha256") or correction.get(
                "parent_correction_sha256"
            )
            if parent_hash not in known_hashes:
                raise ValueError(
                    "post_generation_verification_correction_parent_mismatch"
                )
            last_correction = correction
            known_hashes.add(sha_file(correction_path))
        if last_correction["corrected_algorithm_sha256"] != sha_file(Path(__file__)):
            raise ValueError("corrected_derivative_algorithm_changed")
        if last_correction["corrected_test_sha256"] != sha_file(
            Path(__file__).with_name("test_seed549_derivatives.py")
        ):
            raise ValueError("corrected_derivative_tests_changed")
    else:
        if receipt["derivative_algorithm_sha256"] != sha_file(Path(__file__)):
            raise ValueError("derivative_algorithm_changed")
        if receipt["derivative_test_sha256"] != sha_file(
            Path(__file__).with_name("test_seed549_derivatives.py")
        ):
            raise ValueError("derivative_tests_changed")
    paths = [out / name for name in receipt["products"]] + [receipt_path]
    paths += [
        out / "registration.json",
        out / "execution-freeze.json",
        out / "execution-correction.json",
        out / "execution-freeze-correction.json",
        out / "attempts.jsonl",
        root / "docs/data/selfplay-evaluator-pilot/registration.json",
        root / "model-artifact/current/search_policy.json",
    ]
    paths.extend(correction_paths)
    for arm in ARMS:
        for shard in range(generation.SHARDS):
            paths.append(out / arm / f"shard-{shard:03d}.jsonl")
    for event in generation.read_jsonl(out / "attempts.jsonl"):
        if event.get("log"):
            paths.append(root / event["log"])
    paths = list(dict.fromkeys(paths))
    before = {path: (sha_file(path), path.stat().st_mtime_ns) for path in paths}
    expected_products = _all_derivative_products(root)
    if set(expected_products) != set(receipt["products"]):
        raise ValueError("derivative_product_population_mismatch")
    for name, payload in expected_products.items():
        if sha_bytes(payload) != receipt["products"][name]:
            raise ValueError(f"post_generation_product_receipt_mismatch:{name}")
        reconcile_product(name, payload, (out / name).read_bytes())
    _verify_attempts(root)
    after = {path: (sha_file(path), path.stat().st_mtime_ns) for path in paths}
    if before != after:
        raise RuntimeError("evidence_changed_during_read_only_verification")
    return {
        "status": "verified",
        "games_per_arm": 512,
        "accepted_rows": {
            arm: sum(
                json.loads((out / f"{arm}-source-manifest.json").read_text())[key]
                for key in ("training_rows", "development_rows")
            )
            for arm in ARMS
        },
        "products": len(receipt["products"]),
        "evidence_preserved": True,
    }


def bind_verification_correction(root: Path = ROOT) -> dict[str, Any]:
    """Append a read-only verifier correction without rewriting post-gen receipt."""
    root = root.resolve()
    out = root / OUT
    receipt = out / "post-generation-receipt.json"
    existing = sorted(
        out.glob("post-generation-verification-correction*.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    target = (
        out / f"post-generation-verification-correction-{len(existing) + 1:02d}.json"
    )
    parent = existing[-1] if existing else receipt
    correction = {
        "schema": "seed549-post-generation-verification-correction-v1",
        ("parent_correction_sha256" if existing else "parent_receipt_sha256"): sha_file(
            parent
        ),
        "reason": "attempt verifier recognizes the explicitly bound zero-output CLI recovery; raw corpus and derivative bytes are unchanged",
        "failed_attempt_log_sha256": json.loads(
            (out / "execution-correction.json").read_text()
        )["failed_attempt"]["log_sha256"],
        "corrected_algorithm_sha256": sha_file(Path(__file__)),
        "corrected_test_sha256": sha_file(
            Path(__file__).with_name("test_seed549_derivatives.py")
        ),
    }
    target.write_bytes(json_bytes(correction))
    return correction


def _verify_attempts(root: Path) -> None:
    out = root / OUT
    registration_hash = sha_file(out / "registration.json")
    rows = generation.read_jsonl(out / "attempts.jsonl")
    per_shard: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        arm, shard = row.get("arm"), row.get("shard")
        if arm not in ARMS or shard not in range(generation.SHARDS):
            raise ValueError("attempt_identity_out_of_range")
        per_shard[(arm, shard)].append(row)
    for (arm, shard), events in per_shard.items():
        event_names = [event.get("event") for event in events]
        expected_names = (
            [
                "started",
                "finished",
                "started",
                "finished",
                "published",
            ]
            if (arm, shard) == ("control", 0)
            else ["started", "finished", "published"]
        )
        if event_names != expected_names:
            raise ValueError(f"attempt_event_accounting_mismatch:{arm}:{shard}")
        for event in events:
            if event.get("event") in {"started", "finished"}:
                validate_attempt_command(root, event, registration_hash)
            if event.get("event") == "finished":
                log = root / event["log"]
                if not log.is_file() or sha_file(log) != event.get("log_sha256"):
                    raise ValueError("attempt_log_receipt_mismatch")
                if event.get("exit_status") == 0 and event.get("output_bytes", 0) <= 0:
                    raise ValueError("successful_attempt_has_empty_output")
                if (
                    event.get("peak_sampled_process_tree_rss_bytes", 0)
                    > generation.pilot.RESOURCE_SCOPE["memory_max_bytes"]
                ):
                    raise ValueError("sampled_memory_exceeded_registered_limit")
        successful = [
            event
            for event in events
            if event.get("event") == "finished" and event.get("exit_status") == 0
        ]
        if len(successful) != 1:
            raise ValueError(f"successful_attempt_count_mismatch:{arm}:{shard}")
        receipt = successful[0]
        if (arm, shard) == ("control", 0):
            failed = events[1]
            if failed.get("exit_status") != 2 or failed.get("output_bytes") != 0:
                raise ValueError("zero_output_recovery_precondition_mismatch")
            correction = out / "execution-correction.json"
            correction_payload = json.loads(correction.read_text())
            if (
                correction_payload.get("failed_attempt", {}).get("log_sha256")
                != failed.get("log_sha256")
                or correction_payload.get("failed_attempt", {}).get("exit_status")
                != failed.get("exit_status")
                or correction_payload.get("failed_attempt", {}).get("output_bytes") != 0
            ):
                raise ValueError("retry_authorization_correction_mismatch")
        published = [event for event in events if event.get("event") == "published"]
        final = out / arm / f"shard-{shard:03d}.jsonl"
        expected_published = (OUT / arm / f"shard-{shard:03d}.jsonl").as_posix()
        if (
            len(published) != 1
            or published[0].get("output") != expected_published
            or not final.is_file()
            or sha_file(final) != published[0].get("output_sha256")
        ):
            raise ValueError("published_shard_receipt_mismatch")
        if receipt.get("output_sha256") != sha_file(final):
            raise ValueError("finished_shard_hash_mismatch")
    if set(per_shard) != {
        (arm, shard) for arm in ARMS for shard in range(generation.SHARDS)
    }:
        raise ValueError("attempt_shard_population_mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("build", "verify", "bind-verification-correction")
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "build":
        result = build(args.root)
    elif args.action == "bind-verification-correction":
        result = bind_verification_correction(args.root)
    else:
        result = verify(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
