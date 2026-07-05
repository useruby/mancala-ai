#!/usr/bin/env python3
"""Select deterministic residual_v3 opening-suite states for iteration-0 preflight.

This preflight is diagnostic and data-selection only. It does not train, promote,
or overwrite ``model-artifact/current``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import arena  # noqa: E402
from ml.alphazero_lite.build_opening_suite import (  # noqa: E402
    deduplicate_openings,
    enumerate_legal_prefixes,
    load_suite_jsonl,
    stratify_openings,
    write_suite_jsonl,
)
from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    DEFAULT_RUNTIME_C_PUCT,
    resolve_budget_cpuct,
)

EXPECTED_CURRENT_WEIGHTS_SHA256 = (
    "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
)
PREFLIGHT_SCHEMA = "azlite_residual_v3_opening_iteration0_preflight_v1"
PREFLIGHT_CLASSIFICATION = "residual_v3_opening_iteration0_preflight"
DEFAULT_OPENING_PLIES = (2, 4, 6)
DEFAULT_SELECTION_LIMIT = 256

SEARCH_SETTINGS = (
    {
        "label": "sim256_default",
        "budget_pair": "256:768",
        "simulations": 256,
        "role_context": ["384:256/current", "768:256/current", "1200:256/current"],
    },
    {
        "label": "sim384_default",
        "budget_pair": "384:256",
        "simulations": 384,
        "role_context": ["384:256/challenger"],
    },
    {
        "label": "sim768_default",
        "budget_pair": "768:256",
        "simulations": 768,
        "role_context": ["768:256/challenger"],
    },
    {
        "label": "sim768_equal_override",
        "budget_pair": "768:768",
        "simulations": 768,
        "role_context": ["768:768/shared"],
    },
    {
        "label": "sim1200_default",
        "budget_pair": "1200:1200",
        "simulations": 1200,
        "role_context": ["1200:1200/shared", "1200:256/challenger"],
    },
)

_WORKER_EVALUATOR: arena.ArtifactEvaluator | None = None
_WORKER_SEARCH_PROFILE: dict[str, Any] | None = None
_WORKER_SEED: int | None = None


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def stable_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def canonical_state_hash(state: dict[str, Any]) -> str:
    payload = {
        "current_player": int(state["current_player"]),
        "player_pits": [int(seed) for seed in state["player_pits"]],
        "opponent_pits": [int(seed) for seed in state["opponent_pits"]],
        "player_store": int(state["player_store"]),
        "opponent_store": int(state["opponent_store"]),
    }
    return sha256_text(canonical_json(payload))


def prefix_hash(prefix_moves: list[int]) -> str:
    return sha256_text(canonical_json([int(move) for move in prefix_moves]))


def phase_bucket(entry: dict[str, Any]) -> str:
    if "phase_bucket" in entry:
        return str(entry["phase_bucket"])
    pit_sum = int(entry.get("pit_sum", 48))
    if pit_sum <= 12:
        return "late"
    if pit_sum <= 24:
        return "mid"
    return "early"


def sort_entry_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(entry.get("ply", 0)),
        tuple(int(move) for move in entry.get("prefix_moves", [])),
        canonical_state_hash(entry["state"]),
    )


def build_promoted_search_profile() -> dict[str, Any]:
    return {
        "schema": "azlite_opening_iteration0_search_profile_v1",
        "default_c_puct": DEFAULT_RUNTIME_C_PUCT,
        "c_puct_overrides": {"768:768": 0.90},
        "root_policy_mode": "deterministic",
        "root_prior_transform": None,
        "tactical_root_bias": 0.0,
        "value_transform": None,
        "tablebase_overlay": False,
        "seed_lottery_promotion": False,
        "model_type": "residual_v3",
    }


def validate_guardrails(*, search_profile: dict[str, Any], model_type: str) -> None:
    if model_type != "residual_v3":
        raise RuntimeError("guardrail violation: model_type must remain residual_v3")
    if search_profile.get("root_policy_mode") != "deterministic":
        raise RuntimeError(
            "guardrail violation: root_policy_mode must remain deterministic"
        )
    if search_profile.get("root_prior_transform") is not None:
        raise RuntimeError("guardrail violation: root_prior_transform must remain null")
    if float(search_profile.get("tactical_root_bias", 0.0)) != 0.0:
        raise RuntimeError("guardrail violation: tactical_root_bias must remain 0.0")
    if search_profile.get("value_transform") is not None:
        raise RuntimeError("guardrail violation: value_transform must remain null")
    if bool(search_profile.get("tablebase_overlay")):
        raise RuntimeError("guardrail violation: tablebase overlay is not allowed")
    if bool(search_profile.get("seed_lottery_promotion")):
        raise RuntimeError("guardrail violation: seed-lottery promotion is not allowed")


def search_profile_hash(search_profile: dict[str, Any]) -> str:
    return sha256_text(canonical_json(search_profile))


def build_generated_suite(
    workdir: Path, *, opening_plies: tuple[int, ...]
) -> list[dict[str, Any]]:
    all_prefixes: list[dict[str, Any]] = []
    for ply in opening_plies:
        all_prefixes.extend(enumerate_legal_prefixes(ply))
    unique, _duplicates, _duplicate_count = deduplicate_openings(all_prefixes)
    suite_entries = sorted(stratify_openings(unique), key=sort_entry_key)
    write_suite_jsonl(
        suite_entries, str(workdir / "inputs" / "opening_suite_canonical.jsonl")
    )
    return suite_entries


def load_input_entries(
    *,
    suite_paths: list[Path],
    workdir: Path,
    max_input_rows: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if suite_paths:
        source_summaries = []
        loaded_entries: list[dict[str, Any]] = []
        for suite_path in suite_paths:
            rows = load_suite_jsonl(str(suite_path))
            source_sha = sha256_file(suite_path)
            source_summaries.append(
                {
                    "path": str(suite_path),
                    "sha256": source_sha,
                    "rows": len(rows),
                }
            )
            for index, row in enumerate(rows):
                enriched = dict(row)
                enriched["source_suite_path"] = str(suite_path)
                enriched["source_suite_sha256"] = source_sha
                enriched["source_row_index"] = index
                loaded_entries.append(enriched)
        entries = sorted(loaded_entries, key=sort_entry_key)
        if max_input_rows is not None:
            entries = entries[: max(0, int(max_input_rows))]
        return entries, source_summaries

    generated_entries = build_generated_suite(
        workdir, opening_plies=DEFAULT_OPENING_PLIES
    )
    generated_path = workdir / "inputs" / "opening_suite_canonical.jsonl"
    source_sha = sha256_file(generated_path)
    entries = []
    for index, row in enumerate(generated_entries):
        enriched = dict(row)
        enriched["source_suite_path"] = str(generated_path)
        enriched["source_suite_sha256"] = source_sha
        enriched["source_row_index"] = index
        entries.append(enriched)
    if max_input_rows is not None:
        entries = entries[: max(0, int(max_input_rows))]
    return entries, [
        {
            "path": str(generated_path),
            "sha256": source_sha,
            "rows": len(generated_entries),
        }
    ]


def deduplicate_by_state(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    duplicate_state_rows = 0
    duplicate_prefix_rows = 0
    prefix_seen: set[str] = set()
    for entry in entries:
        state_key = canonical_state_hash(entry["state"])
        pfx_key = prefix_hash(entry.get("prefix_moves", []))
        if pfx_key in prefix_seen:
            duplicate_prefix_rows += 1
        else:
            prefix_seen.add(pfx_key)
        if state_key in seen:
            duplicate_state_rows += 1
            continue
        seen.add(state_key)
        deduped.append(entry)
    return deduped, {
        "input_rows": len(entries),
        "unique_state_rows": len(deduped),
        "duplicate_state_rows_removed": duplicate_state_rows,
        "duplicate_prefix_rows_observed": duplicate_prefix_rows,
        "unique_prefix_rows": len(prefix_seen),
        "state_fingerprint_sha256": sha256_text(canonical_json(sorted(seen))),
        "prefix_fingerprint_sha256": sha256_text(canonical_json(sorted(prefix_seen))),
    }


def normalize_visits(visits: list[float], legal_moves: list[int]) -> list[float]:
    policy = [0.0] * 6
    if not legal_moves:
        return policy
    total = sum(float(visits[move]) for move in legal_moves)
    if total <= 0.0:
        uniform = 1.0 / len(legal_moves)
        for move in legal_moves:
            policy[move] = uniform
        return policy
    for move in legal_moves:
        policy[move] = float(visits[move]) / total
    return policy


def policy_entropy(policy: list[float], legal_moves: list[int]) -> float:
    import math

    entropy = 0.0
    for move in legal_moves:
        probability = float(policy[move])
        if probability > 0.0:
            entropy -= probability * math.log(probability, 2)
    return entropy


def top_share_and_margin(
    policy: list[float], legal_moves: list[int]
) -> tuple[float, float]:
    if not legal_moves:
        return 0.0, 0.0
    ranked = sorted((float(policy[move]) for move in legal_moves), reverse=True)
    top_share = ranked[0]
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else 1.0
    return top_share, margin


def evaluate_search_setting(
    *,
    evaluator: arena.ArtifactEvaluator,
    state: dict[str, Any],
    setting: dict[str, Any],
    search_profile: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    c_puct = resolve_budget_cpuct(
        schedule=search_profile["c_puct_overrides"],
        challenger_simulations=int(str(setting["budget_pair"]).split(":", 1)[0]),
        current_simulations=int(str(setting["budget_pair"]).split(":", 1)[1]),
        default_c_puct=float(search_profile["default_c_puct"]),
    )
    summary = arena.evaluate_artifact_position(
        evaluator=evaluator,
        state=state,
        simulations=int(setting["simulations"]),
        seed=seed,
        c_puct=c_puct,
        search_options=arena.build_eval_search_options(
            root_policy_mode=str(search_profile["root_policy_mode"]),
            tactical_root_bias=float(search_profile["tactical_root_bias"]),
        ),
    )
    legal_moves = [int(move) for move in summary["legal_moves"]]
    search_policy = normalize_visits(summary["visits"], legal_moves)
    top_share, margin = top_share_and_margin(search_policy, legal_moves)
    return {
        "label": str(setting["label"]),
        "budget_pair": str(setting["budget_pair"]),
        "simulations": int(setting["simulations"]),
        "role_context": list(setting["role_context"]),
        "c_puct": stable_float(c_puct),
        "selected_move": summary["selected_move"],
        "legal_moves": legal_moves,
        "top_share": stable_float(top_share),
        "margin": stable_float(margin),
        "entropy": stable_float(policy_entropy(search_policy, legal_moves)),
        "root_value": stable_float(float(summary["value"])),
        "search_policy": [stable_float(value) for value in search_policy],
    }


def build_selected_candidate_record(
    entry: dict[str, Any], search_rows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    metrics = selection_metrics(search_rows)
    tags = selection_tags(metrics)
    if not metrics["unstable"] and not metrics["weak"]:
        return None
    return {
        "state": entry["state"],
        "state_hash": canonical_state_hash(entry["state"]),
        "prefix_moves": [int(move) for move in entry.get("prefix_moves", [])],
        "prefix_hash": prefix_hash(entry.get("prefix_moves", [])),
        "ply": int(entry.get("ply", len(entry.get("prefix_moves", [])))),
        "ply_label": str(int(entry.get("ply", len(entry.get("prefix_moves", []))))),
        "phase_bucket": phase_bucket(entry),
        "side_to_move": int(
            entry.get("side_to_move", entry["state"]["current_player"])
        ),
        "side_to_move_label": str(
            int(entry.get("side_to_move", entry["state"]["current_player"]))
        ),
        "first_move_family": str(entry.get("first_move_family", "none")),
        "source_suite_path": str(entry["source_suite_path"]),
        "source_suite_sha256": str(entry["source_suite_sha256"]),
        "source_row_index": int(entry["source_row_index"]),
        "selection_metrics": metrics,
        "selection_tags": tags,
        "primary_tag": tags[0],
        "search_results": search_rows,
    }


def _evaluate_entry_with_evaluator(
    entry: dict[str, Any],
    *,
    evaluator: arena.ArtifactEvaluator,
    search_profile: dict[str, Any],
    seed: int,
) -> dict[str, Any] | None:
    search_rows = [
        evaluate_search_setting(
            evaluator=evaluator,
            state=entry["state"],
            setting=setting,
            search_profile=search_profile,
            seed=seed,
        )
        for setting in SEARCH_SETTINGS
    ]
    return build_selected_candidate_record(entry, search_rows)


def _init_worker(
    current_artifact_path: str, search_profile: dict[str, Any], seed: int
) -> None:
    global _WORKER_EVALUATOR, _WORKER_SEARCH_PROFILE, _WORKER_SEED
    _WORKER_EVALUATOR = arena.ArtifactEvaluator(Path(current_artifact_path))
    _WORKER_SEARCH_PROFILE = dict(search_profile)
    _WORKER_SEED = int(seed)


def _evaluate_entry_worker(entry: dict[str, Any]) -> dict[str, Any] | None:
    if (
        _WORKER_EVALUATOR is None
        or _WORKER_SEARCH_PROFILE is None
        or _WORKER_SEED is None
    ):
        raise RuntimeError("preflight worker not initialized")
    return _evaluate_entry_with_evaluator(
        entry,
        evaluator=_WORKER_EVALUATOR,
        search_profile=_WORKER_SEARCH_PROFILE,
        seed=_WORKER_SEED,
    )


def select_candidates(
    *,
    entries: list[dict[str, Any]],
    current_path: Path,
    search_profile: dict[str, Any],
    seed: int,
    workers: int,
) -> list[dict[str, Any]]:
    if workers <= 1:
        evaluator = arena.ArtifactEvaluator(current_path)
        return [
            record
            for record in (
                _evaluate_entry_with_evaluator(
                    entry,
                    evaluator=evaluator,
                    search_profile=search_profile,
                    seed=seed,
                )
                for entry in entries
            )
            if record is not None
        ]

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(current_path), search_profile, int(seed)),
    ) as executor:
        return [
            record
            for record in executor.map(_evaluate_entry_worker, entries)
            if record is not None
        ]


def selection_metrics(search_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = {row["label"]: row for row in search_rows}
    selected_moves = [
        row["selected_move"] for row in search_rows if row["selected_move"] is not None
    ]
    top_shares = [float(row["top_share"]) for row in search_rows]
    margins = [float(row["margin"]) for row in search_rows]
    entropies = [float(row["entropy"]) for row in search_rows]
    disagreement_count = 0
    for idx, left in enumerate(selected_moves):
        for right in selected_moves[idx + 1 :]:
            if left != right:
                disagreement_count += 1
    primary = by_label["sim384_default"]
    eq768 = by_label["sim768_equal_override"]
    asym768 = by_label["sim768_default"]
    move_set_size = len(set(selected_moves))
    equal_budget_flip = int(eq768["selected_move"] != asym768["selected_move"])
    weak_primary = int(
        float(primary["top_share"]) <= 0.55 or float(primary["margin"]) <= 0.10
    )
    weak_schedule_min = int(min(top_shares) <= 0.52 or min(margins) <= 0.08)
    high_entropy = int(max(entropies) >= 1.45)
    unstable = int(move_set_size > 1 or equal_budget_flip == 1)
    weak = int(weak_primary == 1 or weak_schedule_min == 1 or high_entropy == 1)
    score = (
        (100 * unstable)
        + (20 * disagreement_count)
        + (15 * equal_budget_flip)
        + (10 * weak_primary)
        + (5 * weak_schedule_min)
        + (5 * high_entropy)
        + int(round((1.0 - min(top_shares)) * 1000.0))
        + int(round((1.0 - float(primary["top_share"])) * 1000.0))
        + int(round(max(entropies) * 100.0))
    )
    return {
        "selection_score": int(score),
        "move_set_size": move_set_size,
        "move_disagreement_pairs": disagreement_count,
        "equal_budget_override_move_flip": bool(equal_budget_flip),
        "primary_384_top_share": stable_float(primary["top_share"]),
        "primary_384_margin": stable_float(primary["margin"]),
        "equal_768_top_share": stable_float(eq768["top_share"]),
        "equal_768_margin": stable_float(eq768["margin"]),
        "min_top_share": stable_float(min(top_shares)),
        "min_margin": stable_float(min(margins)),
        "max_entropy": stable_float(max(entropies)),
        "unstable": bool(unstable),
        "weak": bool(weak),
    }


def selection_tags(metrics: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if metrics["unstable"]:
        tags.append("unstable_search")
    if metrics["weak"]:
        tags.append("weak_search")
    if metrics["equal_budget_override_move_flip"]:
        tags.append("schedule_sensitive")
    return tags or ["control"]


def selected_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metrics = row["selection_metrics"]
    return (
        -int(metrics["selection_score"]),
        -int(metrics["unstable"]),
        -int(metrics["move_disagreement_pairs"]),
        -int(metrics["equal_budget_override_move_flip"]),
        float(metrics["primary_384_margin"]),
        float(metrics["min_top_share"]),
        -float(metrics["max_entropy"]),
        str(row["state_hash"]),
        tuple(int(move) for move in row["prefix_moves"]),
    )


def distribute(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts[str(record[field])] += 1
    return {key: int(counts[key]) for key in sorted(counts)}


def build_manifest(
    *,
    current_path: Path,
    current_weights_sha256: str,
    source_summaries: list[dict[str, Any]],
    search_profile: dict[str, Any],
    selected_rows_path: Path,
    selected_rows: list[dict[str, Any]],
    selection_limit: int,
    dedup_stats: dict[str, Any],
) -> dict[str, Any]:
    selected_row_hash = sha256_file(selected_rows_path)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "classification": PREFLIGHT_CLASSIFICATION,
        "current_artifact": {
            "path": str(current_path),
            "weights_sha256": current_weights_sha256,
        },
        "input_suites": source_summaries,
        "search_profile": search_profile,
        "search_profile_hash": search_profile_hash(search_profile),
        "selection_criteria": {
            "selection_limit": int(selection_limit),
            "required_behavior": [
                "unstable_search or weak_search under promoted deterministic schedule",
                "no value_transform",
                "no root_prior_transform",
                "no tablebase overlay",
                "no seed-lottery promotion",
            ],
            "unstable_thresholds": {
                "move_set_size_gt": 1,
                "equal_budget_override_move_flip": True,
            },
            "weak_thresholds": {
                "primary_384_top_share_lte": 0.55,
                "primary_384_margin_lte": 0.10,
                "min_top_share_lte": 0.52,
                "min_margin_lte": 0.08,
                "max_entropy_gte": 1.45,
            },
            "ranking": [
                "selection_score desc",
                "unstable desc",
                "move_disagreement_pairs desc",
                "equal_budget_override_move_flip desc",
                "primary_384_margin asc",
                "min_top_share asc",
                "max_entropy desc",
                "state_hash asc",
            ],
        },
        "dedup_fingerprint_statistics": {
            **dedup_stats,
            "selected_rows": len(selected_rows),
            "selected_rows_sha256": selected_row_hash,
            "selected_state_fingerprint_sha256": sha256_text(
                canonical_json(sorted(record["state_hash"] for record in selected_rows))
            ),
            "selected_prefix_fingerprint_sha256": sha256_text(
                canonical_json(
                    sorted(record["prefix_hash"] for record in selected_rows)
                )
            ),
        },
        "selected_distribution": {
            "phase": distribute(selected_rows, "phase_bucket"),
            "seat": distribute(selected_rows, "side_to_move_label"),
            "opening_prefix_ply": distribute(selected_rows, "ply_label"),
            "first_move_family": distribute(selected_rows, "first_move_family"),
            "selection_tags": distribute(selected_rows, "primary_tag"),
        },
        "training_data": {
            "path": str(selected_rows_path),
            "rows": len(selected_rows),
            "sha256": selected_row_hash,
            "format": "jsonl",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Residual_v3 opening-suite iteration-0 deterministic preflight."
    )
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--suite", action="append", default=[])
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--expected-current-sha256",
        default=EXPECTED_CURRENT_WEIGHTS_SHA256,
    )
    parser.add_argument("--selection-limit", type=int, default=DEFAULT_SELECTION_LIMIT)
    parser.add_argument("--max-input-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    current_path = Path(args.current)
    current_weights_path = current_path / "weights.json"
    metadata_path = current_path / "metadata.json"
    current_weights_sha256 = sha256_file(current_weights_path)
    if current_weights_sha256 != str(args.expected_current_sha256):
        raise RuntimeError(
            "current artifact weights hash mismatch: "
            f"expected {args.expected_current_sha256}, got {current_weights_sha256}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_type = str(metadata.get("architecture", {}).get("model_type"))
    search_profile = build_promoted_search_profile()
    validate_guardrails(search_profile=search_profile, model_type=model_type)

    suite_paths = [Path(path) for path in args.suite]
    entries, source_summaries = load_input_entries(
        suite_paths=suite_paths,
        workdir=workdir,
        max_input_rows=args.max_input_rows,
    )
    deduped_entries, dedup_stats = deduplicate_by_state(entries)

    selected_candidates = select_candidates(
        entries=deduped_entries,
        current_path=current_path,
        search_profile=search_profile,
        seed=int(args.seed),
        workers=max(1, int(args.workers)),
    )

    selected_rows = sorted(selected_candidates, key=selected_sort_key)[
        : args.selection_limit
    ]
    for rank, row in enumerate(selected_rows, start=1):
        row["selection_rank"] = rank

    selected_rows_path = workdir / "manifests" / "iteration0_selected_positions.jsonl"
    write_jsonl(selected_rows_path, selected_rows)
    manifest = build_manifest(
        current_path=current_path,
        current_weights_sha256=current_weights_sha256,
        source_summaries=source_summaries,
        search_profile=search_profile,
        selected_rows_path=selected_rows_path,
        selected_rows=selected_rows,
        selection_limit=int(args.selection_limit),
        dedup_stats=dedup_stats,
    )
    manifest_path = workdir / "manifests" / "iteration0_training_manifest.json"
    write_json(manifest_path, manifest)

    print(f"[preflight] manifest={manifest_path}")
    print(f"[preflight] selected_rows={selected_rows_path}")
    print(f"[preflight] selected_count={len(selected_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
