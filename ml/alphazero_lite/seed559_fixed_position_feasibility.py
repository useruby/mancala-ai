"""Seed559 fixed-position feasibility cohort selection and execution."""

from __future__ import annotations

import json
import math
import os
import random
import statistics
from collections import Counter
from pathlib import Path

from ml.alphazero_lite import arena
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed557_retrospective import (
    ARCHIVE,
    COMMIT,
    DATA as SEED557_DATA,
    authenticate_runtime,
    canonical_hash,
    git_blob,
    load_authenticated_archive,
    sha256,
)
from ml.alphazero_lite.native_exact_root_tablebase import NativeExactRootTablebase

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_DATA = ROOT / "docs/data/seed559-fixed-position-feasibility/run-0003"
P_ARTIFACT = (
    ROOT / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts/P-E4"
)
INITIALIZER_ARTIFACT = (
    ROOT
    / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts/initializer"
)
EVALUATOR_ARTIFACTS = {
    "P": P_ARTIFACT,
    "initializer": INITIALIZER_ARTIFACT,
}
RUNTIME_FILENAMES = (
    "metadata.json",
    "model.npz",
    "search_policy.json",
    "weights.json",
)
SEED = 559
OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "deterministic",
    "root_temperature": 0.0,
    "tactical_root_bias": 0.0,
}


def route_diagnostic_action(
    active_pit_stones: int, baseline_action: int, exact_action: int
) -> int:
    """Proposed isolated route; serving itself remains untouched at cutoff 16."""
    return exact_action if 16 < active_pit_stones <= 21 else baseline_action


def verify_routing_boundaries() -> dict[str, bool]:
    baseline, exact = 2, 4
    return {
        "at_or_below_16_unchanged": all(
            route_diagnostic_action(n, baseline, exact) == baseline
            for n in range(0, 17)
        ),
        "17_through_21_use_exact": all(
            route_diagnostic_action(n, baseline, exact) == exact for n in range(17, 22)
        ),
        "above_21_unchanged": all(
            route_diagnostic_action(n, baseline, exact) == baseline
            for n in range(22, 49)
        ),
    }


def _wdl(value: int) -> int:
    return (value > 0) - (value < 0)


def evaluator_artifact_hashes(
    artifact_dirs: dict[str, Path], expected: dict[str, dict[str, str]]
) -> dict[str, dict[str, str]]:
    """Authenticate the exact files that will be passed to ArtifactEvaluator."""
    observed: dict[str, dict[str, str]] = {}
    for component, directory in artifact_dirs.items():
        if component not in expected:
            raise ValueError(f"unexpected_evaluator_component:{component}")
        observed[component] = {}
        for filename in RUNTIME_FILENAMES:
            path = directory / filename
            if not path.is_file():
                raise ValueError(f"evaluator_artifact_missing:{path}")
            digest = sha256(path.read_bytes())
            if digest != expected[component].get(filename):
                raise ValueError(f"evaluator_artifact_identity_mismatch:{path}")
            observed[component][filename] = digest
    if set(observed) != set(expected):
        raise ValueError("evaluator_component_set_mismatch")
    return observed


def build_evaluators(
    artifact_dirs: dict[str, Path], expected: dict[str, dict[str, str]]
) -> tuple[dict[str, arena.ArtifactEvaluator], dict[str, dict[str, str]]]:
    """Verify exact evaluator inputs before constructing or invoking them."""
    bound = evaluator_artifact_hashes(artifact_dirs, expected)
    return (
        {name: arena.ArtifactEvaluator(path) for name, path in artifact_dirs.items()},
        bound,
    )


def _source_candidates(openings: list[dict], records: list[dict]) -> list[dict]:
    candidates = []
    for source in records:
        opening_index = source["opening_index"]
        opening = openings[opening_index]
        game_row = source["game_row"]
        game = KalahGame.from_state(opening["state"])
        moves = [int(move) for move in game_row["trajectory"].split(",") if move]
        for ply, move in enumerate(moves):
            if game.over():
                raise ValueError(
                    f"terminal-trajectory:{source['source_path']}:{source['source_line']}"
                )
            legal = [game.pit_index(action) for action in game.possible_moves()]
            if 17 <= sum(game.pits) <= 21 and len(legal) >= 2:
                candidates.append(
                    {
                        "opening_index": opening_index,
                        "game_within_opening": int(game_row["game_within_opening"]),
                        "source_game_index_local": int(game_row["game_index"]),
                        "challenger_player": int(game_row["challenger_player"]),
                        "source_side": "T512"
                        if game.current_player == int(game_row["challenger_player"])
                        else "P",
                        "source_player_to_move": game.current_player,
                        "ply": ply,
                        "state": game.to_state(),
                        "state_hash": canonical_hash(game.to_state()),
                        "active_pit_stones": sum(game.pits),
                        "legal_absolute_actions": legal,
                        "opening_state_hash": opening["state_hash"],
                        "opening_prefix_moves": opening["prefix_moves"],
                        "trajectory_prefix_absolute": moves[:ply],
                        "source_path": source["source_path"],
                        "source_line": source["source_line"],
                        "source_record_sha256": source["source_record_sha256"],
                        "source_batch_sha256": source["batch_sha256"],
                        "decision_absolute_action": move,
                    }
                )
                break
            if move not in legal or not game.move(move):
                raise ValueError(
                    f"illegal-source-trajectory:{source['source_path']}:{source['source_line']}"
                )
    return candidates


def _excluded_hashes() -> tuple[set[str], set[str]]:
    old557 = json.loads((SEED557_DATA / "cohort.json").read_text())["states"]
    old418 = json.loads(
        (
            ROOT / "docs/data/seed418-native-root-handoff/cohort-manifest.json"
        ).read_text()
    )["states"]
    return (
        {canonical_hash(row["state"]) for row in old557},
        {canonical_hash(row["state"]) for row in old418},
    )


def select() -> tuple[list[dict], dict]:
    openings, records, receipt = load_authenticated_archive()
    eligible = _source_candidates(openings, records)
    excluded557, excluded418 = _excluded_hashes()
    count557 = sum(row["state_hash"] in excluded557 for row in eligible)
    count418 = sum(row["state_hash"] in excluded418 for row in eligible)
    remaining = [
        row for row in eligible if row["state_hash"] not in excluded557 | excluded418
    ]
    by_hash: dict[str, dict] = {}
    for row in remaining:
        by_hash.setdefault(row["state_hash"], row)
    unique = list(by_hash.values())
    ranked = sorted(
        unique,
        key=lambda row: (
            sha256(f"seed559:{row['state_hash']}".encode()),
            row["state_hash"],
        ),
    )
    chosen: list[dict] = []
    used_openings: set[int] = set()
    # Alternating source-side picks preserve the preregistered balance whenever possible.
    for side in ("T512", "P") * 256:
        candidate = next(
            (
                row
                for row in ranked
                if row["source_side"] == side
                and row["opening_index"] not in used_openings
                and row not in chosen
            ),
            None,
        )
        if candidate is not None:
            chosen.append(candidate)
            used_openings.add(candidate["opening_index"])
        if len(chosen) == 512:
            break
    chosen.sort(
        key=lambda row: (
            sha256(f"seed559:{row['state_hash']}".encode()),
            row["state_hash"],
        )
    )
    audit = {
        "schema": "seed559-cohort-count-audit-v1",
        "source_commit": COMMIT,
        "archive_receipt_sha256": sha256(
            git_blob(ARCHIVE + "verification-receipt.json")
        ),
        "archive_verified_games": int(receipt["verified_games"]),
        "eligible_first_roots": len(eligible),
        "excluded_by_seed557_state_match": count557,
        "excluded_by_seed418_state_match": count418,
        "excluded_by_both_audits": sum(
            row["state_hash"] in excluded557 & excluded418 for row in eligible
        ),
        "unique_remaining_states": len(unique),
        "distinct_remaining_openings": len({row["opening_index"] for row in unique}),
        "selected_roots": len(chosen),
        "selected_by_source_side": dict(Counter(row["source_side"] for row in chosen)),
        "shortfall": max(0, 512 - len(chosen)),
        "ranking": "SHA256(seed559:<canonical_state_hash>), then canonical_state_hash",
        "balance": "alternating T512/P selection from ranked candidates; skip exhausted side; one root per opening",
        "canonical_hash": "seed557_retrospective.canonical_hash(JSON state, sorted keys, compact separators, SHA256)",
    }
    return chosen, audit


def freeze_and_run(output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"diagnostic output must be new: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    cohort, audit = select()
    cohort_doc = {
        "schema": "seed559-cohort-v1",
        "seed": SEED,
        "selection": audit["ranking"],
        "states": cohort,
    }
    (output / "cohort-manifest.json").write_text(
        json.dumps(cohort_doc, sort_keys=True, indent=2) + "\n"
    )
    (output / "count-audit.json").write_text(
        json.dumps(audit, sort_keys=True, indent=2) + "\n"
    )
    if len(cohort) != 512:
        (output / "decision.json").write_text(
            json.dumps(
                {
                    "decision": "no_decision",
                    "reason": "selection_shortfall",
                    "shortfall": audit["shortfall"],
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(audit, sort_keys=True, indent=2))
        return
    # This freeze is written and closed before policy inference or any oracle query.
    allowed_cpus = sorted(os.sched_getaffinity(0))
    if not allowed_cpus:
        raise RuntimeError("no CPU available for pinned diagnostic")
    pinned_cpu = allowed_cpus[0]
    os.sched_setaffinity(0, {pinned_cpu})
    authenticated = authenticate_runtime()
    evaluated_artifacts = evaluator_artifact_hashes(
        EVALUATOR_ARTIFACTS,
        {component: authenticated[component] for component in EVALUATOR_ARTIFACTS},
    )
    native = ROOT / "model-artifact/runtime/kalah_v1_tablebase"
    tablebase = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
    preregistration = json.loads(git_blob(ARCHIVE + "registration.json"))
    preflight = json.loads(git_blob(ARCHIVE + "runtime-policy-preflight.json"))
    if sha256(native.read_bytes()) != preflight["exact_root_native_probe_sha256"]:
        raise ValueError(f"oracle asset digest mismatch:{native}")
    if (
        sha256(tablebase.read_bytes())
        != preregistration["tablebase_materialized_sha256"]
    ):
        raise ValueError(f"oracle asset digest mismatch:{tablebase}")
    freeze = {
        "schema": "seed559-execution-freeze-v1",
        "selection_audit_sha256": sha256((output / "count-audit.json").read_bytes()),
        "cohort_sha256": sha256((output / "cohort-manifest.json").read_bytes()),
        "authenticated_policies": authenticated,
        "evaluator_artifacts": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "files": evaluated_artifacts[name],
            }
            for name, path in EVALUATOR_ARTIFACTS.items()
        },
        "search": {
            "policy": "seed504 E4 P",
            "value": "seed504 initializer",
            "simulations": 128,
            "c_puct": 1.25,
            "workers": 1,
            "fresh_tree_per_root": True,
            "exact_leaf_solving": False,
            "exact_root_solving": False,
            "search_options": OPTIONS,
        },
        "diagnostic_routing": {
            "extended_cutoff": 21,
            "unchanged_at_or_below": 16,
            "unchanged_above": 21,
            "exact_tie_break": "largest exact final-score margin, then highest legal P prior, then lowest move index",
        },
        "latency_protocol": {
            "cpu_affinity": [pinned_cpu],
            "baseline": "warm end-to-end 128-simulation decision, including P and initializer inference",
            "exact": "warm end-to-end P inference, native exact legal-action query, and margin/prior/index tie-break",
            "native_startup_and_prewarm": "reported separately before per-root timing",
        },
        "oracle": {
            "native_sha256": sha256(native.read_bytes()),
            "tablebase_sha256": sha256(tablebase.read_bytes()),
            "coverage": "all legal actions; exact final-score margins",
        },
        "gate": {
            "n_roots": 512,
            "baseline_wdl_inferior_rate_min": 0.05,
            "baseline_mean_regret_min_seeds": 1,
            "exact_latency_mean_ms_max": 200,
            "exact_latency_p95_ms_max": 350,
            "scope": "a pass supports proposing a separate cutoff-21 arena only",
        },
    }
    (output / "execution-freeze.json").write_text(
        json.dumps(freeze, sort_keys=True, indent=2) + "\n"
    )
    evaluators, _ = build_evaluators(
        EVALUATOR_ARTIFACTS,
        {component: authenticated[component] for component in EVALUATOR_ARTIFACTS},
    )
    p = evaluators["P"]
    initializer = evaluators["initializer"]
    composed = arena.ComposedArtifactEvaluator(
        p, initializer, policy_source="current", value_source="candidate"
    )
    search_records, oracle_records, timing_records = [], [], []
    import time

    startup_started = time.perf_counter_ns()
    oracle_context = NativeExactRootTablebase(native, tablebase)
    native_startup_ms = (time.perf_counter_ns() - startup_started) / 1e6
    with oracle_context as oracle:
        prewarm_started = time.perf_counter_ns()
        oracle.warm()
        native_prewarm_ms = (time.perf_counter_ns() - prewarm_started) / 1e6
        for item in cohort:
            game = KalahGame.from_state(item["state"])
            legal = game.possible_moves()
            exact_start = time.perf_counter_ns()
            priors, _ = p.evaluate(game)
            margins = oracle.root_action_margins(game, game.current_player)
            if set(margins) != set(legal):
                raise ValueError(f"oracle legal-action coverage:{item['state_hash']}")
            exact_move = max(
                legal, key=lambda move: (margins[move], float(priors[move]), -move)
            )
            exact_ms = (time.perf_counter_ns() - exact_start) / 1e6
            oracle_records.append(
                {
                    "state_hash": item["state_hash"],
                    "legal_moves": legal,
                    "action_margins": {
                        str(k): int(v) for k, v in sorted(margins.items())
                    },
                    "coverage_complete": True,
                }
            )
            seed = int(sha256(f"seed559:{item['state_hash']}".encode())[:8], 16) % (
                2**31
            )
            start = time.perf_counter_ns()
            result = arena.evaluate_artifact_position(
                evaluator=composed,
                state=item["state"],
                simulations=128,
                seed=seed,
                c_puct=1.25,
                search_options=OPTIONS,
                exact_root_solve_threshold=None,
                exact_solve_stone_threshold=None,
            )
            search_ms = (time.perf_counter_ns() - start) / 1e6
            search_records.append(
                {
                    "state_hash": item["state_hash"],
                    "seed": seed,
                    "legal_moves": legal,
                    "priors": [float(v) for v in priors],
                    "visits": result["visits"],
                    "child_stats": result["child_stats"],
                    "baseline_action": int(result["selected_move"]),
                    "exact_action": int(exact_move),
                }
            )
            timing_records.append(
                {
                    "state_hash": item["state_hash"],
                    "baseline_warm_decision_ms": search_ms,
                    "exact_warm_decision_ms": exact_ms,
                }
            )
    (output / "oracle-records.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in oracle_records)
    )
    (output / "search-records.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in search_records)
    )
    (output / "timings.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in timing_records)
    )
    (output / "native-startup-prewarm.json").write_text(
        json.dumps(
            {
                "startup_ms": native_startup_ms,
                "prewarm_ms": native_prewarm_ms,
                "warm_end_to_end_exact_mean_ms": statistics.fmean(
                    r["exact_warm_decision_ms"] for r in timing_records
                ),
                "warm_end_to_end_exact_p95_ms": sorted(
                    r["exact_warm_decision_ms"] for r in timing_records
                )[int(0.95 * (len(timing_records) - 1))],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "cohort": audit,
                "policy_searches": len(search_records),
                "oracle_roots": len(oracle_records),
            },
            sort_keys=True,
            indent=2,
        )
    )


def analyze_published_run(data: Path = DEFAULT_ANALYSIS_DATA) -> dict:
    cohort = json.loads((data / "cohort-manifest.json").read_text())["states"]
    searches = {
        row["state_hash"]: row
        for row in map(
            json.loads, (data / "search-records.jsonl").read_text().splitlines()
        )
    }
    oracles = {
        row["state_hash"]: row
        for row in map(
            json.loads, (data / "oracle-records.jsonl").read_text().splitlines()
        )
    }
    timings = {
        row["state_hash"]: row
        for row in map(json.loads, (data / "timings.jsonl").read_text().splitlines())
    }
    rows = []
    for root in cohort:
        oracle = oracles[root["state_hash"]]
        margins = {int(k): int(v) for k, v in oracle["action_margins"].items()}
        best = max(margins.values())
        baseline = searches[root["state_hash"]]["baseline_action"]
        exact = searches[root["state_hash"]]["exact_action"]
        rows.append(
            {
                "state_hash": root["state_hash"],
                "opening_index": root["opening_index"],
                "active_pit_stones": root["active_pit_stones"],
                "baseline_regret": best - margins[baseline],
                "baseline_wdl_inferior": _wdl(margins[baseline]) < _wdl(best),
                "exact_regret": best - margins[exact],
                "exact_wdl_inferior": _wdl(margins[exact]) < _wdl(best),
                "baseline_action": baseline,
                "exact_action": exact,
            }
        )
    rng = random.Random(559)
    groups: dict[int, list[dict]] = {}
    for row in rows:
        groups.setdefault(int(row["opening_index"]), []).append(row)
    cluster_rows = list(groups.values())

    def ci(field: str) -> list[float]:
        samples = []
        for _ in range(20000):
            sampled = rng.choices(cluster_rows, k=len(cluster_rows))
            flattened = [row for cluster in sampled for row in cluster]
            samples.append(statistics.fmean(float(row[field]) for row in flattened))
        samples.sort()
        return [samples[499], samples[19499]]

    by_stones = {}
    for stone_count in range(17, 22):
        subset = [row for row in rows if row["active_pit_stones"] == stone_count]
        by_stones[str(stone_count)] = {
            "n": len(subset),
            "mean_exact_margin_regret": statistics.fmean(
                r["baseline_regret"] for r in subset
            ),
            "wdl_inferior_rate": statistics.fmean(
                r["baseline_wdl_inferior"] for r in subset
            ),
            "mean_regret_opening_bootstrap_95_ci": ci_subset(
                subset, "baseline_regret", 559 + stone_count
            ),
            "wdl_rate_opening_bootstrap_95_ci": ci_subset(
                subset, "baseline_wdl_inferior", 569 + stone_count
            ),
        }
    latency_rows = list(timings.values())
    exact_times = sorted(float(r["exact_warm_decision_ms"]) for r in latency_rows)
    baseline_times = sorted(float(r["baseline_warm_decision_ms"]) for r in latency_rows)
    for row in oracles.values():
        if {int(action) for action in row["action_margins"]} != set(row["legal_moves"]):
            raise ValueError("oracle action set does not equal legal action set")
    full_coverage = len(oracles) == 512 and all(
        row["coverage_complete"] for row in oracles.values()
    )
    routing = verify_routing_boundaries()
    exact_regrets = [row["exact_regret"] for row in rows]
    exact_wdl = [row["exact_wdl_inferior"] for row in rows]
    metrics = {
        "n_roots": len(rows),
        "oracle_full_coverage": full_coverage,
        "baseline_mean_exact_margin_regret": statistics.fmean(
            r["baseline_regret"] for r in rows
        ),
        "baseline_wdl_inferior_rate": statistics.fmean(
            r["baseline_wdl_inferior"] for r in rows
        ),
        "baseline_regret_opening_bootstrap_95_ci": ci("baseline_regret"),
        "baseline_wdl_rate_opening_bootstrap_95_ci": ci("baseline_wdl_inferior"),
        "exact_route_mean_regret": statistics.fmean(exact_regrets),
        "exact_route_wdl_inferior_rate": statistics.fmean(exact_wdl),
        "by_active_stones": by_stones,
        "warm_latency_ms": {
            "baseline_mean": statistics.fmean(baseline_times),
            "baseline_p95": baseline_times[int(0.95 * (len(baseline_times) - 1))],
            "exact_mean": statistics.fmean(exact_times),
            "exact_p95": exact_times[int(0.95 * (len(exact_times) - 1))],
        },
        "native_startup_prewarm": {},
        "routing_invariance_tests": routing,
        "gate": {
            "all_512_valid_and_full_oracle_coverage": len(rows) == 512
            and full_coverage,
            "baseline_wdl_inferior_at_least_5_percent": statistics.fmean(
                r["baseline_wdl_inferior"] for r in rows
            )
            >= 0.05,
            "baseline_regret_at_least_1_seed": statistics.fmean(
                r["baseline_regret"] for r in rows
            )
            >= 1.0,
            "exact_mean_latency_at_most_200_ms": statistics.fmean(exact_times) <= 200,
            "exact_p95_latency_at_most_350_ms": exact_times[
                int(0.95 * (len(exact_times) - 1))
            ]
            <= 350,
        },
        "decision": "no_cutoff_21_arena_proposal"
        if not (
            len(rows) == 512
            and full_coverage
            and statistics.fmean(r["baseline_wdl_inferior"] for r in rows) >= 0.05
            and statistics.fmean(r["baseline_regret"] for r in rows) >= 1.0
            and statistics.fmean(exact_times) <= 200
            and exact_times[int(0.95 * (len(exact_times) - 1))] <= 350
        )
        else "supports_proposing_separate_cutoff_21_arena_only",
        "scope": "retrospective fixed-position feasibility diagnostic from #556; not full-game strength evidence",
    }
    native_summary = json.loads((data / "native-startup-prewarm.json").read_text())
    expected_native_warm = {
        "warm_end_to_end_exact_mean_ms": metrics["warm_latency_ms"]["exact_mean"],
        "warm_end_to_end_exact_p95_ms": metrics["warm_latency_ms"]["exact_p95"],
    }
    if any(
        native_summary.get(key) != value for key, value in expected_native_warm.items()
    ):
        raise ValueError("native_warm_latency_summary_does_not_match_per_root_timings")
    if any(
        not math.isfinite(float(native_summary[key])) or float(native_summary[key]) < 0
        for key in ("startup_ms", "prewarm_ms")
    ):
        raise ValueError("invalid_native_startup_or_prewarm_timing")
    metrics["native_startup_prewarm"] = {
        **native_summary,
        **expected_native_warm,
    }
    metrics["root_records"] = rows
    return metrics


def ci_subset(rows: list[dict], field: str, seed: int) -> list[float] | None:
    if not rows:
        return None
    by_opening: dict[int, list[dict]] = {}
    for row in rows:
        by_opening.setdefault(int(row["opening_index"]), []).append(row)
    clusters = list(by_opening.values())
    rng = random.Random(seed)
    values = sorted(
        statistics.fmean(
            float(row[field])
            for cluster in rng.choices(clusters, k=len(clusters))
            for row in cluster
        )
        for _ in range(20000)
    )
    return [values[499], values[19499]]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--output", type=Path, help="new nonexistent output directory")
    actions.add_argument(
        "--analyze", type=Path, help="existing diagnostic package to analyze"
    )
    args = parser.parse_args()
    if args.analyze is not None:
        result = analyze_published_run(args.analyze)
        roots = result.pop("root_records")
        print(json.dumps({"analysis": result, "root_records": roots}, sort_keys=True))
    else:
        freeze_and_run(args.output)
