#!/usr/bin/env python3
"""Read-only exact-oracle neighborhood audit for PR #289 regression anchors.

This module intentionally has no self-play, training, replay-writing, or
promotion imports.  Native solver failures are recorded as failures; it never
substitutes MCTS or the Python solver for an exact label.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.exact_kalah_solver import ExactState  # noqa: E402
from ml.alphazero_lite.forensic_suite import canonical_state_key, load_suite  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state  # noqa: E402
from ml.alphazero_lite.native_mtdf_probe import payload  # noqa: E402
from ml.alphazero_lite.run_native_hybrid_feasibility import (  # noqa: E402
    NativeHybridProcess,
    NativeRequestError,
)

if TYPE_CHECKING:
    from ml.alphazero_lite.arena import ArtifactEvaluator

SCHEMA = "azlite_uniform1200_exact_neighborhood_audit_v1"
PR289_ARTIFACT = (
    ROOT / "docs/data/alphazero-lite-uniform1200-replay-distribution-audit.json"
)
PR290_ARTIFACT = ROOT / "docs/data/alphazero-lite-uniform1200-midgame-2x2-results.json"
SUITE = ROOT / "ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"
REFERENCES = ROOT / "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
RADIUS2_CAP = 7500
COHORT_SHA256 = "2ae5376b06e26ec382e31b4ea9bad49a21dadd055891a863c407fde9fbe164ce"
CANONICAL_TABLEBASE_SHA256 = (
    "6e65399387f4b9c13bd83568c60a35fbe474a6bdb8ec868002b783eb0abb02ac"
)
COMPLETION_RETRY_TIMEOUT_SECONDS = 120.0


def stable_sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cohort_identity(rows: Iterable[dict[str, Any]]) -> str:
    """Hash only frozen cohort fields, never oracle results or attempt history."""
    frozen = [
        {
            key: row[key]
            for key in (
                "canonical_state_key",
                "state",
                "radius",
                "terminal",
                "provenance",
            )
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def active_stones(row: dict[str, Any]) -> int:
    state = row["state"]
    return sum(state["player_pits"]) + sum(state["opponent_pits"])


class NativeLabelError(ValueError):
    """A malformed native response is an explicit oracle error, never a fallback."""


def validate_native_response(
    values: dict[int, int],
    optimal_actions: list[int],
    exact_value: int,
    legal_moves: list[int],
    current_player: int,
) -> int:
    if not values or set(values) != set(legal_moves):
        raise NativeLabelError("native action values do not exactly cover legal moves")
    root = max(values.values()) if current_player == 0 else min(values.values())
    if exact_value != root or sorted(optimal_actions) != sorted(
        action for action, value in values.items() if value == root
    ):
        raise NativeLabelError(
            "native root or optimal actions violate exact-label contract"
        )
    return root


def root_training_value(margin: int, current_player: int) -> float:
    perspective_margin = margin if current_player == 0 else -margin
    return float((perspective_margin > 0) - (perspective_margin < 0))


def preflight_native_oracle(probe: Path, tablebase: Path) -> dict[str, Any]:
    """Reject substitutes or malformed native hybrid inputs before any request."""
    if not probe.is_file() or not os.access(probe, os.X_OK | os.R_OK):
        raise ValueError(f"native probe is not executable/readable: {probe}")
    if not tablebase.is_file() or not os.access(tablebase, os.R_OK):
        raise ValueError(f"tablebase is not readable: {tablebase}")
    tablebase_sha = sha256_file(tablebase)
    if tablebase_sha != CANONICAL_TABLEBASE_SHA256:
        raise ValueError("tablebase SHA does not match canonical tier-18 artifact")
    with tablebase.open("rb") as handle:
        header = handle.read(48)
    if len(header) != 48 or header[:5] != b"KVTB1" or header[18] != 18:
        raise ValueError("tablebase is not a canonical KVTB1 tier-18 artifact")
    return {
        "solver_identity": "native_mtdf_hybrid_canonical_kvtb",
        "probe_path": str(probe),
        "probe_sha256": sha256_file(probe),
        "probe_version": "sha256-pinned native_probe.c canonical KVTB revision 2",
        "tablebase_path": str(tablebase),
        "tablebase_sha256": tablebase_sha,
        "tablebase_manifest_identity": "kalah_v1/KVTB1/schema=1/revision=2/tier=18",
        "tablebase_tier": 18,
        "tablebase_bytes": tablebase.stat().st_size,
    }


def load_anchor_ids(path: Path = PR289_ARTIFACT) -> tuple[list[str], list[str]]:
    """Load, rather than copy, the frozen PR #289 two-of-three anchor sets."""
    forensic = json.loads(path.read_text(encoding="utf-8"))["forensics"]
    regressions = list(forensic["consistent_uniform_regressions"])
    improvements = list(forensic["consistent_uniform_improvements"])
    if len(regressions) != 38:
        raise ValueError(
            f"PR #289 regression anchor count must be 38, got {len(regressions)}"
        )
    if regressions != sorted(set(regressions)) or improvements != sorted(
        set(improvements)
    ):
        raise ValueError("PR #289 anchor sets must be sorted and unique")
    return regressions, improvements


def _transition(
    state: dict[str, Any], action: int
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    consequence = move_consequence_for_state(state, action)
    game = KalahGame.from_state(state)
    if not game.move(game.pit_index(action)):
        raise ValueError(f"illegal neighborhood action {action}")
    return game.to_state(), consequence, game.over()


def generate_neighborhood(
    anchor_id: str, state: dict[str, Any]
) -> list[dict[str, Any]]:
    """Generate radius 0-2 legal successors, deduplicated by canonical key.

    A state reachable via multiple paths retains every provenance record while
    appearing once as an evaluation state.
    """
    rows: dict[str, dict[str, Any]] = {}

    def add(
        candidate: dict[str, Any],
        radius: int,
        path: list[int],
        extra_turn: bool,
        capture: bool,
        terminal: bool,
    ) -> None:
        key = canonical_state_key(candidate)
        item = rows.setdefault(
            key,
            {
                "canonical_state_key": key,
                "state": candidate,
                "radius": radius,
                "terminal": terminal,
                "training_eligible": False,
                "provenance": [],
            },
        )
        item["radius"] = min(item["radius"], radius)
        item["terminal"] = bool(item["terminal"] and terminal)
        item["provenance"].append(
            {
                "anchor_id": anchor_id,
                "radius": radius,
                "action_path": path,
                "extra_turn": extra_turn,
                "capture": capture,
                "terminal": terminal,
                "canonical_state_key": key,
            }
        )

    add(state, 0, [], False, False, False)
    radius1: list[tuple[dict[str, Any], list[int]]] = []
    for action in KalahGame.from_state(state).possible_moves():
        child, info, terminal = _transition(state, action)
        add(
            child,
            1,
            [action],
            bool(info["gives_extra_turn"]),
            bool(info["produces_capture"]),
            terminal,
        )
        if not terminal:
            radius1.append((child, [action]))
    for parent, path in radius1:
        for action in KalahGame.from_state(parent).possible_moves():
            child, info, terminal = _transition(parent, action)
            add(
                child,
                2,
                path + [action],
                bool(info["gives_extra_turn"]),
                bool(info["produces_capture"]),
                terminal,
            )
    return [rows[key] for key in sorted(rows)]


def generate_cohort(
    anchor_ids: Iterable[str], suite_path: Path = SUITE
) -> list[dict[str, Any]]:
    suite = {position.id: position for position in load_suite(suite_path)}
    cohort: dict[str, dict[str, Any]] = {}
    for anchor_id in anchor_ids:
        if anchor_id not in suite:
            raise ValueError(f"PR #289 anchor missing from suite: {anchor_id}")
        for row in generate_neighborhood(anchor_id, suite[anchor_id].state):
            existing = cohort.setdefault(row["canonical_state_key"], row)
            if existing is not row:
                existing["radius"] = min(existing["radius"], row["radius"])
                existing["provenance"].extend(row["provenance"])
    return [cohort[key] for key in sorted(cohort)]


def sample_radius2(
    cohort: list[dict[str, Any]], cap: int = RADIUS2_CAP
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    early = [row for row in cohort if row["radius"] < 2]
    radius2 = [row for row in cohort if row["radius"] == 2]
    selected = sorted(
        radius2,
        key=lambda row: (
            stable_sha(row["canonical_state_key"]),
            row["canonical_state_key"],
        ),
    )[:cap]
    return sorted(early + selected, key=lambda row: row["canonical_state_key"]), {
        "full_unique": len(cohort),
        "radius2_full": len(radius2),
        "radius2_sampled": len(selected),
        "terminal": sum(row.get("terminal", False) for row in cohort),
    }


def ordered_tasks(
    cohort: Iterable[dict[str, Any]], retry_timeouts: bool, retry_errors: bool
) -> list[dict[str, Any]]:
    """Stable priority ensures interrupted runs cover the strict radii first."""
    eligible = []
    for row in cohort:
        status = row.get("oracle_status", "not_attempted")
        if (
            status == "not_attempted"
            or (retry_timeouts and status == "exact_timeout")
            or (retry_errors and status == "exact_error")
        ):
            if not row["terminal"]:
                eligible.append(row)
    return sorted(
        eligible,
        key=lambda row: (
            row["radius"],
            stable_sha(row["canonical_state_key"]),
            row["canonical_state_key"],
        ),
    )


def _attempt(
    row: dict[str, Any], probe: Path, tablebase: Path, timeout: float
) -> dict[str, Any]:
    """One independent native process: no mutable solver state crosses workers."""
    started = time.monotonic()
    worker: NativeHybridProcess | None = None
    attempt = {
        "attempt_number": len(row.get("oracle_attempt_history", [])) + 1,
        "timeout_seconds": timeout,
        "solver_identity": "native_mtdf_hybrid_canonical_kvtb",
        "probe_sha256": sha256_file(probe),
        "tablebase_sha256": sha256_file(tablebase),
    }
    try:
        worker = NativeHybridProcess(probe, tablebase)
        result = worker.request(
            payload(ExactState.from_game_state(row["state"]), "label"), timeout
        )
        game = KalahGame.from_state(row["state"])
        values = {
            int(action): int(value) for action, value in result["action_values"].items()
        }
        root = validate_native_response(
            values,
            result["optimal_actions"],
            int(result["exact_value"]),
            game.possible_moves(),
            game.current_player,
        )
        attempt["status"] = "exact_solved"
        attempt["failure_reason"] = None
        return attempt | {
            "exact_value": root,
            "exact_action_values": values,
            "exact_optimal_actions": sorted(result["optimal_actions"]),
            "exact_root_value": root_training_value(root, game.current_player),
        }
    except TimeoutError as error:
        attempt["status"] = "exact_timeout"
        attempt["failure_reason"] = str(error)
    except (
        NativeRequestError,
        NativeLabelError,
        KeyError,
        ValueError,
        OSError,
    ) as error:
        attempt["status"] = "exact_error"
        attempt["failure_reason"] = str(error)
    finally:
        attempt["wall_duration_seconds"] = time.monotonic() - started
        if worker is not None:
            worker.close()
    return attempt


def solve_exact(
    cohort: list[dict[str, Any]],
    probe: Path,
    tablebase: Path,
    timeout: float,
    retry_timeouts: bool = False,
    retry_errors: bool = False,
    workers: int = 1,
    on_update: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Submit each state once; solved records are immutable across resumes."""
    tasks = ordered_tasks(cohort, retry_timeouts, retry_errors)
    if len({row["canonical_state_key"] for row in tasks}) != len(tasks):
        raise ValueError("duplicate canonical state submitted to native oracle")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_attempt, row, probe, tablebase, timeout): row
            for row in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                attempt = future.result()
            except BaseException as error:
                attempt = {
                    "attempt_number": len(row.get("oracle_attempt_history", [])) + 1,
                    "timeout_seconds": timeout,
                    "wall_duration_seconds": 0.0,
                    "status": "exact_error",
                    "failure_reason": f"worker crash: {error}",
                    "solver_identity": "native_mtdf_hybrid_canonical_kvtb",
                    "probe_sha256": sha256_file(probe),
                    "tablebase_sha256": sha256_file(tablebase),
                }
            row.setdefault("oracle_attempt_history", []).append(attempt)
            # A completed label is never overwritten, even under a corrupted resume file.
            if row.get("oracle_status") != "exact_solved":
                row["oracle_status"] = attempt["status"]
                row["oracle_failure_reason"] = attempt["failure_reason"]
                if attempt["status"] == "exact_solved":
                    row.update(
                        {
                            key: attempt[key]
                            for key in (
                                "exact_value",
                                "exact_action_values",
                                "exact_optimal_actions",
                                "exact_root_value",
                            )
                        }
                    )
            if on_update:
                on_update()
    return cohort


def coverage_by_radius(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for radius in range(3):
        subset = [row for row in rows if row["radius"] == radius]
        solved = sum(row.get("oracle_status") == "exact_solved" for row in subset)
        result[str(radius)] = {
            "states": len(subset),
            "exact_solved": solved,
            "solve_rate": solved / len(subset) if subset else 0.0,
        }
    return result


def oracle_diagnostics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)

    def summarize(subset: Iterable[dict[str, Any]]) -> dict[str, int]:
        return dict(
            sorted(
                Counter(
                    row.get("oracle_status", "not_attempted") for row in subset
                ).items()
            )
        )

    by_radius = {
        str(radius): summarize(row for row in rows if row["radius"] == radius)
        for radius in range(3)
    }
    by_anchor_bucket: dict[str, dict[str, int]] = {}
    for row in rows:
        for anchor in {item["anchor_id"].split("-")[0] for item in row["provenance"]}:
            by_anchor_bucket.setdefault(anchor, Counter())[
                row.get("oracle_status", "not_attempted")
            ] += 1
    by_attempt = {"original": Counter(), "completion": Counter()}
    for row in rows:
        for attempt in row.get("oracle_attempt_history", []):
            lane = "original" if attempt.get("historical_pr291") else "completion"
            by_attempt[lane][attempt["status"]] += 1
    return {
        "by_radius": by_radius,
        "by_forensic_anchor_bucket": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_anchor_bucket.items())
        },
        "by_active_stones": {
            str(stones): summarize(row for row in rows if active_stones(row) == stones)
            for stones in sorted({active_stones(row) for row in rows})
        },
        "by_attempt": {
            key: dict(sorted(value.items())) for key, value in by_attempt.items()
        },
    }


def action_regret(row: dict[str, Any], action: int) -> int:
    values = row["exact_action_values"]
    best = row["exact_value"]
    return (
        (best - values[action])
        if row["state"]["current_player"] == 0
        else (values[action] - best)
    )


def evaluate_network(
    row: dict[str, Any], evaluator: ArtifactEvaluator
) -> dict[str, Any]:
    game = KalahGame.from_state(row["state"])
    policy, value = evaluator.evaluate(game)
    legal = game.possible_moves()
    action = min(legal, key=lambda move: (-float(policy[move]), move))
    return {
        "policy": [float(v) for v in policy],
        "top_action": action,
        "value": float(value),
        "optimal": action in row["exact_optimal_actions"],
        "exact_regret": action_regret(row, action),
        "exact_value_error": abs(float(value) - row["exact_root_value"]),
    }


def local_rates(rows: Iterable[dict[str, Any]], seed: str) -> dict[str, float]:
    solved = [row for row in rows if row.get("oracle_status") == "exact_solved"]
    comparisons = [
        row
        for row in solved
        if seed in row.get("evaluations", {})
        and "control" in row["evaluations"][seed]
        and "uniform1200" in row["evaluations"][seed]
    ]
    if not comparisons:
        return {
            "states": 0,
            "local_regression_rate": 0.0,
            "local_correct_to_wrong_rate": 0.0,
        }
    control = [row["evaluations"][seed]["control"] for row in comparisons]
    uniform = [row["evaluations"][seed]["uniform1200"] for row in comparisons]
    return {
        "states": len(comparisons),
        "local_regression_rate": sum(
            b["exact_regret"] > a["exact_regret"]
            for a, b in zip(control, uniform, strict=True)
        )
        / len(comparisons),
        "local_correct_to_wrong_rate": sum(
            a["optimal"] and not b["optimal"]
            for a, b in zip(control, uniform, strict=True)
        )
        / len(comparisons),
    }


def family_classification(
    *,
    original_regression_seeds: int,
    local_rates_by_seed: list[float],
    solved_states: int,
    same_direction_seeds: int,
) -> str:
    if (
        original_regression_seeds >= 2
        and statistics.median(local_rates_by_seed) >= 0.60
        and solved_states >= 10
        and same_direction_seeds >= 2
    ):
        return "stable_regression_family"
    if original_regression_seeds >= 2 and statistics.median(local_rates_by_seed) < 0.35:
        return "anchor_specific_regression"
    return "mixed"


def frozen_reference_audit(
    rows: Iterable[dict[str, Any]], references_path: Path = REFERENCES
) -> dict[str, Any]:
    references = {
        row["canonical_state"]: row
        for row in json.loads(references_path.read_text(encoding="utf-8"))["rows"]
    }
    audit = []
    for row in rows:
        if row.get("oracle_status") != "exact_solved" or row["radius"] != 0:
            continue
        reference = references.get(row["canonical_state_key"])
        if reference is None:
            continue
        action = int(reference["reference_move"])
        audit.append(
            {
                "anchor_id": row["provenance"][0]["anchor_id"],
                "reference_move": action,
                "optimal": action in row["exact_optimal_actions"],
                "regret": action_regret(row, action),
            }
        )
    by_bucket = Counter(
        item["anchor_id"].split("-")[0] for item in audit if not item["optimal"]
    )
    return {
        "anchors_exact_solved": len(audit),
        "disagreement_count": sum(not item["optimal"] for item in audit),
        "disagreement_by_bucket_prefix": dict(sorted(by_bucket.items())),
        "rows": audit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-probe", type=Path)
    parser.add_argument("--tablebase", type=Path)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retry-timeouts", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--finalize-partial",
        action="store_true",
        help="persist a resumable exact run as partial without submitting new requests",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.retry_timeouts and args.timeout != COMPLETION_RETRY_TIMEOUT_SECONDS:
        parser.error(
            f"completion retries are pre-registered at {COMPLETION_RETRY_TIMEOUT_SECONDS:g}s"
        )
    regressions, improvements = load_anchor_ids()
    cohort = generate_cohort(regressions)
    sampled, counts = sample_radius2(cohort)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "pr290_status": "merged",
        "pr290_classification": "midgame_2x2_no_gain",
        "training_eligible": False,
        "anchor_counts": {
            "regressions": len(regressions),
            "improvements": len(improvements),
        },
        "inputs": {
            "pr289_artifact_sha256": sha256_file(PR289_ARTIFACT),
            "pr290_artifact_sha256": sha256_file(PR290_ARTIFACT),
            "suite_sha256": sha256_file(SUITE),
        },
        "neighborhood_counts": counts,
        "frozen_cohort_sha256": cohort_identity(sampled),
        "cohort": sampled,
    }
    expected_inputs = {
        "pr289_artifact_sha256": "cdec9522efe250c3128b7b53861f9f3b14e01f91fb20a80a3e1e4e119641a60d",
        "pr290_artifact_sha256": "0871dcfbf5afc5d07676780c021cd790a9373b9736e3690e297b041873260234",
        "suite_sha256": "68506f12685e32c868cc1a21c2e7fa16b9f57478ee3fb885bbe1ba5ef475f4fd",
    }
    if (
        report["inputs"] != expected_inputs
        or report["frozen_cohort_sha256"] != COHORT_SHA256
    ):
        raise ValueError("committed PR #291 input or frozen cohort identity mismatch")
    if args.output.exists():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        for key in ("native_oracle_preflight", "completion_policy"):
            if key in prior:
                report[key] = prior[key]
        prior_rows = {
            row["canonical_state_key"]: row for row in prior.get("cohort", [])
        }
        for row in sampled:
            previous = prior_rows.get(row["canonical_state_key"])
            if previous and previous.get("oracle_status"):
                row.update(
                    {
                        key: value
                        for key, value in previous.items()
                        if key.startswith("oracle_") or key.startswith("exact_")
                    }
                )
                # PR #291 predates attempt history. Preserve its timeout as a
                # historical 30-second attempt rather than treating it as new.
                if "oracle_attempt_history" not in row:
                    row["oracle_attempt_history"] = [
                        {
                            "attempt_number": 1,
                            "timeout_seconds": 30.0,
                            "wall_duration_seconds": None,
                            "status": row["oracle_status"],
                            "failure_reason": row.get("oracle_failure_reason"),
                            "solver_identity": "native_mtdf_hybrid_canonical_kvtb",
                            "probe_sha256": None,
                            "tablebase_sha256": None,
                            "historical_pr291": True,
                        }
                    ]

    def persist() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=args.output.parent, delete=False
        ) as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(args.output)

    if args.finalize_partial:
        statuses = Counter(row.get("oracle_status", "not_attempted") for row in sampled)
        report["oracle_status"] = "partial: fixed solver timeout exhausted"
        report["oracle_coverage"] = dict(sorted(statuses.items()))
        report["classification"] = "forensic_neighborhood_audit_inconclusive"
        report["blocker"] = "solver/tablebase coverage and solver latency"
    elif args.native_probe and args.tablebase:
        if report["frozen_cohort_sha256"] != COHORT_SHA256:
            raise ValueError("regenerated frozen cohort differs from PR #291")
        report["native_oracle_preflight"] = preflight_native_oracle(
            args.native_probe, args.tablebase
        )
        report["completion_policy"] = {
            "original_timeout_seconds": 30.0,
            "retry_timeout_seconds": COMPLETION_RETRY_TIMEOUT_SECONDS,
            "retry_timeouts": args.retry_timeouts,
            "retry_errors": args.retry_errors,
            "workers": args.workers,
            "priority": "radius then canonical_state_sha256",
        }
        report["oracle_status"] = "running"
        persist()
        solve_exact(
            sampled,
            args.native_probe,
            args.tablebase,
            args.timeout,
            retry_timeouts=args.retry_timeouts,
            retry_errors=args.retry_errors,
            workers=args.workers,
            on_update=persist,
        )
        report["oracle_status"] = "completed"
        report["frozen_reference_audit"] = frozen_reference_audit(sampled)
    else:
        report["oracle_status"] = (
            "not_run: --native-probe and --tablebase are required; no MCTS fallback"
        )
    report["oracle_coverage"] = dict(
        sorted(
            Counter(
                row.get("oracle_status", "not_attempted") for row in sampled
            ).items()
        )
    )
    report["coverage_by_radius"] = coverage_by_radius(sampled)
    report["oracle_diagnostics"] = oracle_diagnostics(sampled)
    persist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
