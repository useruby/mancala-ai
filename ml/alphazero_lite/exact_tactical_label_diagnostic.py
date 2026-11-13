"""Frozen retrospective audit of missed moves in the #585 P archive.

This diagnostic never starts games or changes training/serving.  The ``freeze``
command authenticates the merged #585 evidence and records the cohort and
per-root limits; ``solve`` is intentionally separate and only consumes that
already-frozen manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed551_controlled_policy_strength import (
    canonical_identity,
    production_float32_identity,
)
from ml.alphazero_lite.exact_kalah_solver import (
    ExactKalahSolver,
    ExactState,
    SearchTimeout,
)

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "docs/data/broader-opening-coverage-pilot"
OUT = ROOT / "docs/data/exact-tactical-label-diagnostic"
ARCHIVE_EVIDENCE_SHA256 = (
    "e1cf047b1e8587d673ae711c11e29eb9125240391d945a581a27ce47ce37cd75"
)
ARCHIVE_FREEZE_SHA256 = (
    "3f2ccca4fae605f826b4fb00e0adb6f88c8c87ad8ca1373152aea051133f75b5"
)
TABLEBASE_SHA256 = "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7"
COHORT_SIZE = 256
SELECTION_SEED = "585-exact-tactical-v1"
PER_ROOT_SECONDS = 30
PER_ACTION_SECONDS = 5
MAX_TT_ENTRIES = 500_000


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _state_from_row(row: dict[str, Any]) -> dict[str, Any]:
    state = row["canonical_state"]
    return {
        "player_pits": list(state["player_pits"]),
        "opponent_pits": list(state["opponent_pits"]),
        "player_store": int(state["player_store"]),
        "opponent_store": int(state["opponent_store"]),
        "current_player": int(state["current_player"]),
    }


def _row_game_pairs() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted((ARCHIVE / "pairs").glob("*.json")):
        payload = _json(path)
        for game in payload["result"]["games"]:
            trajectory = {
                step.get("row_id"): step
                for step in game["trajectory"]
                if step.get("row_id")
            }
            for row in game["rows"]:
                if row["designated_p_seat"] != row["player"]:
                    continue
                step = trajectory.get(row["row_id"])
                if step is None:
                    raise ValueError(f"archive_trajectory_missing:{row['row_id']}")
                if step["state"] != row["canonical_state"]:
                    raise ValueError(f"archive_state_replay_mismatch:{row['row_id']}")
                state = _state_from_row(row)
                if int(row["stone_count"]) != sum(
                    state["player_pits"] + state["opponent_pits"]
                ):
                    raise ValueError(f"archive_stone_count_mismatch:{row['row_id']}")
                result.append({"row": row, "move": int(step["move"]), "state": state})
    return result


def freeze() -> dict[str, Any]:
    # Full source authentication and replay are performed by the independently
    # published #585 verifier before this command; bind its externally supplied
    # digest plus all source file hashes into our own freeze.
    if (
        _json(ARCHIVE / "evidence-digest.json")["digest_sha256"]
        != ARCHIVE_EVIDENCE_SHA256
    ):
        raise ValueError("585_external_evidence_digest_mismatch")
    if _json(ARCHIVE / "freeze.json").get("freeze_sha256") != ARCHIVE_FREEZE_SHA256:
        raise ValueError("585_freeze_hash_mismatch")
    tablebase = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
    if _sha(tablebase) != TABLEBASE_SHA256:
        raise ValueError("authenticated_tablebase_hash_mismatch")

    candidates = [
        item
        for item in _row_game_pairs()
        if sum(item["state"]["player_pits"] + item["state"]["opponent_pits"]) > 32
    ]
    # Deduplicate against both identities and sample equally across opening
    # groups. Hash ordering is deterministic and independent of solver outcomes.
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen_canonical: set[str] = set()
    seen_float32: set[str] = set()
    for item in candidates:
        state = item["state"]
        can = canonical_identity(state)
        f32 = production_float32_identity(state)
        if can in seen_canonical or f32 in seen_float32:
            continue
        seen_canonical.add(can)
        seen_float32.add(f32)
        item["canonical_identity"] = can
        item["production_float32_identity"] = f32
        item["opening_group"] = item["row"]["opening_id"]
        item["row_id"] = item["row"]["row_id"]
        grouped.setdefault(item["opening_group"], []).append(item)
    for group, items in grouped.items():
        items.sort(
            key=lambda item: hashlib.sha256(
                f"{SELECTION_SEED}:{group}:{item['row_id']}".encode()
            ).hexdigest()
        )
    groups = sorted(grouped)
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < COHORT_SIZE:
        added = False
        for group in groups:
            if depth < len(grouped[group]):
                selected.append(grouped[group][depth])
                added = True
                if len(selected) == COHORT_SIZE:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != COHORT_SIZE:
        raise ValueError(f"insufficient_unique_positions:{len(selected)}")

    roots = []
    for index, item in enumerate(selected):
        row = item["row"]
        roots.append(
            {
                "index": index,
                "opening_group": item["opening_group"],
                "row_id": item["row_id"],
                "state": item["state"],
                "canonical_identity": item["canonical_identity"],
                "production_float32_identity": item["production_float32_identity"],
                "archived_move": item["move"],
                "legal_moves": row["legal_moves"],
                "active_stones": row["stone_count"],
                "simulations": row["simulations"],
                "budget_seconds": PER_ROOT_SECONDS,
                "per_action_seconds": PER_ACTION_SECONDS,
                "tt_entries": MAX_TT_ENTRIES,
            }
        )
    cohort = {
        "schema": "exact_tactical_label_cohort_v1",
        "source_pr": 585,
        "source_evidence_digest_sha256": ARCHIVE_EVIDENCE_SHA256,
        "source_freeze_sha256": ARCHIVE_FREEZE_SHA256,
        "tablebase_sha256": TABLEBASE_SHA256,
        "tablebase_use": "authenticated Kalah v1 21-stone tablebase is eligible for descendants; root-level callback is not integrated in this first bounded solver implementation",
        "selection_seed": SELECTION_SEED,
        "selection_rule": "Within each opening_id select by SHA256(seed:opening_group:row_id), then round-robin sorted opening groups; require uniqueness under canonical-state and production kalah_v3-float32 identities.",
        "deduplication": {
            "candidate_rows": len(candidates),
            "unique_roots_under_both_schemes": len(seen_canonical),
            "opening_groups": len(groups),
        },
        "solver_limits": {
            "per_root_wall_seconds": PER_ROOT_SECONDS,
            "per_action_wall_seconds": PER_ACTION_SECONDS,
            "tt_entries": MAX_TT_ENTRIES,
            "move_order": "ascending",
            "search": "full-depth alpha-beta; only exact completed margins count as certified; timeout is unresolved",
            "no_games_training_or_serving_changes": True,
        },
        "roots": roots,
    }
    cohort["cohort_sha256"] = hashlib.sha256(_stable(cohort)).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cohort.json").write_bytes(
        json.dumps(cohort, sort_keys=True, indent=2).encode() + b"\n"
    )
    return cohort


def solve() -> dict[str, Any]:
    cohort = _json(OUT / "cohort.json")
    recorded_hash = cohort.pop("cohort_sha256")
    if hashlib.sha256(_stable(cohort)).hexdigest() != recorded_hash:
        raise ValueError("cohort_hash_mismatch")
    tablebase = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
    if _sha(tablebase) != TABLEBASE_SHA256:
        raise ValueError("authenticated_tablebase_hash_mismatch")
    evidence = []
    for root in cohort["roots"]:
        start = time.monotonic()
        state = ExactState.from_game_state(root["state"])
        actions: dict[str, Any] = {}
        for move in root["legal_moves"]:
            remaining = min(
                root["per_action_seconds"],
                max(0.001, root["budget_seconds"] - (time.monotonic() - start)),
            )
            solver = ExactKalahSolver(
                tt_size=root["tt_entries"], move_order="ascending"
            )
            try:
                margin = solver.solve(
                    state.play(int(move)), time_limit_seconds=remaining
                )
                actions[str(move)] = {
                    "lower": margin,
                    "upper": margin,
                    "exact": True,
                    "nodes": solver.nodes,
                }
            except SearchTimeout:
                actions[str(move)] = {
                    "lower": None,
                    "upper": None,
                    "exact": False,
                    "nodes": solver.nodes,
                }
            finally:
                solver.close()
            if time.monotonic() - start >= root["budget_seconds"]:
                for pending in root["legal_moves"][
                    root["legal_moves"].index(move) + 1 :
                ]:
                    actions[str(pending)] = {
                        "lower": None,
                        "upper": None,
                        "exact": False,
                        "nodes": 0,
                    }
                break
        archived = actions[str(root["archived_move"])]
        sign = 1 if int(root["state"]["current_player"]) == 0 else -1
        proving_moves = [
            int(move)
            for move, bounds in actions.items()
            if move != str(root["archived_move"])
            and bounds["exact"]
            and archived["exact"]
            and sign * bounds["lower"] > sign * archived["upper"]
        ]
        complete = all(bounds["exact"] for bounds in actions.values())
        exact_margins = [
            bounds["lower"] for bounds in actions.values() if bounds["exact"]
        ]
        best = (max if sign == 1 else min)(exact_margins, default=None)
        evidence.append(
            {
                "index": root["index"],
                "actions": actions,
                "archived_move": root["archived_move"],
                "proven_inferior": bool(proving_moves),
                "proving_moves": proving_moves,
                "certified_best_action": complete,
                "best_margin": best,
                "archived_margin": archived["lower"] if archived["exact"] else None,
                "status": "solved" if complete else "unresolved",
                "elapsed_seconds": time.monotonic() - start,
            }
        )
    result = {
        "schema": "exact_tactical_label_evidence_v1",
        "cohort_sha256": recorded_hash,
        "roots": evidence,
    }
    result["evidence_sha256"] = hashlib.sha256(_stable(result)).hexdigest()
    (OUT / "evidence.json").write_bytes(
        json.dumps(result, sort_keys=True, indent=2).encode() + b"\n"
    )
    return result


def verify() -> dict[str, Any]:
    cohort = _json(OUT / "cohort.json")
    evidence = _json(OUT / "evidence.json")
    frozen_hash = cohort.pop("cohort_sha256")
    if hashlib.sha256(_stable(cohort)).hexdigest() != frozen_hash:
        raise ValueError("cohort_hash_mismatch")
    ev_hash = evidence.pop("evidence_sha256")
    if hashlib.sha256(_stable(evidence)).hexdigest() != ev_hash:
        raise ValueError("evidence_hash_mismatch")
    if (
        evidence["cohort_sha256"] != frozen_hash
        or len(evidence["roots"]) != COHORT_SIZE
    ):
        raise ValueError("cohort_evidence_binding_mismatch")
    certified = 0
    downgrades = {"win_to_draw": 0, "win_to_loss": 0, "draw_to_loss": 0}
    for root, result in zip(cohort["roots"], evidence["roots"], strict=True):
        if (
            result["index"] != root["index"]
            or result["archived_move"] != root["archived_move"]
        ):
            raise ValueError("root_identity_mismatch")
        bounds = result["actions"]
        for move, action in bounds.items():
            if action["exact"] and action["lower"] != action["upper"]:
                raise ValueError("invalid_exact_margin")
        certified += int(result["certified_best_action"])
        if result["proven_inferior"]:
            actual = result["archived_margin"]
            best = result["best_margin"]
            if actual is None or best is None or actual >= best:
                raise ValueError("inferiority_not_certified")
            # Outcome classes are from the perspective of the player to move.
            player = int(root["state"]["current_player"])
            player_margin = actual if player == 0 else -actual
            best_player_margin = best if player == 0 else -best
            if best_player_margin > 0 and player_margin == 0:
                downgrades["win_to_draw"] += 1
            elif best_player_margin > 0 and player_margin < 0:
                downgrades["win_to_loss"] += 1
            elif best_player_margin == 0 and player_margin < 0:
                downgrades["draw_to_loss"] += 1
    return {
        "verified": True,
        "cohort_sha256": frozen_hash,
        "evidence_sha256": ev_hash,
        "certified_best_action_roots": certified,
        "proven_inferior_archived_moves": sum(
            int(row["proven_inferior"]) for row in evidence["roots"]
        ),
        "unresolved_roots": COHORT_SIZE - certified,
        "outcome_downgrades": downgrades,
        "planning_gate": {
            "certified_best_action_at_least_32": certified >= 32,
            "inferior_moves_at_least_8": sum(
                int(row["proven_inferior"]) for row in evidence["roots"]
            )
            >= 8,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "solve", "verify"))
    args = parser.parse_args()
    result = {"freeze": freeze, "solve": solve, "verify": verify}[args.command]()
    print(
        json.dumps(
            result if args.command != "verify" else result, indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
