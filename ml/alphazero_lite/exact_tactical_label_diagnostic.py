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
import os
import select
import subprocess
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
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.native_exact_root_tablebase import NativeExactRootTablebase
from ml.alphazero_lite.verify_broader_opening_coverage_results import (
    verify as verify_585_archive,
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
TABLEBASE_BINARY_SHA256 = (
    "d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48"
)
COHORT_SIZE = 256
SELECTION_SEED = "585-exact-tactical-v1"
PER_ROOT_SECONDS = 30
PER_ACTION_SECONDS = 5
MAX_TT_ENTRIES = 500_000


class DeadlineNativeExactRootTablebase(NativeExactRootTablebase):
    """Add deadline-aware, complete-line reads without changing #585 code."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stdout_buffer = ""
        self._probe_poisoned = False

    def _launch_fresh_probe(self) -> None:
        binary, tablebase = self.binary, self.tablebase
        calls = self.calls
        NativeExactRootTablebase.__init__(self, binary, tablebase)
        self.calls = calls
        self._stdout_buffer = ""
        self._probe_poisoned = False

    def _retire_timed_out_probe(self) -> None:
        self._probe_poisoned = True
        process = self.process
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass

    def _raise_probe_timeout(self) -> None:
        self._retire_timed_out_probe()
        raise TimeoutError("native exact-root probe deadline exceeded")

    def root_action_margins(
        self,
        game: KalahGame,
        perspective_player: int,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[int, int]:
        raw_actions = self._probe(
            game.pits, game.current_player, timeout_seconds=timeout_seconds
        )
        if not isinstance(raw_actions, dict):
            raise RuntimeError(f"native exact-root probe failed: {raw_actions}")
        self.calls += 1
        store_margin_p0 = game.captured_seeds[0] - game.captured_seeds[1]
        root_sign = 1 if int(perspective_player) == 0 else -1
        return {
            int(move): root_sign * (int(raw_margin) + store_margin_p0)
            for move, raw_margin in raw_actions.items()
        }

    def _probe(
        self,
        pits: list[int],
        player: int,
        *,
        timeout_seconds: float | None = None,
    ) -> object:
        if self._probe_poisoned:
            self._launch_fresh_probe()
        if self.process.poll() is not None:
            raise RuntimeError("native exact-root probe exited")
        assert self.process.stdin is not None and self.process.stdout is not None
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + max(0.0, timeout_seconds)
        )
        request = {"pits": pits, "player": player}
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        while "\n" not in self._stdout_buffer:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                self._raise_probe_timeout()
            readable, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not readable:
                self._raise_probe_timeout()
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("native exact-root probe closed stdout")
            self._stdout_buffer += chunk.decode()
        line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
        if deadline is not None and time.monotonic() > deadline:
            self._raise_probe_timeout()
        response = json.loads(line)
        return response.get("actions")


class TablebaseBoundedSolver(ExactKalahSolver):
    """Use exact KVTB values at every eligible nonterminal descendant."""

    def __init__(
        self, tablebase: DeadlineNativeExactRootTablebase, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.tablebase = tablebase

    def _search(self, state: ExactState, alpha: int, beta: int) -> int:
        if sum(state.pits) <= 21 and not state.is_terminal():
            if self._deadline is not None and time.monotonic() >= self._deadline:
                raise SearchTimeout()
            key = state.key()
            entry = self._lookup(key)
            if entry is not None and entry[1] == 0:
                return entry[0]
            game = KalahGame.from_state(state.to_game_state())
            remaining = (
                None if self._deadline is None else self._deadline - time.monotonic()
            )
            if remaining is not None and remaining <= 0:
                raise SearchTimeout()
            try:
                margins = self.tablebase.root_action_margins(
                    game, perspective_player=0, timeout_seconds=remaining
                )
            except TimeoutError as exc:
                raise SearchTimeout() from exc
            expected_moves = set(state.legal_moves())
            if set(margins) != expected_moves or any(
                type(value) is not int for value in margins.values()
            ):
                raise ValueError("incomplete_or_invalid_tablebase_action_margins")
            if self._deadline is not None and time.monotonic() >= self._deadline:
                raise SearchTimeout()
            value = (
                max(margins.values())
                if state.current_player == 0
                else min(margins.values())
            )
            self._store(key, value, 0)
            return value
        return super()._search(state, alpha, beta)


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _reject_published_output(path: Path) -> Path:
    output = path.resolve()
    if output == OUT.resolve():
        raise ValueError("published_output_directory_is_read_only")
    return output


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


def _build_frozen_roots() -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Rebuild deterministic source roots from the already authenticated archive."""
    candidates = [
        item
        for item in _row_game_pairs()
        if sum(item["state"]["player_pits"] + item["state"]["opponent_pits"]) > 32
    ]
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
    return roots, {
        "candidate_rows": len(candidates),
        "unique_roots_under_both_schemes": len(seen_canonical),
        "opening_groups": len(groups),
    }


def freeze(output_dir: Path) -> dict[str, Any]:
    # Full source authentication and replay are performed by the independently
    # published #585 verifier before this command; bind its externally supplied
    # digest plus all source file hashes into our own freeze.
    verify_585_archive(ARCHIVE, ARCHIVE_EVIDENCE_SHA256)
    if _json(ARCHIVE / "freeze.json").get("freeze_sha256") != ARCHIVE_FREEZE_SHA256:
        raise ValueError("585_freeze_hash_mismatch")
    tablebase = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
    if _sha(tablebase) != TABLEBASE_SHA256:
        raise ValueError("authenticated_tablebase_hash_mismatch")
    binary = ROOT / "model-artifact/runtime/kalah_v1_tablebase"
    if _sha(binary) != TABLEBASE_BINARY_SHA256:
        raise ValueError("authenticated_tablebase_binary_hash_mismatch")

    output_dir = _reject_published_output(output_dir)
    if output_dir.exists():
        raise ValueError("freeze_output_directory_must_be_new")
    roots, deduplication = _build_frozen_roots()
    cohort = {
        "schema": "exact_tactical_label_cohort_v1",
        "source_pr": 585,
        "source_evidence_digest_sha256": ARCHIVE_EVIDENCE_SHA256,
        "source_freeze_sha256": ARCHIVE_FREEZE_SHA256,
        "tablebase_sha256": TABLEBASE_SHA256,
        "tablebase_binary_sha256": TABLEBASE_BINARY_SHA256,
        "tablebase_use": "exact native Kalah v1 21-stone tablebase values terminate descendant search at every eligible nonterminal state",
        "solver_source_sha256": {
            "diagnostic": _sha(Path(__file__).resolve()),
            "exact_kalah_solver": _sha(
                ROOT / "ml/alphazero_lite/exact_kalah_solver.py"
            ),
            "native_root_tablebase_adapter": _sha(
                ROOT / "ml/alphazero_lite/native_exact_root_tablebase.py"
            ),
        },
        "selection_seed": SELECTION_SEED,
        "selection_rule": "Within each opening_id select by SHA256(seed:opening_group:row_id), then round-robin sorted opening groups; require uniqueness under canonical-state and production kalah_v3-float32 identities.",
        "deduplication": deduplication,
        "solver_limits": {
            "per_root_wall_seconds": PER_ROOT_SECONDS,
            "per_action_wall_seconds": PER_ACTION_SECONDS,
            "tt_entries": MAX_TT_ENTRIES,
            "move_order": "ascending",
            "search": "full-depth alpha-beta with exact tablebase values at descendant states of 21 or fewer active stones; only exact completed margins count as certified; timeout is unresolved",
            "no_games_training_or_serving_changes": True,
        },
        "roots": roots,
    }
    cohort["cohort_sha256"] = hashlib.sha256(_stable(cohort)).hexdigest()
    output_dir.mkdir(parents=True)
    (output_dir / "cohort.json").write_bytes(
        json.dumps(cohort, sort_keys=True, indent=2).encode() + b"\n"
    )
    return cohort


def solve(output_dir: Path) -> dict[str, Any]:
    output_dir = _reject_published_output(output_dir)
    if any(
        (output_dir / name).exists()
        for name in ("evidence.json", "tablebase-recheck.json")
    ):
        raise ValueError("solver_final_evidence_already_exists")
    cohort = _json(output_dir / "cohort.json")
    recorded_hash = cohort.pop("cohort_sha256")
    if hashlib.sha256(_stable(cohort)).hexdigest() != recorded_hash:
        raise ValueError("cohort_hash_mismatch")
    source_paths = {
        "diagnostic": Path(__file__).resolve(),
        "exact_kalah_solver": ROOT / "ml/alphazero_lite/exact_kalah_solver.py",
        "native_root_tablebase_adapter": ROOT
        / "ml/alphazero_lite/native_exact_root_tablebase.py",
    }
    if {name: _sha(path) for name, path in source_paths.items()} != cohort[
        "solver_source_sha256"
    ]:
        raise ValueError("frozen_solver_source_hash_mismatch")
    tablebase = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
    if _sha(tablebase) != TABLEBASE_SHA256:
        raise ValueError("authenticated_tablebase_hash_mismatch")
    binary = ROOT / "model-artifact/runtime/kalah_v1_tablebase"
    if _sha(binary) != TABLEBASE_BINARY_SHA256:
        raise ValueError("authenticated_tablebase_binary_hash_mismatch")
    evidence = []
    progress_path = output_dir / "evidence-progress.json"
    if progress_path.is_file():
        progress = _json(progress_path)
        if progress.get("cohort_sha256") != recorded_hash:
            raise ValueError("solver_progress_cohort_mismatch")
        evidence = list(progress.get("roots", []))
        if len(evidence) > len(cohort["roots"]):
            raise ValueError("solver_progress_root_count_invalid")
    with DeadlineNativeExactRootTablebase(binary, tablebase) as native:
        for root in cohort["roots"][len(evidence) :]:
            start = time.monotonic()
            state = ExactState.from_game_state(root["state"])
            actions: dict[str, Any] = {}
            solver = TablebaseBoundedSolver(
                native, tt_size=root["tt_entries"], move_order="ascending"
            )
            for move in root["legal_moves"]:
                remaining = min(
                    root["per_action_seconds"],
                    max(0.001, root["budget_seconds"] - (time.monotonic() - start)),
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
            solver.close()
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
            progress_record = {
                "schema": "exact_tactical_label_progress_v1",
                "cohort_sha256": recorded_hash,
                "roots": evidence,
            }
            progress_path.write_bytes(
                json.dumps(progress_record, sort_keys=True, indent=2).encode() + b"\n"
            )
    result = {
        "schema": "exact_tactical_label_evidence_v1",
        "cohort_sha256": recorded_hash,
        "roots": evidence,
    }
    result["evidence_sha256"] = hashlib.sha256(_stable(result)).hexdigest()
    (output_dir / "evidence.json").write_bytes(
        json.dumps(result, sort_keys=True, indent=2).encode() + b"\n"
    )
    return result


def recheck_exact_claims(output_dir: Path) -> dict[str, Any]:
    """Re-solve every previously exact action claim with hardened tablebase IO."""
    output_dir = _reject_published_output(output_dir)
    recheck_path = output_dir / "tablebase-recheck.json"
    if recheck_path.exists():
        raise ValueError("tablebase_recheck_output_already_exists")
    cohort = _json(output_dir / "cohort.json")
    evidence = _json(output_dir / "evidence.json")
    cohort_hash = cohort.pop("cohort_sha256")
    evidence_hash = evidence.pop("evidence_sha256")
    if hashlib.sha256(_stable(cohort)).hexdigest() != cohort_hash:
        raise ValueError("cohort_hash_mismatch")
    if hashlib.sha256(_stable(evidence)).hexdigest() != evidence_hash:
        raise ValueError("evidence_hash_mismatch")
    claims = [
        (root, result, int(move), action)
        for root, result in zip(cohort["roots"], evidence["roots"], strict=True)
        for move, action in result["actions"].items()
        if action["exact"]
    ]
    tablebase = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
    binary = ROOT / "model-artifact/runtime/kalah_v1_tablebase"
    if _sha(tablebase) != TABLEBASE_SHA256 or _sha(binary) != TABLEBASE_BINARY_SHA256:
        raise ValueError("authenticated_tablebase_hash_mismatch")
    checked = []
    with DeadlineNativeExactRootTablebase(binary, tablebase) as native:
        for root, _old_root, move, old_action in claims:
            solver = TablebaseBoundedSolver(
                native, tt_size=root["tt_entries"], move_order="ascending"
            )
            started = time.monotonic()
            tablebase_calls_before = native.calls
            try:
                margin = solver.solve(
                    ExactState.from_game_state(root["state"]).play(move),
                    time_limit_seconds=root["per_action_seconds"],
                )
                checked.append(
                    {
                        "root_index": root["index"],
                        "row_id": root["row_id"],
                        "move": move,
                        "old_margin": old_action["lower"],
                        "rechecked_margin": margin,
                        "matches": margin == old_action["lower"],
                        "status": "exact",
                        "nodes": solver.nodes,
                        "tablebase_calls": native.calls - tablebase_calls_before,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
            except SearchTimeout:
                checked.append(
                    {
                        "root_index": root["index"],
                        "row_id": root["row_id"],
                        "move": move,
                        "old_margin": old_action["lower"],
                        "rechecked_margin": None,
                        "matches": False,
                        "status": "unresolved",
                        "nodes": solver.nodes,
                        "tablebase_calls": native.calls - tablebase_calls_before,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
            finally:
                solver.close()
    recheck = {
        "schema": "exact_tactical_label_tablebase_recheck_v1",
        "cohort_sha256": cohort_hash,
        "original_evidence_sha256": evidence_hash,
        "recheck_source_sha256": {
            "diagnostic": _sha(Path(__file__).resolve()),
            "exact_kalah_solver": _sha(
                ROOT / "ml/alphazero_lite/exact_kalah_solver.py"
            ),
            "native_root_tablebase_adapter": _sha(
                ROOT / "ml/alphazero_lite/native_exact_root_tablebase.py"
            ),
        },
        "claims": checked,
    }
    recheck["recheck_sha256"] = hashlib.sha256(_stable(recheck)).hexdigest()
    recheck_path.write_bytes(
        json.dumps(recheck, sort_keys=True, indent=2).encode() + b"\n"
    )
    return recheck


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
            sign = 1 if int(root["state"]["current_player"]) == 0 else -1
            if actual is None or best is None or sign * actual >= sign * best:
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
    parser.add_argument(
        "command", choices=("freeze", "solve", "recheck-exact-claims", "verify")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="required new output directory for freeze, solve, and recheck commands",
    )
    args = parser.parse_args()
    if args.command == "verify":
        if args.output_dir is not None:
            parser.error(
                "verify reads the pinned publication and accepts no output directory"
            )
        print(json.dumps(verify(), indent=2, sort_keys=True))
        return
    if args.output_dir is None:
        parser.error(f"{args.command} requires --output-dir with a new run directory")
    result = {
        "freeze": freeze,
        "solve": solve,
        "recheck-exact-claims": recheck_exact_claims,
    }[args.command](args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
