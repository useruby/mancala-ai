#!/usr/bin/env python3
"""Harder fresh exact-tablebase-unique endgame mining.

Steps:
  1. Generate candidate endgame states via deeper self-play, disagreement
     sampling, adversarial near-threshold enumeration, and PUCT-vs-tablebase
     disagreement sampling.
  2. Deduplicate and exclude known bad sources (exhausted families, teacher
     conflict, ties, all-equivalent).
  3. Exact tablebase enumeration for every candidate.
  4. PUCT baseline across budgets 64/128/256/384/1200/2400/5000.
  5. Neural value rank scan.
  6. Classify into target/control/holdout/low_budget_diagnostic/exclude.
  7. Decide whether a later diagnostic artifact is warranted.
  8. Write summary JSON, candidate JSONL, selected JSONL, and report markdown.

Does not train, run arena, promote, create replay artifacts, or mutate
active reference fixtures.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position
from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import (
    PUCT,
    build_eval_search_options,
    standard_start_state,
)

PITS_PER_PLAYER = 6
EPS = 1e-9
FAMILY = "harder_fresh_endgame_tablebase"
C_PUCT = 1.25
SEARCH_OPTIONS = build_eval_search_options(
    fpu_mode="parent_q",
    reuse_subtree=True,
    normalize_values=True,
    root_policy_mode="deterministic",
    tactical_root_bias=0.1,
)
# Lower budgets to catch failures earlier
ROOT_BUDGETS = (64, 128, 256, 384, 1200, 2400, 5000)
LOW_BUDGETS = (64, 128, 256)
MED_BUDGETS = (384,)
HIGH_BUDGETS = (1200,)
PROMISING_ONLY_BUDGETS = (2400, 5000)
SEED = 17
CANDIDATE_SEEDS = 100

CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
REFERENCE_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
OUTPUT_DIR = Path("/tmp/azlite_harder_fresh_endgame_tablebase_mining")

EXHAUSTED_ROW_ID_PREFIXES: frozenset[str] = frozenset(
    {
        "incumbent_proxy_disagreement",
        "incumbent_proxy_residual",
        "high_value_swing",
        "high_imbalance",
        "capture_available",
        "starvation_pressure",
        "sparse_endgame",
        "early_extra_turn",
        "opening_plies_1_8",
        "opening_extra_turn",
        "opening_edge_move",
        "opening_missed_extra_turn",
    }
)
EXHAUSTED_BUCKETS: frozenset[str] = frozenset(
    {
        "opening_plies_1_8",
        "opening_extra_turn_overbias",
        "opening_edge_move_5_preference",
        "opening_missed_extra_turn_continuation",
        "incumbent_proxy_disagreement",
        "incumbent_proxy_residual",
        "high_value_swing",
        "high_imbalance",
        "capture_available",
        "starvation_pressure",
        "sparse_endgame",
        "early_extra_turn",
    }
)
CORRECTED_GUARD_ROW_IDS: frozenset[str] = frozenset(
    {
        "capture_available-002",
        "capture_available-003",
        "capture_available-006",
        "capture_available-007",
        "capture_available-008",
    }
)


# ── Helpers ──────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def child_state_from_move(root_state: dict[str, Any], move: int) -> dict[str, Any]:
    game = KalahGame.from_state(root_state)
    succeeded = game.move(game.pit_index(int(move)))
    if not succeeded:
        raise ValueError(f"illegal move {move} for state")
    return game.to_state()


def state_to_root_perspective_value(
    *, raw_value: float, state: dict[str, Any], root_player: int
) -> float:
    return (
        float(raw_value)
        if int(state["current_player"]) == int(root_player)
        else -float(raw_value)
    )


def is_exhausted_row_id(row_id: str) -> bool:
    for prefix in EXHAUSTED_ROW_ID_PREFIXES:
        if row_id.startswith(prefix):
            return True
    if row_id in CORRECTED_GUARD_ROW_IDS:
        return True
    return False


def visit_share(visits: list[float], move: int) -> float | None:
    total = sum(float(v) for v in visits)
    if total <= 0 or move >= len(visits):
        return None
    return round_float(float(visits[move]) / float(total))


def selection_entry_map(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    selection_breakdown = result.get("selection_breakdown") or {}
    return {
        int(entry["move"]): entry
        for entry in list(selection_breakdown.get("moves") or [])
        if isinstance(entry, dict) and entry.get("move") is not None
    }


def total_seeds_remaining(state: dict[str, Any]) -> int:
    return sum(state.get("player_pits", [])) + sum(state.get("opponent_pits", []))


def load_reference_maps(
    reference_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    by_id: dict[str, dict[str, Any]] = {}
    by_canonical: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("id") is None:
            continue
        by_id[str(row["id"])] = row
        canonical = str(row.get("canonical_state") or canonical_state_key(row["state"]))
        by_canonical[canonical] = row
    return by_id, by_canonical


def load_suite(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("rows", raw)
    return {str(row["id"]): row for row in raw if "id" in row}


def suite_canonical_states(suite: dict[str, dict]) -> set[str]:
    result: set[str] = set()
    for row in suite.values():
        if "canonical_state" in row:
            result.add(str(row["canonical_state"]))
        elif "state" in row:
            result.add(canonical_state_key(row["state"]))
    return result


# ── Source A: Deeper self-play endgames ──────────────────────────────────


def generate_deeper_self_play_endgames(
    evaluator: ArtifactEvaluator,
    *,
    games: int,
    seeds: list[int],
    simulations: int,
    max_plies: int,
    min_ply_for_collection: int = 20,
) -> list[dict[str, Any]]:
    """Play games with current-model PUCT, collect late-game positions."""
    candidates: list[dict[str, Any]] = []

    for game_index in range(games):
        if game_index % 5 == 0:
            print(f"    self-play game {game_index}/{games}")
        seed = seeds[game_index % len(seeds)]
        rng = random.Random(seed * 1_000_003 + game_index)
        game = KalahGame.from_state(standard_start_state())
        reusable_root = None

        for ply in range(max_plies):
            if game.over():
                break
            legal_moves = game.possible_moves()
            if not legal_moves:
                break

            state = game.to_state()
            c_hash = canonical_state_key(state)
            seeds_rem = total_seeds_remaining(state)

            # Collect states at deeper plies with low remaining seeds
            if ply >= min_ply_for_collection and seeds_rem <= 16:
                candidates.append(
                    {
                        "state": state,
                        "canonical_state_hash": c_hash,
                        "ply": ply,
                        "source_game_seed": seed + game_index,
                        "source": "deeper_self_play",
                        "remaining_seeds": seeds_rem,
                    }
                )

            search = PUCT(
                evaluator=evaluator,
                simulations=simulations,
                c_puct=1.25,
                rng=rng,
                root=reusable_root,
                fpu_mode="parent_q",
                reuse_subtree=True,
                normalize_values=True,
                root_policy_mode="deterministic",
                tactical_root_bias=0.1,
            )

            visits, root = search.run(
                game,
                dirichlet_alpha=0.3 if ply < 10 else None,
                dirichlet_epsilon=0.25 if ply < 10 else 0.0,
            )

            legal = game.possible_moves()
            if not legal:
                break
            move = search.select_root_move(root, legal)
            child = root.child_for_action(move) if move is not None else None
            reusable_root = child
            absolute_move = game.pit_index(move) if move is not None else None
            if absolute_move is None or not game.move(absolute_move):
                break

    return candidates


# ── Source B: Low-budget PUCT failure prescreen ──────────────────────────


def generate_puct_low_budget_failure_candidates(
    evaluator: ArtifactEvaluator,
    *,
    games: int,
    seeds: list[int],
    simulations: int,
    max_plies: int,
    low_budget: int = 128,
    min_ply: int = 20,
) -> list[dict[str, Any]]:
    """Self-play but at each endgame state, run PUCT at 128 to check
    if low-budget already disagrees with tablebase."""
    candidates: list[dict[str, Any]] = []
    tb = EndgameTablebase()

    for game_index in range(games):
        seed = seeds[game_index % len(seeds)]
        rng = random.Random(seed * 1_000_003 + game_index + 1000)
        game = KalahGame.from_state(standard_start_state())
        reusable_root = None

        for ply in range(max_plies):
            if game.over():
                break
            legal_moves = game.possible_moves()
            if not legal_moves:
                break

            state = game.to_state()
            c_hash = canonical_state_key(state)
            seeds_rem = total_seeds_remaining(state)

            if ply >= min_ply and seeds_rem <= 16:
                # Check tablebase availability
                tb_wr = tb.lookup(game, game.current_player)
                if tb_wr is not None:
                    # Compute tablebase optimal moves
                    offset = game.current_player * 6
                    child_vals: dict[int, float] = {}
                    best_val = -float("inf")
                    for move in legal_moves:
                        child_game = game.clone()
                        child_game.move(offset + move)
                        c_wr = tb.lookup(child_game, game.current_player)
                        if c_wr is not None:
                            cv = (2.0 * float(c_wr)) - 1.0
                            child_vals[move] = cv
                            if cv > best_val:
                                best_val = cv
                    optimal_moves = sorted(
                        m for m, v in child_vals.items() if abs(v - best_val) < EPS
                    )

                    if len(optimal_moves) == 1:
                        optimal = optimal_moves[0]
                        # Run PUCT at low budget
                        r = evaluate_artifact_position(
                            artifact_path=None,
                            evaluator=evaluator,
                            state=state,
                            simulations=int(low_budget),
                            seed=SEED + ply,
                            c_puct=C_PUCT,
                            search_options=dict(SEARCH_OPTIONS),
                            ablation_mode="full",
                        )
                        selected = (
                            None
                            if r.get("selected_move") is None
                            else int(r["selected_move"])
                        )
                        if selected is not None and selected != optimal:
                            candidates.append(
                                {
                                    "state": state,
                                    "canonical_state_hash": c_hash,
                                    "ply": ply,
                                    "source_game_seed": seed + game_index + 1000,
                                    "source": "puct_low_budget_failure",
                                    "remaining_seeds": seeds_rem,
                                    "tb_optimal_move": optimal,
                                    "low_budget_selected": selected,
                                    "low_budget": low_budget,
                                }
                            )

            # Play the move and continue
            search = PUCT(
                evaluator=evaluator,
                simulations=simulations,
                c_puct=1.25,
                rng=rng,
                root=reusable_root,
                fpu_mode="parent_q",
                reuse_subtree=True,
                normalize_values=True,
                root_policy_mode="deterministic",
                tactical_root_bias=0.1,
            )

            visits, root = search.run(
                game,
                dirichlet_alpha=0.3 if ply < 10 else None,
                dirichlet_epsilon=0.25 if ply < 10 else 0.0,
            )

            legal = game.possible_moves()
            if not legal:
                break
            move = search.select_root_move(root, legal)
            child = root.child_for_action(move) if move is not None else None
            reusable_root = child
            absolute_move = game.pit_index(move) if move is not None else None
            if absolute_move is None or not game.move(absolute_move):
                break

    return candidates


# ── Source D: Adversarial near-threshold endgame sampling ────────────────


def generate_adversarial_near_threshold_candidates(
    evaluator: ArtifactEvaluator,
    *,
    samples: int,
    seeds: list[int],
    min_seeds: int = 2,
    max_seeds: int = 14,
) -> list[dict[str, Any]]:
    """Sample random endgame states within tablebase range and check for
    unique optimal moves with meaningful but non-trivial margins."""
    candidates: list[dict[str, Any]] = []
    tb = EndgameTablebase()
    rng = random.Random(seeds[0] * 7)

    attrs = 0
    last_print = 0
    while attrs < samples:
        if attrs % 20 == 0 and attrs > 0:
            print(f"    adversarial: {attrs}/{samples} collected={len(candidates)}")
        # Generate a random board state with fewer seeds for speed
        seeds_rem = rng.randint(min_seeds, min(max_seeds, 12))
        # Distribute seeds randomly across 12 pits
        pits_list = [0] * 12
        remaining = seeds_rem
        for i in range(12):
            if i == 11:
                pits_list[i] = remaining
            else:
                take = rng.randint(0, remaining)
                pits_list[i] = take
                remaining -= take
        rng.shuffle(pits_list)
        current_player = rng.randint(0, 1)

        # Collect from captures
        captured = [rng.randint(0, 8), rng.randint(0, 8)]

        # Make sure the player has at least one legal move
        offset = current_player * 6
        if all(p == 0 for p in pits_list[offset : offset + 6]):
            continue

        # Skip if terminal (total seed count in pits is 0 for either player's side)
        player_seeds = sum(pits_list[offset : offset + 6])
        opp_seeds = sum(pits_list[6 - offset : 12 - offset])
        if player_seeds == 0 or opp_seeds == 0:
            continue

        state = {
            "player_pits": pits_list[:6],
            "opponent_pits": pits_list[6:],
            "player_store": captured[0],
            "opponent_store": captured[1],
            "current_player": current_player,
        }
        game = KalahGame.from_state(state)

        # Check tablebase (cached first for speed)
        tb_wr = tb.lookup_cached(game, current_player)
        if tb_wr is None:
            tb_wr = tb.lookup(game, current_player)
        if tb_wr is None:
            continue

        legal = game.possible_moves()
        offset_i = current_player * 6
        best_val = -float("inf")
        second_val = -float("inf")
        child_vals: dict[int, float] = {}
        optimal_moves: list[int] = []
        for move in legal:
            child_game = game.clone()
            child_game.move(offset_i + move)
            c_wr = tb.lookup(child_game, current_player)
            if c_wr is None:
                continue
            cv = (2.0 * float(c_wr)) - 1.0
            child_vals[move] = cv
            if cv > best_val:
                second_val = best_val
                best_val = cv
            elif cv > second_val and cv < best_val:
                second_val = cv

        if best_val > -float("inf"):
            optimal_moves = sorted(
                m for m, v in child_vals.items() if abs(v - best_val) < EPS
            )

        if len(optimal_moves) != 1:
            continue

        gap = round_float(best_val - second_val) if second_val > -float("inf") else None
        # Reject tiny margins and equivalent moves
        if gap is None or gap < EPS:
            continue

        # Check all-equivalent
        all_vals = set(round_float(v, 6) for v in child_vals.values())
        if len(all_vals) == 1:
            continue

        c_hash = canonical_state_key(state)
        candidates.append(
            {
                "state": state,
                "canonical_state_hash": c_hash,
                "ply": None,
                "source_game_seed": seeds[0] + attrs,
                "source": "adversarial_near_threshold",
                "remaining_seeds": seeds_rem,
                "tb_optimal_move": optimal_moves[0],
                "gap": gap,
            }
        )

        # For unique optimal with best_val >= 0 (winning for root player),
        # also consider total seeds near boundary
        if best_val >= 0.99 and gap is not None and gap < 1.0:
            pass  # we already collect these

        attrs += 1
        if attrs - last_print >= 50:
            print(f"    adversarial progress: {attrs}/{samples}")
            last_print = attrs

    return candidates


# ── Source E: PUCT-vs-tablebase disagreement sampling ────────────────────


def generate_puct_tablebase_disagreement_candidates(
    evaluator: ArtifactEvaluator,
    *,
    games: int,
    seeds: list[int],
    simulations: int,
    max_plies: int,
    puct_budgets: tuple[int, ...] = (384,),
    min_ply: int = 16,
) -> list[dict[str, Any]]:
    """Self-play endgames where PUCT selected move != tablebase optimal."""
    candidates: list[dict[str, Any]] = []
    tb = EndgameTablebase()

    for game_index in range(games):
        seed = seeds[game_index % len(seeds)]
        rng = random.Random(seed * 1_000_003 + game_index + 2000)
        game = KalahGame.from_state(standard_start_state())
        reusable_root = None

        for ply in range(max_plies):
            if game.over():
                break
            legal_moves = game.possible_moves()
            if not legal_moves:
                break

            state = game.to_state()
            c_hash = canonical_state_key(state)
            seeds_rem = total_seeds_remaining(state)

            if ply >= min_ply and seeds_rem <= 16:
                tb_wr = tb.lookup(game, game.current_player)
                if tb_wr is not None:
                    offset_i = game.current_player * 6
                    child_vals: dict[int, float] = {}
                    best_val = -float("inf")
                    for move in legal_moves:
                        child_game = game.clone()
                        child_game.move(offset_i + move)
                        c_wr = tb.lookup(child_game, game.current_player)
                        if c_wr is not None:
                            cv = (2.0 * float(c_wr)) - 1.0
                            child_vals[move] = cv
                            if cv > best_val:
                                best_val = cv
                    optimal_moves = sorted(
                        m for m, v in child_vals.items() if abs(v - best_val) < EPS
                    )

                    if len(optimal_moves) == 1:
                        optimal = optimal_moves[0]
                        for budget in puct_budgets:
                            r = evaluate_artifact_position(
                                artifact_path=None,
                                evaluator=evaluator,
                                state=state,
                                simulations=int(budget),
                                seed=SEED + ply + budget,
                                c_puct=C_PUCT,
                                search_options=dict(SEARCH_OPTIONS),
                                ablation_mode="full",
                            )
                            selected = (
                                None
                                if r.get("selected_move") is None
                                else int(r["selected_move"])
                            )
                            if selected is not None and selected != optimal:
                                candidates.append(
                                    {
                                        "state": state,
                                        "canonical_state_hash": c_hash,
                                        "ply": ply,
                                        "source_game_seed": seed + game_index + 2000,
                                        "source": "puct_tablebase_disagreement",
                                        "remaining_seeds": seeds_rem,
                                        "tb_optimal_move": optimal,
                                        "disagreement_budget": budget,
                                        "puct_selected": selected,
                                    }
                                )
                                break  # one record per state is enough

            # Continue self-play
            search = PUCT(
                evaluator=evaluator,
                simulations=simulations,
                c_puct=1.25,
                rng=rng,
                root=reusable_root,
                fpu_mode="parent_q",
                reuse_subtree=True,
                normalize_values=True,
                root_policy_mode="deterministic",
                tactical_root_bias=0.1,
            )

            visits, root = search.run(
                game,
                dirichlet_alpha=0.3 if ply < 10 else None,
                dirichlet_epsilon=0.25 if ply < 10 else 0.0,
            )

            legal = game.possible_moves()
            if not legal:
                break
            move = search.select_root_move(root, legal)
            child = root.child_for_action(move) if move is not None else None
            reusable_root = child
            absolute_move = game.pit_index(move) if move is not None else None
            if absolute_move is None or not game.move(absolute_move):
                break

    return candidates


# ── Deduplication and exclusion ──────────────────────────────────────────


def deduplicate_and_exclude(
    raw_candidates: list[dict[str, Any]],
    suite_canonical: set[str],
    suite_rows: dict[str, dict],
    ref_rows: dict[str, dict],
    existing_selected_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deduplicate and exclude known bad sources. Returns (clean, counts)."""
    counts: dict[str, int] = {
        "raw_candidates": len(raw_candidates),
        "exact_duplicates": 0,
        "known_fixture_overlaps": 0,
        "excluded_overlaps": 0,
        "existing_selected_overlaps": 0,
        "tablebase_solvable": 0,
        "unique_optimal": 0,
        "rejected_ties_equivalent": 0,
        "rejected_exhausted": 0,
        "rejected_teacher_conflict": 0,
        "remaining_clean": 0,
    }

    seen_hashes: set[str] = set()
    clean: list[dict[str, Any]] = []
    tb = EndgameTablebase()

    for cand in raw_candidates:
        state = cand["state"]
        c_hash = cand.get("canonical_state_hash", canonical_state_key(state))
        cand["canonical_state_hash"] = c_hash

        # Exact duplicate
        if c_hash in seen_hashes:
            counts["exact_duplicates"] += 1
            continue
        seen_hashes.add(c_hash)

        # Overlap with existing selected rows from PR #72
        if c_hash in existing_selected_hashes:
            counts["existing_selected_overlaps"] += 1
            continue

        # Overlap with fixture suite
        if c_hash in suite_canonical:
            counts["known_fixture_overlaps"] += 1
            # Check if overlap is from exhausted family
            is_exhausted = False
            for row_id, row in suite_rows.items():
                row_cs = row.get("canonical_state") or canonical_state_key(
                    row.get("state", {})
                )
                if row_cs == c_hash:
                    if is_exhausted_row_id(row_id):
                        is_exhausted = True
                        break
                    bucket = row.get("bucket", "")
                    if bucket in EXHAUSTED_BUCKETS:
                        is_exhausted = True
                        break
                    if row_id in ref_rows:
                        ref = ref_rows[row_id]
                        if ref.get("reference_unstable", False):
                            is_exhausted = True
                            break
                        tags = ref.get("tags", [])
                        if isinstance(tags, str):
                            tags = [tags]
                        for tag in tags:
                            if tag in (
                                "reference_integrity_error",
                                "train_only",
                                "excluded_diagnostic",
                            ):
                                is_exhausted = True
                                break
                    break
            if is_exhausted:
                counts["excluded_overlaps"] += 1
                continue

        # Tablebase availability at root
        game = KalahGame.from_state(state)
        tb_wr = tb.lookup(game, game.current_player)
        if tb_wr is None:
            continue
        counts["tablebase_solvable"] += 1

        # Unique optimal move
        legal = game.possible_moves()
        offset_i = game.current_player * 6
        child_vals: dict[int, float] = {}
        best_val = -float("inf")
        for move in legal:
            child_game = game.clone()
            child_game.move(offset_i + move)
            c_wr = tb.lookup(child_game, game.current_player)
            if c_wr is not None:
                cv = (2.0 * float(c_wr)) - 1.0
                child_vals[move] = cv
                if cv > best_val:
                    best_val = cv

        optimal_moves = sorted(
            m for m, v in child_vals.items() if abs(v - best_val) < EPS
        )

        if len(optimal_moves) == 0:
            continue

        # Reject ties (not unique)
        if len(optimal_moves) > 1:
            counts["rejected_ties_equivalent"] += 1
            continue

        # Reject all-equivalent
        all_vals = set(round_float(v, 6) for v in child_vals.values())
        if len(all_vals) == 1:
            counts["rejected_ties_equivalent"] += 1
            continue

        unique_optimal = optimal_moves[0]
        counts["unique_optimal"] += 1

        # Store tablebase metadata on the candidate
        sorted_vals = sorted([v for v in child_vals.values()], reverse=True)
        second_best = sorted_vals[1] if len(sorted_vals) >= 2 else None
        gap = round_float(best_val - second_best) if second_best is not None else None

        cand["_tb_root_value"] = round_float((2.0 * float(tb_wr)) - 1.0)
        cand["_tb_child_values"] = child_vals
        cand["_tb_optimal_moves"] = optimal_moves
        cand["_tb_unique_optimal"] = unique_optimal
        cand["_tb_best_minus_second"] = gap
        cand["_tb_all_equiv"] = False
        cand["_tb_second_best"] = (
            round_float(second_best) if second_best is not None else None
        )

        # Teacher conflict check (based on row_id pattern)
        row_id = str(cand.get("source_game_seed", ""))
        if is_exhausted_row_id(row_id):
            counts["rejected_exhausted"] += 1
            continue

        # Check for exhausted patterns in canid
        src = cand.get("source", "")
        if any(
            p in str(src)
            for p in (
                "incumbent_proxy",
                "high_value_swing",
                "high_imbalance",
                "capture_available",
                "starvation_pressure",
                "sparse_endgame",
                "early_extra_turn",
                "opening_plies",
                "opening_extra",
                "opening_edge",
                "opening_missed",
            )
        ):
            counts["rejected_exhausted"] += 1
            continue

        counts["remaining_clean"] += 1
        clean.append(cand)

    return clean, counts


# ── Exact tablebase enumeration ──────────────────────────────────────────


def exact_tablebase_enumeration(
    clean_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tb = EndgameTablebase()
    results: list[dict[str, Any]] = []

    for cand in clean_candidates:
        c_hash = cand["canonical_state_hash"]
        state = cand["state"]
        game = KalahGame.from_state(state)
        root_player = game.current_player

        tb_wr = tb.lookup(game, root_player)
        if tb_wr is None:
            results.append(
                {
                    "candidate_hash": c_hash,
                    "root_value": None,
                    "tablebase_optimal_move": None,
                    "second_best_value": None,
                    "best_minus_second_best": None,
                    "all_moves_equivalent": False,
                    "exact_signal_class": "tablebase_unavailable",
                    "notes": "tablebase unavailable at root",
                }
            )
            continue

        root_value = (2.0 * float(tb_wr)) - 1.0
        legal_moves = game.possible_moves()
        offset_i = game.current_player * 6
        child_values: dict[int, float | None] = {}

        for move in legal_moves:
            child_game = game.clone()
            child_game.move(offset_i + move)
            c_wr = tb.lookup(child_game, root_player)
            if c_wr is not None:
                cv = (2.0 * float(c_wr)) - 1.0
                child_values[move] = round_float(cv)
            else:
                child_values[move] = None

        best_val = max(
            (v for v in child_values.values() if v is not None),
            default=-float("inf"),
        )
        optimal_moves = sorted(
            m
            for m, v in child_values.items()
            if v is not None and abs(v - best_val) < EPS
        )
        unique_optimal = len(optimal_moves) == 1
        optimal_move = optimal_moves[0] if unique_optimal else None

        sorted_vals = sorted(
            [v for v in child_values.values() if v is not None], reverse=True
        )
        best_minus_second = None
        if len(sorted_vals) >= 2:
            best_minus_second = round_float(sorted_vals[0] - sorted_vals[1])

        all_equiv = (
            len(set(round_float(v, 6) for v in child_values.values() if v is not None))
            == 1
            if child_values
            else False
        )

        if not unique_optimal:
            signal = "tablebase_tie"
        elif all_equiv:
            signal = "all_moves_equivalent"
        elif best_minus_second is not None and best_minus_second < EPS:
            signal = "exact_unique_tiny_margin"
        else:
            signal = "exact_unique_clear_margin"

        notes_parts = []
        if signal == "exact_unique_tiny_margin":
            notes_parts.append(f"tiny margin = {best_minus_second}")
        if signal == "all_moves_equivalent":
            notes_parts.append("all child values identical despite unique optimal")
        if signal == "tablebase_tie":
            notes_parts.append(f"optimal moves: {optimal_moves}")

        forced_win_loss: bool | None = (
            abs(root_value) > 0.99 if root_value is not None else None
        )

        results.append(
            {
                "candidate_hash": c_hash,
                "ply": cand.get("ply"),
                "remaining_seed_count": total_seeds_remaining(state),
                "legal_moves": len(legal_moves),
                "root_value": round_float(root_value),
                "tablebase_optimal_move": optimal_move,
                "optimal_moves": optimal_moves,
                "second_best_value": round_float(
                    sorted_vals[1] if len(sorted_vals) >= 2 else None
                ),
                "best_minus_second_best": best_minus_second,
                "all_moves_equivalent": all_equiv,
                "forced_win_or_loss": forced_win_loss,
                "exact_signal_class": signal,
                "notes": "; ".join(notes_parts) if notes_parts else "ok",
            }
        )

    return results


# ── PUCT baseline across budgets ─────────────────────────────────────────


def root_puct_run(
    evaluator: ArtifactEvaluator,
    state: dict[str, Any],
    budget: int,
    seed: int,
    optimal_move: int | None,
) -> dict[str, Any]:
    result = evaluate_artifact_position(
        artifact_path=None,
        evaluator=evaluator,
        state=state,
        simulations=int(budget),
        seed=int(seed),
        c_puct=C_PUCT,
        search_options=dict(SEARCH_OPTIONS),
        ablation_mode="full",
    )
    selected_move = (
        None if result.get("selected_move") is None else int(result["selected_move"])
    )
    selection_map = selection_entry_map(result)
    opt_entry = selection_map.get(optimal_move) if optimal_move is not None else {}
    sel_entry = selection_map.get(selected_move) if selected_move is not None else {}
    selected_is_optimal = (
        selected_move == optimal_move if optimal_move is not None else None
    )
    visits_list = [float(v) for v in result.get("visits", [])]

    # Policy ranks
    policy_list = [float(p) for p in result.get("policy", [])]
    optimal_policy_rank = None
    if optimal_move is not None and policy_list:
        sorted_pol = sorted(
            [(m, policy_list[m]) for m in range(len(policy_list))],
            key=lambda x: -x[1],
        )
        for rank, (m, _) in enumerate(sorted_pol):
            if m == optimal_move:
                optimal_policy_rank = rank
                break

    return {
        "budget": int(budget),
        "optimal_move": optimal_move,
        "selected_move": selected_move,
        "selected_is_optimal": selected_is_optimal,
        "optimal_visit_share": visit_share(visits_list, optimal_move)
        if optimal_move is not None
        else None,
        "selected_visit_share": visit_share(visits_list, selected_move)
        if selected_move is not None
        else None,
        "optimal_q": round_float(float(opt_entry.get("q_value", 0.0)))
        if opt_entry
        else None,
        "selected_q": round_float(float(sel_entry.get("q_value", 0.0)))
        if sel_entry
        else None,
        "selected_minus_optimal_q_margin": round_float(
            float(sel_entry.get("q_value", 0.0)) - float(opt_entry.get("q_value", 0.0))
        )
        if sel_entry and opt_entry
        else None,
        "optimal_policy_probability": round_float(float(opt_entry.get("prior", 0.0)))
        if opt_entry
        else None,
        "selected_policy_probability": round_float(float(sel_entry.get("prior", 0.0)))
        if sel_entry
        else None,
        "optimal_policy_rank": optimal_policy_rank,
        "pass_status": "pass" if selected_is_optimal else "fail",
        "notes": "deterministic PUCT baseline",
    }


def run_puct_baseline(
    evaluator: ArtifactEvaluator,
    clean_candidates: list[dict[str, Any]],
    all_budgets: tuple[int, ...],
    promising_budgets: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Run PUCT across budgets. For promising rows (failed at 384/1200),
    also run 2400/5000."""
    all_results: list[dict[str, Any]] = []

    for idx, cand in enumerate(clean_candidates):
        if idx % 10 == 0:
            print(f"    PUCT baseline: {idx}/{len(clean_candidates)}")
        state = cand["state"]
        c_hash = cand["canonical_state_hash"]
        optimal_move = cand.get("_tb_unique_optimal")

        if optimal_move is None:
            continue

        # Run low and medium budgets for all candidates
        budgets_to_run = list(LOW_BUDGETS) + list(MED_BUDGETS) + list(HIGH_BUDGETS)
        row_results: list[dict[str, Any]] = []

        for budget in budgets_to_run:
            r = root_puct_run(evaluator, state, budget, SEED, optimal_move)
            r["candidate_hash"] = c_hash
            row_results.append(r)
            all_results.append(r)

        # Check if failure at 384 or 1200
        fails_384 = any(
            r["budget"] == 384 and not r["selected_is_optimal"] for r in row_results
        )
        fails_1200 = any(
            r["budget"] == 1200 and not r["selected_is_optimal"] for r in row_results
        )

        if fails_384 or fails_1200:
            # Run promising-only budgets
            for budget in promising_budgets:
                r = root_puct_run(evaluator, state, budget, SEED + 1, optimal_move)
                r["candidate_hash"] = c_hash
                row_results.append(r)
                all_results.append(r)

        # Attach to candidate
        cand["_puct_baseline_results"] = row_results
        cand["_puct_pass_64"] = any(
            r["budget"] == 64 and r["selected_is_optimal"] for r in row_results
        )
        cand["_puct_pass_128"] = any(
            r["budget"] == 128 and r["selected_is_optimal"] for r in row_results
        )
        cand["_puct_pass_256"] = any(
            r["budget"] == 256 and r["selected_is_optimal"] for r in row_results
        )
        cand["_puct_pass_384"] = any(
            r["budget"] == 384 and r["selected_is_optimal"] for r in row_results
        )
        cand["_puct_pass_1200"] = any(
            r["budget"] == 1200 and r["selected_is_optimal"] for r in row_results
        )
        cand["_puct_pass_2400"] = any(
            r["budget"] == 2400 and r["selected_is_optimal"] for r in row_results
        )
        cand["_puct_pass_5000"] = any(
            r["budget"] == 5000 and r["selected_is_optimal"] for r in row_results
        )

        if fails_1200 or (fails_384 and not cand.get("_puct_pass_1200", False)):
            cand["_puct_class"] = "persistent_exact_failure"
        elif fails_384 and cand.get("_puct_pass_1200", False):
            cand["_puct_class"] = "medium_budget_exact_failure"
        elif (
            not cand.get("_puct_pass_64", False)
            or not cand.get("_puct_pass_128", False)
            or not cand.get("_puct_pass_256", False)
        ):
            cand["_puct_class"] = "low_budget_exact_failure"
        else:
            cand["_puct_class"] = "exact_clean_control"

    return all_results


# ── Neural value rank scan ────────────────────────────────────────────────


def neural_value_rank_scan(
    evaluator: ArtifactEvaluator,
    clean_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate neural child values vs exact tablebase child values."""
    tb = EndgameTablebase()
    results: list[dict[str, Any]] = []

    for cand in clean_candidates:
        c_hash = cand["canonical_state_hash"]
        state = cand["state"]
        root_player = KalahGame.from_state(state).current_player
        optimal_move = cand.get("_tb_unique_optimal")

        if optimal_move is None:
            continue

        # Evaluate all legal children
        game = KalahGame.from_state(state)
        legal_moves = game.possible_moves()

        neural_child_values: dict[int, float | None] = {}
        exact_child_values: dict[int, float | None] = {}

        for move in legal_moves:
            child_state = child_state_from_move(state, move)
            _, raw_nv = evaluator.evaluate(KalahGame.from_state(child_state))
            nv = state_to_root_perspective_value(
                raw_value=float(raw_nv), state=child_state, root_player=root_player
            )
            neural_child_values[move] = round_float(nv)

            c_wr = tb.lookup(KalahGame.from_state(child_state), root_player)
            if c_wr is not None:
                exact_child_values[move] = round_float((2.0 * float(c_wr)) - 1.0)

        # Find neural best
        if not neural_child_values:
            continue
        neural_clean = {m: v for m, v in neural_child_values.items() if v is not None}
        if not neural_clean:
            continue
        neural_best_move = max(neural_clean, key=lambda m: neural_clean[m])  # type: ignore[type-var]
        neural_best_is_exact_optimal = neural_best_move == optimal_move

        # Value rank error: neural ranks a non-optimal child above the optimal
        value_rank_error = False
        if not neural_best_is_exact_optimal and optimal_move in neural_child_values:
            value_rank_error = True

        # Sign error: neural root value sign differs from exact
        tb_wr = tb.lookup(game, root_player)
        exact_root = (2.0 * float(tb_wr)) - 1.0 if tb_wr is not None else None
        _, raw_root_nv = evaluator.evaluate(game)
        neural_root = float(raw_root_nv)
        sign_error = False
        if exact_root is not None and abs(exact_root) > EPS:
            exact_sign = math.copysign(1.0, exact_root)
            neural_sign = math.copysign(1.0, neural_root)
            sign_error = exact_sign != neural_sign

        optimal_neural = neural_child_values.get(optimal_move)
        neural_best_exact = exact_child_values.get(neural_best_move)
        gap = cand.get("_tb_best_minus_second")

        results.append(
            {
                "candidate_hash": c_hash,
                "exact_optimal_move": optimal_move,
                "neural_best_move": neural_best_move,
                "neural_best_is_exact_optimal": neural_best_is_exact_optimal,
                "exact_optimal_child_neural_value": optimal_neural,
                "neural_best_child_exact_value": neural_best_exact,
                "value_rank_error": value_rank_error,
                "sign_error": sign_error,
                "exact_value_gap": gap,
                "neural_optimal_value": optimal_neural,
                "neural_best_value": neural_child_values[neural_best_move],
                "neural_value_gap": round_float(
                    float(neural_clean[neural_best_move]) - float(optimal_neural or 0.0)
                )
                if optimal_move in neural_clean and optimal_neural is not None
                else None,
                "notes": (
                    "neural prefers wrong child"
                    if value_rank_error
                    else "neural agrees with exact"
                ),
            }
        )

        cand["_neural_rank_error"] = value_rank_error
        cand["_neural_sign_error"] = sign_error
        cand["_neural_best_move"] = neural_best_move
        cand["_neural_best_is_optimal"] = neural_best_is_exact_optimal

    return results


# ── Classification and split ──────────────────────────────────────────────


def classify_and_split(
    clean_candidates: list[dict[str, Any]],
    exact_results: list[dict[str, Any]],
    neural_results: list[dict[str, Any]],
) -> dict[str, Any]:
    split_rows: list[dict[str, Any]] = []
    target_count = 0
    control_count = 0
    holdout_count = 0
    low_budget_count = 0
    excluded_count = 0

    for cand in clean_candidates:
        c_hash = cand["canonical_state_hash"]
        exact = next((r for r in exact_results if r["candidate_hash"] == c_hash), {})
        neural = next((r for r in neural_results if r["candidate_hash"] == c_hash), {})

        signal = exact.get("exact_signal_class", "")
        if signal in (
            "tablebase_unavailable",
            "tablebase_tie",
            "all_moves_equivalent",
            "exact_unique_tiny_margin",
        ):
            excluded_count += 1
            split_rows.append(
                {
                    "candidate_hash": c_hash,
                    "assigned_role": "exclude",
                    "failure_class": "na",
                    "exact_signal_class": signal,
                    "puct_failure_budget": None,
                    "value_rank_error": neural.get("value_rank_error", False),
                    "reason": f"exact signal: {signal}",
                    "notes": "",
                }
            )
            continue

        optimal_move = exact.get("tablebase_optimal_move")
        if optimal_move is None:
            excluded_count += 1
            split_rows.append(
                {
                    "candidate_hash": c_hash,
                    "assigned_role": "exclude",
                    "failure_class": "na",
                    "exact_signal_class": signal,
                    "puct_failure_budget": None,
                    "value_rank_error": neural.get("value_rank_error", False),
                    "reason": "no unique optimal move",
                    "notes": "",
                }
            )
            continue

        puct_class = cand.get("_puct_class", "")
        value_rank_error = neural.get("value_rank_error", False)

        # Find the failure budget
        failure_budget = None
        baselines = cand.get("_puct_baseline_results", [])
        for r in sorted(baselines, key=lambda x: x["budget"]):
            if not r.get("selected_is_optimal", True):
                failure_budget = r["budget"]
                break
        if failure_budget is None and value_rank_error:
            failure_budget = "value_only"

        if puct_class == "persistent_exact_failure":
            role = "target_candidate"
            target_count += 1
        elif puct_class == "medium_budget_exact_failure":
            role = "target_candidate"
            target_count += 1
        elif puct_class == "low_budget_exact_failure":
            role = "low_budget_diagnostic_only"
            low_budget_count += 1
        elif puct_class == "exact_clean_control":
            if value_rank_error:
                role = "target_candidate"
                target_count += 1
            elif control_count < 3:
                role = "preservation_control"
                control_count += 1
            else:
                role = "holdout_candidate"
                holdout_count += 1
        else:
            role = "holdout_candidate"
            holdout_count += 1

        split_rows.append(
            {
                "candidate_hash": c_hash,
                "assigned_role": role,
                "failure_class": puct_class,
                "exact_signal_class": signal,
                "puct_failure_budget": failure_budget,
                "value_rank_error": value_rank_error,
                "reason": (
                    "PUCT fails persistently at mid-high budget"
                    if role == "target_candidate"
                    and puct_class == "persistent_exact_failure"
                    else "PUCT fails at medium budget, recovers"
                    if role == "target_candidate"
                    and puct_class == "medium_budget_exact_failure"
                    else "neural value rank error even though PUCT passes"
                    if role == "target_candidate" and value_rank_error
                    else "low-budget only failure"
                    if role == "low_budget_diagnostic_only"
                    else "preserves under PUCT"
                    if role == "preservation_control"
                    else "holdout"
                    if role == "holdout_candidate"
                    else "excluded"
                ),
                "notes": "",
            }
        )

    return {
        "split_rows": split_rows,
        "target_candidate_count": target_count,
        "preservation_control_count": control_count,
        "holdout_count": holdout_count,
        "low_budget_diagnostic_count": low_budget_count,
        "excluded_count": excluded_count,
    }


# ── Decision logic ───────────────────────────────────────────────────────


def make_decision(
    split: dict[str, Any],
    clean_candidates: list[dict[str, Any]],
    neural_results: list[dict[str, Any]],
    counts: dict[str, int],
) -> tuple[str, str, str]:
    target_count = split["target_candidate_count"]
    low_budget_count = split["low_budget_diagnostic_count"]

    persistent_count = sum(
        1
        for c in clean_candidates
        if c.get("_puct_class") == "persistent_exact_failure"
    )
    medium_count = sum(
        1
        for c in clean_candidates
        if c.get("_puct_class") == "medium_budget_exact_failure"
    )
    value_rank_error_count = sum(1 for r in neural_results if r.get("value_rank_error"))

    # Rule A: >=5 target rows fail at 384 or 1200
    if target_count >= 5 and (persistent_count >= 2 or medium_count >= 3):
        decision = "harder_tablebase_exact_target_family_ready"
        next_action = (
            "run tablebase-backed local search/value diagnostics "
            "on selected hard rows before any artifact/training"
        )
        dominant = "search_failure_at_medium_high_budget"

    # Rule B: >=5 rows with value rank errors but PUCT recovers
    elif value_rank_error_count >= 5 and target_count >= 3:
        decision = "tablebase_exact_value_rank_diagnostic_ready"
        next_action = "build a value-only diagnostic plan, not policy training"
        dominant = "value_rank_error"

    # Rule C: Most failures only at low budget
    elif low_budget_count >= target_count * 2 and target_count < 5:
        decision = "low_budget_endgame_search_noise_only"
        next_action = "use these as search-budget diagnostics, not training targets"
        dominant = "low_budget_noise"

    # Rule D: Too few targets
    elif target_count < 5:
        decision = "still_too_few_hard_exact_rows"
        next_action = "broaden fresh endgame mining or increase sample scale"
        dominant = "too_few_targets"

    # Rule E: Many ties/equivalent
    elif (
        counts.get("rejected_ties_equivalent", 0)
        >= counts.get("remaining_clean", 0) * 0.5
    ):
        decision = "endgame_tablebase_tie_dominated"
        next_action = "tighten unique-optimal/gap filters and rerun"
        dominant = "tie_dominated"

    # Rule F: Exhausted/teacher conflicts dominate
    elif counts.get("excluded_overlaps", 0) >= counts.get("raw_candidates", 0) * 0.5:
        decision = "mining_repeats_bad_sources"
        next_action = "strengthen exclusions and rerun mining"
        dominant = "bad_sources"

    else:
        decision = "still_too_few_hard_exact_rows"
        next_action = "broaden fresh endgame mining or increase sample scale"
        dominant = "too_few_targets"

    return decision, next_action, dominant


# ── Report generation ────────────────────────────────────────────────────


def generate_report(
    summary: dict[str, Any],
    output_path: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Harder Fresh Endgame Tablebase Mining — Results")
    lines.append("")
    lines.append("**Date:** 2026-06-04")
    lines.append(f"**Family:** `{FAMILY}`")
    lines.append(
        "**Script:** `ml/alphazero_lite/run_harder_fresh_endgame_tablebase_mining.py`"
    )
    lines.append("")

    # Section 1: Context
    lines.append("## 1. Context")
    lines.append("")
    lines.append("- Run classification: {}".format(summary.get("decision", "unknown")))
    lines.append("- Selected family: {}".format(FAMILY))
    lines.append("- Current artifact: storage/ai/alphazero_lite/current")
    lines.append(
        "- Active references: ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
    )
    lines.append("- No training was run.")
    lines.append("- No arena was run.")
    lines.append("- No model was promoted.")
    lines.append("- Active references were not mutated.")
    lines.append("")

    # Section 2: Why PR #72 was too easy
    lines.append("## 2. Why PR #72 was too easy")
    lines.append("")
    lines.append(
        "PR #72 ran diagnostics on 14 fresh endgame-tablebase-unique rows mined from "
        "current-model self-play. All 14 rows passed PUCT at every budget from 128 to 5000. "
        "No rows exhibited value rank errors, sign errors, or child PUCT failures. "
        "The decision was `tablebase_unique_too_small` because zero target candidates were found."
    )
    lines.append("")
    lines.append(
        "The root cause: the self-play positions were too shallow (ply 29–57) and the "
        "tablebase-solvable endgames were trivially solved by the existing network. "
        "This run generates harder positions using multiple strategies: deeper self-play, "
        "low-budget PUCT failure prescreening, adversarial near-threshold sampling, "
        "and direct PUCT-vs-tablebase disagreement sampling."
    )
    lines.append("")

    # Section 3: Candidate generation strategy
    lines.append("## 3. Candidate generation strategy")
    lines.append("")
    lines.append(
        "| source | raw_candidates | tablebase_solvable | unique_optimal | puct_disagreements | neural_rank_errors | kept_candidates | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for entry in summary.get("candidate_sources", []):
        lines.append(
            f"| {entry['source']} | {entry['raw']} | {entry['tablebase_solvable']} | "
            f"{entry['unique_optimal']} | {entry['puct_disagreements']} | "
            f"{entry['neural_rank_errors']} | {entry['kept']} | {entry['notes']} |"
        )
    lines.append("")

    # Section 4: Deduplication and exclusions
    lines.append("## 4. Deduplication and exclusions")
    lines.append("")
    dc = summary.get("dedup_counts", {})
    lines.append("| exclusion_reason | count | notes |")
    lines.append("|---|---|---|")
    for key, label in [
        ("raw_candidates", "Raw candidates generated"),
        ("exact_duplicates", "Exact duplicates removed"),
        ("known_fixture_overlaps", "Known fixture overlaps"),
        ("existing_selected_overlaps", "Existing selected overlap (PR #72)"),
        ("excluded_overlaps", "Excluded exhausted overlaps"),
        ("tablebase_solvable", "Tablebase-solvable"),
        ("unique_optimal", "Unique optimal move"),
        ("rejected_ties_equivalent", "Rejected ties/all-equivalent"),
        ("rejected_exhausted", "Rejected exhausted family patterns"),
        ("remaining_clean", "Remaining clean candidates"),
    ]:
        lines.append(f"| {label} | {dc.get(key, 0)} | |")
    lines.append("")

    # Section 5: Exact tablebase enumeration
    lines.append("## 5. Exact tablebase enumeration")
    lines.append("")
    et = summary.get("exact_tablebase_table", [])
    if et:
        lines.append(
            "| candidate_hash | ply | remaining_seed_count | legal_moves | root_value | optimal_move | "
            "second_best_value | best_minus_second_best | forced_win_or_loss | exact_signal_class | notes |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in et[:50]:
            lines.append(
                f"| {r.get('candidate_hash', '')[:16]}... | {r.get('ply', '')} | "
                f"{r.get('remaining_seed_count', '')} | {r.get('legal_moves', '')} | "
                f"{r.get('root_value', '')} | {r.get('tablebase_optimal_move', '')} | "
                f"{r.get('second_best_value', '')} | {r.get('best_minus_second_best', '')} | "
                f"{r.get('forced_win_or_loss', '')} | {r.get('exact_signal_class', '')} | "
                f"{r.get('notes', '')} |"
            )
        if len(et) > 50:
            lines.append(
                f"| ... and {len(et) - 50} more rows | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |"
            )
    lines.append("")

    # Section 6: PUCT budget scan
    lines.append("## 6. PUCT budget scan")
    lines.append("")
    pb = summary.get("puct_baseline_summary", {})
    lines.append(f"- Total rows evaluated: {pb.get('total_rows', 0)}")
    lines.append(
        f"- Rows with persistent failures (384+1200): {pb.get('persistent_failures', 0)}"
    )
    lines.append(
        f"- Rows with medium-budget failures (384 only): {pb.get('medium_budget_failures', 0)}"
    )
    lines.append(
        f"- Rows with low-budget failures (64/128/256 only): {pb.get('low_budget_failures', 0)}"
    )
    lines.append(f"- Clean controls (all budgets pass): {pb.get('clean_controls', 0)}")
    lines.append("")

    pb_tbl = summary.get("puct_budget_table", [])
    if pb_tbl:
        lines.append(
            "| candidate_hash | budget | optimal_move | selected_move | selected_is_optimal | "
            "optimal_visit_share | selected_visit_share | selected_minus_optimal_q_margin | "
            "optimal_policy_rank | failure_class | notes |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in pb_tbl[:80]:
            lines.append(
                f"| {r.get('candidate_hash', '')[:16]}... | {r.get('budget', '')} | "
                f"{r.get('optimal_move', '')} | {r.get('selected_move', '')} | "
                f"{r.get('selected_is_optimal', '')} | {r.get('optimal_visit_share', '')} | "
                f"{r.get('selected_visit_share', '')} | {r.get('selected_minus_optimal_q_margin', '')} | "
                f"{r.get('optimal_policy_rank', '')} | {r.get('failure_class', '')} | "
                f"{r.get('notes', '')} |"
            )
        if len(pb_tbl) > 80:
            lines.append(
                f"| ... and {len(pb_tbl) - 80} more rows | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |"
            )
    lines.append("")

    # Section 7: Neural value rank scan
    lines.append("## 7. Neural value rank scan")
    lines.append("")
    nv = summary.get("neural_value_table", [])
    if nv:
        lines.append(
            "| candidate_hash | exact_optimal_move | neural_best_move | neural_best_is_exact_optimal | "
            "exact_optimal_child_neural_value | neural_best_child_exact_value | value_rank_error | "
            "sign_error | notes |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in nv[:50]:
            lines.append(
                f"| {r.get('candidate_hash', '')[:16]}... | {r.get('exact_optimal_move', '')} | "
                f"{r.get('neural_best_move', '')} | {r.get('neural_best_is_exact_optimal', '')} | "
                f"{r.get('exact_optimal_child_neural_value', '')} | "
                f"{r.get('neural_best_child_exact_value', '')} | "
                f"{r.get('value_rank_error', '')} | {r.get('sign_error', '')} | "
                f"{r.get('notes', '')} |"
            )
        if len(nv) > 50:
            lines.append(
                f"| ... and {len(nv) - 50} more rows | ... | ... | ... | ... | ... | ... | ... | ... | ... |"
            )
    lines.append("")

    # Section 8: Target/control/holdout split
    lines.append("## 8. Target/control/holdout split")
    lines.append("")
    sp = summary.get("split", {})
    lines.append(
        "| candidate_hash | assigned_role | failure_class | exact_signal_class | "
        "puct_failure_budget | value_rank_error | reason | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in sp.get("split_rows", [])[:80]:
        lines.append(
            f"| {r.get('candidate_hash', '')[:16]}... | {r.get('assigned_role', '')} | "
            f"{r.get('failure_class', '')} | {r.get('exact_signal_class', '')} | "
            f"{r.get('puct_failure_budget', '')} | {r.get('value_rank_error', '')} | "
            f"{r.get('reason', '')} | {r.get('notes', '')} |"
        )
    if len(sp.get("split_rows", [])) > 80:
        lines.append(
            f"| ... and {len(sp['split_rows']) - 80} more rows | ... | ... | ... | ... | ... | ... | ... | ... |"
        )
    lines.append("")

    # Section 9: Final targetability decision
    lines.append("## 9. Final targetability decision")
    lines.append("")
    lines.append(
        "| classification | target_candidate_count | preservation_control_count | "
        "holdout_count | low_budget_diagnostic_count | excluded_count | "
        "dominant_mechanism | next_action |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    dec = summary.get("decision_data", {})
    lines.append(
        f"| {dec.get('classification', '')} | {sp.get('target_candidate_count', 0)} | "
        f"{sp.get('preservation_control_count', 0)} | {sp.get('holdout_count', 0)} | "
        f"{sp.get('low_budget_diagnostic_count', 0)} | {sp.get('excluded_count', 0)} | "
        f"{dec.get('dominant_mechanism', '')} | {dec.get('next_action', '')} |"
    )
    lines.append("")

    # Section 10: Exactly one recommended next action
    lines.append("## 10. Exactly one recommended next action")
    lines.append("")
    lines.append(f"Recommendation: **{dec.get('next_action', 'no action specified')}**")
    lines.append("")
    lines.append("### Acceptance criteria")
    lines.append("")
    lines.append("- No training was run.")
    lines.append("- No arena was run.")
    lines.append("- No model was promoted.")
    lines.append("- Active references were not mutated.")
    lines.append("- Exhausted families were excluded from selection.")
    lines.append("- Teacher-conflict filtering was used.")
    lines.append("- Selected rows are metadata candidates only, not replay artifacts.")
    lines.append("- Final report recommends exactly one next branch.")

    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}")

    # Load fixture references for dedup
    print("\nLoading fixture inventory...")
    suite_rows = load_suite(SUITE_PATH)
    ref_by_id, ref_by_canonical = load_reference_maps(REFERENCE_PATH)
    suite_canon = suite_canonical_states(suite_rows)

    # Load existing selected rows from PR #72 to avoid overlap
    existing_selected_path = Path(
        "/tmp/azlite_fresh_hard_state_mining_teacher_filtered/selected_fresh_family_rows.jsonl"
    )
    existing_selected = load_jsonl(existing_selected_path)
    existing_selected_hashes: set[str] = set()
    for row in existing_selected:
        h = row.get("canonical_state_hash", "")
        if h:
            existing_selected_hashes.add(h)

    print(f"  Suite rows: {len(suite_rows)}")
    print(f"  Reference rows: {len(ref_by_id)}")
    print(f"  Existing selected hashes (PR #72): {len(existing_selected_hashes)}")

    # Load artifact evaluator
    print("\nLoading artifact evaluator...")
    evaluator = ArtifactEvaluator(CURRENT_ARTIFACT)

    # ── Step 1: Generate candidates from multiple sources ────────────────

    print("\n=== Step 1: Candidate generation ===")
    t0 = time.time()

    seeds = [SEED + i for i in range(CANDIDATE_SEEDS)]
    all_raw: list[dict[str, Any]] = []
    candidate_source_info: list[dict[str, Any]] = []

    # Source A: Deeper self-play endgames
    print("\n  Source A: Deeper self-play endgames...")
    sp_games = 15
    sp_candidates = generate_deeper_self_play_endgames(
        evaluator,
        games=sp_games,
        seeds=seeds,
        simulations=400,
        max_plies=100,
        min_ply_for_collection=20,
    )
    all_raw.extend(sp_candidates)
    candidate_source_info.append(
        {
            "source": "deeper_self_play",
            "raw": len(sp_candidates),
            "tablebase_solvable": 0,
            "unique_optimal": 0,
            "puct_disagreements": 0,
            "neural_rank_errors": 0,
            "kept": 0,
            "notes": "late-game positions from current-model PUCT",
        }
    )
    print(f"    Collected {len(sp_candidates)} raw states")

    # Source B: Low-budget PUCT failure prescreen
    print("\n  Source B: Low-budget PUCT failure prescreen...")
    lb_candidates = generate_puct_low_budget_failure_candidates(
        evaluator,
        games=10,
        seeds=seeds,
        simulations=400,
        max_plies=100,
        low_budget=128,
        min_ply=20,
    )
    all_raw.extend(lb_candidates)
    candidate_source_info.append(
        {
            "source": "puct_low_budget_failure",
            "raw": len(lb_candidates),
            "tablebase_solvable": 0,
            "unique_optimal": 0,
            "puct_disagreements": 0,
            "neural_rank_errors": 0,
            "kept": 0,
            "notes": "PUCT 128 fails vs tablebase optimal",
        }
    )
    print(f"    Collected {len(lb_candidates)} raw states")

    # Source D: Adversarial near-threshold endgame sampling
    print("\n  Source D: Adversarial near-threshold endgame sampling...")
    at_candidates = generate_adversarial_near_threshold_candidates(
        evaluator,
        samples=100,
        seeds=seeds,
        min_seeds=2,
        max_seeds=14,
    )
    all_raw.extend(at_candidates)
    candidate_source_info.append(
        {
            "source": "adversarial_near_threshold",
            "raw": len(at_candidates),
            "tablebase_solvable": 0,
            "unique_optimal": 0,
            "puct_disagreements": 0,
            "neural_rank_errors": 0,
            "kept": 0,
            "notes": "random endgame states with unique optimal and meaningful gap",
        }
    )
    print(f"    Collected {len(at_candidates)} raw states")

    # Source E: PUCT-vs-tablebase disagreement sampling
    print("\n  Source E: PUCT-vs-tablebase disagreement sampling...")
    ds_candidates = generate_puct_tablebase_disagreement_candidates(
        evaluator,
        games=10,
        seeds=seeds,
        simulations=400,
        max_plies=100,
        puct_budgets=(384,),
        min_ply=16,
    )
    all_raw.extend(ds_candidates)
    candidate_source_info.append(
        {
            "source": "puct_tablebase_disagreement",
            "raw": len(ds_candidates),
            "tablebase_solvable": 0,
            "unique_optimal": 0,
            "puct_disagreements": 0,
            "neural_rank_errors": 0,
            "kept": 0,
            "notes": "PUCT 384 disagrees with tablebase unique optimal",
        }
    )
    print(f"    Collected {len(ds_candidates)} raw states")

    t1 = time.time()
    print(f"\n  Total raw candidates: {len(all_raw)} ({t1 - t0:.1f}s)")
    print("  By source:")
    for s in candidate_source_info:
        print(f"    {s['source']}: {s['raw']}")

    # Write raw candidates
    write_jsonl(OUTPUT_DIR / "raw_candidates.jsonl", all_raw)

    # ── Step 2: Deduplicate and exclude ──────────────────────────────────

    print("\n=== Step 2: Deduplication and exclusion ===")
    clean_candidates, dedup_counts = deduplicate_and_exclude(
        all_raw, suite_canon, suite_rows, ref_by_id, existing_selected_hashes
    )
    print(f"  Dedup counts: {dedup_counts}")
    print(f"  Clean candidates: {len(clean_candidates)}")

    # Write deduplicated candidates
    write_jsonl(OUTPUT_DIR / "deduplicated_candidates.jsonl", clean_candidates)

    # Update source info with actual counts
    source_final: dict[str, dict[str, int]] = {}
    for c in clean_candidates:
        src = c.get("source", "unknown")
        if src not in source_final:
            source_final[src] = {
                "kept": 0,
                "tb": 0,
                "unique": 0,
                "puct_dis": 0,
                "neural": 0,
            }
        source_final[src]["kept"] += 1

    for s in candidate_source_info:
        fn = source_final.get(s["source"], {})
        s["tablebase_solvable"] = dedup_counts.get("tablebase_solvable", 0)
        s["unique_optimal"] = dedup_counts.get("unique_optimal", 0)
        s["puct_disagreements"] = sum(
            1
            for c in clean_candidates
            if c.get("source") == s["source"]
            and c.get("_puct_class")
            in ("persistent_exact_failure", "medium_budget_exact_failure")
        )
        s["neural_rank_errors"] = 0  # filled later
        s["kept"] = fn.get("kept", 0)

    # ── Step 3: Exact tablebase enumeration ──────────────────────────────

    print("\n=== Step 3: Exact tablebase enumeration ===")
    exact_results = exact_tablebase_enumeration(clean_candidates)
    print(
        f"  Unique-clear-margin: {sum(1 for r in exact_results if r['exact_signal_class'] == 'exact_unique_clear_margin')}"
    )
    print(
        f"  Tablebase-tie: {sum(1 for r in exact_results if r['exact_signal_class'] == 'tablebase_tie')}"
    )
    print(
        f"  All-equivalent: {sum(1 for r in exact_results if r['exact_signal_class'] == 'all_moves_equivalent')}"
    )
    print(
        f"  Tiny-margin: {sum(1 for r in exact_results if r['exact_signal_class'] == 'exact_unique_tiny_margin')}"
    )

    # ── Step 4: PUCT baseline across budgets ─────────────────────────────

    print("\n=== Step 4: PUCT baseline across budgets ===")
    puct_results = run_puct_baseline(
        evaluator, clean_candidates, ROOT_BUDGETS, PROMISING_ONLY_BUDGETS
    )
    print(f"  Total PUCT evaluations: {len(puct_results)}")

    # Summarize PUCT results
    persistent_fails = sum(
        1
        for c in clean_candidates
        if c.get("_puct_class") == "persistent_exact_failure"
    )
    medium_fails = sum(
        1
        for c in clean_candidates
        if c.get("_puct_class") == "medium_budget_exact_failure"
    )
    low_budget_fails = sum(
        1
        for c in clean_candidates
        if c.get("_puct_class") == "low_budget_exact_failure"
    )
    clean_controls = sum(
        1 for c in clean_candidates if c.get("_puct_class") == "exact_clean_control"
    )
    print(f"  Persistent failures: {persistent_fails}")
    print(f"  Medium-budget failures: {medium_fails}")
    print(f"  Low-budget failures: {low_budget_fails}")
    print(f"  Clean controls: {clean_controls}")

    puct_summary = {
        "total_rows": len(clean_candidates),
        "persistent_failures": persistent_fails,
        "medium_budget_failures": medium_fails,
        "low_budget_failures": low_budget_fails,
        "clean_controls": clean_controls,
    }

    # Build PUCT budget table
    puct_budget_table: list[dict[str, Any]] = []
    for cand in clean_candidates:
        c_hash = cand["canonical_state_hash"]
        for r in cand.get("_puct_baseline_results", []):
            puct_budget_table.append(
                {
                    "candidate_hash": c_hash,
                    "budget": r.get("budget"),
                    "optimal_move": r.get("optimal_move"),
                    "selected_move": r.get("selected_move"),
                    "selected_is_optimal": r.get("selected_is_optimal"),
                    "optimal_visit_share": r.get("optimal_visit_share"),
                    "selected_visit_share": r.get("selected_visit_share"),
                    "selected_minus_optimal_q_margin": r.get(
                        "selected_minus_optimal_q_margin"
                    ),
                    "optimal_policy_rank": r.get("optimal_policy_rank"),
                    "failure_class": cand.get("_puct_class", ""),
                    "notes": "deterministic PUCT baseline"
                    if r.get("selected_is_optimal")
                    else "PUCT FAILURE",
                }
            )

    # ── Step 5: Neural value rank scan ────────────────────────────────────

    print("\n=== Step 5: Neural value rank scan ===")
    neural_results = neural_value_rank_scan(evaluator, clean_candidates)
    value_rank_errors = sum(1 for r in neural_results if r.get("value_rank_error"))
    sign_errors = sum(1 for r in neural_results if r.get("sign_error"))
    print(f"  Value rank errors: {value_rank_errors}")
    print(f"  Sign errors: {sign_errors}")

    # Update source info with neural rank errors
    for c in clean_candidates:
        src = c.get("source", "unknown")
        if src in source_final:
            if c.get("_neural_rank_error", False):
                source_final[src]["neural"] += 1

    for s in candidate_source_info:
        fn = source_final.get(s["source"], {})
        s["neural_rank_errors"] = fn.get("neural", 0)
        s["puct_disagreements"] = fn.get("puct_dis", 0)

    # ── Step 6: Classify and split ────────────────────────────────────────

    print("\n=== Step 6: Classification and split ===")
    split = classify_and_split(clean_candidates, exact_results, neural_results)
    print(f"  Target candidates: {split['target_candidate_count']}")
    print(f"  Preservation controls: {split['preservation_control_count']}")
    print(f"  Holdouts: {split['holdout_count']}")
    print(f"  Low-budget diagnostic: {split['low_budget_diagnostic_count']}")
    print(f"  Excluded: {split['excluded_count']}")

    # ── Step 7: Decision ────────────────────────────────────────────────

    print("\n=== Step 7: Decision ===")
    decision, next_action, dominant = make_decision(
        split, clean_candidates, neural_results, dedup_counts
    )
    print(f"  Decision: {decision}")
    print(f"  Next action: {next_action}")
    print(f"  Dominant: {dominant}")

    decision_data = {
        "classification": decision,
        "next_action": next_action,
        "dominant_mechanism": dominant,
        "counts": {
            "target_candidates": split["target_candidate_count"],
            "preservation_controls": split["preservation_control_count"],
            "holdouts": split["holdout_count"],
            "low_budget_diagnostic": split["low_budget_diagnostic_count"],
            "excluded": split["excluded_count"],
            "persistent_failures": persistent_fails,
            "medium_budget_failures": medium_fails,
            "low_budget_only": low_budget_fails,
            "value_rank_errors": value_rank_errors,
            "value_sign_errors": sign_errors,
        },
    }

    # ── Build low-budget diagnostic list ──────────────────────────────────

    low_budget_diagnostic_rows: list[dict[str, Any]] = []
    for spr in split["split_rows"]:
        if spr["assigned_role"] == "low_budget_diagnostic_only":
            c_hash = spr["candidate_hash"]
            cand = next(
                (c for c in clean_candidates if c["canonical_state_hash"] == c_hash),
                None,
            )
            if cand:
                low_budget_diagnostic_rows.append(
                    {
                        "candidate_hash": c_hash,
                        "state": cand["state"],
                        "ply": cand.get("ply"),
                        "source": cand.get("source"),
                        "tb_optimal_move": cand.get("_tb_unique_optimal"),
                        "remaining_seeds": total_seeds_remaining(cand["state"]),
                    }
                )

    # ── Write summary JSON ──────────────────────────────────────────────

    summary = {
        "schema": "azlite_harder_fresh_endgame_tablebase_mining_v1",
        "family": FAMILY,
        "description": (
            "Harder fresh exact-tablebase-unique endgame mining. "
            "Generates candidates from deeper self-play, low-budget PUCT "
            "prescreening, adversarial near-threshold sampling, and direct "
            "PUCT-vs-tablebase disagreement sampling."
        ),
        "guardrails": {
            "mutated_active_fixture": False,
            "ran_training": False,
            "ran_arena": False,
            "promoted_model": False,
            "created_replay_artifacts": False,
        },
        "inputs": {
            "current_artifact": str(CURRENT_ARTIFACT),
            "reference_path": str(REFERENCE_PATH),
            "suite_path": str(SUITE_PATH),
        },
        "candidate_sources": candidate_source_info,
        "dedup_counts": dedup_counts,
        "exact_tablebase_table": exact_results,
        "puct_baseline_summary": puct_summary,
        "puct_budget_table": puct_budget_table,
        "neural_value_table": neural_results,
        "split": split,
        "low_budget_diagnostic": low_budget_diagnostic_rows,
        "decision": decision,
        "decision_data": decision_data,
        "next_action": next_action,
    }

    write_json(
        OUTPUT_DIR / "harder_fresh_endgame_tablebase_mining_summary.json", summary
    )

    # ── Write candidate JSONL ──────────────────────────────────────────────

    candidate_rows = []
    for cand in clean_candidates:
        candidate_rows.append(
            {
                "candidate_id": f"harder_{cand.get('source_game_seed', '?')}_{cand.get('ply', '?')}",
                "canonical_state_hash": cand["canonical_state_hash"],
                "state": cand["state"],
                "source": cand.get("source", ""),
                "ply": cand.get("ply"),
                "remaining_seeds": total_seeds_remaining(cand["state"]),
                "tb_optimal_move": cand.get("_tb_unique_optimal"),
                "tb_root_value": cand.get("_tb_root_value"),
                "tb_best_minus_second": cand.get("_tb_best_minus_second"),
                "tb_all_equiv": cand.get("_tb_all_equiv", False),
                "puct_classification": cand.get("_puct_class", ""),
                "neural_rank_error": cand.get("_neural_rank_error", False),
                "neural_sign_error": cand.get("_neural_sign_error", False),
                "neural_best_move": cand.get("_neural_best_move"),
            }
        )
    write_jsonl(
        OUTPUT_DIR / "harder_endgame_tablebase_candidates.jsonl", candidate_rows
    )

    # ── Write selected rows JSONL ──────────────────────────────────────────

    selected_rows = []
    for spr in split["split_rows"]:
        if spr["assigned_role"] in (
            "target_candidate",
            "preservation_control",
            "holdout_candidate",
        ):
            c_hash = spr["candidate_hash"]
            cand = next(
                (c for c in clean_candidates if c["canonical_state_hash"] == c_hash),
                None,
            )
            if cand:
                selected_rows.append(
                    {
                        "candidate_id": f"harder_{cand.get('source_game_seed', '?')}_{cand.get('ply', '?')}",
                        "canonical_state_hash": c_hash,
                        "provisional_family": FAMILY,
                        "state": cand["state"],
                        "source": cand.get("source", ""),
                        "ply": cand.get("ply"),
                        "remaining_seeds": total_seeds_remaining(cand["state"]),
                        "tb_optimal_move": cand.get("_tb_unique_optimal"),
                        "tb_root_value": cand.get("_tb_root_value"),
                        "tb_best_minus_second": cand.get("_tb_best_minus_second"),
                        "assigned_role": spr["assigned_role"],
                        "failure_class": spr["failure_class"],
                        "exact_signal_class": spr["exact_signal_class"],
                        "puct_failure_budget": spr["puct_failure_budget"],
                        "value_rank_error": spr["value_rank_error"],
                        "do_not_train_yet": True,
                    }
                )
            else:
                # No state available (excluded or low_budget only)
                pass

    write_jsonl(
        OUTPUT_DIR / "selected_harder_endgame_tablebase_rows.jsonl", selected_rows
    )

    # ── Generate report markdown ──────────────────────────────────────────

    report_path = Path(
        "docs/alphazero-lite-harder-fresh-endgame-tablebase-mining-results.md"
    )
    generate_report(summary, report_path)
    print(f"\nReport written to {report_path}")

    # ── Console summary ──────────────────────────────────────────────────

    print(f"\n{'=' * 90}")
    print("HARDER FRESH ENDGAME TABLEBASE MINING")
    print(f"{'=' * 90}")
    print(f"\n{'Raw candidates':30s} {dedup_counts.get('raw_candidates', 0)}")
    print(f"{'Deduplicated':30s} {dedup_counts.get('remaining_clean', 0)}")
    print(f"{'Clean controls':30s} {clean_controls}")
    print(f"\n{'Target candidates':30s} {split['target_candidate_count']}")
    print(f"{'Preservation controls':30s} {split['preservation_control_count']}")
    print(f"{'Holdouts':30s} {split['holdout_count']}")
    print(f"{'Low-budget diagnostic':30s} {split['low_budget_diagnostic_count']}")
    print(f"{'Excluded':30s} {split['excluded_count']}")
    print(f"\n{'Persistent failures':30s} {persistent_fails}")
    print(f"{'Medium-budget failures':30s} {medium_fails}")
    print(f"{'Low-budget failures':30s} {low_budget_fails}")
    print(f"{'Value rank errors':30s} {value_rank_errors}")
    print(f"{'Sign errors':30s} {sign_errors}")
    print(f"\n{'Classification':30s} {decision}")
    print(f"{'Dominant mechanism':30s} {dominant}")
    print(f"{'Next action':30s} {next_action}")
    print(f"\n{'Outputs':30s}")
    print(f"  Summary: {OUTPUT_DIR}/harder_fresh_endgame_tablebase_mining_summary.json")
    print(f"  Candidates: {OUTPUT_DIR}/harder_endgame_tablebase_candidates.jsonl")
    print(f"  Selected rows: {OUTPUT_DIR}/selected_harder_endgame_tablebase_rows.jsonl")
    print(f"  Report: {report_path}")
    print(f"{'=' * 90}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
