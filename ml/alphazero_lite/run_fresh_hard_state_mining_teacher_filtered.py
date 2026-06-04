#!/usr/bin/env python3
"""Fresh hard-state mining with teacher-conflict filtering.

Generates candidate positions from current-model play, evaluates with PUCT and
ClassicMCTS teachers, filters out exhausted and teacher-conflict families, and
selects exactly one clean fresh family or reports none_safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import (
    PUCT,
    standard_start_state,
)


PITS_PER_PLAYER = 6

# ── Paths ──────────────────────────────────────────────────────────────────

DEFAULT_CURRENT_ARTIFACT = Path("storage/ai/alphazero_lite/current")
DEFAULT_REFERENCE_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
DEFAULT_OUTPUT_DIR = Path("/tmp/azlite_fresh_hard_state_mining_teacher_filtered")
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "fresh_hard_state_mining_summary.json"
DEFAULT_CANDIDATE_ROWS_PATH = DEFAULT_OUTPUT_DIR / "mined_candidate_rows.jsonl"
DEFAULT_SELECTED_ROWS_PATH = DEFAULT_OUTPUT_DIR / "selected_fresh_family_rows.jsonl"
DEFAULT_REPORT_PATH = Path(
    "docs/alphazero-lite-fresh-hard-state-mining-teacher-filtered-results.md"
)

# ── Exclusion buckets (exhausted / diagnostic-only) ────────────────────────

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

CORRECTED_GUARD_ROW_IDS: frozenset[str] = frozenset(
    {
        "capture_available-002",
        "capture_available-003",
        "capture_available-006",
        "capture_available-007",
        "capture_available-008",
    }
)


# ── Search profiles ────────────────────────────────────────────────────────


def build_standard_search_options() -> dict[str, Any]:
    return {
        "fpu_mode": "parent_q",
        "reuse_subtree": True,
        "normalize_values": True,
        "root_policy_mode": "deterministic",
        "tactical_root_bias": 0.1,
    }


def build_puct_profile(simulations: int, seed: int) -> dict[str, Any]:
    opts = build_standard_search_options()
    profile = {
        "kind": "fresh_mining_puct",
        "player_mode": "puct",
        "simulations": simulations,
        "c_puct": 1.25,
        "search_options": opts,
    }
    encoded = json.dumps(
        profile, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    profile["hash"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return profile


# ── Candidate representation ───────────────────────────────────────────────


@dataclass
class CandidatePosition:
    state: dict[str, Any]
    canonical_hash: str
    ply: int
    source_game_seed: int
    legal_moves: list[int]
    current_puct_visit_dist: list[float]
    current_puct_q_by_move: dict[int, float]
    current_puct_selected_move: int | None
    current_value: float
    prior: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": f"fresh_{self.source_game_seed}_{self.ply}",
            "source_game_seed": self.source_game_seed,
            "ply": self.ply,
            "canonical_state_hash": self.canonical_hash,
            "state": self.state,
            "legal_moves": self.legal_moves,
            "current_puct_selected_move": self.current_puct_selected_move,
            "current_puct_visit_distribution": self.current_puct_visit_dist,
            "current_puct_q_by_move": {
                str(m): float(q) for m, q in self.current_puct_q_by_move.items()
            },
            "current_value": float(self.current_value),
            "current_puct_prior": [float(p) for p in self.prior],
            "source": "fresh_hard_state_mining_teacher_filtered",
        }


# ── Self-play candidate generation ─────────────────────────────────────────


def generate_candidates_via_self_play(
    evaluator: ArtifactEvaluator,
    *,
    games: int,
    seeds: list[int],
    simulations: int,
    max_plies: int,
    output_dir: Path,
) -> list[CandidatePosition]:
    """Play games using current-artifact PUCT, collect positions."""
    candidates: list[CandidatePosition] = []

    for game_index in range(games):
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

            if ply < 8:
                candidates.append(
                    CandidatePosition(
                        state=state,
                        canonical_hash=c_hash,
                        ply=ply,
                        source_game_seed=seed + game_index,
                        legal_moves=legal_moves,
                        current_puct_visit_dist=[0.0] * PITS_PER_PLAYER,
                        current_puct_q_by_move={},
                        current_puct_selected_move=None,
                        current_value=0.0,
                        prior=[0.0] * PITS_PER_PLAYER,
                    )
                )
                absolute_move = game.pit_index(legal_moves[0])
                game.move(absolute_move)
                reusable_root = None
                continue

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

            prior = np.zeros(PITS_PER_PLAYER, dtype=np.float32)
            for move, child in root.children.items():
                prior[move] = child.prior

            q_by_move: dict[int, float] = {}
            selected_move = None
            if legal_moves:
                selected_move = search.select_root_move(root, legal_moves)
                for move, child in root.children.items():
                    q_by_move[move] = child.q_value

            candidates.append(
                CandidatePosition(
                    state=state,
                    canonical_hash=c_hash,
                    ply=ply,
                    source_game_seed=seed + game_index,
                    legal_moves=legal_moves,
                    current_puct_visit_dist=[float(v) for v in visits.tolist()],
                    current_puct_q_by_move=q_by_move,
                    current_puct_selected_move=int(selected_move)
                    if selected_move is not None
                    else None,
                    current_value=root.q_value,
                    prior=[float(p) for p in prior.tolist()],
                )
            )

            move = search.select_root_move(root, legal_moves)
            child = root.child_for_action(move)
            reusable_root = child
            absolute_move = game.pit_index(move)
            if not game.move(absolute_move):
                break

    return candidates


def generate_search_disagreement_candidates(
    evaluator: ArtifactEvaluator,
    *,
    games: int,
    seeds: list[int],
    puct_simulations: int,
    classic_simulations: int,
    max_plies: int,
    output_dir: Path,
) -> list[CandidatePosition]:
    """Collect positions where PUCT and ClassicMCTS disagree."""
    candidates: list[CandidatePosition] = []

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

            search = PUCT(
                evaluator=evaluator,
                simulations=puct_simulations,
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

            prior = np.zeros(PITS_PER_PLAYER, dtype=np.float32)
            for move, child in root.children.items():
                prior[move] = child.prior

            q_by_move: dict[int, float] = {}
            puct_selected = None
            if legal_moves:
                puct_selected = search.select_root_move(root, legal_moves)
                for move, child in root.children.items():
                    q_by_move[move] = child.q_value

            classic_seed = (seed * 1_000_003) + (game_index * 10_007) + ply
            mcts = ClassicMCTS(
                game.clone(),
                simulations=classic_simulations,
                seed=classic_seed,
            )
            classic_root = mcts.search_root()
            classic_selected = mcts._choose_move_from_root(classic_root)

            if puct_selected is not None and classic_selected is not None:
                if ply >= 8 and puct_selected != classic_selected:
                    candidates.append(
                        CandidatePosition(
                            state=state,
                            canonical_hash=c_hash,
                            ply=ply,
                            source_game_seed=seed + game_index + 1000,
                            legal_moves=legal_moves,
                            current_puct_visit_dist=[float(v) for v in visits.tolist()],
                            current_puct_q_by_move=q_by_move,
                            current_puct_selected_move=int(puct_selected),
                            current_value=root.q_value,
                            prior=[float(p) for p in prior.tolist()],
                        )
                    )

            move = search.select_root_move(root, legal_moves)
            child = root.child_for_action(move)
            reusable_root = child
            absolute_move = game.pit_index(move)
            if not game.move(absolute_move):
                break

    return candidates


# ── Teacher evaluation ─────────────────────────────────────────────────────


@dataclass
class TeacherEval:
    puct_384_move: int | None
    puct_384_q: dict[int, float] | None
    puct_384_visits: list[float] | None
    puct_1200_move: int | None
    puct_1200_q: dict[int, float] | None
    puct_1200_visits: list[float] | None
    classic_1200_move: int | None
    classic_2400_move: int | None
    tablebase_optimal_moves: list[int] | None
    tablebase_value: float | None
    teacher_agreement: (
        str  # agree, disagree, tablebase_confirms, tablebase_tie, inconclusive
    )
    teacher_conflict: bool
    reference_confidence: str  # high, medium, low, reject

    def puct_classic_disagree(self) -> bool:
        return (
            self.puct_1200_move is not None
            and self.classic_2400_move is not None
            and self.puct_1200_move != self.classic_2400_move
        )


def evaluate_with_puct(
    evaluator: ArtifactEvaluator,
    game: KalahGame,
    simulations: int,
    seed: int,
) -> tuple[int | None, dict[int, float] | None, list[float] | None]:
    rng = random.Random(seed)
    search = PUCT(
        evaluator=evaluator,
        simulations=simulations,
        c_puct=1.25,
        rng=rng,
        fpu_mode="parent_q",
        reuse_subtree=True,
        normalize_values=True,
        root_policy_mode="deterministic",
        tactical_root_bias=0.1,
    )
    visits, root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    legal_moves = game.possible_moves()
    selected = search.select_root_move(root, legal_moves) if legal_moves else None
    q_by_move: dict[int, float] = {}
    for move, child in root.children.items():
        q_by_move[move] = child.q_value
    return (
        int(selected) if selected is not None else None,
        q_by_move,
        [float(v) for v in visits.tolist()],
    )


def evaluate_with_classic_mcts(
    game: KalahGame,
    simulations: int,
    seed: int,
) -> int | None:
    mcts = ClassicMCTS(
        game.clone(),
        simulations=simulations,
        seed=seed,
    )
    root = mcts.search_root()
    return mcts._choose_move_from_root(root)


def evaluate_with_tablebase(
    game: KalahGame,
    tb: EndgameTablebase,
) -> tuple[list[int] | None, float | None]:
    perspective = game.current_player
    tb_value = tb.lookup(game, perspective)
    if tb_value is None:
        return None, None

    legal_moves = game.possible_moves()
    optimal_moves: list[int] = []
    offset = game.current_player * 6
    for move in legal_moves:
        child = game.clone()
        child.move(offset + move)
        child_value = tb.lookup(child, perspective)
        if child_value is None:
            continue
        if game.current_player == perspective:
            if child_value == tb_value:
                optimal_moves.append(move)
        else:
            if child_value == tb_value:
                optimal_moves.append(move)
    if not optimal_moves and legal_moves:
        return legal_moves, tb_value
    return optimal_moves or None, tb_value


def run_teacher_labeling(
    candidate: CandidatePosition,
    evaluator: ArtifactEvaluator,
    tb: EndgameTablebase,
) -> TeacherEval:
    game = KalahGame.from_state(candidate.state)

    puct_384_move, puct_384_q, puct_384_visits = evaluate_with_puct(
        evaluator, game.clone(), 384, 11
    )
    puct_1200_move, puct_1200_q, puct_1200_visits = evaluate_with_puct(
        evaluator, game.clone(), 1200, 11
    )

    classic_1200_move = evaluate_with_classic_mcts(game.clone(), 1200, 11)
    classic_2400_move = evaluate_with_classic_mcts(game.clone(), 2400, 23)

    tablebase_optimal, tablebase_value = evaluate_with_tablebase(game.clone(), tb)

    puct_classic_agree = (
        puct_1200_move is not None
        and classic_2400_move is not None
        and puct_1200_move == classic_2400_move
    )

    teacher_conflict = False
    teacher_agreement = "inconclusive"
    reference_confidence = "medium"

    if tablebase_optimal and len(tablebase_optimal) == 1:
        optimal = tablebase_optimal[0]
        if puct_1200_move == optimal and classic_2400_move == optimal:
            teacher_agreement = "tablebase_confirms"
            reference_confidence = "high"
        elif puct_1200_move == optimal or classic_2400_move == optimal:
            teacher_agreement = "tablebase_confirms"
            reference_confidence = "medium"
        else:
            teacher_agreement = "tablebase_conflict"
            teacher_conflict = True
            reference_confidence = "low"
    elif tablebase_optimal and len(tablebase_optimal) > 1:
        teacher_agreement = "tablebase_tie"
        reference_confidence = "medium"
    elif puct_classic_agree:
        teacher_agreement = "agree"
        reference_confidence = "high"
    else:
        teacher_agreement = "disagree"
        teacher_conflict = True
        reference_confidence = "low"

    if classic_1200_move is not None and classic_2400_move is not None:
        if classic_1200_move != classic_2400_move:
            teacher_conflict = True
            if teacher_agreement not in ("tablebase_confirms", "tablebase_tie"):
                reference_confidence = "low"

    return TeacherEval(
        puct_384_move=puct_384_move,
        puct_384_q=puct_384_q,
        puct_384_visits=puct_384_visits,
        puct_1200_move=puct_1200_move,
        puct_1200_q=puct_1200_q,
        puct_1200_visits=puct_1200_visits,
        classic_1200_move=classic_1200_move,
        classic_2400_move=classic_2400_move,
        tablebase_optimal_moves=tablebase_optimal,
        tablebase_value=tablebase_value,
        teacher_agreement=teacher_agreement,
        teacher_conflict=teacher_conflict,
        reference_confidence=reference_confidence,
    )


# ── Exclusion / dedup logic ────────────────────────────────────────────────


def load_suite_rows(path: Path) -> dict[str, dict]:
    """Load forensic suite, return {id: row}."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("rows", raw)
    return {str(row["id"]): row for row in raw if "id" in row}


def load_reference_rows(path: Path) -> dict[str, dict]:
    """Load forensic references, return {id: row}."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "rows" in payload:
        return {str(row["id"]): row for row in payload["rows"] if "id" in row}
    return {}


def suite_canonical_states(suite: dict[str, dict]) -> set[str]:
    result: set[str] = set()
    for row in suite.values():
        if "canonical_state" in row:
            result.add(str(row["canonical_state"]))
        elif "state" in row:
            result.add(canonical_state_key(row["state"]))
    return result


def is_exhausted_row_id(row_id: str) -> bool:
    for prefix in EXHAUSTED_ROW_ID_PREFIXES:
        if row_id.startswith(prefix):
            return True
    if row_id in CORRECTED_GUARD_ROW_IDS:
        return True
    return False


def is_exhausted_bucket(bucket: str | None) -> bool:
    if bucket is None:
        return False
    return bucket in EXHAUSTED_BUCKETS


def is_reference_unstable(refs: dict[str, dict], row_id: str) -> bool:
    ref = refs.get(row_id, {})
    return bool(ref.get("reference_unstable", False))


def is_excluded_row(refs: dict[str, dict], row_id: str) -> bool:
    ref = refs.get(row_id, {})
    tags = ref.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    for tag in tags:
        tag_s = str(tag)
        if tag_s in ("reference_integrity_error", "train_only", "excluded_diagnostic"):
            return True
    return False


def deduplicate_candidates(
    candidates: list[CandidatePosition],
    suite_canonical: set[str],
    suite_rows: dict[str, dict],
    ref_rows: dict[str, dict],
) -> tuple[list[CandidatePosition], dict[str, int]]:
    """Deduplicate. Returns (unique_novel, counts)."""
    counts: dict[str, int] = {
        "raw": len(candidates),
        "exact_duplicates": 0,
        "known_fixture_overlaps": 0,
        "excluded_overlaps": 0,
        "remaining_novel": 0,
    }

    seen_hashes: set[str] = set()
    unique: list[CandidatePosition] = []

    for c in candidates:
        if c.canonical_hash in seen_hashes:
            counts["exact_duplicates"] += 1
            continue
        seen_hashes.add(c.canonical_hash)

        if c.canonical_hash in suite_canonical:
            counts["known_fixture_overlaps"] += 1
            for row_id, row in suite_rows.items():
                row_cs = row.get("canonical_state") or canonical_state_key(
                    row.get("state", {})
                )
                if row_cs == c.canonical_hash:
                    if is_exhausted_row_id(row_id):
                        counts["excluded_overlaps"] += 1
                        break
                    bucket = row.get("bucket")
                    if is_exhausted_bucket(bucket):
                        counts["excluded_overlaps"] += 1
                        break
                    if is_reference_unstable(ref_rows, row_id):
                        counts["excluded_overlaps"] += 1
                        break
                    if is_excluded_row(ref_rows, row_id):
                        counts["excluded_overlaps"] += 1
                        break
            continue

        counts["remaining_novel"] += 1
        unique.append(c)

    return unique, counts


# ── Provisional family clustering ──────────────────────────────────────────


def cluster_into_families(
    candidates: list[CandidatePosition],
    teacher_evals: list[TeacherEval],
) -> dict[str, list[tuple[CandidatePosition, TeacherEval]]]:
    """Cluster candidates into provisional families."""
    families: dict[str, list[tuple[CandidatePosition, TeacherEval]]] = {}
    # Default family for unclustered
    default_family = "fresh_midgame_capture_swing"
    families[default_family] = []

    for c, te in zip(candidates, teacher_evals):
        family = _assign_family(c, te)
        if family not in families:
            families[family] = []
        families[family].append((c, te))

    return families


def _assign_family(c: CandidatePosition, te: TeacherEval) -> str:
    if te.tablebase_optimal_moves and len(te.tablebase_optimal_moves) == 1:
        return "fresh_endgame_tablebase_unique"
    if te.tablebase_optimal_moves and len(te.tablebase_optimal_moves) > 1:
        return "fresh_endgame_tablebase_tie"

    has_extra_turn = (
        any(_has_immediate_extra_turn(c.state, move) for move in c.legal_moves)
        if hasattr(c, "state")
        else False
    )
    has_capture = (
        any(_has_immediate_capture(c.state, move) for move in c.legal_moves)
        if hasattr(c, "state")
        else False
    )

    if has_capture and te.puct_classic_disagree():
        return "fresh_capture_swing"
    if has_extra_turn and te.puct_classic_disagree():
        return "fresh_extra_turn_handoff"

    persistent_1200 = (
        te.puct_384_move is not None
        and te.puct_1200_move is not None
        and te.puct_384_move != te.puct_1200_move
    )
    if persistent_1200:
        return "fresh_search_selection_pressure"

    if (
        c.current_value is not None
        and abs(c.current_value) < 0.3
        and te.teacher_agreement == "disagree"
    ):
        return "fresh_value_q_disagreement"

    return "fresh_midgame_capture_swing"


def _has_immediate_extra_turn(state: dict, move: int) -> bool:
    game = KalahGame.from_state(state)
    simulated = game.clone()
    player = simulated.current_player
    if not simulated.move(simulated.pit_index(move)):
        return False
    return simulated.current_player == player and not simulated.over()


def _has_immediate_capture(state: dict, move: int) -> bool:
    game = KalahGame.from_state(state)
    absolute_index = game.pit_index(move)
    seeds = game.pits[absolute_index]
    store_before = game.captured_seeds[game.current_player]
    simulated = game.clone()
    if not simulated.move(simulated.pit_index(move)):
        return False
    store_gain = simulated.captured_seeds[game.current_player] - store_before
    return store_gain > _own_store_passes(move, seeds)


def _own_store_passes(move: int, seeds: int) -> int:
    distance_to_store = PITS_PER_PLAYER - move
    if seeds < distance_to_store:
        return 0
    return 1 + ((seeds - distance_to_store) // ((PITS_PER_PLAYER * 2) + 1))


# ── Family scoring / targetability ─────────────────────────────────────────


@dataclass
class FamilyScore:
    name: str
    rows: int
    stable_teacher_rows: int
    rejected_rows: int
    persistent_1200_failures: int
    teacher_conflict_rate: float
    tablebase_confirmed_rows: int
    value_q_rows: int
    policy_prior_rows: int
    search_selection_rows: int
    duplicate_rate: float
    dominant_failure_mode: str
    targetability: str
    enough_controls: bool
    notes: str


def score_families(
    families: dict[str, list[tuple[CandidatePosition, TeacherEval]]],
) -> list[FamilyScore]:
    scores: list[FamilyScore] = []
    for name, members in families.items():
        if not members:
            continue

        total = len(members)
        stable = sum(
            1 for _, te in members if te.reference_confidence in ("high", "medium")
        )
        rejected = sum(1 for _, te in members if te.reference_confidence == "reject")
        persistent = sum(
            1
            for _, te in members
            if te.puct_384_move is not None
            and te.puct_1200_move is not None
            and te.puct_384_move != te.puct_1200_move
        )
        conflict = sum(1 for _, te in members if te.teacher_conflict)
        conflict_rate = conflict / total if total > 0 else 0.0
        tb_confirmed = sum(
            1 for _, te in members if te.teacher_agreement == "tablebase_confirms"
        )
        vq_looking = sum(
            1
            for _, te in members
            if te.teacher_agreement == "disagree" and _likely_value_q(te)
        )
        pp_looking = sum(
            1
            for _, te in members
            if te.teacher_agreement == "disagree" and _likely_policy_prior(te)
        )
        ss_looking = sum(
            1
            for _, te in members
            if persistent > 0 and te.puct_384_move != te.puct_1200_move
        )

        # Determine dominant failure mode
        if tb_confirmed > total * 0.5:
            dominant = "tablebase_exact"
        elif conflict_rate > 0.4:
            dominant = "teacher_conflict"
        elif persistent > total * 0.3:
            dominant = "search_selection"
        elif vq_looking > total * 0.3:
            dominant = "value_q"
        elif pp_looking > total * 0.3:
            dominant = "policy_prior"
        else:
            dominant = "mixed"

        # Targetability classification
        enough = total >= 6
        targetability = _classify_targetability(
            total, conflict_rate, persistent, tb_confirmed, dominant, enough
        )

        scores.append(
            FamilyScore(
                name=name,
                rows=total,
                stable_teacher_rows=stable,
                rejected_rows=rejected,
                persistent_1200_failures=persistent,
                teacher_conflict_rate=round(conflict_rate, 4),
                tablebase_confirmed_rows=tb_confirmed,
                value_q_rows=vq_looking,
                policy_prior_rows=pp_looking,
                search_selection_rows=ss_looking,
                duplicate_rate=0.0,
                dominant_failure_mode=dominant,
                targetability=targetability,
                enough_controls=enough,
                notes="",
            )
        )

    scores.sort(key=lambda s: _ranking_key(s), reverse=True)
    return scores


def _likely_value_q(te: TeacherEval) -> bool:
    if te.teacher_agreement != "disagree":
        return False
    if te.puct_1200_q is None or te.puct_1200_move is None:
        return False
    n_legal = len(te.puct_1200_q)
    if n_legal <= 1:
        return False
    q_values = sorted(te.puct_1200_q.values(), reverse=True)
    if len(q_values) < 2:
        return False
    return (q_values[0] - q_values[1]) > 0.15


def _likely_policy_prior(te: TeacherEval) -> bool:
    if te.teacher_agreement != "disagree":
        return False
    if te.puct_1200_visits is None or te.puct_1200_move is None:
        return False
    visits = te.puct_1200_visits
    total = sum(visits)
    if total <= 0:
        return False
    best_visit_share = max(visits) / total
    return best_visit_share < 0.3


def _classify_targetability(
    total: int,
    conflict_rate: float,
    persistent: int,
    tb_confirmed: int,
    dominant: str,
    enough: bool,
) -> str:
    if not enough:
        return "too_sparse"
    if conflict_rate > 0.5:
        return "teacher_conflict_dominant"
    if dominant == "tablebase_exact":
        return "fresh_family_tablebase_exact_ready"
    if dominant == "value_q" and conflict_rate < 0.3:
        return "fresh_family_value_backup_audit_ready"
    if dominant == "policy_prior" and conflict_rate < 0.3:
        return "fresh_family_policy_prior_audit_ready"
    if dominant == "search_selection" and conflict_rate < 0.3:
        return "fresh_family_search_selection_audit_ready"
    if tb_confirmed > 0 and conflict_rate < 0.3:
        return "fresh_family_tablebase_exact_ready"
    if conflict_rate > 0.3:
        return "too_noisy"
    return "no_clean_family"


def _ranking_key(s: FamilyScore) -> float:
    target_order = {
        "fresh_family_value_backup_audit_ready": 6,
        "fresh_family_policy_prior_audit_ready": 5,
        "fresh_family_search_selection_audit_ready": 4,
        "fresh_family_tablebase_exact_ready": 3,
        "teacher_conflict_dominant": 1,
        "too_sparse": 0,
        "too_noisy": 0,
        "no_clean_family": 0,
    }
    base = target_order.get(s.targetability, -1)
    bonus = (
        s.stable_teacher_rows * 2
        + s.tablebase_confirmed_rows
        - int(s.teacher_conflict_rate * 10)
    )
    # Penalize too_sparse and too_noisy
    if s.targetability in ("too_sparse", "too_noisy", "no_clean_family"):
        bonus = -100
    return base * 1000 + bonus


# ── Selection ──────────────────────────────────────────────────────────────


def select_family(
    scores: list[FamilyScore],
    families: dict[str, list[tuple[CandidatePosition, TeacherEval]]],
    rng: random.Random,
) -> tuple[FamilyScore | None, list[dict[str, Any]]]:
    """Select exactly one family, or return None."""
    selected_family = None
    top_scores = [
        s
        for s in scores
        if s.targetability
        not in (
            "too_sparse",
            "too_noisy",
            "no_clean_family",
            "teacher_conflict_dominant",
        )
    ]

    if not top_scores:
        # Fallback: check if any family is tablebase-ready
        for s in scores:
            if s.tablebase_confirmed_rows >= 3 and s.rows >= 4:
                selected_family = s
                break
        if selected_family is None:
            # Stop, no clean family
            return None, []

    if selected_family is None and top_scores:
        selected_family = top_scores[0]

    if selected_family is None:
        return None, []

    members = families.get(selected_family.name, [])
    rng.shuffle(members)

    selected_rows: list[dict[str, Any]] = []
    for i, (c, te) in enumerate(members):
        role = "target_candidate"
        if i == 0 and len(members) > 1:
            role = "preservation_control"
        elif i >= len(members) - 1 and len(members) > 2:
            role = "holdout_candidate"

        teacher_target = te.puct_1200_move
        teacher_source = "artifact_puct"
        if te.tablebase_optimal_moves and len(te.tablebase_optimal_moves) == 1:
            teacher_target = te.tablebase_optimal_moves[0]
            teacher_source = "tablebase"
        elif te.reference_confidence == "high" and not te.teacher_conflict:
            teacher_target = te.puct_1200_move
            teacher_source = "artifact_puct"

        selected_rows.append(
            {
                "candidate_id": f"fresh_{c.source_game_seed}_{c.ply}",
                "provisional_family": selected_family.name,
                "canonical_state_hash": c.canonical_hash,
                "state": c.state,
                "teacher_target_move": teacher_target,
                "teacher_source": teacher_source,
                "teacher_confidence": te.reference_confidence,
                "current_puct_selected_move": c.current_puct_selected_move,
                "classic_selected_move": te.classic_2400_move,
                "teacher_agreement": te.teacher_agreement,
                "teacher_conflict": te.teacher_conflict,
                "failure_mode": selected_family.dominant_failure_mode,
                "severity": "medium",
                "recommended_role": role,
                "do_not_train_yet": True,
            }
        )

    return selected_family, selected_rows


# ── Output ─────────────────────────────────────────────────────────────────


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


# ── Report generation (delegated to main) ─────────────────────────────────


def generate_report(
    summary: dict[str, Any],
    output_path: Path,
) -> str:
    """Generate markdown report, write to output_path, return content."""
    lines: list[str] = []
    lines.append("# AlphaZero-lite Fresh Hard-State Mining (Teacher-Filtered) Results")
    lines.append("")
    lines.append("## 1. Context")
    lines.append("")
    lines.append(f"- Run classification: {summary.get('classification', 'unknown')}")
    lines.append(f"- Selected family: {summary.get('selected_family', 'none_safe')}")
    lines.append(
        f"- Current artifact: {summary.get('artifact_path', 'storage/ai/alphazero_lite/current')}"
    )
    lines.append(
        f"- Active references: {summary.get('reference_path', 'ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json')}"
    )
    lines.append("- No training was run.")
    lines.append("- No arena was run.")
    lines.append("- No model was promoted.")
    lines.append("- Active references were not mutated.")
    lines.append("")

    lines.append("## 2. Why PR #70 stopped incumbent_proxy training")
    lines.append("")
    lines.append(
        "PR #70 concluded that teacher-policy conflict within the incumbent_proxy_disagreement"
    )
    lines.append(
        "family is architectural — no single-head or teacher-conditioned probe variant"
    )
    lines.append(
        "resolved the cross-teacher interference. PR #69 patch bundle was not applied."
    )
    lines.append(
        "The recommendation was to stop training on incumbent_proxy, use teacher-policy"
    )
    lines.append(
        "split only for evaluation, and improve mining/scoring from fresh positions."
    )
    lines.append("")
    lines.append(
        "This run implements that recommendation by mining fresh positions from"
    )
    lines.append("current-model play, not from the exhausted fixture inventory.")
    lines.append("")

    lines.append("## 3. Exclusion policy")
    lines.append("")
    lines.append("| exclusion_group | excluded_count | reason | notes |")
    lines.append("|---|---|---|---|")
    for entry in summary.get("exclusions", []):
        lines.append(
            f"| {entry['group']} | {entry['count']} | {entry['reason']} | {entry['notes']} |"
        )
    lines.append("")

    lines.append("## 4. Fresh candidate generation")
    lines.append("")
    lines.append(
        "| source | raw_candidates | deduplicated_candidates | known_fixture_overlaps | excluded_overlaps | remaining_candidates | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    cand = summary.get("candidate_counts", {})
    lines.append(
        f"| self_play | {cand.get('raw', 0)} | {cand.get('exact_duplicates', 0)} | "
        f"{cand.get('known_fixture_overlaps', 0)} | {cand.get('excluded_overlaps', 0)} | "
        f"{cand.get('remaining_novel', 0)} | fresh positions from current-model PUCT |"
    )
    lines.append("")

    lines.append("## 5. Deduplication and fixture overlap")
    lines.append("")
    lines.append(f"- Raw candidates: {cand.get('raw', 0)}")
    lines.append(f"- Duplicate candidates: {cand.get('exact_duplicates', 0)}")
    lines.append(f"- Known-fixture overlaps: {cand.get('known_fixture_overlaps', 0)}")
    lines.append(f"- Excluded overlaps: {cand.get('excluded_overlaps', 0)}")
    lines.append(f"- Remaining novel candidates: {cand.get('remaining_novel', 0)}")
    lines.append("")

    lines.append("## 6. Teacher labeling / adjudication-lite")
    lines.append("")
    lines.append(
        "| candidate_id | provisional_family | ply | puct_1200_move | classic_2400_move | tablebase_decision | teacher_agreement | teacher_conflict | reference_confidence | status | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in summary.get("teacher_labels", [])[:30]:
        lines.append(
            f"| {t.get('candidate_id', '')} | {t.get('family', '')} | {t.get('ply', '')} | "
            f"{t.get('puct_1200', '')} | {t.get('classic_2400', '')} | "
            f"{t.get('tablebase', '')} | {t.get('agreement', '')} | "
            f"{t.get('conflict', '')} | {t.get('confidence', '')} | "
            f"{t.get('status', '')} | {t.get('notes', '')} |"
        )
    if len(summary.get("teacher_labels", [])) > 30:
        lines.append(
            f"| ... and {len(summary['teacher_labels']) - 30} more rows | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |"
        )
    lines.append("")

    lines.append("## 7. Provisional family clustering")
    lines.append("")
    families = summary.get("family_scores", [])
    if families:
        lines.append(
            "| provisional_family | rows | stable_teacher_rows | rejected_rows | persistent_1200_failures | teacher_conflict_rate | tablebase_confirmed_rows | dominant_failure_mode | targetability | notes |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for s in families:
            lines.append(
                f"| {s['name']} | {s['rows']} | {s['stable_teacher_rows']} | {s['rejected_rows']} | "
                f"{s['persistent_1200_failures']} | {s['teacher_conflict_rate']} | "
                f"{s['tablebase_confirmed_rows']} | {s['dominant_failure_mode']} | "
                f"{s['targetability']} | {s['notes']} |"
            )
    else:
        lines.append("No families formed (no novel candidates survived).")
    lines.append("")

    lines.append("## 8. Targetability scoring")
    lines.append("")
    if families:
        target_counts: dict[str, int] = {}
        for s in families:
            t = s["targetability"]
            target_counts[t] = target_counts.get(t, 0) + 1
        for t, c in sorted(target_counts.items()):
            lines.append(f"- {t}: {c} family/ies")
    lines.append("")

    lines.append("## 9. Selected fresh family")
    lines.append("")
    sel = summary.get("selected_family_info", {})
    if sel.get("name") and sel.get("name") != "none_safe":
        lines.append(
            "| selected_family | target_rows | control_rows | holdout_rows | teacher_source | reason_selected | risks | next_action |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        lines.append(
            f"| {sel.get('name', '')} | {sel.get('target_rows', 0)} | {sel.get('control_rows', 0)} | "
            f"{sel.get('holdout_rows', 0)} | {sel.get('teacher_source', '')} | "
            f"{sel.get('reason', '')} | {sel.get('risks', '')} | {sel.get('next_action', '')} |"
        )
    else:
        lines.append(
            "No clean fresh family was found. All exhausted/diagnostic families were excluded."
        )
        lines.append(
            f"Final classification: {summary.get('classification', 'no_clean_fresh_family_found')}"
        )
    lines.append("")

    lines.append("## 10. Exactly one recommended next action")
    lines.append("")
    lines.append(
        f"Recommendation: **{summary.get('next_action', 'no clean fresh family — broaden candidate generation or rebuild fixture')}**"
    )
    lines.append("")
    lines.append("### Acceptance criteria")
    lines.append("")
    lines.append("- No training was run.")
    lines.append("- No arena was run.")
    lines.append("- No model was promoted.")
    lines.append("- Active references were not mutated.")
    lines.append("- Exhausted families were excluded from selection.")
    lines.append("- Teacher-conflict filtering from PR #70 was used.")
    lines.append("- Selected rows are metadata candidates only, not replay artifacts.")
    lines.append("- Final report recommends exactly one next branch.")

    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report


# ── Main pipeline ──────────────────────────────────────────────────────────


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    artifact_path = Path(args.current_artifact)
    reference_path = Path(args.references)
    suite_path = Path(args.suite)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # Load artifact
    print("Loading artifact evaluator...")
    evaluator = ArtifactEvaluator(artifact_path)

    # Load suite and references for dedup
    print("Loading fixture inventory...")
    suite_rows = load_suite_rows(suite_path)
    ref_rows = load_reference_rows(reference_path)
    suite_canonical = suite_canonical_states(suite_rows)

    # Load tablebase
    tb = EndgameTablebase()

    # Step 1: Exclusion policy (report only)
    print("Applying exclusion policy...")
    exclusion_counts: dict[str, int] = {}
    for bucket in sorted(EXHAUSTED_BUCKETS):
        count = sum(1 for row in suite_rows.values() if row.get("bucket") == bucket)
        if count > 0:
            exclusion_counts[bucket] = count
    exclusion_counts["corrected_guard_rows"] = sum(
        1 for rid in suite_rows if rid in CORRECTED_GUARD_ROW_IDS
    )
    ref_unstable = sum(
        1 for rid, row in ref_rows.items() if row.get("reference_unstable", False)
    )
    exclusion_counts["reference_unstable_rows"] = ref_unstable

    exclusions = []
    for group, count in sorted(exclusion_counts.items()):
        exclusions.append(
            {
                "group": group,
                "count": count,
                "reason": "exhausted or diagnostic-only",
                "notes": f"excluded from training; {count} rows"
                if count > 0
                else "no rows in active inventory",
            }
        )
    # Add explicit exclusion reasons for each group
    excl_notes = {
        "opening_plies_1_8": "opening replay branch is closed",
        "opening_extra_turn_overbias": "opening subfamily — closed",
        "opening_edge_move_5_preference": "opening subfamily — closed",
        "opening_missed_extra_turn_continuation": "opening subfamily — closed",
        "incumbent_proxy_disagreement": "teacher-conflict family; stopped after PR #70",
        "incumbent_proxy_residual": "teacher-conflict residual; exhausted",
        "high_value_swing": "reference suite too noisy after PR #53",
        "high_imbalance": "reference suite too noisy after PR #56",
        "capture_available": "no safe target rows after PR #59",
        "starvation_pressure": "reference suite too noisy after PR #62",
        "sparse_endgame": "dominated by forced/tied positions after PR #67",
        "early_extra_turn": "reference suite too noisy after reference adjudication",
        "corrected_guard_rows": "corrected guard confirmations stay context-only",
        "reference_unstable_rows": "rows marked reference_unstable in active references",
    }
    for entry in exclusions:
        entry["reason"] = excl_notes.get(entry["group"], "exhausted or diagnostic-only")

    # Step 2: Generate fresh candidates
    print(f"Generating {args.self_play_games} self-play candidates...")
    seeds = [args.seed + i for i in range(args.num_seeds)]
    sp_candidates = generate_candidates_via_self_play(
        evaluator,
        games=args.self_play_games,
        seeds=seeds,
        simulations=args.puct_simulations,
        max_plies=args.max_plies,
        output_dir=output_dir,
    )

    print(f"Generating {args.disagreement_games} search-disagreement candidates...")
    ds_candidates = generate_search_disagreement_candidates(
        evaluator,
        games=args.disagreement_games,
        seeds=[args.seed + 100 + i for i in range(args.num_seeds)],
        puct_simulations=args.puct_simulations,
        classic_simulations=args.classic_simulations,
        max_plies=args.max_plies,
        output_dir=output_dir,
    )

    all_candidates = sp_candidates + ds_candidates
    print(f"Total raw candidates: {len(all_candidates)}")

    # Step 3: Deduplicate
    print("Deduplicating...")
    novel_candidates, counts = deduplicate_candidates(
        all_candidates, suite_canonical, suite_rows, ref_rows
    )
    print(f"  Dedup counts: {counts}")

    if not novel_candidates:
        print("No novel candidates remain after dedup.")
        summary = {
            "schema": "fresh_hard_state_mining_v1",
            "classification": "mining_repeats_exhausted_families",
            "selected_family": "none_safe",
            "next_action": "broaden candidate generation or rebuild forensic fixture with stronger teacher criteria",
            "exclusions": exclusions,
            "candidate_counts": counts,
            "teacher_labels": [],
            "family_scores": [],
            "selected_family_info": {},
            "artifact_path": str(artifact_path),
            "reference_path": str(reference_path),
        }
        return summary

    # Step 4: Teacher labeling
    print(f"Running teacher labeling on {len(novel_candidates)} candidates...")
    teacher_labels: list[dict[str, Any]] = []
    teacher_evals: list[TeacherEval] = []
    for idx, c in enumerate(novel_candidates):
        if idx > 0 and idx % 50 == 0:
            print(f"  labeled {idx}/{len(novel_candidates)}")
        te = run_teacher_labeling(c, evaluator, tb)
        teacher_evals.append(te)

    # Build teacher label table data
    for c, te in zip(novel_candidates, teacher_evals):
        status = (
            "rejected"
            if te.reference_confidence == "reject"
            else ("candidate" if not te.teacher_conflict else "conflict")
        )
        teacher_labels.append(
            {
                "candidate_id": f"fresh_{c.source_game_seed}_{c.ply}",
                "family": "",
                "ply": c.ply,
                "puct_1200": te.puct_1200_move,
                "classic_2400": te.classic_2400_move,
                "tablebase": str(te.tablebase_optimal_moves)
                if te.tablebase_optimal_moves
                else "N/A",
                "agreement": te.teacher_agreement,
                "conflict": te.teacher_conflict,
                "confidence": te.reference_confidence,
                "status": status,
                "notes": "",
            }
        )

    # Step 5: Cluster
    print("Clustering into provisional families...")
    families = cluster_into_families(novel_candidates, teacher_evals)

    # Update teacher labels with family
    cand_to_family: dict[str, str] = {}
    for fam, members in families.items():
        for c, _ in members:
            cid = f"fresh_{c.source_game_seed}_{c.ply}"
            cand_to_family[cid] = fam
    for label in teacher_labels:
        label["family"] = cand_to_family.get(label["candidate_id"], "unassigned")

    # Step 6: Score
    print("Scoring targetability...")
    family_scores = score_families(families)

    # Step 7: Select
    print("Selecting family...")
    selected, selected_rows = select_family(family_scores, families, rng)

    # Write candidate rows
    candidate_dicts = []
    for c in novel_candidates:
        cd = c.to_dict()
        cd["provisional_family"] = cand_to_family.get(cd["candidate_id"], "unassigned")
        candidate_dicts.append(cd)
    write_jsonl(output_dir / "mined_candidate_rows.jsonl", candidate_dicts)

    # Build selected family info
    sel_info: dict[str, Any] = {}
    classification = "no_clean_fresh_family_found"
    next_action = "broaden candidate generation or rebuild forensic fixture with stronger teacher criteria"

    if selected is not None:
        target_rows = sum(
            1 for r in selected_rows if r["recommended_role"] == "target_candidate"
        )
        control_rows = sum(
            1 for r in selected_rows if r["recommended_role"] == "preservation_control"
        )
        holdout_rows = sum(
            1 for r in selected_rows if r["recommended_role"] == "holdout_candidate"
        )

        teacher_src = selected_rows[0]["teacher_source"] if selected_rows else "unknown"
        write_jsonl(output_dir / "selected_fresh_family_rows.jsonl", selected_rows)

        sel_info = {
            "name": selected.name,
            "target_rows": target_rows,
            "control_rows": control_rows,
            "holdout_rows": holdout_rows,
            "teacher_source": teacher_src,
            "reason": f"selected_family_{selected.targetability}",
            "risks": "teacher_conflict may reappear at higher budget; monitor holdout rows",
            "next_action": _next_action_for(selected.targetability),
        }
        classification = selected.targetability
        next_action = sel_info["next_action"]
    else:
        # Check which stop condition applies
        if counts["known_fixture_overlaps"] > counts["remaining_novel"] * 2:
            classification = "mining_repeats_exhausted_families"
            next_action = "strengthen exclusion/scoring and rerun fresh mining"
        else:
            classification = "no_clean_fresh_family_found"
            next_action = "broaden candidate generation or rebuild forensic fixture with stronger teacher criteria"
        sel_info = {
            "name": "none_safe",
            "target_rows": 0,
            "control_rows": 0,
            "holdout_rows": 0,
            "teacher_source": "N/A",
            "reason": "no clean fresh family passed targetability filters",
            "risks": "all candidates have teacher-conflict or exhausted-family overlap",
            "next_action": next_action,
        }

    summary = {
        "schema": "fresh_hard_state_mining_v1",
        "classification": classification,
        "selected_family": sel_info.get("name", "none_safe"),
        "next_action": next_action,
        "exclusions": exclusions,
        "candidate_counts": counts,
        "teacher_labels": teacher_labels,
        "family_scores": [
            {
                "name": s.name,
                "rows": s.rows,
                "stable_teacher_rows": s.stable_teacher_rows,
                "rejected_rows": s.rejected_rows,
                "persistent_1200_failures": s.persistent_1200_failures,
                "teacher_conflict_rate": s.teacher_conflict_rate,
                "tablebase_confirmed_rows": s.tablebase_confirmed_rows,
                "value_q_rows": s.value_q_rows,
                "policy_prior_rows": s.policy_prior_rows,
                "search_selection_rows": s.search_selection_rows,
                "dominant_failure_mode": s.dominant_failure_mode,
                "targetability": s.targetability,
                "enough_controls": s.enough_controls,
                "notes": s.notes,
            }
            for s in family_scores
        ],
        "selected_family_info": sel_info,
        "artifact_path": str(artifact_path),
        "reference_path": str(reference_path),
    }

    return summary


def _next_action_for(targetability: str) -> str:
    actions = {
        "fresh_family_value_backup_audit_ready": (
            "run child-afterstate value/backup audit for that fresh family before training"
        ),
        "fresh_family_policy_prior_audit_ready": (
            "run root prior/PUCT pressure diagnostics before training"
        ),
        "fresh_family_search_selection_audit_ready": (
            "run search selection/persistent-1200 audit; check whether PUCT visit allocation is stable"
        ),
        "fresh_family_tablebase_exact_ready": (
            "run tablebase-backed local search/value diagnostics; do not train until target/control split is clean"
        ),
    }
    return actions.get(
        targetability,
        "no training — improve mining/scoring or reconsider teacher policy",
    )


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fresh hard-state mining with teacher-conflict filtering"
    )
    parser.add_argument(
        "--current-artifact",
        default=str(DEFAULT_CURRENT_ARTIFACT),
        help="Path to current model artifact directory",
    )
    parser.add_argument(
        "--references",
        default=str(DEFAULT_REFERENCE_PATH),
        help="Path to active forensic references JSON",
    )
    parser.add_argument(
        "--suite", default=str(DEFAULT_SUITE_PATH), help="Path to forensic suite JSON"
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for mined data",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--num-seeds", type=int, default=4, help="Number of seeds to use for generation"
    )
    parser.add_argument(
        "--self-play-games",
        type=int,
        default=50,
        help="Number of self-play games for candidate generation",
    )
    parser.add_argument(
        "--disagreement-games",
        type=int,
        default=30,
        help="Number of search-disagreement games",
    )
    parser.add_argument(
        "--puct-simulations", type=int, default=96, help="PUCT simulations per position"
    )
    parser.add_argument(
        "--classic-simulations",
        type=int,
        default=384,
        help="Classic MCTS simulations for disagreement detection",
    )
    parser.add_argument(
        "--max-plies", type=int, default=60, help="Maximum plies per game"
    )
    parser.add_argument(
        "--summary-path",
        default=str(DEFAULT_SUMMARY_PATH),
        help="Output path for summary JSON",
    )
    parser.add_argument(
        "--candidate-rows-path",
        default=str(DEFAULT_CANDIDATE_ROWS_PATH),
        help="Output path for mined candidate rows JSONL",
    )
    parser.add_argument(
        "--selected-rows-path",
        default=str(DEFAULT_SELECTED_ROWS_PATH),
        help="Output path for selected family rows JSONL",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="Output path for markdown report",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_pipeline(args)

    # Write summary JSON
    write_json(Path(args.summary_path), summary)

    # Write report
    generate_report(summary, Path(args.report_path))

    print(f"\nSummary written to {args.summary_path}")
    print(f"Report written to {args.report_path}")
    print(f"Selected family: {summary.get('selected_family', 'none_safe')}")
    print(f"Classification: {summary.get('classification', 'unknown')}")
    print(f"Next action: {summary.get('next_action', 'unknown')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
