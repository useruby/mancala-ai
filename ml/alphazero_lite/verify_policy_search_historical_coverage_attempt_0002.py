"""Independently replay Seed 615 games and verify attempt-0002 coverage only."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("docs/data/seed615-source-opening-diversity")
EVAL = DATA / "evaluation-preparation"
RESULTS = EVAL / "results"
POLICY_SOURCE = Path("docs/data/seed633-teacher-target-learning/attempt-0005")
EXACT_SOURCE = Path("docs/data/seed633-exact-oracle-decision-diagnostic/attempt-0003")
ATTEMPT = Path("docs/data/seed633-policy-search-retrospective/attempt-0002")
FREEZE_PATH = ATTEMPT / "freeze.json"
OUTPUT_PATH = ATTEMPT / "coverage.json"
INITIAL = {
    "player_pits": [4, 4, 4, 4, 4, 4],
    "opponent_pits": [4, 4, 4, 4, 4, 4],
    "player_store": 0,
    "opponent_store": 0,
    "current_player": 0,
}
CHECKPOINTS = {
    "B512": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
    "S615": "cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780",
}
P_E4 = "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad"
THRESHOLD = 16
MIN_MCTS_PAIRED_ROOTS = 100
MIN_MCTS_EXACT_ROOTS = 50
MIN_OPENINGS = 20
MIN_MEMBERSHIPS = 2
MIN_PHASES = 1


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def canonical_identity(state: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(state))


def read_json(path: Path) -> Any:
    return json.loads((ROOT / path).read_bytes())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def file_sha256(path: Path) -> str:
    return sha256_bytes((ROOT / path).read_bytes())


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


class ReplayBoard:
    """Small independent Kalah transition model used only for evidence replay."""

    def __init__(self) -> None:
        self.pits = [4] * 12
        self.stores = [0, 0]
        self.current_player = 0
        self.terminal = False

    def to_state(self) -> dict[str, Any]:
        return {
            "player_pits": self.pits[:6],
            "opponent_pits": self.pits[6:],
            "player_store": self.stores[0],
            "opponent_store": self.stores[1],
            "current_player": self.current_player,
        }

    def legal_moves(self) -> list[int]:
        start = self.current_player * 6
        return [index for index in range(6) if self.pits[start + index] > 0]

    def over(self) -> bool:
        return self.terminal or not self.legal_moves()

    def play_absolute(self, absolute_move: int) -> None:
        if self.over() or not 0 <= absolute_move < 12:
            raise ValueError("illegal_absolute_action")
        player = self.current_player
        if absolute_move // 6 != player or self.pits[absolute_move] <= 0:
            raise ValueError("illegal_action_owner_or_empty_pit")

        seeds = self.pits[absolute_move]
        self.pits[absolute_move] = 0
        pit = absolute_move
        pit_owner = player
        extra_turn = False
        while seeds:
            next_pit = (pit + 1) % 12
            next_owner = next_pit // 6
            if pit_owner != next_owner:
                skip_opponent_store = pit_owner == player
                pit_owner = next_owner
                if skip_opponent_store:
                    self.stores[player] += 1
                    extra_turn = True
                    seeds -= 1
                    continue
            extra_turn = False
            pit = next_pit
            self.pits[pit] += 1
            seeds -= 1

        if not extra_turn:
            if pit // 6 == player and self.pits[pit] == 1:
                opposite = 11 - pit
                if self.pits[opposite] > 0:
                    self.stores[player] += self.pits[pit] + self.pits[opposite]
                    self.pits[pit] = 0
                    self.pits[opposite] = 0
            self.current_player = 1 - player

        if not self.legal_moves():
            remaining_player = 1 - self.current_player
            start = remaining_player * 6
            self.stores[remaining_player] += sum(self.pits[start : start + 6])
            self.pits = [0] * 12
            self.terminal = True


def role_for_player(current_player: int, candidate_seat: int) -> str:
    return "candidate" if current_player == candidate_seat else "opponent"


def assign_decision_sources(
    occurrences: list[dict[str, Any]],
    eligible_plies: list[int],
    handoff_count: int,
    first_handoff: int | None,
) -> None:
    """Use observed game-level handoffs; eligibility alone remains unknown."""
    if handoff_count > len(eligible_plies):
        raise ValueError("exact_handoff_count_exceeds_eligible_turns")
    if not eligible_plies:
        if handoff_count != 0 or first_handoff is not None:
            raise ValueError("exact_handoff_without_eligible_turn")
        return
    if handoff_count == 0:
        if first_handoff is not None:
            raise ValueError("first_exact_handoff_without_count")
        return
    if first_handoff not in eligible_plies or first_handoff != eligible_plies[0]:
        raise ValueError("first_exact_handoff_not_first_eligible")
    for row in occurrences:
        if row["ply"] == first_handoff:
            row["decision_source"] = (
                "exact_root_solver"
                if row["actor_role"] == "candidate"
                else "exact_root_solver_opponent"
            )
    if handoff_count == len(eligible_plies):
        for row in occurrences:
            if row["exact_root_eligible"]:
                row["decision_source"] = (
                    "exact_root_solver"
                    if row["actor_role"] == "candidate"
                    else "exact_root_solver_opponent"
                )


def deterministic_occurrence_bytes(occurrences: list[dict[str, Any]]) -> bytes:
    raw = b"".join(canonical_json(row) + b"\n" for row in occurrences)
    return gzip.compress(raw, compresslevel=9, mtime=0)


def validate_freeze(freeze: dict[str, Any]) -> None:
    require(freeze.get("attempt") == "attempt-0002", "attempt_identity_mismatch")
    require(
        freeze.get("status") == "frozen-before-full-trajectory-replay-and-join",
        "attempt_freeze_status_mismatch",
    )
    require(
        freeze.get("adequacy_criteria_unchanged_from_attempt_0001", {}).get(
            "minimum_paired_unique_roots_with_both_raw_policies_and_both_authenticated_mcts_actions"
        )
        == MIN_MCTS_PAIRED_ROOTS,
        "mcts_paired_root_threshold_changed",
    )
    criteria = freeze.get("adequacy_criteria_unchanged_from_attempt_0001", {})
    require(
        criteria.get(
            "minimum_paired_unique_roots_with_exact_margins_and_both_authenticated_mcts_actions"
        )
        == MIN_MCTS_EXACT_ROOTS
        and criteria.get("minimum_seed615_opening_groups") == MIN_OPENINGS
        and criteria.get("minimum_membership_classes_with_nonzero_exact_coverage")
        == MIN_MEMBERSHIPS
        and criteria.get("minimum_phase_strata_with_nonzero_exact_coverage")
        == MIN_PHASES,
        "coverage_threshold_changed",
    )


def verify_source_anchors(freeze: dict[str, Any]) -> None:
    source = freeze["source_inventory"]
    seed615 = source["seed615_evaluation"]
    seed633_policy = source["seed633_policy_metrics"]
    seed633_exact = source["seed633_exact_margins"]
    require(
        file_sha256(EVAL / "execution-freeze-v4.json")
        == seed615["execution_freeze_sha256"],
        "seed615_execution_freeze_identity_mismatch",
    )
    require(
        file_sha256(DATA / "posthoc-policy-diagnostic/diagnostic.json")
        == seed615["diagnostic_sha256"],
        "seed615_diagnostic_identity_mismatch",
    )
    diagnostic = read_json(DATA / "posthoc-policy-diagnostic/diagnostic.json")
    require(
        diagnostic.get("authentication", {}).get("checkpoint_sha256", {}).get("B512")
        == CHECKPOINTS["B512"]
        and diagnostic.get("authentication", {})
        .get("checkpoint_sha256", {})
        .get("S615")
        == CHECKPOINTS["S615"],
        "seed615_diagnostic_checkpoint_identity_mismatch",
    )
    require(
        file_sha256(DATA / "posthoc-policy-diagnostic/first-divergences.jsonl")
        == seed615["pr619_first_divergences_sha256"],
        "seed615_first_divergence_identity_mismatch",
    )
    require(
        file_sha256(POLICY_SOURCE / "freeze.json") == seed633_policy["freeze_sha256"]
        and file_sha256(POLICY_SOURCE / "cohort.jsonl")
        == seed633_policy["cohort_sha256"]
        and file_sha256(POLICY_SOURCE / "per-state-metrics.jsonl")
        == seed633_policy["per_state_metrics_sha256"],
        "seed633_policy_source_identity_mismatch",
    )
    policy_freeze = read_json(POLICY_SOURCE / "freeze.json")
    require(
        policy_freeze.get("checkpoint_sha256") == CHECKPOINTS,
        "seed633_policy_checkpoint_identity_mismatch",
    )
    require(
        file_sha256(EXACT_SOURCE / "freeze.json") == seed633_exact["freeze_sha256"]
        and file_sha256(EXACT_SOURCE / "per-state-results.jsonl")
        == seed633_exact["per_state_results_sha256"],
        "seed633_exact_source_identity_mismatch",
    )
    exact_freeze = read_json(EXACT_SOURCE / "freeze.json")
    require(
        exact_freeze.get("checkpoint_sha256") == CHECKPOINTS,
        "seed633_exact_checkpoint_identity_mismatch",
    )
    require(
        file_sha256(DATA / "evaluation-openings.jsonl")
        == seed615["opening_file_sha256"]
        and file_sha256(DATA / "evaluation-assignments.jsonl")
        == seed615["assignment_file_sha256"],
        "seed615_opening_or_assignment_file_identity_mismatch",
    )
    for source_path, expected in seed615[
        "decision_implementation_source_sha256"
    ].items():
        require(
            file_sha256(Path(source_path)) == expected,
            f"decision_source_code_mismatch:{source_path}",
        )


def unique_source_index(
    rows: list[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = row.get("canonical_identity")
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"{label}_identity_missing")
        require(identity not in indexed, f"{label}_duplicate_identity:{identity}")
        indexed[identity] = row
    return indexed


def expected_assignment_rows(openings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for opening_index, opening in enumerate(openings):
        opening_id = f"seed615-opening-{opening_index:04d}"
        require(opening.get("opening_id") == opening_id, "opening_id_order_mismatch")
        for candidate in ("B512", "S615"):
            for seat in (0, 1):
                seed_hash = hashlib.sha256(
                    f"seed615\0{opening_id}\0{candidate}\0{seat}".encode("utf-8")
                ).digest()
                rows.append(
                    {
                        "candidate": candidate,
                        "candidate_seat": seat,
                        "game_id": f"seed615-{candidate}-{opening_index:04d}-p{seat}",
                        "game_seed": int.from_bytes(seed_hash[:8], "big") % (2**31),
                        "opening_id": opening_id,
                        "opening_identity": opening["canonical_identity"],
                        "opening_index": opening_index,
                        "opponent": "P-E4",
                    }
                )
    return rows


def verify_result_chain(
    assignment: dict[str, Any],
    game_row: dict[str, Any],
    result: dict[str, Any],
    command: list[str],
    report: dict[str, Any],
    game_bytes: bytes,
    command_bytes: bytes,
    report_bytes: bytes,
) -> None:
    require(
        result.get("schema") == "seed615-game-result-v1"
        and all(result.get(key) == value for key, value in assignment.items()),
        "game_assignment_binding_mismatch",
    )
    for name, payload in (
        ("game_sha256", game_bytes),
        ("command_sha256", command_bytes),
        ("report_sha256", report_bytes),
    ):
        require(
            result.get(name) == sha256_bytes(payload),
            f"game_result_content_hash_mismatch:{name}",
        )
    require(
        game_row.get("challenger_player") == assignment["candidate_seat"],
        "candidate_seat_mismatch",
    )
    require(game_row.get("opening_index") == 0, "game_local_opening_index_mismatch")
    require(
        game_row.get("opening_state_hash") == assignment["opening_identity"],
        "game_opening_identity_mismatch",
    )
    require(
        len(command) > 5 and command[1:3] == ["-m", "ml.alphazero_lite.arena"],
        "arena_command_entrypoint_mismatch",
    )
    options = {
        "--challenger": f"runtime-artifacts/{assignment['candidate']}",
        "--current": "runtime-artifacts/P-E4",
    }
    for flag, suffix in options.items():
        require(flag in command, f"arena_command_flag_missing:{flag}")
        require(
            command[command.index(flag) + 1].endswith(suffix),
            f"arena_command_artifact_mismatch:{flag}",
        )
    expected_values = {
        "--challenger-simulations": "128",
        "--current-simulations": "128",
        "--workers": "1",
        "--c-puct": "1.25",
        "--exact-root-solve-threshold": "16",
        "--seed": str(assignment["game_seed"]),
        "--challenger-starts": str(assignment["candidate_seat"]),
        "--max-moves": "200",
    }
    for flag, value in expected_values.items():
        require(flag in command, f"arena_command_flag_missing:{flag}")
        require(
            command[command.index(flag) + 1] == value,
            f"arena_command_value_mismatch:{flag}",
        )
    expected_options = {
        "fpu_mode": "zero",
        "reuse_subtree": False,
        "normalize_values": False,
        "root_policy_mode": "deterministic",
        "root_temperature": 0.0,
        "tactical_root_bias": 0.0,
    }
    for flag in ("--challenger-search-options-json", "--current-search-options-json"):
        require(flag in command, f"arena_command_flag_missing:{flag}")
        require(
            json.loads(command[command.index(flag) + 1]) == expected_options,
            f"arena_search_options_mismatch:{flag}",
        )
    notes = report.get("notes", {})
    profile = notes.get("search_profile", {})
    require(
        report.get("games_played") == 1
        and notes.get("challenger_simulations") == 128
        and notes.get("current_simulations") == 128
        and notes.get("workers_used") == 1
        and notes.get("seed") == assignment["game_seed"]
        and profile.get("exact_root_solve_enabled") is True
        and profile.get("exact_root_solve_threshold") == THRESHOLD
        and profile.get("exact_leaf_solve_mode") == "disabled",
        "arena_report_search_provenance_mismatch",
    )


def replay_trajectory(
    assignment: dict[str, Any],
    opening: dict[str, Any],
    game_row: dict[str, Any],
    result: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, int]:
    require(
        game_row.get("challenger_player") == assignment["candidate_seat"],
        "opponent_attribution_guard",
    )
    game = ReplayBoard()
    prefix = opening.get("prefix_moves")
    if not isinstance(prefix, list) or len(prefix) != 12:
        raise ValueError("opening_prefix_invalid")
    for move in prefix:
        relative_move = int(move)
        require(
            not game.over() and relative_move in game.legal_moves(),
            "illegal_opening_prefix",
        )
        game.play_absolute(game.current_player * 6 + relative_move)
    opening_state = game.to_state()
    require(
        canonical_identity(opening_state) == assignment["opening_identity"],
        "replayed_opening_identity_mismatch",
    )
    require(
        game_row.get("opening_prefix_moves") == prefix,
        "recorded_opening_prefix_mismatch",
    )
    require(
        game_row.get("opening_applied_prefix_length") == len(prefix),
        "recorded_prefix_length_mismatch",
    )

    trajectory = game_row.get("trajectory")
    if not isinstance(trajectory, str):
        raise ValueError("trajectory_missing")
    tokens = trajectory.split(",") if trajectory else []
    require(
        len(tokens) == game_row.get("game_length") and len(tokens) <= 200,
        "trajectory_length_mismatch",
    )
    occurrences: list[dict[str, Any]] = []
    eligible_plies: list[int] = []
    candidate_decisions = 0
    for ply, token in enumerate(tokens):
        try:
            absolute_move = int(token)
        except ValueError as error:
            raise ValueError(f"malformed_move_token:{ply}") from error
        require(not game.over(), f"move_after_terminal:{ply}")
        require(0 <= absolute_move < 12, f"action_out_of_range:{ply}")
        require(
            absolute_move // 6 == game.current_player, f"wrong_player_perspective:{ply}"
        )
        legal = game.legal_moves()
        relative_move = absolute_move % 6
        require(relative_move in legal, f"illegal_recorded_action:{ply}")
        active_stones = sum(game.pits)
        eligible = active_stones <= THRESHOLD
        if eligible:
            eligible_plies.append(ply)
        actor_role = role_for_player(game.current_player, assignment["candidate_seat"])
        if actor_role == "candidate":
            candidate_decisions += 1
            occurrences.append(
                {
                    "canonical_identity": canonical_identity(game.to_state()),
                    "candidate": assignment["candidate"],
                    "candidate_checkpoint_sha256": CHECKPOINTS[assignment["candidate"]],
                    "candidate_seat": assignment["candidate_seat"],
                    "player_to_move": game.current_player,
                    "opening_id": assignment["opening_id"],
                    "opening_identity": assignment["opening_identity"],
                    "opening_index": assignment["opening_index"],
                    "game_id": assignment["game_id"],
                    "game_seed": assignment["game_seed"],
                    "ply": ply,
                    "actor_role": actor_role,
                    "recorded_action": relative_move,
                    "absolute_action": absolute_move,
                    "legal_actions": legal,
                    "active_pit_stones": active_stones,
                    "exact_root_eligible": eligible,
                    "decision_source": "mcts" if not eligible else "unknown",
                }
            )
        else:
            occurrences.append(
                {
                    "canonical_identity": canonical_identity(game.to_state()),
                    "candidate": assignment["candidate"],
                    "candidate_seat": assignment["candidate_seat"],
                    "game_id": assignment["game_id"],
                    "game_seed": assignment["game_seed"],
                    "ply": ply,
                    "actor_role": actor_role,
                    "recorded_action": relative_move,
                    "exact_root_eligible": eligible,
                    "decision_source": "exact_root_eligible_unknown"
                    if eligible
                    else "mcts",
                }
            )
        game.play_absolute(absolute_move)

    handoff_count = game_row.get("exact_root_handoff_count")
    first_handoff = game_row.get("first_exact_root_handoff_ply")
    if type(handoff_count) is not int or handoff_count < 0:
        raise ValueError("exact_handoff_count_invalid")
    if first_handoff is not None and type(first_handoff) is not int:
        raise ValueError("first_exact_handoff_ply_invalid")
    assign_decision_sources(occurrences, eligible_plies, handoff_count, first_handoff)

    terminal = game.over()
    require(terminal or len(tokens) == 200, "trajectory_ended_before_terminal_or_cap")
    margin = int(
        game.stores[assignment["candidate_seat"]]
        - game.stores[1 - assignment["candidate_seat"]]
    )
    winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
    require(game_row.get("margin") == margin, "replayed_terminal_margin_mismatch")
    require(game_row.get("winner") == winner, "replayed_winner_mismatch")
    require(
        result.get("margin") == margin and result.get("winner") == winner,
        "result_outcome_replay_mismatch",
    )
    require(result.get("game_length") == len(tokens), "result_length_replay_mismatch")
    require(
        result.get("trajectory_sha256") == sha256_bytes(trajectory.encode()),
        "result_trajectory_digest_mismatch",
    )
    return occurrences, len(eligible_plies), candidate_decisions


def attach_source_metrics(
    occurrences: list[dict[str, Any]],
    metric_by_id: dict[str, dict[str, Any]],
    exact_by_id: dict[str, dict[str, Any]],
) -> None:
    for row in occurrences:
        if row["actor_role"] != "candidate":
            continue
        identity = row["canonical_identity"]
        metric = metric_by_id.get(identity)
        exact = exact_by_id.get(identity)
        row["has_both_raw_policies"] = bool(
            metric
            and all(
                isinstance(metric.get(model, {}).get("raw_policy"), list)
                and len(metric[model]["raw_policy"]) == 6
                for model in ("B512", "S615")
            )
        )
        row["has_exact_action_margins"] = bool(
            exact and exact.get("exact_action_margins")
        )
        row["membership_class"] = None if exact is None else exact["membership_class"]
        row["phase"] = None if exact is None else exact["phase"]


def pair_occurrences(occurrences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"B512": [], "S615": []}
    )
    for row in occurrences:
        if row["actor_role"] != "candidate":
            continue
        key = (
            row["opening_index"],
            row["candidate_seat"],
            row["canonical_identity"],
        )
        grouped[key][row["candidate"]].append(row)
    pairs: list[dict[str, Any]] = []
    for key, candidates in sorted(grouped.items()):
        b_rows = sorted(candidates["B512"], key=lambda row: row["ply"])
        s_rows = sorted(candidates["S615"], key=lambda row: row["ply"])
        for ordinal, (b512, s615) in enumerate(zip(b_rows, s_rows, strict=False)):
            require(
                b512["opening_identity"] == s615["opening_identity"],
                "paired_opening_identity_mismatch",
            )
            pairs.append(
                {
                    "opening_index": key[0],
                    "candidate_seat": key[1],
                    "canonical_identity": key[2],
                    "occurrence_ordinal": ordinal,
                    "B512": b512,
                    "S615": s615,
                    "both_raw_policies": b512["has_both_raw_policies"]
                    and s615["has_both_raw_policies"],
                    "exact_margins": b512["has_exact_action_margins"]
                    and s615["has_exact_action_margins"],
                    "both_mcts": b512["decision_source"] == "mcts"
                    and s615["decision_source"] == "mcts",
                }
            )
    return pairs


def _unique_roots(rows: list[dict[str, Any]]) -> set[str]:
    return {row["canonical_identity"] for row in rows}


def verify_pr619_trajectory_crosswalk(
    divergence_rows: list[dict[str, Any]],
    assignments: dict[str, dict[str, Any]],
    game_rows: dict[str, dict[str, Any]],
    occurrences: list[dict[str, Any]],
) -> None:
    rows_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for row in divergence_rows:
        opening_index = row.get("opening_index")
        seat = row.get("candidate_seat")
        if type(opening_index) is not int or type(seat) is not int:
            raise ValueError("pr619_pair_key_invalid")
        key = (opening_index, seat)
        require(key not in rows_by_pair, "duplicate_pr619_trajectory_pair")
        rows_by_pair[key] = row
    require(len(rows_by_pair) == 2048, "pr619_trajectory_pair_count_mismatch")
    occurrence_by_game_ply = {(row["game_id"], row["ply"]): row for row in occurrences}
    for (opening_index, seat), diagnostic in rows_by_pair.items():
        game_ids = {
            model: f"seed615-{model}-{opening_index:04d}-p{seat}"
            for model in ("B512", "S615")
        }
        b_assignment = assignments[game_ids["B512"]]
        s_assignment = assignments[game_ids["S615"]]
        b_game = game_rows[game_ids["B512"]]
        s_game = game_rows[game_ids["S615"]]
        same_trajectory = b_game["trajectory"] == s_game["trajectory"]
        require(
            diagnostic.get("identical_trajectory") is same_trajectory,
            "pr619_trajectory_identity_disagreement",
        )
        require(
            diagnostic.get("outcome_b512") == b_game.get("winner")
            and diagnostic.get("outcome_s615") == s_game.get("winner"),
            "pr619_paired_outcome_binding_mismatch",
        )
        if same_trajectory:
            require(
                diagnostic.get("first_differing_move_index") is None,
                "pr619_identical_pair_has_divergence",
            )
            continue
        b_moves = b_game["trajectory"].split(",")
        s_moves = s_game["trajectory"].split(",")
        different_indices = [
            index
            for index, (b_move, s_move) in enumerate(
                zip(b_moves, s_moves, strict=False)
            )
            if b_move != s_move
        ]
        require(bool(different_indices), "pr619_trajectory_length_only_difference")
        first_difference = different_indices[0]
        require(
            diagnostic.get("first_differing_move_index") == first_difference,
            "pr619_first_divergence_ply_mismatch",
        )
        b_decision = occurrence_by_game_ply.get((game_ids["B512"], first_difference))
        s_decision = occurrence_by_game_ply.get((game_ids["S615"], first_difference))
        if b_decision is None or s_decision is None:
            raise ValueError("pr619_divergence_decision_missing")
        require(
            b_decision["actor_role"] == "candidate"
            and s_decision["actor_role"] == "candidate",
            "pr619_divergence_attributed_to_opponent",
        )
        identity = diagnostic.get("canonical_state_identity")
        require(
            b_decision["canonical_identity"]
            == s_decision["canonical_identity"]
            == identity,
            "pr619_divergence_state_identity_mismatch",
        )
        require(
            b_decision["recorded_action"] == diagnostic.get("b512_move")
            and s_decision["recorded_action"] == diagnostic.get("s615_move"),
            "pr619_recorded_action_mismatch",
        )
        require(
            b_decision["player_to_move"]
            == s_decision["player_to_move"]
            == diagnostic.get("player_to_move"),
            "pr619_player_perspective_mismatch",
        )
        require(
            b_assignment["game_seed"] == diagnostic.get("b512_seed")
            and s_assignment["game_seed"] == diagnostic.get("s615_seed"),
            "pr619_candidate_seed_mismatch",
        )


def build_coverage(freeze: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    validate_freeze(freeze)
    verify_source_anchors(freeze)
    seed615 = freeze["source_inventory"]["seed615_evaluation"]
    openings = read_jsonl(DATA / "evaluation-openings.jsonl")
    assignments = read_jsonl(DATA / "evaluation-assignments.jsonl")
    require(
        len(assignments) == seed615["game_record_count"], "assignment_count_mismatch"
    )
    require(
        len({row.get("game_id") for row in assignments}) == len(assignments),
        "duplicate_game_assignment",
    )
    require(
        assignments == expected_assignment_rows(openings),
        "assignment_seed_or_order_mismatch",
    )
    opening_by_id = {row["opening_id"]: row for row in openings}

    game_manifest: list[tuple[str, str]] = []
    result_manifest: list[tuple[str, str]] = []
    assignments_by_id = {row["game_id"]: row for row in assignments}
    game_rows_by_id: dict[str, dict[str, Any]] = {}
    result_by_id: dict[str, dict[str, Any]] = {}
    for game_id, assignment in assignments_by_id.items():
        result_path = RESULTS / game_id / "result.json"
        result = read_json(result_path)
        relative_attempt = result.get("attempt_directory")
        require(
            relative_attempt == "attempts/attempt-0001", "attempt_directory_mismatch"
        )
        attempt = RESULTS / game_id / relative_attempt
        game_path = attempt / "game.jsonl"
        command_path = attempt / "command.json"
        report_path = attempt / "report.json"
        game_bytes = (ROOT / game_path).read_bytes()
        command_bytes = (ROOT / command_path).read_bytes()
        report_bytes = (ROOT / report_path).read_bytes()
        game_manifest.append((game_path.as_posix(), sha256_bytes(game_bytes)))
        result_manifest.append((result_path.as_posix(), file_sha256(result_path)))
        game_lines = game_bytes.splitlines()
        require(len(game_lines) == 1, "game_record_count_mismatch")
        game_row = json.loads(game_lines[0])
        verify_result_chain(
            assignment,
            game_row,
            result,
            json.loads(command_bytes),
            json.loads(report_bytes),
            game_bytes,
            command_bytes,
            report_bytes,
        )
        game_rows_by_id[game_id] = game_row
        result_by_id[game_id] = result
    require(
        sha256_bytes(canonical_json(sorted(game_manifest)))
        == seed615["trajectory_files_manifest_sha256"],
        "full_trajectory_archive_manifest_mismatch",
    )
    require(
        sha256_bytes(canonical_json(sorted(result_manifest)))
        == seed615["result_files_manifest_sha256"],
        "full_result_archive_manifest_mismatch",
    )

    policy_rows = read_jsonl(POLICY_SOURCE / "per-state-metrics.jsonl")
    exact_rows = read_jsonl(EXACT_SOURCE / "per-state-results.jsonl")
    require(len(policy_rows) == 4521, "policy_metric_row_count_mismatch")
    require(len(exact_rows) == 1481, "exact_row_count_mismatch")
    metric_by_id = unique_source_index(policy_rows, label="policy_metric")
    exact_by_id = unique_source_index(exact_rows, label="exact_margin")
    for identity, row in metric_by_id.items():
        require(
            row.get("canonical_identity") == identity, "policy_metric_identity_mismatch"
        )
        for model in ("B512", "S615"):
            raw = row.get(model, {}).get("raw_policy")
            require(
                isinstance(raw, list)
                and len(raw) == 6
                and all(isinstance(item, (int, float)) for item in raw),
                f"raw_policy_vector_invalid:{model}",
            )
    for identity, row in exact_by_id.items():
        require(
            row.get("canonical_identity") == identity, "exact_margin_identity_mismatch"
        )
        margins = row.get("exact_action_margins")
        if not isinstance(margins, dict) or not margins:
            raise ValueError("exact_action_margin_map_missing")
        legal_actions = row.get("legal_actions")
        if not isinstance(legal_actions, list):
            raise ValueError("exact_legal_action_list_missing")
        require(
            set(map(int, margins)) == set(legal_actions),
            "exact_margin_legal_action_coverage_mismatch",
        )

    trajectory_pairs: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(
        dict
    )
    all_occurrences: list[dict[str, Any]] = []
    total_eligible_positions = 0
    telemetry_exact_assignments = Counter()
    per_candidate_decisions = Counter()
    opponent_decisions = Counter()
    for game_id, assignment in assignments_by_id.items():
        row = game_rows_by_id[game_id]
        trajectory_pair_key = (
            assignment["opening_index"],
            assignment["candidate_seat"],
        )
        require(
            assignment["candidate"] not in trajectory_pairs[trajectory_pair_key],
            "duplicate_trajectory_pair_member",
        )
        trajectory_pairs[trajectory_pair_key][assignment["candidate"]] = row
        candidate_rows, eligible_positions, candidate_decisions = replay_trajectory(
            assignment,
            opening_by_id[assignment["opening_id"]],
            row,
            result_by_id[game_id],
        )
        total_eligible_positions += eligible_positions
        per_candidate_decisions[assignment["candidate"]] += candidate_decisions
        opponent_decisions["total"] += (
            len(row.get("trajectory", "").split(",")) - candidate_decisions
        )
        opponent_decisions["exact_root_solver"] += sum(
            item["decision_source"] == "exact_root_solver_opponent"
            for item in candidate_rows
        )
        opponent_decisions["eligible_unknown"] += sum(
            item["decision_source"] == "exact_root_eligible_unknown"
            for item in candidate_rows
        )
        all_occurrences.extend(candidate_rows)
        telemetry_exact_assignments["total_handoff_calls"] += row[
            "exact_root_handoff_count"
        ]
        telemetry_exact_assignments["identified_candidate_solver_actions"] += sum(
            item["decision_source"] == "exact_root_solver" for item in candidate_rows
        )
        telemetry_exact_assignments["identified_opponent_solver_actions"] += sum(
            item["decision_source"] == "exact_root_solver_opponent"
            for item in candidate_rows
        )
        telemetry_exact_assignments["unknown_eligible_total"] += sum(
            item["decision_source"] == "unknown" for item in candidate_rows
        )

    pr619_rows = read_jsonl(DATA / "posthoc-policy-diagnostic/first-divergences.jsonl")
    verify_pr619_trajectory_crosswalk(
        pr619_rows, assignments_by_id, game_rows_by_id, all_occurrences
    )

    identical_trajectories = 0
    divergent_trajectories = 0
    for key, pair in trajectory_pairs.items():
        require(set(pair) == {"B512", "S615"}, "incomplete_paired_trajectory")
        if pair["B512"].get("trajectory") == pair["S615"].get("trajectory"):
            identical_trajectories += 1
        else:
            divergent_trajectories += 1
    require(
        identical_trajectories + divergent_trajectories == 2048,
        "trajectory_pair_count_mismatch",
    )

    attach_source_metrics(all_occurrences, metric_by_id, exact_by_id)
    paired_occurrences = pair_occurrences(all_occurrences)
    matched_with_both_raw = [
        pair for pair in paired_occurrences if pair["both_raw_policies"]
    ]
    matched_with_exact = [
        pair for pair in matched_with_both_raw if pair["exact_margins"]
    ]
    matched_both_mcts = [pair for pair in matched_with_both_raw if pair["both_mcts"]]
    matched_both_solver = [
        pair
        for pair in matched_with_both_raw
        if pair["B512"]["decision_source"] == "exact_root_solver"
        and pair["S615"]["decision_source"] == "exact_root_solver"
    ]
    eligible_mcts_exact = [pair for pair in matched_both_mcts if pair["exact_margins"]]

    occurrences_by_mechanism = Counter(
        row["decision_source"]
        for row in all_occurrences
        if row["actor_role"] == "candidate"
    )
    occurrences_with_raw = [
        row
        for row in all_occurrences
        if row["actor_role"] == "candidate" and row["has_both_raw_policies"]
    ]
    occurrences_with_exact = [
        row
        for row in all_occurrences
        if row["actor_role"] == "candidate" and row["has_exact_action_margins"]
    ]
    eligible_memberships = Counter(
        pair["B512"]["membership_class"] for pair in eligible_mcts_exact
    )
    eligible_phases = Counter(pair["B512"]["phase"] for pair in eligible_mcts_exact)
    solver_exact_pairs = [pair for pair in matched_both_solver if pair["exact_margins"]]
    solver_memberships = Counter(
        pair["B512"]["membership_class"] for pair in solver_exact_pairs
    )
    solver_phases = Counter(pair["B512"]["phase"] for pair in solver_exact_pairs)
    all_exact_memberships = Counter(
        pair["B512"]["membership_class"] for pair in matched_with_exact
    )
    all_exact_phases = Counter(pair["B512"]["phase"] for pair in matched_with_exact)
    exact_cohort_memberships = Counter(row["membership_class"] for row in exact_rows)
    exact_cohort_phases = Counter(row["phase"] for row in exact_rows)
    adequate = (
        len(_unique_roots(matched_both_mcts)) >= MIN_MCTS_PAIRED_ROOTS
        and len(_unique_roots(eligible_mcts_exact)) >= MIN_MCTS_EXACT_ROOTS
        and len({pair["opening_index"] for pair in eligible_mcts_exact}) >= MIN_OPENINGS
        and len(eligible_memberships) >= MIN_MEMBERSHIPS
        and len(eligible_phases) >= MIN_PHASES
    )

    public_occurrences = [
        {
            key: row[key]
            for key in (
                "canonical_identity",
                "candidate",
                "candidate_checkpoint_sha256",
                "candidate_seat",
                "recorded_action",
                "opening_id",
                "opening_identity",
                "opening_index",
                "game_id",
                "game_seed",
                "ply",
                "active_pit_stones",
                "exact_root_eligible",
                "decision_source",
                "has_both_raw_policies",
                "has_exact_action_margins",
                "membership_class",
                "phase",
            )
        }
        for row in all_occurrences
        if row["actor_role"] == "candidate"
    ]
    occurrence_payload = deterministic_occurrence_bytes(public_occurrences)

    result = {
        "schema": "policy-to-search-historical-coverage-v2",
        "attempt": "attempt-0002",
        "scope": "retrospective evidence coverage only",
        "final_classification": "coverage_adequate_quality_analysis_not_run"
        if adequate
        else "historical_evidence_insufficient",
        "source_coverage": {
            "seed615_games": len(assignments),
            "seed615_candidate_paired_trajectories": len(trajectory_pairs),
            "seed615_identical_trajectories": identical_trajectories,
            "seed615_divergent_trajectories": divergent_trajectories,
            "seed633_raw_policy_metric_states": len(metric_by_id),
            "seed633_exact_margin_states": len(exact_by_id),
            "seed633_exact_cohort_by_membership_class": dict(
                sorted(exact_cohort_memberships.items())
            ),
            "seed633_exact_cohort_by_phase": dict(sorted(exact_cohort_phases.items())),
            "pr619_first_divergences_cross_checked": len(pr619_rows),
        },
        "decision_source_coverage": {
            "candidate_decision_occurrences": sum(per_candidate_decisions.values()),
            "candidate_occurrences_by_checkpoint": dict(
                sorted(per_candidate_decisions.items())
            ),
            "candidate_occurrences_by_decision_source": {
                name: occurrences_by_mechanism.get(name, 0)
                for name in ("mcts", "exact_root_solver", "unknown", "other_registered")
            },
            "candidate_occurrences_by_decision_source_and_checkpoint": {
                candidate: {
                    source_name: sum(
                        row["candidate"] == candidate
                        and row["actor_role"] == "candidate"
                        and row["decision_source"] == source_name
                        for row in all_occurrences
                    )
                    for source_name in ("mcts", "exact_root_solver", "unknown")
                }
                for candidate in ("B512", "S615")
            },
            "candidate_occurrences_at_exact_root_eligible_states": sum(
                row["exact_root_eligible"]
                for row in all_occurrences
                if row["actor_role"] == "candidate"
            ),
            "candidate_occurrences_with_both_raw_policies": len(occurrences_with_raw),
            "candidate_occurrences_with_exact_margins": len(occurrences_with_exact),
            "candidate_unique_roots_with_raw_policies": len(
                _unique_roots(occurrences_with_raw)
            ),
            "candidate_unique_roots_with_exact_margins": len(
                _unique_roots(occurrences_with_exact)
            ),
            "opponent_decision_occurrences_not_attributed_to_candidate": opponent_decisions[
                "total"
            ],
            "opponent_exact_root_solver_actions": opponent_decisions[
                "exact_root_solver"
            ],
            "opponent_exact_root_eligible_unknown_actions": opponent_decisions[
                "eligible_unknown"
            ],
            "exact_root_handoff_telemetry": dict(
                sorted(telemetry_exact_assignments.items())
            ),
            "all_player_exact_root_eligible_turn_occurrences": total_eligible_positions,
        },
        "paired_decision_coverage": {
            "one_to_one_same_opening_seat_state_occurrence_pairs": len(
                paired_occurrences
            ),
            "unique_matched_roots": len(_unique_roots(paired_occurrences)),
            "matched_occurrences_with_both_raw_policies": len(matched_with_both_raw),
            "unique_matched_roots_with_both_raw_policies": len(
                _unique_roots(matched_with_both_raw)
            ),
            "matched_occurrences_with_exact_margins": len(matched_with_exact),
            "unique_matched_roots_with_exact_margins": len(
                _unique_roots(matched_with_exact)
            ),
            "opening_groups_with_exact_margins": len(
                {pair["opening_index"] for pair in matched_with_exact}
            ),
            "by_membership_class_for_all_exact_margin_pairs": dict(
                sorted(all_exact_memberships.items())
            ),
            "by_phase_for_all_exact_margin_pairs": dict(
                sorted(all_exact_phases.items())
            ),
            "matched_occurrences_with_both_authenticated_mcts_actions": len(
                matched_both_mcts
            ),
            "unique_roots_with_both_raw_policies_and_mcts_actions": len(
                _unique_roots(matched_both_mcts)
            ),
            "matched_occurrences_with_both_exact_solver_actions": len(
                matched_both_solver
            ),
            "unique_roots_with_both_exact_solver_actions": len(
                _unique_roots(matched_both_solver)
            ),
            "matched_occurrences_with_exact_margins_and_both_exact_solver_actions": len(
                solver_exact_pairs
            ),
            "unique_roots_with_exact_margins_and_both_exact_solver_actions": len(
                _unique_roots(solver_exact_pairs)
            ),
            "opening_groups_with_exact_margins_and_both_exact_solver_actions": len(
                {pair["opening_index"] for pair in solver_exact_pairs}
            ),
            "by_membership_class_for_exact_margin_solver_pairs": dict(
                sorted(solver_memberships.items())
            ),
            "by_phase_for_exact_margin_solver_pairs": dict(
                sorted(solver_phases.items())
            ),
            "matched_occurrences_with_exact_margins_and_both_mcts_actions": len(
                eligible_mcts_exact
            ),
            "unique_roots_with_exact_margins_and_both_mcts_actions": len(
                _unique_roots(eligible_mcts_exact)
            ),
            "opening_groups_with_exact_margins_and_both_mcts_actions": len(
                {pair["opening_index"] for pair in eligible_mcts_exact}
            ),
            "by_membership_class_for_exact_margin_mcts_pairs": dict(
                sorted(eligible_memberships.items())
            ),
            "by_phase_for_exact_margin_mcts_pairs": dict(
                sorted(eligible_phases.items())
            ),
        },
        "decision_occurrence_artifact": {
            "path": (ATTEMPT / "trajectory-decision-coverage.jsonl.gz").as_posix(),
            "candidate_decision_occurrences": len(public_occurrences),
            "sha256": sha256_bytes(occurrence_payload),
            "encoding": "deterministic gzip of compact canonical JSONL; one row per candidate decision occurrence",
        },
        "go_no_go": {
            "minimum_paired_unique_roots_with_both_raw_policies_and_both_authenticated_mcts_actions": MIN_MCTS_PAIRED_ROOTS,
            "minimum_paired_unique_roots_with_exact_margins_and_both_authenticated_mcts_actions": MIN_MCTS_EXACT_ROOTS,
            "minimum_seed615_opening_groups": MIN_OPENINGS,
            "minimum_membership_classes_with_nonzero_exact_coverage": MIN_MEMBERSHIPS,
            "minimum_phase_strata_with_nonzero_exact_coverage": MIN_PHASES,
            "passed": adequate,
        },
        "limits": [
            "No action regret, optimal-action agreement, mechanism effect, or outcome association was calculated.",
            "Seed 615 B512/S615 candidate seeds differ; no causal attribution is made.",
            "The historical source-opening to Seed 624 opening crosswalk remains unresolved and unused.",
            "Repeated occurrences are preserved in occurrence counts; exact-root pairs are matched one-to-one without Cartesian joins.",
        ],
    }
    return result, occurrence_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--freeze", type=Path, default=FREEZE_PATH)
    args = parser.parse_args()
    freeze = read_json(args.freeze)
    result, occurrence_payload = build_coverage(freeze)
    output = ROOT / args.output
    occurrence_output = ROOT / ATTEMPT / "trajectory-decision-coverage.jsonl.gz"
    if args.verify_existing:
        require(output.is_file(), "published_coverage_missing")
        require(
            json.loads(output.read_bytes()) == result, "published_coverage_mismatch"
        )
        require(occurrence_output.is_file(), "published_occurrence_artifact_missing")
        require(
            occurrence_output.read_bytes() == occurrence_payload,
            "published_occurrence_artifact_mismatch",
        )
        print(json.dumps({"status": "verified", "coverage": result}, sort_keys=True))
        return
    if output.exists() or occurrence_output.exists():
        raise FileExistsError(f"create_only_coverage_exists:{args.output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    occurrence_output.write_bytes(occurrence_payload)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
