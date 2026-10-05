"""Build and verify the prospective #415–421 seed422 exclusion union."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
INITIAL = suites.INITIAL_STATE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def add_state(states: set[str], state: dict[str, Any]) -> None:
    states.add(suites.canonical_key(state))


def replay_opening(row: dict[str, Any]) -> KalahGame:
    if row.get("opening_contract") != "arena_player_relative_v2":
        raise ValueError("opening_contract_missing")
    game = KalahGame.from_state(INITIAL)
    actions = [int(action) for action in row["prefix_moves"]]
    for action in actions:
        if game.over() or action not in game.possible_moves():
            raise ValueError("opening_prefix_replay_incomplete")
        if not game.move(game.pit_index(action)):
            raise ValueError("opening_prefix_replay_incomplete")
    if suites.canonical_key(game.to_state()) != suites.canonical_key(row["state"]):
        raise ValueError("opening_prefix_state_mismatch")
    return game


def replay_game(opening: dict[str, Any], game_row: dict[str, Any]) -> set[str]:
    game = replay_opening(opening)
    consumed = {suites.canonical_key(game.to_state())}
    trajectory = [int(value) for value in game_row["trajectory"].split(",") if value]
    if len(trajectory) != int(game_row["game_length"]):
        raise ValueError("trajectory_length_mismatch")
    for ply, absolute in enumerate(trajectory):
        if game.over() or absolute // 6 != game.current_player:
            raise ValueError(f"absolute_action_actor_invalid:{ply}")
        if not game.move(absolute):
            raise ValueError(f"absolute_action_illegal:{ply}")
        consumed.add(suites.canonical_key(game.to_state()))
    if not game.over():
        raise ValueError("trajectory_not_terminal")
    if game.captured_seeds[int(game_row["challenger_player"])] - game.captured_seeds[
        1 - int(game_row["challenger_player"])
    ] != int(game_row["margin"]):
        raise ValueError("trajectory_margin_mismatch")
    margin = int(game_row["margin"])
    winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
    if winner != game_row["winner"]:
        raise ValueError("trajectory_winner_mismatch")
    return consumed


def _source_record(
    path: Path, identity_sets: list[tuple[str, set[str]]], coverage: str
) -> dict[str, Any]:
    identities: set[str] = set()
    for _name, values in identity_sets:
        identities |= values
    overlaps: dict[str, int] = {}
    prior: set[str] = set()
    for name, values in identity_sets:
        overlaps[name] = len(identities & values) if False else len(prior & values)
        prior |= values
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "identity_count": len(identities),
        "contribution": len(
            identities - set().union(*(v for _, v in identity_sets[:-1]))
        )
        if len(identity_sets) > 1
        else len(identities),
        "overlaps": overlaps,
        "coverage": coverage,
    }


def _instrumented_hashes(path: Path) -> set[str]:
    result: set[str] = set()
    for row in rows(path):
        result_row = row.get("result", {})
        for condition in ("off", "on"):
            for item in result_row.get(f"{condition}_request_trace", []):
                state_hash = item.get("state_sha256")
                if not state_hash or len(state_hash) != 64:
                    raise ValueError(f"instrumented_state_hash_invalid:{path}")
                result.add(str(state_hash))
    return result


def build_union() -> tuple[set[str], dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    segments: list[tuple[str, set[str]]] = []

    def record(name: str, path: Path, identities: set[str], coverage: str) -> None:
        overlap_with: dict[str, int] = {}
        for previous_name, previous_set in segments:
            overlap_with[previous_name] = len(identities & previous_set)
        contribution = len(identities - set().union(*(s for _, s in segments)))
        sources.append(
            {
                "name": name,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "identity_count": len(identities),
                "unique_contribution": contribution,
                "overlaps": overlap_with,
                "coverage": coverage,
            }
        )
        segments.append((name, identities))

    corrected_415_path = (
        DATA
        / "seed414-root-budget-confirmation/post-execution-corrected-exclusion-proof.json"
    )
    corrected_415 = json.loads(corrected_415_path.read_text())
    corrected_415_ids = set(corrected_415["excluded_state_identities"])
    if len(corrected_415_ids) != int(corrected_415["corrected_union_count"]):
        raise ValueError("corrected_415_union_count_mismatch")
    record(
        "corrected_415_historical_union",
        corrected_415_path,
        corrected_415_ids,
        "corrected post-execution #415 union, included by #416 v3",
    )

    proof_path = (
        DATA / "seed416-policy-target-softening/opening-exclusion-proof-v3.json"
    )
    proof = json.loads(proof_path.read_text())
    corrected_binding = proof["corrected_415_proof"]
    if Path(
        corrected_binding["path"]
    ).resolve() != corrected_415_path.resolve() or corrected_binding[
        "sha256"
    ] != sha256(corrected_415_path):
        raise ValueError("#416 proof corrected_415_binding_mismatch")
    base_ids = set(proof["excluded_identities"])
    if len(base_ids) != int(proof["combined_union_count"]):
        raise ValueError("validated_416_union_count_mismatch")
    if not corrected_415_ids <= base_ids:
        raise ValueError("validated_416_union_omits_corrected_415_identities")
    record(
        "validated_seed416_historical_union_including_corrected_415",
        proof_path,
        base_ids,
        "validated historical union; prior #415 sources plus #416 registered replay identities",
    )

    suite_path = DATA / "seed416-policy-target-softening/openings-v3.jsonl"
    suite = rows(suite_path)
    suite_ids: set[str] = set()
    for opening in suite:
        game = replay_opening(opening)
        suite_ids.add(suites.canonical_key(game.to_state()))
    if proof.get("suite_overlap") != 0 or suite_ids & base_ids:
        raise ValueError("validated_416_suite_overlap_mismatch")
    record(
        "416_declared_suite",
        suite_path,
        suite_ids,
        "declared opening roots strictly replayed",
    )

    ledger_path = DATA / "seed416-policy-target-softening/outcome-ledger.jsonl"
    ledger = rows(ledger_path)
    if len(ledger) != 2048:
        raise ValueError("416_complete_game_ledger_count_mismatch")
    game_ids: set[str] = set()
    for row in ledger:
        opening_index = int(row["opening_id"])
        game_ids |= replay_game(suite[opening_index], row["game"])
    if {
        (row["lane"], int(row["opening_id"]), int(row["game"]["challenger_player"]))
        for row in ledger
    } != {
        (lane, opening, seat)
        for lane in ("A", "B")
        for opening in range(512)
        for seat in (0, 1)
    }:
        raise ValueError("416_complete_game_ledger_accounting_mismatch")
    record(
        "416_complete_replayed_game_trajectories",
        ledger_path,
        game_ids,
        "all starting, intermediate decision, and terminal states from validated absolute arena trajectories",
    )

    path418 = DATA / "seed418-native-root-handoff/cohort-manifest.json"
    manifest418 = json.loads(path418.read_text())
    registration418_path = path418.with_name("preregistration.json")
    registration418 = json.loads(registration418_path.read_text())
    if registration418["source_suite_sha256"] != sha256(
        DATA / "seed416-policy-target-softening/openings-v3.jsonl"
    ) or registration418["source_ledger_sha256"] != sha256(
        DATA / "seed416-policy-target-softening/outcome-ledger.jsonl"
    ):
        raise ValueError("418_declared_source_binding_mismatch")
    record(
        "418_frozen_preregistration",
        registration418_path,
        set(),
        "binds the retrospective #416 suite and completed trajectory ledger",
    )
    ids418: set[str] = set()
    for entry in manifest418["states"]:
        game = KalahGame.from_state(INITIAL)
        for relative in entry["opening_prefix_moves"]:
            relative = int(relative)
            ids418.add(suites.canonical_key(game.to_state()))
            if game.over() or relative not in game.possible_moves():
                raise ValueError("418_opening_prefix_replay_failed")
            if not game.move(game.pit_index(relative)):
                raise ValueError("418_opening_prefix_replay_failed")
        ids418.add(suites.canonical_key(game.to_state()))
        if suites.canonical_key(game.to_state()) != entry["source_opening_state_hash"]:
            raise ValueError("418_opening_hash_mismatch")
        for ply, absolute in enumerate(entry["trajectory_prefix_absolute"]):
            ids418.add(suites.canonical_key(game.to_state()))
            if (
                game.over()
                or absolute // 6 != game.current_player
                or not game.move(int(absolute))
            ):
                raise ValueError(f"418_absolute_prefix_invalid:{ply}")
        state = game.to_state()
        if suites.canonical_key(state) != entry["state_hash"]:
            raise ValueError("418_root_state_hash_mismatch")
        ids418.add(entry["state_hash"])
    for fname in (
        "search-records.jsonl",
        "oracle-records.jsonl",
        "execution-checkpoint-v2.jsonl",
    ):
        fpath = path418.with_name(fname)
        record_ids: set[str] = set()
        for row in rows(fpath):
            for candidate in (
                row.get("state_hash"),
                row.get("search", {}).get("state_hash"),
                row.get("oracle", {}).get("state_hash"),
            ):
                if candidate:
                    record_ids.add(str(candidate))
        ids418 |= record_ids
        record(
            f"418_published_{fname.removesuffix('.jsonl')}",
            fpath,
            record_ids,
            "published decision/root identities in this artifact; no unrecorded search states reconstructed",
        )
    record(
        "418_declared_roots_and_replayed_prefixes",
        path418,
        ids418,
        "declared roots and every state along published relative/absolute prefixes strictly replayed",
    )

    for seed in (420, 421):
        dirname = (
            "seed420-artifact-evaluator-memoization"
            if seed == 420
            else "seed421-memoization-timing-correction"
        )
        root = DATA / dirname
        cohort_path = root / "cohort.json"
        cohort = json.loads(cohort_path.read_text())
        ids: set[str] = set()
        for entry in cohort:
            state_hash = str(entry["state_hash"])
            if suites.canonical_key(entry["state"]) != state_hash:
                raise ValueError(f"{seed}_cohort_state_hash_mismatch")
            ids.add(state_hash)
        record(
            f"{seed}_declared_instrumented_roots",
            cohort_path,
            ids,
            "published instrumented root states; no unrecorded internal search states inferred",
        )
        ledger_path = root / "benchmark-ledger.jsonl"
        instrumented = _instrumented_hashes(ledger_path)
        record(
            f"{seed}_published_instrumented_evaluation_state_hashes",
            ledger_path,
            instrumented,
            "state_sha256 values in recorded off/on evaluator request traces; traces cover recorded instrumentation only",
        )
    union = set().union(*(values for _, values in segments))
    proof_record = {
        "schema": "seed422-corrected-historical-exclusion-proof-v1",
        "base": "#416 validated v3 historical union including corrected #415 evidence",
        "coverage": "#416 declared openings and every state on its complete published game trajectories; #418 declared roots and published state identities; #420–421 declared instrumented roots and every published evaluator request state hash. No claim is made about unrecorded internal search states.",
        "sources": sources,
        "identity_count": len(union),
        "identity_set_sha256": hashlib.sha256(
            "\n".join(sorted(union)).encode()
        ).hexdigest(),
        "excluded_identities": sorted(union),
        "suite_overlap": 0,
    }
    return union, proof_record


def make_suite(excluded: set[str]) -> list[dict[str, Any]]:
    population = suites.deduplicate_openings(suites.enumerate_legal_prefixes(8))[0]
    eligible = [
        row
        for row in population
        if int(row["pit_sum"]) > 32
        and not KalahGame.from_state(row["state"]).over()
        and suites.canonical_key(row["state"]) not in excluded
    ]
    selected = suites.select_diverse(suites.stratify_openings(eligible), 512, 422)
    result = [suites.export_arena_entry(row) for row in selected]
    identities = suites.validate_arena_entries(result)
    if len(set(identities)) != 512 or set(identities) & excluded:
        raise ValueError("seed422_suite_identity_or_exclusion_failure")
    return result
