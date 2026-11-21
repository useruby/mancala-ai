"""Seed 624 deterministic, model-independent prospective state population.

The module has no model, search, solver, or training imports.  Production
generation is deliberately guarded by an explicit authorization record.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import encode_state

SCHEMA = "seed624-prospective-population-v1"
MASTER_SEED = 62320261121
SEED_CONTRACT = "seed624-sha256-v1; big-endian u64 prefix; Python random.Random"
OPENING_RESERVE = 512
OPENING_TARGET = 256
GAMES_PER_OPENING = 8
PREFIX_ACTIONS = 8
MAX_CONTINUATION_ACTIONS = 200
STONE_TARGETS = {17: 103, 18: 103, 19: 102, 20: 102, 21: 102}
BIN_ORDER = tuple(STONE_TARGETS)
MIN_OPENING_CLUSTERS = 128
ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "ml/alphazero_lite/seed624_prospective_benchmark.py",
    "ml/alphazero_lite/verify_seed624_prospective_benchmark.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/self_play.py",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def identities(state: dict[str, Any]) -> tuple[str, str]:
    """Match Seed 622 canonical JSON and production kalah_v3 float32 identity."""
    canonical_id = sha256(canonical_bytes(state))
    encoded = np.asarray(encode_state(state, input_encoding="kalah_v3"), dtype="<f4")
    return canonical_id, sha256(encoded.tobytes())


def runtime_identity() -> dict[str, str]:
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "prng": f"random.Random/MT19937; Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "optimizer": "seed624-dfs-v1",
    }


def current_source_hashes() -> dict[str, str]:
    return {name: sha256((ROOT / name).read_bytes()) for name in SOURCE_FILES}


def current_git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def derive_seed(
    master_seed: int, opening_index: int, trajectory_index: int, phase: str
) -> int:
    if not 0 <= opening_index < OPENING_RESERVE:
        raise ValueError("opening_index_outside_frozen_reserve")
    if trajectory_index < 0 or phase not in {"opening", "continuation"}:
        raise ValueError("invalid_seed_coordinates")
    message = canonical_bytes(
        {
            "domain": "useruby.mancala-ai/seed624/seed/v1",
            "master_seed": int(master_seed),
            "opening_index": int(opening_index),
            "trajectory_index": int(trajectory_index),
            "phase": phase,
        }
    )
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "big")


def standard_game() -> KalahGame:
    return KalahGame(
        pits=[4] * 12,
        captured_seeds=[0, 0],
        current_player=0,
    )


def _random_game(game: KalahGame, rng: random.Random, limit: int) -> list[int]:
    actions: list[int] = []
    for _ in range(limit):
        legal = game.possible_moves()
        if not legal:
            break
        action = rng.choice(legal)
        if not game.move(game.pit_index(action)):
            raise RuntimeError("rules_engine_rejected_its_legal_action")
        actions.append(action)
        if game.over():
            break
    return actions


def generate_opening(master_seed: int, opening_index: int) -> dict[str, Any]:
    game = standard_game()
    moves = _random_game(
        game,
        random.Random(derive_seed(master_seed, opening_index, 0, "opening")),
        PREFIX_ACTIONS,
    )
    if len(moves) != PREFIX_ACTIONS or game.over():
        raise ValueError("opening_prefix_not_eight_nonterminal_actions")
    canonical_id, float_id = identities(game.to_state())
    return {
        "opening_index": opening_index,
        "seed": derive_seed(master_seed, opening_index, 0, "opening"),
        "state": game.to_state(),
        "canonical_identity": canonical_id,
        "production_float32_identity": float_id,
        "prefix_actions": moves,
    }


def choose_distinct_openings(
    opening_factory: Any, *, reserve: int, target: int
) -> list[dict[str, Any]]:
    """Consume reserve indexes in order and stop rather than drawing replacements."""
    chosen: list[dict[str, Any]] = []
    canonical_seen: set[str] = set()
    float_seen: set[str] = set()
    for index in range(reserve):
        opening = opening_factory(index)
        cid = opening["canonical_identity"]
        fid = opening["production_float32_identity"]
        if cid in canonical_seen or fid in float_seen:
            continue
        canonical_seen.add(cid)
        float_seen.add(fid)
        chosen.append(opening)
        if len(chosen) == target:
            return chosen
    raise ShortfallError(
        {
            "schema": "seed624-generation-shortfall-v1",
            "reason": "opening_reserve_exhausted",
            "required": target,
            "available": len(chosen),
            "reserve": reserve,
        }
    )


def _candidate_rank(
    master_seed: int, opening_id: str, game_id: str, stones: int, cid: str
) -> str:
    return sha256(
        canonical_bytes(
            {
                "domain": "useruby.mancala-ai/seed624/candidate-rank/v1",
                "master_seed": master_seed,
                "opening_id": opening_id,
                "game_id": game_id,
                "stone_bin": stones,
                "canonical_identity": cid,
            }
        )
    )


def generate_game(
    opening: dict[str, Any], master_seed: int, trajectory_index: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    opening_index = int(opening["opening_index"])
    game_id = f"o{opening_index:04d}-g{trajectory_index:02d}"
    game = KalahGame.from_state(opening["state"])
    rng = random.Random(
        derive_seed(master_seed, opening_index, trajectory_index, "continuation")
    )
    trajectory: list[int] = []
    candidates: list[dict[str, Any]] = []
    seen_bins: set[int] = set()
    for ply in range(MAX_CONTINUATION_ACTIONS):
        if game.over():
            break
        legal = game.possible_moves()
        stones = sum(game.pits)
        if stones in STONE_TARGETS and len(legal) >= 2 and stones not in seen_bins:
            state = game.to_state()
            canonical_id, float_id = identities(state)
            candidates.append(
                {
                    "state": state,
                    "canonical_identity": canonical_id,
                    "production_float32_identity": float_id,
                    "active_pit_stones": stones,
                    "legal_actions": legal,
                    "side_to_move": game.current_player,
                    "opening_index": opening_index,
                    "opening_identity": opening["canonical_identity"],
                    "game_id": game_id,
                    "continuation_ply": ply,
                    "trajectory": list(trajectory),
                    "rank": _candidate_rank(
                        master_seed,
                        opening["canonical_identity"],
                        game_id,
                        stones,
                        canonical_id,
                    ),
                }
            )
            seen_bins.add(stones)
        action = rng.choice(legal)
        if not game.move(game.pit_index(action)):
            raise RuntimeError("rules_engine_rejected_its_legal_action")
        trajectory.append(action)
    return {
        "game_id": game_id,
        "opening_index": opening_index,
        "opening_identity": opening["canonical_identity"],
        "seed": derive_seed(
            master_seed, opening_index, trajectory_index, "continuation"
        ),
        "actions": trajectory,
        "full_actions": list(opening["prefix_actions"]) + trajectory,
        "terminal": game.over(),
        "action_limit_reached": len(trajectory) == MAX_CONTINUATION_ACTIONS,
        "final_state": game.to_state(),
    }, candidates


class ShortfallError(ValueError):
    """A hard generation or selection gate failed; carries machine-readable data."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(json.dumps(report, sort_keys=True))


def deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first ranked row; reject canonical OR encoded collisions globally."""
    ordered = sorted(
        candidates, key=lambda row: (row["rank"], row["canonical_identity"])
    )
    canonical_seen: set[str] = set()
    float_seen: set[str] = set()
    result: list[dict[str, Any]] = []
    by_canonical: dict[str, dict[str, Any]] = {}
    by_float: dict[str, dict[str, Any]] = {}
    for row in ordered:
        cid, fid = row["canonical_identity"], row["production_float32_identity"]
        if cid in canonical_seen:
            by_canonical[cid].setdefault("occurrences", []).append(
                {
                    key: row[key]
                    for key in ("opening_index", "game_id", "continuation_ply", "rank")
                }
            )
            continue
        if fid in float_seen:
            by_float[fid].setdefault(
                "float32_identity_collision_occurrences", []
            ).append(
                {
                    "canonical_identity": cid,
                    "opening_index": row["opening_index"],
                    "game_id": row["game_id"],
                    "continuation_ply": row["continuation_ply"],
                    "rank": row["rank"],
                }
            )
            continue
        canonical_seen.add(cid)
        float_seen.add(fid)
        item = {
            **row,
            "occurrences": [
                {
                    key: row[key]
                    for key in ("opening_index", "game_id", "continuation_ply", "rank")
                }
            ],
        }
        by_canonical[cid] = item
        by_float[fid] = item
        result.append(item)
    return result


def screen_known_exposure(
    candidates: list[dict[str, Any]],
    evidence: dict[str, Any],
    *,
    expected_snapshot_hashes: dict[str, str] | None = None,
    expected_provenance_hash: str | None = None,
) -> list[dict[str, Any]]:
    """Fail closed unless every required identity inventory is authenticated."""
    required = ("B512", "S615", "protected_holdout")
    inventories: dict[str, tuple[set[str], set[str]]] = {}
    evidence_hashes: dict[str, str] = {}
    for name in required:
        source = evidence.get(name)
        if not isinstance(source, dict) or source.get("authenticated") is not True:
            raise ValueError(f"invalid_or_unauthenticated_membership:{name}")
        rows = source.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"missing_membership_rows:{name}")
        actual_hash = sha256(canonical_bytes(rows))
        if (
            source.get("sha256") != actual_hash
            or expected_snapshot_hashes is None
            or expected_snapshot_hashes.get(name) != actual_hash
        ):
            raise ValueError(f"membership_hash_mismatch:{name}")
        canonical_ids: set[str] = set()
        float_ids: set[str] = set()
        for row in rows:
            state = row.get("canonical_state")
            if not isinstance(state, dict):
                raise ValueError(f"membership_state_missing:{name}")
            cid, fid = identities(state)
            if cid != row.get("canonical_identity") or fid != row.get(
                "production_float32_identity"
            ):
                raise ValueError(f"membership_identity_mismatch:{name}")
            canonical_ids.add(cid)
            float_ids.add(fid)
        inventories[name] = canonical_ids, float_ids
        evidence_hashes[name] = actual_hash

    provenance = evidence.get("source_provenance", {})
    provenance_complete = provenance.get("authenticated") is True
    if provenance_complete:
        provenance_payload = {
            key: value
            for key, value in provenance.items()
            if key not in {"authenticated", "sha256"}
        }
        actual_provenance_hash = sha256(canonical_bytes(provenance_payload))
        if (
            expected_provenance_hash is None
            or provenance.get("sha256") != actual_provenance_hash
            or expected_provenance_hash != actual_provenance_hash
        ):
            raise ValueError("source_provenance_hash_mismatch")
    screened = []
    for candidate in candidates:
        cid, fid = (
            candidate["canonical_identity"],
            candidate["production_float32_identity"],
        )
        hits = {
            name: cid in ids or fid in floats
            for name, (ids, floats) in inventories.items()
        }
        provenance_match = False
        if provenance_complete:
            for name in required:
                sources = provenance.get(name)
                if not isinstance(sources, list):
                    raise ValueError(f"source_provenance_incomplete:{name}")
                if any(
                    item.get("opening_identity") == candidate["opening_identity"]
                    or item.get("game_id") == candidate["game_id"]
                    for item in sources
                ):
                    provenance_match = True
                    break
        confirmed = any(hits.values()) or provenance_match
        classification = (
            "confirmed_known_exposure"
            if confirmed
            else "known_exposure_nonmatch_inherited_PE4_unresolved"
            if provenance_complete
            else "invalid_or_insufficiently_authenticated_evidence"
        )
        screened.append(
            {
                **candidate,
                "exposure": hits,
                "source_provenance_match": provenance_match,
                "classification": classification,
                "membership_evidence_sha256": evidence_hashes,
                "inherited_PE4_exposure": "unresolved",
            }
        )
    return screened


def select_cohort(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Pinned deterministic rank-first constrained selection for frozen inventories.

    This exact search visits candidates in the declared within-bin rank order.
    It returns the first feasible lexicographic rank vector and has a hard
    1-worker, 3600-second node budget. Exhaustion is a structured shortfall.
    """
    eligible = [
        row
        for row in candidates
        if row.get("classification")
        == "known_exposure_nonmatch_inherited_PE4_unresolved"
    ]
    by_bin = {stones: [] for stones in BIN_ORDER}
    for row in eligible:
        stones = int(row["active_pit_stones"])
        if stones in by_bin:
            by_bin[stones].append(row)
    supply = {stones: len(rows) for stones, rows in by_bin.items()}
    hard_supply = {stones: 2 * target for stones, target in STONE_TARGETS.items()}
    short = {
        stones: {"required": hard_supply[stones], "available": supply[stones]}
        for stones in BIN_ORDER
        if supply[stones] < hard_supply[stones]
    }
    if short:
        raise ShortfallError(
            {
                "schema": "seed624-selection-shortfall-v1",
                "reason": "hard_supply",
                "bins": short,
            }
        )

    # Exact lexicographic depth-first feasibility search. Branches obey every
    # quota and cap; memoization removes equivalent capacity states.
    from time import monotonic

    start = monotonic()
    budget_seconds = 3600
    ordered = {
        stones: sorted(rows, key=lambda row: (row["rank"], row["canonical_identity"]))
        for stones, rows in by_bin.items()
    }
    nodes = 0
    chosen: list[dict[str, Any]] = []
    game_counts: Counter[str] = Counter()
    opening_counts: Counter[int] = Counter()
    opening_set: set[int] = set()
    failed: set[tuple[Any, ...]] = set()

    def search(bin_index: int, start_index: int) -> bool:
        nonlocal nodes
        nodes += 1
        if nodes % 1000 == 0 and monotonic() - start > budget_seconds:
            raise TimeoutError
        if bin_index == len(BIN_ORDER):
            return len(opening_set) >= MIN_OPENING_CLUSTERS
        stones = BIN_ORDER[bin_index]
        target = STONE_TARGETS[stones]
        in_bin = sum(row["active_pit_stones"] == stones for row in chosen)
        remaining = target - in_bin
        rows = ordered[stones]
        if remaining == 0:
            return search(bin_index + 1, 0)
        available = [
            (index, row)
            for index, row in enumerate(rows[start_index:], start_index)
            if game_counts[row["game_id"]] < 2
            and opening_counts[int(row["opening_index"])] < 4
        ]
        if len(available) < remaining:
            return False
        key = (
            bin_index,
            start_index,
            remaining,
            tuple(sorted(game_counts.items())),
            tuple(sorted(opening_counts.items())),
        )
        if key in failed:
            return False
        # Preserve rank order and explore include before skip. The per-bin
        # selected rank vector is therefore lexicographically minimal.
        for offset, (index, row) in enumerate(available):
            chosen.append(row)
            game_counts[row["game_id"]] += 1
            opening = int(row["opening_index"])
            opening_counts[opening] += 1
            was_new = opening not in opening_set
            opening_set.add(opening)
            if search(bin_index, index + 1):
                return True
            if was_new and opening_counts[opening] == 1:
                opening_set.remove(opening)
            opening_counts[opening] -= 1
            game_counts[row["game_id"]] -= 1
            chosen.pop()
            if len(available) - offset - 1 < remaining - 1:
                break
        failed.add(key)
        return False

    try:
        feasible = search(0, 0)
    except TimeoutError:
        raise ShortfallError(
            {
                "schema": "seed624-selection-shortfall-v1",
                "reason": "optimizer_resource_limit",
                "nodes": nodes,
                "runtime_limit_seconds": budget_seconds,
            }
        ) from None
    if not feasible:
        raise ShortfallError(
            {
                "schema": "seed624-selection-shortfall-v1",
                "reason": "constraints_infeasible",
                "supply": supply,
                "distinct_openings": len(
                    {int(row["opening_index"]) for row in eligible}
                ),
                "nodes": nodes,
            }
        )
    selected = sorted(
        chosen,
        key=lambda row: (
            row["active_pit_stones"],
            row["rank"],
            row["canonical_identity"],
        ),
    )
    selected_inventory_sha256 = sha256(canonical_bytes(selected))
    counts = Counter(int(row["active_pit_stones"]) for row in selected)
    openings = Counter(int(row["opening_index"]) for row in selected)
    games = Counter(str(row["game_id"]) for row in selected)
    return {
        "schema": "seed624-selected-cohort-v1",
        "targets": STONE_TARGETS,
        "selection_contract": "exact rank-lexicographic feasible set; canonical identity breaks hash ties; depth-first exhaustive feasibility; implementation seed624-dfs-v1",
        "optimizer": {
            "name": "seed624-dfs-v1",
            "version": 1,
            "workers": 1,
            "runtime_limit_seconds": budget_seconds,
            "nodes": nodes,
        },
        "sampling_contract": "deterministic standardized selected-state population; fixed 1/5 weights per stone bin; no design-unbiased inference or inclusion-probability claim",
        "eligible_supply": supply,
        "preferred_four_x_supply": {
            str(stones): supply[stones] >= 4 * target
            for stones, target in STONE_TARGETS.items()
        },
        "counts_by_stone": dict(counts),
        "selected_inventory_sha256": selected_inventory_sha256,
        "minimum_openings": MIN_OPENING_CLUSTERS,
        "distinct_openings": len(openings),
        "max_per_opening": max(openings.values()),
        "max_per_game": max(games.values()),
        "candidate_inventory_sha256": sha256(canonical_bytes(eligible)),
        "states": selected,
    }


def generate_population(authorization: dict[str, Any], output: Path) -> dict[str, Any]:
    """Production entry point; cannot run absent explicit owner authorization."""
    if (
        authorization.get("authorized") is not True
        or not authorization.get("project_owner_identity")
        or not authorization.get("execution_freeze_sha256")
        or authorization.get("scope") != "model-independent-generation-and-screening"
    ):
        raise PermissionError("explicit_validated_generation_authorization_required")
    freeze_path = authorization.get("execution_freeze_path")
    if not freeze_path or not Path(freeze_path).is_file():
        raise PermissionError("execution_freeze_file_required")
    freeze_raw = Path(freeze_path).read_bytes()
    if sha256(freeze_raw) != authorization["execution_freeze_sha256"]:
        raise PermissionError("execution_freeze_hash_mismatch")
    freeze = json.loads(freeze_raw)
    expected_sources = freeze.get("source_file_sha256")
    expected_runtime = freeze.get("runtime")
    rules_hash = sha256((ROOT / "ml/alphazero_lite/kalah_rules.py").read_bytes())
    encoder_hash = sha256((ROOT / "ml/alphazero_lite/input_encodings.py").read_bytes())
    generation_contract = freeze.get("generation")
    selection_contract = freeze.get("selection")
    if (
        freeze.get("schema") != "seed624-execution-freeze-v1"
        or freeze.get("status") != "authorized"
        or freeze.get("project_owner_authorization_identity")
        != authorization["project_owner_identity"]
        or str(output) not in freeze.get("create_only_output_paths", [])
        or freeze.get("git_commit") != current_git_commit()
        or not isinstance(expected_sources, dict)
        or expected_sources != current_source_hashes()
        or expected_runtime != runtime_identity()
        or freeze.get("rules_identity")
        != {
            "rule_id": "historical-kalah-rules",
            "rule_version": "repository-kalah-rules-v1",
            "source_sha256": rules_hash,
        }
        or freeze.get("encoder_identity")
        != {"encoder": "kalah_v3", "source_sha256": encoder_hash}
        or generation_contract
        != {
            "master_seed": MASTER_SEED,
            "seed_contract": SEED_CONTRACT,
            "opening_reserve": OPENING_RESERVE,
            "opening_target": OPENING_TARGET,
            "prefix_actions": PREFIX_ACTIONS,
            "games_per_opening": GAMES_PER_OPENING,
            "continuation_action_limit": MAX_CONTINUATION_ACTIONS,
        }
        or selection_contract
        != {
            "targets": {str(key): value for key, value in STONE_TARGETS.items()},
            "max_per_game": 2,
            "max_per_opening": 4,
            "minimum_openings": MIN_OPENING_CLUSTERS,
            "objective": "first lexicographic feasible rank vector",
            "optimizer": "seed624-dfs-v1",
            "runtime_limit_seconds": 3600,
            "workers": 1,
        }
        or freeze.get("sampling_contract")
        != "deterministic standardized selected-state population; fixed 1/5 weights per stone bin; no design-unbiased inference or inclusion-probability claim"
        or freeze.get("output_schema") != SCHEMA
    ):
        raise PermissionError("execution_freeze_authorization_binding_mismatch")
    if output.exists():
        raise FileExistsError("production_artifact_path_must_be_create_only")
    evidence_path = authorization.get("membership_evidence_path")
    if not evidence_path or not Path(evidence_path).is_file():
        raise PermissionError("authenticated_membership_evidence_required")
    evidence = json.loads(Path(evidence_path).read_bytes())
    openings = choose_distinct_openings(
        lambda index: generate_opening(MASTER_SEED, index),
        reserve=OPENING_RESERVE,
        target=OPENING_TARGET,
    )
    games: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for opening in openings:
        for trajectory_index in range(GAMES_PER_OPENING):
            game, occurrences = generate_game(opening, MASTER_SEED, trajectory_index)
            game["opening_prefix_actions"] = opening["prefix_actions"]
            games.append(game)
            candidates.extend(occurrences)
    unique = deduplicate_candidates(candidates)
    screened = screen_known_exposure(
        unique,
        evidence,
        expected_snapshot_hashes=freeze.get("membership_snapshot_sha256"),
        expected_provenance_hash=freeze.get("source_provenance_sha256"),
    )
    selection = select_cohort(screened)
    bundle = {
        "schema": SCHEMA,
        "authorization": {
            "project_owner_identity": authorization["project_owner_identity"],
            "execution_freeze_sha256": authorization["execution_freeze_sha256"],
            "execution_freeze_raw_utf8": freeze_raw.decode("utf-8"),
        },
        "generation": {
            "master_seed": MASTER_SEED,
            "seed_contract": SEED_CONTRACT,
            "opening_seeds_consumed": list(
                range(max(int(item["opening_index"]) for item in openings) + 1)
            ),
        },
        "source_identity": {
            "git_commit": current_git_commit(),
            "runtime": runtime_identity(),
            "files": {
                name: {"sha256": digest, "source": (ROOT / name).read_text()}
                for name, digest in current_source_hashes().items()
            },
        },
        "opening_inventory_sha256": sha256(canonical_bytes(openings)),
        "game_inventory_sha256": sha256(canonical_bytes(games)),
        "candidate_inventory_sha256": sha256(canonical_bytes(screened)),
        "aggregate_counts": {
            "openings": len(openings),
            "games": len(games),
            "candidates": len(screened),
            "confirmed_known_exposure": sum(
                row["classification"] == "confirmed_known_exposure" for row in screened
            ),
            "known_exposure_nonmatch_inherited_PE4_unresolved": sum(
                row["classification"]
                == "known_exposure_nonmatch_inherited_PE4_unresolved"
                for row in screened
            ),
            "invalid_or_insufficiently_authenticated_evidence": sum(
                row["classification"]
                == "invalid_or_insufficiently_authenticated_evidence"
                for row in screened
            ),
        },
        "openings": openings,
        "games": games,
        "candidates": screened,
        "selection": selection,
        "evidence": evidence,
    }
    output.mkdir(parents=True)
    (output / "population.json").write_bytes(canonical_bytes(bundle) + b"\n")
    (output / "population.sha256").write_text(
        sha256((output / "population.json").read_bytes()) + "\n"
    )
    return {
        "schema": SCHEMA,
        "openings": len(openings),
        "games": len(games),
        "eligible_candidates": len(screened),
        "selected": len(selection["states"]),
        "output": str(output),
    }


def proposed_freeze_schema() -> dict[str, Any]:
    """Return a schema template, never a production-ready execution freeze."""
    return {
        "schema": "seed624-proposed-execution-freeze-v1",
        "status": "proposal_not_ready_or_approved",
        "required": {
            "git_commit": None,
            "source_file_sha256": {name: None for name in SOURCE_FILES},
            "rules_identity": {
                "rule_id": "historical-kalah-rules",
                "rule_version": "repository-kalah-rules-v1",
                "source_sha256": None,
            },
            "encoder_identity": {"encoder": "kalah_v3", "source_sha256": None},
            "runtime": runtime_identity(),
            "generation": {
                "master_seed": MASTER_SEED,
                "seed_contract": SEED_CONTRACT,
                "opening_reserve": OPENING_RESERVE,
                "opening_target": OPENING_TARGET,
                "prefix_actions": PREFIX_ACTIONS,
                "games_per_opening": GAMES_PER_OPENING,
                "continuation_action_limit": MAX_CONTINUATION_ACTIONS,
            },
            "membership_snapshot_sha256": {
                "B512": None,
                "S615": None,
                "protected_holdout": None,
            },
            "source_provenance_sha256": None,
            "selection": {
                "targets": STONE_TARGETS,
                "max_per_game": 2,
                "max_per_opening": 4,
                "minimum_openings": MIN_OPENING_CLUSTERS,
                "objective": "first lexicographic feasible rank vector",
                "optimizer": "seed624-dfs-v1",
                "runtime_limit_seconds": 3600,
                "workers": 1,
            },
            "sampling_contract": "deterministic standardized selected-state population; fixed 1/5 weights per stone bin; no design-unbiased inference or inclusion-probability claim",
            "output_schema": SCHEMA,
            "create_only_output_paths": [],
            "outputs": {"schema": SCHEMA},
            "verification_commands": [],
            "project_owner_authorization_identity": None,
        },
    }
