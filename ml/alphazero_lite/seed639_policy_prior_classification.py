"""Descriptive B512/S615 policy comparison on the frozen Seed 636 cohort.

The functions in this module are deterministic, side-effect-free analysis
primitives. They do not regenerate states, search, solve, or assess playing
strength.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENT_ID = "seed639-prospective-policy-prior-classification"
POPULATION_SIZE = 512
SELECTED_INVENTORY_SHA256 = (
    "6c6b1f680b2177cb67c79a9a642216703f160131034fbf8fe86e8130910ed077"
)
SUBSET_SIZE = 64
MINIMUM_SUBSET_GAMES = 32
MAX_PER_GAME = 1
MAX_PER_OPENING = 4
STONE_BIN_WEIGHTS = {17: 0.2, 18: 0.2, 19: 0.2, 20: 0.2, 21: 0.2}
STONE_BIN_COUNTS = {17: 103, 18: 103, 19: 102, 20: 102, 21: 102}


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    """Serialize JSON values canonically for stable input identities."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def authenticate_inputs(
    *,
    seed615_root: Path,
    selected_manifest_path: Path,
    selected_verification_path: Path,
    archive_path: Path,
    archive_manifest_path: Path,
    population_member_path: Path,
) -> dict[str, Any]:
    """Authenticate the fixed cohort bundle and historical model evaluation."""
    from ml.alphazero_lite import experiment_lifecycle
    from ml.alphazero_lite import seed615_policy_divergence_diagnostic as seed615

    expected_hashes = {
        "selected_manifest": "d7e095f749c14a429162be91dd3bb5f67b685fab04808658f7f26ac7882c8fab",
        "selected_verification": "4c4c177d2fd4bf8b4e0b37e65584b9c6a62ca1949c6c114b6bf891fefd15c09a",
        "archive": "d6e454f889e6155d7dccf3bf81e6769989896389d401fc5bf4cbd058059c8bf1",
        "archive_manifest": "057358aad2d72b733a29019c21f7a3f0a9b7d68e37cd5ea2954660e59c366bc5",
        "population_member": "689269182824d86d55261b8c89141bbc578867339cc1adce695f57da2c7fc6fc",
    }
    actual_paths = {
        "selected_manifest": selected_manifest_path,
        "selected_verification": selected_verification_path,
        "archive": archive_path,
        "archive_manifest": archive_manifest_path,
        "population_member": population_member_path,
    }
    actual_hashes = {
        name: sha256_bytes(path.read_bytes()) for name, path in actual_paths.items()
    }
    if actual_hashes != expected_hashes:
        raise ValueError("seed636_registered_input_hash_mismatch")

    selected_manifest = json.loads(selected_manifest_path.read_bytes())
    selected_verification = json.loads(selected_verification_path.read_bytes())
    if (
        selected_manifest.get("schema") != "seed636-selected-state-manifest-v1"
        or selected_manifest.get("selected_count") != POPULATION_SIZE
        or selected_manifest.get("selected_inventory_sha256")
        != SELECTED_INVENTORY_SHA256
        or selected_manifest.get("exposure_classification")
        != "known-exposure-screened; inherited P-E4 exposure unresolved"
    ):
        raise ValueError("seed636_selected_manifest_contract_mismatch")
    if (
        selected_verification.get("status") != "verified"
        or selected_verification.get("selected_count") != POPULATION_SIZE
        or selected_verification.get("selected_inventory_sha256")
        != SELECTED_INVENTORY_SHA256
        or selected_verification.get("manifest_sha256")
        != expected_hashes["selected_manifest"]
        or selected_verification.get("population_member_sha256")
        != expected_hashes["population_member"]
    ):
        raise ValueError("seed636_selected_manifest_verification_mismatch")

    archive_verification = experiment_lifecycle.verify_archive(
        archive_path, archive_manifest_path
    )
    if (
        archive_verification.get("experiment_id") != "seed636-prospective-population"
        or archive_verification.get("attempt_id") != "attempt-0001"
    ):
        raise ValueError("seed636_archive_identity_mismatch")
    members = {member["path"]: member for member in archive_verification["files"]}
    population_member = members.get("population/population.json")
    if (
        population_member is None
        or population_member["sha256"] != expected_hashes["population_member"]
        or population_member["size_bytes"] != population_member_path.stat().st_size
    ):
        raise ValueError("seed636_population_member_archive_binding_mismatch")
    population = json.loads(population_member_path.read_bytes())
    selected_rows = population.get("selection", {}).get("states")
    if (
        not isinstance(population, dict)
        or not isinstance(selected_rows, list)
        or len(selected_rows) != POPULATION_SIZE
        or sha256_bytes(canonical_json(selected_rows)) != SELECTED_INVENTORY_SHA256
    ):
        raise ValueError("seed636_population_inventory_mismatch")

    seed615_authentication = seed615.authenticate(seed615_root)
    if (
        seed615_authentication.get("freeze_sha256")
        != "ba30f5e3efadf3358db9f03c7b4eba4f69b23e2f95dcb4089723a1fb9664f32b"
        or seed615_authentication.get("checkpoint_sha256")
        != {
            "B512": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
            "S615": "cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780",
            "P-E4": "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
        }
    ):
        raise ValueError("seed615_historical_authentication_mismatch")
    return {
        "selected_inventory_sha256": SELECTED_INVENTORY_SHA256,
        "selected_count": POPULATION_SIZE,
        "selected_population_rows": selected_rows,
        "input_sha256": actual_hashes,
        "seed636_archive_verification": archive_verification,
        "seed615_authentication": seed615_authentication,
        "exposure_classification": selected_manifest["exposure_classification"],
    }


def compare_legal_priors(
    policy_b512: np.ndarray | list[float],
    policy_s615: np.ndarray | list[float],
    legal_actions: list[int],
) -> dict[str, Any]:
    """Validate, mask, and compare full six-action policy outputs.

    Input vectors must be finite nonnegative probability distributions. Their
    legal components are renormalized after masking; illegal output probability
    is not retained in either returned legal distribution. Ties use NumPy's
    first-index argmax rule, matching the existing prior tie convention.
    """
    if not legal_actions or len(set(legal_actions)) != len(legal_actions):
        raise ValueError("invalid_legal_action_set")
    if any(
        type(action) is not int or action < 0 or action >= 6 for action in legal_actions
    ):
        raise ValueError("invalid_legal_action_set")

    vectors = []
    for label, raw in (("B512", policy_b512), ("S615", policy_s615)):
        vector = np.asarray(raw, dtype=np.float64)
        if (
            vector.shape != (6,)
            or not np.all(np.isfinite(vector))
            or np.any(vector < 0)
            or not math.isclose(float(vector.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-7)
        ):
            raise ValueError(f"invalid_raw_probability_vector:{label}")
        legal = np.zeros(6, dtype=bool)
        legal[legal_actions] = True
        masked = np.where(legal, vector, 0.0)
        mass = float(masked.sum())
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError(f"zero_legal_probability_mass:{label}")
        masked /= mass
        if np.any(masked[~legal] != 0) or not math.isclose(
            float(masked.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"invalid_masked_probability_vector:{label}")
        vectors.append(masked)

    b512, s615 = vectors
    b_top, s_top = int(np.argmax(b512)), int(np.argmax(s615))
    return {
        "legal_actions": list(legal_actions),
        "B512_legal_probabilities": b512.tolist(),
        "S615_legal_probabilities": s615.tolist(),
        "B512_top_action": b_top,
        "S615_top_action": s_top,
        "top_action_disagreement": b_top != s_top,
        "total_variation": float(0.5 * np.abs(b512 - s615).sum()),
        "B512_entropy_bits": _entropy_bits(b512),
        "S615_entropy_bits": _entropy_bits(s615),
        "entropy_difference_S615_minus_B512_bits": _entropy_bits(s615)
        - _entropy_bits(b512),
    }


def evaluate_population(
    population_rows: list[dict[str, Any]], evaluator_b512: Any, evaluator_s615: Any
) -> list[dict[str, Any]]:
    """Evaluate exactly the frozen 512 rows and preserve source provenance."""
    from ml.alphazero_lite.kalah_rules import KalahGame
    from ml.alphazero_lite.self_play import encode_state

    if len(population_rows) != POPULATION_SIZE:
        raise ValueError("population_must_contain_exactly_512_rows")
    records = []
    canonical_ids: set[str] = set()
    float32_ids: set[str] = set()
    for index, source in enumerate(population_rows):
        state = source.get("state")
        if not isinstance(state, dict):
            raise ValueError(f"population_state_missing:{index}")
        canonical_id = sha256_bytes(canonical_json(state))
        encoded = np.asarray(
            encode_state(state, input_encoding="kalah_v3"), dtype="<f4"
        )
        float32_id = sha256_bytes(encoded.tobytes())
        if canonical_id != source.get("canonical_identity"):
            raise ValueError(f"canonical_identity_mismatch:{index}")
        if float32_id != source.get("production_float32_identity"):
            raise ValueError(f"production_float32_identity_mismatch:{index}")
        if canonical_id in canonical_ids or float32_id in float32_ids:
            raise ValueError(f"duplicate_population_identity:{index}")
        canonical_ids.add(canonical_id)
        float32_ids.add(float32_id)

        game = KalahGame.from_state(state)
        legal = [int(action) for action in game.possible_moves()]
        if len(legal) < 2:
            raise ValueError(f"insufficient_legal_actions:{index}")
        source_legal = source.get("legal_actions")
        if not isinstance(source_legal, list) or sorted(source_legal) != sorted(legal):
            raise ValueError(f"legal_action_set_mismatch:{index}")
        policy_b512, _ = evaluator_b512.evaluate(game)
        policy_s615, _ = evaluator_s615.evaluate(game)
        comparison = compare_legal_priors(policy_b512, policy_s615, legal)
        stones = int(sum(game.pits))
        if stones != source.get("active_pit_stones"):
            raise ValueError(f"active_pit_stone_count_mismatch:{index}")
        record = {
            "population_index": index,
            "state": state,
            "canonical_identity": canonical_id,
            "production_float32_identity": float32_id,
            "active_pit_stones": stones,
            "game_id": source.get("game_id"),
            "opening_identity": source.get("opening_identity"),
            "opening_index": source.get("opening_index"),
            "continuation_ply": source.get("continuation_ply"),
            "side_to_move": int(game.current_player),
            **comparison,
        }
        if not record["game_id"] or not record["opening_identity"]:
            raise ValueError(f"population_provenance_missing:{index}")
        records.append(record)
    return records


def _entropy_bits(vector: np.ndarray) -> float:
    positive = vector[vector > 0]
    return float(-np.sum(positive * np.log2(positive)))


def select_diagnostic_subset(
    rows: list[dict[str, Any]],
    *,
    subset_size: int = SUBSET_SIZE,
    minimum_games: int = MINIMUM_SUBSET_GAMES,
) -> dict[str, Any]:
    """Select the Seed 623 high-TV subset without modifying the input cohort.

    TV is sorted descending. Exact TV ties are resolved by SHA-256 of the
    frozen canonical state identity, then by that identity as a collision
    fallback. Greedy traversal applies the registered one-per-game and
    four-per-opening caps.
    """
    if subset_size <= 0 or minimum_games <= 0:
        raise ValueError("invalid_subset_contract")
    identities: set[str] = set()
    ranked: list[tuple[float, str, str, dict[str, Any]]] = []
    for row in rows:
        state_id = row.get("canonical_identity")
        game_id = row.get("game_id")
        opening_id = row.get("opening_identity")
        tv = row.get("total_variation")
        valid_tv = type(tv) in (int, float)
        tv_value = float(tv) if isinstance(tv, (int, float)) else float("nan")
        if (
            not isinstance(state_id, str)
            or not isinstance(game_id, str)
            or not isinstance(opening_id, str)
            or not valid_tv
            or not math.isfinite(tv_value)
            or not 0.0 <= tv_value <= 1.0
        ):
            raise ValueError("invalid_subset_row")
        if state_id in identities:
            raise ValueError("duplicate_state_identity")
        identities.add(state_id)
        tie_digest = sha256_bytes(state_id.encode("utf-8"))
        ranked.append((-tv_value, tie_digest, state_id, row))

    selected: list[dict[str, Any]] = []
    games: set[str] = set()
    opening_counts: Counter[str] = Counter()
    for _, _, _, row in sorted(ranked):
        game_id = row["game_id"]
        opening_id = row["opening_identity"]
        if game_id in games or opening_counts[opening_id] >= MAX_PER_OPENING:
            continue
        selected.append(row)
        games.add(game_id)
        opening_counts[opening_id] += 1
        if len(selected) == subset_size:
            break
    available = len(selected) == subset_size and len(games) >= minimum_games
    return {
        "available": available,
        "unavailable_reason": None
        if available
        else "fewer_than_64_selectable_states_or_32_distinct_games",
        "selected_rows": selected if available else [],
        "selected_state_ids": [row["canonical_identity"] for row in selected]
        if available
        else [],
        "selected_count": len(selected) if available else 0,
        "distinct_continuation_games": len(games) if available else 0,
        "distinct_opening_clusters": len(opening_counts) if available else 0,
        "states_per_opening_cluster": dict(sorted(opening_counts.items()))
        if available
        else {},
        "selection_contract": {
            "sort": "descending_total_variation",
            "tie_break": "sha256(utf8(canonical_identity)); canonical_identity collision fallback",
            "max_per_continuation_game": MAX_PER_GAME,
            "max_per_opening_cluster": MAX_PER_OPENING,
            "subset_size": subset_size,
            "minimum_distinct_continuation_games": minimum_games,
        },
    }


def summarize_population(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute descriptive complete-population and fixed-bin summaries."""
    if len(rows) != POPULATION_SIZE:
        raise ValueError("population_must_contain_exactly_512_rows")
    bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stones = row.get("active_pit_stones")
        if type(stones) is not int or stones not in STONE_BIN_WEIGHTS:
            raise ValueError("invalid_stone_bin")
        bins[stones].append(row)
    if any(not bins[stones] for stones in STONE_BIN_WEIGHTS):
        raise ValueError("missing_stone_bin")
    if {stones: len(bins[stones]) for stones in STONE_BIN_COUNTS} != STONE_BIN_COUNTS:
        raise ValueError("primary_population_stone_bin_quota_mismatch")

    tv = np.asarray([float(row["total_variation"]) for row in rows])
    disagreement = np.asarray(
        [bool(row["top_action_disagreement"]) for row in rows], dtype=np.float64
    )
    entropy_delta = np.asarray(
        [float(row["entropy_difference_S615_minus_B512_bits"]) for row in rows]
    )
    entropy_b512 = np.asarray([float(row["B512_entropy_bits"]) for row in rows])
    entropy_s615 = np.asarray([float(row["S615_entropy_bits"]) for row in rows])

    def metrics(group: list[dict[str, Any]]) -> dict[str, float | int]:
        group_tv = np.asarray([float(row["total_variation"]) for row in group])
        group_disagreement = np.asarray(
            [bool(row["top_action_disagreement"]) for row in group], dtype=float
        )
        group_entropy = np.asarray(
            [float(row["entropy_difference_S615_minus_B512_bits"]) for row in group]
        )
        group_entropy_b512 = np.asarray(
            [float(row["B512_entropy_bits"]) for row in group]
        )
        group_entropy_s615 = np.asarray(
            [float(row["S615_entropy_bits"]) for row in group]
        )
        return {
            "states": len(group),
            "top_action_disagreements": int(group_disagreement.sum()),
            "top_action_disagreement_rate": float(group_disagreement.mean()),
            "mean_total_variation": float(group_tv.mean()),
            "mean_B512_entropy_bits": float(group_entropy_b512.mean()),
            "mean_S615_entropy_bits": float(group_entropy_s615.mean()),
            "mean_entropy_difference_S615_minus_B512_bits": float(group_entropy.mean()),
        }

    quantiles = {
        str(q): float(np.quantile(tv, q))
        for q in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)
    }
    by_bin = {str(stones): metrics(group) for stones, group in sorted(bins.items())}
    standardized = {
        metric: float(
            sum(
                STONE_BIN_WEIGHTS[stones] * float(by_bin[str(stones)][metric])
                for stones in STONE_BIN_WEIGHTS
            )
        )
        for metric in (
            "top_action_disagreement_rate",
            "mean_total_variation",
            "mean_entropy_difference_S615_minus_B512_bits",
        )
    }
    return {
        "population_size": len(rows),
        "overall": {
            "top_action_disagreements": int(disagreement.sum()),
            "top_action_disagreement_rate": float(disagreement.mean()),
            "total_variation_mean": float(tv.mean()),
            "total_variation_quantiles": quantiles,
            "B512_mean_entropy_bits": float(entropy_b512.mean()),
            "S615_mean_entropy_bits": float(entropy_s615.mean()),
            "entropy_difference_S615_minus_B512_mean_bits": float(entropy_delta.mean()),
        },
        "by_stone_bin": by_bin,
        "fixed_one_fifth_bin_standardized": standardized,
    }


def summarize_subset_relationship(
    population_rows: list[dict[str, Any]], subset: dict[str, Any]
) -> dict[str, Any]:
    """Describe the enriched subset's coverage and differences from its parent."""
    if len(population_rows) != POPULATION_SIZE:
        raise ValueError("population_must_contain_exactly_512_rows")
    if not subset.get("available"):
        return {
            "available": False,
            "reason": subset.get("unavailable_reason"),
            "primary_population_size": len(population_rows),
            "subset_size": 0,
        }
    selected_ids = subset.get("selected_state_ids")
    if not isinstance(selected_ids, list) or len(selected_ids) != SUBSET_SIZE:
        raise ValueError("diagnostic_subset_identity_invalid")
    primary_by_id = {row.get("canonical_identity"): row for row in population_rows}
    if len(primary_by_id) != POPULATION_SIZE or any(
        state_id not in primary_by_id for state_id in selected_ids
    ):
        raise ValueError("diagnostic_subset_not_contained_in_primary_population")
    selected = [primary_by_id[state_id] for state_id in selected_ids]

    def describe(group: list[dict[str, Any]]) -> dict[str, Any]:
        by_bin: Counter[int] = Counter(int(row["active_pit_stones"]) for row in group)
        return {
            "states": len(group),
            "stone_bin_counts": {str(k): by_bin.get(k, 0) for k in STONE_BIN_WEIGHTS},
            "distinct_continuation_games": len({row["game_id"] for row in group}),
            "distinct_opening_clusters": len(
                {row["opening_identity"] for row in group}
            ),
            "top_action_disagreement_rate": float(
                np.mean([bool(row["top_action_disagreement"]) for row in group])
            ),
            "mean_total_variation": float(
                np.mean([float(row["total_variation"]) for row in group])
            ),
            "mean_entropy_difference_S615_minus_B512_bits": float(
                np.mean(
                    [
                        float(row["entropy_difference_S615_minus_B512_bits"])
                        for row in group
                    ]
                )
            ),
        }

    return {
        "available": True,
        "primary_population_size": len(population_rows),
        "subset_size": len(selected),
        "subset_is_unweighted_descriptive_only": True,
        "primary_population": describe(population_rows),
        "diagnostic_subset": describe(selected),
    }
