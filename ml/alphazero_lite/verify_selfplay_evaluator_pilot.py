"""Independent replay and accounting verifier for the evaluator pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import train
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import encode_state
from ml.alphazero_lite.validate_seed500_pilot import STONE_BUCKETS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_shard(path: Path, *, arm: str, expected_seed: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["game_index"])].append(row)
    completed = capped = incomplete = 0
    lengths: list[int] = []
    outcomes: Counter[str] = Counter()
    buckets: dict[str, dict[str, Any]] = {
        name: {
            "rows": 0,
            "canonical": set(),
            "encoded": set(),
            "targets": defaultdict(list),
            "entropy_sum": 0.0,
        }
        for name, _, _ in STONE_BUCKETS
    }
    identities: set[tuple[Any, ...]] = set()
    consumed_canonical: set[str] = set()
    consumed_encoded: set[str] = set()
    for game_index in range(64):
        game_rows = grouped.get(game_index, [])
        if not game_rows:
            incomplete += 1
            continue
        sequence = sorted(game_rows, key=lambda r: int(r["move_index"]))
        first = sequence[0]
        expected_moves = first.get("game_moves") or []
        if any(row.get("game_capped") is True for row in sequence):
            capped += 1
            continue
        game = KalahGame.from_state(first["game_start_state"])
        if int(first["game_seed"]) != expected_seed:
            raise ValueError(f"seed_association_mismatch:{arm}:{game_index}")
        if len(sequence) != len(expected_moves) or not all(
            row.get("game_completed") is True for row in sequence
        ):
            incomplete += 1
            continue
        if [int(row["move_index"]) for row in sequence] != list(range(len(sequence))):
            raise ValueError(f"move_index_mismatch:{arm}:{game_index}")
        lengths.append(len(sequence))
        for ply, (move, row) in enumerate(zip(expected_moves, sequence, strict=True)):
            state = game.to_state()
            if row.get("game_index") != game_index or row.get("game_seed") != first.get(
                "game_seed"
            ):
                raise ValueError(
                    f"row_game_association_mismatch:{arm}:{game_index}:{ply}"
                )
            if int(row["player"]) != game.current_player:
                raise ValueError(f"row_player_mismatch:{arm}:{game_index}:{ply}")
            active = sum(state["player_pits"] + state["opponent_pits"])
            encoded = np.asarray(row["state"], dtype=np.float32)
            expected_encoded = np.asarray(
                encode_state(state, input_encoding="kalah_v3"), dtype=np.float32
            )
            if not np.array_equal(encoded, expected_encoded):
                raise ValueError(f"trajectory_state_mismatch:{arm}:{game_index}:{ply}")
            legal = game.possible_moves()
            policy = np.asarray(row["policy"], dtype=np.float64)
            if row.get("legal_moves") != legal or int(move) not in legal:
                raise ValueError(f"illegal_move_or_legal_set:{arm}:{game_index}:{ply}")
            if (
                policy.shape != (6,)
                or not np.isfinite(policy).all()
                or np.any(policy < 0)
            ):
                raise ValueError(f"invalid_target:{arm}:{game_index}:{ply}")
            if any(
                policy[action] > 1e-8 for action in range(6) if action not in legal
            ) or not math.isclose(float(policy.sum()), 1.0, abs_tol=1e-5):
                raise ValueError(
                    f"illegal_or_unnormalized_target:{arm}:{game_index}:{ply}"
                )
            if int(row["top_target_move"]) != int(np.argmax(policy)):
                raise ValueError(f"top_target_move_mismatch:{arm}:{game_index}:{ply}")
            if (
                row.get("policy_target_noise_mode") != "denoised"
                or row.get("policy_target_mode") != "default"
                or row.get("value_target_mode") != "default"
            ):
                raise ValueError(f"target_generation_contract:{arm}:{game_index}:{ply}")
            noise_enabled = active > 16 and ply < 10
            expected_alpha = 0.3 if noise_enabled else 0.0
            expected_sampling_epsilon = 0.25 if noise_enabled else 0.0
            if (
                not math.isclose(
                    float(row["dirichlet_alpha"]), expected_alpha, abs_tol=1e-12
                )
                or not math.isclose(
                    float(row["dirichlet_epsilon_for_sampling"]),
                    expected_sampling_epsilon,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(row["dirichlet_epsilon_for_target"]), 0.0, abs_tol=1e-12
                )
                or row.get("action_sampling_noise_enabled") is not noise_enabled
            ):
                raise ValueError(f"noise_telemetry_contract:{arm}:{game_index}:{ply}")
            if active <= 16:
                if (
                    row.get("policy_target_actual_mode") != "exact_root_one_hot"
                    or int(row["simulations"]) != 0
                    or np.count_nonzero(policy > 1e-8) != 1
                ):
                    raise ValueError(
                        f"exact_root_target_contract:{arm}:{game_index}:{ply}"
                    )
            elif (
                row.get("policy_target_actual_mode") != "default"
                or int(row["simulations"]) != 128
            ):
                raise ValueError(f"puct_target_contract:{arm}:{game_index}:{ply}")
            profile = row["search_profile"]
            if (
                profile["simulations"] != 128
                or profile["c_puct"] != 1.25
                or profile["search_options"]
                != {
                    "fpu_mode": "zero",
                    "reuse_subtree": False,
                    "normalize_values": False,
                    "root_policy_mode": "visit_count",
                    "tactical_root_bias": 0.0,
                    "root_temperature": 0.0,
                }
            ):
                raise ValueError(f"search_telemetry_contract:{arm}:{game_index}:{ply}")
            if (
                profile.get("exact_root_solve_threshold") != 16
                or profile.get("exact_leaf_solve_mode") != "disabled"
            ):
                raise ValueError(f"solver_telemetry_contract:{arm}:{game_index}:{ply}")
            profile_body = {
                key: value for key, value in profile.items() if key != "hash"
            }
            profile_hash = hashlib.sha256(
                json.dumps(
                    profile_body,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode()
            ).hexdigest()
            if (
                profile_hash != profile.get("hash")
                or row.get("search_profile_hash") != profile_hash
            ):
                raise ValueError(
                    f"search_profile_hash_mismatch:{arm}:{game_index}:{ply}"
                )
            canonical_state = tuple(
                state["player_pits"]
                + state["opponent_pits"]
                + [
                    state["player_store"],
                    state["opponent_store"],
                    state["current_player"],
                ]
            )
            consumed_canonical.add(
                hashlib.sha256(
                    json.dumps(canonical_state, separators=(",", ":")).encode()
                ).hexdigest()
            )
            consumed_encoded.add(
                hashlib.sha256(
                    np.asarray(encoded, dtype=np.float32).tobytes()
                ).hexdigest()
            )
            identities.add(canonical_state)
            bucket_name = next(
                name for name, low, high in STONE_BUCKETS if low <= active <= high
            )
            bucket = buckets[bucket_name]
            bucket["rows"] += 1
            bucket["canonical"].add(canonical_state)
            bucket["encoded"].add(tuple(float(x) for x in encoded))
            bucket["targets"][canonical_state].append(tuple(float(x) for x in policy))
            bucket["entropy_sum"] += -sum(
                float(x) * math.log(float(x)) for x in policy if x > 0
            )
            if not game.move(game.pit_index(int(move))):
                raise ValueError(f"replay_move_rejected:{arm}:{game_index}:{ply}")
        if not game.over():
            raise ValueError(f"nonterminal_game:{arm}:{game_index}")
        winner = game.winner
        outcome = "draw" if winner is None else str(winner)
        outcomes[outcome] += 1
        for row in sequence:
            target = (
                0.0
                if winner is None
                else (1.0 if int(row["player"]) == winner else -1.0)
            )
            if (
                not math.isclose(float(row["value"]), target, abs_tol=1e-6)
                or row.get("winner") != winner
            ):
                raise ValueError(f"value_or_outcome_mismatch:{arm}:{game_index}")
        completed += 1
    if sorted(grouped) != list(range(max(grouped, default=-1) + 1)):
        raise ValueError(f"noncontiguous_game_ids:{arm}")
    loaded = train.load_jsonl(
        path, policy_target_mode="default", value_target_mode="default"
    )
    if int(loaded[0].shape[0]) != len(rows):
        raise ValueError(f"loader_row_count_mismatch:{arm}")
    report_buckets = {}
    for name, bucket in buckets.items():
        duplicate_groups = [
            group for group in bucket["targets"].values() if len(group) > 1
        ]
        disagreement_count = sum(
            len({target for target in group}) > 1 for group in duplicate_groups
        )
        report_buckets[name] = {
            "rows": bucket["rows"],
            "canonical_unique": len(bucket["canonical"]),
            "production_float32_unique": len(bucket["encoded"]),
            "mean_target_entropy_nats": bucket["entropy_sum"] / bucket["rows"]
            if bucket["rows"]
            else 0.0,
            "duplicate_target_disagreement_states": disagreement_count,
            "duplicate_target_groups": len(duplicate_groups),
        }
    return {
        "arm": arm,
        "sha256": digest(path),
        "rows": len(rows),
        "games": {"completed": completed, "capped": capped, "incomplete": incomplete},
        "game_lengths": {
            "per_completed_game": lengths,
            "min": min(lengths, default=0),
            "max": max(lengths, default=0),
            "mean": float(np.mean(lengths)) if lengths else 0.0,
        },
        "outcomes_by_winner": dict(outcomes),
        "seed_schedule": [
            {
                "game_index": index,
                "recorded_game_seed": expected_seed,
                "worker_rng_seed": (expected_seed * 1_000_003) + index,
            }
            for index in range(64)
        ],
        "canonical_unique": len(identities),
        "stone_buckets": report_buckets,
        "consumed_state_identities": {
            "canonical": sorted(consumed_canonical),
            "production_float32": sorted(consumed_encoded),
        },
        "loader": "accepted",
        "loader_shapes": [list(array.shape) for array in loaded],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    result = {
        arm: verify_shard(
            root / "docs/data/selfplay-evaluator-pilot" / f"{arm}.jsonl",
            arm=arm,
            expected_seed=542500,
        )
        for arm in ("control", "treatment")
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
