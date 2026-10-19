"""Read-only integrity and accounting report for the archived seed500 pilot."""

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


EXPECTED_SHA256 = "5d532603debd431e399b0ec32265a2c6b43064e5b1f7ecb1f0cfca32c7e266f9"
STONE_BUCKETS = ((">32", 33, 48), ("17–32", 17, 32), ("≤16", 0, 16))


def _canonical_state(state: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        state["player_pits"]
        + state["opponent_pits"]
        + [state["player_store"], state["opponent_store"], state["current_player"]]
    )


def _validate_and_account(rows: list[dict[str, Any]]) -> dict[str, Any]:
    games: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        games[int(row["game_index"])].append((index, row))
    if sorted(games) != list(range(len(games))):
        raise ValueError("game_ids_not_complete_zero_based")

    seen_canonical: Counter[tuple[Any, ...]] = Counter()
    seen_encoded: Counter[tuple[float, ...]] = Counter()
    subgroup_inputs: dict[tuple[str, str], Counter[tuple[float, ...]]] = defaultdict(
        Counter
    )
    groups: dict[str, dict[str, dict[str, float]]] = {
        mode: {
            name: {
                "rows": 0,
                "usable_rows": 0,
                "entropy_sum": 0.0,
                "one_hot_rows": 0,
                "duplicate_rows": 0,
            }
            for name, _, _ in STONE_BUCKETS
        }
        for mode in ("PUCT", "exact-root")
    }
    outcomes: Counter[str] = Counter()
    actions = 0
    extra_turns = 0
    for game_id, indexed_rows in sorted(games.items()):
        sequence = sorted(
            (r for _, r in indexed_rows), key=lambda r: int(r["move_index"])
        )
        first = sequence[0]
        game = KalahGame.from_state(first["game_start_state"])
        moves = first["game_moves"]
        if len(sequence) != len(moves) or not first.get("game_completed"):
            raise ValueError(f"incomplete_game:{game_id}")
        if any(
            int(row["game_length"]) != len(moves)
            or row.get("game_completed") is not True
            for row in sequence
        ):
            raise ValueError(f"game_completion_metadata_mismatch:{game_id}")
        if any(int(r["game_index"]) != game_id for r in sequence):
            raise ValueError(f"game_metadata_mismatch:{game_id}")
        for ply, (move, row) in enumerate(zip(moves, sequence, strict=True)):
            if int(row["move_index"]) != ply or game.over():
                raise ValueError(f"row_action_order:{game_id}:{ply}")
            state = game.to_state()
            encoded = np.asarray(row["state"], dtype=np.float32)
            if encoded.shape != (27,) or not np.allclose(
                encoded, encode_state(state, input_encoding="kalah_v3"), atol=1e-7
            ):
                raise ValueError(f"state_replay_mismatch:{game_id}:{ply}")
            legal = game.possible_moves()
            action = int(move)
            if action not in legal or list(row.get("legal_moves", [])) != legal:
                raise ValueError(f"illegal_action:{game_id}:{ply}")
            if int(row["player"]) != game.current_player:
                raise ValueError(f"player_mismatch:{game_id}:{ply}")
            policy = np.asarray(row["policy"], dtype=float)
            if (
                policy.shape != (6,)
                or not np.isfinite(policy).all()
                or np.any(policy < 0)
            ):
                raise ValueError(f"invalid_policy:{game_id}:{ply}")
            if any(
                policy[i] > 1e-8 for i in range(6) if i not in legal
            ) or not math.isclose(float(policy.sum()), 1, abs_tol=1e-5):
                raise ValueError(f"illegal_policy_mass:{game_id}:{ply}")
            if not game.move(game.pit_index(action)):
                raise ValueError(f"move_rejected:{game_id}:{ply}")
            extra_turns += int(
                game.current_player == int(row["player"]) and not game.over()
            )
            actions += 1
            active = sum(state["player_pits"] + state["opponent_pits"])
            bucket = next(
                name for name, low, high in STONE_BUCKETS if low <= active <= high
            )
            mode = "exact-root" if active <= 16 else "PUCT"
            target = groups[mode][bucket]
            target["rows"] += 1
            target["usable_rows"] += 1
            entropy = -sum(float(x) * math.log(float(x)) for x in policy if x > 0)
            target["entropy_sum"] += entropy
            target["one_hot_rows"] += int(np.count_nonzero(policy > 1e-8) == 1)
            key = _canonical_state(state)
            seen_canonical[key] += 1
            input_key = tuple(float(x) for x in encoded)
            seen_encoded[input_key] += 1
            subgroup_seen = subgroup_inputs[(mode, bucket)]
            if subgroup_seen[input_key]:
                target["duplicate_rows"] += 1
            subgroup_seen[input_key] += 1
        if not game.over():
            raise ValueError(f"game_not_terminal:{game_id}")
        outcome = "draw" if game.winner is None else str(game.winner)
        outcomes[outcome] += 1
        for _, row in indexed_rows:
            expected = (
                0.0
                if game.winner is None
                else (1.0 if int(row["player"]) == game.winner else -1.0)
            )
            if not math.isclose(float(row["value"]), expected, abs_tol=1e-6):
                raise ValueError(f"value_target_mismatch:{game_id}")
            if row.get("winner") != game.winner:
                raise ValueError(f"winner_mismatch:{game_id}")

    for mode_groups in groups.values():
        for bucket in mode_groups.values():
            bucket["duplicate_rows"] = int(bucket["duplicate_rows"])
            n = bucket["rows"]
            entropy_sum = bucket.pop("entropy_sum")
            one_hot_rows = bucket.pop("one_hot_rows")
            bucket["mean_legal_policy_entropy_nats"] = entropy_sum / n if n else 0.0
            bucket["one_hot_fraction"] = one_hot_rows / n if n else 0.0
            for key in ("usable_rows", "rows"):
                bucket[key] = int(bucket[key])
    return {
        "rows": len(rows),
        "games": len(games),
        "actions": actions,
        "outcomes_by_winner": dict(outcomes),
        "extra_turns": extra_turns,
        "canonical_unique_states": len(seen_canonical),
        "canonical_duplicate_rows": sum(n - 1 for n in seen_canonical.values()),
        "encoded_input_unique_states": len(seen_encoded),
        "encoded_input_duplicate_rows": sum(n - 1 for n in seen_encoded.values()),
        "subgroups_by_teacher_and_active_stones": groups,
        "generation_duration": "unavailable (no original timing record)",
    }


def validate(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.name == "seed500-selfplay-64.jsonl" and digest != EXPECTED_SHA256:
        raise ValueError(f"archived_corpus_sha256_mismatch:{digest}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    accounting = _validate_and_account(rows)
    x, p, v = train.load_jsonl(
        path, policy_target_mode="default", value_target_mode="default"
    )
    accounting["production_loader"] = "accepted"
    accounting["loader_shapes"] = {
        "states": list(x.shape),
        "policies": list(p.shape),
        "values": list(v.shape),
    }
    accounting["corpus_sha256"] = digest
    return accounting


def validate_shards(paths: list[Path]) -> dict[str, Any]:
    """Replay and account for separate shards as one corpus without changing bytes."""
    rows: list[dict[str, Any]] = []
    shard_reports = []
    digest = hashlib.sha256()
    seen_encoded: set[tuple[float, ...]] = set()
    diversity_growth = []
    for shard_index, path in enumerate(paths):
        report = validate(path)
        if report["games"] != 64:
            raise ValueError(f"shard_game_count_mismatch:{path}:{report['games']}")
        shard_reports.append({"path": str(path), **report})
        shard_rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        for row in shard_rows:
            row["game_index"] = int(row["game_index"]) + shard_index * 64
        rows.extend(shard_rows)
        shard_encoded = {tuple(float(x) for x in row["state"]) for row in shard_rows}
        prior_unique = len(seen_encoded)
        seen_encoded.update(shard_encoded)
        diversity_growth.append(
            {
                "seed": int(shard_rows[0]["game_seed"]),
                "shard_unique_inputs": len(shard_encoded),
                "cumulative_unique_inputs": len(seen_encoded),
                "new_unique_inputs": len(seen_encoded) - prior_unique,
            }
        )
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    accounting = _validate_and_account(rows)
    loaded = [
        train.load_jsonl(
            path, policy_target_mode="default", value_target_mode="default"
        )
        for path in paths
    ]
    accounting["production_loader"] = "accepted"
    accounting["loader_shapes"] = {
        "states": [sum(int(x.shape[0]) for x, _, _ in loaded), 27],
        "policies": [sum(int(p.shape[0]) for _, p, _ in loaded), 6],
        "values": [sum(int(v.shape[0]) for _, _, v in loaded), 1],
    }
    accounting["corpus_sha256"] = digest.hexdigest()
    accounting["shards"] = shard_reports
    accounting["loader_accepted_rows"] = accounting["rows"]
    accounting["game_status"] = {
        "complete": accounting["games"],
        "capped": 0,
        "incomplete": 0,
    }
    accounting["diversity_growth_by_shard"] = diversity_growth
    return accounting


def partition_overlap(paths: list[Path], training_seeds: set[int]) -> dict[str, Any]:
    """Compare development identities to the designated training-source shards."""
    training_canonical: set[tuple[Any, ...]] = set()
    training_encoded: set[tuple[float, ...]] = set()
    development: list[tuple[tuple[Any, ...], tuple[float, ...]]] = []
    for path in paths:
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        games: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            games[int(row["game_index"])].append(row)
        for game_rows in games.values():
            sequence = sorted(game_rows, key=lambda row: int(row["move_index"]))
            game = KalahGame.from_state(sequence[0]["game_start_state"])
            moves = sequence[0]["game_moves"]
            for row in sequence:
                canonical = _canonical_state(game.to_state())
                encoded = tuple(float(x) for x in row["state"])
                if int(row["game_seed"]) in training_seeds:
                    training_canonical.add(canonical)
                    training_encoded.add(encoded)
                else:
                    development.append((canonical, encoded))
                game.move(game.pit_index(int(moves[int(row["move_index"])])))
    filtered = [
        (canonical, encoded)
        for canonical, encoded in development
        if canonical not in training_canonical and encoded not in training_encoded
    ]
    return {
        "training_source_rows": sum(
            1
            for path in paths
            for line in path.read_text(encoding="utf-8").splitlines()
            if int(json.loads(line)["game_seed"]) in training_seeds
        ),
        "development_original": {
            "rows": len(development),
            "canonical_unique": len({canonical for canonical, _ in development}),
            "encoded_input_unique": len({encoded for _, encoded in development}),
        },
        "development_after_training_identity_exclusion": {
            "rows": len(filtered),
            "canonical_unique": len({canonical for canonical, _ in filtered}),
            "encoded_input_unique": len({encoded for _, encoded in filtered}),
        },
        "development_rows_excluded_by_training_canonical_or_exact_input": len(
            development
        )
        - len(filtered),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, nargs="+")
    args = parser.parse_args()
    report = (
        validate(args.corpus[0])
        if len(args.corpus) == 1
        else validate_shards(args.corpus)
    )
    if len(args.corpus) > 1:
        report["training_development_identity_overlap"] = partition_overlap(
            args.corpus, set(range(503000, 503006))
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
