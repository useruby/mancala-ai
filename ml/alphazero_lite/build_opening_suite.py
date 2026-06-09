#!/usr/bin/env python3
"""Build deduplicated opening-prefix suites for reproducible seat-aware benchmarking.

Enumerates all legal opening move sequences (plies 2, 4, 6) from the initial
board, deduplicates by resulting board state, and stratifies diverse subsets
into small / medium / large suites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    from ml.alphazero_lite.kalah_rules import KalahGame, PITS_PER_PLAYER
except ModuleNotFoundError:
    from kalah_rules import KalahGame, PITS_PER_PLAYER

INITIAL_STATE = {
    "player_pits": [4, 4, 4, 4, 4, 4],
    "opponent_pits": [4, 4, 4, 4, 4, 4],
    "player_store": 0,
    "opponent_store": 0,
    "current_player": 0,
}


def canonical_payload(state: dict) -> dict:
    return {
        "current_player": int(state["current_player"]),
        "player_pits": [int(v) for v in state["player_pits"]],
        "opponent_pits": [int(v) for v in state["opponent_pits"]],
        "player_store": int(state["player_store"]),
        "opponent_store": int(state["opponent_store"]),
    }


def canonical_key(state: dict) -> str:
    encoded = json.dumps(
        canonical_payload(state), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def enumerate_legal_prefixes(max_ply: int) -> list[dict]:
    initial_game = KalahGame.from_state(INITIAL_STATE)
    results: list[dict] = []

    def dfs(game: KalahGame, moves: list[int], ply: int) -> None:
        if ply >= max_ply:
            return
        if game.over():
            return
        legal = game.possible_moves()
        if not legal:
            return
        for move in legal:
            child = game.clone()
            abs_idx = child.pit_index(move)
            if not child.move(abs_idx):
                continue
            child_moves = moves + [abs_idx]
            child_state = child.to_state()
            side_to_move = int(child_state["current_player"])
            stores = [child_state["player_store"], child_state["opponent_store"]]
            player_pits = child_state["player_pits"]
            opponent_pits = child_state["opponent_pits"]

            store_diff = (
                stores[0] - stores[1] if side_to_move == 0 else stores[1] - stores[0]
            )
            pit_sum = sum(player_pits) + sum(opponent_pits)

            first_move_family = str(child_moves[0]) if child_moves else "none"

            results.append(
                {
                    "ply": ply + 1,
                    "prefix_moves": [int(m) for m in child_moves],
                    "state": child_state,
                    "side_to_move": side_to_move,
                    "store_diff": store_diff,
                    "pit_sum": pit_sum,
                    "capture_available": False,
                    "extra_turn_available": False,
                    "first_move_family": first_move_family,
                }
            )
            dfs(child, child_moves, ply + 1)

    dfs(initial_game, [], 0)
    return results


def compute_capture_extra_turn(state: dict) -> tuple[bool, bool]:
    from ml.alphazero_lite.kalah_rules import move_consequence_for_state

    capture = False
    extra = False
    for m in range(PITS_PER_PLAYER):
        cons = move_consequence_for_state(state, m)
        if not cons.get("legal"):
            continue
        if cons.get("gives_extra_turn"):
            extra = True
        if cons.get("produces_capture"):
            capture = True
    return capture, extra


def compute_phase_bucket(pit_sum: int) -> str:
    if pit_sum <= 12:
        return "late"
    if pit_sum <= 24:
        return "mid"
    return "early"


def deduplicate_openings(
    prefixes: list[dict],
) -> tuple[list[dict], list[dict], int]:
    by_board: dict[str, list[dict]] = defaultdict(list)
    for entry in prefixes:
        key = canonical_key(entry["state"])
        by_board[key].append(entry)

    unique: list[dict] = []
    duplicates: list[dict] = []
    duplicate_count = 0

    for key, entries in by_board.items():
        canonical = dict(entries[0])
        if len(entries) > 1:
            alternate = entries[1:]
            canonical["alternate_prefixes"] = [
                [int(m) for m in e["prefix_moves"]] for e in alternate
            ]
            duplicate_count += len(alternate)
            duplicates.extend(alternate)
        unique.append(canonical)

    return unique, duplicates, duplicate_count


def stratify_openings(unique: list[dict]) -> list[dict]:
    for entry in unique:
        state = entry["state"]
        cap, extra = compute_capture_extra_turn(state)
        entry["capture_available"] = cap
        entry["extra_turn_available"] = extra
        entry["phase_bucket"] = compute_phase_bucket(entry["pit_sum"])
    return unique


def bucket_label(entry: dict) -> str:
    ply = entry["ply"]
    side = entry["side_to_move"]
    store_bucket = 0
    sd = entry["store_diff"]
    if sd < 0:
        store_bucket = -1
    elif sd > 0:
        store_bucket = 1
    cap = 1 if entry["capture_available"] else 0
    extra = 1 if entry["extra_turn_available"] else 0
    first = entry.get("first_move_family", "none")
    return f"ply{ply}_side{side}_store{store_bucket}_cap{cap}_ext{extra}_first{first}"


def select_diverse(
    unique: list[dict],
    target_size: int,
    rng_seed: int,
) -> list[dict]:
    import random

    rng = random.Random(rng_seed)

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for entry in unique:
        by_bucket[bucket_label(entry)].append(entry)

    bucket_names = sorted(by_bucket.keys())
    bucket_quotas: dict[str, int] = {}
    remaining = target_size
    for name in bucket_names:
        quota = max(1, target_size // len(bucket_names))
        bucket_quotas[name] = quota
        remaining -= quota

    for name in bucket_names:
        bucket_quotas[name] = min(bucket_quotas[name], len(by_bucket[name]))
    while remaining > 0:
        for name in bucket_names:
            if remaining <= 0:
                break
            current = bucket_quotas[name]
            if current < len(by_bucket[name]):
                bucket_quotas[name] += 1
                remaining -= 1

    selected: list[dict] = []
    for name in bucket_names:
        pool = list(by_bucket[name])
        rng.shuffle(pool)
        quota = bucket_quotas[name]
        selected.extend(pool[:quota])

    if len(selected) > target_size:
        rng.shuffle(selected)
        selected = selected[:target_size]

    rng.shuffle(selected)
    return selected


def build_suite(
    unique: list[dict],
    sizes: list[int],
    rng_seed: int,
) -> dict[int, list[dict]]:
    suites: dict[int, list[dict]] = {}
    for size in sorted(sizes):
        if size >= len(unique):
            suites[size] = list(unique)
        else:
            suites[size] = select_diverse(unique, size, rng_seed)
    return suites


def write_suite_jsonl(entries: list[dict], path: str) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            record = {
                "ply": entry["ply"],
                "prefix_moves": entry["prefix_moves"],
                "state": entry["state"],
                "side_to_move": entry["side_to_move"],
                "store_diff": entry["store_diff"],
                "pit_sum": entry["pit_sum"],
                "capture_available": entry["capture_available"],
                "extra_turn_available": entry["extra_turn_available"],
                "first_move_family": entry.get("first_move_family", "none"),
            }
            if entry.get("alternate_prefixes"):
                record["alternate_prefixes"] = entry["alternate_prefixes"]
            f.write(json.dumps(record) + "\n")
    return str(out_path)


def load_suite_jsonl(path: str) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def suite_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deduplicated opening-prefix benchmark suites."
    )
    parser.add_argument(
        "--out-dir", required=True, help="Output directory for suite JSONL files."
    )
    parser.add_argument(
        "--opening-plies",
        default="2,4,6",
        help="Comma-separated ply counts to enumerate.",
    )
    parser.add_argument(
        "--suite-sizes",
        default="32,128,384",
        help="Comma-separated target suite sizes.",
    )
    parser.add_argument(
        "--seed", type=int, default=49, help="RNG seed for subset selection."
    )
    return parser.parse_args()


def size_label(size: int) -> str:
    if size <= 32:
        return "small_smoke"
    if size <= 128:
        return "medium_eval"
    return "large_eval"


def main() -> int:
    args = parse_args()
    ply_counts = [int(x.strip()) for x in args.opening_plies.split(",")]
    suite_sizes = [int(x.strip()) for x in args.suite_sizes.split(",")]

    print(f"Enumerating legal opening prefixes for plies: {ply_counts}")
    all_prefixes: list[dict] = []
    for max_ply in ply_counts:
        prefixes = enumerate_legal_prefixes(max_ply)
        print(f"  ply<={max_ply}: {len(prefixes)} legal prefixes enumerated")
        all_prefixes.extend(prefixes)

    total_enumerated = len(all_prefixes)
    print(f"Total legal prefixes enumerated: {total_enumerated}")

    unique, duplicates, duplicate_count = deduplicate_openings(all_prefixes)
    unique = stratify_openings(unique)

    print(f"Unique resulting boards: {len(unique)}")
    print(f"Duplicate prefix count: {duplicate_count}")

    ply_dist: Counter = Counter()
    store_buckets: Counter = Counter()
    for entry in unique:
        ply_dist[entry["ply"]] += 1
        sd = entry["store_diff"]
        bucket = "negative" if sd < 0 else ("positive" if sd > 0 else "zero")
        store_buckets[bucket] += 1

    print(f"Unique openings by ply: {dict(sorted(ply_dist.items()))}")
    print(f"Unique openings by store-diff bucket: {dict(store_buckets)}")

    suites = build_suite(unique, suite_sizes, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suite_paths: dict[int, str] = {}
    for size, entries in suites.items():
        label = size_label(size)
        path = str(out_dir / f"{label}.jsonl")
        write_suite_jsonl(entries, path)
        sha = suite_sha256(path)
        suite_paths[size] = path

        ply_sub: Counter = Counter()
        for e in entries:
            ply_sub[e["ply"]] += 1
        store_sub: Counter = Counter()
        for e in entries:
            sd = e["store_diff"]
            bucket = "negative" if sd < 0 else ("positive" if sd > 0 else "zero")
            store_sub[bucket] += 1

        print(f"\nSuite {label} ({size} openings):")
        print(f"  Path: {path}")
        print(f"  SHA256: {sha}")
        print(f"  By ply: {dict(sorted(ply_sub.items()))}")
        print(f"  By store-diff: {dict(store_sub)}")

    summary = {
        "schema": "opening_suite_v1",
        "total_legal_prefixes_enumerated": total_enumerated,
        "unique_resulting_boards": len(unique),
        "duplicate_prefix_count": duplicate_count,
        "ply_distribution": dict(sorted(ply_dist.items())),
        "store_diff_bucket_distribution": dict(store_buckets),
        "suites": {
            size_label(size): {
                "size": size,
                "path": suite_paths[size],
                "sha256": suite_sha256(suite_paths[size]),
            }
            for size in sorted(suites.keys())
        },
    }

    summary_path = out_dir / "suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
