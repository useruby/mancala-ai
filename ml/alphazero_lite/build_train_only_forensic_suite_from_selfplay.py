#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite import build_forensic_suite
from ml.alphazero_lite.forensic_suite import REQUIRED_BUCKETS, canonical_state_key, load_suite
from ml.alphazero_lite.kalah_rules import KalahGame


DEFAULT_HOLDOUT_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
DEFAULT_TARGET_SIZE = 224
DEFAULT_MIN_PER_BUCKET = 12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout-path", default=str(DEFAULT_HOLDOUT_PATH))
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--min-per-bucket", type=int, default=DEFAULT_MIN_PER_BUCKET)
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def discover_self_play_files(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw_input in inputs:
        path = Path(raw_input)
        if path.is_dir():
            paths.extend(sorted(path.rglob("self_play.jsonl")))
            continue
        if path.is_file() and path.name == "self_play.jsonl":
            paths.append(path)
    deduplicated: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduplicated.append(path)
    if not deduplicated:
        raise ValueError("no self_play.jsonl files found in inputs")
    return deduplicated


def decode_state(features: list[Any]) -> dict[str, Any]:
    if not isinstance(features, list) or len(features) < 15:
        raise ValueError("self-play row state must be a feature list with at least 15 values")
    return {
        "player_pits": [max(0, int(round(float(features[idx]) * 48.0))) for idx in range(6)],
        "opponent_pits": [
            max(0, int(round(float(features[6 + idx]) * 48.0))) for idx in range(6)
        ],
        "player_store": max(0, int(round(float(features[12]) * 48.0))),
        "opponent_store": max(0, int(round(float(features[13]) * 48.0))),
        "current_player": int(round(float(features[14]))),
    }


def holdout_keys(path: Path) -> set[str]:
    return {position.canonical_key for position in load_suite(path)}


def row_from_self_play(*, row: dict[str, Any], source_path: Path, line_number: int) -> dict[str, Any] | None:
    state = decode_state(row["state"])
    game = KalahGame.from_state(state)
    legal_moves = game.possible_moves()
    if not legal_moves:
        return None
    ply = int(row.get("move_index", 0))
    bucket = build_forensic_suite._choose_bucket(state, ply)
    if bucket is None:
        return None
    phase = build_forensic_suite._phase_for_state(state, ply)
    return {
        "state": state,
        "legal_moves": legal_moves,
        "bucket": bucket,
        "phase": phase,
        "ply": ply,
        "source": f"batch1-self-play:{source_path.parent.name}",
        "source_path": str(source_path),
        "line_number": line_number,
    }


def collect_candidates(self_play_files: list[Path], *, excluded_keys: set[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_path in self_play_files:
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                candidate = row_from_self_play(
                    row=payload,
                    source_path=source_path,
                    line_number=line_number,
                )
                if candidate is None:
                    continue
                key = canonical_state_key(candidate["state"])
                if key in excluded_keys or key in seen:
                    continue
                seen.add(key)
                candidate["canonical_state"] = key
                candidates.append(candidate)
    if not candidates:
        raise ValueError("no train-only forensic candidates found after holdout exclusion")
    return candidates


def collect_proxy_candidates(
    *, excluded_keys: set[str], seen_keys: set[str], limit: int
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if limit <= 0:
        return candidates

    for ply, state in build_forensic_suite._candidate_states():
        if ply <= 8:
            continue
        key = canonical_state_key(state)
        if key in excluded_keys or key in seen_keys:
            continue
        if not build_forensic_suite._is_proxy_disagreement_candidate(state, ply):
            continue
        candidates.append(
            {
                "state": state,
                "legal_moves": KalahGame.from_state(state).possible_moves(),
                "bucket": build_forensic_suite.PROXY_DISAGREEMENT_BUCKET,
                "phase": build_forensic_suite._phase_for_state(state, ply),
                "ply": ply,
                "source": "batch1-train-only:generated_proxy",
                "source_path": "generated",
                "line_number": 0,
                "canonical_state": key,
            }
        )
        seen_keys.add(key)
        if len(candidates) >= limit:
            break
    return candidates


def build_suite_rows(
    candidates: list[dict[str, Any]], *, target_size: int, min_per_bucket: int
) -> list[dict[str, Any]]:
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if min_per_bucket < 0:
        raise ValueError("min_per_bucket must be non-negative")

    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    row_ids: Counter[str] = Counter()

    def append_candidate(candidate: dict[str, Any]) -> None:
        bucket = str(candidate["bucket"])
        row_ids[bucket] += 1
        rows.append(
            {
                "id": f"{bucket}-{row_ids[bucket]:03d}",
                "state": candidate["state"],
                "side_to_move": int(candidate["state"]["current_player"]),
                "legal_moves": list(candidate["legal_moves"]),
                "phase": str(candidate["phase"]),
                "bucket": bucket,
                "tags": [bucket, "train_only", f"ply_{int(candidate['ply'])}"],
                "source": str(candidate["source"]),
            }
        )
        counts[bucket] += 1

    prioritized = sorted(
        candidates,
        key=lambda candidate: (
            int(candidate["bucket"] != build_forensic_suite.PROXY_DISAGREEMENT_BUCKET),
            int(candidate["ply"]),
            candidate["canonical_state"],
        ),
    )
    for bucket in sorted(REQUIRED_BUCKETS):
        for candidate in prioritized:
            if len(rows) >= target_size:
                break
            if candidate["bucket"] != bucket or counts[bucket] >= min_per_bucket:
                continue
            append_candidate(candidate)

    for candidate in prioritized:
        if len(rows) >= target_size:
            break
        if any(existing["state"] == candidate["state"] for existing in rows):
            continue
        append_candidate(candidate)

    present_buckets = {row["bucket"] for row in rows}
    missing = sorted(REQUIRED_BUCKETS - present_buckets)
    if missing:
        raise ValueError(
            "train-only forensic suite is missing required buckets: " + ", ".join(missing)
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    holdout_path = resolve_path(root, args.holdout_path)
    self_play_files = discover_self_play_files(
        [value.strip() for value in args.inputs.split(",") if value.strip()]
    )
    candidates = collect_candidates(
        self_play_files,
        excluded_keys=holdout_keys(holdout_path),
    )
    seen_keys = {candidate["canonical_state"] for candidate in candidates}
    excluded_keys = holdout_keys(holdout_path)
    candidates.extend(
        collect_proxy_candidates(
            excluded_keys=excluded_keys,
            seen_keys=seen_keys,
            limit=max(0, int(args.min_per_bucket)),
        )
    )
    rows = build_suite_rows(
        candidates,
        target_size=int(args.target_size),
        min_per_bucket=int(args.min_per_bucket),
    )
    out_path = resolve_path(root, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out_path),
                "rows": len(rows),
                "buckets": dict(sorted(Counter(row["bucket"] for row in rows).items())),
                "self_play_files": len(self_play_files),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
