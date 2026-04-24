#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.opening_cache import DEFAULT_OPENING_GATE, SCHEMA, canonical_key, canonical_payload, state_qualifies_for_opening_cache


DEFAULT_ARTIFACT_PATH = Path("storage/ai/alphazero_lite/opening_cache_v1.json")
DEFAULT_GENERATION_GATE = dict(DEFAULT_OPENING_GATE)
MAX_EXAMPLE_ORIGINS = 3


def build_search_profile(*, seed=0):
    profile = {
        "version": "v1",
        "kind": "opening_cache_teacher",
        "player_mode": "classic_mcts",
        "classic_mcts_simulations": 1200,
        "root_policy_mode": "deterministic",
        "deterministic_seed": int(seed),
    }
    encoded_profile = json.dumps(profile, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        **profile,
        "hash": hashlib.sha256(encoded_profile.encode("utf-8")).hexdigest(),
    }


DEFAULT_SEARCH_PROFILE = build_search_profile()


def validate_search_profile_for_default_teacher(search_profile):
    if search_profile != DEFAULT_SEARCH_PROFILE:
        raise ValueError("search_profile must match the built-in default teacher settings when teacher is omitted")


def default_teacher(state):
    search = ClassicMCTS(KalahGame.from_state(state), simulations=1200, seed=0)
    summary = search.root_summary()
    root = search.search_root()
    selected_move = summary["selected_move"]
    policy = [0.0] * 6
    if selected_move is not None and 0 <= int(selected_move) < len(policy):
        policy[int(selected_move)] = 1.0
    value = 0.0 if root.visits == 0 else (2.0 * (root.wins / float(root.visits)) - 1.0)
    return {
        "selected_move": selected_move,
        "policy": policy,
        "value": value,
        "budget": summary["budget"],
    }


def provenance_teacher_kind(search_profile):
    return str(search_profile.get("player_mode") or search_profile.get("kind") or "unknown")


def build_opening_cache_artifact(source_positions, *, teacher=None, opening_gate=None, search_profile=None):
    gate = dict(DEFAULT_GENERATION_GATE)
    if opening_gate:
        gate.update(opening_gate)

    retained = {}
    source_count = 0
    qualifying_count = 0
    for source in source_positions:
        source_count += 1
        state = source["state"]
        ply = int(source.get("ply", 0))
        if not state_qualifies_for_opening_cache(state, ply=ply, opening_gate=gate):
            continue

        qualifying_count += 1
        normalized_state = canonical_payload(state)
        key = canonical_key(normalized_state)
        retained_entry = retained.setdefault(
            key,
            {
                "state": normalized_state,
                "side_to_move": int(normalized_state["current_player"]),
                "support_count": 0,
                "origins": [],
            },
        )
        retained_entry["support_count"] += 1
        origin = source.get("origin")
        if origin is not None:
            retained_entry["origins"].append(str(origin))

    teacher_fn = teacher or default_teacher
    profile = dict(search_profile or DEFAULT_SEARCH_PROFILE)
    if teacher is None:
        validate_search_profile_for_default_teacher(profile)
    teacher_kind = provenance_teacher_kind(profile)
    entries = {}
    sorted_keys = sorted(retained.keys())
    for key in sorted_keys:
        retained_entry = retained[key]
        example_origins = sorted(set(retained_entry["origins"]))[:MAX_EXAMPLE_ORIGINS]
        teacher_result = teacher_fn(retained_entry["state"])
        entries[key] = {
            "state": retained_entry["state"],
            "side_to_move": retained_entry["side_to_move"],
            "selected_move": teacher_result["selected_move"],
            "policy": list(teacher_result["policy"]),
            "value": float(teacher_result["value"]),
            "budget": dict(teacher_result["budget"]),
            "provenance": {
                "teacher_kind": teacher_kind,
                "search_profile_hash": profile["hash"],
                "hit_count_in_sources": retained_entry["support_count"],
                "example_origins": example_origins,
            },
        }

    return {
        "schema": SCHEMA,
        "opening_gate": gate,
        "search_profile": profile,
        "generation_metrics": {
            "source_positions": source_count,
            "qualifying_positions": qualifying_count,
            "retained_positions": len(sorted_keys),
            "teacher_evaluations": len(sorted_keys),
        },
        "entries": entries,
    }


def write_opening_cache_artifact(
    source_positions,
    *,
    teacher=None,
    opening_gate=None,
    search_profile=None,
    artifact_path=DEFAULT_ARTIFACT_PATH,
):
    artifact = build_opening_cache_artifact(
        source_positions,
        teacher=teacher,
        opening_gate=opening_gate,
        search_profile=search_profile,
    )
    destination = Path(artifact_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate deterministic opening cache artifact")
    parser.add_argument("input", type=Path, help="Path to JSON list of source positions")
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH, help="Artifact output path")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source_positions = json.loads(args.input.read_text(encoding="utf-8"))
    artifact = write_opening_cache_artifact(source_positions, artifact_path=args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "generation_metrics": artifact["generation_metrics"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
