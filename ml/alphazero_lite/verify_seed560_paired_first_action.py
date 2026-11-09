"""Read-only verifier for the published seed560 retrospective diagnostic."""

from __future__ import annotations

import json
import math
from pathlib import Path

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed557_retrospective import canonical_hash, sha256
from ml.alphazero_lite.seed560_paired_first_action import (
    INITIALIZER,
    NATIVE,
    P_ARTIFACT,
    TABLEBASE,
    analyze,
    candidates,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed560-paired-first-action"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _asset_sha(path: Path) -> str:
    """Authenticate either materialized bytes or a Git LFS SHA-256 pointer."""
    contents = path.read_bytes()
    marker = b"version https://git-lfs.github.com/spec/v1\n"
    if contents.startswith(marker):
        for line in contents.decode("ascii").splitlines()[1:]:
            if line.startswith("oid sha256:"):
                return line.removeprefix("oid sha256:")
        raise ValueError("invalid_lfs_asset_pointer")
    return sha256(contents)


def _expected_seed(index: int, ply: int, role: str) -> int:
    label = f"seed560:{role}:{index}:{ply}"
    return int(sha256(label.encode())[:8], 16) % (2**31)


def validate_continuation(root: dict, record: dict, outcome: dict) -> None:
    """Replay one forced action and trajectory, including actor/seed bindings."""
    game = KalahGame.from_state(root["state"])
    root_player = game.current_player
    action = int(record["action"])
    if action not in game.possible_moves() or not game.move(game.pit_index(action)):
        raise ValueError("forced_action_not_legal")
    if int(outcome["root_player"]) != root_player:
        raise ValueError("continuation_root_perspective_mismatch")
    trajectory = outcome.get("trajectory")
    if not isinstance(trajectory, list):
        raise ValueError("continuation_trajectory_missing")
    for ply, step in enumerate(trajectory):
        if game.over():
            raise ValueError("trajectory_moves_after_terminal")
        state = game.to_state()
        expected = _expected_seed(int(record["opening_index"]), ply, "continuation")
        if (
            step.get("ply") != ply
            or step.get("actor") != game.current_player
            or step.get("state_hash") != canonical_hash(state)
            or step.get("seed") != expected
            or step.get("simulations") != 128
            or step.get("exact_root_cutoff") != 16
            or step.get("exact_leaf_solving") is not False
            or step.get("exact_root_solver_calls") != int(sum(game.pits) <= 16)
        ):
            raise ValueError("continuation_seed_or_runtime_binding_mismatch")
        move = int(step["action"])
        if move not in game.possible_moves() or not game.move(game.pit_index(move)):
            raise ValueError("continuation_action_illegal")
        if game.to_state() != step.get("state_after"):
            raise ValueError("continuation_state_transition_conflict")
    if not game.over():
        raise ValueError("continuation_missing_or_capped")
    margin = game.captured_seeds[root_player] - game.captured_seeds[1 - root_player]
    score = 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
    if (
        outcome.get("margin") != margin
        or outcome.get("score") != score
        or outcome.get("final_stores") != list(game.captured_seeds)
    ):
        raise ValueError("continuation_terminal_evidence_contradiction")


def verify(data: Path = DATA) -> dict:
    cohort_doc = json.loads((data / "cohort-manifest.json").read_text())
    cohort, audit = candidates()
    if cohort_doc.get("states") != cohort:
        raise ValueError("cohort_does_not_reconstruct_from_authenticated_archive")
    if json.loads((data / "selection-audit.json").read_text()) != audit:
        raise ValueError("selection_audit_mismatch")
    freeze = json.loads((data / "execution-freeze.json").read_text())
    if freeze.get("cohort_sha256") != sha256(
        (data / "cohort-manifest.json").read_bytes()
    ) or freeze.get("selection_audit_sha256") != sha256(
        (data / "selection-audit.json").read_bytes()
    ):
        raise ValueError("freeze_hash_mismatch")
    if (
        freeze.get("source_commit") != "f38838e6ffe20a44c1e8c161a9daed0f1243f0ae"
        or freeze.get("serving_model") != "#556 P-E4 policy + initializer composition"
        or freeze.get("root_search", {}).get("budgets") != [128, 512]
        or freeze.get("root_search", {}).get("exact_root_cutoff") != 16
        or freeze.get("root_search", {}).get("exact_leaf_solving") is not False
        or freeze.get("root_search", {}).get("fresh_tree") is not True
        or freeze.get("continuation", {}).get("simulations") != 128
        or freeze.get("continuation", {}).get("exact_root_cutoff") != 16
        or freeze.get("continuation", {}).get("exact_leaf_solving") is not False
        or freeze.get("analysis", {}).get("opening_bootstrap_resamples") != 20000
        or freeze.get("analysis", {}).get("bootstrap_seed") != 560
    ):
        raise ValueError("frozen_protocol_semantics_mismatch")
    for name, directory in (("P", P_ARTIFACT), ("initializer", INITIALIZER)):
        actual = {
            p.name: sha256(p.read_bytes())
            for p in sorted(directory.iterdir())
            if p.is_file()
        }
        if actual != freeze["artifacts"][name]:
            raise ValueError(f"serving_artifact_identity_mismatch:{name}")
    if freeze.get("exact_root_assets") != {
        "native_sha256": sha256(NATIVE.read_bytes()),
        "tablebase_sha256": _asset_sha(TABLEBASE),
    }:
        raise ValueError("exact_root_asset_identity_mismatch")
    for relative, digest in freeze.get("source_sha256", {}).items():
        if sha256((ROOT / relative).read_bytes()) != digest:
            raise ValueError(f"execution_source_identity_mismatch:{relative}")
    roots = _jsonl(data / "root-searches.jsonl")
    continuations = _jsonl(data / "continuations.jsonl")
    if len(roots) != 256 or len(continuations) != 512:
        raise ValueError("incomplete_root_or_continuation_evidence")
    if [r["opening_index"] for r in roots] != list(range(256)):
        raise ValueError("root_search_coverage_or_order_mismatch")
    outcome_by_key = {}
    for row in continuations:
        key = (int(row["opening_index"]), int(row["budget"]))
        if key in outcome_by_key:
            raise ValueError("duplicate_continuation_record")
        outcome_by_key[key] = row
    if set(outcome_by_key) != {(i, b) for i in range(256) for b in (128, 512)}:
        raise ValueError("continuation_pair_coverage_mismatch")
    action_changes = 0
    for index, (root_search, root_state) in enumerate(zip(roots, cohort, strict=True)):
        if root_search.get("state_hash") != root_state["state_hash"]:
            raise ValueError("root_search_state_mismatch")
        expected_root_seed = int(
            sha256(f"seed560:{root_state['state_hash']}:root".encode())[:8], 16
        ) % (2**31)
        if root_search.get("seed") != expected_root_seed:
            raise ValueError("root_search_seed_mismatch")
        legal = [
            game_move
            for game_move in KalahGame.from_state(root_state["state"]).possible_moves()
        ]
        if set(root_search["actions"].values()) - set(legal):
            raise ValueError("root_search_selected_illegal_action")
        if any(
            not math.isfinite(float(root_search["latency_seconds"][str(budget)]))
            or float(root_search["latency_seconds"][str(budget)]) <= 0
            for budget in (128, 512)
        ):
            raise ValueError("invalid_root_search_latency")
        for budget in (128, 512):
            row = outcome_by_key[(index, budget)]
            if (
                row.get("state_hash") != root_state["state_hash"]
                or row.get("action") != root_search["actions"][str(budget)]
            ):
                raise ValueError("continuation_root_action_binding_mismatch")
            validate_continuation(root_state, row, row["outcome"])
        same = root_search["actions"]["128"] == root_search["actions"]["512"]
        action_changes += not same
        if (
            same
            and outcome_by_key[(index, 128)]["outcome"]
            != outcome_by_key[(index, 512)]["outcome"]
        ):
            raise ValueError("identical_action_continuation_reuse_conflict")
    calculated = analyze(data, persist=False)
    saved = json.loads((data / "analysis.json").read_text())
    if calculated != saved:
        raise ValueError("analysis_recomputation_mismatch")
    return {
        "status": "verified",
        "roots": len(roots),
        "continuation_arm_records": len(continuations),
        "action_changes": int(action_changes),
        "decision": saved["decision"],
        "mean_score_difference_512_minus_128": saved[
            "mean_score_difference_512_minus_128"
        ],
        "paired_opening_bootstrap_95_ci": saved["paired_opening_bootstrap_95_ci"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    print(json.dumps(verify(args.data), sort_keys=True, indent=2))
