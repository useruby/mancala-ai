"""Audit frozen arena evidence without invoking an arena subprocess.

This module intentionally only reads reports and already-recorded game rows.  A
search-contract mismatch is a blocker: callers must not infer suite effects
from scores produced under different search semantics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ml.alphazero_lite.export_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]
H4_WEIGHTS_SHA = "3629ba489409baf7c7f194d104101dcffde4ac7ccbfd20409cd112214095b323"
H4_CHECKPOINT_SHA = "ced2eb4b35ed4593408baaf6176832735e6d1667ee33b4f0d769ec91d80de3b9"
PARENT_WEIGHTS_SHA = "f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c"


class ArenaAlignmentAuditError(ValueError):
    """Raised when frozen arena evidence is internally inconsistent."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def recompute_score(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute challenger-oriented score and validate paired seat accounting."""
    game_ids = [int(row["game_index"]) for row in records]
    if len(game_ids) != len(set(game_ids)):
        raise ArenaAlignmentAuditError("duplicated_game_ids")
    if sorted(game_ids) != list(range(len(records))):
        raise ArenaAlignmentAuditError("missing_game_ids")

    outcomes = Counter(str(row["winner"]) for row in records)
    invalid = set(outcomes) - {"challenger", "current", "draw"}
    if invalid:
        raise ArenaAlignmentAuditError(f"malformed_winners:{sorted(invalid)}")
    pairs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        pairs[int(row["opening_index"])].append(row)
    for opening, rows in pairs.items():
        seats = {int(row["challenger_player"]) for row in rows}
        within = {int(row["game_within_opening"]) for row in rows}
        if len(rows) != 2 or seats != {0, 1} or within != {0, 1}:
            raise ArenaAlignmentAuditError(f"invalid_seat_pair:{opening}")

    wins, draws, losses = (
        outcomes["challenger"],
        outcomes["draw"],
        outcomes["current"],
    )
    games = len(records)
    p0 = [row for row in records if int(row["challenger_player"]) == 0]
    p1 = [row for row in records if int(row["challenger_player"]) == 1]

    def score(rows: list[dict[str, Any]]) -> float:
        return sum(
            1.0
            if row["winner"] == "challenger"
            else 0.5
            if row["winner"] == "draw"
            else 0.0
            for row in rows
        ) / len(rows)

    pair_scores = [score(rows) for _, rows in sorted(pairs.items())]
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "games": games,
        "score": (wins + 0.5 * draws) / games,
        "p0_score": score(p0),
        "p1_score": score(p1),
        "opening_pair_score_distribution": dict(Counter(pair_scores)),
    }


def semantic_contract(report: dict[str, Any]) -> dict[str, Any]:
    """Select fields that affect an arena decision, excluding suite identity."""
    notes = report["notes"]
    profile = notes["search_profile"]
    return {
        "challenger_simulations": notes["challenger_simulations"],
        "current_simulations": notes["current_simulations"],
        "c_puct": profile["c_puct"],
        "search_options": profile["search_options"],
        "exact_root_solve_enabled": profile["exact_root_solve_enabled"],
        "exact_root_solve_threshold": profile["exact_root_solve_threshold"],
        "exact_root_solver": profile["exact_root_solver"],
        "exact_root_native_probe_sha256": profile.get("exact_root_native_probe_sha256"),
        "exact_root_tablebase_sha256": profile.get("exact_root_tablebase_sha256"),
        "exact_root_objective": profile["exact_root_objective"],
        "exact_root_tie_rule": profile["exact_root_tie_rule"],
        "exact_leaf_solve_mode": profile["exact_leaf_solve_mode"],
        "seed_contract": notes["seed_contract"],
        "base_seed": notes["base_seed"],
    }


def semantic_diff(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        key: {"diagnostic": left.get(key), "canonical": right.get(key)}
        for key in left.keys() | right.keys()
        if left.get(key) != right.get(key)
    }


def classify(
    identity_ok: bool, contract_diff: dict[str, Any], accounting_ok: bool
) -> str:
    if not identity_ok:
        return "diagnostic_canonical_candidate_identity_mismatch"
    if contract_diff:
        return "diagnostic_canonical_search_contract_mismatch"
    if not accounting_ok:
        return "arena_score_accounting_bug"
    return "diagnostic_canonical_mismatch_unresolved"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs/data/diagnostic-vs-canonical-arena-alignment-audit.json",
    )
    args = parser.parse_args()
    diagnostic_dir = ROOT / ".tmp/seed467-parent-anchor-diagnostic"
    canonical_dir = (
        ROOT
        / ".tmp/seed455-nextgen-s467-parent-anchor-w010/runs/seed455-nextgen-s467-parent-anchor-w010-iter1"
    )
    diagnostic_report = json.loads(
        (diagnostic_dir / "H4_vs_E0.json").read_text(encoding="utf-8")
    )
    canonical_report = json.loads(
        (canonical_dir / "candidate_vs_current_arena.json").read_text(encoding="utf-8")
    )
    diagnostic_rows = load_jsonl(diagnostic_dir / "H4_vs_E0.games.jsonl")
    canonical_rows = load_jsonl(canonical_dir / "shadow_prefilter_games.jsonl")
    diagnostic_score = recompute_score(diagnostic_rows)
    canonical_score = recompute_score(canonical_rows)
    if (
        diagnostic_score["score"] != diagnostic_report["score"]
        or canonical_score["score"] != canonical_report["score"]
    ):
        raise ArenaAlignmentAuditError("recorded_score_disagrees_with_game_rows")

    h4_path = diagnostic_dir / "artifacts/H4/weights.json"
    candidate_path = canonical_dir / "weights.json"
    h4_metadata = json.loads(
        (diagnostic_dir / "artifacts/H4/metadata.json").read_text(encoding="utf-8")
    )
    generation = json.loads(
        (
            ROOT
            / "docs/data/alphazero-lite-generations/seed455-nextgen-s467-parent-anchor-w010/generation.json"
        ).read_text(encoding="utf-8")
    )
    identity = {
        "h4_checkpoint_sha256": h4_metadata["artifacts"]["weights_sha256"],
        "h4_diagnostic_weights_sha256": sha256_file(h4_path),
        "h4_canonical_candidate_weights_sha256": sha256_file(candidate_path),
        "generation_checkpoint_sha256": generation["candidate"]["checkpoint"]["sha256"],
        "generation_exported_weights_sha256": generation["candidate"]["weights"][
            "sha256"
        ],
        "expected_checkpoint_sha256": H4_CHECKPOINT_SHA,
        "expected_weights_sha256": H4_WEIGHTS_SHA,
        "parent_weights_sha256": PARENT_WEIGHTS_SHA,
        "diagnostic_parent_path": "model-artifact/current/weights.json",
        "canonical_parent_path": "model-artifact/current/weights.json",
    }
    identity_ok = (
        all(
            identity[key] == H4_WEIGHTS_SHA
            for key in (
                "h4_diagnostic_weights_sha256",
                "h4_canonical_candidate_weights_sha256",
            )
        )
        and identity["h4_checkpoint_sha256"] == H4_CHECKPOINT_SHA
        and identity["generation_checkpoint_sha256"] == H4_CHECKPOINT_SHA
        and identity["generation_exported_weights_sha256"] == H4_WEIGHTS_SHA
        and sha256_file(ROOT / "model-artifact/current/weights.json")
        == PARENT_WEIGHTS_SHA
    )
    diff = semantic_diff(
        semantic_contract(diagnostic_report), semantic_contract(canonical_report)
    )
    payload = {
        "schema": "diagnostic_vs_canonical_arena_alignment_audit_v1",
        "semantic_identity": "diagnostic-vs-canonical-arena-alignment-audit",
        "frozen_inputs": {
            "diagnostic_suite_sha256": diagnostic_report["notes"]["suite_sha256"],
            "canonical_suite_sha256": canonical_report["notes"]["suite_sha256"],
            "diagnostic_report_sha256": sha256_file(diagnostic_dir / "H4_vs_E0.json"),
            "canonical_report_sha256": sha256_file(
                canonical_dir / "candidate_vs_current_arena.json"
            ),
        },
        "identity": identity,
        "identity_ok": identity_ok,
        "diagnostic_score_recomputation": diagnostic_score,
        "canonical_score_recomputation": canonical_score,
        "search_contract_diff": diff,
        "classification": classify(identity_ok, diff, True),
        "classification_reason": "The diagnostic exact-root solver is Python EndgameTablebase without recorded native/tablebase identities; canonical uses native_kvtb_root_action_probe_v1 bound to both native and tablebase SHA256 values.",
        "stop_rule_applied": "No suite-distribution, root-probe, or new-game analysis was performed after the material contract mismatch.",
        "scope": {
            "training_runs": 0,
            "self_play_games": 0,
            "canonical_games": 0,
            "promotions": 0,
        },
    }
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
