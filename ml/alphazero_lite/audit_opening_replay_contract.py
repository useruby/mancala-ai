"""Reconstruct the declared-versus-actual opening audit for seed461 suites."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ml.alphazero_lite.arena import apply_opening_moves
from ml.alphazero_lite.build_opening_suite import INITIAL_STATE, canonical_key
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
SUITES = {
    "#383": "seed461-batch-order-sensitivity-confirmation-openings.jsonl",
    "#384": "seed461-lr-sensitivity-openings-v2.jsonl",
    "#385 retrospective fixed-E4 diagnostic": "seed461-lr-sensitivity-openings-v2.jsonl",
    "#386": "seed461-cosine-lr-ablation-openings.jsonl",
    "#387 fixed E2-E4 averaging registration": "seed461-e2-e4-average-openings.jsonl",
    "#388": "seed461-e2-e4-average-openings.jsonl",
    "#389": "seed461-e3-e4-openings.jsonl",
    "#390": "seed461-cross-order-e4-average-openings.jsonl",
    "#391 seed391": "order38615-a5-confirmation-seed391-openings.jsonl",
    "#391 seed392": "order38615-a5-confirmation-seed392-openings.jsonl",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct() -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "opening-replay-reconstruction-v1",
        "suites": {},
    }
    identities: dict[str, set[str]] = {}
    for label, filename in SUITES.items():
        path = DATA / filename
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        ids: list[str] = []
        lengths: Counter[int] = Counter()
        declared_match = 0
        prefix_match = 0
        for row in rows:
            game = KalahGame.from_state(INITIAL_STATE)
            applied = apply_opening_moves(
                game, [int(move) for move in row["prefix_moves"]]
            )
            lengths[applied] += 1
            identity = canonical_key(game.to_state())
            ids.append(identity)
            if "state" in row:
                declared_match += identity == canonical_key(row["state"])
            prefix_match += applied == len(row["prefix_moves"])
        identities[label] = set(ids)
        report["suites"][label] = {
            "path": f"docs/data/{filename}",
            "suite_sha256": sha(path),
            "declared_count": len(rows),
            "actual_unique_count": len(set(ids)),
            "applied_prefix_length_counts": {
                str(k): lengths[k] for k in sorted(lengths)
            },
            "complete_prefix_count": prefix_match,
            "declared_state_matches": declared_match,
            "actual_state_identities": sorted(set(ids)),
        }
    report["pairwise_overlap"] = {
        f"{left}|{right}": len(identities[left] & identities[right])
        for i, left in enumerate(identities)
        for right in list(identities)[i + 1 :]
    }
    binding_path = DATA / "order38615-a5-confirmation-evaluation-binding.json"
    binding = json.loads(binding_path.read_text())
    report["raw_evidence_hashes"] = {}
    for seed, item in binding["reports"].items():
        recovered = {}
        for kind in ("games", "report"):
            path = Path(item[kind])
            actual = sha(path)
            if actual != item[f"{kind}_sha256"]:
                raise ValueError(f"raw_evidence_hash_mismatch:{seed}:{kind}")
            recovered[kind] = {"path": str(path), "sha256": actual, "verified": True}
        report["raw_evidence_hashes"][seed] = recovered
    return report


if __name__ == "__main__":
    print(json.dumps(reconstruct(), indent=2, sort_keys=True))
