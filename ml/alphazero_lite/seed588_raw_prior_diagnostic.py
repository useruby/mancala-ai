"""Read-only source verifier for the retrospective #559 raw-prior diagnostic.

This module verifies frozen inputs only. It deliberately does not load model
artifacts into evaluators or perform policy inference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed557_retrospective import canonical_hash

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed559-fixed-position-feasibility/run-0003"
ARTIFACTS = {
    "T512": ROOT
    / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts/T512",
    "P": ROOT
    / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts/P-E4",
}
EXPECTED_ARTIFACTS = {
    "T512": {
        "metadata.json": "b37f3aa75d828b9b9a0743930c73c8e168b159f6b37434eba1d9871e769af1b8",
        "model.npz": "8940cbff36976801e74542f5ee2d6c5caffe0cafd57c7e787cdb375f3bbe485e",
        "search_policy.json": "280c33d88266969da72a7984dabdcfea180bacb5cb690cb759af1ebdbaac65bc",
        "weights.json": "70af73e6f8ee3f4fb76981584c1f2068a4b669300be90bd71cfdd8f71b369997",
    },
    "P": {
        "metadata.json": "a9d133043c64ca41e3d77354f4ccf3403ab5f94286608f01589387159ac0258c",
        "model.npz": "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
        "search_policy.json": "280c33d88266969da72a7984dabdcfea180bacb5cb690cb759af1ebdbaac65bc",
        "weights.json": "89dead525a255cff4760867645fd9ce05cb2bbadcf1d72c5ac9089868c4a8136",
    },
}
EXPECTED_INPUTS = {
    "cohort-manifest.json": "f2e03fb9491474056e7a7cb3ace225cd2310f9069c9e48600d45039ed63158fb",
    "execution-freeze.json": "3b1ab386c33534130c7e15df024f21ef75359ccd00d453e4c6aea104c8181927",
    "oracle-records.jsonl": "83bec1bb348cbc6d1183112363f27d0612ee204a48d14ee5dbb7530e9b5fb8c2",
    "search-records.jsonl": "b2653c41fe34328386ae432a3e019568eba9df7c5dc82a596a18917e8c8fb12e",
}


class VerificationError(ValueError):
    """Frozen retrospective evidence failed verification."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.splitlines() if line]


def verify_documents(
    cohort_doc: dict[str, Any],
    freeze: dict[str, Any],
    oracle_rows: list[dict[str, Any]],
    search_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate all roots, legal actions, oracle coverage, and archived P priors."""
    states = cohort_doc.get("states")
    if not isinstance(states, list) or len(states) != 512:
        raise VerificationError("cohort_root_count")
    if len({row["state_hash"] for row in states}) != 512:
        raise VerificationError("duplicate_state")
    if len({row["opening_index"] for row in states}) != 512:
        raise VerificationError("duplicate_opening")
    if len(oracle_rows) != 512 or len(search_rows) != 512:
        raise VerificationError("evidence_row_count")
    oracle_by_hash = {row.get("state_hash"): row for row in oracle_rows}
    search_by_hash = {row.get("state_hash"): row for row in search_rows}
    if len(oracle_by_hash) != 512 or len(search_by_hash) != 512:
        raise VerificationError("duplicate_evidence_state")
    source_freeze = freeze.get("authenticated_policies", {})
    if source_freeze.get("T512") != EXPECTED_ARTIFACTS["T512"]:
        raise VerificationError("T512_freeze_artifact_identity")
    if source_freeze.get("P") != EXPECTED_ARTIFACTS["P"]:
        raise VerificationError("P_freeze_artifact_identity")

    for root in states:
        state = root["state"]
        state_hash = canonical_hash(state)
        if root.get("state_hash") != state_hash:
            raise VerificationError("root_state_hash")
        active = sum(state["player_pits"]) + sum(state["opponent_pits"])
        if active != root.get("active_pit_stones") or not 17 <= active <= 21:
            raise VerificationError("active_pit_stones")
        game = KalahGame.from_state(state)
        legal_relative = sorted(game.possible_moves())
        legal_absolute = sorted(game.pit_index(move) for move in legal_relative)
        if legal_absolute != sorted(root["legal_absolute_actions"]):
            raise VerificationError("root_legal_actions")
        oracle = oracle_by_hash.get(state_hash)
        if oracle is None or not oracle.get("coverage_complete"):
            raise VerificationError("oracle_missing_or_incomplete")
        margins = oracle.get("action_margins", {})
        if sorted(map(int, oracle.get("legal_moves", []))) != legal_relative:
            raise VerificationError("oracle_legal_actions")
        if sorted(map(int, margins)) != legal_relative:
            raise VerificationError("oracle_action_margins")
        if any(not isinstance(margins[str(move)], int) for move in legal_relative):
            raise VerificationError("oracle_margin_not_integer")
        archived = search_by_hash.get(state_hash)
        if (
            archived is None
            or sorted(archived.get("legal_moves", [])) != legal_relative
        ):
            raise VerificationError("archived_prior_missing_or_legal_actions")
        priors = archived.get("priors")
        if not isinstance(priors, list) or len(priors) != 6:
            raise VerificationError("archived_prior_shape")
        if any(float(priors[i]) != 0.0 for i in range(6) if i not in legal_relative):
            raise VerificationError("archived_illegal_prior_nonzero")
        if abs(sum(float(priors[i]) for i in legal_relative) - 1.0) > 2e-6:
            raise VerificationError("archived_prior_not_normalized")
    if set(oracle_by_hash) != {row["state_hash"] for row in states}:
        raise VerificationError("oracle_membership_mismatch")
    if set(search_by_hash) != {row["state_hash"] for row in states}:
        raise VerificationError("search_membership_mismatch")
    return {
        "status": "verified",
        "roots": 512,
        "unique_openings": 512,
        "active_pit_stones": {"minimum": 17, "maximum": 21},
        "complete_oracle_roots": 512,
        "archived_p_prior_roots": 512,
        "inference_performed": False,
    }


def verify(root: Path = ROOT) -> dict[str, Any]:
    data = root / "docs/data/seed559-fixed-position-feasibility/run-0003"
    expected_freeze = json.loads((data / "execution-freeze.json").read_bytes())
    actual_hashes = {name: _sha((data / name).read_bytes()) for name in EXPECTED_INPUTS}
    if actual_hashes != EXPECTED_INPUTS:
        raise VerificationError("#559_input_hash_mismatch")
    for model, files in EXPECTED_ARTIFACTS.items():
        actual = {
            name: _sha((root / ARTIFACTS[model].relative_to(ROOT) / name).read_bytes())
            for name in files
        }
        if actual != files:
            raise VerificationError(f"{model}_artifact_hash_mismatch")
    cohort = json.loads((data / "cohort-manifest.json").read_bytes())
    oracles = _jsonl((data / "oracle-records.jsonl").read_bytes())
    searches = _jsonl((data / "search-records.jsonl").read_bytes())
    return verify_documents(cohort, expected_freeze, oracles, searches)


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
