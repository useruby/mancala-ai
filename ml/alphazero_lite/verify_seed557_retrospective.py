"""Read-only verifier for the seed557 #556 retrospective diagnostic."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed557_retrospective import (
    DATA,
    analyze,
    authentication_receipts,
    authenticate_runtime,
    load_authenticated_archive,
    reconstruct_cohort,
    sha256,
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify(data: Path = DATA) -> dict:
    cohort_evidence = json.loads((data / "cohort.json").read_text())
    frozen_states = cohort_evidence["states"]
    openings, records, receipt = load_authenticated_archive()
    reconstructed, selection = reconstruct_cohort(openings, records)
    if frozen_states != reconstructed:
        raise ValueError("frozen cohort reconstruction mismatch")
    freeze = json.loads((data / "execution-freeze.json").read_text())
    if freeze["source_commit"] != "f38838e6ffe20a44c1e8c161a9daed0f1243f0ae":
        raise ValueError("source commit binding mismatch")
    if freeze["selection"] != selection:
        raise ValueError("selection counts mismatch")
    if freeze["authenticated_policies"] != authenticate_runtime():
        raise ValueError("authenticated policy artifact hash mismatch")
    if freeze["authentication_receipts"] != authentication_receipts():
        raise ValueError("policy publication receipt binding mismatch")
    expected_batches = {
        f"games-{index:04d}.jsonl": receipt["evidence_sha256"][
            f"batches/games-{index:04d}.jsonl"
        ]
        for index in range(1024)
    }
    if freeze["verified_game_batch_sha256"] != expected_batches:
        raise ValueError("game batch receipt binding mismatch")
    if freeze["verification_receipt_sha256"] != sha256(
        subprocess.check_output(
            [
                "git",
                "show",
                "f38838e6ffe20a44c1e8c161a9daed0f1243f0ae:docs/data/seed556-exploratory-arena/screen-0005/verification-receipt.json",
            ]
        )
    ):
        raise ValueError("#556 receipt binding mismatch")
    searches = _jsonl(data / "search-records.jsonl")
    oracles = _jsonl(data / "oracle-records.jsonl")
    if len(searches) != len(frozen_states) * 3 or len(oracles) != len(frozen_states):
        raise ValueError("incomplete root evidence")
    by_hash = {item["state_hash"]: item for item in frozen_states}
    oracle_hashes: set[str] = set()
    for oracle in oracles:
        if oracle["state_hash"] in oracle_hashes:
            raise ValueError("duplicate oracle root")
        oracle_hashes.add(oracle["state_hash"])
        item = by_hash.get(oracle["state_hash"])
        if item is None:
            raise ValueError("oracle root is outside frozen cohort")
        legal = KalahGame.from_state(item["state"]).possible_moves()
        margins = {
            int(key): int(value) for key, value in oracle["action_margins"].items()
        }
        if oracle.get("coverage_complete") is not True or set(legal) != set(margins):
            raise ValueError(
                f"incomplete exact legal-action coverage:{item['state_hash']}"
            )
        if sorted(oracle["legal_moves"]) != sorted(legal):
            raise ValueError(
                "oracle legal action list differs from reconstructed state"
            )
    seen: set[tuple[str, str]] = set()
    seeds_by_root: dict[str, set[int]] = {}
    for search in searches:
        key = (search["state_hash"], search["policy"])
        if key in seen or search["state_hash"] not in by_hash:
            raise ValueError("duplicate or out-of-cohort search record")
        if search["policy"] not in {"T512", "C128", "P"}:
            raise ValueError("unknown policy in search evidence")
        seen.add(key)
        seeds_by_root.setdefault(search["state_hash"], set()).add(int(search["seed"]))
        item = by_hash[search["state_hash"]]
        legal = KalahGame.from_state(item["state"]).possible_moves()
        if sorted(search["legal_moves"]) != sorted(legal):
            raise ValueError(
                "search legal action list differs from reconstructed state"
            )
        if (
            search["highest_prior_action"] not in legal
            or search["puct_action"] not in legal
        ):
            raise ValueError("search selected an illegal action")
        priors = [float(value) for value in search["priors"]]
        if search["highest_prior_action"] != max(
            legal, key=lambda move: (priors[move], -move)
        ):
            raise ValueError("highest-prior action does not match recorded priors")
    expected_search_keys = {
        (state["state_hash"], policy)
        for state in frozen_states
        for policy in ("T512", "C128", "P")
    }
    if seen != expected_search_keys or oracle_hashes != set(by_hash):
        raise ValueError("per-root policy or oracle coverage is incomplete")
    if any(len(seeds) != 1 for seeds in seeds_by_root.values()):
        raise ValueError("policy searches do not share matched per-root seeds")
    reconstructed_analysis = analyze(frozen_states, searches, oracles)
    reconstructed_roots = reconstructed_analysis.pop("root_records")
    published_roots = _jsonl(data / "root-metrics.jsonl")
    if reconstructed_roots != published_roots:
        raise ValueError("root metric reconstruction mismatch")
    published = json.loads((data / "analysis.json").read_text())
    if reconstructed_analysis != published:
        raise ValueError("aggregate metric reconstruction mismatch")
    return {
        "status": "verified",
        "selected_roots": len(frozen_states),
        "search_records": len(searches),
        "oracle_roots_with_full_coverage": len(oracles),
        "verified_game_batches": 1024,
        "selection_summary": selection,
        "receipt_games": receipt["verified_games"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
