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
    expected_oracle_identities,
    load_authenticated_archive,
    search_seed,
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
    identity_record_path = data / "oracle-identity-verification.json"
    if not identity_record_path.is_file():
        raise ValueError("authenticated oracle identity record missing")
    oracle_identity_record = json.loads(identity_record_path.read_text())
    expected_oracle, oracle_source_hashes = expected_oracle_identities()
    if (
        oracle_identity_record.get("schema")
        != "seed557-authenticated-oracle-identities-v1"
    ):
        raise ValueError("authenticated oracle identity schema mismatch")
    if oracle_identity_record.get("source_commit") != freeze["source_commit"]:
        raise ValueError("oracle identity source commit mismatch")
    if (
        oracle_identity_record.get("authenticated_source_sha256")
        != oracle_source_hashes
    ):
        raise ValueError("oracle identity source evidence mismatch")
    if oracle_identity_record.get("expected_from_seed556") != expected_oracle:
        raise ValueError("oracle identity differs from authenticated #556")
    frozen_oracle_identities = {
        "native_probe_sha256": freeze["oracle"]["native_probe_sha256"],
        "tablebase_sha256": freeze["oracle"]["tablebase_sha256"],
        "tablebase_size_bytes": expected_oracle["tablebase_size_bytes"],
    }
    if (
        oracle_identity_record.get("oracle_identities_in_frozen_execution")
        != frozen_oracle_identities
    ):
        raise ValueError("frozen oracle identities differ from authenticated #556")
    if frozen_oracle_identities != expected_oracle:
        raise ValueError("frozen oracle identities differ from authenticated #556")
    if (
        freeze["oracle"]["native_probe_sha256"]
        != expected_oracle["native_probe_sha256"]
    ):
        raise ValueError("frozen native probe identity mismatch")
    if freeze["oracle"]["tablebase_sha256"] != expected_oracle["tablebase_sha256"]:
        raise ValueError("frozen tablebase identity mismatch")
    if oracle_identity_record.get("execution_freeze_sha256") != sha256(
        (data / "execution-freeze.json").read_bytes()
    ):
        raise ValueError("oracle identity record is not bound to execution freeze")
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
    selection_ties = _jsonl(data / "puct-selection-tie-verification.jsonl")
    tie_by_key = {(row["state_hash"], row["policy"]): row for row in selection_ties}
    if len(tie_by_key) != len(selection_ties):
        raise ValueError("duplicate PUCT tie-selection evidence")
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
        if int(search["seed"]) != search_seed(search["state_hash"]):
            raise ValueError("search seed does not derive from frozen state hash")
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
        visits = [float(value) for value in search["visits"]]
        if len(visits) != 6 or any(
            value < 0 or not value.is_integer() for value in visits
        ):
            raise ValueError("invalid PUCT root visit vector")
        if any(visits[move] != 0 for move in range(6) if move not in legal):
            raise ValueError("PUCT assigned visits to an illegal action")
        if sum(visits) != int(freeze["search"]["simulations"]):
            raise ValueError("PUCT root visits do not match frozen simulation budget")
        max_visit_moves = [
            move for move in legal if visits[move] == max(visits[m] for m in legal)
        ]
        if search["puct_action"] not in max_visit_moves:
            raise ValueError("PUCT action violates maximum-visit selection rule")
        tie_key = (search["state_hash"], search["policy"])
        if len(max_visit_moves) == 1:
            if tie_key in tie_by_key:
                raise ValueError("unexpected PUCT tie evidence for unique maximum")
        else:
            tie = tie_by_key.get(tie_key)
            if tie is None:
                raise ValueError(
                    "PUCT tied visits lack deterministic tie-break evidence"
                )
            if int(tie["seed"]) != int(search["seed"]):
                raise ValueError("PUCT tie evidence seed mismatch")
            if tie["replay"].get("same_frozen_state") is not True:
                raise ValueError("PUCT tie evidence is not bound to the frozen root")
            if int(tie["replay"].get("simulations", -1)) != int(
                freeze["search"]["simulations"]
            ):
                raise ValueError("PUCT tie evidence simulation budget mismatch")
            if float(tie["replay"].get("c_puct", -1)) != float(
                freeze["search"]["c_puct"]
            ):
                raise ValueError("PUCT tie evidence c_puct mismatch")
            if tie["visits"] != search["visits"]:
                raise ValueError("PUCT tie evidence visit vector mismatch")
            q_values = {
                int(row["move"]): float(row["q_value"]) for row in tie["child_stats"]
            }
            tie_visits = {
                int(row["move"]): int(row["visits"]) for row in tie["child_stats"]
            }
            if set(q_values) != set(legal) or set(tie_visits) != set(legal):
                raise ValueError("PUCT tie evidence legal-child coverage mismatch")
            if any(tie_visits.get(move) != visits[move] for move in max_visit_moves):
                raise ValueError("PUCT tie child visit evidence mismatch")
            expected_tie_action = max(
                max_visit_moves,
                key=lambda move: (
                    visits[move],
                    q_values[move],
                    priors[move],
                    -move,
                ),
            )
            if (
                tie["puct_action"] != search["puct_action"]
                or search["puct_action"] != expected_tie_action
            ):
                raise ValueError(
                    "PUCT action violates deterministic visit/Q/prior/index tie-break"
                )
    expected_search_keys = {
        (state["state_hash"], policy)
        for state in frozen_states
        for policy in ("T512", "C128", "P")
    }
    if seen != expected_search_keys or oracle_hashes != set(by_hash):
        raise ValueError("per-root policy or oracle coverage is incomplete")
    if set(tie_by_key) != {
        (row["state_hash"], row["policy"])
        for row in searches
        if len(
            [
                move
                for move in row["legal_moves"]
                if row["visits"][move]
                == max(row["visits"][legal] for legal in row["legal_moves"])
            ]
        )
        > 1
    }:
        raise ValueError("PUCT tie-break evidence coverage mismatch")
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
