"""Validate explicit PR-to-experiment provenance without conflating identifiers."""

from __future__ import annotations

import argparse
import hashlib
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed482_fresh_confirmation import (
    canonical_identity,
    float32_identity,
    sha_bytes,
    state_from_encoded,
)
from ml.alphazero_lite.seed483_exclusion_closure import (
    verify_coverage_document,
    verify_union_file,
)

PROVENANCE_PATH = "docs/data/seed484-pr-experiment-provenance.json"
EXPECTED_MAPPINGS = {
    458: 457,
    459: 457,
    460: 457,
    461: 457,
    463: 462,
    465: 464,
    467: 466,
}
EXPECTED_PR_COMMITS = {
    458: "e68242eb0d35ce7ab8c86f7eedc3c85942958306",
    459: "aad1f1dcbba209e5529994510b98b89d50618213",
    460: "326570b23c2d8235b518af77fad13b0f773b02d1",
    461: "34c657d627592e344a4d399d4365de1fdad8c75d",
    463: "f91521542196fb469b376c7552e555df23d68f1e",
    465: "e1f1dcdc3ce22b4c2158a6793bc81e55157e7c84",
    467: "3be06c6350ee6e4a0ee52364d0a3e88ca2cd2c1b",
}
EXPECTED_PR_ACTIVITY = {
    458: ("preparation_correction", "registration_only_no_probe_launch"),
    459: ("verification_correction", "reuses_seed457_registration_and_population"),
    460: (
        "original_execution_invalid_verification",
        "consumed_registered_seed457_population",
    ),
    461: ("post_execution_correction_replay", "reuses_original_seed457_population"),
    463: ("post_execution_correction", "replays_existing_seed462_population"),
    465: (
        "audit_packet_publication",
        "packages_existing_seed464_evidence_no_new_probe_population",
    ),
    467: (
        "runtime_reader_authentication",
        "read_only_bundle_access_no_new_probe_population",
    ),
}
EXPECTED_PR_ACTIVITY_EVIDENCE = {
    458: "PR description: No real registration was created and no real C/T arm was executed. No searches or games were run and no artifacts were exported.",
    459: "PR description: No real registration, C/T arm execution, searches, games, export, or experiment classification was performed. This PR only corrects verification wiring.",
    460: "PR description: Both arms completed exactly sixteen proposals. The attempt is marked invalid and reports no completed experiment decision.",
    461: "PR description: Reconstructs the original 16-step arms and 256 trials from the registered inputs; no new training arm, games, or experiment was run.",
    463: "PR description: The correction is read-only; no new training, searches, games, export, or promotion.",
    465: "PR description: No training or rerun was performed.",
    467: "PR description: No training or new experiment.",
}
EXPECTED_ORIGIN_PRS = {
    461: [384, 385, 386, 387, 388, 389, 390, 396],
    467: [377, 378],
}
EXPECTED_APPARENT_GAPS = {
    458: "unregistered_not_classified_empty",
    459: "unregistered_not_classified_empty",
    460: "unregistered_not_classified_empty",
    461: "distinct_registered_lineage_continuation_unresolved",
    463: "unregistered_not_classified_empty",
    465: "unregistered_not_classified_empty",
    467: "distinct_registered_lineage_recovered",
}
EXPECTED_GAP_PR_SEEDS = {
    458: 457,
    459: 457,
    460: 457,
    461: 457,
    463: 462,
    465: 464,
    467: 466,
}
EXPECTED_BOUND_RECEIPTS = {
    "docs/data/seed457-fresh-subgroup-projection/registration.json": "58034469262878a7de99658b9b09c71d59369a1960c69d2e20f83b5f31c5a8a8",
    "docs/data/seed457-fresh-subgroup-projection/receipt.json": "32c968cc5bb17d3fc8e749016d586a969d0ed92427cf562dbb5546e74e31c6ef",
    "docs/data/seed457-fresh-subgroup-projection/verification-status.json": "96b0a6d9b25158d534de8bff5a0c47bc19d98b9af49c0468314429a337c7981c",
    "docs/data/seed457-occurrence-verification-correction/receipt.json": "bac3c8c2c760ab8345c527d0f2df5241aff6180f5d9de8b7fb9bcc35f7680f3f",
    "docs/data/seed457-occurrence-verification-correction/reconciliation.json": "5dc5623d247d3005d6d2d0cf26d1bc7879ef5f9e708e97606f685b35d80e6c93",
    "docs/data/seed461-e3-e4-registration.json": "5e888028b8b79549ddbfd73924623d63e8256fe147e804701911b476701d176f",
    "docs/data/seed461-cross-order-e4-average-registration.json": "5c09af5fef62dec497571fe5d0b6956825e9f3d043f43efe0529c57f4efcadf1",
    "docs/data/seed461-lr-sensitivity-registration-v2.json": "b7b0e6011924cd47154cb26f78bc857dadea04247fb55e17c8810a66dd1ec0ab",
    "docs/data/seed483-recovered-seed461-arena-games.tar.gz": "17703bba162596db668c8dff69fc91e6297669c9c0bf70ec2b75de2f06bd1d05",
    "docs/data/alphazero-lite-generation-comparisons/seed467-parent-anchor-w010-vs-unanchored-config-diff.json": "59fd1b0a6527f40bc1dd059d9941f266607227e98a9db38ae46a47a57a36e0c6",
    "docs/data/seed462-historical-policy-removal/registration.json": "6d5f7e7fde229ee51de74c58a179bf0fcd4c0293c44810b89558d204c9f84f3e",
    "docs/data/seed462-historical-policy-removal-correction/correction-receipt.json": "293adc2a256e0f1c4a6749241d9ba7a7f586326b9fab11319dd051cf5b2bfdbb",
    "docs/data/seed464-dependency-closure-audit/packet/packet-manifest.json": "d4ea022be45e6e6e259d163b8ef2deec1ac06761dec09f6f6ce4463fa507ec03",
    "docs/data/seed466-replay-input-bundle/manifest.json": "db80f3bdfcc0a126032c1f5a16e78cda74b160d4fa2d80330cec8bed18085b34",
    "docs/data/alphazero-lite-generations/seed455-nextgen-s467-parent-anchor-w010/generation.json": "3c989868860beed12b9b465f691e4d31466afc75594bfcba6a87d3b96c719f3d",
}
SEED467_GENERATION = (
    "docs/data/alphazero-lite-generations/"
    "seed455-nextgen-s467-parent-anchor-w010/generation.json"
)
SEED467_SOURCE = (
    ".tmp/seed455-nextgen-s467-parent-anchor-w010/runs/"
    "seed455-nextgen-s467-parent-anchor-w010-iter1/self_play.jsonl"
)
SEED467_SHA256 = "6f7038dccf282fb9365a031a06f8c5a8af4a2b8a6eda32725fe125b06d654a76"
SEED467_SIZE = 749_547_469
SEED467_ROW_COUNT = 66_448
SEED467_DISTINCT_PAIRS = 46_276
SEED467_GENERATION_SHA256 = (
    "3c989868860beed12b9b465f691e4d31466afc75594bfcba6a87d3b96c719f3d"
)
SEED467_SNAPSHOT_SHA256 = (
    "cfd3e83038b8f93978e449df81f38a39009e79370d3b97983a68f368a84ef16b"
)


def load_provenance(root: Path) -> dict[str, Any]:
    """Load the checked-in PR graph and require its declared identifiers."""
    document = json.loads((root / PROVENANCE_PATH).read_text(encoding="utf-8"))
    rows = document.get("pr_mappings")
    if not isinstance(rows, list):
        raise ValueError("provenance_pr_mappings_missing")
    seen: dict[int, int] = {}
    for row in rows:
        pr = row.get("pull_request")
        seed = row.get("experiment_seed")
        if pr in seen:
            raise ValueError("provenance_duplicate_pull_request")
        seen[pr] = seed
        if pr not in EXPECTED_MAPPINGS or EXPECTED_MAPPINGS[pr] != seed:
            raise ValueError("provenance_pr_seed_mapping_mismatch")
        if (
            row.get("merge_commit") != EXPECTED_PR_COMMITS.get(pr)
            or row.get("pull_request_url")
            != f"https://github.com/useruby/mancala-ai/pull/{pr}"
        ):
            raise ValueError("provenance_pr_commit_binding_mismatch")
        if (row.get("change_class"), row.get("population_action")) != (
            EXPECTED_PR_ACTIVITY[pr]
        ):
            raise ValueError("provenance_pr_population_claim_mismatch")
        if row.get("activity_evidence") != EXPECTED_PR_ACTIVITY_EVIDENCE[pr]:
            raise ValueError("provenance_pr_activity_evidence_mismatch")
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("provenance_evidence_missing")
        for relative in evidence:
            if not isinstance(relative, str) or not (root / relative).is_file():
                raise ValueError(f"provenance_evidence_unavailable:{relative}")
    if seen != EXPECTED_MAPPINGS:
        raise ValueError("provenance_pr_inventory_mismatch")
    bound_receipts = document.get("bound_receipts")
    if bound_receipts != EXPECTED_BOUND_RECEIPTS:
        raise ValueError("provenance_receipt_inventory_mismatch")
    for relative, expected in bound_receipts.items():
        path = root / relative
        if not path.is_file() or sha_bytes(path.read_bytes()) != expected:
            raise ValueError(f"provenance_receipt_hash_mismatch:{relative}")
    lineage_rows = document.get("recovered_populations", [])
    recovered = [row for row in lineage_rows if row.get("experiment_seed") == 467]
    if len(recovered) != 1:
        raise ValueError("provenance_seed467_recovery_missing")
    seed467 = recovered[0]
    if (
        seed467.get("source_sha256") != SEED467_SHA256
        or seed467.get("receipt_sha256") != SEED467_GENERATION_SHA256
        or seed467.get("source_size_bytes") != SEED467_SIZE
        or seed467.get("occurrences") != SEED467_ROW_COUNT
        or seed467.get("distinct_identity_pairs") != SEED467_DISTINCT_PAIRS
        or seed467.get("portable_snapshot_sha256") != SEED467_SNAPSHOT_SHA256
    ):
        raise ValueError("provenance_seed467_recovery_binding_mismatch")
    lineages = document.get("separate_historical_lineages", [])
    observed_lineages = {}
    for row in lineages:
        seed = row.get("experiment_seed")
        if seed in observed_lineages:
            raise ValueError("provenance_duplicate_historical_lineage")
        observed_lineages[seed] = row.get("origin_pull_requests")
        for relative in row.get("required_because", []):
            if not (root / relative).is_file():
                raise ValueError(f"provenance_lineage_source_unavailable:{relative}")
    if observed_lineages != EXPECTED_ORIGIN_PRS:
        raise ValueError("provenance_historical_lineage_mismatch")
    expected_lineage_status = {
        461: "opening_roots_reconstructed_arena_continuations_replay_unresolved",
        467: "receipt_bound_self_play_recovered_and_identity_reconstructed",
    }
    if {
        row.get("experiment_seed"): row.get("population_status") for row in lineages
    } != expected_lineage_status:
        raise ValueError("provenance_historical_population_status_mismatch")
    gaps = document.get("apparent_seed_number_gaps", [])
    observed_gaps = {row.get("number"): row.get("status") for row in gaps}
    if observed_gaps != EXPECTED_APPARENT_GAPS or len(gaps) != len(observed_gaps):
        raise ValueError("provenance_apparent_gap_classification_mismatch")
    if any(
        row.get("pull_request_maps_to") != EXPECTED_GAP_PR_SEEDS.get(row["number"])
        for row in gaps
    ):
        raise ValueError("provenance_apparent_gap_mapping_mismatch")
    unresolved = [
        row
        for row in document.get("unresolved_populations", [])
        if row.get("experiment_seed") == 461
    ]
    if (
        len(unresolved) != 1
        or unresolved[0].get("uncovered_population")
        != "1,129,643 trajectory action-pre-states; opening roots are already reconstructed"
        or unresolved[0].get("writer_source_sha256")
        != "85971b30fa8579da7eac1fb6896d8165c6a8c8f7d4c00e878c34e8f848769554"
    ):
        raise ValueError("provenance_seed461_unresolved_population_mismatch")
    snapshot = root / "docs/data/seed484-seed467-recovered-identities.json.gz"
    if not snapshot.is_file():
        raise FileNotFoundError("seed467_portable_snapshot_missing")
    verify_identity_snapshot(snapshot)
    if document.get("generation_authorized") is not False:
        raise ValueError("provenance_generation_must_remain_blocked")
    return document


def replay_seed461_game(
    initial_state: dict[str, Any], row: dict[str, Any]
) -> list[dict[str, Any]]:
    """Replay historical absolute pit indices from the verified initial state.

    The seed461 writer stores ``game.pit_index(move)`` per action; this is an
    absolute pit index. No action offset or seat transformation is inferred.
    """
    game = KalahGame.from_state(initial_state)
    states = []
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, str) or not trajectory:
        raise ValueError("seed461_trajectory_missing")
    for ply, token in enumerate(trajectory.split(",")):
        if game.over():
            raise ValueError("seed461_trajectory_after_terminal")
        try:
            action = int(token)
        except ValueError as error:
            raise ValueError("seed461_action_invalid") from error
        if not 0 <= action < 12:
            raise ValueError("seed461_absolute_action_out_of_range")
        states.append(game.to_state())
        if not game.move(action):
            raise ValueError(f"seed461_absolute_action_illegal:{ply}:{action}")
    if row.get("game_length", len(states)) != len(states):
        raise ValueError("seed461_recorded_game_length_mismatch")
    if not game.over():
        raise ValueError("seed461_trajectory_not_terminal")
    challenger = row.get("challenger_player")
    if challenger not in (0, 1):
        raise ValueError("seed461_challenger_seat_invalid")
    margin = game.captured_seeds[challenger] - game.captured_seeds[1 - challenger]
    if row.get("margin") != margin:
        raise ValueError("seed461_recorded_margin_mismatch")
    winner = "draw" if margin == 0 else ("challenger" if margin > 0 else "current")
    if row.get("winner") != winner:
        raise ValueError("seed461_recorded_winner_mismatch")
    return states


def replay_seed461_opening(prefix: list[int]) -> dict[str, Any]:
    """Replay registered seed461 opening prefixes as absolute pit indices."""
    game = KalahGame([4] * 12, [0, 0], 0)
    for ply, action in enumerate(prefix):
        if not 0 <= int(action) < 12 or not game.move(int(action)):
            raise ValueError(f"seed461_opening_absolute_action_illegal:{ply}:{action}")
    return game.to_state()


def seed467_source_receipt(root: Path) -> dict[str, Any]:
    """Authenticate recovered self-play bytes against the committed receipt."""
    generation_path = root / SEED467_GENERATION
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    matches = [
        item
        for item in generation.get("artifacts", [])
        if item.get("path") == SEED467_SOURCE and item.get("role") == "self_play"
    ]
    if not matches:
        raise ValueError("seed467_source_receipt_missing")
    if len(matches) != 1:
        raise ValueError("seed467_source_receipt_duplicate")
    item = matches[0]
    if item.get("sha256") != SEED467_SHA256 or item.get("size_bytes") != SEED467_SIZE:
        raise ValueError("seed467_expected_source_identity_mismatch")
    source = root / SEED467_SOURCE
    if not source.is_file():
        raise FileNotFoundError(f"seed467_source_unavailable:{SEED467_SOURCE}")
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    if digest.hexdigest() != SEED467_SHA256 or size != SEED467_SIZE:
        raise ValueError("seed467_recovered_source_identity_mismatch")
    return {
        "path": SEED467_SOURCE,
        "sha256": SEED467_SHA256,
        "size_bytes": SEED467_SIZE,
        "receipt_path": SEED467_GENERATION,
        "receipt_sha256": sha_bytes(generation_path.read_bytes()),
    }


def reconstruct_seed467(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recover the consumed seed467 state population without search or replay."""
    binding = seed467_source_receipt(root)
    records: list[dict[str, Any]] = []
    with (root / binding["path"]).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            if not line.strip():
                raise ValueError(f"seed467_blank_source_row:{line_number}")
            row = json.loads(line)
            state = state_from_encoded(row["state"])
            encoded_input_sha256 = sha_bytes(
                np.asarray(row["state"], dtype="<f4").tobytes()
            )
            if encoded_input_sha256 != float32_identity(state):
                raise ValueError(f"seed467_state_encoding_mismatch:{line_number}")
            game_index = int(row["game_index"])
            move_index = int(row["move_index"])
            if game_index < 0 or move_index < 0:
                raise ValueError(f"seed467_source_index_invalid:{line_number}")
            records.append(
                {
                    "state": state,
                    "canonical_identity": canonical_identity(state),
                    "float32_identity": float32_identity(state),
                    "provenance": {
                        "experiment": 467,
                        "source": binding["path"],
                        "line": line_number,
                        "game_index": game_index,
                        "move_index": move_index,
                    },
                }
            )
    binding["row_count"] = len(records)
    if len(records) != SEED467_ROW_COUNT:
        raise ValueError("seed467_source_row_count_mismatch")
    return records, binding


def identity_snapshot(records: list[dict[str, Any]], binding: dict[str, Any]) -> bytes:
    """Serialize state identities and source binding deterministically."""
    payload = {
        "schema": "seed484-seed467-recovered-identities-v1",
        "source_binding": binding,
        "records": records,
    }
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return gzip.compress(raw, mtime=0)


def verify_identity_snapshot(path: Path) -> dict[str, Any]:
    """Verify a portable recovered snapshot and every identity pair."""
    raw_snapshot = path.read_bytes()
    if sha_bytes(raw_snapshot) != SEED467_SNAPSHOT_SHA256:
        raise ValueError("seed467_snapshot_source_bytes_mismatch")
    payload = json.loads(gzip.decompress(raw_snapshot))
    if payload.get("schema") != "seed484-seed467-recovered-identities-v1":
        raise ValueError("seed467_snapshot_schema_invalid")
    records = payload.get("records")
    binding = payload.get("source_binding", {})
    if not isinstance(records, list) or binding.get("row_count") != len(records):
        raise ValueError("seed467_snapshot_population_count_invalid")
    if len(records) != SEED467_ROW_COUNT:
        raise ValueError("seed467_snapshot_expected_row_count_mismatch")
    if (
        binding.get("sha256") != SEED467_SHA256
        or binding.get("size_bytes") != SEED467_SIZE
    ):
        raise ValueError("seed467_snapshot_source_binding_invalid")
    if binding.get("receipt_sha256") != SEED467_GENERATION_SHA256:
        raise ValueError("seed467_snapshot_receipt_binding_invalid")
    pairs = set()
    for index, record in enumerate(records):
        state = record["state"]
        pair = (canonical_identity(state), float32_identity(state))
        if pair != (record.get("canonical_identity"), record.get("float32_identity")):
            raise ValueError("seed467_snapshot_identity_mismatch")
        provenance = record.get("provenance", {})
        if (
            provenance.get("experiment") != 467
            or provenance.get("source") != binding.get("path")
            or provenance.get("line") != index
            or not isinstance(provenance.get("game_index"), int)
            or not isinstance(provenance.get("move_index"), int)
        ):
            raise ValueError("seed467_snapshot_provenance_mismatch")
        pairs.add(pair)
    if len(pairs) != SEED467_DISTINCT_PAIRS:
        raise ValueError("seed467_snapshot_distinct_pair_count_mismatch")
    return {
        "status": "verified_recovered_snapshot",
        "source_sha256": binding["sha256"],
        "occurrences": len(records),
        "distinct_pairs": len(pairs),
    }


def rebuild_union(root: Path) -> dict[str, Any]:
    """Independently extend the immutable seed483 union with recovered seed467."""
    base_raw = (root / "docs/data/seed482-exclusion-manifest.json.gz").read_bytes()
    previous_raw = (
        root / "docs/data/seed483-historical-probe-identities.json.gz"
    ).read_bytes()
    snapshot_path = root / "docs/data/seed484-seed467-recovered-identities.json.gz"
    snapshot_summary = verify_identity_snapshot(snapshot_path)
    base = json.loads(gzip.decompress(base_raw))
    previous = json.loads(gzip.decompress(previous_raw))
    independently_verified = verify_union_file(
        root / "docs/data/seed483-historical-probe-identities.json.gz", root
    )
    if independently_verified["union_sha256"] != previous.get("union_sha256"):
        raise ValueError("seed483_previous_union_reconstruction_mismatch")
    base_pairs = {
        (row["canonical_identity"], row["float32_identity"])
        for row in base["identities"]
    }
    previous_pairs = set(base_pairs)
    for record in previous["identities"]:
        previous_pairs.add((record["canonical_identity"], record["float32_identity"]))
    coverage = json.loads(
        (root / "docs/data/seed483-source-coverage.json").read_text(encoding="utf-8")
    )
    verify_coverage_document(coverage, root, previous_pairs)
    snapshot = json.loads(gzip.decompress(snapshot_path.read_bytes()))
    all_pairs = set(previous_pairs)
    additions: dict[tuple[str, str], dict[str, Any]] = {}
    for record in snapshot["records"]:
        pair = (record["canonical_identity"], record["float32_identity"])
        all_pairs.add(pair)
        if pair not in previous_pairs:
            additions.setdefault(pair, record)

    def pair_digest(pairs: set[tuple[str, str]]) -> str:
        encoded = json.dumps(
            sorted([list(pair) for pair in pairs]), separators=(",", ":")
        ).encode()
        return sha_bytes(encoded)

    result = {
        "schema": "seed484-extended-probe-union-v1",
        "status": "incomplete_seed461_replay_gap",
        "seed482_manifest_sha256": sha_bytes(base_raw),
        "seed483_additions_sha256": sha_bytes(previous_raw),
        "seed467_snapshot_sha256": sha_bytes(snapshot_path.read_bytes()),
        "seed467_source_sha256": SEED467_SHA256,
        "seed467_occurrences": snapshot_summary["occurrences"],
        "seed467_distinct_pairs": snapshot_summary["distinct_pairs"],
        "seed467_additional_pairs": len(additions),
        "seed467_additional_pairs_sha256": pair_digest(set(additions)),
        "previous_union_pair_count": len(previous_pairs),
        "previous_union_pairs_sha256": pair_digest(previous_pairs),
        "final_union_pair_count": len(all_pairs),
        "final_union_pairs_sha256": pair_digest(all_pairs),
        "generation_authorized": False,
        "seed467_additional_identities": [additions[key] for key in sorted(additions)],
    }
    return result


def verify_union(path: Path, root: Path) -> dict[str, Any]:
    """Rebuild union from authenticated baseline, seed483 additions, and snapshot."""
    saved = json.loads(gzip.decompress(path.read_bytes()))
    rebuilt = rebuild_union(root)
    if saved != rebuilt:
        raise ValueError("seed484_union_reconstruction_mismatch")
    return {
        "status": saved["status"],
        "seed467_additional_pairs": saved["seed467_additional_pairs"],
        "final_union_pair_count": saved["final_union_pair_count"],
        "final_union_pairs_sha256": saved["final_union_pairs_sha256"],
        "generation_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--reconstruct-seed467", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-snapshot", type=Path)
    parser.add_argument("--build-union", action="store_true")
    parser.add_argument("--verify-union", type=Path)
    args = parser.parse_args()
    if args.verify_snapshot:
        print(
            json.dumps(verify_identity_snapshot(args.verify_snapshot), sort_keys=True)
        )
        return
    graph = load_provenance(args.root.resolve())
    if args.verify_union:
        print(
            json.dumps(
                verify_union(args.verify_union, args.root.resolve()), sort_keys=True
            )
        )
        return
    result: dict[str, Any] = {
        "status": graph["status"],
        "pr_mapping_count": len(graph["pr_mappings"]),
        "generation_authorized": False,
    }
    if args.reconstruct_seed467:
        records, binding = reconstruct_seed467(args.root.resolve())
        unique = {
            (row["canonical_identity"], row["float32_identity"]) for row in records
        }
        result["seed467"] = {
            **binding,
            "occurrences": len(records),
            "distinct_pairs": len(unique),
        }
        if args.output:
            args.output.write_bytes(identity_snapshot(records, binding))
    if args.build_union:
        union = rebuild_union(args.root.resolve())
        if args.output:
            encoded = (
                json.dumps(union, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            args.output.write_bytes(gzip.compress(encoded, mtime=0))
        result["union"] = {
            key: value
            for key, value in union.items()
            if key != "seed467_additional_identities"
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
