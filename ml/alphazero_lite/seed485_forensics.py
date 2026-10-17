"""Read-only, receipt-bound inventory of historical seed461 arena streams.

This module records the evidence available in the checkout without assigning an
execution contract to a stream. In particular, PR #390 source is not treated as
the writer unless a run-specific source binding is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

ARCHIVE = "docs/data/seed483-recovered-seed461-arena-games.tar.gz"
REPORT = "docs/data/seed485-seed461-forensic-inventory.json"
EXPECTED_STREAMS = 58


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_files(root: Path) -> list[Path]:
    return sorted((root / "docs/data").rglob("*.json"))


def build_inventory(
    root: Path, *, expected_streams: int = EXPECTED_STREAMS
) -> dict[str, Any]:
    """Authenticate archive rows against evaluation bindings and registrations."""
    root = root.resolve()
    files = _json_files(root)
    documents = {
        p.relative_to(root).as_posix(): json.loads(p.read_text()) for p in files
    }
    registration_by_hash = {
        sha256(path.read_bytes()): (relative, documents[relative])
        for relative, path in ((p.relative_to(root).as_posix(), p) for p in files)
        if "registration" in path.name
    }
    expected: dict[str, dict[str, Any]] = {}
    for relative, document in documents.items():
        if not isinstance(document, dict):
            continue
        reg_hash = document.get("registration_sha256")
        registration = registration_by_hash.get(reg_hash)
        reports = document.get("reports", {})
        for run_key, report in reports.items():
            game_path = report.get("games")
            game_hash = report.get("games_sha256")
            if (
                not isinstance(game_path, str)
                or not isinstance(game_hash, str)
                or "seed461" not in game_path
            ):
                continue
            archive_member = game_path.split(".tmp/", 1)[-1]
            if (
                archive_member in expected
                and expected[archive_member]["games_sha256"] != game_hash
            ):
                raise ValueError(f"conflicting_game_receipts:{archive_member}")
            expected[archive_member] = {
                "binding_path": relative,
                "binding_sha256": sha256((root / relative).read_bytes()),
                "registration_sha256": reg_hash,
                "registration_path": registration[0] if registration else None,
                "registration_bound": registration is not None,
                "run_key": run_key,
                "games_path": game_path,
                "games_sha256": game_hash,
                "report_sha256": report.get("report_sha256"),
                "suite_sha256": document.get("suite_sha256"),
                "candidate": document.get("candidates", {}).get(run_key),
            }

        # Some historical bindings place game receipts outside a `reports` map.
        def visit(value: Any, context: str = "") -> None:
            if isinstance(value, dict):
                game_path, game_hash = value.get("games"), value.get("games_sha256")
                if (
                    isinstance(game_path, str)
                    and isinstance(game_hash, str)
                    and ".tmp/" in game_path
                    and "seed461" in game_path
                ):
                    member = game_path.split(".tmp/", 1)[1]
                    if (
                        member in expected
                        and expected[member]["games_sha256"] != game_hash
                    ):
                        raise ValueError(f"conflicting_game_receipts:{member}")
                    expected.setdefault(
                        member,
                        {
                            "binding_path": relative,
                            "binding_sha256": sha256((root / relative).read_bytes()),
                            "registration_sha256": reg_hash,
                            "registration_path": registration[0]
                            if registration
                            else None,
                            "registration_bound": registration is not None,
                            "run_key": context
                            or Path(game_path).stem.removesuffix("-games"),
                            "games_path": game_path,
                            "games_sha256": game_hash,
                            "report_sha256": value.get("report_sha256"),
                            "suite_sha256": document.get("suite_sha256"),
                            "candidate": None,
                        },
                    )
                for key, child in value.items():
                    visit(child, str(key))
            elif isinstance(value, list):
                for child in value:
                    visit(child, context)

        visit(document)
    streams = []
    with tarfile.open(root / ARCHIVE, "r:gz") as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        for member_name, binding in sorted(expected.items()):
            member = members.get(member_name)
            if member is None:
                raise ValueError(f"archive_member_missing:{member_name}")
            raw = archive.extractfile(member).read()  # type: ignore[union-attr]
            actual_hash = sha256(raw)
            if actual_hash != binding["games_sha256"]:
                raise ValueError(f"game_log_hash_mismatch:{member_name}")
            rows = [json.loads(line) for line in raw.splitlines() if line]
            for row_number, row in enumerate(rows):
                row["_row_number"] = row_number
            # Bind suite bytes through the registered opening-suite hash where available.
            registration = (
                documents.get(binding["registration_path"], {})
                if binding["registration_path"]
                else {}
            )
            suite = registration.get(
                "suite", registration.get("evaluation", {}).get("suite", {})
            )
            suite_path = suite.get("path") if isinstance(suite, dict) else None
            suite_file_hash = suite.get("sha256") if isinstance(suite, dict) else None
            if not suite_file_hash:
                suite_file_hash = binding.get("suite_sha256")
            if suite_file_hash:
                for opening_path in sorted(
                    (root / "docs/data").glob("seed461*openings*.jsonl")
                ):
                    if sha256(opening_path.read_bytes()) == suite_file_hash:
                        suite_path = opening_path.relative_to(root).as_posix()
                        break
            suite_bytes_match = False
            suite_rows: list[dict[str, Any]] = []
            if suite_path and suite_file_hash:
                suite_local = root / "docs/data" / Path(suite_path).name
                if suite_local.is_file():
                    suite_raw = suite_local.read_bytes()
                    suite_bytes_match = sha256(suite_raw) == suite_file_hash
                    if suite_bytes_match:
                        suite_rows = [
                            json.loads(line) for line in suite_raw.splitlines() if line
                        ]
            # Candidate artifact is resolved by run key in the bound evaluation file.
            candidate_name = binding["run_key"]
            bound_candidate = (
                documents[binding["binding_path"]]
                .get("candidates", {})
                .get(candidate_name)
            )
            candidate_arm = (
                bound_candidate.get("arm") or candidate_name[-1:]
                if isinstance(bound_candidate, dict)
                else candidate_name[-1:]
            )
            streams.append(
                {
                    **binding,
                    "candidate": bound_candidate,
                    "candidate_arm": candidate_arm,
                    "seat_pairing_fields": sorted(
                        {
                            key
                            for row in rows
                            for key in row
                            if "player" in key or "seat" in key
                        }
                    ),
                    "challenger_player_values": sorted(
                        {
                            row.get("challenger_player")
                            for row in rows
                            if "challenger_player" in row
                        }
                    ),
                    "game_within_opening_pairing_counts": {
                        str(value): sum(
                            row.get("game_within_opening") == value for row in rows
                        )
                        for value in sorted(
                            {
                                row.get("game_within_opening")
                                for row in rows
                                if isinstance(row.get("game_within_opening"), int)
                            }
                        )
                    },
                    "game_count": len(rows),
                    "game_indices": [row.get("game_index") for row in rows],
                    "game_index_start": rows[0].get("game_index") if rows else None,
                    "game_index_end": rows[-1].get("game_index") if rows else None,
                    "game_indices_contiguous_in_log_order": all(
                        rows[index].get("game_index")
                        == rows[0].get("game_index") + index
                        for index in range(len(rows))
                    )
                    if rows and isinstance(rows[0].get("game_index"), int)
                    else False,
                    "chunk_or_resume_metadata": "not present in receipt-bound game rows or evaluation binding",
                    "opening_indices": [row.get("opening_index") for row in rows],
                    "game_within_opening_values": sorted(
                        {row.get("game_within_opening") for row in rows}
                    ),
                    "embedded_prefix_rows": sum(
                        "opening_prefix_moves" in row for row in rows
                    ),
                    "embedded_root_rows": sum(
                        "opening_state" in row or "initial_state" in row for row in rows
                    ),
                    "trajectory_encoding": "present; writer/action semantics not authenticated for this run",
                    "suite_path": suite_path,
                    "suite_sha256_registered": suite_file_hash,
                    "suite_bytes_match": suite_bytes_match,
                    "registered_suite_row_count": len(suite_rows),
                    "first_game_registered_suite_index": (
                        rows[0].get("opening_index") if rows else None
                    ),
                    "first_game_registered_root": (
                        suite_rows[rows[0]["opening_index"]].get("state")
                        if rows
                        and suite_rows
                        and isinstance(rows[0].get("opening_index"), int)
                        and rows[0]["opening_index"] < len(suite_rows)
                        else None
                    ),
                    "first_game_embedded_root": (
                        rows[0].get("opening_state", rows[0].get("initial_state"))
                        if rows
                        else None
                    ),
                    "first_game_embedded_prefix": (
                        rows[0].get("opening_prefix_moves") if rows else None
                    ),
                    "writer_source": None,
                    "writer_source_sha256": None,
                    "rules_source": None,
                    "rules_source_sha256": None,
                    "execution_source_snapshot_references": [],
                    "run_command_reference": None,
                    "execution_manifest_reference": None,
                    "writer_version_evidence": "No run-bound source snapshot/hash, command receipt, or execution manifest found in committed seed461 JSON evidence or reachable arena.py history.",
                    "historical_writer_comparison_only": {
                        "commit": "ec87dc4a77dcd9d31ae0e0dcba08c3bf4f20de0a",
                        "path": "ml/alphazero_lite/arena.py",
                        "sha256": "85971b30fa8579da7eac1fb6896d8165c6a8c8f7d4c00e878c34e8f848769554",
                        "bound_to_run": False,
                    },
                    "writer_action_semantics": "unknown",
                    "rules_semantics": "unknown",
                    "first_game": rows[0] if rows else None,
                }
            )
    if len(streams) != expected_streams:
        raise ValueError(f"seed461_stream_count_mismatch:{len(streams)}")
    failing = next(
        (
            row
            for row in streams
            if row["games_path"].endswith("order_38611_A-games.jsonl")
            and row["first_game"]
            and row["first_game"].get("game_index") == 0
        ),
        None,
    )
    first_failure = None
    if failing:
        game = failing["first_game"]
        registered = failing["first_game_registered_root"]
        legal = (
            [
                index
                for index, stones in enumerate(registered["player_pits"])
                if stones > 0
            ]
            if registered and registered.get("current_player") == 0
            else None
        )
        first_failure = {
            "source_path": failing["games_path"],
            "source_sha256": failing["games_sha256"],
            "row_number_zero_based": game["_row_number"],
            "game_index": game.get("game_index"),
            "selected_suite_index": game.get("opening_index"),
            "registered_root": registered,
            "embedded_root": failing["first_game_embedded_root"],
            "embedded_opening_prefix": game.get("opening_prefix_moves"),
            "player_to_move_in_registered_root": registered.get("current_player")
            if registered
            else None,
            "legal_absolute_pits_under_registered_orientation": legal,
            "first_trajectory_value": int(
                str(game.get("trajectory", "")).split(",")[0]
            ),
            "comparison_contract": {
                "source": "ml/alphazero_lite/arena.py",
                "git_commit": "ec87dc4a77dcd9d31ae0e0dcba08c3bf4f20de0a",
                "source_sha256": "85971b30fa8579da7eac1fb6896d8165c6a8c8f7d4c00e878c34e8f848769554",
                "documented_behavior": "applies opening prefix before trajectory and writes game.pit_index(move) as an absolute pit index",
                "binding_to_this_run": "not established; comparison only",
            },
            "interpretation": "Conditional contradiction under the documented #390 contract; execution semantics for this stream remain unknown.",
        }
    return {
        "schema": "seed485-seed461-forensic-inventory-v1",
        "status": "incomplete_writer_and_rules_bindings",
        "stream_count": len(streams),
        "receipt_matched_stream_count": len(streams),
        "registration_bound_stream_count": sum(
            bool(row["registration_bound"]) for row in streams
        ),
        "writer_contract_stream_count": sum(
            row["writer_source_sha256"] is not None for row in streams
        ),
        "streams": streams,
        "first_failure_packet": first_failure,
        "conclusion": "Game logs are receipt-matched. Available committed bindings do not authenticate per-run writer/rules source bytes; action/root interpretation remains unknown and no continuation replay is certified.",
        "generation_authorized": False,
    }


def verify_inventory(root: Path, inventory_path: Path) -> dict[str, Any]:
    saved = json.loads(inventory_path.read_text())
    rebuilt = build_inventory(root)
    if saved != rebuilt:
        raise ValueError("seed485_inventory_reconstruction_mismatch")
    return {
        "status": saved["status"],
        "stream_count": saved["stream_count"],
        "writer_contract_stream_count": saved["writer_contract_stream_count"],
        "generation_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify_inventory(args.root, args.verify)
    else:
        inventory = build_inventory(args.root)
        if args.write:
            args.write.write_text(
                json.dumps(inventory, indent=2, sort_keys=True) + "\n"
            )
        result = {
            key: inventory[key]
            for key in (
                "status",
                "stream_count",
                "receipt_matched_stream_count",
                "registration_bound_stream_count",
                "writer_contract_stream_count",
                "generation_authorized",
            )
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
