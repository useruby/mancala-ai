"""Publish bounded seed461 artifact recovery without inferring run identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed485_forensics import ARCHIVE, REPORT, build_inventory

LEDGER = "docs/data/seed486-seed461-recovery-ledger.json"
RUN_DIRS = (
    ".tmp/seed461-cosine-lr-ablation",
    ".tmp/seed461-lr-sensitivity",
    ".tmp/seed461-order-confirmation",
    ".tmp/seed461-order-exploratory",
    ".tmp/seed461-e2-e4-average",
    ".tmp/seed461-e3-e4",
    ".tmp/seed461-cross-order-e4-average",
    ".tmp/seed461-e1-e4-corrected-diagnostic",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_ledger(root: Path, *, include_local: bool = True) -> dict[str, Any]:
    """Link receipts to locally preserved bytes and state unresolved contracts."""
    root = root.resolve()
    inventory = build_inventory(root)
    with tarfile.open(root / ARCHIVE, "r:gz") as archive:
        members = {
            member.name: member for member in archive.getmembers() if member.isfile()
        }
        records = []
        for stream in inventory["streams"]:
            game_path = Path(stream["games_path"])
            if not game_path.is_absolute():
                game_path = root / game_path
            run_artifacts = []
            if include_local and game_path.is_file():
                raw = game_path.read_bytes()
                if _sha(game_path) != stream["games_sha256"]:
                    raise ValueError(
                        f"local_game_receipt_mismatch:{stream['games_path']}"
                    )
                run_artifacts.append(
                    {
                        "role": "game_log",
                        "path": str(game_path),
                        "sha256": _sha(game_path),
                    }
                )
                report = game_path.with_name(
                    game_path.name.removesuffix("-games.jsonl") + ".json"
                )
                if report.is_file():
                    run_artifacts.append(
                        {
                            "role": "arena_report",
                            "path": str(report),
                            "sha256": _sha(report),
                        }
                    )
                rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
                explicit_state_rows = sum(
                    any(
                        key in row
                        for key in ("states", "state_trace", "explicit_states")
                    )
                    for row in rows
                )
            else:
                explicit_state_rows = 0
            member = members.get(stream["games_path"].split(".tmp/", 1)[-1])
            archive_hash = None
            if member is not None:
                content = archive.extractfile(member)
                if content is not None:
                    archive_hash = hashlib.sha256(content.read()).hexdigest()
            records.append(
                {
                    "stream_key": stream["run_key"],
                    "receipt": {
                        "binding_path": stream["binding_path"],
                        "binding_sha256": stream["binding_sha256"],
                        "game_log_sha256": stream["games_sha256"],
                        "archive_member_sha256": archive_hash,
                    },
                    "local_artifacts": run_artifacts,
                    "registration_path": stream["registration_path"],
                    "registration_sha256": stream["registration_sha256"],
                    "suite_path": stream["suite_path"]
                    if stream["suite_bytes_match"]
                    else None,
                    "suite_sha256": stream["suite_sha256_registered"],
                    "explicit_state_trace_rows": explicit_state_rows,
                    "execution_command": "runner source is available; no run-bound command log/argv receipt located",
                    "working_revision": None,
                    "arena_rules_source_binding": None,
                    "resume_chunk_mapping": None,
                    "unresolved": [
                        "run-bound working revision or execution-source hashes",
                        "run-bound arena writer and rules source hashes",
                        "explicit state traces (game rows contain trajectory strings only)",
                        "resume/chunk mapping beyond recorded game/opening indices",
                    ]
                    + (
                        ["registration and suite receipt"]
                        if not stream["registration_bound"]
                        else []
                    ),
                }
            )

    searched = []
    for relative in RUN_DIRS:
        path = root / relative
        present = path.is_dir()
        searched.append(
            {
                "location": relative,
                "status": "searched; present"
                if present
                else "not found in this checkout",
                "note": "Only receipt-linked run artifacts are associated per stream; no inference from neighboring artifacts.",
            }
        )
    searched.extend(
        [
            {
                "location": "docs/data/*.json and docs/data/*seed461*",
                "status": "searched",
                "note": "Committed bindings, registrations, suites, source snapshot references, and manifests; see per-stream links.",
            },
            {
                "location": f"{ARCHIVE} (sha256 {_sha(root / ARCHIVE)})",
                "status": "searched",
                "note": "58 receipt-matched game logs; no explicit state trace fields found.",
            },
            {
                "location": "reachable Git history: seed461 runner/arena source paths and PR #390 commit ec87dc4a77dcd9d31ae0e0dcba08c3bf4f20de0a",
                "status": "searched",
                "note": "Source implementations are comparisons; no execution receipt links a source revision to these runs.",
            },
        ]
    )
    return {
        "schema": "seed486-seed461-recovery-ledger-v1",
        "status": "bounded_recovery_incomplete",
        "basis_inventory": REPORT,
        "stream_count": len(records),
        "streams": records,
        "searched_locations": searched,
        "uncovered_population": "1,129,643 recorded trajectory action pre-states; no unique identity count asserted",
        "required_artifact_request": [
            "For each stream, provide the execution-time arena.py and rules source files or their SHA-256 hashes, with a run-bound source snapshot manifest or command/runtime receipt linking them to the receipt-bound game log.",
            "Provide explicit state traces tied to the same game-log receipt if trajectory-free identity reconstruction is expected.",
            "For fixed-E4 streams, provide the missing registration and opening-suite receipt/bytes.",
            "Provide command logs and resume/chunk manifests tying run, suite index, seat initialization, and per-game ranges together.",
        ],
        "generation_authorized": False,
    }


def verify_ledger(root: Path, ledger_path: Path) -> dict[str, Any]:
    saved = json.loads(ledger_path.read_text())
    rebuilt = build_ledger(root, include_local=False)
    # Original run paths are historical discoveries, not dependencies of a
    # copied checkout. Compare portable receipt and archive evidence only.
    saved_portable = json.loads(json.dumps(saved))
    for saved_stream, rebuilt_stream in zip(
        saved_portable.get("streams", []), rebuilt["streams"], strict=True
    ):
        local_artifacts = saved_stream.pop("local_artifacts", [])
        if not all(
            artifact.get("sha256") == saved_stream["receipt"]["game_log_sha256"]
            for artifact in local_artifacts
            if artifact.get("role") == "game_log"
        ):
            raise ValueError("seed486_local_artifact_receipt_mismatch")
        rebuilt_stream.pop("local_artifacts", None)
    # Presence of the untracked original run directories is checkout-specific;
    # their saved status documents where they were found, while portability
    # verification compares the searched location set and scope.
    for location in saved_portable["searched_locations"]:
        location.pop("status", None)
    for location in rebuilt["searched_locations"]:
        location.pop("status", None)
    if saved_portable != rebuilt:
        raise ValueError("seed486_recovery_ledger_reconstruction_mismatch")
    return {
        "status": saved["status"],
        "stream_count": saved["stream_count"],
        "generation_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify_ledger(args.root, args.verify)
    else:
        ledger = build_ledger(args.root)
        if args.write:
            args.write.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        result = {
            key: ledger[key]
            for key in ("status", "stream_count", "generation_authorized")
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
