"""Mutation and portability regressions for archived-screen replay."""

from __future__ import annotations

import hashlib
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from ml.alphazero_lite import archived_screen_identity_inventory as inventory
from ml.alphazero_lite import archived_screen_identity_seven as seven

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_RELATIVE = inventory.ARCHIVE_RELATIVE
AUDIT_RELATIVE = inventory.AUDIT_RELATIVE


def _copy_minimal_publication(source: Path, target: Path) -> None:
    """Copy only hash-bound inputs and the pure-Python public verifier."""
    manifest = json.loads(
        (source / AUDIT_RELATIVE / "seed545-source-manifest.json").read_text()
    )
    archive = source / ARCHIVE_RELATIVE
    relatives = {
        *manifest["bindings"],
        *manifest["recovery_receipts"],
        *manifest["game_batches"],
        *(f"batches/report-{index:04d}.json" for index in range(1024)),
        manifest["prelaunch_failure"]["path"],
    }
    for relative in sorted(relatives):
        origin = archive / relative
        destination = target / ARCHIVE_RELATIVE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)

    for relative in (
        "ml/alphazero_lite/archived_screen_identity_inventory.py",
        "ml/alphazero_lite/kalah_rules.py",
        "ml/alphazero_lite/verify_seed542_expanded_policy_strength_completeness.py",
    ):
        origin = source / relative
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)
    manifest_destination = target / AUDIT_RELATIVE / "seed545-source-manifest.json"
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source / AUDIT_RELATIVE / "seed545-source-manifest.json", manifest_destination
    )
    for name in (
        "seed545-canonical-index.jsonl.gz",
        "seed545-kalah-v3-float32-index.jsonl.gz",
        "seed545-replay-report.json",
    ):
        shutil.copy2(source / AUDIT_RELATIVE / name, target / AUDIT_RELATIVE / name)


def _copy_seven_source_publication(source: Path, target: Path) -> None:
    manifest = json.loads((source / AUDIT_RELATIVE / seven.MANIFEST_NAME).read_text())
    relatives: set[str] = set()
    for source_row in manifest["sources"]:
        relatives.update(source_row["anchor_files"])
        relatives.add(source_row["receipt_path"])
        relatives.add(f"{source_row['archive']}/openings.jsonl")
        relatives.add(f"{source_row['archive']}/outcomes.jsonl")
        relatives.update(
            f"{source_row['archive']}/{batch}" for batch in source_row["game_batches"]
        )
        relatives.update(
            f"{source_row['archive']}/batches/report-{index:04d}.json"
            for index in range(1024)
        )
        relatives.update(source_row["failed_or_superseded_attempts"])
    relatives.update(seven.BASELINE_INDEX_BINDINGS)
    relatives.update(
        (
            "ml/alphazero_lite/archived_screen_identity_inventory.py",
            "ml/alphazero_lite/archived_screen_identity_seven.py",
            "ml/alphazero_lite/kalah_rules.py",
            "ml/alphazero_lite/verify_seed542_expanded_policy_strength_completeness.py",
        )
    )
    relatives.add(f"{AUDIT_RELATIVE}/seed545-source-manifest.json")
    relatives.add(f"{AUDIT_RELATIVE}/{seven.MANIFEST_NAME}")
    relatives.update(
        f"{AUDIT_RELATIVE}/{name}"
        for name in (
            "seven-source-canonical-index.jsonl.gz",
            "seven-source-kalah-v3-float32-index.jsonl.gz",
            "seven-source-count-report.json",
        )
    )
    for relative in sorted(relatives):
        origin, destination = source / relative, target / relative
        if not origin.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)


def _snapshot(paths: list[Path]) -> dict[Path, tuple[int, int, str]]:
    return {
        path: (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in paths
    }


def test_public_cli_replays_physical_copy_without_tablebase() -> None:
    with tempfile.TemporaryDirectory(
        prefix="seed545-physical-copy-", dir=ROOT.parent
    ) as work:
        physical = Path(work) / "checkout"
        _copy_minimal_publication(ROOT, physical)
        assert not (physical / ".tmp/seed542-screen-runtime/kalah_v1_21.kvtb").exists()
        published_index = physical / AUDIT_RELATIVE / "seed545-canonical-index.jsonl.gz"
        expected_jsonl = gzip.decompress(published_index.read_bytes())
        equivalent_gzip = gzip.compress(expected_jsonl, compresslevel=1, mtime=1)
        assert equivalent_gzip != published_index.read_bytes()
        published_index.write_bytes(equivalent_gzip)
        published_index_before = _snapshot([published_index])
        archive = physical / ARCHIVE_RELATIVE
        manifest = json.loads(
            (physical / AUDIT_RELATIVE / "seed545-source-manifest.json").read_text()
        )
        evidence = [
            archive / relative
            for relative in (
                *manifest["bindings"],
                *manifest["recovery_receipts"],
                *manifest["game_batches"],
                *(f"batches/report-{index:04d}.json" for index in range(1024)),
                manifest["prelaunch_failure"]["path"],
            )
        ]
        evidence_before = _snapshot(evidence)
        env = {**os.environ, "PYTHONPATH": str(physical)}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ml.alphazero_lite.archived_screen_identity_inventory",
                "--root",
                str(physical),
                "--out",
                str(physical / AUDIT_RELATIVE),
            ],
            cwd=Path(work),
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["archive_authenticated"] is True
        assert report["archive_replayed"] is True
        assert report["replay"] == {"games": 2048, "openings": 1024, "pairs": 1024}
        assert report["tablebase_required"] is False
        assert _snapshot(evidence) == evidence_before
        assert _snapshot([published_index]) == published_index_before
        assert gzip.decompress(published_index.read_bytes()) == expected_jsonl


def test_corrupt_published_identity_index_is_rejected_without_repair() -> None:
    with tempfile.TemporaryDirectory(
        prefix="seed545-index-corruption-", dir=ROOT.parent
    ) as work:
        physical = Path(work) / "checkout"
        _copy_minimal_publication(ROOT, physical)
        index = physical / AUDIT_RELATIVE / "seed545-canonical-index.jsonl.gz"
        original = index.read_bytes()
        index.write_bytes(gzip.compress(gzip.decompress(original) + b" ", mtime=2))
        before = _snapshot([index])
        with pytest.raises(
            inventory.ArchiveVerificationError,
            match="published_index_mismatch:seed545-canonical-index.jsonl.gz",
        ):
            inventory.verify(physical, physical / AUDIT_RELATIVE)
        assert _snapshot([index]) == before


def test_modified_game_and_regenerated_local_receipt_rejected() -> None:
    with tempfile.TemporaryDirectory(
        prefix="seed545-mutated-copy-", dir=ROOT.parent
    ) as work:
        physical = Path(work) / "checkout"
        _copy_minimal_publication(ROOT, physical)
        batch = physical / ARCHIVE_RELATIVE / "batches/games-0002.jsonl"
        row = json.loads(batch.read_text().splitlines()[0])
        row["margin"] += 1
        lines = batch.read_text().splitlines()
        lines[0] = json.dumps(row, separators=(",", ":"))
        batch.write_text("\n".join(lines) + "\n")

        # Simulate a local receipt being regenerated over the altered bytes. It
        # is not a trust root; the manifest digest is fixed in the verifier.
        fake_receipt = {
            "schema": "local-regenerated-receipt",
            "opening_index": 2,
            "games_sha256": hashlib.sha256(batch.read_bytes()).hexdigest(),
        }
        (batch.parent / "completion-0002.json").write_text(
            json.dumps(fake_receipt, sort_keys=True) + "\n"
        )
        with pytest.raises(
            inventory.ArchiveVerificationError,
            match="authenticated_game_bytes_mismatch",
        ):
            inventory.verify(physical, physical / AUDIT_RELATIVE)


def test_kalah_v3_bytes_match_production_encoder_for_registered_opening() -> None:
    import numpy as np

    from ml.alphazero_lite.seed542_expanded_policy_strength_screen import encode_state

    archive = ROOT / ARCHIVE_RELATIVE
    opening = inventory._load_rows(archive / "openings.jsonl")[0]
    game_rows = inventory._load_rows(archive / "batches/games-0000.jsonl")
    checked = 0
    for position, row in enumerate(game_rows):
        trace, _ = inventory.replay_game_trace(
            opening,
            row,
            int(row["challenger_player"]),
            source="PR #545",
            run="seed542-expanded-policy-strength-screen",
            opening_index=0,
            game_position=position,
            batch_path="batches/games-0000.jsonl",
        )
        for occurrence in trace:
            state = occurrence["state"]
            expected = hashlib.sha256(
                np.asarray(
                    encode_state(state, input_encoding="kalah_v3"), dtype="<f4"
                ).tobytes()
            ).hexdigest()
            assert inventory.kalah_v3_identity(state) == expected
            checked += 1
    assert checked > 40


def test_rules_replay_rejects_opening_seat_move_and_result_mutations() -> None:
    archive = ROOT / ARCHIVE_RELATIVE
    opening = inventory._load_rows(archive / "openings.jsonl")[2]
    row = inventory._load_rows(archive / "batches/games-0002.jsonl")[0]
    seat = int(row["challenger_player"])
    inventory.replay_game_trace(
        opening,
        row,
        seat,
        source="PR #545",
        run="seed542-expanded-policy-strength-screen",
        opening_index=2,
        game_position=0,
        batch_path="batches/games-0002.jsonl",
    )

    wrong_opening = dict(opening)
    wrong_opening["state_hash"] = "0" * 64
    with pytest.raises(
        inventory.ArchiveVerificationError, match="opening_state_hash_mismatch"
    ):
        inventory.replay_game_trace(
            wrong_opening,
            row,
            seat,
            source="PR #545",
            run="seed542-expanded-policy-strength-screen",
            opening_index=2,
            game_position=0,
            batch_path="batches/games-0002.jsonl",
        )

    illegal = dict(row)
    moves = [int(value) for value in illegal["trajectory"].split(",")]
    moves[0] = 12
    illegal["trajectory"] = ",".join(map(str, moves))
    with pytest.raises(
        inventory.ArchiveVerificationError, match="illegal_trajectory_move"
    ):
        inventory.replay_game_trace(
            opening,
            illegal,
            seat,
            source="PR #545",
            run="seed542-expanded-policy-strength-screen",
            opening_index=2,
            game_position=0,
            batch_path="batches/games-0002.jsonl",
        )

    wrong_result = dict(row, margin=int(row["margin"]) + 1)
    with pytest.raises(
        inventory.ArchiveVerificationError, match="trajectory_result_mismatch"
    ):
        inventory.replay_game_trace(
            opening,
            wrong_result,
            seat,
            source="PR #545",
            run="seed542-expanded-policy-strength-screen",
            opening_index=2,
            game_position=0,
            batch_path="batches/games-0002.jsonl",
        )
    with pytest.raises(
        inventory.ArchiveVerificationError, match="trajectory_result_mismatch"
    ):
        inventory.replay_game_trace(
            opening,
            row,
            1 - seat,
            source="PR #545",
            run="seed542-expanded-policy-strength-screen",
            opening_index=2,
            game_position=0,
            batch_path="batches/games-0002.jsonl",
        )


def test_duplicate_recovered_pairs_and_replayed_directory_mismatch_rejected() -> None:
    with pytest.raises(
        inventory.ArchiveVerificationError, match="seat_swapped_pair_mismatch"
    ):
        inventory.validate_seat_swapped_pair(
            [{"challenger_player": 0}, {"challenger_player": 0}]
        )
    with pytest.raises(
        seven.SevenSourceError, match="recovery_pair_accounting_invalid"
    ):
        seven.validate_recovered_openings([0, 1, 1])

    manifest = json.loads((ROOT / AUDIT_RELATIVE / seven.MANIFEST_NAME).read_text())
    source = dict(next(row for row in manifest["sources"] if row["key"] == "545"))
    source["archive"] = "docs/data/seed512-fixed-composition-evaluation"
    with pytest.raises(seven.SevenSourceError, match="outcomes_sha_mismatch:545"):
        seven.replay_one_source(ROOT, source)


def test_seat_swap_and_reversed_decisive_ledger_outcome_rejected() -> None:
    archive = ROOT / "docs/data/seed552-c108-serving-p"
    rows = inventory._load_rows(archive / "batches/games-0000.jsonl")
    with pytest.raises(seven.SevenSourceError, match="seat_swapped_pair_mismatch"):
        seven.validate_seat_swapped_pair([rows[0], dict(rows[0])])

    decisive = dict(rows[1])
    assert decisive["winner"] == "challenger"
    assert int(decisive["margin"]) > 0
    with pytest.raises(seven.SevenSourceError, match="ledger_outcome_mismatch:552"):
        seven._ledger_matches_game(
            "552",
            {"outcome": "P", "margin": int(decisive["margin"])},
            decisive,
            int(decisive["margin"]),
            {"challenger": "C", "current": "P", "draw": "draw"},
        )


def test_seven_source_public_cli_physical_copy_is_read_only_and_portable() -> None:
    with tempfile.TemporaryDirectory(
        prefix="seed564-physical-copy-", dir=ROOT.parent
    ) as work:
        physical = Path(work) / "checkout"
        _copy_seven_source_publication(ROOT, physical)
        assert not (physical / "model-artifact/runtime/kalah_v1_21.kvtb").exists()
        assert not (physical / ".tmp/seed542-screen-runtime/kalah_v1_21.kvtb").exists()
        manifest = json.loads(
            (physical / AUDIT_RELATIVE / seven.MANIFEST_NAME).read_text()
        )
        baseline_files = [
            physical / relative for relative in seven.BASELINE_INDEX_BINDINGS
        ]
        evidence = baseline_files.copy()
        evidence.extend(
            (
                physical / AUDIT_RELATIVE / seven.MANIFEST_NAME,
                physical / AUDIT_RELATIVE / "seed545-source-manifest.json",
            )
        )
        for source in manifest["sources"]:
            evidence.extend(seven._source_snapshot(physical, source))
        evidence_before = _snapshot(sorted(set(evidence)))
        env = {**os.environ, "PYTHONPATH": str(physical)}
        expected_indexes = {
            name: (ROOT / AUDIT_RELATIVE / name).read_bytes()
            for name in (
                "seven-source-canonical-index.jsonl.gz",
                "seven-source-kalah-v3-float32-index.jsonl.gz",
                "seven-source-count-report.json",
            )
        }
        index = physical / AUDIT_RELATIVE / "seven-source-canonical-index.jsonl.gz"
        expected_canonical_jsonl = gzip.decompress(index.read_bytes())
        equivalent_gzip = gzip.compress(
            expected_canonical_jsonl, compresslevel=1, mtime=1
        )
        assert equivalent_gzip != index.read_bytes()
        index.write_bytes(equivalent_gzip)
        compressed_index_before = _snapshot([index])
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ml.alphazero_lite.archived_screen_identity_seven",
                "--root",
                str(physical),
                "--out",
                str(physical / AUDIT_RELATIVE),
            ],
            cwd=Path(work),
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        physical_report = json.loads(result.stdout)
        assert physical_report["baseline_verified"] is True
        assert len(physical_report["sources"]) == 7
        assert physical_report["fresh_confirmation_gate_483"] == "closed"
        assert _snapshot(sorted(set(evidence))) == evidence_before
        assert _snapshot([index]) == compressed_index_before
        for name, expected in expected_indexes.items():
            actual = (physical / AUDIT_RELATIVE / name).read_bytes()
            if name.endswith(".gz"):
                assert gzip.decompress(actual) == gzip.decompress(expected)
            else:
                assert actual == expected

        original_jsonl = gzip.decompress(index.read_bytes())
        index.write_bytes(
            gzip.compress(original_jsonl + b" ", compresslevel=1, mtime=2)
        )
        before = _snapshot([index])
        with pytest.raises(
            seven.SevenSourceError,
            match="published_index_mismatch:seven-source-canonical-index.jsonl.gz",
        ):
            seven.verify_seven(physical, physical / AUDIT_RELATIVE)
        assert _snapshot([index]) == before
