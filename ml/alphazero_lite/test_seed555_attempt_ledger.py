"""Tests for committed attempt validation at the seed555 verifier entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ml.alphazero_lite import seed555_policy_strength_screen as protocol
from ml.alphazero_lite import verify_seed555_policy_strength_screen as verifier


def test_verifier_entrypoint_rejects_rebound_failed_attempt_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(protocol, "OPENINGS", 1)
    monkeypatch.setattr(protocol, "GAMES", 2)
    root = tmp_path / "copy"
    output = root / protocol.OUT
    batch = output / "batches"
    batch.mkdir(parents=True)
    (output / "registration.json").write_text('{"runtime":{}}\n')
    verifier_path = "ml/alphazero_lite/verify_seed555_policy_strength_screen.py"
    (output / "execution-freeze.json").write_text(
        json.dumps({"source_sha256": {verifier_path: "frozen-original-verifier-sha"}})
        + "\n"
    )
    command = {"opening_index": 0, "argv": ["python", "arena"]}
    command_path = batch / "command-0000.json"
    command_path.write_text(json.dumps(command) + "\n")
    (output / "command-preflight.jsonl").write_text(json.dumps(command) + "\n")
    games_path = batch / "games-0000.jsonl"
    report_path = batch / "report-0000.json"
    games_path.write_text('{"game":0}\n{"game":1}\n')
    report_path.write_text('{"games_played":2}\n')
    pair = [
        {
            "opening_index": 0,
            "game_position": seat,
            "outcome": "T512",
            "capped_nonterminal": False,
        }
        for seat in (0, 1)
    ]
    completion = {
        "opening_index": 0,
        "command_sha256": protocol.sha256(command_path),
        "games_sha256": protocol.sha256(games_path),
        "report_sha256": protocol.sha256(report_path),
        "outcomes": pair,
    }
    (batch / "completion-0000.json").write_text(json.dumps(completion) + "\n")
    attempt_path = batch / "attempt-0000-00.json"
    attempt = {
        "opening_index": 0,
        "returncode": 0,
        "command_sha256": protocol.sha256(command_path),
    }
    attempt_path.write_text(json.dumps(attempt) + "\n")
    (output / "outcomes.jsonl").write_bytes(
        b"".join(protocol.stable_json(row) + b"\n" for row in pair)
    )
    analysis, scores = verifier.reconstruct_analysis(pair)
    (output / "analysis.json").write_text(json.dumps(analysis) + "\n")
    (output / "per-opening-scores.jsonl").write_bytes(
        b"".join(protocol.stable_json(row) + b"\n" for row in scores)
    )
    verification_path = output / "verification.json"
    verification_path.write_text('{"status":"verified_complete"}\n')
    receipt_path = output / "verification-receipt.json"
    receipt = {
        "schema": "seed555_read_only_verification_receipt_v1",
        "status": "independent_read_only_verified_complete",
        "verification_result": {
            "status": "verified_complete",
            "validated_games": 2,
            "validated_pairs": 1,
        },
        "verification_sha256": protocol.sha256(verification_path),
        "evidence_sha256": {
            "batches/attempt-0000-00.json": protocol.sha256(attempt_path)
        },
    }
    receipt_path.write_text(json.dumps(receipt) + "\n")
    monkeypatch.setattr(verifier, "verify_preflight", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(verifier, "validate_pair", lambda *_args, **_kwargs: pair)
    monkeypatch.setattr(
        verifier, "validate_execution_runtime_checks", lambda *_args, **_kwargs: [{}]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_seed555_policy_strength_screen",
            "--root",
            str(root),
            "--out",
            str(output),
        ],
    )

    verifier.main()
    assert json.loads(capsys.readouterr().out)["attempt_verification"]["status"] == (
        "committed_attempts_semantically_verified"
    )

    # Rebind the receipt hash to altered attempt bytes. The entrypoint must
    # reject their failed exit and unrelated command hash despite re-binding.
    attempt["returncode"] = 1
    attempt["command_sha256"] = "f" * 64
    attempt_path.write_text(json.dumps(attempt) + "\n")
    receipt["evidence_sha256"]["batches/attempt-0000-00.json"] = protocol.sha256(
        attempt_path
    )
    receipt_path.write_text(json.dumps(receipt) + "\n")
    before = {
        path.relative_to(output).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in output.rglob("*")
        if path.is_file()
    }
    with pytest.raises(ValueError, match="failed_or_rebound_attempt"):
        verifier.main()
    assert capsys.readouterr().out == ""
    after = {
        path.relative_to(output).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before
