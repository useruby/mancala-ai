"""Preflight controls for seed512 command, recovery, analysis, and corruption checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.alphazero_lite import seed512_evaluation as protocol


def test_command_preserves_protocol_and_changes_only_simulation_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(protocol, "OUT", Path("seed512-test"))
    out = tmp_path / protocol.OUT
    out.mkdir(parents=True)
    seed511_registration = json.loads(
        (protocol.ROOT / protocol.SEED511_OUT / "registration.json").read_text()
    )
    prior_openings = protocol.read_rows(
        protocol.ROOT / protocol.SEED511_OUT / "openings.jsonl"
    )
    (out / "openings.jsonl").write_bytes(
        protocol.seed511.canonical_json(prior_openings[0]) + b"\n"
    )
    (out / "registration.json").write_text(
        json.dumps(
            {
                "runtime": seed511_registration["runtime"],
                "runtime_assets": seed511_registration["runtime_assets"],
            }
        )
    )
    frozen = tmp_path / protocol.SEED511_OUT
    frozen.mkdir(parents=True)
    (frozen / "registration.json").write_text(json.dumps(seed511_registration))
    argv = protocol.command(tmp_path, 0)
    assert argv[argv.index("--challenger-simulations") + 1] == "1200"
    assert argv[argv.index("--current-simulations") + 1] == "1200"
    assert argv[argv.index("--challenger-policy-artifact") + 1].endswith(
        "/runtime-artifacts/E4"
    )
    assert argv[argv.index("--challenger-value-artifact") + 1].endswith(
        "/runtime-artifacts/initializer"
    )
    for flag, value in (
        ("--workers", "1"),
        ("--c-puct", "1.25"),
        ("--seed-contract", "azlite_eval_seed_v2"),
        ("--seed", "511"),
        ("--max-moves", "200"),
        ("--games", "2"),
    ):
        assert argv[argv.index(flag) + 1] == value


def test_resume_skips_committed_pair_and_preserves_interrupted_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(protocol, "OUT", Path("seed512-test"))
    out = tmp_path / protocol.OUT / "batches"
    out.mkdir(parents=True)
    openings = [{"opening_index": i} for i in range(1024)]
    monkeypatch.setattr(
        protocol, "verify_pair", lambda root, index, opening, rows: None
    )
    ledger = [
        {"opening_index": 0, "challenger_player": 0},
        {"opening_index": 0, "challenger_player": 1},
    ]
    assert protocol.resume_plan(tmp_path, openings, ledger) == list(range(1, 1024))
    (out / "games-0001.jsonl").write_text("outcome-bearing partial batch\n")
    with pytest.raises(ValueError, match="seed512_interrupted_pair_preserved:1"):
        protocol.resume_plan(tmp_path, openings, ledger)


def test_independent_primary_and_matched_analysis_are_deterministic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(protocol, "OUT", Path("seed512-test"))
    monkeypatch.setattr(protocol, "SEED511_OUT", Path("seed511-test"))
    prior = [
        {"opening_index": i, "winner": "draw"} for i in range(1024) for _ in range(2)
    ]
    (tmp_path / "seed511-test").mkdir()
    (tmp_path / "seed511-test" / "outcomes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in prior)
    )
    ledger = [
        {"opening_index": i, "winner": "challenger" if i % 2 else "current"}
        for i in range(1024)
        for _ in range(2)
    ]
    first = protocol.analyze(tmp_path, ledger)
    second = protocol.analyze(tmp_path, ledger)
    assert first == second
    assert first["score"] == 0.5
    assert first["seed511_matched_contrast"]["mean_1200_minus_128"] == 0.0
    assert first["seed511_matched_contrast"]["primary_gate_substitute"] is False


def test_runtime_report_corruption_fails_semantic_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(protocol, "OUT", Path("seed512-test"))
    batches = tmp_path / protocol.OUT / "batches"
    batches.mkdir(parents=True)
    opening = {"opening_index": 0}
    (batches / "opening-0000.jsonl").write_bytes(
        protocol.seed511.canonical_json(opening) + b"\n"
    )
    (batches / "games-0000.jsonl").write_text(
        '{"challenger_player":0}\n{"challenger_player":1}\n'
    )
    seed511_registration = tmp_path / protocol.SEED511_OUT / "registration.json"
    seed511_registration.parent.mkdir(parents=True)
    registered_runtime = json.loads(
        (protocol.ROOT / protocol.SEED511_OUT / "registration.json").read_text()
    )
    seed511_registration.write_text(
        json.dumps(
            {
                "runtime": registered_runtime["runtime"],
                "runtime_assets": {
                    "exact_root_native_probe": {"sha256": "probe"},
                    "exact_root_tablebase": {"sha256": "tablebase"},
                },
            }
        )
    )
    (tmp_path / protocol.OUT / "registration.json").write_text(
        json.dumps({"runtime": {}})
    )
    (batches / "report-0000.json").write_text(
        json.dumps(
            {
                "games_played": 2,
                "notes": {
                    "challenger_simulations": 1199,
                    "current_simulations": 1200,
                },
            }
        )
    )
    with pytest.raises(ValueError, match="seed512_pair_runtime:0"):
        protocol.verify_pair(tmp_path, 0, opening, [{}, {}])
