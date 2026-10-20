"""Pre-execution contracts for seed508 output composition and commands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.alphazero_lite import arena
from ml.alphazero_lite import seed508_policy_value_crossover as protocol
from ml.alphazero_lite import verify_seed508_policy_value_crossover as verifier
from ml.alphazero_lite.kalah_rules import KalahGame


def test_composed_outputs_keep_each_source_trunk_and_skip_terminals() -> None:
    artifacts = protocol.ROOT / protocol.SEED505 / "runtime-artifacts"
    e4 = arena.ArtifactEvaluator(artifacts / "E4")
    init = arena.ArtifactEvaluator(artifacts / "initializer")
    game = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    p = arena.ComposedArtifactEvaluator(
        e4, init, policy_source="current", value_source="candidate"
    )
    v = arena.ComposedArtifactEvaluator(
        init, e4, policy_source="current", value_source="candidate"
    )
    p_policy, p_value = p.evaluate(game)
    e4_policy, _ = e4.evaluate(game)
    _, init_value = init.evaluate(game)
    v_policy, v_value = v.evaluate(game)
    init_policy, _ = init.evaluate(game)
    _, e4_value = e4.evaluate(game)
    np.testing.assert_array_equal(p_policy, e4_policy)
    assert p_value == init_value
    np.testing.assert_array_equal(v_policy, init_policy)
    assert v_value == e4_value

    class Forbidden:
        def evaluate(self, _game):
            raise AssertionError("terminal positions must not evaluate models")

    terminal = KalahGame.from_state(
        {
            "player_pits": [0] * 6,
            "opponent_pits": [0] * 6,
            "player_store": 24,
            "opponent_store": 24,
            "current_player": 0,
        }
    )
    policy, value = arena.ComposedArtifactEvaluator(
        Forbidden(), Forbidden(), policy_source="current", value_source="candidate"
    ).evaluate(terminal)
    assert value == 0.0
    np.testing.assert_array_equal(policy, np.zeros(6, dtype=np.float32))


def test_commands_bind_composition_artifacts_and_full_runtime(tmp_path: Path) -> None:
    opening = json.loads((protocol.ROOT / protocol.SUITE).read_text().splitlines()[0])
    out = tmp_path / protocol.OUT
    (out / "runtime-artifacts" / "E4").mkdir(parents=True)
    (out / "runtime-artifacts" / "initializer").mkdir()
    historical = tmp_path / "docs/data/seed488-frozen-strength-evaluation"
    historical.mkdir(parents=True)
    (historical / "registration.json").write_bytes(
        (
            protocol.ROOT
            / "docs/data/seed488-frozen-strength-evaluation/registration.json"
        ).read_bytes()
    )
    argv_p = protocol.command(tmp_path, "P", 0, opening)
    argv_v = protocol.command(tmp_path, "V", 0, opening)
    for argv, policy, value in (
        (argv_p, "E4", "initializer"),
        (argv_v, "initializer", "E4"),
    ):
        assert argv[argv.index("--challenger-policy-artifact") + 1].endswith(
            f"/{policy}"
        )
        assert argv[argv.index("--challenger-value-artifact") + 1].endswith(f"/{value}")
        assert argv[argv.index("--challenger") + 1].endswith(f"/{policy}")
        assert argv[argv.index("--challenger-simulations") + 1] == "128"
        assert argv[argv.index("--current-simulations") + 1] == "128"
        assert argv[argv.index("--c-puct") + 1] == "1.25"
        assert argv[argv.index("--workers") + 1] == "1"
        assert argv[argv.index("--seed") + 1] == "488"
        assert argv[argv.index("--max-moves") + 1] == "200"
        assert argv[argv.index("--random-opening-plies") + 1] == "0"
        assert argv[argv.index("--games") + 1] == "2"


def test_complete_pair_rejects_incomplete_ledger(tmp_path: Path) -> None:
    opening = json.loads((protocol.ROOT / protocol.SUITE).read_text().splitlines()[0])
    out = tmp_path / protocol.OUT / "batches"
    out.mkdir(parents=True)
    with pytest.raises(ValueError, match="complete_pair_shape:P:0"):
        protocol.validate_pair(tmp_path, "P", 0, opening, [])


def test_resume_validates_and_skips_a_committed_pair_without_arena(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import shutil

    opening = json.loads((protocol.ROOT / protocol.SUITE).read_text().splitlines()[0])
    source = protocol.ROOT / protocol.SEED505 / "batches"
    batch = tmp_path / protocol.OUT / "batches"
    batch.mkdir(parents=True)
    opener = batch / "P-opening-0000.jsonl"
    opener.write_bytes(protocol.base.canonical_json(opening) + b"\n")
    game_path = batch / "P-games-0000.jsonl"
    report_path = batch / "P-report-0000.json"
    shutil.copyfile(source / "games-0000.jsonl", game_path)
    shutil.copyfile(source / "report-0000.json", report_path)
    report = json.loads(report_path.read_text())
    report["seed508_composition"] = {
        "policy_source": "E4",
        "value_source": "initializer",
        "trunk_policy": "policy source's own trunk",
        "trunk_value": "value source's own trunk",
    }
    report_path.write_text(json.dumps(report))

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    games = protocol.rows(game_path)
    ledger = [
        {
            "arm": "P",
            "opening_index": 0,
            "challenger_player": int(game["challenger_player"]),
            "winner": game["winner"],
            "batch_game_position": position,
            "batch_game_sha256": digest(game_path),
            "batch_report_sha256": digest(report_path),
            "batch_opening_sha256": digest(opener),
        }
        for position, game in enumerate(games)
    ]
    monkeypatch.setattr(
        protocol.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("committed pair must not rerun arena"),
    )
    verifier.verify_complete_pair(tmp_path, "P", 0, opening, ledger)
    pending = protocol.resume_plan(tmp_path, [opening] + [{}] * 511, ledger)
    assert ("P", 0) not in pending
    assert len(pending) == 1023


def test_receipt_rebound_component_assignment_corruption_is_rejected(
    tmp_path: Path,
) -> None:
    import hashlib
    import shutil

    opening = json.loads((protocol.ROOT / protocol.SUITE).read_text().splitlines()[0])
    source = protocol.ROOT / protocol.SEED505 / "batches"
    batch = tmp_path / protocol.OUT / "batches"
    batch.mkdir(parents=True)
    opener = batch / "P-opening-0000.jsonl"
    opener.write_bytes(protocol.base.canonical_json(opening) + b"\n")
    game_path = batch / "P-games-0000.jsonl"
    report_path = batch / "P-report-0000.json"
    shutil.copyfile(source / "games-0000.jsonl", game_path)
    shutil.copyfile(source / "report-0000.json", report_path)
    report = json.loads(report_path.read_text())
    report["seed508_composition"] = {
        "policy_source": "initializer",
        "value_source": "initializer",
        "trunk_policy": "policy source's own trunk",
        "trunk_value": "value source's own trunk",
    }
    report_path.write_text(json.dumps(report))

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    ledger = [
        {
            "arm": "P",
            "opening_index": 0,
            "challenger_player": int(game["challenger_player"]),
            "winner": game["winner"],
            "batch_game_position": position,
            "batch_game_sha256": digest(game_path),
            "batch_report_sha256": digest(report_path),
            "batch_opening_sha256": digest(opener),
        }
        for position, game in enumerate(protocol.rows(game_path))
    ]
    with pytest.raises(ValueError, match="complete_pair_runtime:P:0"):
        verifier.verify_complete_pair(tmp_path, "P", 0, opening, ledger)
