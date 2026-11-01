from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.alphazero_lite import arena, self_play, selfplay_evaluator_pilot as pilot
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.verify_selfplay_evaluator_pilot import verify_shard


def test_composed_serving_evaluator_routes_policy_and_value_for_orientations() -> None:
    policy_evaluator = arena.ArtifactEvaluator(pilot.ROOT / pilot.P_ARTIFACT)
    value_evaluator = arena.ArtifactEvaluator(pilot.ROOT / pilot.VALUE_ARTIFACT)
    composed = arena.ComposedArtifactEvaluator(
        policy_evaluator,
        value_evaluator,
        policy_source="current",
        value_source="candidate",
    )
    game = KalahGame(pits=[4] * 12, captured_seeds=[0, 0], current_player=0)
    positions = [game.clone()]
    game.move(game.pit_index(2))  # standard start: an extra turn
    assert game.current_player == 0
    positions.append(game.clone())
    game.move(game.pit_index(0))
    positions.append(game.clone())
    assert any(position.current_player == 1 for position in positions)
    for position in positions:
        actual_policy, actual_value = composed.evaluate(position)
        expected_policy, _ = policy_evaluator.evaluate(position)
        _, expected_value = value_evaluator.evaluate(position)
        np.testing.assert_array_equal(actual_policy, expected_policy)
        assert actual_value == expected_value
    original_arm = pilot._ARM
    pilot._ARM = "treatment"
    try:
        treatment = pilot._evaluator_factory(
            pilot.ROOT / pilot.INITIALIZER, input_encoding="kalah_v3", cache_size=0
        )
        for position in positions:
            actual_policy, actual_value = treatment.evaluate(position)
            expected_policy, _ = policy_evaluator.evaluate(position)
            _, expected_value = value_evaluator.evaluate(position)
            np.testing.assert_array_equal(actual_policy, expected_policy)
            assert actual_value == expected_value
    finally:
        pilot._ARM = original_arm


def test_terminal_composition_does_not_evaluate_components() -> None:
    class Exploding:
        def evaluate(self, game):
            raise AssertionError("terminal position reached a component")

    evaluator = arena.ComposedArtifactEvaluator(
        Exploding(), Exploding(), policy_source="current", value_source="candidate"
    )
    terminal = KalahGame(pits=[0] * 12, captured_seeds=[24, 24], current_player=0)
    policy, value = evaluator.evaluate(terminal)
    np.testing.assert_array_equal(policy, np.zeros(6, dtype=np.float32))
    assert value == 0.0

    policy_artifact = arena.ArtifactEvaluator(pilot.ROOT / pilot.P_ARTIFACT)
    value_artifact = arena.ArtifactEvaluator(pilot.ROOT / pilot.VALUE_ARTIFACT)
    serving_p = arena.ComposedArtifactEvaluator(
        policy_artifact,
        value_artifact,
        policy_source="current",
        value_source="candidate",
    )
    previous_arm = pilot._ARM
    pilot._ARM = "treatment"
    try:
        treatment = pilot._evaluator_factory(
            pilot.ROOT / pilot.INITIALIZER,
            input_encoding="kalah_v3",
            cache_size=0,
        )
        expected_policy, expected_value = serving_p.evaluate(terminal)
        actual_policy, actual_value = treatment.evaluate(terminal)
        np.testing.assert_array_equal(actual_policy, expected_policy)
        assert actual_value == expected_value
    finally:
        pilot._ARM = previous_arm


def test_control_adapter_matches_ordinary_checkpoint_evaluator(monkeypatch) -> None:
    checkpoint = pilot.ROOT / pilot.INITIALIZER
    ordinary = self_play.CheckpointEvaluator(checkpoint, input_encoding="kalah_v3")
    monkeypatch.setattr(pilot, "_ARM", "control")
    adapted = pilot._evaluator_factory(
        checkpoint, input_encoding="kalah_v3", cache_size=0
    )
    for game in (
        KalahGame(pits=[4] * 12, captured_seeds=[0, 0], current_player=0),
        KalahGame(
            pits=[1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            captured_seeds=[0, 0],
            current_player=1,
        ),
    ):
        expected_policy, expected_value = ordinary.evaluate(game)
        actual_policy, actual_value = adapted.evaluate(game)
        np.testing.assert_array_equal(actual_policy, expected_policy)
        assert actual_value == expected_value


def test_control_adapter_reproduces_ordinary_worker(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = pilot.ROOT / pilot.INITIALIZER
    kwargs = {
        "worker_id": 0,
        "start_index": 0,
        "games": 1,
        "seed": 91,
        "seed_pool": [91],
        "checkpoint": str(checkpoint),
        "input_encoding": "kalah_v3",
        "simulations": 2,
        "c_puct": 1.25,
        "temperature_threshold": 10,
        "temperature": 1.0,
        "temperature_late": 0.1,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
        "max_moves": 1,
        "write_game_metadata": True,
    }
    ordinary_path = tmp_path / "ordinary.jsonl"
    adapted_path = tmp_path / "adapted.jsonl"
    self_play.run_self_play_worker(**kwargs, shard_path=str(ordinary_path))
    monkeypatch.setattr(pilot, "_ARM", "control")
    monkeypatch.setattr(
        self_play, "build_checkpoint_evaluator", pilot._evaluator_factory
    )
    self_play.run_self_play_worker(**kwargs, shard_path=str(adapted_path))
    assert adapted_path.read_bytes() == ordinary_path.read_bytes()


def test_prepare_refuses_existing_registration(tmp_path) -> None:
    out = tmp_path / pilot.OUT
    out.mkdir(parents=True)
    (out / "registration.json").write_text("{}\n")
    with pytest.raises(FileExistsError, match="pilot_registration_already_exists"):
        pilot.prepare(tmp_path)


def test_verifier_accounts_for_interrupted_partial_game(tmp_path: Path) -> None:
    source = pilot.ROOT / "docs/data/seed542-selfplay-4096/shard-542002.jsonl"
    row = json.loads(source.read_text().splitlines()[0])
    target = tmp_path / "tampered.jsonl"
    target.write_text(json.dumps(row) + "\n")
    report = verify_shard(target, arm="control", expected_seed=int(row["game_seed"]))
    assert report["games"] == {"completed": 0, "capped": 0, "incomplete": 64}


def test_verifier_accounts_for_capped_game(tmp_path: Path) -> None:
    source = pilot.ROOT / "docs/data/seed542-selfplay-4096/shard-542002.jsonl"
    row = json.loads(source.read_text().splitlines()[0])
    row["game_capped"] = True
    target = tmp_path / "capped.jsonl"
    target.write_text(json.dumps(row) + "\n")
    report = verify_shard(target, arm="control", expected_seed=int(row["game_seed"]))
    assert report["games"] == {"completed": 0, "capped": 1, "incomplete": 63}


def test_verifier_rejects_wrong_seed_association(tmp_path: Path) -> None:
    source = pilot.ROOT / "docs/data/seed542-selfplay-4096/shard-542002.jsonl"
    row = json.loads(source.read_text().splitlines()[0])
    seed_path = tmp_path / "wrong-seed.jsonl"
    seed_path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="seed_association_mismatch"):
        verify_shard(seed_path, arm="control", expected_seed=int(row["game_seed"]) + 1)


def test_verifier_rejects_illegal_target_and_wrong_terminal_value(
    tmp_path: Path,
) -> None:
    source = pilot.ROOT / "docs/data/seed542-selfplay-4096/shard-542002.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    game_rows = [row for row in rows if row["game_index"] == 0]
    target_path = tmp_path / "illegal-target.jsonl"
    target_rows = [dict(row) for row in game_rows]
    target_rows[0]["policy"] = [0.0] * 6
    target_path.write_text("".join(json.dumps(row) + "\n" for row in target_rows))
    with pytest.raises(ValueError, match="illegal_or_unnormalized_target"):
        verify_shard(
            target_path, arm="control", expected_seed=int(game_rows[0]["game_seed"])
        )

    terminal_path = tmp_path / "wrong-value.jsonl"
    terminal_rows = [dict(row) for row in game_rows]
    terminal_rows[-1]["value"] = 0.0 if terminal_rows[-1]["value"] != 0 else 1.0
    terminal_path.write_text("".join(json.dumps(row) + "\n" for row in terminal_rows))
    with pytest.raises(ValueError, match="value_or_outcome_mismatch"):
        verify_shard(
            terminal_path, arm="control", expected_seed=int(game_rows[0]["game_seed"])
        )


def test_altered_component_sidecar_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "component.npz"
    checkpoint.write_bytes(b"checkpoint")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    for filename in pilot.ARTIFACT_FILES:
        (artifact / filename).write_bytes(filename.encode())
    component = {
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": pilot.sha(checkpoint),
        "artifact": artifact.name,
        "files": pilot.artifact_hashes(artifact),
    }
    pilot.authenticate_component(tmp_path, component)
    (artifact / "search_policy.json").write_bytes(b"modified sidecar")
    with pytest.raises(ValueError, match="pilot_component_artifact_changed"):
        pilot.authenticate_component(tmp_path, component)
    release_sidecar = tmp_path / "serving-search-policy.json"
    release_sidecar.write_bytes(b"serving release")
    expected = {"serving-search-policy.json": pilot.sha(release_sidecar)}
    pilot.authenticate_files(tmp_path, expected)
    release_sidecar.write_bytes(b"altered serving release")
    with pytest.raises(ValueError, match="pilot_release_file_changed"):
        pilot.authenticate_files(tmp_path, expected)


def test_altered_generation_command_is_rejected(tmp_path: Path) -> None:
    registration = {
        "generation": {
            "command_template": [
                "python",
                "-m",
                "pilot",
                "execute-arm",
                "--arm",
                "{arm}",
                "--out",
                "{output}",
                "--games",
                "64",
            ]
        }
    }
    correct = ["--out", str(pilot.OUT / "control.jsonl"), "--games", "64"]
    pilot.authenticate_worker_command(registration, "control", correct, tmp_path)
    with pytest.raises(ValueError, match="pilot_worker_command_mismatch"):
        pilot.authenticate_worker_command(
            registration, "control", [*correct[:-1], "63"], tmp_path
        )


def test_completed_output_is_skipped_only_after_hash_and_corpus_check(
    tmp_path: Path,
) -> None:
    data = tmp_path / "arm.jsonl"
    log = tmp_path / "arm.log"
    data.write_bytes(b"corpus")
    log.write_bytes(b"log")
    events = [
        {"event": "started"},
        {
            "event": "finished",
            "output_sha256": pilot.sha(data),
            "log_sha256": pilot.sha(log),
        },
    ]
    report = {"games": {"completed": 64, "capped": 0, "incomplete": 0}}
    assert pilot.authenticate_completed_output(
        "control", events, data, log, verify_corpus=lambda *_a, **_k: report
    )
    data.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="pilot_prior_output_receipt_mismatch"):
        pilot.authenticate_completed_output(
            "control", events, data, log, verify_corpus=lambda *_a, **_k: report
        )


@pytest.mark.parametrize("tamper", ["command", "evidence"])
def test_attempt_verifier_rejects_altered_command_or_evidence(
    tmp_path: Path, monkeypatch, tamper: str
) -> None:
    monkeypatch.setattr(pilot, "OUT", Path("pilot"))
    out = tmp_path / pilot.OUT
    out.mkdir()
    (out / "registration.json").write_text("{}\n")
    registration = {
        "resource_launcher": [],
        "generation": {
            "command_template": ["python", "--arm", "{arm}", "--out", "{output}"]
        },
    }
    monkeypatch.setattr(pilot, "preflight", lambda _root: registration)
    arm = "control"
    evidence = out / f"{arm}.jsonl"
    log = out / f"{arm}.log"
    evidence.write_bytes(b"immutable output")
    log.write_bytes(b"immutable log")
    command = ["python", "--arm", arm, "--out", str(evidence)]
    if tamper == "command":
        command[0] = "altered-python"
    record = {
        "arm": arm,
        "argv": command,
        "registration_sha256": pilot.sha(out / "registration.json"),
        "event": "started",
    }
    finished = {
        **record,
        "event": "finished",
        "output_sha256": pilot.sha(evidence),
        "log_sha256": pilot.sha(log),
    }
    (out / "attempts.jsonl").write_text(
        json.dumps(record) + "\n" + json.dumps(finished) + "\n"
    )
    if tamper == "evidence":
        evidence.write_bytes(b"changed output")
    with pytest.raises(ValueError):
        pilot.verify_attempts(tmp_path)
