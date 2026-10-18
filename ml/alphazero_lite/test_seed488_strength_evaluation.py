"""Synthetic tests for seed488 pairing, accounting, and endpoint export parity."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.alphazero_lite import seed488_strength_evaluation as protocol
from ml.alphazero_lite import verify_seed488_strength_evaluation as verifier
from ml.alphazero_lite import train


def _paired_rows(count: int = protocol.OPENING_COUNT) -> list[dict]:
    rows = []
    for opening in range(count):
        rows.extend(
            [
                {
                    "opening_index": opening,
                    "opening_state_hash": f"{opening:064x}",
                    "F_player": 0,
                    "winner": "F",
                },
                {
                    "opening_index": opening,
                    "opening_state_hash": f"{opening:064x}",
                    "F_player": 1,
                    "winner": "initializer",
                },
            ]
        )
    return rows


def test_seat_pairing_opening_identity_and_complete_accounting():
    rows = _paired_rows()
    score, clusters = protocol.paired_scores(rows)
    assert score == 0.5
    assert clusters == [0.5] * protocol.OPENING_COUNT
    with pytest.raises(ValueError, match="duplicate"):
        protocol.paired_scores([*rows, rows[0]])
    with pytest.raises(ValueError, match="missing"):
        protocol.paired_scores(rows[:-1])
    with pytest.raises(ValueError, match="invalid"):
        protocol.paired_scores([{**rows[0], "F_player": 2}, *rows[1:]])


def test_outcome_and_score_accounting():
    assert protocol.score_winner("F") == 1
    assert protocol.score_winner("draw") == 0.5
    assert protocol.score_winner("initializer") == 0
    with pytest.raises(ValueError, match="outcome"):
        protocol.score_winner("unknown")
    rows = _paired_rows()
    rows[0]["winner"] = "draw"
    assert protocol.paired_scores(rows)[0] == pytest.approx(0.49951171875)


def test_opening_bootstrap_is_fixed_and_paired():
    clusters = [0.0, 0.5, 1.0] * (protocol.OPENING_COUNT // 3) + [0.25, 0.75]
    a = protocol.opening_bootstrap(clusters)
    b = protocol.opening_bootstrap(clusters)
    assert a == b
    assert a["replicates"] == 20_000
    assert a["seed"] == 488
    assert a["one_sided_lower"] <= np.mean(clusters)
    with pytest.raises(ValueError, match="opening_count"):
        protocol.opening_bootstrap(clusters[:-1])


def test_interrupted_resume_batch_accounting(tmp_path: Path):
    rows = _paired_rows(1)
    ledger = tmp_path / "outcomes.jsonl"
    protocol.append_rows_atomic(ledger, rows[:1])
    with pytest.raises(ValueError, match="incomplete"):
        protocol.completed_opening_indices(
            [json.loads(line) for line in ledger.read_text().splitlines()]
        )
    protocol.append_rows_atomic(ledger, rows)
    assert protocol.completed_opening_indices(
        [json.loads(line) for line in ledger.read_text().splitlines()]
    ) == {0}


def test_deterministic_opening_identities(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(protocol, "reconstructable_exclusions", lambda _root: set())
    first = protocol.make_suite(Path("."))
    second = protocol.make_suite(Path("."))
    assert first == second
    identities = [row["state_hash"] for row in first]
    assert len(identities) == protocol.OPENING_COUNT
    assert len(set(identities)) == protocol.OPENING_COUNT


def test_synthetic_export_preserves_frozen_policy_and_value_outputs(tmp_path: Path):
    initializer_path = protocol.ROOT / protocol.INITIALIZER
    trajectory_path = protocol.ROOT / protocol.TRAJECTORY
    with np.load(trajectory_path, allow_pickle=False) as archive:
        f_parameters = [archive[f"F_step_16_post_parameter_{i:02d}"] for i in range(22)]
    fixture = torch.linspace(-0.25, 0.25, 54, dtype=torch.float32).reshape(2, 27)
    legal = torch.tensor([[1, 1, 0, 1, 0, 1], [0, 1, 1, 0, 1, 1]], dtype=torch.bool)
    for arm, checkpoint in (("initializer", initializer_path), ("F", None)):
        source = train.PolicyValueNet((96, 3), "residual_v3", 27).eval()
        if arm == "initializer":
            train.load_checkpoint_into_model(source, checkpoint)
        else:
            for parameter, value in zip(source.parameters(), f_parameters, strict=True):
                parameter.data.copy_(torch.from_numpy(value.copy()))
        checkpoint_path = tmp_path / f"{arm}-source.npz"
        np.savez(checkpoint_path, **train.checkpoint_from_model(source))
        artifact_path = tmp_path / f"{arm}-artifact"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ml.alphazero_lite.export_artifact",
                "--checkpoint",
                str(checkpoint_path),
                "--out-dir",
                str(artifact_path),
                "--version",
                f"seed488-fixture-{arm}",
                "--model-type",
                "residual_v3",
                "--input-encoding",
                "kalah_v3",
            ],
            cwd=protocol.ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        with np.load(artifact_path / "model.npz", allow_pickle=False) as archive:
            exported_json = json.loads((artifact_path / "weights.json").read_text())
            assert set(archive.files) == set(exported_json)
            for name in archive.files:
                np.testing.assert_array_equal(archive[name], exported_json[name])
        exported = train.PolicyValueNet((96, 3), "residual_v3", 27).eval()
        train.load_checkpoint_into_model(exported, artifact_path / "model.npz")
        with torch.inference_mode():
            source_logits, source_value = source(fixture)
            exported_logits, exported_value = exported(fixture)
            source_policy = torch.softmax(source_logits.masked_fill(~legal, -1e9), 1)
            exported_policy = torch.softmax(
                exported_logits.masked_fill(~legal, -1e9), 1
            )
        torch.testing.assert_close(exported_policy, source_policy, rtol=0, atol=0)
        torch.testing.assert_close(exported_value, source_value, rtol=0, atol=0)


def _write_complete_fixture(root: Path) -> Path:
    source = protocol.ROOT
    registration_source = source / protocol.OUT / "registration.json"
    registration = json.loads(registration_source.read_text())
    registration["schema"] = "seed488-frozen-strength-evaluation-v2"
    registration["supersedes"] = (
        "PR #487 seed488_prepared_not_executed registration and suite; this is a preparation-only correction before any outcomes"
    )
    registration["implementation_sha256"] = protocol.sha256_file(
        source / "ml/alphazero_lite/seed488_strength_evaluation.py"
    )
    registration["independent_verifier_sha256"] = protocol.sha256_file(
        source / "ml/alphazero_lite/verify_seed488_strength_evaluation.py"
    )
    registration["synthetic_tests_sha256"] = protocol.sha256_file(
        source / "ml/alphazero_lite/test_seed488_strength_evaluation.py"
    )
    registration["runtime"]["max_moves_policy"] = (
        "At 200 decision plies, stop without sweeping remaining pits; compare captured stores at the cap. A game ending earlier uses Kalah terminal sweeping. Recorded trajectory length must be terminal before 200 or exactly 200."
    )
    registration["trajectory_contract"] = (
        "arena.py at the runtime source hash; opening prefixes and trajectories are player-relative v2 moves converted by KalahGame.pit_index at each ply; rules use KalahGame.move including extra turns and terminal sweep; stored margin is challenger captured store minus current captured store; winner is the sign of that margin"
    )
    paths = {
        *registration["runtime"]["identities_sha256"],
        *registration["generator_dependency_sha256"],
        "ml/alphazero_lite/seed488_strength_evaluation.py",
        "ml/alphazero_lite/verify_seed488_strength_evaluation.py",
        "ml/alphazero_lite/test_seed488_strength_evaluation.py",
        str(protocol.INITIALIZER),
        str(protocol.TRAJECTORY),
        str(protocol.BASE_EXCLUSIONS),
        str(protocol.RECONSTRUCTED_EXCLUSIONS),
        "docs/data/seed484-extended-probe-union.json.gz",
        *verifier.SEED479_RECEIPTS,
    }
    for relative in paths:
        source_path = source / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source_path.resolve())
    for relative in registration["runtime"]["identities_sha256"]:
        registration["runtime"]["identities_sha256"][relative] = protocol.sha256_file(
            source / relative
        )
    out = root / protocol.OUT
    out.mkdir(parents=True)
    suite_path = out / "openings.jsonl"
    suite_path.write_bytes(protocol.suite_bytes(protocol.make_suite(source)))
    registration["opening_suite"]["opening_suite_sha256"] = protocol.sha256_file(
        suite_path
    )
    registration["opening_suite"]["exclusions"].update(
        {
            "seed484_extended_union_sha256": verifier.EXTENDED_UNION_SHA256,
            "seed484_extended_union_file_sha256": verifier.EXTENDED_EXCLUSION_SHA256,
            "seed484_extended_union_pair_count": 149_756,
        }
    )
    reg_path = out / "registration.json"
    reg_path.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n")
    preserved = {}
    for name in ("registration.json", "openings.jsonl", "receipt.json"):
        target = (
            out
            / f"{name.removesuffix('.json').removesuffix('.jsonl')}-pr487{Path(name).suffix}"
        )
        shutil.copy2(source / protocol.OUT / name, target)
        preserved[name] = {
            "preserved_as": target.name,
            "sha256": protocol.sha256_file(target),
            "mtime_ns": target.stat().st_mtime_ns,
        }
    (out / "supersession.json").write_text(
        json.dumps(
            {
                "status": "seed488_preparation_superseded_before_execution",
                "supersedes_pr": 487,
                "preserved_originals": preserved,
                "no_outcomes_existed_at_supersession": True,
            },
            sort_keys=True,
        )
    )
    rows = _paired_rows()
    suite_rows = [json.loads(line) for line in suite_path.read_text().splitlines()]
    resume = out / "resume"
    resume.mkdir()
    for opening_index in range(protocol.OPENING_COUNT):
        opening_row = suite_rows[opening_index]
        opening_file = resume / f"opening-{opening_index:04d}.jsonl"
        opening_file.write_text(
            json.dumps(opening_row, sort_keys=True, separators=(",", ":")) + "\n"
        )
        arena_rows = []
        for seat in (0, 1):
            game = protocol.KalahGame.from_state(
                {
                    "player_pits": [4] * 6,
                    "opponent_pits": [4] * 6,
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }
            )
            for move in opening_row["prefix_moves"]:
                assert game.move(game.pit_index(int(move)))
            trajectory = []
            while not game.over() and len(trajectory) < 200:
                move = game.possible_moves()[0]
                trajectory.append(move)
                assert game.move(game.pit_index(move))
            margin = game.captured_seeds[seat] - game.captured_seeds[1 - seat]
            winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
            arena_rows.append(
                {
                    "challenger_player": seat,
                    "winner": winner,
                    "margin": margin,
                    "game_length": len(trajectory),
                    "trajectory": ",".join(map(str, trajectory)),
                    "opening_state_hash": opening_row["state_hash"],
                    "opening_prefix_moves": opening_row["prefix_moves"],
                }
            )
            rows[opening_index * 2 + seat]["winner"] = {
                "challenger": "F",
                "current": "initializer",
                "draw": "draw",
            }[winner]
        game_log = resume / f"games-{opening_index:04d}.jsonl"
        game_log.write_text("".join(json.dumps(item) + "\n" for item in arena_rows))
        report = resume / f"report-{opening_index:04d}.json"
        report.write_text(
            json.dumps(
                {
                    "opening_index": opening_index,
                    "games_played": 2,
                    "wins": sum(item["winner"] == "challenger" for item in arena_rows),
                    "losses": sum(item["winner"] == "current" for item in arena_rows),
                    "draws": sum(item["winner"] == "draw" for item in arena_rows),
                    "notes": {
                        "challenger_simulations": 1200,
                        "current_simulations": 1200,
                        "workers_used": 1,
                        "workers": 1,
                        "games": 2,
                        "games_per_opening": 2,
                        "max_moves": 200,
                        "random_opening_plies": 0,
                        "seed_contract": "azlite_eval_seed_v2",
                        "base_seed": 488 + opening_index,
                        "suite_sha256": registration["opening_suite"][
                            "opening_suite_sha256"
                        ],
                        "search_profile": {"c_puct": 1.25},
                        "exact_root_solve_threshold": 16,
                        "exact_root_native_probe_sha256": registration["runtime"][
                            "identities_sha256"
                        ]["model-artifact/runtime/kalah_v1_tablebase"],
                        "exact_root_tablebase_sha256": registration["runtime"][
                            "identities_sha256"
                        ]["model-artifact/runtime/kalah_v1_21.kvtb"],
                        "runtime_search_policy_sha256": registration["runtime"][
                            "identities_sha256"
                        ]["model-artifact/current/search_policy.json"],
                        "runtime_search_policy_mode": "puct_exact_root_hybrid",
                        "exact_root_objective": "final_score_margin",
                        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
                        "exact_leaf_solve_mode": "disabled",
                        "search_options": registration["runtime"]["search_options"],
                        "seed": 488 + opening_index,
                    },
                }
            )
        )
        for row in rows[opening_index * 2 : opening_index * 2 + 2]:
            row.update(
                {
                    "opening_state_hash": opening_row["state_hash"],
                    "batch_game_position": row["F_player"],
                    "batch_game_log_sha256": protocol.sha256_file(game_log),
                    "batch_report_sha256": protocol.sha256_file(report),
                    "batch_opening_sha256": protocol.sha256_file(opening_file),
                }
            )
    outcomes = out / "outcomes.jsonl"
    outcomes.write_text("".join(json.dumps(row) + "\n" for row in rows))
    score, opening_scores = protocol.paired_scores(rows)
    boot = protocol.opening_bootstrap(opening_scores)
    result = {
        "game_count": len(rows),
        "score": score,
        "bootstrap": boot,
        "advance": boot["one_sided_lower"] > 0.5,
        "promotion": False,
    }
    result_path = out / "results.json"
    result_path.write_text(json.dumps(result))
    artifact_manifest = {}
    for arm in ("F", "initializer"):
        artifact_dir = out / "runtime-artifacts" / arm
        artifact_dir.mkdir(parents=True)
        checkpoint = protocol._materialize_endpoint(root, arm, out)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ml.alphazero_lite.export_artifact",
                "--checkpoint",
                str(checkpoint),
                "--out-dir",
                str(artifact_dir),
                "--version",
                f"seed488-{arm}",
                "--model-type",
                "residual_v3",
                "--input-encoding",
                "kalah_v3",
            ],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        )
        shutil.copy2(
            source / protocol.RUNTIME_POLICY, artifact_dir / "search_policy.json"
        )
        artifact_manifest[arm] = {
            name: protocol.sha256_file(artifact_dir / name)
            for name in (
                "model.npz",
                "weights.json",
                "metadata.json",
                "search_policy.json",
            )
        }
    (out / "runtime-artifacts.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n"
    )
    artifact_manifest_sha256 = protocol.sha256_file(out / "runtime-artifacts.json")
    for row in rows:
        row["batch_runtime_contract"] = verifier._expected_batch_runtime_contract(
            registration, int(row["opening_index"]), artifact_manifest_sha256
        )
    (out / "outcomes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = {
        "status": "seed488_strength_evaluation_complete",
        **{
            name: protocol.sha256_file(out / name)
            for name in (
                "registration.json",
                "openings.jsonl",
                "supersession.json",
                "runtime-artifacts.json",
                "outcomes.jsonl",
                "results.json",
            )
        },
    }
    (out / "receipt.json").write_text(json.dumps(receipt))
    return root


def test_independent_verifier_accepts_rebound_complete_synthetic_fixture(
    tmp_path: Path,
):
    root = _write_complete_fixture(tmp_path)
    assert verifier.verify(root)["status"] == "verified_complete"
    opening_index = 2
    opening = json.loads(
        (root / protocol.OUT / "openings.jsonl").read_text().splitlines()[opening_index]
    )
    game = protocol.KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    for move in opening["prefix_moves"]:
        game.move(game.pit_index(int(move)))
    trajectory = json.loads(
        (root / protocol.OUT / "resume" / f"games-{opening_index:04d}.jsonl")
        .read_text()
        .splitlines()[0]
    )["trajectory"]
    extra_turn_seen = False
    for move in map(int, trajectory.split(",")):
        player = game.current_player
        game.move(game.pit_index(move))
        extra_turn_seen |= game.current_player == player
    assert extra_turn_seen


def test_independent_verifier_rejects_receipt_rebound_result_tampering(tmp_path: Path):
    root = _write_complete_fixture(tmp_path)
    out = root / protocol.OUT
    result_path = out / "results.json"
    result = json.loads(result_path.read_text())
    result["score"] = 1.0
    result_path.write_text(json.dumps(result))
    receipt = json.loads((out / "receipt.json").read_text())
    receipt["results.json"] = protocol.sha256_file(result_path)
    (out / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="result_mismatch"):
        verifier.verify(root)


def _reseal_outcomes(out: Path) -> None:
    outcome_path = out / "outcomes.jsonl"
    rows = [json.loads(line) for line in outcome_path.read_text().splitlines()]
    for row in rows:
        index = int(row["opening_index"])
        row["batch_game_log_sha256"] = protocol.sha256_file(
            out / "resume" / f"games-{index:04d}.jsonl"
        )
        row["batch_report_sha256"] = protocol.sha256_file(
            out / "resume" / f"report-{index:04d}.json"
        )
    outcome_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt_path = out / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["outcomes.jsonl"] = protocol.sha256_file(outcome_path)
    receipt_path.write_text(json.dumps(receipt))


def test_rebound_corrupt_trajectory_fails_semantic_replay(tmp_path: Path):
    root = _write_complete_fixture(tmp_path)
    out = root / protocol.OUT
    game_path = out / "resume/games-0000.jsonl"
    games = [json.loads(line) for line in game_path.read_text().splitlines()]
    moves = games[0]["trajectory"].split(",")
    moves[0] = "99"
    games[0]["trajectory"] = ",".join(moves)
    game_path.write_text("".join(json.dumps(row) + "\n" for row in games))
    _reseal_outcomes(out)
    with pytest.raises(ValueError, match="trajectory_illegal"):
        verifier.verify(root)


def test_rebound_false_winner_fails_replay_derived_result(tmp_path: Path):
    root = _write_complete_fixture(tmp_path)
    out = root / protocol.OUT
    game_path = out / "resume/games-0000.jsonl"
    games = [json.loads(line) for line in game_path.read_text().splitlines()]
    games[0]["winner"] = (
        "current" if games[0]["winner"] == "challenger" else "challenger"
    )
    game_path.write_text("".join(json.dumps(row) + "\n" for row in games))
    outcomes_path = out / "outcomes.jsonl"
    outcomes = [json.loads(line) for line in outcomes_path.read_text().splitlines()]
    outcomes[0]["winner"] = "initializer" if games[0]["winner"] == "current" else "F"
    outcomes_path.write_text("".join(json.dumps(row) + "\n" for row in outcomes))
    _reseal_outcomes(out)
    with pytest.raises(ValueError, match="trajectory_outcome_mismatch"):
        verifier.verify(root)


def test_rebound_report_runtime_mismatch_fails(tmp_path: Path):
    root = _write_complete_fixture(tmp_path)
    out = root / protocol.OUT
    report_path = out / "resume/report-0000.json"
    report = json.loads(report_path.read_text())
    report["notes"]["challenger_simulations"] = 4
    report_path.write_text(json.dumps(report))
    _reseal_outcomes(out)
    with pytest.raises(ValueError, match="arena_report_runtime_contract"):
        verifier.verify(root)


def test_rebound_artifact_mutation_fails_frozen_state_binding(tmp_path: Path):
    root = _write_complete_fixture(tmp_path)
    out = root / protocol.OUT
    artifact = out / "runtime-artifacts/F/weights.json"
    artifact.write_text("{}")
    manifest_path = out / "runtime-artifacts.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["F"]["weights.json"] = protocol.sha256_file(artifact)
    manifest_path.write_text(json.dumps(manifest))
    receipt_path = out / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["runtime-artifacts.json"] = protocol.sha256_file(manifest_path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="artifact_weights_keys:F"):
        verifier.verify(root)


def test_missing_and_duplicate_games_rejected_by_verifier(tmp_path: Path):
    root = _write_complete_fixture(tmp_path)
    out = root / protocol.OUT
    path = out / "outcomes.jsonl"
    rows = path.read_text().splitlines()
    path.write_text("\n".join(rows[:-1]) + "\n")
    receipt = json.loads((out / "receipt.json").read_text())
    receipt["outcomes.jsonl"] = protocol.sha256_file(path)
    (out / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="game_coverage"):
        verifier.verify(root)

    root = _write_complete_fixture(tmp_path / "duplicate")
    out = root / protocol.OUT
    path = out / "outcomes.jsonl"
    with path.open("a") as stream:
        stream.write(path.read_text().splitlines()[0] + "\n")
    receipt = json.loads((out / "receipt.json").read_text())
    receipt["outcomes.jsonl"] = protocol.sha256_file(path)
    (out / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="duplicate_or_invalid_game"):
        verifier.verify(root)


def test_runner_rejects_precommit_interruption_and_skips_after_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    interrupted = _write_complete_fixture(tmp_path / "before-commit")
    interrupted_out = interrupted / protocol.OUT
    (interrupted_out / "outcomes.jsonl").unlink()
    (interrupted_out / "outcomes.jsonl.pending").write_text('{"staged":"pair"}\n')
    with pytest.raises(ValueError, match="interrupted_pending_commit"):
        protocol.run(interrupted, execute=True)

    committed = _write_complete_fixture(tmp_path / "after-commit")
    calls = []

    def mock_preflight(command, **kwargs):
        calls.append(command)
        if any(str(item).endswith("ml.alphazero_lite.arena") for item in command):
            raise AssertionError("committed pair must not be rerun")

    monkeypatch.setattr(protocol.subprocess, "run", mock_preflight)
    protocol.run(committed, execute=True)
    assert len(calls) == 1


def test_prepared_verifier_runs_from_physical_copy(tmp_path: Path):
    source = protocol.ROOT
    copied = tmp_path / "copied-checkout"
    shutil.copytree(
        source / "ml", copied / "ml", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(source / "model-artifact", copied / "model-artifact")
    paths = (
        protocol.OUT / "registration.json",
        protocol.OUT / "openings.jsonl",
        protocol.OUT / "receipt.json",
        protocol.INITIALIZER,
        protocol.TRAJECTORY,
        protocol.SEED479_RECEIPT,
        Path("docs/data/seed479-population-ablation/registration.json"),
        Path("docs/data/seed479-population-ablation/evidence.json"),
        protocol.BASE_EXCLUSIONS,
        protocol.RECONSTRUCTED_EXCLUSIONS,
        "docs/data/seed484-extended-probe-union.json.gz",
    )
    source_snapshots = {
        relative: (
            protocol.sha256_file(source / relative),
            (source / relative).stat().st_mtime_ns,
        )
        for relative in paths
    }
    for relative in paths:
        destination = copied / relative
        source_path = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(copied)
    unrelated = tmp_path / "unrelated-directory"
    unrelated.mkdir()
    prepared = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.seed488_strength_evaluation",
            "--root",
            str(copied),
            "--prepare",
        ],
        cwd=unrelated,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"schema": "seed488-frozen-strength-evaluation-v2"' in prepared.stdout
    copied_out = copied / protocol.OUT
    preserved_snapshot = {
        name: ((copied_out / name).read_bytes(), (copied_out / name).stat().st_mtime_ns)
        for name in (
            "registration-pr487.json",
            "openings-pr487.jsonl",
            "receipt-pr487.json",
        )
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed488_strength_evaluation",
            "--root",
            str(copied),
        ],
        cwd=unrelated,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"status": "prepared_not_executed"' in result.stdout
    for relative, (digest, mtime_ns) in source_snapshots.items():
        assert protocol.sha256_file(source / relative) == digest
        assert (source / relative).stat().st_mtime_ns == mtime_ns
    for name, snapshot in preserved_snapshot.items():
        assert (
            (copied_out / name).read_bytes(),
            (copied_out / name).stat().st_mtime_ns,
        ) == snapshot
    receipt = copied_out / "receipt.json"
    receipt_data = json.loads(receipt.read_text())
    receipt_data["registration.json"] = "0" * 64
    receipt.write_text(json.dumps(receipt_data))
    failed_snapshot = (receipt.read_bytes(), receipt.stat().st_mtime_ns)
    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed488_strength_evaluation",
            "--root",
            str(copied),
        ],
        cwd=unrelated,
        env=env,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert (receipt.read_bytes(), receipt.stat().st_mtime_ns) == failed_snapshot
    for name, snapshot in preserved_snapshot.items():
        assert (
            (copied_out / name).read_bytes(),
            (copied_out / name).stat().st_mtime_ns,
        ) == snapshot


def test_outcome_resume_rejects_duplicate_or_missing_games():
    rows = _paired_rows()
    completed = rows[:2]
    assert protocol.paired_scores(completed + rows[2:])[0] == 0.5
    with pytest.raises(ValueError, match="missing"):
        protocol.paired_scores(completed[:1])
    with pytest.raises(ValueError, match="duplicate"):
        protocol.paired_scores(completed + completed[:1] + rows[2:])


def test_run_entry_point_reaches_mocked_arena_with_prepared_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = protocol.ROOT
    shutil.copytree(
        source / "ml", tmp_path / "ml", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(source / "model-artifact", tmp_path / "model-artifact")
    inputs = (
        protocol.INITIALIZER,
        protocol.TRAJECTORY,
        protocol.SEED479_RECEIPT,
        Path("docs/data/seed479-population-ablation/registration.json"),
        Path("docs/data/seed479-population-ablation/evidence.json"),
        protocol.BASE_EXCLUSIONS,
        protocol.RECONSTRUCTED_EXCLUSIONS,
        protocol.EXTENDED_EXCLUSIONS,
    )
    for relative in inputs:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    protocol.prepare(tmp_path)
    out = tmp_path / protocol.OUT
    monkeypatch.setattr(protocol, "_validate_exported_artifacts", lambda *_: None)
    calls = []
    actual_subprocess_run = subprocess.run

    class MockArenaReached(RuntimeError):
        pass

    def mock_run(command, **kwargs):
        calls.append(command)
        if any(str(item).endswith("ml.alphazero_lite.arena") for item in command):
            raise MockArenaReached
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(tmp_path)
        kwargs["env"] = environment
        return actual_subprocess_run(command, **kwargs)

    monkeypatch.setattr(protocol.subprocess, "run", mock_run)
    # Pre-populated exports exercise the real registration/preflight path while
    # the arena launch itself is intercepted before a game can start.
    for arm in ("F", "initializer"):
        artifact = out / "runtime-artifacts" / arm
        artifact.mkdir(parents=True)
        for name in (
            "model.npz",
            "weights.json",
            "metadata.json",
            "search_policy.json",
        ):
            (artifact / name).write_text("fixture")
    with pytest.raises(MockArenaReached):
        protocol.run(tmp_path, execute=True)
    assert len(calls) == 2
    assert any("ml.alphazero_lite.arena" in call for call in calls[-1])
