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
        *verifier.SEED479_RECEIPTS,
    }
    for relative in paths:
        source_path = source / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source_path.resolve())
    out = root / protocol.OUT
    out.mkdir(parents=True)
    suite_path = out / "openings.jsonl"
    shutil.copy2(source / protocol.OUT / "openings.jsonl", suite_path)
    reg_path = out / "registration.json"
    shutil.copy2(registration_source, reg_path)
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
        arena_rows = [
            {"challenger_player": 0, "winner": "challenger"},
            {"challenger_player": 1, "winner": "current"},
        ]
        game_log = resume / f"games-{opening_index:04d}.jsonl"
        game_log.write_text("".join(json.dumps(item) + "\n" for item in arena_rows))
        report = resume / f"report-{opening_index:04d}.json"
        report.write_text(json.dumps({"opening_index": opening_index}))
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
        artifact_manifest[arm] = {}
        artifact_dir = out / "runtime-artifacts" / arm
        artifact_dir.mkdir(parents=True)
        for filename in (
            "model.npz",
            "weights.json",
            "metadata.json",
            "search_policy.json",
        ):
            artifact_path = artifact_dir / filename
            artifact_path.write_text(f"synthetic {arm} {filename}")
            artifact_manifest[arm][filename] = protocol.sha256_file(artifact_path)
    (out / "runtime-artifacts.json").write_text(json.dumps(artifact_manifest))
    receipt = {
        "status": "seed488_strength_evaluation_complete",
        **{
            name: protocol.sha256_file(out / name)
            for name in (
                "registration.json",
                "openings.jsonl",
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
    )
    for relative in paths:
        destination = copied / relative
        source_path = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(copied)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed488_strength_evaluation",
            "--root",
            str(copied),
        ],
        cwd=copied,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"status": "prepared_not_executed"' in result.stdout


def test_outcome_resume_rejects_duplicate_or_missing_games():
    rows = _paired_rows()
    completed = rows[:2]
    assert protocol.paired_scores(completed + rows[2:])[0] == 0.5
    with pytest.raises(ValueError, match="missing"):
        protocol.paired_scores(completed[:1])
    with pytest.raises(ValueError, match="duplicate"):
        protocol.paired_scores(completed + completed[:1] + rows[2:])
