"""Lifecycle and portable-evidence acceptance tests for seed429."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite import run_seed429_canonical_policy_normalization as runner
from ml.alphazero_lite.verify_seed429_publication import _verify_components
from ml.alphazero_lite.verify_seed429_publication import _independent_coefficients
from ml.alphazero_lite.verify_seed429_publication import EXECUTION_SOURCE_INVENTORY

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed429-canonical-policy-normalization"


def _finished_row(opening_index: int, game_index: int) -> dict:
    suite = suites.load_suite_jsonl(str(DATA / "openings.jsonl"))
    opening = suite[opening_index]
    game = KalahGame.from_state(suites.INITIAL_STATE)
    for relative in opening["prefix_moves"]:
        assert game.move(game.pit_index(int(relative)))
    seat = game_index % 2
    actions = []
    while not game.over():
        moves = game.possible_moves()
        assert moves
        absolute = game.pit_index(moves[0])
        actions.append(absolute)
        assert game.move(absolute)
        assert len(actions) < 2048
    margin = game.captured_seeds[seat] - game.captured_seeds[1 - seat]
    return {
        "game_index": game_index,
        "opening_index": opening_index,
        "game_within_opening": game_index % 2,
        "challenger_player": seat,
        "opening_contract": "arena_player_relative_v2",
        "opening_state_hash": opening["state_hash"],
        "trajectory": ",".join(map(str, actions)),
        "game_length": len(actions),
        "margin": margin,
        "winner": "challenger" if margin > 0 else "current" if margin < 0 else "draw",
    }


def test_public_pre_registration_cli_succeeds_with_no_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    relocated_data = tmp_path / "seed429"
    shutil.copytree(DATA, relocated_data)
    registration = relocated_data / "registration.json"
    if registration.exists():
        registration.unlink()
    monkeypatch.setattr(runner, "DATA", relocated_data)
    monkeypatch.setattr(sys, "argv", ["seed429", "pre-registration-readiness"])
    runner.main()
    output = json.loads(capsys.readouterr().out)
    assert output["ready"] is True
    assert output["no_training_or_games_started"] is True
    assert not registration.exists()


def test_portable_independent_vectors_match_published_census() -> None:
    reconstructed = _independent_coefficients(ROOT)
    census = json.loads((DATA / "coefficient-census.json").read_text())
    assert (
        reconstructed["control_coefficients_sha256"]
        == census["control_coefficients_sha256"]
    )
    assert (
        reconstructed["treatment_coefficients_sha256"]
        == census["treatment_coefficients_sha256"]
    )
    assert reconstructed["rows_changed"] == census["rows_changed"]
    assert reconstructed["validation_unchanged"]
    assert reconstructed["le16_unchanged"]


def test_execution_source_snapshot_inventory_accepts_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runner.py"
    source.write_bytes(b"registered source bytes")
    snapshot = (
        tmp_path
        / "docs/data/seed429-canonical-policy-normalization/execution-source-snapshots/runner.py"
    )
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(source.read_bytes())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    bindings = {str(snapshot.relative_to(tmp_path)): digest}
    runner.verify_execution_snapshots(tmp_path, bindings, ("runner.py",))
    with pytest.raises(ValueError, match="inventory_incomplete"):
        runner.verify_execution_snapshots(tmp_path, {}, ("runner.py",))
    snapshot.write_bytes(b"tampered snapshot")
    with pytest.raises(ValueError, match="snapshot_mismatch"):
        runner.verify_execution_snapshots(tmp_path, bindings, ("runner.py",))


def test_candidate_runtime_identity_rejects_changed_artifact_bytes(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "candidate"
    artifact.mkdir()
    path = artifact / "weights.json"
    path.write_bytes(b"bound candidate weights")
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    runner.verify_artifact_files(artifact, {"weights.json": identity})
    path.write_bytes(b"altered weights")
    with pytest.raises(ValueError, match="artifact_identity_mismatch"):
        runner.verify_artifact_files(artifact, {"weights.json": identity})


def test_final_exclusion_component_hash_tampering_is_rejected(tmp_path: Path) -> None:
    proof = json.loads((DATA / "opening-exclusion-proof.json").read_text())
    for component in proof["components"]:
        relocated = tmp_path / component["path"]
        relocated.parent.mkdir(parents=True, exist_ok=True)
        original = ROOT / component["path"]
        if relocated.exists() or relocated.is_symlink():
            continue
        if component["name"] == "manifest.json":
            relocated.write_bytes(original.read_bytes() + b"tampered")
            component["path"] = str(relocated.relative_to(tmp_path))
        else:
            relocated.symlink_to(original)
    with pytest.raises(ValueError, match="exclusion_component_hash_mismatch"):
        _verify_components(tmp_path, proof)


def test_missing_exclusion_component_is_rejected(tmp_path: Path) -> None:
    proof = json.loads((DATA / "opening-exclusion-proof.json").read_text())
    component = proof["components"][0]
    component["path"] = "docs/data/missing-component.json"
    with pytest.raises(ValueError, match="exclusion_component_hash_mismatch"):
        _verify_components(tmp_path, proof)


def test_relocated_read_only_preparation_verifier_and_component_tamper() -> None:
    fixture = Path(
        tempfile.mkdtemp(prefix="seed429-portable-fixture-", dir=ROOT / ".tmp")
    )
    try:
        source_data = DATA
        fixture_data = fixture / "docs/data/seed429-canonical-policy-normalization"
        shutil.copytree(source_data, fixture_data)
        proof = json.loads((fixture_data / "opening-exclusion-proof.json").read_text())
        paths = {
            "docs/data/seed416-policy-target-softening/registration-v3.json",
            "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz",
            "docs/data/seed422-adam-first-moment/opening-exclusion-proof.json",
            "docs/data/seed422-adam-first-moment/openings.jsonl",
            "docs/data/seed422-adam-first-moment/outcome-ledger.jsonl",
        }
        paths.update(component["path"] for component in proof["components"])
        paths.update(
            f"docs/data/seed426-canonical-overlap/sources/{name}.jsonl.gz"
            for name in (
                "fresh",
                "generic_bootstrap",
                "random_teacher",
                "opening_disagreement",
                "stability",
            )
        )
        for relative in paths:
            source = ROOT / relative
            destination = fixture / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(source, destination)
        snapshots = {}
        for relative in EXECUTION_SOURCE_INVENTORY:
            source = ROOT / relative
            live_copy = fixture / relative
            live_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, live_copy)
            snapshot = fixture_data / "execution-source-snapshots" / Path(relative).name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, snapshot)
            snapshots[str(snapshot.relative_to(fixture))] = hashlib.sha256(
                snapshot.read_bytes()
            ).hexdigest()
        registration = {
            "status": "registered_before_training_and_model_probing",
            "execution_source_inventory": list(EXECUTION_SOURCE_INVENTORY),
            "execution_source_snapshots": snapshots,
            "census_sha256": hashlib.sha256(
                (fixture_data / "coefficient-census.json").read_bytes()
            ).hexdigest(),
            "suite_path": "docs/data/seed429-canonical-policy-normalization/openings.jsonl",
            "suite_sha256": hashlib.sha256(
                (fixture_data / "openings.jsonl").read_bytes()
            ).hexdigest(),
            "exclusion_proof_path": "docs/data/seed429-canonical-policy-normalization/opening-exclusion-proof.json",
            "exclusion_proof_sha256": hashlib.sha256(
                (fixture_data / "opening-exclusion-proof.json").read_bytes()
            ).hexdigest(),
        }
        (fixture_data / "registration.json").write_text(
            json.dumps(registration, indent=2, sort_keys=True) + "\n"
        )
        command = [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed429_publication",
            "--root",
            str(fixture),
            "--preparation-only",
        ]
        checked = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr
        assert json.loads(checked.stdout)["valid"] is True
        row_accounting = (
            fixture / "docs/data/seed426-canonical-overlap/row-accounting.jsonl.gz"
        )
        row_accounting.write_bytes(row_accounting.read_bytes() + b"tamper")
        tampered = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert tampered.returncode != 0
        assert "exclusion_component_hash_mismatch" in tampered.stderr
    finally:
        shutil.rmtree(fixture)


def test_final_seed429_suite_reconstructs_and_has_zero_final_overlap() -> None:
    proof = json.loads((DATA / "opening-exclusion-proof.json").read_text())
    reg = {
        "suite_path": "docs/data/seed429-canonical-policy-normalization/openings.jsonl",
        "suite_sha256": runner.sha256_file(DATA / "openings.jsonl"),
    }
    verifier = __import__(
        "ml.alphazero_lite.verify_seed429_publication",
        fromlist=["_verify_suite"],
    )
    verifier._verify_suite(ROOT, reg, set(proof["excluded_identities"]))


def test_chunk_resume_accepts_bound_chunk_and_rejects_binding_or_outcome_tamper(
    tmp_path: Path,
) -> None:
    row = _finished_row(0, 0)
    chunk = {
        "lane": "A",
        "start_index": 0,
        "game_count": 1,
        "registration_sha256": "r" * 64,
        "runtime_binding_sha256": "b" * 64,
        "suite_sha256": runner.sha256_file(DATA / "openings.jsonl"),
        "evaluation_seed": 429,
        "seed_contract": "azlite_eval_seed_v2",
        "challenger_simulations": 384,
        "current_simulations": 384,
        "c_puct": 1.25,
        "game_entries": [row],
    }
    path = tmp_path / "chunk.json"
    encoded = json.dumps(chunk, sort_keys=True)
    path.write_text(encoded)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = runner._load_completed_chunk(
        path,
        expected_sha256=digest,
        lane="A",
        start=0,
        count=1,
        registration_sha256="r" * 64,
        runtime_binding_sha256="b" * 64,
        suite_sha256=runner.sha256_file(DATA / "openings.jsonl"),
    )
    assert len(loaded["game_entries"]) == 1
    with pytest.raises(ValueError, match="identity_mismatch"):
        runner._load_completed_chunk(
            path,
            expected_sha256=digest,
            lane="A",
            start=0,
            count=1,
            registration_sha256="x" * 64,
            runtime_binding_sha256="b" * 64,
            suite_sha256=runner.sha256_file(DATA / "openings.jsonl"),
        )
    altered = json.loads(path.read_text())
    altered["game_entries"][0]["winner"] = "draw"
    path.write_text(json.dumps(altered, sort_keys=True))
    with pytest.raises(ValueError, match="resume_hash_mismatch"):
        runner._load_completed_chunk(
            path,
            expected_sha256=digest,
            lane="A",
            start=0,
            count=1,
            registration_sha256="r" * 64,
            runtime_binding_sha256="b" * 64,
            suite_sha256=runner.sha256_file(DATA / "openings.jsonl"),
        )


def test_chunk_validator_rejects_duplicate_outcomes_and_changed_opening() -> None:
    row = _finished_row(0, 0)
    with pytest.raises(ValueError, match="duplicate_or_seat_mismatch"):
        runner._validate_chunk_rows([row, row], 0, 2)
    changed = dict(row, opening_state_hash="0" * 64)
    with pytest.raises(ValueError, match="opening_identity_mismatch"):
        runner._validate_chunk_rows([changed], 0, 1)


def test_analyze_refuses_incomplete_arena_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "DATA", tmp_path)
    with pytest.raises((FileNotFoundError, ValueError)):
        runner.analyze()
