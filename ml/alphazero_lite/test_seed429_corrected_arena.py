"""Acceptance tests for the separately frozen seed429 arena correction."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from ml.alphazero_lite import seed429_corrected_arena as corrected
from ml.alphazero_lite import verify_seed429_correction as correction_verifier
from ml.alphazero_lite.test_seed429_workflow import _finished_row
from ml.alphazero_lite import run_seed429_canonical_policy_normalization as frozen

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed429-canonical-policy-normalization"


def test_real_arena_parser_accepts_required_deterministic_chunk_output(
    tmp_path: Path,
) -> None:
    binding = json.loads((DATA / "runtime-binding.json").read_text())
    report, chunk = corrected.deterministic_output_paths("A", 64)
    existing_output = report.read_bytes() if report.is_file() else None
    candidate = ROOT / binding["candidates"]["A"]["artifact"]
    opponent = ROOT / binding["opponent"]
    parsed = corrected.construct_parser_args(
        "A", 64, 32, candidate, opponent, DATA / "openings.jsonl", report
    )
    assert parsed.out == str(report)
    assert report != chunk
    assert report != corrected.deterministic_output_paths("A", 96)[0]
    assert corrected.deterministic_output_paths("A", 64) == (report, chunk)
    assert (report.read_bytes() if report.is_file() else None) == existing_output


def test_corrected_chunk_resumes_once_and_rejects_altered_binding_or_outcome(
    tmp_path: Path,
) -> None:
    row = _finished_row(0, 0)
    identities = {
        "registration_sha256": "r" * 64,
        "runtime_binding_sha256": "b" * 64,
        "suite_sha256": frozen.sha256_file(DATA / "openings.jsonl"),
        "correction_receipt_sha256": "c" * 64,
        "correction_source_hashes_sha256": "s" * 64,
        "seed": 429,
        "seed_contract": "azlite_eval_seed_v2",
        "challenger_simulations": 384,
        "current_simulations": 384,
        "c_puct": 1.25,
        "chunk_size": 32,
    }
    chunk = {
        "lane": "A",
        "start": 0,
        "count": 1,
        **identities,
        "worker_result": {"game_entries": [row]},
    }
    path = tmp_path / "chunk.json"
    path.write_text(json.dumps(chunk, sort_keys=True))
    hash_value = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = {"lane": "A", "start": 0, "count": 1, **identities}
    first = corrected._load_chunk(path, expected=expected, source_hash=hash_value)
    resumed = corrected._load_chunk(path, expected=expected, source_hash=hash_value)
    assert len(first["worker_result"]["game_entries"]) == 1
    assert len(resumed["worker_result"]["game_entries"]) == 1
    assert (
        len({item["game_index"] for item in resumed["worker_result"]["game_entries"]})
        == 1
    )
    changed_seed = {**expected, "seed": 430}
    with pytest.raises(ValueError, match="chunk_identity_mismatch"):
        corrected._load_chunk(path, expected=changed_seed, source_hash=hash_value)
    altered = json.loads(path.read_text())
    altered["worker_result"]["game_entries"][0]["winner"] = "draw"
    path.write_text(json.dumps(altered, sort_keys=True))
    with pytest.raises(ValueError, match="chunk_hash_mismatch"):
        corrected._load_chunk(path, expected=expected, source_hash=hash_value)


def test_correction_receipt_rejects_changed_registration_runtime_suite_or_sources() -> (
    None
):
    receipt = {
        "registration_sha256": "r" * 64,
        "runtime_binding_sha256": "b" * 64,
        "suite_sha256": "u" * 64,
        "corrected_source_hashes": {"launcher.py": "s" * 64},
        "games_completed_before_correction": 0,
    }
    valid = {
        "registration_sha256": "r" * 64,
        "runtime_binding_sha256": "b" * 64,
        "suite_sha256": "u" * 64,
        "launcher.py": "s" * 64,
    }
    corrected.verify_correction_bindings(
        receipt,
        registration_sha256=valid["registration_sha256"],
        runtime_binding_sha256=valid["runtime_binding_sha256"],
        suite_sha256=valid["suite_sha256"],
        source_hashes={"launcher.py": valid["launcher.py"]},
    )
    for field, value, error in (
        ("registration_sha256", "x" * 64, "registration_mismatch"),
        ("runtime_binding_sha256", "x" * 64, "runtime_binding_mismatch"),
        ("suite_sha256", "x" * 64, "suite_mismatch"),
    ):
        altered = {**valid, field: value}
        with pytest.raises(ValueError, match=error):
            corrected.verify_correction_bindings(
                receipt,
                registration_sha256=altered["registration_sha256"],
                runtime_binding_sha256=altered["runtime_binding_sha256"],
                suite_sha256=altered["suite_sha256"],
                source_hashes={"launcher.py": valid["launcher.py"]},
            )
    with pytest.raises(ValueError, match="source_binding_mismatch"):
        corrected.verify_correction_bindings(
            receipt,
            registration_sha256=valid["registration_sha256"],
            runtime_binding_sha256=valid["runtime_binding_sha256"],
            suite_sha256=valid["suite_sha256"],
            source_hashes={"launcher.py": "x" * 64},
        )


def test_correction_protocol_gate_rejects_seed_or_search_changes() -> None:
    registration = json.loads((DATA / "registration.json").read_text())
    receipt = {
        "protocol_identity": registration["evaluation"],
        "protocol_unchanged": True,
    }
    correction_verifier.verify_protocol_identity(receipt, registration)
    changed = json.loads(json.dumps(registration))
    changed["evaluation"]["seed"] = 430
    with pytest.raises(ValueError, match="protocol_seed_or_search_change"):
        correction_verifier.verify_protocol_identity(receipt, changed)


def test_corrected_resume_rejects_duplicate_game_rows() -> None:
    row = _finished_row(0, 0)
    chunk = {
        "lane": "A",
        "start": 0,
        "count": 2,
        "registration_sha256": "r" * 64,
        "runtime_binding_sha256": "b" * 64,
        "suite_sha256": frozen.sha256_file(DATA / "openings.jsonl"),
        "correction_receipt_sha256": "c" * 64,
        "correction_source_hashes_sha256": "s" * 64,
        "seed": 429,
        "seed_contract": "azlite_eval_seed_v2",
        "challenger_simulations": 384,
        "current_simulations": 384,
        "c_puct": 1.25,
        "chunk_size": 32,
        "worker_result": {"game_entries": [row, row]},
    }
    path = ROOT / ".tmp/seed429-duplicate-outcome-fixture.json"
    try:
        path.write_text(json.dumps(chunk))
        expected = {
            "lane": "A",
            "start": 0,
            "count": 2,
            "registration_sha256": "r" * 64,
            "runtime_binding_sha256": "b" * 64,
            "suite_sha256": frozen.sha256_file(DATA / "openings.jsonl"),
            "correction_receipt_sha256": "c" * 64,
            "correction_source_hashes_sha256": "s" * 64,
            "seed": 429,
            "seed_contract": "azlite_eval_seed_v2",
            "challenger_simulations": 384,
            "current_simulations": 384,
            "c_puct": 1.25,
            "chunk_size": 32,
        }
        with pytest.raises(ValueError, match="duplicate_or_seat_mismatch"):
            corrected._load_chunk(
                path,
                expected=expected,
                source_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    finally:
        path.unlink(missing_ok=True)


def test_relocated_read_only_complete_publication_and_tamper_rejection() -> None:
    fixture = Path(
        tempfile.mkdtemp(prefix="seed429-complete-relocation-", dir=ROOT / ".tmp")
    )
    try:
        registration = json.loads((DATA / "registration.json").read_text())
        correction_receipt = json.loads(
            (DATA / "arena-correction-receipt.json").read_text()
        )
        proof = json.loads((DATA / "opening-exclusion-proof.json").read_text())
        paths = {
            "docs/data/seed429-canonical-policy-normalization",
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
        paths.update(registration["execution_source_inventory"])
        paths.update(correction_receipt["corrected_source_hashes"])
        for relative in sorted(paths):
            source = ROOT / relative
            destination = fixture / relative
            if source.is_dir():
                if not destination.exists():
                    shutil.copytree(source, destination)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(source, destination)
        valid = correction_verifier.verify_correction(fixture, require_complete=True)
        assert valid["valid"] and valid["complete"]
        assert valid["decision"] == "stop_normalization_branch"
        relocated_suite = fixture / registration["suite_path"]
        relocated_suite.write_bytes(relocated_suite.read_bytes() + b"tamper")
        with pytest.raises(ValueError):
            correction_verifier.verify_correction(fixture, require_complete=True)
    finally:
        shutil.rmtree(fixture)
