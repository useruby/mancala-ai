"""Semantic and portability gates for the append-only seed430 correction."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ml.alphazero_lite import verify_seed430

ROOT = Path(__file__).resolve().parents[2]
DATA_REL = Path("docs/data/seed429-canonical-policy-normalization")


def test_actual_supplemental_entrypoint_accepts_publication() -> None:
    result = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-m", "ml.alphazero_lite.verify_seed430"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["decision"] == "stop_normalization_branch"
    assert payload["reconciliation"]["chunks"] == 47
    assert payload["reconciliation"]["pre_repair_games"] == 1504


def test_semantic_checks_reject_refreshed_report_hash(tmp_path: Path) -> None:
    data = ROOT / DATA_REL
    report_path = data / "arena-evidence/A/A-arena.json"
    outcome_path, analysis_path = data / "outcome-binding.json", data / "analysis.json"
    originals = {p: p.read_bytes() for p in (report_path, outcome_path, analysis_path)}
    try:
        report = json.loads(originals[report_path])
        report["score"] += 0.01
        report_path.write_text(json.dumps(report))
        outcome = json.loads(originals[outcome_path])
        outcome["lanes"]["A"]["report_sha256"] = verify_seed430.sha(report_path)
        outcome_path.write_text(json.dumps(outcome))
        analysis = json.loads(originals[analysis_path])
        analysis["outcome_binding_sha256"] = verify_seed430.sha(outcome_path)
        analysis_path.write_text(json.dumps(analysis))
        with pytest.raises(ValueError, match="A_report_score_invalid"):
            verify_seed430.verify(ROOT)
    finally:
        for path, content in originals.items():
            path.write_bytes(content)


@pytest.mark.parametrize("field", ["registration_sha256", "runtime_binding_sha256"])
def test_outcome_binding_identity_is_semantic(field: str) -> None:
    path = ROOT / DATA_REL / "outcome-binding.json"
    original = path.read_bytes()
    try:
        outcome = json.loads(original)
        outcome[field] = "0" * 64
        path.write_text(json.dumps(outcome))
        with pytest.raises(
            ValueError,
            match=f"outcome_{'runtime_binding' if field == 'runtime_binding_sha256' else 'registration'}_mismatch",
        ):
            verify_seed430.verify(ROOT)
    finally:
        path.write_bytes(original)


def test_reconciliation_claims_are_portable_and_counts_checked(tmp_path: Path) -> None:
    data = tmp_path / DATA_REL
    data.mkdir(parents=True)
    for name in (
        "arena-resume-reconciliation.json",
        "registration.json",
        "runtime-binding.json",
        "arena-correction-receipt.json",
        "training-results.json",
    ):
        shutil.copy2(ROOT / DATA_REL / name, data / name)
    reg = json.loads((data / "registration.json").read_text())
    runtime = json.loads((data / "runtime-binding.json").read_text())
    correction = json.loads((data / "arena-correction-receipt.json").read_text())
    portable = verify_seed430.verify_portable_reconciliation(
        tmp_path, reg, runtime, correction
    )
    assert portable["byte_level_chunk_claims"].startswith("unverifiable")
    receipt_path = data / "arena-resume-reconciliation.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["chunks"].pop()
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="chunk_count_invalid"):
        verify_seed430.verify_portable_reconciliation(
            tmp_path, reg, runtime, correction
        )


def test_relocated_publication_from_unrelated_cwd_is_read_only(tmp_path: Path) -> None:
    registration = json.loads((ROOT / DATA_REL / "registration.json").read_text())
    proof = json.loads((ROOT / DATA_REL / "opening-exclusion-proof.json").read_text())
    correction = json.loads(
        (ROOT / DATA_REL / "seed430-verification-correction.json").read_text()
    )
    paths = {
        str(DATA_REL),
        "docs/data/seed416-policy-target-softening/registration-v3.json",
        "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz",
        "docs/data/seed422-adam-first-moment/opening-exclusion-proof.json",
        "docs/data/seed422-adam-first-moment/openings.jsonl",
        "docs/data/seed422-adam-first-moment/outcome-ledger.jsonl",
        *registration["execution_source_inventory"],
        *correction["supplemental_sources"],
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
    relocated = tmp_path / "relocated"
    for relative in sorted(paths):
        source, target = ROOT / relative, relocated / relative
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    publication = relocated / DATA_REL
    before = {
        p.relative_to(publication): verify_seed430.sha(p)
        for p in publication.rglob("*")
        if p.is_file()
    }
    unrelated = tmp_path / "unrelated-working-directory"
    unrelated.mkdir()
    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "ml.alphazero_lite.verify_seed430",
            "--root",
            str(relocated),
        ],
        cwd=unrelated,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout)["decision"] == "stop_normalization_branch"
    after = {
        p.relative_to(publication): verify_seed430.sha(p)
        for p in publication.rglob("*")
        if p.is_file()
    }
    assert before == after
