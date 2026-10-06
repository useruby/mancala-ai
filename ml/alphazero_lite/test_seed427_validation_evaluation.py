"""Focused tests for the frozen seed427 evaluation and its portable evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest
import torch

from ml.alphazero_lite import verify_seed427_evaluation
from ml.alphazero_lite.run_seed427_validation_evaluation import (
    ABS_TOL,
    CHECKPOINTS,
    REL_TOL,
    forward_chunks,
    load_compact_targets,
    make_model,
)
from ml.alphazero_lite.seed427_validation_metrics import aggregate, row_losses
from ml.alphazero_lite.seed427_validation_subsets import construct, read_evidence

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("docs/data/seed426-canonical-overlap")


def _sample(identity: str, policy: float, value: float, weight: float = 1.0) -> dict:
    return {
        "membership": {
            "subset": "unseen",
            "active_stones": 40,
            "canonical_identity": identity,
        },
        "losses": {
            "policy_loss": policy,
            "value_loss": value,
            "total_loss": policy + 0.3 * value,
            "policy_weight": weight,
        },
    }


def test_legal_masked_loss_and_huber_match_torch() -> None:
    logits = [0.2, -0.4, 1000.0, 0.6, -2.0, 0.1]
    target = [0.25, 0.75, 0.0, 0.0, 0.0, 0.0]
    actual = row_losses(logits, 0.4, target, -0.2, [1, 1, 0, 1, 0, 1], 1.0)
    masked = torch.tensor(logits).masked_fill(
        torch.tensor([0, 0, 1, 0, 1, 0], dtype=torch.bool), -1e9
    )
    expected_policy = torch.nn.functional.cross_entropy(
        masked.unsqueeze(0), torch.tensor(target).unsqueeze(0), reduction="none"
    ).item()
    expected_value = torch.nn.functional.smooth_l1_loss(
        torch.tensor([0.4]), torch.tensor([-0.2]), beta=1, reduction="none"
    ).item()
    assert actual["policy_loss"] == pytest.approx(expected_policy, abs=1e-7)
    assert actual["value_loss"] == pytest.approx(expected_value, abs=1e-7)
    changed_illegal = row_losses(
        [*logits[:2], -900.0, *logits[3:]],
        0.4,
        target,
        -0.2,
        [1, 1, 0, 1, 0, 1],
        1.0,
    )
    assert changed_illegal["policy_loss"] == pytest.approx(actual["policy_loss"])


def test_weighted_accounting_and_equal_identity_averaging() -> None:
    rows = [
        _sample("id-a", 1.0, 0.2),
        _sample("id-a", 3.0, 0.4),
        _sample("id-b", 5.0, 0.8),
    ]
    summary = aggregate(rows)["unseen/>32"]
    assert summary["weighted_positions"] == 3
    assert summary["canonical_identity_denominator"] == 2
    assert summary["exposure_weighted"]["policy_loss"] == pytest.approx(3.0)
    assert summary["equal_canonical_identity"]["policy_loss"] == pytest.approx(3.5)
    assert summary["equal_canonical_identity"]["value_loss"] == pytest.approx(0.55)
    weighted = [_sample("id-a", 1.0, 0.2, 2), _sample("id-b", 3.0, 0.4, 1)]
    assert aggregate(weighted)["unseen/>32"]["policy_weight_denominator"] == 3


def test_empty_subsets_are_finite_and_zero_denominator() -> None:
    empty = aggregate([])["unseen/>32"]
    assert empty["weighted_positions"] == 0
    assert empty["policy_weight_denominator"] == 0
    assert empty["equal_canonical_identity"] == {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "total_loss": 0.0,
    }


def test_forward_chunk_size_invariance() -> None:
    registration = json.loads(
        (
            ROOT / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    model, _identity = make_model(registration, CHECKPOINTS["e4"])
    _, compact = load_compact_targets()
    membership = read_evidence(ROOT / DATA / "seed427-validation-membership.jsonl.gz")
    validation_indexes = sorted({row["compact_row"] for row in membership})[:23]
    inputs = np.stack([compact[index]["state"] for index in validation_indexes])
    whole, whole_values = forward_chunks(model, inputs, 64)
    chunked, chunked_values = forward_chunks(model, inputs, 4)
    np.testing.assert_allclose(whole, chunked, rtol=REL_TOL, atol=ABS_TOL)
    np.testing.assert_allclose(whole_values, chunked_values, rtol=REL_TOL, atol=ABS_TOL)


def test_membership_rebuild_reconciles_published_counts() -> None:
    root_data = ROOT / DATA
    rows = read_evidence(root_data / "row-accounting.jsonl.gz")
    expected, counts = construct(rows)
    actual = read_evidence(root_data / "seed427-validation-membership.jsonl.gz")
    assert expected == actual
    assert counts["all"]["weighted_positions"] == 14946
    assert counts["seen"]["weighted_positions"] == 4240
    assert counts["unseen"]["weighted_positions"] == 10706
    assert (
        counts["input_identity_definition_difference"][
            "validation_positions_canonical_unseen_but_input_seen"
        ]
        == 0
    )


def test_frozen_checkpoint_parity_and_immutability_receipt() -> None:
    receipt = json.loads((ROOT / DATA / "seed427-evaluation-results.json").read_text())
    assert receipt["status"] == "completed"
    assert receipt["checkpoint_sha256_before"] == receipt["checkpoint_sha256_after"]
    for metric, expected in {
        "policy_loss": 0.9556283950805664,
        "value_loss": 0.20779633522033691,
        "total_loss": 1.0179673433303833,
    }.items():
        check = receipt["e4_full_validation_parity"][metric]
        assert check["expected"] == expected
        assert check["passed"]
        assert abs(check["actual"] - expected) <= 1e-6 + 1e-5 * abs(expected)


def test_checkpoint_free_verifier_rejects_altered_bindings(tmp_path: Path) -> None:
    # Avoid replaying Phase 1 in each micro-mutation case; its relocated actual
    # entry-point test separately exercises the complete publication verifier.
    data = ROOT / DATA
    temp_data = tmp_path / DATA
    shutil.copytree(data, temp_data)
    lineage_files = (
        "docs/data/seed416-policy-target-softening/registration-v3.json",
        "docs/data/seed416-policy-target-softening/training-results.json",
        "docs/data/seed422-adam-first-moment/registration.json",
        "docs/data/seed422-adam-first-moment/training-results.json",
        "docs/data/seed422-adam-first-moment/supplemental-verification-receipt.json",
    )
    for relative in lineage_files:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    shutil.copytree(ROOT / DATA / "sources", temp_data / "sources", dirs_exist_ok=True)
    for relative in (
        "ml/alphazero_lite/run_seed427_validation_evaluation.py",
        "ml/alphazero_lite/seed427_validation_metrics.py",
        "ml/alphazero_lite/seed427_validation_subsets.py",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    with mock.patch.object(
        verify_seed427_evaluation,
        "verify_publication",
        return_value={"status": "verified"},
    ):
        for path_name, expected in (
            (
                "seed427-prediction-evidence.jsonl.gz",
                "prediction_evidence_binding_mismatch",
            ),
            ("seed427-validation-membership.jsonl.gz", "membership_binding_mismatch"),
        ):
            target = temp_data / path_name
            original = target.read_bytes()
            try:
                target.write_bytes(original + b"x")
                with pytest.raises(ValueError, match=expected):
                    verify_seed427_evaluation.verify(tmp_path)
            finally:
                target.write_bytes(original)
        manifest_path = temp_data / "seed427-evaluation-manifest.json"
        original_manifest = manifest_path.read_bytes()
        try:
            manifest = json.loads(original_manifest)
            manifest["architecture"]["hidden_sizes"][0] += 1
            manifest_path.write_text(json.dumps(manifest))
            with pytest.raises(
                ValueError, match="evaluation_manifest_binding_mismatch"
            ):
                verify_seed427_evaluation.verify(tmp_path)
        finally:
            manifest_path.write_bytes(original_manifest)

        evidence_path = temp_data / "seed427-prediction-evidence.jsonl.gz"
        receipt_path = temp_data / "seed427-evaluation-results.json"
        original_evidence = evidence_path.read_bytes()
        original_receipt = receipt_path.read_bytes()
        for alteration, expected in (
            ("prediction", "prediction_loss_reconstruction_mismatch"),
            ("target", "prediction_target_binding_mismatch"),
            ("weight", "prediction_weight_binding_mismatch"),
        ):
            rows = [
                json.loads(line)
                for line in gzip.decompress(original_evidence).splitlines()
            ]
            row = rows[0]
            if alteration == "prediction":
                row["logits"][0] += 0.25
            elif alteration == "target":
                row["target_value"] += 0.05
                row["losses"] = row_losses(
                    row["logits"],
                    row["value_prediction"],
                    row["target_policy"],
                    row["target_value"],
                    row["legal_mask"],
                    row["policy_weight"],
                )
            else:
                row["policy_weight"] = 0.5
                row["losses"] = row_losses(
                    row["logits"],
                    row["value_prediction"],
                    row["target_policy"],
                    row["target_value"],
                    row["legal_mask"],
                    row["policy_weight"],
                )
            encoded = b"".join(
                json.dumps(item, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                for item in rows
            )
            evidence_path.write_bytes(gzip.compress(encoded, mtime=0))
            receipt = json.loads(original_receipt)
            receipt["prediction_evidence_sha256"] = hashlib.sha256(
                evidence_path.read_bytes()
            ).hexdigest()
            receipt_path.write_text(json.dumps(receipt))
            with pytest.raises(ValueError, match=expected):
                verify_seed427_evaluation.verify(tmp_path)
        evidence_path.write_bytes(original_evidence)
        receipt_path.write_bytes(original_receipt)


def test_actual_evaluation_verifier_relocated_read_only(tmp_path: Path) -> None:
    root = tmp_path / "relocated checkout with spaces"
    root.mkdir()
    paths = [
        "docs/data/seed426-canonical-overlap",
        "docs/data/seed416-policy-target-softening/registration-v3.json",
        "docs/data/seed416-policy-target-softening/training-results.json",
        "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz",
        "docs/data/seed422-adam-first-moment/registration.json",
        "docs/data/seed422-adam-first-moment/training-results.json",
        "docs/data/seed422-adam-first-moment/supplemental-verification-receipt.json",
        "ml/alphazero_lite/seed426_overlap_analysis.py",
        "ml/alphazero_lite/verify_seed426_overlap_audit.py",
        "ml/alphazero_lite/verify_seed427_publication.py",
        "ml/alphazero_lite/verify_seed427_evaluation.py",
        "ml/alphazero_lite/verify_seed428_supplemental.py",
        "ml/alphazero_lite/seed427_validation_metrics.py",
        "ml/alphazero_lite/seed427_validation_subsets.py",
        "ml/alphazero_lite/fresh_p1_adapter_teacher_audit.py",
        "ml/alphazero_lite/kalah_rules.py",
        "ml/alphazero_lite/run_seed427_validation_evaluation.py",
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/self_play.py",
        "ml/alphazero_lite/fresh_p1_adapter_teacher_audit.py",
        "ml/alphazero_lite/run_seed426_overlap_audit.py",
    ]
    for item in paths:
        source = ROOT / item
        target = root / item
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
    before = {
        path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
    env = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    for entrypoint in (
        "verify_seed427_evaluation.py",
        "verify_seed428_supplemental.py",
    ):
        result = subprocess.run(
            [
                sys.executable,
                str(root / "ml/alphazero_lite" / entrypoint),
                "--root",
                str(root),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["status"] == "verified"
    after = {
        path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert before == after
