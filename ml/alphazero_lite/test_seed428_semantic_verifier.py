"""Semantic mutation tests using refreshed derived bindings and authority."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
from unittest import mock

import pytest

from ml.alphazero_lite import verify_seed428_supplemental as verifier
from ml.alphazero_lite.seed427_validation_metrics import row_losses

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("docs/data/seed426-canonical-overlap")


@pytest.fixture
def publication(tmp_path: Path) -> Path:
    root = tmp_path / "relocated"
    data = root / DATA
    data.mkdir(parents=True)
    for name in (
        "seed427-evaluation-manifest.json",
        "seed427-evaluation-results.json",
        "seed427-prediction-evidence.jsonl.gz",
        "seed428-semantic-verification-receipt.json",
    ):
        shutil.copy2(ROOT / DATA / name, data / name)
    for relative in (
        "docs/data/seed416-policy-target-softening/registration-v3.json",
        "docs/data/seed416-policy-target-softening/training-results.json",
        "ml/alphazero_lite/verify_seed428_supplemental.py",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return root


def _refresh(receipt: Path, root: Path) -> None:
    data = root / DATA

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    record = json.loads(receipt.read_text())
    record["manifest_sha256"] = digest(data / "seed427-evaluation-manifest.json")
    record["results_sha256"] = digest(data / "seed427-evaluation-results.json")
    record["prediction_evidence_sha256"] = digest(
        data / "seed427-prediction-evidence.jsonl.gz"
    )
    receipt.write_text(json.dumps(record))


def test_authoritative_checkpoint_hashes_not_just_equality(publication: Path) -> None:
    data = publication / DATA
    result_path = data / "seed427-evaluation-results.json"
    results = json.loads(result_path.read_text())
    results["checkpoint_sha256_before"] = {"e4": "f" * 64, "initializer": "e" * 64}
    results["checkpoint_sha256_after"] = dict(results["checkpoint_sha256_before"])
    result_path.write_text(json.dumps(results))
    _refresh(data / "seed428-semantic-verification-receipt.json", publication)
    with mock.patch.object(
        verifier, "verify_seed427", return_value={"status": "verified"}
    ):
        with pytest.raises(ValueError, match="authoritative_checkpoint_hash_mismatch"):
            verifier.verify(publication)


def test_fabricated_failed_parity_is_rejected(publication: Path) -> None:
    data = publication / DATA
    result_path = data / "seed427-evaluation-results.json"
    results = json.loads(result_path.read_text())
    results["e4_full_validation_parity"]["policy_loss"] = {
        "actual": 99.0,
        "expected": 0.9556283950805664,
        "passed": False,
    }
    result_path.write_text(json.dumps(results))
    _refresh(data / "seed428-semantic-verification-receipt.json", publication)
    with mock.patch.object(
        verifier, "verify_seed427", return_value={"status": "verified"}
    ):
        with pytest.raises(
            ValueError, match="full_validation_parity_semantic_mismatch"
        ):
            verifier.verify(publication)


def test_empty_pit_mask_is_rejected_after_refreshing_evidence_hash(
    publication: Path,
) -> None:
    data = publication / DATA
    evidence = data / "seed427-prediction-evidence.jsonl.gz"
    rows = [
        json.loads(line) for line in gzip.decompress(evidence.read_bytes()).splitlines()
    ]
    row = rows[0]
    encoded = list(
        struct.unpack("<27f", bytes.fromhex(row["membership"]["input_identity"]))
    )
    legal = [int(value > 0) for value in encoded[:6]]
    illegal_pit = legal.index(0) if 0 in legal else 0
    row["legal_mask"] = list(row["legal_mask"])
    row["legal_mask"][illegal_pit] = 1 - row["legal_mask"][illegal_pit]
    row["losses"] = row_losses(
        row["logits"],
        row["value_prediction"],
        row["target_policy"],
        row["target_value"],
        row["legal_mask"],
        row["policy_weight"],
    )
    evidence.write_bytes(
        gzip.compress(
            b"".join(
                json.dumps(item, separators=(",", ":")).encode() + b"\n"
                for item in rows
            ),
            mtime=0,
        )
    )
    _refresh(data / "seed428-semantic-verification-receipt.json", publication)
    with mock.patch.object(
        verifier, "verify_seed427", return_value={"status": "verified"}
    ):
        with pytest.raises(ValueError, match="prediction_legal_mask_rule_mismatch"):
            verifier.verify(publication)


def test_nonfinite_prediction_rejected_after_refreshing_receipt(
    publication: Path,
) -> None:
    data = publication / DATA
    evidence = data / "seed427-prediction-evidence.jsonl.gz"
    rows = [
        json.loads(line) for line in gzip.decompress(evidence.read_bytes()).splitlines()
    ]
    rows[0]["value_prediction"] = float("inf")
    evidence.write_bytes(
        gzip.compress(
            b"".join(
                json.dumps(item, separators=(",", ":")).encode() + b"\n"
                for item in rows
            ),
            mtime=0,
        )
    )
    _refresh(data / "seed428-semantic-verification-receipt.json", publication)
    with mock.patch.object(
        verifier, "verify_seed427", return_value={"status": "verified"}
    ):
        with pytest.raises(ValueError, match="nonfinite_prediction_or_target"):
            verifier.verify(publication)


def test_weighted_copies_must_have_consistent_model_outputs(publication: Path) -> None:
    data = publication / DATA
    evidence = data / "seed427-prediction-evidence.jsonl.gz"
    rows = [
        json.loads(line) for line in gzip.decompress(evidence.read_bytes()).splitlines()
    ]
    seen: dict[tuple[str, int], int] = {}
    pair: tuple[int, int] | None = None
    for index, item in enumerate(rows):
        key = (item["model"], item["membership"]["compact_row"])
        if key in seen:
            pair = (seen[key], index)
            break
        seen[key] = index
    assert pair is not None
    changed = rows[pair[1]]
    changed["logits"][0] += 0.1
    changed["losses"] = row_losses(
        changed["logits"],
        changed["value_prediction"],
        changed["target_policy"],
        changed["target_value"],
        changed["legal_mask"],
        changed["policy_weight"],
    )
    evidence.write_bytes(
        gzip.compress(
            b"".join(
                json.dumps(item, separators=(",", ":")).encode() + b"\n"
                for item in rows
            ),
            mtime=0,
        )
    )
    _refresh(data / "seed428-semantic-verification-receipt.json", publication)
    with mock.patch.object(
        verifier, "verify_seed427", return_value={"status": "verified"}
    ):
        with pytest.raises(ValueError, match="weighted_copy_output_mismatch"):
            verifier.verify(publication)
