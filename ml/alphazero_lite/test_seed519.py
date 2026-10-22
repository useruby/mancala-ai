"""Real-path regression tests for the seed519 publication verifier."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.alphazero_lite import verify_seed519  # noqa: E402

SOURCE = verify_seed519.PUBLICATION


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def publication(tmp_path: Path) -> Path:
    copied = tmp_path / "seed518"
    shutil.copytree(SOURCE, copied)
    return copied


def test_real_supplement_verifier_accepts_unchanged_publication() -> None:
    assert verify_seed519.verify()["status"] == "verified"


def test_rejects_published_parity_byte_change(publication: Path) -> None:
    path = publication / "parity-outputs.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="seed519_receipt_parity_hash"):
        verify_seed519.verify(publication)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("seed", -1, "seed518_parity_seed"),
        ("evaluator", {"mutated": True}, "seed518_evaluator_output_mismatch"),
        ("search", {"mutated": True}, "seed518_search_output_mismatch"),
    ],
)
def test_rejects_semantic_parity_mutations(
    publication: Path, field: str, value: object, reason: str
) -> None:
    parity_path = publication / "parity-outputs.json"
    rows = json.loads(parity_path.read_text())
    row = rows[0]
    if field == "seed":
        row[field] = value
    elif field == "evaluator":
        row["evaluator_outputs"]["selective_P"] = value
    else:
        row["search_outputs"]["selective_P"] = value
    parity_path.write_text(json.dumps(rows))
    rebind_publication(publication, parity=True)
    with pytest.raises(ValueError, match=reason):
        verify_seed519.verify(publication)


def test_rejects_bad_timing_after_receipt_rebind(publication: Path) -> None:
    raw_path = publication / "raw-measurements.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    rows[0]["seconds"] = float("nan")
    raw_path.write_text("".join(json.dumps(row, allow_nan=True) + "\n" for row in rows))
    rebind_publication(publication)
    with pytest.raises(ValueError, match="seed518_invalid_timing"):
        verify_seed519.verify(publication)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing", "seed518_measurement_coverage_duplicate"),
        ("duplicate", "seed518_measurement_coverage_duplicate"),
        ("seed", "seed518_measurement_seed"),
        ("order", "seed518_alternating_order"),
        ("bucket_route", "seed518_bucket_routing"),
        ("path", "seed518_search_path_routing"),
        ("puct_simulations", "seed518_puct_simulation_accounting"),
        ("exact_simulations", "seed519_exact_root_simulation_accounting"),
        ("zero", "seed518_invalid_timing"),
        ("infinity", "seed518_invalid_timing"),
        ("summary", "seed518_summary_arithmetic"),
        ("aggregate", "seed518_summary_arithmetic"),
        ("per_state", "seed518_summary_arithmetic"),
        ("bucket_summary", "seed518_summary_arithmetic"),
        ("decision", "seed518_summary_arithmetic"),
    ],
)
def test_measurement_and_summary_mutations(
    publication: Path, mutation: str, reason: str
) -> None:
    raw_path = publication / "raw-measurements.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0].copy())
    elif mutation == "seed":
        rows[0]["seed"] += 1
    elif mutation == "order":
        rows[0]["order_position"] = 1 - rows[0]["order_position"]
    elif mutation == "bucket_route":
        rows[0]["bucket"] = "le16" if rows[0]["bucket"] != "le16" else "gt32"
    elif mutation == "path":
        rows[0]["dispatch_path"] = (
            "exact_root" if rows[0]["dispatch_path"] == "puct" else "puct"
        )
    elif mutation == "puct_simulations":
        row = next(row for row in rows if row["dispatch_path"] == "puct")
        row["puct_simulations_executed"] = 1199
    elif mutation == "exact_simulations":
        row = next(row for row in rows if row["dispatch_path"] == "exact_root")
        row["puct_simulations_executed"] = 1
    elif mutation in {"zero", "infinity"}:
        rows[0]["seconds"] = 0 if mutation == "zero" else float("inf")
    raw_path.write_text("".join(json.dumps(row, allow_nan=True) + "\n" for row in rows))
    if mutation in {
        "summary",
        "aggregate",
        "per_state",
        "bucket_summary",
        "decision",
    }:
        evidence_path = publication / "evidence.json"
        evidence = json.loads(evidence_path.read_text())
        if mutation == "summary":
            evidence["summary"]["configs"]["reference_P"]["count"] += 1
        elif mutation == "aggregate":
            evidence["summary"]["configs"]["reference_P"]["p50_seconds"] += 0.01
        elif mutation == "per_state":
            evidence["summary"]["per_state"][0]["selective_over_reference"] += 0.01
        elif mutation == "bucket_summary":
            evidence["summary"]["buckets"]["gt32"]["puct"]["paired_ratio_p50"] += 0.01
        else:
            evidence["summary"]["decision"]["minimum_improvement"] = 0.06
        evidence_path.write_text(json.dumps(evidence))
    rebind_publication(publication)
    with pytest.raises(ValueError, match=reason):
        verify_seed519.verify(publication)


@pytest.mark.parametrize(
    ("artifact", "reason"),
    [
        ("receipt", "seed519_receipt_freeze_hash"),
        ("evidence", "seed519_receipt_evidence_hash"),
        ("parity", "seed519_receipt_parity_hash"),
        ("raw", "seed519_receipt_raw_hash"),
        ("reanalysis", "seed519_receipt_reanalysis_hash"),
        ("freeze", "seed519_receipt_freeze_hash"),
        ("amendment", "seed519_freeze_amendment_hash"),
    ],
)
def test_rejects_broken_publication_links(
    publication: Path, artifact: str, reason: str
) -> None:
    path = {
        "receipt": publication / "seed518-receipt.json",
        "evidence": publication / "evidence.json",
        "parity": publication / "parity-outputs.json",
        "raw": publication / "raw-measurements.jsonl",
        "amendment": publication / "comparison-contract-amendment.json",
        "reanalysis": publication / "post-execution-reanalysis.json",
        "freeze": publication / "amended-timing-freeze.json",
    }[artifact]
    if artifact == "receipt":
        data = json.loads(path.read_text())
        data["identities"]["amended_freeze_sha256"] = "0" * 64
        path.write_text(json.dumps(data))
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match=reason):
        verify_seed519.verify(publication)


@pytest.mark.parametrize(
    ("target", "reason"),
    [
        *(
            (f"component:{component}:{name}", "seed519_registration_component")
            for component in ("initializer", "E4")
            for name in (
                "model.npz",
                "weights.json",
                "metadata.json",
                "search_policy.json",
            )
        ),
        ("registration", "seed519_registration_identity"),
        ("source", "seed519_execution_source"),
        ("runtime", "runtime_asset_hash"),
    ],
)
def test_rejects_altered_frozen_identities(
    publication: Path, target: str, reason: str
) -> None:
    freeze_path = publication / "amended-timing-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if target.startswith("component:"):
        _, component, filename = target.split(":")
        freeze["components"][component][filename] = "0" * 64
    elif target == "registration":
        freeze["registration_sha256"] = "0" * 64
    elif target == "runtime":
        freeze["runtime_identities"]["exact_root_native_probe"]["sha256"] = "0" * 64
    else:
        freeze["source_sha256"]["ml/alphazero_lite/arena.py"] = "0" * 64
    freeze_path.write_text(json.dumps(freeze))
    rebind_publication(publication)
    evidence_path = publication / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["protocol_freeze_sha256"] = sha(freeze_path)
    evidence_path.write_text(json.dumps(evidence))
    receipt_path = publication / "seed518-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["identities"]["amended_freeze_sha256"] = sha(freeze_path)
    receipt["identities"]["evidence_sha256"] = sha(evidence_path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=reason):
        verify_seed519.verify(publication)


def rebind_publication(directory: Path, *, parity: bool = False) -> None:
    """Rebind integrity links deliberately so semantic checks are reached."""
    receipt_path = directory / "seed518-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    evidence_path = directory / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    freeze_path = directory / "amended-timing-freeze.json"
    reanalysis_path = directory / "post-execution-reanalysis.json"
    reanalysis = json.loads(reanalysis_path.read_text())
    parity_path = directory / "parity-outputs.json"
    raw_path = directory / "raw-measurements.jsonl"
    if parity:
        parity_hash = sha(parity_path)
        evidence["parity_sha256"] = parity_hash
        reanalysis["parity_archive_sha256"] = parity_hash
        reanalysis_path.write_text(json.dumps(reanalysis))
    evidence["protocol_freeze_sha256"] = sha(freeze_path)
    evidence["parity_sha256"] = sha(parity_path)
    evidence["post_execution_reanalysis_sha256"] = sha(reanalysis_path)
    evidence_path.write_text(json.dumps(evidence))
    receipt["identities"].update(
        {
            "amended_freeze_sha256": sha(freeze_path),
            "evidence_sha256": sha(evidence_path),
            "parity_archive_sha256": sha(parity_path),
            "raw_measurements_sha256": sha(raw_path),
            "post_execution_reanalysis_sha256": sha(reanalysis_path),
        }
    )
    # amendment file remains independently bound by the freeze
    receipt_path.write_text(json.dumps(receipt))
