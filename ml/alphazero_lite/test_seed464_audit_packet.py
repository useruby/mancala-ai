"""Adversarial entry-point tests for the portable seed464 audit packet."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ml.alphazero_lite.verify_seed464_audit_packet import verify

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "docs/data/seed464-dependency-closure-audit/packet"
VERIFIER = ROOT / "ml/alphazero_lite/verify_seed464_audit_packet.py"


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _copy_packet(destination: Path) -> Path:
    # Hardlinks keep test fixtures storage-cheap; replacement edits below never
    # write through to the preserved source packet.
    def link_or_copy(source: str, target: str) -> str:
        try:
            os.link(source, target)
            return target
        except OSError:
            return shutil.copy2(source, target)

    shutil.copytree(PACKET, destination, copy_function=link_or_copy)
    return destination


def _manifest(packet: Path) -> dict:
    path = packet / "packet-manifest.json"
    return json.loads(path.read_text())


def _save_manifest(packet: Path, manifest: dict) -> None:
    path = packet / "packet-manifest.json"
    temporary = path.with_suffix(".replacement")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _replace(packet: Path, relative: str, data: bytes) -> None:
    path = packet / relative
    temporary = path.with_name(path.name + ".replacement")
    temporary.write_bytes(data)
    temporary.replace(path)


def _member(
    manifest: dict, *, original: str | None = None, portable: str | None = None
) -> dict:
    return next(
        item
        for item in manifest["members"]
        if (original is not None and item["original_path"] == original)
        or (portable is not None and item["portable_path"] == portable)
    )


def _rebind_member(packet: Path, member: dict, raw: bytes) -> None:
    archived = (
        gzip.compress(raw, compresslevel=9, mtime=0)
        if member["archive_encoding"] == "gzip"
        else raw
    )
    _replace(packet, member["portable_path"], archived)
    member["sha256"] = _hash(raw)
    member["actual_sha256"] = _hash(raw)
    if member["identity_class"] == "prospective":
        member["expected_sha256"] = _hash(raw)
    member["archived_sha256"] = _hash(archived)


def _tree_state(root: Path) -> dict[str, tuple[str, int, int]]:
    result = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        data = path.read_bytes()
        stat = path.stat()
        result[str(path.relative_to(root))] = (
            _hash(data),
            stat.st_size,
            stat.st_mtime_ns,
        )
    return result


@pytest.fixture
def packet_copy(tmp_path: Path) -> Path:
    return _copy_packet(tmp_path / "packet")


def test_verifier_accepts_complete_packet_without_modifying_it(
    packet_copy: Path,
) -> None:
    source_before = _tree_state(PACKET)
    before = _tree_state(packet_copy)
    result = verify(packet_copy)
    assert result["status"] == "verified"
    assert result["classification"] == "invalid_attempt_dependency_closure_incomplete"
    assert result["registered_decision"] is None
    assert _tree_state(packet_copy) == before
    assert _tree_state(PACKET) == source_before


def test_missing_member_rejected_without_modifying_remaining_evidence(
    packet_copy: Path,
) -> None:
    (
        packet_copy
        / "docs/data/seed464-dependency-closure-audit/dependency-closure-report.md"
    ).unlink()
    before = _tree_state(packet_copy)
    with pytest.raises(ValueError, match="missing_packet_member"):
        verify(packet_copy)
    assert _tree_state(packet_copy) == before


def test_altered_member_without_manifest_rebinding_rejected(packet_copy: Path) -> None:
    relative = (
        "docs/data/seed464-dependency-closure-audit/dependency-closure-report.json"
    )
    _replace(packet_copy, relative, (packet_copy / relative).read_bytes() + b" ")
    before = _tree_state(packet_copy)
    with pytest.raises(ValueError, match="archived_member_hash_mismatch"):
        verify(packet_copy)
    assert _tree_state(packet_copy) == before


def test_altered_original_registration_rejected_after_member_rebinding(
    packet_copy: Path,
) -> None:
    manifest = _manifest(packet_copy)
    member = _member(
        manifest,
        original="docs/data/seed464-historical-value-removal/registration.json",
    )
    raw = (packet_copy / member["portable_path"]).read_bytes() + b" "
    _rebind_member(packet_copy, member, raw)
    _save_manifest(packet_copy, manifest)
    before = _tree_state(packet_copy)
    with pytest.raises(ValueError, match="original_registration_identity_mismatch"):
        verify(packet_copy)
    assert _tree_state(packet_copy) == before


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (
            "docs/data/seed416-policy-target-softening/registration-v3.json",
            "seed416_registration_binding_mismatch",
        ),
        (
            "docs/data/seed462-historical-policy-removal/registration.json",
            "direct_input_mismatch",
        ),
    ],
)
def test_altered_nested_registration_rejected_after_rebinding(
    packet_copy: Path, path: str, message: str
) -> None:
    manifest = _manifest(packet_copy)
    member = _member(manifest, original=path)
    raw = (packet_copy / member["portable_path"]).read_bytes() + b" "
    _rebind_member(packet_copy, member, raw)
    _save_manifest(packet_copy, manifest)
    before = _tree_state(packet_copy)
    with pytest.raises(ValueError, match=message):
        verify(packet_copy)
    assert _tree_state(packet_copy) == before


def test_altered_registered_source_snapshot_rejected_after_rebinding(
    packet_copy: Path,
) -> None:
    manifest = _manifest(packet_copy)
    member = _member(
        manifest,
        portable="source-snapshots/ml/alphazero_lite/seed435_adam_direction.py",
    )
    raw = (packet_copy / member["portable_path"]).read_bytes() + b"\n# changed\n"
    _rebind_member(packet_copy, member, raw)
    _save_manifest(packet_copy, manifest)
    before = _tree_state(packet_copy)
    with pytest.raises(ValueError, match="registered_source_snapshot_mismatch"):
        verify(packet_copy)
    assert _tree_state(packet_copy) == before


def test_altered_derivative_rejected_after_member_rebinding(packet_copy: Path) -> None:
    manifest = _manifest(packet_copy)
    member = _member(manifest, portable="derivatives/A/fresh.jsonl.gz")
    path = packet_copy / member["portable_path"]
    raw = gzip.decompress(path.read_bytes()) + b"\n"
    _rebind_member(packet_copy, member, raw)
    _save_manifest(packet_copy, manifest)
    before = _tree_state(packet_copy)
    with pytest.raises(ValueError, match="derivative_commitment_mismatch:fresh"):
        verify(packet_copy)
    assert _tree_state(packet_copy) == before


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        ("binding", "reported_transitive_binding_mismatch"),
        ("helper_chain", "helper_binding_chain_claim_mismatch"),
        ("classification", "reported_classification_mismatch"),
    ],
)
def test_rebound_audit_report_semantics_rejected(
    packet_copy: Path, mutate: str, message: str
) -> None:
    manifest = _manifest(packet_copy)
    member = _member(
        manifest,
        original="docs/data/seed464-dependency-closure-audit/dependency-closure-report.json",
    )
    report_path = packet_copy / member["portable_path"]
    report = json.loads(report_path.read_text())
    if mutate == "binding":
        report["specified_dependencies"][0]["frozen_expected_input_sha256"] = "0" * 64
    elif mutate == "helper_chain":
        report["imported_helper_sources"][0]["transitive_commitment"] = "invented"
    else:
        report["classification"] = "valid"
    raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    _rebind_member(packet_copy, member, raw)
    audit_receipt_member = _member(
        manifest,
        original="docs/data/seed464-dependency-closure-audit/dependency-closure-receipt.json",
    )
    audit_receipt_path = packet_copy / audit_receipt_member["portable_path"]
    audit_receipt = json.loads(audit_receipt_path.read_text())
    audit_receipt["report_sha256"] = _hash(raw)
    _rebind_member(
        packet_copy,
        audit_receipt_member,
        (json.dumps(audit_receipt, indent=2, sort_keys=True) + "\n").encode(),
    )
    _save_manifest(packet_copy, manifest)
    before = _tree_state(packet_copy)
    with pytest.raises(ValueError, match=message):
        verify(packet_copy)
    assert _tree_state(packet_copy) == before


def test_cli_runs_from_unrelated_directory_without_original_paths(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "unrelated-copy"
    checkout.mkdir()
    packet_copy = shutil.copytree(PACKET, checkout / "packet")
    copied_verifier = checkout / "verify_seed464_audit_packet.py"
    shutil.copyfile(VERIFIER, copied_verifier)
    unrelated_cwd = tmp_path / "elsewhere"
    unrelated_cwd.mkdir()
    before = _tree_state(checkout)
    completed = subprocess.run(
        [sys.executable, str(copied_verifier), "--packet-root", str(packet_copy)],
        cwd=unrelated_cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["status"] == "verified"
    assert _tree_state(checkout) == before
