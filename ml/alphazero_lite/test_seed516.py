"""Semantic mutation tests for the append-only seed516 correction verifier."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ml.alphazero_lite import verify_seed516 as correction
from ml.alphazero_lite import verify_seed515 as historical

_REBUILT = None
_OCCURRENCES = None


@pytest.fixture(scope="module", autouse=True)
def unchanged_positive_control() -> None:
    """Pass the real unchanged publication before enabling cached replay for mutations."""
    global _REBUILT, _OCCURRENCES
    report = correction.verify()
    assert report["status"] == "verified"
    assert report["portability_execution"] is None
    _REBUILT, _OCCURRENCES = historical.reconstruct(
        json.loads((historical.PUBLICATION / "protocol-freeze.json").read_text())
    )


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    destination = tmp_path / "publication"
    shutil.copytree(historical.PUBLICATION, destination)
    monkeypatch.setattr(correction, "PUBLICATION", destination)
    monkeypatch.setattr(historical, "PUBLICATION", destination)
    monkeypatch.setattr(
        historical, "reconstruct", lambda _freeze: (_REBUILT, _OCCURRENCES)
    )
    return destination


def _json(path: Path):
    return json.loads(path.read_text())


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize(
    ("area", "value", "reason"),
    [
        ("paired", float("nan"), "nonfinite_paired:paired_ratio_p50"),
        ("paired", float("inf"), "nonfinite_paired:paired_ratio_p50"),
        ("state", float("nan"), "nonfinite_per_state_ratio"),
        ("state", float("inf"), "nonfinite_per_state_ratio"),
    ],
)
def test_nonfinite_ratios_are_rejected_by_real_path(
    area: str,
    value: float,
    reason: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _fixture(tmp_path, monkeypatch)
    evidence_path = directory / "evidence.json"
    evidence = _json(evidence_path)
    summary = evidence["summary"]
    if area == "paired":
        summary["paired_ratio_p50"] = value
    else:
        summary["per_state_latency_ratios"][0]["P_over_baseline"] = value
    _write_json(evidence_path, evidence)
    with pytest.raises(ValueError, match=reason):
        correction.verify()


def test_duplicate_per_state_records_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _fixture(tmp_path, monkeypatch)
    evidence_path = directory / "evidence.json"
    evidence = _json(evidence_path)
    evidence["summary"]["per_state_latency_ratios"].append(
        evidence["summary"]["per_state_latency_ratios"][0].copy()
    )
    _write_json(evidence_path, evidence)
    with pytest.raises(ValueError, match="duplicate_per_state_record"):
        correction.verify()


@pytest.mark.parametrize(
    "mutation",
    [
        "swap_configs",
        "swap_strata",
        "wrong_cell",
        "unit",
        "proposal_swap_configs",
        "proposal_swap_strata",
        "proposal_wrong_cell",
        "proposal_unit",
    ],
)
def test_table_semantics_rejected_by_real_path(
    mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _fixture(tmp_path, monkeypatch)
    proposal = mutation.startswith("proposal_")
    results_path = (
        tmp_path / "deployment-proposal.md" if proposal else directory / "results.md"
    )
    if proposal:
        shutil.copyfile(correction.PROPOSAL, results_path)
        monkeypatch.setattr(correction, "PROPOSAL", results_path)
    text = results_path.read_text()
    rows = text.splitlines()
    if proposal:
        offset = 32
        if mutation == "proposal_swap_configs":
            first, second = rows[offset].split("|"), rows[offset + 1].split("|")
            rows[offset] = "|".join(first[:1] + second[1:-1] + [""])
            rows[offset + 1] = "|".join(second[:1] + first[1:-1] + [""])
        elif mutation == "proposal_swap_strata":
            rows[offset + 2], rows[offset + 4] = rows[offset + 4], rows[offset + 2]
            rows[offset + 3], rows[offset + 5] = rows[offset + 5], rows[offset + 3]
        elif mutation == "proposal_wrong_cell":
            rows[offset] = rows[offset].replace("106.2 ms", "999.9 ms", 1)
            rows.append("Control retains expected value 106.2 ms elsewhere.")
        else:
            rows[offset] = rows[offset].replace("106.2 ms", "106.2 sec", 1)
            rows.append("Control retains expected value 106.2 ms elsewhere.")
    elif mutation == "swap_configs":
        first, second = rows[20].split("|"), rows[21].split("|")
        rows[20] = "|".join(first[:3] + second[3:-1] + [""])
        rows[21] = "|".join(second[:3] + first[3:-1] + [""])
    elif mutation == "swap_strata":
        rows[22], rows[24] = rows[24], rows[22]
        rows[23], rows[25] = rows[25], rows[23]
    elif mutation in {"wrong_cell", "unit"}:
        replacement = "999.9 ms" if mutation == "wrong_cell" else "106.2 sec"
        rows[20] = rows[20].replace("106.2 ms", replacement, 1)
        rows.append("Control retains expected value 106.2 ms elsewhere.")
    results_path.write_text("\n".join(rows) + "\n")
    reason = (
        "table_cell_or_unit:proposal" if proposal else "table_cell_or_unit:results.md"
    )
    with pytest.raises(ValueError, match=reason):
        correction.verify()


def test_ordinary_invocation_makes_no_portability_or_preservation_claim() -> None:
    result = correction.verify()
    assert result["portability_execution"] is None
    assert not any("preservation" in key for key in result)
    assert not any("inventory" in key for key in result)


def test_cli_prints_by_default_and_explicit_output_is_create_only(
    tmp_path: Path,
) -> None:
    entrypoint = Path(correction.__file__).resolve()
    environment = os.environ.copy()
    default_run = subprocess.run(
        [sys.executable, str(entrypoint)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert default_run.returncode == 0, default_run.stderr
    assert json.loads(default_run.stdout)["status"] == "verified"

    output = tmp_path / "report.json"
    explicit_run = subprocess.run(
        [sys.executable, str(entrypoint), "--output", str(output)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert explicit_run.returncode == 0, explicit_run.stderr
    assert json.loads(output.read_text())["status"] == "verified"
    repeated_run = subprocess.run(
        [sys.executable, str(entrypoint), "--output", str(output)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert repeated_run.returncode != 0
    assert "FileExistsError" in repeated_run.stderr


def test_lfs_and_materialized_bytes_are_authenticated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(historical, "ROOT", tmp_path)
    pointer = tmp_path / "tablebase.kvtb"
    external = tmp_path / "external-object"
    pointer_bytes = b"version https://git-lfs.github.com/spec/v1\n"
    object_bytes = b"registered tablebase object bytes"
    pointer.write_bytes(pointer_bytes)
    external.write_bytes(object_bytes)
    asset = {
        "artifact": "tablebase.kvtb",
        "source_pointer_sha256": historical.hashlib.sha256(pointer_bytes).hexdigest(),
        "sha256": historical.hashlib.sha256(object_bytes).hexdigest(),
        "runtime_materialized_path": str(external),
        "lfs_oid": historical.hashlib.sha256(object_bytes).hexdigest(),
    }
    frozen = {"sha256": asset["sha256"]}

    # A pointer is accepted only with its matching object, and a local object
    # is independently accepted without consulting the external cache.
    historical.verify_runtime_asset("exact_root_tablebase", asset, frozen)
    pointer.write_bytes(object_bytes)
    historical.verify_runtime_asset("exact_root_tablebase", asset, frozen)

    pointer.write_bytes(b"altered local artifact")
    with pytest.raises(ValueError, match="runtime_tablebase_artifact_hash"):
        historical.verify_runtime_asset("exact_root_tablebase", asset, frozen)
    pointer.unlink()
    with pytest.raises(ValueError, match="runtime_tablebase_artifact_missing"):
        historical.verify_runtime_asset("exact_root_tablebase", asset, frozen)

    pointer.write_bytes(pointer_bytes)
    external.write_bytes(b"corrupt external object")
    with pytest.raises(ValueError, match="runtime_asset_hash:exact_root_tablebase"):
        historical.verify_runtime_asset("exact_root_tablebase", asset, frozen)
