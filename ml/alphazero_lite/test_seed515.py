"""Mutation checks for the independent seed515 verifier."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ml.alphazero_lite import verify_seed515 as verifier

_REBUILT = None
_OCCURRENCES = None
_DIGEST = verifier.digest


@pytest.fixture(scope="module", autouse=True)
def positive_publication() -> None:
    """Run the unchanged publication first and cache replay for mutation cases."""
    global _REBUILT, _OCCURRENCES
    verifier.verify()
    _REBUILT, _OCCURRENCES = verifier.reconstruct(
        json.loads((verifier.PUBLICATION / "protocol-freeze.json").read_text())
    )


def _copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    destination = tmp_path / "publication"
    shutil.copytree(verifier.PUBLICATION, destination)
    monkeypatch.setattr(
        verifier, "reconstruct", lambda _freeze: (_REBUILT, _OCCURRENCES)
    )
    cache: dict[Path, str] = {}

    def cached_digest(path: Path) -> str:
        resolved = path.resolve()
        if resolved.is_relative_to(destination.resolve()):
            return _DIGEST(path)
        if resolved not in cache:
            cache[resolved] = _DIGEST(path)
        return cache[resolved]

    monkeypatch.setattr(verifier, "digest", cached_digest)
    return destination


def _load(path: Path):
    return json.loads(path.read_text())


def _save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _rebind_cohort(directory: Path) -> None:
    freeze_path = directory / "protocol-freeze.json"
    freeze = _load(freeze_path)
    freeze["cohort_sha256"] = _DIGEST(directory / "cohort.json")
    _save(freeze_path, freeze)
    _rebind_protocol_evidence(directory)


def _rebind_protocol_evidence(directory: Path) -> None:
    evidence_path = directory / "evidence.json"
    evidence = _load(evidence_path)
    evidence["protocol_freeze_sha256"] = _DIGEST(directory / "protocol-freeze.json")
    _save(evidence_path, evidence)
    results_path = directory / "results.md"
    results = results_path.read_text()
    old_hash = "7c2f1bcc682e98505a0a4e7c4423094e932ce0dbcf887606b465fc5bf644f88f"
    results_path.write_text(
        results.replace(old_hash, evidence["protocol_freeze_sha256"])
    )


def test_00_unchanged_positive_fixture_passes() -> None:
    result = verifier.verify()
    assert result["cohort_states"] == 64
    assert result["measurement_keys"] == 640
    assert result["matching_occurrences"] == 69976
    assert result["unique_candidate_states"] == 60333


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("ply", "provenance_ply"),
        ("provenance_order", "provenance_order"),
        ("state", "state_hash_or_contents"),
        ("bucket", "bucket:"),
        ("duplicate", "duplicate_measurement"),
        ("missing", "missing_measurement"),
        ("seed", "registered_seed"),
        ("order", "alternating_order"),
        ("routing", "routing"),
        ("simulation_accounting", "simulation_accounting"),
        ("nan", "measurement_timing"),
        ("infinity", "measurement_timing"),
        ("zero", "measurement_timing"),
        ("negative", "measurement_timing"),
        ("aggregate", "summary_statistics"),
        ("paired_summary", "summary_paired_ratios"),
        ("per_state", "per_state_statistics"),
        ("bucket_summary", "displayed_bucket_ratios"),
        ("proposal_summary", "proposal_displayed_statistics"),
        ("source_hash", "source_hash"),
        ("artifact_hash", "artifact_hash"),
        ("native_probe_hash", "runtime_asset_hash"),
        ("tablebase_hash", "runtime_asset_hash"),
        ("memory_identity", "memory_artifact_identity"),
        ("memory_source_hash", "memory_implementation_hash"),
        ("memory_arithmetic", "memory_incremental_arithmetic"),
    ],
)
def test_mutation_is_rejected_with_specific_reason(
    mutation: str, reason: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _copy(tmp_path, monkeypatch)
    raw_path = directory / "raw-measurements.jsonl"
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
    cohort_path = directory / "cohort.json"
    cohort = _load(cohort_path)
    freeze_path = directory / "protocol-freeze.json"
    freeze = _load(freeze_path)
    evidence_path = directory / "evidence.json"
    evidence = _load(evidence_path)
    memory_path = directory / "memory-snapshot.json"
    memory = _load(memory_path)
    if mutation == "ply":
        cohort[0]["ply"] += 1
        _save(cohort_path, cohort)
        _rebind_cohort(directory)
    elif mutation == "provenance_order":
        cohort[0]["selection_order_sha256"] = "0" * 64
        _save(cohort_path, cohort)
        _rebind_cohort(directory)
    elif mutation == "state":
        cohort[0]["state"]["player_store"] += 1
        _save(cohort_path, cohort)
        _rebind_cohort(directory)
    elif mutation == "bucket":
        cohort[0]["bucket"] = "le16" if cohort[0]["bucket"] != "le16" else "gt32"
        _save(cohort_path, cohort)
        _rebind_cohort(directory)
    elif mutation == "duplicate":
        raw.append(raw[0])
    elif mutation == "missing":
        raw.pop()
    elif mutation == "seed":
        raw[0]["seed"] += 1
    elif mutation == "order":
        raw[0], raw[1] = raw[1], raw[0]
    elif mutation == "routing":
        raw[0]["exact_root"] = not raw[0]["exact_root"]
    elif mutation == "simulation_accounting":
        raw[0]["puct_simulations_executed"] += 1
    elif mutation in {"nan", "infinity", "zero", "negative"}:
        raw[0]["seconds"] = {
            "nan": float("nan"),
            "infinity": float("inf"),
            "zero": 0,
            "negative": -1,
        }[mutation]
    elif mutation == "aggregate":
        evidence["summary"]["configs"]["P"]["p50_seconds"] += 1
        _save(evidence_path, evidence)
    elif mutation == "paired_summary":
        evidence["summary"]["paired_ratio_p50"] += 1
        _save(evidence_path, evidence)
    elif mutation == "per_state":
        evidence["summary"]["per_state_latency_ratios"][0]["P_over_baseline"] += 1
        _save(evidence_path, evidence)
    elif mutation == "bucket_summary":
        results = directory / "results.md"
        results.write_text(results.read_text().replace("1.621", "9.999", 1))
    elif mutation == "proposal_summary":
        proposal = verifier.ROOT / "docs/seed512-deployment-evaluation-proposal.md"
        original_read_text = Path.read_text
        monkeypatch.setattr(
            Path,
            "read_text",
            lambda path, *args, **kwargs: (
                original_read_text(path, *args, **kwargs).replace("1.621", "9.999", 1)
                if path == proposal
                else original_read_text(path, *args, **kwargs)
            ),
        )
    elif mutation == "source_hash":
        freeze["source_sha256"]["ml/alphazero_lite/arena.py"] = "0" * 64
        _save(freeze_path, freeze)
        _rebind_protocol_evidence(directory)
    elif mutation == "artifact_hash":
        freeze["artifacts"]["components"]["E4"]["weights.json"] = "0" * 64
        _save(freeze_path, freeze)
        _rebind_protocol_evidence(directory)
    elif mutation in {"native_probe_hash", "tablebase_hash"}:
        key = (
            "exact_root_native_probe"
            if mutation == "native_probe_hash"
            else "exact_root_tablebase"
        )
        freeze["artifacts"]["runtime_assets"][key]["sha256"] = "0" * 64
        _save(freeze_path, freeze)
        _rebind_protocol_evidence(directory)
    elif mutation == "memory_identity":
        memory["artifact_sha256"]["E4"] = "0" * 64
        _save(memory_path, memory)
    elif mutation == "memory_source_hash":
        memory["implementation_sha256"] = "0" * 64
        _save(memory_path, memory)
    elif mutation == "memory_arithmetic":
        memory["configurations"]["P"]["incremental_over_baseline_bytes"] += 1
        _save(memory_path, memory)
    if mutation in {
        "duplicate",
        "missing",
        "seed",
        "order",
        "routing",
        "simulation_accounting",
        "nan",
        "infinity",
        "zero",
        "negative",
    }:
        raw_path.write_text(
            "".join(json.dumps(row, allow_nan=True) + "\n" for row in raw)
        )
    with pytest.raises(ValueError, match=reason):
        verifier.verify(directory)
