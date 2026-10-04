"""Portable model-free verifier for the frozen E4 reference diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed398_e4_report
from ml.alphazero_lite.seed398_e4_report import calculate_report, validate_outcome

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed398-e4-reference-diagnostic"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict[str, Any]:
    reg_path = DATA / "registration.json"
    reg_bytes = reg_path.read_bytes()
    reg = json.loads(reg_bytes)
    require(
        reg["schema"] == "seed398-e4-reference-diagnostic-registration-v1",
        "registration_schema",
    )
    require(
        reg["source_registration_sha256"]
        == sha(ROOT / "docs/data/seed398-paired-first-action/registration.json"),
        "source_registration_identity",
    )
    base_registration = json.loads(
        (ROOT / "docs/data/seed398-paired-first-action/registration.json").read_text()
    )
    require(reg["states"] == base_registration["states"], "registered_state_order")
    sources = {
        "runner": ROOT / "ml/alphazero_lite/seed398_e4_reference_diagnostic.py",
        "paired_runner": ROOT / "ml/alphazero_lite/seed398_paired_first_action.py",
        "arena": ROOT / "ml/alphazero_lite/arena.py",
        "rules": ROOT / "ml/alphazero_lite/kalah_rules.py",
        "seed_contract": ROOT / "ml/alphazero_lite/evaluation_seed_contract.py",
        "native_adapter": ROOT / "ml/alphazero_lite/native_exact_root_tablebase.py",
        "exact_root_decision": ROOT / "ml/alphazero_lite/exact_root_decision.py",
        "runtime_search_policy": ROOT / "ml/alphazero_lite/runtime_search_policy.py",
    }
    require(
        reg["source_sha256"] == {key: sha(path) for key, path in sources.items()},
        "source_hash_binding",
    )
    ledger_path = DATA / "outcomes.jsonl"
    ledger_bytes = ledger_path.read_bytes()
    rows = [json.loads(line) for line in ledger_bytes.decode().splitlines() if line]
    states = {s["opening_index"]: s for s in reg["states"]}
    keys = set()
    for row in rows:
        validate_outcome(row, states[row["opening_index"]])
        key = (row["opening_index"], row["action"], row["budget"])
        require(key not in keys, "duplicate_outcome")
        keys.add(key)
    expected = {
        (s["opening_index"], action, budget)
        for s in reg["states"]
        for action in ("SS", "FF")
        for budget in (1536, 384)
    }
    require(len(rows) == 256 and keys == expected, "incomplete_or_extra_outcomes")
    analysis_path = DATA / "analysis.json"
    require(analysis_path.is_file(), "analysis_missing")
    analysis_bytes = analysis_path.read_bytes()
    published_analysis = json.loads(analysis_bytes)
    baseline = json.loads(
        (ROOT / "docs/data/seed398-paired-first-action/analysis.json").read_bytes()
    )
    expected_analysis = calculate_report(reg, rows, baseline)
    require(
        published_analysis == expected_analysis,
        "analysis_reconciliation",
    )
    binding_path = DATA / "publication-binding.json"
    binding_bytes = binding_path.read_bytes()
    binding = json.loads(binding_bytes)
    require(
        binding["registration_sha256"] == sha(reg_path),
        "publication_registration_binding",
    )
    require(
        binding["outcomes_sha256"] == sha(ledger_path), "publication_outcomes_binding"
    )
    require(
        binding["analysis_sha256"] == sha(analysis_path), "publication_analysis_binding"
    )
    correction_path = DATA / "verifier-correction-receipt.json"
    successor_path = DATA / "verifier-correction-receipt-v2.json"
    require(correction_path.is_file(), "correction_receipt_missing")
    require(successor_path.is_file(), "correction_successor_receipt_missing")
    receipt = json.loads(correction_path.read_bytes())
    successor = json.loads(successor_path.read_bytes())
    helper_sha256 = sha(Path(seed398_e4_report.__file__))
    current_verifier_sha256 = sha(Path(__file__))
    evidence = {
        "analysis_sha256": hashlib.sha256(analysis_bytes).hexdigest(),
        "registration_sha256": sha(reg_path),
        "outcomes_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
    }
    require(
        receipt["original_publication_binding_sha256"] == sha(binding_path),
        "correction_binding_identity",
    )
    require(
        receipt["old_verifier_sha256"] == binding["verifier_sha256"],
        "correction_old_verifier_identity",
    )
    require(
        receipt["helper_sha256"] == helper_sha256,
        "correction_helper_identity",
    )
    for name, digest in evidence.items():
        require(receipt[name] == digest, f"correction_{name}_identity")
    require(
        successor["schema"] == "seed398-e4-reference-diagnostic-verifier-correction-v2",
        "correction_successor_schema",
    )
    require(
        successor["predecessor_receipt_sha256"] == sha(correction_path),
        "correction_predecessor_hash",
    )
    require(
        successor["previous_verifier_sha256"] == receipt["new_verifier_sha256"],
        "correction_previous_verifier_identity",
    )
    require(
        successor["current_verifier_sha256"] == current_verifier_sha256,
        "correction_current_verifier_identity",
    )
    require(
        successor["helper_sha256"] == receipt["helper_sha256"] == helper_sha256,
        "correction_successor_helper_identity",
    )
    require(
        successor["original_publication_binding_sha256"]
        == receipt["original_publication_binding_sha256"]
        == sha(binding_path),
        "correction_successor_binding_identity",
    )
    for name, digest in evidence.items():
        require(
            successor[name] == digest == receipt[name], f"successor_{name}_identity"
        )
    require(
        reg_path.read_bytes() == reg_bytes,
        "registration_bytes_changed_during_verification",
    )
    require(
        ledger_path.read_bytes() == ledger_bytes,
        "ledger_bytes_changed_during_verification",
    )
    require(
        analysis_path.read_bytes() == analysis_bytes,
        "analysis_bytes_changed_during_verification",
    )
    require(
        binding_path.read_bytes() == binding_bytes,
        "binding_bytes_changed_during_verification",
    )
    return {
        "status": "verified",
        "states": 64,
        "outcomes": len(rows),
        "registration_sha256": sha(reg_path),
        "outcomes_sha256": sha(ledger_path),
        "analysis_sha256": sha(analysis_path),
    }


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
