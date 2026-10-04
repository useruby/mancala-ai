"""Portable model-free verifier for the frozen E4 reference diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed398_paired_first_action as paired
from ml.alphazero_lite.seed398_e4_reference_diagnostic import analyze

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed398-e4-reference-diagnostic"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict[str, Any]:
    reg_path = DATA / "registration.json"
    reg = json.loads(reg_path.read_text())
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
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line]
    states = {s["opening_index"]: s for s in reg["states"]}
    keys = set()
    for row in rows:
        paired.validate_outcome(row, states[row["opening_index"]])
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
    expected_analysis = analyze(reg, rows)
    analysis_path = DATA / "analysis.json"
    require(
        json.loads(analysis_path.read_text()) == expected_analysis,
        "analysis_reconciliation",
    )
    binding_path = DATA / "publication-binding.json"
    binding = json.loads(binding_path.read_text())
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
