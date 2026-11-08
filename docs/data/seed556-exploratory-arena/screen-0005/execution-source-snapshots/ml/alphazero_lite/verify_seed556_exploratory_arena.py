"""Read-only semantic verification of seed556 indexed arena evidence."""

from __future__ import annotations

import json
from pathlib import Path

from ml.alphazero_lite import run_seed556_exploratory_arena as runner
from ml.alphazero_lite import seed556_exploratory_arena as protocol


def verify(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    registration = json.loads((output / "registration.json").read_text())
    freeze = json.loads((output / "execution-freeze.json").read_text())
    if freeze.get("registration_sha256") != protocol.sha256(output / "registration.json"):
        raise ValueError("seed556_registration_freeze_mismatch")
    if freeze.get("source_sha256") != registration.get("source_sha256"):
        raise ValueError("seed556_frozen_source_binding_mismatch")
    if protocol.sha256(output / "command-preflight.jsonl") != freeze.get("command_preflight_sha256"):
        raise ValueError("seed556_frozen_commands_hash_mismatch")
    commands = [json.loads(line) for line in (output / "command-preflight.jsonl").read_text().splitlines()]
    ledger = [json.loads(line) for line in (output / "outcomes.jsonl").read_text().splitlines()]
    if len(commands) != protocol.OPENINGS or len(ledger) != protocol.GAMES:
        raise ValueError("seed556_evidence_incomplete")
    reconstructed = []
    for index in range(protocol.OPENINGS):
        pair = runner.validate_pair(root, output, index)
        receipt_path = output / "batches" / f"completion-{index:04d}.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("outcomes") != pair or receipt.get("command_sha256") != protocol.sha256(output / "batches" / f"command-{index:04d}.json"):
            raise ValueError(f"seed556_pair_receipt_mismatch:{index}")
        reconstructed.extend(pair)
    if reconstructed != ledger:
        raise ValueError("seed556_outcome_ledger_reconstruction_mismatch")
    analysis, scores = protocol.analyze(reconstructed)
    saved = json.loads((output / "analysis.json").read_text())
    saved_scores = [json.loads(line) for line in (output / "per-opening-scores.jsonl").read_text().splitlines()]
    if analysis != saved or scores != saved_scores:
        raise ValueError("seed556_analysis_reconstruction_mismatch")
    return {"status": "verified_complete", "validated_pairs": protocol.OPENINGS,
            "validated_games": protocol.GAMES, "analysis": analysis}
