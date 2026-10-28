"""Read-only verifier for seed542 corpus, membership, and training schedules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "docs/data/seed542-selfplay-4096"
sys.path.insert(0, str(ROOT))

from ml.alphazero_lite import prepare_seed542_experiment as rebuild  # noqa: E402
from ml.alphazero_lite import train, validate_seed500_pilot  # noqa: E402


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def verify_inventory() -> None:
    inventory = CORPUS / "execution-inventory.jsonl"
    events = [json.loads(line) for line in inventory.read_text().splitlines()]
    finished = {
        int(event["seed"]): event for event in events if event["event"] == "finished"
    }
    if not set(finished).issubset(set(range(542000, 542064))):
        raise ValueError("inventory_finished_seed_out_of_range")
    for seed, event in finished.items():
        path = CORPUS / f"shard-{seed}.jsonl"
        if event.get("exit_status") != 0 or not path.is_file():
            raise ValueError(f"inventory_shard_incomplete:{seed}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if event.get("output_sha256") != digest:
            raise ValueError(f"inventory_shard_hash_mismatch:{seed}")


def verify_partial() -> dict[str, str]:
    accounting_path = CORPUS / "accounting.json"
    accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
    interruption = json.loads(
        (CORPUS / "execution-interruption.json").read_text(encoding="utf-8")
    )
    status = json.loads((CORPUS / "partial-status.json").read_text(encoding="utf-8"))
    verify_inventory()
    shard = CORPUS / "shard-542000.jsonl"
    report = validate_seed500_pilot.validate_shards([shard])
    expected = accounting["completed_shards"][0]
    for accounting_key, report_key in (
        ("completed_games", "games"),
        ("rows", "rows"),
        ("canonical_unique_states", "canonical_unique_states"),
        ("production_float32_unique_inputs", "encoded_input_unique_states"),
    ):
        if expected[accounting_key] != report[report_key]:
            raise ValueError(f"partial_accounting_mismatch:{accounting_key}")
    if expected["sha256"] != hashlib.sha256(shard.read_bytes()).hexdigest():
        raise ValueError("partial_accounting_mismatch:sha256")
    staged = CORPUS / interruption["staging_output"]["path"].removeprefix(
        "docs/data/seed542-selfplay-4096/"
    )
    if (
        hashlib.sha256(staged.read_bytes()).hexdigest()
        != interruption["staging_output"]["sha256"]
    ):
        raise ValueError("partial_staging_hash_mismatch")
    rows = [
        json.loads(line) for line in staged.read_text(encoding="utf-8").splitlines()
    ]
    complete_rows = [row for row in rows if int(row["game_index"]) < 51]
    partial_report = validate_seed500_pilot._validate_and_account(complete_rows)
    x, policies, values = train.load_jsonl(
        staged, policy_target_mode="default", value_target_mode="default"
    )
    partial_expected = accounting["partial_shards"][0]
    canonical_keys = {rebuild.canonical_key(row) for row in rows}
    input_keys = {rebuild.input_key(row) for row in rows}
    if (
        len(rows) != partial_expected["staged_rows"]
        or len(canonical_keys) != partial_expected["canonical_unique_staged_inputs"]
        or len(input_keys) != partial_expected["float32_unique_staged_inputs"]
    ):
        raise ValueError("partial_staged_row_accounting_mismatch")
    if len(x) != interruption["staging_output"]["production_loader_accepted_rows"]:
        raise ValueError("partial_loader_count_mismatch")
    if partial_report["games"] != interruption["staging_output"]["completed_games"]:
        raise ValueError("partial_completed_game_count_mismatch")
    if status["status"] != "incomplete; training unauthorized":
        raise ValueError("partial_status_mismatch")
    if (
        status["completed_games"] != accounting["game_status"]["completed"]
        or status["preserved_partial_attempt"]["sha256"]
        != interruption["staging_output"]["sha256"]
    ):
        raise ValueError("partial_count_or_membership_mismatch")
    return {
        "accounting.json": hashlib.sha256(accounting_path.read_bytes()).hexdigest(),
        "partial-status.json": hashlib.sha256(
            (CORPUS / "partial-status.json").read_bytes()
        ).hexdigest(),
        "completed-shard.sha256": report["corpus_sha256"],
        "partial-staging.sha256": interruption["staging_output"]["sha256"],
    }


def verify() -> dict[str, str]:
    accounting = json.loads((CORPUS / "accounting.json").read_text(encoding="utf-8"))
    if accounting.get("status", "").startswith("partial corpus"):
        return verify_partial()
    verify_inventory()
    expected = rebuild.build_artifacts()
    verified = {}
    for name, rebuilt in expected.items():
        path = CORPUS / name
        archived = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json(archived) != canonical_json(rebuilt):
            raise ValueError(f"archived_evidence_mismatch:{name}")
        verified[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return verified


def main() -> None:
    for name, digest in verify().items():
        print(f"{name} sha256={digest}")


if __name__ == "__main__":
    main()
