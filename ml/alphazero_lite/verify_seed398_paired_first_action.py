"""Portable, model-free verifier for the native paired-continuation bundle."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed398-paired-first-action"
SOURCE = ROOT / "docs/data/seed398-search-budget-persistence"
BUDGETS = (1536, 384)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def verify_trajectory(row: dict[str, Any], selected: dict[str, Any]) -> None:
    require(
        row["opening_index"] == selected["opening_index"]
        and row["state_hash"] == selected["state_hash"],
        "state_binding_mismatch",
    )
    root = KalahGame.from_state(selected["state"])
    root_player = root.current_player
    moves = row["trajectory"]
    forced_action = selected[
        "ss_action_384" if row["action"] == "SS" else "ff_action_384"
    ]
    require(
        bool(moves)
        and moves[0]["forced"] is True
        and moves[0]["action_relative"] == forced_action,
        "forced_action_mismatch",
    )
    for index, record in enumerate(moves):
        require(
            not root.over() and record["actor"] == root.current_player,
            f"illegal_trajectory_actor:{index}",
        )
        relative = int(record["action_relative"])
        absolute = int(record["action_absolute"])
        require(
            absolute == root.pit_index(relative) and relative in root.possible_moves(),
            f"illegal_trajectory_move:{index}",
        )
        if index:
            active_stones = sum(root.pits)
            expected_backend = (
                "native_kvtb_root_action_probe_v1" if active_stones <= 16 else None
            )
            require(
                record["active_pit_stones_before_move"] == active_stones,
                f"active_stone_count_mismatch:{index}",
            )
            require(
                record["exact_root_backend"] == expected_backend,
                f"exact_root_backend_mismatch:{index}",
            )
            require(
                record["exact_root_solver_calls"] == (1 if active_stones <= 16 else 0),
                f"exact_root_solver_call_mismatch:{index}",
            )
            require(
                isinstance(record["seed"], int)
                and len(record["seed_context_hash"]) == 64,
                f"seed_telemetry_missing:{index}",
            )
        require(
            root.move(absolute) and record["state"] == root.to_state(),
            f"trajectory_state_mismatch:{index}",
        )
    require(root.over(), "trajectory_not_terminal")
    require(
        row["stores"] == root.captured_seeds and sum(row["stores"]) == 48,
        "terminal_store_reconciliation_mismatch",
    )
    margin = root.captured_seeds[root_player] - root.captured_seeds[1 - root_player]
    score = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
    require(
        row["root_player"] == root_player
        and row["store_margin_root_perspective"] == margin
        and row["score"] == score,
        "terminal_score_mismatch",
    )


def recompute(reg: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    keyed = {(r["opening_index"], r["action"], r["budget"]): r for r in rows}
    paired = []
    for s in reg["states"]:
        item = {
            "opening_index": s["opening_index"],
            "state_hash": s["state_hash"],
            "groups": s.get("groups"),
        }
        for budget in BUDGETS:
            ss, ff = (
                keyed[(s["opening_index"], "SS", budget)],
                keyed[(s["opening_index"], "FF", budget)],
            )
            item[str(budget)] = {
                "ss_score": ss["score"],
                "ff_score": ff["score"],
                "delta": ff["score"] - ss["score"],
                "ss_margin": ss["store_margin_root_perspective"],
                "ff_margin": ff["store_margin_root_perspective"],
            }
        paired.append(item)
    primary_deltas = [x["1536"]["delta"] for x in paired]
    rng = random.Random(406)
    boot = sorted(
        sum(primary_deltas[rng.randrange(64)] for _ in range(64)) / 64
        for _ in range(10000)
    )
    primary, secondary = (
        sum(primary_deltas) / 64,
        sum(x["384"]["delta"] for x in paired) / 64,
    )
    low, high = boot[249], boot[9749]
    classification = (
        "FF-action advantage"
        if primary >= 0.03 and low > 0 and secondary >= 0
        else "SS-action advantage"
        if primary <= -0.03 and high < 0 and secondary <= 0
        else "reference-dependent"
        if ((low > 0 and secondary < 0) or (high < 0 and secondary > 0))
        else "no clear action advantage"
    )
    return {
        "classification": classification,
        "primary": {
            "budget": 1536,
            "mean_delta": primary,
            "paired_bootstrap_95_interval": [low, high],
        },
        "secondary": {"budget": 384, "mean_delta": secondary},
        "paired_matrix": paired,
    }


def verify() -> dict[str, Any]:
    reg_path = DATA / "registration.json"
    amendment_path = DATA / "execution-amendment-v1.json"
    previous_amendment_path = DATA / "execution-amendment-v3.json"
    intermediate_amendment_path = DATA / "execution-amendment-v2.json"
    final_amendment_path = DATA / "execution-amendment-v4.json"
    correction_receipt_path = DATA / "analysis-correction-receipt.json"
    original_path = DATA / "outcomes.jsonl"
    archive_path = DATA / "fallback-archive/outcomes.python-fallback.jsonl"
    native_path = DATA / "native-outcomes.jsonl"
    analysis_path = DATA / "analysis.json"
    reg = json.loads(reg_path.read_text())
    amendment = json.loads(amendment_path.read_text())
    previous_amendment = json.loads(previous_amendment_path.read_text())
    intermediate_amendment = json.loads(intermediate_amendment_path.read_text())
    final_amendment = json.loads(final_amendment_path.read_text())
    original_lines = original_path.read_text().splitlines()
    archive_lines = archive_path.read_text().splitlines()
    require(
        len(original_lines)
        == len(archive_lines)
        == amendment["fallback_outcomes_count"]
        == 10,
        "fallback_archive_count_mismatch",
    )
    require(original_lines == archive_lines, "fallback_archive_content_mismatch")
    require(
        sha256(original_path)
        == amendment["fallback_outcomes_sha256"]
        == sha256(archive_path),
        "fallback_archive_hash_mismatch",
    )
    require(
        amendment["original_source_sha256"] == reg["source_sha256"]
        and final_amendment["original_source_sha256"] == reg["source_sha256"],
        "amendment_source_binding_mismatch",
    )
    require(
        final_amendment["prior_amendment_sha256"] == sha256(previous_amendment_path)
        and previous_amendment["prior_amendment_sha256"]
        == sha256(intermediate_amendment_path)
        and intermediate_amendment["prior_amendment_sha256"] == sha256(amendment_path)
        and final_amendment["executed_source_sha256"]
        == amendment["corrected_source_sha256"],
        "amendment_chain_mismatch",
    )
    correction_receipt = json.loads(correction_receipt_path.read_text())
    require(
        correction_receipt["executed_source_sha256"]
        == final_amendment["executed_source_sha256"]
        and correction_receipt["post_execution_source_sha256"]
        == final_amendment["corrected_source_sha256"]
        and correction_receipt["prior_execution_amendment_sha256"]
        == sha256(final_amendment_path),
        "analysis_correction_executed_source_mismatch",
    )
    source_paths = {
        "diagnostic": ROOT / "ml/alphazero_lite/seed398_paired_first_action.py",
        "arena": ROOT / "ml/alphazero_lite/arena.py",
        "rules": ROOT / "ml/alphazero_lite/kalah_rules.py",
        "seed_contract": ROOT / "ml/alphazero_lite/evaluation_seed_contract.py",
        "native_adapter": ROOT / "ml/alphazero_lite/native_exact_root_tablebase.py",
        "exact_root_decision": ROOT / "ml/alphazero_lite/exact_root_decision.py",
        "runtime_search_policy": ROOT / "ml/alphazero_lite/runtime_search_policy.py",
    }
    require(
        correction_receipt["current_source_sha256"]
        == {name: sha256(path) for name, path in source_paths.items()},
        "analysis_correction_source_hash_mismatch",
    )
    require(
        correction_receipt["registration_sha256"] == sha256(reg_path)
        and correction_receipt["native_ledger_sha256"] == sha256(native_path)
        and correction_receipt["fallback_ledger_sha256"] == sha256(original_path),
        "analysis_correction_evidence_identity_mismatch",
    )
    require(
        reg["source_registration_sha256"] == sha256(SOURCE / "registration.json"),
        "seed398_registration_hash_mismatch",
    )
    states = reg["states"]
    require(
        len(states) == 64 and len({s["opening_index"] for s in states}) == 64,
        "registered_state_count_mismatch",
    )
    rows = [json.loads(line) for line in native_path.read_text().splitlines() if line]
    expected = {
        (s["opening_index"], a, b)
        for s in states
        for a in ("SS", "FF")
        for b in BUDGETS
    }
    actual = {(r["opening_index"], r["action"], r["budget"]) for r in rows}
    require(
        len(rows) == 256 and actual == expected, "native_ledger_reconciliation_mismatch"
    )
    by_index = {s["opening_index"]: s for s in states}
    for row in rows:
        require(
            row["budget"] in BUDGETS and row["action"] in ("SS", "FF"),
            "outcome_key_invalid",
        )
        verify_trajectory(row, by_index[row["opening_index"]])
    summary = recompute(reg, rows)
    report = json.loads(analysis_path.read_text())
    for field, value in summary.items():
        require(report[field] == value, f"analysis_mismatch:{field}")
    require(
        correction_receipt["analysis_sha256"] == sha256(analysis_path),
        "analysis_correction_analysis_hash_mismatch",
    )
    require(
        report["seed398_groups_descriptive"]
        == {
            "same_1536_action": {"n": 32},
            "persistent_model_disagreement_with_margin": {"n": 17},
        },
        "descriptive_groups_mismatch",
    )
    return {
        "status": "verified",
        "states": 64,
        "native_outcomes": len(rows),
        "fallback_outcomes_archived": len(archive_lines),
        "native_solver_calls": sum(
            m.get("exact_root_solver_calls", 0) for r in rows for m in r["trajectory"]
        ),
        "registration_sha256": sha256(reg_path),
        "amendment_sha256": sha256(amendment_path),
        "final_amendment_sha256": sha256(final_amendment_path),
        "native_outcomes_sha256": sha256(native_path),
        "analysis_sha256": sha256(analysis_path),
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
