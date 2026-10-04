"""Exploratory comparison of FF's frozen 1536- and 384-search actions.

This module reuses validated #406/#408 outcomes wherever the forced action
matches and runs only the pre-registered unmatched continuations.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed398_paired_first_action as paired
from ml.alphazero_lite import arena
from ml.alphazero_lite.native_exact_root_tablebase import NativeExactRootTablebase

ROOT = Path(__file__).resolve().parents[2]
PROBES = ROOT / "docs/data/seed398-search-budget-persistence/raw-probes.json"
PERSISTENCE_REG = ROOT / "docs/data/seed398-search-budget-persistence/registration.json"
SEED455_DIR = ROOT / "docs/data/seed398-paired-first-action"
E4_DIR = ROOT / "docs/data/seed398-e4-reference-diagnostic"
OUT = ROOT / "docs/data/seed398-ff1536-confirmation"
STATES_REG = SEED455_DIR / "registration.json"
RUNTIME = ROOT / "model-artifact/runtime"
ARTIFACTS = {
    "seed455": ROOT / ".tmp/seed461-order-confirmation/opponent-artifact",
    "original_O0_E4": ROOT / ".tmp/seed461-order-confirmation/artifacts/O0-E4",
}
CHECKPOINTS = {
    "seed455": ROOT
    / ".tmp/seed48-nextgen-s455-default-value/runs/seed48-nextgen-s455-default-value-iter1/checkpoint.npz",
    "original_O0_E4": ROOT / ".tmp/seed461-order-confirmation/training/O0/E4.npz",
}
REFERENCES = {
    "seed455": SEED455_DIR / "native-outcomes.jsonl",
    "original_O0_E4": E4_DIR / "outcomes.jsonl",
}
NATIVE = RUNTIME / "kalah_v1_tablebase"
TABLEBASE = RUNTIME / "kalah_v1_21.kvtb"
BUDGET = 1536


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def frozen_actions() -> dict[int, int]:
    """Read only the published 1536 snapshots; never invoke a root search."""
    states = _json(STATES_REG)["states"]
    raw = _json(PROBES)["probes"]
    indexed = {}
    for probe in raw:
        if probe["treatment"] != "FF":
            continue
        index = probe["opening_index"]
        if index in indexed:
            raise ValueError("duplicate_frozen_ff_probe")
        final = [s for s in probe["snapshots"] if s["simulations"] == BUDGET]
        if len(final) != 1 or probe["final_action"] != final[0]["selected_action"]:
            raise ValueError("frozen_action_snapshot_mismatch")
        indexed[index] = final[0]["selected_action"]
    if len(indexed) != 64 or set(indexed) != {s["opening_index"] for s in states}:
        raise ValueError("frozen_action_state_coverage_mismatch")
    for state in states:
        if indexed[state["opening_index"]] not in state["legal_actions"]:
            raise ValueError("frozen_action_illegal")
    return indexed


def verify_runtime_inputs() -> dict[str, Any]:
    """Fail closed on every checkpoint, sidecar, and native runtime identity."""
    regs = {
        "seed455": _json(STATES_REG),
        "original_O0_E4": _json(E4_DIR / "registration.json"),
    }
    expected_artifact_hashes = {
        "seed455": regs["seed455"]["reference"]["artifact_file_sha256"],
        "original_O0_E4": regs["original_O0_E4"]["reference"]["artifact_file_sha256"],
    }
    identities: dict[str, Any] = {}
    for reference in ARTIFACTS:
        actual = {
            name: sha(ARTIFACTS[reference] / name)
            for name in expected_artifact_hashes[reference]
        }
        if actual != expected_artifact_hashes[reference]:
            raise ValueError(f"artifact_identity_mismatch:{reference}")
        expected_checkpoint = regs[reference]["reference"]["checkpoint_sha256"]
        if sha(CHECKPOINTS[reference]) != expected_checkpoint:
            raise ValueError(f"checkpoint_identity_mismatch:{reference}")
        identities[reference] = {
            "artifact_files": actual,
            "checkpoint_sha256": expected_checkpoint,
        }
    native_hash, tablebase_hash = sha(NATIVE), sha(TABLEBASE)
    for reference, reg in regs.items():
        ref = reg["reference"]
        expected_native = ref.get("native_probe_sha256", ref.get("native_probe_sha256"))
        if expected_native and native_hash != expected_native:
            raise ValueError(f"native_probe_identity_mismatch:{reference}")
        if tablebase_hash != ref["tablebase_sha256"]:
            raise ValueError(f"tablebase_identity_mismatch:{reference}")
    return {
        "references": identities,
        "native_sha256": native_hash,
        "tablebase_sha256": tablebase_hash,
    }


def source_identities() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "paired_runner": ROOT / "ml/alphazero_lite/seed398_paired_first_action.py",
        "arena": ROOT / "ml/alphazero_lite/arena.py",
        "rules": ROOT / "ml/alphazero_lite/kalah_rules.py",
        "seed_contract": ROOT / "ml/alphazero_lite/evaluation_seed_contract.py",
        "native_adapter": ROOT / "ml/alphazero_lite/native_exact_root_tablebase.py",
        "exact_root_decision": ROOT / "ml/alphazero_lite/exact_root_decision.py",
        "runtime_search_policy": ROOT / "ml/alphazero_lite/runtime_search_policy.py",
    }
    return {name: sha(path) for name, path in paths.items()}


def build_reuse_plan() -> list[dict[str, Any]]:
    """Build 128 reference/state assignments with source-ledger provenance."""
    states = _json(STATES_REG)["states"]
    actions = frozen_actions()
    refs = {
        name: _json(path)
        for name, path in (
            ("seed455", STATES_REG),
            ("original_O0_E4", E4_DIR / "registration.json"),
        )
    }
    outcomes: dict[str, dict[tuple[int, str, int], dict[str, Any]]] = {}
    ledger_hashes = {name: sha(path) for name, path in REFERENCES.items()}
    for name, path in REFERENCES.items():
        reg = refs[name]
        state_by_index = {row["opening_index"]: row for row in reg["states"]}
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        keyed = {}
        for row in rows:
            paired.validate_outcome(row, state_by_index[row["opening_index"]])
            key = row["opening_index"], row["action"], row["budget"]
            if key in keyed:
                raise ValueError(f"duplicate_source_outcome:{name}:{key}")
            keyed[key] = row
        outcomes[name] = keyed
    plan = []
    for state in states:
        idx = state["opening_index"]
        for reference in ("seed455", "original_O0_E4"):
            action = actions[idx]
            source_treatment = next(
                (
                    treatment
                    for treatment, field in (
                        ("FF", "ff_action_384"),
                        ("SS", "ss_action_384"),
                    )
                    if state[field] == action
                ),
                None,
            )
            if source_treatment is None:
                plan.append(
                    {
                        "opening_index": idx,
                        "state_hash": state["state_hash"],
                        "reference": reference,
                        "forced_action": action,
                        "source": "new",
                    }
                )
                continue
            outcome = outcomes[reference][(idx, source_treatment, BUDGET)]
            plan.append(
                {
                    "opening_index": idx,
                    "state_hash": state["state_hash"],
                    "reference": reference,
                    "forced_action": action,
                    "source": "reused",
                    "source_ledger": str(REFERENCES[reference].relative_to(ROOT)),
                    "source_ledger_sha256": ledger_hashes[reference],
                    "source_case_identity": {
                        "opening_index": idx,
                        "action": source_treatment,
                        "budget": BUDGET,
                    },
                    "outcome": outcome,
                }
            )
    if (
        len(plan) != 128
        or sum(row["source"] == "reused" for row in plan) != 120
        or sum(row["source"] == "new" for row in plan) != 8
    ):
        raise ValueError("reuse_plan_accounting_mismatch")
    return plan


def analysis(
    plan: list[dict[str, Any]], new_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Pure analysis from frozen/reused rows and the eight new outcomes."""
    by_case = {(row["opening_index"], row["reference"]): row for row in plan}
    for row in new_rows:
        key = (row["opening_index"], row["reference"])
        if key not in by_case or by_case[key]["source"] != "new":
            raise ValueError("unexpected_new_case")
        if "outcome" in by_case[key]:
            raise ValueError("duplicate_new_case")
        by_case[key]["outcome"] = row["outcome"]
    if len(new_rows) != 8 or any("outcome" not in row for row in plan):
        raise ValueError("incomplete_outcome_evidence")
    state_results = []
    means_by_reference: dict[str, list[float]] = {"seed455": [], "original_O0_E4": []}
    for state in _json(STATES_REG)["states"]:
        idx = state["opening_index"]
        ref_gains = {}
        ref_margins = {}
        for reference in means_by_reference:
            row = by_case[(idx, reference)]["outcome"]
            reference_rows = _json_lines(REFERENCES[reference])
            baseline = next(
                r
                for r in reference_rows
                if (r["opening_index"], r["action"], r["budget"]) == (idx, "FF", BUDGET)
            )
            gain = row["score"] - baseline["score"]
            ref_gains[reference] = gain
            ref_margins[reference] = (
                row["store_margin_root_perspective"]
                - baseline["store_margin_root_perspective"]
            )
            means_by_reference[reference].append(gain)
        state_results.append(
            {
                "opening_index": idx,
                "reference_gains": ref_gains,
                "paired_gain": sum(ref_gains.values()) / 2,
                "reference_store_margin_changes_vs_ff384": ref_margins,
                "reference_difference_vs_ss384": {
                    reference: by_case[(idx, reference)]["outcome"]["score"]
                    - next(
                        r["score"]
                        for r in _json_lines(REFERENCES[reference])
                        if (r["opening_index"], r["action"], r["budget"])
                        == (idx, "SS", BUDGET)
                    )
                    for reference in means_by_reference
                },
            }
        )
    paired = [row["paired_gain"] for row in state_results]
    rng = random.Random(411)
    samples = sorted(
        sum(paired[rng.randrange(64)] for _ in range(64)) / 64 for _ in range(10000)
    )
    mean = sum(paired) / 64
    lower = samples[249]
    return {
        "schema": "seed398-ff1536-confirmation-analysis-v1",
        "exploratory": True,
        "primary": {
            "mean_gain": mean,
            "bootstrap_95_interval": [lower, samples[9749]],
            "resamples": 10000,
            "seed": 411,
        },
        "reference_mean_gains": {
            key: sum(values) / 64 for key, values in means_by_reference.items()
        },
        "state_results": state_results,
        "decision": "nominate_fresh_confirmation_experiment"
        if mean >= 0.03
        and lower > 0
        and all(sum(values) / 64 >= 0 for values in means_by_reference.values())
        else "no_supported_budget_follow_up",
        "interpretation": "Exploratory retrospective comparison using previously observed outcomes. No promotion or overall-strength evidence.",
    }


def register_draft() -> dict[str, Any]:
    """Freeze registration only after bundle/runtime identities have passed."""
    runtime = verify_runtime_inputs()
    plan = build_reuse_plan()
    new_cases = [
        {
            key: row[key]
            for key in ("opening_index", "state_hash", "reference", "forced_action")
        }
        for row in plan
        if row["source"] == "new"
    ]
    expected_new = [(372, 4), (260, 1), (164, 2), (217, 0)]
    observed = [
        (row["opening_index"], row["forced_action"])
        for row in new_cases
        if row["reference"] == "seed455"
    ]
    if observed != expected_new or len(new_cases) != 8:
        raise ValueError("registered_new_continuation_cases_mismatch")
    reg = {
        "schema": "seed398-ff1536-confirmation-registration-v1",
        "scope": "exploratory retrospective comparison; no promotion or overall-strength evidence",
        "state_registration_sha256": sha(STATES_REG),
        "probe_sha256": sha(PROBES),
        "persistence_registration_sha256": sha(PERSISTENCE_REG),
        "action_mapping": [
            {
                "opening_index": state["opening_index"],
                "state_hash": state["state_hash"],
                "ff1536_action": frozen_actions()[state["opening_index"]],
            }
            for state in _json(STATES_REG)["states"]
        ],
        "runtime_identities": runtime,
        "source_sha256": source_identities(),
        "search": {
            "budget": 1536,
            "c_puct": 1.25,
            "options": paired.OPTIONS,
            "exact_root_solve_threshold": 16,
            "exact_leaf_solving": "disabled",
            "base_seed": 406,
        },
        "seed_contract": "#406 azlite_eval_seed_v2 context and suite identity; no branch/action/reference labels in seed inputs.",
        "analysis_rules": {
            "primary": "per-state mean across two reference gains; mean over 64 states including zeros",
            "bootstrap": {
                "units": 64,
                "resamples": 10000,
                "seed": 411,
                "interval": [0.025, 0.975],
            },
            "nominate_confirmation_if": "primary mean >= 0.03, lower interval > 0, and both reference means >= 0",
        },
        "reuse_plan": plan,
        "new_cases": new_cases,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "registration.json"
    payload = json.dumps(reg, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text() != payload:
        raise ValueError("immutable_registration_conflict")
    target.write_text(payload)
    return reg


def run_new_continuations() -> list[dict[str, Any]]:
    reg = _json(OUT / "registration.json")
    if source_identities() != reg["source_sha256"]:
        raise ValueError("frozen_execution_sources_changed")
    if verify_runtime_inputs() != reg["runtime_identities"]:
        raise ValueError("frozen_runtime_identities_changed")
    ledger = OUT / "new-outcomes.jsonl"
    if ledger.exists() and ledger.stat().st_size:
        raise ValueError("new_continuation_ledger_must_start_empty")
    states = {row["opening_index"]: row for row in _json(STATES_REG)["states"]}
    adapter = NativeExactRootTablebase(NATIVE, TABLEBASE, warm_on_start=True)
    rows: list[dict[str, Any]] = []
    try:
        with ledger.open("a") as output:
            for case in reg["new_cases"]:
                reference = case["reference"]
                paired.ARTIFACT = ARTIFACTS[reference]
                state = dict(states[case["opening_index"]])
                state["ff_action_384"] = case["forced_action"]
                result = paired.play(
                    state,
                    "FF",
                    BUDGET,
                    arena.ArtifactEvaluator(ARTIFACTS[reference]),
                    endgame_tablebase=adapter,
                )
                row = {
                    "opening_index": case["opening_index"],
                    "reference": reference,
                    "forced_action": case["forced_action"],
                    "outcome": result,
                }
                output.write(json.dumps(row, sort_keys=True) + "\n")
                output.flush()
                rows.append(row)
    finally:
        adapter.close()
    if len(rows) != 8:
        raise ValueError("new_continuation_count_mismatch")
    return rows


def _json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]
