"""Retrospective #406 first-action continuation diagnostic under original O0 E4."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ml.alphazero_lite import arena, seed398_paired_first_action as paired
from ml.alphazero_lite.native_exact_root_tablebase import NativeExactRootTablebase

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed398-e4-reference-diagnostic"
SOURCE = ROOT / "docs/data/seed398-paired-first-action"
PERSISTENCE = ROOT / "docs/data/seed398-search-budget-persistence"
E4 = ROOT / ".tmp/seed461-order-confirmation/artifacts/O0-E4"
CHECKPOINT = ROOT / ".tmp/seed461-order-confirmation/training/O0/E4.npz"
NATIVE = ROOT / "model-artifact/runtime/kalah_v1_tablebase"
TABLEBASE = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
LEDGER = DATA / "outcomes.jsonl"
BUDGETS = (1536, 384)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    base = json.loads((SOURCE / "registration.json").read_text())
    persistence = json.loads((PERSISTENCE / "registration.json").read_text())
    e4 = persistence["components"]["original_o0_e4"]
    expected = e4["artifact_sha256"]
    actual = {name: sha(E4 / name) for name in expected}
    if actual != expected or sha(CHECKPOINT) != e4["checkpoint_sha256"]:
        raise ValueError("E4_checkpoint_or_artifact_identity_mismatch")
    runtime = persistence["components"]["native_runtime_contract"]
    for path, key in (
        (NATIVE, "exact_root_native_probe_sha256"),
        (TABLEBASE, "exact_root_tablebase_sha256"),
    ):
        if sha(path) != runtime[key]:
            raise ValueError("native_runtime_component_identity_mismatch")
    reg = {
        "schema": "seed398-e4-reference-diagnostic-registration-v1",
        "scope": "retrospective reference-dependence diagnostic; no training, state reselection, promotion, or overall-strength claim",
        "source_registration_sha256": sha(SOURCE / "registration.json"),
        "states": base["states"],
        "reference": {
            "name": "original_O0_E4",
            "artifact": str(E4.relative_to(ROOT)),
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "checkpoint_sha256": e4["checkpoint_sha256"],
            "artifact_file_sha256": actual,
            "native_probe_sha256": sha(NATIVE),
            "tablebase_sha256": sha(TABLEBASE),
        },
        "search": {
            "budgets": list(BUDGETS),
            "c_puct": 1.25,
            "options": paired.OPTIONS,
            "exact_root_solve_threshold": 16,
            "exact_root_objective": "final_score_margin",
            "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
            "exact_leaf_solving": "disabled",
        },
        "seed_contract": "#406 azlite_eval_seed_v2 contexts, base seed 406; action, branch, and reference labels excluded from search seeds.",
        "analysis": {
            "paired_bootstrap_samples": 10000,
            "paired_bootstrap_seed": 406,
            "interval": [0.025, 0.975],
            "primary_budget": 1536,
            "secondary_budget": 384,
            "primary_decision": "SS advantage under both tested references iff E4 primary <= -0.03, upper 95% interval < 0, and secondary <= 0; reverse iff primary >= +0.03, lower interval > 0, and secondary >= 0; otherwise Reference robustness unresolved",
            "interaction": "E4 delta minus #406 seed455 delta; descriptive paired bootstrap interval",
        },
        "source_sha256": {
            "runner": sha(Path(__file__)),
            "paired_runner": sha(
                ROOT / "ml/alphazero_lite/seed398_paired_first_action.py"
            ),
            "arena": sha(ROOT / "ml/alphazero_lite/arena.py"),
            "rules": sha(ROOT / "ml/alphazero_lite/kalah_rules.py"),
            "seed_contract": sha(
                ROOT / "ml/alphazero_lite/evaluation_seed_contract.py"
            ),
            "native_adapter": sha(
                ROOT / "ml/alphazero_lite/native_exact_root_tablebase.py"
            ),
            "exact_root_decision": sha(
                ROOT / "ml/alphazero_lite/exact_root_decision.py"
            ),
            "runtime_search_policy": sha(
                ROOT / "ml/alphazero_lite/runtime_search_policy.py"
            ),
        },
    }
    path = DATA / "registration.json"
    payload = json.dumps(reg, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != payload:
        raise ValueError("immutable_registration_conflict")
    path.write_text(payload)
    return reg


def analyze(
    reg: dict[str, Any] | None = None, rows: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    reg = reg or json.loads((DATA / "registration.json").read_text())
    rows = (
        rows
        if rows is not None
        else [json.loads(line) for line in LEDGER.read_text().splitlines() if line]
    )
    states = {s["opening_index"]: s for s in reg["states"]}
    keyed = {}
    for row in rows:
        paired.validate_outcome(row, states[row["opening_index"]])
        key = (row["opening_index"], row["action"], row["budget"])
        if key in keyed:
            raise ValueError("duplicate_outcome")
        keyed[key] = row
    expected = {
        (s["opening_index"], a, b)
        for s in reg["states"]
        for a in ("SS", "FF")
        for b in BUDGETS
    }
    if len(keyed) != 256 or set(keyed) != expected:
        raise ValueError("incomplete_outcome_evidence")
    matrix = []
    for state in reg["states"]:
        item = {
            "opening_index": state["opening_index"],
            "state_hash": state["state_hash"],
        }
        for budget in BUDGETS:
            ss, ff = (
                keyed[(state["opening_index"], "SS", budget)],
                keyed[(state["opening_index"], "FF", budget)],
            )
            item[str(budget)] = {
                "ss_score": ss["score"],
                "ff_score": ff["score"],
                "delta": ff["score"] - ss["score"],
                "ss_margin": ss["store_margin_root_perspective"],
                "ff_margin": ff["store_margin_root_perspective"],
            }
        matrix.append(item)
    baseline = json.loads((SOURCE / "analysis.json").read_text())["paired_matrix"]
    base_by_index = {x["opening_index"]: x for x in baseline}
    primary = [x["1536"]["delta"] for x in matrix]
    secondary = [x["384"]["delta"] for x in matrix]
    interaction = [
        x - base_by_index[m["opening_index"]]["1536"]["delta"]
        for x, m in zip(primary, matrix)
    ]

    def bootstrap(values: list[float]) -> list[float]:
        rng_local = random.Random(406)
        samples = sorted(
            sum(values[rng_local.randrange(64)] for _ in range(64)) / 64
            for _ in range(10000)
        )
        return [samples[249], samples[9749]]

    mean_p, mean_s = sum(primary) / 64, sum(secondary) / 64
    interval = bootstrap(primary)
    if mean_p <= -0.03 and interval[1] < 0 and mean_s <= 0:
        decision = "SS advantage under both tested references"
    elif mean_p >= 0.03 and interval[0] > 0 and mean_s >= 0:
        decision = "Advantage reverses with reference"
    else:
        decision = "Reference robustness unresolved"
    report = {
        "schema": "seed398-e4-reference-diagnostic-analysis-v1",
        "decision": decision,
        "primary": {
            "budget": 1536,
            "mean_delta": mean_p,
            "paired_bootstrap_95_interval": interval,
        },
        "secondary": {"budget": 384, "mean_delta": mean_s},
        "interaction_E4_minus_seed455": {
            "mean_delta": sum(interaction) / 64,
            "paired_bootstrap_95_interval": bootstrap(interaction),
            "interpretation": "descriptive",
        },
        "paired_matrix": matrix,
        "interpretation": "Retrospective first-action comparison; does not establish minimax quality, overall strength, or promotion eligibility.",
    }
    (DATA / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def run() -> dict[str, Any]:
    reg = json.loads((DATA / "registration.json").read_text())
    if reg["source_sha256"] != register()["source_sha256"]:
        raise ValueError("registered_source_identity_changed")
    paired.ARTIFACT = E4
    evaluator = arena.ArtifactEvaluator(E4)
    states = {s["opening_index"]: s for s in reg["states"]}
    complete = {}
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            paired.validate_outcome(row, states[row["opening_index"]])
            key = (row["opening_index"], row["action"], row["budget"])
            if key in complete:
                raise ValueError("duplicate_resumed_outcome")
            complete[key] = row
    adapter = NativeExactRootTablebase(NATIVE, TABLEBASE, warm_on_start=True)
    try:
        with LEDGER.open("a") as ledger:
            for state in reg["states"]:
                for budget in BUDGETS:
                    for action in ("SS", "FF"):
                        key = (state["opening_index"], action, budget)
                        if key not in complete:
                            result = paired.play(
                                state,
                                action,
                                budget,
                                evaluator,
                                endgame_tablebase=adapter,
                            )
                            paired.validate_outcome(result, state)
                            ledger.write(json.dumps(result, sort_keys=True) + "\n")
                            ledger.flush()
                            complete[key] = result
    finally:
        adapter.close()
    if len(complete) != 256:
        raise ValueError("incomplete_outcome_evidence")
    return analyze(reg, list(complete.values()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("register", "run", "analyze"))
    stage = parser.parse_args().stage
    if stage == "register":
        register()
    elif stage == "run":
        run()
    else:
        print(json.dumps(analyze(), indent=2))
