"""Registered paired first-action continuation diagnostic for #405 states."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from ml.alphazero_lite import arena
from ml.alphazero_lite.evaluation_seed_contract import derive_search_seed
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.native_exact_root_tablebase import NativeExactRootTablebase
from ml.alphazero_lite.runtime_search_policy import load_runtime_search_policy

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/data/seed398-search-budget-persistence"
OUT = ROOT / "docs/data/seed398-paired-first-action"
ARTIFACT = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
CHECKPOINT = (
    ROOT
    / ".tmp/seed48-nextgen-s455-default-value/runs/seed48-nextgen-s455-default-value-iter1/checkpoint.npz"
)
NATIVE = ROOT / "model-artifact/runtime/kalah_v1_tablebase"
TABLEBASE = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
NATIVE_LEDGER = OUT / "native-outcomes.jsonl"
AMENDMENT = OUT / "execution-amendment-v4.json"
EXECUTION_AMENDMENT = OUT / "execution-amendment-v1.json"
PREVIOUS_AMENDMENT = OUT / "execution-amendment-v3.json"
FALLBACK_LEDGER = OUT / "outcomes.jsonl"
FALLBACK_ARCHIVE = OUT / "fallback-archive/outcomes.python-fallback.jsonl"
ANALYSIS_CORRECTION = OUT / "analysis-correction-receipt.json"
BUDGETS = (1536, 384)
OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "deterministic",
    "tactical_root_bias": 0.0,
    "root_temperature": 0.0,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((SOURCE / "registration.json").read_text())
    bind = json.loads(
        (
            ROOT / "docs/data/seed398-policy-value-composition/registration.json"
        ).read_text()
    )
    spec = {
        "schema": "seed398-paired-first-action-registration-v1",
        "scope": "retrospective reference-policy-dependent action comparison; not promotion evidence",
        "source_registration_sha256": digest(SOURCE / "registration.json"),
        "states": prior["states"],
        "reference": {
            "artifact": str(ARTIFACT.relative_to(ROOT)),
            "artifact_file_sha256": {
                name: digest(ARTIFACT / name)
                for name in ("metadata.json", "search_policy.json", "weights.json")
            },
            "checkpoint_sha256": digest(CHECKPOINT),
            "metadata": bind["component_sources"]["seed455"]["architecture"],
            "runtime_search_policy_sha256": bind["component_sources"][
                "native_runtime_contract"
            ]["runtime_search_policy_sha256"],
            "native_probe_sha256": digest(NATIVE),
            "tablebase_sha256": digest(TABLEBASE),
        },
        "search": {
            "budgets": list(BUDGETS),
            "c_puct": 1.25,
            "options": OPTIONS,
            "exact_root_solve_threshold": 16,
            "exact_leaf_solving": "disabled",
        },
        "seed_contract": "For each opening, budget, decision ply, canonical state hash, and acting player, derive azlite_eval_seed_v2 seed from SHA256 context. Branch/action identity is deliberately absent; identical contexts get identical seeds.",
        "analysis": {
            "paired_bootstrap_samples": 10000,
            "paired_bootstrap_seed": 406,
            "interval": [0.025, 0.975],
            "primary_budget": 1536,
            "decision_rule": {
                "FF": "mean >= 0.03, lower CI > 0, secondary mean >= 0",
                "SS": "mean <= -0.03, upper CI < 0, secondary mean <= 0",
                "reference-dependent": "significant primary direction reverses at secondary budget",
                "otherwise": "no clear action advantage",
            },
        },
        "source_sha256": {
            "diagnostic": digest(Path(__file__)),
            "arena": digest(ROOT / "ml/alphazero_lite/arena.py"),
            "rules": digest(ROOT / "ml/alphazero_lite/kalah_rules.py"),
            "seed_contract": digest(
                ROOT / "ml/alphazero_lite/evaluation_seed_contract.py"
            ),
        },
    }
    path = OUT / "registration.json"
    content = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != content:
        raise ValueError("immutable_registration_conflict")
    path.write_text(content)
    return spec


def verify_frozen(reg: dict[str, Any]) -> None:
    expected = reg["reference"]
    actual = {
        name: digest(ARTIFACT / name) for name in expected["artifact_file_sha256"]
    }
    if (
        actual != expected["artifact_file_sha256"]
        or digest(CHECKPOINT) != expected["checkpoint_sha256"]
    ):
        raise ValueError("reference_artifact_identity_changed")
    if (
        digest(NATIVE) != expected["native_probe_sha256"]
        or digest(TABLEBASE) != expected["tablebase_sha256"]
    ):
        raise ValueError("native_runtime_identity_changed")
    amendment = json.loads(AMENDMENT.read_text())
    if amendment["original_source_sha256"] != reg["source_sha256"]:
        raise ValueError("amendment_original_source_binding_mismatch")
    if amendment["fallback_outcomes_count"] != 10:
        raise ValueError("amendment_fallback_count_mismatch")
    if digest(FALLBACK_LEDGER) != amendment["fallback_outcomes_sha256"]:
        raise ValueError("fallback_ledger_changed_after_amendment")
    if digest(FALLBACK_ARCHIVE) != amendment["fallback_archive_sha256"]:
        raise ValueError("fallback_archive_changed_after_amendment")
    run_amendment = json.loads(EXECUTION_AMENDMENT.read_text())
    previous_amendment = json.loads(PREVIOUS_AMENDMENT.read_text())
    if digest(PREVIOUS_AMENDMENT) != amendment["prior_amendment_sha256"]:
        raise ValueError("prior_execution_amendment_changed")
    if amendment["executed_source_sha256"] != run_amendment["corrected_source_sha256"]:
        raise ValueError("executed_source_amendment_binding_mismatch")
    if (
        previous_amendment["executed_source_sha256"]
        != run_amendment["corrected_source_sha256"]
    ):
        raise ValueError("postrun_amendment_chain_mismatch")
    current = source_hashes()
    receipt = json.loads(ANALYSIS_CORRECTION.read_text())
    if receipt["executed_source_sha256"] != amendment["executed_source_sha256"]:
        raise ValueError("analysis_correction_execution_identity_mismatch")
    if receipt["post_execution_source_sha256"] != amendment["corrected_source_sha256"]:
        raise ValueError("analysis_correction_post_execution_identity_mismatch")
    if receipt["current_source_sha256"] != current:
        raise ValueError("analysis_correction_source_identity_mismatch")
    policy = load_runtime_search_policy(ARTIFACT)
    runtime = reg["reference"]
    if policy is None or policy["exact_root_threshold"] != 16:
        raise ValueError("registered_native_runtime_policy_unavailable")
    required_policy = {
        "solver_implementation_identity": "native_kvtb_root_action_probe_v1",
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve": "disabled",
    }
    if any(policy.get(key) != value for key, value in required_policy.items()):
        raise ValueError("registered_native_runtime_policy_semantics_mismatch")
    if (
        Path(policy["native_probe"]["resolved_path"]).resolve() != NATIVE.resolve()
        or Path(policy["tablebase"]["resolved_path"]).resolve() != TABLEBASE.resolve()
        or policy["native_probe"]["sha256"] != runtime["native_probe_sha256"]
        or policy["tablebase"]["sha256"] != runtime["tablebase_sha256"]
        or digest(ARTIFACT / "search_policy.json")
        != runtime["runtime_search_policy_sha256"]
    ):
        raise ValueError("registered_native_runtime_identity_mismatch")


def source_hashes() -> dict[str, str]:
    names = (
        "diagnostic",
        "arena",
        "rules",
        "seed_contract",
        "native_adapter",
        "exact_root_decision",
        "runtime_search_policy",
    )
    paths = {
        "diagnostic": Path(__file__),
        "arena": ROOT / "ml/alphazero_lite/arena.py",
        "rules": ROOT / "ml/alphazero_lite/kalah_rules.py",
        "seed_contract": ROOT / "ml/alphazero_lite/evaluation_seed_contract.py",
        "native_adapter": ROOT / "ml/alphazero_lite/native_exact_root_tablebase.py",
        "exact_root_decision": ROOT / "ml/alphazero_lite/exact_root_decision.py",
        "runtime_search_policy": ROOT / "ml/alphazero_lite/runtime_search_policy.py",
    }
    return {name: digest(paths[name]) for name in names}


def write_execution_amendment(reg: dict[str, Any]) -> dict[str, Any]:
    if not FALLBACK_ARCHIVE.is_file():
        raise FileNotFoundError("fallback outcomes must be archived before amendment")
    archive_rows = [line for line in FALLBACK_ARCHIVE.read_text().splitlines() if line]
    original_rows = [line for line in FALLBACK_LEDGER.read_text().splitlines() if line]
    if len(archive_rows) != 10 or archive_rows != original_rows:
        raise ValueError("fallback_archive_does_not_match_original_10_outcomes")
    prior = json.loads(EXECUTION_AMENDMENT.read_text())
    amendment = {
        "schema": "seed398-paired-first-action-execution-amendment-v4",
        "prior_amendment_sha256": digest(PREVIOUS_AMENDMENT),
        "executed_source_sha256": prior["corrected_source_sha256"],
        "original_registration_sha256": digest(OUT / "registration.json"),
        "original_source_sha256": reg["source_sha256"],
        "corrected_source_sha256": source_hashes(),
        "reason": "evaluate_artifact_position skips sidecar loading when an explicit exact_root_solve_threshold is supplied. With no endgame_tablebase argument, eligible roots used the Python EndgameTablebase fallback. Corrected runner supplies one registered NativeExactRootTablebase adapter for its lifetime, hash-verifies the native probe/tablebase/runtime sidecar, requires exact native handoff at every <=16-stone continuation decision, and closes the adapter in finally.",
        "post_run_source_note": "After the durable 256-game run completed, Ruff formatting/check cleanup changed formatting only; the executed source hash is retained above and the final source hash below. Experiment logic and all outcomes are unchanged.",
        "fallback_outcomes_count": len(original_rows),
        "fallback_outcomes_sha256": digest(FALLBACK_LEDGER),
        "fallback_archive_sha256": digest(FALLBACK_ARCHIVE),
        "corrected_ledger": NATIVE_LEDGER.name,
        "scope_preserved": {
            "states": 64,
            "actions": ["SS", "FF"],
            "budgets": [1536, 384],
            "games": 256,
            "analysis": reg["analysis"],
            "seed_contract": reg["seed_contract"],
        },
        "created_before_corrected_run": False,
        "execution_run_completed_before_amendment": True,
    }
    payload = json.dumps(amendment, indent=2, sort_keys=True) + "\n"
    if AMENDMENT.exists() and AMENDMENT.read_text() != payload:
        raise ValueError("execution_amendment_conflict")
    AMENDMENT.write_text(payload)
    return amendment


def play(
    state_row: dict[str, Any],
    action_name: str,
    budget: int,
    evaluator: arena.ArtifactEvaluator,
    *,
    endgame_tablebase: NativeExactRootTablebase,
) -> dict[str, Any]:
    started = time.perf_counter()
    game = KalahGame.from_state(state_row["state"])
    root_player = game.current_player
    forced = state_row["ss_action_384" if action_name == "SS" else "ff_action_384"]
    if forced not in game.possible_moves() or not game.move(game.pit_index(forced)):
        raise ValueError("forced_action_illegal")
    trajectory = [
        {
            "actor": root_player,
            "action_relative": forced,
            "action_absolute": forced + root_player * 6,
            "forced": True,
            "state": game.to_state(),
        }
    ]
    ply = 1
    while not game.over():
        actor = game.current_player
        state_hash = arena.canonical_game_state_hash(game)
        context = {
            "contract_version": "azlite_eval_seed_v2",
            "base_seed": 406,
            "suite_sha256": SUITE_SHA256,
            "opening_index": state_row["opening_index"],
            "opening_state_hash": state_row["state_hash"],
            "challenger_player": root_player,
            "game_within_opening": 0,
            "ply": ply,
            "canonical_current_state_hash": state_hash,
            "acting_role": "challenger" if actor == root_player else "current",
        }
        seed, context_hash = derive_search_seed(**context)
        result = arena.evaluate_artifact_position(
            evaluator=evaluator,
            state=game.to_state(),
            simulations=budget,
            seed=seed,
            c_puct=1.25,
            search_options=OPTIONS,
            exact_root_solve_threshold=16,
            endgame_tablebase=endgame_tablebase,
        )
        active_stones = sum(game.pits)
        exact = result.get("exact_root_decision")
        if active_stones <= 16 and (exact is None or exact.get("solver_calls") != 1):
            raise ValueError("registered_native_exact_root_handoff_missing")
        relative = int(result["selected_move"])
        legal = game.possible_moves()
        if relative not in legal:
            raise ValueError("reference_selected_illegal_action")
        absolute = game.pit_index(relative)
        if not game.move(absolute):
            raise ValueError("reference_move_rejected")
        trajectory.append(
            {
                "actor": actor,
                "action_relative": relative,
                "action_absolute": absolute,
                "forced": False,
                "seed": seed,
                "seed_context_hash": context_hash,
                "active_pit_stones_before_move": active_stones,
                "exact_root_backend": (
                    "native_kvtb_root_action_probe_v1" if active_stones <= 16 else None
                ),
                "exact_root_solver_calls": (exact["solver_calls"] if exact else 0),
                "state": game.to_state(),
            }
        )
        ply += 1
        if ply > 1000:
            raise ValueError("trajectory_safety_limit")
    margin = game.captured_seeds[root_player] - game.captured_seeds[1 - root_player]
    score = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
    return {
        "opening_index": state_row["opening_index"],
        "state_hash": state_row["state_hash"],
        "action": action_name,
        "budget": budget,
        "root_player": root_player,
        "score": score,
        "store_margin_root_perspective": margin,
        "stores": list(game.captured_seeds),
        "trajectory": trajectory,
        "elapsed_wall_seconds": time.perf_counter() - started,
    }


def validate_outcome(row: dict[str, Any], state_row: dict[str, Any]) -> None:
    if (
        row["opening_index"] != state_row["opening_index"]
        or row["state_hash"] != state_row["state_hash"]
    ):
        raise ValueError("resumed_outcome_state_binding_mismatch")
    if row["action"] not in ("SS", "FF") or row["budget"] not in BUDGETS:
        raise ValueError("resumed_outcome_key_invalid")
    game = KalahGame.from_state(state_row["state"])
    root_player = game.current_player
    moves = row["trajectory"]
    expected_forced = state_row[
        "ss_action_384" if row["action"] == "SS" else "ff_action_384"
    ]
    if (
        not moves
        or not moves[0].get("forced")
        or moves[0]["action_relative"] != expected_forced
    ):
        raise ValueError("resumed_outcome_forced_action_mismatch")
    for index, move in enumerate(moves):
        if game.over() or move["actor"] != game.current_player:
            raise ValueError("resumed_outcome_trajectory_after_terminal_or_wrong_actor")
        relative = int(move["action_relative"])
        absolute = int(move["action_absolute"])
        if (
            absolute != game.pit_index(relative)
            or relative not in game.possible_moves()
        ):
            raise ValueError("resumed_outcome_illegal_or_misindexed_action")
        if not game.move(absolute) or move["state"] != game.to_state():
            raise ValueError("resumed_outcome_state_transition_mismatch")
        if index:
            before_count = int(move.get("active_pit_stones_before_move", -1))
            expected_backend = (
                "native_kvtb_root_action_probe_v1" if before_count <= 16 else None
            )
            if before_count != sum(
                KalahGame.from_state(moves[index - 1]["state"]).pits
            ):
                raise ValueError("resumed_outcome_active_stone_count_mismatch")
            if move.get("exact_root_backend") != expected_backend:
                raise ValueError("resumed_outcome_native_backend_identity_mismatch")
            if before_count <= 16 and move.get("exact_root_solver_calls") != 1:
                raise ValueError("resumed_outcome_native_solver_call_missing")
    if not game.over() or row["stores"] != game.captured_seeds:
        raise ValueError("resumed_outcome_not_terminal_or_store_mismatch")
    margin = game.captured_seeds[root_player] - game.captured_seeds[1 - root_player]
    score = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
    if (
        row["root_player"] != root_player
        or row["store_margin_root_perspective"] != margin
        or row["score"] != score
    ):
        raise ValueError("resumed_outcome_terminal_accounting_mismatch")


def run() -> dict[str, Any]:
    reg = json.loads((OUT / "registration.json").read_text())
    verify_frozen(reg)
    evaluator = arena.ArtifactEvaluator(ARTIFACT)
    ledger_path = NATIVE_LEDGER
    completed = {}
    states = {state["opening_index"]: state for state in reg["states"]}
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            row = json.loads(line)
            validate_outcome(row, states[row["opening_index"]])
            key = (row["opening_index"], row["action"], row["budget"])
            if key in completed:
                raise ValueError("duplicate_resumed_outcome")
            completed[key] = row
    adapter = NativeExactRootTablebase(NATIVE, TABLEBASE, warm_on_start=True)
    try:
        with ledger_path.open("a") as ledger:
            for state in reg["states"]:
                for budget in BUDGETS:
                    for action in ("SS", "FF"):
                        key = (state["opening_index"], action, budget)
                        if key in completed:
                            continue
                        outcome = play(
                            state, action, budget, evaluator, endgame_tablebase=adapter
                        )
                        validate_outcome(outcome, state)
                        ledger.write(json.dumps(outcome, sort_keys=True) + "\n")
                        ledger.flush()
                        completed[key] = outcome
    finally:
        adapter.close()
    if len(completed) != 256:
        raise ValueError("outcome_ledger_count_mismatch")
    return analyze(reg, list(completed.values()))


def analyze(
    reg: dict[str, Any] | None = None, rows: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    reg = reg or json.loads((OUT / "registration.json").read_text())
    if rows is None:
        rows = [json.loads(x) for x in NATIVE_LEDGER.read_text().splitlines() if x]
    states = {state["opening_index"]: state for state in reg["states"]}
    grouped = {}
    for row in rows:
        try:
            state = states[row["opening_index"]]
        except (KeyError, TypeError) as exc:
            raise ValueError("outcome_ledger_state_invalid") from exc
        validate_outcome(row, state)
        key = (row["opening_index"], row["action"], row["budget"])
        if key in grouped:
            raise ValueError("duplicate_outcome")
        grouped[key] = row
    expected_keys = {
        (s["opening_index"], a, b)
        for s in reg["states"]
        for a in ("SS", "FF")
        for b in BUDGETS
    }
    if len(grouped) != 256 or set(grouped) != expected_keys or len(rows) != 256:
        raise ValueError("outcome_ledger_count_mismatch")
    paired = []
    for s in reg["states"]:
        result = {
            "opening_index": s["opening_index"],
            "state_hash": s["state_hash"],
            "groups": s.get("groups"),
        }
        for budget in BUDGETS:
            ss, ff = (
                grouped[(s["opening_index"], "SS", budget)],
                grouped[(s["opening_index"], "FF", budget)],
            )
            result[str(budget)] = {
                "ss_score": ss["score"],
                "ff_score": ff["score"],
                "delta": ff["score"] - ss["score"],
                "ss_margin": ss["store_margin_root_perspective"],
                "ff_margin": ff["store_margin_root_perspective"],
            }
        paired.append(result)
    rng = random.Random(406)
    deltas = [row["1536"]["delta"] for row in paired]
    means = [
        sum(deltas[rng.randrange(64)] for _ in range(64)) / 64 for _ in range(10000)
    ]
    means.sort()
    low, high = means[249], means[9749]
    primary, secondary = sum(deltas) / 64, sum(r["384"]["delta"] for r in paired) / 64
    cls = (
        "FF-action advantage"
        if primary >= 0.03 and low > 0 and secondary >= 0
        else "SS-action advantage"
        if primary <= -0.03 and high < 0 and secondary <= 0
        else "reference-dependent"
        if ((low > 0 and secondary < 0) or (high < 0 and secondary > 0))
        else "no clear action advantage"
    )
    report = {
        "schema": "seed398-paired-first-action-analysis-v1",
        "classification": cls,
        "primary": {
            "budget": 1536,
            "mean_delta": primary,
            "paired_bootstrap_95_interval": [low, high],
        },
        "secondary": {"budget": 384, "mean_delta": secondary},
        "paired_matrix": paired,
        "seed398_groups_descriptive": {
            "same_1536_action": {"n": 32},
            "persistent_model_disagreement_with_margin": {"n": 17},
        },
        "interpretation": "Retrospective reference-policy-dependent action comparison; does not establish minimax quality, overall FF strength, or promotion eligibility.",
    }
    (OUT / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


SUITE_SHA256 = "10dcb48c9a6bc268ab45ec0a67895b335bb9b669b309115d6ff37b6218131423"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("register", "amend", "run", "analyze"))
    a = p.parse_args()
    if a.stage == "register":
        register()
    elif a.stage == "amend":
        reg = json.loads((OUT / "registration.json").read_text())
        print(json.dumps(write_execution_amendment(reg), indent=2))
    elif a.stage == "run":
        run()
    else:
        print(json.dumps(analyze(), indent=2))


if __name__ == "__main__":
    main()
