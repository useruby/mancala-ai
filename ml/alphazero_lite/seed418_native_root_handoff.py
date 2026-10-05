"""Retrospective seed418 native exact-root feasibility diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ml.alphazero_lite import arena
from ml.alphazero_lite.evaluation_seed_contract import derive_search_seed
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.native_exact_root_tablebase import NativeExactRootTablebase
from ml.alphazero_lite.seed418_analysis import (
    analyze,
    choose_exact_action as _choose_exact_action,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed418-native-root-handoff"
SOURCE = ROOT / "docs/data/seed416-policy-target-softening"
SEED = 418
SIMULATIONS = 384
OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "deterministic",
    "tactical_root_bias": 0.0,
    "root_temperature": 0.0,
}
ARTIFACT = ROOT / ".tmp/seed416-policy-target-softening/seed455-init-runtime-check"
NATIVE = ROOT / "model-artifact/runtime/kalah_v1_tablebase"
TABLEBASE = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(state: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def choose_exact_action(
    margins: dict[int, int], priors: list[float], legal: list[int]
) -> int:
    return _choose_exact_action(margins, priors, legal)


def replay_cohort(
    suite: list[dict[str, Any]], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    openings = {index: row for index, row in enumerate(suite)}
    candidates: list[dict[str, Any]] = []
    for record in records:
        game_row = record["game"]
        if record["lane"] != "A" or int(game_row["challenger_player"]) != 0:
            continue
        opening_index = int(record["opening_id"])
        opening = openings[opening_index]
        game = KalahGame.from_state(opening["state"])
        trajectory = [int(x) for x in game_row["trajectory"].split(",") if x]
        for ply, absolute in enumerate(trajectory):
            if game.over():
                raise ValueError("source_moves_after_terminal")
            state = game.to_state()
            active = sum(game.pits)
            if 17 <= active <= 21:
                candidates.append(
                    {
                        "opening_index": opening_index,
                        "source_opening_state_hash": openings[opening_index][
                            "state_hash"
                        ],
                        "decision_ply": ply,
                        "state": state,
                        "state_hash": canonical_hash(state),
                        "active_pit_stones": active,
                        "source_record_sha256": hashlib.sha256(
                            json.dumps(
                                record, sort_keys=True, separators=(",", ":")
                            ).encode()
                        ).hexdigest(),
                        "source_game_index": int(game_row["game_index"]),
                        "source_game_within_opening": int(
                            game_row["game_within_opening"]
                        ),
                        "opening_prefix_moves": list(opening["prefix_moves"]),
                        "trajectory_prefix_absolute": trajectory[:ply],
                        "decision_absolute_action": absolute,
                    }
                )
                break
            if absolute // 6 != game.current_player or not game.move(absolute):
                raise ValueError("source_trajectory_illegal")
        else:
            if not game.over():
                raise ValueError("source_trajectory_not_terminal")
    unique: dict[str, dict[str, Any]] = {}
    for row in candidates:
        unique.setdefault(row["state_hash"], row)
    ordered = sorted(
        unique.values(),
        key=lambda row: (
            hashlib.sha256(f"418:{row['state_hash']}".encode()).hexdigest(),
            row["state_hash"],
        ),
    )
    chosen: list[dict[str, Any]] = []
    used_openings: set[int] = set()
    for row in ordered:
        if row["opening_index"] in used_openings:
            continue
        chosen.append(row)
        used_openings.add(row["opening_index"])
        if len(chosen) == 128:
            break
    if len(chosen) != 128:
        raise ValueError(f"eligible_cohort_too_small:{len(chosen)}")
    return chosen


def _source_paths() -> list[Path]:
    return [
        Path(__file__),
        ROOT / "ml/alphazero_lite/arena.py",
        ROOT / "ml/alphazero_lite/kalah_rules.py",
        ROOT / "ml/alphazero_lite/evaluation_seed_contract.py",
        ROOT / "ml/alphazero_lite/native_exact_root_tablebase.py",
        ROOT / "ml/alphazero_lite/runtime_search_policy.py",
        ROOT / "ml/alphazero_lite/exact_root_decision.py",
        ROOT / "ml/alphazero_lite/endgame_tablebase.py",
        ROOT / "ml/alphazero_lite/seed416_public_validation.py",
        ROOT / "ml/alphazero_lite/verify_seed416_policy_target_softening.py",
        ROOT / "ml/alphazero_lite/verify_seed418_native_root_handoff.py",
        ROOT / "ml/alphazero_lite/test_seed418_native_root_handoff.py",
        ROOT / "ml/alphazero_lite/seed418_analysis.py",
        ROOT / "native/kalah_v1_tablebase/kalah_v1_tablebase.cc",
    ]


def register() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    suite_path = SOURCE / "openings-v3.jsonl"
    ledger_path = SOURCE / "outcome-ledger.jsonl"
    suite = [json.loads(line) for line in suite_path.read_text().splitlines() if line]
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines() if line]
    cohort = replay_cohort(suite, ledger)
    manifest = {
        "schema": "seed418-cohort-manifest-v1",
        "selection": "lane A, challenger seat 0; first nonterminal decision at 17-21 active stones; canonical-state dedup; SHA256(418:state_hash); at most one/opening; first 128",
        "source_suite_sha256": digest(suite_path),
        "source_ledger_sha256": digest(ledger_path),
        "states": cohort,
    }
    manifest_path = DATA / "cohort-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    identities = {str(path.relative_to(ROOT)): digest(path) for path in _source_paths()}
    artifacts = {
        "artifact_dir": str(ARTIFACT.relative_to(ROOT)),
        "metadata_sha256": digest(ARTIFACT / "metadata.json"),
        "weights_sha256": digest(ARTIFACT / "weights.json"),
        "native_probe_sha256": digest(NATIVE),
        "tablebase_sha256": digest(TABLEBASE),
        "tablebase_generation_metadata_sha256": digest(
            ROOT / "docs/data/alphazero-lite-kvtb21-generation.json"
        ),
        "runtime_policy_sha256": digest(
            ROOT / ".tmp/seed416-policy-target-softening/artifacts/A/search_policy.json"
        ),
    }
    registration = {
        "schema": "seed418-preregistration-v1",
        "scope": "retrospective native-root-handoff feasibility only",
        "execution_amendment": {
            "rejected_preliminary_run": "A preliminary execution was rejected because its azlite_eval_seed_v2 opening_state_hash mistakenly used the decision-state hash; it is excluded from this evidence.",
            "accepted_run_accounting_correction": "The corrected run used the actual #416 opening-state hash and completed 128 searches plus exact queries. Its post-run aggregation initially failed because in-memory search rows were not appended, despite complete paired checkpoint records. This aggregation-only correction does not alter searches, queries, states, seeds, or rules.",
            "resume_compatible_registration_sha256": "80765bdb69541b1a74e657f3130dcfac144916c484b2a9ff07f86ad637974ccb",
        },
        "sources": identities,
        "manifest_sha256": digest(manifest_path),
        "source_suite_sha256": digest(suite_path),
        "source_ledger_sha256": digest(ledger_path),
        "artifacts": artifacts,
        "search": {
            "simulations": SIMULATIONS,
            "c_puct": 1.25,
            "seed": SEED,
            "seed_contract": "azlite_eval_seed_v2",
            "options": OPTIONS,
            "exact_root_solve_threshold": 16,
            "exact_leaf_solve": "disabled",
        },
        "native": {
            "implementation_identity": "native_kvtb_root_action_probe_v1",
            "tablebase_declared_max_tier": 21,
            "domain_evidence": "KVTB1 generation metadata header declares tier/max_tier 21; native adapter queries all states without the Python threshold-16 handoff.",
        },
        "decision_rules": {
            "require_complete_coverage": True,
            "advance_if": [
                "wdl_inferior_choice_rate>=0.05",
                "mean_final_margin_regret>=1",
                "warm_native_decision_mean_ms<=200",
                "warm_native_decision_p95_ms<=350",
            ],
            "sample_extension": False,
            "threshold_sweep": False,
        },
    }
    (DATA / "preregistration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    return registration


def verify_frozen_inputs(registration: dict[str, Any]) -> None:
    for relative, expected in registration["sources"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"frozen_execution_source_changed:{relative}")
    paths = {
        "metadata_sha256": ARTIFACT / "metadata.json",
        "weights_sha256": ARTIFACT / "weights.json",
        "native_probe_sha256": NATIVE,
        "tablebase_sha256": TABLEBASE,
        "tablebase_generation_metadata_sha256": ROOT
        / "docs/data/alphazero-lite-kvtb21-generation.json",
        "runtime_policy_sha256": ROOT
        / ".tmp/seed416-policy-target-softening/artifacts/A/search_policy.json",
    }
    for name, path in paths.items():
        if digest(path) != registration["artifacts"][name]:
            raise ValueError(f"frozen_input_changed:{name}")
    if digest(SOURCE / "openings-v3.jsonl") != registration["source_suite_sha256"]:
        raise ValueError("frozen_source_suite_changed")
    if digest(SOURCE / "outcome-ledger.jsonl") != registration["source_ledger_sha256"]:
        raise ValueError("frozen_source_ledger_changed")
    if digest(DATA / "cohort-manifest.json") != registration["manifest_sha256"]:
        raise ValueError("frozen_cohort_manifest_changed")


def run() -> None:
    reg = register()
    verify_frozen_inputs(reg)
    manifest = json.loads((DATA / "cohort-manifest.json").read_text())
    binding_path = DATA / "evidence-binding.json"
    if binding_path.exists():
        bound = json.loads(binding_path.read_text())
        if any(digest(DATA / name) != expected for name, expected in bound.items()):
            raise ValueError("completed_evidence_binding_mismatch")
        searches = sum(
            bool(line)
            for line in (DATA / "search-records.jsonl").read_text().splitlines()
        )
        oracles = sum(
            bool(line)
            for line in (DATA / "oracle-records.jsonl").read_text().splitlines()
        )
        if searches == oracles == len(manifest["states"]) == 128:
            return
        raise ValueError("bound_evidence_case_count_incomplete")
    checkpoint_path = DATA / "execution-checkpoint-v2.jsonl"
    checkpoint = (
        [json.loads(line) for line in checkpoint_path.read_text().splitlines() if line]
        if checkpoint_path.exists()
        else []
    )
    registration_hash = digest(DATA / "preregistration.json")
    allowed_registration_hashes = {registration_hash}
    amendment_resume_hash = reg["execution_amendment"].get(
        "resume_compatible_registration_sha256"
    )
    if amendment_resume_hash:
        allowed_registration_hashes.add(amendment_resume_hash)
    if any(
        row.get("preregistration_sha256") not in allowed_registration_hashes
        for row in checkpoint
    ):
        raise ValueError("resume_preregistration_mismatch")
    search_records = [row["search"] for row in checkpoint]
    oracle_records = [row["oracle"] for row in checkpoint]
    if len(checkpoint) > len(manifest["states"]):
        raise ValueError("partial_evidence_case_count_mismatch")
    for index, (srow, orow) in enumerate(
        zip(search_records, oracle_records, strict=True)
    ):
        expected_hash = manifest["states"][index]["state_hash"]
        if (
            srow.get("state_hash") != expected_hash
            or orow.get("state_hash") != expected_hash
        ):
            raise ValueError("resume_evidence_state_mismatch")
    start_index = len(search_records)
    checkpoint_stream = checkpoint_path.open("a", encoding="utf-8")
    evaluator = arena.ArtifactEvaluator(ARTIFACT)
    process_started = time.perf_counter()
    with NativeExactRootTablebase(NATIVE, TABLEBASE) as oracle:
        oracle.warm()
        startup_warm_ms = (time.perf_counter() - process_started) * 1000
        for item in manifest["states"][start_index:]:
            game = KalahGame.from_state(item["state"])
            state_hash = arena.canonical_game_state_hash(game)
            if state_hash != item["state_hash"]:
                raise ValueError("manifest_canonical_state_hash_mismatch")
            context = {
                "contract_version": "azlite_eval_seed_v2",
                "base_seed": SEED,
                "suite_sha256": reg["source_suite_sha256"],
                "opening_index": item["opening_index"],
                "opening_state_hash": item["source_opening_state_hash"],
                "challenger_player": 0,
                "game_within_opening": 0,
                "ply": item["decision_ply"],
                "canonical_current_state_hash": state_hash,
                "acting_role": "challenger" if game.current_player == 0 else "current",
            }
            seed, context_hash = derive_search_seed(**context)
            started = time.perf_counter()
            result = arena.evaluate_artifact_position(
                evaluator=evaluator,
                state=item["state"],
                simulations=384,
                seed=seed,
                c_puct=1.25,
                search_options=OPTIONS,
                exact_root_solve_threshold=16,
            )
            elapsed = (time.perf_counter() - started) * 1000
            search_record = {
                "state_hash": item["state_hash"],
                "seed": seed,
                "seed_context": context,
                "seed_context_hash": context_hash,
                "search_selected_move": int(result["selected_move"]),
                "legal_moves": result["legal_moves"],
                "policy": result["policy"],
                "visits": result["visits"],
                "child_stats": result["child_stats"],
                "search_latency_ms": elapsed,
                "budget": result.get("budget"),
            }
            native_started = time.perf_counter()
            probe_started = time.perf_counter()
            margins = oracle.root_action_margins(game, game.current_player)
            probe_latency = (time.perf_counter() - probe_started) * 1000
            priors, _ = evaluator.evaluate(game)
            legal = [int(m) for m in game.possible_moves()]
            if set(margins) != set(legal):
                raise ValueError(f"exact_root_coverage_gap:{item['state_hash']}")
            best = max(margins.values())
            optimal = [m for m in legal if margins[m] == best]
            selected = choose_exact_action(margins, [float(x) for x in priors], legal)
            decision_latency = (time.perf_counter() - native_started) * 1000
            oracle_record = {
                "state_hash": item["state_hash"],
                "root_player": game.current_player,
                "legal_moves": legal,
                "action_margins": {str(k): v for k, v in sorted(margins.items())},
                "network_priors": [float(x) for x in priors.tolist()],
                "optimal_moves": optimal,
                "selected_move": selected,
                "native_decision_latency_ms": decision_latency,
                "native_probe_latency_ms": probe_latency,
                "neural_tie_break_latency_ms": decision_latency - probe_latency,
                "coverage_complete": True,
            }
            oracle_records.append(oracle_record)
            search_records.append(search_record)
            checkpoint_stream.write(
                json.dumps(
                    {
                        "preregistration_sha256": registration_hash,
                        "search": search_record,
                        "oracle": oracle_record,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            checkpoint_stream.flush()
    checkpoint_stream.close()
    verify_frozen_inputs(reg)
    (DATA / "search-records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in search_records)
    )
    (DATA / "oracle-records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in oracle_records)
    )
    rows = []
    for item, search, oracle in zip(
        manifest["states"], search_records, oracle_records, strict=True
    ):
        rows.append(
            {
                **item,
                **search,
                "action_margins": oracle["action_margins"],
                "native_decision_latency_ms": oracle["native_decision_latency_ms"],
                "coverage_complete": oracle["coverage_complete"],
            }
        )
    analysis = analyze(rows)
    (DATA / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    )
    (DATA / "latency-accounting.json").write_text(
        json.dumps(
            {
                "startup_and_prewarm_ms": startup_warm_ms,
                "warm_native_decision_mean_ms": analysis["native_decision_latency_ms"][
                    "mean"
                ],
                "warm_native_decision_p95_ms": analysis["native_decision_latency_ms"][
                    "p95_nearest_rank"
                ],
                "scope": "warm adapter query plus neural tie-breaking; process startup/prewarm excluded and reported separately",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    bound_files = (
        "cohort-manifest.json",
        "preregistration.json",
        "search-records.jsonl",
        "oracle-records.jsonl",
        "analysis.json",
        "latency-accounting.json",
    )
    (DATA / "evidence-binding.json").write_text(
        json.dumps(
            {name: digest(DATA / name) for name in bound_files},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    if args.register_only:
        register()
    else:
        run()


if __name__ == "__main__":
    main()
