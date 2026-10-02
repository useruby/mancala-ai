"""Register, bind, execute, and analyze the frozen O0 E1/E4 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import seed461_arena_validation as arena_validation
from ml.alphazero_lite import seed461_order_population as population
from ml.alphazero_lite.arena import apply_opening_moves
from ml.alphazero_lite.frozen_opponent_identity import validate_frozen_opponent_identity
from ml.alphazero_lite.opening_exclusion_contract import (
    create_manifest,
    historical_opening_identities,
    validate_suite_against_manifest,
    verify_manifest,
    write_immutable_json,
)
from ml.alphazero_lite.runtime_search_policy import (
    resolve_strength_comparison_runtime_contract,
)
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed461-e1-e4-corrected-diagnostic"
WORK = ROOT / ".tmp/seed461-e1-e4-corrected-diagnostic"
HISTORY = ROOT / "docs/data/seed461-batch-order-sensitivity-confirmation.json"
EPOCH_BINDING = (
    ROOT / "docs/data/seed461-batch-order-sensitivity-candidate-artifact-binding.json"
)
EPOCHS = {
    "E1": {
        "checkpoint": ROOT / ".tmp/seed461-order-confirmation/training/O0/E1.npz",
        "checkpoint_sha256": "643407f0b070acc1603e90e14e4ad8fd4579e4e34e3c109d9b0691f08267a1dc",
    },
    "E4": {
        "checkpoint": ROOT / ".tmp/seed461-order-confirmation/training/O0/E4.npz",
        "checkpoint_sha256": "4bc05f8284d5adb5beac5090dbf2c09686c84831131cf3ceede606587ec127f4",
    },
}
EPOCH_BINDINGS = {
    row["epoch"]: row
    for row in json.loads(EPOCH_BINDING.read_text())["candidates"]
    if row["trajectory"] == "O0" and row["epoch"] in EPOCHS
}
OPPONENT_DIR = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
OPPONENT = {
    "artifact": str(OPPONENT_DIR),
    "metadata_sha256": "8b9f4d02395271accd5accb5cde20ec6ddc4ecd63ded192b0b84d9ecf6c649ac",
    "sidecar_sha256": "b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d",
    "tablebase_sha256": "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7",
    "native_probe_sha256": "d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48",
    "weights_sha256": "f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c",
}
SEED455_CHECKPOINT = (
    ROOT
    / ".tmp/seed48-nextgen-s455-default-value/runs/seed48-nextgen-s455-default-value-iter1/checkpoint.npz"
)
SEED455_CHECKPOINT_SHA256 = (
    "c18beeaa16ebaa038cd383cfd222705c636e2b6398c7270e6674d33231aa9dd1"
)
PRIOR_SUITES = (
    "seed461-batch-order-sensitivity-confirmation-openings.jsonl",
    "seed461-lr-sensitivity-openings-v2.jsonl",
    "seed461-cosine-lr-ablation-openings.jsonl",
    "seed461-e2-e4-average-openings.jsonl",
    "seed461-e3-e4-openings.jsonl",
    "seed461-cross-order-e4-average-openings.jsonl",
    "order38615-a5-confirmation-seed391-openings.jsonl",
    "order38615-a5-confirmation-seed392-openings.jsonl",
    "order38615-corrected-diagnostic/seed393-openings-v2.jsonl",
    "order38615-corrected-diagnostic/seed394-openings-v2.jsonl",
    "order38615-a5-frozen-diagnostic-v4/seed395-openings-v2.jsonl",
    "order38615-a5-frozen-diagnostic-v4/seed396-openings-v2.jsonl",
)
SOURCE_PATHS = (
    "ml/alphazero_lite/seed461_e1_e4_diagnostic.py",
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/seed461_arena_validation.py",
    "ml/alphazero_lite/opening_exclusion_contract.py",
    "ml/alphazero_lite/seed461_order_population.py",
    "ml/alphazero_lite/build_opening_suite.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/runtime_search_policy.py",
    "ml/alphazero_lite/frozen_opponent_identity.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def source_hashes() -> dict[str, str]:
    return {name: sha(ROOT / name) for name in SOURCE_PATHS}


def registered_inputs() -> dict[str, Any]:
    history = read_json(HISTORY)
    records = {}
    for epoch, expected in EPOCHS.items():
        checkpoint = expected["checkpoint"]
        binding = EPOCH_BINDINGS[epoch]
        if sha(checkpoint) != expected["checkpoint_sha256"]:
            raise ValueError(f"source_checkpoint_hash_mismatch:{epoch}")
        if binding["checkpoint_sha256"] != expected["checkpoint_sha256"]:
            raise ValueError(f"epoch_binding_checkpoint_mismatch:{epoch}")
        artifact = Path(binding["artifact"])
        files = {
            name: sha(artifact / name)
            for name in (
                "model.npz",
                "weights.json",
                "metadata.json",
                "search_policy.json",
            )
        }
        if files["model.npz"] != expected["checkpoint_sha256"]:
            raise ValueError(f"runtime_export_model_mismatch:{epoch}")
        if files["metadata.json"] != binding["metadata_sha256"]:
            raise ValueError(f"runtime_export_metadata_mismatch:{epoch}")
        if files["weights.json"] != binding["weights_sha256"]:
            raise ValueError(f"runtime_export_weights_mismatch:{epoch}")
        if (
            files["search_policy.json"]
            != binding["runtime_contract"]["runtime_search_policy_sha256"]
        ):
            raise ValueError(f"runtime_export_sidecar_mismatch:{epoch}")
        resolved_contract = resolve_strength_comparison_runtime_contract(
            current_artifact=OPPONENT_DIR, challenger_artifact=artifact
        )
        if resolved_contract != binding["runtime_contract"]:
            raise ValueError(f"resolved_runtime_contract_mismatch:{epoch}")
        records[epoch] = {
            "source_checkpoint": str(checkpoint.relative_to(ROOT)),
            "source_checkpoint_sha256": sha(checkpoint),
            "source_checkpoint_available": True,
            "epoch_binding_sha256": binding["checkpoint_sha256"],
            "artifact": str(artifact),
            "artifact_sha256": files,
            "runtime_contract": binding["runtime_contract"],
        }
    if "E1" not in records or "E4" not in records:
        raise ValueError("original_o0_epoch_binding_missing")
    opponent_contract = records["E1"]["runtime_contract"]
    for epoch in ("E1", "E4"):
        if records[epoch]["runtime_contract"] != opponent_contract:
            raise ValueError(f"epoch_runtime_contract_mismatch:{epoch}")
    validate_frozen_opponent_identity(OPPONENT_DIR, OPPONENT, opponent_contract)
    if sha(SEED455_CHECKPOINT) != SEED455_CHECKPOINT_SHA256:
        raise ValueError("seed455_checkpoint_hash_mismatch")
    # Bind the published seed455 parent lineage as well as its runtime files.
    if (
        sha(ROOT / "model-artifact/runtime/kalah_v1_tablebase")
        != OPPONENT["native_probe_sha256"]
    ):
        raise ValueError("native_probe_hash_mismatch")
    if (
        sha(ROOT / "model-artifact/runtime/kalah_v1_21.kvtb")
        != OPPONENT["tablebase_sha256"]
    ):
        raise ValueError("tablebase_hash_mismatch")
    return {
        "epochs": records,
        "opponent": OPPONENT,
        "runtime_contract": opponent_contract,
        "history_registration_sha256": sha(HISTORY),
        "epoch_binding_sha256": sha(EPOCH_BINDING),
        "source_hashes": source_hashes(),
        "historical_training": history["training"],
    }


def exclusion_sources() -> list[dict[str, str]]:
    history = read_json(HISTORY)
    specs = [
        {
            "path": ".tmp/canonical-reconstruction/medium_eval.jsonl",
            "kind": "state_jsonl",
        },
        {
            "path": "docs/data/seed461-batch-order-sensitivity-exploratory-openings.jsonl",
            "kind": "state_jsonl",
        },
        {
            "path": "ml/alphazero_lite/run_pr249_fresh_suite_generalization.py",
            "kind": "historical_population",
        },
        *({"path": source, "kind": "source_code"} for source in SOURCE_PATHS),
        *(
            {
                "path": str(Path(row["path"]).resolve().relative_to(ROOT)),
                "kind": "training_replay",
            }
            for row in history["training"]["replays"]
        ),
        *(
            {"path": f"docs/data/{name}", "kind": "historical_suite"}
            for name in PRIOR_SUITES
        ),
    ]
    return specs


def _strict_suite(rows: list[dict[str, Any]], excluded: set[str]) -> set[str]:
    identities = suites.validate_arena_entries(rows)
    actual = set(identities)
    validate_suite_against_manifest(
        rows, verify_manifest(DATA / "opening-exclusion-manifest.json")
    )
    if len(rows) != 512 or len(actual) != 512:
        raise ValueError("suite_count_or_uniqueness_failure")
    for index, row in enumerate(rows):
        game = KalahGame.from_state(suites.INITIAL_STATE)
        prefix = [int(move) for move in row["prefix_moves"]]
        if apply_opening_moves(game, prefix) != len(prefix):
            raise ValueError(f"suite_prefix_replay_failure:{index}")
        if len(prefix) > 8 or game.over() or sum(game.pits) <= 32:
            raise ValueError(f"suite_opening_ineligible:{index}")
        if suites.canonical_key(game.to_state()) != row["state_hash"]:
            raise ValueError(f"suite_declared_state_mismatch:{index}")
    if actual & excluded:
        raise ValueError("suite_exclusion_overlap")
    return actual


def register() -> None:
    inputs = registered_inputs()
    manifest = create_manifest(exclusion_sources())
    DATA.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA / "opening-exclusion-manifest.json"
    write_immutable_json(manifest_path, manifest)
    excluded = set(manifest["excluded_state_identities"])
    suite_path = DATA / "seed397-openings-v2.jsonl"
    selected = population.select_holdout(excluded, seed=397, size=512)
    exported = [suites.export_arena_entry(row) for row in selected]
    suite_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in exported)
    if suite_path.exists() and suite_path.read_text() != suite_text:
        raise ValueError("immutable_suite_conflict")
    if not suite_path.exists():
        suite_path.write_text(suite_text, encoding="utf-8")
    suite_rows = suites.load_suite_jsonl(str(suite_path))
    suite_ids = _strict_suite(suite_rows, excluded)
    if len(suite_ids) != 512:
        raise ValueError("suite_unique_identity_count_mismatch")
    suite_exclusions = {}
    for name in PRIOR_SUITES:
        path = ROOT / "docs/data" / name
        declared_ids, actual_ids = historical_opening_identities(
            suites.load_suite_jsonl(str(path))
        )
        suite_exclusions[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha(path),
            "declared_openings": len(suites.load_suite_jsonl(str(path))),
            "declared_unique_starts": len(declared_ids),
            "actual_unique_starts": len(actual_ids),
            "declared_identities": sorted(declared_ids),
            "actual_identities": sorted(actual_ids),
        }
    evaluation = {
        "opening_count": 512,
        "games_per_checkpoint": 1024,
        "games_total": 2048,
        "games_per_opening": 2,
        "both_seats_per_opening": True,
        "simulations_per_side": 384,
        "c_puct": 1.25,
        "base_seed": 397,
        "seed_contract": "azlite_eval_seed_v2",
        "outcome_dependent_extensions": False,
        "opening_contract": "arena_player_relative_v2",
        "workers": 24,
    }
    value = {
        "schema": "seed461-o0-e1-e4-corrected-diagnostic-registration-v1",
        "status": "registered_before_games",
        "hypotheses": {
            "useful_early_gain": {
                "checkpoint": "E1",
                "score_at_least": 0.55,
                "lower_95_strictly_above": 0.50,
            },
            "subsequent_weakening": {
                "paired_e1_minus_e4_at_least": 0.03,
                "lower_95_strictly_above": 0.0,
            },
            "decision": "report separately; propose replicated duration experiment only if both pass; otherwise do not adopt early stopping",
        },
        "input_binding": inputs,
        "exclusion_manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": sha(manifest_path),
            "excluded_state_count": len(excluded),
            "excluded_identity_sha256": manifest["excluded_identity_sha256"],
            "sources": suite_exclusions,
        },
        "evaluation": {
            **evaluation,
            "suite": {
                "path": str(suite_path.relative_to(ROOT)),
                "sha256": sha(suite_path),
                "opening_count": 512,
                "unique_states": len(suite_ids),
                "selection_seed": 397,
            },
        },
        "analysis": {
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 397,
            "cluster": "opening; both seats and both checkpoints resampled together",
            "interval": "95% percentile",
        },
    }
    value["execution_sources_sha256"] = source_hashes()
    reg_path = DATA / "registration.json"
    write_immutable_json(reg_path, value)
    print(
        json.dumps(
            {
                "registration_sha256": sha(reg_path),
                "suite_sha256": sha(suite_path),
                "excluded_states": len(excluded),
                "selected_openings": len(suite_ids),
            },
            indent=2,
        )
    )


def validate_frozen_inputs(
    registration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if source_hashes() != registration["execution_sources_sha256"]:
        raise ValueError("execution_source_hash_mismatch")
    inputs = registered_inputs()
    if inputs != registration["input_binding"]:
        raise ValueError("frozen_input_binding_mismatch")
    manifest_path = ROOT / registration["exclusion_manifest"]["path"]
    manifest = verify_manifest(manifest_path)
    if sha(manifest_path) != registration["exclusion_manifest"]["sha256"]:
        raise ValueError("manifest_hash_mismatch")
    suite_path = ROOT / registration["evaluation"]["suite"]["path"]
    if sha(suite_path) != registration["evaluation"]["suite"]["sha256"]:
        raise ValueError("suite_hash_mismatch")
    rows = suites.load_suite_jsonl(str(suite_path))
    ids = _strict_suite(rows, set(manifest["excluded_state_identities"]))
    if len(ids) != 512:
        raise ValueError("suite_identity_mismatch")
    binding_path = DATA / "evaluation-binding.json"
    binding = read_json(binding_path)
    if binding["source_hashes"] != source_hashes():
        raise ValueError("binding_source_hash_mismatch")
    expected_artifacts = {
        epoch: registration["input_binding"]["epochs"][epoch]["artifact_sha256"]
        for epoch in EPOCHS
    }
    if binding["artifact_hashes"] != expected_artifacts:
        raise ValueError("binding_artifact_hash_mismatch")
    if binding["registration_sha256"] != sha(DATA / "registration.json"):
        raise ValueError("registration_binding_mismatch")
    if binding["suite_sha256"] != registration["evaluation"]["suite"]["sha256"]:
        raise ValueError("binding_suite_mismatch")
    if binding["manifest_sha256"] != sha(manifest_path):
        raise ValueError("binding_manifest_mismatch")
    return inputs, binding, rows


def bind() -> None:
    reg_path = DATA / "registration.json"
    registration = read_json(reg_path)
    if registration.get("status") != "registered_before_games":
        raise ValueError("registration_status_invalid")
    inputs = registered_inputs()
    if (
        inputs != registration["input_binding"]
        or source_hashes() != registration["execution_sources_sha256"]
    ):
        raise ValueError("frozen_input_binding_mismatch")
    manifest_path = ROOT / registration["exclusion_manifest"]["path"]
    manifest = verify_manifest(manifest_path)
    suite = ROOT / registration["evaluation"]["suite"]["path"]
    rows = suites.load_suite_jsonl(str(suite))
    ids = _strict_suite(rows, set(manifest["excluded_state_identities"]))
    binding = {
        "schema": "seed461-o0-e1-e4-corrected-diagnostic-binding-v1",
        "registration_sha256": sha(reg_path),
        "manifest_sha256": sha(manifest_path),
        "suite_sha256": sha(suite),
        "source_hashes": source_hashes(),
        "artifact_hashes": {
            epoch: registration["input_binding"]["epochs"][epoch]["artifact_sha256"]
            for epoch in EPOCHS
        },
        "reports": {},
        "status": "bound_before_games",
    }
    if len(ids) != 512:
        raise ValueError("binding_suite_identity_mismatch")
    write_immutable_json(DATA / "evaluation-binding.json", binding)
    print(f"binding_sha256={sha(DATA / 'evaluation-binding.json')}")


def run() -> None:
    registration = read_json(DATA / "registration.json")
    inputs, binding, openings = validate_frozen_inputs(registration)
    ev = registration["evaluation"]
    validate_frozen_opponent_identity(
        OPPONENT_DIR, OPPONENT, inputs["runtime_contract"]
    )
    for epoch in ("E1", "E4"):
        artifact = inputs["epochs"][epoch]["artifact"]
        report = WORK / f"{epoch}.json"
        games = WORK / f"{epoch}-games.jsonl"
        cached = binding["reports"].get(epoch)
        if report.exists() or games.exists():
            if cached is None or not report.is_file() or not games.is_file():
                raise ValueError(f"unbound_or_partial_cached_evidence:{epoch}")
            if cached.get("state") != "running" and (
                sha(report) != cached.get("report_sha256")
                or sha(games) != cached.get("games_sha256")
            ):
                raise ValueError(f"cached_evidence_hash_mismatch:{epoch}")
        else:
            binding["reports"][epoch] = {
                "state": "running",
                "report": str(report),
                "games": str(games),
            }
            write_json(DATA / "evaluation-binding.json", binding)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "ml/alphazero_lite/arena.py"),
                    "--challenger",
                    artifact,
                    "--current",
                    OPPONENT["artifact"],
                    "--games",
                    "1024",
                    "--games-per-opening",
                    "2",
                    "--opening-prefixes-jsonl",
                    str(ROOT / ev["suite"]["path"]),
                    "--suite-sha256",
                    ev["suite"]["sha256"],
                    "--challenger-simulations",
                    "384",
                    "--current-simulations",
                    "384",
                    "--seed",
                    "397",
                    "--workers",
                    "24",
                    "--c-puct",
                    "1.25",
                    "--seed-contract",
                    "azlite_eval_seed_v2",
                    "--game-jsonl",
                    str(games),
                    "--out",
                    str(report),
                ],
                cwd=ROOT,
                check=True,
            )
        report_value = read_json(report)
        game_rows = [
            json.loads(line) for line in games.read_text().splitlines() if line
        ]
        score = arena_validation.validate_arena_evidence(
            report_value,
            game_rows,
            openings,
            epoch,
            {"artifact": artifact, "runtime_contract": inputs["runtime_contract"]},
            OPPONENT,
            {
                **ev,
                "games_per_candidate": 1024,
                "suite": ev["suite"],
                "arena_seed": 397,
                "seed_contract": ev["seed_contract"],
            },
        )
        binding["reports"][epoch] = {
            "report": str(report),
            "report_sha256": sha(report),
            "games": str(games),
            "games_sha256": sha(games),
            "score": float(score.mean()),
        }
        write_json(DATA / "evaluation-binding.json", binding)
    binding["status"] = "completed_2048_games"
    write_json(DATA / "evaluation-binding.json", binding)


def analyze() -> dict[str, Any]:
    registration = read_json(DATA / "registration.json")
    inputs, binding, openings = validate_frozen_inputs(registration)
    if binding.get("status") != "completed_2048_games":
        raise ValueError("diagnostic_not_complete")
    all_scores: dict[str, np.ndarray] = {}
    raw_rows: dict[str, list[dict[str, Any]]] = {}
    for epoch in ("E1", "E4"):
        record = binding["reports"][epoch]
        report_path, games_path = Path(record["report"]), Path(record["games"])
        if (
            sha(report_path) != record["report_sha256"]
            or sha(games_path) != record["games_sha256"]
        ):
            raise ValueError(f"raw_evidence_hash_mismatch:{epoch}")
        report = read_json(report_path)
        rows = [
            json.loads(line) for line in games_path.read_text().splitlines() if line
        ]
        all_scores[epoch] = arena_validation.validate_arena_evidence(
            report,
            rows,
            openings,
            epoch,
            {
                "artifact": inputs["epochs"][epoch]["artifact"],
                "runtime_contract": inputs["runtime_contract"],
            },
            OPPONENT,
            {
                **registration["evaluation"],
                "games_per_candidate": 1024,
                "suite": registration["evaluation"]["suite"],
                "arena_seed": 397,
                "seed_contract": "azlite_eval_seed_v2",
            },
        )
        raw_rows[epoch] = rows
    matrix_rows = []
    for index in range(512):
        matrix_rows.append(
            {
                "opening_index": index,
                "opening_state_hash": openings[index]["state_hash"],
                "E1_score": float(all_scores["E1"][index]),
                "E4_score": float(all_scores["E4"][index]),
                "paired_E1_minus_E4": float(
                    all_scores["E1"][index] - all_scores["E4"][index]
                ),
            }
        )
    rng = np.random.default_rng(397)
    draw_indices = rng.integers(0, 512, size=(10000, 512))
    e1_draws = all_scores["E1"][draw_indices].mean(axis=1)
    e4_draws = all_scores["E4"][draw_indices].mean(axis=1)
    paired = all_scores["E1"] - all_scores["E4"]
    paired_draws = paired[draw_indices].mean(axis=1)
    e1_ci = [float(value) for value in np.quantile(e1_draws, [0.025, 0.975])]
    e4_ci = [float(value) for value in np.quantile(e4_draws, [0.025, 0.975])]
    paired_ci = [float(value) for value in np.quantile(paired_draws, [0.025, 0.975])]
    useful_gain = float(all_scores["E1"].mean()) >= 0.55 and e1_ci[0] > 0.50
    weakening = float(paired.mean()) >= 0.03 and paired_ci[0] > 0.0
    accounting = []
    for epoch, rows in raw_rows.items():
        accounting.extend({"checkpoint": epoch, **row} for row in rows)
    accounting.sort(key=lambda row: (row["checkpoint"], row["game_index"]))
    account_path = DATA / "game-outcome-accounting.jsonl"
    account_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in accounting),
        encoding="utf-8",
    )
    matrix = {
        "schema": "seed461-o0-e1-e4-paired-score-matrix-v1",
        "registration_sha256": sha(DATA / "registration.json"),
        "evaluation_binding_sha256": sha(DATA / "evaluation-binding.json"),
        "game_outcome_accounting_sha256": sha(account_path),
        "opening_scores": matrix_rows,
    }
    matrix_path = DATA / "paired-opening-score-matrix.json"
    write_json(matrix_path, matrix)
    results = {
        "schema": "seed461-o0-e1-e4-corrected-diagnostic-results-v1",
        "status": "valid",
        "registration_sha256": sha(DATA / "registration.json"),
        "evaluation_binding_sha256": sha(DATA / "evaluation-binding.json"),
        "game_outcome_accounting_sha256": sha(account_path),
        "paired_score_matrix_sha256": sha(matrix_path),
        "total_games": len(accounting),
        "games_per_checkpoint": {epoch: len(raw_rows[epoch]) for epoch in ("E1", "E4")},
        "scores": {epoch: float(all_scores[epoch].mean()) for epoch in ("E1", "E4")},
        "intervals_95": {"E1": e1_ci, "E4": e4_ci},
        "paired_E1_minus_E4": {"mean": float(paired.mean()), "interval_95": paired_ci},
        "bootstrap": {
            "resamples": 10000,
            "seed": 397,
            "cluster": "opening; both seats and checkpoint pairing preserved",
        },
        "conclusions": {
            "useful_early_gain": useful_gain,
            "subsequent_weakening": weakening,
            "classification": "early_gain_and_subsequent_weakening"
            if useful_gain and weakening
            else "insufficient_evidence_for_early_stopping",
            "replicated_duration_experiment_proposed": bool(useful_gain and weakening),
            "adopt_early_stopping": False,
        },
        "source_checkpoint_hashes": {
            epoch: EPOCHS[epoch]["checkpoint_sha256"] for epoch in EPOCHS
        },
        "runtime_artifact_bindings": inputs["epochs"],
    }
    write_json(DATA / "results.json", results)
    write_json(DATA / "per-opening-score-matrix.json", {"opening_scores": matrix_rows})
    lines = [
        "# Seed461 original O0 E1 versus E4 corrected-arena diagnostic",
        "",
        "| Checkpoint | Score | 95% opening-cluster interval | Games | Useful-gain gate |",
        "|---|---:|---:|---:|---|",
        f"| E1 | {results['scores']['E1']:.4f} | {e1_ci[0]:.4f}–{e1_ci[1]:.4f} | 1,024 | {'PASS' if useful_gain else 'FAIL'} |",
        f"| E4 | {results['scores']['E4']:.4f} | {e4_ci[0]:.4f}–{e4_ci[1]:.4f} | 1,024 | — |",
        "",
        f"Paired mean E1−E4: **{paired.mean():.4f}**, 95% interval [{paired_ci[0]:.4f}, {paired_ci[1]:.4f}] ({'PASS' if weakening else 'FAIL'} weakening gate).",
        "",
        f"Conclusion: useful early gain {'supported' if useful_gain else 'not established'}; subsequent weakening {'supported' if weakening else 'not established'}.",
        "Early stopping is not adopted. A replicated training-duration experiment is proposed only if both preregistered conditions pass.",
        "",
        "Per-opening scores are in `per-opening-score-matrix.json`; paired cluster bootstrap is reproducible from this matrix with 10,000 draws and seed 397.",
    ]
    (DATA / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("register", "bind", "run", "analyze"))
    action = parser.parse_args().stage
    {"register": register, "bind": bind, "run": run, "analyze": analyze}[action]()


if __name__ == "__main__":
    main()
