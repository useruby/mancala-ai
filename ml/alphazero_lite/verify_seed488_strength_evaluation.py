"""Independent read-only verification for seed488 strength outcomes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed482_fresh_confirmation import encode_state

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed488-frozen-strength-evaluation")
OPENINGS = 512
REPLICATES = 20_000
SEED = 488
SEED479_RECEIPTS = {
    "docs/data/seed479-population-ablation/receipt.json": "f7c5edd6e6bedd496b145721edcd77196ccdb32e127340c81b1546a6db816314",
    "docs/data/seed479-population-ablation/registration.json": "4e2a03af089593f1ae32abc26f8a407bb5d980be7100776c091224a7ae0b2fb4",
    "docs/data/seed479-population-ablation/evidence.json": "9056f7654b1dca824eccc783c04b7ad1a68140667e28068f665d304eea9ce2fe",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _points(winner: str) -> float:
    if winner in ("challenger", "F"):
        return 1.0
    if winner == "draw":
        return 0.5
    if winner in ("current", "initializer"):
        return 0.0
    raise ValueError("outcome_unknown")


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    reg_path = out / "registration.json"
    suite_path = out / "openings.jsonl"
    receipt_path = out / "receipt.json"
    registration = json.loads(reg_path.read_text(encoding="utf-8"))
    suite_digest = _sha(suite_path)
    if registration.get("schema") != "seed488-frozen-strength-evaluation-v1":
        raise ValueError("registration_schema")
    if registration.get("status") != "frozen_before_execution":
        raise ValueError("registration_not_frozen")
    if registration.get("fresh_confirmation_authorized") is not False:
        raise ValueError("confirmation_gate_open")
    runtime = registration.get("runtime", {})
    expected_search_options = {
        "fpu_mode": "zero",
        "reuse_subtree": False,
        "normalize_values": False,
        "root_policy_mode": "deterministic",
        "tactical_root_bias": 0.0,
        "root_temperature": 0.0,
        "root_prior_transform": None,
        "random_opening_plies": 0,
    }
    if (
        runtime.get("simulations_per_move") != 1200
        or runtime.get("c_puct") != 1.25
        or runtime.get("workers") != 1
        or runtime.get("max_moves") != 200
        or runtime.get("search_options") != expected_search_options
        or runtime.get("seed_contract")
        != "azlite_eval_seed_v2; per-opening base seed = 488 + opening_index; arena's deterministic seed contract binds opening state, seat, game-within-opening, and ply"
    ):
        raise ValueError("runtime_protocol_mismatch")
    suite_contract = registration.get("opening_suite", {})
    if (
        suite_contract.get("generator_seed") != SEED
        or suite_contract.get("count") != OPENINGS
        or suite_contract.get("games_per_opening") != 2
        or suite_contract.get("game_count") != OPENINGS * 2
        or suite_contract.get("eligible_against_reconstructable_exclusions") is not True
        or "non-overlap with unresolved seed461 continuation population cannot be certified"
        not in suite_contract.get("exclusions", {}).get("limitation", "")
    ):
        raise ValueError("opening_protocol_mismatch")
    if registration.get("bootstrap") != {
        "unit": "512 opening clusters; resample with replacement and carry both seat games together",
        "replicates": REPLICATES,
        "seed": SEED,
        "advance_rule": "advance only if one-sided 95% percentile lower score bound > 0.50",
        "two_sided_interval": "95% percentile interval also reported descriptively",
    }:
        raise ValueError("bootstrap_protocol_mismatch")
    if (
        registration.get("primary_score")
        != "F wins + 0.5 * draws, divided by all 1,024 games"
        or registration.get("promotion") != "No promotion from this experiment alone."
        or registration.get("seed479_and_480_classification")
        != "promising exploratory results; preserved as exploratory"
        or registration.get("interpretation")
        != "exploratory direct strength evaluation on a newly registered suite; not fresh confirmation or an independently untouched holdout"
    ):
        raise ValueError("decision_protocol_mismatch")
    if registration["opening_suite"]["opening_suite_sha256"] != suite_digest:
        raise ValueError("suite_digest")
    impl = root / "ml/alphazero_lite/seed488_strength_evaluation.py"
    if registration.get("implementation_sha256") != _sha(impl):
        raise ValueError("implementation_binding")
    if registration.get("independent_verifier_sha256") != _sha(
        root / "ml/alphazero_lite/verify_seed488_strength_evaluation.py"
    ):
        raise ValueError("verifier_binding")
    if registration.get("synthetic_tests_sha256") != _sha(
        root / "ml/alphazero_lite/test_seed488_strength_evaluation.py"
    ):
        raise ValueError("synthetic_test_binding")
    for relative, digest in registration.get("generator_dependency_sha256", {}).items():
        if _sha(root / relative) != digest:
            raise ValueError(f"generator_dependency_binding:{relative}")
    expected_runtime = registration["runtime"].get("identities_sha256", {})
    if not expected_runtime:
        raise ValueError("runtime_identity_inventory")
    if registration["runtime"].get("python_version") != sys.version:
        raise ValueError("python_runtime_version")
    if registration["runtime"].get("numpy_version") != np.__version__:
        raise ValueError("numpy_runtime_version")
    for relative, digest in expected_runtime.items():
        if _sha(root / relative) != digest:
            raise ValueError(f"runtime_binding:{relative}")
    f_endpoint = registration["arms"]["F"]
    trajectory_path = root / f_endpoint["checkpoint_source"]
    if _sha(trajectory_path) != f_endpoint["trajectory_sha256"]:
        raise ValueError("f_trajectory_binding")
    if (
        _sha(
            root
            / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
        )
        != registration["arms"]["initializer"]["checkpoint_sha256"]
    ):
        raise ValueError("initializer_binding")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        parameter_arrays = [
            archive[f"F_step_16_post_parameter_{i:02d}"] for i in range(22)
        ]
    flattened = np.concatenate([x.reshape(-1).astype("<f4") for x in parameter_arrays])
    if (
        hashlib.sha256(flattened.tobytes()).hexdigest()
        != f_endpoint["parameter_state_sha256"]
    ):
        raise ValueError("f_parameter_identity")
    if registration["arms"].get("seed479_receipts") != {
        "receipt_sha256": SEED479_RECEIPTS[
            "docs/data/seed479-population-ablation/receipt.json"
        ],
        "registration_sha256": SEED479_RECEIPTS[
            "docs/data/seed479-population-ablation/registration.json"
        ],
        "evidence_sha256": SEED479_RECEIPTS[
            "docs/data/seed479-population-ablation/evidence.json"
        ],
    }:
        raise ValueError("seed479_receipt_binding")
    for relative, digest in SEED479_RECEIPTS.items():
        if _sha(root / relative) != digest:
            raise ValueError(f"seed479_receipt_source:{relative}")
    openings = _rows(suite_path)
    if len(openings) != OPENINGS or [int(x["opening_index"]) for x in openings] != list(
        range(OPENINGS)
    ):
        raise ValueError("opening_coverage")
    if any(
        row.get("opening_contract") != "arena_player_relative_v2" for row in openings
    ):
        raise ValueError("opening_contract")
    if all(
        row.get("opening_contract") == "arena_player_relative_v2" for row in openings
    ):
        if (
            _sha(root / "docs/data/seed482-exclusion-manifest.json.gz")
            != "129391fc88118a99552029faafb1880f1daed62476b2d1eafd7e931f621dc60d"
        ):
            raise ValueError("base_exclusion_file_hash")
        if (
            _sha(root / "docs/data/seed483-historical-probe-identities.json.gz")
            != "cb11ea5912c530f4a6c01a913b9e5083c454f639d24669a4675ef6d1c2e45c9b"
        ):
            raise ValueError("reconstructed_exclusion_file_hash")
        excluded_base = json.loads(
            gzip.decompress(
                (root / "docs/data/seed482-exclusion-manifest.json.gz").read_bytes()
            )
        )
        excluded_add = json.loads(
            gzip.decompress(
                (
                    root / "docs/data/seed483-historical-probe-identities.json.gz"
                ).read_bytes()
            )
        )
        excluded_canonical = {
            row["canonical_identity"] for row in excluded_base["identities"]
        }
        excluded_canonical.update(
            row["canonical_identity"] for row in excluded_add["identities"]
        )
        excluded_float32 = {
            row["float32_identity"] for row in excluded_base["identities"]
        }
        excluded_float32.update(
            row["float32_identity"] for row in excluded_add["identities"]
        )
        seen_states: set[str] = set()
        seen_float32: set[str] = set()
        for index, row in enumerate(openings):
            game = KalahGame.from_state(
                {
                    "player_pits": [4] * 6,
                    "opponent_pits": [4] * 6,
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }
            )
            absolute_prefix = row.get("source_prefix_moves_absolute")
            if not isinstance(absolute_prefix, list) or len(absolute_prefix) != len(
                row["prefix_moves"]
            ):
                raise ValueError("opening_absolute_prefix")
            for relative, absolute in zip(
                row["prefix_moves"], absolute_prefix, strict=True
            ):
                if int(absolute) != game.pit_index(int(relative)):
                    raise ValueError("opening_prefix_coordinate_mismatch")
                if relative not in game.possible_moves() or not game.move(
                    game.pit_index(int(relative))
                ):
                    raise ValueError("opening_prefix_replay")
            state = game.to_state()
            canonical = hashlib.sha256(
                json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            float_identity = hashlib.sha256(
                np.asarray(
                    encode_state(state, input_encoding="kalah_v3"), dtype="<f4"
                ).tobytes()
            ).hexdigest()
            if state != row["state"] or canonical != row["state_hash"]:
                raise ValueError(f"opening_state_mismatch:{index}")
            if canonical in excluded_canonical or float_identity in excluded_float32:
                raise ValueError(f"opening_exclusion_overlap:{index}")
            if canonical in seen_states or float_identity in seen_float32:
                raise ValueError("opening_duplicate_state")
            seen_states.add(canonical)
            seen_float32.add(float_identity)
    if not receipt_path.exists():
        return {"status": "prepared_not_executed", "opening_count": len(openings)}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") == "seed488_prepared_not_executed":
        for relative, digest in receipt.items():
            if relative != "status" and _sha(out / relative) != digest:
                raise ValueError(f"prepared_receipt_mismatch:{relative}")
        return {"status": "prepared_not_executed", "opening_count": len(openings)}
    if receipt.get("status") != "seed488_strength_evaluation_complete":
        raise ValueError("completion_receipt_status")
    for relative, digest in receipt.items():
        if relative != "status" and _sha(out / relative) != digest:
            raise ValueError(f"receipt_mismatch:{relative}")
    artifact_manifest = json.loads(
        (out / "runtime-artifacts.json").read_text(encoding="utf-8")
    )
    for arm, artifacts in artifact_manifest.items():
        for name, digest in artifacts.items():
            if _sha(out / "runtime-artifacts" / arm / name) != digest:
                raise ValueError(f"artifact_mismatch:{arm}:{name}")
    results_path = out / "outcomes.jsonl"
    rows = _rows(results_path)
    expected = {(opening, seat) for opening in range(OPENINGS) for seat in (0, 1)}
    observed: dict[tuple[int, int], float] = {}
    for row in rows:
        key = (int(row["opening_index"]), int(row["F_player"]))
        if key not in expected or key in observed:
            raise ValueError("duplicate_or_invalid_game")
        if row.get("opening_state_hash") != openings[key[0]]["state_hash"]:
            raise ValueError("game_opening_identity_mismatch")
        opening = key[0]
        game_log = out / "resume" / f"games-{opening:04d}.jsonl"
        report = out / "resume" / f"report-{opening:04d}.json"
        opening_file = out / "resume" / f"opening-{opening:04d}.jsonl"
        if _sha(game_log) != row.get("batch_game_log_sha256"):
            raise ValueError("game_log_receipt_mismatch")
        if _sha(report) != row.get("batch_report_sha256"):
            raise ValueError("arena_report_receipt_mismatch")
        expected_opening_bytes = (
            json.dumps(openings[opening], sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        if expected_opening_bytes != opening_file.read_bytes() or _sha(
            opening_file
        ) != row.get("batch_opening_sha256"):
            raise ValueError("paired_opening_receipt_mismatch")
        batch_rows = _rows(game_log)
        position = int(row["batch_game_position"])
        if len(batch_rows) != 2 or position not in (0, 1):
            raise ValueError("arena_batch_coverage")
        arena_row = batch_rows[position]
        arena_winner = arena_row.get("winner")
        if arena_winner == "challenger":
            arena_winner = "F"
        elif arena_winner == "current":
            arena_winner = "initializer"
        if int(arena_row.get("challenger_player", -1)) != key[
            1
        ] or arena_winner != row.get("winner"):
            raise ValueError("arena_game_outcome_mismatch")
        observed[key] = _points(str(row["winner"]))
    if observed.keys() != expected:
        raise ValueError("game_coverage")
    clusters = np.asarray(
        [(observed[(i, 0)] + observed[(i, 1)]) / 2 for i in range(OPENINGS)],
        dtype=np.float64,
    )
    score = float(clusters.mean())
    rng = np.random.default_rng(SEED)
    draw_index = rng.integers(0, OPENINGS, size=(REPLICATES, OPENINGS))
    bootstrap_scores = clusters[draw_index].mean(axis=1)
    lower = float(np.quantile(bootstrap_scores, 0.05))
    result = json.loads((out / "results.json").read_text(encoding="utf-8"))
    expected_result = {
        "game_count": OPENINGS * 2,
        "score": score,
        "bootstrap": {
            "method": "opening-paired percentile bootstrap; resample 512 opening clusters, carrying both seats together",
            "replicates": REPLICATES,
            "seed": SEED,
            "confidence_level": 0.95,
            "one_sided_lower": lower,
            "two_sided_interval": [
                float(np.quantile(bootstrap_scores, 0.025)),
                float(np.quantile(bootstrap_scores, 0.975)),
            ],
        },
        "advance": lower > 0.5,
        "promotion": False,
    }
    for key, value in expected_result.items():
        if result.get(key) != value:
            if key == "score" and np.isclose(
                result.get(key), value, rtol=0, atol=1e-15
            ):
                continue
            raise ValueError(f"result_mismatch:{key}")
    return {"status": "verified_complete", **expected_result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
