"""Freeze corrected relative-opening suites for the E4 diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import seed461_order_population as population
from ml.alphazero_lite.arena import apply_opening_moves
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
SOURCE_REG = DATA / "order38615-a5-confirmation-registration.json"
SOURCE_BIND = DATA / "order38615-a5-confirmation-candidate-binding.json"
BASE_REG = DATA / "seed461-cosine-lr-ablation-registration.json"
CANONICAL = ROOT / ".tmp/canonical-reconstruction/medium_eval.jsonl"
EXPLORATORY = DATA / "seed461-batch-order-sensitivity-exploratory-openings.jsonl"
OUT = DATA / "order38615-corrected-diagnostic"
SEEDS = (393, 394)
PRIOR_SUITES = (
    "seed461-batch-order-sensitivity-confirmation-openings.jsonl",
    "seed461-lr-sensitivity-openings-v2.jsonl",
    "seed461-cosine-lr-ablation-openings.jsonl",
    "seed461-e2-e4-average-openings.jsonl",
    "seed461-e3-e4-openings.jsonl",
    "seed461-cross-order-e4-average-openings.jsonl",
    "order38615-a5-confirmation-seed391-openings.jsonl",
    "order38615-a5-confirmation-seed392-openings.jsonl",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_identity(row: dict[str, Any]) -> str:
    game = KalahGame.from_state(suites.INITIAL_STATE)
    apply_opening_moves(game, [int(move) for move in row["prefix_moves"]])
    # Historical runners used arena.apply_opening_moves, whose intentional
    # stop-on-illegal semantics are what this reconstruction audits.
    return suites.canonical_key(game.to_state())


def write_immutable(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != payload:
        raise ValueError(f"immutable_artifact_conflict:{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(payload)


def register() -> None:
    source = json.loads(SOURCE_REG.read_text())
    binding = json.loads(SOURCE_BIND.read_text())
    base = json.loads(BASE_REG.read_text())
    evaluation = source["evaluation"]
    candidate = binding["candidate"]
    if (
        candidate["checkpoint_sha256"]
        != "f9986d511ddf70d76640b866d549e00a86bae6414761176653132e4d5163f351"
    ):
        raise ValueError("frozen_checkpoint_mismatch")
    if candidate["runtime_contract"] != evaluation["runtime_contract"]:
        raise ValueError("frozen_runtime_mismatch")
    for name, digest in candidate["artifact_sha256"].items():
        if sha(Path(candidate["artifact"]) / name) != digest:
            raise ValueError(f"frozen_artifact_hash_mismatch:{name}")

    replay_paths = [Path(row["path"]) for row in base["training"]["replays"]]
    proof, excluded = population.build_proof(CANONICAL, EXPLORATORY, replay_paths)
    prior_rows: dict[str, list[dict[str, Any]]] = {}
    prior_actual: dict[str, set[str]] = {}
    for name in PRIOR_SUITES:
        path = DATA / name
        rows = suites.load_suite_jsonl(str(path))
        prior_rows[name] = rows
        prior_actual[name] = {replay_identity(row) for row in rows}
        excluded |= prior_actual[name]
    registered: dict[str, Any] = {}
    selected_identities: dict[int, set[str]] = {}
    suite_paths: dict[int, Path] = {}
    for seed in SEEDS:
        selected = population.select_holdout(excluded, seed=seed, size=512)
        exported = [suites.export_arena_entry(row) for row in selected]
        identities = set(suites.validate_arena_entries(exported))
        if len(identities) != 512 or identities & excluded:
            raise ValueError(f"corrected_suite_overlap:{seed}")
        if any(
            row["ply"] > 8
            or row["pit_sum"] <= 32
            or KalahGame.from_state(row["state"]).over()
            for row in selected
        ):
            raise ValueError(f"corrected_suite_eligibility:{seed}")
        path = OUT / f"seed{seed}-openings-v2.jsonl"
        suites.write_suite_jsonl(exported, str(path))
        suite_paths[seed] = path
        selected_identities[seed] = identities
        registered[str(seed)] = {
            "seed": seed,
            "path": str(path.relative_to(ROOT)),
            "sha256": sha(path),
            "opening_count": len(exported),
            "actual_unique_states": len(identities),
            "applied_prefix_length_distribution": {
                str(length): sum(len(row["prefix_moves"]) == length for row in exported)
                for length in sorted({len(row["prefix_moves"]) for row in exported})
            },
        }
        excluded |= identities
    if selected_identities[393] & selected_identities[394]:
        raise ValueError("corrected_suites_overlap")

    historical = {
        name: {
            "path": f"docs/data/{name}",
            "sha256": sha(DATA / name),
            "declared_openings": len(prior_rows[name]),
            "actual_unique_states": len(prior_actual[name]),
            "actual_identity_sha256": hashlib.sha256(
                "\n".join(sorted(prior_actual[name])).encode()
            ).hexdigest(),
        }
        for name in PRIOR_SUITES
    }
    value = {
        "schema": "order38615-corrected-frozen-diagnostic-registration-v1",
        "status": "registered_before_games",
        "training_exports_tuning_or_candidate_substitution": False,
        "candidate": {
            "name": "order_38615_A_E4",
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "artifact": candidate["artifact"],
            "artifact_sha256": candidate["artifact_sha256"],
            "runtime_contract": candidate["runtime_contract"],
        },
        "opponent": binding["opponent"],
        "evaluation": {
            "games_per_opening": 2,
            "games_per_suite": 1024,
            "games_total": 2048,
            "simulations_per_side": 384,
            "c_puct": 1.25,
            "seed_contract": "azlite_eval_seed_v2",
            "workers": 24,
            "outcome_dependent_extensions": False,
            "opening_contract": "arena_player_relative_v2",
            "suites": registered,
        },
        "exclusion_proof": {
            **proof,
            "prior_evaluations": historical,
            "prior_actual_state_union": len(set().union(*prior_actual.values())),
            "suite_393_394_cross_overlap": 0,
            "selected_state_identities": {
                str(seed): sorted(selected_identities[seed]) for seed in SEEDS
            },
        },
        "analysis": {
            "per_suite_bootstrap": "opening-cluster; two seats kept together",
            "per_suite_resamples": 10000,
            "per_suite_seeds": {"393": 393, "394": 394},
            "pooled_bootstrap": "equal-weight stratified opening resampling",
            "pooled_resamples": 10000,
            "pooled_seed": 393,
            "success_rule": {
                "each_score_at_least": 0.55,
                "each_interval_lower_strictly_above": 0.50,
                "pooled_score_at_least": 0.58,
                "pooled_interval_lower_strictly_above": 0.55,
            },
            "interpretation": "corrected-distribution performance only; does not override canonical promotion failure",
        },
    }
    out_path = OUT / "registration.json"
    write_immutable(out_path, value)
    print(f"registration={out_path.relative_to(ROOT)}")
    print(f"registration_sha256={sha(out_path)}")
    for seed, path in suite_paths.items():
        print(f"seed{seed}_sha256={sha(path)}")
    print("cross_suite_overlap=0")


if __name__ == "__main__":
    register()
