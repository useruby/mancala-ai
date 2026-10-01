"""Pre-register two fresh seat-paired confirmations of frozen order 38615 A E4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import seed461_order_population as population
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
BASE = DATA / "seed461-cosine-lr-ablation-registration.json"
TRAINING = DATA / "seed461-cosine-lr-ablation-training.json"
TRAINING_BIND = DATA / "seed461-cosine-lr-ablation-training-binding.json"
SOURCE_REG = DATA / "seed461-e2-e4-average-registration.json"
SOURCE_CANDIDATE_BIND = DATA / "seed461-e2-e4-average-candidate-binding.json"
SOURCE_EVAL_BIND = DATA / "seed461-e2-e4-average-evaluation-binding.json"
SOURCE_EVAL_RESULTS = DATA / "seed461-e2-e4-average-results.json"
SOURCE_E3_RESULTS = DATA / "seed461-e3-e4-results.json"
SOURCE_CROSS_RESULTS = DATA / "seed461-cross-order-e4-average-results.json"
CANONICAL = ROOT / ".tmp/canonical-reconstruction/medium_eval.jsonl"
EXPLORATORY = DATA / "seed461-batch-order-sensitivity-exploratory-openings.jsonl"
CONSUMED = [
    DATA / name
    for name in (
        "seed461-batch-order-sensitivity-confirmation-openings.jsonl",
        "seed461-lr-sensitivity-openings-v2.jsonl",
        "seed461-cosine-lr-ablation-openings.jsonl",
        "seed461-e2-e4-average-openings.jsonl",
        "seed461-e3-e4-openings.jsonl",
        "seed461-cross-order-e4-average-openings.jsonl",
    )
]
SEEDS = (391, 392)
SUITES = {
    seed: DATA / f"order38615-a5-confirmation-seed{seed}-openings.jsonl"
    for seed in SEEDS
}
REG = DATA / "order38615-a5-confirmation-registration.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_immutable(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != payload:
        raise ValueError(f"immutable_artifact_conflict:{path.name}")
    if not path.exists():
        path.write_text(payload)


def register() -> None:
    base = json.loads(BASE.read_text())
    training = json.loads(TRAINING.read_text())
    training_bind = json.loads(TRAINING_BIND.read_text())
    source_reg = json.loads(SOURCE_REG.read_text())
    candidate_binding = json.loads(SOURCE_CANDIDATE_BIND.read_text())
    source_eval_binding = json.loads(SOURCE_EVAL_BIND.read_text())
    if training_bind["registration_sha256"] != sha(BASE) or training_bind[
        "training_sha256"
    ] != sha(TRAINING):
        raise ValueError("source_training_binding_mismatch")
    candidate_name = "order_38615_A"
    source_candidate = candidate_binding["candidates"][candidate_name]
    checkpoint = Path(source_candidate["checkpoint"])
    if (
        sha(checkpoint)
        != "f9986d511ddf70d76640b866d549e00a86bae6414761176653132e4d5163f351"
    ):
        raise ValueError("candidate_checkpoint_hash_mismatch")
    if training["trajectories"][candidate_name]["epochs"]["E4"] != sha(checkpoint):
        raise ValueError("source_training_checkpoint_mismatch")
    for filename, digest in source_candidate["artifact_sha256"].items():
        if sha(Path(source_candidate["artifact"]) / filename) != digest:
            raise ValueError(f"candidate_artifact_hash_mismatch:{filename}")
    if source_candidate["artifact_sha256"]["model.npz"] != sha(checkpoint):
        raise ValueError("candidate_model_checkpoint_mismatch")
    opponent = source_reg["evaluation"]["opponent_binding"]
    if (
        source_eval_binding["opponent"] != opponent
        or source_candidate["runtime_contract"]
        != source_reg["evaluation"]["runtime_contract"]
    ):
        raise ValueError("source_frozen_identity_mismatch")

    replay_paths = [Path(row["path"]) for row in base["training"]["replays"]]
    for replay, record in zip(replay_paths, base["training"]["replays"], strict=True):
        if sha(replay) != record["sha256"]:
            raise ValueError(f"frozen_replay_hash_mismatch:{replay}")
    proof, excluded = population.build_proof(CANONICAL, EXPLORATORY, replay_paths)
    consumed = {}
    for path in CONSUMED:
        identities = population.keys(suites.load_suite_jsonl(str(path)))
        consumed[path.name] = {
            "sha256": sha(path),
            "unique_state_identities": len(identities),
        }
        excluded |= identities
    suite_rows = {}
    overlap = {}
    for seed in SEEDS:
        rows = population.select_holdout(excluded, seed=seed, size=512)
        identities = population.keys(rows)
        if len(identities) != 512 or identities & excluded:
            raise ValueError(f"suite_identity_failure:{seed}")
        if any(
            row["pit_sum"] <= 32 or KalahGame.from_state(row["state"]).over()
            for row in rows
        ):
            raise ValueError(f"suite_eligibility_failure:{seed}")
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        if SUITES[seed].exists() and SUITES[seed].read_text() != payload:
            raise ValueError(f"immutable_suite_conflict:{seed}")
        if not SUITES[seed].exists():
            SUITES[seed].write_text(payload)
        suite_rows[seed] = rows
        overlap[seed] = identities
        excluded |= identities
    if overlap[391] & overlap[392]:
        raise ValueError("suite_pair_overlap")
    proof["consumed_suites_through_390"] = consumed
    proof["suite_identity_overlap"] = {
        "canonical_identity_sha256": {
            str(seed): sorted(population.keys(rows))
            for seed, rows in suite_rows.items()
        },
        "opening_counts": {str(seed): len(rows) for seed, rows in suite_rows.items()},
        "overlap_with_frozen_replays_historical_exploratory_and_consumed": {
            str(seed): 0 for seed in SEEDS
        },
        "cross_suite_overlap": len(overlap[391] & overlap[392]),
        "proof": "canonical state SHA-256 identities; empty intersections",
    }
    evaluation = {
        "candidate": candidate_name,
        "candidate_checkpoint_sha256": sha(checkpoint),
        "candidate_binding_source": str(SOURCE_CANDIDATE_BIND),
        "candidate_binding_source_sha256": sha(SOURCE_CANDIDATE_BIND),
        "artifact": source_candidate,
        "opponent_binding": opponent,
        "runtime_contract": source_reg["evaluation"]["runtime_contract"],
        "suite_size": 512,
        "games_per_suite": 1024,
        "games_total": 2048,
        "games_per_opening": 2,
        "suites": {
            str(seed): {
                "path": str(SUITES[seed]),
                "sha256": sha(SUITES[seed]),
                "seed": seed,
                "opening_count": 512,
            }
            for seed in SEEDS
        },
        "simulations_per_side": 384,
        "c_puct": 1.25,
        "seed_contract": "azlite_eval_seed_v2",
        "outcome_dependent_extensions": False,
    }
    value = {
        "schema": "order38615-a5-fresh-confirmation-registration-v1",
        "status": "registered_before_games",
        "selection_statement": "Candidate selection followed inspection of #386, #388, #389, and #390 results; those consumed results are selection evidence only and are excluded from confirmation estimates.",
        "training": {
            "performed": False,
            "source_registration_sha256": sha(BASE),
            "training_record_sha256": sha(TRAINING),
            "checkpoint_sha256": sha(checkpoint),
        },
        "exclusion_proof": proof,
        "evaluation": evaluation,
        "analysis": {
            "per_suite": {
                "bootstrap": "opening-cluster; seats kept together",
                "resamples": 10000,
                "seeds": {"391": 391, "392": 392},
                "interval": "95% percentile",
            },
            "pooled": {
                "bootstrap": "stratified; independently resample 512 openings within each suite, then average suite scores",
                "resamples": 10000,
                "seed": 391,
                "interval": "95% percentile",
            },
            "decision_rule": {
                "each_suite_score_at_least": 0.55,
                "each_suite_interval_lower_strictly_above": 0.50,
                "pooled_score_at_least": 0.58,
                "pooled_interval_lower_strictly_above": 0.55,
            },
            "decision": "advance to promotion review only if all prospective conditions pass; otherwise reject candidate for advancement",
            "scope": "this frozen order_38615_A E4 candidate versus this exact frozen seed455 opponent on the registered opening distribution; does not establish order generalization or validate rejected averaging treatments",
            "preserve_rejections": ["#386", "#388", "#389", "#390"],
        },
    }
    save_immutable(REG, value)
    print(f"registration_sha256={sha(REG)}")
    for seed in SEEDS:
        print(f"suite_{seed}_sha256={sha(SUITES[seed])}")
    print("cross_suite_overlap=0")


if __name__ == "__main__":
    register()
