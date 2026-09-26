"""Contracts and accounting for matched historical replay attribution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_SOURCES = (
    "generic_bootstrap",
    "random_teacher",
    "opening_disagreement",
    "stability",
)
REQUIRED_TRAINING = {
    "training_seed": 461,
    "max_optimizer_updates": 1052,
    "final_checkpoint": "final",
    "epochs": 8,
    "batch_size": 512,
    "lr": 0.001,
    "lr_scheduler": "none",
    "model_type": "residual_v3",
    "hidden_sizes": "96,3",
    "input_encoding": "kalah_v3",
    "value_loss": "huber",
    "huber_delta": 1.0,
    "value_loss_weight": 0.3,
    "grad_clip": 1.0,
    "trainable_scope": "all",
    "exact_root_policy_loss_weight": 1.0,
    "policy_target_mode": "sharpened",
    "value_target_mode": "default",
}


class ReplaySourceAttributionError(ValueError):
    """Raised when an attribution arm violates the frozen causal contract."""


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest without loading a potentially large replay in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_rows(path: Path) -> int:
    """Count JSONL examples by reading rows, never by inferring from byte size."""
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def source_accounting(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return raw and weighted replay-index counts for an ordered source set."""
    rows = [source_rows(ROOT / source["path"]) for source in sources]
    weighted = [
        row_count * int(source["weight"]) for source, row_count in zip(sources, rows)
    ]
    total = sum(weighted)
    if total == 0:
        raise ReplaySourceAttributionError("replay_source_attribution_empty_index")
    return [
        {
            "name": source["name"],
            "raw_row_count": row_count,
            "configured_replay_weight": int(source["weight"]),
            "weighted_replay_index_count": weighted_count,
            "effective_weighted_share": weighted_count / total,
        }
        for source, row_count, weighted_count in zip(sources, rows, weighted)
    ]


def arm_sources(
    plan: dict[str, Any], excluded_source: str | None
) -> list[dict[str, Any]]:
    """Select exactly the requested full or one-source-omitted replay mixture."""
    if excluded_source not in (None, *HISTORICAL_SOURCES):
        raise ReplaySourceAttributionError("replay_source_attribution_unknown_source")
    return [source for source in plan["sources"] if source["name"] != excluded_source]


def verify_frozen_inputs(plan: dict[str, Any]) -> None:
    """Verify every frozen parent and replay byte stream before an optimizer runs."""
    if plan.get("schema") != "azlite_replay_source_attribution_v1":
        raise ReplaySourceAttributionError(
            "replay_source_attribution_plan_schema_invalid"
        )
    if plan.get("training") != REQUIRED_TRAINING:
        raise ReplaySourceAttributionError(
            "replay_source_attribution_training_contract_mismatch"
        )
    source_names = [source["name"] for source in plan.get("sources", [])]
    if source_names != ["fresh", *HISTORICAL_SOURCES]:
        raise ReplaySourceAttributionError("replay_source_attribution_sources_invalid")
    if plan.get("parent", {}).get("weights_sha256") != sha256_file(
        ROOT / plan["parent"]["weights_path"]
    ):
        raise ReplaySourceAttributionError(
            "replay_source_attribution_parent_hash_mismatch"
        )
    runtime = ROOT / plan["parent"]["runtime_policy_path"]
    if plan["parent"].get("runtime_policy_sha256") != sha256_file(runtime):
        raise ReplaySourceAttributionError(
            "replay_source_attribution_runtime_policy_hash_mismatch"
        )
    for source in plan["sources"]:
        path = ROOT / source["path"]
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise ReplaySourceAttributionError(
                f"replay_source_attribution_source_hash_mismatch:{source['name']}"
            )


def validate_arm_contract(baseline: dict[str, Any], treatment: dict[str, Any]) -> None:
    """Require a sibling to omit precisely one historical source and nothing else."""
    if baseline["excluded_source"] is not None:
        raise ReplaySourceAttributionError("replay_source_attribution_baseline_invalid")
    omitted = treatment.get("excluded_source")
    if omitted not in HISTORICAL_SOURCES:
        raise ReplaySourceAttributionError(
            "replay_source_attribution_missing_exclusion"
        )
    for key, value in baseline["training"].items():
        if treatment.get("training", {}).get(key) != value:
            raise ReplaySourceAttributionError(
                "replay_source_attribution_contract_mismatch"
            )
    baseline_sources = {item["name"]: item for item in baseline["sources"]}
    treatment_sources = {item["name"]: item for item in treatment["sources"]}
    if set(baseline_sources) - set(treatment_sources) != {omitted}:
        raise ReplaySourceAttributionError(
            "replay_source_attribution_contract_mismatch"
        )
    if "fresh" not in treatment_sources:
        raise ReplaySourceAttributionError(
            "replay_source_attribution_contract_mismatch"
        )
    for name, source in treatment_sources.items():
        baseline_source = baseline_sources.get(name)
        if baseline_source != source:
            raise ReplaySourceAttributionError(
                "replay_source_attribution_contract_mismatch"
            )


def training_command(
    plan: dict[str, Any], arm: dict[str, Any], output: Path
) -> list[str]:
    """Build the fixed-work trainer invocation for one validated arm."""
    config = arm["training"]
    sources = arm["sources"]
    paths = [str(ROOT / source["path"]) for source in sources]
    modes = [str(source["value_target_mode"]) for source in sources]
    return [
        str(ROOT / ".venv/bin/python"),
        "ml/alphazero_lite/train.py",
        "--data",
        paths[0],
        "--data-files",
        ",".join(paths),
        "--replay-weights",
        ",".join(str(source["weight"]) for source in sources),
        "--replay-value-target-modes",
        ",".join(modes),
        "--init-checkpoint",
        str(ROOT / plan["parent"]["init_checkpoint_path"]),
        "--out",
        str(output),
        "--epochs",
        str(config["epochs"]),
        "--max-optimizer-updates",
        str(config["max_optimizer_updates"]),
        "--final-checkpoint",
        config["final_checkpoint"],
        "--batch-size",
        str(config["batch_size"]),
        "--lr",
        str(config["lr"]),
        "--lr-scheduler",
        config["lr_scheduler"],
        "--hidden-sizes",
        config["hidden_sizes"],
        "--model-type",
        config["model_type"],
        "--input-encoding",
        config["input_encoding"],
        "--value-loss",
        config["value_loss"],
        "--huber-delta",
        str(config["huber_delta"]),
        "--value-loss-weight",
        str(config["value_loss_weight"]),
        "--grad-clip",
        str(config["grad_clip"]),
        "--trainable-scope",
        config["trainable_scope"],
        "--exact-root-policy-loss-weight",
        str(config["exact_root_policy_loss_weight"]),
        "--policy-target-mode",
        config["policy_target_mode"],
        "--value-target-mode",
        config["value_target_mode"],
        "--seed",
        str(config["training_seed"]),
        "--training-metrics-out",
        str(output.parent / "training_metrics.json"),
    ]


def write_arm_contracts(plan: dict[str, Any], out: Path) -> dict[str, Any]:
    """Materialize hash-bound F and F-X contracts before any training is started."""
    verify_frozen_inputs(plan)
    arms: dict[str, dict[str, Any]] = {}
    for name, excluded_source in plan["arms"].items():
        arms[name] = {
            "semantic_id": name,
            "excluded_source": excluded_source,
            "parent": plan["parent"],
            "training": plan["training"],
            "sources": arm_sources(plan, excluded_source),
        }
        arms[name]["replay_accounting"] = source_accounting(arms[name]["sources"])
    baseline = arms["seed461-replay-attribution-full"]
    for name, arm in arms.items():
        if name != baseline["semantic_id"]:
            validate_arm_contract(baseline, arm)
    payload = {
        "schema": "azlite_replay_source_attribution_contracts_v1",
        "semantic_identity": "seed461-historical-replay-leave-one-out-attribution",
        "arms": arms,
        "scope": {"self_play_games": 0, "canonical_games": 0, "promotions": 0},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
