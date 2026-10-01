"""Bind all fixed-E4 candidates before executing the preregistered arena."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite.runtime_search_policy import (
    resolve_strength_comparison_runtime_contract,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
REG = DATA / "seed461-cosine-lr-ablation-registration.json"
TRAINING = ROOT / ".tmp/seed461-cosine-lr-ablation/training.json"
SUITE = DATA / "seed461-cosine-lr-ablation-openings.jsonl"
WORK = ROOT / ".tmp/seed461-cosine-lr-ablation"
OPPONENT = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
RUNTIME_POLICY = ROOT / "model-artifact/current/search_policy.json"
ORDERS = range(38611, 38616)
BINDING = DATA / "seed461-cosine-lr-ablation-evaluation-binding.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    reg = json.loads(REG.read_text())
    training = json.loads(TRAINING.read_text())
    if sha(SUITE) != reg["holdout"]["sha256"] or training["registration_sha256"] != sha(
        REG
    ):
        raise RuntimeError("registered_input_hash_mismatch")
    # Read immutable state before touching artifact paths. Existing exports are
    # evidence and must never be overwritten during a resume.
    previous = json.loads(BINDING.read_text()) if BINDING.exists() else None
    binding: dict[str, Any] = {
        "schema": "seed461-cosine-lr-evaluation-binding-v1",
        "registration_sha256": sha(REG),
        "training_sha256": sha(TRAINING),
        "suite_sha256": sha(SUITE),
        "opponent": {
            "artifact": str(OPPONENT),
            "weights_sha256": sha(OPPONENT / "weights.json"),
            "metadata_sha256": sha(OPPONENT / "metadata.json"),
            "sidecar_sha256": sha(OPPONENT / "search_policy.json"),
            "native_probe_sha256": reg["evaluation"]["runtime_contract"][
                "exact_root_native_probe_sha256"
            ],
            "tablebase_sha256": reg["evaluation"]["runtime_contract"][
                "exact_root_tablebase_sha256"
            ],
        },
        "candidates": {},
        "reports": dict(previous.get("reports", {})) if previous else {},
    }
    if previous is not None:
        for field in (
            "schema",
            "registration_sha256",
            "training_sha256",
            "suite_sha256",
            "opponent",
        ):
            if previous.get(field) != binding[field]:
                raise RuntimeError(f"immutable_binding_mismatch:{field}")
    # Complete and bind all ten exports before launching any arena game.
    for seed in ORDERS:
        for arm in ("A", "B"):
            run = f"order_{seed}_{arm}"
            checkpoint = WORK / "training" / run / "E4.npz"
            expected = training["trajectories"][run]["epochs"]["E4"]
            if sha(checkpoint) != expected:
                raise RuntimeError(f"e4_checkpoint_hash_mismatch:{run}")
            artifact = WORK / "artifacts-e4" / run
            existing = previous.get("candidates", {}).get(run) if previous else None
            if existing is not None:
                if existing.get("checkpoint_sha256") != expected:
                    raise RuntimeError(f"immutable_candidate_binding_mismatch:{run}")
                for name, expected_hash in existing.get("artifact_sha256", {}).items():
                    if sha(Path(existing["artifact"]) / name) != expected_hash:
                        raise RuntimeError(f"bound_artifact_hash_mismatch:{run}:{name}")
                artifact = Path(existing["artifact"])
                hashes = existing["artifact_sha256"]
            else:
                if artifact.exists() and any(artifact.iterdir()):
                    raise RuntimeError(f"unbound_existing_artifact:{run}")
                artifact.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "ml/alphazero_lite/export_artifact.py"),
                        "--checkpoint",
                        str(checkpoint),
                        "--out-dir",
                        str(artifact),
                        "--version",
                        f"seed461-cosine-e4-{run}",
                        "--model-type",
                        "residual_v3",
                        "--rules-version",
                        "kalah_v1",
                        "--input-encoding",
                        "kalah_v3",
                    ],
                    cwd=ROOT,
                    check=True,
                )
                shutil.copy2(RUNTIME_POLICY, artifact / "search_policy.json")
                hashes = {
                    name: sha(artifact / name)
                    for name in (
                        "model.npz",
                        "weights.json",
                        "metadata.json",
                        "search_policy.json",
                    )
                }
            if hashes["model.npz"] != expected:
                raise RuntimeError(f"exported_model_checkpoint_mismatch:{run}")
            contract = resolve_strength_comparison_runtime_contract(
                current_artifact=OPPONENT, challenger_artifact=artifact
            )
            if contract != reg["evaluation"]["runtime_contract"]:
                raise RuntimeError(f"candidate_runtime_contract_mismatch:{run}")
            binding["candidates"][run] = {
                "artifact": str(artifact),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": expected,
                "artifact_sha256": hashes,
                "runtime_contract": contract,
                "epoch": "E4",
            }
    if previous is not None:
        for run, candidate in binding["candidates"].items():
            old = previous.get("candidates", {}).get(run)
            if old != candidate:
                raise RuntimeError(f"immutable_candidate_binding_mismatch:{run}")
    else:
        write_json(BINDING, binding)
    immutable = previous if previous is not None else json.loads(BINDING.read_text())
    for run, candidate in binding["candidates"].items():
        report, games = (
            WORK / "arena-e4" / f"{run}.json",
            WORK / "arena-e4" / f"{run}-games.jsonl",
        )
        report.parent.mkdir(parents=True, exist_ok=True)
        registered_report = immutable.get("reports", {}).get(run)
        if report.exists() or games.exists():
            if registered_report is None or not report.is_file() or not games.is_file():
                raise RuntimeError(f"unbound_cached_evaluation:{run}")
            if (
                sha(report) != registered_report["report_sha256"]
                or sha(games) != registered_report["games_sha256"]
            ):
                raise RuntimeError(f"cached_evaluation_hash_mismatch:{run}")
            binding["reports"][run] = registered_report
            continue
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "ml/alphazero_lite/arena.py"),
                "--challenger",
                candidate["artifact"],
                "--current",
                str(OPPONENT),
                "--games",
                "512",
                "--games-per-opening",
                "2",
                "--opening-prefixes-jsonl",
                str(SUITE),
                "--suite-sha256",
                sha(SUITE),
                "--challenger-simulations",
                "384",
                "--current-simulations",
                "384",
                "--seed",
                "386",
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
        binding["reports"][run] = {
            "report": str(report),
            "report_sha256": sha(report),
            "games": str(games),
            "games_sha256": sha(games),
        }
        write_json(BINDING, binding)
        immutable = json.loads(BINDING.read_text())
    binding["status"] = "completed_fixed_5120_games"
    write_json(BINDING, binding)


if __name__ == "__main__":
    main()
