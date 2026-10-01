"""Run the fixed six-model cross-order E4 arena with immutable recovery."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite.frozen_opponent_identity import validate_frozen_opponent_identity
from ml.alphazero_lite.runtime_search_policy import (
    resolve_strength_comparison_runtime_contract,
)
from ml.alphazero_lite.seed461_arena_validation import validate_arena_evidence

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
REG = DATA / "seed461-cross-order-e4-average-registration.json"
SUITE = DATA / "seed461-cross-order-e4-average-openings.jsonl"
CANDIDATES = DATA / "seed461-cross-order-e4-average-candidate-binding.json"
BINDING = DATA / "seed461-cross-order-e4-average-evaluation-binding.json"
WORK = ROOT / ".tmp/seed461-cross-order-e4-average"
OPPONENT = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
RUNS = ("P", "A1", "A2", "A3", "A4", "A5")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_inputs(reg: dict[str, Any], candidates: dict[str, Any]) -> None:
    evaluation = reg["evaluation"]
    if sha(SUITE) != evaluation["suite"]["sha256"]:
        raise ValueError("registered_suite_hash_mismatch")
    if (
        candidates.get("schema")
        != "seed461-cross-order-e4-average-candidate-binding-v1"
    ):
        raise ValueError("candidate_binding_schema_mismatch")
    for key, expected in (
        ("registration_sha256", sha(REG)),
        ("suite_sha256", sha(SUITE)),
        ("source_training_sha256", reg["training"]["training_record_sha256"]),
        ("opponent", evaluation["opponent_binding"]),
        ("runtime_contract", evaluation["runtime_contract"]),
    ):
        if candidates.get(key) != expected:
            raise ValueError(f"candidate_binding_mismatch:{key}")
    validate_frozen_opponent_identity(
        OPPONENT, evaluation["opponent_binding"], evaluation["runtime_contract"]
    )
    if set(candidates.get("candidates", {})) != set(RUNS):
        raise ValueError("candidate_set_mismatch")
    for run in RUNS:
        row = candidates["candidates"][run]
        artifact = Path(row["artifact"])
        if row["runtime_contract"] != evaluation["runtime_contract"]:
            raise ValueError(f"candidate_runtime_contract_mismatch:{run}")
        if sha(Path(row["checkpoint"])) != row["checkpoint_sha256"]:
            raise ValueError(f"candidate_checkpoint_hash_mismatch:{run}")
        for name, expected in row["artifact_sha256"].items():
            if sha(artifact / name) != expected:
                raise ValueError(f"candidate_artifact_hash_mismatch:{run}:{name}")
        if row["artifact_sha256"].get("model.npz") != row["checkpoint_sha256"]:
            raise ValueError(f"candidate_model_checkpoint_mismatch:{run}")
        if (
            resolve_strength_comparison_runtime_contract(
                current_artifact=OPPONENT, challenger_artifact=artifact
            )
            != evaluation["runtime_contract"]
        ):
            raise ValueError(f"candidate_resolved_runtime_mismatch:{run}")


def main() -> None:
    reg = json.loads(REG.read_text())
    candidates = json.loads(CANDIDATES.read_text())
    validate_inputs(reg, candidates)
    openings = [json.loads(line) for line in SUITE.read_text().splitlines() if line]
    if len(openings) != 256:
        raise ValueError("opening_count_mismatch")
    previous = json.loads(BINDING.read_text()) if BINDING.exists() else None
    binding: dict[str, Any] = {
        "schema": "seed461-cross-order-e4-average-evaluation-binding-v1",
        "registration_sha256": sha(REG),
        "candidate_binding_sha256": sha(CANDIDATES),
        "suite_sha256": sha(SUITE),
        "opponent": reg["evaluation"]["opponent_binding"],
        "candidates": candidates["candidates"],
        "reports": dict(previous.get("reports", {})) if previous else {},
    }
    if previous:
        for key, expected in binding.items():
            if key != "reports" and previous.get(key) != expected:
                raise ValueError(f"immutable_evaluation_binding_mismatch:{key}")
        for run, evidence in previous.get("reports", {}).items():
            if evidence.get("state") == "running":
                continue
            report, games = Path(evidence["report"]), Path(evidence["games"])
            if (
                not report.is_file()
                or not games.is_file()
                or sha(report) != evidence["report_sha256"]
                or sha(games) != evidence["games_sha256"]
            ):
                raise ValueError(f"cached_evidence_hash_mismatch:{run}")

    for run in RUNS:
        candidate = candidates["candidates"][run]
        report, games = (
            WORK / "arena" / f"{run}.json",
            WORK / "arena" / f"{run}-games.jsonl",
        )
        cached = binding["reports"].get(run)
        if report.exists() or games.exists():
            if cached is None:
                raise ValueError(f"unbound_cached_evaluation:{run}")
            if (
                cached.get("state") == "running"
                and report.is_file()
                and games.is_file()
            ):
                report_data = json.loads(report.read_text())
                rows = [
                    json.loads(line) for line in games.read_text().splitlines() if line
                ]
                validate_arena_evidence(
                    report_data,
                    rows,
                    openings,
                    run,
                    candidate,
                    binding["opponent"],
                    reg["evaluation"],
                )
                cached = {
                    "report": str(report),
                    "report_sha256": sha(report),
                    "games": str(games),
                    "games_sha256": sha(games),
                }
                binding["reports"][run] = cached
                write_json(BINDING, binding)
            elif (
                not report.is_file()
                or not games.is_file()
                or cached.get("report") != str(report)
                or cached.get("games") != str(games)
                or sha(report) != cached.get("report_sha256")
                or sha(games) != cached.get("games_sha256")
            ):
                raise ValueError(f"cached_evaluation_hash_mismatch:{run}")
            validate_arena_evidence(
                json.loads(report.read_text()),
                [json.loads(line) for line in games.read_text().splitlines() if line],
                openings,
                run,
                candidate,
                binding["opponent"],
                reg["evaluation"],
            )
            continue
        if cached is not None and cached.get("state") != "running":
            raise ValueError(f"bound_evaluation_missing:{run}")
        report.parent.mkdir(parents=True, exist_ok=True)
        binding["reports"][run] = {
            "state": "running",
            "report": str(report),
            "games": str(games),
        }
        write_json(BINDING, binding)
        evaluation = reg["evaluation"]
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
                "390",
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
        validate_arena_evidence(
            json.loads(report.read_text()),
            [json.loads(line) for line in games.read_text().splitlines() if line],
            openings,
            run,
            candidate,
            binding["opponent"],
            evaluation,
        )
        binding["reports"][run] = {
            "report": str(report),
            "report_sha256": sha(report),
            "games": str(games),
            "games_sha256": sha(games),
        }
        write_json(BINDING, binding)
    binding["status"] = "completed_fixed_3072_games"
    write_json(BINDING, binding)


if __name__ == "__main__":
    main()
