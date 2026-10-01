"""Execute the frozen seed461 E2–E4 averaging arena; no training or exports."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite.runtime_search_policy import (
    resolve_strength_comparison_runtime_contract,
)
from ml.alphazero_lite.seed461_arena_validation import validate_arena_evidence

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
REG = DATA / "seed461-e2-e4-average-registration.json"
SUITE = DATA / "seed461-e2-e4-average-openings.jsonl"
CANDIDATES = DATA / "seed461-e2-e4-average-candidate-binding.json"
BINDING = DATA / "seed461-e2-e4-average-evaluation-binding.json"
WORK = ROOT / ".tmp/seed461-e2-e4-average"
OPPONENT = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_inputs(reg: dict[str, Any], candidates: dict[str, Any]) -> None:
    if sha(SUITE) != reg["evaluation"]["suite"]["sha256"]:
        raise ValueError("registered_suite_hash_mismatch")
    if candidates.get("schema") != "seed461-e2-e4-average-candidate-binding-v1":
        raise ValueError("candidate_binding_schema_mismatch")
    for key, expected in (
        ("registration_sha256", sha(REG)),
        ("source_training_sha256", reg["reused_training"]["training_record_sha256"]),
        ("suite_sha256", sha(SUITE)),
        ("opponent", reg["evaluation"]["opponent_binding"]),
    ):
        if candidates.get(key) != expected:
            raise ValueError(f"candidate_binding_mismatch:{key}")
    opponent = reg["evaluation"]["opponent_binding"]
    if opponent["artifact"] != str(OPPONENT):
        raise ValueError("opponent_path_mismatch")
    for name, field in (
        ("weights.json", "weights_sha256"),
        ("metadata.json", "metadata_sha256"),
        ("search_policy.json", "sidecar_sha256"),
    ):
        if sha(OPPONENT / name) != opponent[field]:
            raise ValueError(f"opponent_hash_mismatch:{name}")
    contract = reg["evaluation"]["runtime_contract"]
    for path_key, hash_key in (
        ("exact_root_native_probe", "native_probe_sha256"),
        ("exact_root_tablebase", "tablebase_sha256"),
    ):
        if (
            sha(Path(contract[path_key])) != contract[f"{path_key}_sha256"]
            or contract[f"{path_key}_sha256"] != opponent[hash_key]
        ):
            raise ValueError(f"runtime_identity_mismatch:{path_key}")
    expected_runs = {
        f"order_{order}_{arm}"
        for order in reg["reused_training"]["orders"]
        for arm in ("A", "B")
    }
    if set(candidates.get("candidates", {})) != expected_runs:
        raise ValueError("candidate_set_mismatch")
    for run, row in candidates["candidates"].items():
        if row["runtime_contract"] != contract:
            raise ValueError(f"bound_runtime_contract_mismatch:{run}")
        if sha(Path(row["checkpoint"])) != row["checkpoint_sha256"]:
            raise ValueError(f"checkpoint_hash_mismatch:{run}")
        for name, digest in row["artifact_sha256"].items():
            if sha(Path(row["artifact"]) / name) != digest:
                raise ValueError(f"candidate_artifact_hash_mismatch:{run}:{name}")
        if row["artifact_sha256"].get("model.npz") != row["checkpoint_sha256"]:
            raise ValueError(f"candidate_model_checkpoint_mismatch:{run}")
        actual = resolve_strength_comparison_runtime_contract(
            current_artifact=OPPONENT, challenger_artifact=Path(row["artifact"])
        )
        if actual != contract:
            raise ValueError(f"resolved_runtime_contract_mismatch:{run}")


def main() -> None:
    reg, candidates = json.loads(REG.read_text()), json.loads(CANDIDATES.read_text())
    validate_inputs(reg, candidates)
    openings = [json.loads(line) for line in SUITE.read_text().splitlines() if line]
    candidate_hash = sha(CANDIDATES)
    previous = json.loads(BINDING.read_text()) if BINDING.exists() else None
    binding: dict[str, Any] = {
        "schema": "seed461-e2-e4-average-evaluation-binding-v1",
        "registration_sha256": sha(REG),
        "candidate_binding_sha256": candidate_hash,
        "suite_sha256": sha(SUITE),
        "opponent": reg["evaluation"]["opponent_binding"],
        "candidates": candidates["candidates"],
        "reports": dict(previous.get("reports", {})) if previous else {},
    }
    if previous:
        for key, expected in binding.items():
            if key == "reports":
                continue
            if previous.get(key) != expected:
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
    for run, candidate in binding["candidates"].items():
        report = WORK / "arena" / f"{run}.json"
        games = WORK / "arena" / f"{run}-games.jsonl"
        cached = binding["reports"].get(run)
        if report.exists() or games.exists():
            if cached is None:
                raise ValueError(f"unbound_cached_evaluation:{run}")
            if (
                cached.get("state") == "running"
                and report.is_file()
                and games.is_file()
            ):
                # Recover only a fully written report/game pair for an evaluation
                # already declared in the immutable binding before launch.
                report_data = json.loads(report.read_text())
                if (
                    report_data.get("games_played") != 512
                    or report_data.get("games") != 512
                ):
                    raise ValueError(f"incomplete_cached_evaluation:{run}")
                rows = [
                    json.loads(line) for line in games.read_text().splitlines() if line
                ]
                validate_arena_evidence(
                    report_data,
                    rows,
                    openings,
                    run,
                    binding["candidates"][run],
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
            report_data = json.loads(report.read_text())
            rows = [json.loads(line) for line in games.read_text().splitlines() if line]
            validate_arena_evidence(
                report_data,
                rows,
                openings,
                run,
                binding["candidates"][run],
                binding["opponent"],
                reg["evaluation"],
            )
            continue
        if cached is not None and cached.get("state") != "running":
            raise ValueError(f"bound_evaluation_missing:{run}")
        evaluation = reg["evaluation"]
        report.parent.mkdir(parents=True, exist_ok=True)
        binding["reports"][run] = {
            "state": "running",
            "report": str(report),
            "games": str(games),
        }
        write_json(BINDING, binding)
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
                str(evaluation["simulations_per_side"]),
                "--current-simulations",
                str(evaluation["simulations_per_side"]),
                "--seed",
                str(evaluation["arena_seed"]),
                "--workers",
                "24",
                "--c-puct",
                str(evaluation["c_puct"]),
                "--seed-contract",
                evaluation["seed_contract"],
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
    binding["status"] = "completed_fixed_5120_games"
    if previous is None or previous != binding:
        write_json(BINDING, binding)


if __name__ == "__main__":
    main()
