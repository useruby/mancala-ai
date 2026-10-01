"""Run or recover both immutable 1,024-game order 38615 A5 suites."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed461_arena_validation as validation
from ml.alphazero_lite.order38615_confirmation_validation import (
    validate_confirmation_inputs,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
REG = DATA / "order38615-a5-confirmation-registration.json"
CANDIDATE = DATA / "order38615-a5-confirmation-candidate-binding.json"
BINDING = DATA / "order38615-a5-confirmation-evaluation-binding.json"
WORK = ROOT / ".tmp/order38615-a5-confirmation"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def inputs(reg: dict[str, Any], cand: dict[str, Any]) -> None:
    validate_confirmation_inputs(
        REG, CANDIDATE, BINDING, reg, cand, json.loads(BINDING.read_text())
    )


def run() -> None:
    reg, cand = json.loads(REG.read_text()), json.loads(CANDIDATE.read_text())
    prior = json.loads(BINDING.read_text())
    inputs(reg, cand)
    for key, value in (
        ("registration_sha256", sha(REG)),
        ("candidate_binding_sha256", sha(CANDIDATE)),
        ("opponent", reg["evaluation"]["opponent_binding"]),
    ):
        if prior.get(key) != value:
            raise ValueError(f"evaluation_binding_mismatch:{key}")
    ev = reg["evaluation"]
    candidate = cand["candidate"]
    opponent = ev["opponent_binding"]
    for seed_text, suite_info in ev["suites"].items():
        seed = int(seed_text)
        suite_path = Path(suite_info["path"])
        openings = [
            json.loads(line) for line in suite_path.read_text().splitlines() if line
        ]
        report = WORK / f"seed{seed}.json"
        games = WORK / f"seed{seed}-games.jsonl"
        reports = prior.setdefault("reports", {})
        cached = reports.get(seed_text)
        if report.exists() or games.exists():
            if cached is None:
                raise ValueError(f"unbound_cached_evaluation:{seed}")
            if (
                cached.get("state") == "running"
                and report.is_file()
                and games.is_file()
            ):
                r = json.loads(report.read_text())
                rows = [json.loads(x) for x in games.read_text().splitlines() if x]
                validation.validate_arena_evidence(
                    r,
                    rows,
                    openings,
                    seed_text,
                    candidate,
                    opponent,
                    {
                        **ev,
                        "games_per_candidate": ev["games_per_suite"],
                        "suite": suite_info,
                        "arena_seed": seed,
                        "seed_contract": ev["seed_contract"],
                    },
                )
                cached = {
                    "report": str(report),
                    "report_sha256": sha(report),
                    "games": str(games),
                    "games_sha256": sha(games),
                }
                reports[seed_text] = cached
                write(BINDING, prior)
            elif (
                not report.is_file()
                or not games.is_file()
                or sha(report) != cached.get("report_sha256")
                or sha(games) != cached.get("games_sha256")
            ):
                raise ValueError(f"cached_evaluation_hash_mismatch:{seed}")
            else:
                r = json.loads(report.read_text())
                rows = [json.loads(x) for x in games.read_text().splitlines() if x]
                validation.validate_arena_evidence(
                    r,
                    rows,
                    openings,
                    seed_text,
                    candidate,
                    opponent,
                    {
                        **ev,
                        "games_per_candidate": ev["games_per_suite"],
                        "suite": suite_info,
                        "arena_seed": seed,
                        "seed_contract": ev["seed_contract"],
                    },
                )
            continue
        if cached and cached.get("state") != "running":
            raise ValueError(f"bound_evaluation_missing:{seed}")
        reports[seed_text] = {
            "state": "running",
            "report": str(report),
            "games": str(games),
        }
        write(BINDING, prior)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "ml/alphazero_lite/arena.py"),
                "--challenger",
                candidate["artifact"],
                "--current",
                opponent["artifact"],
                "--games",
                str(ev["games_per_suite"]),
                "--games-per-opening",
                str(ev["games_per_opening"]),
                "--opening-prefixes-jsonl",
                str(suite_path),
                "--suite-sha256",
                suite_info["sha256"],
                "--challenger-simulations",
                str(ev["simulations_per_side"]),
                "--current-simulations",
                str(ev["simulations_per_side"]),
                "--seed",
                str(seed),
                "--workers",
                "24",
                "--c-puct",
                str(ev["c_puct"]),
                "--seed-contract",
                ev["seed_contract"],
                "--game-jsonl",
                str(games),
                "--out",
                str(report),
            ],
            cwd=ROOT,
            check=True,
        )
        r = json.loads(report.read_text())
        rows = [json.loads(x) for x in games.read_text().splitlines() if x]
        validation.validate_arena_evidence(
            r,
            rows,
            openings,
            seed_text,
            candidate,
            opponent,
            {
                **ev,
                "games_per_candidate": ev["games_per_suite"],
                "suite": suite_info,
                "arena_seed": seed,
                "seed_contract": ev["seed_contract"],
            },
        )
        reports[seed_text] = {
            "report": str(report),
            "report_sha256": sha(report),
            "games": str(games),
            "games_sha256": sha(games),
        }
        write(BINDING, prior)
    if prior.get("status") != "completed_2048_games":
        prior["status"] = "completed_2048_games"
        write(BINDING, prior)


if __name__ == "__main__":
    run()
