"""Execute or verify the registered frozen E4 corrected-opening diagnostic."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed461_arena_validation as validation
from ml.alphazero_lite.bind_corrected_order38615_diagnostic import sha
from ml.alphazero_lite.build_opening_suite import (
    load_suite_jsonl,
    validate_arena_entries,
)
from ml.alphazero_lite.frozen_opponent_identity import validate_frozen_opponent_identity
from ml.alphazero_lite.opening_exclusion_contract import (
    validate_suite_against_manifest,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/order38615-a5-frozen-diagnostic-v4"
REG = DATA / "registration.json"
BIND = DATA / "evaluation-binding.json"
SOURCE_BIND = ROOT / "docs/data/order38615-a5-confirmation-candidate-binding.json"
WORK = ROOT / ".tmp/order38615-corrected-diagnostic"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run() -> None:
    registration = read_json(REG)
    binding = read_json(BIND)
    source = read_json(SOURCE_BIND)
    if binding.get("registration_sha256") != sha(REG):
        raise ValueError("registration_binding_mismatch")
    if binding.get("source_candidate_binding_sha256") != sha(SOURCE_BIND):
        raise ValueError("candidate_source_binding_mismatch")
    manifest_path = ROOT / registration["exclusion_proof"]["manifest_path"]
    if (
        binding.get("exclusion_manifest_sha256") != sha(manifest_path)
        or sha(manifest_path) != registration["exclusion_proof"]["manifest_sha256"]
    ):
        raise ValueError("exclusion_manifest_binding_mismatch")
    manifest = verify_manifest(manifest_path)
    candidate = registration["candidate"]
    opponent = registration["opponent"]
    validate_frozen_opponent_identity(
        Path(opponent["artifact"]), opponent, candidate["runtime_contract"]
    )
    for filename, digest in candidate["artifact_sha256"].items():
        if sha(Path(candidate["artifact"]) / filename) != digest:
            raise ValueError(f"candidate_artifact_hash_mismatch:{filename}")
    if source["candidate"]["checkpoint_sha256"] != candidate["checkpoint_sha256"]:
        raise ValueError("candidate_checkpoint_substitution")
    ev = registration["evaluation"]
    all_suite_states: set[str] = set()
    suite_rows: dict[str, list[dict[str, Any]]] = {}
    for seed_text, spec in ev["suites"].items():
        suite = ROOT / spec["path"]
        if sha(suite) != spec["sha256"]:
            raise ValueError(f"suite_hash_mismatch:{seed_text}")
        rows = load_suite_jsonl(str(suite))
        actual = set(validate_arena_entries(rows))
        excluded = validate_suite_against_manifest(rows, manifest)
        if len(rows) != spec["opening_count"] or actual != excluded:
            raise ValueError(f"suite_exclusion_preflight_failed:{seed_text}")
        if actual & all_suite_states:
            raise ValueError("cross_suite_opening_overlap")
        all_suite_states |= actual
        suite_rows[seed_text] = rows
    reports = binding.setdefault("reports", {})
    for seed_text, spec in ev["suites"].items():
        seed = int(seed_text)
        suite = ROOT / spec["path"]
        if sha(suite) != spec["sha256"]:
            raise ValueError(f"suite_hash_mismatch:{seed}")
        openings = suite_rows[seed_text]
        report = WORK / f"seed{seed}.json"
        games = WORK / f"seed{seed}-games.jsonl"
        cached = reports.get(seed_text)
        if report.exists() or games.exists():
            if cached is None:
                raise ValueError(f"unbound_cached_evidence:{seed}")
            if (
                cached.get("state") == "running"
                and report.is_file()
                and games.is_file()
            ):
                pass
            elif (
                not report.is_file()
                or not games.is_file()
                or sha(report) != cached.get("report_sha256")
                or sha(games) != cached.get("games_sha256")
            ):
                raise ValueError(f"cached_evidence_hash_mismatch:{seed}")
            result = read_json(report)
            rows = [json.loads(line) for line in games.read_text().splitlines() if line]
            scores = validation.validate_arena_evidence(
                result,
                rows,
                openings,
                seed_text,
                {
                    "artifact": candidate["artifact"],
                    "runtime_contract": candidate["runtime_contract"],
                },
                opponent,
                {
                    **ev,
                    "games_per_candidate": ev["games_per_suite"],
                    "suite": spec,
                    "arena_seed": seed,
                    "seed_contract": ev["seed_contract"],
                },
            )
            if cached.get("state") == "running":
                reports[seed_text] = {
                    "report": str(report),
                    "report_sha256": sha(report),
                    "games": str(games),
                    "games_sha256": sha(games),
                    "score": float(scores.mean()),
                }
                write_json(BIND, binding)
            continue

        reports[seed_text] = {
            "state": "running",
            "report": str(report),
            "games": str(games),
        }
        write_json(BIND, binding)
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
                str(suite),
                "--suite-sha256",
                spec["sha256"],
                "--challenger-simulations",
                str(ev["simulations_per_side"]),
                "--current-simulations",
                str(ev["simulations_per_side"]),
                "--seed",
                str(seed),
                "--workers",
                str(ev["workers"]),
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
        result = read_json(report)
        rows = [json.loads(line) for line in games.read_text().splitlines() if line]
        scores = validation.validate_arena_evidence(
            result,
            rows,
            openings,
            seed_text,
            {
                "artifact": candidate["artifact"],
                "runtime_contract": candidate["runtime_contract"],
            },
            opponent,
            {
                **ev,
                "games_per_candidate": ev["games_per_suite"],
                "suite": spec,
                "arena_seed": seed,
                "seed_contract": ev["seed_contract"],
            },
        )
        reports[seed_text] = {
            "report": str(report),
            "report_sha256": sha(report),
            "games": str(games),
            "games_sha256": sha(games),
            "score": float(scores.mean()),
        }
        write_json(BIND, binding)
    binding["status"] = "completed_2048_games"
    write_json(BIND, binding)


if __name__ == "__main__":
    run()
