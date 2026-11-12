"""Build the deterministic, pre-game varied-opponent pilot freeze."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ml.alphazero_lite import varied_opponent_pilot as pilot

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/varied-opponent-coverage-pilot"
SOURCE = ROOT / "docs/data/seed556-exploratory-arena/screen-0005/openings.jsonl"
SEED = 578064


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def build() -> dict[str, Any]:
    source_rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line]
    openings = source_rows[:64]
    rng = random.Random(SEED)
    treatment_order = list(range(64))
    rng.shuffle(treatment_order)
    initializer_indices = set(treatment_order[:32])
    assignments: list[dict[str, Any]] = []
    for index, opening in enumerate(openings):
        opening_id = f"opening-{index:04d}"
        seed = rng.randrange(0, 2**31)
        for arm in ("control", "treatment"):
            pair_id = f"{arm}-{opening_id}"
            opponent = (
                "P"
                if arm == "control"
                else "initializer"
                if index in initializer_indices
                else "T512"
            )
            for seat in (0, 1):
                assignments.append(
                    {
                        "game_id": f"{pair_id}-P{seat}",
                        "pair_id": pair_id,
                        "arm": arm,
                        "opening_id": opening_id,
                        "opening_moves": opening["prefix_moves"],
                        "seed": seed,
                        "designated_p_seat": seat,
                        "opponent": opponent,
                        "search_workers": pilot.WORKERS_PER_GAME,
                    }
                )
    pilot.validate_game_specs(assignments)
    treatment = {
        i: [
            row
            for row in assignments
            if row["opening_id"] == f"opening-{i:04d}" and row["arm"] == "treatment"
        ]
        for i in range(64)
    }
    assert (
        sum(rows[0]["opponent"] == "initializer" for rows in treatment.values()) == 32
    )
    assert len(assignments) == 256
    components = pilot.verify_components(
        ROOT / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts"
    )
    artifact_root = (
        ROOT / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts"
    )
    runtime = pilot.effective_runtime_policy(artifact_root)
    ordered_openings = [
        {
            "opening_id": f"opening-{index:04d}",
            "source_opening_index": opening["opening_index"],
            "prefix_moves": opening["prefix_moves"],
            "state": opening["state"],
            "state_hash": opening["state_hash"],
        }
        for index, opening in enumerate(openings)
    ]
    return {
        "schema": "varied_opponent_coverage_pilot_freeze_v1",
        "source_commit": "f887a7a09c15d28ffd12674aed01aa2ca7319647",
        "seed": SEED,
        "opening_selection": "first 64 rows of the public seed556 screen-0005 openings.jsonl, preserving source order",
        "opening_source": {
            "path": "docs/data/seed556-exploratory-arena/screen-0005/openings.jsonl",
            "sha256": sha(SOURCE),
        },
        "openings": ordered_openings,
        "assignment_rule": "Each opening has a P-vs-P seat-swapped control pair and a seat-swapped treatment pair. One per-opening seed and the identical opening prefix are shared across both arms. Treatment opponent is fixed across the two seat assignments. Seeded shuffle assigns 32 openings to initializer and 32 to T512.",
        "assignments": assignments,
        "artifact_components": components,
        "runtime_assets": runtime["runtime_assets"],
        "execution_source_hashes": {
            relative: sha(ROOT / relative)
            for relative in (
                "ml/alphazero_lite/varied_opponent_pilot.py",
                "ml/alphazero_lite/run_varied_opponent_pilot.py",
                "ml/alphazero_lite/freeze_varied_opponent_pilot.py",
            )
        },
        "search_settings": pilot.SEARCH_SETTINGS,
        "target_settings": pilot.TARGET_SETTINGS,
        "runtime_policy": "Artifact sidecars: PUCT/exact-root hybrid, exact-root threshold 16, final-score-margin objective, highest legal prior then lowest move-index tie rule, exact-leaf solving disabled.",
        "runtime_limits": {
            "pair_concurrency": pilot.PAIR_CONCURRENCY,
            "workers_per_game": pilot.WORKERS_PER_GAME,
            "cpu_quota": pilot.CPU_QUOTA,
            "cpu_weight": pilot.CPU_WEIGHT,
            "memory_max_bytes": pilot.MEMORY_MAX_BYTES,
            "allowed_cpus": pilot.ALLOWED_CPUS,
            "nice": pilot.NICE,
            "torch_intraop_threads": 1,
            "torch_interop_threads": 1,
        },
        "metric": {
            "name": "distinct accepted designated-P turn positions with more than 32 active stones per CPU-hour",
            "identities": ["canonical-state", "production-kalah_v3-float32"],
            "counting": "each identity once per arm",
            "cpu_denominator": "sum of seat-swapped pair CPU, including evaluator/model setup and reaped native-probe child CPU; capped/incomplete games contribute CPU and zero accepted rows",
        },
        "decision_rule": "Propose a larger data-generation run only if treatment yield is at least 1.15 times control yield under both identity schemes. If either control yield is zero or run accounting cannot be reconciled, pilot decision is inconclusive.",
        "resume_contract": "Each verified seat-swapped pair is committed atomically with exact assignment, both full game trajectories, retained P rows, and process/native-child/pair CPU accounting. Resume validates every committed file against the freeze assignment and replay before dispatch; committed pairs are skipped. Uncommitted/interrupted attempts are preserved separately and any unreconciled CPU makes the result inconclusive.",
        "commands": {
            "preflight": "/home/alex/azlite-venv/bin/python -m ml.alphazero_lite.varied_opponent_pilot --preflight --specs docs/data/varied-opponent-coverage-pilot/game-assignments.json --artifacts docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts --tablebase /home/alex/Mancala/ai/model-artifact/runtime/kalah_v1_21.kvtb",
            "launch": "AZLITE_KALAH_V1_TABLEBASE=/home/alex/Mancala/ai/model-artifact/runtime/kalah_v1_21.kvtb systemd-run --user --scope --quiet --property=CPUQuota=1800% --property=CPUWeight=100 --property=MemoryMax=38654705664 --property=AllowedCPUs=14-31 --nice=19 -- /home/alex/azlite-venv/bin/python -m ml.alphazero_lite.run_varied_opponent_pilot --freeze docs/data/varied-opponent-coverage-pilot/freeze.json --checkpoint-dir docs/data/varied-opponent-coverage-pilot/pairs",
            "resume": "AZLITE_KALAH_V1_TABLEBASE=/home/alex/Mancala/ai/model-artifact/runtime/kalah_v1_21.kvtb systemd-run --user --scope --quiet --property=CPUQuota=1800% --property=CPUWeight=100 --property=MemoryMax=38654705664 --property=AllowedCPUs=14-31 --nice=19 -- /home/alex/azlite-venv/bin/python -m ml.alphazero_lite.run_varied_opponent_pilot --freeze docs/data/varied-opponent-coverage-pilot/freeze.json --checkpoint-dir docs/data/varied-opponent-coverage-pilot/pairs --resume",
        },
    }


def write() -> str:
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = build()
    manifest["freeze_sha256"] = digest_bytes(canonical_json(manifest))
    (DATA / "freeze.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (DATA / "game-assignments.json").write_text(
        json.dumps(manifest["assignments"], indent=2) + "\n"
    )
    (DATA / "openings.json").write_text(
        json.dumps(manifest["openings"], indent=2) + "\n"
    )
    return manifest["freeze_sha256"]


if __name__ == "__main__":
    print(write())
