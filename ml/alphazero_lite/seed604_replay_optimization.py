"""Seed 604 preparation contracts for extending Seed 599 Arm B optimization.

This module is preparation-only: it cannot train or launch arena games.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("docs/data/seed604-replay-optimization")
SEED599 = Path("docs/data/seed599-replay-coverage")
UPDATES = 2048
BATCH_SIZE = 32
PARENT_SHA256 = "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad"
B512_SHA256 = "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def batch_schedule(size: int = 2048, updates: int = UPDATES) -> list[list[int]]:
    """Generate the original Seed 599 schedule prefix and its deterministic tail."""
    if size != 2048 or updates < 1:
        raise ValueError("registered_membership_or_update_count_mismatch")
    rng = random.Random(599)
    schedule: list[list[int]] = []
    while len(schedule) < updates:
        indexes = list(range(size))
        rng.shuffle(indexes)
        schedule.extend(
            indexes[start : start + BATCH_SIZE] for start in range(0, size, BATCH_SIZE)
        )
    return schedule[:updates]


def read_only_preflight(root: Path = ROOT) -> dict[str, Any]:
    """Verify immutable inputs, the exact B512 prefix and frozen Seed 604 plan."""
    root = root.resolve()
    prior = root / SEED599
    data = root / DATA
    expected_files = {
        "training-run-B/seed599-arm-B-update-512.npz": B512_SHA256,
        "training-B.json": "8dd20557ac93d7d3c72bb94b50d35e868e85ee6f68bdd84699ee312b54e26405",
        "training-execution-freeze.json": "bd141778461d7046b067966d7fb87188e5b0a2630d82655a321059b4ce17cad9",
        "evaluation-contract.json": "8d29200a27ae87daa7ea9de9a36ebf5a19d1f0f04b4f0237ff48ba1ab1af7bf4",
        "evaluation-openings.json": "3acc395a59d90df57ddf60aa1e9d99aabde220901434f72239eb48a4047bc54a",
    }
    for relative, expected in expected_files.items():
        if sha256(prior / relative) != expected:
            raise ValueError(f"seed599_immutable_identity_mismatch:{relative}")
    registered = json.loads((prior / "training-B.json").read_bytes())
    if registered["states"] != 2048 or registered["updates"] != 512:
        raise ValueError("seed599_B512_configuration_mismatch")
    old_schedule = json.loads((prior / "batch-schedule-B.json").read_bytes())
    schedule = batch_schedule()
    if schedule[:512] != old_schedule or len(schedule) != 2048:
        raise ValueError("B512_schedule_prefix_reproduction_failed")
    if any(
        len(batch) != BATCH_SIZE or len(set(batch)) != BATCH_SIZE for batch in schedule
    ):
        raise ValueError("invalid_minibatch")
    plan_path = data / "registration.json"
    plan = json.loads(plan_path.read_bytes())
    if plan.get("status") != "prepared-not-authorized" or plan.get("seed") != 604:
        raise ValueError("seed604_registration_not_frozen")
    if (
        plan.get("treatment", {}).get("batch_schedule_sha256")
        != hashlib.sha256(canonical_bytes(schedule)).hexdigest()
    ):
        raise ValueError("seed604_schedule_identity_mismatch")
    return {
        "passed": True,
        "training_or_games_started": False,
        "b512_prefix_reproduced": True,
        "schedule_updates": len(schedule),
        "training_authorized": False,
        "evaluation_authorized": False,
    }


def training_authorized(root: Path = ROOT) -> bool:
    """Distinct owner gate; no authorization record is prepared by this change."""
    root = root.resolve()
    freeze_path = root / DATA / "training-execution-freeze.json"
    auth_path = root / DATA / "training-authorization.json"
    if not freeze_path.is_file() or not auth_path.is_file():
        return False
    freeze_raw = freeze_path.read_bytes()
    freeze = json.loads(freeze_raw)
    auth = json.loads(auth_path.read_bytes())
    return (
        freeze.get("status") == "frozen-awaiting-owner-authorization"
        and auth.get("status") == "authorized-for-training"
        and auth.get("execution_freeze_sha256")
        == hashlib.sha256(freeze_raw).hexdigest()
        and auth.get("authorization", {}).get("training_may_start") is True
        and bool(auth.get("authorization", {}).get("authorized_by"))
    )


def evaluation_authorized(root: Path = ROOT) -> bool:
    """Separate evaluation owner gate, bound to its own freeze bytes."""
    root = root.resolve()
    freeze_path = root / DATA / "evaluation-execution-freeze.json"
    auth_path = root / DATA / "evaluation-authorization.json"
    if not freeze_path.is_file() or not auth_path.is_file():
        return False
    freeze_raw = freeze_path.read_bytes()
    freeze = json.loads(freeze_raw)
    auth = json.loads(auth_path.read_bytes())
    return (
        freeze.get("status") == "frozen-awaiting-owner-authorization"
        and auth.get("status") == "authorized-for-evaluation"
        and auth.get("execution_freeze_sha256")
        == hashlib.sha256(freeze_raw).hexdigest()
        and auth.get("authorization", {}).get("run_games") is True
        and bool(auth.get("authorization", {}).get("authorized_by"))
    )
