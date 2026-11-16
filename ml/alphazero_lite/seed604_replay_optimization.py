"""Seed 604 preparation contracts for extending Seed 599 Arm B optimization.

This module is preparation-only: it cannot train or launch arena games.
"""

from __future__ import annotations

import hashlib
import json
import math
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
MEMBERSHIP_SHA256 = "95ecc180e28945702e08815de7a5210e1bce036afdaf76d194bbd774589a4516"
SCHEDULE_512_SHA256 = "62edb14805d20af1cc7988d73aba95dcdf9e67842b6ea1d61a12c98a87c28599"
SEED599_CONFIG_SHA256 = (
    "8dd20557ac93d7d3c72bb94b50d35e868e85ee6f68bdd84699ee312b54e26405"
)
SEED599_FREEZE_SHA256 = (
    "bd141778461d7046b067966d7fb87188e5b0a2630d82655a321059b4ce17cad9"
)
SEED599_EVALUATION_SHA256 = (
    "8d29200a27ae87daa7ea9de9a36ebf5a19d1f0f04b4f0237ff48ba1ab1af7bf4"
)
SEED599_OPENINGS_SHA256 = (
    "3acc395a59d90df57ddf60aa1e9d99aabde220901434f72239eb48a4047bc54a"
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registered_file(root: Path, relative: str, expected: str) -> Path:
    path = root / relative
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"seed599_immutable_identity_mismatch:{relative}")
    return path


def verify_membership_identity(
    registered_sha: Any, membership_sha: str, config_sha: str
) -> None:
    """Bind Seed 604's membership field to both authenticated Seed 599 values."""
    if not isinstance(registered_sha, str) or len(registered_sha) != 64:
        raise ValueError("seed604_membership_sha256_malformed")
    try:
        int(registered_sha, 16)
    except ValueError as error:
        raise ValueError("seed604_membership_sha256_malformed") from error
    if registered_sha.lower() != membership_sha or registered_sha.lower() != config_sha:
        raise ValueError("seed604_membership_sha256_mismatch")


def _verify_published_inputs(root: Path) -> dict[str, Any]:
    """Authenticate the Seed 599 source bytes and the semantic data identities."""
    from ml.alphazero_lite.kalah_rules import KalahGame

    prior = root / SEED599
    _registered_file(
        root,
        "docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4/model.npz",
        PARENT_SHA256,
    )
    config_path = _registered_file(prior, "training-B.json", SEED599_CONFIG_SHA256)
    _registered_file(
        prior,
        "training-execution-freeze.json",
        SEED599_FREEZE_SHA256,
    )
    _registered_file(
        prior,
        "evaluation-contract.json",
        SEED599_EVALUATION_SHA256,
    )
    _registered_file(prior, "evaluation-openings.json", SEED599_OPENINGS_SHA256)
    _registered_file(
        prior,
        "training-run-B/seed599-arm-B-update-512.npz",
        B512_SHA256,
    )
    config = json.loads(config_path.read_bytes())
    freeze = json.loads((prior / "training-execution-freeze.json").read_bytes())
    if (
        config.get("schema") != "seed599_training_configuration_v1"
        or config.get("status") != "frozen-pretraining-not-authorized"
        or config.get("arm") != "B"
        or config.get("states") != 2048
        or config.get("updates") != 512
        or config.get("batch_size") != BATCH_SIZE
        or config.get("parent_checkpoint_sha256") != PARENT_SHA256
        or freeze.get("arms", {}).get("B", {}).get("configuration_sha256")
        != SEED599_CONFIG_SHA256
    ):
        raise ValueError("seed599_B512_frozen_configuration_mismatch")
    membership = json.loads((prior / "treatment-membership.json").read_bytes())
    membership_sha = hashlib.sha256(canonical_bytes(membership)).hexdigest()
    if (
        len(membership) != config["states"]
        or membership_sha != MEMBERSHIP_SHA256
        or config.get("membership_sha256") != MEMBERSHIP_SHA256
    ):
        raise ValueError("seed599_B_membership_identity_mismatch")
    registration = json.loads((root / DATA / "registration.json").read_bytes())
    verify_membership_identity(
        registration.get("treatment", {}).get("membership_sha256"),
        membership_sha,
        config.get("membership_sha256", ""),
    )
    for index, record in enumerate(membership):
        try:
            target = record["target"]
            legal = KalahGame.from_state(record["canonical_state"]).possible_moves()
            valid = (
                len(target) == 6
                and all(
                    math.isfinite(float(value)) and float(value) >= 0
                    for value in target
                )
                and abs(sum(float(value) for value in target) - 1.0) <= 1e-6
                and legal == record["legal_moves"]
                and all(
                    float(target[action]) == 0
                    for action in range(6)
                    if action not in legal
                )
            )
        except (KeyError, TypeError, ValueError, IndexError):
            valid = False
        if not valid:
            raise ValueError(f"seed599_B_target_invalid:{index}")
    schedule = json.loads((prior / "batch-schedule-B.json").read_bytes())
    schedule_sha = hashlib.sha256(canonical_bytes(schedule)).hexdigest()
    if (
        len(schedule) != 512
        or schedule_sha != SCHEDULE_512_SHA256
        or config.get("batch_schedule_sha256") != SCHEDULE_512_SHA256
    ):
        raise ValueError("seed599_B512_schedule_identity_mismatch")
    return {
        "parent_checkpoint_sha256": PARENT_SHA256,
        "membership_sha256": membership_sha,
        "membership_states": len(membership),
        "all_targets_validated": True,
        "published_schedule_sha256": schedule_sha,
        "b512_checkpoint_sha256": B512_SHA256,
        "training_configuration_sha256": SEED599_CONFIG_SHA256,
        "training_freeze_sha256": SEED599_FREEZE_SHA256,
        "evaluation_contract_sha256": SEED599_EVALUATION_SHA256,
        "evaluation_openings_sha256": SEED599_OPENINGS_SHA256,
    }


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
    data = root / DATA
    identities = _verify_published_inputs(root)
    prior = root / SEED599
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
        "authenticated_seed599_inputs": identities,
        "authorization_artifacts": authorization_artifact_status(root),
    }


def authorization_artifact_status(root: Path = ROOT) -> dict[str, bool]:
    """Report actual freeze/authorization file presence and validate each gate."""
    root = root.resolve()
    data = root / DATA
    return {
        "training_freeze_present": (data / "training-execution-freeze.json").is_file(),
        "training_authorization_present": (
            data / "training-authorization.json"
        ).is_file(),
        "training_authorized": training_authorized(root),
        "evaluation_freeze_present": (
            data / "evaluation-execution-freeze.json"
        ).is_file(),
        "evaluation_authorization_present": (
            data / "evaluation-authorization.json"
        ).is_file(),
        "evaluation_authorized": evaluation_authorized(root),
    }


def training_authorized(root: Path = ROOT) -> bool:
    """Distinct owner gate; no authorization record is prepared by this change."""
    root = root.resolve()
    freeze_path = root / DATA / "training-execution-freeze.json"
    auth_path = root / DATA / "training-authorization.json"
    if not freeze_path.is_file() or not auth_path.is_file():
        return False
    try:
        freeze_raw = freeze_path.read_bytes()
        freeze = json.loads(freeze_raw)
        auth = json.loads(auth_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(freeze, dict) or not isinstance(auth, dict):
        return False
    owner = auth.get("authorization", {})
    if not isinstance(owner, dict):
        return False
    return (
        freeze.get("schema") == "seed604-training-execution-freeze-v1"
        and freeze.get("status") == "frozen-awaiting-owner-authorization"
        and freeze.get("execution_authorized") is False
        and auth.get("schema") == "seed604-training-owner-authorization-v1"
        and auth.get("status") == "authorized-for-training"
        and auth.get("execution_freeze_sha256")
        == hashlib.sha256(freeze_raw).hexdigest()
        and owner.get("type") == "explicit-project-owner-authorization"
        and owner.get("authorized_by") == "project owner"
        and owner.get("training_may_start") is True
    )


def evaluation_authorized(root: Path = ROOT) -> bool:
    """Separate evaluation owner gate, bound to its own freeze bytes."""
    root = root.resolve()
    freeze_path = root / DATA / "evaluation-execution-freeze.json"
    auth_path = root / DATA / "evaluation-authorization.json"
    if not freeze_path.is_file() or not auth_path.is_file():
        return False
    try:
        freeze_raw = freeze_path.read_bytes()
        freeze = json.loads(freeze_raw)
        auth = json.loads(auth_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(freeze, dict) or not isinstance(auth, dict):
        return False
    owner = auth.get("authorization", {})
    if not isinstance(owner, dict):
        return False
    return (
        freeze.get("schema") == "seed604-evaluation-execution-freeze-v1"
        and freeze.get("status") == "frozen-awaiting-owner-authorization"
        and freeze.get("execution_authorized") is False
        and auth.get("schema") == "seed604-evaluation-owner-authorization-v1"
        and auth.get("status") == "authorized-for-evaluation"
        and auth.get("execution_freeze_sha256")
        == hashlib.sha256(freeze_raw).hexdigest()
        and owner.get("type") == "explicit-project-owner-authorization"
        and owner.get("authorized_by") == "project owner"
        and owner.get("run_games") is True
    )
