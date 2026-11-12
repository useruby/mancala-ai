"""Synthetic checks for immutable launch bindings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.alphazero_lite import varied_opponent_pilot as pilot
from ml.alphazero_lite import run_varied_opponent_pilot as runner

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "docs/data/varied-opponent-coverage-pilot/freeze.json"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_digest_recomputed_after_assignment_tamper_still_rejected(tmp_path):
    original = json.loads(FREEZE.read_text())
    published_digest = original["freeze_sha256"]
    altered = json.loads(json.dumps(original))
    altered["assignments"][0]["seed"] += 1
    altered["assignments"][1]["seed"] += 1
    altered.pop("freeze_sha256")
    altered["freeze_sha256"] = hashlib.sha256(_canonical(altered)).hexdigest()
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="pilot_freeze_digest_mismatch"):
        runner.load_freeze(path, expected_digest=published_digest)


def test_execution_source_hashes_are_checked_against_live_tree():
    freeze = json.loads(FREEZE.read_text())
    runner.verify_execution_sources(freeze, ROOT)
    freeze["execution_source_hashes"][
        "ml/alphazero_lite/run_varied_opponent_pilot.py"
    ] = "0" * 64
    with pytest.raises(ValueError, match="pilot_execution_source_hash_mismatch"):
        runner.verify_execution_sources(freeze, ROOT)


def test_live_runtime_scope_matches_registered_limits(tmp_path):
    cgroup_root = tmp_path / "sys/fs/cgroup"
    cgroup = cgroup_root / "pilot"
    cgroup.mkdir(parents=True)
    (cgroup / "cpu.max").write_text("1800000 100000")
    (cgroup / "cpu.weight").write_text("100")
    (cgroup / "memory.max").write_text("38654705664")
    (cgroup / "cpuset.cpus.effective").write_text("14-31")
    proc_cgroup = tmp_path / "proc/cgroup"
    proc_cgroup.parent.mkdir()
    proc_cgroup.write_text("0::/pilot\n")
    frozen = {
        "cpu_quota": "1800%",
        "cpu_weight": "100",
        "memory_max_bytes": 38654705664,
        "allowed_cpus": "14-31",
        "nice": 19,
    }
    observed = pilot.verify_live_resource_limits(
        frozen,
        cgroup_root=cgroup_root,
        proc_cgroup=proc_cgroup,
        affinity=set(range(14, 32)),
        nice=19,
    )
    assert observed["cpu_weight"] == 100
    with pytest.raises(ValueError, match="pilot_live_cpu_affinity_mismatch"):
        pilot.verify_live_resource_limits(
            frozen,
            cgroup_root=cgroup_root,
            proc_cgroup=proc_cgroup,
            affinity={14},
            nice=19,
        )


def test_runner_rejects_frozen_resource_drift_before_live_check(monkeypatch):
    freeze = json.loads(FREEZE.read_text())
    monkeypatch.setattr(
        pilot,
        "verify_live_resource_limits",
        lambda _limits: pytest.fail(
            "live scope should not be checked after freeze drift"
        ),
    )
    freeze["runtime_limits"]["pair_concurrency"] = 17
    with pytest.raises(ValueError, match="pilot_resource_freeze_mismatch"):
        runner.verify_resource_bindings(freeze)
