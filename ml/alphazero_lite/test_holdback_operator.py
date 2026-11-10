from __future__ import annotations

import gzip
import hashlib
import shutil

import pytest

from ml.alphazero_lite import holdback_arena as protocol
from ml.alphazero_lite import holdback_operator as operator


def make_inventory(root):
    index_bindings = {}
    for key, filename in (
        ("canonical", "canonical-index.jsonl.gz"),
        ("kalah_v3_float32", "kalah-v3-float32-index.jsonl.gz"),
    ):
        contents = b""
        compressed = gzip.compress(contents, mtime=0)
        (root / filename).write_bytes(compressed)
        index_bindings[key] = {
            "path": filename,
            "sha256_jsonl": hashlib.sha256(contents).hexdigest(),
            "occurrence_rows": 0,
            "distinct_identities": 0,
        }
    manifest = {
        "schema": protocol.INVENTORY_MANIFEST_SCHEMA,
        "identity_schemes": ["canonical_state_sha256", "kalah_v3_float32_sha256"],
        "indexes": index_bindings,
    }
    raw = protocol.canonical_bytes(manifest)
    path = root / "source-manifest.json"
    path.write_bytes(raw)
    return {
        "schema": protocol.INVENTORY_REFERENCE_SCHEMA,
        "manifest_path": "source-manifest.json",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
    }


@pytest.mark.parametrize("point", ["after_seed_saved", "after_suite_before_commitment"])
def test_interrupted_resume_preserves_seed_order_and_public_secrecy(tmp_path, point):
    repo = tmp_path / "repo"
    repo.mkdir()
    inventory_root = repo / "inventory"
    inventory_root.mkdir()
    reference = make_inventory(inventory_root)
    reference["manifest_path"] = "inventory/source-manifest.json"
    custody = tmp_path / "restricted-custody"
    seed = 1234567
    operator.initialize_private_inputs(
        custody, size=3, inventory_reference=reference, root=repo, seed=seed
    )
    public_path = tmp_path / "public" / "commitment.json"
    stopped = False

    def interrupt(name):
        nonlocal stopped
        if name == point and not stopped:
            stopped = True
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        operator.resume_seal(custody, repo, public_path, interruption=interrupt)
    saved_suite = custody / "suite.jsonl"
    before = saved_suite.read_bytes() if saved_suite.exists() else None
    result = operator.resume_seal(custody, repo, public_path)
    suite = saved_suite.read_bytes()
    assert before is None or before == suite
    assert hashlib.sha256(suite).hexdigest() == result["suite_sha256"]
    public = public_path.read_bytes()
    assert str(seed).encode() not in public
    assert suite not in public
    assert b"prefix_moves" not in public
    with pytest.raises(FileExistsError, match="already_exists"):
        operator.resume_seal(custody, repo, public_path)


def test_incomplete_private_seed_state_fails_closed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    custody = tmp_path / "restricted-custody"
    custody.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="private_inputs_missing_refuse_reseed"):
        operator.resume_seal(custody, repo, tmp_path / "public.json")


def test_freeze_gates_reveal_and_consumption_blocks_reuse(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    inventory_root = repo / "inventory"
    inventory_root.mkdir()
    reference = make_inventory(inventory_root)
    reference["manifest_path"] = "inventory/source-manifest.json"
    custody = tmp_path / "restricted-custody"
    operator.initialize_private_inputs(
        custody, size=2, inventory_reference=reference, root=repo, seed=712
    )
    public_path = tmp_path / "public.json"
    operator.resume_seal(custody, repo, public_path)
    bundle = tmp_path / "revealed"
    with pytest.raises(FileNotFoundError):
        operator.reveal_seal(custody, bundle)
    freeze = {
        "candidate_artifact_sha256": "a" * 64,
        "serving_p_artifact_sha256": "b" * 64,
        "arena_settings": {
            "pairs": 2,
            "games": 4,
            "pair_workers": 18,
            "search_workers": 1,
        },
        "analysis_rule": "synthetic paired mean",
    }
    operator.freeze_comparison(custody, freeze)
    trusted = operator.reveal_seal(custody, bundle)
    result = protocol.verify(
        bundle,
        trusted["anchor_sha256"],
        trusted["registry_sha256"],
        trusted_inventory_manifest_sha256=reference["manifest_sha256"],
    )
    assert result["held_back_eligible"]

    copied = tmp_path / "physical-copy"
    shutil.copytree(bundle, copied)
    before = {
        item.relative_to(copied): (item.read_bytes(), item.stat().st_mtime_ns)
        for item in copied.rglob("*")
        if item.is_file()
    }
    copied_result = protocol.verify(
        copied,
        trusted["anchor_sha256"],
        trusted["registry_sha256"],
        trusted_inventory_manifest_sha256=reference["manifest_sha256"],
    )
    after = {
        item.relative_to(copied): (item.read_bytes(), item.stat().st_mtime_ns)
        for item in copied.rglob("*")
        if item.is_file()
    }
    assert copied_result == result
    assert before == after

    consumed = operator.consume_seal(bundle)
    status = protocol.verify(
        bundle,
        consumed["anchor_sha256"],
        consumed["registry_sha256"],
        trusted_inventory_manifest_sha256=reference["manifest_sha256"],
    )
    assert status["consumed"] and not status["held_back_eligible"]
    with pytest.raises(ValueError, match="already_consumed"):
        operator.consume_seal(bundle)
