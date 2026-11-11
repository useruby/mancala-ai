from __future__ import annotations

import gzip
import hashlib
import json
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
        custody,
        size=3,
        inventory_reference=reference,
        root=repo,
        seed=seed,
        synthetic=True,
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
    assert operator.resume_seal(custody, repo, public_path) == result


def test_incomplete_private_seed_state_fails_closed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    custody = tmp_path / "restricted-custody"
    custody.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="private_inputs_missing_refuse_reseed"):
        operator.resume_seal(custody, repo, tmp_path / "public.json")


def test_production_seed_requires_preflight_before_rng(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    reference = {
        "schema": protocol.INVENTORY_REFERENCE_SCHEMA,
        "manifest_path": "missing/source-manifest.json",
        "manifest_sha256": protocol.TRUSTED_HISTORICAL_INVENTORY_MANIFEST_SHA256,
    }
    rng_called = False

    def forbidden_rng(_bits):
        nonlocal rng_called
        rng_called = True
        raise AssertionError("seed draw before preflight")

    monkeypatch.setattr(operator.secrets, "randbits", forbidden_rng)
    with pytest.raises(ValueError, match="verified_preflight_required"):
        operator.initialize_private_inputs(
            tmp_path / "custody", size=1024, inventory_reference=reference, root=repo
        )
    assert not rng_called
    with pytest.raises(ValueError, match="production_seed_override_forbidden"):
        operator.initialize_private_inputs(
            tmp_path / "other-custody",
            size=1024,
            inventory_reference=reference,
            root=repo,
            seed=33,
        )


def test_matching_production_preflight_is_persisted_before_seed_draw(
    monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    repo.mkdir()
    reference = {
        "schema": protocol.INVENTORY_REFERENCE_SCHEMA,
        "manifest_path": "source-manifest.json",
        "manifest_sha256": protocol.TRUSTED_HISTORICAL_INVENTORY_MANIFEST_SHA256,
    }
    record = {
        "preflight_sha256": "f" * 64,
        "inventory_manifest_sha256": reference["manifest_sha256"],
        "generator_binding": protocol.rules_generator_spec(1024),
    }
    calls = []
    monkeypatch.setattr(
        operator, "preflight", lambda root: calls.append("preflight") or record
    )
    monkeypatch.setattr(
        operator.secrets, "randbits", lambda bits: calls.append("rng") or 99
    )
    custody = tmp_path / "custody"
    assert (
        operator.initialize_private_inputs(
            custody,
            size=1024,
            inventory_reference=reference,
            root=repo,
            preflight_record=record,
        )
        == 99
    )
    saved = json.loads((custody / "private-inputs.json").read_bytes())
    assert calls == ["preflight", "rng"]
    assert saved["preflight"] == record


def test_freeze_gates_reveal_and_consumption_blocks_reuse(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    inventory_root = repo / "inventory"
    inventory_root.mkdir()
    reference = make_inventory(inventory_root)
    reference["manifest_path"] = "inventory/source-manifest.json"
    custody = tmp_path / "restricted-custody"
    operator.initialize_private_inputs(
        custody,
        size=2,
        inventory_reference=reference,
        root=repo,
        seed=712,
        synthetic=True,
    )
    public_path = tmp_path / "public.json"
    sealed = operator.resume_seal(custody, repo, public_path)
    bundle = tmp_path / "revealed"
    with pytest.raises(
        ValueError, match="published_commitment_and_independent_freeze_required"
    ):
        operator.reveal_seal(
            custody,
            bundle,
            root=repo,
            trusted_commitment_sha256=sealed["commitment"],
            trusted_freeze_checkpoint_sha256="0" * 64,
        )
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
    frozen = operator.freeze_comparison(custody, freeze, root=repo)
    with pytest.raises(RuntimeError, match="simulated reveal interruption"):
        operator.reveal_seal(
            custody,
            bundle,
            root=repo,
            trusted_commitment_sha256=sealed["commitment"],
            trusted_freeze_checkpoint_sha256=frozen["freeze_checkpoint_sha256"],
            interruption=lambda stage: (
                (_ for _ in ()).throw(RuntimeError("simulated reveal interruption"))
                if stage == "after_reveal_event_before_bundle_publish"
                else None
            ),
        )
    assert not bundle.exists()
    reveals = [
        json.loads(line)
        for line in (custody / "custodian-lifecycle.jsonl").read_bytes().splitlines()
    ]
    assert [event["kind"] for event in reveals].count("reveal") == 1
    trusted = operator.reveal_seal(
        custody,
        bundle,
        root=repo,
        trusted_commitment_sha256=sealed["commitment"],
        trusted_freeze_checkpoint_sha256=frozen["freeze_checkpoint_sha256"],
    )
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
    lifecycle = [
        json.loads(line)
        for line in (bundle / "lifecycle.jsonl").read_bytes().splitlines()
    ]
    anchor = json.loads((bundle / "anchor.json").read_bytes())
    assert [event["kind"] for event in lifecycle] == [
        "commitment",
        "candidate_freeze",
        "reveal",
    ]
    times = [item["anchored_at_utc"] for item in anchor["checkpoints"]]
    assert times == sorted(times) and len(set(times)) == 3

    def stop_after_log(stage):
        if stage == "after_lifecycle_write":
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        operator.consume_seal(bundle, interruption=stop_after_log)
    assert not json.loads((bundle / "custodian-state.json").read_bytes())["consumed"]
    consumed = operator.consume_seal(bundle)
    assert consumed["recovered"] == "true"
    status = protocol.verify(
        bundle,
        consumed["anchor_sha256"],
        consumed["registry_sha256"],
        trusted_inventory_manifest_sha256=reference["manifest_sha256"],
    )
    assert status["consumed"] and not status["held_back_eligible"]
    with pytest.raises(ValueError, match="already_consumed"):
        operator.consume_seal(bundle)


def test_rejected_reveal_leaves_no_suite_at_destination(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    inventory_root = repo / "inventory"
    inventory_root.mkdir()
    reference = make_inventory(inventory_root)
    reference["manifest_path"] = "inventory/source-manifest.json"
    custody = tmp_path / "custody"
    operator.initialize_private_inputs(
        custody,
        size=1,
        inventory_reference=reference,
        root=repo,
        seed=77,
        synthetic=True,
    )
    public_path = tmp_path / "public.json"
    sealed = operator.resume_seal(custody, repo, public_path)
    frozen = operator.freeze_comparison(
        custody,
        {
            "candidate_artifact_sha256": "a" * 64,
            "serving_p_artifact_sha256": "b" * 64,
            "arena_settings": {"pairs": 1},
            "analysis_rule": "synthetic",
        },
        root=repo,
    )
    freeze_path = custody / "comparison-freeze.json"
    freeze = json.loads(freeze_path.read_bytes())
    freeze["commitment"] = "0" * 64
    freeze_path.write_bytes(operator._json_bytes(freeze))
    destination = tmp_path / "rejected-reveal"
    with pytest.raises(ValueError, match="freeze_commitment_mismatch"):
        operator.reveal_seal(
            custody,
            destination,
            root=repo,
            trusted_commitment_sha256=sealed["commitment"],
            trusted_freeze_checkpoint_sha256=frozen["freeze_checkpoint_sha256"],
        )
    assert not destination.exists()
    assert not (destination / "suite.jsonl").exists()


def test_rewritten_local_freeze_history_rejected_by_external_checkpoint_pin(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    inventory_root = repo / "inventory"
    inventory_root.mkdir()
    reference = make_inventory(inventory_root)
    reference["manifest_path"] = "inventory/source-manifest.json"
    custody = tmp_path / "custody"
    operator.initialize_private_inputs(
        custody,
        size=1,
        inventory_reference=reference,
        root=repo,
        seed=213,
        synthetic=True,
    )
    public_path = tmp_path / "commitment.json"
    sealed = operator.resume_seal(custody, repo, public_path)
    frozen = operator.freeze_comparison(
        custody,
        {
            "candidate_artifact_sha256": "a" * 64,
            "serving_p_artifact_sha256": "b" * 64,
            "arena_settings": {"pairs": 1},
            "analysis_rule": "original synthetic rule",
        },
        root=repo,
    )

    freeze_path = custody / "comparison-freeze.json"
    freeze_record = json.loads(freeze_path.read_bytes())
    freeze_record["freeze"]["analysis_rule"] = "rewritten local rule"
    freeze_path.write_bytes(operator._json_bytes(freeze_record))
    original_events = [
        json.loads(line)
        for line in (custody / "custodian-lifecycle.jsonl").read_bytes().splitlines()
    ]
    rewritten = []
    previous = "0" * 64
    for index, event in enumerate(original_events):
        payload = event["payload"]
        if index == 1:
            payload = freeze_record
        replacement = protocol.lifecycle_event(event["kind"], payload, previous)
        rewritten.append(replacement)
        previous = replacement["event_sha256"]
    event_bytes = protocol.event_log_bytes(rewritten)
    (custody / "custodian-lifecycle.jsonl").write_bytes(event_bytes)
    anchor = json.loads((custody / "custodian-anchor.json").read_bytes())
    anchor["lifecycle_sha256"] = hashlib.sha256(event_bytes).hexdigest()
    for checkpoint, event in zip(anchor["checkpoints"], rewritten, strict=True):
        checkpoint["event_sha256"] = event["event_sha256"]
    anchor_bytes = operator._json_bytes(anchor)
    (custody / "custodian-anchor.json").write_bytes(anchor_bytes)
    (custody / "anchor-candidate_freeze.sha256").write_text(
        hashlib.sha256(anchor_bytes).hexdigest() + "\n"
    )

    destination = tmp_path / "tampered-history-reveal"
    with pytest.raises(ValueError, match="trusted_freeze_checkpoint_mismatch"):
        operator.reveal_seal(
            custody,
            destination,
            root=repo,
            trusted_commitment_sha256=sealed["commitment"],
            trusted_freeze_checkpoint_sha256=frozen["freeze_checkpoint_sha256"],
        )
    assert not destination.exists()


def test_local_freeze_and_private_suite_without_published_commitment_rejected(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    inventory_root = repo / "inventory"
    inventory_root.mkdir()
    reference = make_inventory(inventory_root)
    reference["manifest_path"] = "inventory/source-manifest.json"
    custody = tmp_path / "custody"
    operator.initialize_private_inputs(
        custody,
        size=1,
        inventory_reference=reference,
        root=repo,
        seed=81,
        synthetic=True,
    )
    public_path = tmp_path / "commitment.json"
    sealed = operator.resume_seal(custody, repo, public_path)
    frozen = operator.freeze_comparison(
        custody,
        {
            "candidate_artifact_sha256": "a" * 64,
            "serving_p_artifact_sha256": "b" * 64,
            "arena_settings": {"pairs": 1},
            "analysis_rule": "synthetic",
        },
        root=repo,
    )
    public_path.unlink()
    (custody / "public-commitment-publication.json").unlink()
    (custody / "custodian-lifecycle.jsonl").unlink()
    (custody / "custodian-anchor.json").unlink()
    (custody / "anchor-commitment.sha256").unlink()
    (custody / "anchor-candidate_freeze.sha256").unlink()
    destination = tmp_path / "unauthorized-reveal"
    with pytest.raises(
        ValueError, match="published_commitment_and_independent_freeze_required"
    ):
        operator.reveal_seal(
            custody,
            destination,
            root=repo,
            trusted_commitment_sha256=sealed["commitment"],
            trusted_freeze_checkpoint_sha256=frozen["freeze_checkpoint_sha256"],
        )
    assert not destination.exists()
