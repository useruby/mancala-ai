"""Custodian-side, restart-safe private sealing operations for holdback suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import datetime as dt
import base64
import ctypes
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from ml.alphazero_lite import holdback_arena as protocol
from ml.alphazero_lite import seed561_historical_exclusion_inventory as inventory

PRIVATE_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_new(path: Path, data: bytes, mode: int = PRIVATE_FILE_MODE) -> None:
    """Create a file atomically and refuse every form of overwrite."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _json_bytes(value: Any) -> bytes:
    return protocol.canonical_bytes(value) + b"\n"


def preflight(root: Path) -> dict[str, Any]:
    """Authenticate #565 and generator bindings without seed or sampling."""
    root = root.resolve()
    report = inventory.verify(root, root / inventory.OUT_REL)
    reference = {
        "schema": protocol.INVENTORY_REFERENCE_SCHEMA,
        "manifest_path": str(inventory.OUT_REL / "source-manifest.json"),
        "manifest_sha256": protocol.TRUSTED_HISTORICAL_INVENTORY_MANIFEST_SHA256,
    }
    identities = protocol.load_pinned_identity_indexes(
        root, reference, protocol.TRUSTED_HISTORICAL_INVENTORY_MANIFEST_SHA256
    )
    generator = protocol.rules_generator_spec(1024)
    record = {
        "status": "preflight_passed",
        "seed_drawn": False,
        "openings_sampled": False,
        "inventory_report": report,
        "inventory_manifest_sha256": reference["manifest_sha256"],
        "identity_counts": {key: len(value) for key, value in identities.items()},
        "production": {
            "openings": 1024,
            "attempt_limit": protocol.RULES_ATTEMPT_LIMIT,
            "output_format": "compact UTF-8 sorted-key JSONL, one LF per row",
            "custody": "seed and ordered suite in mode-0700 external directory; public commitment only after private verification",
            "pair_workers": 18,
            "workers_per_game": 1,
        },
        "generator_binding": generator,
    }
    record["preflight_sha256"] = _digest(protocol.canonical_bytes(record))
    return record


def _write_atomic(path: Path, data: bytes, mode: int = PRIVATE_FILE_MODE) -> None:
    """Replace one state file atomically using a sibling temporary file."""
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any existing path."""
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OSError("atomic_noreplace_rename_unavailable")
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _timestamp_after(previous: str | None = None) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    if previous is not None:
        prior = dt.datetime.fromisoformat(previous)
        if now <= prior:
            now = prior + dt.timedelta(microseconds=1)
    return now.isoformat()


def _record_private_event(custody: Path, kind: str, payload: dict[str, Any]) -> str:
    """Append and checkpoint a lifecycle event at the time the action occurs."""
    _recover_private_event(custody)
    log_path = custody / "custodian-lifecycle.jsonl"
    events = (
        [json.loads(line) for line in log_path.read_bytes().splitlines() if line]
        if log_path.exists()
        else []
    )
    required = ["commitment"] if not events else ["commitment", "candidate_freeze"]
    if [event["kind"] for event in events] != required[: len(events)]:
        raise ValueError("custodian_lifecycle_order")
    if len(events) == 1 and kind != "candidate_freeze":
        raise ValueError("candidate_freeze_must_follow_commitment")
    if not events and kind != "commitment":
        raise ValueError("commitment_must_be_first_lifecycle_event")
    previous = events[-1]["event_sha256"] if events else "0" * 64
    event = protocol.lifecycle_event(kind, payload, previous)
    events.append(event)
    log_bytes = protocol.event_log_bytes(events)
    prior_anchor = (
        json.loads((custody / "custodian-anchor.json").read_bytes())
        if (custody / "custodian-anchor.json").exists()
        else None
    )
    prior_time = (
        prior_anchor["checkpoints"][-1]["anchored_at_utc"] if prior_anchor else None
    )
    anchor = {
        "lifecycle_path": "custodian-lifecycle.jsonl",
        "lifecycle_sha256": _digest(log_bytes),
        "checkpoints": [
            *([] if prior_anchor is None else prior_anchor["checkpoints"]),
            {
                "event_sha256": event["event_sha256"],
                "anchored_at_utc": _timestamp_after(prior_time),
            },
        ],
    }
    anchor_bytes = _json_bytes(anchor)
    transaction = {
        "kind": kind,
        "lifecycle": base64.b64encode(log_bytes).decode("ascii"),
        "anchor": base64.b64encode(anchor_bytes).decode("ascii"),
    }
    _write_atomic(custody / "event-pending.json", _json_bytes(transaction))
    _recover_private_event(custody)
    return event["event_sha256"]


def _recover_private_event(custody: Path) -> None:
    pending_path = custody / "event-pending.json"
    if not pending_path.exists():
        return
    transaction = json.loads(pending_path.read_bytes())
    log_bytes = base64.b64decode(transaction["lifecycle"], validate=True)
    anchor_bytes = base64.b64decode(transaction["anchor"], validate=True)
    kind = transaction["kind"]
    _write_atomic(custody / "custodian-lifecycle.jsonl", log_bytes)
    _write_atomic(custody / "custodian-anchor.json", anchor_bytes)
    anchor_digest_path = custody / f"anchor-{kind}.sha256"
    digest = (_digest(anchor_bytes) + "\n").encode()
    if anchor_digest_path.exists():
        if anchor_digest_path.read_bytes() != digest:
            raise ValueError("private_anchor_digest_conflict")
    else:
        _write_new(anchor_digest_path, digest)
    pending_path.unlink()


def initialize_private_inputs(
    custody: Path,
    *,
    size: int,
    inventory_reference: dict[str, Any],
    root: Path = Path.cwd(),
    seed: int | None = None,
    preflight_record: dict[str, Any] | None = None,
    synthetic: bool = False,
) -> int:
    """Persist one seed and all generation inputs before any suite sampling."""
    custody = custody.expanduser().absolute()
    root_path = root.resolve()
    try:
        custody.resolve().relative_to(root_path)
    except ValueError:
        pass
    else:
        raise ValueError("custody_must_be_outside_repository")
    custody.mkdir(mode=PRIVATE_MODE, parents=True, exist_ok=False)
    os.chmod(custody, PRIVATE_MODE)
    if custody.stat().st_mode & 0o077:
        raise PermissionError("custody_directory_not_private")
    if synthetic:
        if seed is None:
            raise ValueError("synthetic_seed_required")
        verified_preflight = {"schema": "synthetic_preflight_fixture_v1"}
    else:
        if seed is not None:
            raise ValueError("production_seed_override_forbidden")
        if size != 1024:
            raise ValueError("production_suite_size_must_be_1024")
        if (
            inventory_reference.get("manifest_sha256")
            != protocol.TRUSTED_HISTORICAL_INVENTORY_MANIFEST_SHA256
        ):
            raise ValueError("production_inventory_pin_mismatch")
        if preflight_record is None:
            raise ValueError("verified_preflight_required_before_seed_draw")
        verified_preflight = preflight(root)
        if preflight_record != verified_preflight:
            raise ValueError("preflight_record_stale_or_tampered")
        if (
            inventory_reference.get("manifest_sha256")
            != verified_preflight["inventory_manifest_sha256"]
        ):
            raise ValueError("preflight_inventory_binding_mismatch")
    generator = protocol.rules_generator_spec(size)
    if not synthetic and generator != verified_preflight["generator_binding"]:
        raise ValueError("preflight_generator_binding_changed_before_seed")
    # In production, no operation capable of drawing entropy occurs until the
    # preflight has been recomputed and matched against its retained record.
    actual_seed = secrets.randbits(256) if seed is None else seed
    if isinstance(actual_seed, bool) or not isinstance(actual_seed, int):
        raise ValueError("seed_invalid")
    inputs = {
        "schema": "holdback_private_inputs_v1",
        "seed": actual_seed,
        "generator": generator,
        "inventory_reference": inventory_reference,
        "preflight": verified_preflight,
        "synthetic_fixture": synthetic,
    }
    _write_new(custody / "private-inputs.json", _json_bytes(inputs))
    return actual_seed


def _verify_runtime_binding(root: Path, inputs: dict[str, Any]) -> None:
    if inputs.get("synthetic_fixture"):
        return
    current = preflight(root)
    if inputs.get("preflight") != current:
        raise ValueError("preflight_runtime_or_inventory_binding_changed")


def _load_inputs(custody: Path) -> dict[str, Any]:
    path = custody / "private-inputs.json"
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        raise ValueError("private_inputs_permissions_or_type_invalid")
    data = json.loads(path.read_bytes())
    if data.get("schema") != "holdback_private_inputs_v1":
        raise ValueError("private_inputs_schema_invalid")
    seed = data.get("seed")
    generator = data.get("generator")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("private_seed_invalid")
    if not isinstance(generator, dict):
        raise ValueError("private_generator_invalid")
    size = generator.get("size")
    if protocol.public_generator_spec(generator) != protocol.rules_generator_spec(size):
        raise ValueError("private_generator_inputs_contradictory")
    return data


def _load_inventory(custody: Path, root: Path, inputs: dict[str, Any]):
    reference = inputs["inventory_reference"]
    # Manifest and indexes are copied into custody before generation, so a
    # later repository change cannot redirect a restart to different inputs.
    manifest_path = Path(reference["manifest_path"])
    source_manifest = (root / manifest_path).read_bytes()
    if _digest(source_manifest) != reference["manifest_sha256"]:
        raise ValueError("inventory_manifest_digest_mismatch")
    private_inventory = custody / "inventory"
    if not private_inventory.exists():
        private_inventory.mkdir(mode=PRIVATE_MODE)
        source_dir = (root / manifest_path).parent
        for name in (
            "source-manifest.json",
            "canonical-index.jsonl.gz",
            "kalah-v3-float32-index.jsonl.gz",
        ):
            _write_new(private_inventory / name, (source_dir / name).read_bytes())
    private_reference = dict(reference)
    private_reference["manifest_path"] = "inventory/source-manifest.json"
    return protocol.load_pinned_identity_indexes(
        custody, private_reference, reference["manifest_sha256"]
    ), private_reference


def resume_seal(
    custody: Path,
    root: Path,
    public_output: Path,
    *,
    interruption: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Resume only persisted inputs; verify private bytes before public publish."""
    custody = custody.expanduser().absolute()
    public_output = public_output.expanduser().absolute()
    if public_output.exists():
        existing_public = public_output.read_bytes()
    else:
        existing_public = None
    inputs_path = custody / "private-inputs.json"
    if not inputs_path.exists():
        raise ValueError("private_inputs_missing_refuse_reseed")
    inputs = _load_inputs(custody)
    _verify_runtime_binding(root, inputs)
    if interruption:
        interruption("after_seed_saved")
    inventory, private_reference = _load_inventory(custody, root, inputs)
    suite_path = custody / "suite.jsonl"
    if suite_path.exists():
        if not stat.S_ISREG(suite_path.lstat().st_mode):
            raise ValueError("private_suite_type_invalid")
        suite_raw = suite_path.read_bytes()
        rows = [json.loads(line) for line in suite_raw.splitlines() if line]
        if protocol.suite_bytes(rows) != suite_raw:
            raise ValueError("private_suite_encoding_invalid")
        rebuilt = protocol.build_rules_only_suite(
            inventory,
            inputs["seed"],
            inputs["generator"]["size"],
            attempt_limit=inputs["generator"]["attempt_limit"],
        )
        if protocol.suite_bytes(rebuilt) != suite_raw:
            raise ValueError("private_suite_contradicts_saved_inputs")
    else:
        rows = protocol.build_rules_only_suite(
            inventory,
            inputs["seed"],
            inputs["generator"]["size"],
            attempt_limit=inputs["generator"]["attempt_limit"],
        )
        suite_raw = protocol.suite_bytes(rows)
        _write_new(suite_path, suite_raw)
    receipt = {
        "schema": "holdback_private_generation_receipt_v1",
        "seed_sha256": _digest(str(inputs["seed"]).encode()),
        "suite_sha256": _digest(suite_raw),
        "size": len(rows),
        "inventory_manifest_sha256": private_reference["manifest_sha256"],
    }
    receipt_path = custody / "generation-receipt.json"
    if receipt_path.exists() and receipt_path.read_bytes() != _json_bytes(receipt):
        raise ValueError("private_receipt_contradicts_saved_bundle")
    if not receipt_path.exists():
        _write_new(receipt_path, _json_bytes(receipt))
    if interruption:
        interruption("after_suite_before_commitment")
    # Independently reconstruct and check exact order and both exclusion sets.
    if _digest(suite_raw) != receipt["suite_sha256"]:
        raise ValueError("private_suite_digest_mismatch")
    for row in rows:
        canonical, float32 = protocol.identities(row)
        if canonical in inventory["canonical"] or float32 in inventory["float32"]:
            raise ValueError("private_suite_exclusion_overlap")
    record = protocol.public_commitment_record(
        receipt["suite_sha256"],
        protocol.public_generator_spec(inputs["generator"]),
        private_reference,
    )
    public_bytes = _json_bytes(record)
    if existing_public is not None and existing_public != public_bytes:
        raise FileExistsError("public_commitment_conflicts_with_private_seal")
    # No public path is created until every private artifact has been checked.
    public_output.parent.mkdir(parents=True, exist_ok=True)
    if existing_public is None:
        _write_new(public_output, public_bytes, 0o644)
    publication = {
        "path": str(public_output),
        "sha256": _digest(public_bytes),
        "commitment": record["commitment"],
    }
    publication_path = custody / "public-commitment-publication.json"
    if publication_path.exists():
        if publication_path.read_bytes() != _json_bytes(publication):
            raise ValueError("public_commitment_publication_receipt_mismatch")
    else:
        _write_new(publication_path, _json_bytes(publication))
    _recover_private_event(custody)
    lifecycle = custody / "custodian-lifecycle.jsonl"
    if not lifecycle.exists():
        _record_private_event(
            custody,
            "commitment",
            {
                "commitment": record["commitment"],
                "suite_sha256": receipt["suite_sha256"],
            },
        )
    else:
        events = [
            json.loads(line) for line in lifecycle.read_bytes().splitlines() if line
        ]
        if not events or events[0]["payload"].get("commitment") != record["commitment"]:
            raise ValueError("custodian_commitment_event_mismatch")
    return {"commitment": record["commitment"], "suite_sha256": receipt["suite_sha256"]}


def freeze_comparison(
    custody: Path, freeze: dict[str, Any], root: Path = Path.cwd()
) -> dict[str, str]:
    """Persist candidate/serving and comparison bindings before reveal."""
    required = (
        "candidate_artifact_sha256",
        "serving_p_artifact_sha256",
        "arena_settings",
        "analysis_rule",
    )
    if any(not freeze.get(key) for key in required):
        raise ValueError("candidate_freeze_not_bound")
    for key in required[:2]:
        value = freeze[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("candidate_freeze_digest_invalid")
    inputs = _load_inputs(custody)
    _verify_runtime_binding(root, inputs)
    publication_path = custody / "public-commitment-publication.json"
    publication = json.loads(publication_path.read_bytes())
    published = Path(publication["path"])
    if (
        not published.is_file()
        or _digest(published.read_bytes()) != publication["sha256"]
    ):
        raise ValueError("public_commitment_not_published_or_changed")
    suite = (custody / "suite.jsonl").read_bytes()
    receipt = json.loads((custody / "generation-receipt.json").read_bytes())
    if _digest(suite) != receipt.get("suite_sha256"):
        raise ValueError("private_suite_digest_mismatch")
    inventory_reference = dict(
        inputs["inventory_reference"],
        manifest_path="inventory/source-manifest.json",
    )
    commitment = protocol.public_commitment_record(
        receipt["suite_sha256"],
        protocol.public_generator_spec(inputs["generator"]),
        inventory_reference,
    )["commitment"]
    freeze_document = {"commitment": commitment, "freeze": freeze}
    freeze_path = custody / "comparison-freeze.json"
    freeze_bytes = _json_bytes(freeze_document)
    if freeze_path.exists():
        if freeze_path.read_bytes() != freeze_bytes:
            raise ValueError("candidate_freeze_conflicts_with_saved_freeze")
    else:
        _write_new(freeze_path, freeze_bytes)
    _recover_private_event(custody)
    events_path = custody / "custodian-lifecycle.jsonl"
    events = [
        json.loads(line) for line in events_path.read_bytes().splitlines() if line
    ]
    if len(events) == 1:
        checkpoint_digest = _record_private_event(
            custody, "candidate_freeze", freeze_document
        )
    elif not (
        len(events) == 2
        and events[1]["kind"] == "candidate_freeze"
        and events[1]["payload"] == freeze_document
    ):
        raise ValueError("candidate_freeze_lifecycle_conflict")
    else:
        checkpoint_digest = events[1]["event_sha256"]
    anchor_bytes = (custody / "custodian-anchor.json").read_bytes()
    return {
        "commitment": commitment,
        "freeze_checkpoint_sha256": checkpoint_digest,
        "anchor_sha256": _digest(anchor_bytes),
    }


def reveal_seal(
    custody: Path,
    bundle: Path,
    root: Path = Path.cwd(),
    *,
    trusted_commitment_sha256: str,
    trusted_freeze_checkpoint_sha256: str,
    interruption: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Verify and stage a reveal, then atomically publish the complete bundle."""
    custody = custody.expanduser().absolute()
    bundle = bundle.expanduser().absolute()
    if bundle.exists():
        raise FileExistsError("reveal_bundle_already_exists")
    freeze_path = custody / "comparison-freeze.json"
    _recover_private_event(custody)
    publication_path = custody / "public-commitment-publication.json"
    if not freeze_path.is_file() or not publication_path.is_file():
        raise ValueError("published_commitment_and_independent_freeze_required")
    freeze_record = json.loads(freeze_path.read_bytes())
    publication = json.loads(publication_path.read_bytes())
    published = Path(publication["path"])
    if (
        not published.is_file()
        or _digest(published.read_bytes()) != publication["sha256"]
    ):
        raise ValueError("public_commitment_not_published_or_changed")
    public_record = json.loads(published.read_bytes())
    protocol.verify_public_commitment_record(public_record)
    if public_record.get("commitment") != trusted_commitment_sha256:
        raise ValueError("untrusted_public_commitment")

    inputs = _load_inputs(custody)
    _verify_runtime_binding(root, inputs)
    receipt = json.loads((custody / "generation-receipt.json").read_bytes())
    suite = (custody / "suite.jsonl").read_bytes()
    if _digest(suite) != receipt.get("suite_sha256"):
        raise ValueError("private_suite_digest_mismatch")
    rows = [json.loads(line) for line in suite.splitlines() if line]
    if protocol.suite_bytes(rows) != suite:
        raise ValueError("private_suite_encoding_invalid")
    private_reference = dict(
        inputs["inventory_reference"], manifest_path="inventory/source-manifest.json"
    )
    excluded = protocol.load_pinned_identity_indexes(
        custody,
        private_reference,
        private_reference["manifest_sha256"],
    )
    rebuilt = protocol.build_rules_only_suite(
        excluded,
        inputs["seed"],
        inputs["generator"]["size"],
        attempt_limit=inputs["generator"]["attempt_limit"],
    )
    if protocol.suite_bytes(rebuilt) != suite:
        raise ValueError("private_suite_contradicts_saved_inputs")
    expected_public = protocol.public_commitment_record(
        receipt["suite_sha256"],
        protocol.public_generator_spec(inputs["generator"]),
        private_reference,
    )
    commitment = expected_public["commitment"]
    if public_record != expected_public or publication.get("commitment") != commitment:
        raise ValueError("public_commitment_does_not_match_private_suite")
    if freeze_record.get("commitment") != commitment:
        raise ValueError("freeze_commitment_mismatch")
    events_path = custody / "custodian-lifecycle.jsonl"
    events = [
        json.loads(line) for line in events_path.read_bytes().splitlines() if line
    ]
    if [event["kind"] for event in events] not in (
        ["commitment", "candidate_freeze"],
        ["commitment", "candidate_freeze", "reveal"],
    ):
        raise ValueError("commitment_and_freeze_must_be_independently_recorded")
    anchor = json.loads((custody / "custodian-anchor.json").read_bytes())
    private_event_bytes = events_path.read_bytes()
    previous = "0" * 64
    for event in events:
        expected = protocol.lifecycle_event(event["kind"], event["payload"], previous)
        if event != expected:
            raise ValueError("custodian_lifecycle_chain_invalid")
        previous = event["event_sha256"]
    if anchor.get("lifecycle_sha256") != _digest(private_event_bytes) or [
        item.get("event_sha256") for item in anchor.get("checkpoints", [])
    ] != [event["event_sha256"] for event in events]:
        raise ValueError("custodian_lifecycle_anchor_invalid")
    if events[0]["payload"].get("commitment") != trusted_commitment_sha256:
        raise ValueError("trusted_commitment_checkpoint_mismatch")
    if events[1]["event_sha256"] != trusted_freeze_checkpoint_sha256:
        raise ValueError("trusted_freeze_checkpoint_mismatch")
    if (
        events[0]["payload"].get("commitment") != commitment
        or events[1]["payload"] != freeze_record
    ):
        raise ValueError("custodian_lifecycle_binding_mismatch")

    # Stage beside the destination so final rename is atomic. No destination
    # path, including suite.jsonl, appears until all bundle verification passes.
    bundle.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{bundle.name}.staging-", dir=bundle.parent))
    os.chmod(stage, PRIVATE_MODE)
    try:
        (stage / "inventory").mkdir(mode=PRIVATE_MODE)
        shutil.copy2(custody / "suite.jsonl", stage / "suite.jsonl")
        for name in (
            "source-manifest.json",
            "canonical-index.jsonl.gz",
            "kalah-v3-float32-index.jsonl.gz",
        ):
            shutil.copy2(custody / "inventory" / name, stage / "inventory" / name)
        inv_bytes = _json_bytes(private_reference)
        (stage / "inventory.json").write_bytes(inv_bytes)
        generator = {**inputs["generator"], "seed": inputs["seed"]}
        manifest = {
            "suite_sha256": receipt["suite_sha256"],
            "inventory_path": "inventory.json",
            "inventory_sha256": _digest(inv_bytes),
            "generator": generator,
            "commitment": commitment,
        }
        (stage / "manifest.json").write_bytes(_json_bytes(manifest))

        reveal_payload = {
            "commitment": commitment,
            "suite_sha256": receipt["suite_sha256"],
            "seed": inputs["seed"],
        }
        current_events = [
            json.loads(line) for line in events_path.read_bytes().splitlines() if line
        ]
        if len(current_events) == 2:
            _record_private_event(custody, "reveal", reveal_payload)
            if interruption:
                interruption("after_reveal_event_before_bundle_publish")
        elif not (
            len(current_events) == 3
            and current_events[2]["kind"] == "reveal"
            and current_events[2]["payload"] == reveal_payload
        ):
            raise ValueError("custodian_reveal_event_conflict")
        lifecycle_bytes = (custody / "custodian-lifecycle.jsonl").read_bytes()
        anchor = json.loads((custody / "custodian-anchor.json").read_bytes())
        anchor["lifecycle_path"] = "lifecycle.jsonl"
        anchor_bytes = _json_bytes(anchor)
        registry_bytes = _json_bytes(
            {
                "commitment": commitment,
                "latest_anchor_sha256": _digest(anchor_bytes),
                "consumed": False,
            }
        )
        (stage / "lifecycle.jsonl").write_bytes(lifecycle_bytes)
        (stage / "anchor.json").write_bytes(anchor_bytes)
        (stage / "custodian-state.json").write_bytes(registry_bytes)
        anchor_digest = _digest(anchor_bytes)
        registry_digest = _digest(registry_bytes)
        protocol.verify(
            stage,
            anchor_digest,
            registry_digest,
            trusted_inventory_manifest_sha256=private_reference["manifest_sha256"],
        )
        if bundle.exists():
            raise FileExistsError("reveal_bundle_already_exists")
        _rename_noreplace(stage, bundle)
        _write_new(
            custody / "anchor-revealed-bundle.sha256",
            (anchor_digest + "\n").encode(),
        )
        return {"anchor_sha256": anchor_digest, "registry_sha256": registry_digest}
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _apply_consumption_pending(bundle: Path, interruption=None) -> dict[str, str]:
    pending_path = bundle / "consumption-pending.json"
    transaction = json.loads(pending_path.read_bytes())
    lifecycle_path = bundle / transaction["lifecycle_path"]
    lifecycle_bytes = base64.b64decode(transaction["lifecycle"], validate=True)
    anchor_bytes = base64.b64decode(transaction["anchor"], validate=True)
    registry_bytes = base64.b64decode(transaction["registry"], validate=True)
    _write_atomic(lifecycle_path, lifecycle_bytes)
    if interruption:
        interruption("after_lifecycle_write")
    _write_atomic(bundle / "anchor.json", anchor_bytes)
    anchor_digest = _digest(anchor_bytes)
    anchor_receipt = bundle / "anchor-consumption.sha256"
    digest_bytes = (anchor_digest + "\n").encode()
    if anchor_receipt.exists():
        if anchor_receipt.read_bytes() != digest_bytes:
            raise ValueError("consumption_anchor_receipt_conflict")
    else:
        _write_new(anchor_receipt, digest_bytes)
    if interruption:
        interruption("after_anchor_write")
    _write_atomic(bundle / "custodian-state.json", registry_bytes)
    pending_path.unlink()
    return {
        "anchor_sha256": anchor_digest,
        "registry_sha256": _digest(registry_bytes),
    }


def consume_seal(
    bundle: Path, interruption: Callable[[str], None] | None = None
) -> dict[str, str]:
    """Advance consumption through a durable, replayable transaction intent."""
    bundle = bundle.expanduser().absolute()
    pending_path = bundle / "consumption-pending.json"
    if pending_path.exists():
        result = _apply_consumption_pending(bundle, interruption)
        result["recovered"] = "true"
        return result
    registry_path = bundle / "custodian-state.json"
    registry = json.loads(registry_path.read_bytes())
    if registry.get("consumed"):
        raise ValueError("holdback_already_consumed")
    anchor_path = bundle / "anchor.json"
    anchor = json.loads(anchor_path.read_bytes())
    events_path = bundle / anchor["lifecycle_path"]
    events = [
        json.loads(line) for line in events_path.read_bytes().splitlines() if line
    ]
    commitment = registry["commitment"]
    if [event["kind"] for event in events] != [
        "commitment",
        "candidate_freeze",
        "reveal",
    ]:
        raise ValueError("consumption_requires_revealed_unconsumed_bundle")
    events.append(
        protocol.lifecycle_event(
            "consumption", {"commitment": commitment}, events[-1]["event_sha256"]
        )
    )
    event_bytes = protocol.event_log_bytes(events)
    previous_time = anchor["checkpoints"][-1]["anchored_at_utc"]
    checkpoints = anchor["checkpoints"] + [
        {
            "event_sha256": events[-1]["event_sha256"],
            "anchored_at_utc": _timestamp_after(previous_time),
        }
    ]
    anchor_bytes = _json_bytes(
        {
            "lifecycle_path": anchor["lifecycle_path"],
            "lifecycle_sha256": _digest(event_bytes),
            "checkpoints": checkpoints,
        }
    )
    anchor_digest = _digest(anchor_bytes)
    registry_bytes = _json_bytes(
        {
            "commitment": commitment,
            "latest_anchor_sha256": anchor_digest,
            "consumed": True,
        }
    )
    transaction = {
        "lifecycle_path": anchor["lifecycle_path"],
        "lifecycle": base64.b64encode(event_bytes).decode("ascii"),
        "anchor": base64.b64encode(anchor_bytes).decode("ascii"),
        "registry": base64.b64encode(registry_bytes).decode("ascii"),
    }
    _write_new(pending_path, _json_bytes(transaction))
    return _apply_consumption_pending(bundle, interruption)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--root", type=Path, default=Path.cwd())
    preflight_parser.add_argument("--output", type=Path)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--custody", type=Path, required=True)
    initialize.add_argument("--root", type=Path, default=Path.cwd())
    initialize.add_argument("--size", type=int, default=1024)
    initialize.add_argument("--seed", type=int, help="synthetic fixtures only")
    initialize.add_argument("--synthetic-fixture", action="store_true")
    initialize.add_argument("--preflight-json", type=Path)
    initialize.add_argument("--manifest", type=Path, required=True)
    initialize.add_argument("--manifest-sha256", required=True)
    resume = subparsers.add_parser("resume")
    resume.add_argument("--custody", type=Path, required=True)
    resume.add_argument("--root", type=Path, default=Path.cwd())
    resume.add_argument("--public-output", type=Path, required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--custody", type=Path, required=True)
    freeze.add_argument("--root", type=Path, default=Path.cwd())
    freeze.add_argument("--freeze-json", type=Path, required=True)
    reveal = subparsers.add_parser("reveal")
    reveal.add_argument("--custody", type=Path, required=True)
    reveal.add_argument("--root", type=Path, default=Path.cwd())
    reveal.add_argument("--bundle", type=Path, required=True)
    reveal.add_argument("--trusted-commitment-sha256", required=True)
    reveal.add_argument("--trusted-freeze-checkpoint-sha256", required=True)
    consume = subparsers.add_parser("consume")
    consume.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.root)
        if args.output is not None:
            _write_new(args.output, _json_bytes(result), PRIVATE_FILE_MODE)
        print(json.dumps(result, sort_keys=True, indent=2))
        return
    if args.command == "initialize":
        reference = {
            "schema": protocol.INVENTORY_REFERENCE_SCHEMA,
            "manifest_path": str(args.manifest),
            "manifest_sha256": args.manifest_sha256,
        }
        initialize_private_inputs(
            args.custody,
            size=args.size,
            inventory_reference=reference,
            root=args.root,
            seed=args.seed,
            synthetic=args.synthetic_fixture,
            preflight_record=(
                json.loads(args.preflight_json.read_bytes())
                if args.preflight_json is not None
                else None
            ),
        )
    else:
        if args.command == "resume":
            result = resume_seal(args.custody, args.root, args.public_output)
        elif args.command == "freeze":
            result = freeze_comparison(
                args.custody,
                json.loads(args.freeze_json.read_bytes()),
                root=args.root,
            )
        elif args.command == "reveal":
            result = reveal_seal(
                args.custody,
                args.bundle,
                root=args.root,
                trusted_commitment_sha256=args.trusted_commitment_sha256,
                trusted_freeze_checkpoint_sha256=args.trusted_freeze_checkpoint_sha256,
            )
        else:
            result = consume_seal(args.bundle)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
