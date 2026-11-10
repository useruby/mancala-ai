"""Custodian-side, restart-safe private sealing operations for holdback suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import datetime as dt
import shutil
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
    return {
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


def initialize_private_inputs(
    custody: Path,
    *,
    size: int,
    inventory_reference: dict[str, Any],
    root: Path = Path.cwd(),
    seed: int | None = None,
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
    actual_seed = secrets.randbits(256) if seed is None else seed
    if isinstance(actual_seed, bool) or not isinstance(actual_seed, int):
        raise ValueError("seed_invalid")
    generator = protocol.rules_generator_spec(size)
    inputs = {
        "schema": "holdback_private_inputs_v1",
        "seed": actual_seed,
        "generator": generator,
        "inventory_reference": inventory_reference,
    }
    _write_new(custody / "private-inputs.json", _json_bytes(inputs))
    return actual_seed


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
        raise FileExistsError("public_commitment_already_exists")
    inputs_path = custody / "private-inputs.json"
    if not inputs_path.exists():
        raise ValueError("private_inputs_missing_refuse_reseed")
    inputs = _load_inputs(custody)
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
    # No public path is created until every private artifact has been checked.
    public_output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(public_output, _json_bytes(record), 0o644)
    return {"commitment": record["commitment"], "suite_sha256": receipt["suite_sha256"]}


def freeze_comparison(custody: Path, freeze: dict[str, Any]) -> None:
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
    _write_new(
        custody / "comparison-freeze.json",
        _json_bytes({"commitment": commitment, "freeze": freeze}),
    )


def reveal_seal(custody: Path, bundle: Path) -> dict[str, str]:
    """Publish a portable revealed bundle only after the comparison freeze."""
    custody = custody.expanduser().absolute()
    bundle = bundle.expanduser().absolute()
    if bundle.exists():
        raise FileExistsError("reveal_bundle_already_exists")
    freeze_record = json.loads((custody / "comparison-freeze.json").read_bytes())
    inputs = _load_inputs(custody)
    receipt = json.loads((custody / "generation-receipt.json").read_bytes())
    suite = (custody / "suite.jsonl").read_bytes()
    if _digest(suite) != receipt["suite_sha256"]:
        raise ValueError("private_suite_digest_mismatch")
    bundle.mkdir(mode=PRIVATE_MODE, parents=True, exist_ok=False)
    shutil.copy2(custody / "suite.jsonl", bundle / "suite.jsonl")
    inv_ref = inputs["inventory_reference"]
    source = custody / "inventory"
    (bundle / "inventory").mkdir(mode=PRIVATE_MODE)
    for name in (
        "source-manifest.json",
        "canonical-index.jsonl.gz",
        "kalah-v3-float32-index.jsonl.gz",
    ):
        shutil.copy2(source / name, bundle / "inventory" / name)
    public_ref = dict(inv_ref, manifest_path="inventory/source-manifest.json")
    inv_bytes = _json_bytes(public_ref)
    (bundle / "inventory.json").write_bytes(inv_bytes)
    generator = {**inputs["generator"], "seed": inputs["seed"]}
    public_record = protocol.public_commitment_record(
        receipt["suite_sha256"],
        protocol.public_generator_spec(inputs["generator"]),
        public_ref,
    )
    commitment = public_record["commitment"]
    if freeze_record.get("commitment") != commitment:
        raise ValueError("freeze_commitment_mismatch")
    manifest = {
        "suite_sha256": receipt["suite_sha256"],
        "inventory_path": "inventory.json",
        "inventory_sha256": _digest(inv_bytes),
        "generator": generator,
        "commitment": commitment,
    }
    (bundle / "manifest.json").write_bytes(_json_bytes(manifest))
    events = []
    previous = "0" * 64
    for kind, payload in (
        ("commitment", {"commitment": commitment}),
        (
            "candidate_freeze",
            {"commitment": commitment, "freeze": freeze_record["freeze"]},
        ),
        (
            "reveal",
            {
                "commitment": commitment,
                "suite_sha256": receipt["suite_sha256"],
                "seed": inputs["seed"],
            },
        ),
    ):
        event = protocol.lifecycle_event(kind, payload, previous)
        events.append(event)
        previous = event["event_sha256"]
    event_bytes = protocol.event_log_bytes(events)
    (bundle / "lifecycle.jsonl").write_bytes(event_bytes)
    start = dt.datetime.now(dt.timezone.utc)
    anchor = {
        "lifecycle_path": "lifecycle.jsonl",
        "lifecycle_sha256": _digest(event_bytes),
        "checkpoints": [
            {
                "event_sha256": event["event_sha256"],
                "anchored_at_utc": (start + dt.timedelta(microseconds=i)).isoformat(),
            }
            for i, event in enumerate(events)
        ],
    }
    anchor_bytes = _json_bytes(anchor)
    (bundle / "anchor.json").write_bytes(anchor_bytes)
    anchor_digest = _digest(anchor_bytes)
    registry_bytes = _json_bytes(
        {
            "commitment": commitment,
            "latest_anchor_sha256": anchor_digest,
            "consumed": False,
        }
    )
    (bundle / "custodian-state.json").write_bytes(registry_bytes)
    return {"anchor_sha256": anchor_digest, "registry_sha256": _digest(registry_bytes)}


def consume_seal(bundle: Path) -> dict[str, str]:
    """Append the consumption lifecycle event and atomically advance trust roots."""
    bundle = bundle.expanduser().absolute()
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
    events.append(
        protocol.lifecycle_event(
            "consumption", {"commitment": commitment}, events[-1]["event_sha256"]
        )
    )
    event_bytes = protocol.event_log_bytes(events)
    events_path.write_bytes(event_bytes)
    start = dt.datetime.now(dt.timezone.utc)
    checkpoints = anchor["checkpoints"] + [
        {
            "event_sha256": events[-1]["event_sha256"],
            "anchored_at_utc": (start + dt.timedelta(microseconds=1)).isoformat(),
        }
    ]
    anchor_bytes = _json_bytes(
        {
            "lifecycle_path": anchor["lifecycle_path"],
            "lifecycle_sha256": _digest(event_bytes),
            "checkpoints": checkpoints,
        }
    )
    temporary_anchor = bundle / "anchor.json.new"
    _write_new(temporary_anchor, anchor_bytes)
    os.replace(temporary_anchor, anchor_path)
    anchor_digest = _digest(anchor_bytes)
    registry_bytes = _json_bytes(
        {
            "commitment": commitment,
            "latest_anchor_sha256": anchor_digest,
            "consumed": True,
        }
    )
    temporary_registry = bundle / "custodian-state.json.new"
    _write_new(temporary_registry, registry_bytes)
    os.replace(temporary_registry, registry_path)
    return {"anchor_sha256": anchor_digest, "registry_sha256": _digest(registry_bytes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--root", type=Path, default=Path.cwd())
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--custody", type=Path, required=True)
    initialize.add_argument("--root", type=Path, default=Path.cwd())
    initialize.add_argument("--size", type=int, default=1024)
    initialize.add_argument("--seed", type=int, help="synthetic fixtures only")
    initialize.add_argument("--manifest", type=Path, required=True)
    initialize.add_argument("--manifest-sha256", required=True)
    resume = subparsers.add_parser("resume")
    resume.add_argument("--custody", type=Path, required=True)
    resume.add_argument("--root", type=Path, default=Path.cwd())
    resume.add_argument("--public-output", type=Path, required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--custody", type=Path, required=True)
    freeze.add_argument("--freeze-json", type=Path, required=True)
    reveal = subparsers.add_parser("reveal")
    reveal.add_argument("--custody", type=Path, required=True)
    reveal.add_argument("--bundle", type=Path, required=True)
    consume = subparsers.add_parser("consume")
    consume.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.root)
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
        )
    else:
        if args.command == "resume":
            result = resume_seal(args.custody, args.root, args.public_output)
        elif args.command == "freeze":
            freeze_comparison(args.custody, json.loads(args.freeze_json.read_bytes()))
            result = {"status": "comparison_frozen"}
        elif args.command == "reveal":
            result = reveal_seal(args.custody, args.bundle)
        else:
            result = consume_seal(args.bundle)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
