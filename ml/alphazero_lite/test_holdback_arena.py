from __future__ import annotations

import json
import gzip
import hashlib
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import holdback_arena as protocol
from ml.alphazero_lite import (
    seed561_historical_exclusion_inventory as historical_inventory,
)


def row(n: int, tag: str | None = None) -> dict:
    state = {
        "player_pits": [n, 0, 0, 0, 0, 0],
        "opponent_pits": [0, 0, 0, 0, 0, 0],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    if tag is not None:
        state["synthetic_tag"] = tag
    return {"state": state}


def write_rows(path: Path, rows: list[dict]) -> bytes:
    payload = protocol.suite_bytes(rows)
    path.write_bytes(payload)
    return payload


def setup_bundle(
    path: Path,
    *,
    consumed: bool = True,
    pool: list[dict] | None = None,
    inventory_rows: list[dict] | None = None,
    size: int = 4,
) -> dict[str, str]:
    path.mkdir()
    pool = pool or [row(i) for i in range(10)]
    inventory_rows = inventory_rows or [row(99)]
    generator = {"name": "synthetic-rules-only-v1", "seed": 17, "size": size}
    index_bindings = {}
    identity_sets = {"canonical": set(), "float32": set()}
    for scheme, filename, position in (
        ("canonical", "canonical-index.jsonl.gz", 0),
        ("float32", "float32-index.jsonl.gz", 1),
    ):
        index_rows = []
        for source_row in inventory_rows:
            pair = protocol.identities(source_row)
            value = pair[position]
            identity_sets[scheme].add(value)
            index_rows.append(
                {
                    "identity": value,
                    "source": "synthetic",
                    "provenance": {
                        "fixture_row": source_row["state"].get("synthetic_tag", "known")
                    },
                }
            )
        payload = b"".join(
            protocol.canonical_bytes(index_row) + b"\n" for index_row in index_rows
        )
        compressed = gzip.compress(payload, mtime=0)
        (path / filename).write_bytes(compressed)
        index_bindings["canonical" if scheme == "canonical" else "kalah_v3_float32"] = {
            "path": filename,
            "sha256_gzip": hashlib.sha256(compressed).hexdigest(),
            "sha256_jsonl": hashlib.sha256(payload).hexdigest(),
            "occurrence_rows": len(index_rows),
            "distinct_identities": len(identity_sets[scheme]),
        }
    inventory_manifest = {
        "schema": protocol.INVENTORY_MANIFEST_SCHEMA,
        "identity_schemes": ["canonical_state_sha256", "kalah_v3_float32_sha256"],
        "indexes": index_bindings,
    }
    inventory_manifest_raw = (
        json.dumps(inventory_manifest, sort_keys=True, indent=2).encode() + b"\n"
    )
    (path / "inventory-source-manifest.json").write_bytes(inventory_manifest_raw)
    trusted_inventory_manifest = protocol.sha256(inventory_manifest_raw)
    inventory_doc = {
        "schema": protocol.INVENTORY_REFERENCE_SCHEMA,
        "manifest_path": "inventory-source-manifest.json",
        "manifest_sha256": trusted_inventory_manifest,
    }
    excluded = protocol.load_pinned_identity_indexes(
        path, inventory_doc, trusted_inventory_manifest
    )
    suite = protocol.build_suite(pool, excluded, 17, size)
    suite_raw = write_rows(path / "suite.jsonl", suite)
    inv_raw = protocol.canonical_bytes(inventory_doc)
    (path / "inventory.json").write_bytes(inv_raw)
    pool_raw = write_rows(path / "pool.jsonl", pool)
    generator["pool_sha256"] = protocol.sha256(pool_raw)
    commitment = protocol.commitment(suite_raw, generator, inventory_doc)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "suite_sha256": protocol.sha256(suite_raw),
                "inventory_path": "inventory.json",
                "inventory_sha256": protocol.sha256(inv_raw),
                "pool_path": "pool.jsonl",
                "pool_sha256": protocol.sha256(pool_raw),
                "generator": generator,
                "commitment": commitment,
            }
        )
    )
    freeze = {
        "candidate_artifact_sha256": "a" * 64,
        "serving_p_artifact_sha256": "b" * 64,
        "arena_settings": {"pairs": 4, "workers": 1},
        "analysis_rule": "paired-opening mean",
    }
    events = []
    previous = "0" * 64
    for kind, payload in (
        ("commitment", {"commitment": commitment}),
        ("candidate_freeze", {"commitment": commitment, "freeze": freeze}),
        (
            "reveal",
            {
                "commitment": commitment,
                "suite_sha256": protocol.sha256(suite_raw),
                "seed": 17,
                "pool_sha256": protocol.sha256(pool_raw),
            },
        ),
    ):
        event = protocol.lifecycle_event(kind, payload, previous)
        events.append(event)
        previous = event["event_sha256"]
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def publish_anchor(event_list, log_name, anchor_name, time_offset):
        event_raw = protocol.event_log_bytes(event_list)
        (path / log_name).write_bytes(event_raw)
        anchor = {
            "lifecycle_path": log_name,
            "lifecycle_sha256": protocol.sha256(event_raw),
            "checkpoints": [
                {
                    "event_sha256": event["event_sha256"],
                    "anchored_at_utc": (
                        start + timedelta(seconds=time_offset + index)
                    ).isoformat(),
                }
                for index, event in enumerate(event_list)
            ],
        }
        anchor_raw = protocol.canonical_bytes(anchor)
        (path / anchor_name).write_bytes(anchor_raw)
        return protocol.sha256(anchor_raw)

    reveal_anchor = publish_anchor(
        events, "lifecycle-reveal.jsonl", "anchor-reveal.json", 0
    )
    latest_anchor = reveal_anchor
    if consumed:
        events.append(
            protocol.lifecycle_event(
                "consumption", {"commitment": commitment}, previous
            )
        )
        latest_anchor = publish_anchor(events, "lifecycle.jsonl", "anchor.json", 10)
    else:
        (path / "lifecycle.jsonl").write_bytes(
            (path / "lifecycle-reveal.jsonl").read_bytes()
        )
        (path / "anchor.json").write_bytes((path / "anchor-reveal.json").read_bytes())
    registry_raw = protocol.canonical_bytes(
        {
            "commitment": commitment,
            "latest_anchor_sha256": latest_anchor,
            "consumed": consumed,
        }
    )
    (path / "custodian-state.json").write_bytes(registry_raw)
    return {
        "reveal_anchor": reveal_anchor,
        "latest_anchor": latest_anchor,
        "registry": protocol.sha256(registry_raw),
        "inventory_manifest": trusted_inventory_manifest,
    }


def verify_bundle(bundle: Path, trusted: dict[str, str], **kwargs) -> dict:
    return protocol.verify(
        bundle,
        trusted["latest_anchor"],
        trusted["registry"],
        trusted_inventory_manifest_sha256=trusted["inventory_manifest"],
        **kwargs,
    )


class HoldbackProtocolTest(unittest.TestCase):
    def test_rules_only_generation_is_deterministic_and_serialized(self):
        inventory = {"canonical": set(), "float32": set()}
        first = protocol.build_rules_only_suite(inventory, 1234, 3)
        self.assertEqual(first, protocol.build_rules_only_suite(inventory, 1234, 3))
        self.assertEqual([0, 1, 2], [item["opening_index"] for item in first])
        for opening in first:
            self.assertEqual("arena_player_relative_v2", opening["opening_contract"])
            self.assertEqual(
                len(opening["prefix_moves"]),
                len(opening["source_prefix_moves_absolute"]),
            )
            self.assertIn(len(opening["prefix_moves"]), range(4, 13))
        self.assertTrue(protocol.suite_bytes(first).endswith(b"\n"))

    def test_rules_only_collision_under_either_identity_and_shortfall(self):
        empty = {"canonical": set(), "float32": set()}
        opening = protocol.build_rules_only_suite(empty, 44, 1)[0]
        can, f32 = protocol.identities(opening)
        for scheme, value in (("canonical", can), ("float32", f32)):
            inventory = {"canonical": set(), "float32": set()}
            inventory[scheme].add(value)
            with (
                self.subTest(scheme=scheme),
                self.assertRaisesRegex(ValueError, "rules_only_opening_shortfall"),
            ):
                protocol.build_rules_only_suite(inventory, 44, 1, attempt_limit=1)
        with self.assertRaisesRegex(ValueError, "rules_only_opening_shortfall"):
            protocol.build_rules_only_suite(empty, 1, 1, attempt_limit=0)

    def test_seed_free_commitment_binds_suite_generator_and_inventory(self):
        generator = {
            "name": protocol.RULES_GENERATOR,
            "size": 1024,
            "rng": "python.random.Random/MT19937",
            "attempt_limit": 20_000,
        }
        inventory = {"manifest_sha256": "a" * 64, "schema": "inventory-v1"}
        record = protocol.public_commitment_record("b" * 64, generator, inventory)
        protocol.verify_public_commitment_record(record)
        self.assertNotIn("seed", record["generator"])
        changed = dict(record, suite_sha256="c" * 64)
        with self.assertRaisesRegex(ValueError, "public_commitment_record_mismatch"):
            protocol.verify_public_commitment_record(changed)
        with self.assertRaisesRegex(
            ValueError, "public_generator_must_not_expose_seed"
        ):
            protocol.public_commitment_record(
                "b" * 64, {**generator, "seed": 9}, inventory
            )

    def test_public_loader_authenticates_published_historical_indexes(self):
        reference = {
            "schema": protocol.INVENTORY_REFERENCE_SCHEMA,
            "manifest_path": (
                "docs/data/seed561-historical-exclusion-inventory-v1/"
                "source-manifest.json"
            ),
            "manifest_sha256": protocol.TRUSTED_HISTORICAL_INVENTORY_MANIFEST_SHA256,
        }
        excluded = protocol.load_pinned_identity_indexes(
            historical_inventory.ROOT,
            reference,
            protocol.TRUSTED_HISTORICAL_INVENTORY_MANIFEST_SHA256,
        )
        self.assertEqual(len(excluded["canonical"]), 534033)
        self.assertEqual(len(excluded["float32"]), 534033)

    def test_rebuild_after_reveal_and_physical_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            trusted = setup_bundle(source, consumed=False)
            copied = root / "copy"
            shutil.copytree(source, copied)
            shutil.rmtree(source)
            before = {
                item.relative_to(copied): (item.read_bytes(), item.stat().st_mtime_ns)
                for item in copied.rglob("*")
                if item.is_file()
            }
            result = verify_bundle(copied, trusted)
            self.assertEqual(result["openings"], 4)
            self.assertTrue(result["held_back_eligible"])
            after = {
                item.relative_to(copied): (item.read_bytes(), item.stat().st_mtime_ns)
                for item in copied.rglob("*")
                if item.is_file()
            }
            self.assertEqual(before, after)

    def test_duplicate_and_overlap_rejected_by_both_derived_identities(self):
        with self.assertRaisesRegex(ValueError, "duplicate_exclusion_identity"):
            protocol.validate_inventory([row(1), row(1)])
        # Extra metadata changes canonical identity but is ignored by kalah_v3.
        with self.assertRaisesRegex(ValueError, "eligible_pool_shortfall"):
            protocol.build_suite([row(1), row(1, "different")], [], 1, 2)
        # Same encoded position under a different metadata-only canonical state.
        with self.assertRaisesRegex(ValueError, "eligible_pool_shortfall"):
            protocol.build_suite([row(1, "other")], [row(1)], 1, 1)

    def test_public_verifier_excludes_opening_collision_in_either_identity_scheme(self):
        for collision in ("canonical", "float32"):

            def synthetic_identities(item, collision=collision):
                tag = item["state"]["synthetic_tag"]
                if tag == "known":
                    return "c" * 64, "f" * 64
                if collision == "canonical":
                    return (
                        ("c" * 64, "d" * 64)
                        if tag == "candidate"
                        else ("a" * 64, "b" * 64)
                    )
                return (
                    ("d" * 64, "f" * 64) if tag == "candidate" else ("a" * 64, "b" * 64)
                )

            with (
                self.subTest(collision=collision),
                patch.object(protocol, "identities", side_effect=synthetic_identities),
                tempfile.TemporaryDirectory() as temporary,
            ):
                bundle = Path(temporary) / "bundle"
                trusted = setup_bundle(
                    bundle,
                    pool=[row(1, "candidate"), row(2, "safe")],
                    inventory_rows=[row(99, "known")],
                    size=1,
                )
                result = verify_bundle(bundle, trusted)
                self.assertEqual(result["openings"], 1)
                suite_rows = [
                    json.loads(line)
                    for line in (bundle / "suite.jsonl").read_text().splitlines()
                ]
                self.assertNotIn(
                    "candidate", [x["state"]["synthetic_tag"] for x in suite_rows]
                )

    def test_public_verifier_rejects_tampered_index_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            trusted = setup_bundle(bundle, consumed=False)
            path = bundle / "canonical-index.jsonl.gz"
            original = path.read_bytes()
            rows = [json.loads(line) for line in gzip.decompress(original).splitlines()]
            rows[0]["identity"] = "f" * 64
            changed = b"".join(protocol.canonical_bytes(x) + b"\n" for x in rows)
            path.write_bytes(gzip.compress(changed, mtime=0))
            tampered_bytes = path.read_bytes()
            tampered_mtime = path.stat().st_mtime_ns
            with self.assertRaisesRegex(
                ValueError, "inventory_index_content_mismatch:canonical"
            ):
                verify_bundle(bundle, trusted)
            self.assertEqual(path.read_bytes(), tampered_bytes)
            self.assertEqual(path.stat().st_mtime_ns, tampered_mtime)

    def test_public_verifier_accepts_equivalent_gzip_encoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            trusted = setup_bundle(bundle, consumed=False)
            path = bundle / "canonical-index.jsonl.gz"
            jsonl = gzip.decompress(path.read_bytes())
            path.write_bytes(gzip.compress(jsonl, compresslevel=1, mtime=123))
            result = verify_bundle(bundle, trusted)
            self.assertEqual(result["openings"], 4)

    def test_changed_suite_order_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "bundle"
            trusted = setup_bundle(bundle)
            lines = (bundle / "suite.jsonl").read_bytes().splitlines(keepends=True)
            (bundle / "suite.jsonl").write_bytes(b"".join(reversed(lines)))
            with self.assertRaisesRegex(ValueError, "suite_bytes_changed"):
                verify_bundle(bundle, trusted)

    def test_external_anchor_protects_freeze_and_consumption_lifecycle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "bundle"
            trusted = setup_bundle(bundle, consumed=True)
            result = verify_bundle(bundle, trusted)
            self.assertTrue(result["consumed"])
            self.assertFalse(result["held_back_eligible"])
            historical = verify_bundle(
                bundle,
                {
                    **trusted,
                    "latest_anchor": trusted["reveal_anchor"],
                },
                anchor_name="anchor-reveal.json",
            )
            self.assertTrue(historical["verified_lifecycle_consumed"] is False)
            self.assertTrue(historical["consumed"])
            self.assertFalse(historical["anchor_is_current"])
            self.assertFalse(historical["held_back_eligible"])
            # Removing a convenience marker or replacing a sidecar freeze has no
            # effect: lifecycle is contained in the independently anchored log.
            (bundle / "consumed.json").unlink(missing_ok=True)
            (bundle / "candidate-freeze.json").write_text("{}")
            self.assertFalse(verify_bundle(bundle, trusted)["held_back_eligible"])
            event_path = bundle / "lifecycle.jsonl"
            event_path.write_bytes(event_path.read_bytes().replace(b'"a', b'"z', 1))
            with self.assertRaisesRegex(ValueError, "anchor_lifecycle_mismatch"):
                verify_bundle(bundle, trusted)

    def test_size_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "bundle"
            trusted = setup_bundle(bundle, consumed=False)
            manifest = json.loads((bundle / "manifest.json").read_text())
            manifest["generator"]["size"] = 5
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "suite_declared_size_mismatch"):
                verify_bundle(bundle, trusted)

    def test_reveal_event_before_candidate_freeze_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "bundle"
            trusted = setup_bundle(bundle, consumed=False)
            events = [
                json.loads(line)
                for line in (bundle / "lifecycle.jsonl").read_text().splitlines()
            ]
            payloads = {event["kind"]: event["payload"] for event in events}
            ordered = []
            previous = "0" * 64
            for kind in ("commitment", "reveal", "candidate_freeze"):
                event = protocol.lifecycle_event(kind, payloads[kind], previous)
                ordered.append(event)
                previous = event["event_sha256"]
            log_raw = protocol.event_log_bytes(ordered)
            (bundle / "lifecycle.jsonl").write_bytes(log_raw)
            start = datetime(2026, 1, 1, tzinfo=timezone.utc)
            anchor_doc = {
                "lifecycle_path": "lifecycle.jsonl",
                "lifecycle_sha256": protocol.sha256(log_raw),
                "checkpoints": [
                    {
                        "event_sha256": event["event_sha256"],
                        "anchored_at_utc": (
                            start + timedelta(seconds=index)
                        ).isoformat(),
                    }
                    for index, event in enumerate(ordered)
                ],
            }
            anchor_raw = protocol.canonical_bytes(anchor_doc)
            (bundle / "anchor.json").write_bytes(anchor_raw)
            trusted_anchor = protocol.sha256(anchor_raw)
            registry_raw = protocol.canonical_bytes(
                {
                    "commitment": json.loads((bundle / "manifest.json").read_text())[
                        "commitment"
                    ],
                    "latest_anchor_sha256": trusted_anchor,
                    "consumed": False,
                }
            )
            (bundle / "custodian-state.json").write_bytes(registry_raw)
            with self.assertRaisesRegex(ValueError, "lifecycle_event_order"):
                protocol.verify(
                    bundle,
                    trusted_anchor,
                    protocol.sha256(registry_raw),
                    trusted_inventory_manifest_sha256=trusted["inventory_manifest"],
                )


if __name__ == "__main__":
    unittest.main()
