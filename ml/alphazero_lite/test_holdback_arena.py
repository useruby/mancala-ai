from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ml.alphazero_lite import holdback_arena as protocol


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


def setup_bundle(path: Path, *, consumed: bool = True) -> dict[str, str]:
    path.mkdir()
    pool = [row(i) for i in range(10)]
    inventory_doc = {"rows": [row(99)], "sources": [{"id": "synthetic"}]}
    generator = {"name": "synthetic-rules-only-v1", "seed": 17, "size": 4}
    suite = protocol.build_suite(pool, inventory_doc["rows"], 17, 4)
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
    }


class HoldbackProtocolTest(unittest.TestCase):
    def test_rebuild_after_reveal_and_physical_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            trusted = setup_bundle(source, consumed=False)
            copied = root / "copy"
            shutil.copytree(source, copied)
            shutil.rmtree(source)
            result = protocol.verify(
                copied, trusted["latest_anchor"], trusted["registry"]
            )
            self.assertEqual(result["openings"], 4)
            self.assertTrue(result["held_back_eligible"])

    def test_duplicate_and_overlap_rejected_by_both_derived_identities(self):
        with self.assertRaisesRegex(ValueError, "duplicate_exclusion_identity"):
            protocol.validate_inventory([row(1), row(1)])
        # Extra metadata changes canonical identity but is ignored by kalah_v3.
        with self.assertRaisesRegex(ValueError, "eligible_pool_shortfall"):
            protocol.build_suite([row(1), row(1, "different")], [], 1, 2)
        # Same encoded position under a different metadata-only canonical state.
        with self.assertRaisesRegex(ValueError, "eligible_pool_shortfall"):
            protocol.build_suite([row(1, "other")], [row(1)], 1, 1)

    def test_changed_suite_order_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "bundle"
            trusted = setup_bundle(bundle)
            lines = (bundle / "suite.jsonl").read_bytes().splitlines(keepends=True)
            (bundle / "suite.jsonl").write_bytes(b"".join(reversed(lines)))
            with self.assertRaisesRegex(ValueError, "suite_bytes_changed"):
                protocol.verify(bundle, trusted["latest_anchor"], trusted["registry"])

    def test_external_anchor_protects_freeze_and_consumption_lifecycle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "bundle"
            trusted = setup_bundle(bundle, consumed=True)
            result = protocol.verify(
                bundle, trusted["latest_anchor"], trusted["registry"]
            )
            self.assertTrue(result["consumed"])
            self.assertFalse(result["held_back_eligible"])
            historical = protocol.verify(
                bundle,
                trusted["reveal_anchor"],
                trusted["registry"],
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
            self.assertFalse(
                protocol.verify(bundle, trusted["latest_anchor"], trusted["registry"])[
                    "held_back_eligible"
                ]
            )
            event_path = bundle / "lifecycle.jsonl"
            event_path.write_bytes(event_path.read_bytes().replace(b'"a', b'"z', 1))
            with self.assertRaisesRegex(ValueError, "anchor_lifecycle_mismatch"):
                protocol.verify(bundle, trusted["latest_anchor"], trusted["registry"])

    def test_size_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "bundle"
            trusted = setup_bundle(bundle, consumed=False)
            manifest = json.loads((bundle / "manifest.json").read_text())
            manifest["generator"]["size"] = 5
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "suite_declared_size_mismatch"):
                protocol.verify(bundle, trusted["latest_anchor"], trusted["registry"])

    def test_reveal_event_before_candidate_freeze_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "bundle"
            setup_bundle(bundle, consumed=False)
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
                protocol.verify(bundle, trusted_anchor, protocol.sha256(registry_raw))


if __name__ == "__main__":
    unittest.main()
