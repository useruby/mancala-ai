"""Read-only reconstruction of Seed 621 exposure to historical training."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.self_play import encode_state

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed622-training-provenance-audit"
S599 = ROOT / "docs/data/seed599-replay-coverage"
S615 = ROOT / "docs/data/seed615-source-opening-diversity"
PARENT = ROOT / "docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4/model.npz"
HASHES = {
    "B512_checkpoint": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
    "S615_checkpoint": "cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780",
    "P_E4_parent": "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
    "B512_membership": "95ecc180e28945702e08815de7a5210e1bce036afdaf76d194bbd774589a4516",
    "S615_membership": "5ed236266ed3804a9c0e9f75b7e10b7c81ea6e290f8c72d3cea06dd225a15010",
    "holdout_membership": "4e74d57244ab55b69751c38e4097860c25e541a8db60a2109f2ebd64c3eb81a8",
    "S615_training_receipt": "0dc4d04a6be2b638abf77bb959e93d82c60355f3f9a449a49d0fef0a8bbb81f3",
    "S615_training_trace": "8ee2252f6831ea0589a19dff2c1ebc79a0f3c3447764feb9c53086ba599a8b01",
    "S615_training_report": "c383411a9ce9b896ef49431e502e0047b26de0e8600042d8b2873ae07c796266",
    "B512_training_receipt": "3d81b00c438a678a2e38b9d58dcadc22f4091c0cb8e3ccbb755e1afe19ca41de",
    "B512_training_trace": "9f22328080756aa0fa2fa066164d36f935643938363059e0d0c711eb6f8f8955",
    "B512_independent_verification": "57fd2623874a0064814b71847763f52ad890cc1833d8daffdaa222786673da46",
    "Seed621_inventory": "332792e07accb8856f6cd5c0b65269dcfe55b392c93560369b7b685c2f86ad29",
    "Seed599_preparation": "11c580d2ebb8d4a8fc67b12f7d1f8e6d8bcbffa0d110e6549bf077d259ff751d",
    "B512_freeze": "bd141778461d7046b067966d7fb87188e5b0a2630d82655a321059b4ce17cad9",
    "S615_freeze": "6ac9d4b4e52a6f4789ce406d9c88badd82e9c8ab7cf4b4d4d8e9c8676cd9910d",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def identity(state: dict[str, Any]) -> str:
    return sha(canonical(state))


def float32_identity(state: dict[str, Any]) -> str:
    vector = np.asarray(encode_state(state, input_encoding="kalah_v3"), dtype="<f4")
    return sha(vector.tobytes())


def authenticate_seed621_sources(rows: list[dict[str, Any]]) -> None:
    """Recheck source game batches and canonical source-record digests."""
    cache: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    for row in rows:
        for source in row["source_clusters"]:
            relative = source["source_path"]
            if relative not in cache:
                raw = (ROOT / relative).read_bytes()
                parsed = [json.loads(line) for line in raw.splitlines() if line]
                cache[relative] = (sha(raw), parsed)
            batch_hash, records = cache[relative]
            line_index = int(source["source_line"]) - 1
            if batch_hash != source["source_batch_sha256"] or not 0 <= line_index < len(
                records
            ):
                raise ValueError("Seed621_source_batch_or_line_mismatch")
            record = records[line_index]
            if (
                sha(canonical(record)) != source["source_record_sha256"]
                or Path(relative).name
                != f"games-{int(source['opening_index']):04d}.jsonl"
            ):
                raise ValueError("Seed621_source_record_provenance_mismatch")


def classify_exposure(exposure: dict[str, Any]) -> str:
    """Never clear a state on nonmembership without a complete contract."""
    confirmed = any(
        bool(exposure.get(key))
        for key in (
            "B512_canonical",
            "B512_float32",
            "S615_canonical",
            "S615_float32",
            "protected_holdout_canonical",
            "protected_holdout_float32",
        )
    )
    return (
        "excluded_confirmed_exposure"
        if confirmed
        else "unresolved_insufficient_provenance"
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def authenticated_training() -> tuple[dict[str, list[dict]], dict[str, Any]]:
    """Authenticate memberships/receipts and recover rows touched by actual batches."""
    memberships = {
        "B512": load_json(S599 / "treatment-membership.json"),
        "S615": load_jsonl(S615 / "candidate-membership.jsonl"),
        "protected_holdout": load_json(S599 / "heldout-membership.json"),
    }
    preparation_path = S599 / "preparation.json"
    preparation_raw = preparation_path.read_bytes()
    if sha(preparation_raw) != HASHES["Seed599_preparation"]:
        raise ValueError("seed599_preparation_hash_mismatch")
    preparation = json.loads(preparation_raw)
    source_artifacts = preparation["source_verification"]["source_artifacts"]
    for relative, expected_hash in source_artifacts.items():
        artifact = ROOT / relative
        if not artifact.is_file() or sha(artifact.read_bytes()) != expected_hash:
            raise ValueError(f"seed580_585_source_artifact_hash_mismatch:{relative}")
    for name, rows in memberships.items():
        expected_membership_hash = HASHES[
            "holdout_membership"
            if name == "protected_holdout"
            else f"{name}_membership"
        ]
        if sha(canonical(rows)) != expected_membership_hash:
            raise ValueError(f"membership_hash_mismatch:{name}")
        seen = set()
        for row in rows:
            state = row.get("canonical_state")
            if not isinstance(state, dict) or identity(state) != row.get(
                "canonical_identity"
            ):
                raise ValueError(f"membership_canonical_identity_mismatch:{name}")
            computed = float32_identity(state)
            if computed != row.get("float32_identity"):
                raise ValueError(f"membership_float32_identity_mismatch:{name}")
            seen.add(row["canonical_identity"])
        if name != "protected_holdout" and len(seen) != len(rows):
            raise ValueError(f"duplicate_training_membership:{name}")

    used: dict[str, set[int]] = {}
    lineage: dict[str, Any] = {}
    for name, directory, trace_name, receipt_sha in (
        (
            "B512",
            S599 / "training-run-B",
            "training-trace.jsonl",
            HASHES["B512_training_receipt"],
        ),
        (
            "S615",
            S615 / "training-run-S615",
            "training-trace.jsonl",
            HASHES["S615_training_receipt"],
        ),
    ):
        receipt_path = directory / "training-receipt.json"
        trace_path = directory / trace_name
        receipt = load_json(receipt_path)
        freeze_path = (
            S599 / "training-execution-freeze.json"
            if name == "B512"
            else S615 / "training-execution-freeze-v3.json"
        )
        freeze_raw = freeze_path.read_bytes()
        if sha(freeze_raw) != HASHES[f"{name}_freeze"]:
            raise ValueError(f"training_freeze_hash_mismatch:{name}")
        freeze = json.loads(freeze_raw)
        source_hashes = freeze.get("source_sha256", {})
        if not source_hashes:
            source_hashes = freeze.get("source_hashes", {})
        for relative, expected_hash in source_hashes.items():
            if sha((ROOT / relative).read_bytes()) != expected_hash:
                raise ValueError(
                    f"training_source_code_hash_mismatch:{name}:{relative}"
                )
        if receipt_sha and sha(receipt_path.read_bytes()) != receipt_sha:
            raise ValueError(f"training_receipt_hash_mismatch:{name}")
        trace_raw = trace_path.read_bytes()
        if name == "B512" and sha(trace_raw) != HASHES["B512_training_trace"]:
            raise ValueError("B512_trace_hash_mismatch")
        if name == "S615" and sha(trace_raw) != HASHES["S615_training_trace"]:
            raise ValueError("S615_trace_hash_mismatch")
        trace = [json.loads(line) for line in trace_raw.splitlines() if line]
        expected_updates = 512
        if len(trace) != expected_updates or receipt.get("updates") != expected_updates:
            raise ValueError(f"training_update_count_mismatch:{name}")
        if receipt.get("trace_sha256") != sha(trace_raw):
            raise ValueError(f"training_trace_receipt_mismatch:{name}")
        if receipt.get("checkpoint_sha256") != HASHES[f"{name}_checkpoint"]:
            raise ValueError(f"training_checkpoint_receipt_mismatch:{name}")
        if name == "B512":
            verification_path = directory / "independent-verification.json"
            if (
                sha(verification_path.read_bytes())
                != HASHES["B512_independent_verification"]
            ):
                raise ValueError("B512_independent_verification_hash_mismatch")
        if name == "S615":
            report = directory / "training-integrity-report.json"
            if sha(report.read_bytes()) != HASHES["S615_training_report"]:
                raise ValueError("S615_integrity_report_hash_mismatch")
            schedule = load_json(S599 / "batch-schedule-B.json")
            registered = load_json(S615 / "registration.json")["treatment"]
            if sha(canonical(schedule)) != registered["batch_index_schedule_sha256"]:
                raise ValueError("S615_registered_schedule_mismatch")
        else:
            schedule = load_json(S599 / "batch-schedule-B.json")
        if len(schedule) != expected_updates:
            raise ValueError(f"registered_schedule_length_mismatch:{name}")
        indexes: set[int] = set()
        for update, record in enumerate(trace):
            batch = schedule[update]
            if record.get("update") != update + 1 or record.get("indexes") != batch:
                raise ValueError(f"trace_schedule_mismatch:{name}:{update + 1}")
            if len(batch) != 32 or any(
                type(i) is not int or i < 0 or i >= len(memberships[name])
                for i in batch
            ):
                raise ValueError(f"invalid_batch_membership_index:{name}:{update + 1}")
            indexes.update(batch)
        if receipt.get("membership_sha256") != HASHES[f"{name}_membership"]:
            raise ValueError(f"training_membership_receipt_mismatch:{name}")
        if receipt.get("parent_checkpoint_sha256") != HASHES["P_E4_parent"]:
            raise ValueError(f"training_parent_receipt_mismatch:{name}")
        if receipt.get("batch_schedule_sha256") != sha(canonical(schedule)):
            raise ValueError(f"training_schedule_receipt_mismatch:{name}")
        if receipt.get("execution_freeze_sha256") != sha(freeze_raw):
            raise ValueError(f"training_freeze_receipt_mismatch:{name}")
        used[name] = indexes
        lineage[name] = {
            "updates": len(trace),
            "batch_size": 32,
            "unique_membership_rows_consumed": len(indexes),
            "membership_rows": len(memberships[name]),
            "membership_sha256": HASHES[f"{name}_membership"],
            "batch_schedule_sha256": sha(canonical(schedule)),
            "execution_freeze_sha256": sha(freeze_raw),
            "parent_checkpoint_sha256": HASHES["P_E4_parent"],
            "checkpoint_sha256": HASHES[f"{name}_checkpoint"],
            "trace_sha256": sha(trace_raw),
            "receipt_sha256": sha(receipt_path.read_bytes()),
        }
    if sha(PARENT.read_bytes()) != HASHES["P_E4_parent"]:
        raise ValueError("parent_checkpoint_hash_mismatch")
    # The schedule is a series of full shuffled passes. Verify the exact rows
    # touched by the traces rather than treating source archive rows as consumed.
    consumed = {
        name: [memberships[name][index] for index in sorted(indexes)]
        for name, indexes in used.items()
    }
    return {**memberships, **consumed}, lineage


def audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    populations, lineage = authenticated_training()
    inventory_path = (
        ROOT / "docs/data/seed621-oracle-cohort-audit/eligible-state-inventory.jsonl"
    )
    raw = inventory_path.read_bytes()
    if sha(raw) != HASHES["Seed621_inventory"]:
        raise ValueError("Seed621_inventory_hash_mismatch")
    source = load_jsonl(inventory_path)
    authenticate_seed621_sources(source)
    ids = set()
    ledger = []
    for row in source:
        state = row["state"]
        cid, f32 = identity(state), float32_identity(state)
        if cid != row["state_id"] or cid in ids:
            raise ValueError("Seed621_canonical_identity_or_duplicate_mismatch")
        ids.add(cid)
        exact = {}
        encoded = {}
        for population in ("B512", "S615", "protected_holdout"):
            exact[population] = any(
                x["canonical_identity"] == cid for x in populations[population]
            )
            encoded[population] = any(
                x["float32_identity"] == f32 for x in populations[population]
            )
        # Neither the archived source corpus nor P-E4's inherited training
        # history supplies an identity-complete exclusion set.
        classification = classify_exposure(
            {
                "B512_canonical": exact["B512"],
                "B512_float32": encoded["B512"],
                "S615_canonical": exact["S615"],
                "S615_float32": encoded["S615"],
                "protected_holdout_canonical": exact["protected_holdout"],
                "protected_holdout_float32": encoded["protected_holdout"],
            }
        )
        contamination_categories = []
        for population, label in (
            ("B512", "B512_training_identity_overlap"),
            ("S615", "S615_training_identity_overlap"),
            ("protected_holdout", "protected_holdout_overlap"),
        ):
            if exact[population] or encoded[population]:
                contamination_categories.append(label)
        if not contamination_categories:
            contamination_categories.append("no_direct_membership_overlap")
        ledger.append(
            {
                "state_id": cid,
                "canonical_identity": cid,
                "production_float32_identity": f32,
                "cohorts": row["cohorts"],
                "source_clusters": row["source_clusters"],
                "prior_class": "agreement"
                if row["top_prior_agreement"]
                else "disagreement",
                "contamination_categories": contamination_categories,
                "exposure": {
                    "B512_canonical": exact["B512"],
                    "B512_float32": encoded["B512"],
                    "S615_canonical": exact["S615"],
                    "S615_float32": encoded["S615"],
                    "protected_holdout_canonical": exact["protected_holdout"],
                    "protected_holdout_float32": encoded["protected_holdout"],
                    "source_game_or_opening_relationship": "unresolved_incompatible_historical_identifiers",
                    "parent_checkpoint_exposure": "unresolved_no_complete_parent_training_record",
                },
                "classification": classification,
                "exclusion_scope": "actual consumed batch rows; production kalah_v3 float32 input equivalence; protected holdout membership",
                "limitations": [
                    "P-E4 inherited training membership unavailable as a complete authenticated state ledger",
                    "source-game/opening exposure cannot be exhaustively mapped from the available training membership; archive presence alone is not treated as consumed",
                    "no augmentation is present in the Seed 599/615 executor contracts; any unrecorded historical upstream transformation remains outside authenticated scope",
                ],
            }
        )
    counts = Counter(x["classification"] for x in ledger)
    by_class = Counter((x["classification"], x["prior_class"]) for x in ledger)
    by_contamination_category = Counter(
        category for entry in ledger for category in entry["contamination_categories"]
    )
    by_cohort: dict[str, Counter] = defaultdict(Counter)
    by_source_cluster: dict[str, Counter] = defaultdict(Counter)
    unresolved_cluster_sizes: Counter[str] = Counter()
    for entry in ledger:
        label = entry["classification"]
        for cohort in entry["cohorts"]:
            by_cohort[cohort][(label, entry["prior_class"])] += 1
        for source in entry["source_clusters"]:
            cluster = f"{source.get('source_path')}#opening_index={source.get('opening_index')}"
            by_source_cluster[cluster][(label, entry["prior_class"])] += 1
            if label == "unresolved_insufficient_provenance":
                unresolved_cluster_sizes[cluster] += 1
    clusters: dict[str, set[str]] = defaultdict(set)
    for row in ledger:
        if row["classification"] == "cleared_under_verified_contract":
            for cohort in row["cohorts"]:
                clusters[cohort].add(row["state_id"])
    report = {
        "schema": "seed622-training-provenance-audit-v1",
        "read_only": True,
        "historical_evidence_modified": False,
        "training_lineage": lineage,
        "source_corpus_authentication": {
            "preparation_sha256": HASHES["Seed599_preparation"],
            "authenticated_source_artifacts": len(
                json.loads((S599 / "preparation.json").read_bytes())[
                    "source_verification"
                ]["source_artifacts"]
            ),
            "sources": ["#580", "#585"],
        },
        "training_contract": {
            "parent": "P-E4, shared by B512 and S615",
            "S615": "2,048 membership states, 512 updates, 32-row batches using frozen Seed 599 B index schedule",
            "B512": "2,048 membership states, 512 updates, 32-row batches",
            "augmentation": "none specified by either executor; each row is fed via kalah_v3 encode_state",
        },
        "counts": {
            "inventory": len(ledger),
            **dict(counts),
            "cleared_under_verified_contract": 0,
            "cleared_agreement": 0,
            "cleared_disagreement": 0,
            "by_classification_and_prior": {
                f"{a}:{b}": n for (a, b), n in sorted(by_class.items())
            },
            "by_contamination_category": dict(
                sorted(by_contamination_category.items())
            ),
        },
        "counts_by_historical_cohort": {
            cohort: {
                f"{kind}:{prior}": count
                for (kind, prior), count in sorted(values.items())
            }
            for cohort, values in sorted(by_cohort.items())
        },
        "counts_by_source_game_or_opening_cluster": {
            cluster: {
                f"{kind}:{prior}": count
                for (kind, prior), count in sorted(values.items())
            }
            for cluster, values in sorted(by_source_cluster.items())
        },
        "decision_gate": {
            "numerical_minimum_64_and_16_each_class": False,
            "cleared_disagreement_states": 0,
            "source_cluster_independence": "not estimable for a cleared population; raw inventory is clustered by historical games/openings",
            "recommendation": "stop",
        },
        "source_relationship_status": {
            "unresolved_incompatible_historical_identifiers": len(ledger),
            "interpretation": "training source game/opening identifiers cannot be completely crosswalked to Seed 621 source_path/opening_index; this remains separate from exact-state identity",
        },
        "raw_unresolved_source_cluster_summary": {
            "distinct_source_game_opening_clusters": len(unresolved_cluster_sizes),
            "effective_number_inverse_simpson": float(
                1.0
                / sum(
                    (size / sum(unresolved_cluster_sizes.values())) ** 2
                    for size in unresolved_cluster_sizes.values()
                )
            ),
            "largest_cluster_states": max(unresolved_cluster_sizes.values()),
            "interpretation": "descriptive only; clusters are not an eligible sample because contamination is unresolved",
        },
        "provenance_limitations": [
            "P-E4 inherited training exposure lacks an identity-complete historical training ledger.",
            "#580/#585 source archives authenticate provenance for candidate membership, but archive presence is not evidence that a row was consumed; this audit follows registered membership and observed minibatch traces.",
            "The historical training executors consume membership rows directly and do not implement symmetry or state augmentation. The audit checks the exact canonical identity and encoded-input float32 identity under kalah_v3.",
            "Source-game/source-opening similarity remains a separate risk and does not itself imply identical-state contamination.",
        ],
    }
    return ledger, report


def write() -> dict[str, Any]:
    ledger, report = audit()
    OUT.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical(row) + b"\n" for row in ledger)
    report["ledger_sha256"] = sha(payload)
    (OUT / "state-classification-ledger.jsonl").write_bytes(payload)
    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


if __name__ == "__main__":
    result = write()
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "counts": result["counts"],
                "ledger_sha256": result["ledger_sha256"],
                "decision_gate": result["decision_gate"],
                "historical_evidence_modified": result["historical_evidence_modified"],
            },
            indent=2,
            sort_keys=True,
        )
    )
