"""Independent verifier for the read-only Seed 622 provenance ledger."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.self_play import encode_state

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed622-training-provenance-audit"
S599 = ROOT / "docs/data/seed599-replay-coverage"
S615 = ROOT / "docs/data/seed615-source-opening-diversity"
EXPECTED = {
    "inventory": "332792e07accb8856f6cd5c0b65269dcfe55b392c93560369b7b685c2f86ad29",
    "b_membership": "95ecc180e28945702e08815de7a5210e1bce036afdaf76d194bbd774589a4516",
    "s_membership": "5ed236266ed3804a9c0e9f75b7e10b7c81ea6e290f8c72d3cea06dd225a15010",
    "holdout": "4e74d57244ab55b69751c38e4097860c25e541a8db60a2109f2ebd64c3eb81a8",
    "b_trace": "9f22328080756aa0fa2fa066164d36f935643938363059e0d0c711eb6f8f8955",
    "s_trace": "8ee2252f6831ea0589a19dff2c1ebc79a0f3c3447764feb9c53086ba599a8b01",
    "b_receipt": "3d81b00c438a678a2e38b9d58dcadc22f4091c0cb8e3ccbb755e1afe19ca41de",
    "b_training_verification": "57fd2623874a0064814b71847763f52ad890cc1833d8daffdaa222786673da46",
    "s_receipt": "0dc4d04a6be2b638abf77bb959e93d82c60355f3f9a449a49d0fef0a8bbb81f3",
    "parent": "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
    "seed599_preparation": "11c580d2ebb8d4a8fc67b12f7d1f8e6d8bcbffa0d110e6549bf077d259ff751d",
    "b_freeze": "bd141778461d7046b067966d7fb87188e5b0a2630d82655a321059b4ce17cad9",
    "s_freeze": "6ac9d4b4e52a6f4789ce406d9c88badd82e9c8ab7cf4b4d4d8e9c8676cd9910d",
    "b_checkpoint": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
    "s_checkpoint": "cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780",
}
SEED621_SOURCE_HASHES = {
    "docs/data/seed556-exploratory-arena/screen-0005/verification-receipt.json": "a318933da72b8ed23cb5487fbbd8e57edd0d1d407ba9bc1f8485b0736fcdfbe7",
    "docs/data/seed557-retrospective-diagnostic/cohort.json": "f62539cc3e1b1843a2bd9296453d51b895a1c5e693bd9edc3b3291f883398243",
    "docs/data/seed557-retrospective-diagnostic/execution-freeze.json": "b0892a24872012486ff164d291913e1799737632a816a40c2547210a458bf36d",
    "docs/data/seed557-retrospective-diagnostic/oracle-identity-verification.json": "45f93ce95bf2bc0c58b8a44bdf072324eddfb2c943dd9678e6895613a5a4f60d",
    "docs/data/seed557-retrospective-diagnostic/oracle-records.jsonl": "75453e744a86b9dc8471628e0d133fa7d478550af9c8089ed8661426436a8480",
    "docs/data/seed559-fixed-position-feasibility/run-0003/cohort-manifest.json": "f2e03fb9491474056e7a7cb3ace225cd2310f9069c9e48600d45039ed63158fb",
    "docs/data/seed559-fixed-position-feasibility/run-0003/execution-freeze.json": "3b1ab386c33534130c7e15df024f21ef75359ccd00d453e4c6aea104c8181927",
    "docs/data/seed559-fixed-position-feasibility/run-0003/oracle-records.jsonl": "83bec1bb348cbc6d1183112363f27d0612ee204a48d14ee5dbb7530e9b5fb8c2",
    "docs/data/seed615-source-opening-diversity/evaluation-preparation/execution-freeze-v4.json": "ba30f5e3efadf3358db9f03c7b4eba4f69b23e2f95dcb4089723a1fb9664f32b",
    "docs/data/seed615-source-opening-diversity/training-run-S615/training-integrity-report.json": "c383411a9ce9b896ef49431e502e0047b26de0e8600042d8b2873ae07c796266",
    "docs/data/seed615-source-opening-diversity/training-run-S615/training-receipt.json": "0dc4d04a6be2b638abf77bb959e93d82c60355f3f9a449a49d0fef0a8bbb81f3",
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def state_ids(state: dict[str, Any]) -> tuple[str, str]:
    canonical_id = digest(canonical(state))
    vector = np.asarray(encode_state(state, input_encoding="kalah_v3"), dtype="<f4")
    return canonical_id, digest(vector.tobytes())


def validate_state_identity(
    state: dict[str, Any], canonical_id: Any, float_id: Any
) -> None:
    if state_ids(state) != (canonical_id, float_id):
        raise ValueError("state_identity_mismatch")


def validate_source_provenance(
    published: dict[str, Any], source: dict[str, Any]
) -> None:
    if (
        published.get("cohorts") != source.get("cohorts")
        or published.get("source_clusters") != source.get("source_clusters")
        or published.get("prior_class")
        != ("agreement" if source.get("top_prior_agreement") else "disagreement")
    ):
        raise ValueError("corrupted_source_provenance_or_prior_class")


def validate_complete_consumption(indexes: set[int], membership_size: int) -> None:
    if indexes != set(range(membership_size)):
        raise ValueError("unconsumed_registered_membership_rows")


def read_rows(path: Path, *, jsonl: bool = False) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    return (
        [json.loads(row) for row in raw.splitlines() if row]
        if jsonl
        else json.loads(raw)
    )


def authenticate_seed621_sources(rows: list[dict[str, Any]]) -> None:
    cached: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    for row in rows:
        for source in row["source_clusters"]:
            relative = source["source_path"]
            if relative not in cached:
                raw = (ROOT / relative).read_bytes()
                parsed = [json.loads(line) for line in raw.splitlines() if line]
                cached[relative] = (digest(raw), parsed)
            batch_hash, records = cached[relative]
            index = int(source["source_line"]) - 1
            if batch_hash != source["source_batch_sha256"] or not 0 <= index < len(
                records
            ):
                raise ValueError("Seed621_source_batch_or_line_mismatch")
            record = records[index]
            if (
                digest(canonical(record)) != source["source_record_sha256"]
                or Path(relative).name
                != f"games-{int(source['opening_index']):04d}.jsonl"
            ):
                raise ValueError("Seed621_source_record_provenance_mismatch")


def _consumed(
    directory: Path,
    members: list[dict],
    schedule: list[list[int]],
    expected_trace: str,
    expected_receipt: str,
    checkpoint: str,
    membership_hash: str,
    freeze_hash: str,
) -> set[tuple[str, str]]:
    trace_path = directory / "training-trace.jsonl"
    receipt_path = directory / "training-receipt.json"
    trace_raw, receipt_raw = trace_path.read_bytes(), receipt_path.read_bytes()
    if digest(trace_raw) != expected_trace or digest(receipt_raw) != expected_receipt:
        raise ValueError("training_artifact_digest_mismatch")
    if (
        directory.name == "training-run-B"
        and digest((directory / "independent-verification.json").read_bytes())
        != EXPECTED["b_training_verification"]
    ):
        raise ValueError("B512_independent_verification_hash_mismatch")
    receipt = json.loads(receipt_raw)
    if receipt.get("updates") != 512 or receipt.get("checkpoint_sha256") != checkpoint:
        raise ValueError("training_receipt_checkpoint_or_updates_mismatch")
    if receipt.get("trace_sha256") != digest(trace_raw):
        raise ValueError("training_receipt_trace_mismatch")
    records = [json.loads(line) for line in trace_raw.splitlines() if line]
    if len(records) != 512 or len(schedule) != 512:
        raise ValueError("training_trace_or_schedule_incomplete")
    indexes: set[int] = set()
    for update, (record, batch) in enumerate(zip(records, schedule, strict=True), 1):
        if record.get("update") != update or record.get("indexes") != batch:
            raise ValueError(f"training_schedule_trace_mismatch:{update}")
        if len(batch) != 32 or any(
            type(index) is not int or not 0 <= index < len(members) for index in batch
        ):
            raise ValueError("training_batch_index_invalid")
        indexes.update(batch)
    if (
        receipt.get("membership_sha256") != membership_hash
        or receipt.get("parent_checkpoint_sha256") != EXPECTED["parent"]
        or receipt.get("batch_schedule_sha256") != digest(canonical(schedule))
        or receipt.get("execution_freeze_sha256") != freeze_hash
    ):
        raise ValueError("training_receipt_input_binding_mismatch")
    validate_complete_consumption(indexes, len(members))
    return {state_ids(members[index]["canonical_state"]) for index in indexes}


def verify() -> dict[str, Any]:
    inventory_path = (
        ROOT / "docs/data/seed621-oracle-cohort-audit/eligible-state-inventory.jsonl"
    )
    source_raw = inventory_path.read_bytes()
    if digest(source_raw) != EXPECTED["inventory"]:
        raise ValueError("historical_seed621_inventory_changed")
    preparation_raw = (S599 / "preparation.json").read_bytes()
    if digest(preparation_raw) != EXPECTED["seed599_preparation"]:
        raise ValueError("seed599_preparation_hash_mismatch")
    preparation = json.loads(preparation_raw)
    source_artifacts = preparation["source_verification"]["source_artifacts"]
    for relative, expected_hash in source_artifacts.items():
        if digest((ROOT / relative).read_bytes()) != expected_hash:
            raise ValueError(f"seed580_585_source_artifact_hash_mismatch:{relative}")
    inventory = [json.loads(line) for line in source_raw.splitlines() if line]
    authenticate_seed621_sources(inventory)
    source_report = json.loads(
        (ROOT / "docs/data/seed621-oracle-cohort-audit/report.json").read_bytes()
    )
    if source_report.get("source_artifacts_sha256") != SEED621_SOURCE_HASHES:
        raise ValueError("historical_source_hash_manifest_changed")
    for relative, expected_hash in SEED621_SOURCE_HASHES.items():
        if digest((ROOT / relative).read_bytes()) != expected_hash:
            raise ValueError(f"historical_source_artifact_changed:{relative}")
    b_rows = read_rows(S599 / "treatment-membership.json")
    s_rows = read_rows(S615 / "candidate-membership.jsonl", jsonl=True)
    holdout = read_rows(S599 / "heldout-membership.json")
    if (
        digest(canonical(b_rows)) != EXPECTED["b_membership"]
        or digest(canonical(s_rows)) != EXPECTED["s_membership"]
        or digest(canonical(holdout)) != EXPECTED["holdout"]
    ):
        raise ValueError("training_or_holdout_membership_changed")
    for name, members in (("B512", b_rows), ("S615", s_rows), ("holdout", holdout)):
        canonical_seen: set[str] = set()
        float_seen: set[str] = set()
        for member in members:
            try:
                validate_state_identity(
                    member["canonical_state"],
                    member.get("canonical_identity"),
                    member.get("float32_identity"),
                )
            except ValueError as error:
                raise ValueError(f"membership_identity_mismatch:{name}") from error
            canonical_id, float_id = state_ids(member["canonical_state"])
            if name != "holdout" and (
                canonical_id in canonical_seen or float_id in float_seen
            ):
                raise ValueError(f"training_membership_collision:{name}")
            canonical_seen.add(canonical_id)
            float_seen.add(float_id)
    # The same registered B schedule addresses sorted B512 and S615 manifests.
    schedule = json.loads((S599 / "batch-schedule-B.json").read_bytes())
    freezes = {
        "B512": (S599 / "training-execution-freeze.json", EXPECTED["b_freeze"]),
        "S615": (S615 / "training-execution-freeze-v3.json", EXPECTED["s_freeze"]),
    }
    for label, (path, expected_hash) in freezes.items():
        raw = path.read_bytes()
        if digest(raw) != expected_hash:
            raise ValueError(f"training_freeze_digest_mismatch:{label}")
        freeze = json.loads(raw)
        declared_sources = freeze.get("source_sha256", freeze.get("source_hashes", {}))
        for relative, source_hash in declared_sources.items():
            if digest((ROOT / relative).read_bytes()) != source_hash:
                raise ValueError(f"training_executor_source_changed:{label}:{relative}")
    b_consumed = _consumed(
        S599 / "training-run-B",
        b_rows,
        schedule,
        EXPECTED["b_trace"],
        EXPECTED["b_receipt"],
        EXPECTED["b_checkpoint"],
        EXPECTED["b_membership"],
        EXPECTED["b_freeze"],
    )
    s_consumed = _consumed(
        S615 / "training-run-S615",
        s_rows,
        schedule,
        EXPECTED["s_trace"],
        EXPECTED["s_receipt"],
        EXPECTED["s_checkpoint"],
        EXPECTED["s_membership"],
        EXPECTED["s_freeze"],
    )
    parent = ROOT / "docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4/model.npz"
    if digest(parent.read_bytes()) != EXPECTED["parent"]:
        raise ValueError("parent_checkpoint_changed")
    expected_lineage = {}
    for (
        name,
        directory,
        members,
        trace_hash,
        receipt_hash,
        checkpoint_hash,
        freeze_hash,
        membership_hash,
    ) in (
        (
            "B512",
            S599 / "training-run-B",
            b_rows,
            EXPECTED["b_trace"],
            EXPECTED["b_receipt"],
            EXPECTED["b_checkpoint"],
            EXPECTED["b_freeze"],
            EXPECTED["b_membership"],
        ),
        (
            "S615",
            S615 / "training-run-S615",
            s_rows,
            EXPECTED["s_trace"],
            EXPECTED["s_receipt"],
            EXPECTED["s_checkpoint"],
            EXPECTED["s_freeze"],
            EXPECTED["s_membership"],
        ),
    ):
        expected_lineage[name] = {
            "updates": 512,
            "batch_size": 32,
            "unique_membership_rows_consumed": len(members),
            "membership_rows": len(members),
            "membership_sha256": membership_hash,
            "batch_schedule_sha256": digest(canonical(schedule)),
            "execution_freeze_sha256": freeze_hash,
            "parent_checkpoint_sha256": EXPECTED["parent"],
            "checkpoint_sha256": checkpoint_hash,
            "trace_sha256": trace_hash,
            "receipt_sha256": receipt_hash,
        }
    canonical_sets = {
        "B512": {row["canonical_identity"] for row in b_rows},
        "S615": {row["canonical_identity"] for row in s_rows},
        "holdout": {row["canonical_identity"] for row in holdout},
    }
    float_sets = {
        "B512": {row["float32_identity"] for row in b_rows},
        "S615": {row["float32_identity"] for row in s_rows},
        "holdout": {row["float32_identity"] for row in holdout},
    }
    # Training row sets are taken from trace-verified consumed indexes, not archives.
    consumed = {"B512": b_consumed, "S615": s_consumed}
    published = read_rows(OUT / "state-classification-ledger.jsonl", jsonl=True)
    report = json.loads((OUT / "report.json").read_bytes())
    payload = (OUT / "state-classification-ledger.jsonl").read_bytes()
    if digest(payload) != report.get("ledger_sha256") or len(published) != len(
        inventory
    ):
        raise ValueError("ledger_digest_or_length_mismatch")
    if report.get("training_lineage") != expected_lineage:
        raise ValueError("training_lineage_report_mismatch")
    if report.get("source_corpus_authentication") != {
        "preparation_sha256": EXPECTED["seed599_preparation"],
        "authenticated_source_artifacts": len(source_artifacts),
        "sources": ["#580", "#585"],
    }:
        raise ValueError("source_corpus_report_mismatch")
    if report.get("decision_gate") != {
        "numerical_minimum_64_and_16_each_class": False,
        "cleared_disagreement_states": 0,
        "source_cluster_independence": "not estimable for a cleared population; raw inventory is clustered by historical games/openings",
        "recommendation": "stop",
    }:
        raise ValueError("decision_gate_report_mismatch")
    if [row.get("state_id") for row in published] != [
        row.get("state_id") for row in inventory
    ]:
        raise ValueError("omitted_substituted_or_reordered_state")
    classes = Counter()
    by_class_prior = Counter()
    by_contamination_category = Counter()
    by_cohort: dict[str, Counter] = {}
    by_source_cluster: dict[str, Counter] = {}
    unresolved_cluster_sizes: Counter[str] = Counter()
    for source, row in zip(inventory, published, strict=True):
        state = source["state"]
        cid, f32 = state_ids(state)
        if (
            cid != source["state_id"]
            or row.get("canonical_identity") != cid
            or row.get("production_float32_identity") != f32
        ):
            raise ValueError("inventory_identity_mismatch")
        validate_source_provenance(row, source)
        hit = {
            "B512": cid in canonical_sets["B512"]
            or f32 in float_sets["B512"]
            or (cid, f32) in consumed["B512"],
            "S615": cid in canonical_sets["S615"]
            or f32 in float_sets["S615"]
            or (cid, f32) in consumed["S615"],
            "protected_holdout": cid in canonical_sets["holdout"]
            or f32 in float_sets["holdout"],
        }
        expected_class = (
            "excluded_confirmed_exposure"
            if any(hit.values())
            else "unresolved_insufficient_provenance"
        )
        if row.get("classification") != expected_class:
            raise ValueError("incorrectly_cleared_or_excluded_state")
        expected_categories = [
            label
            for population, label in (
                ("B512", "B512_training_identity_overlap"),
                ("S615", "S615_training_identity_overlap"),
                ("holdout", "protected_holdout_overlap"),
            )
            if cid in canonical_sets[population] or f32 in float_sets[population]
        ]
        if not expected_categories:
            expected_categories = ["no_direct_membership_overlap"]
        if row.get("contamination_categories") != expected_categories:
            raise ValueError("contamination_category_mismatch")
        by_contamination_category.update(expected_categories)
        expected_prior = (
            "agreement" if source["top_prior_agreement"] else "disagreement"
        )
        expected_exposure = {
            "B512_canonical": cid in canonical_sets["B512"],
            "B512_float32": f32 in float_sets["B512"],
            "S615_canonical": cid in canonical_sets["S615"],
            "S615_float32": f32 in float_sets["S615"],
            "protected_holdout_canonical": cid in canonical_sets["holdout"],
            "protected_holdout_float32": f32 in float_sets["holdout"],
        }
        for key, value in expected_exposure.items():
            if row.get("exposure", {}).get(key) != value:
                raise ValueError("exposure_evidence_mismatch")
        classes[expected_class] += 1
        by_class_prior[(expected_class, expected_prior)] += 1
        for cohort in source["cohorts"]:
            by_cohort.setdefault(cohort, Counter())[
                (expected_class, expected_prior)
            ] += 1
        for cluster in source["source_clusters"]:
            cluster_name = f"{cluster.get('source_path')}#opening_index={cluster.get('opening_index')}"
            by_source_cluster.setdefault(cluster_name, Counter())[
                (expected_class, expected_prior)
            ] += 1
            if expected_class == "unresolved_insufficient_provenance":
                unresolved_cluster_sizes[cluster_name] += 1
    if (
        report.get("counts", {}).get("inventory") != len(inventory)
        or report["counts"].get("excluded_confirmed_exposure")
        != classes["excluded_confirmed_exposure"]
        or report["counts"].get("unresolved_insufficient_provenance")
        != classes["unresolved_insufficient_provenance"]
    ):
        raise ValueError("aggregate_count_mismatch")
    if classes["unresolved_insufficient_provenance"] != 764:
        raise ValueError("unexpected_cleared_state_count")
    expected_cohort_counts = {
        cohort: {
            f"{kind}:{prior}": count for (kind, prior), count in sorted(values.items())
        }
        for cohort, values in sorted(by_cohort.items())
    }
    if report.get("counts_by_historical_cohort") != expected_cohort_counts:
        raise ValueError("historical_cohort_count_mismatch")
    expected_class_prior = {
        f"{kind}:{prior}": count
        for (kind, prior), count in sorted(by_class_prior.items())
    }
    if (
        report.get("counts", {}).get("by_classification_and_prior")
        != expected_class_prior
    ):
        raise ValueError("classification_prior_count_mismatch")
    if (
        report["counts"].get("cleared_under_verified_contract") != 0
        or report["counts"].get("cleared_agreement") != 0
        or report["counts"].get("cleared_disagreement") != 0
    ):
        raise ValueError("cleared_state_count_mismatch")
    if report["counts"].get("by_contamination_category") != dict(
        sorted(by_contamination_category.items())
    ):
        raise ValueError("contamination_category_count_mismatch")
    if report.get("source_relationship_status") != {
        "unresolved_incompatible_historical_identifiers": len(inventory),
        "interpretation": "training source game/opening identifiers cannot be completely crosswalked to Seed 621 source_path/opening_index; this remains separate from exact-state identity",
    }:
        raise ValueError("source_relationship_report_mismatch")
    expected_cluster_counts = {
        cluster: {
            f"{kind}:{prior}": count for (kind, prior), count in sorted(values.items())
        }
        for cluster, values in sorted(by_source_cluster.items())
    }
    if (
        report.get("counts_by_source_game_or_opening_cluster")
        != expected_cluster_counts
    ):
        raise ValueError("source_cluster_count_mismatch")
    total_unresolved = sum(unresolved_cluster_sizes.values())
    expected_cluster_summary = {
        "distinct_source_game_opening_clusters": len(unresolved_cluster_sizes),
        "effective_number_inverse_simpson": 1.0
        / sum(
            (size / total_unresolved) ** 2 for size in unresolved_cluster_sizes.values()
        ),
        "largest_cluster_states": max(unresolved_cluster_sizes.values()),
        "interpretation": "descriptive only; clusters are not an eligible sample because contamination is unresolved",
    }
    if report.get("raw_unresolved_source_cluster_summary") != expected_cluster_summary:
        raise ValueError("effective_cluster_summary_mismatch")
    verification_path = OUT / "verification-report.json"
    if verification_path.is_file():
        published_verification = json.loads(verification_path.read_bytes())
        expected_verification = {
            "cleared_under_verified_contract": 0,
            "excluded_confirmed_exposure": classes["excluded_confirmed_exposure"],
            "historical_seed621_inventory_sha256": EXPECTED["inventory"],
            "ledger_sha256": digest(payload),
            "recommendation": "stop",
            "seed599_authenticated_source_artifacts": len(source_artifacts),
            "seed599_preparation_sha256": EXPECTED["seed599_preparation"],
            "states": len(inventory),
            "status": "verified",
            "training_or_search_started": False,
            "unresolved_insufficient_provenance": classes[
                "unresolved_insufficient_provenance"
            ],
        }
        if published_verification != expected_verification:
            raise ValueError("verification_report_mismatch")
    return {
        "status": "verified",
        "states": len(inventory),
        "excluded_confirmed_exposure": classes["excluded_confirmed_exposure"],
        "unresolved_insufficient_provenance": classes[
            "unresolved_insufficient_provenance"
        ],
        "cleared_under_verified_contract": 0,
        "recommendation": "stop",
        "training_or_search_started": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True, indent=2))
