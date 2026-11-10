"""Authenticate and state-index amended seed542 and matched seed549 corpora."""

from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any

from ml.alphazero_lite.self_play import encode_state

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
S542 = DATA / "seed542-selfplay-4096"
S549 = DATA / "seed549-matched-selfplay"
S550 = DATA / "seed549-controlled-training"
S554 = DATA / "seed554-fixed-state-teacher-budget"
OUT = DATA / "seed542-549-exclusion-inventory"
EXPECTED_542_VERIFIED_ARTIFACTS = {
    "amended-accounting.json": "89140844f831ec31b5bcf994d69c677d70f757abb3e18907d07b79817da8d807",
    "continuation-amendment.json": "098a7297d7c73cc3ae8701ea285f0dff26917d8d695310edae986bda17be42e4",
    "development-membership.json": "feeed97c7666a69603620e4e59575fdf02aa206e9e570cde937aa360bd37505b",
    "execution-service.json": "b2fabd500ffbfea986b44506e5dd83bb9b54ea9be57c7af5473e7397b59e0af9",
    "source-manifest.json": "63ffa566b929de4bec6fd445768e5161a53807fc2dcc1b170c8240a28be8f616",
    "training-protocol.json": "bd708bcf3e6fc47b03988f98614c00d4e521374d5cd72cb0d2e33165e6d802f3",
}


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def authenticate(path: Path, expected: str, label: str) -> None:
    actual = digest_file(path) if path.is_file() else "missing"
    if actual != expected:
        raise ValueError(
            f"source_digest_mismatch:{label}:expected={expected}:observed={actual}"
        )


def float_bytes(values: list[float]) -> bytes:
    return b"".join(struct.pack("<f", float(value)) for value in values)


def decode_kalah_v3(values: list[float]) -> dict[str, Any]:
    if len(values) != 27:
        raise ValueError(f"invalid_kalah_v3_state_length:{len(values)}")
    base = [
        struct.unpack("<f", struct.pack("<f", float(value)))[0] for value in values[:15]
    ]
    state = {
        "player_pits": [int(round(value * 48)) for value in base[:6]],
        "opponent_pits": [int(round(value * 48)) for value in base[6:12]],
        "player_store": int(round(base[12] * 48)),
        "opponent_store": int(round(base[13] * 48)),
        "current_player": int(round(base[14])),
    }
    if encode_state(state, input_encoding="kalah_v3") != values:
        # Production inputs are float32. Compare their exact float32 wire values,
        # not Python's JSON decimal representation.
        if float_bytes(encode_state(state, input_encoding="kalah_v3")) != float_bytes(
            values
        ):
            raise ValueError("authenticated_encoded_state_not_reversible")
    return state


def identity_record(state_vector: list[float]) -> tuple[str, str, dict[str, Any]]:
    state = decode_kalah_v3(state_vector)
    canonical = digest_bytes(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    )
    float32 = digest_bytes(float_bytes(state_vector))
    return canonical, float32, state


def legacy_canonical_sha256(state: dict[str, Any]) -> str:
    """Historical seed542/549 occurrence ledgers hash the canonical value tuple."""
    values = (
        state["player_pits"]
        + state["opponent_pits"]
        + [state["player_store"], state["opponent_store"], state["current_player"]]
    )
    return digest_bytes(json.dumps(values, separators=(",", ":")).encode())


def jsonl_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    return [
        (n, json.loads(line))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line
    ]


def add_population(
    indexes: dict[str, dict[str, set[str]]],
    records: list[dict[str, Any]],
    population: str,
    source_path: str,
    line: int,
    vector: list[float],
) -> tuple[str, str]:
    canonical, float32, _ = identity_record(vector)
    indexes.setdefault(population, {"canonical": set(), "kalah_v3_float32": set()})
    indexes[population]["canonical"].add(canonical)
    indexes[population]["kalah_v3_float32"].add(float32)
    records.append(
        {
            "population": population,
            "source_path": source_path,
            "source_line": line,
            "canonical_state_sha256": canonical,
            "kalah_v3_float32_sha256": float32,
        }
    )
    return canonical, float32


def census() -> tuple[
    dict[str, Any], list[dict[str, Any]], dict[str, dict[str, set[str]]]
]:
    indexes: dict[str, dict[str, set[str]]] = {}
    records: list[dict[str, Any]] = []
    binding_manifest: dict[str, Any] = {
        "seed542": {},
        "seed549": {},
        "derivative_reuse": {},
    }

    # #543's final verifier pins the amended manifest and every membership input.
    verification = json.loads((S542 / "amended-verification.json").read_text())
    pinned_542 = verification["independent_read_only_verification"][
        "verified_artifact_sha256"
    ]
    for filename, expected in EXPECTED_542_VERIFIED_ARTIFACTS.items():
        authenticate(
            S542 / filename, expected, f"docs/data/seed542-selfplay-4096/{filename}"
        )
        if pinned_542.get(filename) != expected:
            raise ValueError(f"seed542_verification_binding_disagreement:{filename}")
    manifest = json.loads((S542 / "source-manifest.json").read_text())
    dev_membership = json.loads((S542 / "development-membership.json").read_text())
    source_specs = manifest["sources"]
    raw_dev_rows: set[tuple[str, int]] = set()
    training_rows: set[tuple[str, int]] = set()
    decoded_membership: dict[tuple[str, int], dict[str, Any]] = {}
    for spec in source_specs:
        path = ROOT / spec["path"]
        authenticate(path, spec["sha256"], spec["path"])
        rows = jsonl_rows(path)
        if spec["role"] not in ("training", "development"):
            continue
        population = (
            "seed542_training"
            if spec["role"] == "training"
            else "seed542_raw_development"
        )
        target = training_rows if spec["role"] == "training" else raw_dev_rows
        for line, row in rows:
            target.add((spec["path"], line))
            canonical, float32 = add_population(
                indexes, records, population, spec["path"], line, row["state"]
            )
            decoded_membership[(spec["path"], line)] = {
                "canonical_sha256": canonical,
                "production_float32_input_sha256": float32,
                "legacy_canonical_sha256": legacy_canonical_sha256(
                    decode_kalah_v3(row["state"])
                ),
            }
    retained_rows: set[tuple[str, int]] = set()
    membership_rows: set[tuple[str, int]] = set()
    for row in dev_membership["membership"]:
        source, raw_line = row["row_id"].rsplit(":", 1)
        key = (source, int(raw_line))
        if key in membership_rows:
            raise ValueError(
                f"seed542_duplicate_development_membership:{row['row_id']}"
            )
        membership_rows.add(key)
        if key not in raw_dev_rows:
            raise ValueError(
                f"seed542_membership_not_in_raw_development:{row['row_id']}"
            )
        if (
            decoded_membership[key]["legacy_canonical_sha256"]
            != row["canonical_sha256"]
        ):
            raise ValueError(
                f"seed542_membership_canonical_identity_mismatch:{row['row_id']}"
            )
        if (
            decoded_membership[key]["production_float32_input_sha256"]
            != row["production_float32_input_sha256"]
        ):
            raise ValueError(
                f"seed542_membership_float32_identity_mismatch:{row['row_id']}"
            )
        if row["included"]:
            retained_rows.add(key)
    for source, line in sorted(retained_rows):
        path = ROOT / source
        row = json.loads(path.read_text(encoding="utf-8").splitlines()[line - 1])
        add_population(
            indexes, records, "seed542_retained_development", source, line, row["state"]
        )
    if (len(training_rows), len(raw_dev_rows), len(retained_rows)) != (
        109504,
        36844,
        15359,
    ):
        raise ValueError("seed542_population_occurrence_counts_mismatch")
    if len(training_rows) + len(raw_dev_rows) != 146348:
        raise ValueError("seed542_usable_rows_do_not_reconcile")
    if membership_rows != raw_dev_rows:
        raise ValueError("seed542_development_membership_not_complete")
    amended_accounting = json.loads((S542 / "amended-accounting.json").read_text())
    training_account = amended_accounting["training_corpus"]
    development_account = amended_accounting["development_source"]
    if (
        training_account.get("rows") != 109504
        or training_account.get("loader_accepted_rows") != 109504
        or training_account.get("loader_rejected_rows") != 0
        or development_account.get("rows") != 36844
        or development_account.get("loader_accepted_rows") != 36844
        or development_account.get("loader_rejected_rows") != 0
    ):
        raise ValueError("seed542_training_accounting_mismatch")
    # A recovered derivative represents 51 complete games and is already present
    # once in source-manifest; the incomplete trajectory is not a training source.
    recovery_spec = next(
        spec
        for spec in source_specs
        if spec["path"].endswith("recovered-seed542001-complete.jsonl")
    )
    supplement_spec = next(
        spec for spec in source_specs if spec["path"].endswith("shard-542064.jsonl")
    )
    recovery_rows = len(jsonl_rows(S542 / "recovered-seed542001-complete.jsonl"))
    supplement_rows = len(jsonl_rows(S542 / "shard-542064.jsonl"))
    if (
        recovery_rows != 1768
        or recovery_spec.get("games") != 51
        or supplement_spec.get("games") != 13
    ):
        raise ValueError(
            f"seed542_recovery_or_supplement_mismatch:recovery={recovery_rows}/{recovery_spec.get('games')}:supplement={supplement_rows}/{supplement_spec.get('games')}"
        )
    binding_manifest["seed542"] = {
        "manifest_sha256": digest_file(S542 / "source-manifest.json"),
        "membership_sha256": digest_file(S542 / "development-membership.json"),
        "usable_games": 4096,
        "loader_accepted_rows": 146348,
        "recovered_games": 51,
        "recovered_rows": recovery_rows,
        "supplement_games": 13,
        "supplement_rows": supplement_rows,
        "training_occurrences": len(training_rows),
        "raw_development_occurrences": len(raw_dev_rows),
        "retained_development_occurrences": len(retained_rows),
        "incomplete_and_unstarted_slots_included": False,
    }

    # Authenticate seed549 source shards from each arm-specific source manifest.
    dev_by_row: dict[tuple[str, int], dict[str, Any]] = {}
    training_pairs: dict[str, Counter[tuple[str, str]]] = {
        "control": Counter(),
        "treatment": Counter(),
    }
    shared_membership = []
    for arm in ("control", "treatment"):
        src_manifest_path = S549 / f"{arm}-source-manifest.json"
        src_manifest = json.loads(src_manifest_path.read_text())
        arm_train = set()
        arm_raw_dev = set()
        for shard in src_manifest["shards"]:
            path = ROOT / shard["path"]
            authenticate(path, shard["sha256"], shard["path"])
            part_population = (
                f"seed549_{arm}_training"
                if shard["partition"] == "training"
                else f"seed549_{arm}_raw_development"
            )
            target = arm_train if shard["partition"] == "training" else arm_raw_dev
            for line, row in jsonl_rows(path):
                target.add((shard["path"], line))
                ids = add_population(
                    indexes, records, part_population, shard["path"], line, row["state"]
                )
                if shard["partition"] == "training":
                    training_pairs[arm][
                        (legacy_canonical_sha256(decode_kalah_v3(row["state"])), ids[1])
                    ] += 1
                if shard["partition"] == "development":
                    state = decode_kalah_v3(row["state"])
                    dev_by_row[(shard["path"], line)] = {
                        "canonical_state_sha256": ids[0],
                        "production_float32_sha256": ids[1],
                        "legacy_canonical_sha256": legacy_canonical_sha256(state),
                    }
        if (
            len(arm_train) != src_manifest["training_rows"]
            or len(arm_raw_dev) != src_manifest["development_rows"]
        ):
            raise ValueError(f"seed549_{arm}_manifest_occurrence_count_mismatch")
        binding_manifest["seed549"][arm] = {
            "source_manifest_sha256": digest_file(src_manifest_path),
            "training_occurrences": len(arm_train),
            "raw_development_occurrences": len(arm_raw_dev),
            "training_identity_manifest_sha256": src_manifest[
                "training_identity_sha256"
            ],
            "development_identity_manifest_sha256": src_manifest[
                "development_identity_sha256"
            ],
        }
    for line in (S549 / "shared-development-membership.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = (row["source"]["path"], int(row["source"]["row_number"]) + 1)
        expected = dev_by_row.get(key)
        if expected is None:
            raise ValueError(
                f"seed549_shared_membership_source_missing:{key[0]}:{key[1]}"
            )
        if (
            row["canonical_state_sha256"] != expected["legacy_canonical_sha256"]
            or row["production_float32_sha256"] != expected["production_float32_sha256"]
        ):
            raise ValueError(
                f"seed549_shared_membership_identity_mismatch:{key[0]}:{key[1]}"
            )
        shared_membership.append(row)
    expected_shared_rows = set(dev_by_row)
    observed_shared_rows = {
        (row["source"]["path"], int(row["source"]["row_number"]) + 1)
        for row in shared_membership
    }
    if observed_shared_rows != expected_shared_rows or len(shared_membership) != len(
        observed_shared_rows
    ):
        raise ValueError("seed549_shared_development_membership_not_complete")
    if len(shared_membership) != 9317:
        raise ValueError(
            f"seed549_shared_development_occurrence_count_mismatch:{len(shared_membership)}"
        )
    retained_shared = [row for row in shared_membership if row["included"]]
    for row in retained_shared:
        path = row["source"]["path"]
        line = int(row["source"]["row_number"]) + 1
        shard_row = json.loads((ROOT / path).read_text().splitlines()[line - 1])
        add_population(
            indexes,
            records,
            "seed549_retained_shared_development",
            path,
            line,
            shard_row["state"],
        )
    binding_manifest["seed549"]["shared_membership_sha256"] = digest_file(
        S549 / "shared-development-membership.jsonl"
    )
    binding_manifest["seed549"]["raw_development_occurrences"] = len(shared_membership)
    binding_manifest["seed549"]["retained_shared_development_occurrences"] = len(
        retained_shared
    )

    # #550/#554 bind training input lists, not new state populations. Verify
    # identity pairs against the authenticated #549 training populations.
    registration550 = json.loads((S550 / "registration.json").read_text())
    input_hashes550 = registration550.get("input_sha256", {})
    shared_membership_sha = digest_file(S549 / "shared-development-membership.jsonl")
    if (
        registration550.get("development", {}).get("membership_sha256")
        != shared_membership_sha
    ):
        raise ValueError("seed550_shared_development_membership_binding_mismatch")
    for relative in (
        "docs/data/seed549-matched-selfplay/control-source-manifest.json",
        "docs/data/seed549-matched-selfplay/treatment-source-manifest.json",
        "docs/data/seed549-matched-selfplay/shared-development-membership.jsonl",
    ):
        expected_hash = input_hashes550.get(relative)
        if expected_hash is None:
            raise ValueError(f"seed550_source_binding_missing:{relative}")
        authenticate(ROOT / relative, expected_hash, relative)
    verified_550_lists = {}
    for arm in ("control", "treatment"):
        source_list_rel = (
            f"docs/data/seed549-matched-selfplay/{arm}-training-row-identities.jsonl"
        )
        expected_list_sha = input_hashes550.get(source_list_rel)
        if expected_list_sha is None:
            raise ValueError(
                f"seed550_bound_training_identity_missing:{source_list_rel}"
            )
        source_list = ROOT / source_list_rel
        authenticate(source_list, expected_list_sha, source_list_rel)
        rows = [json.loads(line) for line in source_list.read_text().splitlines()]
        listed_pairs = Counter(
            (row["canonical_state_sha256"], row["production_float32_sha256"])
            for row in rows
        )
        if listed_pairs != training_pairs[arm]:
            raise ValueError(f"seed550_training_membership_changed:{arm}")
        verified_550_lists[arm] = {
            "path": source_list_rel,
            "sha256": expected_list_sha,
            "occurrences": len(rows),
            "membership_matches_seed549": True,
        }
    binding554_path = S554 / "training-input-binding.json"
    binding554 = json.loads(binding554_path.read_text())
    auth554_path = S554 / "input-authentication.json"
    auth554 = json.loads(auth554_path.read_text())
    if auth554.get("development_membership_sha256") != shared_membership_sha:
        raise ValueError("seed554_shared_development_membership_binding_mismatch")
    products554 = auth554["receipt"]["products"]
    if auth554.get("occurrence_identity_sha256") != products554.get(
        "control-training-row-identities.jsonl"
    ):
        raise ValueError("seed554_occurrence_identity_binding_mismatch")
    for filename, expected_hash in products554.items():
        candidates = (S554 / filename, S549 / filename)
        matched = next(
            (
                path
                for path in candidates
                if path.is_file() and digest_file(path) == expected_hash
            ),
            None,
        )
        if matched is None:
            path = next((path for path in candidates if path.exists()), candidates[0])
            observed = digest_file(path) if path.is_file() else "missing"
            raise ValueError(
                f"source_digest_mismatch:{path.relative_to(ROOT)}:expected={expected_hash}:observed={observed}"
            )
    for number, root, evidence, raw in (
        (550, S550, S550 / "registration.json", registration550),
        (554, S554, binding554_path, binding554),
    ):
        binding_manifest["derivative_reuse"][str(number)] = {
            "binding_path": str(evidence.relative_to(ROOT)),
            "binding_sha256": digest_file(evidence),
            "rows": raw.get("rows", raw.get("training", {}).get("rows")),
        }
        if number == 550:
            binding_manifest["derivative_reuse"]["550"][
                "bound_training_state_lists"
            ] = verified_550_lists
            binding_manifest["derivative_reuse"]["550"]["membership_added"] = False
    if auth554.get("occurrences") != 13818 or binding554.get("rows") != 13818:
        raise ValueError("seed554_training_occurrence_binding_mismatch")
    for arm in ("control", "treatment"):
        list_path = S554 / f"{arm}-training-row-identities.jsonl"
        source_path = S549 / f"{arm}-training-row-identities.jsonl"
        expected = products554.get(list_path.name)
        if expected is None or not source_path.is_file():
            raise ValueError(f"seed554_bound_training_identity_missing:{arm}")
        authenticate(
            source_path,
            expected,
            f"docs/data/seed549-matched-selfplay/{source_path.name}",
        )
        listed = Counter(
            (row["canonical_state_sha256"], row["production_float32_sha256"])
            for row in map(json.loads, list_path.read_text().splitlines())
        )
        # #554's corrected treatment labels use the control state membership;
        # confirm that explicitly against the authenticated C training input.
        if listed != training_pairs["control"]:
            raise ValueError(f"seed554_training_membership_changed:{arm}")
    binding_manifest["derivative_reuse"]["554"]["bound_state_occurrences"] = auth554[
        "occurrences"
    ]
    binding_manifest["derivative_reuse"]["554"]["input_authentication_sha256"] = (
        digest_file(auth554_path)
    )
    binding_manifest["derivative_reuse"]["554"]["control_state_membership_reused"] = (
        True
    )
    binding_manifest["derivative_reuse"]["554"]["membership_added"] = False

    report = {
        "sources": binding_manifest,
        "counts": {
            name: {
                "occurrences": sum(1 for row in records if row["population"] == name),
                "canonical_distinct": len(pair["canonical"]),
                "kalah_v3_float32_distinct": len(pair["kalah_v3_float32"]),
            }
            for name, pair in sorted(indexes.items())
        },
    }
    return report, records, indexes


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report, records, _ = census()
    records.sort(
        key=lambda row: (row["population"], row["source_path"], row["source_line"])
    )
    index_bytes = (
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) for row in records
        )
        + "\n"
    ).encode()
    report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "identity-index.jsonl").write_bytes(index_bytes)
        (OUT / "count-report.json").write_bytes(report_bytes)
    else:
        for name, expected in (
            ("identity-index.jsonl", index_bytes),
            ("count-report.json", report_bytes),
        ):
            if not (OUT / name).is_file() or (OUT / name).read_bytes() != expected:
                raise ValueError(f"published_inventory_mismatch:{name}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
