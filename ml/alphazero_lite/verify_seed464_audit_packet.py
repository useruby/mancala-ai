"""Read-only, portable integrity and binding-chain verifier for seed464."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST = "packet-manifest.json"
REPORT = "docs/data/seed464-dependency-closure-audit/dependency-closure-report.json"
REGISTRATION = "docs/data/seed464-historical-value-removal/registration.json"
RECEIPT = "docs/data/seed464-historical-value-removal/receipt.json"
AUDIT_RECEIPT = (
    "docs/data/seed464-dependency-closure-audit/dependency-closure-receipt.json"
)
AUDIT_MARKDOWN = (
    "docs/data/seed464-dependency-closure-audit/dependency-closure-report.md"
)
CLASSIFICATION = "invalid_attempt_dependency_closure_incomplete"
ORIGINAL_REGISTRATION_SHA256 = (
    "0917d52281ba09d7384b8c85ecf3777ea8ab177c0b150b1c7bc8d1bbbcef8dfa"
)
ORIGINAL_RECEIPT_SHA256 = (
    "0f37fd96b593e1a5b2fbab7c1f0c7ed5c5c4486ce788afd898953b34f7afd92d"
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: bytes, label: str) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"invalid_json_object:{label}")
    return value


def _logical(path: str) -> str:
    """Normalize historical absolute derivatives to their logical final path."""
    normalized = path.replace("\\", "/")
    marker = "/.tmp/seed416-policy-target-softening/derivatives/A/"
    if marker in normalized:
        return "derivatives/A/" + normalized.split(marker, 1)[1]
    prefix = ".tmp/seed416-policy-target-softening/derivatives/A/"
    if normalized.startswith(prefix):
        return "derivatives/A/" + normalized.removeprefix(prefix)
    p = PurePosixPath(normalized)
    if p.is_absolute():
        raise ValueError(f"unrecognized_absolute_dependency:{path}")
    return p.as_posix()


def _hash_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _verify_member(packet: Path, member: dict[str, Any]) -> None:
    relative = PurePosixPath(member["portable_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe_packet_path:{relative}")
    path = packet.joinpath(*relative.parts)
    if not path.is_file():
        raise ValueError(f"missing_packet_member:{relative}")
    with path.open("rb") as stream:
        archive_hash = _hash_stream(stream)
    if archive_hash != member["archived_sha256"]:
        raise ValueError(f"archived_member_hash_mismatch:{relative}")
    if member["archive_encoding"] == "gzip":
        with gzip.open(path, "rb") as stream:
            content_hash = _hash_stream(stream)
    else:
        content_hash = archive_hash
    if content_hash != member["sha256"]:
        raise ValueError(f"member_hash_mismatch:{relative}")


def verify(packet_root: Path) -> dict[str, Any]:
    """Verify every member before parsing any evidence and reconstruct claims."""
    packet = packet_root.resolve()
    manifest = _json((packet / MANIFEST).read_bytes(), MANIFEST)
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("invalid_manifest_members")
    by_original: dict[str, dict[str, Any]] = {}
    by_portable: dict[str, dict[str, Any]] = {}
    # Integrity pass: verify all archived and uncompressed bytes before decoding
    # any other packet member as JSON or source text.
    for member in members:
        _verify_member(packet, member)
        if member.get("actual_sha256") != member.get("sha256"):
            raise ValueError(
                f"manifest_actual_identity_mismatch:{member['portable_path']}"
            )
        if member.get("identity_class") == "prospective":
            if member.get("expected_sha256") != member.get("sha256"):
                raise ValueError(
                    f"manifest_expected_identity_mismatch:{member['portable_path']}"
                )
        elif member.get("expected_sha256") is not None:
            raise ValueError(
                f"retrospective_identity_promoted:{member['portable_path']}"
            )
        original_key = member["original_path"]
        portable = member["portable_path"]
        by_original[original_key] = member
        by_portable[portable] = member

    def portable_data(portable: str) -> bytes:
        member = by_portable[portable]
        path = packet.joinpath(*PurePosixPath(portable).parts)
        if member["archive_encoding"] == "gzip":
            with gzip.open(path, "rb") as stream:
                return stream.read()
        return path.read_bytes()

    def original(path: str) -> bytes:
        try:
            return portable_data(by_original[path]["portable_path"])
        except KeyError as exc:
            raise ValueError(f"required_member_not_manifested:{path}") from exc

    reg = _json(original(REGISTRATION), REGISTRATION)
    receipt = _json(original(RECEIPT), RECEIPT)
    if _sha(original(REGISTRATION)) != ORIGINAL_REGISTRATION_SHA256:
        raise ValueError("original_registration_identity_mismatch")
    if _sha(original(RECEIPT)) != ORIGINAL_RECEIPT_SHA256:
        raise ValueError("original_receipt_identity_mismatch")
    if receipt.get("registration_sha256") != ORIGINAL_REGISTRATION_SHA256:
        raise ValueError("receipt_registration_binding_mismatch")
    if reg.get("status") != "frozen_before_C":
        raise ValueError("original_freeze_mismatch")
    if receipt.get("decision", {}).get("status") != "close_configuration":
        raise ValueError("original_outcome_mismatch")
    if (
        manifest.get("classification") != CLASSIFICATION
        or manifest.get("registered_decision") is not None
    ):
        raise ValueError("manifest_classification_mismatch")

    # Validate exact seed464 receipt and registration binding chains.
    for relative, expected in receipt["publication_sha256"].items():
        full = f"docs/data/seed464-historical-value-removal/{relative}"
        if _sha(original(full)) != expected:
            raise ValueError(f"publication_receipt_mismatch:{relative}")
    for relative, expected in reg["input_sha256"].items():
        if _sha(original(relative)) != expected:
            raise ValueError(f"direct_input_mismatch:{relative}")
    for source, expected in reg["source_sha256"].items():
        snap = f"source-snapshots/{source}"
        member = by_portable.get(snap)
        if member is None or member["sha256"] != expected:
            raise ValueError(f"registered_source_snapshot_mismatch:{source}")

    loader_path = "source-snapshots/ml/alphazero_lite/seed435_adam_direction.py"
    loader_text = portable_data(loader_path).decode("utf-8")
    seed416_path = "docs/data/seed416-policy-target-softening/registration-v3.json"
    loader_tree = ast.parse(loader_text)
    seed416_binding: str | None = None
    for node in ast.walk(loader_tree):
        if not isinstance(node, ast.Compare) or not isinstance(node.left, ast.Call):
            continue
        call = node.left
        if not isinstance(call.func, ast.Name) or call.func.id != "sha":
            continue
        call_text = ast.get_source_segment(loader_text, call) or ""
        if seed416_path not in call_text:
            continue
        seed416_binding = next(
            (
                item.value
                for item in node.comparators
                if isinstance(item, ast.Constant)
                and isinstance(item.value, str)
                and re.fullmatch(r"[0-9a-f]{64}", item.value)
            ),
            None,
        )
        if seed416_binding:
            break
    if seed416_binding is None:
        raise ValueError("registered_loader_seed416_binding_unreadable")
    seed416_expected = seed416_binding
    if _sha(original(seed416_path)) != seed416_expected:
        raise ValueError("seed416_registration_binding_mismatch")
    for digest in (seed416_expected,):
        if digest not in loader_text:
            raise ValueError("registered_loader_missing_seed416_digest")
    if "load_jsonl_replay(" not in loader_text or "derivative_sha256" in loader_text:
        raise ValueError("derivative_loader_enforcement_claim_mismatch")

    seed416 = _json(original(seed416_path), seed416_path)
    seed462_path = "docs/data/seed462-historical-policy-removal/registration.json"
    seed462 = _json(original(seed462_path), seed462_path)
    # The two seed427 receipts are committed by exact constants in the registered
    # loader snapshot and verified there. Confirm both code and byte identity.
    receipt_map: dict[str, str] = {}
    for node in ast.walk(loader_tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RECEIPTS"
            for target in node.targets
        ):
            literal = ast.literal_eval(node.value)
            receipt_map = {
                value[0]: value[1]
                for value in literal.values()
                if isinstance(value, tuple) and len(value) == 2
            }
            break
    expected_receipts = {
        "docs/data/seed426-canonical-overlap/seed427-evaluation-verification-receipt.json": receipt_map.get(
            "docs/data/seed426-canonical-overlap/seed427-evaluation-verification-receipt.json"
        ),
        "docs/data/seed426-canonical-overlap/seed427-supplemental-verification-receipt.json": receipt_map.get(
            "docs/data/seed426-canonical-overlap/seed427-supplemental-verification-receipt.json"
        ),
    }
    for path, digest in expected_receipts.items():
        if digest is None:
            raise ValueError(f"registered_loader_receipt_binding_missing:{path}")
        name = (
            "evaluation-verification"
            if "evaluation-verification" in path
            else "supplemental-verification"
        )
        if _sha(original(path)) != digest or digest not in loader_text:
            raise ValueError(f"seed427_receipt_binding_mismatch:{name}")

    report = _json(original(REPORT), REPORT)
    if _sha(original(REPORT)) != by_original[REPORT]["sha256"]:
        raise ValueError("report_member_hash_mismatch")
    direct_claims = {
        row["path"]: row for row in report.get("remaining_direct_input_bindings", [])
    }
    if direct_claims.keys() != reg["input_sha256"].keys():
        raise ValueError("reported_direct_input_set_mismatch")
    for relative, expected in reg["input_sha256"].items():
        row = direct_claims[relative]
        if (
            row.get("frozen_expected_input_sha256") != expected
            or row.get("direct_frozen_input_sha256") != expected
            or row.get("actual_sha256") != expected
            or row.get("actual_matches_direct_input") is not True
        ):
            raise ValueError(f"reported_direct_input_binding_mismatch:{relative}")
    audit_receipt = _json(original(AUDIT_RECEIPT), AUDIT_RECEIPT)
    audit_source_path = audit_receipt["audit_source_path"]
    if _sha(original(REPORT)) != audit_receipt["report_sha256"]:
        raise ValueError("audit_receipt_report_binding_mismatch")
    if _sha(original(AUDIT_MARKDOWN)) != audit_receipt["report_markdown_sha256"]:
        raise ValueError("audit_receipt_markdown_binding_mismatch")
    if by_original[audit_source_path]["sha256"] != audit_receipt["audit_source_sha256"]:
        raise ValueError("audit_receipt_source_binding_mismatch")
    endpoint_path = "docs/data/seed464-historical-value-removal/endpoint-evidence.json"
    endpoint = _json(original(endpoint_path), endpoint_path)
    if report.get("descriptive_results") != {
        "arms": endpoint["arms"],
        "effects_T_minus_C": endpoint["effects_T_minus_C"],
        "gates_as_recorded_descriptively": endpoint["gates"],
    }:
        raise ValueError("reported_descriptive_outcomes_mismatch")

    # Reconstruct all five derivative commitments from seed416 registration.
    derivative_paths = set()
    for replay in seed416["replays"]:
        name = replay["name"]
        spec = seed416["derivatives"][name]["A"]
        logical = f"derivatives/A/{name}.jsonl"
        derivative_paths.add(logical)
        member = by_portable.get(f"{logical}.gz")
        if member is None or member["sha256"] != spec["derivative_sha256"]:
            raise ValueError(f"derivative_commitment_mismatch:{name}")
        if spec.get("lane") != "A" or _logical(spec["derivative"]) != logical:
            raise ValueError(f"derivative_logical_path_mismatch:{name}")
    reported_derivatives = {
        _logical(row["path"]): row for row in report.get("consumed_derivatives", [])
    }
    if reported_derivatives.keys() != derivative_paths:
        raise ValueError("reported_derivative_set_mismatch")
    for replay in seed416["replays"]:
        name = replay["name"]
        spec = seed416["derivatives"][name]["A"]
        row = reported_derivatives[f"derivatives/A/{name}.jsonl"]
        if row.get("frozen_expected_input_sha256") != spec["derivative_sha256"]:
            raise ValueError(f"reported_derivative_binding_mismatch:{name}")
        if (
            row.get("actual_sha256")
            != by_portable[f"derivatives/A/{name}.jsonl.gz"]["sha256"]
        ):
            raise ValueError(f"reported_derivative_actual_mismatch:{name}")
        if row.get("enforcement_location") is not None:
            raise ValueError("derivatives_must_remain_not_enforced")

    # The exact helper and transitive-input claims in the report must follow the
    # nested registration hashes and current packet bytes, not hard-coded values.
    binding_specs = {
        seed416_path: seed416_expected,
        **expected_receipts,
    }
    specified = {row["path"]: row for row in report.get("specified_dependencies", [])}
    if specified.keys() != binding_specs.keys():
        raise ValueError("reported_transitive_dependency_set_mismatch")
    for path, expected in binding_specs.items():
        row = specified[path]
        if row.get("frozen_expected_input_sha256") != expected:
            raise ValueError(f"reported_transitive_binding_mismatch:{path}")
        if row.get("actual_sha256") != _sha(original(path)):
            raise ValueError(f"reported_transitive_actual_mismatch:{path}")
        if row.get("actual_matches_expected") is not True:
            raise ValueError(f"reported_transitive_agreement_mismatch:{path}")
        if (
            row.get("registered_source_path")
            != "ml/alphazero_lite/seed435_adam_direction.py"
        ):
            raise ValueError(f"reported_transitive_source_mismatch:{path}")
        if (
            row.get("registered_source_sha256")
            != reg["source_sha256"]["ml/alphazero_lite/seed435_adam_direction.py"]
        ):
            raise ValueError(f"reported_transitive_source_hash_mismatch:{path}")
        if row.get("direct_frozen_input_sha256") is not None:
            raise ValueError(f"reported_transitive_direct_binding_mismatch:{path}")
        if row.get("enforcement_location") is None:
            raise ValueError(f"reported_transitive_enforcement_missing:{path}")

    # Imported source hashes come from nested registration inventories. The three
    # absent digests remain retrospective observations and are never upgraded.
    helper_rows = {row["path"]: row for row in report["imported_helper_sources"]}

    def local_imports(path: str) -> set[str]:
        source = by_original.get(path)
        if source is None:
            raise ValueError(f"imported_source_missing:{path}")
        source_text = portable_data(source["portable_path"]).decode("utf-8")
        tree = ast.parse(source_text)
        result: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            prefix = "ml.alphazero_lite."
            if node.module.startswith(prefix):
                result.add(node.module.replace(".", "/") + ".py")
            elif node.module == "ml.alphazero_lite":
                result.update(
                    f"ml/alphazero_lite/{alias.name}.py"
                    for alias in node.names
                    if alias.name != "*"
                )
        return result

    reachable_helpers: set[str] = set()
    registered_code_paths = set(reg["source_sha256"])
    pending = [
        imported
        for imported in local_imports("ml/alphazero_lite/seed435_adam_direction.py")
        if imported in by_original and imported not in registered_code_paths
    ]
    while pending:
        current = pending.pop()
        if current in reachable_helpers:
            continue
        reachable_helpers.add(current)
        for imported in local_imports(current):
            if (
                imported in by_original
                and imported not in registered_code_paths
                and imported not in reachable_helpers
            ):
                pending.append(imported)
    if helper_rows.keys() != reachable_helpers:
        raise ValueError(
            "reported_helper_closure_mismatch:"
            f"reported={sorted(helper_rows)}:reachable={sorted(reachable_helpers)}"
        )

    missing_helper_commitments: set[str] = set()
    for path, row in helper_rows.items():
        expected = seed462.get("source_sha256", {}).get(path)
        if expected is None:
            expected = seed416.get("source_hashes", {}).get(path)
        claimed = row.get("transitive_expected_sha256")
        if expected != claimed:
            raise ValueError(f"helper_binding_claim_mismatch:{path}")
        member = by_original.get(path)
        if member is None:
            raise ValueError(f"helper_source_missing:{path}")
        if member["sha256"] != row.get("actual_sha256"):
            raise ValueError(f"helper_source_actual_hash_mismatch:{path}")
        commitment_source = (
            "seed462 registration.json, itself directly input-hash-bound by seed464"
            if path in seed462.get("source_sha256", {})
            else "seed416 registration-v3 source_hashes, whose file SHA is enforced by registered seed435 source"
            if path in seed416.get("source_hashes", {})
            else None
        )
        if row.get("transitive_commitment") != commitment_source:
            raise ValueError(f"helper_binding_chain_claim_mismatch:{path}")
        if row.get("runtime_enforcement") is not None:
            raise ValueError(f"helper_runtime_enforcement_claim_mismatch:{path}")
        if expected is None:
            missing_helper_commitments.add(path)
        if row.get("actual_matches_expected") != (
            member["sha256"] == expected if expected else None
        ):
            raise ValueError(f"helper_actual_agreement_claim_mismatch:{path}")

    missing_derivative_enforcement = any(
        "derivative_sha256" not in loader_text
        and "sha256"
        not in loader_text[
            loader_text.find("paths = [") : loader_text.find("load_jsonl_replay(")
        ]
        for _ in (0,)
    )
    expected_classification = (
        CLASSIFICATION
        if missing_helper_commitments or missing_derivative_enforcement
        else "dependency_closure_complete"
    )
    if report.get("classification") != expected_classification:
        raise ValueError("reported_classification_mismatch")
    if report.get("registered_decision") is not None:
        raise ValueError("registered_decision_must_remain_none")

    # Explicitly check current bytes against nested commitments where present.
    checked_nested = 0
    for path, expected in seed462.get("source_sha256", {}).items():
        member = by_original.get(path)
        if member is not None:
            if member["sha256"] != expected:
                raise ValueError(f"nested_seed462_source_mismatch:{path}")
            checked_nested += 1
    for path, expected in seed416.get("source_hashes", {}).items():
        member = by_original.get(path)
        if member is not None:
            if member["sha256"] != expected:
                raise ValueError(f"nested_seed416_source_mismatch:{path}")
            checked_nested += 1

    return {
        "status": "verified",
        "classification": CLASSIFICATION,
        "registered_decision": None,
        "members": len(members),
        "direct_inputs": len(reg["input_sha256"]),
        "registered_source_snapshots": len(reg["source_sha256"]),
        "nested_source_hashes_checked": checked_nested,
        "derivative_commitments": len(derivative_paths),
        "uncommitted_helper_sources": sorted(missing_helper_commitments),
        "derivatives_runtime_enforced": not missing_derivative_enforcement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().packet_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
