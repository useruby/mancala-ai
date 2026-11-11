"""Deterministic, read-only construction and verification of #570 qualification roots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed551_controlled_policy_strength import (
    canonical_identity,
    production_float32_identity,
)
from ml.alphazero_lite.seed557_retrospective import canonical_hash

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed570-policy-prior-distillation"
TRAINING = ROOT / "docs/data/seed557-retrospective-diagnostic/cohort.json"
QUAL = ROOT / "docs/data/seed559-fixed-position-feasibility/run-0003"
SOURCE_SHA256 = {
    "ml/alphazero_lite/verify_seed557_retrospective.py": "412c314e743d8c2d42423685f0e422f7f9a55c73fdcea222f38d43a1016f6590",
    "ml/alphazero_lite/verify_seed559_fixed_position_feasibility.py": "45b40cf1b46043b08526e21dbb1ec8979b0ec183c78eac7b4dcd24eb20c962a2",
    "ml/alphazero_lite/seed559_fixed_position_feasibility.py": "973b2b884ef6454b0379acbefbcfed903aa47384851d65cfe3815cd7803207fd",
    "docs/data/seed557-retrospective-diagnostic/cohort.json": "f62539cc3e1b1843a2bd9296453d51b895a1c5e693bd9edc3b3291f883398243",
    "docs/data/seed557-retrospective-diagnostic/execution-freeze.json": "b0892a24872012486ff164d291913e1799737632a816a40c2547210a458bf36d",
    "docs/data/seed557-retrospective-diagnostic/physical-copy-verification-correction-01.json": "96f6744701878954cb72290caca9699913b16305adf1d26723a03877951e15cb",
    "docs/data/seed559-fixed-position-feasibility/run-0003/cohort-manifest.json": "f2e03fb9491474056e7a7cb3ace225cd2310f9069c9e48600d45039ed63158fb",
    "docs/data/seed559-fixed-position-feasibility/run-0003/execution-freeze.json": "3b1ab386c33534130c7e15df024f21ef75359ccd00d453e4c6aea104c8181927",
    "docs/data/seed559-fixed-position-feasibility/run-0003/oracle-records.jsonl": "83bec1bb348cbc6d1183112363f27d0612ee204a48d14ee5dbb7530e9b5fb8c2",
    "docs/data/seed559-fixed-position-feasibility/run-0003/README.md": "b537b48b074acfe4f64771707affbacafb06ec5faa5cecf8956a054ba4040bdc",
    "docs/data/seed559-fixed-position-feasibility/run-0003/count-audit.json": "6f106dc36f36182ee09090defe646901b1bce81c2418ac73aa8b76f775fe9b1e",
    "docs/data/seed559-fixed-position-feasibility/run-0003/count-audit-supplement.json": "a26f95daf3e114110b251488240c1291827fb16421d4b22b4d590c3d6edd4874",
}
SOURCE_REVISIONS = {
    "seed557_pr": "44834acaa0d7ee17465e807d10bc90e841a51a89",
    "seed557_verifier_correction": "507846bc",
    "seed557_tie_record_correction": "repository source in #557 merge 44834acaa0d7ee17465e807d10bc90e841a51a89",
    "seed559_pr": "cc4861ce1635afbb0bb4792ee3bfa33b8c09f4da",
    "seed559_run0003_corrections": "70948a58,472e7b8c; applied and published in #559 run-0003 source package",
}


def _bytes(path: Path) -> bytes:
    return path.read_bytes()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(_bytes(path))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in _bytes(path).splitlines() if line]


def authenticate_sources(root: Path = ROOT) -> dict[str, str]:
    """Require the exact frozen source artifacts, including correction records."""
    actual = {name: _sha(_bytes(root / name)) for name in SOURCE_SHA256}
    if actual != SOURCE_SHA256:
        raise ValueError("source_binding_mismatch")
    return actual


def build(
    training: dict[str, Any],
    qualification: list[dict[str, Any]],
    oracle: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply precedence-stable exclusions; oracle data is coverage-only."""
    tr_states = training["states"]
    tr_openings = {int(row["opening_index"]) for row in tr_states}
    tr_canonical = {canonical_identity(row["state"]) for row in tr_states}
    tr_float = {production_float32_identity(row["state"]) for row in tr_states}
    oracle_by_hash: dict[str, dict[str, Any]] = {}
    for record in oracle:
        state_hash = str(record["state_hash"])
        if state_hash in oracle_by_hash:
            raise ValueError("duplicate_oracle_state")
        oracle_by_hash[state_hash] = record

    candidates = []
    reasons = {
        "shared_opening": 0,
        "canonical_collision": 0,
        "float32_collision": 0,
        "duplicate_identity": 0,
        "duplicate_opening": 0,
    }
    seen_canonical: set[str] = set()
    seen_float: set[str] = set()
    seen_opening: set[int] = set()
    for row in qualification:
        state = row["state"]
        ch = canonical_identity(state)
        fh = production_float32_identity(state)
        state_hash = canonical_hash(state)
        record = oracle_by_hash.get(state_hash)
        legal = sorted(
            int(action) for action in KalahGame.from_state(state).possible_moves()
        )
        if (
            record is None
            or not record.get("coverage_complete")
            or sorted(int(action) for action in record.get("legal_moves", [])) != legal
            or sorted(int(action) for action in record.get("action_margins", {}))
            != legal
        ):
            raise ValueError("incomplete_oracle_coverage")
        if int(row["opening_index"]) in tr_openings:
            reasons["shared_opening"] += 1
            continue
        if ch in tr_canonical:
            reasons["canonical_collision"] += 1
            continue
        if fh in tr_float:
            reasons["float32_collision"] += 1
            continue
        if ch in seen_canonical or fh in seen_float:
            reasons["duplicate_identity"] += 1
            continue
        if int(row["opening_index"]) in seen_opening:
            reasons["duplicate_opening"] += 1
            continue
        seen_canonical.add(ch)
        seen_float.add(fh)
        seen_opening.add(int(row["opening_index"]))
        candidates.append(
            {
                "opening_index": int(row["opening_index"]),
                "state": state,
                "canonical_identity": ch,
                "kalah_v3_float32_identity": fh,
                "state_hash": state_hash,
                "legal_moves": legal,
            }
        )
    # Source manifest order is the published qualification ranking (SHA256 of
    # seed559 canonical identity, then identity); preserve it after filtering.
    # Sorting explicitly makes this rule independent of JSONL iteration order.
    candidates.sort(
        key=lambda row: (
            hashlib.sha256(f"seed559:{row['canonical_identity']}".encode()).hexdigest(),
            row["canonical_identity"],
        )
    )
    membership_bytes = json.dumps(
        candidates, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "schema": "seed570-qualification-membership-v1",
        "source_bindings": SOURCE_SHA256,
        "source_revisions": SOURCE_REVISIONS,
        "ordering": "ascending SHA256('seed559:' + canonical_identity), then canonical_identity",
        "source_count": len(qualification),
        "rejections": reasons,
        "population": len(candidates),
        "status": (
            "stop_shortfall"
            if len(candidates) < 256
            else "ready_for_pretraining_freeze"
        ),
        "membership_sha256": _sha(membership_bytes),
        "roots": candidates,
    }


def verify(
    bundle: dict[str, Any],
    training: dict[str, Any],
    qualification: list[dict[str, Any]],
    oracle: list[dict[str, Any]],
    *,
    root: Path = ROOT,
) -> None:
    authenticate_sources(root)
    expected = build(training, qualification, oracle)
    if bundle != expected:
        raise ValueError("membership_mismatch_or_tampering")


def verify_package(root: Path, membership_path: Path) -> dict[str, Any]:
    authenticate_sources(root)
    training = _json(root / "docs/data/seed557-retrospective-diagnostic/cohort.json")
    qual = root / "docs/data/seed559-fixed-position-feasibility/run-0003"
    qualification = _json(qual / "cohort-manifest.json")["states"]
    oracle = _jsonl(qual / "oracle-records.jsonl")
    if (
        len(training["states"]) != 256
        or len(qualification) != 512
        or len(oracle) != 512
    ):
        raise ValueError("source_population_count_mismatch")
    bundle = build(training, qualification, oracle)
    verify(_json(membership_path), training, qualification, oracle, root=root)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=ROOT)
    parser.add_argument("--membership", type=Path)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="write membership when no package exists",
    )
    args = parser.parse_args()
    destination = args.membership or (
        args.evidence_root
        / "docs/data/seed570-policy-prior-distillation/qualification-membership.json"
    )
    if args.publish:
        authenticate_sources(args.evidence_root)
        training = _json(
            args.evidence_root
            / "docs/data/seed557-retrospective-diagnostic/cohort.json"
        )
        qual = (
            args.evidence_root / "docs/data/seed559-fixed-position-feasibility/run-0003"
        )
        bundle = build(
            training,
            _json(qual / "cohort-manifest.json")["states"],
            _jsonl(qual / "oracle-records.jsonl"),
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    else:
        if not destination.is_file():
            raise FileNotFoundError(f"membership package not found: {destination}")
        bundle = verify_package(args.evidence_root, destination)
    print(
        json.dumps(
            {
                key: bundle[key]
                for key in (
                    "population",
                    "rejections",
                    "membership_sha256",
                    "status",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
