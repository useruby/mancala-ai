"""Combined state inventory for authenticated #503/#504, #543, #549 sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed500_504_exclusion_inventory as seed500
from ml.alphazero_lite import seed542_549_exclusion_inventory as later

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
OUT = DATA / "expanded-training-dev-exclusion-inventory"
S562 = DATA / "seed556-560-exclusion-inventory"
S549 = DATA / "seed549-matched-selfplay"
SCHEMES = {
    "canonical": "canonical_state_sha256",
    "kalah_v3_float32": "kalah_v3_float32_sha256",
}
FAMILY_POPULATIONS = {
    "seed500_504": ("training", "raw_development"),
    "seed542": ("seed542_training", "seed542_raw_development"),
    "seed549": (
        "seed549_control_training",
        "seed549_control_raw_development",
        "seed549_treatment_training",
        "seed549_treatment_raw_development",
    ),
}
RETAINED_POPULATIONS = (
    "retained_development",
    "seed542_retained_development",
    "seed549_retained_shared_development",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _normalize_family_records(
    family: str, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    normalized = []
    for row in records:
        if "source_path" in row:
            source_path, source_line = row["source_path"], row["source_line"]
        else:
            source_path, source_line = row["source"], row["line"]
        normalized.append(
            {
                "source_family": family,
                "population": row["population"],
                "source_path": source_path,
                "source_line": source_line,
                "canonical_state_sha256": row["canonical_state_sha256"],
                "kalah_v3_float32_sha256": row.get(
                    "kalah_v3_float32_sha256", row.get("kalah_v3_float32_sha256")
                ),
            }
        )
    return normalized


def _family_sets(
    records: list[dict[str, Any]], families: dict[str, tuple[str, ...]]
) -> dict[str, dict[str, set[str]]]:
    result = {name: {scheme: set() for scheme in SCHEMES} for name in families}
    for row in records:
        for family, populations in families.items():
            if row["source_family"] == family and row["population"] in populations:
                for scheme, field in SCHEMES.items():
                    result[family][scheme].add(row[field])
    return result


def build() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bytes]]:
    report500, records500 = seed500.census()
    reportlater, recordslater, _ = later.census()
    records = _normalize_family_records("seed500_504", records500)
    records.extend(_normalize_family_records("seed542_549", recordslater))
    # The shared source manifest emits seed542 and seed549 records in one build;
    # reclassify populations to their source family while retaining row source.
    for row in records:
        if row["population"].startswith("seed542_"):
            row["source_family"] = "seed542"
        elif row["population"].startswith("seed549_"):
            row["source_family"] = "seed549"
    records.sort(
        key=lambda r: (
            r["source_family"],
            r["population"],
            r["source_path"],
            r["source_line"],
        )
    )

    # #562's publication verifier authenticates the archive sources and exactly
    # reconstructs this deterministic two-scheme identity index before comparison.
    from ml.alphazero_lite.seed556_560_exclusion_inventory import census as census562

    _, ids556, ids560, _, _ = census562()
    holdback = {scheme: set(ids556[scheme]) | set(ids560[scheme]) for scheme in ids556}
    family_sets = _family_sets(records, FAMILY_POPULATIONS)
    family_report = {}
    for family, schemes in family_sets.items():
        family_report[family] = {}
        for scheme, identities in schemes.items():
            overlap = identities & holdback[scheme]
            family_report[family][scheme] = {
                "distinct": len(identities),
                "overlap_with_562": len(overlap),
                "additions_beyond_562": len(identities - holdback[scheme]),
            }

    combined = {
        scheme: set().union(*(family_sets[f][scheme] for f in FAMILY_POPULATIONS))
        for scheme in SCHEMES
    }
    cross_family_overlap = {}
    for scheme in SCHEMES:
        family_names = list(FAMILY_POPULATIONS)
        cross_family_overlap[scheme] = {
            f"{left}_with_{right}": len(
                family_sets[left][scheme] & family_sets[right][scheme]
            )
            for i, left in enumerate(family_names)
            for right in family_names[i + 1 :]
        }
        overlap = combined[scheme] & holdback[scheme]
        family_report.setdefault("combined", {})[scheme] = {
            "distinct": len(combined[scheme]),
            "overlap_with_562": len(overlap),
            "additions_beyond_562": len(combined[scheme] - holdback[scheme]),
            "union_with_562": len(combined[scheme] | holdback[scheme]),
            "562_distinct": len(holdback[scheme]),
        }

    populations = {}
    for name in sorted({r["population"] for r in records}):
        selected = [r for r in records if r["population"] == name]
        populations[name] = {
            "occurrences": len(selected),
            "canonical_distinct": len({r["canonical_state_sha256"] for r in selected}),
            "kalah_v3_float32_distinct": len(
                {r["kalah_v3_float32_sha256"] for r in selected}
            ),
            "retained_cohort_only": name in RETAINED_POPULATIONS,
        }

    # Explicit 503/504 reconciliation values are preserved separately from the
    # retained scored cohort and from its larger raw development set.
    report = {
        "scope": "verified #503/#504, amended #543 seed542, and #549 corpora",
        "source_status": {
            "seed500_504": "verified",
            "seed542": "verified",
            "seed549": "verified",
        },
        "seed500_504_reconciliation": report500,
        "seed542_549_reconciliation": reportlater,
        "per_population": populations,
        "families_vs_562": family_report,
        "cross_family_overlap": cross_family_overlap,
        "known_gaps": {
            "historical_evaluation_populations_not_in_scope": "remain uncovered; this inventory does not close all #512–#555 evidence",
            "seed461_continuations": "unresolved; 1,129,643 continuation actions are not canonically replayable",
            "seed483_confirmation_gate": "closed; this inventory does not authorize suite generation or confirmation",
        },
    }

    source_manifest = {
        "anchor_commit": "3e493173345cd3119d62bf7173db3a512ed80966",
        "publications": {
            "503": "50f42ef63e72d297b66c5f9dbaf52c0bf8f9e1f6",
            "504": "9ff48c82f388887cd1b45e92878525bcc970314f",
            "543": "d680de41f179f5d77447607151b2621df2e0638e",
            "549": "762a98b8457fd284c2e7b02da663b6b6b140e85d",
            "550": "1a81ce5dea82b479b8babc4d90e81c8f45228c4a",
            "554": "18f5edd3167572805771a2b3b68008d38df38fa8",
            "562": "3e493173345cd3119d62bf7173db3a512ed80966",
        },
        "identity_schemes": list(SCHEMES.values()),
        "sources": {
            "seed500_504": {
                "seed504_manifest_sha256": digest(DATA / "seed504/manifest.json"),
                "seed504_split_sha256": digest(DATA / "seed504/split.json"),
                "seed504_scored_results_sha256": report500["scored_results_sha256"],
                "seed500_accounting_sha256": digest(
                    DATA / "seed500-selfplay-512/accounting.json"
                ),
            },
            "seed542": {
                "source_manifest_sha256": digest(
                    DATA / "seed542-selfplay-4096/source-manifest.json"
                ),
                "development_membership_sha256": digest(
                    DATA / "seed542-selfplay-4096/development-membership.json"
                ),
            },
            "seed549": {
                "control_source_manifest_sha256": digest(
                    DATA / "seed549-matched-selfplay/control-source-manifest.json"
                ),
                "treatment_source_manifest_sha256": digest(
                    DATA / "seed549-matched-selfplay/treatment-source-manifest.json"
                ),
                "shared_development_membership_sha256": digest(
                    DATA
                    / "seed549-matched-selfplay/shared-development-membership.jsonl"
                ),
            },
            "562_reference": {
                "source_manifest_sha256": digest(S562 / "source-manifest.json"),
                "identity_inventory_sha256": digest(S562 / "identity-inventory.json"),
            },
        },
        "authenticated_state_files": {
            "seed500_504": {
                item["path"]: item["sha256"]
                for partition in ("training", "development")
                for item in json.loads((DATA / "seed504/manifest.json").read_text())[
                    "sources"
                ][partition]
            },
            "seed542": {
                item["path"]: item["sha256"]
                for item in json.loads(
                    (DATA / "seed542-selfplay-4096/source-manifest.json").read_text()
                )["sources"]
                if item["role"] in ("training", "development")
            },
            "seed549": {
                item["path"]: item["sha256"]
                for arm in ("control", "treatment")
                for item in json.loads(
                    (S549 / f"{arm}-source-manifest.json").read_text()
                )["shards"]
            },
        },
        "derivative_reuse_evidence": reportlater["sources"]["derivative_reuse"],
        "verifier": "python -m ml.alphazero_lite.expanded_training_dev_exclusion_inventory",
    }
    index = (
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) for row in records
        )
        + "\n"
    ).encode()
    outputs = {
        "source-manifest.json": (
            json.dumps(source_manifest, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "identity-inventory.jsonl": index,
        "count-report.json": (
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        ).encode(),
    }
    return report, records, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="rebuild deterministic publication artifacts",
    )
    args = parser.parse_args()
    report, _, outputs = build()
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        for filename, content in outputs.items():
            (OUT / filename).write_bytes(content)
    else:
        for filename, content in outputs.items():
            path = OUT / filename
            if not path.is_file() or path.read_bytes() != content:
                raise ValueError(
                    f"published_artifact_mismatch:{path.relative_to(ROOT)}"
                )
    print(json.dumps(report["families_vs_562"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
