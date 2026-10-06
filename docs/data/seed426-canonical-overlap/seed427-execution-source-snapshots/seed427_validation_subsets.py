"""Pure retrospective identity subset construction from seed426 row evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def construct(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Label frozen validation positions without changing historical partitions."""
    train_canonical = {
        row["canonical_identity"] for row in rows if row["partition"] == "train"
    }
    train_inputs = {
        row["input_identity"] for row in rows if row["partition"] == "train"
    }
    validation = [row for row in rows if row["partition"] == "validation"]
    membership = []
    canonical_seen = set()
    input_seen = set()
    for row in validation:
        seen = row["canonical_identity"] in train_canonical
        input_overlap = row["input_identity"] in train_inputs
        if seen:
            canonical_seen.add(row["canonical_identity"])
        if input_overlap:
            input_seen.add(row["input_identity"])
        membership.append(
            {
                "weighted_position": row["weighted_position"],
                "compact_row": row["compact_row"],
                "source": row["source"],
                "raw_line": row["raw_line"],
                "active_stones": row["active_stones"],
                "canonical_identity": row["canonical_identity"],
                "input_identity": row["input_identity"],
                "canonical_seen_in_train": seen,
                "input_seen_in_train": input_overlap,
                "subset": "seen"
                if seen
                else "unseen"
                if not input_overlap
                else "canonical_unseen_input_seen",
                "input_disjoint_subset": not input_overlap,
            }
        )
    counts: dict[str, Any] = {}
    for scope, selected in (
        ("all", membership),
        ("seen", [r for r in membership if r["canonical_seen_in_train"]]),
        (
            "unseen",
            [
                r
                for r in membership
                if not r["canonical_seen_in_train"] and not r["input_seen_in_train"]
            ],
        ),
    ):
        counts[scope] = {
            "weighted_positions": len(selected),
            "canonical_identities": len(
                {row["canonical_identity"] for row in selected}
            ),
            "network_input_identities": len(
                {row["input_identity"] for row in selected}
            ),
        }
        for bucket, predicate in (
            (">32", lambda n: n > 32),
            ("17-32", lambda n: 17 <= n <= 32),
            ("<=16", lambda n: n <= 16),
        ):
            bucket_rows = [row for row in selected if predicate(row["active_stones"])]
            counts[scope][bucket] = {
                "weighted_positions": len(bucket_rows),
                "canonical_identities": len(
                    {row["canonical_identity"] for row in bucket_rows}
                ),
            }
    counts["input_identity_definition_difference"] = {
        "validation_positions_canonical_unseen_but_input_seen": sum(
            not row["canonical_seen_in_train"] and row["input_seen_in_train"]
            for row in membership
        ),
        "validation_positions_canonical_seen_but_input_unseen": sum(
            row["canonical_seen_in_train"] and not row["input_seen_in_train"]
            for row in membership
        ),
        "strict_unseen_positions": sum(
            not row["canonical_seen_in_train"] and not row["input_seen_in_train"]
            for row in membership
        ),
        "strict_unseen_canonical_identities": len(
            {
                row["canonical_identity"]
                for row in membership
                if not row["canonical_seen_in_train"] and not row["input_seen_in_train"]
            }
        ),
        "input_disjoint_positions": sum(
            not row["input_seen_in_train"] for row in membership
        ),
        "input_disjoint_canonical_identities": len(
            {
                row["canonical_identity"]
                for row in membership
                if not row["input_seen_in_train"]
            }
        ),
    }
    return membership, counts


def read_evidence(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def main(root: Path) -> None:
    data = root / "docs/data/seed426-canonical-overlap"
    rows = read_evidence(data / "row-accounting.jsonl.gz")
    membership, counts = construct(rows)
    output = data / "seed427-validation-membership.jsonl.gz"
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in membership
    ).encode("utf-8")
    output.write_bytes(gzip.compress(content, mtime=0))
    results = {
        "schema": "seed427-validation-subsets-v1",
        "publication": "retrospective subset of frozen seed426 validation; not a new independent holdout",
        "membership_sha256": sha256(output.read_bytes()),
        "membership_rows": len(membership),
        "counts": counts,
    }
    (data / "seed427-validation-subsets.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    main(parser.parse_args().root)
