"""Receipt-rebound semantic tests for seed508 supplement v2."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PUB = REPO / "docs/data/seed508-policy-value-crossover"
SPEC = importlib.util.spec_from_file_location(
    "seed508_v2", PUB / "verify_supplement_v2.py"
)
assert SPEC is not None and SPEC.loader is not None
v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v2)


def copy_pub(tmp_path: Path) -> Path:
    target = tmp_path / "unrelated-copy"
    shutil.copytree(PUB, target)
    return target


def rebind(out: Path) -> None:
    files = {
        p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*"))
        if p.is_file()
        and p.name not in {"fixture-receipt.json", "publication-receipt.json"}
    }
    (out / "fixture-receipt.json").write_text(
        json.dumps({"files": files}, indent=2) + "\n"
    )


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def test_copied_publication_passes(tmp_path: Path) -> None:
    out = copy_pub(tmp_path)
    assert v2.verify(REPO, out)["status"] == "verified"


@pytest.mark.parametrize(
    ("target", "expected", "code"),
    [
        (
            "primary",
            ["primary", "two_sided_95"],
            "published_primary_arithmetic_mismatch",
        ),
        (
            "interpretation",
            ["primary", "interpretation"],
            "published_primary_arithmetic_mismatch",
        ),
        (
            "standalone",
            ["standalone", "P", "descriptive_two_sided_95"],
            "published_standalone_arithmetic_mismatch",
        ),
        (
            "search",
            ["report", "notes", "search_profile", "search_options", "root_temperature"],
            "report_search_options:P:0",
        ),
        (
            "exact_leaf",
            ["report", "notes", "search_profile", "exact_leaf_solve_mode"],
            "report_exact_leaf_mode:P:0",
        ),
        (
            "sidecar",
            ["report", "seed508_composition", "component_artifacts", "policy"],
            "report_sidecar_identity:P:0:policy",
        ),
    ],
)
def test_receipt_rebound_semantic_mutations(
    tmp_path: Path, target: str, expected: list[str], code: str
) -> None:
    out = copy_pub(tmp_path)
    if target in {"primary", "interpretation", "standalone"}:
        path = out / "results.json"
        data = json.loads(path.read_text())
        node = data
        for key in expected[:-1]:
            node = node[key]
        node[expected[-1]] = (
            "altered" if isinstance(node[expected[-1]], str) else [0, 1]
        )
    else:
        path = out / "batches/P-report-0000.json"
        data = json.loads(path.read_text())
        node = data
        for key in expected[1:-1]:
            node = node[key]
        if target == "search":
            node[expected[-1]] = 99.0
        elif target == "exact_leaf":
            node[expected[-1]] = "enabled"
        else:
            node[expected[-1]] = "/wrong/sidecar"
    write_json(path, data)
    rebind(out)
    with pytest.raises(ValueError, match=code):
        v2.verify(REPO, out)
