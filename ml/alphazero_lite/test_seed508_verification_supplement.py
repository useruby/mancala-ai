"""Receipt-rebound semantic regression tests for the seed508 audit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PUBLICATION = REPO / "docs/data/seed508-policy-value-crossover"
_SPEC = importlib.util.spec_from_file_location(
    "seed508_verify_supplement", PUBLICATION / "verify_supplement.py"
)
assert _SPEC is not None and _SPEC.loader is not None
verify_supplement = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify_supplement)


def _copy_publication(path: Path) -> Path:
    target = path / "docs/data/seed508-policy-value-crossover"
    shutil.copytree(PUBLICATION, target)
    return target


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _rebind_fixture_receipt(out: Path) -> None:
    files = {
        path.relative_to(out).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.rglob("*"))
        if path.is_file()
        and path.name not in {"publication-receipt.json", "fixture-receipt.json"}
    }
    _write_json(out / "fixture-receipt.json", {"files": files})


def _mutate_command(out: Path, mutate) -> None:
    path = out / "commands.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows[0]["argv"])
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def test_unmodified_physical_copy_is_accepted_with_real_repository_root(
    tmp_path: Path,
) -> None:
    out = _copy_publication(tmp_path)
    assert verify_supplement.verify(REPO, out) == {
        "commands": 1024,
        "outcomes": 2048,
        "status": "verified",
    }


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda argv: argv.__setitem__(
                argv.index("--current") + 1, "/x/runtime-artifacts/E4"
            ),
            "command_contract:P:0:--current",
        ),
        (
            lambda argv: argv.__setitem__(
                argv.index("--challenger-value-artifact") + 1,
                argv[argv.index("--challenger-policy-artifact") + 1],
            ),
            "command_contract:P:0:--challenger-value-artifact",
        ),
        (
            lambda argv: argv.__setitem__(argv.index("--c-puct") + 1, "1.5"),
            "command_contract:P:0:--c-puct",
        ),
    ],
)
def test_receipt_rebound_command_mutations_fail_semantically(
    tmp_path: Path, mutate, error: str
) -> None:
    out = _copy_publication(tmp_path)
    _mutate_command(out, mutate)
    _rebind_fixture_receipt(out)
    with pytest.raises(ValueError, match=error):
        verify_supplement.verify(REPO, out)


def test_receipt_rebound_exact_root_setting_fails_semantically(tmp_path: Path) -> None:
    out = _copy_publication(tmp_path)
    path = out / "batches/P-report-0000.json"
    report = json.loads(path.read_text())
    report["notes"]["search_profile"]["exact_root_solve_threshold"] = 17
    _write_json(path, report)
    _rebind_fixture_receipt(out)
    with pytest.raises(
        ValueError, match="report_runtime:P:0:exact_root_solve_threshold"
    ):
        verify_supplement.verify(REPO, out)


def test_receipt_rebound_result_mutation_fails_semantically(tmp_path: Path) -> None:
    out = _copy_publication(tmp_path)
    path = out / "results.json"
    result = json.loads(path.read_text())
    result["standalone"]["P"]["wins"] += 1
    _write_json(path, result)
    _rebind_fixture_receipt(out)
    with pytest.raises(ValueError, match="published_result_mismatch"):
        verify_supplement.verify(REPO, out)
