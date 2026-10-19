"""Semantic-corruption checks for seed499's read-only supplement."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from ml.alphazero_lite.seed499_budget_analysis import load_rows
from ml.alphazero_lite.seed499_budget_comparison import HISTORICAL, OUT, _sha
from ml.alphazero_lite.verify_seed499_supplement import _validate_batch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def batch_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    root = tmp_path
    out = root / OUT
    historical = root / HISTORICAL
    batch = out / "batches"
    batch.mkdir(parents=True)
    historical.mkdir(parents=True)
    source_out = ROOT / OUT
    source_historical = ROOT / HISTORICAL
    for name in (
        "runtime-artifacts.json",
        "runtime-artifacts",
    ):
        source = source_out / name
        target = out / name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copyfile(source, target)
    shutil.copyfile(
        source_historical / "runtime-artifacts.json",
        historical / "runtime-artifacts.json",
    )
    for index in (0,):
        for prefix in ("opening", "games", "report"):
            name = f"{prefix}-{index:04d}.{'jsonl' if prefix != 'report' else 'json'}"
            shutil.copyfile(source_out / "batches" / name, batch / name)
    opening = load_rows(source_out / "openings.jsonl")[0]
    pair = [
        row
        for row in load_rows(source_out / "outcomes.jsonl")
        if row["opening_index"] == 0
    ]
    registration = json.loads((source_out / "registration.json").read_text())
    historical_registration = json.loads(
        (source_historical / "registration.json").read_text()
    )
    # Runtime identity source bytes are linked read-only, except the tests do
    # not mutate those source files. The corruption is introduced in report data.
    for relative in registration["runtime"]["identities_sha256"]:
        source = ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    return (
        root,
        {
            "registration": registration,
            "historical": historical_registration,
            "opening": opening,
        },
        pair,
    )


def _rebind_receipts(root: Path, pair: list[dict[str, Any]]) -> None:
    batch = root / OUT / "batches"
    for row in pair:
        row["batch_opening_sha256"] = _sha(batch / "opening-0000.jsonl")
        row["batch_game_sha256"] = _sha(batch / "games-0000.jsonl")
        row["batch_report_sha256"] = _sha(batch / "report-0000.json")


def _run(root: Path, ctx: dict[str, Any], pair: list[dict[str, Any]]) -> None:
    _rebind_receipts(root, pair)
    _validate_batch(
        root,
        0,
        ctx["opening"],
        pair,
        ctx["registration"],
        ctx["historical"],
    )


@pytest.mark.parametrize(
    ("target", "mutation", "reason"),
    [
        ("report", "search", "search_options"),
        ("report", "suite", "suite_association"),
        ("report", "runtime", "report_runtime_identity"),
        ("report", "totals", "report_totals"),
        ("artifact", "identity", "runtime_artifact_manifest_mismatch"),
        ("ledger", "telemetry", "ledger_game_mismatch"),
        ("ledger", "winner", "ledger_game_mismatch"),
    ],
)
def test_semantic_corruption_rejected_after_receipt_rebinding(
    batch_fixture: tuple[Path, dict[str, Any], list[dict[str, Any]]],
    target: str,
    mutation: str,
    reason: str,
) -> None:
    root, ctx, original_pair = batch_fixture
    pair = [dict(row) for row in original_pair]
    batch = root / OUT / "batches"
    report_path = batch / "report-0000.json"
    report = json.loads(report_path.read_text())
    if target == "report":
        if mutation == "search":
            report["notes"]["search_options"]["fpu_mode"] = "altered"
        elif mutation == "suite":
            report["notes"]["suite_sha256"] = "0" * 64
        elif mutation == "runtime":
            report["notes"]["search_profile"]["runtime_search_policy_sha256"] = "0" * 64
        else:
            report["wins"] += 1
        report_path.write_text(json.dumps(report))
    elif target == "artifact":
        artifact = batch.parent / "runtime-artifacts/F/search_policy.json"
        artifact.write_bytes(artifact.read_bytes() + b" ")
        manifest_path = batch.parent / "runtime-artifacts.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["F"]["search_policy.json"] = _sha(artifact)
        manifest_path.write_text(json.dumps(manifest))
    else:
        if mutation == "telemetry":
            pair[0]["normal_puct_time_ms"] += 1.0
        else:
            pair[0]["winner"] = "draw"
    with pytest.raises(ValueError, match=reason):
        _run(root, ctx, pair)
