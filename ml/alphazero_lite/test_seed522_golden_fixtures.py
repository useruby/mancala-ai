"""Real CLI contract tests for the corrected seed522 fixture bundle."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "docs/data/seed522-golden-fixtures"
CLI = "ml.alphazero_lite.seed521_golden_fixtures"


def run(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-m", CLI, *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def copy_packet(tmp_path: Path) -> Path:
    target = tmp_path / "fixture-copy"
    shutil.copytree(PACKET, target)
    return target


def change_case(packet: Path, case_index: int, mutate) -> None:
    path = packet / "fixtures.json"
    fixture = json.loads(path.read_text())
    mutate(fixture["cases"][case_index])
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    index_path = packet / "index.json"
    index = json.loads(index_path.read_text())
    index["fixtures"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize("player", [0, 1])
def test_read_only_cli_accepts_fixed_orientation_and_actor_relative_moves(player):
    result = run(
        "verify", "--repository-root", str(ROOT), "--fixture-root", str(PACKET)
    )
    assert result.returncode == 0, result.stderr
    fixture = json.loads((PACKET / "fixtures.json").read_text())
    case = next(
        row for row in fixture["cases"][:64] if row["state"]["current_player"] == player
    )
    from ml.alphazero_lite.kalah_rules import KalahGame

    game = KalahGame.from_state(case["state"])
    expected = [
        i
        for i, seeds in enumerate(
            case["state"]["player_pits" if player == 0 else "opponent_pits"]
        )
        if seeds
    ]
    assert game.possible_moves() == expected
    assert game.to_state() == case["state"]


@pytest.mark.parametrize(
    ("index", "mutate", "reason"),
    [
        (
            0,
            lambda c: c["state"].update(opponent_pits=c["state"]["player_pits"]),
            "archived_state_mismatch",
        ),
        (
            0,
            lambda c: c["source"].update(record_index=-1),
            "provenance_locator_mismatch",
        ),
        (0, lambda c: c.update(cohort_bucket="wrong"), "archived_state_mismatch"),
        (
            0,
            lambda c: c.update(selection_order_sha256="0" * 64),
            "archived_state_mismatch",
        ),
        (0, lambda c: c.update(policy_f32_hex="00" * 24), "archived_output_mismatch"),
        (64, lambda c: c["state"].update(current_player=1), "terminal_fixture_invalid"),
        (65, lambda c: c["state"].update(player_store=47), "terminal_fixture_invalid"),
        (0, lambda c: c.update(kind="terminal"), "case_kind"),
    ],
)
def test_rebound_integrity_does_not_bypass_semantic_checks(
    tmp_path, index, mutate, reason
):
    packet = copy_packet(tmp_path)
    change_case(packet, index, mutate)
    result = run(
        "verify", "--repository-root", str(ROOT), "--fixture-root", str(packet)
    )
    assert result.returncode != 0
    assert reason in result.stderr


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("schema", "fixture_schema"),
        ("orientation", "orientation"),
        ("encoding", "output_encoding"),
        ("missing", "case_count"),
        ("duplicate", "case_id_order_or_duplicate"),
    ],
)
def test_contract_declarations_and_membership_fail_closed(tmp_path, mutation, reason):
    packet = copy_packet(tmp_path)
    fixture_path = packet / "fixtures.json"
    fixture = json.loads(fixture_path.read_text())
    if mutation == "orientation":
        fixture["field_orientation"]["player_pits"] = "current player"
    elif mutation == "schema":
        fixture["schema"] = "unknown"
    elif mutation == "encoding":
        fixture["output_encoding"] = "big endian"
    elif mutation == "missing":
        fixture["cases"].pop()
        fixture["case_count"] -= 1
    else:
        fixture["cases"][1]["case_id"] = fixture["cases"][0]["case_id"]
    fixture_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    index_path = packet / "index.json"
    index = json.loads(index_path.read_text())
    index["fixtures"]["sha256"] = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    result = run(
        "verify", "--repository-root", str(ROOT), "--fixture-root", str(packet)
    )
    assert result.returncode != 0
    assert reason in result.stderr


def inventory(directory: Path) -> dict[str, tuple[str, int]]:
    return {
        path.name: (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in directory.iterdir()
        if path.is_file()
    }


def test_damaged_fixture_is_not_repaired_and_verify_preserves_publication(tmp_path):
    damaged = copy_packet(tmp_path)
    fixture = damaged / "fixtures.json"
    fixture.write_bytes(fixture.read_bytes() + b"damage")
    before = inventory(damaged)
    result = run(
        "verify", "--repository-root", str(ROOT), "--fixture-root", str(damaged)
    )
    assert result.returncode != 0
    assert "fixture_hash" in result.stderr
    assert inventory(damaged) == before

    before = inventory(PACKET)
    for _ in range(2):
        result = run()
        assert result.returncode == 0, result.stderr
    assert inventory(PACKET) == before


def test_generation_is_explicit_create_only(tmp_path):
    new_dir = tmp_path / "new-bundle"
    result = run("generate", "--repository-root", str(ROOT), "--output", str(new_dir))
    assert result.returncode == 0, result.stderr
    assert (
        run(
            "verify", "--repository-root", str(ROOT), "--fixture-root", str(new_dir)
        ).returncode
        == 0
    )
    before = inventory(new_dir)
    result = run("generate", "--repository-root", str(ROOT), "--output", str(new_dir))
    assert result.returncode != 0
    assert "output_exists" in result.stderr
    assert inventory(new_dir) == before
