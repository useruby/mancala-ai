from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
import torch

from ml.alphazero_lite import seed504_policy_only_screen as protocol
from ml.alphazero_lite import verify_seed504_policy_only_screen as screen_verifier


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[dict]]:
    root = tmp_path / "copy"
    out_rel = Path("screen")
    out = root / out_rel
    batch = out / "batches"
    batch.mkdir(parents=True)
    monkeypatch.setattr(protocol, "OUT", out_rel)
    source = protocol.ROOT / protocol.SEED511
    openings = protocol.read_rows(source / "openings.jsonl")
    (out / "openings.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in openings
        )
    )
    registration = json.loads((source / "registration.json").read_text())
    registration["runtime"].update(
        {
            "native_probe_sha256": protocol.sha(
                protocol.ROOT / protocol.RUNTIME_FILES[0]
            ),
            "tablebase_sha256": protocol.sha(protocol.ROOT / protocol.RUNTIME_FILES[1]),
        }
    )
    protocol.atomic_json(out / "registration.json", registration)
    opening = openings[0]
    (batch / "opening-0000.jsonl").write_bytes(protocol.canonical(opening) + b"\n")
    shutil.copyfile(source / "batches/games-0000.jsonl", batch / "games-0000.jsonl")
    report = json.loads((source / "batches/report-0000.json").read_text())
    report["notes"]["base_seed"] = protocol.BASE_SEED
    report["notes"]["seed"] = protocol.BASE_SEED
    report["notes"]["suite_sha256"] = protocol.sha(batch / "opening-0000.jsonl")
    protocol.atomic_json(batch / "report-0000.json", report)
    return root, openings


def test_resume_recovers_complete_raw_pair_and_skips_committed_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, openings = _fixture(tmp_path, monkeypatch)
    recover, pending = protocol.resume_plan(root, openings, [])
    assert recover == [0]
    assert 0 not in pending
    ledger = protocol._pair_rows(root, 0, openings[0])
    recover_again, pending_again = protocol.resume_plan(root, openings, ledger)
    assert recover_again == []
    assert 0 not in pending_again


def test_resume_rejects_interrupted_partial_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, openings = _fixture(tmp_path, monkeypatch)
    (root / protocol.OUT / "batches/games-0001.jsonl").write_text("[]\n")
    with pytest.raises(ValueError, match="pair_incomplete:1"):
        protocol.resume_plan(root, openings, [])


def test_semantic_corruption_is_rejected_after_receipt_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, openings = _fixture(tmp_path, monkeypatch)
    out = root / protocol.OUT
    rows = protocol._pair_rows(root, 0, openings[0])
    games_path = out / "batches/games-0000.jsonl"
    games = protocol.read_rows(games_path)
    games[0]["trajectory"] = "999"
    games_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in games)
    )

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    for row in rows:
        row["batch_game_sha256"] = digest(games_path)
    # Receipt hashes were updated to the corrupt evidence; legal replay must
    # still reject the changed trajectory.
    with pytest.raises((ValueError, KeyError)):
        protocol._verify_committed(root, openings, rows)


def test_read_only_command_resolution_does_not_recreate_missing_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _fixture(tmp_path, monkeypatch)
    opening_path = root / protocol.OUT / "batches/opening-0000.jsonl"
    opening_path.unlink()
    argv = protocol.command(root, 0, prepare_opening=False)
    assert str(opening_path) in argv
    assert not opening_path.exists()


def test_verifier_relocates_registered_commands_in_physical_checkout(
    tmp_path: Path,
) -> None:
    recorded = [
        "/original/.tmp/venv/bin/python",
        "--challenger",
        "/original/docs/data/seed504-policy-only-ablation/screen/runtime-artifacts/T-E4",
        "--out",
        "/original/docs/data/seed504-policy-only-ablation/screen/batches/report-0000.json",
    ]
    relocated = screen_verifier._relocate_command(recorded, tmp_path)
    assert relocated == [
        "/original/.tmp/venv/bin/python",
        "--challenger",
        str(
            tmp_path
            / "docs/data/seed504-policy-only-ablation/screen/runtime-artifacts/T-E4"
        ),
        "--out",
        str(
            tmp_path
            / "docs/data/seed504-policy-only-ablation/screen/batches/report-0000.json"
        ),
    ]


def test_independent_bootstrap_preserves_duplicate_sampled_openings() -> None:
    scores = np.asarray([0.0, 1.0])
    lower, samples = screen_verifier.bootstrap_lower_bound(
        scores, resamples=8, seed=17, quantile=0.05
    )
    indices = np.random.default_rng(17).integers(0, 2, size=(8, 2))
    expected = scores[indices].mean(axis=1)
    assert np.array_equal(samples, expected)
    assert lower == np.quantile(expected, 0.05, method="linear")
    assert np.any(np.all(indices == 0, axis=1))


def test_advancement_boundary_is_strictly_above_half() -> None:
    assert not screen_verifier.advance_from_lower_bound(0.50)
    assert screen_verifier.advance_from_lower_bound(np.nextafter(0.50, 1.0))


def _copied_screen(root: Path) -> Path:
    source_root = protocol.ROOT
    out_rel = protocol.OUT
    shutil.copytree(source_root / out_rel, root / out_rel)
    out = root / out_rel
    registration = json.loads((out / "registration.json").read_text())
    required = set(registration["source_sha256"])
    required.update(registration["input_sha256"])
    required.update(registration["runtime_asset_sha256"])
    required.update(item["checkpoint"] for item in registration["components"].values())
    for rel in required:
        source = source_root / rel
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return out


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("outcome", "trajectory_length_or_cap"),
        ("seat", "pair_seat_mismatch"),
        ("report", "report_outcome_disagreement"),
        ("exact_leaf", "pair_runtime_or_report_mismatch"),
        ("exact_objective", "pair_runtime_or_report_mismatch"),
        ("exact_tie_rule", "pair_runtime_or_report_mismatch"),
        ("runtime", "runtime_identity_mismatch"),
        ("component", "component_artifact_mismatch"),
        ("score", "result_analysis_mismatch"),
        ("interval", "result_analysis_mismatch"),
        ("advance", "result_analysis_mismatch"),
    ],
)
def test_real_verifier_rejects_semantic_corruption_after_receipt_rebinding(
    tmp_path: Path, mutation: str, error: str
) -> None:
    root = tmp_path / "copy"
    out = _copied_screen(root)
    if mutation in {
        "outcome",
        "seat",
        "report",
        "exact_leaf",
        "exact_objective",
        "exact_tie_rule",
    }:
        game_path = out / "batches/games-0000.jsonl"
        games = protocol.read_rows(game_path)
        report_path = out / "batches/report-0000.json"
        report = json.loads(report_path.read_text())
        if mutation == "outcome":
            games[0]["trajectory"] = "999"
        elif mutation == "seat":
            games[1]["challenger_player"] = games[0]["challenger_player"]
        elif mutation == "exact_leaf":
            report["notes"]["exact_leaf_solve_mode"] = "enabled"
            protocol.atomic_json(report_path, report)
        elif mutation == "exact_objective":
            report["notes"]["exact_root_objective"] = "alternate_score"
            protocol.atomic_json(report_path, report)
        elif mutation == "exact_tie_rule":
            report["notes"]["exact_root_tie_rule"] = "lowest_move_index"
            protocol.atomic_json(report_path, report)
        else:
            report["wins"] += 1
            protocol.atomic_json(report_path, report)
        game_path.write_text(
            "".join(json.dumps(x, sort_keys=True) + "\n" for x in games)
        )
        ledger_path = out / "outcomes.jsonl"
        ledger = protocol.read_rows(ledger_path)
        for receipt in ledger:
            if receipt["opening_index"] == 0:
                receipt["batch_game_sha256"] = protocol.sha(game_path)
                receipt["batch_report_sha256"] = protocol.sha(report_path)
        protocol.atomic_ledger(ledger_path, ledger)
    elif mutation == "runtime":
        rel = next(
            iter(
                json.loads((out / "registration.json").read_text())[
                    "runtime_asset_sha256"
                ]
            )
        )
        (root / rel).write_bytes(b"corrupt runtime")
    elif mutation == "component":
        path = out / "runtime-artifacts/T-E4/model.npz"
        path.write_bytes(path.read_bytes() + b"corrupt")
    else:
        result_path = out / "results.json"
        result = json.loads(result_path.read_text())
        if mutation == "score":
            result["score"] += 0.001
        elif mutation == "interval":
            result["one_sided_95_lower"] += 0.001
        else:
            result["advance"] = True
        protocol.atomic_json(result_path, result)
    with pytest.raises(ValueError, match=error):
        screen_verifier.verify(root)


def test_capped_game_is_not_committed_as_an_ordinary_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, openings = _fixture(tmp_path, monkeypatch)
    out = root / protocol.OUT
    game_path = out / "batches/games-0000.jsonl"
    games = protocol.read_rows(game_path)
    games[0]["game_length"] = 200
    game_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in games)
    )
    with pytest.raises(ValueError, match="capped_or_incomplete_game:0:0"):
        protocol._pair_rows(root, 0, openings[0])


@pytest.mark.parametrize("binding", ["suite", "artifact", "runtime"])
def test_changed_bindings_are_rejected_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: str
) -> None:
    root, openings = _fixture(tmp_path, monkeypatch)
    out = root / protocol.OUT
    registration = json.loads((out / "registration.json").read_text())
    # Construct a minimal self-bound freeze accepted up to the particular
    # changed binding, keeping arena invocation entirely mocked/unstarted.
    registration["status"] = "frozen_before_arena_execution"
    registration["suite"] = {"sha256": protocol.sha(out / "openings.jsonl")}
    registration["source_sha256"] = {}
    registration["input_sha256"] = {}
    registration["runtime_asset_sha256"] = {}
    registration["components"] = {}
    (out / "registration.json").write_text(json.dumps(registration, sort_keys=True))
    protocol.atomic_json(
        out / "execution-freeze.json",
        {
            "status": "frozen_before_arena_execution",
            "registration_sha256": protocol.sha(out / "registration.json"),
            "suite_sha256": protocol.sha(out / "openings.jsonl"),
        },
    )
    if binding == "suite":
        (out / "openings.jsonl").write_text("corrupt\n")
    else:
        # A freeze-bound component/runtime is changed after preparation; the
        # execution preflight must fail closed without starting a game.
        rel = "artifact.bin" if binding == "artifact" else "runtime.bin"
        path = root / rel
        path.write_text("before")
        digest = protocol.sha(path)
        if binding == "artifact":
            checkpoint = root / "checkpoint.npz"
            checkpoint.write_text("checkpoint")
            directory = root / "component"
            directory.mkdir()
            for name in protocol.ARTIFACT_FILES:
                (directory / name).write_text("component")
            registration["components"] = {
                "test": {
                    "checkpoint": "checkpoint.npz",
                    "checkpoint_sha256": protocol.sha(checkpoint),
                    "artifact": "component",
                    "files": protocol.component_files(directory),
                }
            }
            protocol.atomic_json(out / "registration.json", registration)
            protocol.atomic_json(
                out / "execution-freeze.json",
                {
                    "status": "frozen_before_arena_execution",
                    "registration_sha256": protocol.sha(out / "registration.json"),
                    "suite_sha256": protocol.sha(out / "openings.jsonl"),
                },
            )
            (directory / "model.npz").write_text("after")
        else:
            registration["runtime_asset_sha256"] = {rel: digest}
            protocol.atomic_json(out / "registration.json", registration)
            protocol.atomic_json(
                out / "execution-freeze.json",
                {
                    "status": "frozen_before_arena_execution",
                    "registration_sha256": protocol.sha(out / "registration.json"),
                    "suite_sha256": protocol.sha(out / "openings.jsonl"),
                },
            )
            path.write_text("after")
    with pytest.raises(ValueError):
        protocol.preflight(root, allow_started=False)


def test_snapshot_authenticates_history_but_never_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _fixture(tmp_path, monkeypatch)
    out = root / protocol.OUT
    rel = "ml/alphazero_lite/frozen.py"
    source = root / rel
    source.parent.mkdir(parents=True)
    source.write_text("working bytes")
    snapshot = out / "execution-source-snapshots/frozen.py.txt"
    snapshot.parent.mkdir()
    snapshot.write_text("registered historical bytes")
    registration = json.loads((out / "registration.json").read_text())
    registration["status"] = "frozen_before_arena_execution"
    registration["runtime_asset_sha256"] = {}
    registration["input_sha256"] = {}
    registration["components"] = {}
    registration["source_sha256"] = {
        rel: hashlib.sha256(snapshot.read_bytes()).hexdigest()
    }
    registration["suite"]["sha256"] = protocol.sha(out / "openings.jsonl")
    protocol.atomic_json(out / "registration.json", registration)
    protocol.atomic_json(
        out / "execution-freeze.json",
        {
            "status": "frozen_before_arena_execution",
            "registration_sha256": protocol.sha(out / "registration.json"),
            "suite_sha256": protocol.sha(out / "openings.jsonl"),
        },
    )
    with pytest.raises(ValueError, match="frozen_source_changed"):
        protocol.preflight(root, allow_started=False)
    assert protocol.preflight(root, historical=True) == registration


def test_matching_executable_source_passes_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _fixture(tmp_path, monkeypatch)
    out = root / protocol.OUT
    rel = "ml/alphazero_lite/frozen.py"
    source = root / rel
    source.parent.mkdir(parents=True)
    source.write_text("registered bytes")
    registration = json.loads((out / "registration.json").read_text())
    registration["status"] = "frozen_before_arena_execution"
    registration["runtime_asset_sha256"] = {}
    registration["input_sha256"] = {}
    registration["components"] = {}
    registration["source_sha256"] = {rel: protocol.sha(source)}
    registration["suite"]["sha256"] = protocol.sha(out / "openings.jsonl")
    registration["environment"].update(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pip_freeze": subprocess.run(
                [sys.executable, "-m", "pip", "freeze", "--all"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines(),
        }
    )
    protocol.atomic_json(out / "registration.json", registration)
    protocol.atomic_json(
        out / "execution-freeze.json",
        {
            "status": "frozen_before_arena_execution",
            "registration_sha256": protocol.sha(out / "registration.json"),
            "suite_sha256": protocol.sha(out / "openings.jsonl"),
        },
    )
    assert protocol.preflight(root, allow_started=False) == registration


def test_changed_source_rejects_run_before_arena_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _fixture(tmp_path, monkeypatch)
    out = root / protocol.OUT
    rel = "ml/alphazero_lite/frozen.py"
    source = root / rel
    source.parent.mkdir(parents=True)
    source.write_text("working bytes")
    snapshot = out / "execution-source-snapshots/frozen.py.txt"
    snapshot.parent.mkdir()
    snapshot.write_text("registered bytes")
    registration = json.loads((out / "registration.json").read_text())
    registration.update(
        status="frozen_before_arena_execution",
        source_sha256={rel: protocol.sha(snapshot)},
        input_sha256={},
        runtime_asset_sha256={},
        components={},
    )
    registration["suite"]["sha256"] = protocol.sha(out / "openings.jsonl")
    protocol.atomic_json(out / "registration.json", registration)
    protocol.atomic_json(
        out / "execution-freeze.json",
        {
            "status": "frozen_before_arena_execution",
            "registration_sha256": protocol.sha(out / "registration.json"),
            "suite_sha256": protocol.sha(out / "openings.jsonl"),
        },
    )
    calls = []
    monkeypatch.setattr(
        protocol.subprocess, "run", lambda *args, **kwargs: calls.append(args)
    )
    with pytest.raises(ValueError, match="frozen_source_changed"):
        protocol.run(root, execute=True)
    assert calls == []
