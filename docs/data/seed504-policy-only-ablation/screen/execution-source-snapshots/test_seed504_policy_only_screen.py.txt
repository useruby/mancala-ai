from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from ml.alphazero_lite import seed504_policy_only_screen as protocol


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
