"""Self-contained tests for seed549 freezes, recovery, and derivatives."""

from __future__ import annotations

import json
import sys
from unittest.mock import Mock

import numpy as np
import pytest

from ml.alphazero_lite import seed549_corpus as corpus


def test_generation_command_freezes_all_pilot_worker_settings() -> None:
    command = corpus.make_command(
        corpus.ROOT, "control", 6, corpus.OUT / "control/shard-006.jsonl"
    )
    assert command[command.index("--games") + 1] == "64"
    assert command[command.index("--seed") + 1] == "549006"
    assert command[command.index("--simulations") + 1] == "128"
    assert command[command.index("--workers") + 1] == "1"
    assert command[command.index("--max-moves") + 1] == "200"
    assert command[command.index("--start-state-mode") + 1] == "standard_4x6"
    assert command[command.index("--policy-target-noise-mode") + 1] == "denoised"
    assert command[command.index("--value-target-mode") + 1] == "default"


def test_partition_is_frozen_before_generation() -> None:
    assert corpus.TRAIN_SHARDS == {0, 1, 2, 3, 4, 5}
    assert set(range(corpus.SHARDS)) - corpus.TRAIN_SHARDS == {6, 7}
    assert len(corpus.TRAIN_SHARDS) * corpus.GAMES_PER_SHARD == 384
    assert 2 * corpus.GAMES_PER_SHARD == 128


def test_schedule_has_108_updates_54896_exposures_and_pass_uniqueness() -> None:
    schedule = corpus.build_schedule(701, seed=17)
    assert schedule["updates"] == 108
    assert schedule["exposures"] == 54_896
    batches = schedule["batches"]
    assert len(batches) == 108
    assert [
        b["end_exposure_exclusive"] - b["start_exposure"] for b in batches
    ] == corpus.BATCHES * 4
    flattened = [
        index for batch in batches for index in batch["row_occurrence_indices"]
    ]
    assert len(flattened) == 54_896
    for boundary in schedule["pass_boundaries"]:
        values = flattened[
            boundary["start_exposure"] : boundary["end_exposure_exclusive"]
        ]
        assert len(values) == len(set(values))
    assert schedule == corpus.build_schedule(701, seed=17)


def test_schedule_preserves_duplicate_row_occurrences() -> None:
    # The schedule indexes source row occurrences; identical rows are not deduplicated.
    schedule = corpus.build_schedule(2, seed=29)
    assert schedule["exposures"] == 54_896
    first_pass = schedule["pass_boundaries"][0]
    values = [
        i for batch in schedule["batches"] for i in batch["row_occurrence_indices"]
    ]
    pass_values = values[
        first_pass["start_exposure"] : first_pass["end_exposure_exclusive"]
    ]
    assert set(pass_values) == {0, 1}
    assert len(pass_values) == 2


def test_development_exclusions_preserve_identity_reason_and_source() -> None:
    training = {"control": {"canonical-a", "float-b"}, "treatment": {"float-c"}}
    assert corpus.development_exclusion_reasons("canonical-a", "float-c", training) == [
        "canonical_overlap:control",
        "float32_overlap:treatment",
    ]
    assert corpus.development_exclusion_reasons("new", "new-float", training) == []


def test_identity_mutation_changes_float32_identity() -> None:
    a = {"_canonical_state": "[1,2]", "state": [0.1, 0.2]}
    b = {"_canonical_state": "[1,2]", "state": [0.1, 0.3]}
    assert corpus.identity(a)[0] == corpus.identity(b)[0]
    assert corpus.identity(a)[1] != corpus.identity(b)[1]


def test_worker_rejects_mutated_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(corpus, "preflight", lambda: {})
    with pytest.raises(ValueError, match="worker_command_mismatch"):
        corpus.run_worker(
            "control", ["--arm", "control", "--shard", "0", "--out", "bad.jsonl"]
        )


def test_resume_skips_only_receipt_verified_completed_shard(tmp_path) -> None:
    output = tmp_path / "shard.jsonl"
    output.write_text('{"row":1}\n')
    receipt = corpus.digest(output)
    events = [
        {"event": "started"},
        {"event": "finished", "output_sha256": receipt},
        {"event": "published", "output_sha256": receipt},
    ]
    assert corpus.completed_receipt_verified(events, output)
    output.write_text('{"row":2}\n')
    assert not corpus.completed_receipt_verified(events, output)
    assert not corpus.completed_receipt_verified(events[:2], output)
    assert not corpus.completed_receipt_verified(
        [events[0], {"event": "interrupted"}], output
    )


def test_mutated_registration_freeze_is_rejected(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A changed registration byte without a matching freeze is rejected before assets are read.
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"generation": {}}))
    (tmp_path / "execution-freeze.json").write_text(
        json.dumps({"registration_sha256": "bad"})
    )
    monkeypatch.setattr(
        corpus,
        "OUT",
        tmp_path.relative_to(corpus.ROOT)
        if tmp_path.is_relative_to(corpus.ROOT)
        else corpus.OUT,
    )
    with pytest.raises((FileNotFoundError, ValueError)):
        corpus.preflight(tmp_path)


def test_passes_are_deterministic_and_do_not_repeat_within_each_pass() -> None:
    schedule = corpus.build_schedule(997, seed=549000)
    flat = [
        value
        for batch in schedule["batches"]
        for value in batch["row_occurrence_indices"]
    ]
    boundaries = schedule["pass_boundaries"]
    assert boundaries[0]["permutation_seed"] == 549000
    assert boundaries[1]["permutation_seed"] == 549001
    for item in boundaries:
        span = flat[item["start_exposure"] : item["end_exposure_exclusive"]]
        assert len(span) == len(set(span))
    assert np.array_equal(
        np.asarray(flat),
        np.asarray(
            [
                value
                for batch in corpus.build_schedule(997, seed=549000)["batches"]
                for value in batch["row_occurrence_indices"]
            ]
        ),
    )


def _write_prior_attempt(
    root,
    *,
    status: int,
    output_bytes: bytes,
    log_bytes: bytes = b"prior attempt log\n",
):
    out = root / corpus.OUT
    (out / "control").mkdir(parents=True)
    attempts = out / "attempts"
    attempts.mkdir()
    registration_path = out / "registration.json"
    registration_path.write_text("{}\n")
    output = attempts / "control-000-attempt-00.jsonl"
    log = attempts / "control-000-attempt-00.log"
    if output_bytes:
        output.write_bytes(output_bytes)
    log.write_bytes(log_bytes)
    command = corpus.make_command(root, "control", 0, output.relative_to(root))
    common = {
        "arm": "control",
        "shard": 0,
        "seed": corpus.BASE_SEED,
        "argv": command,
        "registration_sha256": corpus.digest(registration_path),
        "output": output.relative_to(root).as_posix(),
        "log": log.relative_to(root).as_posix(),
    }
    started = {**common, "event": "started", "started_at": 1.0}
    finished = {
        **common,
        "event": "finished",
        "exit_status": status,
        "output_bytes": len(output_bytes),
        "output_sha256": corpus.digest(output) if output_bytes else None,
        "log_sha256": corpus.digest(log),
        "elapsed_seconds": 0.1,
        "peak_sampled_process_tree_rss_bytes": 1024,
    }
    (out / "attempts.jsonl").write_text(
        json.dumps(started) + "\n" + json.dumps(finished) + "\n"
    )
    return out, output, log, started, finished


@pytest.mark.parametrize(
    ("status", "output_bytes"),
    [(1, b'{"partial":true}\n'), (1, b""), (2, b'{"partial":true}\n')],
)
def test_execute_rejects_failed_attempt_without_retry_authority_and_preserves_evidence(
    tmp_path, monkeypatch: pytest.MonkeyPatch, status: int, output_bytes: bytes
) -> None:
    out, output, log, _started, _finished = _write_prior_attempt(
        tmp_path, status=status, output_bytes=output_bytes
    )
    monkeypatch.setattr(corpus, "preflight", lambda _root: {"resource_launcher": []})
    launcher = Mock(side_effect=AssertionError("must not launch a replacement shard"))
    monkeypatch.setattr(corpus.subprocess, "Popen", launcher)
    evidence = [out / "attempts.jsonl", out / "registration.json", log]
    if output.exists():
        evidence.append(output)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in evidence}

    with pytest.raises(
        RuntimeError,
        match="seed549_(refuses_failed_or_partial_retry|zero_output_retry_not_registered)",
    ):
        corpus.execute(tmp_path)

    launcher.assert_not_called()
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in evidence
    }


def test_execute_authorizes_only_exact_registered_zero_output_retry(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _output, log, _started, finished = _write_prior_attempt(
        tmp_path, status=2, output_bytes=b""
    )
    registration_path = out / "registration.json"
    correction = {
        "registration_sha256": corpus.digest(registration_path),
        "source_path": "ml/alphazero_lite/seed549_corpus.py",
        "failed_attempt": {
            "arm": "control",
            "shard": 0,
            "seed": corpus.BASE_SEED,
            "exit_status": 2,
            "output_bytes": 0,
            "log_sha256": finished["log_sha256"],
        },
    }
    corpus.write_json(out / "execution-correction.json", correction)
    monkeypatch.setattr(corpus, "preflight", lambda _root: {"resource_launcher": []})

    class ExitedProcess:
        returncode = 1

        @staticmethod
        def poll():
            return 1

    launcher = Mock(return_value=ExitedProcess())
    monkeypatch.setattr(corpus.subprocess, "Popen", launcher)
    with pytest.raises(corpus.subprocess.CalledProcessError):
        corpus.execute(tmp_path)

    launcher.assert_called_once()
    events = corpus.read_jsonl(out / "attempts.jsonl")
    assert [event["event"] for event in events] == [
        "started",
        "finished",
        "retry_authorized",
        "started",
        "finished",
    ]
    assert events[2]["prior_attempt_log_sha256"] == finished["log_sha256"]
    assert events[2]["correction_sha256"] == corpus.digest(
        out / "execution-correction.json"
    )


def test_execute_rejects_tampered_retry_authorization_without_launch(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _output, log, _started, finished = _write_prior_attempt(
        tmp_path, status=2, output_bytes=b""
    )
    corpus.write_json(
        out / "execution-correction.json",
        {
            "registration_sha256": corpus.digest(out / "registration.json"),
            "source_path": "ml/alphazero_lite/seed549_corpus.py",
            "failed_attempt": {
                "arm": "control",
                "shard": 0,
                "seed": corpus.BASE_SEED,
                "exit_status": 2,
                "output_bytes": 0,
                "log_sha256": finished["log_sha256"],
            },
        },
    )
    events = corpus.read_jsonl(out / "attempts.jsonl")
    events.append(
        {
            "event": "retry_authorized",
            "arm": "control",
            "shard": 0,
            "attempt_number": 1,
            "prior_attempt_output_bytes": 0,
            "prior_attempt_exit_status": 2,
            "prior_attempt_log_sha256": finished["log_sha256"],
            "correction_sha256": "tampered",
        }
    )
    (out / "attempts.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    monkeypatch.setattr(corpus, "preflight", lambda _root: {"resource_launcher": []})
    launcher = Mock(side_effect=AssertionError("invalid authorization must not launch"))
    monkeypatch.setattr(corpus.subprocess, "Popen", launcher)
    evidence = [out / "attempts.jsonl", out / "registration.json", log]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in evidence}

    with pytest.raises(RuntimeError, match="seed549_retry_authorization_invalid"):
        corpus.execute(tmp_path)

    launcher.assert_not_called()
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in evidence
    }


def test_legacy_derivative_cli_refuses_to_overwrite_existing_publication(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / corpus.OUT
    out.mkdir(parents=True)
    protected = out / "control-source-manifest.json"
    protected.write_bytes(b"published manifest bytes\n")
    before = (protected.read_bytes(), protected.stat().st_mtime_ns)
    derivative_builder = Mock(
        side_effect=AssertionError("published data must not rebuild")
    )
    monkeypatch.setattr(
        "ml.alphazero_lite.seed549_derivatives.build", derivative_builder
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["seed549_corpus", "build-derivatives", "--root", str(tmp_path)],
    )

    with pytest.raises(FileExistsError, match="seed549_derivatives_already_published"):
        corpus.main()

    derivative_builder.assert_not_called()
    assert (protected.read_bytes(), protected.stat().st_mtime_ns) == before


@pytest.mark.parametrize("action", ["build-derivatives", "verify"])
def test_legacy_derivative_cli_delegates_to_published_implementation(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys, action: str
) -> None:
    sentinel = {"action": action, "delegated": True}
    delegated = Mock(return_value=sentinel)
    target = "build" if action == "build-derivatives" else "verify"
    monkeypatch.setattr(f"ml.alphazero_lite.seed549_derivatives.{target}", delegated)
    monkeypatch.setattr(
        sys, "argv", ["seed549_corpus", action, "--root", str(tmp_path)]
    )

    corpus.main()

    delegated.assert_called_once_with(tmp_path)
    assert json.loads(capsys.readouterr().out) == sentinel
