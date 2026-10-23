"""Deterministic fault-injection tests for the deployment shell wrapper."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def wrapper(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "commands.log"
    for name, content in {
        "ssh": """#!/bin/sh
printf 'ssh %s\\n' "$*" >> "$COMMAND_LOG"
case "$*" in
  *"docker ps"*) printf 'old-container\\n' ;;
  *"docker inspect"*) printf 'KAMAL_VERSION=%s\\n' "${RUNNING_VERSION:-release-a}" ;;
  *"test -d"*) exit "${PREFLIGHT_STATUS:-0}" ;;
esac
""",
        "bundle": """#!/bin/sh
printf 'bundle %s\\n' "$*" >> "$COMMAND_LOG"
case "$*" in
  *"kamal deploy"*) exit "${DEPLOY_STATUS:-0}" ;;
  *"kamal rollback"*)
    if [ -n "${EXPECT_ROLLBACK_IMAGE:-}" ] && [ "$KAMAL_IMAGE" != "$EXPECT_ROLLBACK_IMAGE" ]; then
      echo "rollback used wrong image config: $KAMAL_IMAGE" >&2; exit 9
    fi
    exit "${ROLLBACK_STATUS:-0}" ;;
  *"kamal app stop"*) exit "${STOP_STATUS:-0}" ;;
esac
""",
        "python": f"""#!/bin/sh
case "$*" in
  *urllib.request*)
    if [ -n "${{FAIL_FIRST_READINESS:-}}" ]; then
      count=0; [ ! -f "$READINESS_COUNTER" ] || read count < "$READINESS_COUNTER"
      count=$((count + 1)); printf '%s' "$count" > "$READINESS_COUNTER"
      [ "$count" -ne 1 ] || exit 1
    fi
    exit 0 ;;
esac
exec {sys.executable} "$@"
""",
        "mv": """#!/bin/sh
case "$2" in
  "$MANCALA_AI_DEPLOY_STATE")
    if [ "${FAIL_STATE_WRITE:-0}" = 1 ]; then exit 1; fi ;;
esac
exec /usr/bin/mv "$@"
""",
    }.items():
        executable = bindir / name
        executable.write_text(content, encoding="utf-8")
        executable.chmod(0o755)

    assets = tmp_path / "remote-assets"
    assets.mkdir()
    config_store = tmp_path / "configs"
    previous = config_store / "release-a-config"
    previous.mkdir(parents=True)
    (previous / "deploy.yml").write_text("service: mancala-ai\n", encoding="utf-8")
    (previous / "values.sh").write_text(
        "export KAMAL_HOST=target.example\n"
        "export KAMAL_WIREGUARD_IP=127.0.0.1\n"
        "export KAMAL_SSH_USER=deploy\n"
        "export KAMAL_SSH_PORT=2222\n"
        f"export MANCALA_AI_ASSET_RELEASE={assets}\n"
        "export KAMAL_IMAGE=registry.example/a\n"
        "export KAMAL_REGISTRY_SERVER=registry.example\n"
        "export KAMAL_REGISTRY_USERNAME=robot\n",
        encoding="utf-8",
    )
    state = tmp_path / "state"
    state.write_text(f"release-a\t{assets}\trelease-a-config\t\t\t\n", encoding="utf-8")
    env = os.environ.copy() | {
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "COMMAND_LOG": str(log),
        "KAMAL_HOST": "target.example",
        "KAMAL_WIREGUARD_IP": "127.0.0.1",
        "KAMAL_SSH_USER": "deploy",
        "KAMAL_SSH_PORT": "2222",
        "KAMAL_IMAGE": "registry.example/b",
        "KAMAL_REGISTRY_SERVER": "registry.example",
        "KAMAL_REGISTRY_USERNAME": "robot",
        "KAMAL_REGISTRY_PASSWORD": "test-secret",
        "MANCALA_AI_ASSET_RELEASE": str(assets),
        "MANCALA_AI_DEPLOY_STATE": str(state),
        "MANCALA_AI_CONFIG_STORE": str(config_store),
        "MANCALA_AI_DEPLOY_LOCK": str(tmp_path / "lock"),
        "READINESS_COUNTER": str(tmp_path / "readiness-count"),
        "VERSION": "release-b",
    }
    return env, log


def invoke(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["deploy/deploy.sh", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )


def test_remote_asset_preflight_uses_explicit_ssh_target(wrapper) -> None:
    env, log = wrapper
    env["MANCALA_AI_ASSET_RELEASE"] = "/srv/remote assets/release-b"
    assert not Path(env["MANCALA_AI_ASSET_RELEASE"]).exists()
    result = invoke(env)
    assert result.returncode == 0, result.stderr
    commands = log.read_text(encoding="utf-8")
    assert "-l deploy -p 2222 target.example" in commands
    assert "/srv/remote assets/release-b" in commands


def test_failed_preflight_does_not_stop_incumbent(wrapper) -> None:
    env, log = wrapper
    env["PREFLIGHT_STATUS"] = "1"
    result = invoke(env)
    assert result.returncode != 0
    assert not log.exists() or "kamal app stop" not in log.read_text(encoding="utf-8")


def test_failure_after_replacement_start_restores_previous_release(wrapper) -> None:
    env, log = wrapper
    env["FAIL_FIRST_READINESS"] = "1"
    result = invoke(env)
    assert result.returncode != 0
    commands = log.read_text(encoding="utf-8")
    assert "kamal app stop" in commands
    assert "kamal rollback release-a" in commands
    assert "recovery failed" not in result.stderr
    assert Path(env["READINESS_COUNTER"]).read_text() == "2"


def test_deployment_state_write_failure_restores_previous_release(wrapper) -> None:
    env, log = wrapper
    env["FAIL_STATE_WRITE"] = "1"
    result = invoke(env)
    assert result.returncode != 0
    commands = log.read_text(encoding="utf-8")
    assert "kamal rollback release-a" in commands
    assert "release-a\t" in Path(env["MANCALA_AI_DEPLOY_STATE"]).read_text()


def test_target_drift_is_rejected_before_stop(wrapper) -> None:
    env, log = wrapper
    env["KAMAL_HOST"] = "other.example"
    result = invoke(env)
    assert result.returncode != 0
    assert "target drift" in result.stderr
    assert not log.exists() or "kamal app stop" not in log.read_text(encoding="utf-8")


def test_failed_recovery_is_reported(wrapper) -> None:
    env, log = wrapper
    env["DEPLOY_STATUS"] = "1"
    env["ROLLBACK_STATUS"] = "1"
    result = invoke(env)
    assert result.returncode != 0
    assert "recovery failed: Kamal could not restore" in result.stderr
    assert (
        Path(env["MANCALA_AI_DEPLOY_STATE"]).read_text().startswith("RECOVERY_FAILED\t")
    )


def test_rollback_restores_recorded_image_and_environment(wrapper) -> None:
    env, _ = wrapper
    # First deployment records B while preserving A as the rollback release.
    deployed = invoke(env)
    assert deployed.returncode == 0, deployed.stderr
    env["RUNNING_VERSION"] = "release-b"
    env["KAMAL_IMAGE"] = "registry.changed/now"
    env["KAMAL_REGISTRY_SERVER"] = "registry.changed"
    env["EXPECT_ROLLBACK_IMAGE"] = "registry.example/a"
    rolled_back = invoke(env, "rollback")
    assert rolled_back.returncode == 0, rolled_back.stderr

    state = Path(env["MANCALA_AI_DEPLOY_STATE"]).read_text(encoding="utf-8")
    current_config = state.split("\t")[2]
    restored = Path(env["MANCALA_AI_CONFIG_STORE"]) / current_config / "values.sh"
    contents = restored.read_text(encoding="utf-8")
    assert "KAMAL_IMAGE=registry.example/a" in contents
    assert "KAMAL_REGISTRY_PASSWORD" not in contents
