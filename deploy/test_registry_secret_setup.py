"""Tests for idempotent setup of the encrypted GHCR registry credential."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "setup_registry_secret.sh"
RECIPIENT = "age1testrecipient"


@pytest.fixture
def existing_secret_workspace(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    project = tmp_path / "project"
    deploy = project / "deploy"
    secret_dir = deploy / "secrets"
    fakebin = tmp_path / "bin"
    identity_dir = tmp_path / "home" / ".config" / "mancala-ai"
    secret_dir.mkdir(parents=True)
    fakebin.mkdir()
    identity_dir.mkdir(parents=True)

    shutil.copy2(SCRIPT, deploy / "setup_registry_secret.sh")
    (identity_dir / "age-identity.txt").write_text("test identity\n", encoding="utf-8")
    (secret_dir / "ghcr-recipient.txt").write_text(RECIPIENT + "\n", encoding="utf-8")
    (secret_dir / "ghcr-registry-password.age").write_text(
        "existing encrypted credential\n", encoding="utf-8"
    )

    for name, contents in {
        "age-keygen": """#!/bin/sh
if [ "$1" = "-y" ]; then printf '%s\\n' "$AGE_RECIPIENT"; exit 0; fi
exit 1
""",
        "age": """#!/bin/sh
if [ "${AGE_DECRYPT_STATUS:-0}" -ne 0 ]; then exit "$AGE_DECRYPT_STATUS"; fi
printf '%s' "$AGE_DECRYPTED"
""",
    }.items():
        executable = fakebin / name
        executable.write_text(contents, encoding="utf-8")
        executable.chmod(0o755)

    env = os.environ.copy() | {
        "HOME": str(tmp_path / "home"),
        "PATH": f"{fakebin}:{os.environ['PATH']}",
        "AGE_RECIPIENT": RECIPIENT,
        "AGE_DECRYPTED": "existing-token-value",
    }
    return project, env


def test_setup_reuses_existing_ciphertext_without_prompting_or_rotation(
    existing_secret_workspace: tuple[Path, dict[str, str]],
) -> None:
    project, env = existing_secret_workspace
    secret_file = project / "deploy/secrets/ghcr-registry-password.age"
    original_ciphertext = secret_file.read_bytes()

    result = subprocess.run(
        [str(project / "deploy/setup_registry_secret.sh")],
        cwd=project,
        env=env,
        input="",
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert secret_file.read_bytes() == original_ciphertext
    assert "GHCR token with write:packages permission" not in result.stderr
    assert "existing-token-value" not in result.stdout + result.stderr
    loader = (project / ".kamal/secrets.mancala-ai").read_text(encoding="utf-8")
    assert loader == "KAMAL_REGISTRY_PASSWORD=$KAMAL_REGISTRY_PASSWORD\n"
    assert (project / ".kamal/secrets.mancala-ai").stat().st_mode & 0o777 == 0o600


def test_setup_rejects_existing_ciphertext_that_cannot_be_decrypted(
    existing_secret_workspace: tuple[Path, dict[str, str]],
) -> None:
    project, env = existing_secret_workspace
    env["AGE_DECRYPT_STATUS"] = "1"

    result = subprocess.run(
        [str(project / "deploy/setup_registry_secret.sh")],
        cwd=project,
        env=env,
        input="",
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "could not decrypt existing GHCR registry credential" in result.stderr
    assert not (project / ".kamal/secrets.mancala-ai").exists()
