"""Validate the single deployment destination with pinned Kamal."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LOCAL_GEMS = ROOT / ".tmp/ruby-gems"
BUNDLE = Path(
    shutil.which("bundle")
    or (Path.home() / ".local/share/mise/installs/ruby/3.3.6/bin/bundle")
)


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{BUNDLE.parent}:{env['PATH']}"
    if LOCAL_GEMS.exists():
        env["GEM_HOME"] = str(LOCAL_GEMS)
        env["GEM_PATH"] = str(LOCAL_GEMS)
        env["PATH"] = f"{LOCAL_GEMS / 'bin'}:{env['PATH']}"
    return env


def test_kamal_ai_config_render() -> None:
    env = command_env() | {
        "KAMAL_HOST": "192.0.2.1",
        "KAMAL_WIREGUARD_IP": "192.0.2.1",
        "MANCALA_AI_ASSET_RELEASE": "/srv/mancala-ai/releases/release-a",
        "KAMAL_IMAGE": "useruby/mancala-ai",
        "KAMAL_REGISTRY_SERVER": "ghcr.io",
        "KAMAL_REGISTRY_USERNAME": "ci-user",
        "KAMAL_REGISTRY_PASSWORD": "test-only",
    }
    destination = "mancala-ai"
    result = subprocess.run(
        [str(BUNDLE), "exec", "kamal", "config", "-d", destination],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    rendered = subprocess.run(
        [
            str(BUNDLE),
            "exec",
            "ruby",
            "-e",
            'require "kamal"; require "json"; c=Kamal::Configuration.create_from(config_file: Pathname("config/deploy.yml"), destination: ARGV.fetch(0)); r=c.role("web"); puts JSON.generate({options: r.option_args, proxy: r.proxy, volume_args: c.volume_args, drain_timeout: c.drain_timeout, readiness_delay: c.readiness_delay})',
            destination,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert ":repository: ghcr.io/useruby/mancala-ai" in result.stdout
    assert ":absolute_image: ghcr.io/useruby/mancala-ai:" in result.stdout
    assert re.search(r":version: [0-9a-f]{40}", result.stdout)
    options = json.loads(rendered.stdout)
    assert options["proxy"] is None
    assert "192.0.2.1:80:8080" in " ".join(options["options"])
    assert "/srv/mancala-ai/releases/release-a:/assets:ro" in " ".join(
        options["volume_args"]
    )
    assert options["drain_timeout"] == 45
    assert options["readiness_delay"] == 120
    for docker_option in (
        "--memory",
        "--cpus",
        "--health-cmd",
        "--health-interval",
        "--health-timeout",
        "--health-start-period",
        "--health-retries",
        "--stop-timeout",
    ):
        assert docker_option in options["options"]


@pytest.mark.parametrize(
    "missing_key",
    [
        "KAMAL_HOST",
        "KAMAL_WIREGUARD_IP",
        "MANCALA_AI_ASSET_RELEASE",
        "KAMAL_IMAGE",
        "KAMAL_REGISTRY_SERVER",
        "KAMAL_REGISTRY_USERNAME",
    ],
)
def test_kamal_requires_operator_values(missing_key: str) -> None:
    env = command_env() | {
        "KAMAL_HOST": "192.0.2.1",
        "KAMAL_WIREGUARD_IP": "192.0.2.1",
        "MANCALA_AI_ASSET_RELEASE": "/srv/mancala-ai/releases/release-a",
        "KAMAL_IMAGE": "useruby/mancala-ai",
        "KAMAL_REGISTRY_SERVER": "ghcr.io",
        "KAMAL_REGISTRY_USERNAME": "ci-user",
        "KAMAL_REGISTRY_PASSWORD": "test-only",
    }
    env.pop(missing_key)
    result = subprocess.run(
        [str(BUNDLE), "exec", "kamal", "config", "-d", "mancala-ai"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert missing_key in result.stdout + result.stderr


def test_wireguard_only_publish_and_runtime_controls() -> None:
    config = (ROOT / "config/deploy.yml").read_text(encoding="utf-8")
    assert 'publish: "<%= ENV.fetch("KAMAL_WIREGUARD_IP") %>:80:8080"' in config
    assert "proxy: false" in config
    assert (
        'volumes:\n  - "<%= ENV.fetch("MANCALA_AI_ASSET_RELEASE") %>:/assets:ro"'
        in config
    )
    for setting in (
        "memory:",
        "cpus:",
        "health-cmd:",
        "health-interval:",
        "health-timeout:",
        "health-start-period:",
        "health-retries:",
        "stop-timeout: 45",
        "deploy_timeout: 180",
        "readiness_delay: 120",
        "drain_timeout: 45",
        "minimum_version: 2.5.2",
        "require_destination: true",
    ):
        assert setting in config
    assert 'ENV.fetch("KAMAL_HOST")' in config
    assert 'ENV.fetch("KAMAL_REGISTRY_SERVER")' in config
    assert "- KAMAL_REGISTRY_PASSWORD" in config
