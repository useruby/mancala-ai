"""End-to-end Kamal 2.5.2 lifecycle against a disposable SSH/Docker target."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from deploy.assemble_release import artifact_identity, digest

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, env: dict[str, str], timeout: int = 600) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def request(base: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if data else {}
    with urlopen(
        Request(base + path, data=data, headers=headers), timeout=10
    ) as response:
        return json.loads(response.read())


def wait_ready(base: str, expected_profile: str, timeout: int = 240) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            request(base, "/health/ready")
            identity = request(base, "/health/identity")
            if expected_profile in identity["profiles"]:
                return identity
        except (OSError, URLError, ValueError):
            pass
        time.sleep(1)
    raise AssertionError(f"service failed to become ready as {expected_profile}")


def revision_release(source: Path, destination: Path, suffix: str) -> dict:
    artifact_path = destination / "model-artifact/current"
    artifact_path.mkdir(parents=True)
    for filename in ("metadata.json", "weights.json"):
        shutil.copyfile(source / filename, artifact_path / filename)
    metadata_path = artifact_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["deployment_fixture_revision"] = suffix
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    artifact_digest = artifact_identity(artifact_path)
    profile_name = f"balanced-{suffix}@v1"
    bundle_name = f"fixture-{suffix}"
    config = {
        "asset_root": "/assets",
        "worker_count": 1,
        "startup_timeout": 45,
        "execution_timeout": 30,
        "shutdown_timeout": 10,
        "bundles": {
            "fixture": {
                "bundle_id": bundle_name,
                "kind": "single",
                "artifacts": ["model-artifact/current"],
                "expected_artifact_sha256": [artifact_digest],
                "policy_source": "current",
                "value_source": "current",
            }
        },
        "profiles": {
            profile_name: {
                "name": profile_name,
                "profile_id": profile_name,
                "bundle_id": "fixture",
                "require_runtime_search_policy": False,
                "settings": {
                    "simulations": 2,
                    "c_puct": 1.25,
                    "fpu_mode": "zero",
                    "reuse_subtree": False,
                    "normalize_values": False,
                    "root_policy_mode": "deterministic",
                    "tactical_root_bias": 0.0,
                },
            }
        },
    }
    config_path = destination / "service.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest = {
        "revision": suffix,
        "files": {
            "metadata.json": digest(metadata_path),
            "weights.json": digest(artifact_path / "weights.json"),
            "service.json": digest(config_path),
        },
        "artifact_sha256": artifact_digest,
    }
    (destination / "release-manifest.json").write_text(json.dumps(manifest))
    freeze_release(destination)
    return {
        "profile": profile_name,
        "bundle": bundle_name,
        "artifact": artifact_digest,
    }


def freeze_release(release: Path) -> None:
    for path in release.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted(
        release.rglob("*"), key=lambda item: len(item.parts), reverse=True
    ):
        if path.is_dir():
            path.chmod(0o555)
    release.chmod(0o555)


def thaw_release(release: Path) -> None:
    release.chmod(0o755)
    for path in release.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
        else:
            path.chmod(0o644)


def assert_identity(identity: dict, expected: dict) -> None:
    profile = identity["profiles"][expected["profile"]]
    assert profile["profile_id"] == expected["profile"]
    bundle = identity["bundles"][profile["bundle_id"]]
    assert bundle["bundle_id"] == expected["bundle"]
    assert bundle["artifact_sha256"] == [expected["artifact"]]


def move(base: str, profile: str, request_id: str) -> dict:
    body = {
        "request_id": request_id,
        "game_revision": 1,
        "state": {
            "player0_pits": [4, 3, 2, 1, 0, 5],
            "player1_pits": [1, 2, 3, 4, 5, 6],
            "player0_store": 6,
            "player1_store": 6,
            "current_player": 0,
        },
        "profile": profile,
        "seed": 17,
    }
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            return request(base, "/v1/move", body)
        except HTTPError as error:
            if error.code != 503:
                raise
            time.sleep(0.5)
    raise AssertionError(f"search worker never served {request_id}")


def docker(*args: str) -> str:
    result = subprocess.run(["docker", *args], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"docker {' '.join(args)} failed ({result.returncode}): "
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def running_container() -> tuple[str, dict]:
    identifiers = docker(
        "ps", "--filter", "label=service=mancala-ai", "--format", "{{.ID}}"
    ).splitlines()
    assert len(identifiers) == 1, (
        f"expected one running service container: {identifiers}"
    )
    identifier = identifiers[0]
    return identifier, json.loads(docker("inspect", identifier))[0]


def main() -> None:
    source = Path(sys.argv[1]).resolve()
    port = int(os.environ.get("MANCALA_LIFECYCLE_PORT", "8080"))
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="kamal-lifecycle-") as temporary:
        root = Path(temporary)
        release_a = root / "release-a"
        release_b = root / "release-b"
        release_bad = root / "release-broken"
        expected_a = revision_release(source, release_a, "A")
        expected_b = revision_release(source, release_b, "B")
        shutil.copytree(release_b, release_bad)
        for path in release_bad.rglob("*"):
            if path.is_file():
                path.chmod(0o644)
            elif path.is_dir():
                path.chmod(0o755)
        release_bad.chmod(0o755)
        broken_config = json.loads((release_bad / "service.json").read_text())
        broken_config["profiles"] = {}
        (release_bad / "service.json").write_text(json.dumps(broken_config))
        freeze_release(release_bad)

        env = os.environ.copy()
        env["MANCALA_AI_DEPLOY_STATE"] = str(root / "state")
        env["MANCALA_AI_DEPLOY_LOCK"] = str(root / "lock")
        env["MANCALA_AI_ASSET_RELEASE"] = str(release_a)
        env["VERSION"] = "lifecycle-a"
        run(["deploy/deploy.sh"], env=env)
        identity_a = wait_ready(base, expected_a["profile"])
        assert_identity(identity_a, expected_a)

        container, inspect = running_container()
        mount = next(
            item for item in inspect["Mounts"] if item["Destination"] == "/assets"
        )
        assert mount["RW"] is False
        assert Path(mount["Source"]).resolve() == release_a.resolve()
        assert (
            move(base, expected_a["profile"], "initial-A")["result"]["kind"] == "move"
        )

        # Selecting/preparing B cannot retarget A's mounted release. Kill the
        # supervised search child to force Pebble to create its replacement.
        env["MANCALA_AI_ASSET_RELEASE"] = str(release_b)
        assert json.loads(docker("inspect", container))[0]["Mounts"]
        docker(
            "exec",
            container,
            "python",
            "-c",
            "import os,signal; found=[]; procs=[]\n"
            "for entry in os.listdir('/proc'):\n"
            " if entry.isdigit() and entry != '1':\n"
            "  try:\n"
            "   status=open(f'/proc/{entry}/status').read(); cmd=open(f'/proc/{entry}/cmdline','rb').read().replace(b'\\0',b' ').decode(errors='replace')\n"
            "   ppid=int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))\n"
            "   if ppid > 1: procs.append((int(entry),ppid,cmd))\n"
            "   if 'multiprocessing.spawn' in cmd and 'spawn_main' in cmd: found.append(int(entry))\n"
            "  except (OSError,StopIteration,ValueError): pass\n"
            "assert found, f'no spawned worker process; descendants={procs!r}'; os.kill(found[0],signal.SIGKILL)",
        )
        wait_ready(base, expected_a["profile"])
        assert (
            move(base, expected_a["profile"], "worker-restart-A")["result"]["kind"]
            == "move"
        )
        assert_identity(request(base, "/health/identity"), expected_a)

        env["VERSION"] = "lifecycle-b"
        run(["deploy/deploy.sh"], env=env)
        identity_b = wait_ready(base, expected_b["profile"])
        assert_identity(identity_b, expected_b)
        _, inspect_b = running_container()
        inspect_a = json.loads(docker("inspect", container))[0]
        stopped_at = inspect_a["State"]["FinishedAt"]
        started_at = inspect_b["State"]["StartedAt"]
        assert stopped_at != "0001-01-01T00:00:00Z"
        assert stopped_at <= started_at, (stopped_at, started_at)
        assert docker("port", inspect_b["Id"], "8080/tcp") == "127.0.0.1:8080"

        # Rollback uses the recorded previous image revision and asset path.
        run(["deploy/deploy.sh", "rollback"], env=env)
        identity_a = wait_ready(base, expected_a["profile"])
        assert_identity(identity_a, expected_a)

        env["MANCALA_AI_ASSET_RELEASE"] = str(release_bad)
        env["VERSION"] = "lifecycle-broken"
        try:
            run(["deploy/deploy.sh"], env=env)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "intentionally broken replacement unexpectedly deployed"
            )
        identity_a = wait_ready(base, expected_a["profile"])
        assert_identity(identity_a, expected_a)
        assert (
            move(base, expected_a["profile"], "recovered-A")["result"]["kind"] == "move"
        )

        _, recovered = running_container()
        recovered_mount = next(
            item for item in recovered["Mounts"] if item["Destination"] == "/assets"
        )
        assert recovered_mount["RW"] is False
        assert Path(recovered_mount["Source"]).resolve() == release_a.resolve()
        for release in (release_a, release_b, release_bad):
            thaw_release(release)


if __name__ == "__main__":
    main()
