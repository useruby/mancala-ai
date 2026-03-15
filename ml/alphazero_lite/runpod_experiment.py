"""RunPod experiment orchestration — bundle, launch, run, download, teardown."""

from __future__ import annotations

import glob
import io
import json
import os
import shlex
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Callable

# Repo root: ml/alphazero_lite/runpod_experiment.py → parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]

REMOTE_SCRIPT_PATHS = [
    "script/ai/runpod_remote_bootstrap.sh",
    "script/ai/runpod_remote_run.sh",
]

DEFAULT_IMAGE = "runpod/base:1.0.2-ubuntu2204"
DEFAULT_REMOTE_ROOT = "/root/runpod-experiment"
DEFAULT_SMALL_DISK_GB = 5
DEFAULT_PYTHON_DISK_GB = 20
DEFAULT_CPU_PROFILE = {
    "cpuFlavorIds": ["cpu3c"],
    "vcpuCount": 8,
}
SUPPORTED_CPU_PROFILES = {
    "cpu5c-16-32": {
        "cpuFlavorIds": ["cpu5c"],
        "vcpuCount": 16,
    },
    "cpu3c-16-32": {
        "cpuFlavorIds": ["cpu3c"],
        "vcpuCount": 16,
    },
}
REQUIRED_RAILS_BOOT_PATHS = ["app/middleware"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_bundle(
    bundle_path: str,
    command: str,
    results_path: str,
    include_paths: list[str],
) -> str:
    normalized = [_normalize_repo_path(p) for p in include_paths]
    all_paths = _augment_bundle_paths(normalized)
    for path in all_paths:
        if not (REPO_ROOT / path).exists():
            raise ValueError(f"Missing include path: {path}")
    entries = _expand_bundle_entries(all_paths)

    os.makedirs(os.path.dirname(bundle_path), exist_ok=True)

    manifest = {
        "command": command,
        "results_path": results_path,
        "install_python_requirements": _install_python_requirements(normalized),
        "install_ruby_dependencies": _install_ruby_dependencies(normalized),
        "ruby_version": _ruby_version(),
        "include_paths": normalized,
    }

    with open(bundle_path, "wb") as fh:
        with tarfile.open(fileobj=fh, mode="w:gz") as tar:
            _write_string_entry(tar, "manifest.json", json.dumps(manifest, indent=2))
            seen = set()
            for rel in list(dict.fromkeys(REMOTE_SCRIPT_PATHS + entries)):
                if rel in seen:
                    continue
                seen.add(rel)
                _add_file_entry(tar, rel, REPO_ROOT / rel)

    return bundle_path


def build_pod_request(
    name: str,
    include_paths: list[str] | None = None,
    image_name: str = DEFAULT_IMAGE,
    pod_profile: str | None = None,
) -> dict:
    include_paths = include_paths or []
    disk_gb = _recommended_disk_gb(include_paths)
    cpu_profile = _resolve_cpu_profile(pod_profile) if pod_profile else DEFAULT_CPU_PROFILE

    return {
        "name": name,
        "cloudType": "COMMUNITY",
        "computeType": "CPU",
        "cpuFlavorIds": cpu_profile["cpuFlavorIds"],
        "cpuFlavorPriority": "custom",
        "vcpuCount": cpu_profile["vcpuCount"],
        "containerDiskInGb": disk_gb,
        "volumeInGb": disk_gb,
        "imageName": image_name,
        "ports": ["22/tcp"],
        "supportPublicIp": True,
    }


def build_shell_plan(
    pod_id: str,
    bundle_path: str,
    local_results_path: str,
    remote_root: str,
    remote_results_path: str,
    ssh_info: dict,
) -> dict:
    user = ssh_info["user"]
    host = ssh_info["host"]
    port = ssh_info["port"]
    key_path = ssh_info["keyPath"]

    remote_bundle_path = f"{remote_root}/bundle.tar.gz"
    remote_manifest_path = f"{remote_root}/manifest.json"
    remote_results_full_path = f"{remote_root}/{remote_results_path}"
    ssh_target = f"{user}@{host}"

    ssh_flags = " ".join([
        "-p", shlex.quote(str(port)),
        "-i", shlex.quote(key_path),
        "-o", shlex.quote("IdentitiesOnly=yes"),
        "-o", shlex.quote("StrictHostKeyChecking=no"),
        "-o", shlex.quote("UserKnownHostsFile=/dev/null"),
    ])
    scp_flags = [
        "-P", str(port),
        "-i", key_path,
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
    ]

    return {
        "ssh_info_command": ["runpodctl", "ssh", "info", pod_id],
        "mkdir_command": _ssh_command(
            ssh_flags, ssh_target,
            f"mkdir -p {shlex.quote(remote_root)}"
        ),
        "upload_command": " ".join(
            shlex.quote(p) for p in
            ["scp"] + scp_flags + [bundle_path, f"{ssh_target}:{remote_bundle_path}"]
        ),
        "unpack_command": _ssh_command(
            ssh_flags, ssh_target,
            f"mkdir -p {shlex.quote(remote_root)} && "
            f"tar -xzf {shlex.quote(remote_bundle_path)} -C {shlex.quote(remote_root)}"
        ),
        "bootstrap_command": _ssh_command(
            ssh_flags, ssh_target,
            f"cd {shlex.quote(remote_root)} && "
            f"bash script/ai/runpod_remote_bootstrap.sh {shlex.quote(remote_root)}"
        ),
        "run_command": _ssh_command(
            ssh_flags, ssh_target,
            f"cd {shlex.quote(remote_root)} && "
            f"bash script/ai/runpod_remote_run.sh "
            f"{shlex.quote(remote_root)} {shlex.quote(remote_manifest_path)}"
        ),
        "download_command": " ".join(
            shlex.quote(p) for p in
            ["scp"] + scp_flags + ["-r",
             f"{ssh_target}:{remote_results_full_path}",
             local_results_path]
        ),
        "delete_command": f"runpodctl pod delete {pod_id}",
    }


def build_dry_run_plan(
    name: str,
    command: str,
    bundle_path: str,
    results_path: str,
    local_results_path: str,
    include_paths: list[str],
    api_key_env: str,
    run_timeout_seconds: int | None = None,
    image_name: str = DEFAULT_IMAGE,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    pod_profile: str | None = None,
    keep_pod_on_failure: bool = False,
) -> dict:
    normalized = [_normalize_repo_path(p) for p in include_paths]
    pod_request = build_pod_request(
        name=name,
        include_paths=normalized,
        image_name=image_name,
        pod_profile=pod_profile,
    )
    return {
        "name": name,
        "command": command,
        "bundle_path": bundle_path,
        "results_path": results_path,
        "local_results_path": local_results_path,
        "include_paths": normalized,
        "remote_root": remote_root,
        "run_timeout_seconds": run_timeout_seconds,
        "keep_pod_on_failure": keep_pod_on_failure,
        "pod_request": pod_request,
        "create_pod_command": build_create_pod_command(
            api_key_env=api_key_env,
            pod_request=pod_request,
        ),
        "shell_plan": build_shell_plan(
            pod_id="<pod-id>",
            bundle_path=bundle_path,
            local_results_path=local_results_path,
            remote_root=remote_root,
            remote_results_path=results_path,
            ssh_info={
                "user": "root",
                "host": "<public-host>",
                "port": 22,
                "keyPath": "<ssh-key-path>",
            },
        ),
    }


def build_create_pod_command(api_key_env: str, pod_request: dict) -> str:
    payload = json.dumps(pod_request, separators=(",", ":"))
    return (
        "curl -fsSL"
        " -X POST"
        ' -H "Content-Type: application/json"'
        f' -H "Authorization: Bearer ${{{api_key_env}}}"'
        " https://rest.runpod.io/v1/pods"
        f" --data {_single_quote(payload)}"
    )


def parse_pod_id(response_body: str) -> str:
    return json.loads(response_body)["id"]


def parse_ssh_info(response_body: str) -> dict:
    payload = json.loads(response_body)

    if "host" in payload:
        return {k: payload[k] for k in ("host", "port", "user", "keyPath") if k in payload}

    if "ip" in payload:
        ssh_info = _parse_ssh_command(
            payload.get("ssh_command") or payload.get("command") or payload.get("sshCommand") or ""
        )
        ssh_info["host"] = ssh_info.get("host") or payload["ip"]
        ssh_info["port"] = ssh_info.get("port") or payload.get("port")
        if not ssh_info.get("keyPath"):
            ssh_info["keyPath"] = (payload.get("ssh_key") or {}).get("path")
        return ssh_info

    command = payload.get("command") or payload.get("sshCommand") or ""
    if not command.strip():
        raise ValueError("Missing ssh command")
    return _parse_ssh_command(command)


def run_command(command: str) -> str:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command}\n{detail}"
        )
    return result.stdout


def orchestrate(
    name: str,
    command: str,
    bundle_path: str,
    results_path: str,
    local_results_path: str,
    include_paths: list[str],
    api_key_env: str,
    run_timeout_seconds: int | None = None,
    image_name: str = DEFAULT_IMAGE,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    pod_profile: str | None = None,
    keep_pod_on_failure: bool = False,
    bundle_builder: Callable | None = None,
    command_runner: Callable | None = None,
    timeout_runner: Callable | None = None,
    sleeper: Callable | None = None,
) -> dict:
    if bundle_builder is None:
        bundle_builder = build_bundle
    if command_runner is None:
        command_runner = run_command
    if timeout_runner is None:
        def timeout_runner(seconds, fn):
            return fn()
    if sleeper is None:
        sleeper = time.sleep

    normalized = [_normalize_repo_path(p) for p in include_paths]

    built_bundle_path = bundle_builder(
        bundle_path=bundle_path,
        command=command,
        results_path=results_path,
        include_paths=normalized,
    )

    pod_request = build_pod_request(
        name=name,
        include_paths=normalized,
        image_name=image_name,
        pod_profile=pod_profile,
    )

    pod_id = None
    shell_plan = None
    delete_pod = True

    try:
        create_response = command_runner(
            build_create_pod_command(api_key_env=api_key_env, pod_request=pod_request)
        )
        pod_id = parse_pod_id(create_response)

        ssh_info = wait_for_ssh_info(pod_id, command_runner=command_runner, sleeper=sleeper)
        shell_plan = build_shell_plan(
            pod_id=pod_id,
            bundle_path=built_bundle_path,
            local_results_path=local_results_path,
            remote_root=remote_root,
            remote_results_path=results_path,
            ssh_info=ssh_info,
        )

        for step_command in [
            shell_plan["mkdir_command"],
            shell_plan["upload_command"],
            shell_plan["unpack_command"],
            shell_plan["bootstrap_command"],
        ]:
            command_runner(step_command)

        run_cmd = shell_plan["run_command"]
        if run_timeout_seconds:
            timeout_runner(run_timeout_seconds, lambda: command_runner(run_cmd))
        else:
            command_runner(run_cmd)

        _reset_local_results_dir(local_results_path, results_path)
        command_runner(shell_plan["download_command"])
        downloaded = _inspect_downloaded_results(local_results_path, results_path)

        return {
            "pod_id": pod_id,
            "bundle_path": built_bundle_path,
            "shell_plan": shell_plan,
            "experiment_report_path": downloaded["experiment_report_path"],
            "experiment_passed": downloaded["experiment_passed"],
            "manifest_path": downloaded["manifest_path"],
            "manifest_status": downloaded["manifest_status"],
        }

    except Exception:
        if shell_plan is not None:
            delete_pod = not keep_pod_on_failure
            try:
                _reset_local_results_dir(local_results_path, results_path)
                command_runner(shell_plan["download_command"])
                downloaded = _inspect_downloaded_results(local_results_path, results_path)
                if downloaded["execution_completed"]:
                    delete_pod = True
                    return {
                        "pod_id": pod_id,
                        "bundle_path": built_bundle_path,
                        "shell_plan": shell_plan,
                        "experiment_report_path": downloaded["experiment_report_path"],
                        "experiment_passed": downloaded["experiment_passed"],
                        "manifest_path": downloaded["manifest_path"],
                        "manifest_status": downloaded["manifest_status"],
                    }
                delete_pod = not keep_pod_on_failure
            except Exception:
                pass
        raise

    finally:
        if pod_id and delete_pod:
            command_runner(f"runpodctl pod delete {pod_id}")


def wait_for_ssh_info(
    pod_id: str,
    max_attempts: int = 12,
    sleep_seconds: float = 5,
    command_runner: Callable | None = None,
    sleeper: Callable | None = None,
) -> dict:
    if command_runner is None:
        command_runner = run_command
    if sleeper is None:
        sleeper = time.sleep

    attempt = 0
    while True:
        attempt += 1
        try:
            return parse_ssh_info(command_runner(f"runpodctl ssh info {pod_id}"))
        except Exception:
            if attempt >= max_attempts:
                raise
            sleeper(sleep_seconds)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normalize_repo_path(path: str) -> str:
    return path.lstrip("./").lstrip("/") if path.startswith("./") else path


def _augment_bundle_paths(include_paths: list[str]) -> list[str]:
    if not _install_ruby_dependencies(include_paths):
        return include_paths
    return list(dict.fromkeys(include_paths + REQUIRED_RAILS_BOOT_PATHS))


def _expand_bundle_entries(include_paths: list[str]) -> list[str]:
    entries = []
    for rel in include_paths:
        source = REPO_ROOT / rel
        if source.is_dir():
            for entry in sorted(source.rglob("*")):
                if entry.is_file():
                    entries.append(str(entry.relative_to(REPO_ROOT)))
        else:
            entries.append(rel)
    return entries


def _install_python_requirements(include_paths: list[str]) -> bool:
    for path in include_paths:
        candidate = REPO_ROOT / path
        if path == "ml/alphazero_lite/requirements.txt":
            return True
        if candidate.is_dir() and (candidate / "requirements.txt").exists():
            return True
    return False


def _install_ruby_dependencies(include_paths: list[str]) -> bool:
    ruby_files = {"Gemfile", "Gemfile.lock", "Rakefile", "bin/rails"}
    ruby_dirs = {"app/models", "config", "lib/tasks"}
    for path in include_paths:
        if path in ruby_files:
            return True
        candidate = REPO_ROOT / path
        if candidate.is_dir() and path in ruby_dirs:
            return True
    return False


def _recommended_disk_gb(include_paths: list[str]) -> int:
    return DEFAULT_PYTHON_DISK_GB if _install_python_requirements(include_paths) else DEFAULT_SMALL_DISK_GB


def _resolve_cpu_profile(pod_profile: str) -> dict:
    if pod_profile not in SUPPORTED_CPU_PROFILES:
        raise ValueError(f"Unsupported pod profile: {pod_profile}")
    return SUPPORTED_CPU_PROFILES[pod_profile]


def _ruby_version() -> str:
    version = (REPO_ROOT / ".ruby-version").read_text(encoding="utf-8").strip()
    return version.removeprefix("ruby-")


def _add_file_entry(tar: tarfile.TarFile, rel_path: str, source_path: Path) -> None:
    info = tarfile.TarInfo(name=rel_path)
    data = source_path.read_bytes()
    info.size = len(data)
    info.mode = source_path.stat().st_mode
    tar.addfile(info, io.BytesIO(data))


def _write_string_entry(tar: tarfile.TarFile, rel_path: str, contents: str) -> None:
    data = contents.encode("utf-8")
    info = tarfile.TarInfo(name=rel_path)
    info.size = len(data)
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def _reset_local_results_dir(local_results_path: str, results_path: str) -> None:
    import shutil
    os.makedirs(local_results_path, exist_ok=True)
    stale = os.path.join(local_results_path, os.path.basename(results_path))
    if os.path.exists(stale):
        shutil.rmtree(stale)


def _inspect_downloaded_results(local_results_path: str, results_path: str) -> dict:
    empty = {
        "execution_completed": False,
        "experiment_report_path": None,
        "experiment_passed": None,
        "manifest_path": None,
        "manifest_status": None,
    }
    try:
        results_dir = os.path.join(local_results_path, os.path.basename(results_path))
        report_path = os.path.join(results_dir, "local_promotion_gate.json")
        if os.path.exists(report_path):
            payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
            return {
                "execution_completed": True,
                "experiment_report_path": report_path,
                "experiment_passed": payload.get("passed"),
                "manifest_path": None,
                "manifest_status": None,
            }

        manifests = glob.glob(os.path.join(results_dir, "**", "run_manifest.json"), recursive=True)
        if manifests:
            manifest_path = manifests[0]
            payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            return {
                "execution_completed": False,
                "experiment_report_path": None,
                "experiment_passed": None,
                "manifest_path": manifest_path,
                "manifest_status": payload.get("status"),
            }

        return empty
    except (json.JSONDecodeError, OSError):
        return empty


def _ssh_command(ssh_flags: str, ssh_target: str, remote_command: str) -> str:
    return f"ssh {ssh_flags} {shlex.quote(ssh_target)} {_single_quote(remote_command)}"


def _parse_ssh_command(command: str) -> dict:
    parts = shlex.split(command)
    port = None
    key_path = None
    target = None

    i = 0
    while i < len(parts):
        if parts[i] == "-p" and i + 1 < len(parts):
            port = int(parts[i + 1])
            i += 2
        elif parts[i] == "-i" and i + 1 < len(parts):
            key_path = parts[i + 1]
            i += 2
        elif "@" in parts[i] and not parts[i].startswith("-"):
            target = parts[i]
            i += 1
        else:
            i += 1

    user, _, host = (target or "").partition("@")
    return {
        "host": host or None,
        "port": port or 22,
        "user": user or None,
        "keyPath": key_path,
    }


def _single_quote(value: str) -> str:
    escaped = value.replace("'", "'\\''")
    return f"'{escaped}'"
