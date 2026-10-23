"""Assemble immutable serving assets from an explicitly trusted manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def artifact_identity(directory: Path) -> str:
    value = hashlib.sha256()
    value.update(b"azlite-serving-artifact-v1\0")
    for name in ("metadata.json", "weights.json"):
        path = directory / name
        value.update(name.encode("utf-8"))
        value.update(b"\0")
        value.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
    return value.hexdigest()


def assemble(manifest_path: Path, destination: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("trusted manifest must contain a non-empty files list")
    root = manifest_path.parent.resolve()
    validated: list[tuple[Path, Path, str]] = []
    destinations: set[str] = set()
    kinds: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "source",
            "destination",
            "sha256",
            "kind",
        }:
            raise ValueError(
                "each file entry requires source, destination, sha256, kind"
            )
        source = (root / entry["source"]).resolve()
        relative = Path(entry["destination"])
        expected = entry["sha256"]
        kind = entry["kind"]
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe release path: {relative}")
        if relative.as_posix() in {"", ".", "release-manifest.json"}:
            raise ValueError(f"reserved release path: {relative}")
        if relative.as_posix() in destinations:
            raise ValueError(f"duplicate release path: {relative}")
        destinations.add(relative.as_posix())
        if kind not in {
            "weight",
            "metadata",
            "policy",
            "native_probe",
            "tablebase",
            "config",
        }:
            raise ValueError(f"unsupported release file kind: {kind}")
        if not source.is_file():
            raise ValueError(f"required {kind} file is missing: {source}")
        if source.stat().st_size < 1024 and source.read_bytes().startswith(
            b"version https://git-lfs.github.com/spec/v1"
        ):
            raise ValueError(f"Git LFS pointer supplied instead of object: {source}")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or digest(source) != expected
        ):
            raise ValueError(f"SHA-256 mismatch for {source}")
        if kind == "native_probe":
            if not os.access(source, os.X_OK):
                raise ValueError(f"native probe is not executable: {source}")
        validated.append((source, relative, kind))
        kinds.add(kind)

    required = {"weight", "metadata", "policy", "native_probe", "tablebase", "config"}
    missing_kinds = required - kinds
    if missing_kinds:
        raise ValueError(
            f"manifest is missing required asset kinds: {sorted(missing_kinds)}"
        )

    destination = destination.resolve()
    if destination.exists():
        raise ValueError(f"release destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="release-", dir=destination.parent
    ) as scratch:
        stage = Path(scratch) / "assets"
        stage.mkdir()
        for source, relative, _kind in validated:
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            shutil.copystat(source, target)
        shutil.copyfile(manifest_path, stage / "release-manifest.json")
        native = next(
            stage / path for _src, path, kind in validated if kind == "native_probe"
        )
        tablebase = next(
            stage / path for _src, path, kind in validated if kind == "tablebase"
        )
        try:
            probe = subprocess.run(
                [str(native), "probe", str(tablebase)],
                input='{"pits":[0,0,0,0,0,0,0,0,0,0,0,0],"player":0}\n',
                text=True,
                capture_output=True,
                timeout=15,
                check=True,
            )
            response = json.loads(probe.stdout)
            if response.get("actions") != {}:
                raise ValueError("native exact-root probe failed terminal self-test")
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            raise ValueError("native probe/tablebase pair is unusable") from error
        by_kind = {kind: [] for kind in required}
        for _source, relative, kind in validated:
            by_kind[kind].append(relative)
        for policy_path in by_kind["policy"]:
            policy = json.loads((stage / policy_path).read_text(encoding="utf-8"))
            for name, kind in (
                ("native_probe", "native_probe"),
                ("tablebase", "tablebase"),
            ):
                reference = policy[name]
                resolved = (stage / reference["path"]).resolve()
                if (
                    not resolved.is_relative_to(stage.resolve())
                    or not resolved.is_file()
                ):
                    raise ValueError(
                        f"packaged policy {name} path is outside the release"
                    )
                if resolved not in {(stage / path).resolve() for path in by_kind[kind]}:
                    raise ValueError(
                        f"packaged policy {name} path does not select a manifest asset"
                    )
                if digest(resolved) != reference["sha256"]:
                    raise ValueError(
                        f"packaged policy {name} hash does not match its asset"
                    )
        for config_path in by_kind["config"]:
            config = json.loads((stage / config_path).read_text(encoding="utf-8"))
            required_config = {
                "asset_root",
                "worker_count",
                "startup_timeout",
                "execution_timeout",
                "shutdown_timeout",
                "bundles",
                "profiles",
            }
            if not required_config.issubset(config):
                raise ValueError(
                    f"service config must explicitly define {sorted(required_config)}"
                )
            if config.get("asset_root") != "/releases/current":
                raise ValueError(
                    "packaged service config asset_root must match the read-only deployment mount"
                )
            if not config.get("bundles") or not config.get("profiles"):
                raise ValueError(
                    "packaged service config must define bundles and profiles"
                )
            if (
                type(config.get("worker_count")) is not int
                or config["worker_count"] < 1
            ):
                raise ValueError(
                    "packaged service config requires an explicit positive worker_count"
                )
            if not 0 < config["shutdown_timeout"] <= 10:
                raise ValueError(
                    "service shutdown_timeout must fit the 10-second worker shutdown budget"
                )
            if not 0 < config["startup_timeout"] <= 120:
                raise ValueError(
                    "service startup_timeout must fit the 120-second health-check allowance"
                )
            if config["execution_timeout"] <= 0:
                raise ValueError("service execution_timeout must be positive")
            required_bundle = {
                "bundle_id",
                "kind",
                "artifacts",
                "expected_artifact_sha256",
                "expected_policy_sha256",
                "policy_source",
                "value_source",
            }
            required_profile = {
                "name",
                "profile_id",
                "bundle_id",
                "require_runtime_search_policy",
                "settings",
            }
            search_settings = {
                "simulations",
                "c_puct",
                "fpu_mode",
                "reuse_subtree",
                "normalize_values",
                "root_policy_mode",
                "tactical_root_bias",
            }
            for profile in config["profiles"].values():
                if not required_profile.issubset(
                    profile
                ) or not search_settings.issubset(profile["settings"]):
                    raise ValueError(
                        "each profile must explicitly define identity, policy requirement, and every search setting"
                    )
            for bundle in config["bundles"].values():
                if not required_bundle.issubset(bundle):
                    raise ValueError(
                        "each bundle must explicitly define composition and expected component/policy identities"
                    )
                for artifact in bundle.get("artifacts", []):
                    artifact_dir = (stage / artifact).resolve()
                    if not artifact_dir.is_relative_to(stage.resolve()):
                        raise ValueError(
                            "service config artifact path escapes release root"
                        )
                    if (
                        not (artifact_dir / "metadata.json").is_file()
                        or not (artifact_dir / "weights.json").is_file()
                    ):
                        raise ValueError(
                            f"service config artifact is absent from release: {artifact}"
                        )
                artifacts = bundle.get("artifacts")
                expected = bundle.get("expected_artifact_sha256")
                expected_policy = bundle.get("expected_policy_sha256", [])
                if (
                    not isinstance(artifacts, list)
                    or not isinstance(expected, list)
                    or len(artifacts) != len(expected)
                ):
                    raise ValueError(
                        "service config must authenticate each model component"
                    )
                if expected_policy and (
                    not isinstance(expected_policy, list)
                    or len(expected_policy) != len(artifacts)
                ):
                    raise ValueError(
                        "service config runtime-policy identities do not match components"
                    )
                for index, (artifact, identity) in enumerate(
                    zip(artifacts, expected, strict=True)
                ):
                    artifact_dir = (stage / artifact).resolve()
                    if artifact_identity(artifact_dir) != identity:
                        raise ValueError(
                            f"service config artifact identity mismatch: {artifact}"
                        )
                    policy_path = artifact_dir / "search_policy.json"
                    expected_hash = expected_policy[index] if expected_policy else None
                    if policy_path.is_file():
                        if (
                            not isinstance(expected_hash, str)
                            or digest(policy_path) != expected_hash
                        ):
                            raise ValueError(
                                f"service config policy identity mismatch: {artifact}"
                            )
                    elif expected_hash is not None:
                        raise ValueError(
                            f"service config expected missing search policy: {artifact}"
                        )
        for path in stage.rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
        for path in sorted(
            stage.rglob("*"), key=lambda item: len(item.parts), reverse=True
        ):
            if path.is_dir():
                path.chmod(0o555)
        stage.rename(destination)
        destination.chmod(0o555)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    assemble(args.manifest, args.destination)


if __name__ == "__main__":
    main()
