"""Authenticated runtime reader for the published seed466 bundle."""

from __future__ import annotations

import hashlib
import argparse
from pathlib import Path

import numpy as np

from ml.alphazero_lite.seed466_replay_bundle import read_bundle

MANIFEST_SHA256 = "db80f3bdfcc0a126032c1f5a16e78cda74b160d4fa2d80330cec8bed18085b34"


def read_authenticated_bundle(
    path: Path, expected_manifest_sha256: str
) -> dict[str, dict[str, np.ndarray]]:
    """Authenticate exact manifest bytes, then apply the bundle schema checks."""
    if len(expected_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_manifest_sha256
    ):
        raise ValueError("invalid_expected_manifest_sha256")
    manifest_bytes = (path / "manifest.json").read_bytes()
    actual = hashlib.sha256(manifest_bytes).hexdigest()
    if actual != expected_manifest_sha256:
        raise ValueError("manifest_sha256_mismatch")
    return read_bundle(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    arrays = read_authenticated_bundle(args.bundle, args.expected_manifest_sha256)
    print(
        {
            name: {key: list(value.shape) for key, value in members.items()}
            for name, members in arrays.items()
        }
    )


if __name__ == "__main__":
    main()
