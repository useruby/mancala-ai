"""Uvicorn entry point loaded from a trusted JSON deployment configuration."""

from __future__ import annotations

import json
import os

from ml.alphazero_lite.serving.engine import (
    RuntimeBundle,
    SearchProfile,
    SearchSettings,
)
from ml.alphazero_lite.serving.http import ServiceConfig, create_app
from ml.alphazero_lite.serving.logging import configure_application_logging


configure_application_logging()


def load_config(path: str) -> ServiceConfig:
    """Read server-owned configuration; artifact identities must come from release manifests."""
    with open(path, encoding="utf-8") as stream:
        raw = json.load(stream)
    raw["asset_root"] = os.path.abspath(raw["asset_root"])
    for value in raw["bundles"].values():
        for key in ("artifacts", "expected_artifact_sha256", "expected_policy_sha256"):
            if key in value:
                value[key] = tuple(value[key])
    bundles = {key: RuntimeBundle(**value) for key, value in raw.pop("bundles").items()}
    profiles = {}
    for key, value in raw.pop("profiles").items():
        value["settings"] = SearchSettings(**value["settings"])
        profiles[key] = SearchProfile(**value)
    return ServiceConfig(bundles=bundles, profiles=profiles, **raw)


app = create_app(load_config(os.environ["MANCALA_AI_CONFIG"]))
