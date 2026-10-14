"""Run frozen seed472 verifier tests against a temporary publication copy."""

from __future__ import annotations

import shutil
import os
from pathlib import Path

import pytest

from ml.alphazero_lite import verify_seed472_trajectory_attribution


class Seed472TemporaryPublication:
    """Redirect the unchanged seed472 mutation tests into a private copy."""

    def __init__(self) -> None:
        self.root = Path(__file__).resolve().parents[2]
        self.source = self.root / verify_seed472_trajectory_attribution.OUT
        self.destination = self.root / ".tmp" / f"seed472-test-copy-{os.getpid()}"

    def pytest_configure(self, config: pytest.Config) -> None:
        if self.destination.exists():
            raise RuntimeError(f"temporary_test_copy_exists:{self.destination}")
        shutil.copytree(self.source, self.destination, copy_function=shutil.copy2)
        verify_seed472_trajectory_attribution.OUT = self.destination

    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        for item in items:
            module = getattr(item, "module", None)
            if getattr(module, "__name__", "").endswith(
                "test_seed472_trajectory_attribution"
            ):
                module.OUT = self.destination

    def pytest_unconfigure(self, config: pytest.Config) -> None:
        shutil.rmtree(self.destination, ignore_errors=False)


_plugin = Seed472TemporaryPublication()


def pytest_configure(config: pytest.Config) -> None:
    _plugin.pytest_configure(config)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    _plugin.pytest_collection_modifyitems(items)


def pytest_unconfigure(config: pytest.Config) -> None:
    _plugin.pytest_unconfigure(config)
