"""Regression tests for historical checkout import isolation."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

from ml.alphazero_lite import seed639_seed615_historical_adapter as adapter


class HistoricalImportIsolationTests(unittest.TestCase):
    """Ensure historical verification modules cannot resolve from Seed 639."""

    def _write_checkout(self, root: Path, *, marker: str) -> None:
        package = root / "ml/alphazero_lite"
        package.mkdir(parents=True)
        (root / "ml/__init__.py").write_text("")
        (package / "__init__.py").write_text("")
        (package / "seed615_evaluation.py").write_text(
            f'CHECKOUT_MARKER = "{marker}"\nOPENINGS = "openings"\nASSIGNMENTS = "assignments"\n'
        )
        (package / "verify_seed615_evaluation.py").write_text(
            f'CHECKOUT_MARKER = "{marker}"\n'
        )

    def test_historical_checkout_precedes_seed639_and_clears_cached_project_imports(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            historical = base / "historical"
            conflicting = base / "seed639"
            historical.mkdir()
            conflicting.mkdir()
            self._write_checkout(historical, marker="historical")
            self._write_checkout(conflicting, marker="seed639")
            conflicting_package = conflicting / "ml"
            current_sys_path = list(sys.path)
            try:
                sys.path.insert(0, str(conflicting))
                cached = types.ModuleType("ml.alphazero_lite.injected")
                cached.__file__ = str(conflicting_package / "foreign.py")
                sys.modules[cached.__name__] = cached
                contract, verifier, origins = adapter._historical_imports(historical)
                self.assertEqual(contract.CHECKOUT_MARKER, "historical")
                self.assertEqual(verifier.CHECKOUT_MARKER, "historical")
                self.assertTrue(
                    all(str(historical) in path for path in origins.values())
                )
                self.assertNotIn("ml.alphazero_lite.injected", sys.modules)
            finally:
                for name in tuple(sys.modules):
                    if name == "ml" or name.startswith("ml."):
                        del sys.modules[name]
                sys.path[:] = current_sys_path

    def test_foreign_transitive_project_module_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            historical = Path(temporary) / "historical"
            historical.mkdir()
            foreign = Path(temporary) / "foreign.py"
            foreign.write_text("# foreign checkout module\n")
            module = types.ModuleType("ml.alphazero_lite.foreign")
            module.__file__ = str(foreign)
            sys.modules[module.__name__] = module
            try:
                with self.assertRaisesRegex(
                    RuntimeError, "seed639_historical_module_origin_mismatch"
                ):
                    adapter._historical_module_identities(historical)
            finally:
                sys.modules.pop(module.__name__, None)


if __name__ == "__main__":
    unittest.main()
