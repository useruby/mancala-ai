import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class CheckSuperhumanRegressionsScriptTest(unittest.TestCase):
    def write_fake_superhuman_module(self, root: Path, trace_path: Path) -> None:
        package_root = root / "ml" / "alphazero_lite"
        package_root.mkdir(parents=True)
        (root / "ml" / "__init__.py").write_text("", encoding="utf-8")
        (package_root / "__init__.py").write_text("", encoding="utf-8")
        (package_root / "superhuman_regressions.py").write_text(
            """
from __future__ import annotations

import json
import os
from pathlib import Path


def _trace(payload):
    trace_path = Path(os.environ[\"AZLITE_CHECK_SUPERHUMAN_REGRESSIONS_TRACE\"])
    current = {}
    if trace_path.exists():
        current = json.loads(trace_path.read_text(encoding=\"utf-8\"))
    current.update(payload)
    trace_path.write_text(json.dumps(current), encoding=\"utf-8\")


def load_regression_positions(path):
    _trace({\"positions_path\": str(path), \"loaded_positions_path\": str(path)})
    return [{\"id\": \"capture-1\"}]


def build_search_options(**kwargs):
    normalized = {
        \"normalized_by\": \"shared-module\",
        \"received\": kwargs,
    }
    _trace({\"builder_kwargs\": kwargs, \"normalized_search_options\": normalized})
    return normalized


def evaluate_regression_positions(**kwargs):
    _trace(
        {
            \"positions_path\": os.environ[\"AZLITE_CHECK_SUPERHUMAN_REGRESSIONS_POSITIONS\"],
            \"artifact_path\": str(kwargs[\"artifact_path\"]),
            \"simulations\": kwargs[\"simulations\"],
            \"seed\": kwargs[\"seed\"],
            \"c_puct\": kwargs[\"c_puct\"],
            \"evaluate_search_options\": kwargs[\"search_options\"],
            \"positions\": kwargs[\"positions\"],
        }
    )
    return [
        {
            \"id\": \"capture-1\",
            \"description\": \"Prefer the capture move.\",
            \"expected_move\": 1,
            \"acceptable_moves\": [1],
            \"selected_move\": 1,
            \"passed\": True,
            \"summary\": {\"selected_move\": 1},
        }
    ]


def build_regression_report(*, artifact_path, positions_path, results):
    return {
        \"passed\": bool(results) and all(result[\"passed\"] for result in results),
        \"artifact_path\": str(artifact_path),
        \"positions_path\": str(positions_path),
        \"results\": results,
    }
""".strip()
            + "\n",
            encoding="utf-8",
        )

    def assert_report_contract(self, report: dict) -> None:
        self.assertEqual(
            {"passed", "artifact_path", "positions_path", "results"}, set(report)
        )
        self.assertIsInstance(report["passed"], bool)
        self.assertIsInstance(report["artifact_path"], str)
        self.assertIsInstance(report["positions_path"], str)
        self.assertIsInstance(report["results"], list)

    def test_python_entrypoint_leaves_simulations_unset_when_flag_is_omitted(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "script/ai/check_superhuman_regressions"
        artifact_path = repo_root / "model-artifact/current"
        positions_path = (
            repo_root / "test/fixtures/ai/superhuman_regression_positions.json"
        )

        with tempfile.TemporaryDirectory(prefix="azlite-regressions-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            trace_path = tmp_path / "trace.json"
            shim_root = tmp_path / "shim"
            self.write_fake_superhuman_module(shim_root, trace_path)

            result = subprocess.run(
                [
                    os.environ.get("PYTHON", "python"),
                    str(script_path),
                    "--positions",
                    str(positions_path),
                    "--artifact",
                    str(artifact_path),
                    "--out",
                    str(out_path),
                ],
                cwd=repo_root,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(
                        filter(None, [str(shim_root), os.environ.get("PYTHONPATH")])
                    ),
                    "AZLITE_CHECK_SUPERHUMAN_REGRESSIONS_TRACE": str(trace_path),
                    "AZLITE_CHECK_SUPERHUMAN_REGRESSIONS_POSITIONS": str(
                        positions_path
                    ),
                },
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            seen = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertIsNone(seen["simulations"])

    def test_stub_mode_skips_rails_boot(self):
        with tempfile.TemporaryDirectory(prefix="azlite-regressions-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            result = subprocess.run(
                [
                    str(
                        Path(__file__).resolve().parents[2]
                        / "script/ai/check_superhuman_regressions"
                    ),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={
                    **os.environ,
                    "AZLITE_CHECK_SUPERHUMAN_REGRESSIONS_STUB": "1",
                    "BUNDLE_GEMFILE": str(tmp_path / "missing" / "Gemfile"),
                },
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            stdout_report = json.loads(result.stdout)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(stdout_report, report)
            self.assert_report_contract(report)
            self.assertTrue(report["passed"])
            self.assertEqual("model-artifact/current", report["artifact_path"])
            self.assertEqual(
                "test/fixtures/ai/superhuman_regression_positions.json",
                report["positions_path"],
            )
            self.assertEqual([], report["results"])

    def test_stub_mode_accepts_runtime_search_flags(self):
        with tempfile.TemporaryDirectory(prefix="azlite-regressions-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            result = subprocess.run(
                [
                    str(
                        Path(__file__).resolve().parents[2]
                        / "script/ai/check_superhuman_regressions"
                    ),
                    "--positions",
                    "test/fixtures/ai/superhuman_regression_positions.json",
                    "--artifact",
                    "model-artifact/current",
                    "--simulations",
                    "384",
                    "--reuse-subtree",
                    "--root-policy-mode",
                    "deterministic",
                    "--tactical-root-bias",
                    "0.1",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={
                    **os.environ,
                    "AZLITE_CHECK_SUPERHUMAN_REGRESSIONS_STUB": "1",
                    "BUNDLE_GEMFILE": str(tmp_path / "missing" / "Gemfile"),
                },
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            stdout_report = json.loads(result.stdout)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(stdout_report, report)
            self.assert_report_contract(report)
            self.assertTrue(report["passed"])
            self.assertEqual("model-artifact/current", report["artifact_path"])
            self.assertEqual(
                "test/fixtures/ai/superhuman_regression_positions.json",
                report["positions_path"],
            )
            self.assertEqual([], report["results"])

    def test_stub_mode_emits_contract_without_out_file(self):
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [str(repo_root / "script/ai/check_superhuman_regressions")],
            cwd=repo_root,
            env={
                **os.environ,
                "AZLITE_CHECK_SUPERHUMAN_REGRESSIONS_STUB": "1",
            },
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(0, result.returncode, msg=result.stderr)
        report = json.loads(result.stdout)
        self.assert_report_contract(report)
        self.assertTrue(report["passed"])
        self.assertEqual("model-artifact/current", report["artifact_path"])
        self.assertEqual(
            "test/fixtures/ai/superhuman_regression_positions.json",
            report["positions_path"],
        )
        self.assertEqual([], report["results"])

    def test_python_entrypoint_uses_shared_module_search_option_builder(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "script/ai/check_superhuman_regressions"
        artifact_path = repo_root / "model-artifact/current"
        positions_path = (
            repo_root / "test/fixtures/ai/superhuman_regression_positions.json"
        )

        with tempfile.TemporaryDirectory(prefix="azlite-regressions-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            trace_path = tmp_path / "trace.json"
            shim_root = tmp_path / "shim"
            self.write_fake_superhuman_module(shim_root, trace_path)

            result = subprocess.run(
                [
                    os.environ.get("PYTHON", "python"),
                    str(script_path),
                    "--positions",
                    str(positions_path),
                    "--artifact",
                    str(artifact_path),
                    "--simulations",
                    "384",
                    "--fpu-mode",
                    "absolute",
                    "--no-reuse-subtree",
                    "--no-normalize-values",
                    "--root-policy-mode",
                    "deterministic",
                    "--tactical-root-bias",
                    "0.1",
                    "--out",
                    str(out_path),
                ],
                cwd=repo_root,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(
                        filter(None, [str(shim_root), os.environ.get("PYTHONPATH")])
                    ),
                    "AZLITE_CHECK_SUPERHUMAN_REGRESSIONS_TRACE": str(trace_path),
                    "AZLITE_CHECK_SUPERHUMAN_REGRESSIONS_POSITIONS": str(
                        positions_path
                    ),
                },
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            seen = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(str(positions_path), seen["loaded_positions_path"])
            self.assertEqual(str(artifact_path), seen["artifact_path"])
            self.assertEqual(384, seen["simulations"])
            self.assertEqual(17, seen["seed"])
            self.assertEqual(1.25, seen["c_puct"])
            self.assertEqual(
                {
                    "fpu_mode": "absolute",
                    "reuse_subtree": False,
                    "normalize_values": False,
                    "root_policy_mode": "deterministic",
                    "tactical_root_bias": 0.1,
                },
                seen["builder_kwargs"],
            )
            self.assertEqual(
                {
                    "normalized_by": "shared-module",
                    "received": {
                        "fpu_mode": "absolute",
                        "reuse_subtree": False,
                        "normalize_values": False,
                        "root_policy_mode": "deterministic",
                        "tactical_root_bias": 0.1,
                    },
                },
                seen["normalized_search_options"],
            )
            self.assertEqual(
                seen["normalized_search_options"], seen["evaluate_search_options"]
            )
            self.assertEqual([{"id": "capture-1"}], seen["positions"])

            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assert_report_contract(report)
            self.assertTrue(report["passed"])
            self.assertEqual(str(artifact_path), report["artifact_path"])
            self.assertEqual(str(positions_path), report["positions_path"])
            self.assertEqual(1, len(report["results"]))

    def test_python_entrypoint_resolves_default_positions_outside_repo_root(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "script/ai/check_superhuman_regressions"
        artifact_path = repo_root / "model-artifact/current"
        positions_path = (
            repo_root / "test/fixtures/ai/superhuman_regression_positions.json"
        )

        with tempfile.TemporaryDirectory(prefix="azlite-regressions-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            trace_path = tmp_path / "trace.json"
            shim_root = tmp_path / "shim"
            self.write_fake_superhuman_module(shim_root, trace_path)

            result = subprocess.run(
                [
                    os.environ.get("PYTHON", "python"),
                    str(script_path),
                    "--artifact",
                    str(artifact_path),
                    "--out",
                    str(out_path),
                ],
                cwd=tmp_path,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(
                        filter(None, [str(shim_root), os.environ.get("PYTHONPATH")])
                    ),
                    "AZLITE_CHECK_SUPERHUMAN_REGRESSIONS_TRACE": str(trace_path),
                    "AZLITE_CHECK_SUPERHUMAN_REGRESSIONS_POSITIONS": str(
                        positions_path
                    ),
                },
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            seen = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(str(positions_path), seen["positions_path"])


if __name__ == "__main__":
    unittest.main()
