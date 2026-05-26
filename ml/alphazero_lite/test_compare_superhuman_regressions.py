import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class CompareSuperhumanRegressionsScriptTest(unittest.TestCase):
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
    trace_path = Path(os.environ["AZLITE_COMPARE_SUPERHUMAN_REGRESSIONS_TRACE"])
    current = {}
    if trace_path.exists():
        current = json.loads(trace_path.read_text(encoding="utf-8"))
    current.update(payload)
    trace_path.write_text(json.dumps(current), encoding="utf-8")


def load_regression_positions(path):
    _trace({"positions_path": str(path)})
    return [{"id": "capture-1"}]


def build_search_options(**kwargs):
    normalized = {
        "normalized_by": "shared-module",
        "received": kwargs,
    }
    _trace({"builder_kwargs": kwargs, "normalized_search_options": normalized})
    return normalized


def evaluate_regression_positions(**kwargs):
    trace_key = "baseline" if "current" in str(kwargs["artifact_path"]) else "candidate"
    _trace(
        {
            f"{trace_key}_artifact_path": str(kwargs["artifact_path"]),
            f"{trace_key}_simulations": kwargs["simulations"],
            f"{trace_key}_seed": kwargs["seed"],
            f"{trace_key}_c_puct": kwargs["c_puct"],
            f"{trace_key}_search_options": kwargs["search_options"],
            f"{trace_key}_positions": kwargs["positions"],
        }
    )
    selected_move = 0 if trace_key == "baseline" else 1
    passed = trace_key == "candidate"
    return [
        {
            "id": "capture-1",
            "description": "Prefer the capture move.",
            "expected_move": 1,
            "acceptable_moves": [1],
            "selected_move": selected_move,
            "passed": passed,
            "summary": {"selected_move": selected_move},
        }
    ]


def compare_regression_results(*, baseline_results, candidate_results):
    _trace(
        {
            "compare_baseline_results": baseline_results,
            "compare_candidate_results": candidate_results,
        }
    )
    return [
        {
            "id": "capture-1",
            "description": "Prefer the capture move.",
            "expected_move": 1,
            "acceptable_moves": [1],
            "baseline_selected_move": baseline_results[0]["selected_move"],
            "candidate_selected_move": candidate_results[0]["selected_move"],
            "baseline_passed": baseline_results[0]["passed"],
            "candidate_passed": candidate_results[0]["passed"],
            "improved": True,
            "regressed": False,
        }
    ]
""".strip()
            + "\n",
            encoding="utf-8",
        )

    def assert_report_contract(self, report: dict) -> None:
        self.assertEqual(
            {
                "baseline_artifact_path",
                "candidate_artifact_path",
                "positions_path",
                "comparisons",
            },
            set(report),
        )
        self.assertIsInstance(report["baseline_artifact_path"], str)
        self.assertIsInstance(report["candidate_artifact_path"], str)
        self.assertIsInstance(report["positions_path"], str)
        self.assertIsInstance(report["comparisons"], list)

    def test_direct_execution_stub_contract_writes_comparison_report(self):
        with tempfile.TemporaryDirectory(prefix="azlite-compare-regressions-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            result = subprocess.run(
                [
                    str(
                        Path(__file__).resolve().parents[2]
                        / "script/ai/compare_superhuman_regressions"
                    ),
                    "--positions",
                    "test/fixtures/ai/superhuman_regression_positions.json",
                    "--baseline-artifact",
                    "model-artifact/current",
                    "--candidate-artifact",
                    "model-artifact/candidate",
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
                    "AZLITE_COMPARE_SUPERHUMAN_REGRESSIONS_STUB": "1",
                    "BUNDLE_GEMFILE": str(tmp_path / "missing" / "Gemfile"),
                },
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(1, result.returncode, msg=result.stderr)
            stdout_report = json.loads(result.stdout)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(stdout_report, report)
            self.assert_report_contract(report)
            self.assertEqual("model-artifact/current", report["baseline_artifact_path"])
            self.assertEqual(
                "model-artifact/candidate", report["candidate_artifact_path"]
            )
            self.assertEqual(
                "test/fixtures/ai/superhuman_regression_positions.json",
                report["positions_path"],
            )
            self.assertEqual([], report["comparisons"])

    def test_python_entrypoint_uses_shared_module_search_option_builder(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "script/ai/compare_superhuman_regressions"
        baseline_artifact_path = repo_root / "model-artifact/current"
        candidate_artifact_path = repo_root / "model-artifact/candidate"
        positions_path = (
            repo_root / "test/fixtures/ai/superhuman_regression_positions.json"
        )

        with tempfile.TemporaryDirectory(prefix="azlite-compare-regressions-") as tmp:
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
                    "--baseline-artifact",
                    str(baseline_artifact_path),
                    "--candidate-artifact",
                    str(candidate_artifact_path),
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
                    "AZLITE_COMPARE_SUPERHUMAN_REGRESSIONS_TRACE": str(trace_path),
                },
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            seen = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(str(positions_path), seen["positions_path"])
            self.assertEqual(
                str(baseline_artifact_path), seen["baseline_artifact_path"]
            )
            self.assertEqual(
                str(candidate_artifact_path), seen["candidate_artifact_path"]
            )
            self.assertEqual(384, seen["baseline_simulations"])
            self.assertEqual(384, seen["candidate_simulations"])
            self.assertEqual(17, seen["baseline_seed"])
            self.assertEqual(17, seen["candidate_seed"])
            self.assertEqual(1.25, seen["baseline_c_puct"])
            self.assertEqual(1.25, seen["candidate_c_puct"])
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
                seen["normalized_search_options"], seen["baseline_search_options"]
            )
            self.assertEqual(
                seen["normalized_search_options"], seen["candidate_search_options"]
            )
            self.assertEqual([{"id": "capture-1"}], seen["baseline_positions"])
            self.assertEqual([{"id": "capture-1"}], seen["candidate_positions"])
            self.assertEqual(0, seen["compare_baseline_results"][0]["selected_move"])
            self.assertEqual(1, seen["compare_candidate_results"][0]["selected_move"])

            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assert_report_contract(report)
            self.assertEqual(
                str(baseline_artifact_path), report["baseline_artifact_path"]
            )
            self.assertEqual(
                str(candidate_artifact_path), report["candidate_artifact_path"]
            )
            self.assertEqual(str(positions_path), report["positions_path"])
            self.assertEqual(1, len(report["comparisons"]))

    def test_python_entrypoint_leaves_simulations_unset_when_flag_is_omitted(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "script/ai/compare_superhuman_regressions"
        baseline_artifact_path = repo_root / "model-artifact/current"
        candidate_artifact_path = repo_root / "model-artifact/candidate"
        positions_path = (
            repo_root / "test/fixtures/ai/superhuman_regression_positions.json"
        )

        with tempfile.TemporaryDirectory(prefix="azlite-compare-regressions-") as tmp:
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
                    "--baseline-artifact",
                    str(baseline_artifact_path),
                    "--candidate-artifact",
                    str(candidate_artifact_path),
                    "--out",
                    str(out_path),
                ],
                cwd=repo_root,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(
                        filter(None, [str(shim_root), os.environ.get("PYTHONPATH")])
                    ),
                    "AZLITE_COMPARE_SUPERHUMAN_REGRESSIONS_TRACE": str(trace_path),
                },
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            seen = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertIsNone(seen["baseline_simulations"])
            self.assertIsNone(seen["candidate_simulations"])

    def test_python_entrypoint_resolves_default_positions_outside_repo_root(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "script/ai/compare_superhuman_regressions"
        baseline_artifact_path = repo_root / "model-artifact/current"
        candidate_artifact_path = repo_root / "model-artifact/candidate"
        positions_path = (
            repo_root / "test/fixtures/ai/superhuman_regression_positions.json"
        )

        with tempfile.TemporaryDirectory(prefix="azlite-compare-regressions-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "report.json"
            trace_path = tmp_path / "trace.json"
            shim_root = tmp_path / "shim"
            self.write_fake_superhuman_module(shim_root, trace_path)

            result = subprocess.run(
                [
                    os.environ.get("PYTHON", "python"),
                    str(script_path),
                    "--baseline-artifact",
                    str(baseline_artifact_path),
                    "--candidate-artifact",
                    str(candidate_artifact_path),
                    "--out",
                    str(out_path),
                ],
                cwd=tmp_path,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(
                        filter(None, [str(shim_root), os.environ.get("PYTHONPATH")])
                    ),
                    "AZLITE_COMPARE_SUPERHUMAN_REGRESSIONS_TRACE": str(trace_path),
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
