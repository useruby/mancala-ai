import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite.run_exact_solver_feasibility_preflight import (
    DEFAULT_SEED,
    generate_feasibility_corpus,
)
from ml.alphazero_lite.run_native_hybrid_feasibility import (
    NativeHybridProcess,
    NativeRequestError,
    run_mode,
    verify_frozen_corpus,
)


class NativeHybridFeasibilityTest(unittest.TestCase):
    def test_frozen_seed_271_corpus_identity(self) -> None:
        corpus = generate_feasibility_corpus(DEFAULT_SEED)
        self.assertEqual(
            "fe9c317f6dace7d5c77eb7214db4739643a5b7d3ac0b83a39be885119b459a59",
            verify_frozen_corpus(corpus),
        )

    def test_frozen_corpus_has_the_declared_buckets(self) -> None:
        corpus = generate_feasibility_corpus(DEFAULT_SEED)
        self.assertEqual(96, len(corpus))
        self.assertEqual(32, sum(17 <= row["stones_remaining"] <= 24 for row in corpus))
        self.assertEqual(32, sum(25 <= row["stones_remaining"] <= 32 for row in corpus))
        self.assertEqual(32, sum(33 <= row["stones_remaining"] <= 40 for row in corpus))

    def test_runner_timeout_replaces_process_and_persists_partial_rows(self) -> None:
        corpus = generate_feasibility_corpus(DEFAULT_SEED)[:2]
        executable = Path("native-probe-under-test")
        artifact = Path("artifact-under-test")
        response = {
            "action_values": {"0": 5},
            "exact_value": 5,
            "optimal_actions": [0],
            "metrics": {"cumulative_cache": {"tt_hits": 0}, "tt_hits": 0},
        }
        calls = {"requests": 0, "replaced": 0, "persisted": []}
        real_replace = NativeHybridProcess.replace

        def fake_replace(self) -> None:
            calls["replaced"] += 1
            real_replace(self)

        def fake_request(self, message, timeout):
            calls["requests"] += 1
            if calls["requests"] == 1:
                raise TimeoutError("native request exceeded timeout")
            return json.loads(json.dumps(response))

        def persist(report) -> None:
            calls["persisted"].append(len(report["corpus"]))

        with (
            patch.object(
                NativeHybridProcess,
                "start",
                lambda self: setattr(self, "process", object()),
            ),
            patch.object(
                NativeHybridProcess,
                "close",
                lambda self: setattr(self, "process", None),
            ),
            patch.object(NativeHybridProcess, "replace", fake_replace),
            patch.object(NativeHybridProcess, "request", fake_request),
        ):
            rows = run_mode(
                corpus,
                executable,
                artifact,
                0.01,
                False,
                {"corpus": [], "on_update": persist},
            )
        self.assertEqual([False, True], [row["exact"] for row in rows])
        self.assertEqual([1, 2], calls["persisted"])
        self.assertGreaterEqual(calls["replaced"], 1)

    def test_runner_protocol_failure_replaces_process_and_records_error(self) -> None:
        corpus = generate_feasibility_corpus(DEFAULT_SEED)[:1]
        with (
            patch.object(
                NativeHybridProcess,
                "start",
                lambda self: setattr(self, "process", object()),
            ),
            patch.object(
                NativeHybridProcess,
                "close",
                lambda self: setattr(self, "process", None),
            ),
            patch.object(
                NativeHybridProcess,
                "replace",
                lambda self: setattr(self, "replaced", True),
            ),
            patch.object(
                NativeHybridProcess,
                "request",
                lambda self, message, timeout: (_ for _ in ()).throw(
                    NativeRequestError("malformed native JSON")
                ),
            ),
        ):
            rows = run_mode(
                corpus,
                Path("native-probe-under-test"),
                Path("artifact-under-test"),
                0.01,
                False,
                {"corpus": [], "on_update": lambda report: None},
            )
        self.assertFalse(rows[0]["exact"])
        self.assertIn("malformed native JSON", rows[0]["error"])
        self.assertFalse(rows[0]["reproducible"])

    def test_runner_resume_does_not_change_completed_records(self) -> None:
        corpus = generate_feasibility_corpus(DEFAULT_SEED)[:2]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "partial.json"
            report: dict = {"corpus": [], "on_update": lambda value: None}
            with patch.object(
                NativeHybridProcess,
                "request",
                lambda self, message, timeout: (_ for _ in ()).throw(
                    TimeoutError("slow")
                ),
            ):
                with patch.object(
                    NativeHybridProcess,
                    "start",
                    lambda self: setattr(self, "process", object()),
                ):
                    with patch.object(
                        NativeHybridProcess,
                        "close",
                        lambda self: setattr(self, "process", None),
                    ):
                        with patch.object(
                            NativeHybridProcess, "replace", lambda self: None
                        ):
                            first = run_mode(
                                corpus[:1],
                                Path("native-probe-under-test"),
                                Path("artifact-under-test"),
                                0.01,
                                False,
                                report,
                            )
            output.write_text(json.dumps({"corpus": first}), encoding="utf-8")
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(1, len(saved["corpus"]))
            self.assertEqual(corpus[0]["id"], saved["corpus"][0]["id"])


if __name__ == "__main__":
    unittest.main()
