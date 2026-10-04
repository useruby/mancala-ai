"""Regression checks for the frozen FF1536 continuation plan."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from ml.alphazero_lite import seed398_ff1536_analysis
from ml.alphazero_lite.seed398_ff1536_confirmation import (
    OUT,
    REFERENCES,
    _json,
    _json_lines,
    build_reuse_plan,
    frozen_actions,
)
from ml.alphazero_lite.verify_seed398_ff1536_confirmation import verify
import ml.alphazero_lite.verify_seed398_ff1536_confirmation as verifier
from ml.alphazero_lite.seed398_ff1536_probe_validation import (
    derive_ff1536_actions,
    validate_probe_bundle,
)


class FF1536ConfirmationTests(unittest.TestCase):
    def test_frozen_ff_probe_derivation_rejects_incomplete_or_inconsistent_evidence(
        self,
    ) -> None:
        raw_path = (
            Path(__file__).resolve().parents[2]
            / "docs/data/seed398-search-budget-persistence/raw-probes.json"
        )
        raw = json.loads(raw_path.read_text())
        ff_registration = _json(OUT / "registration.json")
        registration_path = raw_path.with_name("registration.json")
        raw_bytes = raw_path.read_bytes()
        registration_bytes = registration_path.read_bytes()
        self.assertEqual(
            validate_probe_bundle(
                raw_bytes, ff_registration["probe_sha256"], registration_bytes
            ),
            raw,
        )
        states = _json(
            Path(__file__).resolve().parents[2]
            / "docs/data/seed398-paired-first-action/registration.json"
        )["states"]
        publication_names = (
            "registration.json",
            "new-outcomes.jsonl",
            "analysis.json",
            "provenance-matrix.json",
            "results.md",
            "publication-binding.json",
        )
        published = {name: (OUT / name).read_bytes() for name in publication_names}
        result = derive_ff1536_actions(states, raw)
        self.assertEqual(len(result), 64)
        self.assertEqual(
            [
                {
                    key: row[key]
                    for key in ("opening_index", "state_hash", "ff1536_action")
                }
                for row in ff_registration["action_mapping"]
            ],
            result,
        )
        malformed = json.loads(json.dumps(raw))
        malformed["probes"].remove(
            next(row for row in malformed["probes"] if row["treatment"] == "FF")
        )
        with self.assertRaisesRegex(ValueError, "coverage"):
            derive_ff1536_actions(states, malformed)
        malformed = json.loads(json.dumps(raw))
        ff = next(row for row in malformed["probes"] if row["treatment"] == "FF")
        malformed["probes"].append(json.loads(json.dumps(ff)))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            derive_ff1536_actions(states, malformed)
        for mutate, message in (
            (
                lambda probe: probe["snapshots"].remove(
                    next(s for s in probe["snapshots"] if s["simulations"] == 1536)
                ),
                "snapshot_count",
            ),
            (
                lambda probe: probe["snapshots"].append(
                    json.loads(
                        json.dumps(
                            next(
                                s
                                for s in probe["snapshots"]
                                if s["simulations"] == 1536
                            )
                        )
                    )
                ),
                "snapshot_count",
            ),
            (lambda probe: probe.update(state_hash="0" * 64), "state_identity"),
            (
                lambda probe: probe.update(
                    final_action=(probe["final_action"] + 1) % 6
                ),
                "snapshot_mismatch",
            ),
        ):
            malformed = json.loads(json.dumps(raw))
            probe = next(row for row in malformed["probes"] if row["treatment"] == "FF")
            mutate(probe)
            with self.assertRaisesRegex(ValueError, message):
                derive_ff1536_actions(states, malformed)
        for altered, expected_hash, bound_registration, error in (
            (
                b"",
                ff_registration["probe_sha256"],
                registration_bytes,
                "frozen_probe_binding",
            ),
            (
                raw_bytes + b" ",
                ff_registration["probe_sha256"],
                registration_bytes,
                "frozen_probe_binding",
            ),
            (
                raw_bytes,
                ff_registration["probe_sha256"],
                registration_bytes + b" ",
                "probe_registration_binding",
            ),
        ):
            with self.assertRaisesRegex(ValueError, error):
                validate_probe_bundle(altered, expected_hash, bound_registration)
        self.assertEqual(
            {name: (OUT / name).read_bytes() for name in publication_names}, published
        )

    def test_frozen_mapping_and_reuse_accounting(self) -> None:
        actions = frozen_actions()
        plan = build_reuse_plan()
        self.assertEqual(len(actions), 64)
        self.assertEqual(len(plan), 128)
        self.assertEqual(sum(row["source"] == "reused" for row in plan), 120)
        self.assertEqual(sum(row["source"] == "new" for row in plan), 8)
        self.assertTrue(
            all(
                row["source_ledger_sha256"] and row["source_case_identity"]
                for row in plan
                if row["source"] == "reused"
            )
        )

    def test_only_registered_eight_cases_require_games(self) -> None:
        plan = build_reuse_plan()
        selected = [row for row in plan if row["source"] == "new"]
        cases = sorted(
            (row["opening_index"], row["forced_action"])
            for row in selected
            if row["reference"] == "seed455"
        )
        self.assertEqual(cases, [(164, 2), (217, 0), (260, 1), (372, 4)])
        self.assertEqual(
            sum(row["reference"] == "original_O0_E4" for row in selected), 4
        )

    def test_analysis_rejects_missing_and_duplicate_new_outcomes(self) -> None:
        registration = _json(OUT / "registration.json")
        new_rows = _json_lines(OUT / "new-outcomes.jsonl")
        baselines = {name: _json_lines(path) for name, path in REFERENCES.items()}
        with self.assertRaisesRegex(ValueError, "missing_new_case"):
            seed398_ff1536_analysis.calculate(
                registration,
                new_rows[:-1],
                baselines["seed455"],
                baselines["original_O0_E4"],
            )
        with self.assertRaisesRegex(ValueError, "duplicate_new_case"):
            seed398_ff1536_analysis.calculate(
                registration,
                [*new_rows, new_rows[0]],
                baselines["seed455"],
                baselines["original_O0_E4"],
            )

    def test_read_only_verifier_preserves_publication_bytes(self) -> None:
        published = (
            "registration.json",
            "new-outcomes.jsonl",
            "analysis.json",
            "provenance-matrix.json",
            "results.md",
            "publication-binding.json",
        )
        before = {name: (OUT / name).read_bytes() for name in published}
        result = verify()
        after = {name: (OUT / name).read_bytes() for name in published}
        self.assertEqual(before, after)
        self.assertEqual(result["provenance_cases"], 128)
        self.assertEqual(result["primary_mean_gain"], 0.0703125)
        self.assertEqual(result["bootstrap_95_interval"], [0.015625, 0.12890625])
        self.assertEqual(
            result["reference_mean_gains"],
            {"seed455": 0.0859375, "original_O0_E4": 0.0546875},
        )

    def test_failed_verification_preserves_publication_bytes(self) -> None:
        published = (
            "registration.json",
            "new-outcomes.jsonl",
            "analysis.json",
            "provenance-matrix.json",
            "results.md",
            "publication-binding.json",
        )
        before = {name: (OUT / name).read_bytes() for name in published}
        with patch.object(verifier, "validate_probe_bundle", side_effect=ValueError):
            with self.assertRaises(ValueError):
                verify()
        self.assertEqual(
            {name: (OUT / name).read_bytes() for name in published}, before
        )

    def test_verifier_imports_without_execution_stack(self) -> None:
        script = (
            "import sys; import ml.alphazero_lite.verify_seed398_ff1536_confirmation; "
            "assert not any(name in sys.modules for name in "
            "('ml.alphazero_lite.arena', 'ml.alphazero_lite.seed398_ff1536_confirmation', "
            "'ml.alphazero_lite.seed398_paired_first_action'))"
        )
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_verifier_never_reads_runtime_or_checkpoint_files(self) -> None:
        original = Path.read_bytes

        def read_publication_only(path: Path) -> bytes:
            text = str(path)
            if ".tmp/" in text or "model-artifact/runtime" in text:
                raise AssertionError(f"runtime file opened: {text}")
            return original(path)

        with patch.object(Path, "read_bytes", read_publication_only):
            verify()

    def test_verifier_rejects_missing_published_trajectory(self) -> None:
        outcomes = OUT / "new-outcomes.jsonl"
        original = outcomes.read_bytes()
        try:
            outcomes.write_bytes(b"\n".join(original.splitlines()[:-1]) + b"\n")
            with self.assertRaisesRegex(ValueError, "new_outcome_count"):
                verify()
        finally:
            outcomes.write_bytes(original)

    def test_verifier_rejects_altered_analysis_matrix_trajectory_and_provenance(
        self,
    ) -> None:
        cases = (
            "analysis.json",
            "provenance-matrix.json",
            "new-outcomes.jsonl",
            "registration.json",
        )
        for filename in cases:
            path = OUT / filename
            original = path.read_bytes()
            try:
                if filename.endswith(".jsonl"):
                    rows = [json.loads(line) for line in original.splitlines()]
                    rows[0]["outcome"]["trajectory"][1]["action_relative"] = 99
                    payload = b"".join(
                        json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows
                    )
                else:
                    value = json.loads(original)
                    if filename == "analysis.json":
                        value["exploratory"] = False
                    elif filename == "provenance-matrix.json":
                        value[0]["score"] = -1
                    else:
                        value["reuse_plan"][0]["source_ledger_sha256"] = "0" * 64
                    payload = (json.dumps(value) + "\n").encode()
                path.write_bytes(payload)
                with self.assertRaises(ValueError):
                    verify()
            finally:
                path.write_bytes(original)

    def test_verifier_rejects_altered_seeds_and_duplicate_outcomes(self) -> None:
        outcomes = OUT / "new-outcomes.jsonl"
        original = outcomes.read_bytes()
        rows = [json.loads(line) for line in original.splitlines()]
        try:
            rows[0]["outcome"]["trajectory"][1]["seed"] += 1
            outcomes.write_bytes(
                b"".join(
                    json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows
                )
            )
            with self.assertRaises(ValueError):
                verify()
            outcomes.write_bytes(original + original.splitlines()[0] + b"\n")
            with self.assertRaisesRegex(ValueError, "new_outcome_count"):
                verify()
        finally:
            outcomes.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
