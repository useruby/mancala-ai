from __future__ import annotations

import copy
import hashlib

import pytest

from ml.alphazero_lite import (
    verify_seed542_update432_strength_screen_supplement as supplement,
)


def test_substituted_opening_is_rejected_even_when_pair_receipts_are_rebound() -> None:
    suite = [b'{"opening_index":3}\n', b'{"opening_index":4}\n']
    played = suite[0]  # opening-3 bytes copied to the batch for index 4
    preflight = {
        "opening_index": 4,
        "registration_sha256": "registered",
        "suite_sha256": "suite",
        "argv": ["arena", "--seed", "549"],
        "runtime": {"threads": 1},
        "component_assignments": {"challenger_policy": "D-final"},
        "component_hashes": {"candidate": "sha"},
        "runtime_source_sha256": {"runner.py": "source-sha"},
    }
    command = copy.deepcopy(preflight)
    # A forged output receipt can consistently rebind copies and row hashes.
    command["opening_sha256"] = hashlib.sha256(played).hexdigest()
    command["games_sha256"] = "rebound-games"
    command["report_sha256"] = "rebound-report"
    freeze = {"registration_sha256": "registered", "suite_sha256": "suite"}
    receipt = {
        "opening_index": 4,
        "registration_sha256": "registered",
        "suite_sha256": "suite",
        "opening_sha256": hashlib.sha256(played).hexdigest(),
        "command_sha256": "rebound-command",
        "games_sha256": "rebound-games",
        "report_sha256": "rebound-report",
    }
    registration = {
        "runtime": {"threads": 1},
        "component_assignments": {"challenger_policy": "D-final"},
        "component_hashes": {"candidate": "sha"},
        "execution_source_sha256": {"runner.py": "source-sha"},
    }
    with pytest.raises(ValueError, match="opening_not_equal_to_frozen_suite_row:4"):
        supplement.verify_opening_and_command_binding(
            4,
            suite[1],
            played,
            preflight,
            command,
            freeze=freeze,
            registration=registration,
            completion_receipt=receipt,
            command_sha256="rebound-command",
            games_sha256="rebound-games",
            report_sha256="rebound-report",
        )


def test_mutated_command_is_rejected_against_its_frozen_preflight() -> None:
    opening = b'{"opening_index":4}\n'
    preflight = {
        "opening_index": 4,
        "registration_sha256": "registered",
        "suite_sha256": "suite",
        "argv": ["arena", "--seed", "549"],
        "runtime": {"threads": 1},
        "component_assignments": {"challenger_policy": "D-final"},
        "component_hashes": {"candidate": "sha"},
        "runtime_source_sha256": {"runner.py": "source-sha"},
    }
    command = {**preflight, "argv": ["arena", "--seed", "548"]}
    with pytest.raises(
        ValueError, match="played_command_differs_from_frozen_preflight:4:argv"
    ):
        supplement.verify_opening_and_command_binding(
            4,
            opening,
            opening,
            preflight,
            command,
            freeze={"registration_sha256": "registered", "suite_sha256": "suite"},
            registration={
                "runtime": {"threads": 1},
                "component_assignments": {"challenger_policy": "D-final"},
                "component_hashes": {"candidate": "sha"},
                "execution_source_sha256": {"runner.py": "source-sha"},
            },
        )


def test_continuation_registration_must_match_frozen_source_map() -> None:
    registration = {"execution_source_sha256": {"continue.py": "frozen-source"}}
    freeze = {
        "registration_sha256": "registration",
        "execution_source_sha256": {"continue.py": "tampered-source"},
    }
    with pytest.raises(
        ValueError, match="continuation_v6_registration_freeze_mismatch"
    ):
        supplement.verify_continuation_record(6, registration, freeze, "registration")


def test_per_opening_binding_rejects_continuation_freeze_substitution() -> None:
    expected = {
        "opening_index": 7,
        "continuation_version": 6,
        "continuation_registration_sha256": "reg-v6",
        "continuation_freeze_sha256": "freeze-v6",
        "parent_registration_sha256": "parent-reg",
        "parent_execution_freeze_sha256": "parent-freeze",
        "suite_sha256": "suite",
        "frozen_suite_opening_sha256": "opening",
        "played_opening_sha256": "opening",
        "preflight_row_sha256": "preflight",
        "command_sha256": "command",
        "games_sha256": "games",
        "report_sha256": "report",
        "completion_receipt_sha256": "receipt",
    }
    altered = {**expected, "continuation_freeze_sha256": "other-freeze"}
    with pytest.raises(
        ValueError,
        match="continuation_pair_binding_mismatch:continuation_freeze_sha256",
    ):
        supplement.verify_pair_binding(altered, expected)
