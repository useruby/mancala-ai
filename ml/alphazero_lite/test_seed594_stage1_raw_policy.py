from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from ml.alphazero_lite import seed594_stage1_raw_policy as stage1
from ml.alphazero_lite.kalah_rules import KalahGame


def test_synthetic_export_inference_parity_for_both_models() -> None:
    report = stage1.verify_synthetic_export_parity()
    assert report["status"] == "passed"
    assert report["qualification_roots_used_for_inference"] is False
    assert set(report["models"]) == {"P-E4", "Seed 594"}
    assert all(row["synthetic_forward_count"] == 3 for row in report["models"].values())
    assert all(row["legal_masks_exact"] for row in report["models"].values())
    assert all(
        row["max_abs_probability_difference"] <= stage1.PARITY_TOLERANCE
        for row in report["models"].values()
    )


def test_relative_action_coordinates_and_legal_mask_for_player_one() -> None:
    state = stage1._synthetic_states()[1]
    game = KalahGame.from_state(state)
    assert game.current_player == 1
    assert game.possible_moves() == [0, 2, 3, 5]
    assert [game.pit_index(action) for action in game.possible_moves()] == [6, 8, 9, 11]


def test_deterministic_tie_breaking_uses_lowest_legal_action() -> None:
    assert stage1.deterministic_top_action([0.1, 0.5, 0, 0.5, 0, 0], [3, 1]) == 1
    assert stage1.deterministic_top_action([100, 0, 0.4, 0, 0.4, 0], [2, 4]) == 2


def test_deterministic_action_rejects_invalid_legal_coordinates() -> None:
    with pytest.raises(ValueError, match="invalid_policy_or_legal_actions"):
        stage1.deterministic_top_action(np.zeros(6), [6])


def test_root_comparison_uses_legal_relative_actions_and_complete_oracle() -> None:
    row = stage1.compare_root_policies(
        opening_index=12,
        current_player=0,
        p_policy=[0.0, 0.7, 0.0, 0.3, 0.0, 0.0],
        candidate_policy=[0.0, 0.2, 0.0, 0.8, 0.0, 0.0],
        legal_actions=[1, 3],
        action_margins={"1": 0, "3": 4},
    )
    assert row["p_action"] == 1
    assert row["candidate_action"] == 3
    assert row["p_regret"] == 4
    assert row["candidate_regret"] == 0
    assert row["p_wdl_downgrade"] is True
    assert row["candidate_wdl_downgrade"] is False
    with pytest.raises(ValueError, match="incomplete_exact_action_coverage"):
        stage1.compare_root_policies(
            opening_index=12,
            current_player=0,
            p_policy=np.ones(6),
            candidate_policy=np.ones(6),
            legal_actions=[1, 3],
            action_margins={1: 4},
        )


def test_tied_exact_optima_and_policy_choose_lowest_action_deterministically() -> None:
    row = stage1.compare_root_policies(
        opening_index=14,
        current_player=0,
        p_policy=[0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
        candidate_policy=[0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
        legal_actions=[0, 2, 3],
        action_margins={0: 5, 2: 5, 3: 1},
    )
    assert row["p_action"] == 0
    assert row["candidate_action"] == 2
    assert row["p_regret"] == row["candidate_regret"] == 0
    assert row["exact_optimal_actions"] == [0, 2]
    assert row["p_exact_optimal_top_action"] is True
    assert row["candidate_exact_optimal_top_action"] is True
    assert row["p_exact_optimal_prior_mass"] == pytest.approx(1.0)
    assert row["candidate_exact_optimal_prior_mass"] == pytest.approx(0.5)


def test_nonoptimal_top_action_and_prior_mass_are_both_reported() -> None:
    row = stage1.compare_root_policies(
        opening_index=15,
        current_player=0,
        p_policy=[0.8, 0.2, 0, 0, 0, 0],
        candidate_policy=[0.1, 0.9, 0, 0, 0, 0],
        legal_actions=[0, 1],
        action_margins={0: 0, 1: 4},
    )
    assert row["p_exact_optimal_top_action"] is False
    assert row["candidate_exact_optimal_top_action"] is True
    assert row["p_exact_optimal_prior_mass"] == pytest.approx(0.2)
    assert row["candidate_exact_optimal_prior_mass"] == pytest.approx(0.9)


def test_player_one_oracle_margins_are_already_in_acting_player_perspective() -> None:
    row = stage1.compare_root_policies(
        opening_index=13,
        current_player=1,
        p_policy=[0.8, 0.2, 0, 0, 0, 0],
        candidate_policy=[0.2, 0.8, 0, 0, 0, 0],
        legal_actions=[0, 1],
        action_margins={0: 3, 1: -8},
    )
    assert row["p_regret"] == 0
    assert row["candidate_regret"] == 11
    assert row["p_wdl_downgrade"] is False
    assert row["candidate_wdl_downgrade"] is True


def test_checkpoint_identity_fails_closed_before_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stage1, "sha256_file", lambda _path: "0" * 64)
    with pytest.raises(ValueError, match="P_E4_checkpoint_identity_mismatch"):
        stage1.verify_synthetic_export_parity()


def test_candidate_checkpoint_identity_is_independently_authenticated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def digest(path):
        if path == stage1.P_DIR / "model.npz":
            return stage1.P_SHA256
        return "0" * 64

    monkeypatch.setattr(stage1, "sha256_file", digest)
    with pytest.raises(ValueError, match="seed594_checkpoint_identity_mismatch"):
        stage1.verify_synthetic_export_parity()


def test_qualification_runner_requires_review_bound_authorization() -> None:
    with pytest.raises(
        PermissionError, match="reviewed_execution_authorization_required"
    ):
        stage1.evaluate_membership_roots(authorization={})


def test_bootstrap_reconstruction_matches_registered_numpy_procedure() -> None:
    rows = [
        {
            "opening_index": opening,
            "candidate_regret": float((opening % 7) / 10),
            "p_regret": float((opening % 5) / 10),
        }
        for opening in range(423)
    ]
    actual = stage1.paired_opening_bootstrap(rows)
    differences = np.asarray(
        [row["candidate_regret"] - row["p_regret"] for row in rows], dtype=np.float64
    )
    rng = np.random.default_rng(594)
    sampled = rng.integers(0, 423, size=(20_000, 423))
    means = differences[sampled].mean(axis=1)
    assert actual["mean_candidate_minus_p_regret"] == pytest.approx(differences.mean())
    assert actual["paired_opening_bootstrap_ci95"] == pytest.approx(
        np.quantile(means, [0.025, 0.975], method="linear").tolist()
    )
    assert actual["quantile_method"] == "linear"


def test_bootstrap_rejects_missing_or_duplicate_openings() -> None:
    rows = [
        {"opening_index": opening, "candidate_regret": 1.0, "p_regret": 0.0}
        for opening in range(422)
    ]
    with pytest.raises(ValueError, match="incomplete_or_duplicate_opening_results"):
        stage1.paired_opening_bootstrap(rows)
    rows.append({"opening_index": 0, "candidate_regret": 1.0, "p_regret": 0.0})
    with pytest.raises(ValueError, match="incomplete_or_duplicate_opening_results"):
        stage1.paired_opening_bootstrap(rows)


def test_gate_reconstruction_and_missing_results_fail_closed() -> None:
    complete = {
        "opening_clusters": 423,
        "resamples": 20_000,
        "seed": 594,
        "quantile_method": "linear",
        "mean_candidate_minus_p_regret": -0.11,
        "paired_opening_bootstrap_ci95": [-0.2, -0.01],
        "candidate_wdl_downgrade_rate": 0.02,
        "p_wdl_downgrade_rate": 0.02,
    }
    assert stage1.qualify_gate(complete)["decision"] == "keep"
    threshold_case = {
        **complete,
        "mean_candidate_minus_p_regret": -0.10,
        "paired_opening_bootstrap_ci95": [-0.2, 0.0],
    }
    assert stage1.qualify_gate(threshold_case)["decision"] == "reject"
    assert (
        stage1.qualify_gate({**complete, "candidate_wdl_downgrade_rate": 0.03})[
            "decision"
        ]
        == "reject"
    )
    with pytest.raises(ValueError, match="incomplete_analysis"):
        stage1.qualify_gate({**complete, "opening_clusters": 422})
    with pytest.raises(ValueError, match="incomplete_analysis"):
        stage1.qualify_gate({**complete, "seed": 595})
    incomplete_wdl = dict(complete)
    incomplete_wdl.pop("p_wdl_downgrade_rate")
    with pytest.raises(ValueError, match="incomplete_analysis_values"):
        stage1.qualify_gate(incomplete_wdl)


def test_analysis_rejects_incomplete_or_duplicate_paired_results() -> None:
    row = {
        "opening_index": 0,
        "candidate_regret": 0.0,
        "p_regret": 0.0,
        "candidate_wdl_downgrade": False,
        "p_wdl_downgrade": False,
        "candidate_wdl_rank": 1,
        "p_wdl_rank": 1,
    }
    with pytest.raises(ValueError, match="incomplete_or_duplicate_opening_results"):
        stage1.analyze_root_results([row] * 423)


def _synthetic_membership() -> list[dict[str, Any]]:
    return [
        {"opening_index": opening, "state_hash": f"state-{opening:03d}"}
        for opening in range(423)
    ]


def test_paired_result_membership_checks_all_hashes_and_opening_mappings() -> None:
    membership = _synthetic_membership()
    results = list(reversed(membership))
    checked = stage1.verify_paired_result_membership(results, membership)
    assert checked["root_count"] == 423
    assert checked["state_hashes"] == sorted(row["state_hash"] for row in membership)
    assert checked["opening_to_state_hash"] == {
        str(row["opening_index"]): row["state_hash"] for row in membership
    }


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("omit", "result_population_mismatch"),
        ("duplicate", "result_contains_duplicate_identity"),
        ("duplicate_hash", "result_contains_duplicate_identity"),
        ("duplicate_opening", "result_contains_duplicate_identity"),
        ("substitute", "result_membership_mapping_mismatch"),
        ("swap_openings", "result_membership_mapping_mismatch"),
        ("noninteger_opening", "result_identity_field_type_mismatch"),
    ],
)
def test_paired_result_membership_mutations_fail_closed(
    mutation: str, error: str
) -> None:
    membership = _synthetic_membership()
    results = [dict(row) for row in membership]
    if mutation == "omit":
        results.pop()
    elif mutation == "duplicate":
        results[-1] = dict(results[0])
    elif mutation == "duplicate_hash":
        results[-1]["state_hash"] = results[0]["state_hash"]
    elif mutation == "duplicate_opening":
        results[-1]["opening_index"] = results[0]["opening_index"]
    elif mutation == "substitute":
        results[17]["state_hash"] = "not-a-member"
    elif mutation == "noninteger_opening":
        results[17]["opening_index"] = 17.5
    else:
        results[17]["opening_index"], results[18]["opening_index"] = (
            results[18]["opening_index"],
            results[17]["opening_index"],
        )
    with pytest.raises(ValueError, match=error):
        stage1.verify_paired_result_membership(results, membership)


def test_aggregate_reports_exact_optimal_metrics_for_both_models() -> None:
    rows = []
    for opening in range(423):
        rows.append(
            {
                "opening_index": opening,
                "candidate_regret": 1.0,
                "p_regret": 2.0,
                "candidate_exact_optimal_top_action": opening % 2 == 0,
                "p_exact_optimal_top_action": opening % 4 == 0,
                "candidate_exact_optimal_prior_mass": 0.75,
                "p_exact_optimal_prior_mass": 0.25,
                "candidate_wdl_downgrade": False,
                "p_wdl_downgrade": True,
            }
        )
    analysis = stage1.analyze_root_results(rows)
    assert analysis["p_exact_optimal_top_action_rate"] == pytest.approx(106 / 423)
    assert analysis["candidate_exact_optimal_top_action_rate"] == pytest.approx(
        212 / 423
    )
    assert analysis["p_exact_optimal_prior_probability_mass"] == pytest.approx(0.25)
    assert analysis["candidate_exact_optimal_prior_probability_mass"] == pytest.approx(
        0.75
    )
    assert analysis["p_mean_exact_regret"] == pytest.approx(2.0)
    assert analysis["candidate_mean_exact_regret"] == pytest.approx(1.0)
    assert analysis["p_wdl_downgrade_rate"] == pytest.approx(1.0)
    assert analysis["candidate_wdl_downgrade_rate"] == pytest.approx(0.0)


def test_freeze_retains_historical_access_order_disclosure() -> None:
    freeze = json.loads(
        (
            stage1.ROOT / "docs/data/seed594-stage1-raw-policy/execution-freeze.json"
        ).read_bytes()
    )
    deviation = freeze["access_order_deviation"]
    assert (
        deviation["public_membership_file_opened_before_synthetic_parity_checks"]
        is True
    )
    assert deviation["historical_deviation_preserved"] is True
    assert deviation["qualification_inference_performed"] is False
    assert deviation["qualification_outcome_analysis_performed"] is False
