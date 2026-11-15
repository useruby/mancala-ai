from __future__ import annotations

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
