from ml.alphazero_lite.run_internal_cluster_learning_dynamics import (
    assert_lineage_isolation,
    build_lineage_config,
    checkpoint_for_generation,
    classify_trajectory,
    lineage_summary,
    natural_repair_passes,
    policy_metrics,
)


def test_trajectory_classifications() -> None:
    assert classify_trajectory([False, False, False, True]) == "repaired"
    assert classify_trajectory([False, False, False, False]) == "persistent_failure"
    assert classify_trajectory([True, True, False, False]) == "forgotten"
    assert classify_trajectory([True, False, True, False]) == "oscillating"
    assert classify_trajectory([True, True, True, True]) == "stable_good"


def test_exact_policy_scoring_masks_illegal_actions() -> None:
    metrics = policy_metrics(
        [0.2, 0.5, 0.3, 0.9, 0.0, 0.0],
        entry_row={
            "legal_actions": [0, 1, 2],
            "exact_outcome_optimal_actions": [1],
            "degrading_actions": [2],
            "exact_outcome_value": 1.0,
        },
        value=0.25,
    )
    assert metrics["top_action"] == 1
    assert metrics["top_is_outcome_optimal"] is True
    assert metrics["optimal_mass"] == 0.5
    assert metrics["degrading_action_mass"] == 0.3
    assert metrics["exact_wdl_value_error"] == 0.75


def test_natural_repair_rule_requires_two_lineages_and_repair() -> None:
    rows = [
        {
            "final_cluster_delta": 0.2,
            "final_control_delta": 0.1,
            "trajectory_counts": {"repaired": 1},
        },
        {
            "final_cluster_delta": 0.1,
            "final_control_delta": 0.0,
            "trajectory_counts": {},
        },
        {
            "final_cluster_delta": -0.1,
            "final_control_delta": 0.0,
            "trajectory_counts": {},
        },
    ]
    assert natural_repair_passes(rows) is True


def test_lineages_must_share_g0_parent() -> None:
    assert_lineage_isolation([{"g0_sha256": "a"}, {"g0_sha256": "a"}], "a")
    try:
        assert_lineage_isolation([{"g0_sha256": "a"}, {"g0_sha256": "b"}], "a")
    except ValueError as error:
        assert "same G0" in str(error)
    else:
        raise AssertionError("expected isolated-lineage validation failure")


def test_lineage_config_only_binds_existing_parent_checkpoint(tmp_path) -> None:
    parent = tmp_path / "parent" / "checkpoint.npz"
    config = build_lineage_config(parent, seed=61, versions_dir=tmp_path / "versions")
    self_play = next(step for step in config["steps"] if step["name"] == "self_play")
    train = next(step for step in config["steps"] if step["name"] == "train")
    assert self_play["command"][-2:] == ["--checkpoint", "{parent_checkpoint}"]
    assert "--checkpoint" not in train["command"]
    assert config["fixed_replay_sources"][0]["weight"] == 1
    assert config["fixed_replay_sources"][1]["weight"] == 2
    assert "promotion" not in config


def test_generation_checkpoint_uses_pipeline_artifact_contract(tmp_path) -> None:
    parent = tmp_path / "parent.npz"
    parent.write_bytes(b"parent")
    versions = tmp_path / "versions"
    checkpoint = versions / "run-iter1" / "checkpoint.npz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"generation")
    assert (
        checkpoint_for_generation(
            parent=parent, versions_dir=versions, run_id="run", generation=0
        )
        == parent
    )
    assert (
        checkpoint_for_generation(
            parent=parent, versions_dir=versions, run_id="run", generation=1
        )
        == checkpoint
    )


def test_lineage_summary_reports_mass_and_trajectories() -> None:
    def row(identifier, membership, good, mass):
        return {
            "id": identifier,
            "membership": membership,
            "top_is_outcome_optimal": good,
            "optimal_mass": mass,
            "degrading_action_mass": 0.0,
            "policy_entropy": 1.0,
            "exact_wdl_value_error": 0.0,
        }

    evaluations = [
        {
            "rows": [
                row("cluster", "cluster_anchor", False, 0.1),
                row("control", "matched_control", True, 0.5),
            ]
        },
        {
            "rows": [
                row("cluster", "cluster_anchor", False, 0.2),
                row("control", "matched_control", True, 0.5),
            ]
        },
        {
            "rows": [
                row("cluster", "cluster_anchor", False, 0.3),
                row("control", "matched_control", True, 0.5),
            ]
        },
        {
            "rows": [
                row("cluster", "cluster_anchor", True, 0.4),
                row("control", "matched_control", True, 0.5),
            ]
        },
    ]
    summary = lineage_summary(evaluations)
    assert summary["trajectory_counts"] == {"repaired": 1, "stable_good": 1}
    assert abs(summary["final_cluster_delta"] - 0.3) < 1e-12
    assert summary["final_control_delta"] == 0.0
