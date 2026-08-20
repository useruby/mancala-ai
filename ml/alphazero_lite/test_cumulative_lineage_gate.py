from ml.alphazero_lite.cumulative_lineage_gate import evaluate


def entry(lower: float, upper: float) -> dict:
    return {"opening_bootstrap_ci": {"lower_95": lower, "upper_95": upper}}


def test_evaluate_accepts_safe_p0_results_at_both_canonical_budgets() -> None:
    result = evaluate(
        {
            "candidate_vs_p0": {
                "384:256": entry(-0.03, -0.01),
                "1200:1200": entry(0.01, 0.04),
            }
        }
    )

    assert result["passed"]


def test_evaluate_blocks_missing_or_unsafe_p0_evidence() -> None:
    missing = evaluate({"candidate_vs_p0": {"384:256": entry(0.0, 0.01)}})
    unsafe = evaluate(
        {
            "candidate_vs_p0": {
                "384:256": entry(-0.04, -0.01),
                "1200:1200": entry(0.01, 0.04),
            }
        }
    )

    assert missing["failure_reasons"] == ["cumulative_lineage_context_missing"]
    assert unsafe["failure_reasons"] == ["cumulative_lineage_p0_unsafe"]
