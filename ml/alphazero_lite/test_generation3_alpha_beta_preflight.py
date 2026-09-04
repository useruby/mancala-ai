from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ml.alphazero_lite.run_generation3_alpha_beta_preflight import (
    build_corpus,
    classify_lane,
)


ROOT = Path(__file__).resolve().parents[2]


def test_corpus_is_deterministic_and_training_ineligible() -> None:
    left = build_corpus(count=12)
    assert left == build_corpus(count=12)
    assert all(row["diagnostic_only"] and row["not_training_eligible"] for row in left)


def test_current_artifact_and_closeout_identities_remain_frozen() -> None:
    metadata = json.loads((ROOT / "model-artifact/current/metadata.json").read_text())
    digest = hashlib.sha256(
        (ROOT / "model-artifact/current/weights.json").read_bytes()
    ).hexdigest()
    assert metadata["version"] == "azlite-balanced-w8s4-policy-head-e1"
    assert digest == "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
    assert (
        "paused_no_qualified_capability"
        in (ROOT / "docs/data/alphazero-lite-a16-lineage-closeout.json").read_text()
    )


def test_classification_and_diagnostic_guardrail_contract() -> None:
    metric = {
        "alpha_beta": {
            "mean_regret": 0.1,
            "exact_best_agreement": 0.8,
            "catastrophic_miss_rate": 0,
        },
        "ordinary_puct": {
            "mean_regret": 0.2,
            "exact_best_agreement": 0.7,
            "catastrophic_miss_rate": 0,
        },
        "paired_hierarchical_bootstrap": {"upper_95": -0.01},
    }
    slices = {
        "phase": {
            "opening": {
                "n_paired_rows": 9,
                "alpha_beta_catastrophic_miss_rate": 0,
                "ordinary_puct_catastrophic_miss_rate": 0,
            }
        }
    }
    assert (
        classify_lane(metric, slices, budget_ok=True, invariants_ok=True) == "qualified"
    )
    plan = (
        ROOT / "docs/alphazero-lite-generation3-alpha-beta-preflight-plan.md"
    ).read_text()
    assert "diagnostic-only" in plan and "No training" in plan
