"""Semantic and read-only contract tests for seed448."""

from __future__ import annotations

import copy

import pytest

from ml.alphazero_lite import verify_seed448_seed447_publication as verifier


def _fixture() -> tuple[dict, list[dict]]:
    def metric(ce: float, mse: float) -> dict[str, float]:
        return {"policy_ce": ce, "value_mse": mse}

    metrics = {
        "initializer": {
            "full_training_objective": 2,
            "exposure_weighted": metric(1, 0.5),
            "equal_input": metric(1, 0.5),
        },
        "seed442_A": {
            "full_training_objective": 2,
            "exposure_weighted": metric(1, 0.5),
            "equal_input": metric(1, 0.5),
        },
        "T": {
            "full_training_objective": 1,
            "exposure_weighted": metric(0.98, 0.5),
            "equal_input": metric(0.98, 0.5),
        },
    }
    metrics["C"] = copy.deepcopy(metrics["initializer"])
    metrics["seed442_B"] = copy.deepcopy(metrics["initializer"])
    effect_ce = (
        metrics["T"]["exposure_weighted"]["policy_ce"]
        - metrics["C"]["exposure_weighted"]["policy_ce"]
    )
    metrics["T_minus_C"] = {
        "full_training_objective": -1,
        "exposure_weighted": {"policy_ce": effect_ce, "value_mse": 0},
        "equal_input": {"policy_ce": effect_ce, "value_mse": 0},
    }
    return metrics, [
        {
            "proposal_norm": 1,
            "policy_only_scale": 0.5,
            "selected_scale": 0.25,
            "rejected": False,
        }
    ]


def test_independent_gate_arithmetic_and_classification_semantics() -> None:
    metrics, steps = _fixture()
    valid = verifier.classify(metrics, steps)
    assert (
        valid["classification"]
        == "advance_to_separately_preregistered_strength_experiment"
    )
    assert all(valid["checks"].values())
    for weighting in ("exposure_weighted", "equal_input"):
        changed = copy.deepcopy(metrics)
        changed["T"][weighting]["policy_ce"] = 1.0
        assert not verifier.classify(changed, steps)["checks"][
            f"T_minus_C_{weighting}_policy"
        ]
        changed = copy.deepcopy(metrics)
        changed["T"][weighting]["value_mse"] = 0.503
        assert not verifier.classify(changed, steps)["checks"][
            f"T_{weighting}_value_guard"
        ]
        changed = copy.deepcopy(metrics)
        changed["T"][weighting]["policy_ce"] = 0.996
        assert not verifier.classify(changed, steps)["checks"][
            f"T_minus_initializer_{weighting}_policy"
        ]
    changed = copy.deepcopy(metrics)
    changed["T"]["full_training_objective"] = 2
    assert not verifier.classify(changed, steps)["checks"][
        "T_training_objective_decreases"
    ]
    failing = copy.deepcopy(metrics)
    failing["T"]["exposure_weighted"]["policy_ce"] = 1.0
    assert (
        verifier.classify(failing, steps)["classification"]
        == "close_joint_output_cap_branch"
    )
    assert (
        verifier.classify(metrics, [{**steps[0], "selected_scale": 0.5}])[
            "classification"
        ]
        == "value_cap_inactive_no_followup"
    )
    assert (
        verifier.classify(metrics, [{**steps[0], "rejected": True}])["classification"]
        == "value_cap_inactive_no_followup"
    )


def test_amendment_receipt_binding_tamper_is_rejected(tmp_path, monkeypatch) -> None:
    out = tmp_path / verifier.OUT
    out.mkdir(parents=True)
    for path, data in ((out / "registration.json", b"x"), (out / "receipt.json", b"y")):
        path.write_bytes(data)
    with pytest.raises(ValueError, match="registration_binding_invalid"):
        verifier.validate_amendment_bindings(tmp_path)


def test_comparator_effect_gate_and_classification_mutations_reject_semantically() -> (
    None
):
    metrics, steps = _fixture()
    evidence = {"metrics": metrics, "arms": {"T": {"steps": steps}}}
    evidence["decision"] = verifier.classify(metrics, steps)
    authoritative = {"metrics": {"A": metrics["seed442_A"], "B": metrics["seed442_B"]}}
    assert (
        verifier.validate_decision_evidence(evidence, authoritative)
        == evidence["decision"]
    )
    mutations = (
        (
            lambda e: e["metrics"]["seed442_A"]["exposure_weighted"].update(
                policy_ce=1.2
            ),
            "comparator_metrics",
        ),
        (
            lambda e: e["metrics"]["T_minus_C"]["equal_input"].update(policy_ce=0),
            "effects_semantic",
        ),
        (
            lambda e: e["decision"]["checks"].update(value_cap_activates=False),
            "decision_semantics",
        ),
        (
            lambda e: e["decision"].update(
                classification="value_cap_inactive_no_followup"
            ),
            "decision_semantics",
        ),
    )
    for mutate, message in mutations:
        changed = copy.deepcopy(evidence)
        mutate(changed)
        with pytest.raises(ValueError, match=message):
            verifier.validate_decision_evidence(changed, authoritative)


def test_cli_never_publishes_receipt(monkeypatch) -> None:
    monkeypatch.setattr(verifier, "verify", lambda _root: {"status": "valid"})
    monkeypatch.setattr(
        verifier.amendment,
        "publish_amendment_receipt",
        lambda *_: pytest.fail("publisher called"),
    )
    monkeypatch.setattr(
        "sys.argv", ["verify_seed448_seed447_publication", "--root", "."]
    )
    verifier.main()
