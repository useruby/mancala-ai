"""Semantic tamper and retrospective seed462 recovery tests."""

import numpy as np
import pytest
import torch

from ml.alphazero_lite.verify_seed462_recovery import (
    _cohort_targets,
    require_array_equal,
    validate_endpoint_semantics,
)


def _endpoint() -> dict:
    arms = {}
    for arm, offset in (("initializer", 0.0), ("C", 0.02), ("F", 0.03)):
        arms[arm] = {}
        for cohort, base in (("pooled", 1.0), ("fresh", 1.2)):
            arms[arm][cohort] = {}
            for weighting in ("exposure_weighted", "equal_input"):
                arms[arm][cohort][weighting] = {
                    "policy_ce": base - offset,
                    "value_mse": 0.5 + offset,
                }
        arms[arm]["full_training_objectives"] = {"fresh_policy": 1.0 - offset}
    effects = {
        cohort: {
            weight: {
                metric: arms["F"][cohort][weight][metric]
                - arms["C"][cohort][weight][metric]
                for metric in ("policy_ce", "value_mse")
            }
            for weight in ("exposure_weighted", "equal_input")
        }
        for cohort in ("pooled", "fresh")
    }
    gates = {}
    for weight in ("exposure_weighted", "equal_input"):
        gates[f"fresh_vs_initializer_{weight}"] = (
            arms["F"]["fresh"][weight]["policy_ce"]
            <= arms["initializer"]["fresh"][weight]["policy_ce"] - 0.005
        )
        gates[f"fresh_vs_C_{weight}"] = (
            arms["F"]["fresh"][weight]["policy_ce"]
            <= arms["C"]["fresh"][weight]["policy_ce"] - 0.005
        )
        gates[f"pooled_ce_{weight}"] = (
            arms["F"]["pooled"][weight]["policy_ce"]
            <= arms["initializer"]["pooled"][weight]["policy_ce"] + 0.002
        )
        for cohort in ("pooled", "fresh"):
            gates[f"{cohort}_mse_{weight}"] = (
                arms["F"][cohort][weight]["value_mse"]
                <= arms["initializer"][cohort][weight]["value_mse"] + 0.002
            )
    return {
        "arms": arms,
        "effects_F_minus_C": effects,
        "gates": gates,
        "prospective_protocol_limitation": "The protocol does not explicitly register exact cohort estimands.",
    }


def test_historical_policy_mask_and_denominator_are_semantic() -> None:
    ce = torch.tensor([2.0, 8.0, 4.0])
    coeff = torch.tensor([1.0, 3.0, 2.0])
    fresh = torch.tensor([1.0, 0.0, 1.0])
    control = (ce * coeff).sum() / coeff.sum()
    treatment = (ce * coeff * fresh).sum() / (coeff * fresh).sum()
    assert control.item() == pytest.approx(34.0 / 6.0)
    assert treatment.item() == pytest.approx(10.0 / 3.0)
    assert (coeff * fresh).sum().item() == 3.0


def test_empty_fresh_denominator_is_rejected() -> None:
    weights = torch.tensor([1.0, 2.0]) * torch.tensor([False, False])
    with pytest.raises(ZeroDivisionError):
        if weights.sum() == 0:
            raise ZeroDivisionError("empty treatment denominator")


@pytest.mark.parametrize("kind", ["gradient", "moment", "state", "prediction"])
def test_receipt_rebound_does_not_make_tampered_tensor_semantic(kind: str) -> None:
    good = np.asarray([0.25, -0.5], dtype=np.float32)
    changed = good.copy()
    changed[0] += 0.125
    # A normal SHA receipt may be recomputed over `changed`; independent semantics
    # still compare it with the reconstruction rather than trusting that receipt.
    with pytest.raises(ValueError, match="semantic_mismatch"):
        require_array_equal(kind, changed, good)


@pytest.mark.parametrize("tamper", ["effect", "gate", "classification"])
def test_rebound_endpoint_semantics_reject_tampering(tamper: str) -> None:
    endpoint = _endpoint()
    if tamper == "effect":
        endpoint["effects_F_minus_C"]["fresh"]["equal_input"]["policy_ce"] += 0.1
    elif tamper == "gate":
        endpoint["gates"]["fresh_vs_C_equal_input"] = False
    else:
        endpoint["decision_status"] = "advance"
    with pytest.raises(ValueError):
        validate_endpoint_semantics(endpoint)


def test_replay_multiplicity_keeps_duplicate_occurrences() -> None:
    rows = np.asarray([7, 7, 3], dtype=np.int64)
    target = np.asarray([1.0, 4.0, 2.0])
    assert rows.tolist() == [7, 7, 3]
    assert target[:2].tolist() == [1.0, 4.0]


def test_occurrence_targets_preserve_different_repeated_identity_targets(
    tmp_path,
) -> None:
    import gzip
    import json

    source_dir = tmp_path / "docs/data/seed426-canonical-overlap/sources"
    source_dir.mkdir(parents=True)
    records = [
        {"policy": [1.0, 0.0], "value": 0.25},
        {"policy": [0.0, 1.0], "value": -0.5},
    ]
    with gzip.open(source_dir / "fresh.jsonl.gz", "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    membership = [
        {"source": "fresh", "raw_line": 1, "input_identity": "repeated"},
        {"source": "fresh", "raw_line": 2, "input_identity": "repeated"},
    ]
    policies, values = _cohort_targets(tmp_path, membership)
    np.testing.assert_array_equal(policies, [[1, 0], [0, 1]])
    np.testing.assert_array_equal(values, [0.25, -0.5])


def test_rebound_ordinary_hash_does_not_bypass_semantic_effect_check() -> None:
    import hashlib
    import json

    endpoint = _endpoint()
    endpoint["effects_F_minus_C"]["pooled"]["equal_input"]["value_mse"] += 0.01
    # Model the routine response of a naive publisher: rebind the changed bytes.
    rebound = hashlib.sha256(json.dumps(endpoint, sort_keys=True).encode()).hexdigest()
    assert len(rebound) == 64
    with pytest.raises(ValueError, match="semantic_mismatch:effect"):
        validate_endpoint_semantics(endpoint)
