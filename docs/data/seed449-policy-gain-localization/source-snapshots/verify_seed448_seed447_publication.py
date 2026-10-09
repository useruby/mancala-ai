"""Read-only seed448 verification of the seed447 publication and replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import verify_seed447_joint_output_cap_amendment as amendment

OUT = Path("docs/data/seed447-joint-output-cap")
REGISTRATION_SHA256 = "614bd7b158b6590d5d0114534abcdec3bc7edee977220bfcd5e38d58901021cb"
RECEIPT_SHA256 = "1fb596408ff8564871508813aaa1f5f1434e6dfdf921a7ed0ae9b46266c39122"
CLASSIFICATION = "close_joint_output_cap_branch"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(metrics: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Independently calculate seed447 fixed gates and precedence."""
    init, ordinary, treatment = (
        metrics[key] for key in ("initializer", "seed442_A", "T")
    )
    checks: dict[str, bool] = {}
    for weighting in ("exposure_weighted", "equal_input"):
        checks[f"T_minus_C_{weighting}_policy"] = (
            treatment[weighting]["policy_ce"] <= ordinary[weighting]["policy_ce"] - 0.01
        )
        checks[f"T_minus_initializer_{weighting}_policy"] = (
            treatment[weighting]["policy_ce"] <= init[weighting]["policy_ce"] - 0.005
        )
        checks[f"T_{weighting}_value_guard"] = (
            treatment[weighting]["value_mse"] <= init[weighting]["value_mse"] + 0.002
        )
    checks["T_training_objective_decreases"] = (
        treatment["full_training_objective"] < init["full_training_objective"]
    )
    nonzero = [step for step in steps if step["proposal_norm"] > 0]
    activation = any(
        step["policy_only_scale"] is not None
        and step["selected_scale"] is not None
        and step["selected_scale"] < step["policy_only_scale"]
        and not step["rejected"]
        for step in nonzero
    )
    checks["value_cap_activates"] = activation
    checks["no_nonzero_proposal_rejected"] = not any(
        step["rejected"] for step in nonzero
    )
    label = (
        "value_cap_inactive_no_followup"
        if not activation
        else "advance_to_separately_preregistered_strength_experiment"
        if all(checks.values())
        else CLASSIFICATION
    )
    return {"classification": label, "checks": checks}


def validate_amendment_bindings(root: Path) -> None:
    """Reject drift from the identities asserted by the existing amendment receipt."""
    out = root / OUT
    if _sha(out / "registration.json") != REGISTRATION_SHA256:
        raise ValueError("seed447_registration_binding_invalid")
    if _sha(out / "receipt.json") != RECEIPT_SHA256:
        raise ValueError("seed447_original_receipt_binding_invalid")
    receipt = json.loads((out / "verifier-amendment-receipt.json").read_text())
    for field, digest_field, path in (
        (
            "amendment_source",
            "amendment_source_sha256",
            root / "ml/alphazero_lite/verify_seed447_joint_output_cap_amendment.py",
        ),
        (
            "frozen_verifier_sha256",
            "frozen_verifier_sha256",
            root / "ml/alphazero_lite/verify_seed447_joint_output_cap.py",
        ),
    ):
        if _sha(path) != receipt[digest_field]:
            raise ValueError(f"seed447_amendment_binding_invalid:{field}")
    if (
        receipt["amendment_source"]
        != "ml/alphazero_lite/verify_seed447_joint_output_cap_amendment.py"
    ):
        raise ValueError("seed447_amendment_path_invalid")
    if receipt["frozen_receipt_sha256"] != RECEIPT_SHA256:
        raise ValueError("seed447_amendment_receipt_binding_invalid")
    registration = json.loads((out / "registration.json").read_text())
    for relative, digest in registration["source_sha256"].items():
        if (
            _sha(root / relative) != digest
            or _sha(out / "source-snapshots" / Path(relative).name) != digest
        ):
            raise ValueError(f"seed447_frozen_source_binding_invalid:{relative}")
    embedded = registration["seed442_registration"]
    for relative, digest in embedded["source_sha256"].items():
        if _sha(root / relative) != digest:
            raise ValueError(f"seed442_dependency_binding_invalid:{relative}")
    for relative, digest in embedded["frozen_input_sha256"].items():
        if _sha(root / relative) != digest:
            raise ValueError(f"seed442_input_binding_invalid:{relative}")
    publication_amendment = json.loads(
        (out / "post-execution-publication-amendment.json").read_text()
    )
    if (
        publication_amendment["frozen_registration_sha256"] != REGISTRATION_SHA256
        or publication_amendment["frozen_receipt_sha256"] != RECEIPT_SHA256
    ):
        raise ValueError("seed447_publication_amendment_binding_invalid")
    for name, digest in publication_amendment["supplemental_files_sha256"].items():
        if _sha(out / name) != digest:
            raise ValueError(f"seed447_publication_supplemental_binding_invalid:{name}")


def validate_decision_evidence(
    evidence: dict[str, Any], authoritative: dict[str, Any]
) -> dict[str, Any]:
    """Semantically bind comparator/effect fields and independent gate flags."""
    if evidence["metrics"]["seed442_A"] != authoritative["metrics"]["A"]:
        raise ValueError("seed442_comparator_metrics_semantic_invalid")
    if evidence["metrics"]["seed442_B"] != authoritative["metrics"]["B"]:
        raise ValueError("seed442_control_metrics_semantic_invalid")
    expected_effects: dict[str, Any] = {
        "full_training_objective": evidence["metrics"]["T"]["full_training_objective"]
        - evidence["metrics"]["C"]["full_training_objective"]
    }
    for weighting in ("exposure_weighted", "equal_input"):
        expected_effects[weighting] = {
            metric: evidence["metrics"]["T"][weighting][metric]
            - evidence["metrics"]["C"][weighting][metric]
            for metric in ("policy_ce", "value_mse")
        }
    if evidence["metrics"]["T_minus_C"] != expected_effects:
        raise ValueError("treatment_control_effects_semantic_invalid")
    result = classify(evidence["metrics"], evidence["arms"]["T"]["steps"])
    if result != evidence["decision"]:
        raise ValueError("independent_decision_semantics_invalid")
    return result


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    validate_amendment_bindings(root)
    replay = amendment.verify(root)
    out = root / OUT
    evidence = json.loads((out / "evidence.json").read_text())
    base_evidence = json.loads(
        (root / "docs/data/seed442-kl-capped-adam-screen/evidence.json").read_text()
    )
    decision = validate_decision_evidence(evidence, base_evidence)
    if decision["classification"] != CLASSIFICATION:
        raise ValueError("historical_classification_changed")
    if replay["classification"] != CLASSIFICATION:
        raise ValueError("replay_classification_changed")
    return {
        **replay,
        "status": "valid",
        "classification": decision["classification"],
        "independent_checks": decision["checks"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
