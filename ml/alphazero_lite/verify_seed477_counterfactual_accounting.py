"""Read-only verification of the seed477 semantic correction."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from ml.alphazero_lite.seed477_counterfactual_accounting import (
    BASE,
    INPUTS,
    OUT,
    PRIOR,
    SOURCES,
    sha256,
)

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CATEGORIES = (
    "minibatch_transport_failure",
    "adam_coordinate_transport_failure",
    "both_descent",
    "mixed_or_ambiguous",
)


def _receipt_bound(root: Path, directory: Path, *, status: str) -> dict:
    receipt = json.loads((root / directory / "receipt.json").read_text())
    if receipt.get("status") != status:
        raise ValueError(f"historical_receipt_status:{directory}")
    for name, digest in receipt.items():
        if name != "status" and sha256(root / directory / name) != digest:
            raise ValueError(f"historical_receipt_mismatch:{directory}/{name}")
    return receipt


def verify(root: Path) -> None:
    """Validate historical receipts, correction bindings, and exact transformation."""
    root = root.resolve()
    out = root / OUT
    protected_dirs = (root / BASE, root / PRIOR, out)
    before = _snapshot(protected_dirs)
    try:
        # Validate the original seed476 publication and prior supplement first.
        _receipt_bound(root, BASE, status="seed476_publication_bound")
        _receipt_bound(root, PRIOR, status="seed476_supplement_bound")
        prior = json.loads((root / PRIOR / "evidence.json").read_text())
        original = json.loads((root / BASE / "evidence.json").read_text())
        if original.get("schema") != "seed476-direction-transport-evidence-v1":
            raise ValueError("original_seed476_evidence_schema")
        if len(prior.get("steps", [])) != 16 or len(original.get("steps", [])) != 16:
            raise ValueError("historical_record_coverage")

        reg = json.loads((out / "registration.json").read_text())
        if (
            reg.get("schema") != "seed477-counterfactual-accounting-registration-v1"
            or reg.get("status") != "post_execution_semantic_correction_frozen"
        ):
            raise ValueError("seed477_registration_identity")
        if set(reg.get("inputs_sha256", {})) != set(INPUTS):
            raise ValueError("seed477_input_inventory")
        for key, relative in INPUTS.items():
            if sha256(root / relative) != reg["inputs_sha256"][key]:
                raise ValueError(f"seed477_input_binding:{key}")
        for field, key in (
            ("original_seed476_registration_sha256", "seed476_registration"),
            ("original_seed476_receipt_sha256", "seed476_receipt"),
            ("original_seed476_evidence_sha256", "seed476_evidence"),
            ("original_seed476_vectors_sha256", "seed476_vectors"),
            ("prior_supplement_registration_sha256", "prior_registration"),
            ("prior_supplement_receipt_sha256", "prior_receipt"),
            ("prior_supplement_evidence_sha256", "prior_evidence"),
            ("prior_supplement_accounting_sha256", "prior_accounting"),
        ):
            if reg.get(field) != reg["inputs_sha256"][key]:
                raise ValueError(f"historical_binding:{field}")
        if set(reg.get("source_sha256", {})) != set(SOURCES):
            raise ValueError("seed477_source_inventory")
        for name in SOURCES:
            if sha256(root / name) != reg["source_sha256"][name]:
                raise ValueError(f"seed477_source_binding:{name}")

        receipt = json.loads((out / "receipt.json").read_text())
        if (
            set(receipt)
            != {"status", "registration.json", "evidence.json", "results.md"}
            or receipt["status"] != "seed477_counterfactual_accounting_bound"
        ):
            raise ValueError("seed477_receipt_inventory")
        for name, digest in receipt.items():
            if name != "status" and sha256(out / name) != digest:
                raise ValueError(f"seed477_receipt_mismatch:{name}")

        actual = json.loads((out / "evidence.json").read_text())
        expected = _expected_correction(prior)
        if actual != expected:
            raise ValueError("seed477_transformation_mismatch")
        _verify_schema(actual, prior)
    finally:
        if before != _snapshot(protected_dirs):
            raise ValueError("verification_mutated_evidence")


def _verify_schema(report: dict, prior: dict) -> None:
    required = list(range(1, 17))
    steps = report.get("steps")
    if not isinstance(steps, list) or [row.get("step") for row in steps] != required:
        raise ValueError("record_identity_or_coverage")
    if report["record_coverage"].get("required_steps") != required:
        raise ValueError("declared_record_coverage")
    for step in steps:
        if set(step["directions"]) != {"exposure", "equal_input"}:
            raise ValueError("weighting_identity")
        for weighting, record in step["directions"].items():
            if set(record) != {
                "full_negative",
                "minibatch_negative",
                "recorded_adam",
                "category",
            }:
                raise ValueError(
                    f"direction_record_identity:{step['step']}:{weighting}"
                )
            for name in ("full_negative", "minibatch_negative"):
                direction = record[name]
                forbidden = {"observed_loss_change", "remainder"}
                if forbidden.intersection(direction):
                    raise ValueError(
                        f"unevaluated_counterfactual_accounting:{step['step']}:{name}"
                    )
                if set(direction) != {
                    "norm",
                    "dot",
                    "cosine",
                    "sign",
                    "first_order_contribution",
                }:
                    raise ValueError(f"counterfactual_schema:{step['step']}:{name}")
                if direction["dot"] != direction["first_order_contribution"]:
                    raise ValueError("counterfactual_first_order_changed")
            adam = record["recorded_adam"]
            if set(adam) != {
                "norm",
                "dot",
                "cosine",
                "sign",
                "first_order_contribution",
                "observed_loss_change",
                "remainder",
            }:
                raise ValueError(
                    f"recorded_adam_accounting_schema:{step['step']}:{weighting}"
                )
            if adam["dot"] != adam["first_order_contribution"]:
                raise ValueError("recorded_adam_first_order_mismatch")
    for weighting in ("exposure", "equal_input"):
        old = [row["directions"][weighting] for row in prior["steps"]]
        new = [row["directions"][weighting] for row in steps]
        old_counts = {
            category: sum(r["category"] == category for r in old)
            for category in EXPECTED_CATEGORIES
        }
        new_counts = report["totals"][weighting]["category_counts"]
        if new_counts != old_counts:
            raise ValueError(f"category_counts_changed:{weighting}")
        totals = report["totals"][weighting]
        adam_records = [row["directions"][weighting]["recorded_adam"] for row in steps]
        observed = sum(record["observed_loss_change"] for record in adam_records)
        first_order = sum(record["first_order_contribution"] for record in adam_records)
        remainder = sum(record["remainder"] for record in adam_records)
        endpoint = (
            steps[-1]["losses"]["post"][f"unseen_{weighting}_policy_ce"]
            - steps[0]["losses"]["pre"][f"unseen_{weighting}_policy_ce"]
        )
        for name, expected_value in (
            ("observed_change_total", observed),
            ("first_order_total", first_order),
            ("remainder_total", remainder),
            ("endpoint_change", endpoint),
            ("endpoint_reconciliation_residual", observed - endpoint),
        ):
            if not math.isclose(
                totals[name], expected_value, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError(f"recorded_adam_total_mismatch:{weighting}:{name}")
        for direction in ("full_negative", "minibatch_negative", "recorded_adam"):
            for before, after in zip(old, new, strict=True):
                for key in (
                    "norm",
                    "dot",
                    "cosine",
                    "sign",
                    "first_order_contribution",
                ):
                    if before[direction][key] != after[direction][key]:
                        raise ValueError(
                            f"direction_value_changed:{weighting}:{direction}:{key}"
                        )
    training = report["training_totals"]
    training_records = [row["training_accounting"] for row in steps]
    observed = sum(record["observed_change"] for record in training_records)
    first_order = sum(record["first_order_contribution"] for record in training_records)
    remainder = sum(record["remainder"] for record in training_records)
    endpoint = (
        steps[-1]["losses"]["post"]["training_policy_ce"]
        - steps[0]["losses"]["pre"]["training_policy_ce"]
    )
    for name, expected_value in (
        ("observed_change_total", observed),
        ("first_order_total", first_order),
        ("remainder_total", remainder),
        ("endpoint_change", endpoint),
        ("endpoint_reconciliation_residual", observed - endpoint),
    ):
        if not math.isclose(
            training[name], expected_value, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError(f"training_total_mismatch:{name}")


def _expected_correction(prior: dict) -> dict:
    """Rebuild the semantic-only transform without calling publisher code."""
    expected = copy.deepcopy(prior)
    expected["schema"] = "seed477-counterfactual-accounting-correction-evidence-v1"
    expected["correction"] = {
        "status": "post_execution_semantic_correction",
        "reason": (
            "The prior supplement copied recorded-Adam endpoint loss changes into "
            "counterfactual records. Counterfactual endpoints were not evaluated, "
            "so those observed-change and remainder fields are removed."
        ),
        "counterfactual_endpoint_evaluated": False,
        "historical_measurements_changed": False,
    }
    for step in expected["steps"]:
        for weighting in ("exposure", "equal_input"):
            directions = step["directions"][weighting]
            for field in (
                "observed_change",
                "actual_first_order_contribution",
                "actual_remainder",
            ):
                directions.pop(field, None)
            for label in ("full_negative", "minibatch_negative"):
                directions[label].pop("observed_loss_change", None)
                directions[label].pop("remainder", None)
            adam = directions["recorded_adam"]
            adam["observed_loss_change"] = (
                step["losses"]["post"][f"unseen_{weighting}_policy_ce"]
                - step["losses"]["pre"][f"unseen_{weighting}_policy_ce"]
            )
            adam["first_order_contribution"] = adam["dot"]
            adam["remainder"] = adam["observed_loss_change"] - adam["dot"]
    return expected


def _snapshot(directories: tuple[Path, ...]) -> dict:
    result = {}
    for directory in directories:
        for path in directory.iterdir():
            if path.is_file():
                result[str(path)] = (
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    path.stat().st_mtime_ns,
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    verify(args.root)
    print("seed477 counterfactual accounting verification passed")


if __name__ == "__main__":
    main()
