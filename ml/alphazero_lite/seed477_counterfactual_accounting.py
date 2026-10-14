"""Append-only semantic correction for seed476 counterfactual accounting."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

BASE = Path("docs/data/seed476-direction-transport")
PRIOR = Path("docs/data/seed476-direction-transport-supplement-v2")
OUT = Path("docs/data/seed477-counterfactual-accounting")
ROOT = Path(__file__).resolve().parents[2]
PUBLISHER = "ml/alphazero_lite/publish_seed477_counterfactual_accounting.py"
VERIFIER = "ml/alphazero_lite/verify_seed477_counterfactual_accounting.py"
TESTS = "ml/alphazero_lite/test_seed477_counterfactual_accounting.py"
SOURCES = (
    PUBLISHER,
    VERIFIER,
    TESTS,
    "ml/alphazero_lite/seed477_counterfactual_accounting.py",
)
INPUTS = {
    "seed476_registration": BASE / "registration.json",
    "seed476_receipt": BASE / "receipt.json",
    "seed476_evidence": BASE / "evidence.json",
    "seed476_vectors": BASE / "directions.npz",
    "prior_registration": PRIOR / "registration.json",
    "prior_receipt": PRIOR / "receipt.json",
    "prior_evidence": PRIOR / "evidence.json",
    "prior_accounting": PRIOR / "accounting.npz",
    "prior_report": PRIOR / "results.md",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corrected(prior: dict) -> dict:
    """Remove unmeasured endpoint fields; retain recorded displacement accounting."""
    result = copy.deepcopy(prior)
    result["schema"] = "seed477-counterfactual-accounting-correction-evidence-v1"
    result["correction"] = {
        "status": "post_execution_semantic_correction",
        "reason": (
            "The prior supplement copied recorded-Adam endpoint loss changes into "
            "counterfactual records. Counterfactual endpoints were not evaluated, "
            "so those observed-change and remainder fields are removed."
        ),
        "counterfactual_endpoint_evaluated": False,
        "historical_measurements_changed": False,
    }
    for step in result["steps"]:
        for weighting in ("exposure", "equal_input"):
            record = step["directions"][weighting]
            record.pop("observed_change", None)
            record.pop("actual_first_order_contribution", None)
            record.pop("actual_remainder", None)
            for direction in ("full_negative", "minibatch_negative"):
                record[direction].pop("observed_loss_change", None)
                record[direction].pop("remainder", None)
            # The recorded displacement has an evaluated endpoint and keeps the
            # observed change, first-order contribution, and finite-step remainder.
            adam = record["recorded_adam"]
            adam["observed_loss_change"] = (
                step["losses"]["post"][f"unseen_{weighting}_policy_ce"]
                - step["losses"]["pre"][f"unseen_{weighting}_policy_ce"]
            )
            adam["first_order_contribution"] = adam["dot"]
            adam["remainder"] = adam["observed_loss_change"] - adam["dot"]
    result["totals"] = {key: value for key, value in result["totals"].items()}
    return result


def publish(root: Path) -> None:
    root = root.resolve()
    out = root / OUT
    if out.exists() and any(out.iterdir()):
        raise ValueError("seed477_output_exists")
    prior = json.loads((root / PRIOR / "evidence.json").read_text())
    report = corrected(prior)
    out.mkdir(parents=True, exist_ok=True)
    inputs = {key: sha256(root / rel) for key, rel in INPUTS.items()}
    registration = {
        "schema": "seed477-counterfactual-accounting-registration-v1",
        "status": "post_execution_semantic_correction_frozen",
        "timing": "post-execution semantic correction; no new measurements",
        "original_seed476_registration_sha256": inputs["seed476_registration"],
        "original_seed476_receipt_sha256": inputs["seed476_receipt"],
        "original_seed476_evidence_sha256": inputs["seed476_evidence"],
        "original_seed476_vectors_sha256": inputs["seed476_vectors"],
        "prior_supplement_registration_sha256": inputs["prior_registration"],
        "prior_supplement_receipt_sha256": inputs["prior_receipt"],
        "prior_supplement_evidence_sha256": inputs["prior_evidence"],
        "prior_supplement_accounting_sha256": inputs["prior_accounting"],
        "inputs_sha256": inputs,
        "source_sha256": {name: sha256(root / name) for name in SOURCES},
        "transformation": "remove unevaluated counterfactual endpoint loss accounting; retain all direction geometry and evaluated Adam endpoint accounting",
        "claim_scope": "descriptive semantic correction only; no causal, promotion, or strength claim",
    }
    (out / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    (out / "evidence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (out / "results.md").write_text(render(report, registration))
    receipt = {
        "status": "seed477_counterfactual_accounting_bound",
        **{
            name: sha256(out / name)
            for name in ("registration.json", "evidence.json", "results.md")
        },
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def render(report: dict, registration: dict) -> str:
    lines = [
        "# Seed477 counterfactual accounting correction",
        "",
        "Post-execution semantic correction to seed476 supplement evidence. The prior report copied the recorded-Adam endpoint loss change into `full_negative` and `minibatch_negative` records and subtracted their first-order terms. Those counterfactual endpoints were never evaluated; the observed-change and remainder fields are therefore removed. No counterfactual forward evaluations, gradients, training, searches, or games were performed.",
        "",
        "All counterfactual norms, dot products, cosines, signs, and first-order contributions are retained. Only recorded-Adam displacement records retain an evaluated observed change, first-order contribution, and remainder. Original source and evidence files are preserved byte-for-byte.",
        "",
        f"Original seed476 registration SHA-256: `{registration['original_seed476_registration_sha256']}`",
        f"Original seed476 receipt SHA-256: `{registration['original_seed476_receipt_sha256']}`",
        f"Original seed476 evidence SHA-256: `{registration['original_seed476_evidence_sha256']}`",
        f"Prior supplement registration/receipt/evidence SHA-256: `{registration['prior_supplement_registration_sha256']}` / `{registration['prior_supplement_receipt_sha256']}` / `{registration['prior_supplement_evidence_sha256']}`",
        "",
        "## Recorded Adam endpoint reconciliation",
        "",
    ]
    for weighting, values in report["totals"].items():
        lines.append(
            f"- {weighting}: observed total {values['observed_change_total']:.12g}; first-order total {values['first_order_total']:.12g}; remainder total {values['remainder_total']:.12g}; endpoint change {values['endpoint_change']:.12g}; residual {values['endpoint_reconciliation_residual']:.6g}."
        )
        lines.append(f"  Category counts: {values['category_counts']}")
    t = report["training_totals"]
    lines += [
        f"- Training: observed total {t['observed_change_total']:.12g}; first-order total {t['first_order_total']:.12g}; remainder total {t['remainder_total']:.12g}; endpoint change {t['endpoint_change']:.12g}; residual {t['endpoint_reconciliation_residual']:.6g}.",
        "",
        "First-order direction categories and their counts are unchanged from the prior supplement; the correction only removes unsupported counterfactual finite-step fields. Retrospective descriptive evidence only; no causal, promotion, or strength claim.",
        "",
    ]
    return "\n".join(lines)
