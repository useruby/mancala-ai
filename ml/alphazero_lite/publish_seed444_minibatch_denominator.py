"""Publish the read-only seed444 corrected-rerun analysis and hash receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.seed444_minibatch_denominator import (
    AUDIT_DIR,
    CORRECTED_EVIDENCE,
    CORRECTED_REGISTRATION,
    sha,
)


def _analysis(evidence: dict[str, Any]) -> str:
    lines = [
        "# seed444 — frozen-checkpoint minibatch denominator audit",
        "",
        "> **Post-observation corrected rerun.** The original registration and initial evidence are preserved byte-for-byte. The initial E=0 observations preceded discovery of the unused-parameter handling defect and are superseded for measurement purposes. Corrected execution hashes were bound after that observation and before this rerun.",
        "",
        "## Coefficient coverage",
        "",
    ]
    coeff = evidence["coefficient_audit"]
    lines.extend(
        [
            f"- Training exposures N: **{coeff['N']:,}** across {coeff['unique_training_compact_rows']:,} unique compact rows ({coeff['exposure_copies_beyond_unique_rows']:,} repeated exposures retained).",
            f"- Original float32 coefficients: positive {coeff['positive_count']:,}, zero {coeff['zero_count']:,}, min/max {coeff['minimum']:.9g}/{coeff['maximum']:.9g}; Q={coeff['Q']:.9g}.",
            f"- Expanded little-endian float32 coefficient-vector SHA256: `{coeff['expanded_float32_le_sha256']}`.",
            "- Independently enumerating every exposure found that all coefficients equal exactly 1.0f. Therefore S_b=n_b in every batch, Q=N, S_b/Q=n_b/N, and the two policy aggregate objectives are structurally identical at every parameter state. The numerical rerun still computed the differences; no result was forced to zero.",
            "",
            "## Fixed-checkpoint measurements",
            "",
            "Both checkpoints use all 134,502 training exposures in the first frozen epoch order, batch size 512, and the final partial batch (358 exposures). Values below are unclipped fixed-parameter gradients, not Adam updates or expectations over random permutations.",
            "",
            "| Checkpoint | N | Q | ||G_global|| | ||G_batch|| | E | policy cosine | total cosine | direct reconciliation L2 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("initializer", "adam_a16"):
        result = evidence["checkpoints"][name]
        lines.append(
            f"| {name} | {result['N']} | {result['Q']:.1f} | {result['global_norm']:.12g} | {result['batch_norm']:.12g} | {result['E']:.12g} | {result['policy_global_vs_batch']['cosine']:.12g} | {result['total_batch_vs_global']['cosine']:.12g} | {result['direct_reconciliation_l2']:.4g} |"
        )
    lines.extend(
        [
            "",
            "The denominator ratio n_b·Q/(N·S_b) is 1 for every positive-mass batch at both checkpoints. Grouped squared differences are recorded for shared trunk, policy head, and value head and reconcile to the full-vector squared norm. The batch ledger includes exposure and coefficient counts, source/bucket counts, and ordered position/compact-row hashes.",
            "",
            "## Equal-update weighting and unseen alignment",
            "",
            "The equal-update mean across minibatches is reported separately. Its difference from the exposure-weighted batch mean isolates final-partial-batch weighting from denominator choice; per checkpoint its vector norm and relative effect appear in corrected evidence. Strict unseen >32 policy-CE alignment is descriptive only and reuses the corrected #438 vectors after validating their checkpoint, source-registration/split, population, weighting, parameter-layout, protocol, source and archive identities.",
            "",
            "| Checkpoint | Weighting | dot(U,G_global) | cosine | unit-descent alignment |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for name in ("initializer", "adam_a16"):
        for weighting in ("exposure_weighted", "equal_input"):
            alignment = evidence["checkpoints"][name][
                "unseen_policy_alignment_descriptive_only"
            ][weighting]["global_total"]
            lines.append(
                f"| {name} | {weighting} | {alignment['dot_product']:.12g} | {alignment['cosine']:.12g} | {alignment['unit_descent_alignment']:.12g} |"
            )
    lines.extend(
        [
            "",
            "## Fixed decision and interpretation",
            "",
            f"**Original decision classification retained:** `{evidence['decision']['original_classification_retained']}`. The structural finding is `{str(evidence['structural_denominator_identity']).lower()}`: for this frozen dataset, changing between these denominators is an exact objective-aggregation no-op. This does not establish a training bug, causality, reopen a closed branch, or authorize a promotion or experiment. A material mismatch in other coefficient data would require a separately preregistered ablation.",
            "",
            "No training, optimizer update, search, game, export, or promotion was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def publish(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / AUDIT_DIR
    evidence_path = out / CORRECTED_EVIDENCE
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    analysis_path = out / "analysis.md"
    layout_path = out / "corrected-parameter-layout.json"
    registration = json.loads(
        (out / CORRECTED_REGISTRATION).read_text(encoding="utf-8")
    )
    analysis_path.write_text(_analysis(evidence), encoding="utf-8")
    layout_path.write_text(
        json.dumps(registration["inputs"]["parameter_layout"], indent=2) + "\n",
        encoding="utf-8",
    )
    artifact_relatives = [
        f"{AUDIT_DIR}/registration.json",
        f"{AUDIT_DIR}/evidence.json",
        f"{AUDIT_DIR}/corrected-rerun-registration.json",
        f"{AUDIT_DIR}/{CORRECTED_REGISTRATION}",
        f"{AUDIT_DIR}/correction-receipt.json",
        f"{AUDIT_DIR}/correction-receipt-amendment-1.json",
        f"{AUDIT_DIR}/{CORRECTED_EVIDENCE}",
        f"{AUDIT_DIR}/corrected-batch-ledger.json",
        f"{AUDIT_DIR}/corrected-coefficient-audit.json",
        f"{AUDIT_DIR}/corrected-aggregate-vectors.npz",
        f"{AUDIT_DIR}/corrected-parameter-layout.json",
        f"{AUDIT_DIR}/analysis.md",
    ]
    source_relatives = [
        "ml/alphazero_lite/seed444_minibatch_denominator.py",
        "ml/alphazero_lite/verify_seed444_minibatch_denominator.py",
        "ml/alphazero_lite/test_seed444_minibatch_denominator.py",
        "ml/alphazero_lite/publish_seed444_minibatch_denominator.py",
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/verify_seed434_census_source.py",
        "ml/alphazero_lite/seed429_policy_normalization.py",
        "ml/alphazero_lite/exact_root_policy_targets.py",
        "ml/alphazero_lite/seed432_policy_target_census.py",
        "ml/alphazero_lite/seed433_census_correction.py",
        "ml/alphazero_lite/seed438_gradient_correction.py",
    ]
    input_relatives = [
        "docs/data/seed416-policy-target-softening/registration-v3.json",
        "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz",
        "docs/data/seed416-policy-target-softening/training-freeze-v3/epoch-permutations.json.gz",
        "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
        "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
        "docs/data/seed435-adam-direction-screen/A-final.npz",
        "docs/data/seed438-seed437-gradient-correction/protocol.json",
        "docs/data/seed438-seed437-gradient-correction/correction-receipt.json",
        "docs/data/seed438-seed437-gradient-correction/corrected-results.json",
        "docs/data/seed438-seed437-gradient-correction/corrected-gradient-vectors.npz",
        *[
            f"docs/data/seed426-canonical-overlap/sources/{name}.jsonl.gz"
            for name in (
                "fresh",
                "generic_bootstrap",
                "random_teacher",
                "opening_disagreement",
                "stability",
            )
        ],
    ]
    receipt = {
        "schema": "seed444-corrected-rerun-publication-receipt-v1",
        "status": "post-observation corrected diagnostic; read-only verifier bound",
        "original_registration_sha256": sha(out / "registration.json"),
        "original_evidence_sha256": sha(out / "evidence.json"),
        "corrected_rerun_registration_sha256": sha(out / CORRECTED_REGISTRATION),
        "corrected_evidence_sha256": sha(evidence_path),
        "artifact_sha256": {
            relative: sha(root / relative) for relative in artifact_relatives
        },
        "source_sha256": {
            relative: sha(root / relative) for relative in source_relatives
        },
        "input_sha256": {
            relative: sha(root / relative) for relative in input_relatives
        },
        "decision_classification": evidence["decision"]["classification"],
        "no_training_or_optimizer_updates": True,
    }
    receipt_path = out / "receipt.json"
    if receipt_path.exists():
        raise ValueError("publication_receipt_is_immutable")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(publish(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
