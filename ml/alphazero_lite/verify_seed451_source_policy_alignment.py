"""Independent read-only semantic verifier for seed451 publications."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(root: Path) -> dict[str, object]:
    root = root.resolve()
    out = root / "docs/data/seed451-source-policy-alignment"
    registration = json.loads((out / "registration.json").read_text())
    result = json.loads((out / "results.json").read_text())
    receipt = json.loads((out / "receipt.json").read_text())
    for name, expected in receipt["files_sha256"].items():
        digest = hashlib.sha256((out / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"publication_hash_mismatch:{name}")
    for name, expected in registration["source_sha256"].items():
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"source_hash_mismatch:{name}")
    for name, expected in registration["inputs_sha256"].items():
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"input_hash_mismatch:{name}")
    with np.load(out / "gradient-vectors.npz", allow_pickle=False) as archive:
        if set(archive.files) != {
            f"{checkpoint}__{name}"
            for checkpoint in registration["checkpoints"]
            for name in (
                "G_PF",
                "G_PH",
                "G_V",
                "G_P",
                "G_T",
                "U_F_exposure",
                "U_F_equal_exact_input",
                "U_H_exposure",
                "U_H_equal_exact_input",
            )
        }:
            raise ValueError("vector_coverage_mismatch")
        for checkpoint, record in result["checkpoints"].items():
            vec = {
                key: np.asarray(archive[f"{checkpoint}__{key}"], dtype=np.float64)
                for key in (
                    "G_PF",
                    "G_PH",
                    "G_V",
                    "G_P",
                    "G_T",
                    "U_F_exposure",
                    "U_F_equal_exact_input",
                    "U_H_exposure",
                    "U_H_equal_exact_input",
                )
            }
            sizes = {
                item["stop"] - item["start"] for item in result["parameter_layout"]
            }
            total = result["parameter_layout"][-1]["stop"]
            cursor = 0
            for item in result["parameter_layout"]:
                if item["start"] != cursor or item["stop"] - item["start"] != int(
                    np.prod(item["shape"])
                ):
                    raise ValueError("parameter_layout_invalid")
                cursor = item["stop"]
            if (
                result["parameter_layout"]
                != registration["historical_seed438_parameter_layout"]
            ):
                raise ValueError("historical_parameter_layout_mismatch")
            if sizes and any(
                v.shape != (total,) or not np.isfinite(v).all() for v in vec.values()
            ):
                raise ValueError("vector_layout_or_finiteness_invalid")
            if not np.allclose(
                vec["G_P"],
                record["rho_F"] * vec["G_PF"] + record["rho_H"] * vec["G_PH"],
                atol=2e-5,
                rtol=2e-5,
            ):
                raise ValueError("policy_mixture_mismatch")
            if not np.allclose(
                vec["G_T"], vec["G_P"] + 0.3 * vec["G_V"], atol=2e-5, rtol=2e-5
            ):
                raise ValueError("joint_objective_mismatch")
            for cohort in (
                "U_F_exposure",
                "U_F_equal_exact_input",
                "U_H_exposure",
                "U_H_equal_exact_input",
            ):
                u = vec[cohort]
                metric = record["cohorts"][cohort]
                parts = {
                    "fresh": record["rho_F"] * float(np.dot(u, vec["G_PF"])),
                    "historical": record["rho_H"] * float(np.dot(u, vec["G_PH"])),
                    "value": 0.3 * float(np.dot(u, vec["G_V"])),
                }
                dot = float(np.dot(u, vec["G_T"]))
                if not np.isclose(sum(parts.values()), dot, atol=2e-5, rtol=2e-5):
                    raise ValueError("dot_reconciliation_mismatch")
                if not np.isclose(metric["joint_dot"], dot, atol=2e-5, rtol=2e-5):
                    raise ValueError("published_dot_mismatch")
                for group, group_record in result["parameter_groups"].items():
                    group_slice = np.asarray(group_record["indices"], dtype=np.int64)
                    group_report = metric["group_accounting"][group]
                    group_parts = {
                        "fresh": record["rho_F"]
                        * float(np.dot(u[group_slice], vec["G_PF"][group_slice])),
                        "historical": record["rho_H"]
                        * float(np.dot(u[group_slice], vec["G_PH"][group_slice])),
                        "value": 0.3
                        * float(np.dot(u[group_slice], vec["G_V"][group_slice])),
                    }
                    group_total = float(np.dot(u[group_slice], vec["G_T"][group_slice]))
                    if not np.isclose(
                        sum(group_parts.values()), group_total, atol=2e-5, rtol=2e-5
                    ):
                        raise ValueError("group_dot_reconciliation_mismatch")
                    for part, value in group_parts.items():
                        if not np.isclose(
                            group_report[part], value, atol=2e-5, rtol=2e-5
                        ):
                            raise ValueError("group_contribution_mismatch")
                    if not np.isclose(
                        group_report["joint"], group_total, atol=2e-5, rtol=2e-5
                    ):
                        raise ValueError("group_joint_mismatch")
                for objective, gradient in (
                    ("G_PF", vec["G_PF"]),
                    ("G_PH", vec["G_PH"]),
                    ("G_V", vec["G_V"]),
                    ("G_T", vec["G_T"]),
                ):
                    calculated = float(np.dot(u, gradient))
                    reported = metric["metrics"][objective]["dot"]
                    if not np.isclose(calculated, reported, atol=2e-5, rtol=2e-5):
                        raise ValueError("geometry_mismatch")
                    norm = float(np.linalg.norm(gradient))
                    cosine = (
                        None
                        if norm == 0 or np.linalg.norm(u) == 0
                        else calculated / (norm * np.linalg.norm(u))
                    )
                    derivative = None if norm == 0 else -calculated / norm
                    fields = metric["metrics"][objective]
                    if not np.isclose(fields["norm"], norm, atol=2e-5, rtol=2e-5):
                        raise ValueError("norm_mismatch")
                    for key, value in (
                        ("cosine", cosine),
                        ("unit_direction_loss_derivative", derivative),
                    ):
                        observed = fields[key]
                        if (
                            value is None
                            and observed is not None
                            or value is not None
                            and (
                                observed is None
                                or not np.isclose(observed, value, atol=2e-5, rtol=2e-5)
                            )
                        ):
                            raise ValueError(f"{key}_mismatch")
        uf = [
            result["checkpoints"][c]["cohorts"]["U_F_equal_exact_input"]["metrics"]
            for c in registration["checkpoints"]
        ]
        opposition = all(
            m["G_PH"]["cosine"] is not None
            and m["G_PH"]["cosine"] <= -0.1
            and m["G_PF"]["cosine"] is not None
            and m["G_PF"]["cosine"] >= 0.1
            for m in uf
        )
        positive = all(
            m["G_T"]["cosine"] is not None and m["G_T"]["cosine"] >= 0.1 for m in uf
        )
        label = (
            "historical_policy_opposition_present"
            if opposition
            else "fresh_first_order_alignment_positive"
            if positive
            else "mixed_or_checkpoint_dependent_alignment"
        )
        if result["classification"] != label:
            raise ValueError("classification_mismatch")
    return {"status": "valid", "classification": label}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
