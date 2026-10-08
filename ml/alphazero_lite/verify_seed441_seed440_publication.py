"""Independent structural and numeric verifier for seed441 completion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.seed441_seed440_publication import REL, WEIGHTINGS, generate


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numeric_equal(left: Any, right: Any, atol: float, rtol: float, label: str) -> None:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"invalid_numeric_coverage:{label}")
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise ValueError(f"numeric_mismatch:{label}")


def _compare_tree(left: Any, right: Any, atol: float, rtol: float, path: str) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError(f"ledger_keys_mismatch:{path}")
        for key in left:
            _compare_tree(left[key], right[key], atol, rtol, f"{path}.{key}")
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError(f"ledger_length_mismatch:{path}")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _compare_tree(a, b, atol, rtol, f"{path}[{index}]")
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        _numeric_equal([left], [right], atol, rtol, path)
    elif left != right:
        raise ValueError(f"ledger_value_mismatch:{path}")


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / REL
    completion = json.loads((out / "completion.json").read_text())
    expected = generate(root)
    for key in (
        "seed440_protocol_sha256",
        "seed440_bindings_sha256",
        "seed440_ledger_sha256",
        "classification",
        "membership",
    ):
        if completion.get(key) != expected[key]:
            raise ValueError(f"completion_binding_mismatch:{key}")
    bindings_file = (
        root / "docs/data/seed440-recorded-update-attribution/evidence-bindings.json"
    )
    bindings = json.loads(bindings_file.read_text())
    for section in ("seed435",):
        for item in bindings[section].values():
            if sha(root / item["path"]) != item["sha256"]:
                raise ValueError(f"historical_hash_mismatch:{item['path']}")
    for section in ("seed439_U_gradient_archive", "authoritative_membership"):
        item = bindings[section]
        if sha(root / item["path"]) != item["sha256"]:
            raise ValueError(f"historical_hash_mismatch:{item['path']}")
    protocol = json.loads(
        (
            root / "docs/data/seed440-recorded-update-attribution/protocol.json"
        ).read_text()
    )
    supplemental_protocol = json.loads((out / "protocol.json").read_text())
    source_root = Path(__file__).resolve().parents[2]
    for source, digest in supplemental_protocol["source_sha256"].items():
        if sha(source_root / source) != digest:
            raise ValueError(f"supplemental_source_hash_mismatch:{source}")
    for dependency, digest in supplemental_protocol["dependency_sha256"].items():
        base = source_root if dependency.startswith("ml/") else root
        if sha(base / dependency) != digest:
            raise ValueError(f"dependency_hash_mismatch:{dependency}")
    if supplemental_protocol["classification"]["primary_arm"] != "A":
        raise ValueError("primary_arm_mismatch")
    if (
        supplemental_protocol["paths"]["arms"] != ["A", "B"]
        or supplemental_protocol["paths"]["updates_per_arm"] != 16
    ):
        raise ValueError("path_protocol_mismatch")
    atol = protocol["arithmetic"]["tolerance"]["absolute"]
    rtol = protocol["arithmetic"]["tolerance"]["relative"]
    rows = json.loads((out / "row-identities.json").read_text())
    if len(rows["compact_rows"]) != 2607 or len(rows["input_identities"]) != 2607:
        raise ValueError("row_identity_coverage_mismatch")
    expected_state_order = [
        f"{arm}_{kind}_{step:02d}"
        for arm in ("A", "B")
        for kind, first, last in (("state", 0, 16), ("midpoint", 1, 16))
        for step in range(first, last + 1)
    ]
    if set(rows["state_order"]) != set(expected_state_order):
        raise ValueError("prediction_state_coverage_mismatch")
    with np.load(out / "predictions.npz", allow_pickle=False) as predictions:
        expected_prediction_keys = {
            f"{state}_{suffix}"
            for state in expected_state_order
            for suffix in ("policy_logits", "value")
        }
        if set(predictions.files) != expected_prediction_keys:
            raise ValueError("prediction_archive_coverage_mismatch")
        for key in predictions.files:
            if not np.isfinite(predictions[key]).all():
                raise ValueError(f"nonfinite_prediction:{key}")
    expected_gradient_keys = {
        f"{arm}_step_{step:02d}_{weight}"
        for arm in ("A", "B")
        for step in range(1, 17)
        for weight in WEIGHTINGS
    }
    with np.load(out / "weighting-gradients.npz", allow_pickle=False) as gradients:
        if set(gradients.files) != expected_gradient_keys:
            raise ValueError("gradient_archive_coverage_mismatch")
        for key in gradients.files:
            if not np.isfinite(gradients[key]).all():
                raise ValueError(f"nonfinite_gradient:{key}")
    reproduced = generate(root, publish=False)
    # Path and gradient evidence are reproduced directly; ledger comparison uses
    # registered numeric tolerances, never whole-dictionary float equality.
    with np.load(out / "weighting-gradients.npz", allow_pickle=False) as archived:
        for name in expected_gradient_keys:
            _numeric_equal(
                archived[name], reproduced["_gradients"][name], atol, rtol, name
            )
            if (
                archived[name].ndim != 1
                or archived[name].size != reproduced["_layout"][-1]["stop"]
            ):
                raise ValueError(f"gradient_layout_invalid:{name}")
    with np.load(out / "predictions.npz", allow_pickle=False) as archived:
        for name in archived.files:
            _numeric_equal(
                archived[name], reproduced["_predictions"][name], atol, rtol, name
            )
    if rows["state_order"] != reproduced["_state_order"]:
        raise ValueError("prediction_state_order_mismatch")
    layout = json.loads((out / "parameter-layout.json").read_text())
    if layout != reproduced["_layout"]:
        raise ValueError("parameter_layout_mismatch")
    if (
        rows["compact_rows"] != reproduced["_rows"]
        or rows["input_identities"] != reproduced["_identities"]
    ):
        raise ValueError("row_identity_mismatch")
    historical = json.loads(
        (
            root / "docs/data/seed440-recorded-update-attribution/step-ledger.json"
        ).read_text()
    )
    _compare_tree(historical["arms"], reproduced["_ledger"], atol, rtol, "arms")
    if (
        historical["classification"] != reproduced["_classification"]
        or completion["classification"] != historical["classification"]
    ):
        raise ValueError("classification_mismatch")
    _numeric_equal(
        [reproduced["membership"]["selected_exposures"]], [2607], 0, 0, "membership"
    )
    return {
        "status": "valid",
        "classification": completion["classification"],
        "states_per_arm": 17,
        "midpoints_per_arm": 16,
        "gradients_per_arm": 32,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
