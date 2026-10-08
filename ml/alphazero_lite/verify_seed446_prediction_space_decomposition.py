"""Independent read-only semantic verifier for the retrospective seed446 result."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import train

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed446-prediction-space-decomposition")
ARMS = ("A", "B")
WEIGHTINGS = ("exposure_weighted", "equal_input")
ATOL = 2e-7
RTOL = 2e-6


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_targets(root: Path) -> tuple[np.ndarray, list[int], list[str]]:
    """Rebuild targets through the registered loader, not publication values."""
    registration = json.loads(
        (
            root / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    specs = registration["replays"]
    scratch_root = root / ".tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="seed446-loader-", dir=scratch_root
    ) as scratch:
        paths: list[Path] = []
        for spec in specs:
            source = (
                root
                / "docs/data/seed426-canonical-overlap/sources"
                / f"{spec['name']}.jsonl.gz"
            )
            target_path = Path(scratch) / f"{spec['name']}.jsonl"
            with (
                gzip.open(source, "rb") as source_stream,
                target_path.open("wb") as target_stream,
            ):
                while chunk := source_stream.read(1024 * 1024):
                    target_stream.write(chunk)
            paths.append(target_path)
        loaded = train.load_jsonl_replay(
            paths,
            [int(spec["weight"]) for spec in specs],
            policy_target_mode="sharpened",
            value_target_mode="default",
            replay_value_target_modes=[spec["value_target_mode"] for spec in specs],
            include_policy_loss_weights=True,
        )
    all_targets = np.asarray(loaded[2], dtype=np.float32).reshape(-1)
    with gzip.open(
        root
        / "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
        "rt",
        encoding="utf-8",
    ) as stream:
        membership = [json.loads(line) for line in stream]
    selected = [
        row
        for row in membership
        if row["subset"] == "unseen" and row["active_stones"] > 32
    ]
    compact_rows = [int(row["compact_row"]) for row in selected]
    identities = [str(row["input_identity"]) for row in selected]
    if len(compact_rows) != 2607 or len(set(identities)) != 1242:
        raise ValueError("population_invalid")
    source_targets = all_targets[np.asarray(compact_rows, dtype=np.int64)]
    if not np.isfinite(source_targets).all():
        raise ValueError("authoritative_targets_nonfinite")
    archived_ids = json.loads(
        (
            root / "docs/data/seed445-finite-step-attribution/row-identities.json"
        ).read_text()
    )
    if (
        archived_ids["compact_row"] != compact_rows
        or archived_ids["input_identity"] != identities
    ):
        raise ValueError("source_identity_correspondence_invalid")
    return source_targets, compact_rows, identities


def _weight_vectors(identities: list[str]) -> dict[str, np.ndarray]:
    counts: dict[str, int] = {}
    for identity in identities:
        counts[identity] = counts.get(identity, 0) + 1
    return {
        "exposure_weighted": np.full(
            len(identities), 1.0 / len(identities), dtype=np.float64
        ),
        "equal_input": np.asarray(
            [1.0 / (len(counts) * counts[key]) for key in identities], dtype=np.float64
        ),
    }


def _check(condition: Any, code: str) -> None:
    if not bool(condition):
        raise ValueError(code)


def validate_semantics(
    targets: np.ndarray,
    compact_rows: list[int],
    identities: list[str],
    predictions: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    states: dict[str, list[list[np.ndarray]]],
    layout: list[dict[str, Any]],
    seed445_ledger: dict[str, Any],
    published_targets: dict[str, Any],
    published_ledger: dict[str, Any],
) -> dict[str, Any]:
    """Validate arithmetic directly, independent of any receipt hash gate."""
    _check(
        published_targets["compact_row"] == compact_rows,
        "published_compact_rows_mismatch",
    )
    _check(
        published_targets["input_identity"] == identities,
        "published_identities_mismatch",
    )
    _check(
        np.array_equal(
            np.asarray(published_targets["value_target"], dtype=np.float32), targets
        ),
        "published_targets_mismatch",
    )
    _check(
        len(layout) == 22 and all(entry["stop"] > entry["start"] for entry in layout),
        "parameter_layout_invalid",
    )
    baseline_keys = [
        (arm, row["step"], row["weighting"])
        for arm in ARMS
        for row in seed445_ledger["arms"][arm]["step_ledger"]
        if row["objective"] == "value_mse"
    ]
    expected_baseline_keys = {
        (arm, step, weighting)
        for arm in ARMS
        for step in range(1, 17)
        for weighting in WEIGHTINGS
    }
    _check(
        set(baseline_keys) == expected_baseline_keys and len(baseline_keys) == 64,
        "seed445_ledger_coverage_invalid",
    )
    weight_vectors = _weight_vectors(identities)
    for name, vector in weight_vectors.items():
        _check(
            np.isfinite(vector).all()
            and np.isclose(vector.sum(), 1.0, atol=ATOL, rtol=RTOL),
            f"weight_mass_invalid:{name}",
        )
    rows: list[dict[str, Any]] = []
    totals: dict[str, dict[str, dict[str, float]]] = {}
    seen: set[tuple[str, str, int]] = set()
    for arm in ARMS:
        _check(len(states[arm]) == 17, f"state_count_invalid:{arm}")
        for step in range(16):
            pre, post = states[arm][step], states[arm][step + 1]
            _check(
                len(pre) == len(post) == len(layout),
                f"state_layout_invalid:{arm}:{step}",
            )
            for left, right, entry in zip(pre, post, layout, strict=True):
                _check(
                    list(left.shape) == entry["shape"]
                    and list(right.shape) == entry["shape"],
                    "parameter_shape_invalid",
                )
                _check(
                    left.dtype == np.float32
                    and right.dtype == np.float32
                    and np.isfinite(left).all()
                    and np.isfinite(right).all(),
                    "parameter_state_invalid",
                )
            displacement = np.concatenate(
                [
                    (right.astype(np.float64) - left.astype(np.float64)).reshape(-1)
                    for left, right in zip(pre, post, strict=True)
                ]
            )
            for weighting in WEIGHTINGS:
                key = (arm, weighting, step + 1)
                _check(key not in seen, "duplicate_step")
                seen.add(key)
                before = np.asarray(predictions[f"{arm}_{step:02d}_values"])
                after = np.asarray(predictions[f"{arm}_{step + 1:02d}_values"])
                grad = np.asarray(gradients[f"{arm}_value_mse_{weighting}_{step:02d}"])
                _check(
                    before.shape == after.shape == targets.shape
                    and np.isfinite(before).all()
                    and np.isfinite(after).all(),
                    "prediction_invalid",
                )
                _check(
                    grad.ndim == displacement.ndim == 1
                    and grad.shape == displacement.shape
                    and np.isfinite(grad).all(),
                    "gradient_invalid",
                )
                weight = weight_vectors[weighting]
                delta = after.astype(np.float64) - before.astype(np.float64)
                e = before.astype(np.float64) - targets.astype(np.float64)

                def mean(value: np.ndarray) -> float:
                    return float(
                        np.sum(
                            np.asarray(value, dtype=np.float64) * weight,
                            dtype=np.float64,
                        )
                        / np.sum(weight, dtype=np.float64)
                    )

                c = 2.0 * mean(e * delta)
                q = mean(delta * delta)
                s = float(np.dot(grad.astype(np.float64), displacement))
                n = c - s
                d = mean((after.astype(np.float64) - targets) ** 2) - mean(e**2)
                r = d - s
                _check(
                    np.isclose(d, c + q, atol=ATOL, rtol=RTOL), "identity_d_c_q_failed"
                )
                _check(
                    np.isclose(r, q + n, atol=ATOL, rtol=RTOL), "identity_r_q_n_failed"
                )
                archived = next(
                    (
                        record
                        for record in seed445_ledger["arms"][arm]["step_ledger"]
                        if record["step"] == step + 1
                        and record["objective"] == "value_mse"
                        and record["weighting"] == weighting
                    ),
                    None,
                )
                _check(archived is not None, "seed445_step_missing")
                assert archived is not None
                _check(
                    np.isclose(d, archived["d"], atol=ATOL, rtol=RTOL)
                    and np.isclose(s, archived["s"], atol=ATOL, rtol=RTOL),
                    "seed445_dot_or_displacement_mismatch",
                )
                rows.append(
                    {
                        "arm": arm,
                        "weighting": weighting,
                        "step": step + 1,
                        "c": c,
                        "q": q,
                        "s": s,
                        "n": n,
                        "d": d,
                    }
                )
    _check(len(seen) == 64 and len(rows) == 64, "step_coverage_invalid")
    totals: dict[str, dict[str, dict[str, float]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for weighting in WEIGHTINGS:
            selected = [
                row
                for row in rows
                if row["arm"] == arm and row["weighting"] == weighting
            ]
            _check(len(selected) == 16, "step_count_invalid")
            summed = {
                name: float(sum(row[name] for row in selected))
                for name in ("c", "q", "s", "n", "d")
            }
            _check(
                np.isclose(
                    summed["d"], summed["c"] + summed["q"], atol=ATOL, rtol=RTOL
                ),
                "total_d_c_q_failed",
            )
            _check(
                np.isclose(
                    summed["d"],
                    summed["s"] + summed["q"] + summed["n"],
                    atol=ATOL,
                    rtol=RTOL,
                ),
                "total_d_s_q_n_failed",
            )
            totals[arm][weighting] = summed
    rows.sort(
        key=lambda row: (
            ARMS.index(row["arm"]),
            WEIGHTINGS.index(row["weighting"]),
            row["step"],
        )
    )
    expected_rows = published_ledger["rows"]
    _check(len(expected_rows) == len(rows), "published_ledger_coverage_invalid")
    for actual, expected in zip(rows, expected_rows, strict=True):
        _check(
            {k: actual[k] for k in ("arm", "weighting", "step")}
            == {k: expected[k] for k in ("arm", "weighting", "step")},
            "published_ledger_identity_invalid",
        )
        for term in ("c", "q", "s", "n", "d"):
            _check(
                np.isclose(actual[term], expected[term], atol=ATOL, rtol=RTOL),
                f"published_ledger_{term}_invalid",
            )
    for arm in ARMS:
        for weighting in WEIGHTINGS:
            for term, value in totals[arm][weighting].items():
                _check(
                    np.isclose(
                        value,
                        published_ledger["totals"][arm][weighting][term],
                        atol=ATOL,
                        rtol=RTOL,
                    ),
                    f"published_total_{term}_invalid",
                )
    component_class = {
        weighting: _classify(totals["B"][weighting]["q"], totals["B"][weighting]["n"])
        for weighting in WEIGHTINGS
    }
    classification = (
        component_class[WEIGHTINGS[0]]
        if len(set(component_class.values())) == 1
        else "mixed_or_no_common_residual_component"
    )
    _check(
        published_ledger["weighting_classifications"] == component_class
        and published_ledger["classification"] == classification,
        "classification_invalid",
    )
    return {
        "status": "valid",
        "classification": classification,
        "weighting_classifications": component_class,
        "exposures": len(targets),
    }


def _classify(q: float, n: float) -> str:
    residual = q + n
    if residual > 0 and q >= 0.75 * residual:
        return "squared_prediction_movement_dominant"
    if residual > 0 and n >= 0.75 * residual:
        return "output_response_remainder_dominant"
    return "mixed_or_no_common_residual_component"


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Reject changes to the frozen weighting/arithmetic/classification contract."""
    _check(
        protocol.get("weightings")
        == {
            "exposure_weighted": "uniform across all exposure rows",
            "equal_input": "equal identity mass, retaining within-identity exposures",
        },
        "weighting_definition_invalid",
    )
    _check(
        protocol.get("arithmetic")
        == "float64 from archived float32 predictions/targets; c=2*weighted_mean((pre-target)*delta); q=weighted_mean(delta^2); s=dot(archived full gradient,float64 adjacent-state difference); n=c-s; d=c+q",
        "arithmetic_protocol_invalid",
    )
    _check(
        protocol.get("classification")
        == "B per weighting: R=Q+N; if R>0 and Q>=.75R squared_prediction_movement_dominant; else if R>0 and N>=.75R output_response_remainder_dominant; else mixed_or_no_common_residual_component",
        "classification_protocol_invalid",
    )


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    validate_protocol(json.loads((out / "protocol.json").read_text()))
    supplement = json.loads((out / "supplemental-receipt.json").read_text())

    def check_binding(condition: Any, message: str) -> None:
        if not bool(condition):
            raise ValueError(message)

    check_binding(
        supplement["status"] == "post_execution_verifier_and_test_binding",
        "supplement_status_invalid",
    )
    for rel, expected in supplement["bound_sha256"].items():
        check_binding(_sha(root / rel) == expected, f"supplement_hash_mismatch:{rel}")
    publication = json.loads((out / "receipt.json").read_text())
    for rel, expected in publication["artifacts_sha256"].items():
        check_binding(
            _sha(out / rel) == expected, f"publication_artifact_hash_mismatch:{rel}"
        )
    targets, compact_rows, identities = _load_targets(root)
    pub_targets = json.loads((out / "ordered-targets.json").read_text())
    published = json.loads((out / "step-ledger.json").read_text())
    layout = json.loads(
        (
            root / "docs/data/seed445-finite-step-attribution/parameter-layout.json"
        ).read_text()
    )
    predictions_npz = np.load(
        root / "docs/data/seed445-finite-step-attribution/predictions.npz",
        allow_pickle=False,
    )
    gradients_npz = np.load(
        root / "docs/data/seed445-finite-step-attribution/gradients.npz",
        allow_pickle=False,
    )
    predictions = {
        key: predictions_npz[key]
        for key in predictions_npz.files
        if key.endswith("_values")
    }
    if not np.array_equal(predictions_npz["row_identities"], np.asarray(identities)):
        raise ValueError("prediction_row_identities_invalid")
    gradients = {key: gradients_npz[key] for key in gradients_npz.files}
    model = train.PolicyValueNet((96, 3), "residual_v3", 27)
    train.load_checkpoint_into_model(
        model,
        root
        / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
    )
    initial = [
        parameter.detach().cpu().numpy().copy() for parameter in model.parameters()
    ]
    expected_layout: list[dict[str, Any]] = []
    offset = 0
    for name, parameter in model.named_parameters():
        group = (
            "shared_trunk"
            if name.startswith(("input_layer.", "residual_layers."))
            else "policy_head"
            if name.startswith(("policy_hidden_layer.", "policy_head."))
            else "value_head"
            if name.startswith(("value_hidden_layer.", "value_head."))
            else None
        )
        if group is None:
            raise ValueError(f"unclassified_parameter:{name}")
        stop = offset + parameter.numel()
        expected_layout.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "start": offset,
                "stop": stop,
                "group": group,
            }
        )
        offset = stop
    if layout != expected_layout:
        raise ValueError("parameter_layout_mismatch")
    archive = np.load(
        root / "docs/data/seed442-kl-capped-adam-screen/step-tensors.npz",
        allow_pickle=False,
    )
    states: dict[str, list[list[np.ndarray]]] = {}
    for arm in ARMS:
        arm_states = [[item.copy() for item in initial]]
        for step in range(16):
            pre = [
                archive[f"{arm}_{step:02d}_pre_{i:02d}"] for i in range(len(initial))
            ]
            post = [
                archive[f"{arm}_{step:02d}_post_{i:02d}"] for i in range(len(initial))
            ]
            if any(
                not np.array_equal(x, y)
                for x, y in zip(arm_states[-1], pre, strict=True)
            ):
                raise ValueError(f"trajectory_discontinuity:{arm}:{step}")
            arm_states.append([x.copy() for x in post])
        train.load_checkpoint_into_model(
            model,
            root / f"docs/data/seed442-kl-capped-adam-screen/{arm}-final.npz",
        )
        if any(
            not np.array_equal(state, parameter.detach().cpu().numpy())
            for state, parameter in zip(arm_states[-1], model.parameters(), strict=True)
        ):
            raise ValueError(f"trajectory_endpoint_invalid:{arm}")
        states[arm] = arm_states
    seed445_ledger = json.loads(
        (
            root / "docs/data/seed445-finite-step-attribution/step-ledger.json"
        ).read_text()
    )
    report = validate_semantics(
        targets,
        compact_rows,
        identities,
        predictions,
        gradients,
        states,
        layout,
        seed445_ledger,
        pub_targets,
        published,
    )
    seed442_evidence = json.loads(
        (root / "docs/data/seed442-kl-capped-adam-screen/evidence.json").read_text()
    )
    if seed442_evidence["decision"]["classification"] != "close_kl_capped_step_branch":
        raise ValueError("seed442_historical_classification_changed")
    report["preserved_historical_classifications"] = {
        "seed445": "finite_step_value_residual_dominant",
        "seed442": seed442_evidence["decision"]["classification"],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
