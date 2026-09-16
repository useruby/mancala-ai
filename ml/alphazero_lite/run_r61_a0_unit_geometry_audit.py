#!/usr/bin/env python3
"""Diagnostic-only R61 A0-unit geometry and causal-patching audit.

This replays the historical R61 trajectories only when their required snapshots
are unavailable, then evaluates frozen models offline.  It does not change the
training procedure, replay, optimizer, or search.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

# ruff: noqa: E402
from ml.alphazero_lite.input_encodings import (
    BASE_FEATURE_ORDER,
    KALAH_V3_EXTRA_FEATURE_ORDER,
)
from ml.alphazero_lite.run_g1_replay_optimizer_crossover import (
    ANCHOR_ID,
    FROZEN_SET_SHA,
    G0_SHA,
    artifact_paths,
)
from ml.alphazero_lite.run_internal_cluster_learning_dynamics import MANIFEST_SCHEMA
from ml.alphazero_lite.run_r61_activation_representation_audit import (
    EPS,
    TOLERANCE,
    continue_policy_from_stage,
    legal_policy_metrics,
    residual_v3_activations,
    state_inputs,
)
from ml.alphazero_lite.run_r61_earliest_a0_divergence_audit import (
    EXPECTED_SHA,
    reconstruct_lane,
)
from ml.alphazero_lite.run_r61_early_trunk_replay_provenance_audit import (
    model_from_snapshot,
    tensor_snapshot,
    verify_r61_artifacts,
)
from ml.alphazero_lite.train import PolicyValueNet, load_checkpoint_into_model

SCHEMA = "azlite_r61_a0_unit_geometry_audit_v1"
LANDMARKS = ("g0", "pre80", "80", "81", "82", "108", "final")
PATCH_LANDMARKS = ("82", "108", "final")
PREFIX_SIZES = (1, 2, 4, 8, 16)
FINAL_SHAS = {
    "T61": "50736a6063cd0558329c433ad7a6c60b741c83ceee75d36e21e703749af8c9de",
    "T63": "bab60465a9bc93ac569ed28b356ef2abde2c1cbb0987a4a474e26117f9a57324",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def frozen_cohort(entry: dict[str, Any]) -> str:
    if entry["membership"] == "cluster_anchor":
        return "anchor"
    if entry["membership"] == "matched_control":
        return "matched_controls"
    return "cluster"


def a0_width(model: PolicyValueNet, x: torch.Tensor | None = None) -> int:
    """Read the loaded architecture and mechanically verify its A0 width."""
    if model.input_layer is None:
        raise RuntimeError("a0_input_layer_missing")
    width = int(model.input_layer.out_features)
    if x is not None and residual_v3_activations(model, x)["A0"].shape[-1] != width:
        raise RuntimeError("a0_architecture_shape_mismatch")
    if width != 96:
        raise RuntimeError(f"unexpected_r61_a0_width:{width}")
    return width


def patch_a0(
    recipient: PolicyValueNet, donor: PolicyValueNet, x: torch.Tensor, units: list[int]
) -> torch.Tensor:
    with torch.no_grad():
        left = residual_v3_activations(recipient, x)["A0"]
        right = residual_v3_activations(donor, x)["A0"]
        result = left.clone()
        result[:, units] = right[:, units]
        return result


def margin_logits(logits: torch.Tensor, entry: dict[str, Any]) -> torch.Tensor:
    optimal = entry["exact_outcome_optimal_actions"]
    legal = entry["legal_actions"]
    alternatives = [action for action in legal if action not in optimal]
    return logits[0, optimal].max() - logits[0, alternatives].max()


def metric_rows(
    recipient: PolicyValueNet,
    donor: PolicyValueNet | None,
    units: list[int],
    entries: list[dict[str, Any]],
    inputs: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows = []
    with torch.no_grad():
        for entry in entries:
            x = inputs[entry["id"]]
            activation = (
                residual_v3_activations(recipient, x)["A0"]
                if donor is None
                else patch_a0(recipient, donor, x, units)
            )
            metrics = legal_policy_metrics(
                continue_policy_from_stage(recipient, "A0", activation), entry
            )
            rows.append({"id": entry["id"], "cohort": frozen_cohort(entry), **metrics})
    return rows


def mean_metric(rows: list[dict[str, Any]], name: str, group: str) -> float:
    return statistics.fmean(float(row[name]) for row in rows if row["cohort"] == group)


def patch_summary(
    native: list[dict[str, Any]], patched: list[dict[str, Any]]
) -> dict[str, Any]:
    by_id = {row["id"]: row for row in native}
    deltas = {
        row["id"]: {
            "margin": float(row["margin"] - by_id[row["id"]]["margin"]),
            "optimal_mass": float(
                row["optimal_mass"] - by_id[row["id"]]["optimal_mass"]
            ),
            "top1_repair": int(
                not by_id[row["id"]]["top_is_outcome_optimal"]
                and row["top_is_outcome_optimal"]
            ),
            "top1_degradation": int(
                by_id[row["id"]]["top_is_outcome_optimal"]
                and not row["top_is_outcome_optimal"]
            ),
        }
        for row in patched
    }

    def average(key: str, group: str) -> float:
        return statistics.fmean(
            value[key]
            for ident, value in deltas.items()
            if by_id[ident]["cohort"] == group
        )

    return {
        "cluster_margin_delta": average("margin", "cluster"),
        "anchor_margin_delta": deltas[ANCHOR_ID]["margin"],
        "cluster_optimal_mass_delta": average("optimal_mass", "cluster"),
        "control_margin_delta": average("margin", "matched_controls"),
        "cluster_top1_repair_count": sum(
            value["top1_repair"]
            for ident, value in deltas.items()
            if by_id[ident]["cohort"] == "cluster"
        ),
        "control_degradation_count": sum(
            value["top1_degradation"]
            for ident, value in deltas.items()
            if by_id[ident]["cohort"] == "matched_controls"
        ),
        "state_margin_deltas": {
            ident: value["margin"] for ident, value in deltas.items()
        },
    }


def unit_trajectory(
    models: dict[str, dict[str, PolicyValueNet]],
    entries: list[dict[str, Any]],
    inputs: dict[str, torch.Tensor],
    width: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for landmark in LANDMARKS:
        values: dict[str, Any] = {
            "anchor": {},
            "cluster": {},
            "matched_controls": {},
        }
        for group in values:
            values[group] = {
                key: [[] for _ in range(width)]
                for key in ("z61", "z63", "a61", "a63", "g0")
            }
        for entry in entries:
            group, x = frozen_cohort(entry), inputs[entry["id"]]
            with torch.no_grad():
                z61 = models["T61"][landmark].input_layer(x)[0]
                z63 = models["T63"][landmark].input_layer(x)[0]
                base = residual_v3_activations(models["T61"]["g0"], x)["A0"][0]
            for key, tensor in (
                ("z61", z61),
                ("z63", z63),
                ("a61", torch.relu(z61)),
                ("a63", torch.relu(z63)),
                ("g0", base),
            ):
                for unit, value in enumerate(tensor.tolist()):
                    values[group][key][unit].append(float(value))
        rows = []
        for unit in range(width):
            c, control = values["cluster"], values["matched_controls"]
            cgap = float(
                np.mean(np.abs(np.asarray(c["a61"][unit]) - np.asarray(c["a63"][unit])))
            )
            ogap = float(
                np.mean(
                    np.abs(
                        np.asarray(control["a61"][unit])
                        - np.asarray(control["a63"][unit])
                    )
                )
            )
            rows.append(
                {
                    "unit": unit,
                    "cluster_abs_gap": cgap,
                    "control_abs_gap": ogap,
                    "cluster_specific_gap": cgap - ogap,
                    "signed_cluster_mean_difference": float(
                        np.mean(np.asarray(c["a63"][unit]) - np.asarray(c["a61"][unit]))
                    ),
                    "active_rate_difference": float(
                        np.mean(np.asarray(c["a63"][unit]) > 0)
                        - np.mean(np.asarray(c["a61"][unit]) > 0)
                    ),
                    "support_flip_rate": float(
                        np.mean(
                            (np.asarray(c["a61"][unit]) > 0)
                            != (np.asarray(c["a63"][unit]) > 0)
                        )
                    ),
                    "pre_relu": {
                        "t61_mean": float(np.mean(c["z61"][unit])),
                        "t63_mean": float(np.mean(c["z63"][unit])),
                    },
                    "activation": {
                        "t61_mean": float(np.mean(c["a61"][unit])),
                        "t63_mean": float(np.mean(c["a63"][unit])),
                        "g0_mean": float(np.mean(c["g0"][unit])),
                    },
                    "drift_from_g0": {
                        "t61": float(
                            np.mean(
                                np.asarray(c["a61"][unit]) - np.asarray(c["g0"][unit])
                            )
                        ),
                        "t63": float(
                            np.mean(
                                np.asarray(c["a63"][unit]) - np.asarray(c["g0"][unit])
                            )
                        ),
                    },
                    "gating": [
                        gating_classification(left, right)
                        for left, right in zip(
                            c["z61"][unit], c["z63"][unit], strict=True
                        )
                    ],
                }
            )
        output[landmark] = rows
    return output


def consistency(trajectory: dict[str, Any], width: int) -> list[dict[str, Any]]:
    result = []
    for unit in range(width):
        vectors = []
        signs = []
        for landmark in ("80", "81", "82", "108", "final"):
            value = trajectory[landmark][unit]["signed_cluster_mean_difference"]
            vectors.append(value)
            signs.append(np.sign(value))
        # Signed mean is the registered cross-landmark direction; pairwise state
        # correlations are retained in the detailed trajectory rather than invented.
        final_gap = trajectory["final"][unit]["cluster_abs_gap"]
        result.append(
            {
                "unit": unit,
                "sign_consistency": float(np.mean(np.asarray(signs[:-1]) == signs[-1])),
                "signed_difference_cosine_to_final": float(
                    np.dot(vectors[:-1], [vectors[-1]] * 4)
                    / max(np.linalg.norm(vectors[:-1]) * abs(vectors[-1]) * 2, EPS)
                ),
                "early_fraction": trajectory["82"][unit]["cluster_abs_gap"]
                / (final_gap + EPS),
            }
        )
    return result


def sensitivities(
    models: dict[str, dict[str, PolicyValueNet]],
    entries: list[dict[str, Any]],
    inputs: dict[str, torch.Tensor],
    width: int,
) -> dict[str, Any]:
    output = {}
    for landmark in LANDMARKS:
        rows = []
        for group in ("cluster", "matched_controls"):
            sums = np.zeros(width)
            count = 0
            for entry in entries:
                if frozen_cohort(entry) != group:
                    continue
                x = inputs[entry["id"]]
                activation = (
                    residual_v3_activations(models["T61"][landmark], x)["A0"]
                    .detach()
                    .requires_grad_(True)
                )
                margin_logits(
                    continue_policy_from_stage(
                        models["T61"][landmark], "A0", activation
                    ),
                    entry,
                ).backward()
                sums += np.abs(activation.grad.detach().numpy()[0])
                count += 1
            rows.append(sums / max(count, 1))
        output[landmark] = [
            {
                "unit": unit,
                "cluster_margin_sensitivity": float(rows[0][unit]),
                "control_margin_sensitivity": float(rows[1][unit]),
                "cluster_specific_sensitivity": float(rows[0][unit] - rows[1][unit]),
            }
            for unit in range(width)
        ]
    return output


def causal_score(forward: float, reverse: float, control: float) -> float:
    return min(max(0.0, forward), max(0.0, -reverse)) - max(0.0, abs(control) - 0.01)


def eligible(row: dict[str, Any]) -> bool:
    return bool(
        row["final"]["bidirectional_unit_support"]
        and row["final"]["causal_score"] > 0
        and row["108"]["same_direction"]
        and abs(row["final"]["forward"]["control_margin_delta"]) <= 0.02
        and row["82"]["forward"]["cluster_margin_delta"] >= 0
        and row["82"]["reverse"]["cluster_margin_delta"] <= 0
    )


def ranked_units(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (row for row in rows if row["eligible"]),
        key=lambda row: (
            -row["final"]["causal_score"],
            -row["108"]["causal_score"],
            -min(
                row["82"]["forward"]["cluster_margin_delta"],
                -row["82"]["reverse"]["cluster_margin_delta"],
            ),
            row["unit"],
        ),
    )


def prefix_sizes(count: int) -> list[int]:
    return [size for size in PREFIX_SIZES if size <= count]


def transfer_fraction(native: float, patched: float, donor_native: float) -> float:
    return (patched - native) / (donor_native - native + EPS)


def minimal_prefix(prefixes: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in prefixes:
        if (
            row["forward_transfer_fraction"] >= 0.60
            and row["reverse_transfer_fraction"] >= 0.60
            and row["forward_correct_direction_fraction"] >= 0.60
            and row["control_degradation_fraction"] < 0.20
            and row["step108_directional"]
        ):
            return row
    return None


def feature_groups() -> dict[str, list[str]]:
    return {
        "player_pits": list(BASE_FEATURE_ORDER[:6]),
        "opponent_pits": list(BASE_FEATURE_ORDER[6:12]),
        "stores": list(BASE_FEATURE_ORDER[12:14]),
        "player_to_move": [BASE_FEATURE_ORDER[14]],
        "v3_tactical": list(KALAH_V3_EXTRA_FEATURE_ORDER),
    }


def gating_classification(z61: float, z63: float) -> str:
    if z61 <= 0 and z63 <= 0:
        return "both_inactive"
    if (z61 > 0) != (z63 > 0):
        return "relu_threshold_crossing"
    return "both_active_magnitude_difference"


def selected_feature_geometry(
    models: dict[str, dict[str, PolicyValueNet]], selected: list[int]
) -> list[dict[str, Any]]:
    names = list(BASE_FEATURE_ORDER + KALAH_V3_EXTRA_FEATURE_ORDER)
    groups = feature_groups()
    result = []
    for unit in selected:
        snapshots = {}
        for landmark in ("g0", "pre80", "82", "108", "final"):
            left = models["T61"][landmark].input_layer
            right = models["T63"][landmark].input_layer
            assert left is not None and right is not None
            lw, rw = left.weight[unit].detach(), right.weight[unit].detach()
            base = models["T61"]["g0"].input_layer
            assert base is not None
            bw = base.weight[unit].detach()
            snapshots[landmark] = {
                "row_l2_difference": float(torch.linalg.vector_norm(lw - rw)),
                "t61_drift_from_g0": float(torch.linalg.vector_norm(lw - bw)),
                "t63_drift_from_g0": float(torch.linalg.vector_norm(rw - bw)),
                "t61_t63_drift_cosine": float(
                    torch.dot(lw - bw, rw - bw)
                    / max(
                        torch.linalg.vector_norm(lw - bw)
                        * torch.linalg.vector_norm(rw - bw),
                        torch.tensor(EPS),
                    )
                ),
                "bias_difference": float(right.bias[unit] - left.bias[unit]),
            }
        weights = {}
        for label, model in (
            ("g0", models["T61"]["g0"]),
            ("t61_final", models["T61"]["final"]),
            ("t63_final", models["T63"]["final"]),
        ):
            assert model.input_layer is not None
            vector = model.input_layer.weight[unit].detach().cpu().numpy()
            weights[label] = [
                {"feature": names[index], "weight": float(vector[index])}
                for index in np.argsort(np.abs(vector))[::-1][:8]
            ]
        final61, final63 = (
            models["T61"]["final"].input_layer,
            models["T63"]["final"].input_layer,
        )
        assert final61 is not None and final63 is not None
        vector = final63.weight[unit].detach() - final61.weight[unit].detach()
        total = float(torch.sum(vector.square())) + EPS
        result.append(
            {
                "unit": unit,
                "snapshots": snapshots,
                "top_absolute_weights": weights,
                "weight_deltas": [
                    {"feature": name, "t63_minus_t61": float(vector[index])}
                    for index, name in enumerate(names)
                ],
                "squared_weight_group_fraction": {
                    group: float(
                        sum(vector[names.index(name)].square() for name in features)
                        / total
                    )
                    for group, features in groups.items()
                },
            }
        )
    return result


def ordinary_replay_response(
    lanes: dict[str, Any],
    models: dict[str, dict[str, PolicyValueNet]],
    selected: list[int],
    trajectory: dict[str, Any],
) -> list[dict[str, Any]]:
    """Sample replay by fixed row index only, without forensic membership or labels."""
    indexes = np.linspace(
        0, len(lanes["T61"]["x"]) - 1, min(512, len(lanes["T61"]["x"])), dtype=int
    )
    x = torch.from_numpy(lanes["T61"]["x"][indexes])
    with torch.no_grad():
        a0 = {
            label: residual_v3_activations(model, x)["A0"].cpu().numpy()
            for label, model in (
                ("g0", models["T61"]["g0"]),
                ("t61", models["T61"]["final"]),
                ("t63", models["T63"]["final"]),
            )
        }
    return [
        {
            "unit": unit,
            "sample_size": len(indexes),
            "sampling": "fixed_evenly_spaced_replay_rows_no_forensic_labels",
            "g0": {
                "mean": float(a0["g0"][:, unit].mean()),
                "std": float(a0["g0"][:, unit].std()),
                "active_rate": float((a0["g0"][:, unit] > 0).mean()),
            },
            "t61": {
                "mean": float(a0["t61"][:, unit].mean()),
                "std": float(a0["t61"][:, unit].std()),
                "active_rate": float((a0["t61"][:, unit] > 0).mean()),
            },
            "t63": {
                "mean": float(a0["t63"][:, unit].mean()),
                "std": float(a0["t63"][:, unit].std()),
                "active_rate": float((a0["t63"][:, unit] > 0).mean()),
            },
            "drift_distribution": {
                "t61_minus_g0": np.quantile(
                    a0["t61"][:, unit] - a0["g0"][:, unit], [0.0, 0.5, 1.0]
                ).tolist(),
                "t63_minus_g0": np.quantile(
                    a0["t63"][:, unit] - a0["g0"][:, unit], [0.0, 0.5, 1.0]
                ).tolist(),
            },
            "specificity_ratio": trajectory["final"][unit]["cluster_abs_gap"]
            / (float(np.abs(a0["t61"][:, unit] - a0["t63"][:, unit]).mean()) + EPS),
        }
        for unit in selected
    ]


def snapshot_models(
    paths: dict[str, Any], lanes: dict[str, Any]
) -> dict[str, dict[str, PolicyValueNet]]:
    g0 = PolicyValueNet((96, 3), "residual_v3", lanes["T61"]["x"].shape[1])
    load_checkpoint_into_model(g0, paths["parent"])
    models = {}
    for seed, lane in lanes.items():
        states = {
            "g0": tensor_snapshot(g0),
            "pre80": lane["snapshots"][80]["before"],
            **{
                str(step): lane["snapshots"][step]["after"]
                for step in (80, 81, 82, 108)
            },
            "final": lane["final"],
        }
        models[seed] = {
            name: model_from_snapshot(state, lane["x"].shape[1]).eval()
            for name, state in states.items()
        }
    return models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-result", type=Path, required=True)
    parser.add_argument("--out-unit-matrix", type=Path, required=True)
    parser.add_argument("--out-prefixes", type=Path, required=True)
    parser.add_argument("--out-feature-geometry", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "inherited_classification": "early_divergence_local_decomposition_mixed",
        "g0_sha256": G0_SHA,
        "guardrails": {
            "training": False,
            "replay_mutation": False,
            "optimizer_mutation": False,
            "search_mutation": False,
            "frozen_cluster_training": False,
            "promotion": False,
        },
    }
    if not args.execute:
        write_json(args.out_result, result | {"classification": "planned"})
        return 0
    paths = artifact_paths()
    verify_r61_artifacts(paths)
    manifest = json.loads(
        (
            ROOT
            / "docs/data/alphazero-lite-internal-cluster-frozen-evaluation-set.json"
        ).read_text()
    )
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("set_sha256") != FROZEN_SET_SHA
        or manifest.get("training_injection") is not False
    ):
        raise RuntimeError("frozen_manifest_guard_failed")
    lanes = {
        seed: reconstruct_lane(paths, manifest, seed, {80, 81, 82, 108}, args.workdir)
        for seed in ("T61", "T63")
    }
    if not all(
        lanes[seed]["sha"] == FINAL_SHAS[seed] == EXPECTED_SHA[seed] for seed in lanes
    ):
        raise RuntimeError("historical_snapshot_reproduction_failed")
    models, inputs = snapshot_models(paths, lanes), state_inputs(manifest)
    width = a0_width(models["T61"]["final"], next(iter(inputs.values())))
    if a0_width(models["T63"]["final"], next(iter(inputs.values()))) != width:
        raise RuntimeError("a0_width_seed_mismatch")
    entries = manifest["entries"]
    trajectory, sensitivity = (
        unit_trajectory(models, entries, inputs, width),
        sensitivities(models, entries, inputs, width),
    )
    native = {
        landmark: {
            seed: metric_rows(models[seed][landmark], None, [], entries, inputs)
            for seed in ("T61", "T63")
        }
        for landmark in PATCH_LANDMARKS
    }
    # Mandatory endpoint invariants before interpreting any unit result.
    full = {}
    for landmark in PATCH_LANDMARKS:
        full[landmark] = {}
        for recipient, donor, direction in (
            ("T61", "T63", "forward"),
            ("T63", "T61", "reverse"),
        ):
            patched = metric_rows(
                models[recipient][landmark],
                models[donor][landmark],
                list(range(width)),
                entries,
                inputs,
            )
            full[landmark][direction] = patch_summary(
                native[landmark][recipient], patched
            )
    self_rows = metric_rows(
        models["T61"]["final"],
        models["T61"]["final"],
        list(range(width)),
        entries,
        inputs,
    )
    if any(
        abs(row["margin"] - base["margin"]) > TOLERANCE
        for row, base in zip(self_rows, native["final"]["T61"], strict=True)
    ):
        raise RuntimeError("a0_self_patch_identity_failed")
    if (
        full["final"]["forward"]["cluster_margin_delta"] == 0
        or full["final"]["reverse"]["cluster_margin_delta"] == 0
    ):
        raise RuntimeError("full_a0_patch_parity_failed")
    matrix = []
    for unit in range(width):
        row: dict[str, Any] = {"unit": unit}
        for landmark in PATCH_LANDMARKS:
            forward = patch_summary(
                native[landmark]["T61"],
                metric_rows(
                    models["T61"][landmark],
                    models["T63"][landmark],
                    [unit],
                    entries,
                    inputs,
                ),
            )
            reverse = patch_summary(
                native[landmark]["T63"],
                metric_rows(
                    models["T63"][landmark],
                    models["T61"][landmark],
                    [unit],
                    entries,
                    inputs,
                ),
            )
            row[landmark] = {
                "forward": forward,
                "reverse": reverse,
                "same_direction": forward["cluster_margin_delta"] >= 0
                and reverse["cluster_margin_delta"] <= 0,
                "bidirectional_unit_support": forward["cluster_margin_delta"] > 0
                and reverse["cluster_margin_delta"] < 0,
                "causal_score": causal_score(
                    forward["cluster_margin_delta"],
                    reverse["cluster_margin_delta"],
                    forward["control_margin_delta"],
                ),
            }
        row["eligible"] = eligible(row)
        matrix.append(row)
    ranking = ranked_units(matrix)
    prefixes = []
    for size in prefix_sizes(len(ranking)):
        units = [row["unit"] for row in ranking[:size]]
        forward = patch_summary(
            native["final"]["T61"],
            metric_rows(
                models["T61"]["final"], models["T63"]["final"], units, entries, inputs
            ),
        )
        reverse = patch_summary(
            native["final"]["T63"],
            metric_rows(
                models["T63"]["final"], models["T61"]["final"], units, entries, inputs
            ),
        )
        underperform = [
            row["id"]
            for row in native["final"]["T61"]
            if row["cohort"] == "cluster"
            and next(
                value for value in native["final"]["T63"] if value["id"] == row["id"]
            )["margin"]
            > row["margin"]
        ]
        prefixes.append(
            {
                "size": size,
                "units": units,
                "forward": forward,
                "reverse": reverse,
                "forward_transfer_fraction": transfer_fraction(
                    mean_metric(native["final"]["T61"], "margin", "cluster"),
                    mean_metric(native["final"]["T61"], "margin", "cluster")
                    + forward["cluster_margin_delta"],
                    mean_metric(native["final"]["T63"], "margin", "cluster"),
                ),
                "reverse_transfer_fraction": transfer_fraction(
                    mean_metric(native["final"]["T63"], "margin", "cluster"),
                    mean_metric(native["final"]["T63"], "margin", "cluster")
                    + reverse["cluster_margin_delta"],
                    mean_metric(native["final"]["T61"], "margin", "cluster"),
                ),
                "forward_correct_direction_fraction": float(
                    np.mean(
                        [
                            forward["state_margin_deltas"][ident] > 0
                            for ident in underperform
                        ]
                    )
                )
                if underperform
                else 0.0,
                "control_degradation_fraction": forward["control_degradation_count"]
                / max(
                    sum(frozen_cohort(row) == "matched_controls" for row in entries), 1
                ),
                "step108_directional": all(
                    matrix[unit]["108"]["same_direction"] for unit in units
                ),
            }
        )
    minimum = minimal_prefix(prefixes)
    selected = minimum["units"] if minimum else []
    loo = []
    if minimum:
        for unit in selected:
            remaining = [value for value in selected if value != unit]
            f = patch_summary(
                native["final"]["T61"],
                metric_rows(
                    models["T61"]["final"],
                    models["T63"]["final"],
                    remaining,
                    entries,
                    inputs,
                ),
            )
            r = patch_summary(
                native["final"]["T63"],
                metric_rows(
                    models["T63"]["final"],
                    models["T61"]["final"],
                    remaining,
                    entries,
                    inputs,
                ),
            )
            loo.append(
                {
                    "unit": unit,
                    "forward_loss": minimum["forward_transfer_fraction"]
                    - transfer_fraction(
                        mean_metric(native["final"]["T61"], "margin", "cluster"),
                        mean_metric(native["final"]["T61"], "margin", "cluster")
                        + f["cluster_margin_delta"],
                        mean_metric(native["final"]["T63"], "margin", "cluster"),
                    ),
                    "reverse_loss": minimum["reverse_transfer_fraction"]
                    - transfer_fraction(
                        mean_metric(native["final"]["T63"], "margin", "cluster"),
                        mean_metric(native["final"]["T63"], "margin", "cluster")
                        + r["cluster_margin_delta"],
                        mean_metric(native["final"]["T61"], "margin", "cluster"),
                    ),
                }
            )
    for row in loo:
        row["necessary_within_prefix"] = (
            row["forward_loss"] >= 0.10 or row["reverse_loss"] >= 0.10
        )
    classification, next_experiment = (
        (
            "distributed_a0_causal_subspace",
            "run ONE full-A0 G0 feature-retention ablation on ORDINARY replay states using a single pre-registered coefficient.",
        )
        if not minimum
        else (
            "compact_a0_causal_subspace_identified",
            "run ONE G0 feature-retention training ablation on ORDINARY replay states, regularizing ONLY this pre-registered A0 unit subset with one coefficient.",
        )
    )
    if not minimum and abs(full["final"]["forward"]["anchor_margin_delta"]) >= 0.10:
        classification, next_experiment = (
            "a0_unit_geometry_state_specific",
            "expand the frozen exact subcluster before modifying training.",
        )
    elif minimum and all(
        trajectory["final"][unit]["support_flip_rate"] > 0.5 for unit in selected
    ):
        classification, next_experiment = (
            "a0_gating_failure_primary",
            "audit preactivation-margin retention around zero for those units on ordinary replay before selecting a training regularizer.",
        )
    elif minimum:
        classification, next_experiment = (
            "a0_magnitude_drift_primary",
            "run subset A0 feature retention toward G0 on ordinary replay states.",
        )
    geometry = selected_feature_geometry(models, selected)
    ordinary = ordinary_replay_response(lanes, models, selected, trajectory)
    result |= {
        "historical_checkpoint_sha256": {seed: lanes[seed]["sha"] for seed in lanes},
        "a0_width": width,
        "snapshot_reproduction": True,
        "full_a0_patch": full,
        "trajectory": trajectory,
        "early_to_late_consistency": consistency(trajectory, width),
        "sensitivity": sensitivity,
        "ranking": ranking,
        "minimal_causal_prefix": minimum,
        "leave_one_out": loo,
        "ordinary_replay_response": ordinary,
        "classification": classification,
        "next_experiment": next_experiment,
    }
    write_json(args.out_result, result)
    write_json(args.out_unit_matrix, {"schema": SCHEMA, "single_unit_patches": matrix})
    write_json(
        args.out_prefixes,
        {
            "schema": SCHEMA,
            "prefixes": prefixes,
            "minimal_causal_prefix": minimum,
            "leave_one_out": loo,
        },
    )
    write_json(
        args.out_feature_geometry,
        {
            "schema": SCHEMA,
            "selected_units": selected,
            "feature_names": list(BASE_FEATURE_ORDER + KALAH_V3_EXTRA_FEATURE_ORDER),
            "feature_groups": feature_groups(),
            "units": geometry,
            "ordinary_replay_response": ordinary,
        },
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        "\n".join(
            [
                "# R61 A0 Unit Geometry Audit",
                "",
                "Inherited #322 classification: `early_divergence_local_decomposition_mixed`.",
                "",
                "## Snapshot And Full-A0 Parity",
                "",
                json.dumps(
                    {
                        "snapshots": result["historical_checkpoint_sha256"],
                        "full_a0_patch": full,
                    },
                    indent=2,
                ),
                "",
                "## Unitwise Trajectories And Sensitivity",
                "",
                "All 96 units at G0, pre-80, 80, 81, 82, 108, and final are in the result artifact; every single-unit forward/reverse patch is in the patch matrix.",
                "",
                "## Causal Ranking And Prefixes",
                "",
                json.dumps(
                    {
                        "ranking": ranking,
                        "prefixes": prefixes,
                        "minimal": minimum,
                        "leave_one_out": loo,
                    },
                    indent=2,
                ),
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Exactly one next experiment: {next_experiment}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
