#!/usr/bin/env python3
# ruff: noqa: E402
"""Test preregistered cross-seed policy-adapter strength residuals.

This runner never trains, takes optimizer steps, generates self-play, changes
search settings, or promotes a model.  It only constructs the preregistered
adapter substitutions, seals fresh suites D/E/F, and evaluates them with the
PR #249 paired ordinary-PUCT protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import build_opening_suite as suites  # noqa: E402
from ml.alphazero_lite import run_pr249_fresh_suite_generalization as pr249  # noqa: E402
from ml.alphazero_lite import run_pr250_cross_seed_adapter_gradient_audit as pr250  # noqa: E402
from ml.alphazero_lite.evaluation_metrics import (
    paired_effect_difference,
    paired_opening_candidate_effect,
)  # noqa: E402
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import sha256_file  # noqa: E402
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (  # noqa: E402
    ADAPTER_KEYS,
    export,
    new_model,
)
from ml.alphazero_lite.self_play import encode_state  # noqa: E402
from ml.alphazero_lite.train import legal_mask_matrix_for_encoded_states  # noqa: E402

SUITE_SEEDS = {"D": 4042, "E": 5042, "F": 6042}
CONSUMED_SUITES = {"A": 1042, "B": 2042, "C": 3042}
EXPECTED_CONSUMED_SUITE_SHA = {
    "A": "c8277e659c7a4e137140d83c187781f40e6b25c4b1dff5ec4da3f2e09fdcc6ab",
    "B": "1f4c17eb7df21af75bc29c3274b3951899ae5fb2522f762d5270d58ddf93b37e",
    "C": "b56783a2a2bbf63168cfb642f2b878badc80ace0279a9bbb7778757a4e4ba90d",
}
EXPECTED_FULL_MODEL_SHA = {
    "seed45_positive": "47177e32cb8f9aba51210f42bee78e65d603a61ac9cda7909ca1ab96f95d0d2f",
    "seed45_negative": "af860b80d33b28bd16b7259db8a265f320c209bc392285b2b83c92c0be123e17",
    "seed46_positive": "82e375a1a78f4d24d3002eafe031cc0a174f004ba4efd7ae5e1507578c954107",
    "seed46_negative": "39c97170f1e045b2d7506d5fa4ae8a8b5f4bc087f6033b952de683b50d89115c",
}
SOURCE = {
    "seed45_positive": pr249.CANDIDATES["seed45_fixed768_positive"],
    "seed45_negative": pr249.CANDIDATES["seed45_fixed1024_negative"],
    "seed46_positive": pr249.CANDIDATES["seed46_fresh1024_positive"],
    "seed46_negative": pr249.CANDIDATES["seed46_fresh768_negative"],
}


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def model_sha(state: dict[str, torch.Tensor]) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                key: value.detach().cpu().numpy().tolist()
                for key, value in state.items()
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load_candidates() -> dict[str, dict[str, torch.Tensor]]:
    states = {}
    for name, candidate in SOURCE.items():
        checkpoint = candidate["checkpoint"]
        if not checkpoint.is_file():
            fail(f"missing checkpoint: {name}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = {
            key: value.detach().cpu().clone() for key, value in saved["model"].items()
        }
        actual = model_sha(state)
        if actual != EXPECTED_FULL_MODEL_SHA[name]:
            fail(f"candidate hash mismatch: {name}={actual}")
        states[name] = state
    return states


def vector(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([state[key].reshape(-1) for key in ADAPTER_KEYS])


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm, right_norm = (
        torch.linalg.vector_norm(left),
        torch.linalg.vector_norm(right),
    )
    if left_norm == 0 or right_norm == 0:
        fail("zero residual or gradient vector")
    return float(torch.dot(left, right) / (left_norm * right_norm))


def residual_report(
    residuals: dict[str, dict[str, torch.Tensor]],
    states: dict[str, dict[str, torch.Tensor]],
    a16: dict[str, torch.Tensor],
) -> dict[str, Any]:
    r45, r46 = vector(residuals["r45"]), vector(residuals["r46"])
    positive_deltas = {
        "r45": vector(states["seed45_positive"]) - vector(a16),
        "r46": vector(states["seed46_positive"]) - vector(a16),
    }
    return {
        "adapter_key_order": list(ADAPTER_KEYS),
        "r45_norm": float(torch.linalg.vector_norm(r45)),
        "r46_norm": float(torch.linalg.vector_norm(r46)),
        "cosine": cosine(r45, r46),
        "dot_product": float(torch.dot(r45, r46)),
        "norm_ratio_r45_over_r46": float(
            torch.linalg.vector_norm(r45) / torch.linalg.vector_norm(r46)
        ),
        "per_tensor_norms": {
            name: {
                key: float(torch.linalg.vector_norm(value))
                for key, value in residual.items()
            }
            for name, residual in residuals.items()
        },
        "fraction_of_full_positive_delta_norm": {
            name: float(
                torch.linalg.vector_norm(vector(residuals[name]))
                / torch.linalg.vector_norm(positive_deltas[name])
            )
            for name in residuals
        },
    }


def make_state(
    recipient: dict[str, torch.Tensor], change: dict[str, torch.Tensor], sign: int
) -> dict[str, torch.Tensor]:
    state = {key: value.clone() for key, value in recipient.items()}
    for key in ADAPTER_KEYS:
        state[key] = state[key] + sign * change[key]
    for key in state:
        if key not in ADAPTER_KEYS and not torch.equal(state[key], recipient[key]):
            fail(f"non-adapter mutation: {key}")
    return state


def reconstruction_report(
    states: dict[str, dict[str, torch.Tensor]],
    residuals: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    report = {}
    for seed, positive, negative, residual in (
        ("seed45", "seed45_positive", "seed45_negative", "r45"),
        ("seed46", "seed46_positive", "seed46_negative", "r46"),
    ):
        reconstructed = make_state(states[negative], residuals[residual], 1)
        removed = make_state(states[positive], residuals[residual], -1)
        errors = [
            max(
                float((reconstructed[key] - states[positive][key]).abs().max())
                for key in ADAPTER_KEYS
            ),
            max(
                float((removed[key] - states[negative][key]).abs().max())
                for key in ADAPTER_KEYS
            ),
        ]
        # Float32 subtract/add is allowed one rounding unit; it must still be exact enough
        # to preserve the intended checkpoint arithmetic and all inherited tensors.
        if max(errors) > 1e-9:
            fail(f"same-seed reconstruction failure: {seed}, max_abs={max(errors)}")
        report[seed] = {
            "negative_plus_residual_max_abs_error": errors[0],
            "positive_minus_residual_max_abs_error": errors[1],
            "reconstructed_full_model_sha256": model_sha(reconstructed),
            "positive_full_model_sha256": model_sha(states[positive]),
            "removed_full_model_sha256": model_sha(removed),
            "negative_full_model_sha256": model_sha(states[negative]),
        }
    return report


def scrambled(residual: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {}
    for key in ADAPTER_KEYS:
        generator = torch.Generator(device="cpu").manual_seed(251)
        result[key] = (
            residual[key]
            .reshape(-1)[torch.randperm(residual[key].numel(), generator=generator)]
            .reshape_as(residual[key])
        )
        if not torch.equal(
            torch.sort(result[key].reshape(-1)).values,
            torch.sort(residual[key].reshape(-1)).values,
        ):
            fail(f"scramble distribution changed: {key}")
    return result


def intervention_states(
    states: dict[str, dict[str, torch.Tensor]],
    residuals: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    perm45, perm46 = scrambled(residuals["r45"]), scrambled(residuals["r46"])
    return {
        **states,
        "seed45_cross_rescue": make_state(
            states["seed45_negative"], residuals["r46"], 1
        ),
        "seed46_cross_rescue": make_state(
            states["seed46_negative"], residuals["r45"], 1
        ),
        "seed45_cross_remove": make_state(
            states["seed45_positive"], residuals["r46"], -1
        ),
        "seed46_cross_remove": make_state(
            states["seed46_positive"], residuals["r45"], -1
        ),
        "seed45_scrambled_control": make_state(states["seed45_negative"], perm46, 1),
        "seed46_scrambled_control": make_state(states["seed46_negative"], perm45, 1),
    }


def consumed_suite_paths(workdir: Path) -> dict[str, Path]:
    root = Path("/tmp/azlite_pr249_fresh_suite_generalization/suites")
    paths = {label: root / f"suite_{label}.jsonl" for label in CONSUMED_SUITES}
    for label, path in paths.items():
        if not path.is_file():
            fail(f"missing consumed suite {label}: {path}")
    return paths


def prefix_keys(entries: list[dict[str, Any]]) -> set[tuple[int, ...]]:
    return {
        tuple(prefix)
        for entry in entries
        for prefix in [entry["prefix_moves"], *entry.get("alternate_prefixes", [])]
    }


def seal_suites(workdir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    consumed = consumed_suite_paths(workdir)
    old_entries = {
        label: suites.load_suite_jsonl(str(path)) for label, path in consumed.items()
    }
    canonical = suites.load_suite_jsonl(str(pr249.CANONICAL_SUITE))
    used = set().union(
        pr249.suite_keys(canonical),
        *(pr249.suite_keys(rows) for rows in old_entries.values()),
    )
    training = set().union(*pr249.replay_states().values())
    universe = [
        entry
        for entry in pr249.all_openings()
        if tuple(encode_state(entry["state"], input_encoding="kalah_v3"))
        not in training
    ]
    paths, manifest = (
        {},
        {
            "canonical_sha256": sha256_file(pr249.CANONICAL_SUITE),
            "consumed_suites": {},
            "suites": {},
        },
    )
    for label, path in consumed.items():
        actual = sha256_file(path)
        if actual != EXPECTED_CONSUMED_SUITE_SHA[label]:
            fail(f"consumed suite SHA mismatch: {label}")
        manifest["consumed_suites"][label] = {
            "seed": CONSUMED_SUITES[label],
            "sha256": actual,
        }
    for label, seed in SUITE_SEEDS.items():
        available = [
            entry
            for entry in universe
            if suites.canonical_key(entry["state"]) not in used
        ]
        selected = suites.select_diverse(available, 128, seed)
        keys = pr249.suite_keys(selected)
        if len(keys) != 128 or keys & used:
            fail(f"cannot construct disjoint suite {label}")
        path = workdir / "suites" / f"suite_{label}.jsonl"
        suites.write_suite_jsonl(selected, str(path))
        paths[label] = path
        manifest["suites"][label] = {
            "preregistered_seed": seed,
            "sha256": sha256_file(path),
            "openings": 128,
            "consumed": True,
        }
        used |= keys
    return paths, manifest


def overlap_report(paths: dict[str, Path]) -> dict[str, Any]:
    old = {"canonical": pr249.CANONICAL_SUITE, **consumed_suite_paths(Path("."))}
    old_entries = {
        label: suites.load_suite_jsonl(str(path)) for label, path in old.items()
    }
    replays = pr249.replay_states()
    report, all_keys, all_prefixes = (
        {
            "within_suite_duplicates": {},
            "cross_suite_duplicates": 0,
            "old_suite_duplicates": {},
            "replay_state_duplicates": {},
            "prefix_overlaps": {},
        },
        set(),
        set(),
    )
    for label, path in paths.items():
        entries, keys = (
            suites.load_suite_jsonl(str(path)),
            pr249.suite_keys(suites.load_suite_jsonl(str(path))),
        )
        encoded = {
            tuple(encode_state(entry["state"], input_encoding="kalah_v3"))
            for entry in entries
        }
        prefixes = prefix_keys(entries)
        report["within_suite_duplicates"][label] = len(entries) - len(keys)
        report["cross_suite_duplicates"] += len(keys & all_keys)
        report["old_suite_duplicates"][label] = {
            name: len(keys & pr249.suite_keys(rows))
            for name, rows in old_entries.items()
        }
        report["replay_state_duplicates"][label] = {
            seed: len(encoded & values) for seed, values in replays.items()
        }
        report["prefix_overlaps"][label] = {
            "D_E_F": len(prefixes & all_prefixes),
            **{
                name: len(prefixes & prefix_keys(rows))
                for name, rows in old_entries.items()
            },
        }
        all_keys |= keys
        all_prefixes |= prefixes
    if (
        any(report["within_suite_duplicates"].values())
        or report["cross_suite_duplicates"]
        or any(any(item.values()) for item in report["old_suite_duplicates"].values())
        or any(
            any(item.values()) for item in report["replay_state_duplicates"].values()
        )
        or any(any(item.values()) for item in report["prefix_overlaps"].values())
    ):
        fail("suite overlap")
    report["passed"] = True
    return report


def diagnostic_states(paths: dict[str, Path]) -> np.ndarray:
    old_paths = {
        "canonical": pr249.CANONICAL_SUITE,
        **consumed_suite_paths(Path(".")),
    }
    forbidden = set().union(
        *(
            pr249.suite_keys(suites.load_suite_jsonl(str(path)))
            for path in paths.values()
        ),
        *(
            pr249.suite_keys(suites.load_suite_jsonl(str(path)))
            for path in old_paths.values()
        ),
    )
    candidates = [
        entry
        for entry in pr249.all_openings()
        if suites.canonical_key(entry["state"]) not in forbidden
    ]
    selected = suites.select_diverse(candidates, 128, 251)
    return np.asarray(
        [encode_state(entry["state"], input_encoding="kalah_v3") for entry in selected],
        dtype=np.float32,
    )


def policies_and_logits(
    state: dict[str, torch.Tensor], x: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    model = new_model(torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(x))
    raw = logits.numpy().astype(np.float64)
    legal = raw.copy()
    legal[~mask] = -1e9
    legal -= legal.max(axis=1, keepdims=True)
    policy = np.exp(legal)
    policy /= policy.sum(axis=1, keepdims=True)
    return policy, raw


def diagnostics(
    models: dict[str, dict[str, torch.Tensor]], paths: dict[str, Path]
) -> dict[str, Any]:
    x = diagnostic_states(paths)
    mask = legal_mask_matrix_for_encoded_states(x).astype(bool)
    pairs = {
        "seed45_cross_rescue": "seed45_negative",
        "seed46_cross_rescue": "seed46_negative",
        "seed45_cross_remove": "seed45_positive",
        "seed46_cross_remove": "seed46_positive",
        "seed45_scrambled_control": "seed45_negative",
        "seed46_scrambled_control": "seed46_negative",
        "seed45_cross_rescue_vs_positive": "seed45_positive",
        "seed46_cross_rescue_vs_positive": "seed46_positive",
    }
    result = {"states": len(x), "selector_seed": 251, "comparisons": {}}
    for candidate, recipient in pairs.items():
        left, left_logits = policies_and_logits(
            models[candidate.replace("_vs_positive", "")], x, mask
        )
        right, right_logits = policies_and_logits(models[recipient], x, mask)
        l1 = np.abs(left - right).sum(axis=1)
        midpoint = (left + right) / 2
        js = 0.5 * (
            (
                left * np.log(np.maximum(left, 1e-300) / np.maximum(midpoint, 1e-300))
            ).sum(axis=1)
            + (
                right * np.log(np.maximum(right, 1e-300) / np.maximum(midpoint, 1e-300))
            ).sum(axis=1)
        )
        adapter_change = vector(models[candidate.replace("_vs_positive", "")]) - vector(
            models[recipient]
        )
        result["comparisons"][candidate] = {
            "adapter_l2_change": float(torch.linalg.vector_norm(adapter_change)),
            "legal_policy_l1": {
                "mean": float(l1.mean()),
                "p50": float(np.percentile(l1, 50)),
                "p90": float(np.percentile(l1, 90)),
                "p99": float(np.percentile(l1, 99)),
            },
            "mean_js": float(js.mean()),
            "top1_disagreement": float(
                np.mean(np.argmax(left, axis=1) != np.argmax(right, axis=1))
            ),
            "max_logit_shift": float(
                np.abs(left_logits[mask] - right_logits[mask]).max()
            ),
        }
    return result


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def cached_records(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 512:
        return None
    return records


def evaluate(
    models: dict[str, dict[str, torch.Tensor]], paths: dict[str, Path], workdir: Path
) -> dict[str, Any]:
    artifacts = {
        name: export(state, workdir / "models" / name, name)
        for name, state in models.items()
    }
    result = {}
    for label, suite in paths.items():
        control_path = workdir / "records" / label / "p1_control.json"
        control = cached_records(control_path) or pr249.arena(
            workdir / "arena" / label, pr249.P1_ARTIFACT, suite, "p1_control"
        )
        write_records(control_path, control)
        result[label] = {"control_wdl": pr249.wdl(control), "candidates": {}}
        for name, artifact in artifacts.items():
            record_path = workdir / "records" / label / f"{name}.json"
            records = cached_records(record_path) or pr249.arena(
                workdir / "arena" / label, artifact, suite, name
            )
            write_records(record_path, records)
            result[label]["candidates"][name] = {
                "effect": paired_opening_candidate_effect(records, control),
                "win_draw_loss": pr249.wdl(records),
                "records": records,
            }
    return result


CONTRASTS = {
    "seed45_positive_minus_negative": ("seed45_positive", "seed45_negative"),
    "seed46_positive_minus_negative": ("seed46_positive", "seed46_negative"),
    "seed45_cross_rescue_minus_negative": ("seed45_cross_rescue", "seed45_negative"),
    "seed46_cross_rescue_minus_negative": ("seed46_cross_rescue", "seed46_negative"),
    "seed45_positive_minus_cross_remove": ("seed45_positive", "seed45_cross_remove"),
    "seed46_positive_minus_cross_remove": ("seed46_positive", "seed46_cross_remove"),
    "seed45_cross_rescue_minus_scrambled": (
        "seed45_cross_rescue",
        "seed45_scrambled_control",
    ),
    "seed46_cross_rescue_minus_scrambled": (
        "seed46_cross_rescue",
        "seed46_scrambled_control",
    ),
    "seed45_cross_rescue_minus_positive": ("seed45_cross_rescue", "seed45_positive"),
    "seed46_cross_rescue_minus_positive": ("seed46_cross_rescue", "seed46_positive"),
}


def bootstrap(values: np.ndarray, seed: int = 42) -> dict[str, float]:
    draws = values[
        np.random.default_rng(seed).integers(0, len(values), (10_000, len(values)))
    ].mean(axis=1)
    return {
        "effect": float(values.mean()),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "samples": 10_000,
    }


def analysis(evaluations: dict[str, Any]) -> dict[str, Any]:
    output = {"contrasts": {}}
    for name, (left, right) in CONTRASTS.items():
        suites_values, suite_effects = [], {}
        for label, item in evaluations.items():
            diff = paired_effect_difference(
                item["candidates"][left]["effect"], item["candidates"][right]["effect"]
            )
            values = np.asarray(list(diff["per_opening_effect"].values()), dtype=float)
            suites_values.append(values)
            suite_effects[label] = float(values.mean())
        pooled = np.concatenate(suites_values)
        rng = np.random.default_rng(42)
        draws = np.asarray(
            [
                rng.choice(suites_values[rng.integers(0, 3)], 128, replace=True).mean()
                for _ in range(10_000)
            ]
        )
        pooled_result = bootstrap(pooled)
        output["contrasts"][name] = {
            "per_suite": suite_effects,
            "pooled": {
                **pooled_result,
                "positive_suites": sum(value > 0 for value in suite_effects.values()),
            },
            "hierarchical_suite_opening_bootstrap": {
                "effect": float(pooled.mean()),
                "lower_95": float(np.quantile(draws, 0.025)),
                "upper_95": float(np.quantile(draws, 0.975)),
                "samples": 10_000,
            },
            "practical_behavioral_equivalence": abs(float(pooled.mean())) <= 0.02
            and pooled_result["lower_95"] <= 0 <= pooled_result["upper_95"]
            if name.endswith("minus_positive")
            else None,
        }
    return output


def divergence(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any]:
    left_by_key, right_by_key = (
        {pr249.record_key(row): row for row in rows} for rows in (left, right)
    )
    if set(left_by_key) != set(right_by_key):
        fail("record key mismatch")
    plies, seats, divergent, changed = [], Counter(), 0, 0
    for key, row in left_by_key.items():
        other = right_by_key[key]
        moves, other_moves = (
            row["trajectory"].split(","),
            other["trajectory"].split(","),
        )
        first = next(
            (
                index
                for index, pair in enumerate(zip(moves, other_moves))
                if pair[0] != pair[1]
            ),
            None,
        )
        if first is not None or len(moves) != len(other_moves):
            divergent += 1
            plies.append(
                first if first is not None else min(len(moves), len(other_moves))
            )
            seats[str(row["challenger_player"])] += 1
        changed += row["winner"] != other["winner"]
    return {
        "games": len(left_by_key),
        "fraction_with_move_divergence": divergent / len(left_by_key),
        "first_divergent_ply_distribution": dict(sorted(Counter(plies).items())),
        "divergent_challenger_seat": dict(sorted(seats.items())),
        "final_outcome_changed_fraction": changed / len(left_by_key),
    }


def classify(report: dict[str, Any]) -> str:
    items = report["contrasts"]

    def passed(name: str) -> bool:
        return (
            items[name]["pooled"]["lower_95"] > 0
            and items[name]["pooled"]["positive_suites"] >= 2
        )

    controls = passed("seed45_positive_minus_negative") and passed(
        "seed46_positive_minus_negative"
    )
    rescues = [
        passed("seed45_cross_rescue_minus_negative"),
        passed("seed46_cross_rescue_minus_negative"),
    ]
    removes = [
        passed("seed45_positive_minus_cross_remove"),
        passed("seed46_positive_minus_cross_remove"),
    ]
    directional = all(
        items[name]["pooled"]["lower_95"] > 0
        for name in (
            "seed45_cross_rescue_minus_scrambled",
            "seed46_cross_rescue_minus_scrambled",
        )
    )
    if not controls:
        return "fresh_strength_signal_fails_again"
    if all(rescues) and all(removes) and directional:
        return "cross_seed_strength_residual_transfers"
    if all(rescues) and not directional:
        return "residual_magnitude_not_direction"
    if all(rescues) and not all(removes):
        return "residual_sufficient_not_necessary"
    if sum(rescues) == 1 and sum(removes) == 1:
        return "one_seed_residual_transfers"
    return "strength_residual_not_transferable"


def gradient_audit(states: dict[str, dict[str, torch.Tensor]]) -> dict[str, float]:
    snapshot = torch.load(pr250.A16, map_location="cpu", weights_only=False)
    a16, _ = pr250.replay.immutable_initial_state(snapshot)
    parent_model = new_model(torch.device("cpu"))
    pr250.load_checkpoint_into_model(parent_model, pr250.replay.P1_CHECKPOINT)
    parent = {
        key: value.detach().cpu().clone()
        for key, value in parent_model.state_dict().items()
    }
    gradients = {}
    for name, lane in pr250.LANES.items():
        if sha256_file(lane["replay"]) != lane["replay_sha"]:
            fail(f"replay SHA mismatch: {name}")
        rows, _ = pr250.load_rows(lane["replay"])
        gradients[name] = pr250.full_batch_gradient(rows, a16, parent)["vector"]
    r45, r46 = (
        vector(states["seed45_positive"]) - vector(states["seed45_negative"]),
        vector(states["seed46_positive"]) - vector(states["seed46_negative"]),
    )
    dg45 = (
        gradients["seed45_fixed768_positive"] - gradients["seed45_fixed1024_negative"]
    )
    dg46 = (
        gradients["seed46_fresh1024_positive"] - gradients["seed46_fresh768_negative"]
    )
    return {
        "cosine_r45_negative_dg45": cosine(r45, -dg45),
        "cosine_r46_negative_dg46": cosine(r46, -dg46),
        "cosine_r45_negative_dg46": cosine(r45, -dg46),
        "cosine_r46_negative_dg45": cosine(r46, -dg45),
        "cosine_dg45_dg46": cosine(dg45, dg46),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_pr251_cross_seed_strength_residual_transfer"),
    )
    parser.add_argument("--freeze-suites-only", action="store_true")
    args = parser.parse_args()
    if sha256_file(pr249.CANONICAL_SUITE) != pr249.CANONICAL_SHA:
        fail("canonical suite SHA")
    states = load_candidates()
    residuals = {
        "r45": {
            key: states["seed45_positive"][key] - states["seed45_negative"][key]
            for key in ADAPTER_KEYS
        },
        "r46": {
            key: states["seed46_positive"][key] - states["seed46_negative"][key]
            for key in ADAPTER_KEYS
        },
    }
    snapshot = torch.load(pr250.A16, map_location="cpu", weights_only=False)
    a16, _ = pr250.replay.immutable_initial_state(snapshot)
    args.workdir.mkdir(parents=True, exist_ok=True)
    paths, suite_manifest = seal_suites(args.workdir)
    overlap = overlap_report(paths)
    frozen = {
        "candidate_hashes": {name: model_sha(state) for name, state in states.items()},
        "residual_geometry": residual_report(residuals, states, a16),
        "same_seed_reconstruction": reconstruction_report(states, residuals),
        "suite_manifest": suite_manifest,
        "suite_overlap_report": overlap,
        "guardrails": {
            "residual_scaling": False,
            "permutation_seed": 251,
            "training": False,
            "optimizer_steps": False,
            "self_play": False,
            "primary_suites": list(SUITE_SEEDS),
            "context": "1200:1200",
            "c_puct": 1.25,
            "arena_seed": 42,
        },
    }
    (args.workdir / "frozen_manifest.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.freeze_suites_only:
        return
    models = intervention_states(states, residuals)
    diagnostic = diagnostics(models, paths)
    evaluations = evaluate(models, paths, args.workdir)
    report = analysis(evaluations)
    telemetry_pairs = {
        name: pair
        for name, pair in CONTRASTS.items()
        if "cross_rescue_minus_negative" in name
        or "positive_minus_cross_remove" in name
        or "cross_rescue_minus_scrambled" in name
    }
    summary = {
        **frozen,
        "parameter_output_diagnostics": diagnostic,
        "evaluation_context": {
            "budget": "1200:1200",
            "ordinary_puct": True,
            "c_puct": 1.25,
            "seat_swapping": True,
            "arena_seed": 42,
            "matched_p1_control_per_suite": True,
        },
        "analysis": report,
        "first_divergence_telemetry": {
            label: {
                name: divergence(
                    data["candidates"][pair[0]]["records"],
                    data["candidates"][pair[1]]["records"],
                )
                for name, pair in telemetry_pairs.items()
            }
            for label, data in evaluations.items()
        },
        "residual_vs_gradient_difference_audit": gradient_audit(states),
    }
    summary["classification"] = classify(report)
    (args.workdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary["classification"])


if __name__ == "__main__":
    main()
