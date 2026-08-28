import random
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.alphazero_lite import run_pr247_fixed_target_budget as fixed_budget
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import Evaluator, standard_start_state


class UniformEvaluator(Evaluator):
    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        priors = np.zeros(6, dtype=np.float32)
        legal = game.possible_moves()
        priors[legal] = 1.0 / len(legal)
        return priors, 0.0


def _row() -> dict:
    return {
        "teacher_search_profile": {
            "c_puct": 1.25,
            "search_options": {
                "fpu_mode": "zero",
                "normalize_values": False,
                "root_policy_mode": "visit_count",
                "tactical_root_bias": 0.0,
                "root_temperature": 0.0,
            },
        },
        "action_sampling_noise_enabled": True,
        "dirichlet_alpha": 0.3,
        "target_dirichlet_epsilon": 0.3,
        "legal_moves": list(range(6)),
        "move_index": 0,
        "policy_target_mode": "default",
    }


def test_all_fixed_target_budgets_receive_identical_noisy_priors(monkeypatch):
    monkeypatch.setattr(fixed_budget, "_EVALUATOR", UniformEvaluator())
    state = random.Random(45).getstate()
    priors = []
    for budget in (384, 768, 1024, 1280, 1536, 777):
        rng = random.Random()
        rng.setstate(state)
        _policy, _root, prior = fixed_budget.search(
            KalahGame.from_state(standard_start_state()),
            _row(),
            rng,
            root=None,
            reuse=False,
            simulations=budget,
        )
        priors.append(prior)
    assert all(np.array_equal(priors[0], prior) for prior in priors[1:])


def test_fixed_target_lane_contract_excludes_reused_training_lane():
    assert fixed_budget.LANES == (
        "fresh384",
        "fixed768",
        "fixed1024",
        "fixed1280",
        "fixed1536",
        "fresh_equiv",
    )
    assert "reused" not in fixed_budget.LANES
    assert fixed_budget.FIXED_BUDGETS["fixed1024"] == 1024


def test_materialize_views_writes_source_order_and_gates_fresh_controls(
    monkeypatch, tmp_path: Path
):
    source = [
        {"game_index": 1, "move_index": 0, "state": [1], "policy": [0] * 6},
        {"game_index": 0, "move_index": 0, "state": [2], "policy": [0] * 6},
    ]
    reconstructed = {
        (1, 0): {
            "policies": {
                lane: [1, 0, 0, 0, 0, 0] for lane in fixed_budget.LANES + ("reused",)
            }
        },
        (0, 0): {
            "policies": {
                lane: [0, 1, 0, 0, 0, 0] for lane in fixed_budget.LANES + ("reused",)
            }
        },
    }
    monkeypatch.setattr(
        fixed_budget,
        "sha256_file",
        lambda path: (
            fixed_budget.FRESH384_SHA
            if path.name == "fresh384.jsonl"
            else fixed_budget.FRESH_EQUIV_SHA
            if path.name == "fresh_equiv.jsonl"
            else "derived"
        ),
    )

    views, hashes = fixed_budget.materialize_views(source, reconstructed, tmp_path)

    assert list(views) == [*fixed_budget.LANES, "reused"]
    assert [row["state"] for row in views["fresh384"]] == [[1], [2]]
    assert hashes["fresh384"] == fixed_budget.FRESH384_SHA
    assert (tmp_path / "derived" / "fixed1024.jsonl").is_file()


def test_resume_derived_views_and_telemetry_are_validated(monkeypatch, tmp_path: Path):
    source = [
        {"game_index": 0, "move_index": 0, "state": [1], "policy": [0] * 6},
        {"game_index": 0, "move_index": 1, "state": [2], "policy": [0] * 6},
    ]
    views = {
        lane: [{**row, "policy": [index == 0 for index in range(6)]} for row in source]
        for lane in fixed_budget.LANES + ("reused",)
    }
    for lane, rows in views.items():
        fixed_budget.write_jsonl(tmp_path / "derived" / f"{lane}.jsonl", rows)
    monkeypatch.setattr(
        fixed_budget,
        "sha256_file",
        lambda path: (
            fixed_budget.FRESH384_SHA
            if path.name == "fresh384.jsonl"
            else fixed_budget.FRESH_EQUIV_SHA
            if path.name == "fresh_equiv.jsonl"
            else "derived"
        ),
    )
    loaded, _hashes = fixed_budget.load_derived_views(source, tmp_path)
    telemetry = {
        (0, index): {
            "game_index": 0,
            "move_index": index,
            "inherited_child_visit_mass": index,
            "fresh_equiv_budget": fixed_budget.BASE + index,
            "policies": {lane: rows[index]["policy"] for lane, rows in loaded.items()},
        }
        for index in range(len(source))
    }
    fixed_budget.write_telemetry(tmp_path / "telemetry.jsonl", source, telemetry)

    assert fixed_budget.load_telemetry(source, loaded, tmp_path) == telemetry


def test_resume_uses_hash_gated_frozen_telemetry_in_source_order(
    monkeypatch, tmp_path: Path
):
    source = [
        {"game_index": 1, "move_index": 0, "state": [1], "policy": [0] * 6},
        {"game_index": 0, "move_index": 2, "state": [2], "policy": [0] * 6},
    ]
    views = {
        lane: [
            {**row, "policy": [index == row_index for index in range(6)]}
            for row_index, row in enumerate(source)
        ]
        for lane in fixed_budget.LANES + ("reused",)
    }
    frozen = tmp_path / "frozen-telemetry.jsonl"
    records = [
        {
            "game_index": row["game_index"],
            "move_index": row["move_index"],
            "inherited_child_visit_mass": index + 3,
            "fresh_equiv_budget": fixed_budget.BASE + index + 3,
        }
        for index, row in enumerate(source)
    ]
    fixed_budget.write_jsonl(frozen, records)
    monkeypatch.setattr(fixed_budget, "FROZEN_TELEMETRY", frozen)
    monkeypatch.setattr(
        fixed_budget,
        "sha256_file",
        lambda path: fixed_budget.FROZEN_TELEMETRY_SHA if path == frozen else "derived",
    )

    loaded = fixed_budget.load_telemetry(source, views, tmp_path)

    assert [loaded[key]["inherited_child_visit_mass"] for key in loaded] == [3, 4]
    assert [loaded[key]["fresh_equiv_budget"] for key in loaded] == [387, 388]
    assert loaded[(1, 0)]["policies"] == {
        lane: views[lane][0]["policy"] for lane in views
    }
    fixed_budget.write_jsonl(frozen, list(reversed(records)))
    with pytest.raises(RuntimeError, match="telemetry ordering mismatch row=0"):
        fixed_budget.load_telemetry(source, views, tmp_path)
    monkeypatch.setattr(fixed_budget, "sha256_file", lambda _path: "wrong")
    with pytest.raises(RuntimeError, match="frozen PR #247 telemetry is unavailable"):
        fixed_budget.load_telemetry(source, views, tmp_path)


def test_resume_derived_rejects_changed_metadata(monkeypatch, tmp_path: Path):
    source = [{"game_index": 0, "move_index": 0, "state": [1], "policy": [0] * 6}]
    for lane in fixed_budget.LANES + ("reused",):
        fixed_budget.write_jsonl(
            tmp_path / "derived" / f"{lane}.jsonl",
            [{**source[0], "state": [2], "policy": [1] + [0] * 5}],
        )
    monkeypatch.setattr(fixed_budget, "sha256_file", lambda _path: "derived")

    with pytest.raises(RuntimeError, match="metadata or ordering mismatch"):
        fixed_budget.load_derived_views(source, tmp_path)


def test_cached_lane_requires_every_checkpoint_and_export(tmp_path: Path):
    lane = "fixed1024"
    (tmp_path / "train" / lane).mkdir(parents=True)
    for step in fixed_budget.replay_train.STEPS:
        torch.save(
            {"model": {"weight": torch.tensor([step])}, "optimizer": {}},
            tmp_path / "train" / lane / f"step_{step:04d}.pt",
        )

    assert fixed_budget.load_cached_lane(tmp_path, lane) is None

    export = tmp_path / "train" / lane / "step_0016" / "checkpoint.npz"
    export.parent.mkdir(parents=True, exist_ok=True)
    export.touch()
    snapshots, optimizers = fixed_budget.load_cached_lane(tmp_path, lane) or ({}, {})

    assert snapshots[16]["weight"].item() == 16
    assert optimizers == {1: {}, 4: {}, 16: {}}


def _arena(effect_by_lane: dict[str, float], fresh_ci_lower: float = 0.01) -> dict:
    def contrast(lower: float, upper: float) -> dict:
        return {"opening_bootstrap_ci": {"lower_95": lower, "upper_95": upper}}

    arena = {
        "fresh384": {"1200:1200": {"effect": 0.0}},
        "fresh_equiv": {"1200:1200": {"effect": 0.04}},
        "paired_contrasts": {},
    }
    for lane in fixed_budget.FIXED_BUDGETS:
        arena[lane] = {"1200:1200": {"effect": effect_by_lane[lane]}}
        arena["paired_contrasts"][f"{lane}_minus_fresh384"] = {
            "1200:1200": contrast(fresh_ci_lower, 0.03)
        }
        arena["paired_contrasts"][f"{lane}_minus_fresh_equiv"] = {
            "1200:1200": contrast(-0.01, 0.01)
        }
    arena["paired_contrasts"]["fresh_equiv_minus_fresh384"] = {
        "1200:1200": contrast(0.01, 0.05)
    }
    return arena


def _training() -> dict:
    return {
        lane: {"metrics": {"16": {"fit_fraction": 0.25}}} for lane in fixed_budget.LANES
    }


def test_strict_selection_chooses_smallest_passing_budget():
    arena = _arena(
        {"fixed768": 0.04, "fixed1024": 0.04, "fixed1280": 0.04, "fixed1536": 0.04}
    )
    geometry = {"monotonic_mean_js_vs_fresh_equiv_decreasing": True}

    classification, passes, curve = fixed_budget.classify(
        True, geometry, _training(), arena
    )

    assert classification == "fixed_768_sufficient"
    assert passes["fixed768"]["passes"]
    assert curve["fixed1024"]["marginal_gain_per_plus_256"] == 0.0


def test_selection_requires_positive_fresh_contrast_and_zero_equiv_contrast():
    arena = _arena(
        {"fixed768": 0.04, "fixed1024": 0.04, "fixed1280": 0.04, "fixed1536": 0.04},
        fresh_ci_lower=0.0,
    )
    geometry = {"monotonic_mean_js_vs_fresh_equiv_decreasing": True}

    classification, passes, _curve = fixed_budget.classify(
        True, geometry, _training(), arena
    )

    assert not any(row["passes"] for row in passes.values())
    assert classification == "adaptive_budget_materially_better"
