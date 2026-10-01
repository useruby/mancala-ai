"""Run and analyze PR #384's retrospective fixed-E4 checkpoint diagnostic."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
SPEC = DATA / "seed461-lr-fixed-e4-diagnostic-spec.json"
REG = DATA / "seed461-lr-sensitivity-registration-v2.json"
AMEND = DATA / "seed461-lr-sensitivity-training-amendment.json"
BIND = DATA / "seed461-lr-sensitivity-evaluation-binding.json"
SUITE = DATA / "seed461-lr-sensitivity-openings-v2.jsonl"
WORK = ROOT / ".tmp/seed461-lr-sensitivity"
OPPONENT = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
ORDERS = range(38411, 38416)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def e4_checkpoint_binding(amendment: dict[str, Any], run: str) -> tuple[Path, str]:
    """Return the registered E4 file/hash, independently of selected epoch."""
    checkpoint_hash = amendment["checkpoints"][run]["epochs"]["E4"]
    return WORK / "training" / run / "E4.npz", checkpoint_hash


def verify_cached_file(path: Path, expected_sha256: str, identity: str) -> None:
    if sha(path) != expected_sha256:
        raise ValueError(f"cached_evidence_identity_mismatch:{identity}")


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify_spec(spec: dict[str, Any]) -> None:
    for row in spec["provenance"].values():
        if "path" in row and sha(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"frozen_provenance_hash_mismatch:{row['path']}")
    opponent = spec["provenance"]["opponent"]
    for filename, key in (
        ("weights.json", "weights_sha256"),
        ("metadata.json", "metadata_sha256"),
        ("search_policy.json", "search_policy_sha256"),
    ):
        if sha(ROOT / opponent["artifact"] / filename) != opponent[key]:
            raise ValueError(f"frozen_opponent_hash_mismatch:{filename}")


def validate_games(path: Path, openings: list[dict[str, Any]]) -> np.ndarray:
    rows = jsonl(path)
    if len(rows) != 512:
        raise ValueError(f"game_count_mismatch:{path.name}")
    by_opening: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        index = int(row["opening_index"])
        if (
            not 0 <= index < 256
            or row.get("opening_prefix_moves") != openings[index]["prefix_moves"]
        ):
            raise ValueError(f"suite_identity_mismatch:{path.name}:{index}")
        by_opening.setdefault(index, []).append(row)
    scores = np.empty(256, dtype=np.float64)
    for index, pair in by_opening.items():
        if len(pair) != 2 or sorted(int(r["challenger_player"]) for r in pair) != [
            0,
            1,
        ]:
            raise ValueError(f"seat_pairing_mismatch:{path.name}:{index}")
        scores[index] = (
            sum(
                1.0
                if r["winner"] == "challenger"
                else 0.5
                if r["winner"] == "draw"
                else 0.0
                for r in pair
            )
            / 2
        )
    if set(by_opening) != set(range(256)):
        raise ValueError(f"opening_coverage_mismatch:{path.name}")
    return scores


def verify_report_identity(
    report_path: Path,
    candidate_path: str,
    opponent_path: str,
    suite_hash: str,
    runtime_contract: dict[str, Any],
) -> None:
    report = json.loads(report_path.read_text())
    notes = report.get("notes", {})
    expected = {
        "challenger_path": candidate_path,
        "current_path": opponent_path,
        "suite_sha256": suite_hash,
        "challenger_simulations": 384,
        "current_simulations": 384,
        "seed": 384,
    }
    if report.get("games_played") != 512:
        raise ValueError(f"report_game_count_mismatch:{report_path.name}")
    for key, value in expected.items():
        if notes.get(key) != value:
            raise ValueError(f"report_identity_mismatch:{report_path.name}:{key}")
    profile = notes.get("search_profile", {})
    if profile.get("c_puct") != 1.25 or profile.get("simulations") != 384:
        raise ValueError(f"report_search_contract_mismatch:{report_path.name}")
    if notes.get("search_profile_hash") != profile.get("hash"):
        raise ValueError(f"report_profile_hash_mismatch:{report_path.name}")
    for key, value in runtime_contract.items():
        if notes.get(key) != value:
            raise ValueError(
                f"report_runtime_identity_mismatch:{report_path.name}:{key}"
            )


def ensure_e4_evaluations(
    spec: dict[str, Any], amendment: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    runtime = spec["provenance"]["runtime_contract"]
    runtime_contract = json.loads(REG.read_text())["evaluation"]["runtime_contract"]
    if (
        runtime_contract["runtime_search_policy_sha256"]
        != runtime["search_policy_sha256"]
    ):
        raise ValueError("registered_runtime_identity_mismatch")
    from ml.alphazero_lite.runtime_search_policy import (
        resolve_strength_comparison_runtime_contract,
    )

    resolved_runtime = resolve_strength_comparison_runtime_contract(
        current_artifact=OPPONENT,
        challenger_artifact=ROOT / ".tmp/seed461-order-confirmation/artifacts/T1-E4",
    )
    if resolved_runtime != runtime_contract:
        raise ValueError("runtime_contract_mismatch")
    e4_binding: dict[str, Any] = {
        "schema": "seed461-lr-fixed-e4-binding-v1",
        "spec_sha256": sha(SPEC),
        "runs": {},
    }
    for seed in ORDERS:
        for arm in ("A", "B"):
            run = f"order_{seed}_{arm}"
            checkpoint, checkpoint_hash = e4_checkpoint_binding(amendment, run)
            verify_cached_file(checkpoint, checkpoint_hash, f"checkpoint:{run}:E4")
            if sha(checkpoint) != checkpoint_hash:
                raise ValueError(f"e4_checkpoint_mismatch:{run}")
            source_binding = binding["candidates"][run]
            artifact = (
                Path(source_binding["artifact"])
                if source_binding["selected_epoch"] == "E4"
                else WORK / "fixed-e4-artifacts" / run
            )
            report = WORK / "fixed-e4-arena" / f"{run}.json"
            games = WORK / "fixed-e4-arena" / f"{run}-games.jsonl"
            if not artifact.exists():
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "ml/alphazero_lite/export_artifact.py"),
                        "--checkpoint",
                        str(checkpoint),
                        "--out-dir",
                        str(artifact),
                        "--version",
                        f"seed461-fixed-e4-{run}",
                        "--model-type",
                        "residual_v3",
                        "--rules-version",
                        "kalah_v1",
                        "--input-encoding",
                        "kalah_v3",
                    ],
                    cwd=ROOT,
                    check=True,
                )
                (artifact / "search_policy.json").write_bytes(
                    (ROOT / "model-artifact/current/search_policy.json").read_bytes()
                )
            candidate_identity = {
                name: sha(artifact / name)
                for name in (
                    "model.npz",
                    "weights.json",
                    "metadata.json",
                    "search_policy.json",
                )
            }
            if (
                candidate_identity["search_policy.json"]
                != runtime["search_policy_sha256"]
            ):
                raise ValueError(f"candidate_runtime_policy_mismatch:{run}")
            e4_binding["runs"][run] = {
                "epoch": "E4",
                "checkpoint_sha256": checkpoint_hash,
                "artifact": str(artifact),
                "candidate_artifacts": candidate_identity,
                "report": str(report),
                "games": str(games),
                "reused_selected_record": source_binding["selected_epoch"] == "E4",
            }
            if source_binding["selected_epoch"] == "E4":
                # Copy existing validated evaluation evidence to the dedicated E4 evidence path.
                old_report = Path(binding["reports"][run]["report"])
                old_games = Path(binding["reports"][run]["games"])
                verify_cached_file(
                    old_report,
                    binding["reports"][run]["report_sha256"],
                    f"report:{run}",
                )
                verify_cached_file(
                    old_games, binding["reports"][run]["games_sha256"], f"games:{run}"
                )
                verify_report_identity(
                    old_report,
                    source_binding["artifact"],
                    binding["opponent"]["artifact"],
                    sha(SUITE),
                    runtime_contract,
                )
                for filename, key in (
                    ("model.npz", "model_sha256"),
                    ("weights.json", "weights_sha256"),
                    ("metadata.json", "metadata_sha256"),
                    ("search_policy.json", "search_policy_sha256"),
                ):
                    if (
                        sha(Path(source_binding["artifact"]) / filename)
                        != source_binding[key]
                    ):
                        raise ValueError(
                            f"cached_candidate_artifact_mismatch:{run}:{filename}"
                        )
                if source_binding["model_sha256"] != checkpoint_hash:
                    raise ValueError(f"cached_e4_checkpoint_artifact_mismatch:{run}")
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_bytes(old_report.read_bytes())
                games.write_bytes(old_games.read_bytes())
            elif not games.exists() or not report.exists():
                report.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "ml/alphazero_lite/arena.py"),
                        "--challenger",
                        str(artifact),
                        "--current",
                        str(OPPONENT),
                        "--games",
                        "512",
                        "--games-per-opening",
                        "2",
                        "--opening-prefixes-jsonl",
                        str(SUITE),
                        "--suite-sha256",
                        sha(SUITE),
                        "--challenger-simulations",
                        "384",
                        "--current-simulations",
                        "384",
                        "--seed",
                        "384",
                        "--workers",
                        "24",
                        "--c-puct",
                        "1.25",
                        "--seed-contract",
                        "azlite_eval_seed_v2",
                        "--game-jsonl",
                        str(games),
                        "--out",
                        str(report),
                    ],
                    cwd=ROOT,
                    check=True,
                )
            evaluated_path = (
                source_binding["artifact"]
                if source_binding["selected_epoch"] == "E4"
                else str(artifact)
            )
            verify_report_identity(
                report,
                evaluated_path,
                binding["opponent"]["artifact"],
                sha(SUITE),
                runtime_contract,
            )
            e4_binding["runs"][run]["report_sha256"] = sha(report)
            e4_binding["runs"][run]["games_sha256"] = sha(games)
    return e4_binding


def analyze(
    spec: dict[str, Any],
    amendment: dict[str, Any],
    original_binding: dict[str, Any],
    e4_binding: dict[str, Any],
) -> dict[str, Any]:
    openings = jsonl(SUITE)
    if len(openings) != 256:
        raise ValueError("suite_count_mismatch")
    selected: dict[str, np.ndarray] = {}
    fixed: dict[str, np.ndarray] = {}
    for run, info in e4_binding["runs"].items():
        fixed[run] = validate_games(Path(info["games"]), openings)
        # The original evaluation binding is the immutable selected-checkpoint arm.
        old_path = Path(original_binding["reports"][run]["games"])
        old_report_path = Path(original_binding["reports"][run]["report"])
        original_candidate = original_binding["candidates"][run]
        selected_epoch = original_candidate["selected_epoch"]
        checkpoint_record = amendment["checkpoints"][run]
        selected_hash = checkpoint_record["epochs"][selected_epoch]
        if (
            checkpoint_record["selected_sha256"] != selected_hash
            or original_candidate["checkpoint_sha256"] != selected_hash
        ):
            raise ValueError(f"selected_checkpoint_binding_mismatch:{run}")
        verify_cached_file(
            old_path,
            original_binding["reports"][run]["games_sha256"],
            f"selected_games:{run}",
        )
        verify_cached_file(
            old_report_path,
            original_binding["reports"][run]["report_sha256"],
            f"selected_report:{run}",
        )
        for filename, key in (
            ("model.npz", "model_sha256"),
            ("weights.json", "weights_sha256"),
            ("metadata.json", "metadata_sha256"),
            ("search_policy.json", "search_policy_sha256"),
        ):
            verify_cached_file(
                Path(original_candidate["artifact"]) / filename,
                original_candidate[key],
                f"selected_candidate:{run}:{filename}",
            )
        verify_report_identity(
            old_report_path,
            original_candidate["artifact"],
            original_binding["opponent"]["artifact"],
            sha(SUITE),
            json.loads(REG.read_text())["evaluation"]["runtime_contract"],
        )
        selected[run] = validate_games(old_path, openings)
        report = json.loads(Path(info["report"]).read_text())
        if not np.isclose(float(fixed[run].mean()), float(report["score"])):
            raise ValueError(f"report_score_accounting_mismatch:{run}")
        old_report = json.loads(old_report_path.read_text())
        if not np.isclose(float(selected[run].mean()), float(old_report["score"])):
            raise ValueError(f"selected_score_accounting_mismatch:{run}")
    per_order = {}
    fixed_vectors = []
    interaction_vectors = []
    for seed in ORDERS:
        a, b = f"order_{seed}_A", f"order_{seed}_B"
        selected_effect = selected[b] - selected[a]
        fixed_effect = fixed[b] - fixed[a]
        interaction = selected_effect - fixed_effect
        selected_epochs = {
            arm: original_binding["candidates"][f"order_{seed}_{arm}"]["selected_epoch"]
            for arm in ("A", "B")
        }
        per_order[str(seed)] = {
            "selected_effect": float(selected_effect.mean()),
            "fixed_effect": float(fixed_effect.mean()),
            "selection_interaction": float(interaction.mean()),
            "A_selected_minus_E4": float(selected[a].mean() - fixed[a].mean()),
            "B_selected_minus_E4": float(selected[b].mean() - fixed[b].mean()),
            "selected_epochs": selected_epochs,
            "e4_checkpoint_sha256": {
                arm: amendment["checkpoints"][f"order_{seed}_{arm}"]["epochs"]["E4"]
                for arm in ("A", "B")
            },
        }
        fixed_vectors.append(fixed_effect)
        interaction_vectors.append(interaction)
    rng = np.random.default_rng(385)
    ix = rng.integers(0, 256, size=(10_000, 256))

    def interval(vectors: list[np.ndarray]) -> dict[str, float]:
        matrix = np.stack(vectors)
        draws = matrix[:, ix].mean(axis=2).mean(axis=0)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return {
            "mean": float(matrix.mean()),
            "lower_95": float(lo),
            "upper_95": float(hi),
        }

    fixed_summary = interval(fixed_vectors)
    interaction_summary = interval(interaction_vectors)
    material = abs(interaction_summary["mean"]) >= 0.03 and (
        interaction_summary["lower_95"] > 0 or interaction_summary["upper_95"] < 0
    )
    return {
        "schema": "seed461-lr-fixed-e4-diagnostic-results-v1",
        "status": "retrospective_diagnostic",
        "original_rejection_preserved": True,
        "holdout_consumed": True,
        "fixed_effect": fixed_summary,
        "selection_interaction": interaction_summary,
        "interaction_classification": "material_average_selection_interaction"
        if material
        else "no_clear_average_interaction",
        "per_order": per_order,
        "between_order_ranges": {
            name: [
                min(row[name] for row in per_order.values()),
                max(row[name] for row in per_order.values()),
            ]
            for name in ("selected_effect", "fixed_effect", "selection_interaction")
        },
        "bootstrap": spec["analysis"]["bootstrap"],
        "evidence_identity": {
            "spec_sha256": sha(SPEC),
            "registration_sha256": sha(REG),
            "training_amendment_sha256": sha(AMEND),
            "suite_sha256": sha(SUITE),
            "original_binding_sha256": sha(BIND),
            "e4_binding_sha256": sha(DATA / "seed461-lr-fixed-e4-evidence.json"),
        },
        "game_accounting": {
            "per_run": 512,
            "runs": 10,
            "total": 5120,
            "additional": 2560,
            "extensions": 0,
        },
    }


def main() -> None:
    spec = json.loads(SPEC.read_text())
    verify_spec(spec)
    amendment = json.loads(AMEND.read_text())
    original_binding = json.loads(BIND.read_text())
    e4_binding = ensure_e4_evaluations(spec, amendment, original_binding)
    binding_path = DATA / "seed461-lr-fixed-e4-evidence.json"
    binding_path.write_text(json.dumps(e4_binding, indent=2, sort_keys=True) + "\n")
    result = analyze(spec, amendment, original_binding, e4_binding)
    out = DATA / "seed461-lr-fixed-e4-diagnostic-results.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# PR #384 retrospective fixed-E4 diagnostic",
        "",
        "Retrospective only; the shared holdout is consumed. PR #384's rejection of LR 0.0005 is preserved.",
        "",
        f"Mean fixed-E4 effect (B−A): **{result['fixed_effect']['mean']:+.4f}** "
        f"(95% opening-cluster bootstrap interval {result['fixed_effect']['lower_95']:+.4f} to {result['fixed_effect']['upper_95']:+.4f}).",
        f"Mean selection interaction: **{result['selection_interaction']['mean']:+.4f}** "
        f"(95% interval {result['selection_interaction']['lower_95']:+.4f} to {result['selection_interaction']['upper_95']:+.4f}); **{result['interaction_classification']}**.",
        "",
        "| Order | Selected effect | Fixed E4 effect | Interaction | A selected−E4 | B selected−E4 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, row in result["per_order"].items():
        lines.append(
            f"| {seed} | {row['selected_effect']:+.4f} | {row['fixed_effect']:+.4f} | "
            f"{row['selection_interaction']:+.4f} | {row['A_selected_minus_E4']:+.4f} | "
            f"{row['B_selected_minus_E4']:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"Between-order ranges: selected [{result['between_order_ranges']['selected_effect'][0]:+.4f}, {result['between_order_ranges']['selected_effect'][1]:+.4f}]; "
            f"fixed E4 [{result['between_order_ranges']['fixed_effect'][0]:+.4f}, {result['between_order_ranges']['fixed_effect'][1]:+.4f}]; "
            f"interaction [{result['between_order_ranges']['selection_interaction'][0]:+.4f}, {result['between_order_ranges']['selection_interaction'][1]:+.4f}].",
            "",
            "Epoch-fixed heterogeneity persists descriptively: the five fixed-E4 effects span both signs and a wide range. A subsequent experiment should vary training order/permutation seed at fixed LR and epoch, using a fresh preregistered holdout.",
            "",
            "No checkpoint was selected from these games, and this diagnostic does not change the original rejection or imply promotion.",
            "",
            "Hash-bound inputs and per-run checkpoint/artifact/report/game identities are in `seed461-lr-fixed-e4-evidence.json`; compact numerical output is in `seed461-lr-fixed-e4-diagnostic-results.json`. Detailed games and exports are preserved under `.tmp/seed461-lr-sensitivity/`.",
        ]
    )
    (DATA / "seed461-lr-fixed-e4-diagnostic-results.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
