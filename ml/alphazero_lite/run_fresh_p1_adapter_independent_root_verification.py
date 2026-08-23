#!/usr/bin/env python3
"""Preregistered independent per-root-action PUCT verification screen.

The verifier is post-search only: each legal action receives a new A16 tree
rooted after that action. P1 is used only for frozen offline references and the
root_qsync positive control, never by either verification lane.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator, canonical_game_state_hash
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.root_action_verification import (
    select_verified_move,
    verify_root_actions,
)
from ml.alphazero_lite.run_fresh_p1_adapter_matched_q_feedback import (
    _bootstrap,
    _game_hash,
)
from ml.alphazero_lite.run_fresh_p1_adapter_q_feedback_necessity import (
    C_PUCT,
    SIMULATIONS,
    _seed,
    decode_kalah_v3_base_state,
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_fresh_p1_adapter_synchronized_backup import _control_subset
from ml.alphazero_lite.self_play import Node, PUCT


REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN = (
    REPO_ROOT
    / "docs/data/alphazero-lite-fresh-p1-adapter-q-feedback-necessity-frozen-amplified-roots.json"
)
PR222 = (
    REPO_ROOT
    / "docs/data/alphazero-lite-fresh-p1-adapter-postdivergence-amplification-summary.json"
)
MANIFEST = (
    REPO_ROOT
    / "docs/data/alphazero-lite-fresh-p1-adapter-margin-sensitivity-manifest.json"
)
PR220 = (
    REPO_ROOT / "docs/data/alphazero-lite-fresh-p1-adapter-puct-divergence-summary.json"
)
BUDGETS = (32, 64)
HARD_ROOT = "362958a9d30519f98e27a71256c89f1acba0a61e3d93e609adb59609bff61674"


def _search(
    evaluator: ArtifactEvaluator,
    state: dict[str, Any],
    seed: int,
    *,
    override: Any = None,
    hook: Any = None,
) -> dict[str, Any]:
    search = PUCT(
        evaluator=evaluator,
        simulations=SIMULATIONS,
        c_puct=C_PUCT,
        rng=random.Random(seed),
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="deterministic",
        root_temperature=0.0,
        tactical_root_bias=0.0,
        selection_q_override=override,
        pre_simulation_hook=hook,
    )
    search.run(KalahGame.from_state(state), dirichlet_alpha=None, dirichlet_epsilon=0.0)
    return search.root_summary()


def _root_reference_hook(references: dict[int, dict[int, float]]) -> Any:
    def hook(simulation: int, root: Node) -> None:
        references[simulation] = {
            int(move): float(child.q_value)
            for move, child in root.children.items()
            if child.visit_count
        }

    return hook


def _root_qsync(
    root_hash: str, references: dict[int, dict[int, float]], start: int
) -> Any:
    def override(
        simulation: int, state_hash: str, move: int, _q: float, visits: int
    ) -> float | None:
        if simulation < start or state_hash != root_hash or visits <= 0:
            return None
        return references.get(simulation, {}).get(move)

    return override


def _q_rows(
    main: dict[str, Any], p1: dict[str, Any], verification: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    main_stats = {int(row["move"]): row for row in main["child_stats"]}
    p1_stats = {int(row["move"]): row for row in p1["child_stats"]}
    return [
        {
            "action": action,
            "normal_a16_q": float(main_stats[action]["q_value"]),
            "matched_p1_q": float(p1_stats[action]["q_value"]),
            "verification_q": float(verification[action]["verification_q"]),
            "normal_a16_visits": int(main_stats[action]["visits"]),
            "p1_visits": int(p1_stats[action]["visits"]),
        }
        for action in sorted(verification)
    ]


def _ranking(q_rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = {
        name: {int(row["action"]): float(row[name]) for row in q_rows}
        for name in ("normal_a16_q", "matched_p1_q", "verification_q")
    }

    def compare(left: str, right: str) -> dict[str, Any]:
        actions = sorted(values[left])
        left_order = sorted(actions, key=lambda action: (-values[left][action], action))
        right_order = sorted(
            actions, key=lambda action: (-values[right][action], action)
        )
        diffs = [
            abs(values[left][action] - values[right][action]) for action in actions
        ]
        pairs = [
            (a, b) for index, a in enumerate(actions) for b in actions[index + 1 :]
        ]
        agreement = sum(
            (values[left][a] - values[left][b]) * (values[right][a] - values[right][b])
            >= 0
            for a, b in pairs
        )
        return {
            "best_q_action_agreement": left_order[0] == right_order[0],
            "pairwise_rank_agreement": None if not pairs else agreement / len(pairs),
            "q_l1": sum(diffs),
            "max_abs_q_difference": max(diffs, default=0.0),
        }

    return {
        "verification_vs_p1": compare("verification_q", "matched_p1_q"),
        "a16_main_q_vs_p1": compare("normal_a16_q", "matched_p1_q"),
        "verification_vs_a16_main_q": compare("verification_q", "normal_a16_q"),
    }


def _audit(
    state: dict[str, Any],
    metadata: dict[str, Any],
    *,
    p1: ArtifactEvaluator,
    a16: ArtifactEvaluator,
    verifier_mode: str,
) -> dict[str, Any]:
    root_hash, seed = str(metadata["state_hash"]), _seed(str(metadata["state_hash"]))
    if canonical_game_state_hash(KalahGame.from_state(state)) != root_hash:
        raise RuntimeError(f"state hash mismatch: {root_hash}")
    main, references = _search(a16, state, seed), {}
    p1_summary = _search(p1, state, seed, hook=_root_reference_hook(references))
    qsync_start = (
        int(
            metadata.get(
                "actual_first_divergence", metadata.get("first_divergence", {})
            ).get("simulation", 0)
        )
        + 1
    )
    qsync = _search(
        a16,
        state,
        seed,
        override=_root_qsync(
            _game_hash(KalahGame.from_state(state)), references, qsync_start
        ),
    )
    lanes: dict[str, Any] = {
        "main_full": {"final_move": main["selected_move"]},
        "root_qsync": {"final_move": qsync["selected_move"]},
    }
    for budget in BUDGETS:
        verification = verify_root_actions(
            KalahGame.from_state(state),
            evaluator=a16,
            root_hash=root_hash,
            budget=budget,
            ablation_mode=verifier_mode,
        )
        q_rows = _q_rows(main, p1_summary, verification)
        final_move = select_verified_move(verification, main_summary=main)
        main_q_winner = max(
            q_rows, key=lambda row: (row["normal_a16_q"], -row["action"])
        )["action"]
        verification_q_winner = max(
            q_rows, key=lambda row: (row["verification_q"], -row["action"])
        )["action"]
        lanes[f"verify_{budget}"] = {
            "final_move": final_move,
            "actions": verification,
            "q_rows": q_rows,
            "ranking": _ranking(q_rows),
            "rescue_mechanism": None
            if final_move != p1_summary["selected_move"]
            else "ranking_correction"
            if verification_q_winner == p1_summary["selected_move"]
            and main_q_winner != p1_summary["selected_move"]
            else "visit_cascade_correction"
            if main_q_winner == p1_summary["selected_move"]
            and main["selected_move"] != p1_summary["selected_move"]
            else "other",
        }
    return {
        **metadata,
        "search_seed": seed,
        "p1_final_move": p1_summary["selected_move"],
        "main_summary": main,
        "p1_summary": p1_summary,
        "lanes": lanes,
        "root_qsync_start": qsync_start,
    }


def _screen(
    records: list[dict[str, Any]], lanes: tuple[str, ...], *, rescue: bool
) -> dict[str, Any]:
    result = {}
    for lane in lanes:
        matches = [
            row["lanes"][lane]["final_move"] == row["p1_final_move"] for row in records
        ]
        result[lane] = {
            "rescue_rate" if rescue else "new_divergence_rate": _bootstrap(
                matches if rescue else [not match for match in matches]
            ),
            "rescued_hashes" if rescue else "divergent_hashes": [
                row["state_hash"]
                for row, match in zip(records, matches, strict=True)
                if match == rescue
            ],
            "retained_hashes": []
            if not rescue
            else [
                row["state_hash"]
                for row, match in zip(records, matches, strict=True)
                if not match
            ],
        }
    return result


def _compute(records: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    extras = [
        sum(
            int(row["simulations"])
            for row in record["lanes"][f"verify_{budget}"]["actions"].values()
        )
        for record in records
    ]
    legal = [len(record["lanes"][f"verify_{budget}"]["actions"]) for record in records]
    return {
        "mean_legal_actions_per_root": float(np.mean(legal)),
        "mean_extra_simulations_per_root": float(np.mean(extras)),
        "p50_extra_simulations": float(np.percentile(extras, 50)),
        "p90_extra_simulations": float(np.percentile(extras, 90)),
        "max_extra_simulations": int(max(extras)),
        "total_verification_simulations": int(sum(extras)),
        "overhead_vs_1200_main": float(np.mean(extras) / SIMULATIONS),
    }


def _classification(
    invariants: dict[str, bool],
    screen: dict[str, Any],
    controls: dict[str, Any],
    *,
    verifier_mode: str,
) -> tuple[str, int | None, str]:
    if not all(invariants.values()):
        return (
            "invariant_failure",
            None,
            "Repair the frozen reproduction or perspective contract.",
        )
    passing = [
        budget
        for budget in BUDGETS
        if screen[f"verify_{budget}"]["rescue_rate"]["estimate"] >= 0.70
        and controls[f"verify_{budget}"]["new_divergence_rate"]["estimate"] <= 0.10
    ]
    if not passing:
        follow_up = (
            "Test a deliberately decorrelated value-only or flat-prior independent verifier."
            if verifier_mode == "full"
            else "Stop pursuing independent root-action verification; the parent reference contains useful information not recovered by either coupled or flat-prior A16 search."
        )
        return "parent_reference_contains_unique_information", None, follow_up
    selected = min(passing)
    return (
        "inconclusive",
        selected,
        "Run the preregistered canonical arena and P1 parent-search quality gate before classifying the selected verifier.",
    )


def _report(summary: dict[str, Any]) -> str:
    lanes = ("verify_32", "verify_64", "root_qsync")
    screen = ["| Lane | Rescue | New divergence |", "| --- | ---: | ---: |"]
    for lane in lanes:
        rescue, control = (
            summary["frozen_screen"][lane]["rescue_rate"],
            summary["negative_controls"][lane]["new_divergence_rate"],
        )
        screen.append(
            f"| `{lane}` | {rescue['estimate']:.3f} [{rescue['lower_95']:.3f}, {rescue['upper_95']:.3f}] | {control['estimate']:.3f} [{control['lower_95']:.3f}, {control['upper_95']:.3f}] |"
        )
    root_rows = [
        "| Root hash | verify_32 | verify_64 | root_qsync |",
        "| --- | --- | --- | --- |",
    ]
    by_hash = {row["state_hash"]: row for row in summary["records"]}
    for state_hash in summary["frozen_amplified_roots"]:
        row = by_hash[state_hash]
        root_rows.append(
            "| `{}` | {} | {} | {} |".format(
                state_hash,
                *(
                    "rescued"
                    if row["lanes"][lane]["final_move"] == row["p1_final_move"]
                    else "retained"
                    for lane in lanes
                ),
            )
        )
    control_rows = [
        "| Root hash | verify_32 | verify_64 | root_qsync |",
        "| --- | --- | --- | --- |",
    ]
    for state_hash in summary["negative_control_roots"]:
        row = by_hash[state_hash]
        control_rows.append(
            "| `{}` | {} | {} | {} |".format(
                state_hash,
                *(
                    "new divergence"
                    if row["lanes"][lane]["final_move"] != row["p1_final_move"]
                    else "stable"
                    for lane in lanes
                ),
            )
        )
    rank_rows = [
        "| Root hash | Budget | Verify/P1 best-Q | A16/P1 best-Q | Verify/P1 pairwise |",
        "| --- | ---: | --- | --- | ---: |",
    ]
    for row in summary["records"]:
        for budget in BUDGETS:
            ranking = row["lanes"][f"verify_{budget}"]["ranking"]
            rank_rows.append(
                f"| `{row['state_hash']}` | {budget} | {ranking['verification_vs_p1']['best_q_action_agreement']} | {ranking['a16_main_q_vs_p1']['best_q_action_agreement']} | {ranking['verification_vs_p1']['pairwise_rank_agreement']:.3f} |"
            )
    mode = summary["guardrails"]["verifier_mode"]
    title = (
        "Value-Only Flat-Prior Root-Action Verification"
        if mode == "value_only"
        else "Independent Root-Action Verification"
    )
    return "\n".join(
        [
            f"# {title}",
            "",
            f"**Classification:** `{summary['classification']}`",
            "",
            f"**Recommended follow-up:** {summary['recommended_follow_up']}",
            "",
            "## Frozen Screen",
            "",
            *screen,
            "",
            "## Amplified Rescue Table",
            "",
            *root_rows,
            "",
            "## Washed-Out Controls",
            "",
            *control_rows,
            "",
            "## Compute Overhead",
            "",
            "```json",
            json.dumps(summary["compute_overhead"], indent=2, sort_keys=True),
            "```",
            "",
            "## Q Ranking Table",
            "",
            *rank_rows,
            "",
            "## Verification Q Diagnostics",
            "",
            "Complete per-root/action normal A16, matched offline P1, and independent verification Q values, visits, L1, and maximum differences are in `records[].lanes[].q_rows` and `ranking`.",
            "",
            f"Persistent hard root: `{HARD_ROOT}`.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p1-workdir", type=Path, default=Path("/tmp/azlite_fresh_selfplay_anchor")
    )
    parser.add_argument(
        "--adapter-workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_parent_adapter"),
    )
    parser.add_argument(
        "--verifier-mode",
        choices=("full", "value_only"),
        default="full",
        help="Verifier-only PUCT mode; value_only uses flat legal priors and A16 values.",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-adapter-independent-root-verification-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-adapter-independent-root-verification-results.md",
    )
    args = parser.parse_args()
    frozen, prior, manifest, held = (
        json.loads(path.read_text()) for path in (FROZEN, PR222, MANIFEST, PR220)
    )
    manifest_by_hash, prior_by_hash = (
        {row["state_hash"]: row for row in manifest["rows"]},
        {row["state_hash"]: row for row in prior["records"]},
    )
    amplified = list(frozen["full_amplified_1200"])
    controls_meta = _control_subset(
        prior["records"], [prior_by_hash[key] for key in amplified]
    )
    replay_path = args.adapter_workdir / "fresh_p1_self_play.jsonl"
    replay = read_jsonl(replay_path)
    p1_artifact, a16_artifact = (
        args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/artifact",
        args.adapter_workdir / "artifacts/step_0016/artifact",
    )
    p1, a16 = ArtifactEvaluator(p1_artifact), ArtifactEvaluator(a16_artifact)

    def state_for(meta: dict[str, Any]) -> dict[str, Any]:
        return decode_kalah_v3_base_state(
            list(replay[int(meta["replay_index"])]["state"])
        )

    selected_meta = [manifest_by_hash[key] for key in amplified] + [
        manifest_by_hash[row["state_hash"]] for row in controls_meta
    ]
    records = [
        _audit(state_for(meta), meta, p1=p1, a16=a16, verifier_mode=args.verifier_mode)
        for meta in selected_meta
    ]
    by_hash = {row["state_hash"]: row for row in records}
    primary, controls = (
        [by_hash[key] for key in amplified],
        [by_hash[row["state_hash"]] for row in controls_meta],
    )
    lane_names = ("verify_32", "verify_64", "root_qsync")
    screen, negative = (
        _screen(primary, lane_names, rescue=True),
        _screen(controls, lane_names, rescue=False),
    )
    invariants = {
        "exact_40_amplified": len(primary) == 40
        and all(
            row["lanes"]["main_full"]["final_move"] != row["p1_final_move"]
            for row in primary
        ),
        "root_qsync_39_of_40": len(screen["root_qsync"]["rescued_hashes"]) == 39,
        "root_qsync_controls_stable": not negative["root_qsync"]["divergent_hashes"],
        "frozen_identity": frozen["identity_hash"]
        == "60ebaa037dc931f9d9a7d71e0171c5708db995ae88a119346ffa639812c5b9d1",
    }
    classification, selected_budget, follow_up = _classification(
        invariants, screen, negative, verifier_mode=args.verifier_mode
    )
    held_records = (
        []
        if selected_budget is None
        else [
            _audit(
                row["state"],
                {"state_hash": row["state_hash"], **row},
                p1=p1,
                a16=a16,
                verifier_mode=args.verifier_mode,
            )
            for row in held["first_game_divergences"]
        ]
    )
    summary = {
        "schema": "azlite_independent_root_action_verification_v1",
        "guardrails": {
            "training": False,
            "self_play": False,
            "main_search_modified": False,
            "verification_uses_p1": False,
            "verification_budgets": list(BUDGETS),
            "verifier_mode": args.verifier_mode,
            "c_puct": C_PUCT,
            "fpu_mode": "zero",
            "normalize_values": False,
            "root_noise": False,
        },
        "hashes": {
            "frozen": sha256_file(FROZEN),
            "replay": sha256_file(replay_path),
            "p1_artifact_weights": sha256_file(p1_artifact / "weights.json"),
            "a16_artifact_weights": sha256_file(a16_artifact / "weights.json"),
        },
        "frozen_amplified_roots": amplified,
        "negative_control_roots": [row["state_hash"] for row in controls_meta],
        "frozen_screen": screen,
        "negative_controls": negative,
        "compute_overhead": {
            str(budget): _compute(records, budget) for budget in BUDGETS
        },
        "selected_budget": selected_budget,
        "invariants": invariants,
        "persistent_hard_root": {
            "hash": HARD_ROOT,
            "verify_32": by_hash[HARD_ROOT]["lanes"]["verify_32"]["final_move"]
            == by_hash[HARD_ROOT]["p1_final_move"],
            "verify_64": by_hash[HARD_ROOT]["lanes"]["verify_64"]["final_move"]
            == by_hash[HARD_ROOT]["p1_final_move"],
        },
        "records": records,
        "held_out_pr220": held_records,
        "canonical_arena": None,
        "parent_search_quality": None,
        "classification": classification,
        "recommended_follow_up": follow_up,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.out_report.write_text(_report(summary))
    print(classification)


if __name__ == "__main__":
    main()
