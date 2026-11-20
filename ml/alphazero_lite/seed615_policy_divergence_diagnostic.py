"""Read-only descriptive retrospective for the authenticated Seed 615 ledger.

This module replays only recorded game trajectories and evaluates network policy
priors on recorded states.  It never invokes search, a solver, or a game runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed615_evaluation as contract
from ml.alphazero_lite import verify_seed615_evaluation as verifier
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_table

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path("docs/data/seed615-source-opening-diversity/evaluation-preparation")
OUTPUT = Path("docs/data/seed615-source-opening-diversity/posthoc-policy-diagnostic")
FREEZE_SHA256 = "ba30f5e3efadf3358db9f03c7b4eba4f69b23e2f95dcb4089723a1fb9664f32b"
CHECKPOINTS = {
    "B512": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
    "S615": "cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780",
}
SERVING = "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad"
FROZEN_RESULTS = {
    "B512_score": 0.502197265625,
    "S615_score": 0.496337890625,
    "delta": -0.005859375,
    "interval": [-0.01611328125, 0.00439453125],
    "S615_vs_P_E4_lower_bound": 0.486572265625,
    "interpretation": "neither",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def authenticate(
    root: Path = Path("/home/alex/Mancala/ai-seed615-authorized-eval"),
) -> dict[str, Any]:
    """Run the existing independent verifier and bind every registered identity."""
    root = root.resolve()
    evidence = root / EVIDENCE
    freeze = evidence / "execution-freeze-v4.json"
    if sha256_bytes(freeze.read_bytes()) != FREEZE_SHA256:
        raise ValueError("evaluation_freeze_identity_mismatch")
    result = verifier.verify_results(root)
    if result != {
        "passed": True,
        "completed_games": 4096,
        "capped_games": 0,
        "evaluation_authorized": True,
        "analysis_present": True,
    }:
        raise ValueError("independent_evaluation_verification_mismatch")
    freeze_doc = json.loads(freeze.read_bytes())
    for label, digest in CHECKPOINTS.items():
        path = root / freeze_doc["registration"]["models"][label]["path"]
        if sha256_bytes(path.read_bytes()) != digest:
            raise ValueError(f"checkpoint_identity_mismatch:{label}")
    if freeze_doc["serving_model_sha256"] != SERVING:
        raise ValueError("serving_checkpoint_identity_mismatch")
    for label, digest in CHECKPOINTS.items():
        path = (
            root
            / "docs/data/seed615-source-opening-diversity/evaluation-preparation/runtime-artifacts"
            / label
            / "model.npz"
        )
        if sha256_bytes(path.read_bytes()) != digest:
            raise ValueError(f"runtime_checkpoint_identity_mismatch:{label}")
    analysis = json.loads((evidence / "analysis.json").read_bytes())
    actual = analysis["results"]
    validate_published_results(actual)
    # Check the registered manifests against the active freeze byte hashes.
    for field, relative in (
        ("file_sha256", contract.OPENINGS),
        ("file_sha256_assignments", contract.ASSIGNMENTS),
    ):
        if sha256_bytes((root / relative).read_bytes()) != freeze_doc["cohort"][field]:
            raise ValueError(f"registered_manifest_identity_mismatch:{relative}")
    return {
        "verification": result,
        "freeze_sha256": FREEZE_SHA256,
        "checkpoint_sha256": {**CHECKPOINTS, "P-E4": SERVING},
        "analysis_sha256": sha256_bytes((evidence / "analysis.json").read_bytes()),
        "opening_manifest_sha256": freeze_doc["cohort"]["file_sha256"],
        "assignment_manifest_sha256": freeze_doc["cohort"]["file_sha256_assignments"],
    }


def _score(margin: int) -> float:
    return 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0


def validate_published_results(actual: dict[str, Any]) -> None:
    """Fail closed unless the frozen confirmatory numbers match exactly."""
    observed = {
        "B512_score": actual["B512_score"],
        "S615_score": actual["S615_score"],
        "delta": actual["S615_score"] - actual["B512_score"],
        "interval": actual["S615_minus_B512_interval"],
        "S615_vs_P_E4_lower_bound": actual["S615_vs_P_E4_one_sided_lower_bound"],
        "interpretation": actual["interpretation"],
    }
    if observed != FROZEN_RESULTS:
        raise ValueError("published_analysis_identity_or_result_mismatch")


def reconstruct_openings(
    root: Path = Path("/home/alex/Mancala/ai-seed615-authorized-eval"),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconstruct seat-wise and paired opening results from verified receipts."""
    root = root.resolve()
    authentication = authenticate(root)
    openings = _jsonl(root / contract.OPENINGS)
    assignments = _jsonl(root / contract.ASSIGNMENTS)
    expected = contract.expected_assignments(openings)
    verifier.validate_assignment_rows(assignments, expected)
    outcomes: dict[tuple[int, str], dict[int, dict[str, Any]]] = {}
    trajectories: dict[
        tuple[int, str, int], tuple[dict[str, Any], dict[str, Any], list[int]]
    ]
    trajectories = {}
    for assignment in assignments:
        game_id = assignment["game_id"]
        row = verifier.verify_one_result(root, assignment)
        game_path = (
            root / contract.RESULTS / game_id / "attempts/attempt-0001/game.jsonl"
        )
        raw = json.loads(game_path.read_text(encoding="utf-8"))
        key = (assignment["opening_index"], assignment["candidate"])
        seat = assignment["candidate_seat"]
        outcomes.setdefault(key, {})[seat] = {
            "score": _score(row["margin"]),
            "margin": row["margin"],
            "winner": row["winner"],
        }
        trajectories[(assignment["opening_index"], assignment["candidate"], seat)] = (
            assignment,
            raw,
            [int(move) for move in raw["trajectory"].split(",")],
        )
    paired = []
    for opening_index in range(1024):
        b = outcomes[(opening_index, "B512")]
        s = outcomes[(opening_index, "S615")]
        b_avg = (b[0]["score"] + b[1]["score"]) / 2
        s_avg = (s[0]["score"] + s[1]["score"]) / 2
        paired.append(
            {
                "opening_index": opening_index,
                "opening_id": openings[opening_index]["opening_id"],
                "B512": {str(seat): b[seat] for seat in (0, 1)},
                "S615": {str(seat): s[seat] for seat in (0, 1)},
                "B512_average": b_avg,
                "S615_average": s_avg,
                "S615_minus_B512": s_avg - b_avg,
            }
        )
    b_score = sum(row["B512_average"] for row in paired) / 1024
    s_score = sum(row["S615_average"] for row in paired) / 1024
    if (
        b_score != FROZEN_RESULTS["B512_score"]
        or s_score != FROZEN_RESULTS["S615_score"]
    ):
        raise ValueError("reconstructed_score_disagrees_with_frozen_result")
    return paired, [
        build_divergence(
            trajectories[(i, "B512", seat)],
            trajectories[(i, "S615", seat)],
            authentication,
        )
        for i in range(1024)
        for seat in (0, 1)
    ]


def build_divergence(
    b_game: tuple[dict[str, Any], dict[str, Any], list[int]],
    s_game: tuple[dict[str, Any], dict[str, Any], list[int]],
    authentication: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay the common opening and report the first different recorded move."""
    b_assignment, b_row, b_moves = b_game
    s_assignment, s_row, s_moves = s_game
    if (
        (b_assignment["opening_index"], b_assignment["candidate_seat"])
        != (s_assignment["opening_index"], s_assignment["candidate_seat"])
        or b_assignment["candidate"] != "B512"
        or s_assignment["candidate"] != "S615"
    ):
        raise ValueError("incorrect_opening_or_seat_pairing")
    if b_assignment["opening_identity"] != s_assignment["opening_identity"]:
        raise ValueError("opening_identity_pairing_mismatch")
    prefix = b_row["opening_prefix_moves"]
    if prefix != s_row["opening_prefix_moves"]:
        raise ValueError("opening_prefix_pairing_mismatch")
    game = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    for move in prefix:
        if (
            game.over()
            or move not in game.possible_moves()
            or not game.move(game.pit_index(int(move)))
        ):
            raise ValueError("illegal_registered_opening")
    common_prefix_state = game.to_state()
    same = 0
    for left, right in zip(b_moves, s_moves):
        if left != right:
            break
        absolute = left
        if (
            game.over()
            or absolute // 6 != game.current_player
            or not game.move(absolute)
        ):
            raise ValueError("illegal_common_trajectory")
        same += 1
    if same == min(len(b_moves), len(s_moves)) and len(b_moves) == len(s_moves):
        return {
            "opening_index": b_assignment["opening_index"],
            "candidate_seat": b_assignment["candidate_seat"],
            "identical_trajectory": True,
            "first_differing_move_index": None,
            "candidate_seed_b512": b_assignment["game_seed"],
            "candidate_seed_s615": s_assignment["game_seed"],
            "outcome_b512": b_row["winner"],
            "outcome_s615": s_row["winner"],
        }
    if same >= min(len(b_moves), len(s_moves)):
        raise ValueError("trajectory_prefix_or_length_mismatch")
    if same == 0:
        game = KalahGame.from_state(common_prefix_state)
    else:
        # The shared moves were already replayed above.
        pass
    b_move, s_move = b_moves[same], s_moves[same]
    actor = game.current_player
    candidate_seat = b_assignment["candidate_seat"]
    legal = game.possible_moves()
    state = game.to_state()
    stones = sum(game.pits)
    b_consequences = move_consequence_table(state)
    row = {
        "opening_index": b_assignment["opening_index"],
        "candidate_seat": candidate_seat,
        "identical_trajectory": False,
        "first_differing_move_index": same,
        "state_before_divergence": state,
        "canonical_state_identity": sha256_bytes(canonical_json(state)),
        "player_to_move": actor,
        "actor": "candidate" if actor == candidate_seat else "shared_P-E4_opponent",
        "legal_move_count": len(legal),
        "active_pit_stone_count": stones,
        "extra_turn_available": any(
            move["gives_extra_turn"] for move in b_consequences if move["legal"]
        ),
        "capture_available": any(
            move["produces_capture"] for move in b_consequences if move["legal"]
        ),
        "exact_root_eligible": stones <= 16,
        "b512_move": b_move % 6,
        "s615_move": s_move % 6,
        "b512_seed": b_assignment["game_seed"],
        "s615_seed": s_assignment["game_seed"],
        "outcome_b512": b_row["winner"],
        "outcome_s615": s_row["winner"],
        "b512_move_consequence": b_consequences[b_move % 6],
        "s615_move_consequence": b_consequences[s_move % 6],
    }
    return row


def canonical_state_identity(state: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(state))


def compare_priors(
    policy_b: np.ndarray, policy_s: np.ndarray, legal: list[int]
) -> dict[str, Any]:
    """Compute descriptive divergences after strict legal masking/normalization."""
    b = np.asarray(policy_b, dtype=np.float64).copy()
    s = np.asarray(policy_s, dtype=np.float64).copy()
    if b.shape != (6,) or s.shape != (6,) or not legal:
        raise ValueError("policy_shape_or_legal_moves_invalid")
    mask = np.zeros(6, dtype=bool)
    mask[legal] = True
    b[~mask] = 0
    s[~mask] = 0
    for vector in (b, s):
        if not np.all(np.isfinite(vector)) or np.any(vector < 0) or vector.sum() <= 0:
            raise ValueError("invalid_policy_probabilities")
        vector /= vector.sum()
    midpoint = (b + s) / 2
    positive_b, positive_s = b > 0, s > 0
    js = 0.5 * np.sum(b[positive_b] * np.log2(b[positive_b] / midpoint[positive_b]))
    js += 0.5 * np.sum(s[positive_s] * np.log2(s[positive_s] / midpoint[positive_s]))

    def entropy(probabilities: np.ndarray) -> float:
        positive = probabilities[probabilities > 0]
        return float(-np.sum(positive * np.log2(positive)))

    return {
        "B512_legal_probabilities": b.tolist(),
        "S615_legal_probabilities": s.tolist(),
        "top_prior_agreement": int(np.argmax(b)) == int(np.argmax(s)),
        "B512_top_action": int(np.argmax(b)),
        "S615_top_action": int(np.argmax(s)),
        "B512_preferred_action_probability": float(b[np.argmax(b)]),
        "S615_preferred_action_probability": float(s[np.argmax(s)]),
        "B512_mass_on_S615_preferred_action": float(b[np.argmax(s)]),
        "S615_mass_on_B512_preferred_action": float(s[np.argmax(b)]),
        "total_variation": float(0.5 * np.abs(b - s).sum()),
        "jensen_shannon_bits": float(js),
        "B512_entropy_bits": entropy(b),
        "S615_entropy_bits": entropy(s),
        "legal_action_count": len(legal),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/alex/Mancala/ai-seed615-authorized-eval"),
        help="Original execution worktree; recorded absolute command paths remain unchanged",
    )
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output
    if not output.is_absolute():
        output = ROOT / output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"create_only_output_exists:{output}")
    auth = authenticate(root)
    paired, divergences = reconstruct_openings(root)
    # Policy evaluation is performed by the pinned NumPy artifact evaluator.
    from ml.alphazero_lite.arena import ArtifactEvaluator

    artifacts = (
        root
        / "docs/data/seed615-source-opening-diversity/evaluation-preparation/runtime-artifacts"
    )
    evaluators = {
        name: ArtifactEvaluator(artifacts / name) for name in ("B512", "S615")
    }
    opening_rows = _jsonl(root / contract.OPENINGS)
    primary: dict[str, tuple[dict[str, Any], str]] = {}
    for divergence in divergences:
        if divergence["identical_trajectory"]:
            continue
        identity = divergence["canonical_state_identity"]
        primary.setdefault(
            identity, (divergence["state_before_divergence"], "first_divergence")
        )
    for opening in opening_rows:
        state = opening["state"]
        primary[canonical_state_identity(state)] = (state, "registered_opening")
    if sum(cohort == "registered_opening" for _, cohort in primary.values()) != 1024:
        raise ValueError("primary_registered_state_cohort_mismatch")
    state_records = []
    for identity, (state, cohort) in sorted(primary.items()):
        game = KalahGame.from_state(state)
        b, _ = evaluators["B512"].evaluate(game)
        s, _ = evaluators["S615"].evaluate(game)
        comparison = compare_priors(b, s, game.possible_moves())
        state_records.append(
            {
                "canonical_state_identity": identity,
                "cohort": cohort,
                "state": state,
                **comparison,
            }
        )
    prior_by_id = {row["canonical_state_identity"]: row for row in state_records}
    for divergence in divergences:
        if not divergence["identical_trajectory"]:
            divergence["policy_prior_comparison"] = prior_by_id[
                divergence["canonical_state_identity"]
            ]
    source_hashes = {path: contract.sha(root / path) for path in contract.SOURCE_PATHS}
    source_hashes["ml/alphazero_lite/seed615_policy_divergence_diagnostic.py"] = (
        contract.sha(Path(__file__))
    )
    provenance = {
        "evaluation_freeze_sha256": auth["freeze_sha256"],
        "B512_checkpoint_sha256": CHECKPOINTS["B512"],
        "S615_checkpoint_sha256": CHECKPOINTS["S615"],
        "P_E4_checkpoint_sha256": SERVING,
        "diagnostic_source_sha256": contract.sha(Path(__file__)),
        "source_sha256": source_hashes,
    }
    for rows in (paired, divergences, state_records):
        for row in rows:
            row["provenance"] = provenance
    files = {
        "paired-opening-outcomes.jsonl": paired,
        "first-divergences.jsonl": divergences,
        "policy-prior-comparison.jsonl": state_records,
    }
    divergence_associations = {}
    for feature in (
        "actor",
        "extra_turn_available",
        "capture_available",
        "exact_root_eligible",
    ):
        values = sorted({str(row.get(feature, "identical")) for row in divergences})
        divergence_associations[feature] = {
            value: {
                "matched_game_pairs": sum(
                    str(row.get(feature, "identical")) == value for row in divergences
                ),
                "B512_candidate_score": float(
                    np.mean(
                        [
                            1.0
                            if row["outcome_b512"] == "challenger"
                            else 0.5
                            if row["outcome_b512"] == "draw"
                            else 0.0
                            for row in divergences
                            if str(row.get(feature, "identical")) == value
                        ]
                    )
                ),
                "S615_candidate_score": float(
                    np.mean(
                        [
                            1.0
                            if row["outcome_s615"] == "challenger"
                            else 0.5
                            if row["outcome_s615"] == "draw"
                            else 0.0
                            for row in divergences
                            if str(row.get(feature, "identical")) == value
                        ]
                    )
                ),
            }
            for value in values
        }
    summary = {
        "schema": "seed615-posthoc-policy-divergence-v1",
        "read_only_posthoc": True,
        "authentication": auth,
        "source_sha256": source_hashes,
        "opening_pairs": len(paired),
        "matched_game_pairs": len(divergences),
        "identical_trajectory_pairs": sum(
            row["identical_trajectory"] for row in divergences
        ),
        "different_trajectory_pairs": sum(
            not row["identical_trajectory"] for row in divergences
        ),
        "prior_states": len(state_records),
        "prior_states_by_cohort": dict(Counter(row["cohort"] for row in state_records)),
        "policy_metrics_by_cohort": {
            cohort: {
                "states": len(group),
                "top_action_disagreements": sum(
                    not row["top_prior_agreement"] for row in group
                ),
                "top_action_disagreement_rate": float(
                    np.mean([not row["top_prior_agreement"] for row in group])
                ),
                "mean_total_variation": float(
                    np.mean([row["total_variation"] for row in group])
                ),
                "mean_jensen_shannon_bits": float(
                    np.mean([row["jensen_shannon_bits"] for row in group])
                ),
            }
            for cohort in ("registered_opening", "first_divergence")
            if (group := [row for row in state_records if row["cohort"] == cohort])
        },
        "outcome_difference_counts": dict(
            Counter(
                "better"
                if row["S615_minus_B512"] > 0
                else "worse"
                if row["S615_minus_B512"] < 0
                else "equal"
                for row in paired
            )
        ),
        "outcome_difference_by_candidate_seat": {
            str(seat): dict(
                Counter(
                    "better"
                    if row["S615"][str(seat)]["score"] > row["B512"][str(seat)]["score"]
                    else "worse"
                    if row["S615"][str(seat)]["score"] < row["B512"][str(seat)]["score"]
                    else "equal"
                    for row in paired
                )
            )
            for seat in (0, 1)
        },
        "matched_assignment_outcome_transitions": {
            f"{b}_to_{s}": sum(
                row["B512"][str(seat)]["winner"] == b
                and row["S615"][str(seat)]["winner"] == s
                for row in paired
                for seat in (0, 1)
            )
            for b in ("challenger", "current", "draw")
            for s in ("challenger", "current", "draw")
        },
        "divergence_by_actor": dict(
            Counter(row.get("actor", "identical") for row in divergences)
        ),
        "divergence_tactical_counts": {
            key: sum(bool(row.get(key)) for row in divergences)
            for key in (
                "extra_turn_available",
                "capture_available",
                "exact_root_eligible",
            )
        },
        "descriptive_divergence_associations": divergence_associations,
        "registered_result": FROZEN_RESULTS,
        "limitations": [
            "Descriptive post-hoc analysis; no new significance tests.",
            "Candidate-identity game seeds differ; opponent divergence and seed effects are not isolated.",
            "Network priors are not searched actions or action-quality measurements.",
            "No new exact solving; move quality beyond ordinary immediate Kalah consequences is unmeasured.",
        ],
    }
    # Bind all sibling outputs to canonical bytes and publish create-only.
    payloads = {
        name: b"".join(canonical_json(row) + b"\n" for row in rows)
        for name, rows in files.items()
    }
    summary["output_sha256"] = {
        name: sha256_bytes(payload) for name, payload in payloads.items()
    }
    summary_payload = json.dumps(summary, sort_keys=True, indent=2).encode() + b"\n"
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (output / name).open("xb").write(payload)
    (output / "diagnostic.json").open("xb").write(summary_payload)
    policy_summary = summary["policy_metrics_by_cohort"]
    outcome_counts = summary["outcome_difference_counts"]
    (output / "README.md").open("xb").write(
        (
            "# Seed 615 post-hoc policy-divergence retrospective\n\n"
            "Read-only replay and deterministic inference on authenticated historical evidence. "
            "This is exploratory and does not alter the registered confirmatory result.\n\n"
            "## Findings\n\n"
            f"- **Identical-position priors:** the registered-opening cohort has {policy_summary['registered_opening']['top_action_disagreements']} top-action disagreements across {policy_summary['registered_opening']['states']} states; mean TV is {policy_summary['registered_opening']['mean_total_variation']:.6f} and mean Jensen–Shannon divergence is {policy_summary['registered_opening']['mean_jensen_shannon_bits']:.6f} bits. The deduplicated first-divergence cohort has {policy_summary.get('first_divergence', {}).get('top_action_disagreements', 0)} disagreements across {policy_summary.get('first_divergence', {}).get('states', 0)} states (mean TV {policy_summary.get('first_divergence', {}).get('mean_total_variation', 0):.6f}; JS {policy_summary.get('first_divergence', {}).get('mean_jensen_shannon_bits', 0):.6f} bits).\n"
            f"- **Recorded trajectories:** {summary['identical_trajectory_pairs']} of {summary['matched_game_pairs']} matched opening/seat game pairs were identical; {summary['different_trajectory_pairs']} first diverged. Of those, {summary['divergence_by_actor'].get('candidate', 0)} first differing turns were by the candidate and {summary['divergence_by_actor'].get('shared_P-E4_opponent', 0)} by P-E4.\n"
            f"- **Tactical associations:** among first-divergence states, {summary['divergence_tactical_counts']['capture_available']} had an immediate capture available, {summary['divergence_tactical_counts']['extra_turn_available']} had an extra turn available, and {summary['divergence_tactical_counts']['exact_root_eligible']} were within the 16-stone exact-root region. For example, the capture-available stratum's recorded candidate scores were B512 {summary['descriptive_divergence_associations']['capture_available']['True']['B512_candidate_score']:.3f} and S615 {summary['descriptive_divergence_associations']['capture_available']['True']['S615_candidate_score']:.3f}; this descriptive selection is not an action-quality or causal comparison.\n"
            f"- **Opening outcomes:** S615 was better on {outcome_counts.get('better', 0)} openings, worse on {outcome_counts.get('worse', 0)}, and equal on {outcome_counts.get('equal', 0)}. The frozen aggregate remains B512 {FROZEN_RESULTS['B512_score']}, S615 {FROZEN_RESULTS['S615_score']} (difference {FROZEN_RESULTS['delta']}); paired 95% interval {FROZEN_RESULTS['interval']}; S615 lower bound vs P-E4 {FROZEN_RESULTS['S615_vs_P_E4_lower_bound']}; registered interpretation **neither**.\n"
            "- **Mechanism:** tactical/phase strata and actor-specific outcome summaries are descriptive associations. They do not identify causal effects or establish action quality; available evidence is insufficient to identify a specific mechanism.\n\n"
            f"Evidence bindings: freeze `{auth['freeze_sha256']}`; B512 `{CHECKPOINTS['B512']}`; S615 `{CHECKPOINTS['S615']}`; P-E4 `{SERVING}`.\n\n"
            "Recorded outcome labels `challenger`, `current`, and `draw` are from each game's candidate perspective. `diagnostic.json` contains the full authenticated source-hash map and hashes of all JSONL outputs; each JSONL row carries the same source/checkpoint provenance.\n\n"
            "The recorded games were independently verified before reconstruction. Matched games share opening and candidate seat, but their numeric seeds differ because candidate identity is part of the seed contract. First action divergence is therefore not uniquely attributable to the model intervention.\n\n"
            "The training intervention increased effective source-opening-group diversity while also changing actual training-state membership. This evaluation therefore compares a changed sampling/allocation strategy; it does not isolate a single diversity statistic.\n\n"
            "Policy-prior metrics describe the original network policy distributions on identical states. They do not describe MCTS-selected actions, counterfactual outcomes, or action quality. Tactical indicators use ordinary Kalah consequences only; no new exact solve was run. Post-hoc associations are descriptive, not significance tests or causal evidence. No candidate is selected or promoted.\n"
        ).encode()
    )
    print(json.dumps({"output": str(output), **summary}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
