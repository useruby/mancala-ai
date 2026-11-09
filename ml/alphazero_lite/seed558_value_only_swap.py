"""Paired retrospective seed558 diagnostic for swapping only P's value route."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite import arena
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed558-value-only-swap"
BASE = ROOT / "docs/data/seed557-retrospective-diagnostic"
ARTIFACTS = ROOT / "docs/data/seed542-update-432-strength-screen-v3/runtime-artifacts"
P_ARTIFACT = (
    ROOT / "docs/data/seed504-policy-only-ablation/screen/runtime-artifacts/P-E4"
)
SEED = 558
OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "deterministic",
    "root_temperature": 0.0,
    "tactical_root_bias": 0.0,
}
ARMS = ("initializer_value", "D432_value")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def frozen_inputs() -> dict[str, Any]:
    cohort = (BASE / "cohort.json").read_bytes()
    oracle = (BASE / "oracle-records.jsonl").read_bytes()
    source = ROOT / "docs/data/seed542-update-432-strength-screen-v3"
    runtime = read_json(source / "runtime-authentication.json")
    # Bind the immutable source identities and complete component exports.
    files: dict[str, Any] = {}
    for label, directory in (
        ("P_E4", P_ARTIFACT),
        ("initializer", ARTIFACTS / "initializer"),
        ("D432", ARTIFACTS / "D-final"),
    ):
        files[label] = {
            name: sha((directory / name).read_bytes())
            for name in (
                "metadata.json",
                "model.npz",
                "search_policy.json",
                "weights.json",
            )
        }
    if files["D432"]["model.npz"] != (
        "4555cba73d2c908d1eca829836f361b226ccc1072be28affd816b2a240123163"
    ):
        raise ValueError("D432 training checkpoint identity mismatch")
    if files["D432"] != runtime["D_final"]["files"]:
        raise ValueError(
            "D432 export does not match authenticated #546 runtime identity"
        )
    reg = read_json(source / "registration.json")
    seed557_freeze = read_json(BASE / "execution-freeze.json")
    serving_auth = read_json(
        ROOT
        / "docs/data/seed542-expanded-policy-strength-screen/runtime-authentication.json"
    )
    for label, frozen_name in (("P_E4", "E4"), ("initializer", "initializer")):
        authenticated_files = serving_auth["components"][frozen_name]["files"]
        if any(
            files[label][key] != value for key, value in authenticated_files.items()
        ):
            raise ValueError(
                f"{label} export differs from authenticated serving publication"
            )
    return {
        "schema": "seed558-value-only-freeze-v1",
        "seed": SEED,
        "cohort_sha256": sha(cohort),
        "oracle_records_sha256": sha(oracle),
        "cohort_roots": 256,
        "oracle_roots": 256,
        "oracle_action_margins_reused_verbatim": True,
        "publications": {
            "seed557_merge_commit": "44834acaa0d7ee17465e807d10bc90e841a51a89",
            "seed546_merge_commit": "54ca41c2f499452a424dfdd32f0aede1b584fcb7",
            "seed546_registration_sha256": sha(
                (source / "registration.json").read_bytes()
            ),
            "seed546_execution_freeze_sha256": sha(
                (source / "execution-freeze.json").read_bytes()
            ),
            "seed546_runtime_authentication_sha256": sha(
                (source / "runtime-authentication.json").read_bytes()
            ),
            "seed546_checkpoint_sha256": reg["candidate"]["training_checkpoint_sha256"],
            "seed557_policy_publication_receipts": seed557_freeze[
                "authentication_receipts"
            ],
            "seed557_oracle_identity_record_sha256": sha(
                (BASE / "oracle-identity-verification.json").read_bytes()
            ),
            "seed557_execution_freeze_sha256": sha(
                (BASE / "execution-freeze.json").read_bytes()
            ),
            "serving_runtime_authentication_sha256": sha(
                (
                    ROOT
                    / "docs/data/seed542-expanded-policy-strength-screen/runtime-authentication.json"
                ).read_bytes()
            ),
        },
        "components": files,
        "value_routes": {
            "initializer_value": "seed542 frozen initializer value head and own trunk",
            "D432_value": "complete #546 D-update-432 value head and D432 own trunk",
            "shared_policy": "serving P seed504 E4 policy head and own trunk",
            "only_route_changed": "value head and value trunk",
        },
        "training_overlap": {
            "seed542_expanded_training": reg.get("disclosures", {}).get(
                "historical_seed542_expanded_training_overlap"
            ),
            "seed461_historical_overlap_limitation_unresolved": reg.get(
                "disclosures", {}
            ).get("seed461_historical_overlap_limitation_unresolved"),
            "cohort_overlap": "Not authenticated by the #546 publication; no claim of disjointness.",
        },
        "search": {
            "simulations": 128,
            "c_puct": 1.25,
            "workers": 1,
            "fresh_tree_per_root_arm": True,
            "same_seed_per_root": True,
            "exact_root_solving": False,
            "exact_leaf_solving": False,
            "search_options": OPTIONS,
            "bootstrap": {
                "method": "paired opening-cluster bootstrap, percentile 95%",
                "resamples": 20000,
                "seed": 558,
            },
        },
        "planning_rule": {
            "contrast": "D432-value minus initializer-value mean exact final-margin regret",
            "advance_if": "mean <= -0.25 seeds; 95% interval upper < 0; WDL downgrade rate does not increase",
            "decision_scope": "propose a separate 128-simulation value-only arena only; near-cutoff roots do not establish full-game strength",
            "preserve_decision": "do_not_advance_T512",
        },
    }


def run_searches(cohort: list[dict[str, Any]]) -> list[dict[str, Any]]:
    policy = arena.ArtifactEvaluator(P_ARTIFACT)
    initializer = arena.ArtifactEvaluator(ARTIFACTS / "initializer")
    d432 = arena.ArtifactEvaluator(ARTIFACTS / "D-final")
    evaluators = {
        "initializer_value": arena.ComposedArtifactEvaluator(
            policy, initializer, policy_source="current", value_source="candidate"
        ),
        "D432_value": arena.ComposedArtifactEvaluator(
            policy, d432, policy_source="current", value_source="candidate"
        ),
    }
    rows = []
    for item in cohort:
        state = item["state"]
        game = KalahGame.from_state(state)
        legal = game.possible_moves()
        seed = int(sha(f"seed558:{item['state_hash']}".encode())[:8], 16) % (2**31)
        arm_priors = {}
        for arm, evaluator in evaluators.items():
            priors, _ = evaluator.evaluate(game)
            arm_priors[arm] = [float(value) for value in priors]
        if arm_priors["initializer_value"] != arm_priors["D432_value"]:
            raise ValueError(f"paired policy priors differ at {item['state_hash']}")
        for arm, evaluator in evaluators.items():
            result = arena.evaluate_artifact_position(
                evaluator=evaluator,
                state=state,
                simulations=128,
                seed=seed,
                c_puct=1.25,
                search_options=OPTIONS,
                exact_root_solve_threshold=None,
                exact_solve_stone_threshold=None,
            )
            rows.append(
                {
                    "state_hash": item["state_hash"],
                    "opening_index": item["opening_index"],
                    "arm": arm,
                    "seed": seed,
                    "legal_moves": legal,
                    "priors": arm_priors[arm],
                    "visits": result["visits"],
                    "selected_action": int(result["selected_move"]),
                    "child_stats": result["child_stats"],
                }
            )
    return rows


def score(
    cohort: list[dict[str, Any]],
    searches: list[dict[str, Any]],
    oracles: list[dict[str, Any]],
) -> dict[str, Any]:
    oracle_by_hash = {row["state_hash"]: row for row in oracles}
    searches_by_key = {(row["state_hash"], row["arm"]): row for row in searches}
    per_root = []
    for item in cohort:
        oracle = oracle_by_hash[item["state_hash"]]
        margins = {int(k): int(v) for k, v in oracle["action_margins"].items()}
        best = max(margins.values())
        pair = {}
        for arm in ARMS:
            row = searches_by_key[item["state_hash"], arm]
            move = row["selected_action"]
            margin = margins[move]
            regret = best - margin
            best_wdl = (best > 0) - (best < 0)
            move_wdl = (margin > 0) - (margin < 0)
            pair[arm] = {
                "regret": regret,
                "optimal": regret == 0,
                "wdl_downgrade": move_wdl < best_wdl,
                "action": move,
            }
        per_root.append(
            {
                "state_hash": item["state_hash"],
                "opening_index": item["opening_index"],
                **pair,
            }
        )
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in per_root:
        grouped.setdefault(int(row["opening_index"]), []).append(row)
    opening_diffs = [
        statistics.fmean(
            r["D432_value"]["regret"] - r["initializer_value"]["regret"] for r in rows
        )
        for rows in grouped.values()
    ]
    rng = random.Random(558)
    bootstrap = sorted(
        statistics.fmean(rng.choices(opening_diffs, k=len(opening_diffs)))
        for _ in range(20000)
    )
    contrast = statistics.fmean(
        r["D432_value"]["regret"] - r["initializer_value"]["regret"] for r in per_root
    )
    metrics = {
        arm: {
            "exact_optimal_rate": statistics.fmean(r[arm]["optimal"] for r in per_root),
            "wdl_downgrade_rate": statistics.fmean(
                r[arm]["wdl_downgrade"] for r in per_root
            ),
            "mean_exact_final_margin_regret": statistics.fmean(
                r[arm]["regret"] for r in per_root
            ),
        }
        for arm in ARMS
    }
    changed = sum(
        r["initializer_value"]["action"] != r["D432_value"]["action"] for r in per_root
    )
    interval = [
        bootstrap[int(0.025 * len(bootstrap))],
        bootstrap[int(0.975 * len(bootstrap)) - 1],
    ]
    pass_rule = (
        contrast <= -0.25
        and interval[1] < 0
        and metrics["D432_value"]["wdl_downgrade_rate"]
        <= metrics["initializer_value"]["wdl_downgrade_rate"]
    )
    return {
        "n_roots": len(per_root),
        "primary_contrast_D432_minus_initializer_mean_regret": contrast,
        "paired_opening_bootstrap_95_percent_interval": interval,
        "bootstrap_seed": 558,
        "bootstrap_resamples": 20000,
        "metrics": metrics,
        "selected_action_swap_rate": changed / len(per_root),
        "selected_action_changed_roots": changed,
        "decision": "supports_proposing_separate_128_simulation_value_only_arena"
        if pass_rule
        else "diagnostic_does_not_support_advancing_swap",
        "preserved_decision": "do_not_advance_T512",
        "scope": "Near-cutoff retrospective roots; does not establish full-game strength.",
        "root_records": per_root,
    }


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    freeze = frozen_inputs()
    freeze_path = DATA / "execution-freeze.json"
    freeze_bytes = (json.dumps(freeze, sort_keys=True, indent=2) + "\n").encode()
    if len(sys.argv) > 1 and sys.argv[1] == "refreeze-and-run":
        freeze_path.write_bytes(freeze_bytes)
    elif freeze_path.exists() and freeze_path.read_bytes() != freeze_bytes:
        raise ValueError("frozen execution inputs changed")
    elif not freeze_path.exists():
        freeze_path.write_bytes(freeze_bytes)
    # Preserve the #557 cohort and margins byte-for-byte in this self-contained package.
    for name in ("cohort.json", "oracle-records.jsonl"):
        source = BASE / name
        target = DATA / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"frozen #557 evidence changed: {name}")
        if not target.exists():
            target.write_bytes(source.read_bytes())
    for label, directory in (
        ("P_E4", P_ARTIFACT),
        ("initializer", ARTIFACTS / "initializer"),
        ("D432", ARTIFACTS / "D-final"),
    ):
        target = DATA / "artifacts" / label
        target.mkdir(parents=True, exist_ok=True)
        for filename in (
            "metadata.json",
            "model.npz",
            "search_policy.json",
            "weights.json",
        ):
            source = directory / filename
            dest = target / filename
            if (
                dest.exists()
                and sha(dest.read_bytes()) != freeze["components"][label][filename]
            ):
                raise ValueError(f"bundled component changed: {label}/{filename}")
            if not dest.exists():
                shutil.copyfile(source, dest)
    cohort = read_json(DATA / "cohort.json")["states"]
    searches = run_searches(cohort)
    (DATA / "search-records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in searches)
    )
    result = score(
        cohort,
        searches,
        [
            json.loads(line)
            for line in (DATA / "oracle-records.jsonl").read_text().splitlines()
        ],
    )
    (DATA / "root-metrics.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in result.pop("root_records")
        )
    )
    (DATA / "analysis.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
