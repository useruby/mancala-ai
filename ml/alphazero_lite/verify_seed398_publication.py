"""Verify the public seed398 evidence without checkpoints or native searches.

Run from the repository root with:
    python -m ml.alphazero_lite.verify_seed398_publication

This checks public, hash-bound evidence. Runtime/checkpoint file identity checks
are intentionally separate because those files are not part of the publication.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.seed461_arena_validation import validate_arena_evidence

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed398-policy-value-composition"
TREATMENTS = ("FF", "FS", "SF", "SS")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_command(
    name: str,
    command: list[str],
    treatment: dict[str, Any],
    opponent: str,
    registration: dict[str, Any],
) -> None:
    """Check each argument of the recorded arena invocation, including sources."""
    require(len(command) == 38, f"command_length_mismatch:{name}")
    args = dict(zip(command[2::2], command[3::2]))
    # The executable and arena script occupy the first two argv slots; the rest
    # are flag/value pairs in the registered runner's stable invocation.
    execution_root = Path(command[1]).resolve().parents[2]
    suite_path = registration["evaluation"]["suite"]["path"]
    expected = {
        "--challenger": treatment["policy_artifact"],
        "--current": opponent,
        "--challenger-policy-artifact": treatment["policy_artifact"],
        "--challenger-value-artifact": treatment["value_artifact"],
        "--current-policy-artifact": opponent,
        "--current-value-artifact": opponent,
        "--games": "1024",
        "--games-per-opening": "2",
        "--opening-prefixes-jsonl": str(execution_root / suite_path),
        "--suite-sha256": "10dcb48c9a6bc268ab45ec0a67895b335bb9b669b309115d6ff37b6218131423",
        "--challenger-simulations": "384",
        "--current-simulations": "384",
        "--seed": "398",
        "--workers": "24",
        "--c-puct": "1.25",
        "--seed-contract": "azlite_eval_seed_v2",
        "--game-jsonl": str(
            execution_root
            / ".tmp/seed398-policy-value-composition"
            / f"{name}-games.jsonl"
        ),
        "--out": str(
            execution_root / ".tmp/seed398-policy-value-composition" / f"{name}.json"
        ),
    }
    require(
        command[0] == str(execution_root / ".venv-azlite/bin/python"),
        f"command_executable_mismatch:{name}",
    )
    require(
        command[1] == str(execution_root / "ml/alphazero_lite/arena.py"),
        f"command_script_mismatch:{name}",
    )
    for flag, value in expected.items():
        require(args.get(flag) == value, f"command_identity_mismatch:{name}:{flag}")


def verify_binding_identity(
    registration: dict[str, Any], binding: dict[str, Any], registration_path: Path
) -> None:
    require(
        registration["schema"] == "seed398-policy-value-composition-registration-v1",
        "registration_schema_mismatch",
    )
    require(
        binding["registration_sha256"] == sha256(registration_path),
        "binding_registration_mismatch",
    )
    require(
        binding["execution_source_hashes"] == registration["execution_source_hashes"],
        "binding_execution_sources_mismatch",
    )
    require(
        binding["treatments"] == registration["treatments"],
        "binding_treatment_identity_mismatch",
    )


def verify_publication() -> dict[str, Any]:
    registration_path = DATA / "registration.json"
    binding_path = DATA / "evaluation-binding.json"
    registration = read_json(registration_path)
    binding = read_json(binding_path)
    verify_binding_identity(registration, binding, registration_path)
    require(
        binding["suite_sha256"] == sha256(DATA / "seed398-openings-v2.jsonl"),
        "binding_suite_mismatch",
    )
    require(
        binding["suite_sha256"] == registration["evaluation"]["suite"]["sha256"],
        "registration_suite_mismatch",
    )
    require(
        binding["exclusion_proof_sha256"]
        == sha256(DATA / "opening-exclusion-proof.json"),
        "binding_exclusion_proof_mismatch",
    )
    require(
        binding["exclusion_proof_sha256"] == registration["exclusion_proof"]["sha256"],
        "registration_exclusion_proof_mismatch",
    )

    for name, digest in registration["execution_source_hashes"].items():
        snapshot = DATA / "execution-source-snapshots" / name
        require(
            snapshot.is_file() and sha256(snapshot) == digest,
            f"execution_snapshot_mismatch:{name}",
        )
    proof = read_json(DATA / "opening-exclusion-proof.json")
    require(
        proof["excluded_state_count"] == binding["excluded_state_count"],
        "exclusion_proof_count_mismatch",
    )
    require(
        proof["excluded_identity_sha256"]
        == registration["exclusion_proof"]["excluded_identity_sha256"],
        "exclusion_proof_identity_mismatch",
    )
    openings = [
        json.loads(line)
        for line in (DATA / "seed398-openings-v2.jsonl").read_text().splitlines()
    ]
    require(
        len(openings) == 512 and len({row["state_hash"] for row in openings}) == 512,
        "suite_count_or_uniqueness_mismatch",
    )

    runtime = registration["component_sources"]["native_runtime_contract"]
    all_rows: list[dict[str, Any]] = []
    scores: dict[str, np.ndarray] = {}
    ledger_rows = [
        json.loads(line)
        for line in (DATA / "validated-outcome-ledger.jsonl").read_text().splitlines()
    ]
    require(len(ledger_rows) == 4096, "ledger_game_count_mismatch")
    for name in TREATMENTS:
        record = binding["reports"][name]
        treatment = registration["treatments"][name]
        require(
            record["state"] == "completed_validated", f"report_state_invalid:{name}"
        )
        verify_command(
            name,
            record["command"],
            treatment,
            registration["component_sources"]["seed455"]["artifact"],
            registration,
        )
        report_path = DATA / "original-arena-reports" / f"{name}.json"
        require(
            sha256(report_path) == record["report_sha256"],
            f"report_hash_mismatch:{name}",
        )
        report = read_json(report_path)
        rows = [row for row in ledger_rows if row["treatment"] == name]
        require(len(rows) == 1024, f"treatment_ledger_count_mismatch:{name}")
        scores[name] = validate_arena_evidence(
            report,
            rows,
            openings,
            name,
            {"artifact": treatment["policy_artifact"], "runtime_contract": runtime},
            {"artifact": registration["component_sources"]["seed455"]["artifact"]},
            {
                **registration["evaluation"],
                "games_per_candidate": 1024,
                "suite": registration["evaluation"]["suite"],
                "arena_seed": 398,
                "seed_contract": "azlite_eval_seed_v2",
            },
        )
        all_rows.extend(rows)
    require(
        sha256(DATA / "validated-outcome-ledger.jsonl")
        == read_json(DATA / "analysis.original.json")["outcome_ledger_sha256"],
        "ledger_hash_mismatch",
    )

    matrix = read_json(DATA / "four-treatment-opening-score-matrix.json")
    original = read_json(DATA / "analysis.original.json")
    corrected = read_json(DATA / "analysis.corrected.json")
    matrix_hash = sha256(DATA / "four-treatment-opening-score-matrix.json")
    expected_matrix_identities = {
        "registration_sha256": binding["registration_sha256"],
        "evaluation_binding_sha256": sha256(binding_path),
        "outcome_ledger_sha256": sha256(DATA / "validated-outcome-ledger.jsonl"),
    }
    for identity, digest in expected_matrix_identities.items():
        require(matrix.get(identity) == digest, f"matrix_identity_mismatch:{identity}")
    for analysis in (original, corrected):
        require(
            analysis["opening_matrix_sha256"] == matrix_hash,
            "analysis_matrix_hash_mismatch",
        )
    require(len(matrix["opening_scores"]) == 512, "matrix_count_mismatch")
    for i, row in enumerate(matrix["opening_scores"]):
        require(
            row["opening_index"] == i
            and row["opening_state_hash"] == openings[i]["state_hash"],
            f"matrix_opening_mismatch:{i}",
        )
        for name in TREATMENTS:
            require(
                row[name] == float(scores[name][i]), f"matrix_score_mismatch:{name}:{i}"
            )

    # Frozen analyzer bootstrap: openings are the resampling clusters; treatment
    # scores and both seats travel together in every draw.
    rng = np.random.default_rng(398)
    draws = rng.integers(0, 512, size=(10_000, 512))
    contrasts = {
        "policy_FS_minus_SS": scores["FS"] - scores["SS"],
        "value_SF_minus_SS": scores["SF"] - scores["SS"],
        "FF_minus_SS": scores["FF"] - scores["SS"],
        "interaction_FF_minus_FS_minus_SF_plus_SS": scores["FF"]
        - scores["FS"]
        - scores["SF"]
        + scores["SS"],
    }
    intervals = {}
    for i, (label, values) in enumerate(contrasts.items()):
        quantiles = [0.0125, 0.9875] if i < 2 else [0.025, 0.975]
        intervals[label] = [
            float(x) for x in np.quantile(values[draws].mean(axis=1), quantiles)
        ]
    require(
        intervals["policy_FS_minus_SS"] == [-0.03125, 0.016607666015625355],
        "policy_primary_interval_mismatch",
    )
    require(
        intervals["value_SF_minus_SS"] == [-0.02294921875, 0.025390625],
        "value_primary_interval_mismatch",
    )
    require(
        float(contrasts["value_SF_minus_SS"].mean()) == 0.0009765625,
        "value_point_estimate_mismatch",
    )
    require(
        not (
            contrasts["policy_FS_minus_SS"].mean() >= 0.03
            and intervals["policy_FS_minus_SS"][0] > 0
        ),
        "policy_followup_should_be_false",
    )
    require(
        not (
            contrasts["value_SF_minus_SS"].mean() >= 0.03
            and intervals["value_SF_minus_SS"][0] > 0
        ),
        "value_followup_should_be_false",
    )

    require(
        (DATA / "analysis.json").read_bytes()
        == (DATA / "analysis.corrected.json").read_bytes(),
        "published_analysis_derivative_mismatch",
    )
    for label in original["contrasts"]:
        require(
            {
                k: v
                for k, v in corrected["contrasts"][label].items()
                if k != "classification"
            }
            == {
                k: v
                for k, v in original["contrasts"][label].items()
                if k != "classification"
            },
            f"corrected_analysis_effect_values_changed:{label}",
        )
    for label, values in contrasts.items():
        frozen_classification = (
            "merits_follow_up_training_study"
            if label in ("policy_FS_minus_SS", "value_SF_minus_SS")
            and values.mean() >= 0.03
            and intervals[label][0] > 0
            else "supported_harmful_effect"
            if values.mean() <= -0.03 and intervals[label][1] < 0
            else "positive_but_threshold_not_met"
            if values.mean() > 0
            and label in ("policy_FS_minus_SS", "value_SF_minus_SS")
            else "uncertain"
        )
        require(
            corrected["contrasts"][label]["classification"] == frozen_classification,
            f"classification_mismatch:{label}",
        )
        require(
            corrected["contrasts"][label]["mean"] == float(values.mean()),
            f"analysis_mean_mismatch:{label}",
        )
        require(
            corrected["contrasts"][label]["interval"] == intervals[label],
            f"analysis_interval_mismatch:{label}",
        )
    for name in TREATMENTS:
        treatment_rows = [row for row in ledger_rows if row["treatment"] == name]
        summary = corrected["outcomes"][name]
        for outcome, winner in (
            ("wins", "challenger"),
            ("draws", "draw"),
            ("losses", "current"),
        ):
            require(
                summary[outcome]
                == sum(row["winner"] == winner for row in treatment_rows),
                f"analysis_outcome_mismatch:{name}:{outcome}",
            )
        require(
            summary["score"] == float(scores[name].mean()),
            f"analysis_score_mismatch:{name}",
        )
    require(
        corrected["decision"] == {"policy_follow_up": False, "value_follow_up": False},
        "followup_decision_mismatch",
    )
    receipt = read_json(DATA / "correction-receipt.json")
    require(
        receipt["original_analysis_sha256"] == sha256(DATA / "analysis.original.json"),
        "correction_original_hash_mismatch",
    )
    require(
        receipt["corrected_analysis_sha256"]
        == sha256(DATA / "analysis.corrected.json"),
        "correction_corrected_hash_mismatch",
    )
    require(
        receipt["original_report_hashes"]
        == {n: binding["reports"][n]["report_sha256"] for n in TREATMENTS},
        "correction_report_hash_mismatch",
    )
    return {
        "status": "verified",
        "games": 4096,
        "treatments": len(TREATMENTS),
        "primary_97_5_intervals": intervals,
    }


def main() -> None:
    print(json.dumps(verify_publication(), indent=2))


if __name__ == "__main__":
    main()
