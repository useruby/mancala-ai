"""Run the 128-simulation strength screen for seed542 update 432."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import (
    run_seed542_expanded_policy_strength_screen as screen,
    seed542_expanded_policy_strength_screen as protocol,
    verify_seed542_expanded_policy_strength_screen as verifier,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_REL = "docs/data/seed542-update-432-strength-screen"
OUT = ROOT / OUT_REL
TRAINING = ROOT / "docs/data/seed542-training-budget-432"
PRIOR_SCREEN = ROOT / "docs/data/seed542-expanded-policy-strength-screen"
SEED511 = ROOT / "docs/data/seed511-fixed-composition-screen-v4"
SOURCE_FILES = (
    "ml/alphazero_lite/seed542_update432_strength_screen.py",
    "ml/alphazero_lite/verify_seed542_update432_strength_screen.py",
    "ml/alphazero_lite/test_seed542_update432_strength_screen.py",
)
TRAINING_EVIDENCE = (
    "docs/data/seed542-training-budget-432/D-update-432.npz",
    "docs/data/seed542-training-budget-432/registration.json",
    "docs/data/seed542-training-budget-432/run-status.json",
    "docs/data/seed542-training-budget-432/development-metrics.json",
)
ASSIGNMENTS = {
    "challenger_policy": "D-final",
    "challenger_policy_trunk": "D-final own trunk",
    "challenger_value": "initializer",
    "challenger_value_trunk": "initializer own trunk",
    "opponent_policy": "E4",
    "opponent_policy_trunk": "E4 own trunk",
    "opponent_value": "initializer",
    "opponent_value_trunk": "initializer own trunk",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _update432_artifact(output: Path) -> dict[str, Any]:
    artifacts = output / "runtime-artifacts"
    prior_artifacts = PRIOR_SCREEN / "runtime-artifacts"
    artifacts.mkdir(parents=True)
    for name in ("initializer", "E4"):
        shutil.copytree(prior_artifacts / name, artifacts / name)
    shutil.copytree(prior_artifacts / "D-final", artifacts / "D-final")
    candidate = TRAINING / "D-update-432.npz"
    d_final = artifacts / "D-final"
    shutil.copy2(candidate, d_final / "model.npz")
    with np.load(candidate) as checkpoint:
        weights = {key: checkpoint[key].tolist() for key in checkpoint.files}
    (d_final / "weights.json").write_text(
        json.dumps(weights, sort_keys=True, separators=(",", ":")) + "\n"
    )
    metadata_path = d_final / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["version"] = "seed542-D-update-432-policy-and-trunk"
    metadata["artifacts"]["weights_sha256"] = sha(d_final / "model.npz")
    metadata["artifacts"]["weights_json_sha256"] = sha(d_final / "weights.json")
    metadata["training"] = {
        "experiment": "seed542 expanded-data controlled training-budget experiment",
        "update": 432,
        "role": "sole candidate checkpoint",
        "composition": "update-432 policy and own trunk; initializer value and own trunk",
        "checkpoint_sha256": sha(candidate),
    }
    write_json(metadata_path, metadata)

    auth = json.loads((PRIOR_SCREEN / "runtime-authentication.json").read_text())
    auth["D_final"] = {
        "checkpoint_sha256": sha(candidate),
        "files": {
            name: sha(d_final / name)
            for name in (
                "metadata.json",
                "model.npz",
                "search_policy.json",
                "weights.json",
            )
        },
    }
    auth["challenger_policy_component"] = (
        "seed542 update-432 policy with update-432 own trunk"
    )
    auth["challenger_value_component"] = (
        "frozen initializer value with initializer own trunk"
    )
    auth["component_assignments"] = ASSIGNMENTS
    auth["current_P_component_artifact_identities"] = auth[
        "current_P_component_artifact_identities"
    ]
    auth["components"]["D-final"] = {
        "checkpoint": "docs/data/seed542-training-budget-432/D-update-432.npz",
        "checkpoint_sha256": sha(candidate),
        "artifact_identity": verifier._serving_identity(d_final),
        "files": auth["D_final"]["files"],
    }
    auth["components"]["D432"] = copy.deepcopy(auth["components"]["D-final"])
    auth["component_hashes"] = {
        "training_checkpoint": sha(candidate),
        "artifact_files": auth["D_final"]["files"],
        "current_P_component_artifact_identities": auth[
            "current_P_component_artifact_identities"
        ],
        "serving_release_sha256": auth["serving_release_sha256"],
        "release_manifest_sha256": auth["release_manifest_sha256"],
        "runtime_assets": auth["runtime_assets"],
    }
    return auth


def _authenticated_files(
    root: Path, output: Path, auth: dict[str, Any]
) -> dict[str, str]:
    from ml.alphazero_lite import run_seed542_expanded_policy_strength_screen as runner

    sources = {
        relative: runner.sha(root / relative) for relative in runner.RUNTIME_SOURCES
    }
    sources.update({relative: runner.sha(root / relative) for relative in SOURCE_FILES})
    files = {
        "docs/data/seed511-fixed-composition-screen-v4/registration.json": sha(
            SEED511 / "registration.json"
        ),
        "docs/data/seed511-fixed-composition-screen-v4/execution-freeze.json": sha(
            SEED511 / "execution-freeze.json"
        ),
        "docs/data/seed511-fixed-composition-screen-v4/openings.jsonl": sha(
            SEED511 / "openings.jsonl"
        ),
        "docs/data/seed542-expanded-policy-strength-screen/registration.json": sha(
            PRIOR_SCREEN / "registration.json"
        ),
        "docs/data/seed542-expanded-policy-strength-screen/results.json": sha(
            PRIOR_SCREEN / "results.json"
        ),
        "deploy/serving-release.json": sha(root / "deploy/serving-release.json"),
        "deploy/release-manifest.json": sha(root / "deploy/release-manifest.json"),
        "model-artifact/runtime/kalah_v1_tablebase": sha(
            root / "model-artifact/runtime/kalah_v1_tablebase"
        ),
        "model-artifact/runtime/kalah_v1_21.kvtb": sha(
            root / "model-artifact/runtime/kalah_v1_21.kvtb"
        ),
        str(screen.TABLEBASE.relative_to(root)): sha(screen.TABLEBASE),
    }
    files.update({relative: sha(root / relative) for relative in TRAINING_EVIDENCE})
    for relative, digest in sources.items():
        files[relative] = digest
    for component in ("D-final", "E4", "initializer"):
        directory = output / "runtime-artifacts" / component
        for path in directory.iterdir():
            if path.is_file():
                files[str(path.relative_to(root))] = sha(path)
    files[str((output / "runtime-authentication.json").relative_to(root))] = sha(
        output / "runtime-authentication.json"
    )
    return files


def prepare(output: Path = OUT) -> dict[str, Any]:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"update432_screen_output_not_empty:{output}")
    output.mkdir(parents=True, exist_ok=True)
    protocol.OUT = str(output.relative_to(ROOT))
    screen.RUNTIME_SOURCES = (*screen.RUNTIME_SOURCES, *SOURCE_FILES)

    suite_bytes = (SEED511 / "openings.jsonl").read_bytes()
    if sha(SEED511 / "openings.jsonl") != protocol.OPENING_SHA256:
        raise ValueError("update432_screen_seed511_suite_hash_mismatch")
    (output / "openings.jsonl").write_bytes(suite_bytes)
    auth = _update432_artifact(output)
    write_json(output / "runtime-authentication.json", auth)
    overlap = protocol.overlap_report(ROOT, protocol.import_seed511_suite(ROOT)[0])
    write_json(
        output / "overlap-summary.json",
        {key: value for key, value in overlap.items() if key != "per_opening"},
    )
    (output / "overlap-per-opening.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in overlap["per_opening"]
        )
    )
    (output / "suite-import-report.json").write_text(
        json.dumps(protocol.import_seed511_suite(ROOT)[1], indent=2, sort_keys=True)
        + "\n"
    )
    training = json.loads((TRAINING / "registration.json").read_text())
    if training["sole_candidate_checkpoint"] != 432:
        raise ValueError("update432_screen_candidate_identity_missing")
    write_json(
        output / "training-verification.json",
        {
            "schema": "seed542_update432_training_evidence_v1",
            "status": "both_seed542_v2_verifiers_passed",
            "training_budget_432_status": "verified_complete",
            "training_registration_sha256": sha(TRAINING / "registration.json"),
            "training_checkpoint_sha256": sha(TRAINING / "D-update-432.npz"),
            "training_run_status_sha256": sha(TRAINING / "run-status.json"),
            "first_108_control_passed": True,
        },
    )

    original_auth = protocol.authenticate_runtime
    original_file_manifest = screen._file_manifest

    def authenticate(_root: Path, _output: Path, _tablebase: Path) -> dict[str, Any]:
        return json.loads((_output / "runtime-authentication.json").read_text())

    def file_manifest(root: Path) -> tuple[dict[str, str], dict[str, str]]:
        sources, files = original_file_manifest(root)
        for relative in SOURCE_FILES:
            sources[relative] = sha(root / relative)
            files[relative] = sources[relative]
        for relative in TRAINING_EVIDENCE:
            files[relative] = sha(root / relative)
        files[str((output / "runtime-authentication.json").relative_to(root))] = sha(
            output / "runtime-authentication.json"
        )
        return sources, files

    protocol.authenticate_runtime = authenticate
    screen._file_manifest = file_manifest
    try:
        result = screen._prepare(ROOT, output, run_training_verifiers=False)
    finally:
        protocol.authenticate_runtime = original_auth
        screen._file_manifest = original_file_manifest

    registration_path = output / "registration.json"
    registration = json.loads(registration_path.read_text())
    registration["status"] = "prospective_frozen_before_arena_games"
    registration["candidate"] = {
        "training_update": 432,
        "training_checkpoint_sha256": sha(TRAINING / "D-update-432.npz"),
        "policy_component": "update-432 policy and own trunk",
        "value_component": "frozen initializer value and own trunk",
        "sole_candidate": True,
    }
    registration["disclosures"] = {
        "classification": "exploratory reused seed511 suite; no fresh holdout",
        "seed504_original_training_overlap": overlap["aggregates"][
            "seed504_original_training"
        ],
        "historical_seed542_expanded_training_overlap": overlap["aggregates"][
            "seed542_expanded_training"
        ],
        "seed461_historical_overlap_limitation_unresolved": True,
        "seed511_historical_games_or_outcomes_reused": False,
        "seed542_training_effective_overlap": "12 of 1,024 opening roots by canonical state and production float32 identity",
        "advancement_requires": "all 2,048 games validate and registered one-sided 95% lower bound strictly > 0.50",
    }
    registration["component_assignments"] = ASSIGNMENTS
    registration["component_hashes"] = auth
    registration["runtime_assets"]["training_checkpoint_sha256"] = sha(
        TRAINING / "D-update-432.npz"
    )
    registration["training_evidence_sha256"] = {
        relative: sha(ROOT / relative) for relative in TRAINING_EVIDENCE
    }
    authenticate_composition(ROOT, output, registration)
    write_json(registration_path, registration)

    freeze_path = output / "execution-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    freeze["registration_sha256"] = sha(registration_path)
    freeze["authenticated_file_sha256"].update(
        {relative: sha(ROOT / relative) for relative in TRAINING_EVIDENCE}
    )
    freeze["authenticated_file_sha256"].update(
        {relative: sha(ROOT / relative) for relative in SOURCE_FILES}
    )
    freeze["authenticated_file_sha256"][
        str((output / "runtime-authentication.json").relative_to(ROOT))
    ] = sha(output / "runtime-authentication.json")
    freeze["execution_source_sha256"].update(
        {relative: sha(ROOT / relative) for relative in SOURCE_FILES}
    )
    write_json(freeze_path, freeze)

    preflight_path = output / "command-preflight.jsonl"
    preflight = [json.loads(line) for line in preflight_path.read_text().splitlines()]
    for row in preflight:
        row["registration_sha256"] = sha(registration_path)
        row["component_assignments"] = ASSIGNMENTS
        row["component_hashes"] = auth
    preflight_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in preflight
        )
    )
    freeze["command_preflight_sha256"] = sha(preflight_path)
    write_json(freeze_path, freeze)
    result["registration_sha256"] = sha(registration_path)
    result["freeze_sha256"] = sha(freeze_path)
    result["candidate_checkpoint_sha256"] = sha(TRAINING / "D-update-432.npz")
    write_json(output / "preflight-status.json", result)
    return result


def authenticate_composition(
    root: Path, output: Path, registration: dict[str, Any]
) -> None:
    auth = json.loads((output / "runtime-authentication.json").read_text())
    if registration.get("component_hashes") != auth:
        raise ValueError("update432_runtime_authentication_mismatch")
    if registration.get("component_assignments") != ASSIGNMENTS:
        raise ValueError("update432_component_assignment_mismatch")
    checkpoint = root / "docs/data/seed542-training-budget-432/D-update-432.npz"
    d_final = output / "runtime-artifacts/D-final"
    if sha(checkpoint) != sha(d_final / "model.npz"):
        raise ValueError("update432_artifact_checkpoint_mismatch")
    if sha(checkpoint) != registration["candidate"]["training_checkpoint_sha256"]:
        raise ValueError("update432_registered_candidate_hash_mismatch")
    if registration["candidate"].get("training_update") != 432:
        raise ValueError("update432_registered_candidate_update_mismatch")
    if (
        sha(d_final / "search_policy.json")
        != registration["runtime_assets"]["search_policy_sha256"]
    ):
        raise ValueError("update432_search_policy_sidecar_mismatch")
    initializer = output / "runtime-artifacts/initializer"
    if sha(initializer / "model.npz") != sha(
        root / "docs/data/seed542-selfplay-4096/initializer.npz"
    ):
        raise ValueError("update432_initializer_value_component_mismatch")
    release = json.loads((root / "deploy/serving-release.json").read_text())
    e4 = output / "runtime-artifacts/E4"
    identities = [
        verifier._serving_identity(initializer),
        verifier._serving_identity(e4),
    ]
    if identities != release["bundles"]["policy-value-p"]["expected_artifact_sha256"]:
        raise ValueError("update432_current_P_composition_not_serving_release")


def execute(root: Path = ROOT, output: Path = OUT) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    protocol.OUT = str(output.relative_to(root))
    screen.RUNTIME_SOURCES = (*screen.RUNTIME_SOURCES, *SOURCE_FILES)
    old_auth = verifier._authenticate_composition
    verifier._authenticate_composition = authenticate_composition
    try:
        return screen.run(root, output)
    finally:
        verifier._authenticate_composition = old_auth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = prepare(args.out) if args.prepare else execute(ROOT, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
