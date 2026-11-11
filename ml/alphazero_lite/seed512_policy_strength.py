"""Guarded root-level #571 policy-prior strength evaluation.

The default command verifies the immutable freeze only. Qualification requires
an explicit post-review authorization record and writes results exclusively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import arena
from ml.alphazero_lite.seed551_controlled_policy_strength import (
    canonical_identity,
    production_float32_identity,
)
from ml.alphazero_lite.seed557_retrospective import canonical_hash, search_seed
from ml.alphazero_lite.seed570_qualification_cohort import verify_package

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed512-policy-strength-evaluation"
FREEZE_PATH = DATA / "freeze.json"
SEARCH_OPTIONS = {
    "fpu_mode": "zero",
    "normalize_values": False,
    "reuse_subtree": False,
    "root_policy_mode": "deterministic",
    "root_temperature": 0.0,
    "tactical_root_bias": 0.0,
}
ARTIFACT_HASHES = {
    "P-E4/metadata.json": "a9d133043c64ca41e3d77354f4ccf3403ab5f94286608f01589387159ac0258c",
    "P-E4/model.npz": "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
    "P-E4/search_policy.json": "b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d",
    "P-E4/weights.json": "89dead525a255cff4760867645fd9ce05cb2bbadcf1d72c5ac9089868c4a8136",
    "T512/metadata.json": "703957f2b2da2966ec7e7f67975e775fb9daf7407dfd68f50785550bca143c61",
    "T512/model.npz": "4e16fed2550a64a9413b588f84e8df1eadde7e0b5889b4932cb3783e3cb648e4",
    "T512/search_policy.json": "b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d",
    "T512/weights.json": "171c897e27ec4967d33cd28366f4c0114b25d766455ae41e777c5692fd64e892",
    "initializer/metadata.json": "f3b9a7d665be7b1219b4a3412bf4a499e5bafd525905b50d93deac4e6ad9ccef",
    "initializer/model.npz": "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c",
    "initializer/search_policy.json": "b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d",
    "initializer/weights.json": "f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c",
}


def assert_artifact_hashes(observed: dict[str, str]) -> None:
    if observed != ARTIFACT_HASHES:
        raise ValueError("artifact_hash_manifest_mismatch")


def assert_membership_binding(bundle: dict[str, Any]) -> None:
    if (
        bundle.get("population") != 423
        or bundle.get("membership_sha256")
        != "94e126af6ea88fb99a4230f84e5cf2a20248daa39ea66d25b68730ea0426d668"
    ):
        raise ValueError("qualification_membership_binding_mismatch")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _verify_runtime(freeze: dict[str, Any]) -> None:
    expected = freeze["runtime"]
    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    if observed != {key: expected[key] for key in observed}:
        raise ValueError(f"runtime_version_mismatch:{observed}")
    if torch.cuda.is_available():
        raise ValueError("runtime_device_mismatch:cuda_available")


def _digest_list(items: list[Any]) -> str:
    payload = json.dumps(items, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return sha256(payload)


def verify_runtime_source_bindings(root: Path, bindings: dict[str, str]) -> None:
    for relative, expected in bindings.items():
        path = root / relative
        if not path.is_file() or sha256(path.read_bytes()) != expected:
            raise ValueError(f"runtime_source_hash_mismatch:{relative}")


def verify_authorization(
    freeze_bytes: bytes,
    authorization_bytes: bytes,
    trusted_authorization_sha256: str,
) -> dict[str, Any]:
    if sha256(authorization_bytes) != trusted_authorization_sha256:
        raise PermissionError("trusted_authorization_hash_mismatch")
    auth = json.loads(authorization_bytes)
    if (
        auth.get("schema") != "seed512-qualification-authorization-v1"
        or auth.get("status") != "approved_after_review"
        or auth.get("qualification_authorized") is not True
        or auth.get("freeze_sha256") != sha256(freeze_bytes)
        or type(auth.get("reviewed_pull_request")) is not int
        or auth["reviewed_pull_request"] < 1
        or not isinstance(auth.get("reviewed_merge_commit"), str)
        or len(auth["reviewed_merge_commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in auth["reviewed_merge_commit"]
        )
    ):
        raise PermissionError("reviewed_authorization_binding_mismatch")
    return auth


def configure_evaluation_runtime() -> None:
    """Apply the preregistered single-thread CPU runtime before model work."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("single_thread_runtime_configuration_failed")


def verify_freeze(root: Path = ROOT) -> dict[str, Any]:
    """Verify every input before any evaluator is constructed or search starts."""
    data = root / "docs/data/seed512-policy-strength-evaluation"
    freeze = _json(data / "freeze.json")
    _verify_runtime(freeze)
    if freeze.get("schema") != "seed512-policy-strength-evaluation-freeze-v1":
        raise ValueError("freeze_schema_mismatch")
    if freeze.get("qualification_started") is not False:
        raise ValueError("qualification_already_started")
    if freeze.get("qualification_authorized") is not False:
        raise ValueError("freeze_authorization_flag_must_remain_false")
    verify_runtime_source_bindings(root, freeze["runtime_source_bindings"])
    for relative, expected in freeze["provenance_bindings"].items():
        path = root / relative
        if not path.is_file() or sha256(path.read_bytes()) != expected:
            raise ValueError(f"provenance_hash_mismatch:{relative}")
    for key, relative in (
        ("runner_sha256", "ml/alphazero_lite/seed512_policy_strength.py"),
        ("verifier_sha256", "ml/alphazero_lite/verify_seed512_policy_strength.py"),
        ("synthetic_tests_sha256", "ml/alphazero_lite/test_seed512_policy_strength.py"),
    ):
        actual = sha256((root / relative).read_bytes())
        if freeze.get(key) != actual:
            raise ValueError(f"execution_code_hash_mismatch:{key}")
    observed_artifacts = {}
    for relative, expected in ARTIFACT_HASHES.items():
        path = data / "runtime-artifacts" / relative
        if not path.is_file() or sha256(path.read_bytes()) != expected:
            raise ValueError(f"artifact_hash_mismatch:{relative}")
        observed_artifacts[relative] = expected
    assert_artifact_hashes(observed_artifacts)

    membership_path = root / freeze["cohort"]["source"]
    membership = verify_package(root, membership_path)
    assert_membership_binding(membership)

    training = _json(root / freeze["cohort"]["training_source"])
    roots = training["states"]
    exclusion_sets = {
        "opening_indices": sorted({int(row["opening_index"]) for row in roots}),
        "canonical_identities": sorted(
            {canonical_identity(row["state"]) for row in roots}
        ),
        "kalah_v3_float32_identities": sorted(
            {production_float32_identity(row["state"]) for row in roots}
        ),
    }
    if {key: _digest_list(values) for key, values in exclusion_sets.items()} != freeze[
        "cohort"
    ]["exclusion_set_digests"]:
        raise ValueError("training_exclusion_digest_mismatch")

    oracle_path = root / freeze["cohort"]["oracle_records"]
    if sha256(oracle_path.read_bytes()) != freeze["cohort"]["oracle_records_sha256"]:
        raise ValueError("oracle_records_hash_mismatch")
    gate_path = root / freeze["gate"]["source"]
    if "0.10" not in gate_path.read_text(
        encoding="utf-8"
    ) or "20,000" not in gate_path.read_text(encoding="utf-8"):
        raise ValueError("registered_gate_source_mismatch")
    if freeze["search"]["options"] != SEARCH_OPTIONS:
        raise ValueError("search_options_mismatch")
    pit_counts = [
        sum(row["state"]["player_pits"]) + sum(row["state"]["opponent_pits"])
        for row in membership["roots"]
    ]
    if (
        min(pit_counts) != 17
        or max(pit_counts) != 21
        or any(
            count <= freeze["search"]["shared_policy_exact_root_threshold"]
            for count in pit_counts
        )
        or freeze["search"]["exact_root_handoff_roots"] != 0
    ):
        raise ValueError("qualification_root_exact_handoff_scope_mismatch")
    search_policy = root / freeze["shared_search_policy"]["source"]
    if sha256(search_policy.read_bytes()) != freeze["shared_search_policy"]["sha256"]:
        raise ValueError("shared_search_policy_hash_mismatch")
    return {"freeze": freeze, "membership": membership}


def reconstruct_gate(evidence: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct #570's gate from paired synthetic or qualification evidence."""
    control = evidence["control"]
    treatment = evidence["treatment"]
    if set(control) != set(treatment) or not control:
        raise ValueError("incomplete_or_unpaired_evidence")
    rows = []
    for key in sorted(control):
        left, right = control[key], treatment[key]
        if left["opening_index"] != right["opening_index"]:
            raise ValueError("paired_opening_mismatch")
        rows.append(
            {
                "opening_index": int(left["opening_index"]),
                "regret_delta": float(right["regret"] - left["regret"]),
                "control_downgrade": bool(left["wdl"] < left["oracle_wdl"]),
                "treatment_downgrade": bool(right["wdl"] < right["oracle_wdl"]),
            }
        )
    opening_values: dict[int, list[float]] = {}
    for row in rows:
        opening_values.setdefault(row["opening_index"], []).append(row["regret_delta"])
    clusters = np.asarray(
        [np.mean(values) for _, values in sorted(opening_values.items())],
        dtype=np.float64,
    )
    rng = np.random.default_rng(570)
    draws = rng.integers(0, len(clusters), size=(20_000, len(clusters)))
    means = clusters[draws].mean(axis=1)
    mean_delta = float(clusters.mean())
    interval = np.quantile(means, [0.025, 0.975], method="linear").tolist()
    downgrade_rate = float(np.mean([row["treatment_downgrade"] for row in rows]))
    control_downgrade = float(np.mean([row["control_downgrade"] for row in rows]))
    advances = (
        mean_delta <= -0.10
        and interval[1] < 0.0
        and downgrade_rate <= control_downgrade
    )
    return {
        "mean_treatment_minus_control_regret": mean_delta,
        "opening_bootstrap_95": interval,
        "bootstrap_resamples": 20_000,
        "bootstrap_seed": 570,
        "quantile_method": "linear",
        "treatment_wdl_downgrade_rate": downgrade_rate,
        "control_wdl_downgrade_rate": control_downgrade,
        "decision": "advance_to_separate_public_exploratory_strength_screen"
        if advances
        else "do_not_advance",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def evaluate(
    root: Path,
    out: Path,
    authorization: Path,
    trusted_authorization_sha256: str,
) -> dict[str, Any]:
    """Run only under a separate reviewed authorization; output is create-only."""
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"evaluation_output_already_exists:{out}")
    if not out.parent.is_dir():
        raise FileNotFoundError(f"evaluation_output_parent_missing:{out.parent}")
    verified = verify_freeze(root)
    freeze = verified["freeze"]
    if not authorization.is_file():
        raise PermissionError("post_review_authorization_missing")
    freeze_bytes = (
        root / "docs/data/seed512-policy-strength-evaluation/freeze.json"
    ).read_bytes()
    authorization_bytes = authorization.read_bytes()
    verify_authorization(
        freeze_bytes, authorization_bytes, trusted_authorization_sha256
    )
    configure_evaluation_runtime()

    data = root / "docs/data/seed512-policy-strength-evaluation"
    membership = verified["membership"]
    oracle_path = root / freeze["cohort"]["oracle_records"]
    oracles = {row["state_hash"]: row for row in _read_jsonl(oracle_path)}
    artifact_root = data / "runtime-artifacts"
    evaluators = {
        name: arena.ArtifactEvaluator(artifact_root / name)
        for name in ("P-E4", "T512", "initializer")
    }
    results: dict[str, Any] = {"control": {}, "treatment": {}}
    for item in membership["roots"]:
        key = item["state_hash"]
        oracle = oracles.get(key)
        if oracle is None or oracle.get("coverage_complete") is not True:
            raise ValueError(f"missing_oracle_before_search:{key}")
        margins = {
            int(move): int(value) for move, value in oracle["action_margins"].items()
        }
        if set(margins) != set(item["legal_moves"]):
            raise ValueError(f"oracle_action_coverage_mismatch:{key}")
        seed = int(search_seed(canonical_hash(item["state"])))
        for arm, policy_name in (("control", "P-E4"), ("treatment", "T512")):
            composed = arena.ComposedArtifactEvaluator(
                evaluators[policy_name],
                evaluators["initializer"],
                policy_source="current",
                value_source="candidate",
            )
            result = arena.evaluate_artifact_position(
                evaluator=composed,
                state=item["state"],
                simulations=128,
                seed=seed,
                c_puct=1.25,
                search_options=SEARCH_OPTIONS,
                exact_root_solve_threshold=None,
                exact_solve_stone_threshold=None,
            )
            action = int(result["selected_move"])
            best = max(margins.values())
            margin = margins[action]
            results[arm][key] = {
                "opening_index": int(item["opening_index"]),
                "seed": seed,
                "selected_action": action,
                "search_witness": {
                    "selected_action": action,
                    "legal_moves": [int(move) for move in result["legal_moves"]],
                    "visit_counts": [int(value) for value in result["visits"]],
                    "child_stats": [
                        {
                            "move": int(item["move"]),
                            "visits": int(item["visits"]),
                            "q_value": float(item["q_value"]),
                        }
                        for item in result["child_stats"]
                    ],
                    "network_priors": [float(value) for value in result["policy"]],
                },
                "exact_final_margin": margin,
                "regret": best - margin,
                "wdl": (margin > 0) - (margin < 0),
                "oracle_wdl": (best > 0) - (best < 0),
            }
    if (
        set(results["control"]) != set(results["treatment"])
        or len(results["control"]) != 423
    ):
        raise ValueError("incomplete_pairs_no_result")
    result = {
        "schema": "seed512-policy-strength-evidence-v1",
        "freeze_sha256": sha256((data / "freeze.json").read_bytes()),
        "authorization_sha256": sha256(authorization_bytes),
        "membership_sha256": freeze["cohort"]["membership_sha256"],
        "artifact_hashes": ARTIFACT_HASHES,
        "pairs": results,
        "gate": reconstruct_gate(results),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--trusted-authorization-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.execute:
        if (
            args.authorization is None
            or args.output is None
            or args.trusted_authorization_sha256 is None
        ):
            parser.error(
                "--execute requires --authorization, --trusted-authorization-sha256, and --output"
            )
        evaluate(
            args.root,
            args.output,
            args.authorization,
            args.trusted_authorization_sha256,
        )
        return
    checked = verify_freeze(args.root)
    print(
        json.dumps(
            {
                "status": "freeze_verified",
                "roots": checked["membership"]["population"],
                "qualification_started": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
