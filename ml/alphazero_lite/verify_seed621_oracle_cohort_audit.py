"""Independent structural verifier for the published Seed 621 audit package."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed621-oracle-cohort-audit"
MEMBERSHIP_EVIDENCE = {
    "B512": (
        ROOT / "docs/data/seed599-replay-coverage/treatment-membership.json",
        "95ecc180e28945702e08815de7a5210e1bce036afdaf76d194bbd774589a4516",
    ),
    "S615": (
        ROOT / "docs/data/seed615-source-opening-diversity/candidate-membership.jsonl",
        "5ed236266ed3804a9c0e9f75b7e10b7c81ea6e290f8c72d3cea06dd225a15010",
    ),
    "protected_holdout": (
        ROOT / "docs/data/seed599-replay-coverage/heldout-membership.json",
        "4e74d57244ab55b69751c38e4097860c25e541a8db60a2109f2ebd64c3eb81a8",
    ),
}
CHECKPOINTS = {
    "B512": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
    "S615": "cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780",
}
EXPECTED_SOURCE_HASHES = {
    "docs/data/seed556-exploratory-arena/screen-0005/verification-receipt.json": "a318933da72b8ed23cb5487fbbd8e57edd0d1d407ba9bc1f8485b0736fcdfbe7",
    "docs/data/seed557-retrospective-diagnostic/cohort.json": "f62539cc3e1b1843a2bd9296453d51b895a1c5e693bd9edc3b3291f883398243",
    "docs/data/seed557-retrospective-diagnostic/execution-freeze.json": "b0892a24872012486ff164d291913e1799737632a816a40c2547210a458bf36d",
    "docs/data/seed557-retrospective-diagnostic/oracle-identity-verification.json": "45f93ce95bf2bc0c58b8a44bdf072324eddfb2c943dd9678e6895613a5a4f60d",
    "docs/data/seed557-retrospective-diagnostic/oracle-records.jsonl": "75453e744a86b9dc8471628e0d133fa7d478550af9c8089ed8661426436a8480",
    "docs/data/seed559-fixed-position-feasibility/run-0003/cohort-manifest.json": "f2e03fb9491474056e7a7cb3ace225cd2310f9069c9e48600d45039ed63158fb",
    "docs/data/seed559-fixed-position-feasibility/run-0003/execution-freeze.json": "3b1ab386c33534130c7e15df024f21ef75359ccd00d453e4c6aea104c8181927",
    "docs/data/seed559-fixed-position-feasibility/run-0003/oracle-records.jsonl": "83bec1bb348cbc6d1183112363f27d0612ee204a48d14ee5dbb7530e9b5fb8c2",
    "docs/data/seed615-source-opening-diversity/evaluation-preparation/execution-freeze-v4.json": "ba30f5e3efadf3358db9f03c7b4eba4f69b23e2f95dcb4089723a1fb9664f32b",
    "docs/data/seed615-source-opening-diversity/training-run-S615/training-integrity-report.json": "c383411a9ce9b896ef49431e502e0047b26de0e8600042d8b2873ae07c796266",
    "docs/data/seed615-source-opening-diversity/training-run-S615/training-receipt.json": "0dc4d04a6be2b638abf77bb959e93d82c60355f3f9a449a49d0fef0a8bbb81f3",
}
SEED615_FREEZE_SHA256 = (
    "ba30f5e3efadf3358db9f03c7b4eba4f69b23e2f95dcb4089723a1fb9664f32b"
)
RUNTIME_HASHES = {
    "B512": {
        "model.npz": CHECKPOINTS["B512"],
        "metadata.json": "510d4d5056d1b68ddd83eed89a4e8da13e9305d2fa39a46eefcebcf13334dce6",
        "weights.json": "d3effd234fe657fd8be3e3896153441d75ba338e83e83b66d7541936b3c01ccd",
    },
    "S615": {
        "model.npz": CHECKPOINTS["S615"],
        "metadata.json": "24e42998824856d0242e04ff8a98431b139375e0bc4540d759373ccb335983b3",
        "weights.json": "8f36140461390d317e7e5bcf604357ca65fa1b605bd1e9562bf923b840e13433",
    },
}


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha(rows: Any) -> str:
    return sha(
        json.dumps(
            rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    )


def canonical_state_hash(state: dict[str, Any]) -> str:
    return sha(json.dumps(state, sort_keys=True, separators=(",", ":")).encode())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _authenticate_inputs(
    report: dict[str, Any],
) -> tuple[dict[str, set[str]], dict[str, ArtifactEvaluator]]:
    if report["source_artifacts_sha256"] != EXPECTED_SOURCE_HASHES:
        raise ValueError("published_source_hash_map_changed")
    for relative, expected in EXPECTED_SOURCE_HASHES.items():
        if sha((ROOT / relative).read_bytes()) != expected:
            raise ValueError(f"source_artifact_digest_mismatch:{relative}")

    observed_memberships: dict[str, set[str]] = {}
    membership_hashes: dict[str, str] = {}
    for name, (path, expected_hash) in MEMBERSHIP_EVIDENCE.items():
        raw = path.read_bytes()
        rows = (
            [json.loads(line) for line in raw.splitlines() if line]
            if path.suffix == ".jsonl"
            else json.loads(raw)
        )
        digest = canonical_sha(rows)
        if digest != expected_hash:
            raise ValueError(f"membership_source_digest_mismatch:{name}")
        identities: set[str] = set()
        for row in rows:
            identity = row.get("canonical_identity")
            state = row.get("canonical_state")
            if not identity or (
                state is not None and canonical_state_hash(state) != identity
            ):
                raise ValueError(f"membership_state_identity_mismatch:{name}")
            identities.add(identity)
        observed_memberships[name] = identities
        membership_hashes[name] = digest
    if report["membership_sha256"] != membership_hashes:
        raise ValueError("published_membership_hash_mismatch")

    freeze_path = (
        ROOT
        / "docs/data/seed615-source-opening-diversity/evaluation-preparation/execution-freeze-v4.json"
    )
    if sha(freeze_path.read_bytes()) != SEED615_FREEZE_SHA256:
        raise ValueError("seed615_evaluation_freeze_hash_mismatch")
    freeze = json.loads(freeze_path.read_bytes())
    artifact_root = freeze_path.parent / "runtime-artifacts"
    evaluators: dict[str, ArtifactEvaluator] = {}
    for name in CHECKPOINTS:
        directory = artifact_root / name
        observed = {
            filename: sha((directory / filename).read_bytes())
            for filename in RUNTIME_HASHES[name]
        }
        if (
            observed != RUNTIME_HASHES[name]
            or observed != report["runtime_artifact_sha256"][name]
        ):
            raise ValueError(f"runtime_artifact_identity_mismatch:{name}")
        expected_freeze = freeze["runtime_artifacts"][name]
        if (
            observed["model.npz"] != expected_freeze["model_npz_sha256"]
            or observed["metadata.json"] != expected_freeze["metadata_sha256"]
            or observed["weights.json"] != expected_freeze["weights_json_sha256"]
        ):
            raise ValueError(f"runtime_artifact_freeze_mismatch:{name}")
        evaluators[name] = ArtifactEvaluator(directory)
    if report["checkpoint_sha256"] != CHECKPOINTS:
        raise ValueError("published_checkpoint_hashes_changed")
    return observed_memberships, evaluators


def _source_inventory(
    membership: dict[str, set[str]], evaluators: dict[str, ArtifactEvaluator]
) -> list[dict[str, Any]]:
    roots: dict[str, dict[str, Any]] = {}
    oracles: dict[str, dict[str, Any]] = {}
    provenance: dict[str, dict[str, list]] = {}
    source_specs = (
        (
            "seed557",
            ROOT / "docs/data/seed557-retrospective-diagnostic/cohort.json",
            ROOT / "docs/data/seed557-retrospective-diagnostic/oracle-records.jsonl",
        ),
        (
            "seed559",
            ROOT
            / "docs/data/seed559-fixed-position-feasibility/run-0003/cohort-manifest.json",
            ROOT
            / "docs/data/seed559-fixed-position-feasibility/run-0003/oracle-records.jsonl",
        ),
    )
    source_record_count = 0
    for cohort, cohort_path, oracle_path in source_specs:
        cohort_rows = json.loads(cohort_path.read_bytes())["states"]
        oracle_rows = _load_jsonl(oracle_path)
        if len(cohort_rows) != len(oracle_rows):
            raise ValueError(f"{cohort}_oracle_row_count_mismatch")
        cohort_by_id = {row["state_hash"]: row for row in cohort_rows}
        oracle_by_id = {row["state_hash"]: row for row in oracle_rows}
        if (
            len(cohort_by_id) != len(cohort_rows)
            or len(oracle_by_id) != len(oracle_rows)
            or set(cohort_by_id) != set(oracle_by_id)
        ):
            raise ValueError(f"{cohort}_source_identity_join_mismatch")
        source_record_count += len(cohort_rows)
        for identity, root in cohort_by_id.items():
            state = root["state"]
            if canonical_state_hash(state) != identity:
                raise ValueError(f"{cohort}_canonical_state_hash_mismatch")
            game = KalahGame.from_state(state)
            legal = game.possible_moves()
            oracle = oracle_by_id[identity]
            margins = {
                int(action): int(value)
                for action, value in oracle["action_margins"].items()
            }
            if (
                oracle.get("coverage_complete") is not True
                or set(margins) != set(legal)
                or set(oracle["legal_moves"]) != set(legal)
            ):
                raise ValueError(f"{cohort}_oracle_legal_coverage_mismatch:{identity}")
            if game.current_player != state["current_player"]:
                raise ValueError(
                    f"{cohort}_current_player_orientation_mismatch:{identity}"
                )
            if (
                identity in oracles
                and oracles[identity]["action_margins"] != oracle["action_margins"]
            ):
                raise ValueError(f"duplicate_state_oracle_disagreement:{identity}")
            roots.setdefault(identity, root)
            oracles.setdefault(identity, oracle)
            provenance.setdefault(identity, {"cohorts": [], "sources": []})
            provenance[identity]["cohorts"].append(cohort)
            provenance[identity]["sources"].append(
                {
                    "opening_index": root.get("opening_index"),
                    "source_path": root.get("source_path"),
                    "source_line": root.get("source_line"),
                    "source_record_sha256": root.get("source_record_sha256"),
                    "source_batch_sha256": root.get("source_batch_sha256"),
                }
            )
    if source_record_count != 768:
        raise ValueError("authenticated_source_record_count_mismatch")

    rebuilt: list[dict[str, Any]] = []
    for identity in sorted(roots):
        root = roots[identity]
        state = root["state"]
        game = KalahGame.from_state(state)
        legal = game.possible_moves()
        reasons = []
        if identity in membership["B512"]:
            reasons.append("B512_training_state_overlap")
        if identity in membership["S615"]:
            reasons.append("S615_training_state_overlap")
        if identity in membership["protected_holdout"]:
            reasons.append("protected_holdout_overlap")

        # Re-run deterministic model inference independently of the publisher.
        policy_vectors = {}
        for name in ("B512", "S615"):
            prior, _unused_value = evaluators[name].evaluate(game)
            vector = np.asarray(prior, dtype=np.float64).copy()
            mask = np.zeros(6, dtype=bool)
            mask[legal] = True
            vector[~mask] = 0.0
            if not np.isfinite(vector).all() or (vector < 0).any() or vector.sum() <= 0:
                raise ValueError(f"invalid_{name}_legal_prior:{identity}")
            vector /= vector.sum()
            policy_vectors[name] = vector
        b, s = policy_vectors["B512"], policy_vectors["S615"]
        b_top, s_top = int(np.argmax(b)), int(np.argmax(s))
        midpoint = (b + s) / 2.0
        b_positive, s_positive = b > 0, s > 0
        js = 0.5 * float(
            np.sum(b[b_positive] * np.log2(b[b_positive] / midpoint[b_positive]))
        )
        js += 0.5 * float(
            np.sum(s[s_positive] * np.log2(s[s_positive] / midpoint[s_positive]))
        )
        b_entropy = -float(np.sum(b[b > 0] * np.log2(b[b > 0])))
        s_entropy = -float(np.sum(s[s > 0] * np.log2(s[s > 0])))
        record = {
            "state_id": identity,
            "state": state,
            "player_to_move": game.current_player,
            "legal_moves": legal,
            "exact_action_margins_to_move_perspective": oracles[identity][
                "action_margins"
            ],
            "cohorts": provenance[identity]["cohorts"],
            "source_clusters": provenance[identity]["sources"],
            "eligible_by_exact_membership_rule": not reasons,
            "exclusion_reasons": reasons,
            "contamination_status": "excluded_exact_membership"
            if reasons
            else "unresolved_nonmembership_not_proven",
            "B512_legal_probabilities": b.tolist(),
            "S615_legal_probabilities": s.tolist(),
            "top_prior_agreement": b_top == s_top,
            "B512_top_action": b_top,
            "S615_top_action": s_top,
            "B512_preferred_action_probability": float(b[b_top]),
            "S615_preferred_action_probability": float(s[s_top]),
            "B512_mass_on_S615_preferred_action": float(b[s_top]),
            "S615_mass_on_B512_preferred_action": float(s[b_top]),
            "total_variation": 0.5 * float(np.abs(b - s).sum()),
            "jensen_shannon_bits": js,
            "B512_entropy_bits": b_entropy,
            "S615_entropy_bits": s_entropy,
            "legal_action_count": len(legal),
        }
        rebuilt.append(record)
    return rebuilt


def assert_inventory_matches(
    published: list[dict[str, Any]], rebuilt: list[dict[str, Any]]
) -> None:
    published_ids = [row.get("state_id") for row in published]
    rebuilt_ids = [row["state_id"] for row in rebuilt]
    if len(published_ids) != len(set(published_ids)):
        raise ValueError("duplicate_published_inventory_state")
    if published_ids != rebuilt_ids:
        raise ValueError("missing_substituted_or_reordered_inventory_state")
    for index, (actual, expected) in enumerate(zip(published, rebuilt, strict=True)):
        for key in expected:
            if key not in actual or not _equivalent(actual[key], expected[key]):
                raise ValueError(f"reconstructed_inventory_mismatch:{index}:{key}")
        if set(actual) != set(expected):
            raise ValueError(f"inventory_schema_mismatch:{index}")


def _equivalent(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return isinstance(actual, (float, int)) and math.isclose(
            float(actual), expected, rel_tol=1e-11, abs_tol=1e-12
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_equivalent(a, e) for a, e in zip(actual, expected, strict=True))
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_equivalent(actual[key], expected[key]) for key in expected)
        )
    return actual == expected


def verify() -> dict[str, Any]:
    report = json.loads((OUT / "report.json").read_bytes())
    if (
        report.get("schema") != "seed621-oracle-cohort-audit-v1"
        or report.get("read_only") is not True
    ):
        raise ValueError("report_schema_or_execution_scope_mismatch")
    if report.get("source_commits") != {
        "seed557_and_seed559_reconstruction": "f38838e6ffe20a44c1e8c161a9daed0f1243f0ae"
    }:
        raise ValueError("published_source_commit_binding_mismatch")
    inventory_payload = (OUT / "eligible-state-inventory.jsonl").read_bytes()
    if sha(inventory_payload) != report["inventory_sha256"]:
        raise ValueError("inventory_digest_mismatch")
    published = [json.loads(line) for line in inventory_payload.splitlines() if line]
    membership, evaluators = _authenticate_inputs(report)
    rebuilt = _source_inventory(membership, evaluators)
    assert_inventory_matches(published, rebuilt)

    strata = Counter()
    exclusions = Counter()
    cohort_metrics: dict[str, list[dict[str, Any]]] = {}
    for row in rebuilt:
        if row["eligible_by_exact_membership_rule"]:
            stratum = "agreement" if row["top_prior_agreement"] else "disagreement"
            strata[stratum] += 1
            for cohort in row["cohorts"]:
                cohort_metrics.setdefault(cohort, []).append(row)
        exclusions.update(row["exclusion_reasons"])
    expected_counts = {
        "source_records": 768,
        "unique_states": len(rebuilt),
        "duplicated_across_cohorts": 0,
        "exact_covered": len(rebuilt),
        "exact_covered_excluded_by_authenticated_membership": sum(
            bool(row["exclusion_reasons"]) for row in rebuilt
        ),
        "eligible_by_exact-membership-rule": sum(strata.values()),
        "unresolved_nonmembership": sum(strata.values()),
        "agreement": strata["agreement"],
        "disagreement": strata["disagreement"],
        "exclusion_reasons": dict(exclusions),
    }
    if report["counts"] != expected_counts:
        raise ValueError("aggregate_count_reconstruction_mismatch")
    expected_metrics = {
        cohort: {
            "n": len(rows),
            "agreements": sum(row["top_prior_agreement"] for row in rows),
            "disagreements": sum(not row["top_prior_agreement"] for row in rows),
            "mean_tv": float(np.mean([row["total_variation"] for row in rows])),
            "mean_js_bits": float(
                np.mean([row["jensen_shannon_bits"] for row in rows])
            ),
        }
        for cohort, rows in cohort_metrics.items()
    }
    if not _equivalent(report["cohort_policy_metrics"], expected_metrics):
        raise ValueError("aggregate_policy_metric_reconstruction_mismatch")
    expected_gate = {
        "minimum_64_and_16_each_class_met": sum(strata.values()) >= 64
        and strata["agreement"] >= 16
        and strata["disagreement"] >= 16,
        "training_contamination_excluded": False,
        "recommendation": "revise",
    }
    if report["decision_gate"] != expected_gate:
        raise ValueError("decision_gate_reconstruction_mismatch")
    return {
        "status": "verified",
        "unique_states": len(rebuilt),
        "top_prior_agreement": strata["agreement"],
        "top_prior_disagreement": strata["disagreement"],
        "contamination_clearance": "unresolved",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True, indent=2))
