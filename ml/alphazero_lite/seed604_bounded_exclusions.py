"""Authenticate the proposed, bounded Seed 604 evaluation exclusions.

This adapter only verifies identities and checks candidate states. It has no
opening-generation, training, evaluation-game, freeze, or authorization path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import holdback_arena
from ml.alphazero_lite.seed551_controlled_policy_strength import (
    canonical_identity,
    production_float32_identity,
)
from ml.alphazero_lite.seed561_historical_exclusion_inventory import (
    verify as verify_561,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("docs/data/seed604-replay-optimization")
SEED599 = Path("docs/data/seed599-replay-coverage")
SEED561 = Path("docs/data/seed561-historical-exclusion-inventory-v1")
SEED599_TRAINING_FREEZE_SHA256 = (
    "bd141778461d7046b067966d7fb87188e5b0a2630d82655a321059b4ce17cad9"
)
SEED599_EVALUATION_FREEZE_SHA256 = (
    "467cb1ecf4acb47013fce9cd9418b3ac6b1d3929752713e9bd029fceabbcea11"
)
SEED599_EVALUATION_CONTRACT_SHA256 = (
    "8d29200a27ae87daa7ea9de9a36ebf5a19d1f0f04b4f0237ff48ba1ab1af7bf4"
)
SEED599_OPENINGS_SHA256 = (
    "3acc395a59d90df57ddf60aa1e9d99aabde220901434f72239eb48a4047bc54a"
)
SEED599_CONFIG_SHA256 = {
    "A": "413a8a2b128391580a505dfa71e5772e44fd3c8c955577558ffdf92324f4f41a",
    "B": "8dd20557ac93d7d3c72bb94b50d35e868e85ee6f68bdd84699ee312b54e26405",
}
SEED599_MEMBERSHIP_FILE_SHA256 = {
    "A": "5f066de2d3c0e988aec137a8c64d66d89ea66562bc2d011da2cd6319fed21135",
    "B": "aa1139a4bd7ed31e35ea7870a0f9c2571db46b4b9fe79ae0f2ff10622e0668d2",
}
SEED599_HELDOUT_FILE_SHA256 = (
    "c1de9881bd8f997254908c188f00165c33d029f477580e0b8b75868cde53edc2"
)
SEED599_PREPARATION_SHA256 = (
    "11c580d2ebb8d4a8fc67b12f7d1f8e6d8bcbffa0d110e6549bf077d259ff751d"
)
SEED599_HELDOUT_IDENTITY_SHA256 = (
    "4e74d57244ab55b69751c38e4097860c25e541a8db60a2109f2ebd64c3eb81a8"
)
SEED561_MANIFEST_SHA256 = (
    "946fc52c75fa2e6c0499811fcb97ff773ea00ec6ec9b6d1bfb3ec1c2466a95ad"
)
SEED561_INDEX_SHA256 = {
    "canonical": "0bbf129eca3f747375eb3bb81417b6b698fa2c733ce1ffcd4431b89d24541f3f",
    "kalah_v3_float32": "d18d45a0caa028ba956bba78c87ea68c0cfb6378ef93cb303731f6be1f9ce5e3",
}
REQUIRED_SEED599_POPULATIONS = {
    "seed599_training_A",
    "seed599_training_B",
    "seed599_heldout",
    "seed599_evaluation_roots",
}
REQUIRED_POLICY_POPULATIONS = REQUIRED_SEED599_POPULATIONS | {
    "seed561_historical_inventory"
}


def sha256_bytes(raw: bytes) -> str:
    """Return the SHA-256 digest for a byte string."""
    return hashlib.sha256(raw).hexdigest()


def identity_set_sha256(identities: set[str]) -> str:
    """Hash a set as sorted lowercase identity hex strings separated by LF."""
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in identities
    ):
        raise ValueError("malformed_identity_set")
    return sha256_bytes("\n".join(sorted(identities)).encode())


def require_complete_population_manifest(
    populations: dict[str, Any], required: set[str] = REQUIRED_POLICY_POPULATIONS
) -> None:
    """Fail closed if a required source population is omitted."""
    missing = required - populations.keys()
    if missing:
        raise ValueError(
            f"required_exclusion_population_missing:{','.join(sorted(missing))}"
        )


def verify_source_bytes(path: Path, expected_sha256: str, label: str) -> bytes:
    """Read a registered source only when its exact file bytes are authenticated."""
    if not path.is_file():
        raise ValueError(f"exclusion_source_missing:{label}")
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"exclusion_source_hash_mismatch:{label}")
    return raw


def verify_identity_union_bindings(
    policy: dict[str, Any], excluded: dict[str, set[str]]
) -> None:
    """Verify supplied identity sets against the recorded set counts and hashes."""
    expected_unions = policy.get("identity_unions")
    if not isinstance(expected_unions, dict):
        raise ValueError("exclusion_identity_union_missing")
    for scheme in ("canonical", "float32"):
        identities = excluded.get(scheme)
        binding = expected_unions.get(scheme)
        if not isinstance(identities, set) or not isinstance(binding, dict):
            raise ValueError(f"exclusion_identity_scheme_missing:{scheme}")
        if len(identities) != binding.get("distinct") or identity_set_sha256(
            identities
        ) != binding.get("identity_set_sha256"):
            raise ValueError(f"exclusion_identity_manifest_mismatch:{scheme}")


def validate_candidate_identities(
    canonical: str,
    float32: str,
    excluded: dict[str, set[str]],
) -> None:
    """Reject candidate roots colliding in either identity scheme."""
    if "canonical" not in excluded or "float32" not in excluded:
        raise ValueError("exclusion_identity_scheme_missing")
    identity_set_sha256({canonical})
    identity_set_sha256({float32})
    if canonical in excluded["canonical"]:
        raise ValueError("candidate_canonical_exclusion_collision")
    if float32 in excluded["float32"]:
        raise ValueError("candidate_float32_exclusion_collision")


def _read_pinned_json(root: Path, relative: str, expected_sha256: str) -> Any:
    path = root / relative
    raw = verify_source_bytes(path, expected_sha256, relative)
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"exclusion_source_invalid_json:{relative}") from error


def _state_identity_sets(
    rows: list[dict[str, Any]], population: str
) -> dict[str, set[str]]:
    canonical: set[str] = set()
    float32: set[str] = set()
    for index, row in enumerate(rows):
        state = row.get("canonical_state", row.get("state"))
        if not isinstance(state, dict):
            raise ValueError(f"exclusion_state_missing:{population}:{index}")
        canonical_id = canonical_identity(state)
        float32_id = production_float32_identity(state)
        if row.get("canonical_identity", canonical_id) != canonical_id:
            raise ValueError(f"canonical_identity_mismatch:{population}:{index}")
        if row.get("float32_identity", float32_id) != float32_id:
            raise ValueError(f"float32_identity_mismatch:{population}:{index}")
        canonical.add(canonical_id)
        float32.add(float32_id)
    if len(canonical) != len(rows) or len(float32) != len(rows):
        raise ValueError(f"duplicate_identity_within_population:{population}")
    return {"canonical": canonical, "float32": float32}


def build_bounded_policy(root: Path = ROOT) -> dict[str, Any]:
    """Authenticate registered inputs and report the proposed bounded union.

    The Seed 561 verifier is read-only and must pass before its identity indexes
    are accepted. No openings are generated or written.
    """
    root = root.resolve()
    training_freeze = _read_pinned_json(
        root,
        str(SEED599 / "training-execution-freeze.json"),
        SEED599_TRAINING_FREEZE_SHA256,
    )
    evaluation_freeze = _read_pinned_json(
        root,
        str(SEED599 / "evaluation-readiness/execution-freeze.json"),
        SEED599_EVALUATION_FREEZE_SHA256,
    )
    evaluation_contract = _read_pinned_json(
        root,
        str(SEED599 / "evaluation-contract.json"),
        SEED599_EVALUATION_CONTRACT_SHA256,
    )
    preparation = _read_pinned_json(
        root, str(SEED599 / "preparation.json"), SEED599_PREPARATION_SHA256
    )
    if (
        training_freeze.get("schema") != "seed599-training-execution-freeze-v1"
        or training_freeze.get("status") != "frozen-training-not-authorized"
    ):
        raise ValueError("seed599_training_freeze_status_mismatch")
    if (
        evaluation_freeze.get("schema") != "seed599-evaluation-execution-freeze-v1"
        or evaluation_freeze.get("execution_authorized") is not False
        or evaluation_freeze.get("registered_contract_sha256")
        != SEED599_EVALUATION_CONTRACT_SHA256
        or evaluation_freeze.get("registered_openings_identity")
        != "6ed3c99b35c6f51d03ea59ee2c9ed1c5fb52cf76a51b4cccd50f74ba8f962962"
        or evaluation_freeze.get("training_execution_freeze_sha256")
        != SEED599_TRAINING_FREEZE_SHA256
    ):
        raise ValueError("seed599_evaluation_freeze_binding_mismatch")

    populations: dict[str, dict[str, Any]] = {}
    identity_sets: dict[str, dict[str, set[str]]] = {}
    for arm in ("A", "B"):
        config_path = SEED599 / f"training-{arm}.json"
        config = _read_pinned_json(root, str(config_path), SEED599_CONFIG_SHA256[arm])
        if (
            training_freeze.get("arms", {}).get(arm, {}).get("configuration_sha256")
            != (SEED599_CONFIG_SHA256[arm])
        ):
            raise ValueError(f"seed599_training_freeze_binding_mismatch:{arm}")
        membership_name = (
            "control-membership.json" if arm == "A" else "treatment-membership.json"
        )
        membership_relative = str(SEED599 / membership_name)
        raw = (root / membership_relative).read_bytes()
        if sha256_bytes(raw) != SEED599_MEMBERSHIP_FILE_SHA256[arm]:
            raise ValueError(f"exclusion_source_hash_mismatch:{membership_relative}")
        rows = json.loads(raw)
        if (
            len(rows) != config["states"]
            or sha256_bytes(
                json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
            )
            != config["membership_sha256"]
        ):
            raise ValueError(f"seed599_membership_binding_mismatch:{arm}")
        name = f"seed599_training_{arm}"
        identity_sets[name] = _state_identity_sets(rows, name)
        populations[name] = {
            "path": membership_relative,
            "source_commit": "a5557fe4890b919b8f4fa5c009070e02de95085b",
            "registration_path": str(config_path),
            "registration_sha256": SEED599_CONFIG_SHA256[arm],
            "format": "JSON array of canonical_state training rows",
            "extraction_rule": "recompute canonical and production kalah_v3 float32 identities from each canonical_state",
            "file_sha256": SEED599_MEMBERSHIP_FILE_SHA256[arm],
            "records": len(rows),
            "canonical_distinct": len(identity_sets[name]["canonical"]),
            "canonical_identity_set_sha256": identity_set_sha256(
                identity_sets[name]["canonical"]
            ),
            "float32_distinct": len(identity_sets[name]["float32"]),
            "float32_identity_set_sha256": identity_set_sha256(
                identity_sets[name]["float32"]
            ),
            "verification_status": "passed",
        }

    heldout_relative = str(SEED599 / "heldout-membership.json")
    heldout_raw = (root / heldout_relative).read_bytes()
    if sha256_bytes(heldout_raw) != SEED599_HELDOUT_FILE_SHA256:
        raise ValueError("exclusion_source_hash_mismatch:seed599_heldout")
    heldout_rows = json.loads(heldout_raw)
    heldout_semantic_sha = sha256_bytes(
        json.dumps(heldout_rows, sort_keys=True, separators=(",", ":")).encode()
    )
    if (
        heldout_semantic_sha != SEED599_HELDOUT_IDENTITY_SHA256
        or preparation["heldout"]["membership_manifest"]["sha256"]
        != SEED599_HELDOUT_IDENTITY_SHA256
    ):
        raise ValueError("seed599_heldout_binding_mismatch")
    identity_sets["seed599_heldout"] = _state_identity_sets(
        heldout_rows, "seed599_heldout"
    )
    populations["seed599_heldout"] = {
        "path": heldout_relative,
        "source_commit": "a5557fe4890b919b8f4fa5c009070e02de95085b",
        "registration_path": str(SEED599 / "preparation.json"),
        "registration_sha256": SEED599_PREPARATION_SHA256,
        "format": "JSON array of provenance-held-out training examples",
        "extraction_rule": "recompute canonical and production kalah_v3 float32 identities from each canonical_state",
        "file_sha256": SEED599_HELDOUT_FILE_SHA256,
        "records": len(heldout_rows),
        "canonical_distinct": len(identity_sets["seed599_heldout"]["canonical"]),
        "canonical_identity_set_sha256": identity_set_sha256(
            identity_sets["seed599_heldout"]["canonical"]
        ),
        "float32_distinct": len(identity_sets["seed599_heldout"]["float32"]),
        "float32_identity_set_sha256": identity_set_sha256(
            identity_sets["seed599_heldout"]["float32"]
        ),
        "verification_status": "passed",
    }

    openings_relative = str(SEED599 / "evaluation-openings.json")
    openings_raw = (root / openings_relative).read_bytes()
    if sha256_bytes(openings_raw) != SEED599_OPENINGS_SHA256:
        raise ValueError("seed599_evaluation_openings_file_hash_mismatch")
    openings = json.loads(openings_raw)
    openings_semantic_sha = sha256_bytes(
        json.dumps(openings, sort_keys=True, separators=(",", ":")).encode()
    )
    if (
        len(openings) != 1024
        or evaluation_contract.get("opening_count") != 1024
        or evaluation_contract.get("opening_manifest_sha256") != openings_semantic_sha
    ):
        raise ValueError("seed599_evaluation_openings_binding_mismatch")
    identity_sets["seed599_evaluation_roots"] = _state_identity_sets(
        openings, "seed599_evaluation_roots"
    )
    populations["seed599_evaluation_roots"] = {
        "path": openings_relative,
        "source_commit": "7f3f9d4c53877602b15939a766a3a04a60081ee4",
        "registration_path": str(SEED599 / "evaluation-contract.json"),
        "registration_sha256": SEED599_EVALUATION_CONTRACT_SHA256,
        "format": "JSON array of opening records with state and legal prefix replay identity",
        "extraction_rule": "recompute canonical and production kalah_v3 float32 identities from each state; verify declared identity fields",
        "file_sha256": SEED599_OPENINGS_SHA256,
        "records": len(openings),
        "canonical_distinct": len(
            identity_sets["seed599_evaluation_roots"]["canonical"]
        ),
        "canonical_identity_set_sha256": identity_set_sha256(
            identity_sets["seed599_evaluation_roots"]["canonical"]
        ),
        "float32_distinct": len(identity_sets["seed599_evaluation_roots"]["float32"]),
        "float32_identity_set_sha256": identity_set_sha256(
            identity_sets["seed599_evaluation_roots"]["float32"]
        ),
        "verification_status": "passed",
    }
    require_complete_population_manifest(populations, REQUIRED_SEED599_POPULATIONS)

    inventory_path = root / SEED561
    verify_561(root, inventory_path, write=False)
    inventory_manifest_path = inventory_path / "source-manifest.json"
    inventory_manifest_raw = inventory_manifest_path.read_bytes()
    if sha256_bytes(inventory_manifest_raw) != SEED561_MANIFEST_SHA256:
        raise ValueError("seed561_inventory_manifest_hash_mismatch")
    inventory_manifest = json.loads(inventory_manifest_raw)
    if (
        inventory_manifest.get("anchor_commit")
        != "0139cbd70d70184fc86cabd92afafdd849f28ef0"
    ):
        raise ValueError("seed561_inventory_anchor_mismatch")
    reference = {
        "schema": "seed561_holdback_inventory_reference_v1",
        "manifest_path": "source-manifest.json",
        "manifest_sha256": SEED561_MANIFEST_SHA256,
    }
    historical = holdback_arena.load_pinned_identity_indexes(
        inventory_path, reference, SEED561_MANIFEST_SHA256
    )
    for scheme, expected in SEED561_INDEX_SHA256.items():
        path = inventory_path / inventory_manifest["indexes"][scheme]["path"]
        if sha256_bytes(path.read_bytes()) != expected:
            raise ValueError(f"seed561_inventory_index_hash_mismatch:{scheme}")
    identity_sets["seed561_historical_inventory"] = historical
    populations["seed561_historical_inventory"] = {
        "path": str(SEED561),
        "source_commit": "2d6c5092e0947ffad59bd631ae7eafbb40eef6e7",
        "registration_path": str(SEED561 / "source-manifest.json"),
        "registration_sha256": SEED561_MANIFEST_SHA256,
        "format": "published seed561 canonical and kalah_v3_float32 JSONL gzip indexes",
        "extraction_rule": "load both externally pinned distinct identity indexes after independent Seed 561 verifier; preserve separate scheme namespaces",
        "file_sha256": SEED561_MANIFEST_SHA256,
        "canonical_index_sha256": SEED561_INDEX_SHA256["canonical"],
        "float32_index_sha256": SEED561_INDEX_SHA256["kalah_v3_float32"],
        "records": inventory_manifest["indexes"]["canonical"]["occurrence_rows"],
        "canonical_distinct": len(historical["canonical"]),
        "canonical_identity_set_sha256": identity_set_sha256(historical["canonical"]),
        "float32_distinct": len(historical["float32"]),
        "float32_identity_set_sha256": identity_set_sha256(historical["float32"]),
        "verification_status": "passed",
    }
    require_complete_population_manifest(populations)

    union = {
        scheme: set().union(*(ids[scheme] for ids in identity_sets.values()))
        for scheme in ("canonical", "float32")
    }
    seed599_union = {
        scheme: set().union(
            *(
                identity_sets[name][scheme]
                for name in sorted(REQUIRED_SEED599_POPULATIONS)
            )
        )
        for scheme in ("canonical", "float32")
    }
    intersections = {
        population: {
            scheme: len(ids[scheme] & historical[scheme])
            for scheme in ("canonical", "float32")
        }
        for population, ids in identity_sets.items()
        if population != "seed561_historical_inventory"
    }
    return {
        "schema": "seed604-bounded-exclusion-policy-v1",
        "status": "proposed-awaiting-project-owner-review",
        "seed604_registration_sha256": sha256_bytes(
            (root / DATA / "registration.json").read_bytes()
        ),
        "boundary": "all authenticated Seed 599 A/B training membership states, Seed 599 held-out states, all 1,024 Seed 599 evaluation opening roots, and the full authenticated Seed 561 bounded historical identity inventory",
        "populations": populations,
        "required_population_names": sorted(REQUIRED_SEED599_POPULATIONS),
        "identity_set_rule": "candidate roots must be disjoint from the union in their matching canonical and production kalah_v3 float32 identity scheme",
        "identity_unions": {
            scheme: {
                "distinct": len(identities),
                "identity_set_sha256": identity_set_sha256(identities),
            }
            for scheme, identities in union.items()
        },
        "seed599_to_seed561_intersections": intersections,
        "known_overlaps_within_included_seed599_populations": {
            scheme: {
                "A_with_B": len(
                    identity_sets["seed599_training_A"][scheme]
                    & identity_sets["seed599_training_B"][scheme]
                ),
                "A_with_heldout": len(
                    identity_sets["seed599_training_A"][scheme]
                    & identity_sets["seed599_heldout"][scheme]
                ),
                "B_with_heldout": len(
                    identity_sets["seed599_training_B"][scheme]
                    & identity_sets["seed599_heldout"][scheme]
                ),
                "evaluation_roots_with_A": len(
                    identity_sets["seed599_evaluation_roots"][scheme]
                    & identity_sets["seed599_training_A"][scheme]
                ),
                "evaluation_roots_with_B": len(
                    identity_sets["seed599_evaluation_roots"][scheme]
                    & identity_sets["seed599_training_B"][scheme]
                ),
                "evaluation_roots_with_heldout": len(
                    identity_sets["seed599_evaluation_roots"][scheme]
                    & identity_sets["seed599_heldout"][scheme]
                ),
            }
            for scheme in ("canonical", "float32")
        },
        "seed599_distinct_union_per_scheme": {
            scheme: len(values) for scheme, values in seed599_union.items()
        },
        "seed599_protocol_scope": {
            "training_exclusion": "Seed 599 generator rejects canonical or production-float32 matches against selected treatment B membership; A is nested within B.",
            "evaluation_exclusion": "Seed 599 evaluation generated roots disjoint from selected training states; historical Seed 561 inventory and held-out membership were not registered root-exclusion inputs.",
            "bounded_seed604_additions": "explicitly excludes A, B, held-out states, every Seed 599 evaluation root, and Seed 561 inventory under both identity schemes",
        },
        "seed561_historical_scope": {
            "historical_source_families": [
                "authenticated Seed 482-484 historical snapshots",
                "Seed 488, 499, 505, 508 evaluation roots and replayed trajectory identities",
                "Seed 500 pilot and self-play rows",
                "Seed 562-564 published historical source indexes including Seed 556-560 and Seed 503-504 source families",
            ],
            "source_scope": "exact Seed 511 registration exclusion_source_sha256 list plus authenticated Seed 562-564 subtotal",
            "distinct_per_scheme": len(historical["canonical"]),
            "identity_manifest_sha256": SEED561_MANIFEST_SHA256,
            "limitations": [
                "Seed 461 continuation identities are unresolved and not included.",
                "Other historical source populations outside the Seed 511-pinned scope remain uncovered with unknown counts.",
                "Seed 599 evaluation-game trajectory states are not included; only its 1,024 registered opening roots are added.",
                "This boundary is not a complete historical-independence claim.",
            ],
        },
        "seed461_scientific_treatment": {
            "blocks_bounded_paired_comparison": False,
            "restriction": "The paired B2048-versus-B512 contrast and B2048-versus-P-E4 score may be reported only for this registered cohort and source boundary; no claim of never-before-seen or fully historically independent states is permitted.",
            "reason": "The same frozen cohort, seat-swapped design, common P-E4 evaluator, and preregistered gates still define the bounded contrasts. Unknown Seed 461 continuation overlap leaves cohort novelty unproven and limits external generalization; it must not be imputed by guessing replay semantics.",
        },
        "seed604_experiment_unchanged": {
            "training_intervention": "extend authenticated Seed 599 B schedule from P-E4 with fresh Adam, exact B512 reproduction before update 513",
            "updates": 2048,
            "parent_checkpoint_sha256": "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
            "B512_checkpoint_sha256": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
            "comparison": "B2048 and B512 both versus serving P-E4; common unchanged P-E4 value evaluator",
            "statistical_gates": "unchanged from docs/data/seed604-replay-optimization/registration.json",
        },
        "execution": {
            "evaluation_openings_generated": False,
            "evaluation_execution_freeze_created": False,
            "training_started": False,
            "evaluation_games_started": False,
            "authorization_created": False,
        },
    }


def verify_published_policy(root: Path = ROOT) -> dict[str, Any]:
    """Recompute authenticated inputs and compare them to the proposed record."""
    root = root.resolve()
    expected = build_bounded_policy(root)
    path = root / DATA / "bounded-exclusion-policy.json"
    if not path.is_file():
        raise ValueError("bounded_exclusion_policy_missing")
    try:
        saved = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("bounded_exclusion_policy_invalid_json") from error
    if (
        saved.get("schema") != expected["schema"]
        or saved.get("status") != expected["status"]
        or saved.get("seed604_registration_sha256")
        != expected["seed604_registration_sha256"]
    ):
        raise ValueError("bounded_exclusion_policy_identity_mismatch")
    expected_union = expected["identity_unions"]
    saved_accounting = saved.get("observed_overlap_accounting", {})
    expected_hashes = saved_accounting.get("bounded_union_identity_set_sha256", {})
    if (
        saved_accounting.get("bounded_union_per_scheme")
        != expected_union["canonical"]["distinct"]
        or expected_hashes.get("canonical")
        != expected_union["canonical"]["identity_set_sha256"]
        or expected_hashes.get("production_float32")
        != expected_union["float32"]["identity_set_sha256"]
    ):
        raise ValueError("bounded_exclusion_policy_union_mismatch")
    if (
        saved_accounting.get("distinct_seed599_union_per_scheme")
        != expected["seed599_distinct_union_per_scheme"]
        or saved_accounting.get("seed599_cross_population_overlaps")
        != expected["known_overlaps_within_included_seed599_populations"]
    ):
        raise ValueError("bounded_exclusion_policy_overlap_accounting_mismatch")
    saved_hist_intersections = saved_accounting.get(
        "seed599_population_intersections_with_seed561", {}
    )
    expected_hist_intersections = {
        population.removeprefix("seed599_"): values["canonical"]
        for population, values in expected["seed599_to_seed561_intersections"].items()
    }
    if saved_hist_intersections.get(
        "canonical"
    ) != expected_hist_intersections or saved_hist_intersections.get(
        "production_float32"
    ) != {
        population.removeprefix("seed599_"): values["float32"]
        for population, values in expected["seed599_to_seed561_intersections"].items()
    }:
        raise ValueError("bounded_exclusion_policy_historical_overlap_mismatch")
    saved_sources = saved.get("sources", {})
    require_complete_population_manifest(saved_sources)
    if set(saved_sources) != set(expected["populations"]):
        raise ValueError("bounded_exclusion_policy_population_set_mismatch")
    limitations = saved.get("limitations", {})
    if (
        "unresolved" not in limitations.get("seed461_continuation_states", "")
        or "unknown counts" not in limitations.get("other_historical_populations", "")
        or "not included" not in limitations.get("seed599_evaluation_trajectories", "")
    ):
        raise ValueError("bounded_exclusion_policy_limitations_incomplete")
    execution = saved.get("execution_status", {})
    if any(execution.get(key) is not False for key in execution):
        raise ValueError("bounded_exclusion_policy_execution_status_invalid")
    invariants = saved.get("seed604_experiment_invariants", {})
    if (
        invariants.get("parent_checkpoint_sha256")
        != "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad"
        or invariants.get("B512_checkpoint_sha256")
        != "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c"
        or invariants.get("update_budget") != 2048
        or "unchanged" not in invariants.get("statistical_gates", "")
    ):
        raise ValueError("bounded_exclusion_policy_experiment_invariant_mismatch")
    for population, source in expected["populations"].items():
        saved_source = saved_sources[population]
        saved_path = saved_source.get("directory", saved_source.get("path"))
        saved_registration_path = saved_source.get(
            "registration_path",
            saved_source.get(
                "configuration_path",
                saved_source.get(
                    "evaluation_contract_path", saved_source.get("source_manifest_path")
                ),
            ),
        )
        saved_registration_sha = saved_source.get(
            "registration_sha256",
            saved_source.get(
                "configuration_sha256",
                saved_source.get(
                    "evaluation_contract_sha256",
                    saved_source.get("source_manifest_sha256"),
                ),
            ),
        )
        if (
            saved_path != source["path"]
            or saved_source.get("source_commit") != source["source_commit"]
            or saved_registration_path != source["registration_path"]
            or saved_registration_sha != source["registration_sha256"]
            or not str(saved_source.get("verification_status", "")).startswith("passed")
        ):
            raise ValueError(
                f"bounded_exclusion_policy_provenance_mismatch:{population}"
            )
        for key in (
            "file_sha256",
            "records",
            "canonical_identity_set",
            "production_float32_identity_set",
        ):
            expected_key = {
                "canonical_identity_set": "canonical_identity_set_sha256",
                "production_float32_identity_set": "float32_identity_set_sha256",
            }.get(key)
            if key in {"canonical_identity_set", "production_float32_identity_set"}:
                count_key = (
                    "canonical_distinct"
                    if key == "canonical_identity_set"
                    else "float32_distinct"
                )
                if (
                    saved_source[key].get("distinct") != source[count_key]
                    or saved_source[key].get("sha256") != source[expected_key]
                ):
                    raise ValueError(
                        f"bounded_exclusion_policy_source_mismatch:{population}:{key}"
                    )
            elif saved_source.get(key) != source.get(key):
                raise ValueError(
                    f"bounded_exclusion_policy_source_mismatch:{population}:{key}"
                )
    return {
        "passed": True,
        "status": expected["status"],
        "identity_unions": expected_union,
        "evaluation_openings_generated": False,
        "training_or_games_started": False,
    }


def verify_candidate_against_bounded_policy(
    canonical: str,
    float32: str,
    policy: dict[str, Any],
    excluded_identity_sets: dict[str, set[str]],
) -> None:
    """Validate source completeness and identity bindings before overlap checks."""
    populations = policy.get("populations")
    if not isinstance(populations, dict):
        raise ValueError("exclusion_population_manifest_missing")
    require_complete_population_manifest(populations)
    verify_identity_union_bindings(policy, excluded_identity_sets)
    validate_candidate_identities(canonical, float32, excluded_identity_sets)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify_published_policy(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
