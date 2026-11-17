"""Authenticate the proposed, bounded Seed 604 evaluation exclusions.

This adapter only verifies identities and checks candidate states. It has no
opening-generation, training, evaluation-game, freeze, or authorization path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Set as AbstractSet
from pathlib import Path
from types import MappingProxyType
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
SEED604_REGISTRATION_SHA256 = (
    "b2a8eb494fefa97ee454cd52760612a92f6f766b053dfb730a6c94bb962e3d71"
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
EXECUTION_STATUS_KEYS = frozenset(
    {
        "evaluation_openings_generated",
        "execution_freeze_created",
        "training_started",
        "evaluation_games_started",
        "authorization_created",
    }
)
_AUTHENTICATED_POLICY_SEAL = object()


def _freeze_json(value: Any) -> Any:
    """Recursively freeze a parsed JSON document for authenticated reuse."""
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


class AuthenticatedBoundedPolicy:
    """Verified exclusion document and immutable identity sets loaded from it."""

    __slots__ = ("document", "_excluded", "_seal")

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, name):
            raise AttributeError("authenticated_bounded_policy_is_immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        document: dict[str, Any],
        identity_sets: dict[str, dict[str, set[str]]],
        seal: object,
    ) -> None:
        if seal is not _AUTHENTICATED_POLICY_SEAL:
            raise ValueError("bounded_policy_must_be_loaded_and_authenticated")
        self.document = _freeze_json(document)
        self._excluded = MappingProxyType(
            {
                scheme: frozenset().union(
                    *(population[scheme] for population in identity_sets.values())
                )
                for scheme in ("canonical", "float32")
            }
        )
        self._seal = seal

    def validate_candidate_state(self, state: dict[str, Any]) -> dict[str, str]:
        """Recompute both identities and reject overlap with authenticated sources."""
        if self._seal is not _AUTHENTICATED_POLICY_SEAL:
            raise ValueError("bounded_policy_authentication_seal_invalid")
        if not isinstance(state, dict):
            raise TypeError("candidate_state_must_be_a_mapping")
        canonical = canonical_identity(state)
        float32 = production_float32_identity(state)
        _validate_candidate_identities(
            canonical,
            float32,
            self._excluded,
        )
        return {"canonical": canonical, "float32": float32}


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


def _validate_candidate_identities(
    canonical: str,
    float32: str,
    excluded: Mapping[str, AbstractSet[str]],
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


def _build_bounded_evidence(
    root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, set[str]]]]:
    """Authenticate all inputs and build the normalized policy and identity sets.

    The Seed 561 verifier is read-only and must pass before its identity indexes
    are accepted. No openings are generated or written.
    """
    root = root.resolve()
    registration = _read_pinned_json(
        root, str(DATA / "registration.json"), SEED604_REGISTRATION_SHA256
    )
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
    policy = {
        "schema": "seed604-bounded-exclusion-policy-v1",
        "status": "proposed-awaiting-project-owner-review",
        "seed604_registration_sha256": SEED604_REGISTRATION_SHA256,
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
                "Seed 482-484 authenticated historical snapshots",
                "Seed 488, 499, 505 and 508 evaluation roots and replayed trajectory identities",
                "Seed 500 pilot and self-play rows",
                "Seed 562-564 historical source indexes, including Seed 556-560 and Seed 503-504 source families",
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
    if registration.get("seed") != 604:
        raise ValueError("seed604_registration_seed_mismatch")
    return policy, identity_sets


def build_bounded_policy(root: Path = ROOT) -> dict[str, Any]:
    """Build internal evidence; candidate validation never accepts this mapping."""
    policy, _identity_sets = _build_bounded_evidence(root)
    return policy


def _identity_record(source: dict[str, Any], scheme: str) -> dict[str, Any]:
    count_key = "canonical_distinct" if scheme == "canonical" else "float32_distinct"
    digest_key = (
        "canonical_identity_set_sha256"
        if scheme == "canonical"
        else "float32_identity_set_sha256"
    )
    return {"distinct": source[count_key], "sha256": source[digest_key]}


def _published_sources(internal: dict[str, Any]) -> dict[str, Any]:
    """Map authenticated internal populations to the published policy schema."""
    populations = internal["populations"]
    sources: dict[str, Any] = {}
    for arm in ("A", "B"):
        name = f"seed599_training_{arm}"
        source = populations[name]
        sources[name] = {
            "path": source["path"],
            "source_commit": source["source_commit"],
            "configuration_path": source["registration_path"],
            "configuration_sha256": source["registration_sha256"],
            "training_freeze_path": str(SEED599 / "training-execution-freeze.json"),
            "training_freeze_sha256": SEED599_TRAINING_FREEZE_SHA256,
            "file_sha256": source["file_sha256"],
            "format": "JSON array; extract canonical_state from each training row.",
            "extraction_rule": (
                "Recompute canonical SHA-256 from sorted compact JSON canonical_state "
                "and production float32 SHA-256 from the registered kalah_v3 encoder; "
                "require any stored identity fields to agree."
            ),
            "records": source["records"],
            "canonical_identity_set": _identity_record(source, "canonical"),
            "production_float32_identity_set": _identity_record(source, "float32"),
            "verification_status": (
                "passed; file bytes, training configuration, freeze binding, stored "
                "and recomputed identities agree"
            ),
        }

    heldout = populations["seed599_heldout"]
    sources["seed599_heldout"] = {
        "path": heldout["path"],
        "source_commit": heldout["source_commit"],
        "registration_path": heldout["registration_path"],
        "registration_sha256": heldout["registration_sha256"],
        "published_membership_sha256": SEED599_HELDOUT_IDENTITY_SHA256,
        "file_sha256": heldout["file_sha256"],
        "format": "JSON array of provenance-held-out training examples; extract canonical_state.",
        "extraction_rule": (
            "Recompute canonical SHA-256 from sorted compact JSON canonical_state "
            "and production float32 SHA-256 from the registered kalah_v3 encoder; "
            "require any stored identity fields to agree."
        ),
        "records": heldout["records"],
        "canonical_identity_set": _identity_record(heldout, "canonical"),
        "production_float32_identity_set": _identity_record(heldout, "float32"),
        "verification_status": (
            "passed; canonical membership bytes match the published preparation "
            "identity and both schemes were recomputed"
        ),
    }

    roots = populations["seed599_evaluation_roots"]
    sources["seed599_evaluation_roots"] = {
        "path": roots["path"],
        "source_commit": roots["source_commit"],
        "evaluation_contract_path": roots["registration_path"],
        "evaluation_contract_sha256": roots["registration_sha256"],
        "evaluation_freeze_path": str(
            SEED599 / "evaluation-readiness/execution-freeze.json"
        ),
        "evaluation_freeze_sha256": SEED599_EVALUATION_FREEZE_SHA256,
        "file_sha256": roots["file_sha256"],
        "format": (
            "JSON array of registered opening roots; extract state and recompute "
            "identities, checking declared identity fields."
        ),
        "extraction_rule": (
            "Use each authenticated resulting root state; recompute canonical and "
            "production float32 identities and reject mismatches with declared "
            "identity fields. Do not use game outcomes."
        ),
        "records": roots["records"],
        "canonical_identity_set": _identity_record(roots, "canonical"),
        "production_float32_identity_set": _identity_record(roots, "float32"),
        "verification_status": (
            "passed; file hash, contract opening-manifest digest, count and "
            "stored/recomputed identities agree"
        ),
    }

    historical = populations["seed561_historical_inventory"]
    sources["seed561_historical_inventory"] = {
        "directory": str(SEED561),
        "source_commit": historical["source_commit"],
        "publication_commit": "b2f2bb092ededaad07e1c2d7dda57ce0d24d4409",
        "anchor_commit": "0139cbd70d70184fc86cabd92afafdd849f28ef0",
        "source_manifest_path": historical["registration_path"],
        "source_manifest_sha256": historical["registration_sha256"],
        "registration_sha256": historical["registration_sha256"],
        "source_scope": (
            "exact Seed 511 registration exclusion_source_sha256 list plus the "
            "independently verified Seed 562-564 subtotal"
        ),
        "extraction_rule": (
            "Run the published Seed 561 verifier; load the externally pinned "
            "distinct canonical and kalah_v3 float32 indexes as separate identity namespaces."
        ),
        "file_sha256": historical["file_sha256"],
        "occurrence_rows_per_scheme": historical["records"],
        "records": historical["records"],
        "canonical_index_sha256": historical["canonical_index_sha256"],
        "canonical_identity_set": _identity_record(historical, "canonical"),
        "float32_index_sha256": historical["float32_index_sha256"],
        "production_float32_identity_set": _identity_record(historical, "float32"),
        "included_families": internal["seed561_historical_scope"][
            "historical_source_families"
        ],
        "verification_status": (
            "passed using the published Seed 561 verifier and externally pinned "
            "index loader"
        ),
    }
    return sources


def _published_policy_document(internal: dict[str, Any]) -> dict[str, Any]:
    """Serialize the sole published schema from verified internal evidence."""
    unions = internal["identity_unions"]
    canonical_count = unions["canonical"]["distinct"]
    float32_count = unions["float32"]["distinct"]
    if canonical_count != float32_count:
        raise ValueError("bounded_union_scheme_count_mismatch")
    intersections = internal["seed599_to_seed561_intersections"]
    return {
        "schema": "seed604-bounded-exclusion-policy-v1",
        "status": "proposed-awaiting-project-owner-review",
        "seed604_registration_sha256": internal["seed604_registration_sha256"],
        "boundary": (
            "All authenticated Seed 599 A/B training membership states, Seed 599 "
            "held-out states, all 1,024 Seed 599 evaluation opening roots, and the "
            "full authenticated Seed 561 bounded historical identity inventory."
        ),
        "identity_rule": {
            "schemes": [
                "canonical_state_sha256",
                "production_kalah_v3_float32_sha256",
            ],
            "candidate_requirement": (
                "Every future Seed 604 opening root must be disjoint from every "
                "included population in both matching identity schemes; roots must "
                "also be unique within each scheme."
            ),
            "identity_set_sha256": (
                "SHA-256 of sorted lowercase 64-character identity strings joined "
                "by LF without a terminal LF."
            ),
        },
        "sources": _published_sources(internal),
        "observed_overlap_accounting": {
            "seed599_population_intersections_with_seed561": {
                "canonical" if scheme == "canonical" else "production_float32": {
                    name.removeprefix("seed599_"): counts[scheme]
                    for name, counts in intersections.items()
                }
                for scheme in ("canonical", "float32")
            },
            "seed599_cross_population_overlaps": internal[
                "known_overlaps_within_included_seed599_populations"
            ],
            "distinct_seed599_union_per_scheme": internal[
                "seed599_distinct_union_per_scheme"
            ],
            "bounded_union_per_scheme": canonical_count,
            "bounded_union_identity_set_sha256": {
                "canonical": unions["canonical"]["identity_set_sha256"],
                "production_float32": unions["float32"]["identity_set_sha256"],
            },
        },
        "limitations": {
            "seed461_continuation_states": (
                "unresolved and not included; see PR #605 and "
                "docs/data/seed604-replay-optimization/exclusion-inventory-audit.json"
            ),
            "other_historical_populations": (
                "outside the exact Seed 511-pinned scope remain uncovered with unknown counts"
            ),
            "seed599_evaluation_trajectories": (
                "not included; this boundary adds the 1,024 authenticated evaluation "
                "opening roots, not all post-root states from Seed 599 games"
            ),
            "claim_boundary": (
                "Do not claim complete historical independence, a globally untouched "
                "holdout, or unseen-state generalization. Report conclusions as bounded "
                "to these authenticated exclusions and the registered paired cohort."
            ),
            "seed461_comparison_effect": (
                "The unresolved states do not prevent the bounded paired B2048-versus-B512 "
                "or B2048-versus-P-E4 cohort contrasts, but prevent claims that the cohort "
                "is independent of all historical experience. No missing states may be "
                "reconstructed by guessing the execution source version."
            ),
        },
        "seed604_experiment_invariants": {
            "parent_checkpoint_sha256": "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
            "B512_checkpoint_sha256": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
            "treatment": (
                "continue from authenticated P-E4 with fresh Adam; exact Seed 599 "
                "B512 tensor reproduction before update 513"
            ),
            "update_budget": 2048,
            "comparison": (
                "B512 and B2048 versus the same serving P-E4; unchanged common P-E4 value evaluator"
            ),
            "statistical_gates": (
                "unchanged from the Seed 604 registration: primary one-sided 95% lower "
                "bound > 0.50; secondary paired two-sided 95% lower endpoint for B2048 "
                "minus B512 > 0"
            ),
        },
        "execution_status": {name: False for name in sorted(EXECUTION_STATUS_KEYS)},
    }


def validate_published_policy_document(saved: Any, expected: dict[str, Any]) -> None:
    """Fail closed on schema, source-registration, index, status or digest changes."""
    if not isinstance(saved, dict):
        raise ValueError("bounded_exclusion_policy_invalid_document")
    if saved.get("schema") != expected.get("schema") or saved.get(
        "status"
    ) != expected.get("status"):
        raise ValueError("bounded_exclusion_policy_schema_or_status_mismatch")
    status = saved.get("execution_status")
    if not isinstance(status, dict) or set(status) != EXECUTION_STATUS_KEYS:
        raise ValueError("bounded_exclusion_policy_execution_status_schema_mismatch")
    if any(type(value) is not bool or value is not False for value in status.values()):
        raise ValueError("bounded_exclusion_policy_execution_status_invalid")
    expected_sources = expected.get("sources", {})
    saved_sources = saved.get("sources")
    if not isinstance(saved_sources, dict):
        raise ValueError("bounded_exclusion_policy_sources_missing")
    missing = REQUIRED_POLICY_POPULATIONS - saved_sources.keys()
    if missing:
        raise ValueError(
            f"bounded_exclusion_policy_source_registration_missing:{','.join(sorted(missing))}"
        )
    if set(saved_sources) != set(expected_sources):
        raise ValueError("bounded_exclusion_policy_source_registration_set_mismatch")
    saved_historical = saved_sources["seed561_historical_inventory"]
    expected_historical = expected_sources["seed561_historical_inventory"]
    for field in ("canonical_index_sha256", "float32_index_sha256"):
        if saved_historical.get(field) != expected_historical.get(field):
            raise ValueError(f"seed561_policy_index_hash_mismatch:{field}")
    for population, expected_source in expected_sources.items():
        saved_source = saved_sources[population]
        if set(saved_source) != set(expected_source):
            raise ValueError(
                f"bounded_exclusion_policy_source_field_set_mismatch:{population}"
            )
        for field, expected_value in expected_source.items():
            if saved_source[field] != expected_value:
                raise ValueError(
                    f"bounded_exclusion_policy_source_binding_mismatch:{population}:{field}"
                )
    if set(saved) != set(expected):
        raise ValueError("bounded_exclusion_policy_manifest_key_set_mismatch")
    for key, expected_value in expected.items():
        if saved[key] != expected_value:
            if key == "observed_overlap_accounting":
                observed = saved[key]
                for accounting_key, accounting_value in expected_value.items():
                    if observed.get(accounting_key) != accounting_value:
                        raise ValueError(
                            "bounded_exclusion_policy_accounting_mismatch:"
                            f"{accounting_key}:saved={observed.get(accounting_key)!r}:"
                            f"expected={accounting_value!r}"
                        )
            raise ValueError(f"bounded_exclusion_policy_manifest_mismatch:{key}")


def load_authenticated_published_policy(
    root: Path = ROOT,
) -> AuthenticatedBoundedPolicy:
    """Load the one registered policy path, authenticate it, and retain its sets.

    Callers cannot provide a policy document or exclusion sets. The returned
    object can validate multiple candidate roots against the source-backed sets.
    """
    root = root.resolve()
    internal, identity_sets = _build_bounded_evidence(root)
    expected = _published_policy_document(internal)
    path = root / DATA / "bounded-exclusion-policy.json"
    if not path.is_file():
        raise ValueError("bounded_exclusion_policy_missing")
    try:
        saved = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("bounded_exclusion_policy_invalid_json") from error
    validate_published_policy_document(saved, expected)
    return AuthenticatedBoundedPolicy(
        saved,
        identity_sets,
        _AUTHENTICATED_POLICY_SEAL,
    )


def verify_published_policy(root: Path = ROOT) -> dict[str, Any]:
    """Recompute and verify the checked-in policy without producing outputs."""
    authenticated = load_authenticated_published_policy(root)
    return {
        "passed": True,
        "status": authenticated.document["status"],
        "identity_unions": dict(
            authenticated.document["observed_overlap_accounting"][
                "bounded_union_identity_set_sha256"
            ]
        ),
        "evaluation_openings_generated": False,
        "training_or_games_started": False,
    }


def validate_candidate_against_published_policy(
    candidate_state: dict[str, Any],
    policy: AuthenticatedBoundedPolicy,
) -> dict[str, str]:
    """Recompute and validate a root using a loaded authenticated policy object."""
    if not isinstance(policy, AuthenticatedBoundedPolicy):
        raise TypeError("candidate_validation_requires_authenticated_published_policy")
    return policy.validate_candidate_state(candidate_state)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify_published_policy(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
