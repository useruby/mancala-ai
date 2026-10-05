"""Repository-root-aware rebuilding of the published seed422 exclusions.

The registered historical builder is intentionally left byte-identical. This
entry point evaluates that frozen implementation in an isolated namespace,
with the caller's root and the one historical absolute source identity mapped
explicitly to its repository-relative evidence file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType
from typing import Any

from ml.alphazero_lite import seed422_exclusions

EXPECTED_COUNT = 266_575
EXPECTED_IDENTITY_SHA256 = (
    "f6713bebfe1ded09ed15e45f17a11566d0012f03fe7719138fdc832bbb1339f7"
)
HISTORICAL_415_PATH = (
    "/home/alex/Mancala/ai/docs/data/seed414-root-budget-confirmation/"
    "post-execution-corrected-exclusion-proof.json"
)
HISTORICAL_415_RELATIVE = (
    "docs/data/seed414-root-budget-confirmation/"
    "post-execution-corrected-exclusion-proof.json"
)
HISTORICAL_415_SHA256 = (
    "8e6829c75bb8884bd45b9ee86f583345fadf68e034ff97f0ac96a5420d33f697"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_root_bound_builder(root: Path) -> ModuleType:
    """Load the registered builder while replacing its path identity check."""
    source_path = Path(seed422_exclusions.__file__)
    source = source_path.read_text(encoding="utf-8")
    root_declaration = (
        'ROOT = Path(__file__).resolve().parents[2]\nDATA = ROOT / "docs/data"'
    )
    if source.count(root_declaration) != 1:
        raise ValueError("registered_exclusion_builder_root_declaration_unrecognized")
    source = source.replace(
        root_declaration, 'ROOT = Path(ROOT)\nDATA = ROOT / "docs/data"'
    )
    old = """    if Path(\n        corrected_binding["path"]\n    ).resolve() != corrected_415_path.resolve() or corrected_binding[\n        "sha256"\n    ] != sha256(corrected_415_path):\n        raise ValueError("#416 proof corrected_415_binding_mismatch")"""
    new = """    historical_path = corrected_binding["path"]
    if historical_path != HISTORICAL_415_PATH or corrected_binding["sha256"] != HISTORICAL_415_SHA256:\n        raise ValueError("#416 proof corrected_415_binding_mismatch")\n    if corrected_415_path != ROOT / HISTORICAL_415_RELATIVE:\n        raise ValueError("#416 proof corrected_415_repository_mapping_mismatch")\n    if sha256(corrected_415_path) != HISTORICAL_415_SHA256:\n        raise ValueError("#416 proof corrected_415_binding_mismatch")"""
    if source.count(old) != 1:
        raise ValueError("registered_exclusion_builder_path_check_unrecognized")
    source = source.replace(old, new)
    namespace = {
        "__name__": "seed422_root_bound_exclusion_builder",
        "__file__": str(source_path),
        "__package__": "ml.alphazero_lite",
        "ROOT": root,
        "DATA": root / "docs/data",
        "HISTORICAL_415_PATH": HISTORICAL_415_PATH,
        "HISTORICAL_415_RELATIVE": HISTORICAL_415_RELATIVE,
        "HISTORICAL_415_SHA256": HISTORICAL_415_SHA256,
    }
    exec(compile(source, str(source_path), "exec"), namespace)
    module = ModuleType("seed422_root_bound_exclusion_builder")
    module.__dict__.update(namespace)
    return module


def build_union(root: Path) -> tuple[set[str], dict[str, Any]]:
    """Rebuild all #415–421 identity sources beneath ``root``."""
    root = root.resolve()
    legacy = _load_root_bound_builder(root)
    identities, rebuilt = legacy.build_union()
    if len(identities) != EXPECTED_COUNT:
        raise ValueError(f"historical_union_count_mismatch:{len(identities)}")
    if rebuilt["identity_set_sha256"] != EXPECTED_IDENTITY_SHA256:
        raise ValueError("historical_union_digest_mismatch")
    proof = legacy.DATA / "seed422-adam-first-moment/opening-exclusion-proof.json"
    recorded = __import__("json").loads(proof.read_text(encoding="utf-8"))
    for key in (
        "sources",
        "identity_count",
        "identity_set_sha256",
        "excluded_identities",
    ):
        if rebuilt.get(key) != recorded.get(key):
            raise ValueError(f"historical_union_proof_mismatch:{key}")
    return identities, rebuilt
