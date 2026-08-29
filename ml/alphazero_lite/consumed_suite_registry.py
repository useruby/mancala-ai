"""Immutable consumed-opening registry for sealed AlphaZero evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import run_pr249_fresh_suite_generalization as pr249
from ml.alphazero_lite import run_pr251_cross_seed_strength_residual_transfer as pr251
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import sha256_file


@dataclass(frozen=True)
class ConsumedSuite:
    label: str
    path: Path
    seed: int | None
    sha256: str
    status: str = "consumed"


_ROOT = Path("/tmp")
_SPECS = (
    ConsumedSuite(
        "canonical",
        pr249.CANONICAL_SUITE,
        None,
        "57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04",
    ),
    ConsumedSuite(
        "A",
        _ROOT / "azlite_pr249_fresh_suite_generalization/suites/suite_A.jsonl",
        1042,
        "c8277e659c7a4e137140d83c187781f40e6b25c4b1dff5ec4da3f2e09fdcc6ab",
    ),
    ConsumedSuite(
        "B",
        _ROOT / "azlite_pr249_fresh_suite_generalization/suites/suite_B.jsonl",
        2042,
        "1f4c17eb7df21af75bc29c3274b3951899ae5fb2522f762d5270d58ddf93b37e",
    ),
    ConsumedSuite(
        "C",
        _ROOT / "azlite_pr249_fresh_suite_generalization/suites/suite_C.jsonl",
        3042,
        "b56783a2a2bbf63168cfb642f2b878badc80ace0279a9bbb7778757a4e4ba90d",
    ),
    ConsumedSuite(
        "D",
        _ROOT
        / "azlite_pr251_cross_seed_strength_residual_transfer/suites/suite_D.jsonl",
        4042,
        "e9d931031e75d39d188699d0b77ecc91d429c6998d06aada5f04e34b0384b1e2",
    ),
    ConsumedSuite(
        "E",
        _ROOT
        / "azlite_pr251_cross_seed_strength_residual_transfer/suites/suite_E.jsonl",
        5042,
        "3e78f5425370bb17ed2280e850a5f93b62cb86d94a0222263d481fcf866def37",
    ),
    ConsumedSuite(
        "F",
        _ROOT
        / "azlite_pr251_cross_seed_strength_residual_transfer/suites/suite_F.jsonl",
        6042,
        "8cba0ea407dde34696877d5ee2fe7110cd5610b2392f446df1bc02c1900d7c0e",
    ),
    ConsumedSuite(
        "G",
        _ROOT / "azlite_pr252_phase_target_delta_attribution/suites/suite_G.jsonl",
        7042,
        "ec331fc5672d0af95083443620ede6aac68755280ba375a9727bdd781918b216",
    ),
    ConsumedSuite(
        "H",
        _ROOT / "azlite_pr252_phase_target_delta_attribution/suites/suite_H.jsonl",
        8042,
        "24e4ae8cb9ab336959f243c1dedd903201497d94e04fbc99ddaeed8894c58682",
    ),
    ConsumedSuite(
        "I",
        _ROOT / "azlite_pr252_phase_target_delta_attribution/suites/suite_I.jsonl",
        9042,
        "5c8265a4c387c1a2f40ddc492da0435db548b98e02692582ccbf4b119f621b8e",
    ),
    ConsumedSuite(
        "J",
        _ROOT / "azlite_pr253_semantic_receiver_target_surgery/suites/suite_J.jsonl",
        10042,
        "7c0b097e18949a2d0ac5657116c565f3b9c0bb942ae69c9741dcd328915bf98b",
    ),
    ConsumedSuite(
        "K",
        _ROOT / "azlite_pr253_semantic_receiver_target_surgery/suites/suite_K.jsonl",
        11042,
        "64725ca85c6657eeefbbc329bc8a03f762bbbb7b615ff4fefeb4356fa158e1fd",
    ),
    ConsumedSuite(
        "L",
        _ROOT / "azlite_pr253_semantic_receiver_target_surgery/suites/suite_L.jsonl",
        12042,
        "147ff9d503975641da2ad366e1df93aa07a296190f850ed0514ecbf903b43ac0",
    ),
    ConsumedSuite(
        "M",
        _ROOT / "azlite_pr254_third_seed_budget_replay/suites/suite_M.jsonl",
        13042,
        "2d5552eaa68c2e5ad8397fb8a0c3eb1e9831a8a668991457dd9afe0c2b2960aa",
        "consumed_invalid_due_to_overlap",
    ),
    ConsumedSuite(
        "N",
        _ROOT / "azlite_pr254_third_seed_budget_replay/suites/suite_N.jsonl",
        14042,
        "5c37f39a1ea50478e3bbb17f5c0c02cb55068d8a9180c324e39b30cb83fc464e",
        "consumed_invalid_due_to_overlap",
    ),
    ConsumedSuite(
        "O",
        _ROOT / "azlite_pr254_third_seed_budget_replay/suites/suite_O.jsonl",
        15042,
        "0e066bbd3e6288efe4a7ba7bc26c29b27358bf8d85d686801a0a0521c8a4425b",
        "consumed_invalid_due_to_overlap",
    ),
)


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def _reconstruct(labels: tuple[str, ...], output: Path) -> dict[str, Path]:
    """Recreate J/L or M/O using their originally committed selector contract."""
    preceding = [
        spec
        for spec in _SPECS
        if spec.label in ("canonical", "A", "B", "C", "D", "E", "F", "G", "H", "I")
    ]
    entries = {
        spec.label: suites.load_suite_jsonl(str(spec.path)) for spec in preceding
    }
    used = set().union(*(pr249.suite_keys(value) for value in entries.values()))
    selected_paths: dict[str, Path] = {}
    for spec in (spec for spec in _SPECS if spec.label in labels):
        selected = suites.select_diverse(
            [
                entry
                for entry in pr249.all_openings()
                if suites.canonical_key(entry["state"]) not in used
            ],
            128,
            spec.seed,
        )
        path = output / f"suite_{spec.label}.jsonl"
        suites.write_suite_jsonl(selected, str(path))
        if sha256_file(path) != spec.sha256:
            fail(f"{spec.label} deterministic reconstruction SHA mismatch")
        selected_paths[spec.label] = path
        used |= pr249.suite_keys(selected)
    return selected_paths


def load(workdir: Path) -> dict[str, ConsumedSuite]:
    """Load the authoritative canonical-through-O registry, validating every SHA."""
    if len({spec.label for spec in _SPECS}) != len(_SPECS):
        fail("duplicate consumed-suite label")
    missing_jkl = any(
        not spec.path.is_file() for spec in _SPECS if spec.label in ("J", "K", "L")
    )
    recovered = (
        _reconstruct(("J", "K", "L"), workdir / "recovered_consumed")
        if missing_jkl
        else {}
    )
    missing_mno = any(
        not spec.path.is_file() for spec in _SPECS if spec.label in ("M", "N", "O")
    )
    if missing_mno:
        recovered.update(_reconstruct(("M", "N", "O"), workdir / "recovered_consumed"))
    result = {}
    for spec in _SPECS:
        path = recovered.get(spec.label, spec.path)
        if not path.is_file() or sha256_file(path) != spec.sha256:
            fail(f"consumed suite SHA mismatch: {spec.label}")
        result[spec.label] = ConsumedSuite(
            spec.label, path, spec.seed, spec.sha256, spec.status
        )
    # M/N/O must retain the known defective overlap fingerprint even when their files exist.
    jkl = set().union(
        *(
            pr249.suite_keys(suites.load_suite_jsonl(str(result[label].path)))
            for label in ("J", "K", "L")
        )
    )
    if {
        label: len(
            pr249.suite_keys(suites.load_suite_jsonl(str(result[label].path))) & jkl
        )
        for label in ("M", "N", "O")
    } != {"M": 13, "N": 14, "O": 22}:
        fail("M/N/O historical overlap fingerprint mismatch")
    validate(result)
    return result


def validate(registry: dict[str, ConsumedSuite]) -> None:
    """Reject partial registries before either replay filtering or suite selection."""
    expected = {spec.label: spec.sha256 for spec in _SPECS}
    actual = {label: spec.sha256 for label, spec in registry.items()}
    if actual != expected:
        fail("consumed-suite registry mismatch")


def entries(registry: dict[str, ConsumedSuite]) -> dict[str, list[dict[str, Any]]]:
    return {
        label: suites.load_suite_jsonl(str(spec.path))
        for label, spec in registry.items()
    }


def final_keys(registry: dict[str, ConsumedSuite]) -> set[str]:
    return set().union(*(pr249.suite_keys(rows) for rows in entries(registry).values()))


def prefix_keys(registry: dict[str, ConsumedSuite]) -> set[tuple[int, ...]]:
    return set().union(
        *(pr251.prefix_keys(rows) for rows in entries(registry).values())
    )


def manifest(registry: dict[str, ConsumedSuite]) -> dict[str, Any]:
    return {
        label: {"seed": spec.seed, "sha256": spec.sha256, "status": spec.status}
        for label, spec in registry.items()
    }
