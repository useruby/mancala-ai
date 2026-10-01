from pathlib import Path

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import seed461_order_population as population


def test_canonical_reconstruction_and_population_coverage() -> None:
    root = Path(__file__).resolve().parents[2]
    canonical = root / ".tmp/canonical-reconstruction/medium_eval.jsonl"
    assert population.sha256(canonical) == population.CANONICAL_SHA
    universe = population.pr249_population()
    assert len(universe) == len(population.keys(universe)) == 28_961
    assert population.keys(suites.load_suite_jsonl(str(canonical))) <= population.keys(
        universe
    )


def test_holdout_has_zero_overlap_with_complete_source_population() -> None:
    excluded = population.keys(population.pr249_population())
    selected = population.select_holdout(excluded)
    assert len(selected) == 256
    assert not population.keys(selected) & excluded
    assert all(row["pit_sum"] > 32 for row in selected)


def test_512_opening_holdouts_are_disjoint() -> None:
    excluded = population.keys(population.pr249_population())
    first = population.select_holdout(excluded, seed=391, size=512)
    second = population.select_holdout(
        excluded | population.keys(first), seed=392, size=512
    )
    assert len(first) == len(population.keys(first)) == 512
    assert len(second) == len(population.keys(second)) == 512
    assert not population.keys(first) & population.keys(second)
    assert all(row["pit_sum"] > 32 for row in first + second)
