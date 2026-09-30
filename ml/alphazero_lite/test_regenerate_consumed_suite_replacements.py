from pathlib import Path
from types import SimpleNamespace

import pytest

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import regenerate_consumed_suite_replacements as recovery


def test_replacements_are_disjoint_and_explicitly_non_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical_dir = tmp_path / "canonical"
    canonical = canonical_dir / "medium_eval.jsonl"
    suites.write_suite_jsonl(
        suites.select_diverse(recovery.all_openings(), 128, 49), str(canonical)
    )
    monkeypatch.setattr(
        recovery.historical,
        "_SPECS",
        (
            SimpleNamespace(sha256=suites.suite_sha256(str(canonical))),
            SimpleNamespace(label="A", seed=1042, sha256="historical-a"),
        ),
    )
    result = recovery.regenerate(
        canonical=canonical,
        out_dir=tmp_path / "replacements",
        manifest=tmp_path / "replacement_registry.json",
    )
    assert result["status"] == "exploratory_only_not_historical_registry"
    assert all(
        entry["status"] == "replacement_not_original"
        for label, entry in result["entries"].items()
        if label != "canonical"
    )
