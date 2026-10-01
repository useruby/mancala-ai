import json

import numpy as np
import pytest

from ml.alphazero_lite.run_seed461_lr_fixed_e4_diagnostic import (
    e4_checkpoint_binding,
    validate_games,
    verify_cached_file,
)


def test_e4_game_accounting_requires_full_suite_and_both_seats(tmp_path):
    openings = [{"prefix_moves": [index]} for index in range(256)]
    rows = [
        {
            "opening_index": index,
            "opening_prefix_moves": [index],
            "challenger_player": seat,
            "winner": "draw",
        }
        for index in range(256)
        for seat in (0, 1)
    ]
    path = tmp_path / "games.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert np.all(validate_games(path, openings) == 0.5)
    path.write_text("\n".join(json.dumps(row) for row in rows[:-1]))
    with pytest.raises(ValueError, match="game_count_mismatch"):
        validate_games(path, openings)


def test_checkpoint_binding_uses_e4_even_when_selection_varies(tmp_path, monkeypatch):
    import ml.alphazero_lite.run_seed461_lr_fixed_e4_diagnostic as diagnostic

    monkeypatch.setattr(diagnostic, "WORK", tmp_path)
    amendment = {
        "checkpoints": {
            "A": {"epochs": {"E1": "a", "E4": "fixed-a"}, "selected_sha256": "a"},
            "B": {"epochs": {"E1": "b", "E4": "fixed-b"}, "selected_sha256": "b"},
        }
    }
    for run, expected in (("A", "fixed-a"), ("B", "fixed-b")):
        path, digest = e4_checkpoint_binding(amendment, run)
        assert path == tmp_path / "training" / run / "E4.npz"
        assert digest == expected


def test_cached_evidence_hash_must_match(tmp_path):
    import hashlib

    path = tmp_path / "record.json"
    path.write_text("bound evidence")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    verify_cached_file(path, expected, "record")
    with pytest.raises(ValueError, match="cached_evidence_identity_mismatch"):
        verify_cached_file(path, "0" * 64, "record")
