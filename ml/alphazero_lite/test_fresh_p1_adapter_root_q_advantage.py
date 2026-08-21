from __future__ import annotations

from ml.alphazero_lite.run_fresh_p1_adapter_root_q_advantage import matched_random


def test_matched_random_has_exact_positive_count_and_uses_no_q_fields() -> None:
    records = {
        f"state-{index}": {
            "group": "robust_positive" if index < 3 else "robust_nonpositive",
            "player": index % 2,
            "move_bucket": "00-04",
            "legal_count": 3 + (index % 2),
            "teacher_entropy_quartile": 1 + (index % 4),
        }
        for index in range(12)
    }
    selected = matched_random(records)
    assert len(selected) == 3
    assert selected == matched_random(records)
