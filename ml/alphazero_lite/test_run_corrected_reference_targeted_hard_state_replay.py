from __future__ import annotations

import json
import tempfile
import unittest


class RunCorrectedReferenceTargetedHardStateReplayTest(unittest.TestCase):
    def test_apply_family_quota_prioritizes_capture_family(self) -> None:
        from ml.alphazero_lite.run_corrected_reference_targeted_hard_state_replay import (
            apply_family_quota,
        )

        rows = []
        for index in range(10):
            rows.append(
                {
                    "canonical_state": f"capture-{index}",
                    "metadata": {
                        "max_regret": 0.9 - (index * 0.01),
                        "max_value_error": 0.5,
                    },
                }
            )
        for family in (
            "opening_plies_1_8",
            "incumbent_proxy_disagreement",
            "high_value_swing",
            "high_imbalance",
        ):
            for index in range(4):
                rows.append(
                    {
                        "canonical_state": f"{family}-{index}",
                        "metadata": {
                            "max_regret": 0.7 - (index * 0.01),
                            "max_value_error": 0.4,
                        },
                    }
                )

        canonical_to_family = {
            row["canonical_state"]: (
                "capture_available"
                if row["canonical_state"].startswith("capture-")
                else row["canonical_state"].rsplit("-", 1)[0]
            )
            for row in rows
        }
        selected, summary = apply_family_quota(
            rows, top_n=20, canonical_to_family=canonical_to_family
        )

        self.assertEqual(20, len(selected))
        self.assertEqual(8, summary["family_selected_counts"]["capture_available"])
        self.assertEqual(3, summary["family_selected_counts"]["opening_plies_1_8"])
        self.assertEqual(
            3, summary["family_selected_counts"]["incumbent_proxy_disagreement"]
        )

    def test_write_filtered_forensic_artifact_keeps_only_selected_rows(self) -> None:
        from pathlib import Path

        from ml.alphazero_lite.run_corrected_reference_targeted_hard_state_replay import (
            _write_filtered_forensic_artifact,
        )

        payload = {
            "schema": "azlite_forensic_suite_v1",
            "systems": {
                "challenger": {
                    "artifact_path": "storage/ai/alphazero_lite/current",
                    "rows": [
                        {
                            "id": "capture-1",
                            "canonical_state": "capture-1-state",
                            "state": {},
                            "side_to_move": 0,
                            "legal_moves": [0],
                        },
                        {
                            "id": "capture-2",
                            "canonical_state": "capture-2-state",
                            "state": {},
                            "side_to_move": 0,
                            "legal_moves": [0],
                        },
                    ],
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "source.json"
            out_path = tmp_path / "filtered.json"
            source_path.write_text(json.dumps(payload), encoding="utf-8")

            filtered = _write_filtered_forensic_artifact(
                source_artifact_path=source_path,
                selected_canonical_states={"capture-2-state"},
                output_path=out_path,
            )

        self.assertEqual(
            ["capture-2"],
            [row["id"] for row in filtered["systems"]["challenger"]["rows"]],
        )

    def test_apply_family_quota_backfills_from_primary_families(self) -> None:
        from ml.alphazero_lite.run_corrected_reference_targeted_hard_state_replay import (
            apply_family_quota,
        )

        rows = [
            {
                "canonical_state": f"capture-{index}",
                "metadata": {
                    "max_regret": 0.9 - (index * 0.01),
                    "max_value_error": 0.4,
                },
            }
            for index in range(4)
        ]
        rows.extend(
            {
                "canonical_state": f"opening-{index}",
                "metadata": {
                    "max_regret": 0.8 - (index * 0.01),
                    "max_value_error": 0.3,
                },
            }
            for index in range(3)
        )
        rows.extend(
            {
                "canonical_state": f"proxy-{index}",
                "metadata": {
                    "max_regret": 0.7 - (index * 0.01),
                    "max_value_error": 0.2,
                },
            }
            for index in range(3)
        )

        canonical_to_family = {
            **{f"capture-{index}": "capture_available" for index in range(4)},
            **{f"opening-{index}": "opening_plies_1_8" for index in range(3)},
            **{f"proxy-{index}": "incumbent_proxy_disagreement" for index in range(3)},
        }

        selected, summary = apply_family_quota(
            rows,
            top_n=8,
            canonical_to_family=canonical_to_family,
        )

        self.assertEqual(8, len(selected))
        self.assertEqual(8, summary["selected_top_n"])
        self.assertEqual(
            1,
            summary["family_selected_counts"]["incumbent_proxy_disagreement"],
        )


if __name__ == "__main__":
    unittest.main()
