import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


class BuildTacticalStableFailureFamilyReplayTest(unittest.TestCase):
    OPENING_CAPTURE_SUITE_ROWS = [
        {
            "id": "capture_available-016",
            "state": {
                "player_pits": [5, 1, 5, 5, 5, 0],
                "opponent_pits": [1, 6, 0, 7, 6, 5],
                "player_store": 1,
                "opponent_store": 1,
                "current_player": 0,
            },
            "side_to_move": 0,
            "legal_moves": [0, 1, 2, 3, 4],
            "phase": "opening",
            "bucket": "capture_available",
            "tags": ["capture_available", "generated", "ply_4"],
            "source": "generated",
        },
        {
            "id": "capture_available-017",
            "state": {
                "player_pits": [5, 1, 6, 5, 5, 0],
                "opponent_pits": [1, 6, 6, 0, 6, 5],
                "player_store": 1,
                "opponent_store": 1,
                "current_player": 0,
            },
            "side_to_move": 0,
            "legal_moves": [0, 1, 2, 3, 4],
            "phase": "opening",
            "bucket": "capture_available",
            "tags": ["capture_available", "generated", "ply_4"],
            "source": "generated",
        },
        {
            "id": "capture_available-018",
            "state": {
                "player_pits": [5, 1, 6, 5, 5, 0],
                "opponent_pits": [1, 6, 6, 6, 0, 5],
                "player_store": 1,
                "opponent_store": 1,
                "current_player": 0,
            },
            "side_to_move": 0,
            "legal_moves": [0, 1, 2, 3, 4],
            "phase": "opening",
            "bucket": "capture_available",
            "tags": ["capture_available", "generated", "ply_4"],
            "source": "generated",
        },
        {
            "id": "capture_available-019",
            "state": {
                "player_pits": [5, 1, 6, 5, 5, 0],
                "opponent_pits": [1, 6, 6, 6, 5, 0],
                "player_store": 1,
                "opponent_store": 1,
                "current_player": 0,
            },
            "side_to_move": 0,
            "legal_moves": [0, 1, 2, 3, 4],
            "phase": "opening",
            "bucket": "capture_available",
            "tags": ["capture_available", "generated", "ply_4"],
            "source": "generated",
        },
        {
            "id": "capture_available-020",
            "state": {
                "player_pits": [1, 0, 7, 7, 6, 5],
                "opponent_pits": [5, 4, 0, 5, 5, 0],
                "player_store": 1,
                "opponent_store": 2,
                "current_player": 1,
            },
            "side_to_move": 1,
            "legal_moves": [0, 1, 3, 4],
            "phase": "opening",
            "bucket": "capture_available",
            "tags": ["capture_available", "generated", "ply_4"],
            "source": "generated",
        },
        {
            "id": "capture_available-021",
            "state": {
                "player_pits": [1, 6, 0, 7, 6, 5],
                "opponent_pits": [5, 5, 0, 5, 5, 0],
                "player_store": 1,
                "opponent_store": 2,
                "current_player": 1,
            },
            "side_to_move": 1,
            "legal_moves": [0, 1, 3, 4],
            "phase": "opening",
            "bucket": "capture_available",
            "tags": ["capture_available", "generated", "ply_4"],
            "source": "generated",
        },
        {
            "id": "capture_available-022",
            "state": {
                "player_pits": [1, 6, 6, 0, 6, 5],
                "opponent_pits": [5, 5, 1, 5, 5, 0],
                "player_store": 1,
                "opponent_store": 2,
                "current_player": 1,
            },
            "side_to_move": 1,
            "legal_moves": [0, 1, 3, 4],
            "phase": "opening",
            "bucket": "capture_available",
            "tags": ["capture_available", "generated", "ply_4"],
            "source": "generated",
        },
    ]

    HIGH_IMBALANCE_SUITE_ROWS = [
        {
            "id": "high_imbalance-001",
            "state": {
                "player_pits": [0, 5, 5, 5, 0, 5],
                "opponent_pits": [0, 0, 5, 5, 5, 5],
                "player_store": 8,
                "opponent_store": 0,
                "current_player": 1,
            },
            "side_to_move": 1,
            "legal_moves": [2, 3, 4, 5],
            "phase": "opening",
            "bucket": "high_imbalance",
            "tags": ["high_imbalance", "seed", "ply_3"],
            "source": "seed",
        },
        {
            "id": "high_imbalance-002",
            "state": {
                "player_pits": [0, 0, 5, 5, 5, 4],
                "opponent_pits": [0, 5, 5, 5, 0, 5],
                "player_store": 0,
                "opponent_store": 9,
                "current_player": 0,
            },
            "side_to_move": 0,
            "legal_moves": [2, 3, 4, 5],
            "phase": "opening",
            "bucket": "high_imbalance",
            "tags": ["high_imbalance", "seed", "ply_4"],
            "source": "seed",
        },
        {
            "id": "high_imbalance-003",
            "state": {
                "player_pits": [0, 0, 0, 7, 7, 6],
                "opponent_pits": [1, 6, 0, 6, 0, 5],
                "player_store": 9,
                "opponent_store": 1,
                "current_player": 1,
            },
            "side_to_move": 1,
            "legal_moves": [0, 1, 3, 5],
            "phase": "opening",
            "bucket": "high_imbalance",
            "tags": ["high_imbalance", "seed"],
            "source": "seed",
        },
    ]

    def encode_state(self, raw_state):
        from ml.alphazero_lite import self_play

        return self_play.encode_state(raw_state, input_encoding="kalah_v3")

    def preservation_capture_row(self, canonical_state, *, priority, variant):
        rows = {
            "gain_4_5": {
                "raw_state": {
                    "player_pits": [5, 5, 1, 0, 6, 6],
                    "opponent_pits": [5, 5, 4, 4, 4, 0],
                    "player_store": 2,
                    "opponent_store": 1,
                    "current_player": 0,
                },
                "legal_moves": [0, 1, 2, 4, 5],
                "policy": [
                    0.0013778912834823132,
                    0.24870026111602783,
                    0.059389978647232056,
                    0.0,
                    0.4638570547103882,
                    0.22667483985424042,
                ],
                "teacher_selected_move": 4,
            },
            "gain_2_3": {
                "raw_state": {
                    "player_pits": [4, 4, 4, 4, 0, 5],
                    "opponent_pits": [0, 1, 6, 6, 6, 6],
                    "player_store": 1,
                    "opponent_store": 1,
                    "current_player": 0,
                },
                "legal_moves": [0, 1, 2, 3, 5],
                "policy": [
                    0.01769360341131687,
                    0.03087548352777958,
                    0.44671961665153503,
                    0.3265913426876068,
                    0.0,
                    0.17811991274356842,
                ],
                "teacher_selected_move": 2,
            },
            "gain_6_plus": {
                "raw_state": {
                    "player_pits": [0, 6, 3, 1, 2, 0],
                    "opponent_pits": [6, 6, 5, 0, 3, 0],
                    "player_store": 0,
                    "opponent_store": 16,
                    "current_player": 0,
                },
                "legal_moves": [1, 2, 3, 4],
                "policy": [0.0, 0.02, 0.6, 0.02, 0.3, 0.0],
                "teacher_selected_move": 2,
            },
            "gain_6_plus_alt": {
                "raw_state": {
                    "player_pits": [2, 5, 0, 2, 2, 0],
                    "opponent_pits": [5, 3, 0, 6, 0, 4],
                    "player_store": 12,
                    "opponent_store": 1,
                    "current_player": 0,
                },
                "legal_moves": [0, 1, 3, 4],
                "policy": [0.6, 0.3, 0.0, 0.02, 0.02, 0.0],
                "teacher_selected_move": 0,
            },
        }
        selected = rows[variant]
        return {
            "canonical_state": canonical_state,
            "state": self.encode_state(selected["raw_state"]),
            "raw_state": selected["raw_state"],
            "side_to_move": 0,
            "legal_moves": selected["legal_moves"],
            "bucket": "capture_available",
            "bucket_group": "tactical",
            "input_encoding": "kalah_v3",
            "policy_target_mode": "sharpened",
            "value_target_mode": "sharpened",
            "value": 0.5,
            "priority_score": priority,
            "policy": selected["policy"],
            "teacher_selected_move": selected["teacher_selected_move"],
        }

    def nearby_row(self, bucket, index, *, priority):
        bucket_states = {
            "high_imbalance": [
                {
                    "player_pits": [0, 4, 1, 0, 3, 5],
                    "opponent_pits": [2, 0, 5, 4, 0, 6],
                    "player_store": 16,
                    "opponent_store": 11,
                    "current_player": 0,
                },
                {
                    "player_pits": [3, 0, 4, 2, 0, 5],
                    "opponent_pits": [0, 5, 1, 6, 0, 4],
                    "player_store": 15,
                    "opponent_store": 12,
                    "current_player": 0,
                },
                {
                    "player_pits": [2, 5, 0, 4, 1, 0],
                    "opponent_pits": [6, 0, 3, 0, 5, 2],
                    "player_store": 17,
                    "opponent_store": 10,
                    "current_player": 0,
                },
            ],
            "high_value_swing": [
                {
                    "player_pits": [4, 0, 2, 5, 0, 1],
                    "opponent_pits": [0, 6, 3, 0, 4, 5],
                    "player_store": 13,
                    "opponent_store": 14,
                    "current_player": 0,
                },
                {
                    "player_pits": [0, 3, 5, 0, 4, 2],
                    "opponent_pits": [5, 0, 1, 6, 0, 4],
                    "player_store": 14,
                    "opponent_store": 13,
                    "current_player": 0,
                },
                {
                    "player_pits": [1, 4, 0, 3, 5, 0],
                    "opponent_pits": [0, 5, 4, 0, 2, 6],
                    "player_store": 12,
                    "opponent_store": 15,
                    "current_player": 0,
                },
            ],
            "starvation_pressure": [
                {
                    "player_pits": [0, 0, 1, 6, 0, 7],
                    "opponent_pits": [4, 5, 3, 2, 6, 1],
                    "player_store": 19,
                    "opponent_store": 8,
                    "current_player": 0,
                },
                {
                    "player_pits": [0, 2, 0, 5, 7, 0],
                    "opponent_pits": [5, 4, 6, 1, 3, 2],
                    "player_store": 18,
                    "opponent_store": 9,
                    "current_player": 0,
                },
                {
                    "player_pits": [1, 0, 0, 4, 0, 8],
                    "opponent_pits": [6, 3, 5, 2, 4, 1],
                    "player_store": 20,
                    "opponent_store": 7,
                    "current_player": 0,
                },
            ],
        }
        raw_state = json.loads(
            json.dumps(bucket_states[bucket][index % len(bucket_states[bucket])])
        )
        raw_state["player_store"] += index // len(bucket_states[bucket])
        raw_state["opponent_store"] += (index + 1) // len(bucket_states[bucket])
        side_pits = (
            raw_state["player_pits"]
            if raw_state["current_player"] == 0
            else raw_state["opponent_pits"]
        )
        legal_moves = [pit for pit, seeds in enumerate(side_pits) if seeds > 0]
        preferred_move = legal_moves[index % len(legal_moves)]
        secondary_move = legal_moves[(index + 1) % len(legal_moves)]
        policy = [0.0] * 6
        base_share = 0.12 / max(1, len(legal_moves) - 2)
        for move in legal_moves:
            policy[move] = base_share
        policy[preferred_move] = 0.58
        policy[secondary_move] = 0.3
        return {
            "canonical_state": f"{bucket}-{index}",
            "state": self.encode_state(raw_state),
            "raw_state": raw_state,
            "side_to_move": raw_state["current_player"],
            "legal_moves": legal_moves,
            "bucket": bucket,
            "bucket_group": "preservation"
            if bucket == "starvation_pressure"
            else "tactical",
            "input_encoding": "kalah_v3",
            "policy_target_mode": "sharpened",
            "value_target_mode": "sharpened",
            "value": 0.25 + (0.01 * (index % 3)),
            "priority_score": priority,
            "policy": policy,
        }

    def opening_capture_suite_rows(self):
        return json.loads(json.dumps(self.OPENING_CAPTURE_SUITE_ROWS))

    def high_imbalance_suite_rows(self):
        return json.loads(json.dumps(self.HIGH_IMBALANCE_SUITE_ROWS))

    def rebalance_suite_rows(self):
        return self.high_imbalance_suite_rows() + self.opening_capture_suite_rows()

    def rebalance_reference_payload(
        self, module, *, unstable_opening_ids=None, unstable_high_imbalance_ids=None
    ):
        unstable_opening_ids = set(unstable_opening_ids or [])
        unstable_high_imbalance_ids = set(unstable_high_imbalance_ids or [])

        opening_reference_moves = {
            "capture_available-016": 4,
            "capture_available-017": 3,
            "capture_available-018": 4,
            "capture_available-019": 4,
            "capture_available-020": 0,
            "capture_available-021": 1,
            "capture_available-022": 3,
        }
        imbalance_reference_moves = {
            "high_imbalance-001": 2,
            "high_imbalance-002": 3,
            "high_imbalance-003": 5,
        }

        rows = []
        for row in self.rebalance_suite_rows():
            row_id = row["id"]
            reference_move = opening_reference_moves.get(
                row_id, imbalance_reference_moves.get(row_id)
            )
            unstable = (
                row_id in unstable_opening_ids or row_id in unstable_high_imbalance_ids
            )
            alternate_reference_move = next(
                (move for move in row["legal_moves"] if move != reference_move),
                None,
            )
            observed_reference_moves = (
                [reference_move]
                if not unstable
                else [reference_move, alternate_reference_move]
            )
            seed_samples = [
                {"seed": 2040, "reference_move": reference_move, "teacher_value": 0.42},
            ]
            if unstable:
                if alternate_reference_move is None:
                    raise ValueError(
                        f"unstable test fixture requires alternate legal move for {row_id}"
                    )
                seed_samples.append(
                    {
                        "seed": 3040,
                        "reference_move": observed_reference_moves[1],
                        "teacher_value": 0.41,
                    }
                )
            rows.append(
                {
                    "id": row_id,
                    "canonical_state": module.canonical_state_key(row["state"]),
                    "state": row["state"],
                    "reference_move": None if unstable else reference_move,
                    "reference_unstable": unstable,
                    "observed_reference_moves": observed_reference_moves,
                    "seed_samples": seed_samples,
                }
            )

        return {
            "schema": "azlite_forensic_references_v1",
            "reference": {
                "policy_simulations": 1200,
                "value_simulations": 1800,
                "sample_seeds": [2040, 3040],
            },
            "rows": rows,
        }

    def rebalance_source_rows(self, *, nearby_counts=None):
        nearby_counts = dict(
            nearby_counts
            or {
                "high_imbalance": 4,
                "high_value_swing": 6,
                "starvation_pressure": 2,
            }
        )
        nearby_priority_starts = {
            "high_imbalance": 30,
            "high_value_swing": 40,
            "starvation_pressure": 20,
        }
        source_rows = [
            self.preservation_capture_row(
                "preserve-capture-a", priority=9.0, variant="gain_4_5"
            ),
            self.preservation_capture_row(
                "preserve-capture-b", priority=8.0, variant="gain_2_3"
            ),
            self.preservation_capture_row(
                "preserve-capture-c", priority=7.0, variant="gain_6_plus"
            ),
            self.preservation_capture_row(
                "preserve-capture-d", priority=6.0, variant="gain_6_plus_alt"
            ),
        ]
        for bucket in ("high_imbalance", "high_value_swing", "starvation_pressure"):
            for index in range(nearby_counts[bucket]):
                source_rows.append(
                    self.nearby_row(
                        bucket,
                        index,
                        priority=nearby_priority_starts[bucket] - index,
                    )
                )
        return source_rows

    def test_selects_structural_opening_capture_rows_for_reference_moves_three_and_four(
        self,
    ):
        from ml.alphazero_lite import (
            build_tactical_stable_failure_family_replay as module,
        )

        suite_rows = [
            {
                "id": "capture_available-016",
                "state": {
                    "player_pits": [5, 1, 5, 5, 5, 0],
                    "opponent_pits": [1, 6, 0, 7, 6, 5],
                    "player_store": 1,
                    "opponent_store": 1,
                    "current_player": 0,
                },
                "side_to_move": 0,
                "legal_moves": [0, 1, 2, 3, 4],
                "phase": "opening",
                "bucket": "capture_available",
                "tags": ["capture_available", "generated", "ply_4"],
                "source": "generated",
            },
            {
                "id": "capture_available-017",
                "state": {
                    "player_pits": [5, 1, 6, 5, 5, 0],
                    "opponent_pits": [1, 6, 6, 0, 6, 5],
                    "player_store": 1,
                    "opponent_store": 1,
                    "current_player": 0,
                },
                "side_to_move": 0,
                "legal_moves": [0, 1, 2, 3, 4],
                "phase": "opening",
                "bucket": "capture_available",
                "tags": ["capture_available", "generated", "ply_4"],
                "source": "generated",
            },
            {
                "id": "capture_available-018",
                "state": {
                    "player_pits": [5, 1, 5, 5, 0, 6],
                    "opponent_pits": [1, 6, 0, 7, 6, 5],
                    "player_store": 1,
                    "opponent_store": 1,
                    "current_player": 0,
                },
                "side_to_move": 0,
                "legal_moves": [0, 1, 2, 3, 5],
                "phase": "opening",
                "bucket": "capture_available",
                "tags": ["capture_available", "generated", "ply_4"],
                "source": "generated",
            },
        ]
        reference_rows = [
            {
                "id": "capture_available-016",
                "canonical_state": module.canonical_state_key(suite_rows[0]["state"]),
                "state": suite_rows[0]["state"],
                "reference_move": 4,
                "reference_unstable": False,
                "observed_reference_moves": [4],
                "seed_samples": [
                    {"seed": 2040, "reference_move": 4, "teacher_value": 0.42}
                ],
            },
            {
                "id": "capture_available-017",
                "canonical_state": module.canonical_state_key(suite_rows[1]["state"]),
                "state": suite_rows[1]["state"],
                "reference_move": 3,
                "reference_unstable": False,
                "observed_reference_moves": [3],
                "seed_samples": [
                    {"seed": 2040, "reference_move": 3, "teacher_value": 0.46}
                ],
            },
            {
                "id": "capture_available-018",
                "canonical_state": module.canonical_state_key(suite_rows[2]["state"]),
                "state": suite_rows[2]["state"],
                "reference_move": 4,
                "reference_unstable": False,
                "observed_reference_moves": [4],
                "seed_samples": [
                    {"seed": 2040, "reference_move": 4, "teacher_value": 0.41}
                ],
            },
        ]

        selected = module.select_opening_capture_family_rows(
            suite_rows=suite_rows, reference_rows=reference_rows
        )

        self.assertEqual(
            ["capture_available-016", "capture_available-017"],
            [row["id"] for row in selected],
        )
        self.assertEqual([4, 3], [row["reference_move"] for row in selected])
        self.assertEqual(
            ["opening_capture_family", "opening_capture_family"],
            [row["replay_role"] for row in selected],
        )

    def test_selects_only_stable_high_imbalance_rows_from_requested_ids(self):
        from ml.alphazero_lite import (
            build_tactical_stable_failure_family_replay as module,
        )

        suite_rows = [
            {
                "id": "high_imbalance-001",
                "state": {
                    "player_pits": [4, 4, 4, 4, 0, 5],
                    "opponent_pits": [0, 1, 6, 6, 6, 6],
                    "player_store": 1,
                    "opponent_store": 1,
                    "current_player": 0,
                },
                "side_to_move": 0,
                "legal_moves": [0, 1, 2, 3, 5],
                "phase": "midgame",
                "bucket": "high_imbalance",
                "tags": ["high_imbalance"],
                "source": "generated",
            },
            {
                "id": "high_imbalance-002",
                "state": {
                    "player_pits": [5, 4, 4, 4, 4, 0],
                    "opponent_pits": [5, 5, 0, 5, 5, 5],
                    "player_store": 1,
                    "opponent_store": 1,
                    "current_player": 0,
                },
                "side_to_move": 0,
                "legal_moves": [0, 1, 2, 3, 4],
                "phase": "midgame",
                "bucket": "high_imbalance",
                "tags": ["high_imbalance"],
                "source": "generated",
            },
            {
                "id": "high_imbalance-003",
                "state": {
                    "player_pits": [3, 5, 5, 5, 0, 5],
                    "opponent_pits": [0, 2, 6, 6, 6, 5],
                    "player_store": 2,
                    "opponent_store": 1,
                    "current_player": 0,
                },
                "side_to_move": 0,
                "legal_moves": [0, 1, 2, 3, 5],
                "phase": "midgame",
                "bucket": "high_imbalance",
                "tags": ["high_imbalance"],
                "source": "generated",
            },
        ]
        reference_rows = [
            {
                "id": "high_imbalance-001",
                "canonical_state": module.canonical_state_key(suite_rows[0]["state"]),
                "state": suite_rows[0]["state"],
                "reference_move": 2,
                "reference_unstable": False,
                "observed_reference_moves": [2],
                "seed_samples": [
                    {"seed": 2040, "reference_move": 2, "teacher_value": 0.57}
                ],
            },
            {
                "id": "high_imbalance-002",
                "canonical_state": module.canonical_state_key(suite_rows[1]["state"]),
                "state": suite_rows[1]["state"],
                "reference_move": None,
                "reference_unstable": True,
                "observed_reference_moves": [2, 4],
                "seed_samples": [
                    {"seed": 2040, "reference_move": 2, "teacher_value": 0.57},
                    {"seed": 3040, "reference_move": 4, "teacher_value": 0.51},
                ],
            },
            {
                "id": "high_imbalance-003",
                "canonical_state": module.canonical_state_key(suite_rows[2]["state"]),
                "state": suite_rows[2]["state"],
                "reference_move": 1,
                "reference_unstable": False,
                "observed_reference_moves": [1],
                "seed_samples": [
                    {"seed": 2040, "reference_move": 1, "teacher_value": 0.61}
                ],
            },
        ]

        selected = module.select_high_imbalance_stable_rows(
            suite_rows=suite_rows,
            reference_rows=reference_rows,
            selected_ids={"high_imbalance-001", "high_imbalance-002"},
        )

        self.assertEqual(["high_imbalance-001"], [row["id"] for row in selected])
        self.assertEqual(
            ["high_imbalance_stable"], [row["replay_role"] for row in selected]
        )

    def test_build_dataset_treats_empty_high_imbalance_ids_as_select_none(self):
        from ml.alphazero_lite import (
            build_tactical_stable_failure_family_replay as module,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tactical_replay_path = tmp_path / "tactical_replay.jsonl"
            suite_path = tmp_path / "suite.json"
            reference_path = tmp_path / "reference_moves.json"
            out_path = tmp_path / "stable_failure_replay.jsonl"
            summary_out_path = (
                tmp_path / "tactical_stable_failure_family_replay_summary.json"
            )

            source_rows = self.rebalance_source_rows()
            suite_rows = self.rebalance_suite_rows()
            reference_payload = self.rebalance_reference_payload(module)

            tactical_replay_path.write_text(
                "\n".join(json.dumps(row) for row in source_rows) + "\n",
                encoding="utf-8",
            )
            suite_path.write_text(json.dumps(suite_rows), encoding="utf-8")
            reference_path.write_text(json.dumps(reference_payload), encoding="utf-8")

            rows, summary = module.build_stable_failure_family_replay_dataset(
                tactical_replay_path=tactical_replay_path,
                suite_path=suite_path,
                reference_path=reference_path,
                regression_positions_path=module.DEFAULT_REGRESSION_POSITIONS,
                high_imbalance_ids=set(),
                out_path=out_path,
                summary_out_path=summary_out_path,
            )

        self.assertEqual(
            [],
            [
                row["id"]
                for row in rows
                if row["replay_role"] == "high_imbalance_stable"
            ],
        )
        self.assertEqual(0, summary["role_counts"]["high_imbalance_stable"])
        self.assertEqual(3, summary["shortfalls_by_role"]["high_imbalance_stable"])

    def test_bool_reference_move_is_not_treated_as_stable_reference(self):
        from ml.alphazero_lite import (
            build_tactical_stable_failure_family_replay as module,
        )

        suite_row = {
            "id": "capture_available-016",
            "state": {
                "player_pits": [5, 1, 5, 5, 5, 0],
                "opponent_pits": [1, 6, 0, 7, 6, 5],
                "player_store": 1,
                "opponent_store": 1,
                "current_player": 0,
            },
            "side_to_move": 0,
            "legal_moves": [0, 1, 2, 3, 4],
            "phase": "opening",
            "bucket": "capture_available",
            "tags": ["capture_available"],
            "source": "generated",
        }
        reference_rows = [
            {
                "id": "capture_available-016",
                "canonical_state": module.canonical_state_key(suite_row["state"]),
                "state": suite_row["state"],
                "reference_move": True,
                "reference_unstable": False,
                "observed_reference_moves": [True],
                "seed_samples": [
                    {"seed": 2040, "reference_move": True, "teacher_value": 0.42}
                ],
            }
        ]

        selected = module.select_opening_capture_family_rows(
            suite_rows=[suite_row], reference_rows=reference_rows
        )

        self.assertEqual([], selected)

    def test_build_dataset_rejects_missing_stable_failure_family_targets(self):
        from ml.alphazero_lite import (
            build_tactical_stable_failure_family_replay as module,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tactical_replay_path = tmp_path / "tactical_replay.jsonl"
            suite_path = tmp_path / "suite.json"
            reference_path = tmp_path / "reference_moves.json"
            out_path = tmp_path / "stable_failure_replay.jsonl"

            source_rows = [
                self.preservation_capture_row(
                    "preserve-capture-a", priority=7.0, variant="gain_4_5"
                ),
                self.preservation_capture_row(
                    "preserve-capture-b", priority=6.0, variant="gain_2_3"
                ),
            ]
            for bucket in ("high_imbalance", "high_value_swing", "starvation_pressure"):
                source_rows.extend(
                    self.nearby_row(bucket, index, priority=10 - index)
                    for index in range(2)
                )
            tactical_replay_path.write_text(
                "\n".join(json.dumps(row) for row in source_rows) + "\n",
                encoding="utf-8",
            )

            suite_rows = [
                {
                    "id": "capture_available-016",
                    "state": {
                        "player_pits": [5, 1, 5, 5, 5, 0],
                        "opponent_pits": [1, 6, 0, 7, 6, 5],
                        "player_store": 1,
                        "opponent_store": 1,
                        "current_player": 0,
                    },
                    "side_to_move": 0,
                    "legal_moves": [0, 1, 2, 3, 4],
                    "phase": "opening",
                    "bucket": "capture_available",
                    "tags": ["capture_available"],
                    "source": "generated",
                },
                {
                    "id": "high_imbalance-001",
                    "state": {
                        "player_pits": [4, 4, 4, 4, 0, 5],
                        "opponent_pits": [0, 1, 6, 6, 6, 6],
                        "player_store": 1,
                        "opponent_store": 1,
                        "current_player": 0,
                    },
                    "side_to_move": 0,
                    "legal_moves": [0, 1, 2, 3, 5],
                    "phase": "midgame",
                    "bucket": "high_imbalance",
                    "tags": ["high_imbalance"],
                    "source": "generated",
                },
            ]
            suite_path.write_text(json.dumps(suite_rows), encoding="utf-8")
            reference_payload = {
                "schema": "azlite_forensic_references_v1",
                "reference": {
                    "policy_simulations": 1200,
                    "value_simulations": 1800,
                    "sample_seeds": [2040],
                },
                "rows": [
                    {
                        "id": "capture_available-016",
                        "canonical_state": module.canonical_state_key(
                            suite_rows[0]["state"]
                        ),
                        "state": suite_rows[0]["state"],
                        "reference_move": None,
                        "reference_unstable": True,
                        "observed_reference_moves": [3, 4],
                        "seed_samples": [
                            {"seed": 2040, "reference_move": 3, "teacher_value": 0.42},
                            {"seed": 3040, "reference_move": 4, "teacher_value": 0.41},
                        ],
                    },
                    {
                        "id": "high_imbalance-001",
                        "canonical_state": module.canonical_state_key(
                            suite_rows[1]["state"]
                        ),
                        "state": suite_rows[1]["state"],
                        "reference_move": None,
                        "reference_unstable": True,
                        "observed_reference_moves": [1, 2],
                        "seed_samples": [
                            {"seed": 2040, "reference_move": 2, "teacher_value": 0.57},
                            {"seed": 3040, "reference_move": 1, "teacher_value": 0.51},
                        ],
                    },
                ],
            }
            reference_path.write_text(json.dumps(reference_payload), encoding="utf-8")
            summary_out_path = tmp_path / "stable_failure_replay_summary.json"

            with self.assertRaisesRegex(
                ValueError, "nearby_preservation does not exceed prior 6"
            ):
                module.build_stable_failure_family_replay_dataset(
                    tactical_replay_path=tactical_replay_path,
                    suite_path=suite_path,
                    reference_path=reference_path,
                    regression_positions_path=module.DEFAULT_REGRESSION_POSITIONS,
                    high_imbalance_ids={"high_imbalance-001"},
                    out_path=out_path,
                    summary_out_path=summary_out_path,
                )

            summary = json.loads(summary_out_path.read_text(encoding="utf-8"))

        self.assertEqual(0, summary["role_counts"]["opening_capture_family"])
        self.assertEqual(0, summary["role_counts"]["high_imbalance_stable"])
        self.assertEqual(1, summary["role_counts"]["capture_protection"])
        self.assertEqual(2, summary["role_counts"]["capture_preservation"])
        self.assertEqual(6, summary["role_counts"]["nearby_preservation"])
        self.assertEqual(7, summary["shortfalls_by_role"]["opening_capture_family"])
        self.assertEqual(3, summary["shortfalls_by_role"]["high_imbalance_stable"])
        self.assertIn(
            "nearby_preservation does not exceed prior 6", summary["invalid_reasons"]
        )

    def test_build_capture_protection_row_rejects_empty_regression_fixture(self):
        from ml.alphazero_lite import (
            build_tactical_stable_failure_family_replay as module,
        )

        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "empty_regression_positions.json"
            fixture_path.write_text(json.dumps([]), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "regression fixture must contain at least one position"
            ):
                module.build_capture_protection_row(fixture_path)

    def test_build_dataset_writes_explicit_stable_failure_roles(self):
        from ml.alphazero_lite import (
            build_tactical_stable_failure_family_replay as module,
        )
        from ml.alphazero_lite import train as train_module

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tactical_replay_path = tmp_path / "tactical_replay.jsonl"
            suite_path = tmp_path / "suite.json"
            reference_path = tmp_path / "reference_moves.json"
            out_path = tmp_path / "stable_failure_replay.jsonl"

            source_rows = [
                self.preservation_capture_row(
                    "preserve-capture-a", priority=7.0, variant="gain_4_5"
                ),
                self.preservation_capture_row(
                    "preserve-capture-b", priority=6.0, variant="gain_2_3"
                ),
            ]
            for bucket in ("high_imbalance", "high_value_swing", "starvation_pressure"):
                source_rows.extend(
                    self.nearby_row(bucket, index, priority=10 - index)
                    for index in range(2)
                )
            tactical_replay_path.write_text(
                "\n".join(json.dumps(row) for row in source_rows) + "\n",
                encoding="utf-8",
            )

            suite_rows = [
                {
                    "id": "capture_available-016",
                    "state": {
                        "player_pits": [5, 1, 5, 5, 5, 0],
                        "opponent_pits": [1, 6, 0, 7, 6, 5],
                        "player_store": 1,
                        "opponent_store": 1,
                        "current_player": 0,
                    },
                    "side_to_move": 0,
                    "legal_moves": [0, 1, 2, 3, 4],
                    "phase": "opening",
                    "bucket": "capture_available",
                    "tags": ["capture_available"],
                    "source": "generated",
                },
                {
                    "id": "high_imbalance-001",
                    "state": {
                        "player_pits": [4, 4, 4, 4, 0, 5],
                        "opponent_pits": [0, 1, 6, 6, 6, 6],
                        "player_store": 1,
                        "opponent_store": 1,
                        "current_player": 0,
                    },
                    "side_to_move": 0,
                    "legal_moves": [0, 1, 2, 3, 5],
                    "phase": "midgame",
                    "bucket": "high_imbalance",
                    "tags": ["high_imbalance"],
                    "source": "generated",
                },
            ]
            suite_path.write_text(json.dumps(suite_rows), encoding="utf-8")
            summary_out_path = tmp_path / "stable_failure_replay_summary.json"
            reference_payload = {
                "schema": "azlite_forensic_references_v1",
                "reference": {
                    "policy_simulations": 1200,
                    "value_simulations": 1800,
                    "sample_seeds": [2040],
                },
                "rows": [
                    {
                        "id": "capture_available-016",
                        "canonical_state": module.canonical_state_key(
                            suite_rows[0]["state"]
                        ),
                        "state": suite_rows[0]["state"],
                        "reference_move": 4,
                        "reference_unstable": False,
                        "observed_reference_moves": [4],
                        "seed_samples": [
                            {"seed": 2040, "reference_move": 4, "teacher_value": 0.42}
                        ],
                    },
                    {
                        "id": "high_imbalance-001",
                        "canonical_state": module.canonical_state_key(
                            suite_rows[1]["state"]
                        ),
                        "state": suite_rows[1]["state"],
                        "reference_move": 2,
                        "reference_unstable": False,
                        "observed_reference_moves": [2],
                        "seed_samples": [
                            {"seed": 2040, "reference_move": 2, "teacher_value": 0.57}
                        ],
                    },
                ],
            }
            reference_path.write_text(json.dumps(reference_payload), encoding="utf-8")
            summary_out_path = tmp_path / "stable_failure_replay_summary.json"

            with self.assertRaisesRegex(
                ValueError, "nearby_preservation does not exceed prior 6"
            ):
                module.build_stable_failure_family_replay_dataset(
                    tactical_replay_path=tactical_replay_path,
                    suite_path=suite_path,
                    reference_path=reference_path,
                    regression_positions_path=module.DEFAULT_REGRESSION_POSITIONS,
                    high_imbalance_ids={"high_imbalance-001"},
                    out_path=out_path,
                    summary_out_path=summary_out_path,
                )

            summary = json.loads(summary_out_path.read_text(encoding="utf-8"))
            persisted_rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
            ]
            x, p_target, v_target = train_module.load_jsonl(
                out_path,
                policy_target_mode="sharpened",
                value_target_mode="sharpened",
            )

        self.assertEqual(
            [
                "capture_protection",
                "capture_preservation",
                "capture_preservation",
                "opening_capture_family",
                "high_imbalance_stable",
            ]
            + ["nearby_preservation"] * 6,
            [row["replay_role"] for row in persisted_rows],
        )
        self.assertEqual(2, summary["shortfalls_by_role"]["high_imbalance_stable"])
        self.assertEqual(str(summary_out_path), summary["summary_artifact_path"])
        self.assertEqual(str(out_path), summary["replay_artifact_path"])
        self.assertEqual(6, summary["role_counts"]["nearby_preservation"])
        self.assertIn(
            "nearby_preservation does not exceed prior 6", summary["invalid_reasons"]
        )
        self.assertEqual((11, 27), x.shape)
        self.assertEqual((11, 6), p_target.shape)
        self.assertEqual((11, 1), v_target.shape)

    def test_build_dataset_records_shortfalls_when_source_rows_are_undersized(self):
        from ml.alphazero_lite import (
            build_tactical_stable_failure_family_replay as module,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tactical_replay_path = tmp_path / "tactical_replay.jsonl"
            suite_path = tmp_path / "suite.json"
            reference_path = tmp_path / "reference_moves.json"
            out_path = tmp_path / "stable_failure_replay.jsonl"

            tactical_replay_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            self.preservation_capture_row(
                                "preserve-capture-a", priority=7.0, variant="gain_4_5"
                            )
                        ),
                        json.dumps(
                            self.nearby_row("high_value_swing", 0, priority=5.0)
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            suite_rows = [
                {
                    "id": "capture_available-016",
                    "state": {
                        "player_pits": [5, 1, 5, 5, 5, 0],
                        "opponent_pits": [1, 6, 0, 7, 6, 5],
                        "player_store": 1,
                        "opponent_store": 1,
                        "current_player": 0,
                    },
                    "side_to_move": 0,
                    "legal_moves": [0, 1, 2, 3, 4],
                    "phase": "opening",
                    "bucket": "capture_available",
                    "tags": ["capture_available"],
                    "source": "generated",
                },
                {
                    "id": "high_imbalance-001",
                    "state": {
                        "player_pits": [4, 4, 4, 4, 0, 5],
                        "opponent_pits": [0, 1, 6, 6, 6, 6],
                        "player_store": 1,
                        "opponent_store": 1,
                        "current_player": 0,
                    },
                    "side_to_move": 0,
                    "legal_moves": [0, 1, 2, 3, 5],
                    "phase": "midgame",
                    "bucket": "high_imbalance",
                    "tags": ["high_imbalance"],
                    "source": "generated",
                },
            ]
            suite_path.write_text(json.dumps(suite_rows), encoding="utf-8")
            reference_payload = {
                "schema": "azlite_forensic_references_v1",
                "reference": {
                    "policy_simulations": 1200,
                    "value_simulations": 1800,
                    "sample_seeds": [2040],
                },
                "rows": [
                    {
                        "id": "capture_available-016",
                        "canonical_state": module.canonical_state_key(
                            suite_rows[0]["state"]
                        ),
                        "state": suite_rows[0]["state"],
                        "reference_move": 4,
                        "reference_unstable": False,
                        "observed_reference_moves": [4],
                        "seed_samples": [
                            {"seed": 2040, "reference_move": 4, "teacher_value": 0.42}
                        ],
                    },
                    {
                        "id": "high_imbalance-001",
                        "canonical_state": module.canonical_state_key(
                            suite_rows[1]["state"]
                        ),
                        "state": suite_rows[1]["state"],
                        "reference_move": 2,
                        "reference_unstable": False,
                        "observed_reference_moves": [2],
                        "seed_samples": [
                            {"seed": 2040, "reference_move": 2, "teacher_value": 0.57}
                        ],
                    },
                ],
            }
            reference_path.write_text(json.dumps(reference_payload), encoding="utf-8")
            summary_out_path = tmp_path / "stable_failure_replay_summary.json"

            with self.assertRaisesRegex(
                ValueError, "nearby_preservation does not exceed prior 6"
            ):
                module.build_stable_failure_family_replay_dataset(
                    tactical_replay_path=tactical_replay_path,
                    suite_path=suite_path,
                    reference_path=reference_path,
                    regression_positions_path=module.DEFAULT_REGRESSION_POSITIONS,
                    high_imbalance_ids={"high_imbalance-001"},
                    out_path=out_path,
                    summary_out_path=summary_out_path,
                )

            summary = json.loads(summary_out_path.read_text(encoding="utf-8"))
            rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(5, len(rows))
        self.assertEqual(1, summary["role_counts"]["capture_preservation"])
        self.assertEqual(3, summary["shortfalls_by_role"]["capture_preservation"])
        self.assertEqual(1, summary["role_counts"]["nearby_preservation"])
        self.assertEqual(7, summary["shortfalls_by_role"]["nearby_preservation"])
        self.assertIn(
            "nearby_preservation does not exceed prior 6", summary["invalid_reasons"]
        )

    def test_build_dataset_records_capture_preservation_shortfall_without_early_failure(
        self,
    ):
        from ml.alphazero_lite import (
            build_tactical_stable_failure_family_replay as module,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tactical_replay_path = tmp_path / "tactical_replay.jsonl"
            suite_path = tmp_path / "suite.json"
            reference_path = tmp_path / "reference_moves.json"
            out_path = tmp_path / "stable_failure_replay.jsonl"
            summary_out_path = (
                tmp_path / "tactical_stable_failure_family_replay_summary.json"
            )

            source_rows = [
                self.preservation_capture_row(
                    "preserve-capture-a", priority=9.0, variant="gain_4_5"
                ),
                self.preservation_capture_row(
                    "preserve-capture-b", priority=8.0, variant="gain_2_3"
                ),
            ]
            for bucket, count in (
                ("high_imbalance", 4),
                ("high_value_swing", 6),
                ("starvation_pressure", 2),
            ):
                for index in range(count):
                    source_rows.append(
                        self.nearby_row(bucket, index, priority=50 - index)
                    )

            tactical_replay_path.write_text(
                "\n".join(json.dumps(row) for row in source_rows) + "\n",
                encoding="utf-8",
            )
            suite_rows = self.rebalance_suite_rows()
            reference_payload = self.rebalance_reference_payload(module)
            suite_path.write_text(json.dumps(suite_rows), encoding="utf-8")
            reference_path.write_text(json.dumps(reference_payload), encoding="utf-8")

            rows, summary = module.build_stable_failure_family_replay_dataset(
                tactical_replay_path=tactical_replay_path,
                suite_path=suite_path,
                reference_path=reference_path,
                regression_positions_path=module.DEFAULT_REGRESSION_POSITIONS,
                out_path=out_path,
                summary_out_path=summary_out_path,
            )

        self.assertEqual(21, len(rows))
        self.assertEqual(2, summary["role_counts"]["capture_preservation"])
        self.assertEqual(2, summary["shortfalls_by_role"]["capture_preservation"])
        self.assertEqual([], summary["invalid_reasons"])

    def test_build_dataset_records_capture_preservation_duplicate_shape_collapse_without_rejecting_run(
        self,
    ):
        from ml.alphazero_lite import (
            build_tactical_stable_failure_family_replay as module,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tactical_replay_path = tmp_path / "tactical_replay.jsonl"
            suite_path = tmp_path / "suite.json"
            reference_path = tmp_path / "reference_moves.json"
            out_path = tmp_path / "stable_failure_replay.jsonl"
            summary_out_path = (
                tmp_path / "tactical_stable_failure_family_replay_summary.json"
            )

            source_rows = [
                self.preservation_capture_row(
                    "preserve-capture-a", priority=9.0, variant="gain_4_5"
                ),
                self.preservation_capture_row(
                    "preserve-capture-b", priority=8.0, variant="gain_2_3"
                ),
                self.preservation_capture_row(
                    "preserve-capture-c", priority=7.0, variant="gain_4_5"
                ),
                self.preservation_capture_row(
                    "preserve-capture-d", priority=6.0, variant="gain_2_3"
                ),
            ]
            for bucket, count in (
                ("high_imbalance", 2),
                ("high_value_swing", 4),
                ("starvation_pressure", 2),
            ):
                for index in range(count):
                    source_rows.append(
                        self.nearby_row(bucket, index, priority=50 - index)
                    )

            tactical_replay_path.write_text(
                "\n".join(json.dumps(row) for row in source_rows) + "\n",
                encoding="utf-8",
            )
            suite_rows = self.rebalance_suite_rows()
            reference_payload = self.rebalance_reference_payload(module)
            suite_path.write_text(json.dumps(suite_rows), encoding="utf-8")
            reference_path.write_text(json.dumps(reference_payload), encoding="utf-8")

            rows, summary = module.build_stable_failure_family_replay_dataset(
                tactical_replay_path=tactical_replay_path,
                suite_path=suite_path,
                reference_path=reference_path,
                regression_positions_path=module.DEFAULT_REGRESSION_POSITIONS,
                out_path=out_path,
                summary_out_path=summary_out_path,
            )

        self.assertEqual(23, len(rows))
        self.assertEqual(4, summary["role_counts"]["capture_preservation"])
        self.assertEqual(0, summary["shortfalls_by_role"]["capture_preservation"])
        self.assertEqual(2, summary["capture_preservation_distinct_shape_count"])
        self.assertEqual(2, summary["capture_preservation_shape_shortfall"])
        self.assertEqual(4, sum(summary["capture_preservation_shape_counts"].values()))
        self.assertNotIn(
            "capture_preservation shape diversity shortfall 2",
            summary["invalid_reasons"],
        )

    def test_build_dataset_records_nearby_zero_bucket_shortfall_without_early_failure(
        self,
    ):
        from ml.alphazero_lite import (
            build_tactical_stable_failure_family_replay as module,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tactical_replay_path = tmp_path / "tactical_replay.jsonl"
            suite_path = tmp_path / "suite.json"
            reference_path = tmp_path / "reference_moves.json"
            out_path = tmp_path / "stable_failure_replay.jsonl"
            summary_out_path = (
                tmp_path / "tactical_stable_failure_family_replay_summary.json"
            )

            source_rows = self.rebalance_source_rows(
                nearby_counts={
                    "high_imbalance": 0,
                    "high_value_swing": 4,
                    "starvation_pressure": 2,
                }
            )
            suite_rows = self.rebalance_suite_rows()
            reference_payload = self.rebalance_reference_payload(module)

            tactical_replay_path.write_text(
                "\n".join(json.dumps(row) for row in source_rows) + "\n",
                encoding="utf-8",
            )
            suite_path.write_text(json.dumps(suite_rows), encoding="utf-8")
            reference_path.write_text(json.dumps(reference_payload), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "nearby_preservation does not exceed prior 6"
            ):
                module.build_stable_failure_family_replay_dataset(
                    tactical_replay_path=tactical_replay_path,
                    suite_path=suite_path,
                    reference_path=reference_path,
                    regression_positions_path=module.DEFAULT_REGRESSION_POSITIONS,
                    out_path=out_path,
                    summary_out_path=summary_out_path,
                )

            summary = json.loads(summary_out_path.read_text(encoding="utf-8"))

        self.assertEqual(6, summary["role_counts"]["nearby_preservation"])
        self.assertEqual(2, summary["shortfalls_by_role"]["nearby_preservation"])
        self.assertIn(
            "nearby_preservation does not exceed prior 6", summary["invalid_reasons"]
        )
        self.assertEqual(0, summary["nearby_bucket_counts"]["high_imbalance"])
        self.assertIn("high_imbalance", summary["nearby_missing_buckets"])

    def test_build_dataset_rejects_full_nearby_count_when_bucket_missing(self):
        from ml.alphazero_lite import (
            build_tactical_stable_failure_family_replay as module,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tactical_replay_path = tmp_path / "tactical_replay.jsonl"
            suite_path = tmp_path / "suite.json"
            reference_path = tmp_path / "reference_moves.json"
            out_path = tmp_path / "stable_failure_replay.jsonl"
            summary_out_path = (
                tmp_path / "tactical_stable_failure_family_replay_summary.json"
            )

            source_rows = self.rebalance_source_rows(
                nearby_counts={
                    "high_imbalance": 0,
                    "high_value_swing": 4,
                    "starvation_pressure": 4,
                }
            )
            suite_rows = self.rebalance_suite_rows()
            reference_payload = self.rebalance_reference_payload(module)

            tactical_replay_path.write_text(
                "\n".join(json.dumps(row) for row in source_rows) + "\n",
                encoding="utf-8",
            )
            suite_path.write_text(json.dumps(suite_rows), encoding="utf-8")
            reference_path.write_text(json.dumps(reference_payload), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "nearby_preservation missing bucket participation: high_imbalance",
            ):
                module.build_stable_failure_family_replay_dataset(
                    tactical_replay_path=tactical_replay_path,
                    suite_path=suite_path,
                    reference_path=reference_path,
                    regression_positions_path=module.DEFAULT_REGRESSION_POSITIONS,
                    out_path=out_path,
                    summary_out_path=summary_out_path,
                )

            summary = json.loads(summary_out_path.read_text(encoding="utf-8"))

        self.assertEqual(8, summary["role_counts"]["nearby_preservation"])
        self.assertEqual(0, summary["shortfalls_by_role"]["nearby_preservation"])
        self.assertEqual(
            {
                "high_imbalance": 0,
                "high_value_swing": 4,
                "starvation_pressure": 4,
            },
            summary["nearby_bucket_counts"],
        )
        self.assertEqual(["high_imbalance"], summary["nearby_missing_buckets"])
        self.assertIn(
            "nearby_preservation missing bucket participation: high_imbalance",
            summary["invalid_reasons"],
        )

    def test_build_dataset_returns_rebalanced_23_row_mix_with_summary_diagnostics(self):
        from ml.alphazero_lite import (
            build_tactical_stable_failure_family_replay as module,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tactical_replay_path = tmp_path / "tactical_replay.jsonl"
            suite_path = tmp_path / "suite.json"
            reference_path = tmp_path / "reference_moves.json"
            out_path = tmp_path / "stable_failure_replay.jsonl"
            summary_out_path = (
                tmp_path / "tactical_stable_failure_family_replay_summary.json"
            )

            source_rows = self.rebalance_source_rows()
            suite_rows = self.rebalance_suite_rows()
            reference_payload = self.rebalance_reference_payload(module)

            tactical_replay_path.write_text(
                "\n".join(json.dumps(row) for row in source_rows) + "\n",
                encoding="utf-8",
            )
            suite_path.write_text(json.dumps(suite_rows), encoding="utf-8")
            reference_path.write_text(json.dumps(reference_payload), encoding="utf-8")

            rows, summary = module.build_stable_failure_family_replay_dataset(
                tactical_replay_path=tactical_replay_path,
                suite_path=suite_path,
                reference_path=reference_path,
                regression_positions_path=module.DEFAULT_REGRESSION_POSITIONS,
                out_path=out_path,
                summary_out_path=summary_out_path,
            )
            self.assertTrue(summary_out_path.exists())
            summary_payload = json.loads(summary_out_path.read_text(encoding="utf-8"))

        actual_role_counts = {
            "capture_protection": 1,
            "capture_preservation": 4,
            "opening_capture_family": 7,
            "high_imbalance_stable": 3,
            "nearby_preservation": 8,
        }
        self.assertEqual(23, len(rows))
        self.assertEqual(
            actual_role_counts,
            summary["target_counts"],
        )
        self.assertEqual(actual_role_counts, summary["role_counts"])
        self.assertEqual(
            ["high_imbalance-001", "high_imbalance-002", "high_imbalance-003"],
            [
                row["id"]
                for row in rows
                if row["replay_role"] == "high_imbalance_stable"
            ],
        )
        self.assertEqual(
            {
                "capture_protection": [
                    row["id"]
                    for row in rows
                    if row["replay_role"] == "capture_protection"
                ],
                "capture_preservation": [
                    row["canonical_state"]
                    for row in rows
                    if row["replay_role"] == "capture_preservation"
                ],
                "opening_capture_family": [
                    row["id"]
                    for row in rows
                    if row["replay_role"] == "opening_capture_family"
                ],
                "high_imbalance_stable": [
                    row["id"]
                    for row in rows
                    if row["replay_role"] == "high_imbalance_stable"
                ],
                "nearby_preservation": [
                    row["canonical_state"]
                    for row in rows
                    if row["replay_role"] == "nearby_preservation"
                ],
            },
            summary["selected_ids_by_role"],
        )
        self.assertEqual(
            {
                "capture_protection": 0,
                "capture_preservation": 0,
                "opening_capture_family": 0,
                "high_imbalance_stable": 0,
                "nearby_preservation": 0,
            },
            summary["shortfalls_by_role"],
        )
        self.assertEqual(4, summary["capture_preservation_distinct_shape_count"])
        self.assertEqual(0, summary["capture_preservation_shape_shortfall"])
        self.assertEqual(
            {
                "high_imbalance": 2,
                "high_value_swing": 4,
                "starvation_pressure": 2,
            },
            summary["nearby_bucket_counts"],
        )
        self.assertEqual([], summary["nearby_missing_buckets"])
        self.assertEqual([], summary["invalid_reasons"])
        self.assertEqual(str(summary_out_path), summary["summary_artifact_path"])
        self.assertEqual(str(out_path), summary["replay_artifact_path"])
        self.assertEqual(summary, summary_payload)

        nearby_rows = [
            row for row in rows if row["replay_role"] == "nearby_preservation"
        ]
        nearby_bucket_counts = {
            bucket: sum(1 for row in nearby_rows if row["bucket"] == bucket)
            for bucket in ("high_imbalance", "high_value_swing", "starvation_pressure")
        }
        self.assertGreaterEqual(nearby_bucket_counts["high_imbalance"], 2)
        self.assertGreaterEqual(nearby_bucket_counts["high_value_swing"], 2)
        self.assertGreaterEqual(nearby_bucket_counts["starvation_pressure"], 2)
        self.assertEqual(
            [
                "high_value_swing-2",
                "high_value_swing-3",
            ],
            sorted(
                row["canonical_state"]
                for row in nearby_rows
                if row["bucket"] == "high_value_swing"
                and row["canonical_state"]
                in {"high_value_swing-2", "high_value_swing-3"}
            ),
        )

    def test_build_dataset_writes_summary_before_raising_when_opening_family_remains_dominant(
        self,
    ):
        from ml.alphazero_lite import (
            build_tactical_stable_failure_family_replay as module,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tactical_replay_path = tmp_path / "tactical_replay.jsonl"
            suite_path = tmp_path / "suite.json"
            reference_path = tmp_path / "reference_moves.json"
            out_path = tmp_path / "stable_failure_replay.jsonl"
            summary_out_path = (
                tmp_path / "tactical_stable_failure_family_replay_summary.json"
            )

            source_rows = self.rebalance_source_rows(
                nearby_counts={
                    "high_imbalance": 2,
                    "high_value_swing": 2,
                    "starvation_pressure": 2,
                }
            )
            suite_rows = self.rebalance_suite_rows()
            reference_payload = self.rebalance_reference_payload(module)

            tactical_replay_path.write_text(
                "\n".join(json.dumps(row) for row in source_rows) + "\n",
                encoding="utf-8",
            )
            suite_path.write_text(json.dumps(suite_rows), encoding="utf-8")
            reference_path.write_text(json.dumps(reference_payload), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "opening_capture_family remains dominant|nearby_preservation does not exceed prior 6",
            ):
                module.build_stable_failure_family_replay_dataset(
                    tactical_replay_path=tactical_replay_path,
                    suite_path=suite_path,
                    reference_path=reference_path,
                    regression_positions_path=module.DEFAULT_REGRESSION_POSITIONS,
                    out_path=out_path,
                    summary_out_path=summary_out_path,
                )

            self.assertTrue(summary_out_path.exists())
            summary_payload = json.loads(summary_out_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {
                    "capture_protection": 1,
                    "capture_preservation": 4,
                    "opening_capture_family": 7,
                    "high_imbalance_stable": 3,
                    "nearby_preservation": 6,
                },
                summary_payload["role_counts"],
            )
            self.assertEqual(
                {
                    "capture_protection": 0,
                    "capture_preservation": 0,
                    "opening_capture_family": 0,
                    "high_imbalance_stable": 0,
                    "nearby_preservation": 2,
                },
                summary_payload["shortfalls_by_role"],
            )
            self.assertIn(
                "opening_capture_family remains dominant",
                summary_payload["invalid_reasons"],
            )
            self.assertIn(
                "nearby_preservation does not exceed prior 6",
                summary_payload["invalid_reasons"],
            )

    def test_build_dataset_records_nearby_shortfall_without_early_failure(self):
        from ml.alphazero_lite import (
            build_tactical_stable_failure_family_replay as module,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tactical_replay_path = tmp_path / "tactical_replay.jsonl"
            suite_path = tmp_path / "suite.json"
            reference_path = tmp_path / "reference_moves.json"
            out_path = tmp_path / "stable_failure_replay.jsonl"
            summary_out_path = (
                tmp_path / "tactical_stable_failure_family_replay_summary.json"
            )

            source_rows = self.rebalance_source_rows(
                nearby_counts={
                    "high_imbalance": 2,
                    "high_value_swing": 3,
                    "starvation_pressure": 2,
                }
            )
            suite_rows = self.rebalance_suite_rows()
            reference_payload = self.rebalance_reference_payload(module)

            tactical_replay_path.write_text(
                "\n".join(json.dumps(row) for row in source_rows) + "\n",
                encoding="utf-8",
            )
            suite_path.write_text(json.dumps(suite_rows), encoding="utf-8")
            reference_path.write_text(json.dumps(reference_payload), encoding="utf-8")

            rows, summary = module.build_stable_failure_family_replay_dataset(
                tactical_replay_path=tactical_replay_path,
                suite_path=suite_path,
                reference_path=reference_path,
                regression_positions_path=module.DEFAULT_REGRESSION_POSITIONS,
                out_path=out_path,
                summary_out_path=summary_out_path,
            )

        self.assertEqual(22, len(rows))
        self.assertEqual(7, summary["role_counts"]["nearby_preservation"])
        self.assertEqual(1, summary["shortfalls_by_role"]["nearby_preservation"])
        self.assertEqual([], summary["invalid_reasons"])

    def test_parse_args_requires_summary_out(self):
        from ml.alphazero_lite import (
            build_tactical_stable_failure_family_replay as module,
        )

        with mock.patch.object(
            sys,
            "argv",
            [
                "build_tactical_stable_failure_family_replay.py",
                "--reference",
                "reference.json",
                "--out",
                "replay.jsonl",
            ],
        ):
            with self.assertRaises(SystemExit):
                module.parse_args()

    def test_main_treats_omitted_high_imbalance_ids_as_none(self):
        from ml.alphazero_lite import (
            build_tactical_stable_failure_family_replay as module,
        )

        args = mock.Mock(
            tactical_replay="tactical.jsonl",
            suite="suite.json",
            reference="reference.json",
            regression_positions="regression.json",
            high_imbalance_ids=[],
            out="out.jsonl",
            summary_out="summary.json",
        )

        with (
            mock.patch.object(module, "parse_args", return_value=args),
            mock.patch.object(
                module, "build_stable_failure_family_replay_dataset"
            ) as build_dataset,
        ):
            module.main()

        build_dataset.assert_called_once()
        self.assertIsNone(build_dataset.call_args.kwargs["high_imbalance_ids"])
