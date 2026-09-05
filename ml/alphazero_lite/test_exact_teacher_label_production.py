"""Unit/integration tests for production native-hybrid exact labeling."""

import json
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite import exact_teacher_labeling as exact
from ml.alphazero_lite.forensic_suite import canonical_state_key


def _source(
    state: dict | None = None,
    *,
    source_id: str = "exact-prod-frozen-17-24-00000",
    move_index: int = 3,
) -> dict:
    raw = state or {
        "player_pits": [4, 4, 4, 4, 4, 4],
        "opponent_pits": [4, 4, 4, 4, 4, 4],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    return {
        "source_id": source_id,
        "state": dict(raw),
        "canonical_state": canonical_state_key(raw),
        "active_stones": 48,
        "stones_bucket": "33-40",
        "phase": "early",
        "move_index": move_index,
        "player": raw["current_player"],
        "legal_moves": [0, 1, 2, 3, 4, 5],
        "training_eligible": True,
        "provenance": {"generator": "test"},
    }


PROVENANCE = {"identity": "native_hybrid_exact", "tablebase_artifact": "x"}


class ExactConversionTest(unittest.TestCase):
    def test_uniform_policy_over_multiple_optima(self) -> None:
        policy = exact.policy_target_from_optimal(
            optimal_actions=[1, 5], legal_moves=[1, 2, 3, 5]
        )
        self.assertEqual([0.0, 0.5, 0.0, 0.0, 0.0, 0.5], policy)
        self.assertAlmostEqual(1.0, sum(policy))

    def test_single_optimum_is_one_hot_on_legal_action(self) -> None:
        policy = exact.policy_target_from_optimal(
            optimal_actions=[2], legal_moves=[0, 2, 4]
        )
        self.assertEqual(1.0, policy[2])
        self.assertEqual(0.0, policy[0] + policy[1] + policy[3] + policy[4] + policy[5])

    def test_illegal_optimal_action_rejected(self) -> None:
        with self.assertRaises(exact.ExactLabelError):
            exact.policy_target_from_optimal(optimal_actions=[4], legal_moves=[0, 1, 2])

    def test_empty_optimal_set_rejected(self) -> None:
        with self.assertRaises(exact.ExactLabelError):
            exact.policy_target_from_optimal(optimal_actions=[], legal_moves=[0])

    def test_illegal_action_mass_rejected_by_train_validator(self) -> None:
        import numpy as np

        from ml.alphazero_lite import train as train_module
        from ml.alphazero_lite.self_play import encode_state

        state = {
            "player_pits": [4, 0, 0, 0, 0, 0],
            "opponent_pits": [0, 0, 0, 0, 0, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        encoded = encode_state(state, input_encoding="kalah_v3")
        bad = np.zeros(6, dtype=np.float32)
        bad[5] = 1.0  # pit 5 is empty -> illegal
        with self.assertRaises(ValueError):
            train_module.validate_policy_target(
                bad,
                state=encoded,
                path=Path("test.jsonl"),
                row_number=1,
                policy_target_mode="default",
                declared_mode="default",
            )

    def test_produced_rows_pass_train_validation(self) -> None:
        import numpy as np

        from ml.alphazero_lite import train as train_module

        source = _source()
        row = exact.build_training_row(
            source=source,
            action_values={0: 1, 1: 3, 2: 0, 3: -1, 4: 3, 5: 2},
            optimal_actions=[1, 4],
            exact_value=3,
            label_wall_seconds=0.01,
            label_cpu_seconds=0.01,
            teacher_provenance=PROVENANCE,
        )
        policy = np.asarray(row["policy"], dtype=np.float32)
        train_module.validate_policy_target(
            policy,
            state=row["state"],
            path=Path("test.jsonl"),
            row_number=1,
            policy_target_mode="default",
            declared_mode=row["policy_target_mode"],
        )
        self.assertEqual("default", row["value_target_mode"])
        self.assertIn(row["value"], (-1.0, 0.0, 1.0))
        # raw exact margins preserved verbatim
        self.assertEqual(3, row["exact_root_margin"])
        self.assertEqual([1, 4], row["exact_optimal_actions"])

    def test_value_sign_player_zero_winning_margin(self) -> None:
        # +6 for player zero: player 0 to move sees a win, player 1 a loss.
        self.assertEqual(
            1.0, exact.exact_margin_to_training_value(exact_margin=6, current_player=0)
        )
        self.assertEqual(
            -1.0, exact.exact_margin_to_training_value(exact_margin=6, current_player=1)
        )
        # draw stays a draw for both sides to move
        self.assertEqual(
            0.0, exact.exact_margin_to_training_value(exact_margin=0, current_player=0)
        )
        self.assertEqual(
            0.0, exact.exact_margin_to_training_value(exact_margin=0, current_player=1)
        )
        # negative margin is a player-1 win
        self.assertEqual(
            -1.0,
            exact.exact_margin_to_training_value(exact_margin=-4, current_player=0),
        )
        self.assertEqual(
            1.0, exact.exact_margin_to_training_value(exact_margin=-4, current_player=1)
        )

    def test_pr281_warm_fixture_value_mapping(self) -> None:
        # First row of the frozen PR #281 warm report: player 1 to move,
        # player-zero margin -4 => root player 1 wins => +1.0.
        fixture = json.loads(
            Path("docs/data/alphazero-lite-kalah-v1-tier18-hybrid-warm.json").read_text(
                encoding="utf-8"
            )
        )["corpus"][0]
        self.assertEqual(1, fixture["state"]["current_player"])
        self.assertEqual(-4, fixture["final_margin"])
        self.assertEqual(
            1.0,
            exact.exact_margin_to_training_value(
                exact_margin=fixture["final_margin"],
                current_player=fixture["state"]["current_player"],
            ),
        )

    def test_native_semantics_player_one_min(self) -> None:
        root = exact.validate_native_label(
            action_values={1: -4, 2: 4, 3: -2, 5: -4},
            optimal_actions=[1, 5],
            exact_value=-4,
            legal_moves=[1, 2, 3, 5],
            current_player=1,
        )
        self.assertEqual(-4, root)
        with self.assertRaises(exact.ExactLabelError):
            exact.validate_native_label(
                action_values={1: -4, 2: 4},
                optimal_actions=[1],
                exact_value=4,  # wrong extremum for player 1
                legal_moves=[1, 2],
                current_player=1,
            )

    def test_native_semantics_wrong_optimal_set_rejected(self) -> None:
        with self.assertRaises(exact.ExactLabelError):
            exact.validate_native_label(
                action_values={0: 1, 1: 3, 2: 3},
                optimal_actions=[1],  # missing tied action 2
                exact_value=3,
                legal_moves=[0, 1, 2],
                current_player=0,
            )

    def test_native_semantics_illegal_values_rejected(self) -> None:
        with self.assertRaises(exact.ExactLabelError):
            exact.validate_native_label(
                action_values={0: 1, 4: 2},  # 4 not legal
                optimal_actions=[4],
                exact_value=2,
                legal_moves=[0, 1],
                current_player=0,
            )

    def test_deterministic_labeling_same_inputs(self) -> None:
        kwargs: dict = {
            "source": _source(),
            "action_values": {0: 1, 1: 3, 2: 0, 3: -1, 4: 3, 5: 2},
            "optimal_actions": [1, 4],
            "exact_value": 3,
            "label_wall_seconds": 0.01,
            "label_cpu_seconds": 0.005,
            "teacher_provenance": PROVENANCE,
        }
        first = exact.build_training_row(**kwargs)
        second = exact.build_training_row(**kwargs)
        self.assertEqual(first, second)
        # repeat sample hash is stable
        self.assertEqual(exact.sha256_rows([first]), exact.sha256_rows([second]))

    def test_timeout_failure_row_is_explicit_no_fallback(self) -> None:
        row = exact.build_failed_row(
            source=_source(),
            status="timeout",
            error="native request exceeded timeout",
            teacher_provenance=PROVENANCE,
        )
        self.assertEqual("timeout", row["status"])
        self.assertFalse(row["exact"])
        self.assertNotIn("policy", row)
        self.assertNotIn("value", row)

    def test_train_holdout_disjointness(self) -> None:
        rows = [
            {**_source(source_id=f"id-{i:04d}"), "canonical_state": f"key-{i:04d}"}
            for i in range(100)
        ]
        train_rows, holdout_rows = exact.deterministic_split(rows, train_split=0.8)
        self.assertEqual(80, len(train_rows))
        self.assertEqual(20, len(holdout_rows))
        train_keys = {r["canonical_state"] for r in train_rows}
        holdout_keys = {r["canonical_state"] for r in holdout_rows}
        self.assertFalse(train_keys & holdout_keys)
        # deterministic: same input order-independent split
        again_train, _ = exact.deterministic_split(
            list(reversed(rows)), train_split=0.8
        )
        self.assertEqual(train_keys, {r["canonical_state"] for r in again_train})

    def test_freeze_excludes_feasibility_corpus(self) -> None:
        feasibility = exact.load_feasibility_corpus_keys()
        self.assertEqual(96, len(feasibility))
        rows, _ = exact.freeze_source_cohort(
            seed=12345,
            per_bucket=8,
            exclusion_keys={"feasibility_corpus": feasibility},
        )
        keys = {r["canonical_state"] for r in rows}
        self.assertEqual(len(rows), len(keys))
        self.assertFalse(keys & feasibility)

    def test_freeze_rows_unique_and_stratified(self) -> None:
        rows, stats = exact.freeze_source_cohort(seed=999, per_bucket=8)
        self.assertEqual(32, len(rows))
        self.assertEqual(
            {"8-16": 8, "17-24": 8, "25-32": 8, "33-40": 8},
            stats["rows_per_bucket"],
        )
        self.assertEqual(32, len({r["canonical_state"] for r in rows}))
        self.assertTrue(all(r["training_eligible"] for r in rows))

    def test_audit_metrics_primary_set_membership(self) -> None:
        pairs = [
            {
                "exact_optimal_actions": [1, 4],
                "mcts_top": 4,  # tied optimum, not the argmax single
                "mcts_value": 0.2,
                "exact_value": 1.0,
                "mcts_top_visit_share": 0.4,
                "bucket": "17-24",
            },
            {
                "exact_optimal_actions": [2],
                "mcts_top": 3,
                "mcts_value": -0.1,
                "exact_value": 1.0,
                "mcts_top_visit_share": 0.7,
                "bucket": "17-24",
            },
        ]
        metrics = exact.compute_audit_metrics(pairs)
        self.assertEqual(0.5, metrics["mcts_top_in_exact_optimal_set_rate"])
        self.assertEqual(0.5, metrics["exact_multi_optimum_rate"])
        self.assertIn("17-24", metrics["by_bucket"])

    def test_provenance_and_hash_fields_present(self) -> None:
        row = exact.build_training_row(
            source=_source(),
            action_values={0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2},
            optimal_actions=[0, 1, 2, 3, 4, 5],
            exact_value=2,
            label_wall_seconds=0.02,
            label_cpu_seconds=0.01,
            teacher_provenance={
                "identity": "native_hybrid_exact",
                "tablebase_artifact_sha256": "abc123",
                "solver_timeout_seconds": 30.0,
            },
        )
        self.assertEqual("native_hybrid_exact", row["teacher"])
        self.assertEqual("abc123", row["teacher_tablebase_artifact_sha256"])
        self.assertEqual([1 / 6] * 6, row["policy"])
        self.assertIn("source_provenance", row)
        self.assertIn("exact_action_margins", row)


class ResumeBehaviorTest(unittest.TestCase):
    def test_completed_ids_survive_across_membership_scans(self) -> None:
        from ml.alphazero_lite import run_exact_teacher_label_production as runner

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            holdout = root / "holdout.jsonl"
            failures = root / "failures.jsonl"
            first = {
                "source_id": "id-0001",
                "canonical_state": "k1",
                "policy": [1.0, 0, 0, 0, 0, 0],
                "value": 1.0,
                "label_wall_seconds": 0.1,
                "label_cpu_seconds": 0.05,
            }
            bad = {
                "source_id": "id-0002",
                "canonical_state": "k2",
                "status": "timeout",
                "error": "slow",
                "exact": False,
            }
            exact.write_jsonl(train, [first])
            exact.write_jsonl(failures, [bad])
            completed, existing = runner.split_membership(train, holdout, failures)
            self.assertEqual({"id-0001": "train", "id-0002": "failures"}, completed)
            # second scan does not duplicate or mutate
            completed2, existing2 = runner.split_membership(train, holdout, failures)
            self.assertEqual(completed, completed2)
            self.assertEqual(existing, existing2)


if __name__ == "__main__":
    unittest.main()
