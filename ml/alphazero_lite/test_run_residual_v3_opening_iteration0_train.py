import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import (  # noqa: E402
    run_residual_v3_opening_iteration0_train as iteration0_train,
)


class ResidualV3OpeningIteration0TrainTest(unittest.TestCase):
    def _state(self) -> dict:
        return {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }

    def _search_profile(self) -> dict:
        return iteration0_train.build_promoted_search_profile()

    def _search_row(self, *, label: str, budget_pair: str, c_puct: float) -> dict:
        return {
            "label": label,
            "budget_pair": budget_pair,
            "simulations": int(budget_pair.split(":", 1)[0]),
            "role_context": [f"{budget_pair}/shared"],
            "c_puct": c_puct,
            "selected_move": 2,
            "legal_moves": [0, 1, 2, 3, 4, 5],
            "top_share": 0.6,
            "margin": 0.2,
            "entropy": 1.2,
            "root_value": 0.25,
            "search_policy": [0.05, 0.1, 0.45, 0.2, 0.1, 0.1],
        }

    def _selected_row(self) -> dict:
        state = self._state()
        return {
            "state": state,
            "state_hash": iteration0_train.canonical_state_hash(state),
            "prefix_moves": [2],
            "prefix_hash": "prefix-hash",
            "ply": 1,
            "ply_label": "1",
            "phase_bucket": "early",
            "side_to_move": 0,
            "side_to_move_label": "0",
            "first_move_family": "2",
            "selection_rank": 1,
            "selection_tags": ["weak_search"],
            "selection_metrics": {"weak": True, "unstable": False},
            "search_results": [
                self._search_row(
                    label="sim384_default", budget_pair="384:256", c_puct=1.25
                ),
                self._search_row(
                    label="sim768_equal_override", budget_pair="768:768", c_puct=0.9
                ),
                self._search_row(
                    label="sim1200_default", budget_pair="1200:1200", c_puct=1.25
                ),
            ],
        }

    def _write_manifest_and_rows(
        self, tmp_path: Path, *, selected_rows: list[dict] | None = None
    ) -> tuple[Path, Path, dict]:
        rows = selected_rows or [self._selected_row()]
        rows_path = tmp_path / "iteration0_selected_positions.jsonl"
        iteration0_train.write_jsonl(rows_path, rows)
        manifest = {
            "schema": iteration0_train.PREFLIGHT_SCHEMA,
            "classification": "residual_v3_opening_iteration0_preflight",
            "current_artifact": {
                "path": "model-artifact/current",
                "weights_sha256": "current-sha",
            },
            "input_suites": [
                {"path": str(rows_path), "sha256": "suite-sha", "rows": 1}
            ],
            "search_profile": self._search_profile(),
            "search_profile_hash": iteration0_train.search_profile_hash(
                self._search_profile()
            ),
            "training_data": {
                "path": str(rows_path),
                "rows": len(rows),
                "sha256": iteration0_train.sha256_file(rows_path),
                "format": "jsonl",
            },
        }
        manifest_path = tmp_path / "iteration0_training_manifest.json"
        iteration0_train.write_json(manifest_path, manifest)
        return manifest_path, rows_path, manifest

    def _write_current_artifact(
        self, tmp_path: Path, *, model_type: str = "residual_v3"
    ) -> Path:
        artifact_dir = tmp_path / "current"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "weights.json").write_text("{}", encoding="utf-8")
        metadata = {
            "input_encoding": "kalah_v3",
            "architecture": {
                "model_type": model_type,
                "trunk_size": 96,
                "residual_block_count": 3,
            },
        }
        (artifact_dir / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return artifact_dir

    def test_verify_manifest_accepts_matching_selected_rows_sha(self):
        with tempfile.TemporaryDirectory(prefix="azlite-iter0-train-") as tmp:
            tmp_path = Path(tmp)
            manifest_path, rows_path, _manifest = self._write_manifest_and_rows(
                tmp_path
            )

            manifest, verified_rows_path = iteration0_train.verify_preflight_manifest(
                manifest_path=manifest_path
            )

            self.assertEqual(str(rows_path), str(verified_rows_path))
            self.assertEqual(iteration0_train.PREFLIGHT_SCHEMA, manifest["schema"])

    def test_verify_manifest_rejects_selected_position_sha_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="azlite-iter0-train-") as tmp:
            tmp_path = Path(tmp)
            manifest_path, rows_path, _manifest = self._write_manifest_and_rows(
                tmp_path
            )
            rows_path.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(
                RuntimeError, "selected positions hash mismatch"
            ):
                iteration0_train.verify_preflight_manifest(manifest_path=manifest_path)

    def test_target_jsonl_is_train_py_compatible(self):
        with tempfile.TemporaryDirectory(prefix="azlite-iter0-train-") as tmp:
            tmp_path = Path(tmp)
            selected_row = self._selected_row()
            rows = iteration0_train.materialize_target_rows(
                selected_rows=[selected_row],
                input_encoding="kalah_v3",
                preferred_target_lane_label="sim768_equal_override",
                search_profile_hash_value="profile-sha",
            )
            rows_path = tmp_path / "targets.jsonl"
            iteration0_train.write_jsonl(rows_path, rows)

            compatibility = iteration0_train.verify_train_compatibility(rows_path)

            self.assertEqual(1, compatibility["rows"])
            self.assertEqual(27, compatibility["feature_count"])
            self.assertEqual(6, compatibility["policy_size"])
            self.assertEqual("profile-sha", rows[0]["search_profile_hash"])
            self.assertEqual("sim768_equal_override", rows[0]["target_lane_label"])

    def test_target_generation_is_deterministic(self):
        selected_rows = [self._selected_row()]

        first = iteration0_train.materialize_target_rows(
            selected_rows=selected_rows,
            input_encoding="kalah_v3",
            preferred_target_lane_label="sim768_equal_override",
            search_profile_hash_value="hash-a",
        )
        second = iteration0_train.materialize_target_rows(
            selected_rows=selected_rows,
            input_encoding="kalah_v3",
            preferred_target_lane_label="sim768_equal_override",
            search_profile_hash_value="hash-a",
        )

        self.assertEqual(first, second)
        self.assertAlmostEqual(1.0, sum(first[0]["policy"]))
        self.assertEqual(0.9, first[0]["target_c_puct"])

    def test_verify_current_artifact_enforces_residual_v3_and_current_sha(self):
        with tempfile.TemporaryDirectory(prefix="azlite-iter0-train-") as tmp:
            tmp_path = Path(tmp)
            manifest_path, _rows_path, manifest = self._write_manifest_and_rows(
                tmp_path
            )
            current_artifact = self._write_current_artifact(
                tmp_path, model_type="residual_v3"
            )
            actual_sha = iteration0_train.sha256_file(current_artifact / "weights.json")
            manifest["current_artifact"]["weights_sha256"] = actual_sha
            iteration0_train.write_json(manifest_path, manifest)

            metadata = iteration0_train.verify_current_artifact(
                current_path=current_artifact,
                expected_weights_sha256=actual_sha,
                manifest=manifest,
            )
            self.assertEqual("residual_v3", metadata["architecture"]["model_type"])

            bad_artifact = self._write_current_artifact(
                tmp_path / "bad", model_type="residual_v4_move_factorized"
            )
            with self.assertRaisesRegex(RuntimeError, "model_type"):
                iteration0_train.verify_current_artifact(
                    current_path=bad_artifact,
                    expected_weights_sha256=iteration0_train.sha256_file(
                        bad_artifact / "weights.json"
                    ),
                    manifest={
                        **manifest,
                        "current_artifact": {
                            "path": str(bad_artifact),
                            "weights_sha256": iteration0_train.sha256_file(
                                bad_artifact / "weights.json"
                            ),
                        },
                    },
                )

    def test_manifest_guardrails_reject_transforms_and_tablebase(self):
        with tempfile.TemporaryDirectory(prefix="azlite-iter0-train-") as tmp:
            tmp_path = Path(tmp)
            manifest_path, _rows_path, manifest = self._write_manifest_and_rows(
                tmp_path
            )
            manifest["search_profile"]["root_prior_transform"] = {"name": "danger"}
            iteration0_train.write_json(manifest_path, manifest)

            with self.assertRaisesRegex(RuntimeError, "root_prior_transform"):
                iteration0_train.verify_preflight_manifest(manifest_path=manifest_path)


if __name__ == "__main__":
    unittest.main()
