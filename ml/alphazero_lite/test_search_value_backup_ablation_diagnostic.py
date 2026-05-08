import io
import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from ml.alphazero_lite import search_interaction_diagnostic as module


class SearchValueBackupAblationFixtureTest(unittest.TestCase):
    FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "diagnostics" / "search_value_backup_ablation"
    MANIFEST_PATH = FIXTURE_DIR / "manifest.json"
    SEARCH_INTERACTION_FIXTURE_PATH = FIXTURE_DIR / "search_interaction_diagnostic.json"
    SEARCH_VALUE_INTERACTION_FIXTURE_PATH = FIXTURE_DIR / "search_value_interaction_diagnostic.json"
    ORIGINAL_SEARCH_INTERACTION_PATH = "/tmp/opencode/stable-failure-family-runs/tactical-stable-failure-family-rebalance-20260508/final/search_interaction_diagnostic.json"
    ORIGINAL_SEARCH_VALUE_INTERACTION_PATH = "/tmp/opencode/stable-failure-family-runs/tactical-stable-failure-family-rebalance-20260508/final/search_value_interaction_diagnostic.json"

    def _fixture_sha256(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_fixture_manifest_preserves_tracked_copies_and_original_sources(self):
        self.assertTrue(self.MANIFEST_PATH.exists(), f"missing fixture manifest: {self.MANIFEST_PATH}")
        self.assertTrue(
            self.SEARCH_INTERACTION_FIXTURE_PATH.exists(),
            f"missing fixture copy: {self.SEARCH_INTERACTION_FIXTURE_PATH}",
        )
        self.assertTrue(
            self.SEARCH_VALUE_INTERACTION_FIXTURE_PATH.exists(),
            f"missing fixture copy: {self.SEARCH_VALUE_INTERACTION_FIXTURE_PATH}",
        )

        manifest = json.loads(self.MANIFEST_PATH.read_text(encoding="utf-8"))

        self.assertEqual("azlite_search_value_backup_ablation_fixture_manifest_v1", manifest["schema"])
        self.assertEqual(
            {
                "search_interaction_diagnostic": {
                    "path": "ml/alphazero_lite/fixtures/diagnostics/search_value_backup_ablation/search_interaction_diagnostic.json",
                    "original_path": self.ORIGINAL_SEARCH_INTERACTION_PATH,
                    "sha256": self._fixture_sha256(self.SEARCH_INTERACTION_FIXTURE_PATH),
                },
                "search_value_interaction_diagnostic": {
                    "path": "ml/alphazero_lite/fixtures/diagnostics/search_value_backup_ablation/search_value_interaction_diagnostic.json",
                    "original_path": self.ORIGINAL_SEARCH_VALUE_INTERACTION_PATH,
                    "sha256": self._fixture_sha256(self.SEARCH_VALUE_INTERACTION_FIXTURE_PATH),
                },
            },
            manifest["fixtures"],
        )
        self.assertEqual(
            json.loads(self.SEARCH_INTERACTION_FIXTURE_PATH.read_text(encoding="utf-8"))["schema"],
            "azlite_search_interaction_diagnostic_v1",
        )
        self.assertEqual(
            json.loads(self.SEARCH_VALUE_INTERACTION_FIXTURE_PATH.read_text(encoding="utf-8"))["schema"],
            "azlite_search_value_interaction_diagnostic_v1",
        )


class SearchValueBackupAblationMatrixTest(unittest.TestCase):
    def test_fixture_candidate_row_ids_match_approved_constants(self):
        source_artifacts = module.build_search_value_backup_source_artifacts()
        payload = source_artifacts["search_value_interaction_diagnostic"]["payload"]

        self.assertEqual(module.SEARCH_VALUE_PRIMARY_ROW_IDS, payload["primary_row_ids"])
        self.assertEqual(module.SEARCH_VALUE_COMPARATOR_ROW_IDS, payload["comparator_row_ids"])

    def test_build_matrix_rejects_fixture_row_id_drift(self):
        source_artifacts = deepcopy(module.build_search_value_backup_source_artifacts())
        source_artifacts["search_value_interaction_diagnostic"]["payload"]["primary_row_ids"] = [
            *module.SEARCH_VALUE_PRIMARY_ROW_IDS,
            "unexpected-row",
        ]

        with self.assertRaisesRegex(ValueError, "primary_row_ids"):
            module.build_search_value_backup_ablation_matrix(source_artifacts=source_artifacts)

    def test_build_matrix_accepts_approved_row_subsets_in_preserved_order(self):
        source_artifacts = deepcopy(module.build_search_value_backup_source_artifacts())
        source_artifacts["search_value_interaction_diagnostic"]["payload"]["primary_row_ids"] = [
            module.SEARCH_VALUE_PRIMARY_ROW_IDS[1],
            module.SEARCH_VALUE_PRIMARY_ROW_IDS[3],
        ]
        source_artifacts["search_value_interaction_diagnostic"]["payload"]["comparator_row_ids"] = []

        matrix = module.build_search_value_backup_ablation_matrix(source_artifacts=source_artifacts)

        self.assertTrue(
            all(
                row["target_row_ids"]
                == [
                    module.SEARCH_VALUE_PRIMARY_ROW_IDS[1],
                    module.SEARCH_VALUE_PRIMARY_ROW_IDS[3],
                ]
                for row in matrix
                if row["classification_role"] == "candidate"
            )
        )
        self.assertEqual(
            [],
            next(row for row in matrix if row["stable_key"] == "classic_only_comparator")["target_row_ids"],
        )

    def test_build_source_artifacts_and_matrix_use_stable_metadata(self):
        source_artifacts = module.build_search_value_backup_source_artifacts()

        self.assertEqual(
            "azlite_search_value_backup_ablation_v1",
            module.SEARCH_VALUE_BACKUP_ABLATION_SCHEMA,
        )
        self.assertEqual(
            "azlite_search_value_backup_ablation_fixture_manifest_v1",
            module.SEARCH_VALUE_BACKUP_ABLATION_FIXTURE_MANIFEST_SCHEMA,
        )
        self.assertEqual(
            {
                "enabled": True,
                "opening": 0.25,
                "midgame": 1.0,
                "late": 1.0,
            },
            module.LOW_OPENING_VALUE_TRUST_SCHEDULE,
        )

        self.assertEqual(
            module.SEARCH_VALUE_BACKUP_ABLATION_FIXTURE_DIR / "manifest.json",
            source_artifacts["fixture_manifest_path"],
        )
        self.assertEqual(
            module.SEARCH_VALUE_BACKUP_ABLATION_FIXTURE_MANIFEST_SCHEMA,
            source_artifacts["fixture_manifest"]["schema"],
        )
        self.assertEqual(
            "azlite_search_interaction_diagnostic_v1",
            source_artifacts["search_interaction_diagnostic"]["payload"]["schema"],
        )
        self.assertEqual(
            "azlite_search_value_interaction_diagnostic_v1",
            source_artifacts["search_value_interaction_diagnostic"]["payload"]["schema"],
        )

        matrix = module.build_search_value_backup_ablation_matrix(source_artifacts=source_artifacts)
        self.assertEqual(
            [
                "full_default",
                "policy_only_default",
                "value_only_default",
                "full_fpu_zero",
                "full_fpu_parent_q",
                "full_root_visit_count",
                "full_normalize_values",
                "full_value_trust_low_opening",
                "classic_only_comparator",
            ],
            [row["stable_key"] for row in matrix],
        )

        by_key = {row["stable_key"]: row for row in matrix}
        self.assertEqual(
            module.LOW_OPENING_VALUE_TRUST_SCHEDULE,
            by_key["full_value_trust_low_opening"]["search_options_overrides"]["value_trust_schedule"],
        )
        self.assertEqual("parent_q", by_key["full_fpu_parent_q"]["search_options_overrides"]["fpu_mode"])
        self.assertEqual("candidate", by_key["full_fpu_parent_q"]["classification_role"])
        self.assertEqual(
            {
                "is_supported": True,
                "support_tier": "native",
                "resolved_mode": "parent_q",
            },
            by_key["full_fpu_parent_q"]["support_metadata"],
        )
        self.assertEqual("full", by_key["full_default"]["ablation_mode"])
        self.assertEqual("policy_only", by_key["policy_only_default"]["ablation_mode"])
        self.assertEqual("value_only", by_key["value_only_default"]["ablation_mode"])
        self.assertEqual("visit_count", by_key["full_root_visit_count"]["search_options_overrides"]["root_policy_mode"])
        self.assertTrue(by_key["full_normalize_values"]["search_options_overrides"]["normalize_values"])
        self.assertEqual("comparator_only", by_key["classic_only_comparator"]["classification_role"])
        self.assertEqual(
            ["opening_plies_1_8-057"],
            by_key["classic_only_comparator"]["target_row_ids"],
        )

        self.assertEqual(
            Path("/tmp/rebalanced/final/search_value_backup_ablation_diagnostic.json"),
            module.search_value_backup_ablation_out_path(rebalanced_run_dir=Path("/tmp/rebalanced")),
        )


class SearchValueBackupAblationPayloadTest(unittest.TestCase):
    def test_build_payload_probes_selected_rebalanced_artifact_per_configuration(self):
        source_artifacts = module.build_search_value_backup_source_artifacts()

        captured_calls = []
        selected_artifact_path = "/tmp/rebalanced-selected-artifact"
        requested_simulations = 512
        requested_c_puct = 1.75
        requested_seed = 99
        stable_key_to_value = {
            "full_default": 0.11,
            "policy_only_default": 0.12,
            "value_only_default": 0.13,
            "full_fpu_zero": 0.14,
            "full_fpu_parent_q": 0.15,
            "full_root_visit_count": 0.16,
            "full_normalize_values": 0.17,
            "full_value_trust_low_opening": 0.18,
            "classic_only_comparator": 0.19,
        }

        class FakeArena:
            @staticmethod
            def build_eval_search_options():
                return {
                    "fpu_mode": "zero",
                    "reuse_subtree": False,
                    "normalize_values": False,
                    "root_policy_mode": "deterministic",
                    "tactical_root_bias": 0.1,
                }

            @staticmethod
            def ArtifactEvaluator(path):
                return {"evaluator_path": str(path)}

        def fake_load_row_context(*, row_id, original_run_dir, rebalanced_run_dir):
            del original_run_dir, rebalanced_run_dir
            return {
                "row_id": row_id,
                "bucket": "fixture_bucket",
                "phase": "opening",
                "reference_move": 2,
                "teacher_value": 0.5,
                "current_row": {
                    "state": {"row_id": row_id},
                    "legal_moves": [0, 1, 2, 3, 4, 5],
                },
                "rebalanced_row": {
                    "system_value": 0.44,
                    "value_error": 0.06,
                },
            }

        def fake_probe_artifact_position(*, artifact_path, state, simulations, seed, c_puct, evaluator=None, search_options=None, ablation_mode="full"):
            stable_key = search_options["stable_key"]
            resolved_options = search_options["search_options"]
            captured_calls.append(
                {
                    "artifact_path": artifact_path,
                    "row_id": state["row_id"],
                    "stable_key": stable_key,
                    "simulations": simulations,
                    "seed": seed,
                    "c_puct": c_puct,
                    "search_options": resolved_options,
                    "ablation_mode": ablation_mode,
                    "evaluator": evaluator,
                }
            )
            policy_top_move = 4 if stable_key == "full_fpu_parent_q" else 2
            visit_top_move = 3 if stable_key == "full_root_visit_count" else 1
            q_top_move = 0 if stable_key == "policy_only_default" else 5
            return {
                "selected_move": 1,
                "value": stable_key_to_value[stable_key],
                "policy": [0.05, 0.1, 0.45, 0.15, 0.2, 0.05],
                "visits": [10.0, 15.0, 90.0, 25.0, 30.0, 5.0],
                "child_stats": [
                    {"move": 0, "visits": 10, "q_value": 0.1},
                    {"move": 1, "visits": 15, "q_value": 0.2},
                    {"move": 2, "visits": 90, "q_value": 0.3},
                    {"move": 3, "visits": 25, "q_value": 0.4},
                    {"move": 4, "visits": 30, "q_value": 0.5},
                    {"move": 5, "visits": 5, "q_value": 0.6},
                ],
                "selection_breakdown": {
                    "fpu_mode": resolved_options.get("fpu_mode"),
                    "value_trust_multiplier": resolved_options.get("value_trust_schedule", {}).get("opening", 1.0),
                    "policy_top_move": policy_top_move,
                    "visit_top_move": visit_top_move,
                    "q_top_move": q_top_move,
                },
                "visit_snapshots": [{"simulation": 128, "visits": [10.0, 15.0, 90.0, 25.0, 30.0, 5.0]}],
            }

        original_load_row_context = module.load_row_context
        original_load_selected_artifact_path = module.load_selected_artifact_path
        original_load_arena_module = module.load_arena_module
        original_probe_artifact_position = module.probe_artifact_position
        module.load_row_context = fake_load_row_context
        module.load_selected_artifact_path = lambda _run_dir: selected_artifact_path
        module.load_arena_module = lambda: FakeArena
        module.probe_artifact_position = fake_probe_artifact_position
        self.addCleanup(setattr, module, "load_row_context", original_load_row_context)
        self.addCleanup(setattr, module, "load_selected_artifact_path", original_load_selected_artifact_path)
        self.addCleanup(setattr, module, "load_arena_module", original_load_arena_module)
        self.addCleanup(setattr, module, "probe_artifact_position", original_probe_artifact_position)

        payload = module.build_search_value_backup_ablation_payload(
            source_artifacts=source_artifacts,
            artifact_simulations=requested_simulations,
            c_puct=requested_c_puct,
            seed=requested_seed,
        )

        self.assertEqual(module.SEARCH_VALUE_BACKUP_ABLATION_SCHEMA, payload["schema"])
        self.assertEqual(
            [
                "full_default",
                "policy_only_default",
                "value_only_default",
                "full_fpu_zero",
                "full_fpu_parent_q",
                "full_root_visit_count",
                "full_normalize_values",
                "full_value_trust_low_opening",
                "classic_only_comparator",
            ],
            [row["stable_key"] for row in payload["matrix_configurations"]],
        )
        self.assertEqual(module.SEARCH_VALUE_PRIMARY_ROW_IDS, payload["primary_row_ids"])
        self.assertEqual(module.SEARCH_VALUE_COMPARATOR_ROW_IDS, payload["comparator_row_ids"])

        matrix_by_key = {row["stable_key"]: row for row in payload["matrix_configurations"]}
        self.assertEqual("classic_only", matrix_by_key["classic_only_comparator"]["ablation_mode"])
        self.assertEqual(
            module.LOW_OPENING_VALUE_TRUST_SCHEDULE,
            matrix_by_key["full_value_trust_low_opening"]["search_options_overrides"]["value_trust_schedule"],
        )

        rows = payload["rows"]
        self.assertEqual(
            [*module.SEARCH_VALUE_PRIMARY_ROW_IDS, *module.SEARCH_VALUE_COMPARATOR_ROW_IDS],
            list(rows.keys()),
        )

        primary_row = rows["incumbent_proxy_disagreement-031"]
        self.assertEqual("primary", primary_row["row_role"])
        self.assertEqual("search_overrides_prior", primary_row["classification_outcome"])
        self.assertEqual(
            "policy leans to 4, visits finish on 2, q-values favor 3, snapshots available",
            primary_row["row_mechanism_summary"],
        )
        self.assertEqual("model-artifact/current", primary_row["source_artifacts"]["current"])
        self.assertEqual(
            selected_artifact_path,
            primary_row["source_artifacts"]["selected_rebalanced_artifact"],
        )
        self.assertIn("configurations", primary_row)
        self.assertNotIn("primary_id", primary_row["configurations"]["full_default"])
        self.assertNotIn("comparator_id", primary_row["configurations"]["full_default"])
        self.assertEqual(
            {
                "selected_move": 1,
                "probe_value": 0.11,
                "probe_value_error": 0.39,
                "visit_top_move": 1,
                "q_top_move": 5,
                "policy_top_move": 2,
                "fpu_mode": "zero",
                "value_trust_multiplier": 1.0,
            },
            primary_row["configurations"]["full_default"]["observability"],
        )
        self.assertEqual(0.14, primary_row["configurations"]["full_fpu_zero"]["observability"]["probe_value"])
        self.assertEqual(0.15, primary_row["configurations"]["full_fpu_parent_q"]["observability"]["probe_value"])
        self.assertEqual("parent_q", primary_row["configurations"]["full_fpu_parent_q"]["search_options"]["fpu_mode"])
        self.assertEqual(3, primary_row["configurations"]["full_root_visit_count"]["observability"]["visit_top_move"])
        self.assertTrue(primary_row["configurations"]["full_normalize_values"]["search_options"]["normalize_values"])
        self.assertEqual(
            0.25,
            primary_row["configurations"]["full_value_trust_low_opening"]["observability"]["value_trust_multiplier"],
        )
        self.assertEqual([], primary_row["configurations"]["full_default"]["unsupported_comparisons"])

        comparator_row = rows["opening_plies_1_8-057"]
        self.assertEqual("comparator", comparator_row["row_role"])
        self.assertEqual("bad_priors", comparator_row["classification_outcome"])
        self.assertEqual("comparator_only", comparator_row["configurations"]["classic_only_comparator"]["classification_role"])
        self.assertEqual(
            [
                {
                    "kind": "row_out_of_scope",
                    "reason": "row not approved for this configuration",
                },
            ],
            comparator_row["configurations"]["full_default"]["unsupported_comparisons"],
        )
        self.assertEqual([], comparator_row["configurations"]["classic_only_comparator"]["unsupported_comparisons"])

        primary_calls = [call for call in captured_calls if call["row_id"] == "capture_available-002"]
        self.assertEqual(
            [
                "full_default",
                "policy_only_default",
                "value_only_default",
                "full_fpu_zero",
                "full_fpu_parent_q",
                "full_root_visit_count",
                "full_normalize_values",
                "full_value_trust_low_opening",
            ],
            [call["stable_key"] for call in primary_calls],
        )
        self.assertTrue(all(call["artifact_path"] == selected_artifact_path for call in captured_calls))
        self.assertTrue(all(call["evaluator"] == {"evaluator_path": selected_artifact_path} for call in captured_calls))
        self.assertTrue(all(call["simulations"] == requested_simulations for call in captured_calls))
        self.assertTrue(all(call["c_puct"] == requested_c_puct for call in captured_calls))
        self.assertEqual(requested_seed, captured_calls[0]["seed"])
        self.assertEqual(requested_seed + 100, captured_calls[8]["seed"])
        comparator_calls = [call for call in captured_calls if call["row_id"] == "opening_plies_1_8-057"]
        self.assertEqual(["classic_only_comparator"], [call["stable_key"] for call in comparator_calls])

    def test_build_payload_marks_unsupported_configurations_explicitly(self):
        source_artifacts = module.build_search_value_backup_source_artifacts()
        original_build_matrix = module.build_search_value_backup_ablation_matrix
        original_load_row_context = module.load_row_context
        original_load_selected_artifact_path = module.load_selected_artifact_path
        original_load_arena_module = module.load_arena_module
        original_probe_artifact_position = module.probe_artifact_position
        probe_calls = []

        module.build_search_value_backup_ablation_matrix = lambda *, source_artifacts: [
            {
                "stable_key": "unsupported_parent_value",
                "name": "Unsupported Parent Value",
                "classification_role": "candidate",
                "ablation_mode": "full",
                "search_options_overrides": {"fpu_mode": "parent_value"},
                "target_row_ids": list(module.SEARCH_VALUE_PRIMARY_ROW_IDS),
                "support_metadata": {
                    "is_supported": False,
                    "support_tier": "unsupported",
                    "requested_mode": "parent_value",
                },
            }
        ]
        module.load_row_context = lambda **kwargs: {
            "row_id": kwargs["row_id"],
            "bucket": "fixture_bucket",
            "phase": "opening",
            "reference_move": 2,
            "teacher_value": 0.5,
            "current_row": {"state": {"row_id": kwargs["row_id"]}, "legal_moves": [0, 1, 2, 3, 4, 5]},
            "rebalanced_row": {"system_value": 0.44, "value_error": 0.06},
        }
        module.load_selected_artifact_path = lambda _run_dir: "/tmp/rebalanced-selected-artifact"
        module.load_arena_module = lambda: type(
            "FakeArena",
            (),
            {
                "build_eval_search_options": staticmethod(
                    lambda: {
                        "fpu_mode": "zero",
                        "reuse_subtree": False,
                        "normalize_values": False,
                        "root_policy_mode": "deterministic",
                        "tactical_root_bias": 0.1,
                    }
                ),
                "ArtifactEvaluator": staticmethod(lambda path: {"evaluator_path": str(path)}),
            },
        )
        module.probe_artifact_position = lambda **kwargs: probe_calls.append(kwargs) or {"selected_move": 2}
        self.addCleanup(setattr, module, "build_search_value_backup_ablation_matrix", original_build_matrix)
        self.addCleanup(setattr, module, "load_row_context", original_load_row_context)
        self.addCleanup(setattr, module, "load_selected_artifact_path", original_load_selected_artifact_path)
        self.addCleanup(setattr, module, "load_arena_module", original_load_arena_module)
        self.addCleanup(setattr, module, "probe_artifact_position", original_probe_artifact_position)

        payload = module.build_search_value_backup_ablation_payload(source_artifacts=source_artifacts)

        primary_entry = payload["rows"][module.SEARCH_VALUE_PRIMARY_ROW_IDS[0]]["configurations"]["unsupported_parent_value"]
        self.assertEqual(
            [
                {
                    "kind": "configuration_unsupported",
                    "reason": "configuration is unsupported for probing",
                    "requested_mode": "parent_value",
                }
            ],
            primary_entry["unsupported_comparisons"],
        )
        self.assertEqual([], probe_calls)

    def test_probe_artifact_position_unwraps_diagnostic_search_options_and_ablation_mode(self):
        calls = {}

        class FakeArena:
            class ArtifactEvaluator:
                def __init__(self, _path):
                    raise AssertionError("evaluator should not be constructed in this test")

            @staticmethod
            def evaluate_artifact_position(**kwargs):
                calls.update(kwargs)
                return {"selected_move": 2}

            @staticmethod
            def build_eval_search_options():
                return {"fpu_mode": "zero", "max_depth": 4}

        original_loader = module.load_arena_module
        module.load_arena_module = lambda: FakeArena
        self.addCleanup(setattr, module, "load_arena_module", original_loader)

        wrapper = {
            "stable_key": "policy_only_default",
            "search_options": {"fpu_mode": "parent_q", "max_depth": 6},
            "ablation_mode": "policy_only",
        }

        result = module.probe_artifact_position(
            artifact_path="model-artifact/current",
            state={"board": [4, 4, 4]},
            simulations=64,
            seed=7,
            c_puct=1.5,
            evaluator=object(),
            search_options=wrapper,
            ablation_mode=wrapper,
        )

        self.assertEqual({"selected_move": 2}, result)
        self.assertEqual({"fpu_mode": "parent_q", "max_depth": 6}, calls["search_options"])
        self.assertEqual("policy_only", calls["ablation_mode"])


class SearchValueBackupAblationMainTest(unittest.TestCase):
    def test_main_writes_ablation_artifact_from_existing_source_and_sibling_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            original_run = tmp_path / "original"
            rebalanced_run = tmp_path / "rebalanced"
            (original_run / "final").mkdir(parents=True)
            (rebalanced_run / "final").mkdir(parents=True)
            stdout = io.StringIO()

            source_payload = {
                "schema": module.SEARCH_INTERACTION_SCHEMA,
                "rows": {},
                "summary": {},
                "row_source": {"resolved_rows": ["capture_available-002"]},
            }
            sibling_payload = {
                "schema": module.SEARCH_VALUE_INTERACTION_SCHEMA,
                "source_diagnostic_path": str(module.diagnostic_out_path(rebalanced_run_dir=rebalanced_run)),
                "primary_row_ids": ["capture_available-002"],
                "comparator_row_ids": [],
                "rows": {},
                "summary": {"primary_row_count": 1, "comparator_row_count": 0},
            }
            ablation_payload = {
                "schema": module.SEARCH_VALUE_BACKUP_ABLATION_SCHEMA,
                "matrix_configurations": [],
                "primary_row_ids": ["capture_available-002"],
                "comparator_row_ids": [],
                "rows": {},
            }

            def fake_build_ablation_payload(*, source_artifacts, artifact_simulations, c_puct, seed):
                self.assertEqual(
                    module.diagnostic_out_path(rebalanced_run_dir=rebalanced_run),
                    source_artifacts["search_interaction_diagnostic"]["path"],
                )
                self.assertIs(source_payload, source_artifacts["search_interaction_diagnostic"]["payload"])
                self.assertEqual(
                    module.search_value_interaction_diagnostic_out_path(rebalanced_run_dir=rebalanced_run),
                    source_artifacts["search_value_interaction_diagnostic"]["path"],
                )
                self.assertIs(
                    sibling_payload,
                    source_artifacts["search_value_interaction_diagnostic"]["payload"],
                )
                self.assertEqual(512, artifact_simulations)
                self.assertEqual(1.75, c_puct)
                self.assertEqual(99, seed)
                return ablation_payload

            with mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.build_search_interaction_payload",
                return_value=source_payload,
            ), mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.build_search_value_interaction_payload",
                side_effect=AssertionError("main should not rebuild sibling from scratch"),
            ), mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.build_search_value_interaction_payload_from_source_payload",
                return_value=sibling_payload,
            ), mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.build_search_value_backup_source_artifacts",
                side_effect=AssertionError("main should not rebuild ablation sources from scratch"),
            ), mock.patch(
                "ml.alphazero_lite.search_interaction_diagnostic.build_search_value_backup_ablation_payload",
                side_effect=fake_build_ablation_payload,
            ) as build_ablation, mock.patch("sys.stdout", stdout):
                exit_code = module.main(
                    [
                        "--original-run",
                        str(original_run),
                        "--rebalanced-run",
                        str(rebalanced_run),
                        "--current-artifact",
                        "model-artifact/current",
                        "--artifact-simulations",
                        "512",
                        "--c-puct",
                        "1.75",
                        "--seed",
                        "99",
                    ]
                )

            source_path = module.diagnostic_out_path(rebalanced_run_dir=rebalanced_run)
            sibling_path = module.search_value_interaction_diagnostic_out_path(rebalanced_run_dir=rebalanced_run)
            ablation_path = module.search_value_backup_ablation_out_path(rebalanced_run_dir=rebalanced_run)
            reported = json.loads(stdout.getvalue())

            self.assertEqual(0, exit_code)
            self.assertEqual(source_payload, json.loads(source_path.read_text(encoding="utf-8")))
            self.assertEqual(sibling_payload, json.loads(sibling_path.read_text(encoding="utf-8")))
            self.assertEqual(ablation_payload, json.loads(ablation_path.read_text(encoding="utf-8")))
            build_ablation.assert_called_once()
            self.assertEqual(str(source_path), reported["artifact_path"])
            self.assertEqual(str(sibling_path), reported["search_value_interaction_artifact_path"])
            self.assertEqual(str(ablation_path), reported["search_value_backup_ablation_artifact_path"])
