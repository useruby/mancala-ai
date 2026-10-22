"""Serving engine authentication, routing, lifecycle, and backend parity tests."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from ml.alphazero_lite import arena
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.serving import engine as engine_module
from ml.alphazero_lite.serving.contract import MoveRequest
from ml.alphazero_lite.serving.engine import (
    MoveEngine,
    RuntimeBundle,
    RuntimeConfigurationError,
    SearchProfile,
    SearchSettings,
    serving_artifact_identity,
)

FIXTURE = Path(__file__).parent / "fixtures/incumbent_forensic_proxy_current"


def _request(current_player: int, *, pits0=None, pits1=None, stores=(4, 2)):
    return MoveRequest.model_validate(
        {
            "request_id": f"request-{current_player}",
            "game_revision": 1,
            "state": {
                "player0_pits": pits0 or [1, 2, 3, 4, 5, 6],
                "player1_pits": pits1 or [6, 5, 4, 3, 2, 1],
                "player0_store": stores[0],
                "player1_store": stores[1],
                "current_player": current_player,
            },
            "profile": "default",
            "seed": 17,
        }
    )


def _artifact(root: Path, name: str) -> tuple[str, str]:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE / "metadata.json", path / "metadata.json")
    shutil.copy(FIXTURE / "weights.json", path / "weights.json")
    return name, serving_artifact_identity(
        path / "metadata.json", path / "weights.json"
    )


def _engine(root: Path, *, names=("model",), kind="single", **bundle_options):
    artifacts = [_artifact(root, name) for name in names]
    bundle = RuntimeBundle(
        bundle_id="bundle-v1",
        kind=kind,
        artifacts=tuple(item[0] for item in artifacts),
        expected_artifact_sha256=tuple(item[1] for item in artifacts),
        **bundle_options,
    )
    return MoveEngine(
        root,
        {"bundle": bundle},
        {
            "default": SearchProfile(
                "default", "profile-v1", "bundle", SearchSettings(simulations=3)
            )
        },
    )


class NativeDouble:
    created = 0
    closed = 0

    def __init__(self, *_args, **_kwargs):
        type(self).created += 1
        self.calls = 0

    def close(self):
        type(self).closed += 1

    def root_action_margins(self, game, perspective_player):
        self.calls += 1
        return {move: 0 for move in game.possible_moves()}


def _write_policy(root: Path, artifact_dirs: tuple[str, ...], *, threshold=12):
    probe = root / "probe"
    probe.write_bytes(b"controlled native fixture")
    probe.chmod(0o755)
    tablebase = root / "tablebase.kvtb"
    tablebase.write_bytes(b"controlled tablebase fixture")
    policy = {
        "schema": "azlite_runtime_search_policy_v1",
        "mode": "puct_exact_root_hybrid",
        "exact_root_enabled": True,
        "exact_root_threshold": threshold,
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve": "disabled",
        "solver_implementation_identity": "native_kvtb_root_action_probe_v1",
        "native_probe": {
            "path": "probe",
            "sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
        },
        "tablebase": {
            "path": "tablebase.kvtb",
            "sha256": hashlib.sha256(tablebase.read_bytes()).hexdigest(),
        },
    }
    encoded = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    policy_hash = hashlib.sha256(encoded.encode()).hexdigest()
    for artifact_dir in artifact_dirs:
        directory = root / artifact_dir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "search_policy.json").write_text(encoded)
    return policy_hash


def test_backend_receives_fixed_ownership_for_both_players(monkeypatch, tmp_path):
    engine = MoveEngine.__new__(MoveEngine)
    engine.bundles = {"bundle": RuntimeBundle("bundle", "single", ("unused",))}
    engine.profiles = {
        "default": SearchProfile("default", "default", "bundle", SearchSettings())
    }
    engine._evaluators = {"bundle": object()}
    engine._bundle_ids = {"bundle": "bundle:identity"}
    engine._profile_ids = {"default": "default:identity"}
    engine._runtime_policies = {"bundle": None}
    engine._native_resources = {}
    engine._closed = False
    passed = []

    def backend(**kwargs):
        passed.append(kwargs["state"])
        return {"selected_move": 0, "child_stats": [{"visits": 1}]}

    monkeypatch.setattr(engine_module, "evaluate_artifact_position", backend)
    for player in (0, 1):
        engine.select_move(_request(player))
        assert passed[-1] == {
            "player_pits": [1, 2, 3, 4, 5, 6],
            "opponent_pits": [6, 5, 4, 3, 2, 1],
            "player_store": 4,
            "opponent_store": 2,
            "current_player": player,
        }


def test_real_backend_matches_direct_search_for_both_players(tmp_path):
    engine = _engine(tmp_path)
    states = (
        _request(0).state,
        _request(1).state,
        # Player 1's relative pit 5 has one seed and yields an extra turn.
        _request(
            1,
            pits0=[1, 1, 1, 1, 1, 1],
            pits1=[0, 0, 0, 0, 0, 1],
            stores=(33, 8),
        ).state,
    )
    extra_turn = KalahGame.from_state(
        {
            "player_pits": states[2].player0_pits,
            "opponent_pits": states[2].player1_pits,
            "player_store": states[2].player0_store,
            "opponent_store": states[2].player1_store,
            "current_player": 1,
        }
    )
    assert extra_turn.move(11)
    assert extra_turn.current_player == 1
    try:
        for index, state in enumerate(states):
            request = _request(state.current_player)
            request.state = state
            response = engine.select_move(request)
            original = {
                "player_pits": list(state.player0_pits),
                "opponent_pits": list(state.player1_pits),
                "player_store": state.player0_store,
                "opponent_store": state.player1_store,
                "current_player": state.current_player,
            }
            direct = arena.evaluate_artifact_position(
                evaluator=engine._evaluators["bundle"],
                state=original,
                simulations=3,
                seed=request.seed,
                c_puct=1.25,
                search_options=SearchSettings(simulations=3).options(),
                ablation_mode="full",
            )
            assert response.result.move == direct["selected_move"], index
            assert response.completed_simulations == sum(
                item["visits"] for item in direct["child_stats"]
            )
    finally:
        engine.close()


def test_terminal_request_bypasses_backend_and_search(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    monkeypatch.setattr(
        engine_module,
        "evaluate_artifact_position",
        lambda **_kwargs: pytest.fail("terminal position entered search"),
    )
    try:
        request = _request(0, pits0=[0] * 6, pits1=[0] * 6, stores=(24, 24))
        result = engine.select_move(request)
        assert result.result.kind == "terminal"
        assert result.completed_simulations == 0
    finally:
        engine.close()


def test_composed_routing_and_search_policy_are_identity_bound(monkeypatch, tmp_path):
    NativeDouble.created = NativeDouble.closed = 0
    monkeypatch.setattr(engine_module, "NativeExactRootTablebase", NativeDouble)
    policy_hash = _write_policy(tmp_path, ("policy", "value"))
    engine = _engine(
        tmp_path,
        names=("policy", "value"),
        kind="composed",
        expected_policy_sha256=(policy_hash, policy_hash),
        policy_source="candidate",
        value_source="current",
    )
    calls = []

    def backend(**kwargs):
        calls.append(kwargs)
        exact = kwargs["state"]["player_pits"] + kwargs["state"]["opponent_pits"]
        if sum(exact) <= kwargs["exact_root_solve_threshold"]:
            return {
                "selected_move": 0,
                "exact_root_decision": {},
                "child_stats": [{"visits": 0}],
            }
        return {"selected_move": 0, "child_stats": [{"visits": 3}]}

    monkeypatch.setattr(engine_module, "evaluate_artifact_position", backend)
    assert NativeDouble.created == 1
    ordinary_response = engine.select_move(_request(1))
    assert ordinary_response.completed_simulations == 3
    assert calls[-1]["exact_root_solve_threshold"] == 12
    assert calls[-1]["exact_root_tablebase"] is engine._native_resources["bundle"]
    assert calls[-1]["evaluator"].policy_source == "candidate"
    assert calls[-1]["evaluator"].value_source == "current"
    exact_request = _request(
        1,
        pits0=[0, 0, 0, 0, 0, 0],
        pits1=[1, 1, 2, 0, 0, 0],
        stores=(36, 8),
    )
    exact_response = engine.select_move(exact_request)
    assert exact_response.completed_simulations == 0
    engine.close()
    engine.close()
    assert NativeDouble.closed == 1
    with pytest.raises(engine_module.EngineClosedError):
        engine.select_move(_request(1))


def test_single_artifact_exact_root_routing(monkeypatch, tmp_path):
    NativeDouble.created = NativeDouble.closed = 0
    monkeypatch.setattr(engine_module, "NativeExactRootTablebase", NativeDouble)
    policy_hash = _write_policy(tmp_path, ("model",), threshold=8)
    engine = _engine(tmp_path, expected_policy_sha256=(policy_hash,))
    seen = []

    def backend(**kwargs):
        seen.append(kwargs)
        return {
            "selected_move": 0,
            "exact_root_decision": {},
            "child_stats": [{"visits": 0}],
        }

    monkeypatch.setattr(engine_module, "evaluate_artifact_position", backend)
    response = engine.select_move(
        _request(
            1,
            pits0=[0] * 6,
            pits1=[1, 1, 2, 0, 0, 0],
            stores=(36, 8),
        )
    )
    assert response.completed_simulations == 0
    assert seen[0]["exact_root_solve_threshold"] == 8
    assert seen[0]["exact_root_tablebase"] is engine._native_resources["bundle"]
    engine.close()


def test_real_search_uses_puct_outside_exact_root_threshold(monkeypatch, tmp_path):
    NativeDouble.created = NativeDouble.closed = 0
    monkeypatch.setattr(engine_module, "NativeExactRootTablebase", NativeDouble)
    policy_hash = _write_policy(tmp_path, ("model",), threshold=8)
    engine = _engine(tmp_path, expected_policy_sha256=(policy_hash,))
    native = engine._native_resources["bundle"]
    try:
        response = engine.select_move(_request(1))
        assert response.completed_simulations == 3
        assert native.calls == 0
    finally:
        engine.close()


def test_illegal_backend_move_is_rejected(monkeypatch):
    engine = MoveEngine.__new__(MoveEngine)
    engine.profiles = {
        "default": SearchProfile("default", "default", "bundle", SearchSettings())
    }
    engine._evaluators = {"bundle": object()}
    engine._bundle_ids = {"bundle": "bundle:identity"}
    engine._profile_ids = {"default": "default:identity"}
    engine._runtime_policies = {"bundle": None}
    engine._native_resources = {}
    engine._closed = False
    monkeypatch.setattr(
        engine_module,
        "evaluate_artifact_position",
        lambda **_kwargs: {"selected_move": 6, "child_stats": []},
    )
    with pytest.raises(engine_module.InvalidBackendResultError):
        engine.select_move(_request(0))


def test_initialization_failure_closes_previously_created_native_resource(
    monkeypatch, tmp_path
):
    NativeDouble.created = NativeDouble.closed = 0
    monkeypatch.setattr(engine_module, "NativeExactRootTablebase", NativeDouble)
    policy_hash = _write_policy(tmp_path, ("good",))
    good_name, good_identity = _artifact(tmp_path, "good")
    bad_name, _bad_identity = _artifact(tmp_path, "bad")
    first = RuntimeBundle(
        "good-id", "single", (good_name,), (good_identity,), (policy_hash,)
    )
    second = RuntimeBundle("bad-id", "single", (bad_name,), ("0" * 64,), ())
    with pytest.raises(RuntimeConfigurationError):
        MoveEngine(
            tmp_path,
            {"good": first, "bad": second},
            {"default": SearchProfile("default", "v1", "good", SearchSettings())},
        )
    assert NativeDouble.created == NativeDouble.closed == 1


def test_bundle_and_profile_identities_include_routing_threshold_and_asset_hash(
    monkeypatch, tmp_path
):
    NativeDouble.created = NativeDouble.closed = 0
    monkeypatch.setattr(engine_module, "NativeExactRootTablebase", NativeDouble)
    base = tmp_path / "base"
    base.mkdir()
    threshold_hash = _write_policy(base, ("model",), threshold=8)
    _artifact(base, "model")
    first = _engine(
        base,
        names=("model",),
        expected_policy_sha256=(threshold_hash,),
    )
    first_id = first._bundle_ids["bundle"]
    first_profile_id = first._profile_ids["default"]
    first.close()

    changed_root = tmp_path / "changed"
    changed_root.mkdir()
    shutil.copytree(base / "model", changed_root / "model")
    shutil.copy(base / "probe", changed_root / "probe")
    shutil.copy(base / "tablebase.kvtb", changed_root / "tablebase.kvtb")
    (changed_root / "probe").chmod(0o755)
    policy = json.loads((changed_root / "model" / "search_policy.json").read_text())
    policy["exact_root_threshold"] = 9
    encoded = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    (changed_root / "model" / "search_policy.json").write_text(encoded)
    changed_policy_hash = hashlib.sha256(encoded.encode()).hexdigest()
    changed = _engine(
        changed_root,
        expected_policy_sha256=(changed_policy_hash,),
    )
    assert changed._bundle_ids["bundle"] != first_id
    assert changed._profile_ids["default"] != first_profile_id
    changed.close()

    relocated = tmp_path / "relocated"
    relocated.mkdir()
    shutil.copytree(base / "model", relocated / "model")
    shutil.copy(base / "probe", relocated / "probe")
    shutil.copy(base / "tablebase.kvtb", relocated / "tablebase.kvtb")
    (relocated / "probe").chmod(0o755)
    same = _engine(relocated, expected_policy_sha256=(threshold_hash,))
    assert same._bundle_ids["bundle"] == first_id
    same.close()

    asset_changed_root = tmp_path / "asset_changed"
    asset_changed_root.mkdir()
    shutil.copytree(base / "model", asset_changed_root / "model")
    shutil.copy(base / "probe", asset_changed_root / "probe")
    (asset_changed_root / "probe").chmod(0o755)
    (asset_changed_root / "tablebase.kvtb").write_bytes(b"different tablebase")
    asset_policy = json.loads(
        (asset_changed_root / "model" / "search_policy.json").read_text()
    )
    asset_policy["tablebase"]["sha256"] = hashlib.sha256(
        (asset_changed_root / "tablebase.kvtb").read_bytes()
    ).hexdigest()
    asset_encoded = json.dumps(asset_policy, sort_keys=True, separators=(",", ":"))
    (asset_changed_root / "model" / "search_policy.json").write_text(asset_encoded)
    asset_changed = _engine(
        asset_changed_root,
        expected_policy_sha256=(hashlib.sha256(asset_encoded.encode()).hexdigest(),),
    )
    assert asset_changed._bundle_ids["bundle"] != first_id
    asset_changed.close()

    routing_root = tmp_path / "routing"
    routing_root.mkdir()
    artifacts = [
        _artifact(routing_root, "current"),
        _artifact(routing_root, "candidate"),
    ]

    def routed_engine(policy_source):
        return MoveEngine(
            routing_root,
            {
                "bundle": RuntimeBundle(
                    "routed-v1",
                    "composed",
                    tuple(item[0] for item in artifacts),
                    tuple(item[1] for item in artifacts),
                    policy_source=policy_source,
                    value_source="current",
                )
            },
            {
                "default": SearchProfile(
                    "default", "v1", "bundle", SearchSettings(simulations=3)
                )
            },
        )

    current_policy = routed_engine("current")
    candidate_policy = routed_engine("candidate")
    assert (
        current_policy._bundle_ids["bundle"] != candidate_policy._bundle_ids["bundle"]
    )
    current_policy.close()
    candidate_policy.close()


def test_missing_sidecar_and_incompatible_composed_policy_fail(tmp_path, monkeypatch):
    NativeDouble.created = 0
    monkeypatch.setattr(engine_module, "NativeExactRootTablebase", NativeDouble)
    first_hash = _write_policy(tmp_path, ("first",), threshold=8)
    _, _ = _artifact(tmp_path, "second")
    first = tmp_path / "first" / "search_policy.json"
    second = tmp_path / "second" / "search_policy.json"
    second.write_text(
        first.read_text().replace(
            '"exact_root_threshold":8', '"exact_root_threshold":9'
        )
    )
    expected_second = hashlib.sha256(second.read_bytes()).hexdigest()
    with pytest.raises(RuntimeConfigurationError):
        _engine(
            tmp_path,
            names=("first", "second"),
            kind="composed",
            expected_policy_sha256=(first_hash, expected_second),
        )


@pytest.mark.parametrize(
    "mutation", ["expected_hash", "weights", "metadata", "missing_policy"]
)
def test_hashes_and_metadata_are_enforced(tmp_path, mutation):
    _, identity = _artifact(tmp_path, "model")
    bundle_args = {}
    if mutation == "expected_hash":
        identity = "0" * 64
    elif mutation == "weights":
        (tmp_path / "model" / "weights.json").write_text("{}")
        identity = serving_artifact_identity(
            tmp_path / "model/metadata.json", tmp_path / "model/weights.json"
        )
    elif mutation == "metadata":
        (tmp_path / "model" / "metadata.json").write_text(
            '{"input_encoding":"not-an-encoding",'
            '"architecture":{"model_type":"mlp_v1"}}'
        )
        identity = serving_artifact_identity(
            tmp_path / "model/metadata.json", tmp_path / "model/weights.json"
        )
    else:
        bundle_args["expected_policy_sha256"] = ("a" * 64,)
    bundle = RuntimeBundle(
        "bundle-v1", "single", ("model",), (identity,), **bundle_args
    )
    with pytest.raises(RuntimeConfigurationError):
        MoveEngine(
            tmp_path,
            {"bundle": bundle},
            {"default": SearchProfile("default", "v1", "bundle", SearchSettings())},
        )


def test_directories_and_lfs_pointers_are_rejected(tmp_path):
    _, identity = _artifact(tmp_path, "model")
    policy_hash = _write_policy(tmp_path, ("model",))
    (tmp_path / "tablebase.kvtb").unlink()
    (tmp_path / "tablebase.kvtb").mkdir()
    bundle = RuntimeBundle(
        "bundle-v1", "single", ("model",), (identity,), (policy_hash,)
    )
    with pytest.raises(RuntimeConfigurationError):
        MoveEngine(
            tmp_path,
            {"bundle": bundle},
            {"default": SearchProfile("default", "v1", "bundle", SearchSettings())},
        )

    (tmp_path / "tablebase.kvtb").rmdir()
    (tmp_path / "tablebase.kvtb").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n"
    )
    policy = json.loads((tmp_path / "model" / "search_policy.json").read_text())
    policy["tablebase"]["sha256"] = hashlib.sha256(
        (tmp_path / "tablebase.kvtb").read_bytes()
    ).hexdigest()
    encoded = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    (tmp_path / "model" / "search_policy.json").write_text(encoded)
    bundle = RuntimeBundle(
        "bundle-v1",
        "single",
        ("model",),
        (identity,),
        (hashlib.sha256(encoded.encode()).hexdigest(),),
    )
    with pytest.raises(RuntimeConfigurationError):
        MoveEngine(
            tmp_path,
            {"bundle": bundle},
            {"default": SearchProfile("default", "v1", "bundle", SearchSettings())},
        )
