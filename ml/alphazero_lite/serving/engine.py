"""Synchronous, initialized Python move engine for the v1 serving contract."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from pydantic import TypeAdapter

from ml.alphazero_lite.native_exact_root_tablebase import NativeExactRootTablebase
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.input_encodings import feature_count_for
from ml.alphazero_lite.exact_root_decision import exact_root_decision

from ml.alphazero_lite.arena import (
    ArtifactEvaluator,
    ComposedArtifactEvaluator,
    evaluate_artifact_position,
)
from ml.alphazero_lite.serving.contract import (
    Identifier,
    MoveRequest,
    MoveResponse,
    MoveResult,
    TerminalResult,
    state_sha256,
    to_kalah_game,
    validate_response,
)


class EngineError(RuntimeError):
    """Base class for typed engine failures."""


class UnknownProfileError(EngineError):
    """The request selected no configured profile."""


class RuntimeConfigurationError(EngineError):
    """A configured runtime bundle is missing, corrupt, or incompatible."""


class EngineClosedError(EngineError):
    """A request was submitted after engine resources were released."""


class InvalidBackendResultError(EngineError):
    """The Python backend returned an invalid action or identity."""


@dataclass(frozen=True)
class SearchSettings:
    simulations: int = 128
    c_puct: float = 1.25
    fpu_mode: str = "zero"
    reuse_subtree: bool = False
    normalize_values: bool = False
    root_policy_mode: str = "deterministic"
    tactical_root_bias: float = 0.0

    def __post_init__(self) -> None:
        if (
            type(self.simulations) is not int
            or self.simulations < 1
            or type(self.c_puct) not in (int, float)
            or not math.isfinite(self.c_puct)
            or self.c_puct <= 0
            or not isinstance(self.fpu_mode, str)
            or self.fpu_mode not in {"zero", "parent_q", "parent_value"}
            or type(self.reuse_subtree) is not bool
            or type(self.normalize_values) is not bool
            or not isinstance(self.root_policy_mode, str)
            or self.root_policy_mode not in {"deterministic", "visit_count"}
            or type(self.tactical_root_bias) not in (int, float)
            or not math.isfinite(self.tactical_root_bias)
        ):
            raise ValueError("search simulations and c_puct must be positive")

    def options(self) -> dict:
        return {
            "fpu_mode": self.fpu_mode,
            "reuse_subtree": self.reuse_subtree,
            "normalize_values": self.normalize_values,
            "root_policy_mode": self.root_policy_mode,
            "tactical_root_bias": self.tactical_root_bias,
        }


@dataclass(frozen=True)
class RuntimeBundle:
    bundle_id: str
    kind: str
    artifacts: tuple[str, ...]
    expected_artifact_sha256: tuple[str, ...] = ()
    expected_policy_sha256: tuple[str | None, ...] = ()
    policy_source: str = "current"
    value_source: str = "current"


@dataclass(frozen=True)
class SearchProfile:
    name: str
    profile_id: str
    bundle_id: str
    settings: SearchSettings
    require_runtime_search_policy: bool = False


class _ServingEvaluator:
    """Delegate evaluations without exposing research artifact paths to arena."""

    def __init__(self, evaluator: Any) -> None:
        self.evaluator = evaluator

    def evaluate(self, game: KalahGame):
        return self.evaluator.evaluate(game)

    def reset_telemetry(self) -> None:
        reset = getattr(self.evaluator, "reset_telemetry", None)
        if callable(reset):
            reset()

    @property
    def telemetry(self):
        return getattr(self.evaluator, "telemetry", None)

    @property
    def last_evaluation(self):
        return getattr(self.evaluator, "last_evaluation", None)


def _exact_root_outcome(
    game: KalahGame, evaluator: Any, native_adapter: object, threshold: int
) -> dict:
    decision = exact_root_decision(
        game,
        evaluator,
        tablebase=native_adapter,
        threshold=threshold,
    )
    if decision is None:
        raise RuntimeConfigurationError("eligible exact-root position was not solved")
    return {
        "selected_move": decision.selected_move,
        "legal_moves": decision.legal_moves,
        "policy": decision.network_priors,
        "value": 0.0,
        "child_stats": [
            {"move": move, "visits": 0, "q_value": 0.0} for move in decision.legal_moves
        ],
        "visits": [0.0] * 6,
        "exact_root_decision": {
            "active_pit_stones": decision.active_pit_stones,
            "root_player": decision.root_player,
            "action_margins": {
                str(move): margin for move, margin in decision.action_margins.items()
            },
            "wdl": {str(move): value for move, value in decision.wdl.items()},
            "optimal_moves": decision.optimal_moves,
            "network_priors": decision.network_priors,
            "solver_calls": decision.solver_calls,
            "root_latency_ms": decision.root_latency_ms,
            "puct_simulations_executed": 0,
        },
        "root_latency_ms": decision.root_latency_ms,
    }


class MoveEngine:
    """Load all resources once, then validate, search, and bind every response."""

    def __init__(
        self,
        asset_root: str | Path,
        bundles: Mapping[str, RuntimeBundle],
        profiles: Mapping[str, SearchProfile],
    ) -> None:
        self.asset_root = Path(asset_root).resolve()
        self.bundles = MappingProxyType(dict(bundles))
        self.profiles = MappingProxyType(dict(profiles))
        self._evaluators: dict[str, object] = {}
        self._bundle_ids: dict[str, str] = {}
        self._profile_ids: dict[str, str] = {}
        self._runtime_policy_available: dict[str, bool] = {}
        self._runtime_policies: dict[str, dict | None] = {}
        self._native_resources: dict[str, object] = {}
        self._native_resources_by_identity: dict[tuple[str, str, str], object] = {}
        self._closed = False
        try:
            for key, bundle in self.bundles.items():
                _validate_identifier(key, "bundle key")
                _validate_identifier(bundle.bundle_id, "bundle id")
                if len(f"{bundle.bundle_id}:{'0' * 64}") > 128:
                    raise RuntimeConfigurationError("bundle id is too long")
                if bundle.kind == "composed" and (
                    bundle.policy_source not in {"current", "candidate"}
                    or bundle.value_source not in {"current", "candidate"}
                ):
                    raise RuntimeConfigurationError("invalid composed output routing")
                if bundle.kind == "single" and (
                    bundle.policy_source != "current"
                    or bundle.value_source != "current"
                ):
                    raise RuntimeConfigurationError(
                        "single bundles do not accept composition routing"
                    )
                evaluator, identity, runtime_policy_available = self._load_bundle(
                    key, bundle
                )
                self._evaluators[key] = evaluator
                self._bundle_ids[key] = identity
                self._runtime_policy_available[key] = runtime_policy_available
            for name, profile in self.profiles.items():
                _validate_identifier(name, "profile key")
                _validate_identifier(profile.name, "profile name")
                _validate_identifier(profile.profile_id, "profile id")
                if len(f"{profile.name}:{'0' * 64}") > 128:
                    raise RuntimeConfigurationError("profile name is too long")
                if type(profile.require_runtime_search_policy) is not bool:
                    raise RuntimeConfigurationError("invalid runtime policy setting")
                if profile.bundle_id not in self.bundles:
                    raise RuntimeConfigurationError("profile references unknown bundle")
                if (
                    profile.require_runtime_search_policy
                    and not self._runtime_policy_available[profile.bundle_id]
                ):
                    raise RuntimeConfigurationError(
                        "profile requires a runtime search policy sidecar"
                    )
                raw = json.dumps(
                    {
                        "profile": profile.name,
                        "profile_id": profile.profile_id,
                        "bundle": self._bundle_ids[profile.bundle_id],
                        "settings": profile.settings.__dict__,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                self._profile_ids[name] = (
                    f"{profile.name}:{hashlib.sha256(raw).hexdigest()}"
                )
        except Exception as error:
            self.close()
            if isinstance(error, RuntimeConfigurationError):
                raise
            raise RuntimeConfigurationError(
                "runtime configuration initialization failed"
            ) from error

    def _load_bundle(self, key: str, bundle: RuntimeBundle) -> tuple[object, str, bool]:
        if not isinstance(bundle.artifacts, tuple) or not all(
            isinstance(path, str) and path for path in bundle.artifacts
        ):
            raise RuntimeConfigurationError("artifact paths must be nonempty strings")
        if not isinstance(bundle.expected_artifact_sha256, tuple):
            raise RuntimeConfigurationError(
                "expected artifact identities must be a tuple"
            )
        if not isinstance(bundle.expected_policy_sha256, tuple):
            raise RuntimeConfigurationError(
                "expected policy identities must be a tuple"
            )
        if bundle.kind not in {"single", "composed"}:
            raise RuntimeConfigurationError("unsupported bundle kind")
        if bundle.kind == "single" and len(bundle.artifacts) != 1:
            raise RuntimeConfigurationError("single bundle requires one artifact")
        if bundle.kind == "composed" and len(bundle.artifacts) != 2:
            raise RuntimeConfigurationError(
                "composed bundle requires policy and value artifacts"
            )
        if len(bundle.expected_artifact_sha256) != len(bundle.artifacts):
            raise RuntimeConfigurationError(
                "every artifact requires an expected identity"
            )
        expected_policy_hashes = bundle.expected_policy_sha256 or tuple(
            None for _ in bundle.artifacts
        )
        if len(expected_policy_hashes) != len(bundle.artifacts):
            raise RuntimeConfigurationError("invalid expected policy hash list")
        evaluators = []
        policies = []
        component_digests = []
        for relative, expected_identity, expected_policy_hash in zip(
            bundle.artifacts,
            bundle.expected_artifact_sha256,
            expected_policy_hashes,
            strict=True,
        ):
            if not _is_sha256(expected_identity):
                raise RuntimeConfigurationError("invalid expected artifact identity")
            if not isinstance(relative, str) or not relative:
                raise RuntimeConfigurationError("invalid artifact path")
            path = (self.asset_root / relative).resolve()
            if not path.is_relative_to(self.asset_root):
                raise RuntimeConfigurationError("artifact path escapes asset root")
            metadata = path / "metadata.json"
            weights = path / "weights.json"
            if (
                not metadata.is_file()
                or not weights.is_file()
                or not metadata.resolve().is_relative_to(self.asset_root)
                or not weights.resolve().is_relative_to(self.asset_root)
            ):
                raise RuntimeConfigurationError(
                    f"missing artifact metadata or weights: {relative}"
                )
            metadata_bytes = metadata.read_bytes()
            metadata_obj = json.loads(metadata_bytes)
            weight_hash = _sha256_file(weights)
            declared_hash = metadata_obj.get("artifacts", {}).get("weights_sha256")
            if (
                declared_hash not in (None, "REPLACE_WITH_SHA256")
                and declared_hash != weight_hash
            ):
                raise RuntimeConfigurationError("metadata weight hash mismatch")
            if serving_artifact_identity(metadata, weights) != expected_identity:
                raise RuntimeConfigurationError("artifact identity mismatch")
            sidecar = path / "search_policy.json"
            policy = None
            if sidecar.exists():
                try:
                    if (
                        not sidecar.is_file()
                        or not sidecar.resolve().is_relative_to(self.asset_root)
                        or not _is_sha256(expected_policy_hash)
                    ):
                        raise ValueError("policy requires an expected SHA-256")
                    if _sha256_file(sidecar) != expected_policy_hash:
                        raise ValueError("runtime policy identity mismatch")
                    policy = json.loads(sidecar.read_text(encoding="utf-8"))
                    _validate_runtime_policy(policy, self.asset_root)
                    for key_name in ("native_probe", "tablebase"):
                        reference = policy[key_name]
                        reference["resolved_path"] = str(
                            (self.asset_root / reference["path"]).resolve()
                        )
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    OSError,
                    json.JSONDecodeError,
                ) as error:
                    raise RuntimeConfigurationError(
                        f"invalid runtime search policy: {relative}"
                    ) from error
            elif expected_policy_hash is not None:
                raise RuntimeConfigurationError("expected runtime policy is missing")
            evaluator = ArtifactEvaluator(path)
            if evaluator.input_encoding != metadata_obj.get("input_encoding"):
                raise RuntimeConfigurationError("artifact metadata encoding mismatch")
            architecture = metadata_obj.get("architecture", {})
            if (
                not isinstance(architecture, dict)
                or architecture.get("model_type", evaluator.model_type)
                != evaluator.model_type
            ):
                raise RuntimeConfigurationError("artifact architecture mismatch")
            if metadata_obj.get(
                "feature_count", feature_count_for(evaluator.input_encoding)
            ) != feature_count_for(evaluator.input_encoding):
                raise RuntimeConfigurationError("artifact feature count mismatch")
            if metadata_obj.get("policy_size", 6) != 6:
                raise RuntimeConfigurationError("artifact policy size mismatch")
            try:
                policy_vector, value = evaluator.evaluate(
                    KalahGame([4] * 12, [0, 0], 0)
                )
                if getattr(policy_vector, "shape", None) != (6,) or not math.isfinite(
                    float(value)
                ):
                    raise ValueError("incompatible policy/value output shape")
            except Exception as error:
                raise RuntimeConfigurationError(
                    "artifact architecture or tensor compatibility failure"
                ) from error
            evaluators.append(evaluator)
            policies.append(policy)
            component_digests.append(serving_artifact_identity(metadata, weights))
        present_policies = [policy for policy in policies if policy is not None]
        runtime_policy = None
        if present_policies:
            if len(present_policies) != len(policies):
                raise RuntimeConfigurationError(
                    "runtime policy must be present on every composed component"
                )
            canonical = [
                json.dumps(policy, sort_keys=True, separators=(",", ":"))
                for policy in present_policies
            ]
            if any(item != canonical[0] for item in canonical[1:]):
                raise RuntimeConfigurationError(
                    "composed component runtime policies are incompatible"
                )
            runtime_policy = present_policies[0]
            native_key = (
                runtime_policy["native_probe"]["sha256"],
                runtime_policy["tablebase"]["sha256"],
                runtime_policy["solver_implementation_identity"],
            )
            resource = self._native_resources_by_identity.get(native_key)
            if resource is None:
                resource = NativeExactRootTablebase(
                    runtime_policy["native_probe"]["resolved_path"],
                    runtime_policy["tablebase"]["resolved_path"],
                    warm_on_start=False,
                )
                try:
                    resource.warm()
                except Exception:
                    resource.close()
                    raise
                self._native_resources_by_identity[native_key] = resource
            self._native_resources[key] = resource
        self._runtime_policies[key] = runtime_policy
        if bundle.kind == "single":
            result = evaluators[0]
        else:
            result = ComposedArtifactEvaluator(
                evaluators[0],
                evaluators[1],
                policy_source=bundle.policy_source,
                value_source=bundle.value_source,
            )
        identity_obj = {
            "components": component_digests,
            "kind": bundle.kind,
            "policy_source": bundle.policy_source,
            "value_source": bundle.value_source,
            "runtime_policy": _identity_policy(runtime_policy),
        }
        bundle_hash = hashlib.sha256(_canonical(identity_obj)).hexdigest()
        return result, f"{bundle.bundle_id}:{bundle_hash}", runtime_policy is not None

    def select_move(self, request: MoveRequest | dict) -> MoveResponse:
        """Select a legal actor-relative move; execution deadlines are external."""
        if self._closed:
            raise EngineClosedError("move engine is closed")
        try:
            req = MoveRequest.model_validate(
                request.model_dump(mode="python")
                if isinstance(request, MoveRequest)
                else request
            )
            # Snapshot nested mutable input before any backend work.
            state = req.state.model_copy(deep=True)
            game = to_kalah_game(state)
        except Exception as error:
            raise EngineError("invalid move request") from error
        profile = self.profiles.get(req.profile)
        if profile is None:
            raise UnknownProfileError(f"unknown profile: {req.profile}")
        started = time.perf_counter()
        if not game.possible_moves():
            response = MoveResponse(
                request_id=req.request_id,
                game_revision=req.game_revision,
                state_sha256=state_sha256(state),
                result=TerminalResult(kind="terminal", move=None),
                model_bundle_id=self._bundle_ids[profile.bundle_id],
                search_profile_id=self._profile_ids[req.profile],
                elapsed_ms=0,
                completed_simulations=0,
                termination_reason="terminal",
            )
            return validate_response(
                req,
                response,
                model_bundle_id=self._bundle_ids[profile.bundle_id],
                search_profile_id=self._profile_ids[req.profile],
            )
        # The arena adapter uses fixed ownership: player-relative field names do
        # not imply that the acting player is player 0.
        backend_state = game.to_state()
        runtime_policy = self._runtime_policies[profile.bundle_id]
        threshold = (
            None if runtime_policy is None else runtime_policy["exact_root_threshold"]
        )
        if threshold is not None and sum(game.pits) <= threshold:
            native = self._native_resources.get(profile.bundle_id)
            if native is None:
                raise RuntimeConfigurationError(
                    "exact-root policy has no initialized native adapter"
                )
            outcome = _exact_root_outcome(
                game, self._evaluators[profile.bundle_id], native, threshold
            )
        else:
            outcome = evaluate_artifact_position(
                evaluator=_ServingEvaluator(self._evaluators[profile.bundle_id]),
                state=backend_state,
                simulations=profile.settings.simulations,
                seed=req.seed,
                c_puct=profile.settings.c_puct,
                search_options=profile.settings.options(),
                ablation_mode="full",
            )
        move = outcome.get("selected_move")
        legal = game.possible_moves()
        if type(move) is not int or move not in legal:
            raise InvalidBackendResultError("backend returned an illegal move")
        if (
            outcome.get("model_bundle_id", self._bundle_ids[profile.bundle_id])
            != self._bundle_ids[profile.bundle_id]
        ):
            raise InvalidBackendResultError(
                "backend returned mismatched model identity"
            )
        if "exact_root_decision" in outcome:
            completed = 0
        else:
            completed = int(
                sum(item.get("visits", 0) for item in outcome.get("child_stats", []))
            )
        response = MoveResponse(
            request_id=req.request_id,
            game_revision=req.game_revision,
            state_sha256=state_sha256(state),
            result=MoveResult(kind="move", move=move),
            model_bundle_id=self._bundle_ids[profile.bundle_id],
            search_profile_id=self._profile_ids[req.profile],
            elapsed_ms=max(0, int((time.perf_counter() - started) * 1000)),
            completed_simulations=completed,
            termination_reason="completed",
        )
        try:
            return validate_response(
                req,
                response,
                model_bundle_id=self._bundle_ids[profile.bundle_id],
                search_profile_id=self._profile_ids[req.profile],
            )
        except Exception as error:
            raise InvalidBackendResultError(
                "backend response failed contextual validation"
            ) from error

    def close(self) -> None:
        """Release initialized native resources; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        resources, self._native_resources = self._native_resources, {}
        failures = []
        self._native_resources_by_identity = {}
        unique_resources = {id(resource): resource for resource in resources.values()}
        for resource in unique_resources.values():
            try:
                getattr(resource, "close")()
            except Exception as error:  # cleanup must continue for all resources
                failures.append(error)
        if failures:
            raise RuntimeConfigurationError(
                "failed to close native resources"
            ) from failures[0]

    def __enter__(self) -> MoveEngine:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_identifier(value: object, label: str) -> None:
    try:
        TypeAdapter(Identifier).validate_python(value)
    except Exception as error:
        raise RuntimeConfigurationError(f"invalid {label}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serving_artifact_identity(metadata: str | Path, weights: str | Path) -> str:
    """Return the serving manifest identity for a metadata/weights pair."""
    metadata, weights = Path(metadata), Path(weights)
    digest = hashlib.sha256()
    digest.update(b"azlite-serving-artifact-v1\0")
    for path in (metadata, weights):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_runtime_policy(policy: object, asset_root: Path) -> None:
    from ml.alphazero_lite.runtime_search_policy import EXACT_ROOT_MODE, SCHEMA

    if not isinstance(policy, dict):
        raise ValueError("runtime policy must be an object")
    required_keys = {
        "schema",
        "mode",
        "exact_root_enabled",
        "exact_root_threshold",
        "exact_root_objective",
        "exact_root_tie_rule",
        "exact_leaf_solve",
        "solver_implementation_identity",
        "native_probe",
        "tablebase",
    }
    if set(policy) != required_keys:
        raise ValueError("runtime policy has missing or unsupported fields")
    expected = {
        "schema": SCHEMA,
        "mode": EXACT_ROOT_MODE,
        "exact_root_enabled": True,
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve": "disabled",
        "solver_implementation_identity": "native_kvtb_root_action_probe_v1",
    }
    if any(policy.get(key) != value for key, value in expected.items()):
        raise ValueError("unsupported runtime policy semantics")
    threshold = policy.get("exact_root_threshold")
    if type(threshold) is not int or not 0 <= threshold <= 48:
        raise ValueError("invalid exact-root threshold")
    for name in ("native_probe", "tablebase"):
        reference = policy.get(name)
        if not isinstance(reference, dict):
            raise ValueError(f"missing {name} reference")
        relative, expected_hash = reference.get("path"), reference.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or not _is_sha256(expected_hash)
        ):
            raise ValueError(f"invalid {name} reference")
        resolved = (asset_root / relative).resolve()
        if not resolved.is_relative_to(asset_root) or not resolved.is_file():
            raise ValueError(f"missing or invalid {name} file")
        with resolved.open("rb") as handle:
            prefix = handle.read(128)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"{name} is an unresolved LFS pointer")
        if _sha256_file(resolved) != expected_hash:
            raise ValueError(f"{name} hash mismatch")
        if name == "native_probe" and not os.access(resolved, os.X_OK):
            raise ValueError("native probe is not executable")


def _identity_policy(policy: dict | None) -> object:
    if policy is None:
        return None
    return {key: value for key, value in policy.items() if key != "resolved_path"} | {
        "native_probe": {
            "sha256": policy["native_probe"]["sha256"],
            "identity": "native_kvtb_root_action_probe_v1",
        },
        "tablebase": {"sha256": policy["tablebase"]["sha256"]},
    }
