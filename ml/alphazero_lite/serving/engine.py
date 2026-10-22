"""Synchronous, initialized Python move engine for the v1 serving contract."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ml.alphazero_lite.arena import (
    ArtifactEvaluator,
    ComposedArtifactEvaluator,
    evaluate_artifact_position,
)
from ml.alphazero_lite.serving.contract import (
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
        if self.simulations < 1 or self.c_puct <= 0:
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
    policy_source: str = "current"
    value_source: str = "current"


@dataclass(frozen=True)
class SearchProfile:
    name: str
    profile_id: str
    bundle_id: str
    settings: SearchSettings
    deployment: bool = False
    require_runtime_search_policy: bool = False


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
        try:
            for key, bundle in self.bundles.items():
                evaluator, identity, runtime_policy_available = self._load_bundle(
                    bundle
                )
                self._evaluators[key] = evaluator
                self._bundle_ids[key] = identity
                self._runtime_policy_available[key] = runtime_policy_available
            for name, profile in self.profiles.items():
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
            if isinstance(error, RuntimeConfigurationError):
                raise
            raise RuntimeConfigurationError(
                "runtime configuration initialization failed"
            ) from error

    def _load_bundle(self, bundle: RuntimeBundle) -> tuple[object, str, bool]:
        if bundle.kind not in {"single", "composed"}:
            raise RuntimeConfigurationError("unsupported bundle kind")
        if bundle.kind == "single" and len(bundle.artifacts) != 1:
            raise RuntimeConfigurationError("single bundle requires one artifact")
        if bundle.kind == "composed" and len(bundle.artifacts) != 2:
            raise RuntimeConfigurationError(
                "composed bundle requires policy and value artifacts"
            )
        evaluators = []
        runtime_policy_available = False
        digest = hashlib.sha256()
        for relative in bundle.artifacts:
            path = (self.asset_root / relative).resolve()
            if not path.is_relative_to(self.asset_root):
                raise RuntimeConfigurationError("artifact path escapes asset root")
            metadata = path / "metadata.json"
            weights = path / "weights.json"
            if not metadata.is_file() or not weights.is_file():
                raise RuntimeConfigurationError(
                    f"missing artifact metadata or weights: {relative}"
                )
            metadata_obj = json.loads(metadata.read_text(encoding="utf-8"))
            sidecar = path / "search_policy.json"
            if sidecar.exists():
                try:
                    policy = json.loads(sidecar.read_text(encoding="utf-8"))
                    if policy.get("schema") != "azlite_runtime_search_policy_v1":
                        raise ValueError("unsupported runtime policy schema")
                    if policy.get("mode") != "puct_exact_root_hybrid":
                        raise ValueError("unsupported runtime policy mode")
                    for key_name in ("native_probe", "tablebase"):
                        reference = policy[key_name]
                        artifact_path = (self.asset_root / reference["path"]).resolve()
                        if not artifact_path.is_relative_to(self.asset_root):
                            raise ValueError("runtime artifact escapes asset root")
                        if not artifact_path.exists():
                            raise FileNotFoundError(artifact_path)
                        raw = (
                            artifact_path.read_bytes()
                            if artifact_path.is_file()
                            else b""
                        )
                        if (
                            not artifact_path.is_dir()
                            and hashlib.sha256(raw).hexdigest() != reference["sha256"]
                        ):
                            raise ValueError(
                                f"runtime artifact hash mismatch: {key_name}"
                            )
                        if key_name == "tablebase" and raw.startswith(
                            b"version https://git-lfs.github.com/spec/v1"
                        ):
                            raise ValueError(
                                "tablebase is an LFS pointer, not materialized data"
                            )
                        digest.update(raw)
                    runtime_policy_available = True
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
            evaluator = ArtifactEvaluator(path)
            if evaluator.input_encoding != metadata_obj.get(
                "input_encoding", evaluator.input_encoding
            ):
                raise RuntimeConfigurationError("artifact metadata encoding mismatch")
            evaluators.append(evaluator)
            for file_path in (metadata, weights):
                digest.update(file_path.name.encode())
                digest.update(file_path.read_bytes())
        if bundle.kind == "single":
            result = evaluators[0]
        else:
            result = ComposedArtifactEvaluator(
                evaluators[0],
                evaluators[1],
                policy_source=bundle.policy_source,
                value_source=bundle.value_source,
            )
        return (
            result,
            f"{bundle.bundle_id}:{digest.hexdigest()}",
            runtime_policy_available,
        )

    def select_move(self, request: MoveRequest | dict) -> MoveResponse:
        """Select a legal actor-relative move; execution deadlines are external."""
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
        # Arena KalahGame consumes actor-relative fields with current_player=0.
        offset = game.current_player * 6
        opponent_offset = (1 - game.current_player) * 6
        backend_state = {
            "player_pits": list(game.pits[offset : offset + 6]),
            "opponent_pits": list(game.pits[opponent_offset : opponent_offset + 6]),
            "player_store": game.captured_seeds[game.current_player],
            "opponent_store": game.captured_seeds[1 - game.current_player],
            "current_player": 0,
        }
        outcome = evaluate_artifact_position(
            evaluator=self._evaluators[profile.bundle_id],
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
