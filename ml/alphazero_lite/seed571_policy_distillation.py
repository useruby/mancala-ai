"""Frozen #571 policy-prior distillation pipeline (preparation only).

This module has no import-time search, training, or artifact-writing side effects.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import arena, self_play, train
from ml.alphazero_lite.input_encodings import encode_state
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed557_retrospective import canonical_hash, search_seed

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed570-policy-prior-distillation"
TRAINING_COHORT = ROOT / "docs/data/seed557-retrospective-diagnostic/cohort.json"
ARTIFACT_ROOT = (
    ROOT / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts"
)
SOURCES = {
    "ml/alphazero_lite/seed571_policy_distillation.py": "SELF",
    "ml/alphazero_lite/self_play.py": "76718efde7ccfb708f4bb0a4314f812bab746d1364cb75ce5c8b3551b635fc0c",
    "ml/alphazero_lite/arena.py": "c16a668e8f90bd87b59dbc08f6a1ddb3b5ffc912e21cc0564efaceba37e03fe2",
    "ml/alphazero_lite/train.py": "2dcb42f8dff790d5033e2c9a865f3c24eb1fee6b9b37302200dd176d109dad32",
    "ml/alphazero_lite/kalah_rules.py": "6e5338ef9a85b690f51b6d0d184c8bc0f54c2b28ac084a4a17a88f9f552ad036",
    "ml/alphazero_lite/input_encodings.py": "0a4f56ed60f0c2ed6b93a673dbc0850a966a907912dc26edd092bfd1ac209dbe",
    "docs/data/seed557-retrospective-diagnostic/cohort.json": "f62539cc3e1b1843a2bd9296453d51b895a1c5e693bd9edc3b3291f883398243",
}
ARTIFACTS = {
    f"{component}/{name}": digest
    for component, expected in (
        (
            "P-E4",
            {
                "metadata.json": "a9d133043c64ca41e3d77354f4ccf3403ab5f94286608f01589387159ac0258c",
                "model.npz": "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
                "search_policy.json": "280c33d88266969da72a7984dabdcfea180bacb5cb690cb759af1ebdbaac65bc",
                "weights.json": "89dead525a255cff4760867645fd9ce05cb2bbadcf1d72c5ac9089868c4a8136",
            },
        ),
        (
            "initializer",
            {
                "metadata.json": "f3b9a7d665be7b1219b4a3412bf4a499e5bafd525905b50d93deac4e6ad9ccef",
                "model.npz": "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c",
                "search_policy.json": "280c33d88266969da72a7984dabdcfea180bacb5cb690cb759af1ebdbaac65bc",
                "weights.json": "f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c",
            },
        ),
    )
    for name, digest in expected.items()
}

CONFIG: dict[str, Any] = {
    "schema": "seed571-policy-distillation-config-v1",
    "training_membership": {
        "path": "docs/data/seed557-retrospective-diagnostic/cohort.json",
        "sha256": SOURCES["docs/data/seed557-retrospective-diagnostic/cohort.json"],
        "roots": 256,
    },
    "qualification_membership": {
        "path": "docs/data/seed570-policy-prior-distillation/qualification-membership.json",
        "sha256": "94e126af6ea88fb99a4230f84e5cf2a20248daa39ea66d25b68730ea0426d668",
        "roots": 423,
        "use": "identity binding only; never opened by target generation or training",
    },
    "teacher": "P-E4 policy+trunk; initializer value+trunk, both authenticated against #557 execution freeze",
    "search": {
        "implementation": "ml/alphazero_lite/self_play.py:PUCT",
        "simulations": 128,
        "c_puct": 1.25,
        "workers": 1,
        "fresh_tree": True,
        "fpu_mode": "zero",
        "reuse_subtree": False,
        "normalize_values": False,
        "root_policy_mode": "deterministic",
        "root_temperature": 0.0,
        "tactical_root_bias": 0.0,
        "dirichlet_noise": False,
    },
    "target": "six-action visit counts divided by total visits; illegal actions zero; unvisited legal actions zero",
    "model": {
        "type": "residual_v3",
        "input_encoding": "kalah_v3",
        "hidden_sizes": [96, 3],
        "initialization": "P-E4 exact checkpoint",
        "trainable": ["shared trunk", "policy hidden/head"],
        "frozen": [
            "value hidden/head initialized from initializer",
            "all other parameters",
        ],
    },
    "training": {
        "seed": 571,
        "batch_size": 32,
        "updates": 512,
        "order": "one fixed seeded permutation of 256 roots, cycled in contiguous batches of 32; recorded permutation is immutable",
        "optimizer": "Adam",
        "learning_rate": 1e-5,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "loss": "legal-action masked soft-target cross entropy, mean over batch",
        "regularization": "none",
        "gradient_clip_norm": 1.0,
        "checkpoint": "update-512 only; no interim checkpoint selection",
    },
    "failure_conditions": [
        "any input or configuration digest mismatch",
        "root count or legal-action mismatch",
        "search visit sum differs from 128",
        "nonfinite or negative target/loss/gradient/parameter",
        "any frozen parameter changes",
        "training step count or fixed batch order mismatch",
        "checkpoint output already exists",
    ],
    "outputs": {
        "checkpoint": "seed571-policy-distillation-update-512.npz",
        "teacher_visits": "teacher-visits.jsonl",
        "target_digest": "targets.sha256",
        "training_trace": "training-trace.jsonl",
        "checkpoint_identity": "checkpoint-identity.json",
    },
    "runtime": {"python": "3.12.3", "numpy": "2.2.6", "torch": "2.7.1+cpu"},
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def normalized_visit_target(
    visits: list[int] | np.ndarray, legal: list[int]
) -> np.ndarray:
    values = np.asarray(visits, dtype=np.float64)
    if values.shape != (6,) or not legal or len(set(legal)) != len(legal):
        raise ValueError("invalid_visit_target_shape_or_legal_actions")
    if (
        any(move < 0 or move >= 6 for move in legal)
        or np.any(values < 0)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("invalid_visit_counts")
    if np.any(values[[move for move in range(6) if move not in legal]] != 0):
        raise ValueError("illegal_action_has_visits")
    total = float(values[legal].sum())
    if total <= 0:
        raise ValueError("zero_root_visits")
    target = np.zeros(6, dtype=np.float32)
    target[legal] = values[legal] / total
    return target


def masked_policy_loss(
    logits: torch.Tensor, targets: torch.Tensor, legal: torch.Tensor
) -> torch.Tensor:
    if logits.shape != targets.shape or logits.shape != legal.shape or logits.ndim != 2:
        raise ValueError("policy_batch_shape_mismatch")
    if not torch.all(legal.any(dim=1)) or torch.any(targets.masked_select(~legal) != 0):
        raise ValueError("target_violates_legal_mask")
    if not torch.allclose(
        targets.sum(dim=1), torch.ones(len(targets), device=targets.device), atol=1e-6
    ):
        raise ValueError("target_mass_mismatch")
    masked = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
    return -(targets * torch.log_softmax(masked, dim=1)).sum(dim=1).mean()


def verify_inputs(
    root: Path = ROOT, *, expected: dict[str, str] | None = None
) -> dict[str, str]:
    """Authenticate all inputs before a search/training function can be called."""
    bindings = dict(expected or {})
    for rel, digest in SOURCES.items():
        if digest == "SELF":
            bindings[rel] = sha((root / rel).read_bytes())
        elif digest:
            bindings[rel] = digest
    for rel, digest in ARTIFACTS.items():
        if digest:
            bindings[f"{ARTIFACT_ROOT.relative_to(ROOT)}/{rel}"] = digest
    verify_bindings(root, bindings)
    return bindings


def verify_bindings(root: Path, bindings: dict[str, str]) -> None:
    for rel, digest in bindings.items():
        path = root / rel
        if not path.is_file() or sha(path.read_bytes()) != digest:
            raise ValueError(f"input_binding_mismatch:{rel}")


def verify_configuration(config_path: Path, expected_sha256: str) -> None:
    if not config_path.is_file() or sha(config_path.read_bytes()) != expected_sha256:
        raise ValueError("configuration_binding_mismatch")


def verify_runtime() -> None:
    actual = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    expected = {key: CONFIG["runtime"][key] for key in actual}
    if actual != expected:
        raise ValueError(f"runtime_binding_mismatch:{actual}")
    if os.environ.get("PYTHONHASHSEED") != "571":
        raise ValueError("runtime_binding_mismatch:PYTHONHASHSEED")
    if (
        os.environ.get("OMP_NUM_THREADS") != "1"
        or os.environ.get("MKL_NUM_THREADS") != "1"
    ):
        raise ValueError("runtime_binding_mismatch:thread_environment")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def verify_pretraining_freeze(root: Path = ROOT) -> None:
    freeze_path = (
        root / "docs/data/seed570-policy-prior-distillation/pretraining-freeze.json"
    )
    config_path = (
        root / "docs/data/seed570-policy-prior-distillation/training-configuration.json"
    )
    freeze = json.loads(freeze_path.read_bytes())
    if freeze.get("status") not in {
        "frozen-pretraining-not-authorized",
        "authorized-for-pretraining",
    }:
        raise ValueError("pretraining_freeze_status_mismatch")
    if freeze.get("runner", {}).get("sha256") != sha(
        (root / "ml/alphazero_lite/seed571_policy_distillation.py").read_bytes()
    ):
        raise ValueError("runner_binding_mismatch")
    verify_configuration(config_path, freeze.get("configuration", {}).get("sha256", ""))
    if freeze.get("teacher_artifact_hashes") != ARTIFACTS:
        raise ValueError("teacher_artifact_freeze_mismatch")
    if (
        freeze.get("qualification_identity_only", {}).get("membership_sha256")
        != CONFIG["qualification_membership"]["sha256"]
    ):
        raise ValueError("qualification_membership_freeze_mismatch")
    verify_runtime()


def make_teacher() -> arena.ComposedArtifactEvaluator:
    initializer = arena.ArtifactEvaluator(ARTIFACT_ROOT / "initializer")
    policy = arena.ArtifactEvaluator(ARTIFACT_ROOT / "P-E4")
    return arena.ComposedArtifactEvaluator(
        initializer, policy, policy_source="candidate", value_source="current"
    )


def generate_targets(
    *, root: Path = ROOT, evaluator: Any | None = None
) -> list[dict[str, Any]]:
    # Complete validation occurs before opening cohort data or constructing/running search.
    verify_inputs(root)
    verify_pretraining_freeze(root)
    cohort = json.loads((root / CONFIG["training_membership"]["path"]).read_bytes())
    if len(cohort.get("states", [])) != 256:
        raise ValueError("training_root_count_mismatch")
    teacher = evaluator if evaluator is not None else make_teacher()
    rows = []
    for root_row in cohort["states"]:
        game = KalahGame.from_state(root_row["state"])
        legal = sorted(game.possible_moves())
        if legal != sorted(int(x) for x in root_row["legal_absolute_actions"]):
            raise ValueError("cohort_legal_action_mismatch")
        seed = search_seed(canonical_hash(root_row["state"]))
        search = self_play.PUCT(
            teacher,
            128,
            1.25,
            random.Random(seed),
            fpu_mode="zero",
            reuse_subtree=False,
            normalize_values=False,
            root_policy_mode="deterministic",
            root_temperature=0.0,
            tactical_root_bias=0.0,
        )
        visits, node = search.run(game)
        visit_vector = [
            int(node.children[m].visit_count) if m in node.children else 0
            for m in range(6)
        ]
        if sum(visit_vector) != 128:
            raise ValueError("teacher_visit_budget_mismatch")
        target = normalized_visit_target(visit_vector, legal)
        rows.append(
            {
                "state_hash": canonical_hash(root_row["state"]),
                "search_seed": seed,
                "simulations": 128,
                "c_puct": 1.25,
                "legal_actions": legal,
                "visits": visit_vector,
                "target": target.tolist(),
            }
        )
    return rows


def fixed_order(seed: int = 571) -> list[int]:
    order = list(range(256))
    random.Random(seed).shuffle(order)
    return order


def train_synthetic_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    target: torch.Tensor,
    legal: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)[0]
    loss = masked_policy_loss(logits, target, legal)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite_loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
    )
    optimizer.step()
    return float(loss.detach())


def train_targets(
    target_rows: list[dict[str, Any]], output_dir: Path, *, root: Path = ROOT
) -> dict[str, Any]:
    """Run the frozen 512-update policy/trunk recipe on an authenticated corpus."""
    verify_inputs(root)
    verify_pretraining_freeze(root)
    if output_dir.exists():
        raise ValueError("checkpoint_output_already_exists")
    if len(target_rows) != 256:
        raise ValueError("target_root_count_mismatch")
    cohort = json.loads((root / CONFIG["training_membership"]["path"]).read_bytes())
    expected_hashes = [canonical_hash(row["state"]) for row in cohort["states"]]
    if [row.get("state_hash") for row in target_rows] != expected_hashes:
        raise ValueError("target_root_order_mismatch")
    for source, target in zip(cohort["states"], target_rows, strict=True):
        legal_actions = sorted(KalahGame.from_state(source["state"]).possible_moves())
        if target.get("legal_actions") != legal_actions:
            raise ValueError("target_legal_action_mismatch")
        if sum(target.get("visits", [])) != 128:
            raise ValueError("teacher_visit_budget_mismatch")
        expected_target = normalized_visit_target(target["visits"], legal_actions)
        if not np.array_equal(
            expected_target, np.asarray(target["target"], dtype=np.float32)
        ):
            raise ValueError("target_visit_normalization_mismatch")

    train.set_seed(571)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = train.PolicyValueNet(
        hidden_sizes=(96, 3), model_type="residual_v3", input_size=27
    ).cpu()
    skipped = train.load_checkpoint_into_model(model, ARTIFACT_ROOT / "P-E4/model.npz")
    if skipped:
        raise ValueError(f"teacher_checkpoint_keys_skipped:{skipped}")
    trainable_prefixes = (
        "input_layer.",
        "residual_layers.",
        "policy_hidden_layer.",
        "policy_head.",
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(trainable_prefixes))
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    encoded = [
        encode_state(row["state"], input_encoding="kalah_v3")
        for row in cohort["states"]
    ]
    x = torch.tensor(encoded, dtype=torch.float32)
    targets = torch.tensor([row["target"] for row in target_rows], dtype=torch.float32)
    legal = torch.zeros((256, 6), dtype=torch.bool)
    for index, row in enumerate(target_rows):
        legal[index, row["legal_actions"]] = True
    order = fixed_order()
    batches = [order[index : index + 32] for index in range(0, 256, 32)]
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    output_dir.mkdir(parents=True)
    (output_dir / "teacher-visits.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in target_rows
        )
    )
    target_digest = sha(canonical_json(target_rows))
    (output_dir / "targets.sha256").write_text(target_digest + "\n")
    with (output_dir / "training-trace.jsonl").open("x", encoding="utf-8") as trace:
        for update in range(512):
            indices = batches[update % len(batches)]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x[indices])
            loss = masked_policy_loss(logits, targets[indices], legal[indices])
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite_loss_at_update:{update + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError(f"nonfinite_parameter_at_update:{update + 1}")
            trace.write(
                json.dumps(
                    {
                        "update": update + 1,
                        "root_indexes": indices,
                        "loss": float(loss.detach()),
                        "gradient_norm": float(grad_norm),
                        "seed": 571,
                        "runtime": CONFIG["runtime"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            trace.flush()
    parameters = dict(model.named_parameters())
    if any(
        not torch.equal(parameters[name].detach(), value)
        for name, value in frozen_before.items()
    ):
        raise ValueError("frozen_parameter_changed")
    checkpoint = output_dir / "seed571-policy-distillation-update-512.npz"
    arrays = train.checkpoint_from_model(model)
    np.savez(checkpoint, **arrays)
    identity = {
        "checkpoint": checkpoint.name,
        "sha256": sha(checkpoint.read_bytes()),
        "target_sha256": target_digest,
        "training_roots": 256,
        "updates": 512,
        "frozen_parameters_unchanged": True,
        "parent_policy_checkpoint_sha256": ARTIFACTS["P-E4/model.npz"],
        "initializer_value_checkpoint_sha256": ARTIFACTS["initializer/model.npz"],
        "tensors": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(arrays.items())
        },
    }
    (output_dir / "checkpoint-identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n"
    )
    return identity


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    verify_inputs()
    verify_pretraining_freeze()
    if args.verify_only:
        print("pretraining freeze verified; no search or training performed")
        return
    if not args.run:
        parser.error("choose --verify-only or --run")
    freeze = json.loads((DATA / "pretraining-freeze.json").read_bytes())
    if freeze.get("status") != "authorized-for-pretraining":
        raise SystemExit("pretraining_not_authorized_by_execution_freeze")
    if args.output_dir is None:
        parser.error("--output-dir is required with --run")
    targets = generate_targets()
    print(json.dumps(train_targets(targets, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
