"""Seed 594 exact-oracle policy-target preparation and frozen training path.

Importing this module performs no inference, training, or artifact writes. Target
construction consumes only the authenticated #557 cohort and oracle records.
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

from ml.alphazero_lite import self_play, train
from ml.alphazero_lite.exact_root_policy_targets import (
    OPTIMAL_SET_UNIFORM,
    exact_root_policy_target,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed551_controlled_policy_strength import (
    canonical_identity,
    production_float32_identity,
)
from ml.alphazero_lite.seed557_retrospective import canonical_hash

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed594-exact-oracle-distillation"
ARTIFACTS_ROOT = (
    ROOT / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts"
)
TRAINING_PATH = "docs/data/seed557-retrospective-diagnostic/cohort.json"
ORACLE_PATH = "docs/data/seed557-retrospective-diagnostic/oracle-records.jsonl"
QUALIFICATION_PATH = (
    "docs/data/seed570-policy-prior-distillation/qualification-membership.json"
)
CONFIG_PATH = "docs/data/seed594-exact-oracle-distillation/training-configuration.json"
FREEZE_PATH = "docs/data/seed594-exact-oracle-distillation/pretraining-freeze.json"

INPUT_HASHES = {
    TRAINING_PATH: "f62539cc3e1b1843a2bd9296453d51b895a1c5e693bd9edc3b3291f883398243",
    ORACLE_PATH: "75453e744a86b9dc8471628e0d133fa7d478550af9c8089ed8661426436a8480",
    QUALIFICATION_PATH: "6447d6ae98a750de4046992fd84e53998843dd1f2f20706375de4c91fa798efc",
    "docs/data/seed570-policy-prior-distillation/training-configuration.json": "0941e5d658a65c006f8ccae7b84f13437ebe3476e2c4c61b1fd26733c0882ea8",
}
MODEL_HASHES = {
    "P-E4/metadata.json": "a9d133043c64ca41e3d77354f4ccf3403ab5f94286608f01589387159ac0258c",
    "P-E4/model.npz": "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
    "P-E4/search_policy.json": "280c33d88266969da72a7984dabdcfea180bacb5cb690cb759af1ebdbaac65bc",
    "P-E4/weights.json": "89dead525a255cff4760867645fd9ce05cb2bbadcf1d72c5ac9089868c4a8136",
    "initializer/metadata.json": "f3b9a7d665be7b1219b4a3412bf4a499e5bafd525905b50d93deac4e6ad9ccef",
    "initializer/model.npz": "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c",
    "initializer/search_policy.json": "280c33d88266969da72a7984dabdcfea180bacb5cb690cb759af1ebdbaac65bc",
    "initializer/weights.json": "f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c",
}
IMPLEMENTATION_HASHES = {
    "ml/alphazero_lite/self_play.py": "76718efde7ccfb708f4bb0a4314f812bab746d1364cb75ce5c8b3551b635fc0c",
    "ml/alphazero_lite/arena.py": "c16a668e8f90bd87b59dbc08f6a1ddb3b5ffc912e21cc0564efaceba37e03fe2",
    "ml/alphazero_lite/train.py": "2dcb42f8dff790d5033e2c9a865f3c24eb1fee6b9b37302200dd176d109dad32",
    "ml/alphazero_lite/kalah_rules.py": "6e5338ef9a85b690f51b6d0d184c8bc0f54c2b28ac084a4a17a88f9f552ad036",
    "ml/alphazero_lite/input_encodings.py": "0a4f56ed60f0c2ed6b93a673dbc0850a966a907912dc26edd092bfd1ac209dbe",
    "ml/alphazero_lite/seed557_retrospective.py": "78dbc90f3339a41f932fd07594237fd7a90606251db5e0c49b0070b0586cc3f2",
    "ml/alphazero_lite/seed551_controlled_policy_strength.py": "c589240e2352273e0b86c48c3b37f5fa9737cfc2423255b67d74e7e177f59e63",
    "ml/alphazero_lite/seed571_policy_distillation.py": "186f93e674b6aeeb6396c3df59449c174660494e7924b2f315003c1ada660c4c",
    "ml/alphazero_lite/exact_root_policy_targets.py": "2aca4c1e2e9e9e199fb1d951b143c506785c23121531e26628a59c9afaf79898",
}
TRAINABLE_PREFIXES = (
    "input_layer.",
    "residual_layers.",
    "policy_hidden_layer.",
    "policy_head.",
)
RUNTIME = {
    "python": "3.12.8",
    "numpy": "2.2.6",
    "torch": "2.7.1+cpu",
    "device": "cpu",
    "threads": 1,
    "pythonhashseed": "571",
    "omp_num_threads": "1",
    "mkl_num_threads": "1",
    "deterministic_algorithms": True,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def fixed_order(seed: int = 571) -> list[int]:
    order = list(range(256))
    random.Random(seed).shuffle(order)
    return order


def paired_opening_bootstrap(
    opening_ids: list[int], differences: list[float]
) -> dict[str, Any]:
    """Frozen Stage-1 mean and percentile interval; no model inference occurs."""
    if len(opening_ids) != len(differences) or not opening_ids:
        raise ValueError("bootstrap_input_mismatch")
    clusters: dict[int, list[float]] = {}
    for opening, difference in zip(opening_ids, differences, strict=True):
        value = float(difference)
        if not np.isfinite(value):
            raise ValueError("bootstrap_nonfinite_difference")
        clusters.setdefault(int(opening), []).append(value)
    ids = np.asarray(sorted(clusters), dtype=np.int64)
    cluster_means = np.asarray(
        [np.mean(clusters[int(opening)]) for opening in ids], dtype=np.float64
    )
    if len(ids) != 423:
        raise ValueError("qualification_opening_count_mismatch")
    rng = np.random.default_rng(594)
    sampled = rng.integers(0, len(ids), size=(20_000, len(ids)))
    means = cluster_means[sampled].mean(axis=1)
    return {
        "mean_difference": float(cluster_means.mean()),
        "ci95": [
            float(value)
            for value in np.quantile(means, [0.025, 0.975], method="linear")
        ],
        "resamples": 20_000,
        "seed": 594,
        "cluster_count": len(ids),
        "quantile_method": "linear",
    }


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


def verify_frozen_parameters(
    model: torch.nn.Module, snapshot: dict[str, torch.Tensor]
) -> None:
    parameters = dict(model.named_parameters())
    if set(snapshot) - set(parameters):
        raise ValueError("frozen_parameter_set_changed")
    if any(
        not torch.equal(parameters[name].detach(), value)
        for name, value in snapshot.items()
    ):
        raise ValueError("frozen_parameter_changed")


def verify_hashes(root: Path = ROOT) -> dict[str, str]:
    expected = dict(INPUT_HASHES)
    expected.update(IMPLEMENTATION_HASHES)
    expected.update(
        {f"{ARTIFACTS_ROOT.relative_to(ROOT)}/{p}": h for p, h in MODEL_HASHES.items()}
    )
    expected["ml/alphazero_lite/seed594_exact_oracle_distillation.py"] = sha256_bytes(
        Path(__file__).read_bytes()
    )
    for rel, digest in expected.items():
        path = root / rel
        if not path.is_file() or sha256_bytes(path.read_bytes()) != digest:
            raise ValueError(f"source_or_artifact_hash_mismatch:{rel}")
    return expected


def verify_runtime() -> None:
    actual = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    if actual != {key: RUNTIME[key] for key in actual}:
        raise ValueError(f"runtime_mismatch:{actual}")
    if (
        os.environ.get("PYTHONHASHSEED") != "571"
        or os.environ.get("OMP_NUM_THREADS") != "1"
        or os.environ.get("MKL_NUM_THREADS") != "1"
    ):
        raise ValueError("runtime_environment_mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def exact_targets(
    cohort: dict[str, Any], oracle_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join complete exact margins to frozen roots and make uniform optimum targets."""
    roots = cohort.get("states")
    if not isinstance(roots, list) or len(roots) != 256:
        raise ValueError("training_root_count_mismatch")
    if len(oracle_rows) != 256:
        raise ValueError("oracle_record_count_mismatch")
    oracle_by_hash: dict[str, dict[str, Any]] = {}
    for row in oracle_rows:
        state_hash = row.get("state_hash")
        if not isinstance(state_hash, str) or state_hash in oracle_by_hash:
            raise ValueError("duplicate_or_invalid_oracle_state")
        oracle_by_hash[state_hash] = row
    if len(oracle_by_hash) != len(oracle_rows):
        raise ValueError("duplicate_oracle_state")
    output = []
    for root in roots:
        state = root["state"]
        state_hash = canonical_hash(state)
        if root.get("state_hash") != state_hash:
            raise ValueError("training_state_hash_mismatch")
        oracle = oracle_by_hash.get(state_hash)
        if oracle is None:
            raise ValueError("missing_oracle_state")
        game = KalahGame.from_state(state)
        legal = sorted(int(action) for action in game.possible_moves())
        if sorted(
            int(action) for action in root.get("legal_absolute_actions", [])
        ) != sorted(game.pit_index(a) for a in game.possible_moves()):
            raise ValueError("training_absolute_legal_action_mismatch")
        if oracle.get("coverage_complete") is not True:
            raise ValueError("oracle_coverage_incomplete")
        if oracle.get("legal_moves") != legal:
            raise ValueError("oracle_legal_action_mismatch")
        margins = oracle.get("action_margins")
        if not isinstance(margins, dict):
            raise ValueError("oracle_margins_missing")
        try:
            margin_actions = [int(action) for action in margins]
            margin_values = {
                int(action): int(value) for action, value in margins.items()
            }
        except (TypeError, ValueError) as error:
            raise ValueError("oracle_margin_invalid") from error
        if len(margin_actions) != len(set(margin_actions)) or set(
            margin_actions
        ) != set(legal):
            raise ValueError("oracle_action_coverage_mismatch")
        if any(not isinstance(value, int) for value in margins.values()):
            raise ValueError("oracle_margin_not_integer")
        best = max(margin_values[action] for action in legal)
        optimal = [action for action in legal if margin_values[action] == best]
        target = exact_root_policy_target(
            selected_action=optimal[0],
            optimal_actions=optimal,
            mode=OPTIMAL_SET_UNIFORM,
        )
        if (
            not np.isclose(sum(target), 1.0)
            or any(target[action] <= 0 for action in optimal)
            or any(target[action] != 0 for action in range(6) if action not in optimal)
        ):
            raise ValueError("exact_target_normalization_failure")
        output.append(
            {
                "state_hash": state_hash,
                "legal_actions": legal,
                "action_margins": {str(a): margin_values[a] for a in legal},
                "optimal_actions": optimal,
                "target": target,
            }
        )
    if len(output) != 256:
        raise ValueError("target_coverage_not_one_to_one")
    return output


def verify_separation(root: Path = ROOT) -> dict[str, Any]:
    training = json.loads((root / TRAINING_PATH).read_bytes())["states"]
    qualification_bundle = json.loads((root / QUALIFICATION_PATH).read_bytes())
    qualification = qualification_bundle.get("roots", [])
    if len(training) != 256 or len(qualification) != 423:
        raise ValueError("cohort_count_mismatch")
    training_openings = {int(row["opening_index"]) for row in training}
    training_canonical = {canonical_identity(row["state"]) for row in training}
    training_float = {production_float32_identity(row["state"]) for row in training}
    for row in qualification:
        state = row["state"]
        if int(row["opening_index"]) in training_openings:
            raise ValueError("qualification_training_opening_overlap")
        if canonical_identity(state) in training_canonical:
            raise ValueError("qualification_training_canonical_overlap")
        if production_float32_identity(state) in training_float:
            raise ValueError("qualification_training_float32_overlap")
    return {
        "training_roots": 256,
        "qualification_roots": 423,
        "opening_overlap": 0,
        "canonical_overlap": 0,
        "production_float32_overlap": 0,
        "qualification_membership_sha256": INPUT_HASHES[QUALIFICATION_PATH],
    }


def load_frozen_targets(
    root: Path = ROOT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    verify_hashes(root)
    verify_runtime()
    verify_separation(root)
    cohort = json.loads((root / TRAINING_PATH).read_bytes())
    rows = exact_targets(cohort, _read_jsonl(root / ORACLE_PATH))
    target_path = (
        root / "docs/data/seed594-exact-oracle-distillation/exact-targets.jsonl"
    )
    target_identity = json.loads((root / FREEZE_PATH).read_bytes())["target_identity"]
    expected_digest = target_identity["canonical_sha256"]
    if sha256_bytes(canonical_json(rows)) != expected_digest:
        raise ValueError("exact_target_identity_mismatch")
    if (
        not target_path.is_file()
        or sha256_bytes(target_path.read_bytes()) != target_identity["file_sha256"]
    ):
        raise ValueError("exact_target_file_hash_mismatch")
    if _read_jsonl(target_path) != rows:
        raise ValueError("exact_target_file_content_mismatch")
    return cohort, rows


def verify_pretraining_freeze(root: Path = ROOT) -> None:
    freeze = json.loads((root / FREEZE_PATH).read_bytes())
    config = json.loads((root / CONFIG_PATH).read_bytes())
    if freeze.get("status") != "frozen-pretraining-not-authorized":
        raise ValueError("pretraining_status_mismatch")
    if freeze.get("configuration_sha256") != sha256_bytes(
        (root / CONFIG_PATH).read_bytes()
    ):
        raise ValueError("configuration_hash_mismatch")
    if freeze.get("runner_sha256") != sha256_bytes(
        (root / "ml/alphazero_lite/seed594_exact_oracle_distillation.py").read_bytes()
    ):
        raise ValueError("runner_hash_mismatch")
    if freeze.get("authenticated_input_hashes") != INPUT_HASHES:
        raise ValueError("frozen_input_hashes_mismatch")
    if freeze.get("model_artifact_hashes") != MODEL_HASHES:
        raise ValueError("frozen_model_hashes_mismatch")
    if freeze.get("implementation_hashes") != IMPLEMENTATION_HASHES:
        raise ValueError("frozen_implementation_hashes_mismatch")
    if config.get("training", {}).get("batch_order") != fixed_order():
        raise ValueError("matched_batch_order_mismatch")
    verify_hashes(root)
    verify_runtime()
    verify_separation(root)
    load_frozen_targets(root)


def prepare_target_file(root: Path = ROOT) -> dict[str, Any]:
    """Deterministically materialize the target artifact; this does not train/search."""
    verify_hashes(root)
    verify_runtime()
    verify_separation(root)
    cohort = json.loads((root / TRAINING_PATH).read_bytes())
    rows = exact_targets(cohort, _read_jsonl(root / ORACLE_PATH))
    destination = (
        root / "docs/data/seed594-exact-oracle-distillation/exact-targets.jsonl"
    )
    if destination.exists():
        raise ValueError("target_output_already_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    return {
        "root_count": len(rows),
        "canonical_sha256": sha256_bytes(canonical_json(rows)),
        "file_sha256": sha256_bytes(destination.read_bytes()),
    }


def train_frozen(output_dir: Path, *, root: Path = ROOT) -> dict[str, Any]:
    """Run the frozen 512-update schedule only after a distinct authorization freeze."""
    verify_pretraining_freeze(root)
    if output_dir.exists():
        raise ValueError("output_directory_exists")
    authorization = json.loads(
        (root / DATA.relative_to(ROOT) / "execution-freeze.json").read_bytes()
    )
    expected_authorization = {
        "configuration_sha256": sha256_bytes((root / CONFIG_PATH).read_bytes()),
        "pretraining_freeze_sha256": sha256_bytes((root / FREEZE_PATH).read_bytes()),
        "runner_sha256": sha256_bytes(
            (
                root / "ml/alphazero_lite/seed594_exact_oracle_distillation.py"
            ).read_bytes()
        ),
        "target_file_sha256": json.loads((root / FREEZE_PATH).read_bytes())[
            "target_identity"
        ]["file_sha256"],
        "target_canonical_sha256": json.loads((root / FREEZE_PATH).read_bytes())[
            "target_identity"
        ]["canonical_sha256"],
        "training_cohort_sha256": INPUT_HASHES[TRAINING_PATH],
        "oracle_records_sha256": INPUT_HASHES[ORACLE_PATH],
        "qualification_membership_sha256": INPUT_HASHES[QUALIFICATION_PATH],
        "model_hashes": MODEL_HASHES,
    }
    if (
        authorization.get("schema") != "seed594-execution-freeze-v1"
        or authorization.get("status") != "authorized-for-pretraining"
        or authorization.get("bindings") != expected_authorization
    ):
        raise ValueError("execution_not_authorized")
    cohort, rows = load_frozen_targets(root)
    train.set_seed(571)
    model = train.PolicyValueNet(
        hidden_sizes=(96, 3), model_type="residual_v3", input_size=27
    ).cpu()
    model_path = root / ARTIFACTS_ROOT.relative_to(ROOT) / "P-E4/model.npz"
    if train.load_checkpoint_into_model(model, model_path):
        raise ValueError("parent_checkpoint_tensor_mismatch")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE_PREFIXES))
    frozen = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if not p.requires_grad
    }
    x = torch.tensor(
        [
            self_play.encode_state(r["state"], input_encoding="kalah_v3")
            for r in cohort["states"]
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([r["target"] for r in rows], dtype=torch.float32)
    legal = torch.zeros((256, 6), dtype=torch.bool)
    for i, row in enumerate(rows):
        legal[i, row["legal_actions"]] = True
    order = fixed_order()
    batches = [order[i : i + 32] for i in range(0, 256, 32)]
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    output_dir.mkdir(parents=True)
    trace = (output_dir / "training-trace.jsonl").open("x", encoding="utf-8")
    with trace:
        for update in range(512):
            indexes = batches[update % len(batches)]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x[indexes])
            loss = masked_policy_loss(logits, targets[indexes], legal[indexes])
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite_loss:{update + 1}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError(f"nonfinite_parameter:{update + 1}")
            trace.write(
                json.dumps(
                    {
                        "update": update + 1,
                        "root_indexes": indexes,
                        "loss": float(loss.detach()),
                        "gradient_norm": float(grad),
                        "seed": 571,
                        "runtime": RUNTIME,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            trace.flush()
    verify_frozen_parameters(model, frozen)
    checkpoint = output_dir / "seed594-exact-oracle-update-512.npz"
    arrays = train.checkpoint_from_model(model)
    np.savez(checkpoint, **arrays)
    identity = {
        "checkpoint": checkpoint.name,
        "sha256": sha256_bytes(checkpoint.read_bytes()),
        "target_sha256": json.loads((root / FREEZE_PATH).read_bytes())[
            "target_identity"
        ]["canonical_sha256"],
        "training_roots": 256,
        "updates": 512,
        "frozen_parameters_unchanged": True,
        "parent_policy_checkpoint_sha256": MODEL_HASHES["P-E4/model.npz"],
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
    if args.verify_only:
        verify_pretraining_freeze()
        print(
            "Seed 594 pretraining freeze verified; no training or inference performed"
        )
        return
    if args.run:
        if args.output_dir is None:
            parser.error("--output-dir required with --run")
        print(json.dumps(train_frozen(args.output_dir), sort_keys=True))
        return
    parser.error("select --verify-only or --run")


if __name__ == "__main__":
    main()
