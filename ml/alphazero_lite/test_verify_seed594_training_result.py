from __future__ import annotations

import json

import numpy as np
import pytest

from ml.alphazero_lite import seed594_exact_oracle_distillation as seed594
from ml.alphazero_lite import train
from ml.alphazero_lite.verify_seed594_training_result import verify_training_result


@pytest.fixture
def synthetic_result(tmp_path):
    output = tmp_path / "result"
    output.mkdir()
    config = json.loads((seed594.ROOT / seed594.CONFIG_PATH).read_bytes())
    order = config["training"]["batch_order"]
    batches = [order[i : i + 32] for i in range(0, 256, 32)]
    trace = [
        {
            "update": update + 1,
            "root_indexes": batches[update % 8],
            "loss": 0.5,
            "gradient_norm": 1.0,
            "seed": 571,
            "runtime": seed594.RUNTIME,
        }
        for update in range(512)
    ]
    trace_path = output / "training-trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in trace)
    )
    parent_path = (
        seed594.ROOT
        / seed594.ARTIFACTS_ROOT.relative_to(seed594.ROOT)
        / "P-E4/model.npz"
    )
    model = train.PolicyValueNet(
        hidden_sizes=(96, 3), model_type="residual_v3", input_size=27
    ).cpu()
    assert not train.load_checkpoint_into_model(model, parent_path)
    arrays = train.checkpoint_from_model(model)
    checkpoint_path = output / "seed594-exact-oracle-update-512.npz"
    np.savez(checkpoint_path, **arrays)
    freeze = json.loads((seed594.ROOT / seed594.FREEZE_PATH).read_bytes())
    identity = {
        "checkpoint": checkpoint_path.name,
        "sha256": seed594.sha256_bytes(checkpoint_path.read_bytes()),
        "target_sha256": freeze["target_identity"]["canonical_sha256"],
        "training_roots": 256,
        "updates": 512,
        "frozen_parameters_unchanged": True,
        "parent_policy_checkpoint_sha256": seed594.MODEL_HASHES["P-E4/model.npz"],
        "tensors": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(arrays.items())
        },
    }
    (output / "checkpoint-identity.json").write_text(json.dumps(identity))
    return output


def test_verifies_synthetic_complete_result(synthetic_result):
    receipt = verify_training_result(synthetic_result, root=seed594.ROOT)
    assert receipt["updates"] == 512
    assert receipt["frozen_parameters_unchanged"] is True
    assert (
        receipt["target_file_sha256"]
        == json.loads((seed594.ROOT / seed594.FREEZE_PATH).read_bytes())[
            "target_identity"
        ]["file_sha256"]
    )
    assert receipt["initializer_sha256"] == seed594.MODEL_HASHES["P-E4/model.npz"]


def _rewrite_checkpoint_and_identity(output, arrays, *, dtype_updates=None):
    checkpoint = output / "seed594-exact-oracle-update-512.npz"
    np.savez(checkpoint, **arrays)
    identity_path = output / "checkpoint-identity.json"
    identity = json.loads(identity_path.read_bytes())
    identity["sha256"] = seed594.sha256_bytes(checkpoint.read_bytes())
    if dtype_updates:
        for name, value in dtype_updates.items():
            identity["tensors"][name]["dtype"] = str(value.dtype)
    identity_path.write_text(json.dumps(identity))


def _checkpoint_arrays(output):
    checkpoint = output / "seed594-exact-oracle-update-512.npz"
    with np.load(checkpoint, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def test_modified_trainable_checkpoint_tensor_is_accepted(synthetic_result):
    arrays = _checkpoint_arrays(synthetic_result)
    arrays["w_policy"].flat[0] += 0.25
    _rewrite_checkpoint_and_identity(synthetic_result, arrays)
    receipt = verify_training_result(synthetic_result, root=seed594.ROOT)
    assert receipt["status"] == "verified"


def test_modified_frozen_tensor_is_rejected_with_updated_identity(synthetic_result):
    arrays = _checkpoint_arrays(synthetic_result)
    arrays["w_value"].flat[0] += 0.25
    _rewrite_checkpoint_and_identity(synthetic_result, arrays)
    with pytest.raises(ValueError, match="frozen_parameter_changed:w_value"):
        verify_training_result(synthetic_result, root=seed594.ROOT)


def test_changed_checkpoint_dtype_is_rejected_with_updated_identity(synthetic_result):
    arrays = _checkpoint_arrays(synthetic_result)
    arrays["w_policy"] = arrays["w_policy"].astype(np.float64)
    _rewrite_checkpoint_and_identity(
        synthetic_result, arrays, dtype_updates={"w_policy": arrays["w_policy"]}
    )
    with pytest.raises(ValueError, match="checkpoint_tensor_dtype_mismatch:w_policy"):
        verify_training_result(synthetic_result, root=seed594.ROOT)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda authorization: authorization.update(status="pending-independent-review"),
        lambda authorization: authorization["authorization"].update(
            training_may_start=False
        ),
    ],
)
def test_incomplete_execution_authorization_is_rejected(tmp_path, mutation):
    authorization_path = (
        tmp_path / seed594.DATA.relative_to(seed594.ROOT) / "execution-freeze.json"
    )
    authorization_path.parent.mkdir(parents=True)
    authorization = json.loads(
        (
            seed594.ROOT
            / seed594.DATA.relative_to(seed594.ROOT)
            / "execution-freeze.json"
        ).read_bytes()
    )
    mutation(authorization)
    authorization_path.write_text(json.dumps(authorization))
    with pytest.raises(ValueError, match="execution_not_authorized"):
        seed594.verify_execution_freeze(tmp_path)


def test_execution_freeze_records_owner_approval_without_independent_review_claim():
    authorization = seed594.verify_execution_freeze()
    assert authorization["authorization"]["authorized_by"] == "project owner"
    assert authorization["authorization"]["training_may_start"] is True
    assert authorization["authorization"]["independent_human_review"] == "not-performed"


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("batch", "training_trace_batch_identity_mismatch"),
        ("step", "training_trace_update_identity_mismatch"),
        ("identity", "checkpoint_identity_binding_mismatch"),
        ("nonfinite", "checkpoint_tensor_nonfinite"),
        ("shape", "checkpoint_tensor_shape_mismatch"),
    ],
)
def test_rejects_mutated_result(synthetic_result, mutation, message):
    trace_path = synthetic_result / "training-trace.jsonl"
    identity_path = synthetic_result / "checkpoint-identity.json"
    checkpoint_path = synthetic_result / "seed594-exact-oracle-update-512.npz"
    if mutation in {"batch", "step"}:
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
        if mutation == "batch":
            trace[0]["root_indexes"][0] = 0
        else:
            trace[0]["update"] = 2
        trace_path.write_text("".join(json.dumps(row) + "\n" for row in trace))
    elif mutation == "identity":
        identity = json.loads(identity_path.read_bytes())
        identity["target_sha256"] = "0" * 64
        identity_path.write_text(json.dumps(identity))
    else:
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        key = sorted(arrays)[0]
        if mutation == "nonfinite":
            arrays[key] = arrays[key].copy()
            arrays[key].flat[0] = np.nan
        else:
            arrays[key] = arrays[key].reshape(arrays[key].shape + (1,))
        np.savez(checkpoint_path, **arrays)
        identity = json.loads(identity_path.read_bytes())
        identity["sha256"] = seed594.sha256_bytes(checkpoint_path.read_bytes())
        identity_path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match=message):
        verify_training_result(synthetic_result, root=seed594.ROOT)
