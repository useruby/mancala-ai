"""Real-arithmetic and receipt-rebound entrypoint tests for seed476 supplement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.alphazero_lite import verify_seed476_transport_supplement as verifier


def test_real_reset_adam_step_clipping_bias_correction_and_writeback():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    opt = torch.optim.Adam([parameter], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    parameter.grad = torch.tensor([3.0, 4.0])
    norm = torch.nn.utils.clip_grad_norm_([parameter], 1.0)
    clipped = parameter.grad.detach().clone()
    opt.state.clear()
    opt.step()
    state = opt.state[parameter]
    assert float(norm) == 5.0
    assert torch.allclose(clipped, torch.tensor([0.6, 0.8]), atol=1e-7)
    assert float(state["step"]) == 1.0
    assert torch.equal(state["exp_avg"], clipped * 0.1)
    assert torch.allclose(state["exp_avg_sq"], clipped.square() * 0.001, atol=1e-9)
    # At step one, bias-corrected Adam uses g/(|g|+eps) in each coordinate.
    expected = torch.tensor([1.0, -2.0]) - 0.01 * clipped / (clipped.abs() + 1e-8)
    assert torch.allclose(parameter.detach(), expected, atol=1e-7)


def test_real_duplicate_targets_multiplicity_equal_input_and_zero_gradient():
    logits = torch.nn.Parameter(torch.zeros((3, 2)))
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    # The first two occurrences are identical inputs but retain distinct targets.
    row_input_identity = ["duplicate", "duplicate", "other"]
    weights = torch.tensor([1.0, 2.0, 3.0])
    loss = -(targets * torch.log_softmax(logits, dim=1)).sum(1)
    grad = torch.autograd.grad((loss * weights).sum() / weights.sum(), logits)[0]
    assert row_input_identity[0] == row_input_identity[1]
    assert not torch.equal(targets[0], targets[1])
    assert torch.equal(grad[0], torch.tensor([-1 / 12, 1 / 12]))
    empty_loss = -(targets * torch.log_softmax(logits, dim=1)).sum(1)
    empty = torch.autograd.grad((empty_loss * torch.zeros_like(weights)).sum(), logits)[
        0
    ]
    assert torch.count_nonzero(empty) == 0


def test_float32_norm_match_writeback_and_buffer_preservation():
    pre = torch.tensor([1.0000001, -3.0], dtype=torch.float32)
    raw = torch.tensor([3.0, 4.0], dtype=torch.float32)
    scaled = (raw.double() * (0.5 / torch.linalg.vector_norm(raw.double()))).float()
    post = pre + scaled
    realized = post.double() - pre.double()
    assert torch.isclose(
        torch.linalg.vector_norm(realized),
        torch.tensor(0.5, dtype=torch.float64),
        atol=2e-7,
    )
    zero_post = pre + torch.zeros_like(pre)
    assert torch.equal(
        zero_post.double() - pre.double(), torch.zeros(2, dtype=torch.float64)
    )
    model = torch.nn.BatchNorm1d(2).eval()
    before = [b.detach().clone() for b in model.buffers()]
    with torch.inference_mode():
        model(torch.ones((4, 2)))
    assert all(torch.equal(x, y) for x, y in zip(before, model.buffers(), strict=True))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path):
    for rel in verifier.SOURCE_CLOSURE:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture source {rel}\n")
    for rel in verifier.INPUTS.values():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture input {rel}".encode())
    base = root / verifier.BASE
    base.mkdir(parents=True, exist_ok=True)
    (base / "registration.json").write_text("{}\n")
    (base / "receipt.json").write_text("{}\n")
    (base / "evidence.json").write_text("{}\n")
    np.savez_compressed(base / "directions.npz", v=np.array([0.2, -0.4]))
    hist = root / verifier.SEED474
    hist.mkdir(parents=True, exist_ok=True)
    (hist / "history.json").write_text("immutable\n")
    out = root / verifier.OUT
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "record_coverage": {
            "steps": 16,
            "required_steps": list(range(1, 17)),
            "training_occurrences": 60015,
            "unseen_occurrences": 859,
        },
        "population": {"training_occurrences": 60015, "unseen_occurrences": 859},
        "steps": [
            {
                "optimizer": {
                    "gradient": 1.0,
                    "moment": 0.1,
                    "counter": 1,
                    "pre": [1.0],
                    "post": [0.9],
                },
                "prediction": {"pre": [0.6, 0.4], "post": [0.5, 0.5]},
                "weights": {"exposure": 1 / 859, "equal_input": 1 / 851},
                "counterfactual_scale": 0.5,
                "contributions": {"first_order": -0.1, "observed": -0.08},
                "remainder": 0.02,
                "category": "both_descent",
                "count": 1,
            }
            for _ in range(16)
        ],
        "totals": {
            "contributions": -1.6,
            "remainders": 0.32,
            "counts": {"both_descent": 16},
        },
    }
    arrays = {
        "population_rows": np.array([4, 4, 7], dtype=np.int64),
        "full_gradient": np.array([1.0, 2.0]),
        "minibatch_gradient": np.array([-1.0, 1.0]),
        "moment": np.array([0.1]),
        "optimizer_counter": np.array(1.0),
        "pre_state": np.array([1.0]),
        "post_state": np.array([0.9]),
        "pre_prediction": np.array([[0.6, 0.4]]),
        "post_prediction": np.array([[0.5, 0.5]]),
        "exposure_weights": np.array([1 / 859]),
        "equal_input_weights": np.array([1 / 851]),
        "counterfactual_realized": np.array([-0.05]),
        "contributions": np.array([-0.1]),
        "remainders": np.array([0.02]),
        "totals": np.array([-1.6]),
        "categories": np.array(["both_descent"], dtype="U32"),
        "counts": np.array([16], dtype=np.int64),
        "record_coverage": np.array([16], dtype=np.int64),
    }
    (out / "evidence.json").write_text(json.dumps(report, sort_keys=True) + "\n")
    np.savez_compressed(out / "accounting.npz", **arrays)
    (out / "results.md").write_text("baseline\n")
    source_hashes = {rel: _sha(root / rel) for rel in verifier.SOURCE_CLOSURE}
    input_hashes = {key: _sha(root / rel) for key, rel in verifier.INPUTS.items()}
    dependencies = {
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": __import__("platform").platform(),
    }
    reg = {
        "schema": "seed476-direction-transport-supplement-registration-v1",
        "status": "post_execution_supplement_frozen_before_reconstruction",
        "source_closure_sha256": source_hashes,
        "inputs_sha256": input_hashes,
        "dependencies": dependencies,
        "original_seed476_registration_sha256": input_hashes["seed476_registration"],
        "original_seed476_receipt_sha256": input_hashes["seed476_receipt"],
        "original_seed476_evidence_sha256": input_hashes["seed476_evidence"],
        "original_seed476_vectors_sha256": input_hashes["seed476_vectors"],
    }
    (out / "registration.json").write_text(json.dumps(reg, sort_keys=True) + "\n")
    names = ("registration.json", "evidence.json", "accounting.npz", "results.md")
    receipt = {name: _sha(out / name) for name in names}
    receipt["status"] = "seed476_supplement_bound"
    (out / "receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
    return report, arrays


def _rebound_receipt(root: Path):
    out = root / verifier.OUT
    receipt = json.loads((out / "receipt.json").read_text())
    for name in receipt:
        if name != "status":
            receipt[name] = _sha(out / name)
    (out / "receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")


@pytest.fixture
def bound_baseline(tmp_path, monkeypatch):
    report, arrays = _fixture(tmp_path)
    monkeypatch.setattr(
        verifier, "_independent_reconstruct", lambda _root: (report, arrays)
    )
    # An unchanged receipt-bound publication is a required passing fixture.
    verifier.verify(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    ("kind", "path"),
    [
        ("json", ("population", "training_occurrences")),
        ("json", ("steps", 0, "optimizer", "gradient")),
        ("json", ("steps", 0, "optimizer", "moment")),
        ("json", ("steps", 0, "optimizer", "counter")),
        ("json", ("steps", 0, "optimizer", "pre", 0)),
        ("json", ("steps", 0, "optimizer", "post", 0)),
        ("json", ("steps", 0, "prediction", "pre", 0)),
        ("json", ("steps", 0, "weights", "exposure")),
        ("json", ("steps", 0, "weights", "equal_input")),
        ("json", ("steps", 0, "counterfactual_scale")),
        ("json", ("steps", 0, "contributions", "first_order")),
        ("json", ("steps", 0, "remainder")),
        ("json", ("totals", "contributions")),
        ("json", ("steps", 0, "category")),
        ("json", ("totals", "counts", "both_descent")),
        ("json", ("record_coverage", "steps")),
        ("array", "population_rows"),
        ("array", "full_gradient"),
        ("array", "moment"),
        ("array", "optimizer_counter"),
        ("array", "pre_state"),
        ("array", "post_state"),
        ("array", "pre_prediction"),
        ("array", "exposure_weights"),
        ("array", "equal_input_weights"),
        ("array", "counterfactual_realized"),
        ("array", "contributions"),
        ("array", "remainders"),
        ("array", "totals"),
        ("array", "categories"),
        ("array", "counts"),
        ("array", "record_coverage"),
    ],
)
def test_receipt_rebound_verifier_entrypoint_rejects_semantic_corruption(
    bound_baseline, kind, path
):
    root = bound_baseline
    out = root / verifier.OUT
    evidence_path = out / "evidence.json"
    arrays_path = out / "accounting.npz"
    if kind == "json":
        doc = json.loads(evidence_path.read_text())
        cursor = doc
        for key in path[:-1]:
            cursor = cursor[key]
        key = path[-1]
        cursor[key] = "corrupt" if isinstance(cursor[key], str) else 991
        evidence_path.write_text(json.dumps(doc, sort_keys=True) + "\n")
    else:
        with np.load(arrays_path, allow_pickle=False) as source:
            changed = {key: source[key].copy() for key in source.files}
        changed[path].flat[0] = "altered" if changed[path].dtype.kind in "US" else 991
        np.savez_compressed(arrays_path, **changed)
    _rebound_receipt(root)
    protected = [root / verifier.BASE, root / verifier.SEED474, out]
    before = {
        (str(d), p.name): (p.read_bytes(), p.stat().st_mtime_ns)
        for d in protected
        for p in d.iterdir()
        if p.is_file()
    }
    with pytest.raises(ValueError):
        verifier.verify(root)
    after = {
        (str(d), p.name): (p.read_bytes(), p.stat().st_mtime_ns)
        for d in protected
        for p in d.iterdir()
        if p.is_file()
    }
    assert before == after
