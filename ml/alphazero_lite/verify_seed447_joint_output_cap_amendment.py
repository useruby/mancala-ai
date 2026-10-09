"""Explicit post-execution amendment for seed447 verifier replay dispatch.

The prospectively frozen verifier passed NumPy arrays to seed442's Torch trial
setter. This wrapper preserves that verifier byte-for-byte and adapts only the
trial setter dispatch. The amendment is separately receipted and does not alter
or regenerate experiment evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import seed442_kl_capped_adam as seed442
from ml.alphazero_lite import verify_seed447_joint_output_cap as frozen_verifier


def _numpy_aware_trial_setter(
    pre: list[np.ndarray] | list[torch.Tensor],
    proposal: list[np.ndarray] | list[torch.Tensor],
    scale: float,
) -> None:
    """Apply the registered float32 trial arithmetic to frozen verifier arrays."""
    with torch.no_grad():
        for parameter, before, proposed in zip(
            seed442._CURRENT_PARAMETERS, pre, proposal, strict=True
        ):
            before_tensor = torch.as_tensor(before, dtype=torch.float32)
            proposal_tensor = torch.as_tensor(proposed, dtype=torch.float32)
            if scale == 1.0:
                parameter.copy_(proposal_tensor)
            else:
                parameter.copy_(
                    before_tensor + (proposal_tensor - before_tensor) * scale
                )


def verify(root: Path) -> dict[str, Any]:
    """Run the frozen semantic verifier with the documented setter adaptation."""
    original = seed442._set_trial
    seed442._set_trial = _numpy_aware_trial_setter
    try:
        result = frozen_verifier.verify(root)
    finally:
        seed442._set_trial = original
    return {**result, "verifier_amendment": "numpy_trial_setter_dispatch_v1"}


def publish_amendment_receipt(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Bind the supplemental verifier and result without rewriting frozen files."""
    root = root.resolve()
    out = root / "docs/data/seed447-joint-output-cap"
    relative = "ml/alphazero_lite/verify_seed447_joint_output_cap_amendment.py"
    amendment = {
        "schema": "seed447-verifier-amendment-receipt-v1",
        "reason": "frozen verifier passed NumPy arrays to Torch-only seed442 trial setter",
        "frozen_verifier_sha256": hashlib.sha256(
            (root / "ml/alphazero_lite/verify_seed447_joint_output_cap.py").read_bytes()
        ).hexdigest(),
        "amendment_source": relative,
        "amendment_source_sha256": hashlib.sha256(
            (root / relative).read_bytes()
        ).hexdigest(),
        "frozen_receipt_sha256": hashlib.sha256(
            (out / "receipt.json").read_bytes()
        ).hexdigest(),
        "result": result,
    }
    (out / "verifier-amendment-receipt.json").write_text(
        json.dumps(amendment, indent=2, sort_keys=True) + "\n"
    )
    return amendment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(
        json.dumps(
            publish_amendment_receipt(args.root, verify(args.root)),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
