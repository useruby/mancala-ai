from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.alphazero_lite import (
    verify_seed542_fixed_update_cd_v2 as base_verifier,
    verify_seed542_fixed_update_cd_v2_supplement as supplement,
)

ROOT = Path(__file__).resolve().parents[2]
PUBLICATION = ROOT / "docs/data/seed542-fixed-update-cd-v2"


@pytest.fixture(scope="module")
def replayed_c_boundaries() -> dict[int, dict[str, np.ndarray]]:
    torch.set_num_threads(1)
    return supplement.reconstruct_c_boundary_states(ROOT)


def _scratch() -> tempfile.TemporaryDirectory[str]:
    parent = ROOT / ".tmp"
    parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="seed542-v2-supplement-", dir=parent)


def _copy_boundary_artifacts(output: Path) -> None:
    production = output / "production-reference"
    production.mkdir(parents=True)
    for epoch in range(1, 5):
        shutil.copy2(
            PUBLICATION / "production-reference" / f"seed504_epoch{epoch}.npz",
            production / f"seed504_epoch{epoch}.npz",
        )
    for update in (27, 54, 81, 108):
        shutil.copy2(
            PUBLICATION / f"C-update-{update:03d}.npz",
            output / f"C-update-{update:03d}.npz",
        )
    shutil.copy2(
        PUBLICATION / "historical-seed504-drift.json",
        output / "historical-seed504-drift.json",
    )


def test_mutated_production_boundary_fails_after_descriptive_drift_is_resealed(
    replayed_c_boundaries: dict[int, dict[str, np.ndarray]],
) -> None:
    with _scratch() as scratch:
        output = Path(scratch)
        _copy_boundary_artifacts(output)
        assert (
            supplement.verify_production_boundary_checkpoints(
                output, replayed_c_boundaries
            )["status"]
            == "production_and_C_boundary_arrays_reconciled"
        )

        production_e1 = output / "production-reference/seed504_epoch1.npz"
        with np.load(production_e1) as archive:
            altered = {name: archive[name] for name in archive.files}
        altered["w_policy"] = altered["w_policy"].copy()
        altered["w_policy"][0, 0] = np.nextafter(
            altered["w_policy"][0, 0], np.float32(np.inf)
        )
        np.savez(production_e1, **altered)

        drift = supplement.historical_drift_report(ROOT, output)
        (output / "historical-seed504-drift.json").write_text(
            json.dumps(drift, indent=2, sort_keys=True) + "\n"
        )
        # The descriptive report is internally consistent with the altered E1;
        # it cannot make that production boundary equal the reconstructed C state.
        base_verifier.verify_historical_drift(ROOT, output)
        with pytest.raises(
            ValueError,
            match="supplement_production_boundary_checkpoint_mismatch:27",
        ):
            supplement.verify_production_boundary_checkpoints(
                output, replayed_c_boundaries
            )


def test_mutated_run_status_effect_is_rejected_against_independent_development() -> (
    None
):
    expected, _predictions = base_verifier.independently_recompute_development(
        ROOT, PUBLICATION
    )
    with _scratch() as scratch:
        output = Path(scratch)
        shutil.copy2(PUBLICATION / "run-status.json", output / "run-status.json")
        supplement.verify_status_effects(output, expected["effects"])
        status_path = output / "run-status.json"
        status = json.loads(status_path.read_text())
        status["development_effects"]["D_minus_C"][">32"]["exposure_weighted"][
            "policy_ce"
        ] += 0.125
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        with pytest.raises(
            ValueError, match="supplement_run_status_development_effects_mismatch"
        ):
            supplement.verify_status_effects(output, expected["effects"])
