"""Shared constants for the append-only seed479 value-MSE correction."""

from pathlib import Path

OUT = Path("docs/data/seed480-value-mse-correction")
BASE = Path("docs/data/seed479-population-ablation")
SOURCE_PATHS = (
    "ml/alphazero_lite/seed480_value_mse_correction.py",
    "ml/alphazero_lite/publish_seed480_value_mse_correction.py",
    "ml/alphazero_lite/verify_seed480_value_mse_correction.py",
    "ml/alphazero_lite/test_seed480_value_mse_correction.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/seed466_runtime_bundle.py",
    "ml/alphazero_lite/seed466_replay_bundle.py",
    "ml/alphazero_lite/checkpoint_phase_selection.py",
    "ml/alphazero_lite/seed479_population_ablation.py",
    "ml/alphazero_lite/verify_seed479_population_ablation.py",
)
