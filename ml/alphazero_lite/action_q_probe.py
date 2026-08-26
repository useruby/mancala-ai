"""Standalone diagnostic action-Q head for a frozen PolicyValueNet trunk."""

from __future__ import annotations

import torch
from torch import nn


class ActionQProbe(nn.Module):
    """Predict raw Q values for the six root actions from a trunk feature."""

    def __init__(self, trunk_size: int):
        super().__init__()
        self.q_hidden = nn.Linear(trunk_size, trunk_size)
        self.q_head = nn.Linear(trunk_size, 6)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.q_head(torch.relu(self.q_hidden(features)))
