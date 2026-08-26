"""Search-conditioned root action-Q correction probe."""

from __future__ import annotations

import torch
from torch import nn


CONTEXT_SIZE = 11  # Action one-hot plus A16 root/action selection statistics.


class ContextActionQProbe(nn.Module):
    """Predict a visited A16 root edge's P1-Q correction from pre-search evidence."""

    def __init__(self, trunk_size: int):
        super().__init__()
        self.hidden = nn.Linear(trunk_size + CONTEXT_SIZE, trunk_size)
        self.head = nn.Linear(trunk_size, 1)

    def forward(
        self, trunk_features: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat((trunk_features, context), dim=1)
        return self.head(torch.relu(self.hidden(x))).reshape(-1)
