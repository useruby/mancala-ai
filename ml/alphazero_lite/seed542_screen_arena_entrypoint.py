"""Set deterministic CPU thread counts before dispatching one arena pair."""

from __future__ import annotations

import torch

from ml.alphazero_lite import arena


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    arena.main()


if __name__ == "__main__":
    main()
