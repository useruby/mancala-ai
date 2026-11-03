"""Configure numerical threads before dispatching one seed551 arena pair."""

import torch

from ml.alphazero_lite import arena


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    arena.main()


if __name__ == "__main__":
    main()
