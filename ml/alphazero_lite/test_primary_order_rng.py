import numpy as np
import torch

from ml.alphazero_lite.train import PolicyValueNet, set_seed, train


def test_primary_order_seed_isolated_and_persistent() -> None:
    x = np.zeros((24, 21), dtype=np.float32)
    x[:, 0] = 1 / 48
    x[:, 6] = 1 / 48
    policy = np.zeros((24, 6), dtype=np.float32)
    policy[:, 0] = 1.0
    value = np.zeros((24, 1), dtype=np.float32)

    def permutations(order_seed: int) -> list[list[int]]:
        set_seed(461)
        model = PolicyValueNet((8, 3), "residual_v3", 21)
        observed: list[list[int]] = []
        train(
            model,
            x,
            policy,
            value,
            np.arange(24),
            epochs=2,
            batch_size=8,
            lr=0.001,
            device=torch.device("cpu"),
            value_loss_weight=0.3,
            value_loss="huber",
            huber_delta=1.0,
            val_split=0.1,
            grad_clip=1.0,
            save_top_k=0,
            lr_scheduler="none",
            primary_order_seed=order_seed,
            permutation_callback=lambda _epoch, values: observed.append(values),
        )
        return observed

    first = permutations(38201)
    assert first == permutations(38201)
    assert first != permutations(38202)
    assert first[0] != first[1]
    assert sorted(first[0]) == sorted(first[1])
