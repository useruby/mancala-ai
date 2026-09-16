from ml.alphazero_lite.run_r61_expanded_exact_subcluster import descendants


def test_descendants_are_unique_and_bfs_ordered() -> None:
    anchor = {
        "current_player": 0,
        "player_pits": [1, 1, 1, 1, 1, 1],
        "opponent_pits": [1, 1, 1, 1, 1, 1],
        "player_store": 0,
        "opponent_store": 0,
    }
    rows = descendants(anchor, 2)
    assert rows[0] == (0, anchor)
    assert [depth for depth, _ in rows] == sorted(depth for depth, _ in rows)
    assert len({str(state) for _, state in rows}) == len(rows)
