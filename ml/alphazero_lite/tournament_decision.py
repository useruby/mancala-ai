from __future__ import annotations

from statistics import median


def pick_best_topk(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("at least one candidate is required")
    return max(
        candidates,
        key=lambda row: (
            float(row["mcts_screen_score"]),
            float(row.get("arena_score", -1.0)),
            -int(row["topk_index"]),
        ),
    )


def summarize_tournament(
    seed_winners: list[dict], *, min_mcts_score: float, min_arena_score: float
) -> dict:
    if not seed_winners:
        raise ValueError("seed winners required")

    mcts_scores = [float(row["mcts_confirm_score"]) for row in seed_winners]
    arena_scores = [float(row["arena_score"]) for row in seed_winners]

    median_mcts = float(median(mcts_scores))
    best_arena = max(arena_scores)

    return {
        "median_mcts_score": round(median_mcts, 4),
        "best_arena_score": round(best_arena, 4),
        "pass_mcts_median": median_mcts >= float(min_mcts_score),
        "pass_arena_best_of_3": best_arena >= float(min_arena_score),
        "passed": (median_mcts >= float(min_mcts_score))
        and (best_arena >= float(min_arena_score)),
    }
