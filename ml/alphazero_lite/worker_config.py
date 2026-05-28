from __future__ import annotations

DEFAULT_WORKERS = 24

WORKER_CAPABLE_EXECUTABLES = {
    "ml/alphazero_lite/self_play.py",
    "ml/alphazero_lite/generate_bootstrap_dataset.py",
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/mcts1200_baseline.py",
}


def normalize_command_workers(
    command: list[str], *, workers: int | None = None
) -> list[str]:
    if len(command) < 2:
        return list(command)

    executable = command[1]
    if executable not in WORKER_CAPABLE_EXECUTABLES:
        return list(command)

    target_workers = str(DEFAULT_WORKERS if workers is None else workers)
    rendered = list(command)
    if "--workers" in rendered:
        worker_index = rendered.index("--workers")
        if worker_index + 1 < len(rendered):
            rendered[worker_index + 1] = target_workers
            return rendered
    rendered.extend(["--workers", target_workers])
    return rendered
