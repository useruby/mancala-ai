from __future__ import annotations

DEFAULT_WORKERS = 24
HIGH_MEMORY_LOCAL_PROFILE = "high_memory_local"
HIGH_MEMORY_LOCAL_EVALUATOR_CACHE_SIZE = 200000
SUPPORTED_MEMORY_SPEED_PROFILES = {HIGH_MEMORY_LOCAL_PROFILE}

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
    for index, token in enumerate(rendered):
        if token.startswith("--workers="):
            rendered[index] = f"--workers={target_workers}"
            return rendered
    rendered.extend(["--workers", target_workers])
    return rendered


def normalize_memory_speed_profile(
    command: list[str], *, memory_speed_profile: str | None = None
) -> list[str]:
    def option_value(tokens: list[str], flag: str) -> str | None:
        if flag in tokens:
            value_index = tokens.index(flag) + 1
            if value_index >= len(tokens):
                return None
            value = tokens[value_index]
            if value.startswith("--"):
                return None
            return value

        prefix = f"{flag}="
        for token in tokens:
            if token.startswith(prefix):
                return token[len(prefix) :]
        return None

    def has_flag(tokens: list[str], flag: str) -> bool:
        return flag in tokens or any(token.startswith(f"{flag}=") for token in tokens)

    def set_option_value(tokens: list[str], flag: str, value: str) -> list[str]:
        rendered_tokens = list(tokens)
        if flag in rendered_tokens:
            value_index = rendered_tokens.index(flag) + 1
            if value_index >= len(rendered_tokens):
                rendered_tokens.append(value)
                return rendered_tokens
            if rendered_tokens[value_index].startswith("--"):
                rendered_tokens.insert(value_index, value)
                return rendered_tokens
            rendered_tokens[value_index] = value
            return rendered_tokens

        prefix = f"{flag}="
        for index, token in enumerate(rendered_tokens):
            if token.startswith(prefix):
                rendered_tokens[index] = f"{flag}={value}"
                return rendered_tokens

        rendered_tokens.extend([flag, value])
        return rendered_tokens

    rendered = normalize_command_workers(command)
    if memory_speed_profile is None:
        return rendered
    if memory_speed_profile not in SUPPORTED_MEMORY_SPEED_PROFILES:
        raise ValueError(f"unsupported memory_speed_profile: {memory_speed_profile}")
    if len(rendered) < 2:
        return rendered

    executable = rendered[1]
    if executable == "ml/alphazero_lite/self_play.py":
        if not has_flag(rendered, "--checkpoint"):
            return rendered
        return set_option_value(
            rendered,
            "--evaluator-cache-size",
            str(HIGH_MEMORY_LOCAL_EVALUATOR_CACHE_SIZE),
        )

    if executable == "ml/alphazero_lite/generate_bootstrap_dataset.py":
        teacher_mode = option_value(rendered, "--teacher-mode") or "puct"
        position_mode = option_value(rendered, "--position-selection-mode")
        if (
            position_mode == "hybrid_teacher"
            and teacher_mode == "puct"
            and not has_flag(rendered, "--teacher-search-reuse")
        ):
            rendered.append("--teacher-search-reuse")

    return rendered
