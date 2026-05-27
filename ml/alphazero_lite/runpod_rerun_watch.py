def should_stop_watcher(
    *, result_marker_exists: bool, elapsed_seconds: int, max_seconds: int
) -> bool:
    if result_marker_exists:
        return False
    return elapsed_seconds >= max_seconds
