from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def worktree_workspace_root(root: Path) -> Path | None:
    if root.parent.name != ".worktrees":
        return None
    return root.parent.parent


def python_executable(root: Path | None = None) -> str:
    root = root or repo_root()
    if os.environ.get("AZLITE_EXPERIMENT_PYTHON"):
        return os.environ["AZLITE_EXPERIMENT_PYTHON"]

    candidates = [root / ".venv/bin/python"]
    workspace_root = worktree_workspace_root(root)
    if workspace_root is not None:
        candidates.append(workspace_root / ".venv/bin/python")
    for ancestor in root.parents:
        candidates.append(ancestor / ".venv/bin/python")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    return sys.executable


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def set_flag(command: list[str], flag: str, value: str) -> None:
    if flag in command:
        command[command.index(flag) + 1] = value
    else:
        command.extend([flag, value])


def rewrite_python_commands(config: dict, *, python_bin: str) -> dict:
    updated = copy.deepcopy(config)
    for step in updated.get("steps", []):
        command = step.get("command")
        if isinstance(command, list) and command and command[0] == ".venv/bin/python":
            step["command"] = [python_bin, *command[1:]]
    return updated


def apply_runtime_paths(
    config: dict,
    *,
    versions_dir: Path,
    current_path: Path,
    parent_artifact_path: Path,
) -> dict:
    updated = copy.deepcopy(config)
    updated["versions_dir"] = str(versions_dir)
    updated["current_path"] = str(current_path)
    updated["parent_artifact_path"] = str(parent_artifact_path)
    return updated


def apply_budget_overrides(
    config: dict,
    *,
    challenger_budget: int,
    current_simulations: int,
) -> dict:
    updated = copy.deepcopy(config)
    for step in updated.get("steps", []):
        command = step.get("command")
        if not isinstance(command, list):
            continue

        step_name = step.get("name")
        if step_name == "arena_confirm_report":
            set_flag(command, "--challenger-simulations", str(challenger_budget))
            set_flag(command, "--current-simulations", str(current_simulations))
        elif step_name == "mcts1200_baseline_report":
            set_flag(command, "--az-base-simulations", str(challenger_budget))

    return updated


def final_iteration(config: dict) -> int:
    start_iteration = int(config.get("start_iteration", 1))
    iterations = int(config.get("iterations", 1))
    return start_iteration + iterations - 1
