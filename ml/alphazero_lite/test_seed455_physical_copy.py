"""Run the published read-only verifier against an isolated physical copy."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from ml.alphazero_lite.verify_seed455_fresh_projection_attribution import sha


def _inventory(directory: Path) -> dict[str, tuple[str, int]]:
    return {
        path.relative_to(directory).as_posix(): (
            sha(path),
            path.stat().st_mtime_ns,
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_verifier_cli_from_physical_copy_is_read_only_and_portable(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    publication_rel = Path("docs/data/seed455-fresh-projection-attribution")
    with (root / publication_rel / "freeze-v7.json").open() as stream:
        frozen = json.load(stream)
    copy_root = tmp_path / "physical-checkout"
    copy_root.mkdir()
    (copy_root / ".tmp").mkdir()
    shutil.copytree(root / "ml", copy_root / "ml")
    required = set(frozen["inputs"]) | set(frozen["sources"])
    for path in (root / publication_rel).rglob("*"):
        if path.is_file():
            required.add(path.relative_to(root).as_posix())
    for relative in sorted(required):
        source = root / relative
        destination = copy_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    registration = json.loads(
        (
            copy_root / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_text()
    )
    blocked_paths = [
        str(Path(row["path"]).resolve()) for row in registration["replays"]
    ]
    sitecustomize = copy_root / "sitecustomize.py"
    sitecustomize.write_text(
        "import sys\n"
        f"blocked = {blocked_paths!r}\n"
        f"original_data = {str(root / 'docs/data')!r}\n"
        "def audit(event, args):\n"
        "    if event == 'open' and args and isinstance(args[0], (str, bytes)) and (str(args[0]) in blocked or str(args[0]).startswith(original_data)):\n"
        "        raise PermissionError('historical absolute derivative access forbidden')\n"
        "sys.addaudithook(audit)\n"
    )
    isolated_cwd = tmp_path / "unrelated-working-directory"
    isolated_cwd.mkdir()
    publication = copy_root / publication_rel
    before = _inventory(publication)
    python = root / ".venv-azlite/bin/python"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(copy_root)
    completed = subprocess.run(
        [
            str(python),
            "-m",
            "ml.alphazero_lite.verify_seed455_fresh_projection_attribution",
            "--root",
            str(copy_root),
        ],
        cwd=isolated_cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert '"status": "valid"' in completed.stdout
    assert _inventory(publication) == before
