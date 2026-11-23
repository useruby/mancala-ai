"""Harmless worker used only by Seed 629 supervisor integration tests."""

from __future__ import annotations

import argparse
import errno
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("success", "cpu", "memory", "sleep", "write", "child")
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("scratch_path", type=Path, nargs="?")
    args = parser.parse_args()
    if args.mode == "success":
        args.path.write_text("synthetic-ok\n")
    elif args.mode == "cpu":
        while True:
            pass
    elif args.mode == "memory":
        blocks = []
        while True:
            blocks.append(bytearray(1024 * 1024))
    elif args.mode == "sleep":
        time.sleep(3600)
    elif args.mode == "write":
        paths = [args.path]
        if args.scratch_path is not None:
            paths.append(args.scratch_path)
        try:
            streams = [path.open("wb") for path in paths]
            while True:
                for stream in streams:
                    stream.write(bytes(1024 * 1024))
                    stream.flush()
        except OSError as error:
            reason = (
                "project_quota_exceeded"
                if error.errno == errno.EDQUOT
                else "filesystem_space_exhausted"
                if error.errno == errno.ENOSPC
                else "write_failed"
            )
            print(
                json.dumps(
                    {
                        "schema": "seed630-worker-failure-v1",
                        "reason": reason,
                        "errno": error.errno,
                        "errno_name": errno.errorcode.get(error.errno or 0),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(
                74
                if error.errno == errno.EDQUOT
                else 75
                if error.errno == errno.ENOSPC
                else 76
            )
    elif args.mode == "child":
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(3600)"]
        )
        args.path.write_text(str(child.pid))
        child.wait()


if __name__ == "__main__":
    main()
