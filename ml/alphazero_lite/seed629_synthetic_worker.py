"""Harmless worker used only by Seed 629 supervisor integration tests."""

from __future__ import annotations

import argparse
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
        with args.path.open("wb") as stream:
            while True:
                stream.write(bytes(1024 * 1024))
    elif args.mode == "child":
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(3600)"]
        )
        args.path.write_text(str(child.pid))
        child.wait()


if __name__ == "__main__":
    main()
