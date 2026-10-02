"""Retired entry point for the protocol-invalid seed397 evaluation."""

from __future__ import annotations

import sys


def main() -> None:
    """Fail closed before importing or launching any arena dependencies."""
    if len(sys.argv) != 2 or sys.argv[1] not in {"register", "run", "analyze"}:
        raise SystemExit("usage: run_seed397_o0_e1_e4.py {register|run|analyze}")
    raise SystemExit(
        "seed397_protocol_invalid: register/run/analyze are retired; "
        "the published suite collides with historical declared states and "
        "its evidence does not satisfy the shared validation contract"
    )


if __name__ == "__main__":
    main()
