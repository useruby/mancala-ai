"""Run the actual service entry point against the pinned release assets."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def send(base: str, path: str, body: dict | bytes | None = None):
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    headers = {"Content-Type": "application/json"} if data is not None else {}
    try:
        with urlopen(
            Request(base + path, data=data, headers=headers), timeout=5
        ) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


def main() -> None:
    release = Path(sys.argv[1]).resolve()
    config = json.loads((release / "service.json").read_text(encoding="utf-8"))
    config["asset_root"] = str(release)
    temp_root = Path(os.environ.get("TMPDIR", ".tmp")).resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="real-assets-", dir=temp_root) as tmp:
        config_path = Path(tmp) / "service.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = os.environ | {
            "MANCALA_AI_CONFIG": str(config_path),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "ml.alphazero_lite.service:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                "1",
                "--timeout-graceful-shutdown",
                "25",
                "--no-access-log",
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        base = f"http://127.0.0.1:{port}"
        clean_shutdown = False
        try:
            deadline = time.monotonic() + 150
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    output = process.stdout.read() if process.stdout is not None else ""
                    raise RuntimeError(f"Uvicorn exited at startup: {output}")
                try:
                    if send(base, "/health/ready")[0] == 200:
                        break
                    time.sleep(0.5)
                except (URLError, TimeoutError):
                    time.sleep(0.5)
            else:
                raise RuntimeError("real asset service did not become ready")

            assert send(base, "/health/live")[0] == 200
            ordinary = {
                "game_revision": 1,
                "state": {
                    "player0_pits": [4, 3, 2, 1, 0, 5],
                    "player1_pits": [1, 2, 3, 4, 5, 6],
                    "player0_store": 6,
                    "player1_store": 6,
                    "current_player": 0,
                },
                "profile": "balanced@v1",
                "seed": 17,
            }
            status, response = send(base, "/v1/move", ordinary)
            assert status == 200 and response["result"]["kind"] == "move"
            assert "request_id" not in response
            assert response["result"]["move"] in [0, 1, 2, 3, 5]
            assert (
                response["completed_simulations"]
                == config["profiles"]["balanced@v1"]["settings"]["simulations"]
            )

            exact = {
                "request_id": "real-asset-exact-root",
                "game_revision": 2,
                "state": {
                    "player0_pits": [1, 0, 0, 0, 0, 1],
                    "player1_pits": [1, 1, 1, 1, 1, 1],
                    "player0_store": 20,
                    "player1_store": 20,
                    "current_player": 0,
                },
                "profile": "balanced@v1",
                "seed": 18,
            }
            status, response = send(base, "/v1/move", exact)
            assert status == 200 and response["result"]["kind"] == "move"
            assert response["result"]["move"] in [0, 5]
            assert response["completed_simulations"] == 0

            terminal = {
                "request_id": "real-asset-terminal",
                "game_revision": 3,
                "state": {
                    "player0_pits": [0] * 6,
                    "player1_pits": [0] * 6,
                    "player0_store": 20,
                    "player1_store": 28,
                    "current_player": 0,
                },
                "profile": "balanced@v1",
                "seed": 19,
            }
            status, response = send(base, "/v1/move", terminal)
            assert status == 200 and response["result"] == {
                "kind": "terminal",
                "move": None,
            }
            assert response["completed_simulations"] == 0
            assert response["model_bundle_id"] == "model:none@v1"
            assert send(base, "/v1/move", b"{malformed")[0] == 400
            assert (
                send(base, "/v1/move", {**ordinary, "profile": "unknown@v1"})[0] == 422
            )
            clean_shutdown = True
        finally:
            process.terminate()
            try:
                output, _ = process.communicate(timeout=40)
            except subprocess.TimeoutExpired:
                process.kill()
                output, _ = process.communicate()
                if not clean_shutdown:
                    raise RuntimeError(
                        f"service failed and could not stop cleanly: {output}"
                    )
                raise
            if clean_shutdown:
                assert process.returncode in (0, -15), (
                    f"exit={process.returncode}\n{output}"
                )
                assert "Application shutdown complete" in output
            elif process.returncode != 0:
                print(output, file=sys.stderr)


if __name__ == "__main__":
    main()
