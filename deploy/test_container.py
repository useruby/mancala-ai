"""Exercise the real service image, including process-tree shutdown."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(base: str, path: str, body: dict | bytes | None = None):
    data = None
    headers = {}
    if isinstance(body, dict):
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    elif isinstance(body, bytes):
        data = body
        headers["Content-Type"] = "application/json"
    try:
        with urlopen(
            Request(base + path, data=data, headers=headers), timeout=5
        ) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


def main() -> None:
    image = os.environ["MANCALA_TEST_IMAGE"]
    release_root = Path(sys.argv[1]).resolve()
    process = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "-p",
            "127.0.0.1::8080",
            "-v",
            f"{release_root}:/assets:ro",
            "-e",
            "MANCALA_AI_CONFIG=/assets/service.json",
            image,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    container = process.stdout.strip()
    print(f"Started service container {container}", flush=True)
    stopped = False
    try:
        inspect = json.loads(
            subprocess.run(
                ["docker", "inspect", container],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            ).stdout
        )[0]
        assert inspect["Config"]["User"] == "10001:10001"
        assert inspect["Mounts"][0]["RW"] is False
        environment = dict(item.split("=", 1) for item in inspect["Config"]["Env"])
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            assert environment[variable] == "1"
        python_version = subprocess.run(
            ["docker", "exec", container, "python", "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        version_text = python_version.stdout + python_version.stderr
        assert version_text.startswith("Python 3.12."), version_text
        print(f"Container runtime: {version_text.strip()}", flush=True)
        published = (
            subprocess.run(
                ["docker", "port", container, "8080/tcp"],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        base = f"http://{published}"
        deadline = time.monotonic() + 150
        while time.monotonic() < deadline:
            try:
                if request(base, "/health/ready")[0] == 200:
                    break
                time.sleep(1)
            except (URLError, TimeoutError, OSError):
                state = json.loads(
                    subprocess.run(
                        [
                            "docker",
                            "inspect",
                            container,
                            "--format",
                            "{{json .State}}",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=15,
                    ).stdout
                )
                if not state["Running"]:
                    logs = subprocess.run(
                        ["docker", "logs", container],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    raise RuntimeError(
                        f"service container exited during startup: {logs.stdout}{logs.stderr}"
                    )
                time.sleep(1)
        else:
            state = subprocess.run(
                [
                    "docker",
                    "inspect",
                    container,
                    "--format",
                    "{{json .State}}",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            ).stdout
            logs = subprocess.run(
                ["docker", "logs", container],
                capture_output=True,
                text=True,
                timeout=15,
            )
            raise RuntimeError(
                f"container did not become ready; state={state}\n{logs.stdout}{logs.stderr}"
            )

        print("Container readiness check passed", flush=True)
        assert request(base, "/health/live")[0] == 200
        assert request(base, "/health/ready")[0] == 200
        print("Container liveness check passed", flush=True)
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
        status, moved = request(base, "/v1/move", ordinary)
        assert status == 200 and moved["result"]["kind"] == "move"
        assert "request_id" not in moved
        assert moved["result"]["move"] in [0, 1, 2, 3, 5]
        assert moved["completed_simulations"] > 0
        print("Ordinary search HTTP check passed", flush=True)

        exact = {
            "request_id": "container-native-exact-root",
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
        status, exact_response = request(base, "/v1/move", exact)
        assert status == 200 and exact_response["result"]["kind"] == "move"
        assert exact_response["result"]["move"] in [0, 5]
        assert exact_response["completed_simulations"] == 0
        print("Native exact-root HTTP check passed", flush=True)

        terminal = {
            "request_id": "container-terminal",
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
        status, terminal_response = request(base, "/v1/move", terminal)
        assert status == 200 and terminal_response["result"] == {
            "kind": "terminal",
            "move": None,
        }
        assert terminal_response["completed_simulations"] == 0
        assert terminal_response["model_bundle_id"] == "model:none@v1"

        assert request(base, "/v1/move", b"{malformed")[0] == 400
        assert (
            request(base, "/v1/move", {**ordinary, "profile": "unknown@v1"})[0] == 422
        )
        print("Terminal and malformed-request checks passed", flush=True)
        stop = subprocess.run(
            ["docker", "stop", "--timeout", "45", container],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        stopped = True
        state = json.loads(
            subprocess.run(
                ["docker", "inspect", container, "--format", "{{json .State}}"],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            ).stdout
        )
        assert state["ExitCode"] in (0, 143), (
            f"container exited {state['ExitCode']}: {stop.stdout}"
        )
        logs = subprocess.run(
            ["docker", "logs", container],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert "Application shutdown complete" in logs.stdout + logs.stderr
    finally:
        if not stopped:
            subprocess.run(
                ["docker", "stop", "--timeout", "45", container],
                check=False,
                timeout=60,
            )
        subprocess.run(["docker", "rm", "--force", container], check=False, timeout=15)


if __name__ == "__main__":
    main()
