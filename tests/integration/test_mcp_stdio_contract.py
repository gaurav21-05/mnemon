from __future__ import annotations

import importlib.util
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest


def _has_mcp_dependency() -> bool:
    return importlib.util.find_spec("mcp") is not None


class JsonRpcStdioClient:
    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process
        self._next_id = 1

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        self._send(payload)
        while True:
            response = self._read()
            if response.get("id") == req_id:
                return response

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        self._send(payload)

    def _send(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        assert self._process.stdin is not None
        self._process.stdin.write(body + b"\n")
        self._process.stdin.flush()

    def _read(self, timeout_s: float = 10.0) -> dict[str, Any]:
        assert self._process.stdout is not None

        deadline = time.monotonic() + timeout_s
        line = bytearray()
        while True:
            self._wait_for_stdout(deadline)
            chunk = os.read(self._process.stdout.fileno(), 1)
            if not chunk:
                stderr = self._read_stderr()
                raise RuntimeError(f"MCP server closed stdout early. stderr={stderr}")
            line.extend(chunk)
            if chunk == b"\n":
                break

        text = line.decode("utf-8").strip()
        if not text:
            return self._read(timeout_s=timeout_s)
        return json.loads(text)

    def _wait_for_stdout(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stderr = self._read_stderr()
            raise TimeoutError(f"Timed out waiting for MCP response. stderr={stderr}")
        assert self._process.stdout is not None
        ready, _, _ = select.select([self._process.stdout], [], [], remaining)
        if not ready:
            stderr = self._read_stderr()
            raise TimeoutError(f"Timed out waiting for MCP response. stderr={stderr}")

    def _read_stderr(self) -> str:
        assert self._process.stderr is not None
        try:
            data = os.read(self._process.stderr.fileno(), 4096)
        except OSError:
            return ""
        return data.decode("utf-8", errors="replace")


def _spawn_server(namespace: str) -> tuple[subprocess.Popen[bytes], JsonRpcStdioClient]:
    root = Path(__file__).resolve().parents[2]
    server_script = root / "examples" / "mcp_memory_server.py"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MNEMON_MCP_NAMESPACE"] = namespace

    process = subprocess.Popen(
        [sys.executable, str(server_script)],
        cwd=str(root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    return process, JsonRpcStdioClient(process)


def _initialize(client: JsonRpcStdioClient) -> None:
    init = client.request(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mnemon-test", "version": "0.1.0"},
        },
    )
    assert init.get("error") is None
    assert "result" in init
    client.notify("notifications/initialized", {})


def _is_error_response(response: dict[str, Any]) -> bool:
    if response.get("error") is not None:
        return True
    payload = json.dumps(response.get("result", {})).lower()
    return "error" in payload or "invalid" in payload or "unknown" in payload


@pytest.mark.skipif(not _has_mcp_dependency(), reason="mcp package not installed")
def test_mcp_stdio_jsonrpc_contract() -> None:
    process, client = _spawn_server("mnemon_test")

    try:
        _initialize(client)

        tools = client.request("tools/list", {})
        assert tools.get("error") is None
        tool_names = {
            t["name"] for t in tools.get("result", {}).get("tools", []) if "name" in t
        }
        assert "mnemon_test.memory_state" in tool_names
        assert "mnemon_test.memory_resources_list" in tool_names

        state_call = client.request(
            "tools/call",
            {
                "name": "mnemon_test.memory_state",
                "arguments": {},
            },
        )
        assert state_call.get("error") is None
        # Keep this resilient to MCP server shape variants.
        assert "result" in state_call
        assert "episodic_memories" in json.dumps(state_call["result"])

        resources_list = client.request("resources/list", {})
        if resources_list.get("error") is None:
            uris = {
                item["uri"]
                for item in resources_list.get("result", {}).get("resources", [])
                if "uri" in item
            }
            assert "memory://mnemon_test/state" in uris

            resources_read = client.request(
                "resources/read",
                {"uri": "memory://mnemon_test/state"},
            )
            assert resources_read.get("error") is None
            assert "episodic_memories" in json.dumps(resources_read.get("result", {}))
        else:
            # Fallback route for MCP runtimes without resources API.
            fallback = client.request(
                "tools/call",
                {
                    "name": "mnemon_test.memory_resources_read",
                    "arguments": {"uri": "memory://mnemon_test/state"},
                },
            )
            assert fallback.get("error") is None
            assert "episodic_memories" in json.dumps(fallback.get("result", {}))
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.skipif(not _has_mcp_dependency(), reason="mcp package not installed")
def test_mcp_stdio_unknown_tool_returns_error() -> None:
    process, client = _spawn_server("mnemon_test")
    try:
        _initialize(client)
        response = client.request(
            "tools/call",
            {
                "name": "mnemon_test.tool_that_does_not_exist",
                "arguments": {},
            },
        )
        assert _is_error_response(response)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.skipif(not _has_mcp_dependency(), reason="mcp package not installed")
def test_mcp_stdio_invalid_args_returns_error() -> None:
    process, client = _spawn_server("mnemon_test")
    try:
        _initialize(client)
        response = client.request(
            "tools/call",
            {
                "name": "mnemon_test.memory_retrieve",
                "arguments": {
                    "query": "test",
                    "top_k": "bad-type",
                },
            },
        )
        assert _is_error_response(response)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
