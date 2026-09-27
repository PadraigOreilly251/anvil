"""Minimal MCP client for Anvil.

Two transports:
- streamable HTTP (JSON-RPC 2.0 over POST, e.g. websearch on :8560/mcp)
- stdio subprocess (newline-delimited JSON-RPC 2.0, e.g. cua-mcp.ts)
"""
import json
import os
import subprocess
import threading
import time


class McpError(Exception):
    pass


PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "anvil", "version": "0.1.0"}


class McpHTTPClient:
    """MCP streamable-HTTP transport client."""

    def __init__(self, url, timeout=120):
        import httpx

        self.url = url
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        self.session_id = None
        self._id = 0
        self._lock = threading.Lock()

    def _next_id(self):
        with self._lock:
            self._id += 1
            return self._id

    def _rpc(self, method, params=None, notify=False):
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        rid = None
        if not notify:
            rid = self._next_id()
            payload["id"] = rid
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        r = self.client.post(self.url, json=payload, headers=headers)
        sid = r.headers.get("mcp-session-id")
        if sid:
            self.session_id = sid
        if notify:
            return None
        if r.status_code >= 400:
            raise McpError(f"HTTP {r.status_code}: {r.text[:300]}")
        ct = r.headers.get("content-type", "")
        if "text/event-stream" in ct:
            return self._parse_sse(r.text, rid)
        data = r.json()
        if isinstance(data, list):
            for d in data:
                if d.get("id") == rid:
                    data = d
        if "error" in data:
            raise McpError(str(data["error"]))
        return data.get("result")

    @staticmethod
    def _parse_sse(text, want_id):
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                body = line[5:].strip()
                if not body or body == "[DONE]":
                    continue
                try:
                    d = json.loads(body)
                except Exception:
                    continue
                if isinstance(d, dict) and d.get("id") == want_id:
                    if "error" in d:
                        raise McpError(str(d["error"]))
                    return d.get("result")
        return None

    def connect(self):
        res = self._rpc(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            },
        )
        self._rpc("notifications/initialized", notify=True)
        return res

    def list_tools(self):
        res = self._rpc("tools/list", {})
        return res.get("tools", []) if isinstance(res, dict) else []

    def call_tool(self, name, arguments):
        res = self._rpc("tools/call", {"name": name, "arguments": arguments or {}})
        return self._result_text(res)

    @staticmethod
    def _result_text(res):
        if res is None:
            return ""
        if isinstance(res, str):
            return res
        if isinstance(res, dict):
            parts = []
            for item in res.get("content", []):
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif "text" in item:
                        parts.append(str(item.get("text")))
            if parts:
                txt = "\n".join(parts)
                if res.get("isError"):
                    return f"[tool error] {txt}"
                return txt
            return json.dumps(res)
        return str(res)

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


class McpStdioClient:
    """MCP stdio transport via subprocess (newline JSON-RPC)."""

    def __init__(self, command, cwd=None, env=None, timeout=120):
        import shlex

        argv = command if isinstance(command, list) else shlex.split(command)
        full_env = {**os.environ, **(env or {})}
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            text=True,
            bufsize=1,
            env=full_env,
        )
        self.timeout = timeout
        self._id = 0
        self._lock = threading.Lock()
        self._err_lines = []
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self):
        for line in self.proc.stderr:
            self._err_lines.append(line.rstrip())
            if len(self._err_lines) > 200:
                self._err_lines.pop(0)

    def _send(self, method, params=None, notify=False):
        with self._lock:
            self._id += 1
            rid = self._id
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        if not notify:
            payload["id"] = rid
        if self.proc.poll() is not None:
            raise McpError(f"stdio server exited: {' | '.join(self._err_lines[-5:])[:300]}")
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        if notify:
            return None
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    raise McpError(
                        f"stdio server exited: {' | '.join(self._err_lines[-5:])[:300]}"
                    )
                continue
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if isinstance(d, dict) and d.get("id") == rid:
                if "error" in d:
                    raise McpError(str(d["error"]))
                return d.get("result")
        raise McpError(f"timeout waiting for response to {method}")

    def connect(self):
        res = self._send(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            },
        )
        self._send("notifications/initialized", notify=True)
        return res

    def list_tools(self):
        res = self._send("tools/list", {})
        return res.get("tools", []) if isinstance(res, dict) else []

    def call_tool(self, name, arguments):
        res = self._send("tools/call", {"name": name, "arguments": arguments or {}})
        return McpHTTPClient._result_text(res)

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        except Exception:
            pass


class McpManager:
    """Holds live MCP server connections and exposes a unified tool view."""

    def __init__(self):
        self.servers = {}
        self.status = {}

    def _open(self, spec):
        if spec["transport"] == "http":
            c = McpHTTPClient(spec["url"], timeout=spec.get("timeout", 120))
        elif spec["transport"] == "stdio":
            c = McpStdioClient(
                spec["command"],
                cwd=spec.get("cwd"),
                env=spec.get("env"),
                timeout=spec.get("timeout", 120),
            )
        else:
            raise McpError(f"unknown transport {spec['transport']}")
        c.connect()
        return c

    def load(self, config):
        """Connect to all enabled MCP servers (replaces existing connections)."""
        for name, old in list(self.servers.items()):
            self._close(old)
        self.servers = {}
        self.status = {}
        for spec in config.get("mcp", []):
            name = spec.get("name") or spec.get("url") or "mcp"
            if not spec.get("enabled", True):
                self.status[name] = {"ok": False, "reason": "disabled"}
                continue
            try:
                client = self._open(spec)
                tools = client.list_tools()
                self.servers[name] = client
                self.status[name] = {
                    "ok": True,
                    "tools": len(tools),
                    "names": [t.get("name") for t in tools],
                }
            except Exception as e:
                self.status[name] = {"ok": False, "reason": str(e)[:200]}
        return self.status

    def refresh(self, config):
        return self.load(config)

    def _close(self, entry):
        if isinstance(entry, dict):
            return
        try:
            entry.close()
        except Exception:
            pass

    def all_tools(self):
        """Return OpenAI tool schemas + a name->(server, mcp-tool) index."""
        schemas = []
        index = {}
        for name, entry in self.servers.items():
            if isinstance(entry, dict):
                continue
            try:
                tools = entry.list_tools()
            except Exception:
                continue
            for t in tools:
                tname = t.get("name")
                if not tname:
                    continue
                final = tname if tname not in index else f"{name}__{tname}"
                schema = {
                    "type": "function",
                    "function": {
                        "name": final,
                        "description": t.get("description", "")[:1000],
                        "parameters": t.get("inputSchema")
                        or {"type": "object", "properties": {}},
                    },
                }
                schemas.append(schema)
                index[final] = (name, tname)
        return schemas, index

    def call_by(self, server_name, tname, args):
        server = self.servers.get(server_name)
        if isinstance(server, dict) or server is None:
            raise McpError(f"mcp server {server_name} not connected")
        return server.call_tool(tname, args)
