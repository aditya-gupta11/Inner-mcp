"""Async client pool for the MCP servers used by the inner model."""

import asyncio
import base64
import json
import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log = logging.getLogger("mcp_client")


def _expand_value(value: str, workspace_root: Path) -> str:
    """Expand environment variables and the configured workspace placeholder."""
    return os.path.expandvars(value).replace("{workspace_root}", str(workspace_root))


class MCPServerConnection:
    def __init__(self, config: dict, workspace_root: Path):
        self.name = config["name"]
        self.config = config
        self.workspace_root = workspace_root
        self.session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None

    async def start(self) -> None:
        command_parts = self.config.get("command", [])
        if not command_parts:
            raise ValueError(f"MCP server {self.name!r} has no command")

        command = _expand_value(str(command_parts[0]), self.workspace_root)
        args = [
            _expand_value(str(arg), self.workspace_root)
            for arg in command_parts[1:]
        ]

        configured_env = self.config.get("env")
        env = None
        if configured_env:
            env = {
                **os.environ,
                **{
                    key: _expand_value(str(value), self.workspace_root)
                    for key, value in configured_env.items()
                },
            }

        configured_cwd = self.config.get("cwd")
        cwd = (
            _expand_value(str(configured_cwd), self.workspace_root)
            if configured_cwd
            else str(self.workspace_root)
        )

        params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
            cwd=cwd,
        )

        stack = AsyncExitStack()
        try:
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(params)
            )
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
        except Exception:
            await stack.aclose()
            raise

        self._stack = stack
        self.session = session
        log.info(
            "MCP server %r started — workspace: %s",
            self.name,
            self.workspace_root,
        )

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self.session = None

    async def list_tools(self) -> list[Any]:
        if self.session is None:
            raise RuntimeError(f"MCP server {self.name!r} is not started")

        tools = []
        cursor = None
        while True:
            result = await self.session.list_tools(cursor=cursor)
            tools.extend(result.tools)
            cursor = result.nextCursor
            if not cursor:
                return tools

    async def call_tool(self, name: str, arguments: dict) -> dict:
        if self.session is None:
            raise RuntimeError(f"MCP server {self.name!r} is not started")

        result = await self.session.call_tool(name, arguments=arguments)
        text_parts: list[str] = []
        screenshot_bytes: bytes | None = None

        for item in result.content:
            item_type = getattr(item, "type", None)
            if item_type == "text":
                text_parts.append(item.text)
            elif item_type == "image":
                if screenshot_bytes is None:
                    screenshot_bytes = base64.b64decode(item.data)
                text_parts.append(f"[image: {item.mimeType}]")
            else:
                text_parts.append(
                    json.dumps(item.model_dump(mode="json"), default=str)
                )

        response = {
            "ok": not result.isError,
            "content": "\n".join(text_parts),
        }
        if result.structuredContent is not None:
            response["structured_content"] = result.structuredContent
        if screenshot_bytes is not None:
            response["screenshot_bytes"] = screenshot_bytes
        return response


class MCPClientPool:
    """
    Starts configured MCP servers and exposes namespaced tools.

    workspace_root resolution order:
      1. workspace_root argument passed directly (from Codex roots via Context)
      2. MCP_WORKSPACE_ROOT environment variable
      3. Current working directory (cwd at process start)

    The resolved root is substituted for {workspace_root} in all command args
    and cwd fields in config.yaml, so the filesystem MCP server automatically
    gains access to whichever project Codex is working in.
    """

    def __init__(self, configs: list[dict], workspace_root: str | Path | None = None):
        # Resolve workspace root with explicit fallback chain
        if workspace_root is not None:
            resolved_root = Path(workspace_root).expanduser().resolve()
        elif os.environ.get("MCP_WORKSPACE_ROOT"):
            resolved_root = Path(os.environ["MCP_WORKSPACE_ROOT"]).expanduser().resolve()
        else:
            resolved_root = Path.cwd().resolve()

        self.workspace_root = resolved_root
        log.info("MCPClientPool workspace_root: %s", self.workspace_root)

        self.connections = {
            config["name"]: MCPServerConnection(config, self.workspace_root)
            for config in configs
        }
        self._tool_routes: dict[str, tuple[MCPServerConnection, str]] = {}

    async def start_all(self) -> None:
        names = list(self.connections.keys())
        conns = list(self.connections.values())
        results = await asyncio.gather(
            *(conn.start() for conn in conns),
            return_exceptions=True,
        )
        failures = []
        for name, result in zip(names, results):
            if isinstance(result, Exception):
                failures.append(f"{name}: {result}")
                log.error("Failed to start MCP server %r: %s", name, result)

        if failures:
            if len(failures) == len(conns):
                raise RuntimeError("All MCP servers failed to start: " + "; ".join(failures))
            else:
                log.warning(
                    "%s/%s MCP servers failed to start: %s",
                    len(failures), len(conns), "; ".join(failures),
                )

    async def close_all(self) -> None:
        await asyncio.gather(
            *(conn.close() for conn in self.connections.values()),
            return_exceptions=True,
        )

    async def get_all_tool_schemas(self) -> list[dict]:
        schemas = []
        self._tool_routes.clear()

        for conn in self.connections.values():
            if conn.session is None:
                continue
            try:
                tools = await conn.list_tools()
            except Exception as exc:
                log.error("Failed to list tools from %r: %s", conn.name, exc)
                continue

            for tool in tools:
                public_name = f"{conn.name}__{tool.name}"
                self._tool_routes[public_name] = (conn, tool.name)
                schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": public_name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema or {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    }
                )

        return schemas

    async def call_tool(self, public_name: str, arguments: dict) -> dict:
        route = self._tool_routes.get(public_name)
        if route is None:
            # Routes may be stale — refresh once
            await self.get_all_tool_schemas()
            route = self._tool_routes.get(public_name)
        if route is None:
            raise KeyError(f"Unknown MCP tool: {public_name}")

        conn, original_name = route
        return await conn.call_tool(original_name, arguments)
