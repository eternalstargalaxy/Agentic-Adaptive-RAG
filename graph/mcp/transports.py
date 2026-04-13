from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from graph.mcp.client import MCPToolDefinition, MCPToolResponse


@dataclass(frozen=True)
class StdioMCPServerConfig:
    server_name: str
    command: str
    args: Sequence[str] = field(default_factory=tuple)
    env: Dict[str, str] = field(default_factory=dict)


class StdioMCPTransport:
    """
    Real MCP transport based on stdio.

    This transport connects to an external MCP server process through the
    official Python MCP SDK when available. The rest of the graph can still use
    the lightweight MCP client abstraction without being tied to in-process
    handlers only.
    """

    def __init__(self, config: StdioMCPServerConfig) -> None:
        self.config = config

    async def _with_session(self, callback):
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except Exception as exc:
            raise RuntimeError(
                "The 'mcp' Python package is required for stdio MCP transport."
            ) from exc

        server_params = StdioServerParameters(
            command=self.config.command,
            args=list(self.config.args),
            env=self.config.env or None,
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await callback(session)

    @staticmethod
    def _run(coro):
        return asyncio.run(coro)

    @staticmethod
    def _coerce_tool_list(raw_response: Any) -> List[Any]:
        return list(getattr(raw_response, "tools", raw_response or []))

    @staticmethod
    def _coerce_content_text(raw_content: Any) -> str:
        parts: List[str] = []
        for item in raw_content or []:
            if hasattr(item, "text"):
                parts.append(str(item.text))
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n\n".join(part for part in parts if part)

    def list_tools(self) -> List[MCPToolDefinition]:
        async def _list_tools(session):
            response = await session.list_tools()
            return self._coerce_tool_list(response)

        raw_tools = self._run(self._with_session(_list_tools))
        tools: List[MCPToolDefinition] = []
        for tool in raw_tools:
            tools.append(
                MCPToolDefinition(
                    name=getattr(tool, "name", ""),
                    description=getattr(tool, "description", ""),
                )
            )
        return tools

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPToolResponse:
        async def _call_tool(session):
            return await session.call_tool(name, arguments)

        raw_response = self._run(self._with_session(_call_tool))
        structured_content = (
            getattr(raw_response, "structuredContent", None)
            or getattr(raw_response, "structured_content", None)
            or {}
        )
        content = self._coerce_content_text(getattr(raw_response, "content", []))
        return MCPToolResponse(
            tool_name=name,
            content=content,
            structured_content=dict(structured_content),
        )
