from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass(frozen=True)
class MCPToolDefinition:
    name: str
    description: str


@dataclass
class MCPToolResponse:
    tool_name: str
    content: str
    structured_content: Dict[str, Any] = field(default_factory=dict)


class InProcessMCPClient:
    """
    A lightweight MCP-style client boundary.

    This phase keeps all tools in-process, but the rest of the graph talks to
    them through a uniform client interface so the implementation can later be
    swapped for a real MCP transport.
    """

    def __init__(self) -> None:
        self._definitions: Dict[str, MCPToolDefinition] = {}
        self._handlers: Dict[str, Callable[..., Any]] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable[..., Any],
    ) -> None:
        self._definitions[name] = MCPToolDefinition(name=name, description=description)
        self._handlers[name] = handler

    def list_tools(self) -> List[MCPToolDefinition]:
        return list(self._definitions.values())

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPToolResponse:
        if name not in self._handlers:
            raise KeyError(f"Tool '{name}' is not registered in the MCP client.")

        raw_result = self._handlers[name](**arguments)
        if isinstance(raw_result, MCPToolResponse):
            return raw_result

        if isinstance(raw_result, dict):
            content = str(raw_result.get("content", ""))
            structured_content = {
                key: value for key, value in raw_result.items() if key != "content"
            }
            return MCPToolResponse(
                tool_name=name,
                content=content,
                structured_content=structured_content,
            )

        return MCPToolResponse(tool_name=name, content=str(raw_result))
