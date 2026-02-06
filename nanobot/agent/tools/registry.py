"""Tool registry for dynamic tool management."""

import re
from typing import Any

from nanobot.agent.tools.base import Tool

# Patterns that look like API keys — redacted before results reach the LLM
_KEY_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),   # Anthropic
    re.compile(r"sk-or-[A-Za-z0-9_-]{20,}"),     # OpenRouter
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),         # OpenAI / generic
    re.compile(r"gsk_[A-Za-z0-9_-]{20,}"),        # Groq
    re.compile(r"AIza[A-Za-z0-9_-]{20,}"),        # Google / Gemini
]


def redact_api_keys(text: str) -> str:
    """Replace anything that looks like an API key with ``[REDACTED]``."""
    for pattern in _KEY_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


class ToolRegistry:
    """
    Registry for agent tools.
    
    Allows dynamic registration and execution of tools.
    """
    
    def __init__(self):
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)
    
    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]
    
    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """
        Execute a tool by name with given parameters.
        
        Args:
            name: Tool name.
            params: Tool parameters.
        
        Returns:
            Tool execution result as string.
        
        Raises:
            KeyError: If tool not found.
        """
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found"
        
        try:
            result = await tool.execute(**params)
            return redact_api_keys(result)
        except Exception as e:
            return redact_api_keys(f"Error executing {name}: {str(e)}")
    
    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
