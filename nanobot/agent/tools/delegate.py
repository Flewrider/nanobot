"""Synchronous delegate tool — runs a specialist agent and waits for the result."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class DelegateTool(Tool):
    """
    Delegate a task to a specialist agent and wait for the result.

    Unlike the ``spawn`` tool (async/background), ``delegate`` runs the
    sub-agent **synchronously** and returns its output directly as the
    tool result.  The orchestrator can issue multiple ``delegate`` calls
    in a single response to run agents in parallel.
    """

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager

    @property
    def name(self) -> str:
        return "delegate"

    @property
    def description(self) -> str:
        return (
            "Delegate a task to a specialist agent and wait for the result. "
            "You can call this multiple times in one response to run agents in parallel."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": (
                        "Agent name: explorer, researcher, coder, thinker "
                        "(or any custom role)"
                    ),
                },
                "task": {
                    "type": "string",
                    "description": (
                        "What to do — be specific, provide file paths and context"
                    ),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Optional additional context, file paths, or prior findings"
                    ),
                },
            },
            "required": ["agent", "task"],
        }

    async def execute(self, **kwargs: Any) -> str:
        agent_name: str = kwargs.get("agent", "")
        task: str = kwargs.get("task", "")
        context: str = kwargs.get("context", "")

        if not agent_name:
            return "Error: 'agent' is required (e.g. explorer, researcher, coder, thinker)"
        if not task:
            return "Error: 'task' is required"

        return await self._manager.delegate(
            agent_name=agent_name,
            task=task,
            context=context,
        )
