"""Tests for nanobot.agent.agents — built-in agent definitions and delegation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.agents import (
    AgentDef,
    get_builtin_agents,
    resolve_agent,
)
from nanobot.config.schema import AgentRole, ModelSpec


# ---------------------------------------------------------------------------
# get_builtin_agents()
# ---------------------------------------------------------------------------


def test_builtin_agents_returns_four():
    agents = get_builtin_agents()
    assert set(agents.keys()) == {"explorer", "researcher", "coder", "thinker"}


def test_builtin_agents_have_required_fields():
    for name, agent in get_builtin_agents().items():
        assert isinstance(agent, AgentDef)
        assert agent.name == name
        assert agent.description
        assert agent.system_prompt
        assert agent.default_model
        assert isinstance(agent.tool_allow, list)
        assert agent.temperature > 0
        assert agent.max_iterations > 0


def test_builtin_agents_returns_copy():
    a = get_builtin_agents()
    b = get_builtin_agents()
    assert a is not b
    assert a.keys() == b.keys()


# ---------------------------------------------------------------------------
# resolve_agent()
# ---------------------------------------------------------------------------


def test_resolve_builtin_without_overrides():
    agent = resolve_agent("coder")
    assert agent is not None
    assert agent.name == "coder"
    assert "claude-sonnet" in agent.default_model


def test_resolve_unknown_agent_returns_none():
    assert resolve_agent("nonexistent") is None


def test_resolve_with_model_override():
    roles = {
        "coder": AgentRole(model=ModelSpec(model="claude_max/claude-sonnet-4-5")),
    }
    agent = resolve_agent("coder", roles)
    assert agent is not None
    assert agent.default_model == "claude_max/claude-sonnet-4-5"


def test_resolve_with_system_prompt_override():
    custom_prompt = "You are a custom coder."
    roles = {
        "coder": AgentRole(system_prompt=custom_prompt),
    }
    agent = resolve_agent("coder", roles)
    assert agent is not None
    assert agent.system_prompt == custom_prompt


def test_resolve_with_tool_allow_override():
    roles = {
        "explorer": AgentRole(tool_allow=["read_file"]),
    }
    agent = resolve_agent("explorer", roles)
    assert agent is not None
    assert agent.tool_allow == ["read_file"]


def test_resolve_custom_role():
    """Custom roles not in builtins should still resolve."""
    roles = {
        "security_auditor": AgentRole(
            model=ModelSpec(model="anthropic/claude-opus-4-6"),
            system_prompt="You are a security auditor.",
            tool_allow=["read_file", "exec"],
        ),
    }
    agent = resolve_agent("security_auditor", roles)
    assert agent is not None
    assert agent.name == "security_auditor"
    assert agent.default_model == "anthropic/claude-opus-4-6"
    assert agent.system_prompt == "You are a security auditor."


# ---------------------------------------------------------------------------
# Agent tool restrictions
# ---------------------------------------------------------------------------


def test_explorer_is_read_only():
    agent = resolve_agent("explorer")
    assert agent is not None
    assert "write_file" not in agent.tool_allow
    assert "edit_file" not in agent.tool_allow


def test_researcher_has_web_tools():
    agent = resolve_agent("researcher")
    assert agent is not None
    assert "web_search" in agent.tool_allow
    assert "web_fetch" in agent.tool_allow


def test_coder_has_write_tools():
    agent = resolve_agent("coder")
    assert agent is not None
    assert "write_file" in agent.tool_allow
    assert "edit_file" in agent.tool_allow
    assert "exec" in agent.tool_allow


def test_thinker_is_read_only():
    agent = resolve_agent("thinker")
    assert agent is not None
    assert "write_file" not in agent.tool_allow
    assert "edit_file" not in agent.tool_allow
    assert "exec" not in agent.tool_allow


# ---------------------------------------------------------------------------
# Orchestrator prompt includes agents
# ---------------------------------------------------------------------------


def test_context_builder_includes_agents():
    """ContextBuilder should inject agent descriptions into the system prompt."""
    import tempfile
    from pathlib import Path
    from nanobot.agent.context import ContextBuilder

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "memory").mkdir()
        builder = ContextBuilder(workspace)
        prompt = builder.build_system_prompt("hello")
        assert "@explorer" in prompt
        assert "@coder" in prompt
        assert "@thinker" in prompt
        assert "@researcher" in prompt
        assert "Delegation Rules" in prompt


# ---------------------------------------------------------------------------
# DelegateTool
# ---------------------------------------------------------------------------


def test_delegate_tool_schema():
    from nanobot.agent.tools.delegate import DelegateTool

    manager = MagicMock()
    tool = DelegateTool(manager=manager)
    assert tool.name == "delegate"
    schema = tool.to_schema()
    props = schema["function"]["parameters"]["properties"]
    assert "agent" in props
    assert "task" in props
    assert "context" in props


@pytest.mark.asyncio
async def test_delegate_tool_calls_manager():
    from nanobot.agent.tools.delegate import DelegateTool

    manager = MagicMock()
    manager.delegate = AsyncMock(return_value="result from coder")
    tool = DelegateTool(manager=manager)

    result = await tool.execute(agent="coder", task="write hello.py")
    assert result == "result from coder"
    manager.delegate.assert_called_once_with(
        agent_name="coder", task="write hello.py", context=""
    )


@pytest.mark.asyncio
async def test_delegate_tool_requires_agent():
    from nanobot.agent.tools.delegate import DelegateTool

    manager = MagicMock()
    tool = DelegateTool(manager=manager)
    result = await tool.execute(agent="", task="something")
    assert "required" in result.lower()


@pytest.mark.asyncio
async def test_delegate_tool_requires_task():
    from nanobot.agent.tools.delegate import DelegateTool

    manager = MagicMock()
    tool = DelegateTool(manager=manager)
    result = await tool.execute(agent="coder", task="")
    assert "required" in result.lower()
