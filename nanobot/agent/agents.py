"""Built-in agent definitions for the orchestrator pattern."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nanobot.config.schema import AgentRole, ModelSpec


@dataclass
class AgentDef:
    """Immutable definition for a built-in specialist agent."""

    name: str
    description: str  # One-liner shown to the orchestrator
    system_prompt: str  # Full instructions sent to the agent
    default_model: str  # Default model identifier
    tool_allow: list[str] = field(default_factory=list)
    temperature: float = 0.2
    max_iterations: int = 15


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_EXPLORER_PROMPT = """\
You are Explorer -- a fast codebase navigation specialist.

Role: Quick contextual search across filesystems and codebases. \
Answer "where is X?", "find Y", "which file has Z?".

Capabilities: Read files, list directories, run grep/find via shell.

Behavior:
- Fire multiple searches in parallel if needed
- Return file paths with relevant snippets and line numbers
- Be fast and thorough

Constraints:
- READ-ONLY: Search and report, never modify files
- Be exhaustive but concise
- Include line numbers when relevant

Output Format:
- File paths with brief descriptions of what's at each location
- Concise answer to the question asked"""

_RESEARCHER_PROMPT = """\
You are Researcher -- a documentation and web research specialist.

Role: Find official docs, library APIs, external references, and current best practices.

Capabilities: Web search, fetch URLs, read local files for context.

Behavior:
- Provide evidence-based answers with sources
- Quote relevant snippets from documentation
- Distinguish between official and community patterns

Constraints:
- Research only -- don't modify files or run shell commands
- Prefer official documentation over blog posts
- Note when information may be outdated

Output Format:
- Findings with source URLs
- Key code examples or API signatures
- Confidence level in the information"""

_CODER_PROMPT = """\
You are Coder -- a focused implementation specialist.

Role: Execute code changes efficiently. You receive context and specs, you implement.

Capabilities: Read, write, edit files. Run shell commands for testing/building.

Behavior:
- Read files before using edit/write tools
- Be fast and direct -- no research, no delegation
- Run tests when relevant
- Report completion with summary of changes

Constraints:
- NO external research (no web tools)
- NO delegation to other agents
- If context is insufficient, read the files listed; only ask for \
missing inputs you cannot retrieve

Output Format:
- Summary of what was implemented
- List of files changed with brief description
- Verification status (tests passed/failed/skipped)"""

_THINKER_PROMPT = """\
You are Thinker -- a strategic technical advisor.

Role: Architecture decisions, complex debugging, code review, and engineering guidance.

Capabilities: Deep reasoning about codebases, tradeoffs, and system design.

Behavior:
- Be direct and concise
- Provide actionable recommendations
- Explain reasoning briefly
- Acknowledge uncertainty when present

Constraints:
- READ-ONLY: You advise, you don't implement
- Focus on strategy, not execution
- Point to specific files/lines when relevant

Output Format:
- Analysis of the problem/decision
- Recommended approach with reasoning
- Tradeoffs and risks
- Specific file/line references"""

# ---------------------------------------------------------------------------
# Built-in agents
# ---------------------------------------------------------------------------

_BUILTIN_AGENTS: dict[str, AgentDef] = {
    "explorer": AgentDef(
        name="explorer",
        description="Codebase/filesystem search specialist (cheap, fast model)",
        system_prompt=_EXPLORER_PROMPT,
        default_model="gemini/gemini-2.5-flash-preview-05-20",
        tool_allow=["read_file", "list_dir", "exec"],
        temperature=0.1,
        max_iterations=15,
    ),
    "researcher": AgentDef(
        name="researcher",
        description="Web research & documentation lookup (cheap, fast model)",
        system_prompt=_RESEARCHER_PROMPT,
        default_model="gemini/gemini-2.5-flash-preview-05-20",
        tool_allow=["web_search", "web_fetch", "read_file"],
        temperature=0.1,
        max_iterations=15,
    ),
    "coder": AgentDef(
        name="coder",
        description="Implementation specialist (powerful model)",
        system_prompt=_CODER_PROMPT,
        default_model="anthropic/claude-sonnet-4-5",
        tool_allow=["read_file", "write_file", "edit_file", "list_dir", "exec"],
        temperature=0.2,
        max_iterations=15,
    ),
    "thinker": AgentDef(
        name="thinker",
        description="Strategic advisor (most powerful model -- use sparingly)",
        system_prompt=_THINKER_PROMPT,
        default_model="anthropic/claude-opus-4-6",
        tool_allow=["read_file", "list_dir"],
        temperature=0.1,
        max_iterations=15,
    ),
}


def get_builtin_agents() -> dict[str, AgentDef]:
    """Return a *copy* of the built-in agent definitions."""
    return dict(_BUILTIN_AGENTS)


def resolve_agent(
    name: str,
    roles: dict[str, AgentRole] | None = None,
) -> AgentDef | None:
    """Resolve an agent by name, merging config overrides onto the built-in def.

    Returns ``None`` if the name matches neither a built-in agent nor a
    config-defined custom role.
    """
    builtin = _BUILTIN_AGENTS.get(name)
    role = (roles or {}).get(name)

    if builtin is None and role is None:
        return None

    # Start from built-in or create a bare definition for custom roles
    if builtin:
        agent = AgentDef(
            name=builtin.name,
            description=builtin.description,
            system_prompt=builtin.system_prompt,
            default_model=builtin.default_model,
            tool_allow=list(builtin.tool_allow),
            temperature=builtin.temperature,
            max_iterations=builtin.max_iterations,
        )
    else:
        # Custom role with no built-in base
        agent = AgentDef(
            name=name,
            description=f"Custom agent: {name}",
            system_prompt=f"You are {name}, a specialist agent.",
            default_model="",
            tool_allow=[],
        )

    # Apply config overrides
    if role:
        if role.model:
            agent.default_model = role.model.model
            if role.model.temperature is not None:
                agent.temperature = role.model.temperature
            if role.model.max_tool_iterations is not None:
                agent.max_iterations = role.model.max_tool_iterations
        if role.system_prompt:
            agent.system_prompt = role.system_prompt
        if role.tool_allow:
            agent.tool_allow = list(role.tool_allow)
        if role.max_tool_iterations is not None:
            agent.max_iterations = role.max_tool_iterations

    return agent
