"""Subagent manager for background task execution and synchronous delegation."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.config.schema import ModelSpec, ModelSpecBase, AgentRole
from nanobot.agent.agents import resolve_agent, AgentDef
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool


class SubagentManager:
    """
    Manages background subagent execution and synchronous delegation.

    Subagents are lightweight agent instances that run in the background
    to handle specific tasks. They share the same LLM provider but have
    isolated context and a focused system prompt.

    The ``delegate()`` method runs an agent synchronously (blocking) and
    returns the result directly — used by the ``delegate`` tool.
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: ModelSpec | None = None,
        roles: dict[str, AgentRole] | None = None,
        brave_api_key: str | None = None,
        claude_max_provider: LLMProvider | None = None,
        codex_provider: LLMProvider | None = None,
    ):
        self.provider = provider
        self.claude_max_provider = claude_max_provider
        self.codex_provider = codex_provider
        self.workspace = workspace
        self.bus = bus
        self.model_spec = model or ModelSpec(model=provider.get_default_model())
        self.roles = roles or {}
        self.brave_api_key = brave_api_key
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        role: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> str:
        """
        Spawn a subagent to execute a task in the background.

        Args:
            task: The task description for the subagent.
            label: Optional human-readable label for the task.
            origin_channel: The channel to announce results to.
            origin_chat_id: The chat ID to announce results to.

        Returns:
            Status message indicating the subagent was started.
        """
        if role and role not in self.roles:
            return f"Unknown role '{role}'. Available roles: {', '.join(self.roles) or 'none'}."

        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")

        origin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
        }

        # Create background task
        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin, role)
        )
        self._running_tasks[task_id] = bg_task

        # Cleanup when done
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))

        logger.info(f"Spawned subagent [{task_id}]: {display_label}")
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    def _provider_for_model(self, model: str) -> LLMProvider:
        """Return the correct provider based on model prefix."""
        if model.startswith("claude_max/") and self.claude_max_provider:
            return self.claude_max_provider
        if model.startswith("codex/") and self.codex_provider:
            return self.codex_provider
        return self.provider

    async def delegate(
        self,
        agent_name: str,
        task: str,
        context: str = "",
    ) -> str:
        """
        Run a specialist agent synchronously and return its result.

        This resolves the agent definition (built-in merged with config
        overrides), builds its tool set, and runs it to completion.
        """
        agent_def = resolve_agent(agent_name, self.roles)
        if agent_def is None:
            from nanobot.agent.agents import get_builtin_agents
            available = list(get_builtin_agents().keys())
            # Also include custom roles
            for role_name in self.roles:
                if role_name not in available:
                    available.append(role_name)
            return (
                f"Unknown agent '{agent_name}'. "
                f"Available agents: {', '.join(available)}"
            )

        # Determine model
        model_id = agent_def.default_model or self.model_spec.model
        provider = self._provider_for_model(model_id)

        model_spec = ModelSpec(
            model=model_id,
            max_tokens=self.model_spec.max_tokens,
            temperature=agent_def.temperature,
            max_tool_iterations=agent_def.max_iterations,
        )

        # Build tools restricted to the agent's allow list
        tools = self._build_tools_for_agent(agent_def)

        # Build the prompt
        full_task = task
        if context:
            full_task = f"{task}\n\nContext:\n{context}"

        system_prompt = f"{agent_def.system_prompt}\n\n## Workspace\nYour workspace is at: {self.workspace}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_task},
        ]

        # Run agent loop
        max_iterations = agent_def.max_iterations
        iteration = 0
        final_result: str | None = None

        logger.info(f"Delegate @{agent_name} starting (model={model_id})")

        while iteration < max_iterations:
            iteration += 1

            try:
                response = await provider.chat(
                    messages=messages,
                    tools=tools.get_definitions() if len(tools) > 0 else None,
                    model=model_id,
                    max_tokens=model_spec.max_tokens,
                    temperature=model_spec.temperature,
                )
            except Exception as exc:
                logger.error(f"Delegate @{agent_name} LLM error: {exc}")
                return f"Agent @{agent_name} failed: {exc}"

            if response.finish_reason == "error":
                return f"Agent @{agent_name} error: {response.content}"

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": tool_call_dicts,
                    }
                )

                for tool_call in response.tool_calls:
                    logger.debug(f"Delegate @{agent_name} executing: {tool_call.name}")
                    result = await tools.execute(tool_call.name, tool_call.arguments)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        }
                    )
            else:
                final_result = response.content
                break

        if final_result is None:
            final_result = f"Agent @{agent_name} reached iteration limit without a final response."

        logger.info(f"Delegate @{agent_name} completed ({iteration} iterations)")
        return final_result

    def _build_tools_for_agent(self, agent_def: AgentDef) -> ToolRegistry:
        """Build a tool registry restricted to the agent's allowed tools."""
        tools = ToolRegistry()
        available_tools = [
            ReadFileTool(),
            WriteFileTool(),
            EditFileTool(),
            ListDirTool(),
            ExecTool(working_dir=str(self.workspace)),
            WebSearchTool(api_key=self.brave_api_key),
            WebFetchTool(),
        ]

        allow = set(agent_def.tool_allow) if agent_def.tool_allow else set()

        for tool in available_tools:
            if allow and tool.name not in allow:
                continue
            tools.register(tool)

        return tools

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
        role: str | None,
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info(f"Subagent [{task_id}] starting task: {label}")

        try:
            role_config = self.roles.get(role) if role else None
            model_spec = role_config.model if role_config and role_config.model else self.model_spec

            # Build subagent tools (no message tool, no spawn tool)
            tools = self._build_tools(role_config)

            # Build messages with subagent-specific prompt
            system_prompt = self._build_subagent_prompt(task)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            # Run agent loop (limited iterations)
            max_iterations = (
                role_config.max_tool_iterations
                if role_config and role_config.max_tool_iterations is not None
                else model_spec.max_tool_iterations
            )
            iteration = 0
            final_result: str | None = None

            while iteration < max_iterations:
                iteration += 1

                response = await self._chat_with_fallback(
                    messages=messages,
                    tools=tools.get_definitions(),
                    model_spec=model_spec,
                )

                if response.has_tool_calls:
                    # Add assistant message with tool calls
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ]
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response.content or "",
                            "tool_calls": tool_call_dicts,
                        }
                    )

                    # Execute tools
                    for tool_call in response.tool_calls:
                        logger.debug(f"Subagent [{task_id}] executing: {tool_call.name}")
                        result = await tools.execute(tool_call.name, tool_call.arguments)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.name,
                                "content": result,
                            }
                        )
                else:
                    final_result = response.content
                    break

            if final_result is None:
                final_result = "Task completed but no final response was generated."

            logger.info(f"Subagent [{task_id}] completed successfully")
            await self._announce_result(task_id, label, task, final_result, origin, "ok")

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Subagent [{task_id}] failed: {e}")
            await self._announce_result(task_id, label, task, error_msg, origin, "error")

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}

Result:
{result}

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like "subagent" or task IDs."""

        # Inject as system message to trigger main agent
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)
        logger.debug(
            f"Subagent [{task_id}] announced result to {origin['channel']}:{origin['chat_id']}"
        )

    def _build_subagent_prompt(self, task: str) -> str:
        """Build a focused system prompt for the subagent."""
        return f"""# Subagent

You are a subagent spawned by the main agent to complete a specific task.

## Your Task
{task}

## Rules
1. Stay focused - complete only the assigned task, nothing else
2. Your final response will be reported back to the main agent
3. Do not initiate conversations or take on side tasks
4. Be concise but informative in your findings

## What You Can Do
- Read and write files in the workspace
- Execute shell commands
- Search the web and fetch web pages
- Complete the task thoroughly

## What You Cannot Do
- Send messages directly to users (no message tool available)
- Spawn other subagents
- Access the main agent's conversation history

## Workspace
Your workspace is at: {self.workspace}

When you have completed the task, provide a clear summary of your findings or actions."""

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)

    def _build_tools(self, role: AgentRole | None) -> ToolRegistry:
        tools = ToolRegistry()
        available_tools = [
            ReadFileTool(),
            WriteFileTool(),
            EditFileTool(),
            ListDirTool(),
            ExecTool(working_dir=str(self.workspace)),
            WebSearchTool(api_key=self.brave_api_key),
            WebFetchTool(),
        ]

        allow = set(role.tool_allow) if role and role.tool_allow else set()
        deny = set(role.tool_deny) if role and role.tool_deny else set()

        for tool in available_tools:
            if allow and tool.name not in allow:
                continue
            if deny and tool.name in deny:
                continue
            tools.register(tool)

        return tools

    def _model_candidates(self, model_spec: ModelSpec) -> list[ModelSpecBase]:
        base = ModelSpecBase(
            model=model_spec.model,
            max_tokens=model_spec.max_tokens,
            temperature=model_spec.temperature,
            max_tool_iterations=model_spec.max_tool_iterations,
        )
        return [base, *model_spec.fallbacks]

    async def _chat_with_fallback(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model_spec: ModelSpec,
        max_retries: int = 3,
        retry_delay: float = 15.0,
    ) -> LLMResponse:
        """
        Call LLM with retry logic for transient errors and fallback to other models.

        Transient errors (503, 429, 500, etc.) are retried with delay.
        Permanent failures trigger fallback to next configured model.
        """
        import asyncio

        last_error: LLMResponse | None = None

        for spec in self._model_candidates(model_spec):
            # Retry loop for transient errors on this model
            for attempt in range(max_retries):
                try:
                    response = await self.provider.chat(
                        messages=messages,
                        tools=tools,
                        model=spec.model,
                        max_tokens=spec.max_tokens,
                        temperature=spec.temperature,
                    )
                except Exception as exc:
                    error_str = str(exc).lower()
                    # Check if transient error (503, 429, 500, timeout, etc.)
                    is_transient = any(
                        code in error_str
                        for code in [
                            "503",
                            "429",
                            "500",
                            "502",
                            "504",
                            "timeout",
                            "rate limit",
                            "overloaded",
                            "temporarily unavailable",
                            "internal server error",
                        ]
                    )

                    if is_transient and attempt < max_retries - 1:
                        delay = retry_delay * (attempt + 1)  # Exponential backoff
                        logger.warning(
                            f"Transient error on {spec.model}, retrying in {delay}s "
                            f"(attempt {attempt + 1}/{max_retries}): {exc}"
                        )
                        await asyncio.sleep(delay)
                        continue

                    last_error = LLMResponse(
                        content=f"Error calling LLM: {exc}",
                        finish_reason="error",
                    )
                    break  # Move to next model

                # Check response for error content
                if response.finish_reason == "error":
                    error_content = response.content or ""
                    is_transient = any(
                        code in error_content.lower()
                        for code in [
                            "503",
                            "429",
                            "500",
                            "502",
                            "504",
                            "timeout",
                            "rate limit",
                            "overloaded",
                        ]
                    )

                    if is_transient and attempt < max_retries - 1:
                        delay = retry_delay * (attempt + 1)
                        logger.warning(
                            f"Transient error response on {spec.model}, retrying in {delay}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue

                    last_error = response
                    break  # Move to next model

                if response.content and response.content.startswith("Error calling LLM:"):
                    error_content = response.content.lower()
                    is_transient = any(
                        code in error_content
                        for code in [
                            "503",
                            "429",
                            "500",
                            "502",
                            "504",
                            "timeout",
                            "rate limit",
                            "overloaded",
                        ]
                    )

                    if is_transient and attempt < max_retries - 1:
                        delay = retry_delay * (attempt + 1)
                        logger.warning(
                            f"Transient error on {spec.model}, retrying in {delay}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue

                    last_error = response
                    break  # Move to next model

                # Success!
                return response

            # If we broke out of retry loop, try next model
            if last_error:
                logger.info(f"Falling back from {spec.model} to next model")

        return last_error or LLMResponse(
            content="Error calling LLM: all fallback models failed",
            finish_reason="error",
        )
