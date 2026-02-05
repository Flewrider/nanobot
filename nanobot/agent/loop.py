"""Agent loop: the core processing engine."""

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.config.schema import ModelSpec, ModelSpecBase, AgentRole
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import SessionManager


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: ModelSpec | None = None,
        max_iterations: int | None = None,
        roles: dict[str, AgentRole] | None = None,
        brave_api_key: str | None = None,
    ):
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model_spec = model or ModelSpec(model=provider.get_default_model())
        self.max_iterations = max_iterations or self.model_spec.max_tool_iterations
        self.brave_api_key = brave_api_key

        self.context = ContextBuilder(workspace)
        self.sessions = SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagent_roles = roles or {}
        self._last_user_context: tuple[str, str] | None = None
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model_spec,
            roles=self.subagent_roles,
            brave_api_key=brave_api_key,
        )

        self._running = False
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools
        self.tools.register(ReadFileTool())
        self.tools.register(WriteFileTool())
        self.tools.register(EditFileTool())
        self.tools.register(ListDirTool())

        # Shell tool
        self.tools.register(ExecTool(working_dir=str(self.workspace)))

        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())

        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)

        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")

        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)

                # Process it
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except MemoryError:
                    logger.error("Out of memory while processing message")
                    try:
                        import gc

                        gc.collect()
                    except Exception:
                        pass
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=(
                                "Sorry, I ran out of memory while processing that. "
                                "Please try a shorter request."
                            ),
                        )
                    )
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error response
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=f"Sorry, I encountered an error: {str(e)}",
                        )
                    )
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a single inbound message.

        Args:
            msg: The inbound message to process.

        Returns:
            The response message, or None if no response needed.
        """
        # Handle system messages (subagent announces)
        # The chat_id contains the original "channel:chat_id" to route back to
        if msg.channel == "system":
            return await self._process_system_message(msg)

        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}")

        # Get or create session
        session_key = msg.session_key
        if msg.channel == "heartbeat" and self._last_user_context:
            last_channel, last_chat_id = self._last_user_context
            session_key = f"{last_channel}:{last_chat_id}"
        session = self.sessions.get_or_create(session_key)

        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            if msg.channel == "heartbeat" and self._last_user_context:
                message_tool.set_context(*self._last_user_context)
            else:
                message_tool.set_context(msg.channel, msg.chat_id)

        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            if msg.channel == "heartbeat" and self._last_user_context:
                spawn_tool.set_context(*self._last_user_context)
            else:
                spawn_tool.set_context(msg.channel, msg.chat_id)

        if msg.channel not in {"system", "heartbeat"}:
            self._last_user_context = (msg.channel, msg.chat_id)

        # Build initial messages (use get_history for LLM-formatted messages)
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            media=msg.media if msg.media else None,
        )

        # Agent loop
        iteration = 0
        final_content = None

        auto_continue_limit = 5
        auto_continues = 0
        message_continue_limit = 2
        message_continues = 0
        message_tool_used = False

        while iteration < self.max_iterations:
            iteration += 1

            # If we're at the last iteration, ask for final summary without tools
            at_limit = iteration >= self.max_iterations

            if at_limit:
                # Add a message asking for final summary
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have reached the maximum number of tool calls. "
                            "Please provide a final summary of what you accomplished and any remaining tasks."
                        ),
                    }
                )

            # Call LLM (no tools on final iteration to force a text response)
            response = await self._chat_with_fallback(
                messages=messages,
                tools=None if at_limit else self.tools.get_definitions(),
            )

            # Handle tool calls
            if response.has_tool_calls:
                # Add assistant message with tool calls
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),  # Must be JSON string
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts
                )

                # Execute tools
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments)
                    logger.debug(f"Executing tool: {tool_call.name} with arguments: {args_str}")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                    if tool_call.name == "message":
                        message_tool_used = True
            else:
                # No tool calls, we're done
                if (
                    msg.channel != "system"
                    and auto_continues < auto_continue_limit
                    and self._should_autocontinue(response.content)
                ):
                    auto_continues += 1
                    messages = self.context.add_assistant_message(messages, response.content)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Continue with the execution now. Do not ask for confirmation. "
                                "If the task is done, reply with the final result only."
                            ),
                        }
                    )
                    continue

                if (
                    msg.channel != "system"
                    and message_tool_used
                    and message_continues < message_continue_limit
                ):
                    message_continues += 1
                    message_tool_used = False
                    messages = self.context.add_assistant_message(messages, response.content)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Continue executing the task. Only stop once the work is done."
                            ),
                        }
                    )
                    continue

                if message_tool_used and (not response.content or not response.content.strip()):
                    messages = self.context.add_assistant_message(messages, response.content)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Provide a brief final status update for the user and stop."
                            ),
                        }
                    )
                    message_tool_used = False
                    continue

                final_content = response.content
                break

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # Save to session
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=final_content)

    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).

        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")

        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id

        # Use the origin session for context
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)

        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(origin_channel, origin_chat_id)

        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(origin_channel, origin_chat_id)

        # Build messages with the announce content
        messages = self.context.build_messages(
            history=session.get_history(), current_message=msg.content
        )

        # Agent loop (limited for announce handling)
        iteration = 0
        final_content = None

        while iteration < self.max_iterations:
            iteration += 1

            response = await self._chat_with_fallback(
                messages=messages,
                tools=self.tools.get_definitions(),
            )

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts
                )

                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments)
                    logger.debug(f"Executing tool: {tool_call.name} with arguments: {args_str}")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = response.content
                break

        if final_content is None:
            final_content = "Background task completed."

        # Save to session (mark as system message in history)
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)

        return OutboundMessage(
            channel=origin_channel, chat_id=origin_chat_id, content=final_content
        )

    async def process_direct(self, content: str, session_key: str = "cli:direct") -> str:
        """
        Process a message directly (for CLI usage).

        Args:
            content: The message content.
            session_key: Session identifier.

        Returns:
            The agent's response.
        """
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content=content)

        response = await self._process_message(msg)
        return response.content if response else ""

    async def process_heartbeat(self, content: str) -> str:
        msg = InboundMessage(
            channel="heartbeat",
            sender_id="heartbeat",
            chat_id="heartbeat",
            content=content,
        )
        response = await self._process_message(msg)
        return response.content if response else ""

    def _should_autocontinue(self, content: str | None) -> bool:
        if not content:
            return False
        text = content.lower()
        triggers = [
            "next step",
            "next steps",
            "step 1",
            "step 2",
            "step 3",
            "step 4",
            "plan:",
            "todo",
            "to-do",
            "i will",
            "i'll",
            "i am going to",
            "i'm going to",
            "i am gonna",
            "i'm gonna",
            "let me",
            "first,",
            "second,",
            "third,",
            "then",
            "here's the plan",
            "checklist",
            "- [ ]",
            "1)",
            "2)",
            "1.",
            "2.",
            "3.",
            "1️⃣",
            "2️⃣",
            "3️⃣",
            "doing this now",
            "one sec",
            "hold on",
            "working on it",
            "give me a moment",
            "what i need to do",
        ]
        return any(trigger in text for trigger in triggers)

    def _model_candidates(self) -> list[ModelSpecBase]:
        base = ModelSpecBase(
            model=self.model_spec.model,
            max_tokens=self.model_spec.max_tokens,
            temperature=self.model_spec.temperature,
            max_tool_iterations=self.model_spec.max_tool_iterations,
        )
        return [base, *self.model_spec.fallbacks]

    async def _chat_with_fallback(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
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

        for spec in self._model_candidates():
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
