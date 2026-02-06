"""Agent loop: the core processing engine."""

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage, MSG_SYSTEM_HEARTBEAT
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.config.schema import ModelSpec, ModelSpecBase, AgentRole
from nanobot.agent.compactor import Compactor
from nanobot.agent.context import ContextBuilder
from nanobot.agent.pruner import ContextPruner
from nanobot.agent.tokens import TokenCounter
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.media import (
    AnalyzeMediaTool,
    parse_media_injection,
    build_media_content,
    model_supports_media,
)
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
        self.token_counter = TokenCounter()
        self.compactor = Compactor(provider=provider, token_counter=self.token_counter)
        self.pruner = ContextPruner()
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

        # Media analysis tool (video, audio, image)
        self.tools.register(AnalyzeMediaTool())

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")

        # Check for interrupted sessions (e.g., from OOM/crash) and auto-resume
        await self._recover_interrupted_sessions()

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

    def get_context_usage(self, messages: list[dict[str, Any]], model: str | None = None) -> tuple[int, int]:
        """Get current token usage vs context window limit.

        Args:
            messages: Current message array.
            model: Model identifier. Defaults to the configured model.

        Returns:
            Tuple of (tokens_used, context_limit).
        """
        model = model or self.model_spec.model
        return self.token_counter.get_context_usage(
            messages, model, context_window=self.model_spec.context_window
        )

    async def _recover_interrupted_sessions(self) -> None:
        """
        Check for sessions that were interrupted mid-processing (e.g., OOM/crash).

        Sends a recovery message to continue the interrupted task.

        To disable auto-recovery, create a file: ~/.nanobot/no_auto_continue
        Or run: touch ~/.nanobot/no_auto_continue
        Delete it to re-enable: rm ~/.nanobot/no_auto_continue
        """
        # Check for disable flag
        no_continue_flag = Path.home() / ".nanobot" / "no_auto_continue"
        if no_continue_flag.exists():
            logger.info("Auto-continue disabled (found ~/.nanobot/no_auto_continue)")
            # Clear all interrupted sessions without continuing
            for session_key, _, _ in self.sessions.get_interrupted_sessions():
                session = self.sessions.get_or_create(session_key)
                session.clear_processing()
                self.sessions.save(session)
                logger.info(f"Cleared interrupted flag for {session_key}")
            return

        interrupted = self.sessions.get_interrupted_sessions()

        if not interrupted:
            return

        logger.info(f"Found {len(interrupted)} interrupted session(s), attempting recovery...")

        for session_key, content, last_tools in interrupted:
            try:
                # Parse channel and chat_id from session key
                if ":" in session_key:
                    channel, chat_id = session_key.split(":", 1)
                else:
                    continue

                # Create a recovery message with tool context
                recovery_content = (
                    f"[System: Your previous task was interrupted (possibly due to a crash or memory issue). "
                    f"The task was: '{content}...' "
                )
                if last_tools:
                    recovery_content += f"Last tools executed before crash: {last_tools}. "
                recovery_content += "Please continue where you left off or report what went wrong.]"

                # Create an inbound message to process
                recovery_msg = InboundMessage(
                    channel=channel,
                    chat_id=chat_id,
                    sender_id=chat_id,
                    content=recovery_content,
                )

                logger.info(f"Recovering interrupted session: {session_key}")

                # Process and send response
                response = await self._process_message(recovery_msg)
                if response:
                    await self.bus.publish_outbound(response)

            except Exception as e:
                logger.error(f"Failed to recover session {session_key}: {e}")
                # Clear the processing flag to avoid infinite recovery loops
                session = self.sessions.get_or_create(session_key)
                session.clear_processing()
                self.sessions.save(session)

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

        is_heartbeat = msg.message_type == MSG_SYSTEM_HEARTBEAT

        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}")

        # Get or create session
        session_key = msg.session_key
        if is_heartbeat and self._last_user_context:
            last_channel, last_chat_id = self._last_user_context
            session_key = f"{last_channel}:{last_chat_id}"
        session = self.sessions.get_or_create(session_key)

        # Skip heartbeat if session is currently processing a user message
        if is_heartbeat and session.is_processing:
            logger.info("Heartbeat: skipping, session is currently processing")
            return None

        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            if is_heartbeat and self._last_user_context:
                message_tool.set_context(*self._last_user_context)
            else:
                message_tool.set_context(msg.channel, msg.chat_id)

        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            if is_heartbeat and self._last_user_context:
                spawn_tool.set_context(*self._last_user_context)
            else:
                spawn_tool.set_context(msg.channel, msg.chat_id)

        if not is_heartbeat and msg.channel != "system":
            self._last_user_context = (msg.channel, msg.chat_id)

        # Mark that we're processing this message (for crash recovery)
        if not is_heartbeat and msg.channel != "system":
            session.mark_processing(msg.content)
            self.sessions.save(session)

        # Build initial messages (use get_history for LLM-formatted messages)
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            media=msg.media if msg.media else None,
        )

        # Agent loop
        iteration = 0
        final_content = None
        recent_tools = []  # Track recent tool calls for crash recovery

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

            # Compact context if approaching token limits
            if await self.compactor.should_compact(messages, self.model_spec.model):
                memory_mgr = getattr(self.context, "memory", None)
                messages, comp_meta = await self.compactor.compact(
                    messages, self.model_spec.model, memory_manager=memory_mgr
                )
                session.compaction_count += 1
                self.sessions.save(session)
                logger.info(
                    f"Context compacted: {comp_meta['tokens_before']} -> "
                    f"{comp_meta['tokens_after']} tokens, "
                    f"{comp_meta['turns_removed']} turns removed"
                )

            # Prune stale tool outputs before calling LLM
            current_turn = sum(1 for m in messages if m.get("role") == "user")
            pruned_messages = self.pruner.prune(messages, current_turn)

            # Call LLM (no tools on final iteration to force a text response)
            response = await self._chat_with_fallback(
                messages=pruned_messages,
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
                pending_media = []  # Collect media injections to add after tool results
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments)
                    logger.debug(f"Executing tool: {tool_call.name} with arguments: {args_str}")

                    # Track recent tools for crash recovery
                    recent_tools.append(f"{tool_call.name}: {args_str[:100]}")
                    if len(recent_tools) > 5:
                        recent_tools.pop(0)

                    # Update processing marker with recent tool context
                    if not is_heartbeat and msg.channel != "system":
                        session.mark_processing(msg.content, " | ".join(recent_tools))
                        self.sessions.save(session)

                    result = await self.tools.execute(tool_call.name, tool_call.arguments)

                    # Check if this is a media injection request
                    media_data = parse_media_injection(result)
                    if media_data:
                        # Add a placeholder result, queue the media for injection
                        path = media_data.get("path", "unknown")
                        media_type = media_data.get("media_type", "media")
                        size_mb = media_data.get("size_bytes", 0) / (1024 * 1024)
                        placeholder = f"[Analyzing {media_type}: {path} ({size_mb:.1f}MB)]"
                        messages = self.context.add_tool_result(
                            messages, tool_call.id, tool_call.name, placeholder
                        )
                        pending_media.append(media_data)
                        logger.debug(f"Media injection queued: {media_type} from {path}")
                    else:
                        messages = self.context.add_tool_result(
                            messages, tool_call.id, tool_call.name, result
                        )

                    if tool_call.name == "message":
                        message_tool_used = True

                # Inject any pending media as a user message for the next LLM call
                if pending_media:
                    for media_data in pending_media:
                        media_content = build_media_content(media_data)
                        messages.append({"role": "user", "content": media_content})
                        logger.debug(f"Injected {media_data['media_type']} into messages")
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

        # Heartbeat messages are ephemeral - don't save to session history
        if is_heartbeat:
            logger.info("Heartbeat: processed (ephemeral, not saved to history)")
        else:
            # Save to session and clear processing flag (task completed)
            session.add_message("user", msg.content)
            session.add_message("assistant", final_content)
            session.clear_processing()
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

            # Compact context if approaching token limits
            if await self.compactor.should_compact(messages, self.model_spec.model):
                memory_mgr = getattr(self.context, "memory", None)
                messages, comp_meta = await self.compactor.compact(
                    messages, self.model_spec.model, memory_manager=memory_mgr
                )
                session.compaction_count += 1
                self.sessions.save(session)

            # Prune stale tool outputs before calling LLM
            current_turn = sum(1 for m in messages if m.get("role") == "user")
            pruned_messages = self.pruner.prune(messages, current_turn)

            response = await self._chat_with_fallback(
                messages=pruned_messages,
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

    def _strip_media_from_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove media content from messages for text-only models."""
        stripped = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                # Multi-part content - keep only text parts
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part)
                        elif part.get("type") in ("image_url", "input_audio"):
                            # Skip media, add placeholder
                            text_parts.append(
                                {
                                    "type": "text",
                                    "text": "[Media content not supported by this model]",
                                }
                            )
                    elif isinstance(part, str):
                        text_parts.append({"type": "text", "text": part})

                if text_parts:
                    # If only one text part, flatten to string
                    if len(text_parts) == 1 and text_parts[0].get("type") == "text":
                        stripped.append({**msg, "content": text_parts[0]["text"]})
                    else:
                        stripped.append({**msg, "content": text_parts})
            else:
                stripped.append(msg)
        return stripped

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
        candidates = self._model_candidates()

        for i, spec in enumerate(candidates):
            # Check if model supports media - if not, strip media from messages
            if model_supports_media(spec.model):
                model_messages = messages
            else:
                logger.info(f"Model {spec.model} doesn't support media, stripping media content")
                model_messages = self._strip_media_from_messages(messages)

            # Retry loop for transient errors on this model
            for attempt in range(max_retries):
                try:
                    response = await self.provider.chat(
                        messages=model_messages,
                        tools=tools,
                        model=spec.model,
                        max_tokens=spec.max_tokens,
                        temperature=spec.temperature,
                    )
                except Exception as exc:
                    error_str = str(exc).lower()
                    # Check if transient error (503, 429, 500, timeout, etc.)
                    # Note: 404 is usually permanent (model not found) so we fallback immediately
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

                    # 404 means model not found - fallback immediately without retry
                    is_not_found = "404" in error_str or "not found" in error_str
                    if is_not_found:
                        logger.warning(f"Model {spec.model} not found (404), will fallback")
                        last_error = LLMResponse(
                            content=f"Error calling LLM: {exc}",
                            finish_reason="error",
                        )
                        break  # Move to next model immediately

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

                # Success - log token usage from response
                if response.usage:
                    logger.debug(
                        f"Token usage [{spec.model}]: "
                        f"prompt={response.usage.get('prompt_tokens', 0)}, "
                        f"completion={response.usage.get('completion_tokens', 0)}, "
                        f"total={response.usage.get('total_tokens', 0)}"
                    )
                return response

            # If we broke out of retry loop, try next model
            if last_error:
                next_model = (
                    candidates[i + 1].model if i + 1 < len(candidates) else "none (last model)"
                )
                error_msg = last_error.content[:200] if last_error.content else "unknown error"
                logger.warning(
                    f"Model {spec.model} failed, falling back to {next_model} | {error_msg}"
                )

        return last_error or LLMResponse(
            content="Error calling LLM: all fallback models failed",
            finish_reason="error",
        )
