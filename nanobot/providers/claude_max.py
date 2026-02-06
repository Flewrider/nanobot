"""Claude Max provider — routes through the Claude CLI (Claude Max subscription)."""

import asyncio
import json
import os
import uuid
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class ClaudeMaxProvider(LLMProvider):
    """
    LLM provider that routes through the Claude CLI.

    Uses the user's Claude Max subscription via the ``claude`` CLI tool.
    No API key needed — authentication is handled by the CLI's own OAuth token.

    Prerequisites:
        npm install -g @anthropic-ai/claude-code
        claude setup-token
    """

    def __init__(self, cli_path: str = "claude"):
        super().__init__()
        self.cli_path = cli_path

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send a chat request through the Claude CLI."""
        model = model or "claude-sonnet-4-5"

        # Strip claude_max/ prefix if present
        if model.startswith("claude_max/"):
            model = model[len("claude_max/"):]

        # Convert messages to a single prompt string
        prompt = self._messages_to_prompt(messages)

        # Build CLI command
        cmd = [
            self.cli_path,
            "-p",
            "--output-format", "json",
            "--model", model,
            "--max-turns", "1",
        ]

        # Build a clean environment: inherit current env but strip sensitive keys
        env = os.environ.copy()
        for key in list(env.keys()):
            if key == "ANTHROPIC_API_KEY":
                del env[key]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt.encode("utf-8")),
                timeout=300,
            )

            if process.returncode != 0:
                error_text = stderr.decode("utf-8", errors="replace").strip()
                logger.error(f"Claude CLI failed (exit {process.returncode}): {error_text}")
                return LLMResponse(
                    content=f"Error calling Claude CLI: {error_text}",
                    finish_reason="error",
                )

            output = stdout.decode("utf-8", errors="replace").strip()
            return self._parse_cli_output(output)

        except asyncio.TimeoutError:
            logger.error("Claude CLI timed out after 300s")
            return LLMResponse(
                content="Error: Claude CLI timed out after 300 seconds",
                finish_reason="error",
            )
        except FileNotFoundError:
            return LLMResponse(
                content=(
                    "Error: Claude CLI not found. Install with: "
                    "npm install -g @anthropic-ai/claude-code"
                ),
                finish_reason="error",
            )
        except Exception as e:
            logger.error(f"Claude CLI error: {e}")
            return LLMResponse(
                content=f"Error calling Claude CLI: {e}",
                finish_reason="error",
            )

    def get_default_model(self) -> str:
        return "claude-sonnet-4-5"

    def _messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Convert message list to a single prompt string for the CLI."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                # Multi-part content — extract text only
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "\n".join(text_parts)

            if role == "system":
                parts.append(f"<system>\n{content}\n</system>")
            elif role == "user":
                parts.append(f"<user>\n{content}\n</user>")
            elif role == "assistant":
                # Include assistant content and any tool calls
                if content:
                    parts.append(f"<assistant>\n{content}\n</assistant>")
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        name = func.get("name", "")
                        args = func.get("arguments", "")
                        if isinstance(args, dict):
                            args = json.dumps(args)
                        parts.append(f"<tool_call name=\"{name}\">\n{args}\n</tool_call>")
            elif role == "tool":
                tool_name = msg.get("name", "unknown")
                parts.append(f"<tool_result name=\"{tool_name}\">\n{content}\n</tool_result>")

        return "\n\n".join(parts)

    def _parse_cli_output(self, output: str) -> LLMResponse:
        """Parse JSON output from the Claude CLI."""
        if not output:
            return LLMResponse(content="", finish_reason="stop")

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            # If not valid JSON, treat the raw output as the response text
            return LLMResponse(content=output, finish_reason="stop")

        # The CLI JSON output format has a "result" field with the response text
        # Format: {"type": "result", "subtype": "success", "result": "...", ...}
        if isinstance(data, dict):
            result_text = data.get("result", "")
            cost = data.get("cost_usd", 0)
            session_id = data.get("session_id", "")

            # Extract usage if available
            usage = {}
            if "usage" in data:
                u = data["usage"]
                usage = {
                    "prompt_tokens": u.get("input_tokens", 0),
                    "completion_tokens": u.get("output_tokens", 0),
                    "total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0),
                }

            if result_text:
                return LLMResponse(
                    content=result_text,
                    finish_reason="stop",
                    usage=usage,
                )

            # Fallback: might be a list of content blocks
            content_blocks = data.get("content", [])
            if isinstance(content_blocks, list):
                text_parts = []
                tool_calls = []
                for block in content_blocks:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            tool_calls.append(
                                ToolCallRequest(
                                    id=block.get("id", str(uuid.uuid4())[:8]),
                                    name=block.get("name", ""),
                                    arguments=block.get("input", {}),
                                )
                            )
                return LLMResponse(
                    content="\n".join(text_parts) if text_parts else None,
                    tool_calls=tool_calls,
                    finish_reason="stop",
                    usage=usage,
                )

        # List of content blocks (alternative format)
        if isinstance(data, list):
            text_parts = []
            for block in data:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return LLMResponse(
                content="\n".join(text_parts) if text_parts else str(data),
                finish_reason="stop",
            )

        return LLMResponse(content=str(data), finish_reason="stop")
