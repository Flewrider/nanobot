"""Codex provider — routes through the OpenAI Codex CLI (ChatGPT/Codex subscription)."""

import asyncio
import json
import os
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse


class CodexProvider(LLMProvider):
    """
    LLM provider that routes through the OpenAI Codex CLI.

    Uses the user's ChatGPT Plus/Pro subscription via the ``codex`` CLI tool.
    No API key needed — authentication is handled by the CLI's own OAuth flow.

    Prerequisites:
        Install the Codex CLI:
            npm install -g @openai/codex

        Then authenticate:
            codex login
    """

    def __init__(self, cli_path: str = "codex"):
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
        """Send a chat request through the Codex CLI."""
        model = model or "o4-mini"

        # Strip codex/ prefix if present
        if model.startswith("codex/"):
            model = model[len("codex/"):]

        # Convert messages to a single prompt string
        prompt = self._messages_to_prompt(messages)

        # Build CLI command — use exec mode with JSON output
        cmd = [
            self.cli_path,
            "exec",
            "--json",
            "--model", model,
            prompt,
        ]

        # Build a clean environment: inherit current env but strip sensitive keys
        env = os.environ.copy()
        for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY"):
            env.pop(key, None)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300,
            )

            if process.returncode != 0:
                error_text = stderr.decode("utf-8", errors="replace").strip()
                logger.error(f"Codex CLI failed (exit {process.returncode}): {error_text}")
                return LLMResponse(
                    content=f"Error calling Codex CLI: {error_text}",
                    finish_reason="error",
                )

            output = stdout.decode("utf-8", errors="replace").strip()
            return self._parse_cli_output(output)

        except asyncio.TimeoutError:
            logger.error("Codex CLI timed out after 300s")
            return LLMResponse(
                content="Error: Codex CLI timed out after 300 seconds",
                finish_reason="error",
            )
        except FileNotFoundError:
            return LLMResponse(
                content=(
                    "Error: Codex CLI not found. Install with: "
                    "npm install -g @openai/codex"
                ),
                finish_reason="error",
            )
        except Exception as e:
            logger.error(f"Codex CLI error: {e}")
            return LLMResponse(
                content=f"Error calling Codex CLI: {e}",
                finish_reason="error",
            )

    def get_default_model(self) -> str:
        return "o4-mini"

    def _messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Convert message list to a single prompt string for the CLI."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "\n".join(text_parts)

            if role == "system":
                parts.append(f"[System]\n{content}")
            elif role == "user":
                parts.append(f"[User]\n{content}")
            elif role == "assistant":
                if content:
                    parts.append(f"[Assistant]\n{content}")
            elif role == "tool":
                tool_name = msg.get("name", "unknown")
                parts.append(f"[Tool: {tool_name}]\n{content}")

        return "\n\n".join(parts)

    def _parse_cli_output(self, output: str) -> LLMResponse:
        """Parse JSONL output from the Codex CLI.

        The ``codex exec --json`` command streams newline-delimited JSON events.
        We extract text from ``item.agentMessage`` events.
        """
        if not output:
            return LLMResponse(content="", finish_reason="stop")

        text_parts: list[str] = []
        usage: dict[str, int] = {}

        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # If the whole output isn't JSONL, treat as raw text
                if not text_parts:
                    return LLMResponse(content=output, finish_reason="stop")
                continue

            event_type = event.get("type", "")

            if event_type == "item.agentMessage":
                text = event.get("text", "")
                if text:
                    text_parts.append(text)
            elif event_type == "turn.completed":
                # Extract usage if available
                u = event.get("usage")
                if isinstance(u, dict):
                    usage = {
                        "prompt_tokens": u.get("input_tokens", 0),
                        "completion_tokens": u.get("output_tokens", 0),
                        "total_tokens": (
                            u.get("input_tokens", 0) + u.get("output_tokens", 0)
                        ),
                    }

        if text_parts:
            return LLMResponse(
                content="\n".join(text_parts),
                finish_reason="stop",
                usage=usage,
            )

        # Fallback: try to parse the entire output as a single JSON object
        try:
            data = json.loads(output)
            if isinstance(data, dict):
                result = data.get("result", "") or data.get("text", "")
                if result:
                    return LLMResponse(content=result, finish_reason="stop")
        except json.JSONDecodeError:
            pass

        return LLMResponse(content=output, finish_reason="stop")
