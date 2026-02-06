"""Shell execution tool."""

import asyncio
import os
from typing import Any

from nanobot.agent.tools.base import Tool

# Env vars stripped from subprocesses to prevent credential leakage
SENSITIVE_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "GROQ_API_KEY",
    "ZHIPUAI_API_KEY",
    "OPENCLAW_GATEWAY_TOKEN",
    "BRAVE_API_KEY",
]


def _build_clean_env() -> dict[str, str]:
    """Return a copy of os.environ with sensitive keys removed."""
    env = os.environ.copy()
    for key in SENSITIVE_ENV_VARS:
        env.pop(key, None)
    # Also strip anything matching NANOBOT_PROVIDERS__*
    for key in list(env.keys()):
        if key.startswith("NANOBOT_PROVIDERS__"):
            del env[key]
    return env


class ExecTool(Tool):
    """Tool to execute shell commands."""

    def __init__(self, timeout: int = 60, working_dir: str | None = None):
        self.timeout = timeout
        self.working_dir = working_dir

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Execute a shell command and return its output. Use with caution."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 60, max 300). Use higher values for pip install, builds, etc.",
                },
            },
            "required": ["command"],
        }

    async def execute(self, **kwargs: Any) -> str:
        command: str = kwargs["command"]
        working_dir: str | None = kwargs.get("working_dir")
        timeout: int | None = kwargs.get("timeout")

        cwd = working_dir or self.working_dir or os.getcwd()
        # Allow per-command timeout, capped at 5 minutes
        cmd_timeout = min(timeout or self.timeout, 300)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=_build_clean_env(),
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=cmd_timeout)
            except asyncio.TimeoutError:
                process.kill()
                return f"Error: Command timed out after {cmd_timeout} seconds. Use timeout parameter for long-running commands (max 300)."

            output_parts = []

            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            if process.returncode != 0:
                output_parts.append(f"\nExit code: {process.returncode}")

            result = "\n".join(output_parts) if output_parts else "(no output)"

            # Truncate very long output
            max_len = 10000
            if len(result) > max_len:
                result = result[:max_len] + f"\n... (truncated, {len(result) - max_len} more chars)"

            return result

        except Exception as e:
            return f"Error executing command: {str(e)}"
